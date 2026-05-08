"""Hypothesis Generation Agent - creates diverse molecular reaction pathways.

This agent uses an LLM with chemistry-specific prompt engineering to generate
novel reaction hypotheses at high temperature for maximum diversity.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field

from src.chemistry.rdkit_wrapper import validate_smiles_syntax
from src.data.models import (
    ChemicalEntity,
    ReactionConditions,
    ReactionHypothesis,
    ReactionType,
    Solvent,
)


class GeneratedReaction(BaseModel):
    """Schema for LLM-generated reaction."""

    reaction_type: str = Field(description="Type of chemical reaction")
    reactant_smiles: list[str] = Field(description="SMILES of reactants")
    product_smiles: list[str] = Field(description="SMILES of expected products")
    solvent: Optional[str] = Field(default=None, description="Reaction solvent")
    catalyst: Optional[str] = Field(default=None, description="Catalyst if needed")
    temperature_celsius: Optional[float] = Field(default=None)
    time_hours: Optional[float] = Field(default=None)
    yield_estimate: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    mechanism_steps: Optional[str] = Field(default=None)
    rationale: str = Field(description="Why this reaction should proceed")


SYSTEM_PROMPT = """You are an expert synthetic organic chemist. Generate diverse, creative, and chemically plausible reaction hypotheses.

IMPORTANT: You MUST specify the reaction_type using one of these EXACT names:
- esterification, amide_coupling, diels_alder, suzuki_coupling, wittig
- grignard, aldol_condensation, michael_addition, heck_reaction
- oxidation, reduction, hydrolysis, friedel_crafts, nucleophilic_substitution
- elimination, claisen_rearrangement, mannich, buchwald_hartwig, click_chemistry

Output ONLY a valid JSON object with these fields:
- reaction_type (string): EXACT name from the list above
- reactant_smiles (array of strings): SMILES strings for all reactants
- product_smiles (array of strings): SMILES strings for all products  
- solvent (string or null): reaction solvent if applicable
- catalyst (string or null): catalyst if applicable
- temperature_celsius (number or null): typical reaction temperature
- time_hours (number or null): typical reaction time
- yield_estimate (number or null): estimated yield 0-100
- mechanism_steps (string or null): brief mechanism description
- rationale (string): why this reaction should proceed

No markdown, no commentary — ONLY JSON."""

USER_PROMPT_TEMPLATES = [
    "Generate {n} novel {reaction_type} reactions using diverse aromatic substrates.",
    "Propose {n} creative {reaction_type} reactions involving {functional_group} chemistry.",
    "Design {n} {reaction_type} reactions useful in pharmaceutical synthesis.",
    "Generate {n} {reaction_type} reactions converting {start_material} derivatives to {target_class} compounds.",
    "Propose {n} {reaction_type} reactions under {condition_type} conditions.",
    "Create {n} {reaction_type} reactions exploiting {principle} for selectivity.",
    "Design {n} synthetic routes using {reaction_type} with {reagent_class} reagents.",
    "Generate {n} {reaction_type} reactions featuring {feature}.",
]

REACTION_TYPE_POOL = [
    "esterification",
    "amide_coupling",
    "diels_alder",
    "suzuki_coupling",
    "wittig",
    "grignard",
    "aldol_condensation",
    "michael_addition",
    "heck_reaction",
    "oxidation",
    "reduction",
    "hydrolysis",
    "friedel_crafts",
    "nucleophilic_substitution",
    "elimination",
    "claisen_rearrangement",
    "mannich",
    "buchwald_hartwig",
    "click_chemistry",
]


class HypothesisGenerationAgent:
    """Agent that generates diverse chemical reaction hypotheses using an LLM."""

    def __init__(
        self,
        llm: BaseChatModel,
        temperature: float = 0.9,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        num_generations_per_prompt: int = 5,
    ):
        self.llm = llm
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.num_generations_per_prompt = num_generations_per_prompt
        self._prompt_index = 0

    def generate(
        self,
        session_id: UUID,
        num_hypotheses: int = 5,
        reaction_type: ReactionType | None = None,
        seed_prompt: str | None = None,
        learning_context: str | None = None,
    ) -> list[ReactionHypothesis]:
        """Generate reaction hypotheses.

        Args:
            session_id: Pipeline session ID.
            num_hypotheses: Number of hypotheses to generate.
            reaction_type: Optional specific reaction type to focus on.
            seed_prompt: Optional custom prompt to use.

        Returns:
            List of validated ReactionHypothesis objects.
        """
        hypotheses: list[ReactionHypothesis] = []
        batch_size = self.num_generations_per_prompt

        for batch_start in range(0, num_hypotheses, batch_size):
            batch_n = min(batch_size, num_hypotheses - batch_start)
            batch_hypotheses = self._generate_batch(
                session_id=session_id,
                n=batch_n,
                reaction_type=reaction_type,
                seed_prompt=seed_prompt,
                learning_context=learning_context,
            )
            hypotheses.extend(batch_hypotheses)
            logger.info(
                "Generated batch: {}/{} hypotheses",
                len(hypotheses),
                num_hypotheses,
            )

        logger.info(
            "Hypothesis generation complete: {} total, {} valid SMILES",
            len(hypotheses),
            sum(1 for h in hypotheses if _all_smiles_valid(h)),
        )
        return hypotheses

    def _generate_batch(
        self,
        session_id: UUID,
        n: int,
        reaction_type: ReactionType | None = None,
        seed_prompt: str | None = None,
        learning_context: str | None = None,
    ) -> list[ReactionHypothesis]:
        """Generate a single batch of hypotheses.

        Args:
            learning_context: Optional prompt snippet with accumulated failure
                patterns and success patterns from prior iterations.
        """
        if seed_prompt:
            user_prompt = seed_prompt
        else:
            template_idx = self._prompt_index % len(USER_PROMPT_TEMPLATES)
            self._prompt_index += 1
            # Cycle through specific reaction types for diversity
            rt = reaction_type.value if reaction_type else REACTION_TYPE_POOL[self._prompt_index % len(REACTION_TYPE_POOL)]
            user_prompt = USER_PROMPT_TEMPLATES[template_idx].format(
                n=n,
                reaction_type=rt.replace("_", " "),
                functional_group="carbonyl",
                start_material="aromatic",
                target_class="heterocyclic",
                condition_type="mild",
                principle="neighboring group participation",
                reagent_class="organometallic",
                feature="quaternary carbon centers",
            )

        system_content = SYSTEM_PROMPT
        if learning_context:
            system_content += learning_context

        system_msg = SystemMessage(content=system_content)
        user_msg = HumanMessage(content=user_prompt)

        logger.debug("Calling LLM for hypothesis generation: {}", user_prompt[:100])

        response = self.llm.invoke(
            [system_msg, user_msg],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return self._parse_response(
            response=response.content,
            session_id=session_id,
            prompt_used=user_prompt,
        )

    def _parse_response(
        self,
        response: str,
        session_id: UUID,
        prompt_used: str,
    ) -> list[ReactionHypothesis]:
        """Parse LLM response into ReactionHypothesis objects."""
        hypotheses: list[ReactionHypothesis] = []

        try:
            data = self._extract_json(response)
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = [data]
            else:
                logger.warning("Unexpected LLM response format: {}", type(data))
                return hypotheses

            for item in items:
                try:
                    generated = GeneratedReaction.model_validate(item)
                    hypothesis = self._to_hypothesis(
                        generated, session_id, prompt_used
                    )
                    hypotheses.append(hypothesis)
                except Exception as e:
                    logger.warning("Failed to parse generated reaction: {}", e)
                    continue

        except Exception as e:
            logger.error("Failed to parse LLM response entirely: {}", e)
            logger.debug("Raw response: {}", response[:500])

        return hypotheses

    def _extract_json(self, text: str) -> list | dict:
        """Extract JSON from LLM response (handles markdown code blocks and commentary)."""
        text = text.strip()

        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
                if text.startswith("json"):
                    text = text[4:].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                inner = text[start:end]
                return json.loads(inner)
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                inner = text[start:end]
                return json.loads(inner)
            raise

    def _to_hypothesis(
        self,
        generated: GeneratedReaction,
        session_id: UUID,
        prompt_used: str,
    ) -> ReactionHypothesis:
        """Convert GeneratedReaction to ReactionHypothesis."""
        reactants = [
            ChemicalEntity(smiles=s) for s in generated.reactant_smiles
        ]
        products = [
            ChemicalEntity(smiles=s) for s in generated.product_smiles
        ]

        try:
            reaction_type = _parse_reaction_type(generated.reaction_type)
        except ValueError:
            reaction_type = ReactionType.OTHER

        solvent = None
        if generated.solvent:
            try:
                solvent = Solvent(generated.solvent.lower().replace(" ", "_"))
            except ValueError:
                solvent = Solvent.OTHER

        return ReactionHypothesis(
            id=uuid4(),
            session_id=session_id,
            reactants=reactants,
            products=products,
            reaction_type=reaction_type,
            conditions=ReactionConditions(
                temperature_celsius=generated.temperature_celsius,
                solvent=solvent,
                catalyst=generated.catalyst,
                time_hours=generated.time_hours,
            ),
            yield_estimate=generated.yield_estimate,
            mechanism_steps=generated.mechanism_steps,
            rationale=generated.rationale,
            prompt_used=prompt_used,
            generation_temperature=self.temperature,
            created_at=datetime.now(),
        )


def _all_smiles_valid(hypothesis: ReactionHypothesis) -> bool:
    """Check if all SMILES in a hypothesis pass syntax validation."""
    for mol in hypothesis.reactants + hypothesis.products:
        if not validate_smiles_syntax(mol.smiles):
            return False
    return True


def _parse_reaction_type(raw: str) -> ReactionType:
    """Parse LLM-generated reaction type string into ReactionType enum.

    Handles common variations: 'Diels-Alder' → DIELS_ALDER,
    'Suzuki coupling' → SUZUKI_COUPLING, etc.
    """
    normalized = raw.lower().strip().replace(" ", "_").replace("-", "_").replace("/", "_")

    # Direct mapping for common LLM variations
    aliases = {
        "diels_alder": "diels_alder",
        "diels_alder_reaction": "diels_alder",
        "suzuki_coupling": "suzuki_coupling",
        "suzuki": "suzuki_coupling",
        "suzuki_miyaura": "suzuki_coupling",
        "suzuki_cross_coupling": "suzuki_coupling",
        "heck": "heck_reaction",
        "heck_coupling": "heck_reaction",
        "mizoroki_heck": "heck_reaction",
        "grignard": "grignard",
        "grignard_reaction": "grignard",
        "grignard_addition": "grignard",
        "wittig": "wittig",
        "wittig_reaction": "wittig",
        "friedel_crafts": "friedel_crafts",
        "friedel_crafts_acylation": "friedel_crafts",
        "friedel_crafts_alkylation": "friedel_crafts",
        "aldol": "aldol_condensation",
        "aldol_reaction": "aldol_condensation",
        "michael_addition": "michael_addition",
        "michael": "michael_addition",
        "amide_bond_formation": "amide_coupling",
        "peptide_coupling": "amide_coupling",
        "amide_synthesis": "amide_coupling",
        "sn2": "nucleophilic_substitution",
        "sn1": "nucleophilic_substitution",
        "substitution": "nucleophilic_substitution",
        "nucleophilic": "nucleophilic_substitution",
        "esterification": "esterification",
        "ester_synthesis": "esterification",
        "oxidation": "oxidation",
        "reduction": "reduction",
        "hydrolysis": "hydrolysis",
        "elimination": "elimination",
        "mannich": "mannich",
        "mannich_reaction": "mannich",
        "buchwald": "buchwald_hartwig",
        "buchwald_hartwig": "buchwald_hartwig",
        "click": "click_chemistry",
        "click_chemistry": "click_chemistry",
        "claisen": "claisen_rearrangement",
        "claisen_rearrangement": "claisen_rearrangement",
        "williamson_ether_synthesis": "nucleophilic_substitution",
    }

    mapped = aliases.get(normalized)
    if mapped:
        return ReactionType(mapped)

    try:
        return ReactionType(normalized)
    except ValueError:
        return ReactionType.OTHER
