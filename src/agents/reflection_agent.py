"""Reflection Agent — generates causal reasoning traces for failed reactions."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional
from uuid import UUID

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field

from src.data.models import (
    FailureCategory,
    LearningContext,
    ReactionHypothesis,
    ReflectionTrace,
    VerificationResult,
    VerificationStatus,
)


class ReflectionOutput(BaseModel):
    """Schema for LLM-generated reflection."""

    failure_categories: list[str] = Field(
        description="One or more failure categories: steric_hindrance, electronic_effects, "
        "thermodynamic_unfavorable, kinetic_barrier, solvent_incompatibility, "
        "regioselectivity_conflict, chemoselectivity_issue, valence_violation, "
        "ring_strain, other"
    )
    primary_cause: str = Field(
        description="The single most important reason this reaction fails"
    )
    causal_explanation: str = Field(
        description="Detailed step-by-step causal analysis explaining exactly WHY "
        "the reaction fails, written in the style of a professor correcting a student. "
        "Reference specific atoms, functional groups, steric interactions, "
        "and electronic effects. Minimum 3 sentences."
    )
    chemical_principles: list[str] = Field(
        description="List of relevant chemical principles (e.g., Baldwin's rules, "
        "HSAB theory, Curtin-Hammett principle)"
    )
    fix_suggestion: str = Field(
        description="Concrete modification that could make this reaction work "
        "(e.g., change solvent, use protecting group, alter substituent)"
    )
    confidence: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="How confident are you in this analysis? (0-1)"
    )


SYSTEM_PROMPT = """You are an expert physical organic chemist and chemical educator. 
Your task is to analyze failed chemical reactions and explain WHY they fail in a 
detailed, causally rigorous manner.

Guidelines:
1. Start from the fundamental physical principles (thermodynamics, kinetics, electronic structure)
2. Identify specific problematic atoms, bonds, or functional groups
3. Consider steric effects, electronic effects, and solvent effects
4. Explain using established chemical principles (Baldwin's rules, HSAB, 
   Hammond postulate, Curtin-Hammett, etc.)
5. Be precise — reference specific positions and functional groups
6. Suggest concrete modifications that would fix the problem
7. Avoid vague explanations like "it's unstable" — explain what kind of instability

The reaction data and verification failure details will be provided.

{format_instructions}"""

USER_PROMPT = """Analyze why this chemical reaction fails:

REACTION:
Reactants: {reactants}
Products: {products}
Reaction Type: {reaction_type}
Conditions: {conditions}

VERIFICATION FAILURES:
{errors}

VALIDATION RESULTS:
- SMILES valid: {smiles_valid}
- Valence valid: {valence_valid}
- Steric valid: {steric_valid}
- Energy valid: {energy_valid}

COMPUTED PROPERTIES:
{computed_properties}

Please provide a detailed causal analysis of why this reaction fails."""


class ReflectionAgent:
    """Agent that generates causal reasoning traces for failed reaction hypotheses."""

    def __init__(
        self,
        llm: BaseChatModel,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ):
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens

    def reflect(
        self,
        hypothesis: ReactionHypothesis,
        verification: VerificationResult,
    ) -> Optional[ReflectionTrace]:
        """Generate a reflection trace for a failed reaction.

        Only processes hypotheses where verification status is FAILED.
        Returns None if the hypothesis passed verification.

        Args:
            hypothesis: The failed reaction hypothesis.
            verification: The verification result with failure details.

        Returns:
            ReflectionTrace with causal analysis, or None if passed.
        """
        if verification.status != VerificationStatus.FAILED:
            return None

        reactants_str = ", ".join(
            f"{i+1}. {r.smiles}" for i, r in enumerate(hypothesis.reactants)
        )
        products_str = ", ".join(
            f"{i+1}. {p.smiles}" for i, p in enumerate(hypothesis.products)
        )

        conditions = []
        if hypothesis.conditions.temperature_celsius is not None:
            conditions.append(f"{hypothesis.conditions.temperature_celsius}°C")
        if hypothesis.conditions.solvent:
            conditions.append(f"solvent: {hypothesis.conditions.solvent.value}")
        if hypothesis.conditions.catalyst:
            conditions.append(f"catalyst: {hypothesis.conditions.catalyst}")
        conditions_str = ", ".join(conditions) if conditions else "not specified"

        errors_str = "\n".join(f"- {e}" for e in verification.errors) if verification.errors else "none"

        props = verification.computed_properties
        props_str = f"""- SA Score: {props.sa_score or 'N/A'}
- QED: {props.qed or 'N/A'}
- LogP: {props.logp or 'N/A'}
- Total Energy: {props.total_energy_hartree or 'N/A'} Eh
- HOMO-LUMO Gap: {props.gap_ev or 'N/A'} eV"""

        user_prompt = USER_PROMPT.format(
            reactants=reactants_str,
            products=products_str,
            reaction_type=hypothesis.reaction_type.value,
            conditions=conditions_str,
            errors=errors_str,
            smiles_valid=verification.smiles_valid,
            valence_valid=verification.valence_valid,
            steric_valid=verification.steric_valid,
            energy_valid=verification.energy_valid,
            computed_properties=props_str,
        )

        parser = None
        system_msg = SystemMessage(content=SYSTEM_PROMPT)
        user_msg = HumanMessage(content=user_prompt)

        logger.debug("Requesting reflection for hypothesis {}", hypothesis.id)

        try:
            response = self.llm.invoke(
                [system_msg, user_msg],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            logger.error("LLM call failed in reflection: {}", e)
            return ReflectionTrace(
                hypothesis_id=hypothesis.id,
                verification_result_id=verification.id,
                failure_categories=[FailureCategory.OTHER],
                primary_cause="LLM error during reflection",
                causal_explanation=f"Reflection generation failed: {e}",
                chemical_principles=[],
                confidence=0.1,
                created_at=datetime.now(),
            )

        return self._parse_response(
            response=response.content,
            hypothesis=hypothesis,
            verification=verification,
            prompt_used=user_prompt,
        )

    def reflect_batch(
        self,
        hypotheses: list[ReactionHypothesis],
        results: list[VerificationResult],
    ) -> list[ReflectionTrace]:
        """Generate reflections for a batch of verification results.

        Only reflects on failed verifications.

        Args:
            hypotheses: Original hypotheses.
            results: Corresponding verification results.

        Returns:
            List of ReflectionTrace objects (only for failures).
        """
        traces: list[ReflectionTrace] = []
        results_by_id = {r.hypothesis_id: r for r in results}

        for hyp in hypotheses:
            result = results_by_id.get(hyp.id)
            if result is None:
                continue
            trace = self.reflect(hypothesis=hyp, verification=result)
            if trace is not None:
                traces.append(trace)

        logger.info("Generated {} reflection traces for {} hypotheses", len(traces), len(hypotheses))
        return traces

    def _parse_response(
        self,
        response: str,
        hypothesis: ReactionHypothesis,
        verification: VerificationResult,
        prompt_used: str,
    ) -> ReflectionTrace:
        """Parse LLM response into ReflectionTrace, with fallback."""
        try:
            data = self._extract_json(response)
            reflected = ReflectionOutput.model_validate(data)

            categories: list[FailureCategory] = []
            for cat in reflected.failure_categories:
                try:
                    categories.append(FailureCategory(cat.lower().replace(" ", "_")))
                except ValueError:
                    categories.append(FailureCategory.OTHER)

            return ReflectionTrace(
                hypothesis_id=hypothesis.id,
                verification_result_id=verification.id,
                failure_categories=categories,
                primary_cause=reflected.primary_cause,
                causal_explanation=reflected.causal_explanation,
                chemical_principles=reflected.chemical_principles,
                fix_suggestion=reflected.fix_suggestion,
                confidence=reflected.confidence,
                prompt_used=prompt_used,
                created_at=datetime.now(),
            )
        except Exception as e:
            logger.warning("Failed to parse structured reflection, using raw text: {}", e)
            return ReflectionTrace(
                hypothesis_id=hypothesis.id,
                verification_result_id=verification.id,
                failure_categories=[FailureCategory.OTHER],
                primary_cause="Parse failure",
                causal_explanation=response[:2000],
                chemical_principles=[],
                confidence=0.3,
                prompt_used=prompt_used,
                created_at=datetime.now(),
            )

    def accumulate_learning(
        self,
        traces: list[ReflectionTrace],
        results: list[VerificationResult],
        learning_context: LearningContext | None = None,
    ) -> LearningContext:
        """Accumulate learning from reflection traces into a self-bootstrapping context.

        Distills patterns from failure analyses and successful verifications
        into actionable constraints for the hypothesis agent.

        Args:
            traces: Reflection traces from failed hypotheses.
            results: All verification results (passed + failed).
            learning_context: Existing context to update (creates new if None).

        Returns:
            Updated LearningContext with accumulated patterns.
        """
        from src.data.models import LearningContext

        if learning_context is None:
            learning_context = LearningContext()

        learning_context.iteration_count += 1

        passed = sum(1 for r in results if r.status == VerificationStatus.PASSED)
        failed = sum(1 for r in results if r.status == VerificationStatus.FAILED)
        learning_context.cumulative_hypotheses += len(results)
        learning_context.cumulative_passed += passed
        learning_context.cumulative_failed += failed

        for trace in traces:
            for fc in trace.failure_categories:
                cat = fc.value if hasattr(fc, "value") else str(fc)
                learning_context.failure_categories[cat] = (
                    learning_context.failure_categories.get(cat, 0) + 1
                )

            if trace.causal_explanation:
                excerpt = trace.causal_explanation[:200].strip()
                if excerpt not in learning_context.common_mistakes:
                    learning_context.common_mistakes.append(excerpt)

            if trace.fix_suggestion:
                excerpt = trace.fix_suggestion[:200].strip()
                if excerpt not in learning_context.successful_patterns:
                    learning_context.successful_patterns.append(excerpt)

        for result in results:
            if result.status == VerificationStatus.PASSED and result.computed_properties:
                props = result.computed_properties
                if props.sa_score:
                    learning_context.successful_patterns.append(
                        f"Reactions with SA score ~{props.sa_score:.1f} passed validation"
                    )
                if props.qed:
                    learning_context.successful_patterns.append(
                        f"Reactions with QED ~{props.qed:.2f} passed validation"
                    )

        logger.info(
            "Accumulated learning: {} iterations, {} traces, {} failure categories",
            learning_context.iteration_count,
            len(traces),
            len(learning_context.failure_categories),
        )
        return learning_context

    def _extract_json(self, text: str) -> dict:
        """Extract JSON from LLM response."""
        text = text.strip()
        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
                if text.startswith("json"):
                    text = text[4:].strip()
        return json.loads(text)
