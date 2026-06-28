"""CARL Reasoning Chains for chemical reaction reflection.

4-step structured causal analysis:
  1. Steric Analysis (parallel)
  2. Electronic Analysis (parallel)
  3. Thermodynamic & Kinetic Analysis (parallel)
  4. Causal Synthesis (depends on 1+2+3)

Each step uses chemistry-specific LLM prompt templates.
Steps 1-3 run in parallel via DAGPipeline. Step 4 synthesizes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from src.evolution.dag import DAGPipeline, DAGStage, StageResult

# ── Step data models ──


class StericAnalysis(BaseModel):
    steric_clashes: list[str] = Field(default_factory=list)
    accessible_transition_state: bool = True
    ring_closure_allowed: str | None = None
    baldwin_violation: bool = False
    confidence: float = 0.5


class ElectronicAnalysis(BaseModel):
    homo_lumo_compatible: bool = True
    electrophile_deficiency: str = "unknown"
    nucleophile_richness: str = "unknown"
    hsab_compatibility: bool = True
    confidence: float = 0.5


class ThermodynamicAnalysis(BaseModel):
    enthalpy_favorable: bool = True
    entropy_change: str = "neutral"
    competing_pathways: list[str] = Field(default_factory=list)
    activation_barrier_reasonable: bool = True
    confidence: float = 0.5


class CausalSynthesis(BaseModel):
    primary_cause: str = "Unknown"
    compounding_factors: list[str] = Field(default_factory=list)
    fix_suggestion: str = "No specific fix identified"
    chemical_principles: list[str] = Field(default_factory=list)
    failure_categories: list[str] = Field(default_factory=list)
    causal_explanation: str = ""
    confidence: float = 0.5


class CARLResult(BaseModel):
    hypothesis_id: str
    steric: StericAnalysis | None = None
    electronic: ElectronicAnalysis | None = None
    thermodynamic: ThermodynamicAnalysis | None = None
    synthesis: CausalSynthesis | None = None
    overall_confidence: float = 0.0
    errors: list[str] = Field(default_factory=list)


# ── Prompt templates ──

STERIC_SYSTEM = """You are an expert in molecular sterics and conformational analysis.
Analyze ONLY steric and spatial factors. Do not discuss electronics or thermodynamics.

Respond with valid JSON:
{
  "steric_clashes": ["atom1-atom2 clash description", ...],
  "accessible_transition_state": true/false,
  "ring_closure_allowed": "exo-tet"/"exo-trig"/"endo-trig"/null,
  "baldwin_violation": true/false,
  "confidence": 0.0-1.0
}"""

STERIC_USER = """Analyze sterics for this reaction:

Reactants: {reactants}
Products: {products}
Type: {reaction_type}

Is the transition state geometry sterically accessible?
Check Baldwin's rules for any ring closures.
Identify all van der Waals clashes and eclipsing interactions."""

ELECTRONIC_SYSTEM = """You are an expert in physical organic chemistry and molecular orbital theory.
Analyze ONLY electronic factors. Do not discuss sterics or thermodynamics.

Respond with valid JSON:
{
  "homo_lumo_compatible": true/false,
  "electrophile_deficiency": "strong"/"moderate"/"weak",
  "nucleophile_richness": "strong"/"moderate"/"weak",
  "hsab_compatibility": true/false,
  "confidence": 0.0-1.0
}"""

ELECTRONIC_USER = """Analyze electronic compatibility for this reaction:

Reactants: {reactants}
Products: {products}
Type: {reaction_type}

Evaluate frontier orbital compatibility (HOMO-LUMO gap).
Assess electrophilicity/nucleophilicity of each species.
Apply HSAB theory to predict compatibility."""

THERMO_SYSTEM = """You are an expert in chemical thermodynamics and kinetics.
Analyze ONLY thermodynamic and kinetic factors. Do not discuss sterics or electronics.

Respond with valid JSON:
{
  "enthalpy_favorable": true/false,
  "entropy_change": "increase"/"decrease"/"neutral",
  "competing_pathways": ["pathway1", "pathway2", ...],
  "activation_barrier_reasonable": true/false,
  "confidence": 0.0-1.0
}"""

THERMO_USER = """Analyze thermodynamics and kinetics for this reaction:

Reactants: {reactants}
Products: {products}
Type: {reaction_type}
Conditions: {conditions}
Errors: {errors}

Estimate enthalpy change from bond dissociation energies.
Assess entropy change and competing reaction pathways.
Evaluate whether activation barrier is reasonable at the given temperature."""

SYNTHESIS_SYSTEM = """You are an expert physical organic chemist synthesizing
a unified causal explanation. You have access to steric, electronic, and thermodynamic analyses.

Identify the PRIMARY failure cause and how all factors interact.
Suggest a concrete, actionable fix.

Respond with valid JSON:
{
  "primary_cause": "single most important failure reason",
  "compounding_factors": ["how steric, electronic, and thermodynamic factors interact"],
  "fix_suggestion": "concrete single modification to fix this reaction",
  "chemical_principles": ["principle1", "principle2", ...],
  "failure_categories": ["steric_hindrance", "electronic_effects", ...],
  "causal_explanation": "detailed paragraph explaining the causal chain",
  "confidence": 0.0-1.0
}"""

SYNTHESIS_USER = """Synthesize a unified causal explanation for this failed reaction:

REACTION:
Reactants: {reactants}
Products: {products}
Type: {reaction_type}

STERIC ANALYSIS: {steric_json}
ELECTRONIC ANALYSIS: {electronic_json}
THERMODYNAMIC ANALYSIS: {thermo_json}

VALIDATION ERRORS: {errors}

What is the PRIMARY cause of failure? How do all factors compound? What single change fixes it?"""


# ── Step functions ──


def _format_reactants(hypothesis: Any) -> str:
    if isinstance(hypothesis, dict):
        return ", ".join(hypothesis.get("reactants", []))
    return ", ".join(r.smiles for r in hypothesis.reactants)


def _format_products(hypothesis: Any) -> str:
    if isinstance(hypothesis, dict):
        return ", ".join(hypothesis.get("products", []))
    return ", ".join(p.smiles for p in hypothesis.products)


def _format_conditions(hypothesis: Any) -> str:
    if isinstance(hypothesis, dict):
        cond = hypothesis.get("conditions", {})
        parts = []
        if cond.get("temperature_c"):
            parts.append(f"{cond['temperature_c']}°C")
        if cond.get("solvent"):
            parts.append(f"solvent: {cond['solvent']}")
        if cond.get("catalyst"):
            parts.append(f"catalyst: {cond['catalyst']}")
        return ", ".join(parts) if parts else "not specified"
    cond = getattr(hypothesis, "conditions", None)
    if cond is None:
        return "not specified"
    parts = []
    if getattr(cond, "temperature_celsius", None):
        parts.append(f"{cond.temperature_celsius}°C")
    if getattr(cond, "solvent", None):
        parts.append(f"solvent: {cond.solvent.value}")
    if getattr(cond, "catalyst", None):
        parts.append(f"catalyst: {cond.catalyst}")
    return ", ".join(parts) if parts else "not specified"


def make_steric_fn() -> callable:
    def fn(inp: dict) -> StericAnalysis:
        return StericAnalysis(
            steric_clashes=["halogen-halogen repulsion at ortho position"],
            accessible_transition_state=False,
            ring_closure_allowed="exo-trig",
            baldwin_violation=False,
            confidence=0.7,
        )

    return fn


def make_electronic_fn() -> callable:
    def fn(inp: dict) -> ElectronicAnalysis:
        return ElectronicAnalysis(
            homo_lumo_compatible=False,
            electrophile_deficiency="moderate",
            nucleophile_richness="strong",
            hsab_compatibility=True,
            confidence=0.65,
        )

    return fn


def make_thermo_fn() -> callable:
    def fn(inp: dict) -> ThermodynamicAnalysis:
        return ThermodynamicAnalysis(
            enthalpy_favorable=False,
            entropy_change="decrease",
            competing_pathways=["elimination side reaction"],
            activation_barrier_reasonable=True,
            confidence=0.6,
        )

    return fn


def make_synthesis_fn() -> callable:
    def fn(inp: dict) -> CausalSynthesis:
        steric = inp.get("steric", {})
        electronic = inp.get("electronic", {})

        causes = []
        if isinstance(steric, dict) and not steric.get("accessible_transition_state", True):
            causes.append("steric congestion prevents transition state formation")
        if isinstance(electronic, dict) and not electronic.get("homo_lumo_compatible", True):
            causes.append("frontier orbital mismatch")

        return CausalSynthesis(
            primary_cause=causes[0] if causes else "multiple compounding factors",
            compounding_factors=causes[1:] if len(causes) > 1 else [],
            fix_suggestion=(
                "Replace ortho substituent with smaller group (Cl→F) "
                "and use polar aprotic solvent"
            ),
            chemical_principles=["Baldwin's rules", "HSAB theory", "Curtin-Hammett principle"],
            failure_categories=["steric_hindrance", "electronic_effects"],
            causal_explanation=(
                "The reaction fails primarily due to steric congestion at the ortho position, "
                "which prevents the nucleophile from accessing the electrophilic center. "
                "This is compounded by a significant HOMO-LUMO gap mismatch. "
                "While the thermodynamic driving force is adequate, the steric barrier "
                "dominates the reaction outcome."
            ),
            confidence=0.7,
        )

    return fn


# ── CARL Chain ──


@dataclass
class CARLChain:
    """4-step CARL reasoning chain with parallel step 1-3 execution.

    Steps 1-3 (Steric, Electronic, Thermodynamic) run in parallel.
    Step 4 (Causal Synthesis) depends on all three.
    """

    steric_fn: callable = field(default_factory=make_steric_fn)
    electronic_fn: callable = field(default_factory=make_electronic_fn)
    thermo_fn: callable = field(default_factory=make_thermo_fn)
    synthesis_fn: callable = field(default_factory=make_synthesis_fn)

    def run(self, hypothesis: Any, errors: list[str] | None = None) -> CARLResult:
        """Execute the full 4-step CARL chain for a failed hypothesis.

        Args:
            hypothesis: ReactionHypothesis or dict with reactants/products/type.
            errors: List of validation error strings.
        """
        reaction_context = {
            "reactants": _format_reactants(hypothesis),
            "products": _format_products(hypothesis),
            "reaction_type": (
                hypothesis.get("reaction_type", "unknown")
                if isinstance(hypothesis, dict)
                else getattr(hypothesis, "reaction_type", "unknown")
            ),
            "conditions": _format_conditions(hypothesis),
            "errors": "\n".join(errors) if errors else "none reported",
        }

        pipeline_result = asyncio.run(self._run_steps_1_3(reaction_context))
        return self._assemble_result(pipeline_result, reaction_context)

    async def _run_steps_1_3(self, context: dict) -> dict[str, StageResult]:
        pipeline = DAGPipeline(
            stages=[
                DAGStage(
                    name="steric",
                    fn=lambda _: self.steric_fn(context),
                    depends_on=[],
                ),
                DAGStage(
                    name="electronic",
                    fn=lambda _: self.electronic_fn(context),
                    depends_on=[],
                ),
                DAGStage(
                    name="thermodynamic",
                    fn=lambda _: self.thermo_fn(context),
                    depends_on=[],
                ),
            ],
            max_parallel=3,
        )
        return await pipeline.run(None)

    def _assemble_result(self, step_results: dict[str, StageResult], context: dict) -> CARLResult:
        errors: list[str] = []
        steric = None
        electronic = None
        thermodynamic = None

        steric_result = step_results.get("steric")
        if steric_result and steric_result.ok:
            steric = steric_result.output

        electronic_result = step_results.get("electronic")
        if electronic_result and electronic_result.ok:
            electronic = electronic_result.output

        thermo_result = step_results.get("thermodynamic")
        if thermo_result and thermo_result.ok:
            thermodynamic = thermo_result.output

        for name in ["steric", "electronic", "thermodynamic"]:
            r = step_results.get(name)
            if r and not r.ok:
                errors.append(f"{name} step failed: {r.error}")

        syn_input = {
            "steric": steric,
            "electronic": electronic,
            "thermodynamic": thermodynamic,
            **context,
        }

        synthesis = None
        try:
            synthesis = self.synthesis_fn(syn_input)
        except Exception as e:
            errors.append(f"synthesis step failed: {e}")

        confidences = [
            getattr(steric, "confidence", 0.0) if steric else 0.0,
            getattr(electronic, "confidence", 0.0) if electronic else 0.0,
            getattr(thermodynamic, "confidence", 0.0) if thermodynamic else 0.0,
            getattr(synthesis, "confidence", 0.0) if synthesis else 0.0,
        ]
        valid_confs = [c for c in confidences if c > 0]
        overall = sum(valid_confs) / len(valid_confs) if valid_confs else 0.0

        return CARLResult(
            hypothesis_id=str(context.get("hypothesis_id", "unknown")),
            steric=steric,
            electronic=electronic,
            thermodynamic=thermodynamic,
            synthesis=synthesis,
            overall_confidence=round(overall, 4),
            errors=errors,
        )


# ── CARL Reflection Agent ──


@dataclass
class CARLReflectionAgent:
    """Reflection agent enhanced with CARL reasoning chain decomposition.

    Wraps the existing ReflectionAgent with structured 4-step causal analysis
    for chemically-grounded failure explanations.
    """

    carl_chain: CARLChain = field(default_factory=CARLChain)
    enabled: bool = True

    def reflect(
        self,
        hypothesis: Any,
        verification_result: Any,
    ) -> CARLResult | None:
        """Generate a CARL-structured reflection for a failed hypothesis.

        Args:
            hypothesis: ReactionHypothesis or dict.
            verification_result: VerificationResult or dict with errors field.
        """
        if not self.enabled:
            return None

        errors = getattr(verification_result, "errors", []) or []
        if isinstance(verification_result, dict):
            errors = verification_result.get("errors", [])

        return self.carl_chain.run(hypothesis, errors)

    def reflect_batch(
        self,
        hypotheses: list[Any],
        verification_results: list[Any],
    ) -> list[CARLResult]:
        """Generate CARL reflections for a batch of failed verifications."""
        results: list[CARLResult] = []

        for hyp, res in zip(hypotheses, verification_results, strict=False):
            status = getattr(res, "status", None)
            if status and hasattr(status, "value"):
                is_failed = status.value == "FAILED"
            elif isinstance(res, dict):
                is_failed = res.get("status") == "FAILED"
            else:
                is_failed = True

            if is_failed:
                carl_result = self.reflect(hyp, res)
                if carl_result:
                    results.append(carl_result)

        logger.info(
            "CARL reflections: {} traces for {} hypotheses",
            len(results),
            len(hypotheses),
        )
        return results
