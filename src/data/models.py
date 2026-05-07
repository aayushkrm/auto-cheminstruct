"""Core data models for Auto-ChemInstruct pipeline.

All data flowing through the multi-agent system uses these Pydantic models
for validation, serialization, and type safety.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class ReactionType(str, Enum):
    """Classification of chemical reaction types."""

    NUCLEOPHILIC_SUBSTITUTION = "nucleophilic_substitution"
    ELECTROPHILIC_ADDITION = "electrophilic_addition"
    ELIMINATION = "elimination"
    CONDENSATION = "condensation"
    OXIDATION = "oxidation"
    REDUCTION = "reduction"
    CYCLOADDITION = "cycloaddition"
    REARRANGEMENT = "rearrangement"
    CROSS_COUPLING = "cross_coupling"
    PROTECTION = "protection"
    DEPROTECTION = "deprotection"
    OTHER = "other"


class Solvent(str, Enum):
    """Common organic solvents."""

    WATER = "water"
    ETHANOL = "ethanol"
    METHANOL = "methanol"
    DCM = "dichloromethane"
    THF = "tetrahydrofuran"
    DMF = "dimethylformamide"
    DMSO = "dimethylsulfoxide"
    ACETONE = "acetone"
    ACETONITRILE = "acetonitrile"
    TOLUENE = "toluene"
    ETHYL_ACETATE = "ethyl_acetate"
    HEXANE = "hexane"
    DIETHYL_ETHER = "diethyl_ether"
    OTHER = "other"


class FailureCategory(str, Enum):
    """Categories for reaction failure analysis."""

    STERIC_HINDRANCE = "steric_hindrance"
    ELECTRONIC_EFFECTS = "electronic_effects"
    THERMODYNAMIC_UNFAVORABLE = "thermodynamic_unfavorable"
    KINETIC_BARRIER = "kinetic_barrier"
    SOLVENT_INCOMPATIBILITY = "solvent_incompatibility"
    REGIOSELECTIVITY_CONFLICT = "regioselectivity_conflict"
    CHEMOSELECTIVITY_ISSUE = "chemoselectivity_issue"
    VALENCE_VIOLATION = "valence_violation"
    RING_STRAIN = "ring_strain"
    OTHER = "other"


class VerificationStatus(str, Enum):
    """Status of physical verification."""

    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    SIMULATION_ERROR = "simulation_error"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class ChemicalEntity(BaseModel):
    """A chemical compound with multiple representations."""

    smiles: str = Field(..., description="Canonical SMILES representation")
    inchi: Optional[str] = Field(default=None, description="IUPAC International Chemical Identifier")
    selfies: Optional[str] = Field(default=None, description="SELFIES representation")
    name: Optional[str] = Field(default=None, description="Common or IUPAC name")
    molecular_weight: Optional[float] = Field(default=None, description="Molecular weight in g/mol")
    formula: Optional[str] = Field(default=None, description="Molecular formula")

    @field_validator("smiles")
    @classmethod
    def smiles_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("SMILES string cannot be empty")
        return v.strip()


class ReactionConditions(BaseModel):
    """Reaction conditions for a chemical transformation."""

    temperature_celsius: Optional[float] = Field(default=None, description="Temperature in Celsius")
    pressure_atm: Optional[float] = Field(default=None, description="Pressure in atmospheres")
    solvent: Optional[Solvent] = Field(default=None, description="Reaction solvent")
    catalyst: Optional[str] = Field(default=None, description="Catalyst name or SMILES")
    time_hours: Optional[float] = Field(default=None, description="Reaction time in hours")
    ph: Optional[float] = Field(default=None, description="pH if aqueous")
    atmosphere: Optional[str] = Field(default=None, description="Inert atmosphere (N2, Ar, etc.)")


class ReactionHypothesis(BaseModel):
    """A generated hypothesis for a chemical reaction."""

    id: UUID = Field(default_factory=uuid4, description="Unique hypothesis identifier")
    session_id: UUID = Field(..., description="Pipeline session identifier")
    reactants: list[ChemicalEntity] = Field(..., min_length=1, description="Reactant molecules")
    products: list[ChemicalEntity] = Field(..., min_length=1, description="Product molecules")
    reaction_type: ReactionType = Field(default=ReactionType.OTHER)
    conditions: ReactionConditions = Field(default_factory=ReactionConditions)
    reaction_smarts: Optional[str] = Field(default=None, description="Reaction SMARTS pattern")
    yield_estimate: Optional[float] = Field(default=None, description="Estimated yield (0-100%)")
    mechanism_steps: Optional[str] = Field(default=None, description="Proposed mechanism description")
    rationale: Optional[str] = Field(default=None, description="Why this reaction should work")
    prompt_used: Optional[str] = Field(default=None, description="LLM prompt that generated this")
    generation_temperature: Optional[float] = Field(default=None, description="LLM temperature used")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("yield_estimate")
    @classmethod
    def yield_in_range(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0 <= v <= 100):
            raise ValueError("yield_estimate must be between 0 and 100")
        return v


class ComputedProperties(BaseModel):
    """Physically computed molecular and reaction properties."""

    total_energy_hartree: Optional[float] = Field(default=None)
    reaction_energy_kcal_mol: Optional[float] = Field(default=None)
    activation_barrier_kcal_mol: Optional[float] = Field(default=None)
    dipole_moment_debye: Optional[float] = Field(default=None)
    homo_ev: Optional[float] = Field(default=None)
    lumo_ev: Optional[float] = Field(default=None)
    gap_ev: Optional[float] = Field(default=None)
    sa_score: Optional[float] = Field(default=None, description="Synthetic Accessibility score")
    qed: Optional[float] = Field(default=None, description="Quantitative Estimate of Drug-likeness")
    logp: Optional[float] = Field(default=None, description="Octanol-water partition coefficient")
    tpsa: Optional[float] = Field(default=None, description="Topological Polar Surface Area")
    molecular_weight: Optional[float] = Field(default=None, description="Molecular weight (g/mol)")
    num_rotatable_bonds: Optional[int] = Field(default=None)
    num_h_acceptors: Optional[int] = Field(default=None)
    num_h_donors: Optional[int] = Field(default=None)
    steric_score: Optional[float] = Field(default=None, description="0-1 steric hindrance score")


class VerificationResult(BaseModel):
    """Result of physical verification for a reaction hypothesis."""

    id: UUID = Field(default_factory=uuid4)
    hypothesis_id: UUID = Field(..., description="Reference to the hypothesis")
    status: VerificationStatus = Field(default=VerificationStatus.PENDING)
    smiles_valid: bool = Field(default=False, description="SMILES passed RDKit parsing")
    valence_valid: bool = Field(default=False, description="All valences satisfied")
    steric_valid: bool = Field(default=False, description="No steric clashes detected")
    energy_valid: bool = Field(default=False, description="Reaction energy within threshold")
    computed_properties: ComputedProperties = Field(default_factory=ComputedProperties)
    errors: list[str] = Field(default_factory=list, description="Verification error messages")
    warnings: list[str] = Field(default_factory=list, description="Verification warnings")
    xtb_output: Optional[str] = Field(default=None, description="Raw xTB output")
    verified_at: Optional[datetime] = Field(default=None)
    computation_time_seconds: Optional[float] = Field(default=None)


class ReflectionTrace(BaseModel):
    """Causal reasoning trace explaining why a reaction failed."""

    id: UUID = Field(default_factory=uuid4)
    hypothesis_id: UUID = Field(..., description="Reference to the failed hypothesis")
    verification_result_id: UUID = Field(..., description="Reference to verification data")
    failure_categories: list[FailureCategory] = Field(default_factory=list)
    primary_cause: Optional[str] = Field(default=None, description="Primary failure mechanism")
    causal_explanation: str = Field(..., description="Detailed step-by-step causal analysis")
    chemical_principles: list[str] = Field(default_factory=list, description="Relevant chemical principles")
    fix_suggestion: Optional[str] = Field(default=None, description="How to modify to succeed")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in the reflection")
    prompt_used: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PreferencePair(BaseModel):
    """A DPO/RLHF preference pair (chosen vs rejected)."""

    id: UUID = Field(default_factory=uuid4)
    prompt: str = Field(..., description="The instruction/prompt that elicited the responses")
    chosen: str = Field(..., description="The preferred (successful) reaction or CoT trace")
    rejected: str = Field(..., description="The rejected (failed) reaction or CoT trace")
    chosen_hypothesis_id: UUID = Field(..., description="Reference to winning hypothesis")
    rejected_hypothesis_id: UUID = Field(..., description="Reference to losing hypothesis")
    reflection_trace_id: Optional[UUID] = Field(default=None)
    reaction_type: ReactionType = Field(default=ReactionType.OTHER)
    quality_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Pair quality score")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LearningContext(BaseModel):
    """Accumulated knowledge from reflection traces for self-bootstrapping.

    Tracks failure patterns across iterations and generates contextual
    constraints to guide the hypothesis agent toward higher-quality reactions.
    """

    failure_categories: dict[str, int] = Field(
        default_factory=dict, description="Count of failures by category"
    )
    common_mistakes: list[str] = Field(
        default_factory=list, description="Patterns from prior failures to avoid"
    )
    successful_patterns: list[str] = Field(
        default_factory=list, description="Patterns from prior successes to reinforce"
    )
    iteration_count: int = Field(default=0)
    cumulative_hypotheses: int = Field(default=0)
    cumulative_passed: int = Field(default=0)
    cumulative_failed: int = Field(default=0)

    def build_context_prompt(self) -> str:
        """Build a prompt snippet summarizing learned constraints."""
        if not self.common_mistakes and not self.successful_patterns:
            return ""

        parts = ["\n## Learned from Previous Iterations\n"]
        if self.common_mistakes:
            parts.append("**Avoid these patterns (prior failures):**\n")
            for i, mistake in enumerate(self.common_mistakes[-5:], 1):
                parts.append(f"{i}. {mistake}\n")

        if self.successful_patterns:
            parts.append("\n**Prefer these patterns (prior successes):**\n")
            for i, pattern in enumerate(self.successful_patterns[-5:], 1):
                parts.append(f"{i}. {pattern}\n")

        if self.failure_categories:
            parts.append("\n**Failure category distribution:**\n")
            for cat, count in sorted(
                self.failure_categories.items(), key=lambda x: -x[1]
            ):
                parts.append(f"- {cat}: {count} failures\n")

        return "".join(parts)


class PipelineStatus(str, Enum):
    """Overall pipeline execution status."""

    IDLE = "idle"
    GENERATING = "generating"
    VERIFYING = "verifying"
    REFLECTING = "reflecting"
    COMPILING = "compiling"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class SessionState(BaseModel):
    """Persistent state for a pipeline execution session."""

    id: UUID = Field(default_factory=uuid4)
    status: PipelineStatus = Field(default=PipelineStatus.IDLE)
    config_hash: str = Field(..., description="Hash of configuration used")
    hypotheses_generated: int = Field(default=0)
    hypotheses_verified: int = Field(default=0)
    hypotheses_passed: int = Field(default=0)
    hypotheses_failed: int = Field(default=0)
    reflections_generated: int = Field(default=0)
    pairs_compiled: int = Field(default=0)
    current_batch: int = Field(default=0)
    total_batches: int = Field(default=0)
    errors: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = Field(default=None)
    metadata: dict = Field(default_factory=dict)


class AgentMessage(BaseModel):
    """Typed message for inter-agent communication."""

    source: str = Field(..., description="Sending agent name")
    target: str = Field(..., description="Receiving agent name")
    msg_type: str = Field(..., description="Message type identifier")
    payload: dict = Field(default_factory=dict, description="Message payload")
    session_id: UUID = Field(..., description="Correlation ID")
    hypothesis_id: Optional[UUID] = Field(default=None)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    priority: int = Field(default=3, ge=1, le=5, description="1=highest, 5=lowest")
