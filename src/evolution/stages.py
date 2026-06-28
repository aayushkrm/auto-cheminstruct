"""Type-safe stage wrappers bridging existing agent interfaces to DAGPipeline.

Each wrapper defines explicit input/output types, timeout policies,
retry semantics, and optional flagging.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from src.agents.compilation_agent import CompilationAgent
from src.agents.hypothesis_agent import HypothesisGenerationAgent
from src.agents.reflection_agent import ReflectionAgent
from src.agents.verification_agent import VerificationAgent
from src.carl.chain import CARLReflectionAgent
from src.config import (
    CompilationAgentConfig,
    HypothesisAgentConfig,
    ReflectionAgentConfig,
    VerificationAgentConfig,
)
from src.data.models import (
    PreferencePair,
    ReactionHypothesis,
    ReflectionTrace,
    VerificationResult,
)

# ── Stage input/output models ──


class GenerateInput(BaseModel):
    session_id: UUID
    num_hypotheses: int = 50
    seed_prompt: str | None = None
    learning_context: str | None = None
    temperature: float | None = None


class GenerateOutput(BaseModel):
    hypotheses: list[ReactionHypothesis]
    count: int


class VerifyInput(BaseModel):
    hypotheses: list[ReactionHypothesis]


class VerifyOutput(BaseModel):
    results: list[ReactionHypothesis]
    verification_results: list[VerificationResult]
    passed: int
    failed: int


class ReflectInput(BaseModel):
    hypotheses: list[ReactionHypothesis]
    verification_results: list[VerificationResult]


class ReflectOutput(BaseModel):
    traces: list[ReflectionTrace]
    count: int
    learning_context_prompt: str | None = None


class CompileInput(BaseModel):
    hypotheses: list[ReactionHypothesis]
    verification_results: list[VerificationResult]
    reflection_traces: list[ReflectionTrace]


class CompileOutput(BaseModel):
    pairs: list[PreferencePair]
    train_count: int
    val_count: int
    test_count: int
    metadata: dict = Field(default_factory=dict)


# ── Stage functions ──


def make_generate_fn(
    llm: BaseChatModel,
    config: HypothesisAgentConfig,
) -> callable:
    agent = HypothesisGenerationAgent(
        llm=llm,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        num_generations_per_prompt=config.num_generations_per_prompt,
    )

    def fn(inp: GenerateInput) -> GenerateOutput:
        hypotheses = agent.generate(
            session_id=inp.session_id,
            num_hypotheses=inp.num_hypotheses,
            seed_prompt=inp.seed_prompt,
            learning_context=inp.learning_context,
        )
        return GenerateOutput(hypotheses=hypotheses, count=len(hypotheses))

    return fn


def make_verify_fn(config: VerificationAgentConfig) -> callable:
    agent = VerificationAgent(
        enable_xtb=config.enable_xtb,
        xtb_method=config.xtb_method,
        xtb_timeout=config.xtb_timeout,
        xtb_max_atoms=config.xtb_max_atoms,
        energy_barrier_threshold_kcal=config.energy_barrier_threshold,
        sa_score_min=config.sa_score_min,
        sa_score_max=config.sa_score_max,
        qed_min=config.qed_min,
    )

    def fn(inp: VerifyInput) -> VerifyOutput:
        results = agent.verify_batch(inp.hypotheses)
        passed = sum(1 for r in results if r.status.value == "PASSED")
        failed = sum(1 for r in results if r.status.value == "FAILED")
        combined = [
            h.model_copy(update={"verification": r})
            for h, r in zip(inp.hypotheses, results, strict=False)
        ]
        return VerifyOutput(
            results=combined,
            verification_results=results,
            passed=passed,
            failed=failed,
        )

    return fn


def make_reflect_fn(
    llm: BaseChatModel,
    config: ReflectionAgentConfig,
    use_carl: bool = False,
) -> callable:
    carl_agent = CARLReflectionAgent(enabled=True) if use_carl else None

    agent = ReflectionAgent(
        llm=llm,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    def fn(inp: ReflectInput) -> ReflectOutput:
        failed_h = [
            h
            for h, r in zip(inp.hypotheses, inp.verification_results, strict=False)
            if r.status.value == "FAILED"
        ]
        failed_r = [r for r in inp.verification_results if r.status.value == "FAILED"]

        if not failed_h:
            return ReflectOutput(traces=[], count=0)

        if carl_agent:
            carl_results = carl_agent.reflect_batch(failed_h, failed_r)
            traces = _carl_to_reflection_traces(carl_results)
        else:
            traces = agent.reflect_batch(failed_h, failed_r)

        return ReflectOutput(traces=traces, count=len(traces))

    return fn


def _carl_to_reflection_traces(
    carl_results: list,
) -> list[ReflectionTrace]:
    """Convert CARLResult objects to ReflectionTrace objects.

    Merges the 4-step CARL analysis into the existing ReflectionTrace schema.
    """
    traces: list[ReflectionTrace] = []
    for cr in carl_results:
        from src.data.models import FailureCategory, ReflectionTrace

        synthesis = getattr(cr, "synthesis", None)
        if synthesis is None:
            continue

        failure_cats: list[FailureCategory] = []
        for cat_name in synthesis.failure_categories or []:
            try:
                failure_cats.append(FailureCategory(cat_name))
            except ValueError:
                failure_cats.append(FailureCategory.OTHER)

        trace = ReflectionTrace(
            hypothesis_id=UUID(cr.hypothesis_id) if cr.hypothesis_id != "unknown" else None,
            verification_result_id=None,
            failure_categories=failure_cats or [FailureCategory.OTHER],
            primary_cause=synthesis.primary_cause,
            causal_explanation=synthesis.causal_explanation,
            chemical_principles=synthesis.chemical_principles or [],
            fix_suggestion=synthesis.fix_suggestion,
            confidence=synthesis.confidence,
            prompt_used="carl-4-step-chain",
            created_at=datetime.now(),
        )
        traces.append(trace)

    return traces


def make_compile_fn(config: CompilationAgentConfig) -> callable:
    agent = CompilationAgent(
        output_format=config.output_format,
        train_split=config.train_split,
        val_split=config.val_split,
        test_split=config.test_split,
        min_pairs_per_reaction_type=config.min_pairs_per_reaction_type,
        deduplicate=config.deduplicate,
    )

    def fn(inp: CompileInput) -> CompileOutput:
        compilation = agent.compile(
            inp.hypotheses,
            inp.verification_results,
            inp.reflection_traces,
        )
        return CompileOutput(
            pairs=compilation["train"] + compilation["val"] + compilation["test"],
            train_count=len(compilation["train"]),
            val_count=len(compilation["val"]),
            test_count=len(compilation["test"]),
            metadata=compilation["metadata"],
        )

    return fn
