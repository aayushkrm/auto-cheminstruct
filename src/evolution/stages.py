"""Type-safe stage wrappers bridging existing agent interfaces to DAGPipeline.

Each wrapper defines explicit input/output types, timeout policies,
retry semantics, and optional flagging.
"""

from __future__ import annotations

from uuid import UUID

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from src.agents.compilation_agent import CompilationAgent
from src.agents.hypothesis_agent import HypothesisGenerationAgent
from src.agents.reflection_agent import ReflectionAgent
from src.agents.verification_agent import VerificationAgent
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
) -> callable:
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

        traces = agent.reflect_batch(failed_h, failed_r)
        return ReflectOutput(traces=traces, count=len(traces))

    return fn


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
