"""Tests for async DAG execution engine, stage wrappers, and linear parity validation."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from src.evolution.dag import (
    DAGPipeline,
    DAGStage,
    StageResult,
    StageStatus,
    linear_parity_check,
)


def _identity(x: Any) -> Any:
    return x


def _double(x: int) -> int:
    return x * 2


def _triple(x: int) -> int:
    return x * 3


def _sum_deps(deps: dict[str, int]) -> int:
    return sum(deps.values())


def _slow_fn(x: Any) -> Any:
    time.sleep(2.0)
    return x


def _fail_fn(x: Any) -> None:
    raise ValueError("boom")


# ── DAGPipeline tests ──


class TestDAGPipeline:
    def test_single_stage(self):
        pipeline = DAGPipeline(stages=[DAGStage(name="double", fn=_double)])
        results = asyncio.run(pipeline.run(21))
        assert results["double"].ok
        assert results["double"].output == 42

    def test_linear_chain(self):
        pipeline = DAGPipeline(
            stages=[
                DAGStage(name="step1", fn=lambda x: x + 1),
                DAGStage(name="step2", fn=lambda x: x * 10, depends_on=["step1"]),
                DAGStage(name="step3", fn=lambda x: x + 7, depends_on=["step2"]),
            ]
        )
        results = asyncio.run(pipeline.run(5))
        assert results["step1"].output == 6
        assert results["step2"].output == 60
        assert results["step3"].output == 67
        assert all(r.ok for r in results.values())

    def test_parallel_fan_out(self):
        pipeline = DAGPipeline(
            stages=[
                DAGStage(name="source", fn=_identity),
                DAGStage(name="branch_a", fn=lambda x: x * 2, depends_on=["source"]),
                DAGStage(name="branch_b", fn=lambda x: x * 3, depends_on=["source"]),
            ]
        )
        results = asyncio.run(pipeline.run(10))
        assert results["branch_a"].output == 20
        assert results["branch_b"].output == 30

    def test_diamond_merge(self):
        pipeline = DAGPipeline(
            stages=[
                DAGStage(name="a", fn=lambda x: x * 2),
                DAGStage(name="b", fn=lambda x: x * 3, depends_on=["a"]),
                DAGStage(name="c", fn=lambda x: x * 4, depends_on=["a"]),
                DAGStage(name="d", fn=_sum_deps, depends_on=["b", "c"]),
            ]
        )
        results = asyncio.run(pipeline.run(5))
        assert results["a"].output == 10
        assert results["b"].output == 30
        assert results["c"].output == 40
        assert results["d"].output == 70

    def test_failure_propagation(self):
        pipeline = DAGPipeline(
            stages=[
                DAGStage(name="good", fn=_double),
                DAGStage(name="bad", fn=_fail_fn),
                DAGStage(name="depends_on_bad", fn=_identity, depends_on=["bad"]),
            ]
        )
        results = asyncio.run(pipeline.run(5))
        assert results["good"].ok
        assert results["bad"].status == StageStatus.FAILED
        assert results["depends_on_bad"].status == StageStatus.SKIPPED
        assert "boom" in (results["bad"].error or "")

    def test_optional_stage_skipped_on_dep_failure(self):
        pipeline = DAGPipeline(
            stages=[
                DAGStage(name="required", fn=_fail_fn),
                DAGStage(
                    name="optional_downstream",
                    fn=_identity,
                    depends_on=["required"],
                    optional=True,
                ),
            ]
        )
        results = asyncio.run(pipeline.run(5))
        assert results["required"].status == StageStatus.FAILED
        # Optional stage still runs when dep fails
        assert results["optional_downstream"].status in (
            StageStatus.COMPLETED,
            StageStatus.SKIPPED,
        )

    def test_timeout_triggers(self):
        pipeline = DAGPipeline(
            stages=[
                DAGStage(name="slow", fn=_slow_fn, timeout_seconds=0.1),
            ]
        )
        results = asyncio.run(pipeline.run(42))
        assert results["slow"].status == StageStatus.FAILED
        assert "timed out" in (results["slow"].error or "").lower()

    def test_empty_pipeline(self):
        pipeline = DAGPipeline(stages=[])
        results = asyncio.run(pipeline.run(None))
        assert results == {}

    def test_cycle_detection(self):
        with pytest.raises(ValueError, match="cycle"):
            DAGPipeline(
                stages=[
                    DAGStage(name="x", fn=_identity, depends_on=["y"]),
                    DAGStage(name="y", fn=_identity, depends_on=["x"]),
                ]
            )

    def test_duplicate_name_detection(self):
        with pytest.raises(ValueError, match="Duplicate"):
            DAGPipeline(
                stages=[
                    DAGStage(name="dup", fn=_identity),
                    DAGStage(name="dup", fn=_double),
                ]
            )

    def test_unknown_dependency_detection(self):
        with pytest.raises(ValueError, match="unknown stage"):
            DAGPipeline(
                stages=[
                    DAGStage(name="a", fn=_identity, depends_on=["nonexistent"]),
                ]
            )

    def test_stage_result_timing(self):
        pipeline = DAGPipeline(stages=[DAGStage(name="fast", fn=_identity)])
        results = asyncio.run(pipeline.run(1))
        assert results["fast"].duration_ms >= 0
        assert results["fast"].started_at > 0
        assert results["fast"].completed_at >= results["fast"].started_at

    def test_parallel_execution_is_faster(self):
        slow_stage = DAGStage(name="slow_a", fn=lambda x: (time.sleep(0.15), x * 2)[1])
        pipeline = DAGPipeline(
            stages=[
                slow_stage,
                DAGStage(name="slow_b", fn=lambda x: (time.sleep(0.15), x * 3)[1]),
                DAGStage(name="slow_c", fn=lambda x: (time.sleep(0.15), x * 4)[1]),
            ]
        )
        start = time.monotonic()
        results = asyncio.run(pipeline.run(1))
        elapsed = time.monotonic() - start
        assert elapsed < 0.35  # parallel should be << 0.45 sequential
        assert all(r.ok for r in results.values())

    def test_stage_result_ok_false_on_failure(self):
        pipeline = DAGPipeline(stages=[DAGStage(name="fail", fn=_fail_fn)])
        results = asyncio.run(pipeline.run(1))
        assert not results["fail"].ok

    def test_max_parallel_limit(self):
        import threading

        counter = 0
        lock = threading.Lock()
        peak = 0

        def _tracked(x: int) -> int:
            nonlocal counter, peak
            with lock:
                counter += 1
                peak = max(peak, counter)
            time.sleep(0.05)
            with lock:
                counter -= 1
            return x

        pipeline = DAGPipeline(
            stages=[DAGStage(name=f"s{i}", fn=_tracked) for i in range(8)],
            max_parallel=3,
        )
        asyncio.run(pipeline.run(1))
        assert 1 <= peak <= 3  # bounded by max_parallel semaphore


# ── Linear parity check tests ──


class TestLinearParityCheck:
    def test_match_trivial(self):
        pipeline = DAGPipeline(stages=[DAGStage(name="double", fn=_double)])
        result = linear_parity_check(pipeline, 21, _double)
        assert result["match"] is True
        assert result["dag_stages_total"] == 1
        assert result["dag_stages_passed"] == 1

    def test_match_chain(self):
        pipeline = DAGPipeline(
            stages=[
                DAGStage(name="step1", fn=lambda x: x + 1),
                DAGStage(name="step2", fn=lambda x: x * 10, depends_on=["step1"]),
            ]
        )

        def sequential(x: int) -> int:
            return (x + 1) * 10

        result = linear_parity_check(pipeline, 5, sequential)
        assert result["match"] is True

    def test_mismatch_detected(self):
        pipeline = DAGPipeline(stages=[DAGStage(name="doubled", fn=_double)])

        def sequential(x: int) -> int:
            return x * 3  # tripling, not doubling

        result = linear_parity_check(pipeline, 10, sequential)
        assert result["match"] is False
        assert "mismatches" in result["diff_report"]

    def test_failed_stages_reported(self):
        pipeline = DAGPipeline(
            stages=[
                DAGStage(name="good", fn=_identity),
                DAGStage(name="bad", fn=lambda x: (_fail_fn(x), x)[1]),
            ]
        )

        def sequential(x: int) -> int:
            return x

        result = linear_parity_check(pipeline, 7, sequential)
        assert result["dag_stages_failed"] == 1
        assert "bad" in result["failed_stage_names"]

    def test_dict_parity(self):
        pipeline = DAGPipeline(
            stages=[
                DAGStage(name="enrich", fn=lambda d: {**d, "y": d["x"] * 2}),
            ]
        )

        def sequential(d: dict) -> dict:
            return {"x": d["x"], "y": d["x"] * 2}

        result = linear_parity_check(pipeline, {"x": 10}, sequential)
        assert result["match"] is True


# ── StageResult tests ──


class TestStageResult:
    def test_defaults(self):
        sr = StageResult(stage_name="test")
        assert sr.stage_name == "test"
        assert sr.status == StageStatus.PENDING
        assert sr.output is None
        assert sr.error is None
        assert not sr.ok

    def test_completed_is_ok(self):
        sr = StageResult(stage_name="done", status=StageStatus.COMPLETED, output=42)
        assert sr.ok
        assert sr.output == 42

    def test_failed_is_not_ok(self):
        sr = StageResult(stage_name="fail", status=StageStatus.FAILED, error="err")
        assert not sr.ok
        assert sr.error == "err"
