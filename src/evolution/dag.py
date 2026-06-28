"""Async DAG execution engine for typed pipeline stages.

Lightweight topological executor with futures, error propagation,
and timeout support. No external DAG framework dependency.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

from loguru import logger

T_IN = TypeVar("T_IN")
T_OUT = TypeVar("T_OUT")


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult(Generic[T_OUT]):
    stage_name: str
    status: StageStatus = StageStatus.PENDING
    output: T_OUT | None = None
    error: str | None = None
    duration_ms: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == StageStatus.COMPLETED


@dataclass
class DAGStage(Generic[T_IN, T_OUT]):
    name: str
    fn: Callable[[T_IN], T_OUT]
    timeout_seconds: float = 300.0
    retry_count: int = 0
    depends_on: list[str] = field(default_factory=list)
    optional: bool = False


@dataclass
class DAGPipeline:
    """Topologically-sorted async pipeline executor.

    Usage:
        pipeline = DAGPipeline(stages=[stage_a, stage_b, stage_c])
        result = await pipeline.run(initial_input)
    """

    stages: list[DAGStage]
    max_parallel: int = 4

    def __post_init__(self) -> None:
        self._validate_graph()
        self._sorted_stages = self._topological_sort()

    def _validate_graph(self) -> None:
        names = {s.name for s in self.stages}
        if len(names) != len(self.stages):
            seen: set[str] = set()
            for s in self.stages:
                if s.name in seen:
                    raise ValueError(f"Duplicate stage name: {s.name}")
                seen.add(s.name)
        for s in self.stages:
            for dep in s.depends_on:
                if dep not in names:
                    raise ValueError(f"Stage '{s.name}' depends on unknown stage '{dep}'")

    def _topological_sort(self) -> list[DAGStage]:
        """Topological sort (Kahn's algorithm)."""
        in_degree: dict[str, int] = {}
        name_to_stage: dict[str, DAGStage] = {}
        children: dict[str, list[str]] = {}

        for s in self.stages:
            name_to_stage[s.name] = s
            in_degree.setdefault(s.name, 0)
            children.setdefault(s.name, [])

        for s in self.stages:
            for dep in s.depends_on:
                in_degree[s.name] = in_degree.get(s.name, 0) + 1
                children[dep].append(s.name)

        queue = [name for name, deg in in_degree.items() if deg == 0]
        result: list[DAGStage] = []

        while queue:
            name = queue.pop(0)
            result.append(name_to_stage[name])
            for child in children.get(name, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(result) != len(self.stages):
            raise ValueError("DAG has a cycle")

        return result

    async def run(self, initial_input: Any = None) -> dict[str, StageResult]:
        """Execute all stages in topological order with parallel fan-out.

        Args:
            initial_input: Input passed to stages with no dependencies.
                Stages downstream receive the output of their dependency.

        Returns:
            Dict mapping stage name to StageResult.
        """
        results: dict[str, StageResult] = {}
        completed: set[str] = set()
        failed: set[str] = set()

        semaphore = asyncio.Semaphore(self.max_parallel)

        async def _run_stage(stage: DAGStage) -> None:
            async with semaphore:
                result = StageResult(stage_name=stage.name)
                started = time.monotonic()
                result.started_at = started

                try:
                    deps_failed = [d for d in stage.depends_on if d in failed]
                    if deps_failed and not stage.optional:
                        result.status = StageStatus.SKIPPED
                        result.error = f"Dependencies failed: {deps_failed}"
                        results[stage.name] = result
                        return

                    inp = initial_input
                    if stage.depends_on:
                        dep_outputs = {d: results[d].output for d in stage.depends_on}
                        inp = (
                            dep_outputs[stage.depends_on[0]]
                            if len(stage.depends_on) == 1
                            else dep_outputs
                        )

                    result.status = StageStatus.RUNNING
                    output = await asyncio.wait_for(
                        asyncio.to_thread(stage.fn, inp),
                        timeout=stage.timeout_seconds,
                    )
                    result.status = StageStatus.COMPLETED
                    result.output = output

                except TimeoutError:
                    result.status = StageStatus.FAILED
                    result.error = f"Timed out after {stage.timeout_seconds}s"
                    logger.error("Stage '{}' timed out", stage.name)

                except Exception as e:
                    result.status = StageStatus.FAILED
                    result.error = str(e)
                    logger.error("Stage '{}' failed: {}", stage.name, e)

                finally:
                    result.duration_ms = (time.monotonic() - started) * 1000
                    result.completed_at = time.monotonic()
                    results[stage.name] = result
                    if result.status == StageStatus.FAILED:
                        failed.add(stage.name)
                    elif result.status == StageStatus.COMPLETED:
                        completed.add(stage.name)

                logger.debug(
                    "Stage '{}': {} ({:.1f}ms)",
                    stage.name,
                    result.status.value,
                    result.duration_ms,
                )

        pending: list[asyncio.Task] = []
        for stage in self._sorted_stages:
            deps_ready = all(d in completed or d in failed for d in stage.depends_on)
            while not deps_ready:
                await asyncio.sleep(0.01)
                deps_ready = all(d in completed or d in failed for d in stage.depends_on)

            task = asyncio.create_task(_run_stage(stage))
            pending.append(task)

        await asyncio.gather(*pending, return_exceptions=True)
        return results


def linear_parity_check(
    pipeline: DAGPipeline,
    initial_input: Any,
    sequential_fn: Callable[[Any], Any],
    tolerance: float = 1e-6,
) -> dict:
    """Validate DAG output matches sequential execution output.

    Runs the pipeline in DAG mode, then runs the sequential equivalent,
    and compares results.

    Returns dict with: match, dag_stages, sequential_result, diff_report.
    """
    dag_results = asyncio.run(pipeline.run(initial_input))
    sequential_output = sequential_fn(initial_input)

    dag_final = _extract_final_output(dag_results, pipeline)
    match = _compare_outputs(dag_final, sequential_output, tolerance)

    report: dict[str, list[str]] = {"mismatches": [], "dag_extra": [], "seq_extra": []}

    if not match:
        report["mismatches"] = _diff_keys(dag_final, sequential_output)

    failed_stages = [name for name, r in dag_results.items() if r.status == StageStatus.FAILED]

    return {
        "match": match,
        "dag_stages_total": len(dag_results),
        "dag_stages_passed": sum(1 for r in dag_results.values() if r.ok),
        "dag_stages_failed": len(failed_stages),
        "failed_stage_names": failed_stages,
        "dag_final_output": dag_final,
        "sequential_output": sequential_output,
        "diff_report": report,
    }


def _extract_final_output(results: dict[str, StageResult], pipeline: DAGPipeline) -> Any:
    """Find the sink node(s) output."""
    child_names: set[str] = set()
    for s in pipeline.stages:
        for dep in s.depends_on:
            child_names.add(dep)

    parent_names = {s.name for s in pipeline.stages}
    sinks = parent_names - child_names

    if not sinks:
        sinks = {pipeline._sorted_stages[-1].name}

    outputs: dict[str, Any] = {}
    for name in sinks:
        r = results.get(name)
        if r and r.ok and r.output is not None:
            outputs[name] = r.output

    return outputs if len(outputs) > 1 else next(iter(outputs.values()), None)


def _compare_outputs(dag_output: Any, seq_output: Any, tolerance: float) -> bool:
    """Compare two pipeline outputs with tolerance for floats."""
    if type(dag_output) != type(seq_output):
        return False
    if isinstance(dag_output, dict) and isinstance(seq_output, dict):
        if set(dag_output.keys()) != set(seq_output.keys()):
            return False
        return all(_compare_outputs(dag_output[k], seq_output[k], tolerance) for k in dag_output)
    if isinstance(dag_output, list) and isinstance(seq_output, list):
        if len(dag_output) != len(seq_output):
            return False
        return all(
            _compare_outputs(a, b, tolerance) for a, b in zip(dag_output, seq_output, strict=False)
        )
    if isinstance(dag_output, float) and isinstance(seq_output, float):
        return abs(dag_output - seq_output) < tolerance
    return dag_output == seq_output


def _diff_keys(dag_output: Any, seq_output: Any) -> list[str]:
    """Report key-level differences."""
    issues: list[str] = []
    if isinstance(dag_output, dict) and isinstance(seq_output, dict):
        for k in set(dag_output.keys()) & set(seq_output.keys()):
            if dag_output[k] != seq_output[k]:
                issues.append(f"key '{k}': DAG={dag_output[k]}, seq={seq_output[k]}")
        for k in set(dag_output.keys()) - set(seq_output.keys()):
            issues.append(f"key '{k}': only in DAG output")
        for k in set(seq_output.keys()) - set(dag_output.keys()):
            issues.append(f"key '{k}': only in sequential output")
    return issues
