"""Generate v2.0 dataset with rate-limited MiniMax-M3 via Fireworks."""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.config import load_config
from src.pipeline.orchestrator import PipelineOrchestrator

NUM_HYPOTHESES = 50
BOOTSTRAP = 3
RATE_LIMIT_SLEEP = 8  # seconds between LLM calls


def main():
    cfg = load_config()
    orch = PipelineOrchestrator(cfg)
    orch.start_session()

    print(f"Session: {orch.session.id}")
    print(f"Target: {NUM_HYPOTHESES} hypotheses, {BOOTSTRAP} bootstrap iterations")
    print(f"Rate limit: {RATE_LIMIT_SLEEP}s between calls")

    for boot_iter in range(1, BOOTSTRAP + 1):
        temperature = cfg.temperature_max - (cfg.temperature_max - cfg.temperature_min) * (
            (boot_iter - 1) / max(1, BOOTSTRAP - 1)
        )
        print(f"\n=== Bootstrap {boot_iter}/{BOOTSTRAP} (T={temperature:.3f}) ===")

        result = orch.run_pipeline(
            num_hypotheses=NUM_HYPOTHESES,
            bootstrap_iterations=1,
        )

        summary = result.get("summary", {})
        print(f"  Generated: {summary.get('hypotheses_generated', 0)}")
        print(f"  Passed: {summary.get('hypotheses_passed', 0)}")
        print(f"  Failed: {summary.get('hypotheses_failed', 0)}")
        print(f"  Reflections: {summary.get('reflections_generated', 0)}")
        print(f"  Pairs: {summary.get('pairs_compiled', 0)}")

        time.sleep(5)

    print(f"\n=== Complete ===")
    print(f"Session: {orch.session.id}")
    print(f"Dataset: datasets/autochem-{orch.session.id}/")


if __name__ == "__main__":
    main()
