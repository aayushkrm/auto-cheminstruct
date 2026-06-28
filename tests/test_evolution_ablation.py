"""Tests for evolution ablation framework (7-variant comparison)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.benchmarks.evolution_ablation import (
    VARIANTS,
    EvolutionAblationReport,
    EvolutionAblationResult,
    _run_evolution_variant,
    run_evolution_ablation,
)
from src.config import load_config


class TestEvolutionAblationResult:
    def test_to_dict(self):
        result = EvolutionAblationResult(
            variant="Test",
            description="desc",
            map_elites_enabled=True,
            carl_enabled=False,
            bootstrap_enabled=True,
            rag_enabled=True,
            reflection_enabled=True,
            total_elites=42,
            grid_coverage=0.015,
            generation_count=3,
        )
        d = result.to_dict()
        assert d["variant"] == "Test"
        assert d["map_elites"] is True
        assert d["carl"] is False
        assert d["total_elites"] == 42
        assert d["grid_coverage"] == 0.015

    def test_error_result(self):
        result = EvolutionAblationResult(
            variant="ErrorVariant",
            description="broken",
            map_elites_enabled=True,
            carl_enabled=True,
            bootstrap_enabled=False,
            rag_enabled=False,
            reflection_enabled=False,
            error="Something went wrong",
        )
        d = result.to_dict()
        assert d["error"] == "Something went wrong"
        assert d["total_elites"] == 0


class TestEvolutionAblationReport:
    def test_add_and_summarize(self):
        report = EvolutionAblationReport()
        report.add_variant(
            EvolutionAblationResult(
                variant="Baseline",
                description="b",
                map_elites_enabled=False,
                carl_enabled=False,
                bootstrap_enabled=False,
                rag_enabled=True,
                reflection_enabled=False,
                total_elites=10,
                grid_coverage=0.004,
                pass_rate_simulated=0.65,
            )
        )
        report.add_variant(
            EvolutionAblationResult(
                variant="Full-System",
                description="fs",
                map_elites_enabled=True,
                carl_enabled=True,
                bootstrap_enabled=True,
                rag_enabled=True,
                reflection_enabled=True,
                total_elites=50,
                grid_coverage=0.019,
                pass_rate_simulated=0.85,
            )
        )
        report.compute_improvements()
        assert "Full-System" in report.relative_improvements
        assert report.relative_improvements["Full-System"]["elites_ratio"] == 5.0
        assert report.relative_improvements["Full-System"]["pass_rate_ratio"] == pytest.approx(
            0.85 / 0.65, rel=0.01
        )

    def test_summary_table_output(self):
        report = EvolutionAblationReport()
        report.add_variant(
            EvolutionAblationResult(
                variant="X",
                description="x",
                map_elites_enabled=False,
                carl_enabled=False,
                bootstrap_enabled=False,
                rag_enabled=False,
                reflection_enabled=False,
                total_elites=5,
                grid_coverage=0.002,
            )
        )
        table = report.summary_table()
        assert "X" in table
        assert "ME" in table

    def test_to_dict(self):
        report = EvolutionAblationReport()
        report.add_variant(
            EvolutionAblationResult(
                variant="A",
                description="a",
                map_elites_enabled=False,
                carl_enabled=False,
                bootstrap_enabled=False,
                rag_enabled=False,
                reflection_enabled=False,
                total_elites=1,
            )
        )
        d = report.to_dict()
        assert "timestamp" in d
        assert len(d["variants"]) == 1


class TestEvolutionAblationRunner:
    def test_all_seven_variants_run(self):
        cfg = load_config()
        report = run_evolution_ablation(cfg, generations=2)
        assert len(report.variants) == 7
        for v in report.variants:
            assert v.error is None, f"Variant {v.variant} failed: {v.error}"
            assert v.execution_time_seconds >= 0

    def test_report_to_json(self):
        cfg = load_config()
        report = run_evolution_ablation(cfg, generations=2)
        d = report.to_dict()
        assert len(d["variants"]) == 7
        assert len(d["relative_improvements"]) == 6
        full_system = next(v for v in d["variants"] if v["variant"] == "Full-System")
        assert full_system["map_elites"] is True
        assert full_system["carl"] is True

    def test_variants_config_readonly(self):
        assert len(VARIANTS) == 7
        names = [v[0] for v in VARIANTS]
        assert "Baseline" in names
        assert "Full-System" in names
        assert "MAP-Elites-Only" in names
        assert "CARL-Only" in names

    def test_output_to_file(self):
        cfg = load_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_evolution_ablation(cfg, generations=2, output_dir=tmpdir)
            report_path = Path(tmpdir) / "evolution_ablation.json"
            assert report_path.exists()
            data = json.loads(report_path.read_text())
            assert len(data["variants"]) == 7

    def test_map_elites_variant_has_elites(self):
        result = _run_evolution_variant(
            variant_name="ME-Test",
            description="",
            map_elites=True,
            carl=False,
            bootstrap=False,
            rag=True,
            reflection=False,
            generations=3,
            mutants_per_gen=5,
            seed=42,
        )
        assert result.total_elites > 0
        assert result.grid_coverage > 0
        assert result.map_elites_enabled is True

    def test_full_system_outperforms_baseline(self):
        cfg = load_config()
        report = run_evolution_ablation(cfg, generations=3)
        baseline = next(v for v in report.variants if v.variant == "Baseline")
        full = next(v for v in report.variants if v.variant == "Full-System")
        assert full.total_elites > baseline.total_elites
        assert full.quality_estimate > baseline.quality_estimate
