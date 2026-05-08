"""Ablation study framework for Auto-ChemInstruct.

Rigorously measures the causal contribution of each architectural innovation:
- Self-bootstrapping reflection loop (LearningContext accumulation)
- Temperature scheduling (cosine annealing across iterations)

Implements the paper's ablation methodology:
    Baseline → Bootstrap-only → Temp-schedule-only → Full system
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from src.config import AutoChemConfig, load_config
from src.pipeline.orchestrator import PipelineOrchestrator


@dataclass
class AblationResult:
    """Results from a single ablation variant."""
    variant: str
    config_description: str
    num_hypotheses_target: int
    num_hypotheses_generated: int
    num_passed: int
    num_failed: int
    num_reflections: int
    num_pairs: int
    pass_rate: float
    unique_molecules: int
    unique_scaffolds: int
    scaffold_ratio: float
    tanimoto_diversity: float
    reaction_types: dict[str, int]
    quality_scores: list[float]
    execution_time_seconds: float
    session_id: str
    error: Optional[str] = None


@dataclass
class AblationReport:
    """Comparative analysis across all ablation variants."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    variants: list[AblationResult] = field(default_factory=list)
    num_hypotheses: int = 0
    relative_improvements: dict[str, dict] = field(default_factory=dict)

    def add_variant(self, result: AblationResult) -> None:
        self.variants.append(result)

    def compute_improvements(self) -> dict:
        """Compute relative improvements over baseline for each metric."""
        if not self.variants:
            return {}

        baseline = self.variants[0]
        improvements = {}

        for variant in self.variants[1:]:
            rel = {}
            if baseline.pass_rate > 0:
                rel["pass_rate_delta"] = round(variant.pass_rate - baseline.pass_rate, 4)
            if baseline.tanimoto_diversity > 0:
                rel["diversity_ratio"] = round(variant.tanimoto_diversity / baseline.tanimoto_diversity, 4)
            if baseline.num_pairs > 0:
                rel["pairs_ratio"] = round(variant.num_pairs / baseline.num_pairs, 4)
            if baseline.num_pairs > 0:
                avg_q_baseline = sum(baseline.quality_scores) / len(baseline.quality_scores) if baseline.quality_scores else 0
                avg_q_variant = sum(variant.quality_scores) / len(variant.quality_scores) if variant.quality_scores else 0
                if avg_q_baseline > 0:
                    rel["quality_ratio"] = round(avg_q_variant / avg_q_baseline, 4)
            improvements[variant.variant] = rel

        self.relative_improvements = improvements
        return improvements

    def summary_table(self) -> str:
        """Generate a formatted summary table for the paper."""
        if not self.variants:
            return "No ablation data."

        header = f"{'Variant':<25} {'Pass%':>8} {'Pairs':>6} {'Divers':>7} {'Scaf%':>6} {'Quality':>7} {'Time(s)':>7}"
        sep = "-" * len(header)
        rows = [header, sep]

        for v in self.variants:
            avg_q = sum(v.quality_scores) / len(v.quality_scores) if v.quality_scores else 0
            row = (
                f"{v.variant:<25} "
                f"{v.pass_rate*100:>7.1f}% "
                f"{v.num_pairs:>6} "
                f"{v.tanimoto_diversity:>6.3f} "
                f"{v.scaffold_ratio*100:>5.1f}% "
                f"{avg_q:>6.3f} "
                f"{v.execution_time_seconds:>7.0f}"
            )
            rows.append(row)

        if self.relative_improvements:
            rows.append("")
            rows.append("Relative improvements over baseline:")
            for variant_name, metrics in self.relative_improvements.items():
                parts = [f"  {variant_name}:"]
                if "pass_rate_delta" in metrics:
                    parts.append(f"pass_rate {metrics['pass_rate_delta']:+.1%}")
                if "diversity_ratio" in metrics:
                    parts.append(f"diversity {metrics['diversity_ratio']:.2f}x")
                if "pairs_ratio" in metrics:
                    parts.append(f"pairs {metrics['pairs_ratio']:.2f}x")
                if "quality_ratio" in metrics:
                    parts.append(f"quality {metrics['quality_ratio']:.2f}x")
                rows.append(", ".join(parts))

        return "\n".join(rows)

    def to_dict(self) -> dict:
        self.compute_improvements()
        return {
            "timestamp": self.timestamp,
            "num_hypotheses": self.num_hypotheses,
            "variants": [
                {
                    "variant": v.variant,
                    "config_description": v.config_description,
                    "num_generated": v.num_hypotheses_generated,
                    "passed": v.num_passed,
                    "failed": v.num_failed,
                    "reflections": v.num_reflections,
                    "pairs": v.num_pairs,
                    "pass_rate": v.pass_rate,
                    "unique_molecules": v.unique_molecules,
                    "unique_scaffolds": v.unique_scaffolds,
                    "scaffold_ratio": v.scaffold_ratio,
                    "tanimoto_diversity": v.tanimoto_diversity,
                    "reaction_types": v.reaction_types,
                    "avg_quality": sum(v.quality_scores) / len(v.quality_scores) if v.quality_scores else 0,
                    "execution_time_s": v.execution_time_seconds,
                    "session_id": v.session_id,
                    "error": v.error,
                }
                for v in self.variants
            ],
            "relative_improvements": self.relative_improvements,
        }


def _compute_result_metrics(result: dict) -> dict:
    """Extract metrics from a pipeline result dict."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from rdkit.Chem.Scaffolds import MurckoScaffold

    hypotheses = result.get("hypotheses", [])
    verification_results = result.get("verification_results", [])
    compilation = result.get("compilation", {})
    pairs = compilation.get("train", [])
    summary = result.get("summary", {})

    unique_smiles = set()
    for h in hypotheses:
        for entity in h.reactants + h.products:
            s = entity.smiles if hasattr(entity, "smiles") else str(entity)
            s = s.strip()
            if s and len(s) > 1:
                unique_smiles.add(s)

    fps = []
    scaffolds = set()
    for s in unique_smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps.append(fp)
            try:
                scaff = MurckoScaffold.GetScaffoldForMol(mol)
                if scaff and scaff.GetNumAtoms() > 0:
                    scaffolds.add(Chem.MolToSmiles(scaff))
            except Exception:
                pass

    diversity = 0.0
    if len(fps) > 1:
        sims = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
        diversity = 1.0 - (sum(sims) / len(sims))

    rtypes = Counter()
    quality_scores = []
    for pair in pairs:
        rt = getattr(pair, "reaction_type", None)
        rtypes[str(rt) if rt else "unknown"] += 1
        qs = getattr(pair, "quality_score", 0.0)
        quality_scores.append(float(qs) if qs else 0.0)

    return {
        "unique_molecules": len(fps),
        "unique_scaffolds": len(scaffolds),
        "scaffold_ratio": len(scaffolds) / max(1, len(fps)),
        "tanimoto_diversity": round(diversity, 4),
        "reaction_types": dict(rtypes),
        "quality_scores": quality_scores,
    }


def run_ablation_variant(
    config: AutoChemConfig,
    variant_name: str,
    num_hypotheses: int,
    bootstrap_iterations: int = 1,
    temperature_schedule: str | None = None,
    disable_rag: bool = False,
    disable_reflection: bool = False,
    description: str = "",
) -> AblationResult:
    """Run a single ablation variant.

    Args:
        config: Base configuration.
        variant_name: Label for this variant (e.g., "Baseline").
        num_hypotheses: Number of hypotheses to generate.
        bootstrap_iterations: Bootstrap iterations (1 = off).
        temperature_schedule: Schedule type, or None for fixed temp.
        description: Human-readable description of this variant.
    """
    import copy

    cfg = copy.deepcopy(config)
    if temperature_schedule is not None:
        cfg.pipeline.temperature_schedule = temperature_schedule
    cfg.pipeline.bootstrap_iterations = bootstrap_iterations
    if disable_rag:
        cfg.rag.enabled = False
    if disable_reflection:
        cfg.pipeline.bootstrap_iterations = 1  # No bootstrap = no reflection

    logger.info(
        "Ablation variant '{}': bootstrap={}, schedule={}, rag={}, reflection={}",
        variant_name,
        bootstrap_iterations,
        temperature_schedule or "fixed",
        not disable_rag,
        not disable_reflection,
    )

    start = time.time()

    try:
        orch = PipelineOrchestrator(cfg)
        result = orch.run_pipeline(
            num_hypotheses=num_hypotheses,
            bootstrap_iterations=bootstrap_iterations,
            skip_reflection=disable_reflection,
        )

        metrics = _compute_result_metrics(result)
        summary = result.get("summary", {})

        return AblationResult(
            variant=variant_name,
            config_description=description,
            num_hypotheses_target=num_hypotheses,
            num_hypotheses_generated=summary.get("hypotheses_generated", 0),
            num_passed=summary.get("hypotheses_passed", 0),
            num_failed=summary.get("hypotheses_failed", 0),
            num_reflections=summary.get("reflections_generated", 0),
            num_pairs=summary.get("pairs_compiled", 0),
            pass_rate=(
                summary["hypotheses_passed"] / max(1, summary["hypotheses_generated"])
                if summary.get("hypotheses_generated", 0) > 0
                else 0.0
            ),
            unique_molecules=metrics["unique_molecules"],
            unique_scaffolds=metrics["unique_scaffolds"],
            scaffold_ratio=metrics["scaffold_ratio"],
            tanimoto_diversity=metrics["tanimoto_diversity"],
            reaction_types=metrics["reaction_types"],
            quality_scores=metrics["quality_scores"],
            execution_time_seconds=round(time.time() - start, 1),
            session_id=str(result.get("session_id", "unknown")),
        )

    except Exception as e:
        logger.error("Ablation variant '{}' failed: {}", variant_name, e)
        return AblationResult(
            variant=variant_name,
            config_description=description,
            num_hypotheses_target=num_hypotheses,
            num_hypotheses_generated=0,
            num_passed=0,
            num_failed=0,
            num_reflections=0,
            num_pairs=0,
            pass_rate=0.0,
            unique_molecules=0,
            unique_scaffolds=0,
            scaffold_ratio=0.0,
            tanimoto_diversity=0.0,
            reaction_types={},
            quality_scores=[],
            execution_time_seconds=round(time.time() - start, 1),
            session_id="error",
            error=str(e),
        )


def run_full_ablation(
    config: AutoChemConfig,
    num_hypotheses: int = 5,
    output_dir: str | None = None,
) -> AblationReport:
    """Run all ablation variants and compile comparative report.

    Variants (NeurIPS-style component ablation):
        1. Full-System: bootstrap + temperature + RAG + reflection
        2. No-Bootstrap: fixed temp, no learning feedback loop
        3. No-Reflection: no causal traces, verify only (pass/fail)
        4. No-RAG: full pipeline without retrieval augmentation
    """
    report = AblationReport(num_hypotheses=num_hypotheses)
    variants = [
        ("Full-System", 3, "cosine", False, False,
         "3 bootstrap iterations + cosine annealing + RAG + reflection"),
        ("No-Bootstrap", 1, None, False, False,
         "No bootstrap, fixed temperature=0.8, RAG + reflection active"),
        ("No-Reflection", 3, "cosine", False, True,
         "Bootstrap + cosine, RAG active, no causal reflection"),
        ("No-RAG", 3, "cosine", True, False,
         "Bootstrap + cosine + reflection, RAG disabled"),
    ]

    logger.info(
        "Starting ablation study: {} variants, {} hypotheses each",
        len(variants),
        num_hypotheses,
    )

    for variant_name, bootstrap_iters, temp_schedule, no_rag, no_refl, desc in variants:
        logger.info("--- Ablation: {} ---", variant_name)
        result = run_ablation_variant(
            config=config,
            variant_name=variant_name,
            num_hypotheses=num_hypotheses,
            bootstrap_iterations=bootstrap_iters,
            temperature_schedule=temp_schedule,
            disable_rag=no_rag,
            disable_reflection=no_refl,
            description=desc,
        )
        report.add_variant(result)

        if result.error:
            logger.warning("Variant '{}' failed: {}", variant_name, result.error)

    report.compute_improvements()

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        report_file = output_path / "ablation_report.json"
        with open(report_file, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info("Ablation report saved to {}", report_file)

        summary_file = output_path / "ablation_summary.txt"
        with open(summary_file, "w") as f:
            f.write(report.summary_table())
            f.write("\n")
        logger.info("Ablation summary saved to {}", summary_file)

    return report
