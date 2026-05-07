"""Chemistry-aware quality scoring for DPO preference pairs.

Provides standalone scoring functions with 6 dimensions:
    - Structural validity
    - Drug-likeness (QED)
    - Reflection depth (causal chain quality)
    - Yield differential
    - Scaffold diversity
    - Reaction type specificity

Also exports batch scoring and QualityReport for paper-ready metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from src.data.models import PreferencePair, ReactionHypothesis, ReflectionTrace


@dataclass
class QualityScores:
    """Per-pair chemistry-aware quality scores (0-1 per dimension)."""

    pair_id: str
    structural_validity: float = 0.0
    drug_likeness: float = 0.0
    reflection_depth: float = 0.0
    yield_differential: float = 0.0
    scaffold_diversity: float = 0.0
    reaction_specificity: float = 0.0
    composite: float = 0.0

    def to_dict(self) -> dict[str, float | str]:
        return {
            "pair_id": self.pair_id,
            "structural_validity": round(self.structural_validity, 4),
            "drug_likeness": round(self.drug_likeness, 4),
            "reflection_depth": round(self.reflection_depth, 4),
            "yield_differential": round(self.yield_differential, 4),
            "scaffold_diversity": round(self.scaffold_diversity, 4),
            "reaction_specificity": round(self.reaction_specificity, 4),
            "composite": round(self.composite, 4),
        }


@dataclass
class QualityReport:
    """Aggregate quality statistics across all pairs."""

    total_pairs: int
    mean_composite: float
    mean_validity: float
    mean_druglikeness: float
    mean_reflection: float
    mean_yield: float
    mean_diversity: float
    mean_specificity: float
    min_composite: float
    max_composite: float
    scores: list[QualityScores] = field(default_factory=list)

    def summary_table(self) -> str:
        rows = [
            f"{'Quality Dimension':<28} {'Mean':>8}",
            "-" * 38,
            f"{'Structural Validity':<28} {self.mean_validity:>8.4f}",
            f"{'Drug-Likeness (QED)':<28} {self.mean_druglikeness:>8.4f}",
            f"{'Reflection Depth':<28} {self.mean_reflection:>8.4f}",
            f"{'Yield Differential':<28} {self.mean_yield:>8.4f}",
            f"{'Scaffold Diversity':<28} {self.mean_diversity:>8.4f}",
            f"{'Reaction Specificity':<28} {self.mean_specificity:>8.4f}",
            "-" * 38,
            f"{'Composite':<28} {self.mean_composite:>8.4f}",
            f"{'Range':<28} [{self.min_composite:.4f}, {self.max_composite:.4f}]",
        ]
        return "\n".join(rows)

    def to_dict(self) -> dict:
        return {
            "total_pairs": self.total_pairs,
            "mean_composite": round(self.mean_composite, 4),
            "mean_validity": round(self.mean_validity, 4),
            "mean_druglikeness": round(self.mean_druglikeness, 4),
            "mean_reflection": round(self.mean_reflection, 4),
            "mean_yield": round(self.mean_yield, 4),
            "mean_diversity": round(self.mean_diversity, 4),
            "mean_specificity": round(self.mean_specificity, 4),
            "min_composite": round(self.min_composite, 4),
            "max_composite": round(self.max_composite, 4),
            "scores": [s.to_dict() for s in self.scores],
        }


def compute_quality_scores(
    pair: PreferencePair,
) -> QualityScores:
    """Score a single preference pair across all 6 chemistry-aware dimensions.

    Args:
        pair: A compiled DPO preference pair with chosen/rejected hypotheses.

    Returns:
        QualityScores with per-dimension and composite scores.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Scaffolds

    pair_id = pair.id.hex if hasattr(pair.id, "hex") else str(pair.id)
    scores = QualityScores(pair_id=pair_id)

    # Dimension 1: Structural validity (0-1)
    try:
        chosen_data = json.loads(pair.chosen) if isinstance(pair.chosen, str) else {}
        rejected_data = json.loads(pair.rejected) if isinstance(pair.rejected, str) else {}
    except (json.JSONDecodeError, TypeError):
        chosen_data = {}
        rejected_data = {}

    chosen_smiles = _extract_smiles(chosen_data)
    rejected_has_invalid = any(
        Chem.MolFromSmiles(s) is None for s in _extract_smiles(rejected_data)
    )
    all_valid = all(Chem.MolFromSmiles(s) is not None for s in chosen_smiles)
    if all_valid and len(chosen_smiles) > 0:
        scores.structural_validity = 1.0
    elif all_valid:
        scores.structural_validity = 0.5
    elif rejected_has_invalid:
        scores.structural_validity = 0.3
    else:
        scores.structural_validity = 0.1

    # Dimension 2: Drug-likeness via QED (0-1)
    mols = [Chem.MolFromSmiles(s) for s in chosen_smiles]
    valid_mols = [m for m in mols if m is not None]
    if valid_mols:
        try:
            qed_vals = [Descriptors.qed(m) for m in valid_mols]
            scores.drug_likeness = sum(qed_vals) / len(qed_vals)
        except Exception:
            scores.drug_likeness = 0.1

    # Dimension 3: Reflection depth (0-1)
    if pair.metadata:
        confidence = float(pair.metadata.get("reflection_confidence", 0.3))
        cat_count = len(pair.metadata.get("failure_categories", []))
        scores.reflection_depth = min(confidence * 0.6 + cat_count * 0.1, 1.0)

    # Dimension 4: Yield differential (0-1)
    if pair.metadata and "chosen_yield" in pair.metadata:
        y = pair.metadata["chosen_yield"]
        if y >= 80:
            scores.yield_differential = 1.0
        elif y >= 60:
            scores.yield_differential = 0.8
        elif y >= 40:
            scores.yield_differential = 0.5
        elif y >= 20:
            scores.yield_differential = 0.2
        else:
            scores.yield_differential = 0.05
    else:
        scores.yield_differential = 0.3

    # Dimension 5: Scaffold diversity (0-1)
    if valid_mols:
        try:
            scaffolds: set[str] = set()
            heavy_total = 0
            for m in valid_mols:
                heavy_total += m.GetNumHeavyAtoms()
                scaff = Scaffolds.MurckoScaffold.GetScaffoldForMol(m)
                if scaff and scaff.GetNumAtoms() > 0:
                    scaffolds.add(Chem.MolToSmiles(scaff))
            scaf_ratio = len(scaffolds) / max(1, len(valid_mols))
            heavy_avg = heavy_total / len(valid_mols)
            scores.scaffold_diversity = min(scaf_ratio, 1.0) * 0.6 + min(heavy_avg / 20, 1.0) * 0.4
        except Exception:
            scores.scaffold_diversity = 0.2
    else:
        scores.scaffold_diversity = 0.0

    # Dimension 6: Reaction type specificity (0-1)
    rtype = pair.reaction_type
    if rtype and rtype != "other":
        scores.reaction_specificity = 0.8
    elif rtype == "other":
        scores.reaction_specificity = 0.3
    else:
        scores.reaction_specificity = 0.2

    # Composite (weighted average)
    weights = {
        "sv": 0.20,
        "dl": 0.15,
        "rd": 0.25,
        "yd": 0.10,
        "sd": 0.15,
        "rs": 0.15,
    }
    scores.composite = (
        scores.structural_validity * weights["sv"]
        + scores.drug_likeness * weights["dl"]
        + scores.reflection_depth * weights["rd"]
        + scores.yield_differential * weights["yd"]
        + scores.scaffold_diversity * weights["sd"]
        + scores.reaction_specificity * weights["rs"]
    )

    return scores


def _extract_smiles(data: dict) -> list[str]:
    """Extract SMILES strings from pair data (handles both structured and text formats)."""
    smiles: list[str] = []
    text = str(data)

    import re

    for match in re.finditer(r"[A-Za-z0-9@\[\]\(\)=#/\-+\\\.%]{3,}", text):
        candidate = match.group(0)
        if any(c.isupper() or c in "@[]()=#/" for c in candidate):
            smiles.append(candidate)

    return smiles[:10]


def score_all_pairs(pairs: list[PreferencePair]) -> QualityReport:
    """Score all pairs and generate aggregate report.

    Args:
        pairs: List of compiled DPO preference pairs.

    Returns:
        QualityReport with per-pair scores and aggregate statistics.
    """
    scores = [compute_quality_scores(p) for p in pairs]

    if not scores:
        return QualityReport(
            total_pairs=0,
            mean_composite=0.0,
            mean_validity=0.0,
            mean_druglikeness=0.0,
            mean_reflection=0.0,
            mean_yield=0.0,
            mean_diversity=0.0,
            mean_specificity=0.0,
            min_composite=0.0,
            max_composite=0.0,
        )

    n = len(scores)
    return QualityReport(
        total_pairs=n,
        mean_composite=sum(s.composite for s in scores) / n,
        mean_validity=sum(s.structural_validity for s in scores) / n,
        mean_druglikeness=sum(s.drug_likeness for s in scores) / n,
        mean_reflection=sum(s.reflection_depth for s in scores) / n,
        mean_yield=sum(s.yield_differential for s in scores) / n,
        mean_diversity=sum(s.scaffold_diversity for s in scores) / n,
        mean_specificity=sum(s.reaction_specificity for s in scores) / n,
        min_composite=min(s.composite for s in scores),
        max_composite=max(s.composite for s in scores),
        scores=scores,
    )


def export_quality_report(report: QualityReport, path: str | Path) -> None:
    """Export quality report as JSON.

    Args:
        report: QualityReport to export.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = report.to_dict()
    path.write_text(json.dumps(data, indent=2))
    logger.info("Quality report exported to {}", path)
