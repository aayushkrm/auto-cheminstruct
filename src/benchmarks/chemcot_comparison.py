"""ChemCoTBench comparison module for Auto-ChemInstruct.

Evaluates our generated dataset against ChemCoTBench's benchmark dimensions:
1. Molecular property optimization coverage
2. Chemical reaction prediction diversity
3. Step-wise reasoning quality
4. Physical validity metrics
5. Dataset scale and diversity comparison
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


@dataclass
class ChemCoTComparison:
    """Comparison metrics between our dataset and ChemCoTBench benchmarks."""

    dataset_path: str
    total_pairs: int = 0
    total_molecules: int = 0
    unique_scaffolds: int = 0
    scaffold_diversity: float = 0.0
    tanimoto_diversity: float = 0.0
    tanimoto_mean: float = 0.0
    tanimoto_std: float = 0.0
    reaction_type_counts: dict[str, int] = field(default_factory=dict)
    reaction_type_diversity: int = 0
    avg_quality_score: float = 0.0
    quality_std: float = 0.0
    avg_molecular_weight: float = 0.0
    mw_range: tuple[float, float] = (0, 0)
    avg_num_rings: float = 0.0
    avg_logp: float = 0.0
    avg_tpsa: float = 0.0
    avg_qed: float = 0.0
    avg_sa_score: float = 0.0
    reflection_ratio: float = 0.0
    avg_reflection_length: int = 0
    pair_contamination: float = 0.0

    # ChemCoTBench reference values (from paper)
    chemcot_total_samples: int = 1495
    chemcot_tasks: int = 22
    chemcot_avg_tokens_per_sample: int = 0
    chemcot_human_annotated: int = 0

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset_path,
            "total_pairs": self.total_pairs,
            "total_molecules": self.total_molecules,
            "unique_scaffolds": self.unique_scaffolds,
            "scaffold_diversity": round(self.scaffold_diversity, 4),
            "tanimoto_diversity": round(self.tanimoto_diversity, 4),
            "tanimoto_mean": round(self.tanimoto_mean, 4),
            "tanimoto_std": round(self.tanimoto_std, 4),
            "reaction_type_counts": self.reaction_type_counts,
            "reaction_type_diversity": self.reaction_type_diversity,
            "avg_quality_score": round(self.avg_quality_score, 4),
            "quality_std": round(self.quality_std, 4),
            "avg_molecular_weight": round(self.avg_molecular_weight, 2),
            "mw_range": [round(x, 2) for x in self.mw_range],
            "avg_num_rings": round(self.avg_num_rings, 2),
            "avg_logp": round(self.avg_logp, 2),
            "avg_tpsa": round(self.avg_tpsa, 2),
            "avg_qed": round(self.avg_qed, 2),
            "avg_sa_score": round(self.avg_sa_score, 2),
            "reflection_ratio": round(self.reflection_ratio, 4),
            "avg_reflection_length": self.avg_reflection_length,
            "pair_contamination": round(self.pair_contamination, 4),
            "chemcot_reference": {
                "total_samples": self.chemcot_total_samples,
                "tasks": self.chemcot_tasks,
            },
        }

    def comparison_summary(self) -> str:
        """Generate a formatted comparison summary for the paper."""
        lines = [
            f"{'Metric':<35} {'Our Dataset':>15} {'ChemCoTBench':>15}",
            "-" * 65,
        ]
        metrics = [
            ("Total Samples", str(self.total_pairs), str(self.chemcot_total_samples)),
            ("Reaction Types", str(self.reaction_type_diversity), str(self.chemcot_tasks)),
            ("Unique Molecules", str(self.total_molecules), "—"),
            ("Tanimoto Diversity", f"{self.tanimoto_diversity:.3f}", "—"),
            ("Scaffold Diversity", f"{self.scaffold_diversity:.1%}", "—"),
            ("Avg Quality Score", f"{self.avg_quality_score:.3f}", "—"),
            ("Reflection Traces", f"{self.reflection_ratio:.1%}", "—"),
        ]
        for name, ours, theirs in metrics:
            lines.append(f"{name:<35} {ours:>15} {theirs:>15}")
        return "\n".join(lines)


def analyze_dataset(dataset_dir: str) -> ChemCoTComparison:
    """Analyze a compiled dataset for ChemCoTBench comparison.

    Args:
        dataset_dir: Path to dataset directory (e.g., datasets/autochem-<id>/).

    Returns:
        ChemCoTComparison with computed metrics.
    """
    comp = ChemCoTComparison(dataset_path=dataset_dir)

    train_path = Path(dataset_dir) / "train.jsonl"
    if not train_path.exists():
        logger.warning("No train.jsonl found in {}", dataset_dir)
        return comp

    pairs = []
    with open(train_path) as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    if not pairs:
        return comp

    comp.total_pairs = len(pairs)

    # Extract all molecules from chosen (passed) reactions
    all_smiles: list[str] = []
    reaction_types: list[str] = []
    quality_scores: list[float] = []
    reflection_lengths: list[int] = []
    has_reflection: list[bool] = []

    for pair in pairs:
        reaction_types.append(pair.get("reaction_type", "unknown"))
        quality_scores.append(pair.get("quality_score", 0.0))

        # Extract SMILES from chosen reaction
        chosen = pair.get("chosen", "")
        for line in chosen.split("\n"):
            if "Reactants:" in line or "Products:" in line:
                for s in line.split(":", 1)[1].strip().split(", "):
                    s = s.strip()
                    if s and len(s) > 1 and Chem.MolFromSmiles(s):
                        all_smiles.append(s)

        # Extract reflection info from rejected
        rejected = pair.get("rejected", "")
        has_reflection.append("FAILURE ANALYSIS" in rejected or "failure_category" in pair.get("metadata", {}).get("rejected", ""))
        if "FAILURE ANALYSIS" in rejected:
            reflection_lengths.append(len(rejected))

    comp.total_molecules = len(set(all_smiles))
    comp.reaction_type_counts = dict(Counter(reaction_types))
    comp.reaction_type_diversity = len(comp.reaction_type_counts)
    comp.avg_quality_score = float(np.mean(quality_scores)) if quality_scores else 0.0
    comp.quality_std = float(np.std(quality_scores)) if quality_scores else 0.0
    comp.reflection_ratio = sum(has_reflection) / max(1, len(has_reflection))
    comp.avg_reflection_length = int(np.mean(reflection_lengths)) if reflection_lengths else 0

    # Molecular property statistics
    unique_smiles = list(set(all_smiles))
    mols = [Chem.MolFromSmiles(s) for s in unique_smiles if Chem.MolFromSmiles(s)]
    comp.total_molecules = len(mols)

    if mols:
        mws = [Descriptors.MolWt(m) for m in mols]
        comp.avg_molecular_weight = float(np.mean(mws))
        comp.mw_range = (float(np.min(mws)), float(np.max(mws)))

        rings = [rdMolDescriptors.CalcNumRings(m) for m in mols]
        comp.avg_num_rings = float(np.mean(rings))

        logps = [Descriptors.MolLogP(m) for m in mols]
        comp.avg_logp = float(np.mean(logps))

        tpsas = [Descriptors.TPSA(m) for m in mols]
        comp.avg_tpsa = float(np.mean(tpsas))

        qeds = [Descriptors.qed(m) for m in mols if hasattr(Descriptors, "qed")]
        comp.avg_qed = float(np.mean(qeds)) if qeds else 0.0

        # SA Score approximation (based on molecular complexity)
        sa_scores = [_synthetic_accessibility_score(m) for m in mols]
        comp.avg_sa_score = float(np.mean(sa_scores))

        # Tanimoto diversity
        fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]
        if len(fps) > 1:
            sims = []
            for i in range(len(fps)):
                for j in range(i + 1, len(fps)):
                    sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
            comp.tanimoto_mean = float(np.mean(sims))
            comp.tanimoto_std = float(np.std(sims))
            comp.tanimoto_diversity = 1.0 - comp.tanimoto_mean

        # Scaffold diversity
        scaffolds = set()
        for m in mols:
            try:
                scaff = MurckoScaffold.GetScaffoldForMol(m)
                if scaff and scaff.GetNumAtoms() > 0:
                    scaffolds.add(Chem.MolToSmiles(scaff))
            except Exception:
                pass
        comp.unique_scaffolds = len(scaffolds)
        comp.scaffold_diversity = comp.unique_scaffolds / max(1, comp.total_molecules)

    # Intra-dataset pair contamination (near-duplicate detection)
    if len(pairs) > 1:
        chosen_hashes = set()
        duplicates = 0
        for pair in pairs:
            chosen = pair.get("chosen", "")
            ch = hash(chosen)
            if ch in chosen_hashes:
                duplicates += 1
            chosen_hashes.add(ch)
        comp.pair_contamination = duplicates / len(pairs)

    return comp


def _synthetic_accessibility_score(mol: Chem.Mol) -> float:
    """Approximate SA score based on molecular complexity.

    Formula: fragment contribution + complexity penalty.
    Range: 1 (easy) to 10 (hard).
    """
    # Complexity penalty factors
    complexity = 0.0

    # Ring complexity
    num_rings = rdMolDescriptors.CalcNumRings(mol)
    complexity += num_rings * 0.2

    # Chiral centers
    chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    complexity += chiral * 0.3

    # Spiro atoms (shared between rings)
    ri = mol.GetRingInfo()
    spiro_count = 0
    for atom in mol.GetAtoms():
        ring_count = sum(1 for ring in ri.AtomRings() if atom.GetIdx() in ring)
        if ring_count > 1:
            spiro_count += 1
    complexity += spiro_count * 0.5

    # Bridgehead atoms
    bridgehead_count = 0
    for atom in mol.GetAtoms():
        if atom.IsInRing():
            ring_count = len([r for r in ri.AtomRings() if atom.GetIdx() in r])
            if ring_count >= 3:
                bridgehead_count += 1
    complexity += bridgehead_count * 0.8

    # Macrocycle penalty
    if num_rings > 2 and rdMolDescriptors.CalcNumRotatableBonds(mol) > 6:
        complexity += 2.0

    # Size penalty
    num_atoms = mol.GetNumAtoms()
    complexity += max(0, (num_atoms - 30) * 0.1)

    score = 1.0 + complexity
    return round(min(10.0, score), 2)


def compare_to_chemcot(
    dataset_dir: str,
    output_file: str | None = None,
) -> dict:
    """Run full comparison against ChemCoTBench metrics.

    Args:
        dataset_dir: Path to our compiled dataset.
        output_file: Optional path for JSON output.

    Returns:
        Comparison results dict.
    """
    comp = analyze_dataset(dataset_dir)
    result = comp.to_dict()

    if output_file:
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Comparison report saved to {}", output_file)

    logger.info("\n{}", comp.comparison_summary())
    return result
