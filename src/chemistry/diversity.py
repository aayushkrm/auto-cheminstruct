"""Chemical diversity and analysis utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


def tanimoto_similarity(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    """Compute Tanimoto similarity between two molecules (Morgan fingerprint)."""
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def pairwise_tanimoto(mols: Sequence[Chem.Mol]) -> np.ndarray:
    """Compute pairwise Tanimoto similarity matrix."""
    n = len(mols)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim = tanimoto_similarity(mols[i], mols[j])
            matrix[i][j] = sim
            matrix[j][i] = sim
        matrix[i][i] = 1.0
    return matrix


def compute_diversity_score(mols: Sequence[Chem.Mol]) -> float:
    """Compute internal diversity (1 - mean pairwise similarity)."""
    if len(mols) < 2:
        return 1.0
    matrix = pairwise_tanimoto(mols)
    upper_tri = matrix[np.triu_indices(len(mols), k=1)]
    mean_similarity = np.mean(upper_tri) if len(upper_tri) > 0 else 0.0
    return 1.0 - mean_similarity


def scaffold_diversity(mols: Sequence[Chem.Mol]) -> dict:
    """Analyze scaffold diversity.

    Returns dict with: num_unique_scaffolds, scaffold_counts, scaffold_smiles.
    """
    scaffolds: dict[str, int] = {}

    for mol in mols:
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold is None:
                continue
            scaffold_smiles = Chem.MolToSmiles(scaffold)
            scaffolds[scaffold_smiles] = scaffolds.get(scaffold_smiles, 0) + 1
        except Exception:
            continue

    return {
        "num_unique_scaffolds": len(scaffolds),
        "scaffold_counts": dict(
            sorted(scaffolds.items(), key=lambda x: x[1], reverse=True)
        ),
    }


def filter_by_diversity(
    mols: list[Chem.Mol],
    threshold: float = 0.7,
    max_keep: int | None = None,
) -> list[int]:
    """Filter molecules to maximize diversity (greedy).

    Args:
        mols: List of RDKit Mol objects.
        threshold: Maximum allowed Tanimoto similarity (0-1).
        max_keep: Maximum number of molecules to keep.

    Returns:
        Indices of selected molecules.
    """
    if not mols:
        return []

    selected: list[int] = [0]

    for i in range(1, len(mols)):
        if max_keep and len(selected) >= max_keep:
            break

        max_sim = max(
            tanimoto_similarity(mols[i], mols[j]) for j in selected
        )
        if max_sim <= threshold:
            selected.append(i)

    return selected


def reaction_type_counts(
    pairs: list,
) -> dict[str, int]:
    """Count pairs by reaction type."""
    from collections import Counter
    return dict(Counter(p.reaction_type.value for p in pairs))
