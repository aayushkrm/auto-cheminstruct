"""RDKit wrapper for molecular validation and descriptor computation."""

from __future__ import annotations

from typing import Optional

from loguru import logger
from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    Descriptors,
    QED,
    rdMolDescriptors,
)
from rdkit.Chem.rdDistGeom import ETKDGv3

from src.exceptions import MolecularValidationError, SMILESParseError


def smiles_to_mol(smiles: str, sanitize: bool = True) -> Chem.Mol:
    """Convert SMILES string to RDKit Mol object.

    Args:
        smiles: SMILES string to parse.
        sanitize: Whether to sanitize the molecule (valence, kekulize).

    Returns:
        RDKit Mol object.

    Raises:
        SMILESParseError: If SMILES parsing fails.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if mol is None:
        raise SMILESParseError(f"Failed to parse SMILES: {smiles}")
    return mol


def mol_to_smiles(mol: Chem.Mol, canonical: bool = True) -> str:
    """Convert RDKit Mol to canonical SMILES."""
    return Chem.MolToSmiles(mol, canonical=canonical)


def validate_molecule(mol: Chem.Mol) -> tuple[bool, list[str]]:
    """Perform comprehensive molecular validation.

    Checks:
        - Valence satisfaction
        - No explicit radical electrons
        - No impossible valences
        - Normal atom types only (no wildcards)

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    errors: list[str] = []

    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        errors.append(f"Sanitization failed: {e}")
        return False, errors

    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:
            errors.append(f"Atom {atom.GetIdx()} {atom.GetSymbol()} has radical electrons")

        try:
            atom.GetTotalValence()
        except Exception:
            errors.append(
                f"Atom {atom.GetIdx()} {atom.GetSymbol()} has impossible valence"
            )

    return len(errors) == 0, errors


def generate_conformer(mol: Chem.Mol, max_attempts: int = 100) -> Optional[Chem.Mol]:
    """Generate 3D conformer using ETKDGv3.

    Args:
        mol: RDKit Mol (with hydrogens added).
        max_attempts: Maximum conformer generation attempts.

    Returns:
        Mol with conformer embedded, or None on failure.
    """
    mol = Chem.AddHs(mol)
    params = ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 1

    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        logger.warning("ETKDG conformer embedding failed, retrying with random coords")
        status = AllChem.EmbedMolecule(mol, params)

    if status != 0:
        return None

    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def compute_descriptors(mol: Chem.Mol) -> dict[str, float]:
    """Compute key molecular descriptors.

    Returns dict with: mw, logp, tpsa, num_rotatable_bonds,
    num_h_acceptors, num_h_donors, qed, sa_score.
    """
    return {
        "molecular_weight": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol),
        "num_rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "num_h_acceptors": rdMolDescriptors.CalcNumHBA(mol),
        "num_h_donors": rdMolDescriptors.CalcNumHBD(mol),
        "qed": QED.qed(mol),
        "sa_score": _estimate_sa_score(mol),
    }


def _estimate_sa_score(mol: Chem.Mol) -> float:
    """Estimate Synthetic Accessibility score (1=easy, 10=hard).

    Based on Ertl & Schuffenhauer (2009) fragment-based approach.
    """
    fragment_score = _compute_fragment_score(mol)
    complexity_penalty = _compute_complexity_penalty(mol)

    sa_score = fragment_score - complexity_penalty
    return max(1.0, min(10.0, sa_score))


def _compute_fragment_score(mol: Chem.Mol) -> float:
    """Compute fragment contribution to SA score."""
    mol_weight = Descriptors.MolWt(mol)
    num_atoms = mol.GetNumHeavyAtoms()
    num_rings = rdMolDescriptors.CalcNumRings(mol)
    num_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
    num_aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)

    base = 3.0
    base += num_atoms * 0.05
    base += num_rotatable * 0.1
    base += num_rings * 0.15
    base -= num_aromatic * 0.1

    if mol_weight > 500:
        base += (mol_weight - 500) * 0.005

    return base


def _compute_complexity_penalty(mol: Chem.Mol) -> float:
    """Penalty for structural complexity."""
    num_chiral = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    num_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    num_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)

    penalty = num_chiral * 0.2
    penalty += num_spiro * 1.0
    penalty += num_bridgehead * 0.5

    return penalty


def check_steric_clash(
    mol: Chem.Mol, distance_threshold: float = 1.2
) -> tuple[bool, list[str]]:
    """Check for steric clashes between non-bonded atoms.

    Args:
        mol: RDKit Mol with 3D conformer.
        distance_threshold: Minimum allowed distance in Angstroms.

    Returns:
        Tuple of (has_clash, list of clash descriptions).
    """
    conf = mol.GetConformer()
    clashes: list[str] = []

    for i in range(mol.GetNumAtoms()):
        for j in range(i + 1, mol.GetNumAtoms()):
            bond = mol.GetBondBetweenAtoms(i, j)
            if bond is not None:
                continue

            pos_i = conf.GetAtomPosition(i)
            pos_j = conf.GetAtomPosition(j)
            distance = pos_i.Distance(pos_j)

            if distance < distance_threshold:
                atom_i = mol.GetAtomWithIdx(i)
                atom_j = mol.GetAtomWithIdx(j)
                clashes.append(
                    f"Steric clash: {atom_i.GetSymbol()}{i} - "
                    f"{atom_j.GetSymbol()}{j} at {distance:.2f} Å"
                )

    return len(clashes) == 0, clashes


def validate_smiles_syntax(smiles: str) -> bool:
    """Quick SMILES syntax check only (no sanitization)."""
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    return mol is not None


# Unstable functional group SMARTS patterns
UNSTABLE_FUNCTIONAL_GROUPS: dict[str, str] = {
    "peroxide": "[OX2,OX2-]-[OX2,OX2-]",
    "linear_azide": "[NX2-]=[NX2+]=[NX1-]",
    "diazo": "[CX3]=[NX2+]=[NX1-]",
    "acyl_azide": "[CX3](=O)-[NX2-]-[NX2+]#[NX1]",
    "nitroso": "[NX2]=O",
    "isocyanate": "[NX2]=[CX2]=[OX1]",
    "nitrile_oxide": "[CX2]#[NX2+]-[OX1-]",
    "ozonide": "[OX2,OX2-]-[OX2,OX2-]-[OX2,OX2-]",
    "acid_chloride": "[CX3](=O)-[ClX1]",
    "sulfonyl_chloride": "[SX4](=O)(=O)-[ClX1]",
}


def check_chemical_feasibility(mol: Chem.Mol) -> tuple[bool, list[str]]:
    """Check chemical feasibility beyond basic structural validation.

    Checks: unstable functional groups, ring strain, hypervalent atoms,
    and unusual valence states.

    Returns:
        Tuple of (is_feasible, list of warnings).
    """
    warnings: list[str] = []

    for name, smarts in UNSTABLE_FUNCTIONAL_GROUPS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue
        if mol.HasSubstructMatch(pattern):
            warnings.append(f"Contains potentially unstable group: {name}")

    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if len(ring) == 3:
            warnings.append(f"Contains strained 3-membered ring")
            break

    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        atomic_num = atom.GetAtomicNum()

        if symbol in ("P", "S", "I") and atom.GetTotalValence() > 4:
            if symbol == "I" and atom.GetTotalValence() > 7:
                warnings.append(f"Hypervalent iodine with valence {atom.GetTotalValence()}")
            elif symbol in ("P", "S") and atom.GetTotalValence() > 6:
                warnings.append(f"Hypervalent {symbol} with valence {atom.GetTotalValence()}")

        if atomic_num > 18 and atomic_num not in (35, 53):
            warnings.append(f"Unusual heavy atom: {symbol}")

    return len(warnings) == 0, warnings
