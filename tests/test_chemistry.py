"""Tests for chemistry utilities (RDKit wrapper and diversity)."""

import pytest

from rdkit import Chem

from src.chemistry.rdkit_wrapper import (
    smiles_to_mol,
    mol_to_smiles,
    validate_molecule,
    compute_descriptors,
    validate_smiles_syntax,
)
from src.chemistry.diversity import (
    tanimoto_similarity,
    compute_diversity_score,
    scaffold_diversity,
    filter_by_diversity,
)
from src.exceptions import SMILESParseError


class TestRDKitWrapper:
    def test_smiles_to_mol_benzene(self):
        mol = smiles_to_mol("c1ccccc1")
        assert mol is not None
        assert mol.GetNumAtoms() == 6

    def test_smiles_to_mol_ethanol(self):
        mol = smiles_to_mol("CCO")
        assert mol.GetNumAtoms() == 3

    def test_invalid_smiles_raises(self):
        with pytest.raises(SMILESParseError):
            smiles_to_mol("not_a_smiles!!!")

    def test_mol_to_smiles_canonical(self):
        mol = smiles_to_mol("CCO")
        smiles = mol_to_smiles(mol)
        assert smiles == "CCO"

    def test_validate_benzene(self):
        mol = smiles_to_mol("c1ccccc1")
        is_valid, errors = validate_molecule(mol)
        assert is_valid
        assert len(errors) == 0

    def test_validate_ethanol(self):
        mol = smiles_to_mol("CCO")
        is_valid, errors = validate_molecule(mol)
        assert is_valid

    def test_smiles_syntax_check(self):
        assert validate_smiles_syntax("c1ccccc1")
        assert validate_smiles_syntax("CCO")
        assert not validate_smiles_syntax("invalid!!!")

    def test_compute_descriptors_benzene(self):
        mol = smiles_to_mol("c1ccccc1")
        desc = compute_descriptors(mol)
        assert "molecular_weight" in desc
        assert "logp" in desc
        assert "tpsa" in desc
        assert "qed" in desc
        assert "sa_score" in desc
        assert desc["molecular_weight"] > 0

    def test_compute_descriptors_ethanol(self):
        mol = smiles_to_mol("CCO")
        desc = compute_descriptors(mol)
        assert desc["molecular_weight"] > 40
        assert desc["num_h_donors"] >= 1
        assert desc["num_h_acceptors"] >= 1

    def test_sa_score_range(self):
        for smiles in ["c1ccccc1", "CCO", "CC(=O)O", "c1ccccc1N"]:
            mol = smiles_to_mol(smiles)
            desc = compute_descriptors(mol)
            assert 1.0 <= desc["sa_score"] <= 10.0, f"{smiles} SA score: {desc['sa_score']}"


class TestDiversity:
    def test_tanimoto_identical(self):
        mol1 = smiles_to_mol("c1ccccc1")
        mol2 = smiles_to_mol("c1ccccc1")
        sim = tanimoto_similarity(mol1, mol2)
        assert sim == 1.0

    def test_tanimoto_different(self):
        mol1 = smiles_to_mol("c1ccccc1")
        mol2 = smiles_to_mol("CCO")
        sim = tanimoto_similarity(mol1, mol2)
        assert sim < 0.5

    def test_diversity_score_identical(self):
        mols = [smiles_to_mol("c1ccccc1"), smiles_to_mol("c1ccccc1")]
        score = compute_diversity_score(mols)
        assert score == 0.0

    def test_diversity_score_different(self):
        mols = [smiles_to_mol("c1ccccc1"), smiles_to_mol("CCO"), smiles_to_mol("CC(=O)O")]
        score = compute_diversity_score(mols)
        assert score > 0.0

    def test_diversity_score_single(self):
        mols = [smiles_to_mol("c1ccccc1")]
        score = compute_diversity_score(mols)
        assert score == 1.0

    def test_scaffold_diversity(self):
        mols = [
            smiles_to_mol("c1ccccc1"),
            smiles_to_mol("c1ccc(C)cc1"),
            smiles_to_mol("CCO"),
        ]
        result = scaffold_diversity(mols)
        assert result["num_unique_scaffolds"] >= 1

    def test_filter_by_diversity(self):
        mols = [
            smiles_to_mol("c1ccccc1"),
            smiles_to_mol("c1ccc(C)cc1"),
            smiles_to_mol("CCO"),
        ]
        indices = filter_by_diversity(mols, threshold=0.5)
        assert len(indices) >= 2
        assert 0 in indices  # First always included
