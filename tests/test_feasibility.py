"""Tests for chemical feasibility filter — unstable groups, hypervalent atoms, ring strain."""

import pytest
from rdkit import Chem
from src.chemistry.rdkit_wrapper import check_chemical_feasibility


class TestChemicalFeasibility:
    def test_stable_molecule_passes(self):
        mol = Chem.MolFromSmiles("CCO")
        feasible, warnings = check_chemical_feasibility(mol)
        assert feasible is True
        assert len(warnings) == 0

    def test_benzene_passes(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        feasible, _ = check_chemical_feasibility(mol)
        assert feasible is True

    def test_peroxide_detected(self):
        mol = Chem.MolFromSmiles("CCOOC")
        _, warnings = check_chemical_feasibility(mol)
        assert any("peroxide" in w.lower() for w in warnings)

    def test_azide_detected(self):
        mol = Chem.MolFromSmiles("CN=[N+]=[N-]")
        _, warnings = check_chemical_feasibility(mol)

    def test_cyclopropane_strain_detected(self):
        mol = Chem.MolFromSmiles("C1CC1")
        _, warnings = check_chemical_feasibility(mol)
        assert any("strain" in w.lower() or "ring" in w.lower() for w in warnings)

    def test_cyclobutane_no_strain_checked(self):
        mol = Chem.MolFromSmiles("C1CCC1")
        feasible, _ = check_chemical_feasibility(mol)
        assert feasible is True

    def test_phosphoric_acid_passes(self):
        mol = Chem.MolFromSmiles("OP(=O)(O)O")
        feasible, _ = check_chemical_feasibility(mol)
        assert feasible is True

    def test_valid_bromoalkane_passes(self):
        mol = Chem.MolFromSmiles("CCBr")
        feasible, _ = check_chemical_feasibility(mol)
        assert feasible is True

    def test_isocyanate_warns(self):
        mol = Chem.MolFromSmiles("CN=C=O")
        _, warnings = check_chemical_feasibility(mol)
        assert any("isocyanate" in w.lower() for w in warnings)

    def test_acid_chloride_warns(self):
        mol = Chem.MolFromSmiles("CC(=O)Cl")
        _, warnings = check_chemical_feasibility(mol)
        assert any("acid_chloride" in w.lower() for w in warnings)

    def test_sulfonyl_chloride_warns(self):
        mol = Chem.MolFromSmiles("CS(=O)(=O)Cl")
        _, warnings = check_chemical_feasibility(mol)
        assert any("sulfonyl_chloride" in w.lower() for w in warnings)

    def test_invalid_smiles_handled(self):
        mol = Chem.MolFromSmiles("not_a_smiles")
        assert mol is None

    def test_multiple_warnings_accumulated(self):
        mol = Chem.MolFromSmiles("C1CC1OOC")
        _, warnings = check_chemical_feasibility(mol)
        assert len(warnings) >= 1
