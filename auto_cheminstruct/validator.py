# Add this import at the top of validator.py
from sanitizer import sanitize_reactants, sanitize_reaction_smiles

def validate_route(target_smiles: str, route: dict) -> dict:
    errors = []

    # Sanitize before validating
    route["reactants"] = sanitize_reactants(route.get("reactants", []))
    if route.get("reaction_smiles"):
        route["reaction_smiles"] = sanitize_reaction_smiles(route["reaction_smiles"])

    # ... rest of the function stays exactly the same

import warnings
warnings.filterwarnings("ignore")

# Must suppress RDKit logger BEFORE importing anything from rdkit
import os
os.environ["RDKIT_SILENCE"] = "1"

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

INCOMPATIBLE_PAIRS = [
    (
        "[#6][MgBr,MgCl,Li]", "[OX2H]",
        "Grignard/organolithium destroyed by protic O-H bonds — incompatible in same flask."
    ),
    (
        "[CX3](=O)[OX2H1]", "[NX3;H2,H1][CX4]",
        "Carboxylic acid + amine forms inert ammonium salt — coupling agent required for amide bond."
    ),
    (
        "[CX3](=O)[F,Cl,Br]", "[OX2H]",
        "Acid halide + free hydroxyl causes uncontrolled acylation — protect the hydroxyl first."
    ),
    (
        "[N+](=O)[O-]", "[CX4][Li,Na,K]",
        "Nitro group (strong oxidant) + organometallic reductant — violent redox reaction."
    ),
]


def _check_smiles(smiles: str) -> tuple:
    if not smiles or not smiles.strip():
        return False, "empty_smiles", "SMILES is empty."
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return False, "invalid_smiles_syntax", f"RDKit cannot parse: '{smiles}'"
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        return False, "valence_violation", f"Sanitization failed for '{smiles}': {exc}"
    return True, "", ""


def _check_atom_balance(rxn_smi: str) -> tuple:
    if not rxn_smi or ">>" not in rxn_smi:
        return False, "missing_reaction_arrow", f"No '>>' in: '{rxn_smi}'"
    reactant_block, _, product_block = rxn_smi.partition(">>")

    def counts(block):
        result = {}
        for smi in block.split("."):
            smi = smi.strip()
            if not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return None
            for atom in Chem.AddHs(mol).GetAtoms():
                s = atom.GetSymbol()
                result[s] = result.get(s, 0) + 1
        return result

    r, p = counts(reactant_block), counts(product_block)
    if r is None:
        return False, "unparseable_reactants", f"Cannot parse reactants: '{reactant_block}'"
    if p is None:
        return False, "unparseable_products", f"Cannot parse products: '{product_block}'"

    heavy = {k for k in (set(r) | set(p)) if k != "H"}
    bad = [
        f"{a}: {r.get(a, 0)} → {p.get(a, 0)}"
        for a in sorted(heavy)
        if r.get(a, 0) != p.get(a, 0)
    ]
    if bad:
        return False, "atom_imbalance", "Mass conservation violated — " + "; ".join(bad)
    return True, "", ""


def _check_fg_compatibility(reactants: list) -> tuple:
    if len(reactants) < 2:
        return True, "", ""
    combined = Chem.MolFromSmiles(".".join(r.strip() for r in reactants if r.strip()))
    if combined is None:
        return True, "", ""
    for sma, smb, reason in INCOMPATIBLE_PAIRS:
        pa = Chem.MolFromSmarts(sma)
        pb = Chem.MolFromSmarts(smb)
        if pa and pb and combined.HasSubstructMatch(pa) and combined.HasSubstructMatch(pb):
            return False, "functional_group_incompatibility", reason
    return True, "", ""


def _check_product_match(target_smiles: str, route: dict) -> tuple:
    rxn = route.get("reaction_smiles", "")
    if not rxn or ">>" not in rxn:
        return True, "", ""
    product_smi = rxn.split(">>")[-1].strip()
    pm = Chem.MolFromSmiles(product_smi)
    tm = Chem.MolFromSmiles(target_smiles)
    if pm is None or tm is None:
        return True, "", ""
    pf = rdMolDescriptors.CalcMolFormula(pm)
    tf = rdMolDescriptors.CalcMolFormula(tm)
    if pf != tf:
        return False, "wrong_product", f"Product formula {pf} ≠ target formula {tf}"
    return True, "", ""


def validate_route(target_smiles: str, route: dict) -> dict:
    errors = []

    for smi in route.get("reactants", []):
        ok, etype, detail = _check_smiles(smi)
        if not ok:
            errors.append({"type": etype, "detail": detail})

    rxn = route.get("reaction_smiles", "")
    if rxn and ">>" in rxn:
        product_smi = rxn.split(">>")[-1]
        ok, etype, detail = _check_smiles(product_smi)
        if not ok:
            errors.append({"type": etype, "detail": detail})
        ok, etype, detail = _check_atom_balance(rxn)
        if not ok:
            errors.append({"type": etype, "detail": detail})

    ok, etype, detail = _check_fg_compatibility(route.get("reactants", []))
    if not ok:
        errors.append({"type": etype, "detail": detail})

    ok, etype, detail = _check_product_match(target_smiles, route)
    if not ok:
        errors.append({"type": etype, "detail": detail})

    return {
        "is_valid":      len(errors) == 0,
        "num_errors":    len(errors),
        "errors":        errors,
        "error_types":   [e["type"]   for e in errors],
        "error_details": [e["detail"] for e in errors],
    }