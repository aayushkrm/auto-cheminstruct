import warnings
warnings.filterwarnings("ignore")
import os
os.environ["RDKIT_SILENCE"] = "1"
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Common reagent names that LLMs write as names instead of SMILES
REAGENT_TO_SMILES = {
    "NH3": "N", "H2": "[H][H]", "H2O": "O", "CO2": "O=C=O",
    "HCl": "Cl", "H2SO4": "OS(=O)(=O)O", "HBr": "Br",
    "NaOH": "[Na+].[OH-]", "KOH": "[K+].[OH-]",
    "CH3OH": "CO", "EtOH": "CCO", "MeOH": "CO",
    "CH3I": "CI", "CH3Br": "CBr", "MeI": "CI",
    "Br2": "BrBr", "Cl2": "ClCl", "I2": "II",
    "NaBH4": "[Na+].[BH4-]", "LiAlH4": "[Li+].[AlH4-]",
    "AlCl3": "[Al+3].[Cl-].[Cl-].[Cl-]",
    "NaIO4": "[Na+].[IO4-]", "KMnO4": "[K+].[O-][Mn](=O)(=O)=O",
    "OsO4": "O=[Os](=O)(=O)=O",
    "CH3MgBr": "C[Mg]Br", "EtMgBr": "CC[Mg]Br",
    "CCMgBr": "CC[Mg]Br", "CCMgCl": "CC[Mg]Cl",
    "CH3Li": "C[Li]", "BuLi": "CCCC[Li]",
    "Mg": "[Mg]", "Zn": "[Zn]", "Fe": "[Fe]",
    "Pd": "[Pd]", "Pt": "[Pt]", "Ni": "[Ni]",
    "NaHCO3": "[Na+].OC([O-])=O",
    "Na2CO3": "[Na+].[Na+].[O-]C([O-])=O",
    "K2CO3": "[K+].[K+].[O-]C([O-])=O",
    "Et3N": "CCN(CC)CC", "TEA": "CCN(CC)CC",
    "DMF": "CN(C)C=O", "DMSO": "CS(C)=O",
    "THF": "C1CCOC1", "DCM": "ClCCl",
    "O2": "O=O", "N2": "N#N", "CO": "[C-]#[O+]",
}

def sanitize_reactants(reactants: list[str]) -> list[str]:
    """Convert reagent names to valid SMILES where possible."""
    cleaned = []
    for r in reactants:
        r = r.strip()
        if r in REAGENT_TO_SMILES:
            cleaned.append(REAGENT_TO_SMILES[r])
        else:
            cleaned.append(r)
    return cleaned

def sanitize_reaction_smiles(rxn_smi: str) -> str:
    """Fix reagent names in reaction SMILES blocks."""
    if not rxn_smi or ">>" not in rxn_smi:
        return rxn_smi
    reactant_block, _, product_block = rxn_smi.partition(">>")
    fixed_reactants = ".".join(
        REAGENT_TO_SMILES.get(s.strip(), s.strip())
        for s in reactant_block.split(".")
        if s.strip()
    )
    return f"{fixed_reactants}>>{product_block}"