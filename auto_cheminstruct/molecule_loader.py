import warnings
warnings.filterwarnings("ignore")

TARGET_MOLECULES = [
    {"name": "Aspirin",        "smiles": "CC(=O)Oc1ccccc1C(=O)O",          "class": "analgesic"},
    {"name": "Paracetamol",    "smiles": "CC(=O)Nc1ccc(O)cc1",              "class": "analgesic"},
    {"name": "Ibuprofen",      "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",     "class": "NSAID"},
    {"name": "Caffeine",       "smiles": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",     "class": "stimulant"},
    {"name": "Dopamine",       "smiles": "NCCc1ccc(O)c(O)c1",              "class": "neurotransmitter"},
    {"name": "Benzocaine",     "smiles": "CCOC(=O)c1ccc(N)cc1",            "class": "anesthetic"},
    {"name": "Aniline",        "smiles": "Nc1ccccc1",                       "class": "aromatic amine"},
    {"name": "Vanillin",       "smiles": "COc1cc(C=O)ccc1O",               "class": "aldehyde"},
    {"name": "Lidocaine",      "smiles": "CCN(CC)CC(=O)Nc1c(C)cccc1C",     "class": "anesthetic"},
    {"name": "Metformin",      "smiles": "CN(C)C(=N)NC(=N)N",              "class": "antidiabetic"},
    {"name": "Naproxen",       "smiles": "COc1ccc2cc(C(C)C(=O)O)ccc2c1",  "class": "NSAID"},
    {"name": "Adrenaline",     "smiles": "CNC[C@@H](O)c1ccc(O)c(O)c1",    "class": "hormone"},
    {"name": "Nicotine",       "smiles": "CN1CCC[C@H]1c1cccnc1",           "class": "alkaloid"},
    {"name": "Serotonin",      "smiles": "NCCc1c[nH]c2ccc(O)cc12",         "class": "neurotransmitter"},
    {"name": "Melatonin",      "smiles": "COc1ccc2[nH]cc(CCNC(C)=O)c2c1", "class": "hormone"},
    {"name": "Phenol",         "smiles": "Oc1ccccc1",                       "class": "phenolic"},
    {"name": "Salicylic acid", "smiles": "OC(=O)c1ccccc1O",                "class": "acid"},
    {"name": "Glycine",        "smiles": "NCC(=O)O",                        "class": "amino acid"},
    {"name": "Serotonin",      "smiles": "NCCc1c[nH]c2ccc(O)cc12",         "class": "neurotransmitter"},
    {"name": "Toluene",        "smiles": "Cc1ccccc1",                       "class": "aromatic"},
]

def get_molecules() -> list[dict]:
    return TARGET_MOLECULES