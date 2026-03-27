import warnings
warnings.filterwarnings("ignore")

import os
os.environ["RDKIT_SILENCE"] = "1"

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import json
import matplotlib.pyplot as plt
from config import CHART_FAILURE_MODES, CHART_MOLECULES, OUTPUT_DIR

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    DRAW_OK = True
except Exception:
    DRAW_OK = False

COLORS = {
    "atom_imbalance":                   "#e74c3c",
    "functional_group_incompatibility": "#e67e22",
    "invalid_smiles_syntax":            "#9b59b6",
    "valence_violation":                "#8e44ad",
    "wrong_product":                    "#c0392b",
    "reagent_incompatibility":          "#f39c12",
    "unparseable_reactants":            "#d35400",
    "unparseable_products":             "#d35400",
    "regiochemistry_error":             "#27ae60",
    "stereochemistry_conflict":         "#16a085",
    "thermodynamic_infeasibility":      "#2980b9",
    "other":                            "#7f8c8d",
}


def plot_failure_modes(dataset: dict) -> None:
    dist = dataset.get("failure_mode_distribution", {})
    if not dist:
        print("  No failure mode data to plot.")
        return
    labels = list(dist.keys())
    values = list(dist.values())
    colors = [COLORS.get(l, "#7f8c8d") for l in labels]

    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(range(len(labels)), values, color=colors,
                  edgecolor="white", linewidth=1.2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=8)
    ax.set_ylabel("Number of preference pairs", fontsize=11)
    ax.set_title(
        f"Auto-ChemInstruct v0.1 — Chemical Failure Mode Distribution\n"
        f"({dataset['num_pairs']} preference pairs)",
        fontsize=13, fontweight="bold", pad=15,
    )
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05, str(val),
                ha="center", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(values) * 1.25 + 1)
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(CHART_FAILURE_MODES, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {CHART_FAILURE_MODES}")


def plot_molecule_grid(dataset: dict) -> None:
    if not DRAW_OK:
        print("  Molecule grid skipped — RDKit Draw unavailable")
        return
    seen = {}
    for p in dataset.get("data", []):
        if p["target_name"] not in seen:
            seen[p["target_name"]] = p["target_smiles"]
        if len(seen) >= 12:
            break
    mols, legends = [], []
    for name, smi in seen.items():
        m = Chem.MolFromSmiles(smi)
        if m:
            mols.append(m)
            legends.append(name)
    if mols:
        img = Draw.MolsToGridImage(
            mols, molsPerRow=4, subImgSize=(280, 200), legends=legends
        )
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        img.save(CHART_MOLECULES)
        print(f"  Grid  saved → {CHART_MOLECULES}")


def print_examples(dataset: dict, n: int = 2) -> None:
    print(f"\n{'═'*65}")
    print("  SAMPLE PREFERENCE PAIRS FROM AUTO-CHEMINSTRUCT v0.1")
    print(f"{'═'*65}")
    for i, pair in enumerate(dataset.get("data", [])[:n]):
        t = pair["rejection_trace"]
        print(f"\n  ── PAIR {i+1}: {pair['target_name']} ({pair['target_class']}) ──")
        print(f"  ✅ CHOSEN:   {pair['chosen'].split(chr(10))[0]}")
        print(f"  ❌ REJECTED: {pair['rejected'].split(chr(10))[0]}")
        print(f"  🔬 MODE:     {t['primary_failure_mode']}")
        print(f"  🔬 ANALYSIS: {t['causal_analysis'][:180]}")
        print(f"  💡 FIX:      {t['fix_suggestion'][:120]}")


def run_all(dataset: dict) -> None:
    plot_failure_modes(dataset)
    plot_molecule_grid(dataset)
    print_examples(dataset)