import warnings
warnings.filterwarnings("ignore")

import json, os
from datetime import datetime, timezone
from config import OUTPUT_DIR, DATASET_PATH

def _route_to_text(route: dict) -> str:
    return "\n".join([
        f"Strategy: {route.get('description', 'N/A')}",
        f"Transformation: {route.get('key_transformation', 'N/A')}",
        f"Reactants: {', '.join(route.get('reactants', []))}",
        f"Reagents: {', '.join(route.get('reagents', []))}",
        f"Conditions: {route.get('conditions', 'N/A')}",
        f"Reaction SMILES: {route.get('reaction_smiles', 'N/A')}",
        f"Expected yield: {route.get('expected_yield', 'N/A')}",
    ])

def build_preference_pair(target, chosen, rejected, trace, validation) -> dict:
    return {
        "prompt": (
            f"Propose a valid retrosynthetic route for:\n"
            f"Name: {target['name']}\n"
            f"SMILES: {target['smiles']}\n"
            f"Class: {target.get('class','unknown')}"
        ),
        "chosen":   _route_to_text(chosen),
        "rejected": _route_to_text(rejected),
        "rejection_trace": {
            "causal_analysis":          trace.get("causal_analysis", ""),
            "primary_failure_mode":     trace.get("primary_failure_mode", "other"),
            "problematic_feature":      trace.get("problematic_feature", ""),
            "chemical_principle_violated": trace.get("chemical_principle_violated", ""),
            "severity":                 trace.get("severity", "fatal"),
            "educational_note":         trace.get("educational_note", ""),
            "fix_suggestion":           trace.get("fix_suggestion", ""),
        },
        "validation": {
            "error_types":   validation.get("error_types", []),
            "error_details": validation.get("error_details", []),
        },
        "target_name":  target["name"],
        "target_smiles": target["smiles"],
        "target_class": target.get("class", "unknown"),
        "created_at":   datetime.now(timezone.utc).isoformat(),
    }

def save_dataset(pairs: list[dict]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    modes: dict[str, int] = {}
    for p in pairs:
        m = p["rejection_trace"]["primary_failure_mode"]
        modes[m] = modes.get(m, 0) + 1

    out = {
        "dataset_name": "AutoChemInstruct-v0.1",
        "description":  "Physics-verified DPO preference pairs with causal failure traces",
        "version":      "0.1",
        "num_pairs":    len(pairs),
        "failure_mode_distribution": modes,
        "format":       "DPO — prompt / chosen / rejected / rejection_trace",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data":         pairs,
    }
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved {len(pairs)} pairs → {DATASET_PATH}")
    print(f"  Failure modes: {modes}")