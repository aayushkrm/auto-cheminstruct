import warnings
warnings.filterwarnings("ignore")

import os
os.environ["RDKIT_SILENCE"] = "1"

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import json
from datetime import datetime, timezone
from tqdm import tqdm

from config import OUTPUT_DIR, STATS_PATH, DATASET_PATH
from molecule_loader import get_molecules
from route_generator import generate_routes
from validator import validate_route
from reflection_agent import generate_failure_trace
from dataset_builder import build_preference_pair, save_dataset
import visualizer


def run():
    print("\n" + "═" * 55)
    print("  AUTO-CHEMINSTRUCT v0.1")
    print("  Physics-Verified DPO Dataset Generator")
    print("═" * 55 + "\n")

    molecules = get_molecules()
    pairs = []
    stats = {
        "processed": 0, "routes_generated": 0,
        "passed": 0, "failed": 0, "pairs": 0, "errors": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    for mol in tqdm(molecules, desc="Molecules", unit="mol"):
        print(f"\n  ▶ {mol['name']}")
        routes = generate_routes(mol)
        if not routes:
            stats["errors"] += 1
            continue

        stats["routes_generated"] += len(routes)
        passed, failed = [], []

        for r in routes:
            result = validate_route(mol["smiles"], r)
            r["_val"] = result
            if result["is_valid"]:
                passed.append(r)
                print(f"    ✅ {r.get('key_transformation', '?')[:55]}")
            else:
                failed.append(r)
                print(f"    ❌ {r.get('key_transformation', '?')[:55]}")
                for d in result["error_details"][:1]:
                    print(f"       {d[:88]}")

        stats["passed"] += len(passed)
        stats["failed"] += len(failed)

        if passed and failed:
            for rej in failed:
                trace = generate_failure_trace(mol, rej, rej["_val"])
                print(
                    f"    🔬 [{trace.get('primary_failure_mode', '?')}] "
                    f"{trace.get('causal_analysis', '')[:68]}..."
                )
                pairs.append(
                    build_preference_pair(mol, passed[0], rej, trace, rej["_val"])
                )
                stats["pairs"] += 1
        else:
            reason = "no valid routes" if not passed else "no invalid routes"
            print(f"    ℹ  Skipping: {reason}")

        stats["processed"] += 1

    if pairs:
        save_dataset(pairs)

    stats["finished_at"] = datetime.now(timezone.utc).isoformat()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'═' * 55}")
    print(f"  DONE  |  {stats['pairs']} preference pairs created")
    print(f"  Passed routes : {stats['passed']}")
    print(f"  Failed routes : {stats['failed']}")
    print(f"  API errors    : {stats['errors']}")
    print(f"{'═' * 55}")

    if pairs:
        with open(DATASET_PATH) as f:
            ds = json.load(f)
        print("\n  Generating charts and examples...")
        visualizer.run_all(ds)
        print("\n  ✅ outputs/ ready for mentor presentation")


if __name__ == "__main__":
    run()