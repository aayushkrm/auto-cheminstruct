import warnings
warnings.filterwarnings("ignore")

import json, time
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME, TEMPERATURE_LOW, API_CALL_DELAY

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)

SYSTEM = (
    "You are a senior synthetic organic chemist. "
    "You write precise, mechanistically rigorous causal explanations "
    "of why proposed reactions fail, using IUPAC terminology."
)

PROMPT = """A retrosynthetic route has failed automated chemistry validation.
Write a precise CAUSAL FAILURE TRACE.

Target: {target_name} ({target_smiles})
Route description: {description}
Transformation: {transformation}
Reactants: {reactants}
Conditions: {conditions}
Reaction SMILES: {reaction_smiles}
Validation errors detected:
{errors}

Respond ONLY in this JSON (no markdown):
{{
  "causal_analysis": "2-4 sentences explaining root cause in precise mechanistic terms",
  "primary_failure_mode": "one of: steric_hindrance | electronic_incompatibility | atom_imbalance | valence_violation | functional_group_clash | thermodynamic_infeasibility | regiochemistry_error | stereochemistry_conflict | reagent_incompatibility | wrong_product | other",
  "problematic_feature": "the specific atom, bond, or functional group causing failure",
  "chemical_principle_violated": "name the specific rule violated",
  "severity": "fatal | major | minor",
  "educational_note": "one sentence a chemistry student should remember",
  "fix_suggestion": "minimal change that would make this route viable"
}}"""

def generate_failure_trace(target: dict, route: dict, validation: dict) -> dict:
    errors_str = "\n".join(
        f"  [{i+1}] {e['type']}: {e['detail']}"
        for i, e in enumerate(validation["errors"])
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": PROMPT.format(
                    target_name=target["name"],
                    target_smiles=target["smiles"],
                    description=route.get("description", "N/A"),
                    transformation=route.get("key_transformation", "N/A"),
                    reactants=", ".join(route.get("reactants", [])),
                    conditions=route.get("conditions", "N/A"),
                    reaction_smiles=route.get("reaction_smiles", "N/A"),
                    errors=errors_str or "  (no specific errors — route flagged as low confidence)",
                )},
            ],
            temperature=TEMPERATURE_LOW,
            response_format={"type": "json_object"},
        )
        time.sleep(API_CALL_DELAY)
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"    [reflection_agent] Error: {e}")
        return {
            "causal_analysis": "; ".join(validation["error_details"]),
            "primary_failure_mode": "other",
            "problematic_feature": "unknown",
            "chemical_principle_violated": "unknown",
            "severity": "fatal",
            "educational_note": "",
            "fix_suggestion": "",
        }