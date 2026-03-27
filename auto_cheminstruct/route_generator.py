import warnings
warnings.filterwarnings("ignore")

import json, time
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME, TEMPERATURE_HIGH, API_CALL_DELAY, MAX_ROUTES

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)

SYSTEM_PROMPT = (
    "You are an expert organic chemist specializing in retrosynthetic analysis. "
    "You generate structured reaction routes in valid JSON only. "
    "You deliberately include both valid and flawed routes when asked."
)

ROUTE_PROMPT = """Propose {n} retrosynthetic routes for this target molecule:

Name:   {name}
SMILES: {smiles}

IMPORTANT — vary quality deliberately:
- 1-2 routes must be chemically valid and feasible
- 1-2 routes must contain realistic flaws such as:
  * Atom imbalance in the reaction SMILES (different atom counts left/right of >>)
  * Incompatible functional groups (e.g. Grignard reagent + water/alcohol)
  * Wrong leaving group for the stated mechanism
  * Sterically impossible transformation

Return ONLY valid JSON, no markdown:
{{
  "routes": [
    {{
      "route_id": 1,
      "description": "One sentence describing the synthetic strategy",
      "key_transformation": "Named reaction (e.g. Fischer esterification)",
      "reactants": ["SMILES_1", "SMILES_2"],
      "reagents": ["reagent_1"],
      "solvent": "solvent name",
      "temperature": "e.g. 80°C",
      "conditions": "full conditions description",
      "reaction_smiles": "reactant1.reactant2>>product",
      "expected_yield": "e.g. 75-85%",
      "confidence": "high | medium | low"
    }}
  ]
}}"""


def generate_routes(molecule: dict) -> list[dict]:
    prompt = ROUTE_PROMPT.format(
        n=MAX_ROUTES,
        name=molecule["name"],
        smiles=molecule["smiles"],
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=TEMPERATURE_HIGH,
            response_format={"type": "json_object"},
        )
        routes = json.loads(resp.choices[0].message.content).get("routes", [])
        time.sleep(API_CALL_DELAY)
        return routes
    except Exception as e:
        print(f"    [route_generator] Error: {e}")
        return []