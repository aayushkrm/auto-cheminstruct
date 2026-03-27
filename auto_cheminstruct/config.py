import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY        = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL       = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME            = "gpt-4o"
TEMPERATURE_HIGH      = 0.7
TEMPERATURE_LOW       = 0.2
MAX_ROUTES            = 4
API_CALL_DELAY        = 1.5

OUTPUT_DIR            = "outputs"
DATASET_PATH          = f"{OUTPUT_DIR}/preference_pairs.json"
STATS_PATH            = f"{OUTPUT_DIR}/run_stats.json"
CHART_FAILURE_MODES   = f"{OUTPUT_DIR}/failure_modes.png"
CHART_MOLECULES       = f"{OUTPUT_DIR}/target_molecules.png"