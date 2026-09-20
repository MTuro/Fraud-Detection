from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.15
TARGET = "is_fraud"

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "fraud_model.joblib"
RESULTS_DIR = ROOT / "results"
