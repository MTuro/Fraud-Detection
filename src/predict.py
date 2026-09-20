import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import MODEL_PATH
from src.data.collect import RAW_FEATURES


def predict(data: pd.DataFrame, model_path: str | Path = MODEL_PATH) -> pd.DataFrame:
    missing = sorted(set(RAW_FEATURES) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required features: {', '.join(missing)}")
    artifact = joblib.load(model_path)
    probability = artifact["pipeline"].predict_proba(data[RAW_FEATURES])[:, 1]
    return pd.DataFrame({"prediction": (probability >= artifact["threshold"]).astype(int), "fraud_probability": probability}, index=data.index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score transaction records from a JSON file.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    records = json.loads(args.input.read_text(encoding="utf-8"))
    print(predict(pd.DataFrame(records), args.model).to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
