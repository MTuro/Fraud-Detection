import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.class_weight import compute_sample_weight

from src.config import (
    MODEL_PATH,
    RANDOM_STATE,
    RESULTS_DIR,
    TARGET,
    TEST_SIZE,
    VALIDATION_SIZE,
)
from src.data.collect import RAW_FEATURES, generate_transactions, load_transactions
from src.evaluate import (
    choose_threshold,
    metrics_at_threshold,
    save_dataset_plots,
    save_evaluation_plots,
)
from src.features import add_features
from src.preprocessing import build_preprocessor


def split_data(data: pd.DataFrame):
    features, target = data[RAW_FEATURES], data[TARGET]
    train_x, temporary_x, train_y, temporary_y = train_test_split(
        features, target, test_size=TEST_SIZE + VALIDATION_SIZE, stratify=target, random_state=RANDOM_STATE
    )
    validation_x, test_x, validation_y, test_y = train_test_split(
        temporary_x, temporary_y, test_size=TEST_SIZE / (TEST_SIZE + VALIDATION_SIZE),
        stratify=temporary_y, random_state=RANDOM_STATE,
    )
    return train_x, validation_x, test_x, train_y, validation_y, test_y


def build_pipeline(model) -> Pipeline:
    return Pipeline([
        ("features", FunctionTransformer(add_features, validate=False)),
        ("preprocessing", build_preprocessor()),
        ("model", model),
    ])


def train(data: pd.DataFrame, model_path: Path = MODEL_PATH, results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    train_x, validation_x, test_x, train_y, validation_y, test_y = split_data(data)
    models = {
        "Majority baseline": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1_000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    fitted, rows, validation_scores = {}, [], {}
    results_dir.mkdir(parents=True, exist_ok=True)
    for name, estimator in models.items():
        fit_params = {"model__sample_weight": compute_sample_weight("balanced", train_y)} if name == "Gradient Boosting" else {}
        pipeline = build_pipeline(estimator).fit(train_x, train_y, **fit_params)
        validation_probability = pipeline.predict_proba(validation_x)[:, 1]
        threshold, threshold_table = choose_threshold(validation_y, validation_probability)
        test_probability = pipeline.predict_proba(test_x)[:, 1]
        metrics = metrics_at_threshold(test_y, test_probability, threshold)
        rows.append({"model": name, "threshold": threshold, **metrics})
        fitted[name] = pipeline
        validation_scores[name] = metrics_at_threshold(validation_y, validation_probability, threshold)["pr_auc"]
        threshold_table.to_csv(results_dir / f"thresholds_{name.lower().replace(' ', '_')}.csv", index=False)

    results = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    best_name = max(validation_scores, key=validation_scores.get)
    best_row = results.loc[results["model"].eq(best_name)].iloc[0]
    best_model = fitted[best_name]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": best_model, "threshold": best_row["threshold"], "model_name": best_name}, model_path)
    results.to_csv(results_dir / "model_comparison.csv", index=False)
    save_evaluation_plots(test_y, best_model.predict_proba(test_x)[:, 1], best_row["threshold"], results_dir)
    save_dataset_plots(data, best_model, results_dir)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate fraud classifiers.")
    parser.add_argument("--data", type=Path, help="CSV input; synthetic data is used when omitted.")
    parser.add_argument("--samples", type=int, default=50_000, help="Synthetic sample count.")
    args = parser.parse_args()
    data = load_transactions(args.data) if args.data else generate_transactions(args.samples)
    results = train(data)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
