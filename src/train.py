import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.class_weight import compute_sample_weight

from src.config import (
    MODEL_PATH,
    RANDOM_STATE,
    RESULTS_DIR,
    TARGET,
    TEST_SIZE,
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
    return train_test_split(
        features, target, test_size=TEST_SIZE, stratify=target, random_state=RANDOM_STATE
    )


def build_pipeline(model) -> Pipeline:
    return Pipeline([
        ("features", FunctionTransformer(add_features, validate=False)),
        ("preprocessing", build_preprocessor()),
        ("model", model),
    ])


def fit_model(estimator, features: pd.DataFrame, target: pd.Series) -> Pipeline:
    fit_params = {"model__sample_weight": compute_sample_weight("balanced", target)} if isinstance(estimator, GradientBoostingClassifier) else {}
    return build_pipeline(clone(estimator)).fit(features, target, **fit_params)


def cross_validated_probabilities(estimator, features: pd.DataFrame, target: pd.Series) -> np.ndarray:
    probabilities = np.full(len(target), np.nan)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for train_indices, validation_indices in folds.split(features, target):
        pipeline = fit_model(estimator, features.iloc[train_indices], target.iloc[train_indices])
        probabilities[validation_indices] = pipeline.predict_proba(features.iloc[validation_indices])[:, 1]
    assert np.isfinite(probabilities).all()
    return probabilities


def train(data: pd.DataFrame, model_path: Path = MODEL_PATH, results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    train_x, test_x, train_y, test_y = split_data(data)
    models = {
        "Majority baseline": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1_000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    fitted, rows = {}, []
    results_dir.mkdir(parents=True, exist_ok=True)
    for name, estimator in models.items():
        cv_probability = cross_validated_probabilities(estimator, train_x, train_y)
        threshold, threshold_table = choose_threshold(train_y, cv_probability)
        cv_pr_auc = metrics_at_threshold(train_y, cv_probability, threshold)["pr_auc"]
        pipeline = fit_model(estimator, train_x, train_y)
        test_probability = pipeline.predict_proba(test_x)[:, 1]
        metrics = metrics_at_threshold(test_y, test_probability, threshold)
        rows.append({"model": name, "threshold": threshold, "cv_pr_auc": cv_pr_auc, **{f"test_{key}": value for key, value in metrics.items()}})
        fitted[name] = pipeline
        threshold_table.to_csv(results_dir / f"thresholds_{name.lower().replace(' ', '_')}.csv", index=False)

    results = pd.DataFrame(rows).sort_values("cv_pr_auc", ascending=False).reset_index(drop=True)
    best_name = results.iloc[0]["model"]
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
