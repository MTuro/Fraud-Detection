from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def threshold_scores(y_true, probabilities, thresholds=None) -> pd.DataFrame:
    thresholds = np.arange(.10, .91, .05) if thresholds is None else thresholds
    rows = []
    for threshold in thresholds:
        predicted = np.asarray(probabilities) >= threshold
        rows.append({"threshold": round(float(threshold), 2), "precision": precision_score(y_true, predicted, zero_division=0), "recall": recall_score(y_true, predicted, zero_division=0), "f1": f1_score(y_true, predicted, zero_division=0)})
    return pd.DataFrame(rows)


def choose_threshold(y_true, probabilities) -> tuple[float, pd.DataFrame]:
    scores = threshold_scores(y_true, probabilities)
    return float(scores.loc[scores["f1"].idxmax(), "threshold"]), scores


def metrics_at_threshold(y_true, probabilities, threshold: float) -> dict[str, float]:
    predicted = np.asarray(probabilities) >= threshold
    return {"precision": precision_score(y_true, predicted, zero_division=0), "recall": recall_score(y_true, predicted, zero_division=0), "f1": f1_score(y_true, predicted, zero_division=0), "pr_auc": average_precision_score(y_true, probabilities), "roc_auc": roc_auc_score(y_true, probabilities)}


def save_evaluation_plots(y_true, probabilities, threshold: float, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predicted = np.asarray(probabilities) >= threshold
    ConfusionMatrixDisplay.from_predictions(y_true, predicted, display_labels=["Non-fraud", "Fraud"], cmap="Blues")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=160)
    plt.close()
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probabilities)
    plt.plot(false_positive_rate, true_positive_rate)
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close()

    tn, fp, fn, tp = confusion_matrix(y_true, predicted).ravel()
    pd.DataFrame({"true_negative": [tn], "false_positive": [fp], "false_negative": [fn], "true_positive": [tp]}).to_csv(
        output_dir / "error_analysis.csv", index=False
    )


def save_dataset_plots(data: pd.DataFrame, pipeline, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    counts = data["is_fraud"].value_counts().sort_index()
    counts.plot.bar(color=["steelblue", "tomato"])
    plt.xticks([0, 1], ["Non-fraud", "Fraud"], rotation=0)
    plt.ylabel("Transactions")
    plt.title("Class distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=160)
    plt.close()

    model = pipeline.named_steps["model"]
    importance = getattr(model, "feature_importances_", None)
    if importance is None:
        return
    names = pipeline.named_steps["preprocessing"].get_feature_names_out()
    top = pd.Series(importance, index=names).nlargest(12).sort_values()
    top.plot.barh()
    plt.xlabel("Importance")
    plt.title("Top feature importances")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=160)
    plt.close()
