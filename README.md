# Fraud Detection

An end-to-end **binary classification** project that predicts **fraud vs. non-fraud** in financial transactions. It demonstrates a reproducible ML workflow for imbalanced data: validated collection, in-pipeline feature engineering and preprocessing, train/validation/test isolation, baseline comparison, threshold selection, error analysis, and saved-model inference.

## Problem

Fraud detection is difficult because fraud is rare, false positives inconvenience legitimate customers, false negatives create financial loss, and fraud patterns can drift. Accuracy is therefore misleading: a classifier that labels every transaction as non-fraud reaches 99% accuracy on this dataset while detecting no fraud.

The project prioritizes fraud-class **precision**, **recall**, **F1**, **PR-AUC**, **ROC-AUC**, and the **confusion matrix**. PR-AUC is the main ranking metric because it focuses on the minority class.

## Dataset

The repository uses a reproducible synthetic dataset so it can be run without credentials or private financial data. The default experiment creates 50,000 transactions with 1% fraud and 12 raw features describing amount, time, recent activity, account age, card/channel, location, authentication failures, merchant category, and historical spending.

The generator encodes plausible differences between classes. This makes the repository useful for demonstrating methodology, not for claiming production fraud performance.

## Architecture

```mermaid
flowchart LR
    A[Raw or synthetic data] --> B[Schema validation]
    B --> C[70/15/15 split]
    C --> D[Feature engineering]
    D --> E[Imputation + encoding + scaling]
    E --> F[Model training]
    F --> G[Validation threshold selection]
    G --> H[One-time test evaluation]
    F --> I[Saved pipeline]
    I --> J[Inference]
```

Feature engineering and preprocessing live inside the scikit-learn `Pipeline`, so the same transformations run during training and inference and are fitted only on training data.

## Data collection

[`src/data/collect.py`](src/data/collect.py) generates deterministic synthetic transactions or loads a CSV, then validates its schema, identifiers, timestamp, and binary target. A custom CSV must contain the same columns as the generated data. Missing feature values are accepted and handled by the preprocessing pipeline.

## Preprocessing

- Numeric features: median imputation and standardization.
- Categorical features: most-frequent imputation and one-hot encoding with unknown-category support.
- IDs, timestamps, and the target never enter the model.

## Feature engineering

Three domain features are derived in [`src/features.py`](src/features.py): transaction-to-historical-value ratio, suspicious-hour flag (00:00–06:59), and weekend flag. The transformer does not mutate its input.

## Class imbalance

The majority-class dummy classifier establishes the no-skill baseline. Logistic Regression and Random Forest use balanced class weights; Gradient Boosting receives balanced sample weights. No synthetic oversampling is used. Every probabilistic model is evaluated over validation thresholds from 0.10 to 0.90.

## Models

The experiment intentionally compares only four useful reference points:

1. Majority baseline
2. Logistic Regression
3. Random Forest
4. Gradient Boosting

The best model is selected by validation PR-AUC. Its classification threshold is selected by validation F1, and the untouched test set is used once for the reported metrics.

## Evaluation and results

These results were produced by `python -m src.train --samples 50000` with seed 42 and a stratified 70%/15%/15% split (7,500 test transactions, including 75 frauds).

| Model | Threshold | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Gradient Boosting | 0.50 | 0.9740 | 1.0000 | 0.9868 | 0.9998 | 1.0000 |
| Random Forest | 0.25 | 0.9615 | 1.0000 | 0.9804 | 0.9981 | 1.0000 |
| Logistic Regression | 0.65 | 0.9737 | 0.9867 | 0.9801 | 0.9923 | 0.9999 |
| Majority baseline | 0.10 | 0.0000 | 0.0000 | 0.0000 | 0.0100 | 0.5000 |

The validation search selected Gradient Boosting at threshold **0.50**; this was an evaluated outcome, not an assumed default. Full threshold tables and raw metrics are stored in [`results/`](results/).

![Class distribution](results/class_distribution.png)

![Precision-Recall curve](results/precision_recall_curve.png)

![Confusion matrix](results/confusion_matrix.png)

![Feature importance](results/feature_importance.png)

## Error analysis

The selected model produced 7,422 true negatives, 3 false positives, 0 false negatives, and 75 true positives on the synthetic test set.

- A **false positive** can block a legitimate transaction, harm customer experience, and add review cost.
- A **false negative** lets fraud pass, creating financial and operational risk.

Optimizing F1 treats precision and recall symmetrically. A real deployment should select the threshold from business costs and capacity—for example, favoring higher recall when missed fraud is substantially more expensive than manual review.

## Running locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
```

To train from a compatible CSV:

```bash
python -m src.train --data path/to/transactions.csv
```

Training writes the selected pipeline to `models/fraud_model.joblib` and evaluation artifacts to `results/`. Model binaries and input data are intentionally not committed.

For inference, provide a JSON array containing the 12 raw features:

```bash
python -m src.predict transactions.json
```

The output contains a binary `prediction` and `fraud_probability`. See [`src/data/collect.py`](src/data/collect.py) for the exact input field names.

## Tests and quality

```bash
pip install -r requirements-dev.txt
ruff check .
python -m pytest -q
```

The 11 focused tests cover deterministic collection, imbalance, missing values, unknown categories, feature calculations, split proportions, pipeline probabilities, model loading, inference shape, and invalid inputs. GitHub Actions runs Ruff and pytest on pushes and pull requests.

## Project structure

```text
.
├── .github/workflows/tests.yml
├── models/                    # generated model (ignored)
├── notebooks/exploratory_analysis.ipynb
├── results/                   # metrics and evaluation plots
├── src/
│   ├── data/collect.py
│   ├── config.py
│   ├── evaluate.py
│   ├── features.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
├── tests/
├── requirements.txt
└── requirements-dev.txt
```

## Limitations

- Synthetic data is easier and cleaner than production transaction streams, so the high scores should not be interpreted as production readiness.
- The generated timestamp is not tied to behavioral drift; a random stratified split is appropriate here, while real temporal data should use chronological validation and backtesting.
- Fraud prevalence, feature availability, and error costs differ by institution and market.
- The model is not calibrated, monitored for drift, or evaluated for fairness.
- Threshold choice depends on business costs and review capacity, not metrics alone.

## Future work

The next valuable step is evaluation on a representative time-ordered dataset with cost-based threshold selection and drift monitoring. An API or Docker image should be added only when the model needs to be served; they do not improve the current ML evidence.

## License

MIT — see [`LICENSE`](LICENSE).
