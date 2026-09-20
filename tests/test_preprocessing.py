import numpy as np

from src.data.collect import RAW_FEATURES, generate_transactions
from src.features import add_features
from src.preprocessing import build_preprocessor


def test_preprocessor_handles_missing_values():
    data = add_features(generate_transactions(200)[RAW_FEATURES])
    data.loc[0, "valor_transacao"] = np.nan
    transformed = build_preprocessor().fit_transform(data)
    assert transformed.shape[0] == 200
    assert np.isfinite(transformed).all()


def test_preprocessor_accepts_unseen_category():
    data = add_features(generate_transactions(200)[RAW_FEATURES])
    preprocessor = build_preprocessor().fit(data)
    unseen = data.iloc[[0]].copy()
    unseen["canal_transacao"] = 99
    assert preprocessor.transform(unseen).shape[0] == 1


def test_generated_data_is_reproducible():
    assert generate_transactions(100).equals(generate_transactions(100))


def test_generated_data_is_imbalanced():
    assert generate_transactions(1_000, fraud_ratio=.02)["is_fraud"].mean() == .02
