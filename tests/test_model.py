import joblib
import pytest
from sklearn.dummy import DummyClassifier

from src.data.collect import RAW_FEATURES, generate_transactions
from src.predict import predict
from src.train import build_pipeline, split_data


def test_split_is_70_15_15():
    parts = split_data(generate_transactions(1_000, fraud_ratio=.1))
    assert [len(part) for part in parts[:3]] == [700, 150, 150]


def test_pipeline_predicts_probability():
    data = generate_transactions(500, fraud_ratio=.1)
    pipeline = build_pipeline(DummyClassifier(strategy="prior")).fit(data[RAW_FEATURES], data["is_fraud"])
    assert pipeline.predict_proba(data[RAW_FEATURES].head()).shape == (5, 2)


def test_saved_model_supports_inference(tmp_path):
    data = generate_transactions(500, fraud_ratio=.1)
    pipeline = build_pipeline(DummyClassifier(strategy="prior")).fit(data[RAW_FEATURES], data["is_fraud"])
    path = tmp_path / "model.joblib"
    joblib.dump({"pipeline": pipeline, "threshold": .5, "model_name": "test"}, path)
    result = predict(data.head(3), path)
    assert list(result.columns) == ["prediction", "fraud_probability"]
    assert len(result) == 3


def test_inference_rejects_missing_features(tmp_path):
    with pytest.raises(ValueError, match="Missing required features"):
        predict(generate_transactions(100)[["valor_transacao"]], tmp_path / "unused.joblib")
