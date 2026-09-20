import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from src.data.collect import RAW_FEATURES, generate_transactions
from src.predict import predict
from src.train import build_pipeline, cross_validated_probabilities, split_data, train


def test_split_is_85_15():
    parts = split_data(generate_transactions(1_000, fraud_ratio=.1))
    assert [len(part) for part in parts] == [850, 150, 850, 150]


def test_cross_validation_predicts_every_training_row():
    data = generate_transactions(500, fraud_ratio=.1)
    probabilities = cross_validated_probabilities(DummyClassifier(strategy="prior"), data[RAW_FEATURES], data["is_fraud"])
    assert len(probabilities) == len(data)
    assert np.isfinite(probabilities).all()


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


def test_artifact_matches_best_cross_validated_model(tmp_path):
    model_path = tmp_path / "model.joblib"
    results = train(generate_transactions(500, fraud_ratio=.1), model_path, tmp_path / "results")
    artifact = joblib.load(model_path)
    assert artifact["model_name"] == results.iloc[0]["model"]
    assert artifact["threshold"] == results.iloc[0]["threshold"]
