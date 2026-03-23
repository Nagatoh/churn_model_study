from __future__ import annotations

import numpy as np

from churn_model.inference import predict_records, prepare_inference_frame
from churn_model.training import PersistedModel


class FakeProbabilityModel:
    def predict_proba(self, features):
        assert "tenure_group" in features.columns
        return np.array([[0.35, 0.65]])


def test_prepare_inference_frame_keeps_expected_columns(
    sample_customer_record: dict[str, object],
) -> None:
    frame = prepare_inference_frame(
        [sample_customer_record],
        include_tenure_group=True,
    )

    assert "tenure_group" in frame.columns
    assert "ChurnFlag" not in frame.columns
    assert "Churn" not in frame.columns


def test_predict_records_uses_promoted_threshold(
    monkeypatch,
    sample_customer_record: dict[str, object],
) -> None:
    artifact = PersistedModel(
        model=FakeProbabilityModel(),
        variant_name="logreg_without_tenure_group",
        model_family="logistic_regression",
        include_tenure_group=True,
        threshold=0.60,
        dataset_hash="abc123",
        trained_at_utc="2026-03-22T00:00:00+00:00",
        metrics={},
    )
    monkeypatch.setattr("churn_model.inference.load_persisted_model", lambda _=None: artifact)

    predictions = predict_records([sample_customer_record])

    assert len(predictions) == 1
    assert predictions[0].churn_prediction == 1
    assert predictions[0].threshold == 0.60
    assert predictions[0].variant_name == "logreg_without_tenure_group"
    assert predictions[0].dataset_hash == "abc123"
