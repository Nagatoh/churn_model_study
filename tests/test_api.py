from __future__ import annotations

from fastapi.testclient import TestClient

from churn_model import api
from churn_model.schemas import PredictionResponse


def test_health_reflects_model_file_presence(monkeypatch, tmp_path) -> None:
    model_path = tmp_path / "churn_model.joblib"
    monkeypatch.setattr(api, "MODEL_PATH", model_path)
    client = TestClient(api.app)

    missing_response = client.get("/health")
    assert missing_response.status_code == 200
    assert missing_response.json()["status"] == "model_not_found"

    model_path.write_text("artifact", encoding="utf-8")

    ready_response = client.get("/health")
    assert ready_response.status_code == 200
    assert ready_response.json()["status"] == "ok"


def test_predict_endpoint_returns_model_response(
    monkeypatch,
    tmp_path,
    sample_customer_record: dict[str, object],
) -> None:
    model_path = tmp_path / "churn_model.joblib"
    model_path.write_text("artifact", encoding="utf-8")
    monkeypatch.setattr(api, "MODEL_PATH", model_path)

    def fake_predict_records(records, *, model_path=None):
        assert model_path == api.MODEL_PATH
        assert records[0]["Contract"] == "Month-to-month"
        return [
            PredictionResponse(
                churn_probability=0.83,
                churn_prediction=1,
                threshold=0.55,
                variant_name="logreg_without_tenure_group",
                dataset_hash="abc123",
            )
        ]

    monkeypatch.setattr(api, "predict_records", fake_predict_records)
    client = TestClient(api.app)

    response = client.post("/predict", json=sample_customer_record)

    assert response.status_code == 200
    assert response.json()["churn_prediction"] == 1
    assert response.json()["variant_name"] == "logreg_without_tenure_group"


def test_predict_endpoint_rejects_missing_required_field(
    sample_customer_record: dict[str, object],
) -> None:
    client = TestClient(api.app)
    invalid_payload = sample_customer_record.copy()
    invalid_payload.pop("Contract")

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422
    assert "Contract" in response.text


def test_predict_endpoint_rejects_extra_field(
    sample_customer_record: dict[str, object],
) -> None:
    client = TestClient(api.app)
    invalid_payload = sample_customer_record | {"unexpected_field": "boom"}

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422
    assert "unexpected_field" in response.text


def test_predict_endpoint_rejects_invalid_field_shape(
    sample_customer_record: dict[str, object],
) -> None:
    client = TestClient(api.app)
    invalid_payload = sample_customer_record.copy()
    invalid_payload["MonthlyCharges"] = {"value": 29.85}

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422
    assert "MonthlyCharges" in response.text
