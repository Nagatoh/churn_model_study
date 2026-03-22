from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI

from churn_model.config import MODEL_PATH_ENV_VAR, PRODUCTION_MODEL_PATH
from churn_model.inference import predict_records
from churn_model.schemas import CustomerFeatures, PredictionResponse

app = FastAPI(title="Churn Model API", version="0.1.0")
MODEL_PATH = Path(os.getenv(MODEL_PATH_ENV_VAR, str(PRODUCTION_MODEL_PATH)))


@app.get("/health")
def health() -> dict[str, str]:
    model_exists = MODEL_PATH.exists()
    status = "ok" if model_exists else "model_not_found"
    return {"status": status, "model_path": str(MODEL_PATH)}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CustomerFeatures) -> PredictionResponse:
    prediction = predict_records([payload.model_dump()], model_path=MODEL_PATH)[0]
    return prediction
