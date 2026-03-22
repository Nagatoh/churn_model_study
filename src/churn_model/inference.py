from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from churn_model.config import DEFAULT_THRESHOLD, PRODUCTION_METADATA_PATH, PRODUCTION_MODEL_PATH
from churn_model.data import build_model_frame
from churn_model.schemas import PredictionResponse
from churn_model.training import PersistedModel


def load_persisted_model(model_path: Path | None = None) -> PersistedModel:
    path = model_path or PRODUCTION_MODEL_PATH
    return joblib.load(path)


def read_metadata(metadata_path: Path | None = None) -> dict:
    path = metadata_path or PRODUCTION_METADATA_PATH
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_inference_frame(records: list[dict], *, include_tenure_group: bool) -> pd.DataFrame:
    raw_df = pd.DataFrame(records)
    return build_model_frame(
        raw_df,
        include_tenure_group=include_tenure_group,
        include_target=False,
    )


def predict_records(
    records: list[dict],
    *,
    model_path: Path | None = None,
) -> list[PredictionResponse]:
    artifact = load_persisted_model(model_path)
    threshold = artifact.threshold or DEFAULT_THRESHOLD
    features = prepare_inference_frame(records, include_tenure_group=artifact.include_tenure_group)
    probabilities = artifact.model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    responses = []
    for probability, prediction in zip(probabilities, predictions, strict=True):
        responses.append(
            PredictionResponse(
                churn_probability=float(probability),
                churn_prediction=int(prediction),
                threshold=float(threshold),
                variant_name=artifact.variant_name,
                dataset_hash=artifact.dataset_hash,
            )
        )
    return responses
