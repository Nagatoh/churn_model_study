from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from churn_model.config import DEFAULT_THRESHOLD


class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float | str

    model_config = ConfigDict(extra="forbid")


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    threshold: float = DEFAULT_THRESHOLD
    variant_name: str
    dataset_hash: str
