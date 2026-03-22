from __future__ import annotations

import pytest


@pytest.fixture
def sample_customer_record() -> dict[str, object]:
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": "29.85",
    }


@pytest.fixture
def sample_training_record(sample_customer_record: dict[str, object]) -> dict[str, object]:
    return {
        "customerID": "7590-VHVEG",
        "Churn": "No",
        **sample_customer_record,
    }
