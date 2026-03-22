from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from churn_model.config import DATA_PATH

TENURE_GROUP_COLUMN = "tenure_group"
TARGET_COLUMN = "ChurnFlag"
TARGET_LABEL_COLUMN = "Churn"
ID_COLUMN = "customerID"

RAW_FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


def dataset_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw_dataset(path: Path | None = None) -> pd.DataFrame:
    dataset_path = path or DATA_PATH
    return pd.read_csv(dataset_path)


def clean_telco_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    if TARGET_LABEL_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_LABEL_COLUMN].map({"No": 0, "Yes": 1})
    return df


def add_tenure_group(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    df[TENURE_GROUP_COLUMN] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12m", "13-24m", "25-48m", "49-72m"],
        include_lowest=True,
    )
    return df


def build_model_frame(
    dataframe: pd.DataFrame,
    *,
    include_tenure_group: bool,
    include_target: bool = True,
) -> pd.DataFrame:
    df = clean_telco_dataframe(dataframe)
    if include_tenure_group:
        df = add_tenure_group(df)
    columns = RAW_FEATURE_COLUMNS.copy()
    if include_tenure_group:
        columns.append(TENURE_GROUP_COLUMN)
    if include_target and TARGET_COLUMN in df.columns:
        columns.extend([TARGET_LABEL_COLUMN, TARGET_COLUMN])
    return df.drop(columns=[ID_COLUMN], errors="ignore")[columns]

