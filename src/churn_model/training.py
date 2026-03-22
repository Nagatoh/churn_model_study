from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from shutil import copy2
from typing import Any

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_model.config import (
    DATA_PATH,
    MLFLOW_ARTIFACTS_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_DIR,
    PRODUCTION_METADATA_PATH,
    PRODUCTION_MODEL_PATH,
)
from churn_model.data import TARGET_COLUMN, build_model_frame, dataset_sha256, load_raw_dataset


@dataclass
class PersistedModel:
    model: Any
    variant_name: str
    include_tenure_group: bool
    threshold: float
    dataset_hash: str
    trained_at_utc: str
    metrics: dict[str, float]


@dataclass
class TrainingResult:
    variant_name: str
    include_tenure_group: bool
    dropped_columns: list[str]
    run_id: str
    model: Any
    threshold: float
    dataset_hash: str
    metrics: dict[str, float]
    report_dict: dict[str, Any]
    threshold_df: pd.DataFrame
    best_threshold_row: pd.Series
    top_coefficients: pd.DataFrame
    model_path: Path
    metadata_path: Path


def git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def dvc_metadata_for_dataset(dataset_path: Path) -> dict[str, str]:
    dvc_path = dataset_path.with_name(f"{dataset_path.name}.dvc")
    if not dvc_path.exists():
        return {
            "dataset_dvc_file": "missing",
            "dataset_dvc_hash_name": "missing",
            "dataset_dvc_hash_value": "missing",
            "dataset_dvc_tracked_relpath": "missing",
        }

    # DVC `.dvc` files are YAML-like but simple enough here to parse line by line.
    hash_name = "unknown"
    hash_value = "unknown"
    tracked_relpath = "unknown"

    for raw_line in dvc_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("- md5:"):
            hash_name = "md5"
            hash_value = line.split(":", 1)[1].strip()
        elif line.startswith("md5:"):
            hash_name = "md5"
            hash_value = line.split(":", 1)[1].strip()
        elif line.startswith("hash:"):
            hash_name = line.split(":", 1)[1].strip()
        elif line.startswith("path:"):
            tracked_relpath = line.split(":", 1)[1].strip()

    return {
        "dataset_dvc_file": str(dvc_path),
        "dataset_dvc_hash_name": hash_name,
        "dataset_dvc_hash_value": hash_value,
        "dataset_dvc_tracked_relpath": tracked_relpath,
    }


def dataset_digest_for_mlflow(dataset_hash: str, dvc_metadata: dict[str, str]) -> str:
    dvc_hash_value = dvc_metadata.get("dataset_dvc_hash_value", "missing")
    if dvc_hash_value not in {"missing", "unknown"} and len(dvc_hash_value) <= 36:
        return dvc_hash_value
    return dataset_hash[:36]


def configure_mlflow(
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> str:
    MLFLOW_ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=MLFLOW_ARTIFACTS_PATH.as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id
    return experiment_id


def build_pipeline(X_train: pd.DataFrame) -> Pipeline:
    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
            ),
        ]
    )


def compute_threshold_table(
    y_true: pd.Series,
    y_score: pd.Series,
    threshold_grid: list[float],
) -> tuple[pd.DataFrame, pd.Series]:
    rows: list[dict[str, float]] = []
    for threshold in threshold_grid:
        prediction = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, prediction).ravel()
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": precision_score(y_true, prediction, zero_division=0),
                "recall": recall_score(y_true, prediction, zero_division=0),
                "f1": f1_score(y_true, prediction, zero_division=0),
                "false_positives": float(fp),
                "false_negatives": float(fn),
                "predicted_positive_rate": float(prediction.mean()),
            }
        )
    threshold_df = pd.DataFrame(rows)
    best_row = threshold_df.loc[threshold_df["f1"].idxmax()]
    return threshold_df, best_row


def top_coefficients_dataframe(model: Pipeline, limit: int = 15) -> pd.DataFrame:
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = model.named_steps["classifier"].coef_[0]
    return (
        pd.DataFrame({"feature": feature_names, "coef": coefficients})
        .assign(abs_coef=lambda df: df["coef"].abs())
        .sort_values("abs_coef", ascending=False)
        .head(limit)
    )


def cross_validation_metrics(X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }
    cv_results = cross_validate(
        build_pipeline(X_train),
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=None,
    )
    return {
        "cv_roc_auc_mean": float(np.mean(cv_results["test_roc_auc"])),
        "cv_roc_auc_std": float(np.std(cv_results["test_roc_auc"])),
        "cv_f1_mean": float(np.mean(cv_results["test_f1"])),
        "cv_f1_std": float(np.std(cv_results["test_f1"])),
        "cv_precision_mean": float(np.mean(cv_results["test_precision"])),
        "cv_precision_std": float(np.std(cv_results["test_precision"])),
        "cv_recall_mean": float(np.mean(cv_results["test_recall"])),
        "cv_recall_std": float(np.std(cv_results["test_recall"])),
    }


def save_figures(
    *,
    y_true: pd.Series,
    y_pred: pd.Series,
    y_score: pd.Series,
    top_coefficients: pd.DataFrame,
    threshold_df: pd.DataFrame,
    variant_name: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=axes[0], colorbar=False)
    axes[0].set_title(f"Confusion matrix: {variant_name}")
    RocCurveDisplay.from_predictions(y_true, y_score, ax=axes[1])
    axes[1].set_title(f"ROC curve: {variant_name}")
    plt.tight_layout()
    mlflow.log_figure(fig, f"figures/{variant_name}_evaluation_overview.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top_coefficients, y="feature", x="coef", ax=ax)
    ax.set_title(f"Top 15 coeficientes: {variant_name}")
    plt.tight_layout()
    mlflow.log_figure(fig, f"figures/{variant_name}_top_coefficients.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(threshold_df["threshold"], threshold_df["precision"], marker="o", label="precision")
    axes[0].plot(threshold_df["threshold"], threshold_df["recall"], marker="o", label="recall")
    axes[0].plot(threshold_df["threshold"], threshold_df["f1"], marker="o", label="f1")
    axes[0].set_title(f"Trade-off por threshold: {variant_name}")
    axes[0].set_xlabel("threshold")
    axes[0].set_ylabel("score")
    axes[0].legend()

    axes[1].plot(
        threshold_df["threshold"],
        threshold_df["false_positives"],
        marker="o",
        label="false_positives",
    )
    axes[1].plot(
        threshold_df["threshold"],
        threshold_df["false_negatives"],
        marker="o",
        label="false_negatives",
    )
    axes[1].set_title(f"Erros por threshold: {variant_name}")
    axes[1].set_xlabel("threshold")
    axes[1].set_ylabel("quantidade")
    axes[1].legend()
    plt.tight_layout()
    mlflow.log_figure(fig, f"figures/{variant_name}_threshold_tradeoff.png")
    plt.close(fig)


def persist_model(
    *,
    model: Pipeline,
    variant_name: str,
    include_tenure_group: bool,
    threshold: float,
    dataset_hash: str,
    metrics: dict[str, float],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{variant_name}.joblib"
    metadata_path = output_dir / f"{variant_name}_metadata.json"

    artifact = PersistedModel(
        model=model,
        variant_name=variant_name,
        include_tenure_group=include_tenure_group,
        threshold=threshold,
        dataset_hash=dataset_hash,
        trained_at_utc=datetime.now(UTC).isoformat(),
        metrics=metrics,
    )
    joblib.dump(artifact, model_path)
    metadata = {
        "variant_name": artifact.variant_name,
        "include_tenure_group": artifact.include_tenure_group,
        "threshold": artifact.threshold,
        "dataset_hash": artifact.dataset_hash,
        "trained_at_utc": artifact.trained_at_utc,
        "metrics": artifact.metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path, metadata_path


def train_variant(
    *,
    dataframe: pd.DataFrame,
    dataset_path: Path,
    dataset_hash: str,
    experiment_id: str,
    variant_name: str,
    include_tenure_group: bool,
    output_dir: Path,
) -> TrainingResult:
    dvc_metadata = dvc_metadata_for_dataset(dataset_path)
    dataset_digest = dataset_digest_for_mlflow(dataset_hash, dvc_metadata)
    commit_hash = git_commit_hash()

    model_frame = build_model_frame(
        dataframe,
        include_tenure_group=include_tenure_group,
        include_target=True,
    )
    X = model_frame.drop(columns=["Churn", TARGET_COLUMN])
    y = model_frame[TARGET_COLUMN]
    training_dataset_df = X.copy()
    training_dataset_df[TARGET_COLUMN] = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    model = build_pipeline(X_train)
    cv_metrics = cross_validation_metrics(X_train, y_train)

    with mlflow.start_run(experiment_id=experiment_id, run_name=f"logreg_{variant_name}") as run:
        mlflow.set_tags(
            {
                "dataset": "WA_Fn-UseC_-Telco-Customer-Churn.csv",
                "model_family": "logistic_regression",
                "feature_variant": variant_name,
                "include_tenure_group": str(include_tenure_group).lower(),
                "git_commit": commit_hash,
                "dataset_dvc_file": dvc_metadata["dataset_dvc_file"],
                "dataset_dvc_hash_name": dvc_metadata["dataset_dvc_hash_name"],
                "dataset_dvc_hash_value": dvc_metadata["dataset_dvc_hash_value"],
            }
        )
        mlflow.log_params(
            {
                "dataset_hash": dataset_hash,
                "dataset_path": str(dataset_path),
                "dataset_dvc_tracked_relpath": dvc_metadata["dataset_dvc_tracked_relpath"],
                "test_size": 0.2,
                "random_state": 42,
                "classifier": "LogisticRegression",
                "classifier__max_iter": 2000,
                "classifier__class_weight": "balanced",
                "cv_fold_count": 5,
                "numeric_feature_count": len(X_train.select_dtypes(include=["number"]).columns),
                "categorical_feature_count": len(X_train.select_dtypes(exclude=["number"]).columns),
            }
        )
        mlflow_dataset = mlflow.data.from_pandas(
            training_dataset_df,
            source=dataset_path.resolve().as_uri(),
            targets=TARGET_COLUMN,
            name=f"telco_churn_{variant_name}",
            digest=dataset_digest,
        )
        mlflow.log_input(
            mlflow_dataset,
            context="training",
            tags={
                "feature_variant": variant_name,
                "dataset_dvc_hash_name": dvc_metadata["dataset_dvc_hash_name"],
                "dataset_dvc_hash_value": dvc_metadata["dataset_dvc_hash_value"],
            },
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]

        report_dict = classification_report(y_test, y_pred, output_dict=True, digits=3)
        threshold_df, best_threshold_row = compute_threshold_table(
            y_test,
            y_score,
            threshold_grid=[round(value, 2) for value in np.arange(0.20, 0.81, 0.05)],
        )

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_score),
            "churn_precision": report_dict["1"]["precision"],
            "churn_recall": report_dict["1"]["recall"],
            "churn_f1": report_dict["1"]["f1-score"],
            "best_threshold_by_f1": float(best_threshold_row["threshold"]),
            "best_threshold_precision": float(best_threshold_row["precision"]),
            "best_threshold_recall": float(best_threshold_row["recall"]),
            "best_threshold_f1": float(best_threshold_row["f1"]),
        }
        metrics.update(cv_metrics)
        mlflow.log_metrics(metrics)

        top_coefficients = top_coefficients_dataframe(model)
        save_figures(
            y_true=y_test,
            y_pred=y_pred,
            y_score=y_score,
            top_coefficients=top_coefficients,
            threshold_df=threshold_df,
            variant_name=variant_name,
        )
        mlflow.sklearn.log_model(model, artifact_path="model")

        model_path, metadata_path = persist_model(
            model=model,
            variant_name=variant_name,
            include_tenure_group=include_tenure_group,
            threshold=float(best_threshold_row["threshold"]),
            dataset_hash=dataset_hash,
            metrics=metrics,
            output_dir=output_dir,
        )

        return TrainingResult(
            variant_name=variant_name,
            include_tenure_group=include_tenure_group,
            dropped_columns=[] if include_tenure_group else ["tenure_group"],
            run_id=run.info.run_id,
            model=model,
            threshold=float(best_threshold_row["threshold"]),
            dataset_hash=dataset_hash,
            metrics=metrics,
            report_dict=report_dict,
            threshold_df=threshold_df,
            best_threshold_row=best_threshold_row,
            top_coefficients=top_coefficients,
            model_path=model_path,
            metadata_path=metadata_path,
        )


def comparison_table(results: list[TrainingResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variant": result.variant_name,
                "run_id": result.run_id,
                "include_tenure_group": result.include_tenure_group,
                "accuracy": result.metrics["accuracy"],
                "roc_auc": result.metrics["roc_auc"],
                "churn_precision": result.metrics["churn_precision"],
                "churn_recall": result.metrics["churn_recall"],
                "churn_f1": result.metrics["churn_f1"],
                "cv_roc_auc_mean": result.metrics["cv_roc_auc_mean"],
                "cv_f1_mean": result.metrics["cv_f1_mean"],
                "best_threshold_by_f1": result.metrics["best_threshold_by_f1"],
            }
            for result in results
        ]
    ).sort_values("variant")


def promote_result(result: TrainingResult) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    copy2(result.model_path, PRODUCTION_MODEL_PATH)
    copy2(result.metadata_path, PRODUCTION_METADATA_PATH)


def train_production_pipeline(
    *,
    dataset_path: Path = DATA_PATH,
    selected_variant: str = "without_tenure_group",
    output_dir: Path | None = None,
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> tuple[list[TrainingResult], pd.DataFrame]:
    dataset = load_raw_dataset(dataset_path)
    dataset_hash = dataset_sha256(dataset_path)
    experiment_id = configure_mlflow(tracking_uri=tracking_uri, experiment_name=experiment_name)
    mlflow.end_run()

    training_output_dir = output_dir or MODEL_DIR
    results = [
        train_variant(
            dataframe=dataset,
            dataset_path=dataset_path,
            dataset_hash=dataset_hash,
            experiment_id=experiment_id,
            variant_name="with_tenure_group",
            include_tenure_group=True,
            output_dir=training_output_dir,
        ),
        train_variant(
            dataframe=dataset,
            dataset_path=dataset_path,
            dataset_hash=dataset_hash,
            experiment_id=experiment_id,
            variant_name="without_tenure_group",
            include_tenure_group=False,
            output_dir=training_output_dir,
        ),
    ]
    summary = comparison_table(results)
    selected = next(result for result in results if result.variant_name == selected_variant)
    promote_result(selected)
    return results, summary
