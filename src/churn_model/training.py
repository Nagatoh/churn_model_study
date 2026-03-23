from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
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
from mlflow.data.sources import LocalArtifactDatasetSource
from mlflow.models import infer_signature
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
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

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
    model_family: str
    include_tenure_group: bool
    threshold: float
    dataset_hash: str
    trained_at_utc: str
    metrics: dict[str, float]


@dataclass
class TrainingResult:
    variant_name: str
    model_family: str
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


@dataclass(frozen=True)
class ModelVariant:
    variant_name: str
    model_family: str
    include_tenure_group: bool


RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLD_COUNT = 5
TUNING_CV_FOLD_COUNT = 3
XGBOOST_RANDOM_SEARCH_ITERATIONS = 8
DEFAULT_SELECTION_METRIC = "cv_f1_mean"
SELECTION_METRICS = (
    "cv_f1_mean",
    "cv_roc_auc_mean",
    "churn_f1",
    "roc_auc",
    "accuracy",
)

MODEL_VARIANTS: tuple[ModelVariant, ...] = (
    ModelVariant(
        variant_name="logreg_with_tenure_group",
        model_family="logistic_regression",
        include_tenure_group=True,
    ),
    ModelVariant(
        variant_name="logreg_without_tenure_group",
        model_family="logistic_regression",
        include_tenure_group=False,
    ),
    ModelVariant(
        variant_name="xgboost_with_tenure_group",
        model_family="xgboost",
        include_tenure_group=True,
    ),
    ModelVariant(
        variant_name="xgboost_without_tenure_group",
        model_family="xgboost",
        include_tenure_group=False,
    ),
)

SUPPORTED_VARIANTS = tuple(variant.variant_name for variant in MODEL_VARIANTS)
PRODUCTION_VARIANT = ModelVariant(
    variant_name="xgboost_with_tenure_group",
    model_family="xgboost",
    include_tenure_group=True,
)
PRODUCTION_CLASSIFIER_PARAMS: dict[str, Any] = {
    "n_estimators": 250,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
}


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


def build_classifier(
    model_family: str,
    y_train: pd.Series,
    classifier_params: dict[str, Any] | None = None,
) -> Any:
    classifier_params = classifier_params or {}
    if model_family == "logistic_regression":
        params = {
            "max_iter": 4000,
            "class_weight": "balanced",
            "solver": "saga",
            "random_state": RANDOM_STATE,
        }
        params.update(classifier_params)
        return LogisticRegression(**params)

    if model_family == "xgboost":
        negative_class_count = int((y_train == 0).sum())
        positive_class_count = max(int((y_train == 1).sum()), 1)
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_estimators": 250,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "min_child_weight": 1,
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
            "scale_pos_weight": negative_class_count / positive_class_count,
        }
        params.update(classifier_params)
        return XGBClassifier(**params)

    raise ValueError(f"Unsupported model family: {model_family}")


def logistic_regularization_label(classifier: LogisticRegression) -> str:
    if classifier.l1_ratio == 1.0:
        return "l1"
    if classifier.l1_ratio == 0.0:
        return "l2"
    return "elasticnet"


def classifier_logging_params(classifier: Any, model_family: str) -> dict[str, Any]:
    if model_family == "logistic_regression":
        return {
            "classifier": "LogisticRegression",
            "classifier__max_iter": classifier.max_iter,
            "classifier__class_weight": classifier.class_weight,
            "classifier__solver": classifier.solver,
            "classifier__C": classifier.C,
            "classifier__l1_ratio": classifier.l1_ratio,
            "classifier__regularization": logistic_regularization_label(classifier),
        }

    if model_family == "xgboost":
        return {
            "classifier": "XGBClassifier",
            "classifier__n_estimators": classifier.n_estimators,
            "classifier__max_depth": classifier.max_depth,
            "classifier__learning_rate": classifier.learning_rate,
            "classifier__subsample": classifier.subsample,
            "classifier__colsample_bytree": classifier.colsample_bytree,
            "classifier__min_child_weight": classifier.min_child_weight,
            "classifier__reg_alpha": classifier.reg_alpha,
            "classifier__reg_lambda": classifier.reg_lambda,
            "classifier__scale_pos_weight": classifier.scale_pos_weight,
        }

    raise ValueError(f"Unsupported model family: {model_family}")


def validate_selection_metric(selection_metric: str) -> None:
    if selection_metric not in SELECTION_METRICS:
        supported_metrics = ", ".join(SELECTION_METRICS)
        raise ValueError(
            f"Unsupported selection metric: {selection_metric}. "
            f"Supported metrics: {supported_metrics}"
        )


def installed_package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def model_pip_requirements(model_family: str) -> list[str]:
    package_names = ["mlflow", "scikit-learn", "numpy", "pandas", "skops"]
    if model_family == "xgboost":
        package_names.append("xgboost")

    requirements: list[str] = []
    for package_name in package_names:
        package_version = installed_package_version(package_name)
        if package_version is None:
            continue
        requirements.append(f"{package_name}=={package_version}")
    return requirements


def model_conda_env(model_family: str) -> dict[str, Any]:
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return {
        "name": "churn-model-mlflow",
        "channels": ["conda-forge"],
        "dependencies": [
            f"python={python_version}",
            "pip",
            {"pip": model_pip_requirements(model_family)},
        ],
    }


def build_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_family: str,
    classifier_params: dict[str, Any] | None = None,
) -> Pipeline:
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
    classifier = build_classifier(model_family, y_train, classifier_params=classifier_params)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def logistic_param_grid() -> list[dict[str, list[Any]]]:
    return [
        {
            "classifier__C": [0.01, 0.1, 1.0, 5.0],
            "classifier__l1_ratio": [0.0, 1.0],
        }
    ]


def xgboost_param_distributions() -> dict[str, list[Any]]:
    return {
        "classifier__n_estimators": [150, 250, 350],
        "classifier__max_depth": [3, 4, 5],
        "classifier__learning_rate": [0.03, 0.05, 0.08],
        "classifier__subsample": [0.8, 0.9, 1.0],
        "classifier__colsample_bytree": [0.7, 0.8, 1.0],
        "classifier__min_child_weight": [1, 3, 5],
        "classifier__reg_alpha": [0.0, 0.1, 0.3],
        "classifier__reg_lambda": [0.5, 1.0, 2.0],
    }


def tune_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    model_family: str,
) -> tuple[Pipeline, dict[str, Any], dict[str, float]]:
    pipeline = build_pipeline(X_train, y_train, model_family)
    cv = StratifiedKFold(n_splits=TUNING_CV_FOLD_COUNT, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"f1": "f1", "roc_auc": "roc_auc"}

    if model_family == "logistic_regression":
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=logistic_param_grid(),
            scoring=scoring,
            refit="f1",
            cv=cv,
            n_jobs=None,
        )
        candidate_count = len(list(ParameterGrid(logistic_param_grid())))
        search_strategy = "GridSearchCV"
    elif model_family == "xgboost":
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=xgboost_param_distributions(),
            n_iter=XGBOOST_RANDOM_SEARCH_ITERATIONS,
            scoring=scoring,
            refit="f1",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=None,
        )
        candidate_count = XGBOOST_RANDOM_SEARCH_ITERATIONS
        search_strategy = "RandomizedSearchCV"
    else:
        raise ValueError(f"Unsupported model family: {model_family}")

    search.fit(X_train, y_train)
    best_index = int(search.best_index_)
    tuning_params: dict[str, Any] = {
        "search_strategy": search_strategy,
        "search_refit_metric": "f1",
        "search_cv_fold_count": TUNING_CV_FOLD_COUNT,
        "search_candidate_count": candidate_count,
    }
    tuning_params.update(search.best_params_)
    tuning_metrics = {
        "tuning_best_cv_f1": float(search.cv_results_["mean_test_f1"][best_index]),
        "tuning_best_cv_roc_auc": float(search.cv_results_["mean_test_roc_auc"][best_index]),
    }
    return search.best_estimator_, tuning_params, tuning_metrics


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


def top_feature_importance_dataframe(
    model: Pipeline,
    *,
    model_family: str,
    limit: int = 15,
) -> pd.DataFrame:
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    classifier = model.named_steps["classifier"]

    if model_family == "logistic_regression":
        importance_values = classifier.coef_[0]
        sort_column = "abs_value"
    elif model_family == "xgboost":
        importance_values = classifier.feature_importances_
        sort_column = "value"
    else:
        raise ValueError(f"Unsupported model family: {model_family}")

    feature_importance = pd.DataFrame({"feature": feature_names, "value": importance_values})
    if sort_column == "abs_value":
        feature_importance = feature_importance.assign(abs_value=lambda df: df["value"].abs())
    return feature_importance.sort_values(sort_column, ascending=False).head(limit)


def cross_validation_metrics(
    estimator: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, float]:
    cv = StratifiedKFold(n_splits=CV_FOLD_COUNT, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }
    cv_results = cross_validate(
        estimator,
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


def regularization_and_selection_metadata(
    model: Pipeline,
    model_family: str,
) -> tuple[dict[str, Any], dict[str, float]]:
    classifier = model.named_steps["classifier"]
    if model_family == "logistic_regression":
        coefficients = classifier.coef_[0]
        non_zero_features = int(np.count_nonzero(coefficients))
        params = {
            "regularization_family": "linear_penalty",
            "regularization_penalty": logistic_regularization_label(classifier),
            "selection_method": (
                "embedded_l1" if logistic_regularization_label(classifier) == "l1" else "dense_coefficients"
            ),
        }
        metrics = {
            "selected_feature_count": float(non_zero_features),
            "total_encoded_feature_count": float(len(coefficients)),
        }
        return params, metrics

    if model_family == "xgboost":
        importances = classifier.feature_importances_
        non_zero_features = int(np.count_nonzero(importances))
        params = {
            "regularization_family": "tree_regularization",
            "regularization_penalty": "reg_alpha_reg_lambda",
            "selection_method": "feature_importance_non_zero",
        }
        metrics = {
            "selected_feature_count": float(non_zero_features),
            "total_encoded_feature_count": float(len(importances)),
        }
        return params, metrics

    raise ValueError(f"Unsupported model family: {model_family}")


def mlflow_input_example(features: pd.DataFrame) -> pd.DataFrame:
    input_example = features.head(min(5, len(features))).copy()
    for column in input_example.columns:
        if pd.api.types.is_integer_dtype(input_example[column]):
            input_example[column] = input_example[column].astype(float)
    return input_example


def mlflow_safe_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    safe_dataframe = dataframe.copy()
    for column in safe_dataframe.columns:
        if pd.api.types.is_integer_dtype(safe_dataframe[column]):
            safe_dataframe[column] = safe_dataframe[column].astype(float)
    return safe_dataframe


def save_figures(
    *,
    y_true: pd.Series,
    y_pred: pd.Series,
    y_score: pd.Series,
    top_features: pd.DataFrame,
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
    sns.barplot(data=top_features, y="feature", x="value", ax=ax)
    ax.set_title(f"Top 15 features: {variant_name}")
    plt.tight_layout()
    mlflow.log_figure(fig, f"figures/{variant_name}_top_features.png")
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
    model_family: str,
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
        model_family=model_family,
        include_tenure_group=include_tenure_group,
        threshold=threshold,
        dataset_hash=dataset_hash,
        trained_at_utc=datetime.now(UTC).isoformat(),
        metrics=metrics,
    )
    joblib.dump(artifact, model_path)
    metadata = {
        "variant_name": artifact.variant_name,
        "model_family": artifact.model_family,
        "include_tenure_group": artifact.include_tenure_group,
        "threshold": artifact.threshold,
        "dataset_hash": artifact.dataset_hash,
        "trained_at_utc": artifact.trained_at_utc,
        "metrics": artifact.metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path, metadata_path


def select_result(
    results: list[TrainingResult],
    *,
    selection_metric: str,
    selected_variant: str | None = None,
) -> TrainingResult:
    validate_selection_metric(selection_metric)
    if selected_variant is not None:
        return next(result for result in results if result.variant_name == selected_variant)

    return max(
        results,
        key=lambda result: (
            result.metrics[selection_metric],
            result.metrics["roc_auc"],
            result.metrics["churn_f1"],
        ),
    )


def train_variant(
    *,
    dataframe: pd.DataFrame,
    dataset_path: Path,
    dataset_hash: str,
    experiment_id: str,
    variant_name: str,
    model_family: str,
    include_tenure_group: bool,
    output_dir: Path,
) -> TrainingResult:
    return run_training_variant(
        dataframe=dataframe,
        dataset_path=dataset_path,
        dataset_hash=dataset_hash,
        experiment_id=experiment_id,
        variant_name=variant_name,
        model_family=model_family,
        include_tenure_group=include_tenure_group,
        output_dir=output_dir,
        tuned=True,
        classifier_params=None,
    )


def run_training_variant(
    *,
    dataframe: pd.DataFrame,
    dataset_path: Path,
    dataset_hash: str,
    experiment_id: str,
    variant_name: str,
    model_family: str,
    include_tenure_group: bool,
    output_dir: Path,
    tuned: bool,
    classifier_params: dict[str, Any] | None,
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
    training_dataset_df_for_mlflow = mlflow_safe_dataframe(training_dataset_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    if tuned:
        model, tuning_params, tuning_metrics = tune_pipeline(X_train, y_train, model_family=model_family)
        training_mode = "experiment"
    else:
        model = build_pipeline(
            X_train,
            y_train,
            model_family,
            classifier_params=classifier_params,
        )
        model.fit(X_train, y_train)
        tuning_params = {
            "search_strategy": "frozen_best_params",
            "search_refit_metric": "not_applicable",
            "search_cv_fold_count": 0,
            "search_candidate_count": 1,
        }
        tuning_metrics = {}
        training_mode = "production"
    cv_metrics = cross_validation_metrics(model, X_train, y_train)
    classifier = model.named_steps["classifier"]
    regularization_params, regularization_metrics = regularization_and_selection_metadata(
        model,
        model_family,
    )
    dataset_source = LocalArtifactDatasetSource(uri=str(dataset_path.resolve()))

    with mlflow.start_run(experiment_id=experiment_id, run_name=variant_name) as run:
        mlflow.set_tags(
            {
                "dataset": "WA_Fn-UseC_-Telco-Customer-Churn.csv",
                "model_family": model_family,
                "feature_variant": variant_name,
                "training_mode": training_mode,
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
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "cv_fold_count": CV_FOLD_COUNT,
                "numeric_feature_count": len(X_train.select_dtypes(include=["number"]).columns),
                "categorical_feature_count": len(X_train.select_dtypes(exclude=["number"]).columns),
                **tuning_params,
                **regularization_params,
                **classifier_logging_params(classifier, model_family),
            }
        )
        mlflow_dataset = mlflow.data.from_pandas(
            training_dataset_df_for_mlflow,
            source=dataset_source,
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
        metrics.update(tuning_metrics)
        metrics.update(cv_metrics)
        metrics.update(regularization_metrics)
        mlflow.log_metrics(metrics)

        top_features = top_feature_importance_dataframe(model, model_family=model_family)
        save_figures(
            y_true=y_test,
            y_pred=y_pred,
            y_score=y_score,
            top_features=top_features,
            threshold_df=threshold_df,
            variant_name=variant_name,
        )
        input_example = mlflow_input_example(X_train)
        probability_output = model.predict_proba(input_example)
        signature = infer_signature(input_example, probability_output)
        mlflow.sklearn.log_model(
            model,
            name="model",
            conda_env=model_conda_env(model_family),
            signature=signature,
            input_example=input_example,
            serialization_format="skops",
            skops_trusted_types=[
                "numpy.dtype",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ],
            pyfunc_predict_fn="predict_proba",
        )

        model_path, metadata_path = persist_model(
            model=model,
            variant_name=variant_name,
            model_family=model_family,
            include_tenure_group=include_tenure_group,
            threshold=float(best_threshold_row["threshold"]),
            dataset_hash=dataset_hash,
            metrics=metrics,
            output_dir=output_dir,
        )

        return TrainingResult(
            variant_name=variant_name,
            model_family=model_family,
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
            top_coefficients=top_features,
            model_path=model_path,
            metadata_path=metadata_path,
        )


def comparison_table(
    results: list[TrainingResult],
    *,
    promoted_variant: str | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variant": result.variant_name,
                "model_family": result.model_family,
                "run_id": result.run_id,
                "include_tenure_group": result.include_tenure_group,
                "accuracy": result.metrics["accuracy"],
                "roc_auc": result.metrics["roc_auc"],
                "churn_precision": result.metrics["churn_precision"],
                "churn_recall": result.metrics["churn_recall"],
                "churn_f1": result.metrics["churn_f1"],
                "tuning_best_cv_f1": result.metrics["tuning_best_cv_f1"],
                "selected_feature_count": result.metrics["selected_feature_count"],
                "cv_roc_auc_mean": result.metrics["cv_roc_auc_mean"],
                "cv_f1_mean": result.metrics["cv_f1_mean"],
                "best_threshold_by_f1": result.metrics["best_threshold_by_f1"],
                "is_promoted": result.variant_name == promoted_variant,
            }
            for result in results
        ]
    ).sort_values("variant")


def promote_result(
    result: TrainingResult,
    *,
    selection_metric: str,
    selected_variant: str | None,
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    copy2(result.model_path, PRODUCTION_MODEL_PATH)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    metadata["promotion"] = {
        "selection_metric": selection_metric,
        "selection_mode": "manual" if selected_variant is not None else "automatic",
        "promoted_variant": result.variant_name,
        "promoted_at_utc": datetime.now(UTC).isoformat(),
    }
    PRODUCTION_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def train_production_pipeline(
    *,
    dataset_path: Path = DATA_PATH,
    output_dir: Path | None = None,
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> TrainingResult:
    dataset = load_raw_dataset(dataset_path)
    dataset_hash = dataset_sha256(dataset_path)
    experiment_id = configure_mlflow(tracking_uri=tracking_uri, experiment_name=experiment_name)
    mlflow.end_run()

    training_output_dir = output_dir or MODEL_DIR
    result = run_training_variant(
        dataframe=dataset,
        dataset_path=dataset_path,
        dataset_hash=dataset_hash,
        experiment_id=experiment_id,
        variant_name=PRODUCTION_VARIANT.variant_name,
        model_family=PRODUCTION_VARIANT.model_family,
        include_tenure_group=PRODUCTION_VARIANT.include_tenure_group,
        output_dir=training_output_dir,
        tuned=False,
        classifier_params=PRODUCTION_CLASSIFIER_PARAMS,
    )
    promote_result(
        result,
        selection_metric="frozen_best_params",
        selected_variant=PRODUCTION_VARIANT.variant_name,
    )
    return result


def run_experiment_pipeline(
    *,
    dataset_path: Path = DATA_PATH,
    selected_variant: str | None = None,
    selection_metric: str = DEFAULT_SELECTION_METRIC,
    output_dir: Path | None = None,
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> tuple[list[TrainingResult], pd.DataFrame]:
    validate_selection_metric(selection_metric)
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
            variant_name=variant.variant_name,
            model_family=variant.model_family,
            include_tenure_group=variant.include_tenure_group,
            output_dir=training_output_dir,
        )
        for variant in MODEL_VARIANTS
    ]
    selected = select_result(
        results,
        selection_metric=selection_metric,
        selected_variant=selected_variant,
    )
    summary = comparison_table(results, promoted_variant=selected.variant_name)
    promote_result(
        selected,
        selection_metric=selection_metric,
        selected_variant=selected_variant,
    )
    return results, summary
