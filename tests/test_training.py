from __future__ import annotations

import pandas as pd
import pytest

from churn_model.data import TARGET_COLUMN, build_model_frame
from churn_model.training import (
    TrainingResult,
    build_pipeline,
    comparison_table,
    cross_validation_metrics,
    regularization_and_selection_metadata,
    select_result,
    tune_pipeline,
)


def synthetic_telco_dataset() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx in range(60):
        churn = idx % 2 == 0
        tenure = 2 + (idx % 4) if churn else 24 + (idx % 24)
        monthly_charges = 85.0 + (idx % 5) if churn else 45.0 + (idx % 8)
        rows.append(
            {
                "customerID": f"CUST-{idx:04d}",
                "Churn": "Yes" if churn else "No",
                "gender": "Female" if idx % 3 == 0 else "Male",
                "SeniorCitizen": 1 if churn and idx % 6 == 0 else 0,
                "Partner": "No" if churn else "Yes",
                "Dependents": "No" if churn else "Yes",
                "tenure": tenure,
                "PhoneService": "Yes",
                "MultipleLines": "Yes" if churn else "No",
                "InternetService": "Fiber optic" if churn else "DSL",
                "OnlineSecurity": "No" if churn else "Yes",
                "OnlineBackup": "No" if churn else "Yes",
                "DeviceProtection": "No" if churn else "Yes",
                "TechSupport": "No" if churn else "Yes",
                "StreamingTV": "Yes" if churn else "No",
                "StreamingMovies": "Yes" if churn else "No",
                "Contract": "Month-to-month" if churn else "Two year",
                "PaperlessBilling": "Yes" if churn else "No",
                "PaymentMethod": "Electronic check" if churn else "Bank transfer (automatic)",
                "MonthlyCharges": monthly_charges,
                "TotalCharges": f"{monthly_charges * tenure:.2f}",
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("model_family", ["logistic_regression", "xgboost"])
def test_model_families_keep_metric_floor(model_family: str) -> None:
    model_frame = build_model_frame(
        synthetic_telco_dataset(),
        include_tenure_group=False,
        include_target=True,
    )
    X = model_frame.drop(columns=["Churn", TARGET_COLUMN])
    y = model_frame[TARGET_COLUMN]

    pipeline = build_pipeline(X, y, model_family=model_family)
    cv_metrics = cross_validation_metrics(pipeline, X, y)
    pipeline.fit(X, y)

    assert cv_metrics["cv_roc_auc_mean"] >= 0.95
    assert cv_metrics["cv_f1_mean"] >= 0.90
    assert hasattr(pipeline.named_steps["classifier"], "predict_proba")


@pytest.mark.parametrize("model_family", ["logistic_regression", "xgboost"])
def test_tuned_pipeline_returns_search_metadata(model_family: str) -> None:
    model_frame = build_model_frame(
        synthetic_telco_dataset(),
        include_tenure_group=False,
        include_target=True,
    )
    X = model_frame.drop(columns=["Churn", TARGET_COLUMN])
    y = model_frame[TARGET_COLUMN]

    tuned_pipeline, tuning_params, tuning_metrics = tune_pipeline(
        X,
        y,
        model_family=model_family,
    )
    tuned_pipeline.fit(X, y)
    regularization_params, regularization_metrics = regularization_and_selection_metadata(
        tuned_pipeline,
        model_family,
    )

    assert tuning_params["search_strategy"] in {"GridSearchCV", "RandomizedSearchCV"}
    assert tuning_params["search_cv_fold_count"] == 3
    assert tuning_metrics["tuning_best_cv_f1"] >= 0.90
    assert tuning_metrics["tuning_best_cv_roc_auc"] >= 0.95
    assert regularization_metrics["selected_feature_count"] > 0
    assert regularization_params["selection_method"] in {
        "embedded_l1",
        "dense_coefficients",
        "feature_importance_non_zero",
    }


def test_select_result_uses_best_metric_when_variant_is_not_forced() -> None:
    results = [
        TrainingResult(
            variant_name="logreg_without_tenure_group",
            model_family="logistic_regression",
            include_tenure_group=False,
            dropped_columns=["tenure_group"],
            run_id="run-a",
            model=None,
            threshold=0.55,
            dataset_hash="hash-a",
                metrics={
                    "accuracy": 0.74,
                    "roc_auc": 0.84,
                    "churn_precision": 0.50,
                    "churn_recall": 0.78,
                    "churn_f1": 0.61,
                    "tuning_best_cv_f1": 0.62,
                    "cv_roc_auc_mean": 0.845,
                    "cv_f1_mean": 0.628,
                    "selected_feature_count": 18.0,
                    "best_threshold_by_f1": 0.55,
                },
            report_dict={},
            threshold_df=pd.DataFrame(),
            best_threshold_row=pd.Series(dtype=float),
            top_coefficients=pd.DataFrame(),
            model_path=None,
            metadata_path=None,
        ),
        TrainingResult(
            variant_name="xgboost_without_tenure_group",
            model_family="xgboost",
            include_tenure_group=False,
            dropped_columns=["tenure_group"],
            run_id="run-b",
            model=None,
            threshold=0.55,
            dataset_hash="hash-b",
                metrics={
                    "accuracy": 0.75,
                    "roc_auc": 0.844,
                    "churn_precision": 0.52,
                    "churn_recall": 0.77,
                    "churn_f1": 0.623,
                    "tuning_best_cv_f1": 0.63,
                    "cv_roc_auc_mean": 0.846,
                    "cv_f1_mean": 0.632,
                    "selected_feature_count": 12.0,
                    "best_threshold_by_f1": 0.55,
                },
            report_dict={},
            threshold_df=pd.DataFrame(),
            best_threshold_row=pd.Series(dtype=float),
            top_coefficients=pd.DataFrame(),
            model_path=None,
            metadata_path=None,
        ),
    ]

    selected = select_result(results, selection_metric="cv_f1_mean", selected_variant=None)
    summary = comparison_table(results, promoted_variant=selected.variant_name)

    assert selected.variant_name == "xgboost_without_tenure_group"
    assert summary.loc[summary["variant"] == "xgboost_without_tenure_group", "is_promoted"].item() is True
