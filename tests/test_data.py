from __future__ import annotations

import pandas as pd

from churn_model.data import TARGET_COLUMN, TENURE_GROUP_COLUMN, build_model_frame, dataset_sha256


def test_build_model_frame_adds_target_and_tenure_group(
    sample_training_record: dict[str, object],
) -> None:
    dataframe = pd.DataFrame([sample_training_record])

    result = build_model_frame(
        dataframe,
        include_tenure_group=True,
        include_target=True,
    )

    assert TARGET_COLUMN in result.columns
    assert TENURE_GROUP_COLUMN in result.columns
    assert result.loc[0, TARGET_COLUMN] == 0
    assert str(result.loc[0, TENURE_GROUP_COLUMN]) == "0-12m"
    assert result.loc[0, "TotalCharges"] == 29.85


def test_build_model_frame_excludes_target_columns_for_inference(
    sample_customer_record: dict[str, object],
) -> None:
    dataframe = pd.DataFrame([sample_customer_record])

    result = build_model_frame(
        dataframe,
        include_tenure_group=False,
        include_target=False,
    )

    assert TARGET_COLUMN not in result.columns
    assert "Churn" not in result.columns
    assert TENURE_GROUP_COLUMN not in result.columns
    assert "customerID" not in result.columns


def test_dataset_sha256_is_stable(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text("a,b\n1,2\n", encoding="utf-8")

    first_digest = dataset_sha256(dataset_path)
    second_digest = dataset_sha256(dataset_path)

    assert first_digest == second_digest
    assert len(first_digest) == 64
