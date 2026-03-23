from __future__ import annotations

import argparse
from pathlib import Path

from churn_model.config import DATA_PATH, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODEL_DIR
from churn_model.training import (
    DEFAULT_SELECTION_METRIC,
    SELECTION_METRICS,
    SUPPORTED_VARIANTS,
    run_experiment_pipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Executa experimentos de churn com tuning e comparacao de variantes."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DATA_PATH,
        help="Caminho do CSV de churn usado nos experimentos.",
    )
    parser.add_argument(
        "--selected-variant",
        choices=SUPPORTED_VARIANTS,
        help="Variante promovida manualmente apos os experimentos. Se omitido, usa a melhor run.",
    )
    parser.add_argument(
        "--selection-metric",
        default=DEFAULT_SELECTION_METRIC,
        choices=SELECTION_METRICS,
        help="Metrica usada para promover automaticamente a melhor run experimental.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODEL_DIR,
        help="Diretorio para salvar os artifacts locais de modelo.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=MLFLOW_TRACKING_URI,
        help="Tracking URI do MLflow.",
    )
    parser.add_argument(
        "--experiment-name",
        default=MLFLOW_EXPERIMENT_NAME,
        help="Nome do experimento no MLflow.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _, summary = run_experiment_pipeline(
        dataset_path=args.dataset_path,
        selected_variant=args.selected_variant,
        selection_metric=args.selection_metric,
        output_dir=args.output_dir,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
    )
    promoted_variant = summary.loc[summary["is_promoted"], "variant"].iloc[0]
    print("Resumo das runs registradas no MLflow:")
    print(summary.to_string(index=False))
    print()
    print(f"Variante promovida: {promoted_variant}")


if __name__ == "__main__":
    main()
