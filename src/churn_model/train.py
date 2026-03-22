from __future__ import annotations

import argparse
from pathlib import Path

from churn_model.config import DATA_PATH, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODEL_DIR
from churn_model.training import train_production_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Treina o pipeline de churn e registra runs no MLflow.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DATA_PATH,
        help="Caminho do CSV de churn usado no treino.",
    )
    parser.add_argument(
        "--selected-variant",
        default="without_tenure_group",
        choices=["with_tenure_group", "without_tenure_group"],
        help="Variante promovida para artifact de producao.",
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
    _, summary = train_production_pipeline(
        dataset_path=args.dataset_path,
        selected_variant=args.selected_variant,
        output_dir=args.output_dir,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
    )
    print("Resumo das runs registradas no MLflow:")
    print(summary.to_string(index=False))
    print()
    print(f"Variante promovida: {args.selected_variant}")


if __name__ == "__main__":
    main()
