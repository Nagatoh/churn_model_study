from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from churn_model.config import PRODUCTION_MODEL_PATH
from churn_model.inference import predict_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Executa inferencia batch para churn.")
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="CSV com as features cruas do cliente.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Arquivo JSON de saida com as previsoes.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PRODUCTION_MODEL_PATH,
        help="Caminho do artifact de modelo promovido.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    dataframe = pd.read_csv(args.input_path)
    predictions = predict_records(dataframe.to_dict(orient="records"), model_path=args.model_path)
    payload = [prediction.model_dump() for prediction in predictions]
    if args.output_path:
        args.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Predicoes salvas em {args.output_path}")
        return
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
