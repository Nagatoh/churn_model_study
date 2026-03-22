from __future__ import annotations

import argparse
import os

import uvicorn

from churn_model.config import MODEL_PATH_ENV_VAR, PRODUCTION_MODEL_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sobe a API de inferencia do modelo de churn.")
    parser.add_argument("--host", default="127.0.0.1", help="Host do servidor.")
    parser.add_argument("--port", type=int, default=8000, help="Porta do servidor.")
    parser.add_argument(
        "--model-path",
        default=str(PRODUCTION_MODEL_PATH),
        help="Caminho do artifact de modelo promovido.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    os.environ[MODEL_PATH_ENV_VAR] = args.model_path
    uvicorn.run("churn_model.api:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
