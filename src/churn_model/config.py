from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
MLFLOW_ARTIFACTS_PATH = PROJECT_ROOT / "mlartifacts"
MLFLOW_EXPERIMENT_NAME = "telco-churn-production-pipeline"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "models"
PRODUCTION_MODEL_PATH = MODEL_DIR / "churn_model.joblib"
PRODUCTION_METADATA_PATH = MODEL_DIR / "churn_model_metadata.json"
MODEL_PATH_ENV_VAR = "CHURN_MODEL_PATH"

DEFAULT_THRESHOLD = 0.50
