from pathlib import Path


RANDOM_SEED = 2026

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_RESULTS_PATH = REPORTS_DIR / "validation_results.csv"

ID_COL = "new_id"
MONTH_COL = "Месяц"
TARGET_COL = "РТО"

FUTURE_MONTH = 11
EXPECTED_SUBMISSION_ROWS = 20615

CATEGORICAL_COLS = [
    ID_COL,
    "Дата открытия, категориальный",
    "Торговая площадь, категориальный",
    "Населенный пункт",
    "Регион",
]

FOLDS = [
    {"fold": 1, "train_end_month": 7, "valid_month": 8},
    {"fold": 2, "train_end_month": 8, "valid_month": 9},
    {"fold": 3, "train_end_month": 9, "valid_month": 10},
]

MODEL_NAMES = [
    "baseline_last_month",
    "baseline_global_growth",
    "catboost_mape",
    "catboost_log",
]
