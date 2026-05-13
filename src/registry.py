import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT, REPORTS_DIR
from .submit import validate_saved_csv


LEADERBOARD_COLUMNS = [
    "submitted_at",
    "filename",
    "model_name",
    "local_cv_mape",
    "lb_score",
    "lb_mape",
    "verdict",
    "comment",
]

BEST_SUBMISSION_PATH = REPORTS_DIR / "best_submission.json"
LEADERBOARD_RESULTS_PATH = REPORTS_DIR / "leaderboard_results.csv"
SUBMISSION_REGISTRY_PATH = REPORTS_DIR / "submission_registry.csv"

KNOWN_LEADERBOARD_ROWS = [
    {
        "submitted_at": "",
        "filename": "submissions/test_official_mean_october_baseline.csv",
        "model_name": "official_mean_october_baseline",
        "local_cv_mape": "",
        "lb_score": 54.34,
        "lb_mape": 45.66,
        "verdict": "OK",
        "comment": "официальный бейзлайн, проверка формата",
    },
    {
        "submitted_at": "",
        "filename": "submissions/test_catboost_mape.csv",
        "model_name": "catboost_mape",
        "local_cv_mape": "",
        "lb_score": 95.80,
        "lb_mape": 4.20,
        "verdict": "OK",
        "comment": "первый кандидат catboost/fallback",
    },
    {
        "submitted_at": "",
        "filename": "submissions/test_catboost_log.csv",
        "model_name": "catboost_log",
        "local_cv_mape": "",
        "lb_score": 95.80,
        "lb_mape": 4.20,
        "verdict": "OK",
        "comment": "первый кандидат catboost log/fallback",
    },
    {
        "submitted_at": "",
        "filename": "submissions/test_baseline_last_month.csv",
        "model_name": "baseline_last_month",
        "local_cv_mape": "",
        "lb_score": 95.86,
        "lb_mape": 4.14,
        "verdict": "OK",
        "comment": "текущее лучшее подтвержденное решение",
    },
]

KNOWN_BEST = {
    "filename": "submissions/test_ratio_shrink_b0p06_c97_103.csv",
    "model_name": "ratio_shrink_b0p06_c97_103",
    "lb_score": 95.88,
    "lb_mape": 4.12,
    "verdict": "OK",
    "is_confirmed_by_leaderboard": True,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_registry_files() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if not LEADERBOARD_RESULTS_PATH.exists():
        pd.DataFrame(KNOWN_LEADERBOARD_ROWS, columns=LEADERBOARD_COLUMNS).to_csv(
            LEADERBOARD_RESULTS_PATH, index=False
        )
    if not SUBMISSION_REGISTRY_PATH.exists():
        pd.DataFrame(KNOWN_LEADERBOARD_ROWS, columns=LEADERBOARD_COLUMNS).to_csv(
            SUBMISSION_REGISTRY_PATH, index=False
        )
    if not BEST_SUBMISSION_PATH.exists():
        save_best_submission(KNOWN_BEST)


def load_leaderboard() -> pd.DataFrame:
    ensure_registry_files()
    return pd.read_csv(LEADERBOARD_RESULTS_PATH, encoding="utf-8")


def save_leaderboard(df: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = df.loc[:, LEADERBOARD_COLUMNS]
    df.to_csv(LEADERBOARD_RESULTS_PATH, index=False, encoding="utf-8")


def load_best_submission() -> dict:
    ensure_registry_files()
    return json.loads(BEST_SUBMISSION_PATH.read_text(encoding="utf-8"))


def save_best_submission(best: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    BEST_SUBMISSION_PATH.write_text(
        json.dumps(best, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def record_leaderboard_result(
    filename: str,
    model_name: str,
    lb_score: float,
    verdict: str,
    comment: str = "",
    local_cv_mape: float | None = None,
    submitted_at: str | None = None,
) -> dict:
    ensure_registry_files()
    lb_mape = 100.0 - float(lb_score)
    row = {
        "submitted_at": submitted_at or _now_iso(),
        "filename": filename.replace("\\", "/"),
        "model_name": model_name,
        "local_cv_mape": "" if local_cv_mape is None else float(local_cv_mape),
        "lb_score": float(lb_score),
        "lb_mape": lb_mape,
        "verdict": verdict,
        "comment": comment,
    }

    df = load_leaderboard()
    key_cols = ["filename", "model_name", "lb_score", "verdict"]
    existing = df.copy()
    existing["filename"] = existing["filename"].astype(str)
    existing["model_name"] = existing["model_name"].astype(str)
    existing["lb_score_num"] = pd.to_numeric(existing["lb_score"], errors="coerce")
    existing["verdict"] = existing["verdict"].astype(str)
    is_duplicate = (
        (existing["filename"] == str(row["filename"]))
        & (existing["model_name"] == str(row["model_name"]))
        & (existing["lb_score_num"].round(8) == round(float(row["lb_score"]), 8))
        & (existing["verdict"] == str(row["verdict"]))
    ).any()
    if not is_duplicate:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_leaderboard(df)

    best = load_best_submission()
    if verdict == "OK" and float(lb_score) > float(best.get("lb_score", float("-inf"))):
        best = {
            "filename": row["filename"],
            "model_name": model_name,
            "lb_score": float(lb_score),
            "lb_mape": lb_mape,
            "verdict": verdict,
            "is_confirmed_by_leaderboard": True,
        }
        save_best_submission(best)
    return best


def restore_best_submission() -> Path:
    best = load_best_submission()
    source = PROJECT_ROOT / best["filename"]
    destination = PROJECT_ROOT / "test.csv"
    validate_saved_csv(source)
    shutil.copy2(source, destination)
    validate_saved_csv(destination)
    return destination


def best_warning_text() -> str:
    best = load_best_submission()
    return (
        "WARNING: test.csv будет перезаписан. "
        f"Текущий лучший подтвержденный сабмит: {best['filename']} "
        f"score {best['lb_score']:.2f}."
    )
