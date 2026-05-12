from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.registry import SUBMISSION_REGISTRY_PATH, ensure_registry_files, load_best_submission


def main() -> None:
    ensure_registry_files()
    df = pd.read_csv(SUBMISSION_REGISTRY_PATH)
    best = load_best_submission()
    view = df[["filename", "model_name", "local_cv_mape", "lb_score", "lb_mape", "verdict"]].copy()
    view["is_best"] = view["filename"].astype(str).str.replace("\\", "/", regex=False) == best["filename"]
    sort_score = pd.to_numeric(view["lb_score"], errors="coerce")
    view = view.assign(_sort_score=sort_score).sort_values("_sort_score", ascending=False).drop(columns="_sort_score")
    print(view.to_string(index=False))


if __name__ == "__main__":
    main()
