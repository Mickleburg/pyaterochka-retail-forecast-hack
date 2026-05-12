from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import MODEL_NAMES, TRAIN_PATH, VALIDATION_RESULTS_PATH
from src.features import load_train
from src.validation import run_time_validation, summarize_train


def main() -> None:
    df = load_train(TRAIN_PATH)
    summary = summarize_train(df)
    print("Data summary")
    print(f"shape: {summary['shape']}")
    print(f"columns: {summary['columns']}")
    print("dtypes:")
    for col, dtype in summary["dtypes"].items():
        print(f"  {col}: {dtype}")
    print(f"month min/max: {summary['month_min']} / {summary['month_max']}")
    print(f"unique stores: {summary['unique_stores']}")
    print(f"missing values: {summary['missing']}")
    print(
        "months per store min/max: "
        f"{summary['months_per_store_min']} / {summary['months_per_store_max']}"
    )
    print(f"stores with incomplete history: {summary['stores_with_incomplete_history']}")

    results = run_time_validation(df, MODEL_NAMES)
    VALIDATION_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(VALIDATION_RESULTS_PATH, index=False)
    print()
    print(results.to_string(index=False))
    print(f"\nSaved: {VALIDATION_RESULTS_PATH}")


if __name__ == "__main__":
    main()
