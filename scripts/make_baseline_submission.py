from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import SUBMISSIONS_DIR, TRAIN_PATH
from src.features import load_train
from src.submit import make_predictions_for_month, save_submission


def main() -> None:
    df = load_train(TRAIN_PATH)
    submission = make_predictions_for_month(df, "baseline_last_month")
    out_path = SUBMISSIONS_DIR / "test_baseline_last_month.csv"
    save_submission(submission, out_path)
    print(f"Saved: {out_path}")
    print(f"shape: {submission.shape}")
    print(submission.head().to_string(index=False))


if __name__ == "__main__":
    main()
