import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import SUBMISSIONS_DIR, TRAIN_PATH
from src.features import load_train
from src.submit import make_official_mean_october_baseline, make_predictions_for_month, save_submission


def parse_args():
    parser = argparse.ArgumentParser(description="Generate baseline submissions.")
    parser.add_argument(
        "--official",
        action="store_true",
        help="Generate the organizers' mean October baseline from sample_submission.ipynb.",
    )
    parser.add_argument("--make-test-copy", action="store_true", help="Also copy output to project root as test.csv.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_train(TRAIN_PATH)
    if args.official:
        submission = make_official_mean_october_baseline(df)
        out_path = SUBMISSIONS_DIR / "test_official_mean_october_baseline.csv"
    else:
        submission = make_predictions_for_month(df, "baseline_last_month")
        out_path = SUBMISSIONS_DIR / "test_baseline_last_month.csv"

    save_submission(submission, out_path, make_test_copy=args.make_test_copy)
    print(f"Saved: {out_path}")
    print(f"shape: {submission.shape}")
    print(submission.head().to_string(index=False))
    if args.make_test_copy:
        print(f"Copied to: {ROOT / 'test.csv'}")


if __name__ == "__main__":
    main()
