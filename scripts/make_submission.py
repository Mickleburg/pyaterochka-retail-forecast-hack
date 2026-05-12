import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import MODEL_NAMES, SUBMISSIONS_DIR, TRAIN_PATH
from src.features import load_train
from src.registry import best_warning_text, restore_best_submission
from src.submit import make_predictions_for_month, save_submission


def parse_args():
    parser = argparse.ArgumentParser(description="Generate X5/Pyaterochka RTO submission.")
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--make-test-copy", action="store_true", help="Also copy output to project root as test.csv.")
    parser.add_argument(
        "--restore-best-after",
        action="store_true",
        help="Restore the best confirmed submission to test.csv after generating this candidate.",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Debug only. Do not use for Contest: Contest CSV must contain the new_id,rto header.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.make_test_copy:
        print(best_warning_text())

    df = load_train(TRAIN_PATH)
    submission = make_predictions_for_month(df, args.model)
    out_path = SUBMISSIONS_DIR / f"test_{args.model}.csv"
    save_submission(
        submission,
        out_path,
        make_test_copy=args.make_test_copy,
        write_header=not args.no_header,
    )
    print(f"Saved: {out_path}")
    print(f"shape: {submission.shape}")
    print(submission.head().to_string(index=False))
    if args.make_test_copy:
        print(f"Copied to: {ROOT / 'test.csv'}")
    if args.restore_best_after:
        restored = restore_best_submission()
        print(f"Restored best confirmed submission to: {restored}")


if __name__ == "__main__":
    main()
