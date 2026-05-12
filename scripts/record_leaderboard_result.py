import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.registry import record_leaderboard_result


def parse_args():
    parser = argparse.ArgumentParser(description="Record a leaderboard result.")
    parser.add_argument("--file", required=True, help="Submission file path, e.g. submissions/test.csv")
    parser.add_argument("--model", required=True, help="Model or experiment name.")
    parser.add_argument("--lb-score", required=True, type=float, help="Leaderboard score, e.g. 95.86")
    parser.add_argument("--verdict", required=True, help="Contest verdict, usually OK.")
    parser.add_argument("--comment", default="", help="Free-form note.")
    parser.add_argument("--local-cv-mape", type=float, default=None, help="Optional local CV MAPE.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    best = record_leaderboard_result(
        filename=args.file,
        model_name=args.model,
        lb_score=args.lb_score,
        verdict=args.verdict,
        comment=args.comment,
        local_cv_mape=args.local_cv_mape,
    )
    print("Recorded leaderboard result.")
    print(f"Current best: {best['filename']} score={best['lb_score']:.2f}")


if __name__ == "__main__":
    main()
