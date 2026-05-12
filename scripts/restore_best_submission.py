from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.registry import load_best_submission, restore_best_submission


def main() -> None:
    best = load_best_submission()
    destination = restore_best_submission()
    print(f"Restored {best['filename']} -> {destination}")
    print(f"Best confirmed score: {best['lb_score']:.2f}")


if __name__ == "__main__":
    main()
