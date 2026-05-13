from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.advanced_experiments import run_error_analysis


if __name__ == "__main__":
    run_error_analysis()
