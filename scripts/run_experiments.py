from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stage2_experiments import run_stage2


if __name__ == "__main__":
    run_stage2()
