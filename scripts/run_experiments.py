from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.advanced_experiments import *  # noqa: F401,F403
from src.advanced_experiments import main


if __name__ == "__main__":
    main()
