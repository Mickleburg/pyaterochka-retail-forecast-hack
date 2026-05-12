import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.submit import validate_saved_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Check a Contest submission CSV.")
    parser.add_argument("path", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = args.path
    print(f"path: {path}")

    try:
        file_size = path.stat().st_size
        with path.open("r", encoding="utf-8", newline="") as file:
            first_line = file.readline().rstrip("\r\n")
        df = pd.read_csv(path)

        print(f"file size: {file_size}")
        print(f"first line: {first_line}")
        print(f"columns: {list(df.columns)}")
        print(f"shape: {df.shape}")
        print("dtypes:")
        print(df.dtypes.to_string())
        print(f"number of NaN: {int(df.isna().sum().sum())}")
        print(f"number of negative rto: {int((df['rto'] < 0).sum()) if 'rto' in df else 'n/a'}")
        print(f"number of duplicated new_id: {int(df['new_id'].duplicated().sum()) if 'new_id' in df else 'n/a'}")
        print("head:")
        print(df.head().to_string(index=False))
        print("tail:")
        print(df.tail().to_string(index=False))

        validate_saved_csv(path)
    except Exception as exc:
        print(f"verdict: FORMAT ERROR")
        print(f"error: {exc}")
        return 1

    print("verdict: FORMAT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
