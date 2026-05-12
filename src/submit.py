import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .config import EXPECTED_SUBMISSION_ROWS, FUTURE_MONTH, ID_COL, MONTH_COL, PROJECT_ROOT, TARGET_COL
from .features import add_future_month_rows, build_features, get_cat_features, get_feature_columns
from .models import get_model


def make_predictions_for_month(df: pd.DataFrame, model_name: str, future_month: int = FUTURE_MONTH) -> pd.DataFrame:
    full = add_future_month_rows(df, future_month=future_month)
    feat_df = build_features(full)
    feature_columns = get_feature_columns(feat_df)
    cat_features = get_cat_features(feature_columns)

    train_mask = feat_df[TARGET_COL].notna() & (feat_df[MONTH_COL] < future_month)
    future_mask = feat_df[MONTH_COL] == future_month

    x_train = feat_df.loc[train_mask, feature_columns]
    y_train = feat_df.loc[train_mask, TARGET_COL]
    x_future = feat_df.loc[future_mask, feature_columns]

    model = get_model(model_name)
    model.fit(x_train, y_train, cat_features=cat_features)
    pred = model.predict(x_future)
    pred = np.clip(pred, 0, None)

    submission = pd.DataFrame(
        {
            ID_COL: feat_df.loc[future_mask, ID_COL].to_numpy(),
            "rto": np.round(pred, 2),
        }
    )
    return submission.sort_values(ID_COL).reset_index(drop=True)


def validate_submission(submission: pd.DataFrame, expected_rows: int = EXPECTED_SUBMISSION_ROWS) -> None:
    if list(submission.columns) != [ID_COL, "rto"]:
        raise ValueError(f"Submission columns must be {[ID_COL, 'rto']}, got {list(submission.columns)}")
    if len(submission) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(submission)}")
    if submission.isna().any().any():
        raise ValueError("Submission contains NaN values")
    if (submission["rto"] < 0).any():
        raise ValueError("Submission contains negative rto values")
    if not submission[ID_COL].is_unique:
        raise ValueError("Submission new_id values must be unique")


def save_submission(
    submission: pd.DataFrame,
    path: Path,
    make_test_copy: bool = False,
    write_header: bool = False,
) -> Path:
    validate_submission(submission)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False, header=write_header)
    if path.stat().st_size >= 1_000_000:
        raise ValueError(f"Submission is too large: {path.stat().st_size} bytes")
    if make_test_copy:
        shutil.copy2(path, PROJECT_ROOT / "test.csv")
    return path
