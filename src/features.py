import numpy as np
import pandas as pd

from .config import CATEGORICAL_COLS, FUTURE_MONTH, ID_COL, MONTH_COL, TARGET_COL


LAG_WINDOWS = [1, 2, 3, 4, 5, 6]
ROLLING_MEAN_WINDOWS = [2, 3, 4, 6]


def load_train(path) -> pd.DataFrame:
    return pd.read_csv(path)


def add_future_month_rows(df: pd.DataFrame, future_month: int = FUTURE_MONTH) -> pd.DataFrame:
    """Append one month-ahead rows with static store attributes copied from latest history."""
    latest = (
        df.sort_values([ID_COL, MONTH_COL])
        .groupby(ID_COL, as_index=False, sort=False)
        .tail(1)
        .copy()
    )
    latest[MONTH_COL] = future_month
    latest[TARGET_COL] = np.nan
    return pd.concat([df, latest], ignore_index=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build leakage-free lag, rolling, dynamics, and calendar features."""
    out = df.sort_values([ID_COL, MONTH_COL]).copy()
    grouped = out.groupby(ID_COL, sort=False)[TARGET_COL]

    for lag in LAG_WINDOWS:
        out[f"rto_lag_{lag}"] = grouped.shift(lag)

    shifted = grouped.shift(1)
    for window in ROLLING_MEAN_WINDOWS:
        out[f"rto_mean_{window}"] = shifted.groupby(out[ID_COL], sort=False).rolling(window).mean().reset_index(level=0, drop=True)

    for window in [3, 6]:
        out[f"rto_std_{window}"] = shifted.groupby(out[ID_COL], sort=False).rolling(window).std().reset_index(level=0, drop=True)
        out[f"rto_median_{window}"] = shifted.groupby(out[ID_COL], sort=False).rolling(window).median().reset_index(level=0, drop=True)

    out["rto_min_3"] = shifted.groupby(out[ID_COL], sort=False).rolling(3).min().reset_index(level=0, drop=True)
    out["rto_max_3"] = shifted.groupby(out[ID_COL], sort=False).rolling(3).max().reset_index(level=0, drop=True)

    out["growth_1"] = out["rto_lag_1"] / out["rto_lag_2"] - 1.0
    out["growth_2"] = out["rto_lag_2"] / out["rto_lag_3"] - 1.0
    out["lag1_to_mean3"] = out["rto_lag_1"] / out["rto_mean_3"]
    out["lag1_to_mean6"] = out["rto_lag_1"] / out["rto_mean_6"]

    out["month_num"] = out[MONTH_COL].astype(float)
    out["month_sin"] = np.sin(2.0 * np.pi * out[MONTH_COL] / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * out[MONTH_COL] / 12.0)

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {TARGET_COL}
    return [col for col in df.columns if col not in excluded]


def get_cat_features(feature_columns: list[str]) -> list[str]:
    return [col for col in CATEGORICAL_COLS if col in feature_columns]


def prepare_model_frame(df: pd.DataFrame, feature_columns: list[str], cat_features: list[str]) -> pd.DataFrame:
    """Return a CatBoost/sklearn-friendly feature matrix."""
    x = df.loc[:, feature_columns].copy()
    for col in cat_features:
        x[col] = x[col].astype("string").fillna("__MISSING__")
    return x
