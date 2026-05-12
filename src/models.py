from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .config import MONTH_COL, RANDOM_SEED, TARGET_COL
from .features import prepare_model_frame


def _safe_clip(pred, lower: float = 0.0, upper: float | None = None) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    if upper is None:
        return np.maximum(pred, lower)
    return np.clip(pred, lower, upper)


class BaselineLastMonth:
    backend = "last_month"

    def fit(self, x: pd.DataFrame, y, cat_features=None):
        y = np.asarray(y, dtype=float)
        self.fallback_ = float(np.nanmedian(y))
        self.upper_ = float(np.nanquantile(y, 0.995) * 1.5)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        pred = x["rto_lag_1"].fillna(self.fallback_).to_numpy(dtype=float)
        return _safe_clip(pred, upper=self.upper_)


class BaselineGlobalGrowth:
    backend = "global_growth"

    def fit(self, x: pd.DataFrame, y, cat_features=None):
        train = x.copy()
        train[TARGET_COL] = np.asarray(y, dtype=float)
        latest_month = int(train[MONTH_COL].max())
        latest = train[(train[MONTH_COL] == latest_month) & (train["rto_lag_1"] > 0)]
        ratio = latest[TARGET_COL] / latest["rto_lag_1"]
        ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
        self.growth_ratio_ = float(ratio.median()) if len(ratio) else 1.0
        self.fallback_ = float(np.nanmedian(y))
        self.upper_ = float(np.nanquantile(y, 0.995) * 1.5)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        base = x["rto_lag_1"].fillna(self.fallback_).to_numpy(dtype=float)
        return _safe_clip(base * self.growth_ratio_, upper=self.upper_)


class TrendFallbackRegressor:
    """Deterministic dependency-free fallback when CatBoost/sklearn are unavailable."""

    backend = "trend_fallback"

    def __init__(self, log_target: bool = False):
        self.log_target = log_target

    def fit(self, x: pd.DataFrame, y, cat_features=None):
        y = np.asarray(y, dtype=float)
        train = x.copy()
        train[TARGET_COL] = y
        latest_month = int(train[MONTH_COL].max())
        latest = train[(train[MONTH_COL] == latest_month) & (train["rto_lag_1"] > 0)].copy()
        latest["ratio"] = latest[TARGET_COL] / latest["rto_lag_1"]
        latest["ratio"] = latest["ratio"].replace([np.inf, -np.inf], np.nan)
        latest["ratio"] = latest["ratio"].clip(0.25, 3.0)

        self.global_ratio_ = float(latest["ratio"].median()) if latest["ratio"].notna().any() else 1.0
        self.fallback_ = float(np.nanmedian(y))
        self.upper_ = float(np.nanquantile(y, 0.997) * 1.35)
        self.cat_ratio_maps_ = {}

        for col in ["Регион", "Торговая площадь, категориальный", "Дата открытия, категориальный"]:
            if col not in latest:
                continue
            stats = latest.groupby(col, dropna=False)["ratio"].agg(["median", "count"])
            stats = stats[stats["count"] >= 25]["median"]
            self.cat_ratio_maps_[col] = stats.to_dict()

        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        lag1 = x["rto_lag_1"].fillna(self.fallback_).astype(float).to_numpy()
        mean3 = x.get("rto_mean_3", pd.Series(np.nan, index=x.index)).fillna(x["rto_lag_1"]).astype(float).to_numpy()
        pred = 0.98 * lag1 + 0.02 * mean3
        if self.log_target:
            pred = np.expm1(np.log1p(np.maximum(pred, 0.0)))
        return _safe_clip(pred, upper=self.upper_)


class SklearnHistGradientFallback:
    backend = "hist_gradient_boosting"

    def __init__(self, log_target: bool = False):
        self.log_target = log_target

    def fit(self, x: pd.DataFrame, y, cat_features=None):
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.cat_features_ = cat_features or []
        self.feature_columns_ = list(x.columns)
        x_fit = self._fit_transform(x)
        y_fit = np.asarray(y, dtype=float)
        if self.log_target:
            y_fit = np.log1p(np.maximum(y_fit, 0.0))
            loss = "squared_error"
        else:
            loss = "absolute_error"
        self.model_ = HistGradientBoostingRegressor(
            loss=loss,
            learning_rate=0.05,
            max_iter=450,
            l2_regularization=0.05,
            random_state=RANDOM_SEED,
        )
        self.model_.fit(x_fit, y_fit)
        self.upper_ = float(np.nanquantile(y, 0.997) * 1.35)
        return self

    def _fit_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x_out = x.loc[:, self.feature_columns_].copy()
        self.category_maps_ = {}
        self.medians_ = {}
        for col in self.feature_columns_:
            if col in self.cat_features_:
                values = x_out[col].astype("string").fillna("__MISSING__")
                cats = pd.Index(values.unique())
                self.category_maps_[col] = {value: idx for idx, value in enumerate(cats)}
                x_out[col] = values.map(self.category_maps_[col]).fillna(-1).astype(float)
            else:
                x_out[col] = pd.to_numeric(x_out[col], errors="coerce")
                self.medians_[col] = float(x_out[col].median()) if x_out[col].notna().any() else 0.0
                x_out[col] = x_out[col].fillna(self.medians_[col])
        return x_out

    def _transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x_out = x.loc[:, self.feature_columns_].copy()
        for col in self.feature_columns_:
            if col in self.cat_features_:
                values = x_out[col].astype("string").fillna("__MISSING__")
                x_out[col] = values.map(self.category_maps_[col]).fillna(-1).astype(float)
            else:
                x_out[col] = pd.to_numeric(x_out[col], errors="coerce").fillna(self.medians_[col])
        return x_out

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        pred = self.model_.predict(self._transform(x))
        if self.log_target:
            pred = np.expm1(pred)
        return _safe_clip(pred, upper=self.upper_)


class CatBoostMAPEModel:
    def __init__(self):
        self.backend = "catboost"
        self.model = None
        self.fallback = TrendFallbackRegressor(log_target=False)

    def fit(self, x: pd.DataFrame, y, cat_features=None):
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            try:
                self.fallback = SklearnHistGradientFallback(log_target=False)
                self.fallback.fit(x, y, cat_features)
                self.backend = self.fallback.backend
                warnings.warn("catboost is not installed; using sklearn HistGradientBoostingRegressor.")
            except ImportError:
                warnings.warn("catboost and sklearn are not installed; using deterministic TrendFallbackRegressor.")
                self.fallback = TrendFallbackRegressor(log_target=False)
                self.fallback.fit(x, y, cat_features)
                self.backend = self.fallback.backend
            return self

        self.cat_features_ = cat_features or []
        x_fit = prepare_model_frame(x, list(x.columns), self.cat_features_)
        self.model = CatBoostRegressor(
            iterations=1200,
            learning_rate=0.04,
            depth=8,
            loss_function="MAPE",
            eval_metric="MAPE",
            random_seed=RANDOM_SEED,
            allow_writing_files=False,
            verbose=False,
        )
        self.model.fit(x_fit, y, cat_features=self.cat_features_)
        self.upper_ = float(np.nanquantile(y, 0.997) * 1.35)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return self.fallback.predict(x)
        x_pred = prepare_model_frame(x, list(x.columns), self.cat_features_)
        return _safe_clip(self.model.predict(x_pred), upper=self.upper_)


class CatBoostLogModel:
    def __init__(self):
        self.backend = "catboost_log"
        self.model = None
        self.fallback = TrendFallbackRegressor(log_target=True)

    def fit(self, x: pd.DataFrame, y, cat_features=None):
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            try:
                self.fallback = SklearnHistGradientFallback(log_target=True)
                self.fallback.fit(x, y, cat_features)
                self.backend = self.fallback.backend
                warnings.warn("catboost is not installed; using sklearn HistGradientBoostingRegressor on log1p target.")
            except ImportError:
                warnings.warn("catboost and sklearn are not installed; using deterministic TrendFallbackRegressor.")
                self.fallback = TrendFallbackRegressor(log_target=True)
                self.fallback.fit(x, y, cat_features)
                self.backend = self.fallback.backend
            return self

        self.cat_features_ = cat_features or []
        y_log = np.log1p(np.maximum(np.asarray(y, dtype=float), 0.0))
        x_fit = prepare_model_frame(x, list(x.columns), self.cat_features_)
        self.model = CatBoostRegressor(
            iterations=1400,
            learning_rate=0.035,
            depth=8,
            loss_function="RMSE",
            eval_metric="MAE",
            random_seed=RANDOM_SEED,
            allow_writing_files=False,
            verbose=False,
        )
        self.model.fit(x_fit, y_log, cat_features=self.cat_features_)
        self.upper_ = float(np.nanquantile(y, 0.997) * 1.35)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return self.fallback.predict(x)
        x_pred = prepare_model_frame(x, list(x.columns), self.cat_features_)
        pred = np.expm1(self.model.predict(x_pred))
        return _safe_clip(pred, upper=self.upper_)


def get_model(name: str):
    if name == "baseline_last_month":
        return BaselineLastMonth()
    if name == "baseline_global_growth":
        return BaselineGlobalGrowth()
    if name == "catboost_mape":
        return CatBoostMAPEModel()
    if name == "catboost_log":
        return CatBoostLogModel()
    raise ValueError(f"Unknown model: {name}")
