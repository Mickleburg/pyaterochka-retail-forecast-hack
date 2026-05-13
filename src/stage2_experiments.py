from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from . import advanced_experiments as base
from .config import ID_COL, MONTH_COL, REPORTS_DIR, SUBMISSIONS_DIR, TARGET_COL, TRAIN_PATH, RANDOM_SEED, PROJECT_ROOT
from .features import load_train
from .metrics import mape_percent
from .registry import (
    LEADERBOARD_RESULTS_PATH,
    SUBMISSION_REGISTRY_PATH,
    load_best_submission,
    load_leaderboard,
    restore_best_submission,
    save_best_submission,
)
from .submit import save_submission, validate_saved_csv


FOLDS = [8, 9, 10]
CURRENT_BEST_FILE = "submissions/test_cluster_blend_v1.csv"
CURRENT_BEST_MODEL = "cluster_blend_v1"
PRED_CACHE: dict[tuple, pd.Series] = {}


@dataclass
class Experiment:
    experiment_name: str
    model_name: str
    hypothesis_group: str
    candidate_class: str
    pred_func: callable
    preferred_filename: str
    comment: str
    expected_direction: str = "mixed"


def ensure_new_leaderboard_results() -> None:
    rows = [
        ("submissions/test_cluster_blend_v1.csv", "cluster_blend_v1", 95.91, "OK", "новый best: cluster blend по траектории"),
        ("submissions/test_temporal_ridge_ratio_v1.csv", "temporal_ridge_ratio_v1", 95.91, "OK", "temporal ridge ratio, тот же LB score 95.91"),
        ("submissions/test_exploratory_segment_bias_v1.csv", "exploratory_segment_bias_v1", 95.90, "OK", "исследовательская segment bias correction"),
        ("submissions/test_selector_by_rto_decile_v1.csv", "selector_by_rto_decile_v1", 95.89, "OK", "selector по decile РТО"),
        ("submissions/test_decile_beta_ratio_v1.csv", "decile_beta_ratio_v1", 95.88, "OK", "разная beta ratio по decile РТО"),
        ("submissions/test_weighted_oof_global_v1.csv", "weighted_oof_global_v1", 95.88, "OK", "глобальная OOF mixture"),
        ("submissions/test_temporal_ensemble_ratio_v1.csv", "temporal_ensemble_ratio_v1", 95.87, "OK", "temporal ensemble оказался хуже одиночного ridge"),
        ("submissions/test_segment_calibrated_trend_v1.csv", "segment_calibrated_trend_v1", 95.87, "OK", "segment calibration по тренду"),
        ("submissions/test_segment_calibrated_region_rto_v1.csv", "segment_calibrated_region_rto_v1", 95.86, "OK", "segment calibration region x rto"),
        ("submissions/test_segment_calibrated_rto_decile_v1.csv", "segment_calibrated_rto_decile_v1", 95.86, "OK", "segment calibration по decile РТО"),
        ("submissions/test_october_high_rollback_v1.csv", "october_high_rollback_v1", 95.83, "OK", "high October rollback оказался слишком вредным"),
    ]
    lb = load_leaderboard()
    for filename, model_name, score, verdict, comment in rows:
        mask = (
            (lb["filename"].astype(str) == filename)
            & (lb["model_name"].astype(str) == model_name)
            & (pd.to_numeric(lb["lb_score"], errors="coerce").round(8) == round(score, 8))
            & (lb["verdict"].astype(str) == verdict)
        )
        if not mask.any():
            lb = pd.concat(
                [
                    lb,
                    pd.DataFrame(
                        [
                            {
                                "submitted_at": "",
                                "filename": filename,
                                "model_name": model_name,
                                "local_cv_mape": "",
                                "lb_score": score,
                                "lb_mape": 100.0 - score,
                                "verdict": verdict,
                                "comment": comment,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    lb.to_csv(LEADERBOARD_RESULTS_PATH, index=False, encoding="utf-8")


def ensure_current_best() -> None:
    best = {
        "filename": CURRENT_BEST_FILE,
        "model_name": CURRENT_BEST_MODEL,
        "lb_score": 95.91,
        "lb_mape": 4.09,
        "verdict": "OK",
        "is_confirmed_by_leaderboard": True,
    }
    current = load_best_submission()
    if float(current.get("lb_score", 0.0)) <= best["lb_score"]:
        save_best_submission(best)


def ratio_best_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int) -> pd.Series:
    return base.ratio_shrink_pred(df, pivot, month, 0.06, (0.97, 1.03))


def cluster_blend_v1_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int) -> pd.Series:
    key = ("cluster_blend_v1", month)
    if key in PRED_CACHE:
        return PRED_CACHE[key].copy()
    ratio = ratio_best_pred(df, pivot, month)
    if month < 8:
        PRED_CACHE[key] = ratio.copy()
        return ratio
    cluster = base.cluster_ratio_pred(df, pivot, month, 20, 0.18, (0.97, 1.03))
    pred = base.clip_series(0.70 * ratio + 0.30 * cluster, index=pivot.index)
    PRED_CACHE[key] = pred.copy()
    return pred


def temporal_ridge_v1_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int) -> pd.Series:
    return fast_temporal_model_pred(pivot, month, "ridge", 0.18, (0.97, 1.03))


def current_best_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int) -> pd.Series:
    return cluster_blend_v1_pred(df, pivot, month)


def fast_temporal_model_pred(pivot: pd.DataFrame, month: int, kind: str, beta: float, clip_bounds: tuple[float, float]) -> pd.Series:
    key = ("fast_temporal", kind, month, float(beta), tuple(clip_bounds))
    if key in PRED_CACHE:
        return PRED_CACHE[key].copy()
    try:
        from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return base.baseline_pred(pivot, month)
    xs, ys = [], []
    for t in range(7, month):
        xs.append(base.temporal_features(pivot, t))
        target = np.log(base.safe_ratio(pivot[t], pivot[t - 1], 1.0).clip(0.70, 1.30))
        ys.append(target)
    x_train = pd.concat(xs)
    y_train = pd.concat(ys)
    if kind == "hgb":
        model = HistGradientBoostingRegressor(max_iter=80, learning_rate=0.04, l2_regularization=4.0, random_state=RANDOM_SEED)
    elif kind == "extra":
        model = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=120, random_state=RANDOM_SEED, n_jobs=-1)
    else:
        model = make_pipeline(StandardScaler(), Ridge(alpha=20.0, random_state=RANDOM_SEED))
    model.fit(x_train, y_train)
    raw = pd.Series(model.predict(base.temporal_features(pivot, month)), index=pivot.index)
    raw = raw - raw.median()
    ratio = np.exp(beta * raw).clip(*clip_bounds)
    pred = base.clip_series(base.baseline_pred(pivot, month) * ratio, index=pivot.index)
    PRED_CACHE[key] = pred.copy()
    return pred


def temporal_v2_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int, kind: str, beta: float, clip_bounds: tuple[float, float], blend: float = 1.0) -> pd.Series:
    pred = fast_temporal_model_pred(pivot, month, kind, beta, clip_bounds)
    cur = current_best_pred(df, pivot, month)
    return base.clip_series(blend * pred + (1.0 - blend) * cur, index=pivot.index)


def cluster_v2_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int, k: int, beta: float, clip_bounds: tuple[float, float], blend: float) -> pd.Series:
    key = ("cluster_v2", month, int(k), float(beta), tuple(clip_bounds), float(blend))
    if key in PRED_CACHE:
        return PRED_CACHE[key].copy()
    cur = current_best_pred(df, pivot, month)
    if month < 8:
        return cur
    pred = base.cluster_ratio_pred(df, pivot, month, k, beta, clip_bounds)
    out = base.clip_series(blend * pred + (1.0 - blend) * cur, index=pivot.index)
    PRED_CACHE[key] = out.copy()
    return out


def analog_v2_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int, beta: float, blend: float, clip_bounds=(0.97, 1.03)) -> pd.Series:
    key = ("analog_v2", month, float(beta), float(blend), tuple(clip_bounds))
    if key in PRED_CACHE:
        return PRED_CACHE[key].copy()
    pred = base.analog_knn_pred(df, pivot, month, beta, clip_bounds)
    cur = current_best_pred(df, pivot, month)
    out = base.clip_series(blend * pred + (1.0 - blend) * cur, index=pivot.index)
    PRED_CACHE[key] = out.copy()
    return out


def trajectory_features_v2(df: pd.DataFrame, pivot: pd.DataFrame, month: int) -> pd.DataFrame:
    feats = base.dynamic_features(df, pivot, month).copy()
    tf = base.temporal_features(pivot, month)
    for col in tf.columns:
        feats[f"traj_{col}"] = tf[col]
    feats["regime"] = detect_regime(feats)
    return feats


def detect_regime(feats: pd.DataFrame) -> pd.Series:
    regime = pd.Series("stable", index=feats.index, dtype=object)
    regime.loc[(feats["volatility"] > 0.07) & (feats["signed_outlier"].abs() <= 0.10)] = "volatile"
    regime.loc[(feats["trend"] > 0.04) & (feats["volatility"] <= 0.07)] = "growing"
    regime.loc[(feats["trend"] < -0.04) & (feats["volatility"] <= 0.07)] = "declining"
    regime.loc[feats["signed_outlier"] > 0.10] = "october_spike"
    regime.loc[feats["signed_outlier"] < -0.10] = "october_drop"
    regime.loc[(feats["rto_decile"] >= 8) & (feats["volatility"] <= 0.04)] = "high_stable"
    regime.loc[(feats["rto_decile"] <= 1) & (feats["volatility"] > 0.06)] = "low_volatile"
    return regime


def segment_temporal_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int, segment_col: str) -> pd.Series:
    cur = current_best_pred(df, pivot, month)
    temporal = temporal_ridge_v1_pred(df, pivot, month)
    cluster = cluster_blend_v1_pred(df, pivot, month)
    feats = trajectory_features_v2(df, pivot, month)
    pred = cur.copy()

    rows = []
    for t in FOLDS:
        if t >= month:
            continue
        f = trajectory_features_v2(df, pivot, t)
        y = pivot[t]
        candidates = {
            "current": current_best_pred(df, pivot, t),
            "temporal": temporal_ridge_v1_pred(df, pivot, t),
            "cluster": cluster_blend_v1_pred(df, pivot, t),
            "rolling": base.rolling_median_pred(pivot, t, 3),
        }
        part = f[[segment_col]].copy()
        for name, p in candidates.items():
            part[f"{name}_ape"] = base.ape(y, p)
        rows.append(part)
    if not rows:
        return cur
    hist = pd.concat(rows)
    winners = {}
    for value, part in hist.groupby(segment_col, dropna=False):
        if len(part) < 150:
            continue
        mapes = {name: float(part[f"{name}_ape"].mean()) for name in ["current", "temporal", "cluster", "rolling"]}
        winner = min(mapes, key=mapes.get)
        if mapes["current"] - mapes[winner] > 0.0003:
            winners[value] = winner
    for value, winner in winners.items():
        mask = feats[segment_col].eq(value)
        if winner == "temporal":
            pred.loc[mask] = 0.65 * temporal.loc[mask] + 0.35 * cur.loc[mask]
        elif winner == "cluster":
            pred.loc[mask] = cluster.loc[mask]
        elif winner == "rolling":
            rolling = base.rolling_median_pred(pivot, month, 3)
            pred.loc[mask] = 0.25 * rolling.loc[mask] + 0.75 * cur.loc[mask]
    return base.clip_series(pred, index=pivot.index)


def mape_decile_strategy_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int) -> pd.Series:
    cur = current_best_pred(df, pivot, month)
    temporal = temporal_ridge_v1_pred(df, pivot, month)
    cluster = cluster_blend_v1_pred(df, pivot, month)
    rolling = base.rolling_median_pred(pivot, month, 3)
    feats = trajectory_features_v2(df, pivot, month)
    pred = cur.copy()
    low = feats["rto_decile"] <= 1
    mid = feats["rto_decile"].between(2, 6)
    high = feats["rto_decile"] >= 8
    pred.loc[low] = 0.60 * cur.loc[low] + 0.25 * temporal.loc[low] + 0.15 * rolling.loc[low]
    pred.loc[mid] = 0.70 * cur.loc[mid] + 0.20 * temporal.loc[mid] + 0.10 * cluster.loc[mid]
    pred.loc[high] = 0.75 * cur.loc[high] + 0.25 * temporal.loc[high]
    return base.clip_series(pred, index=pivot.index)


def regime_selector_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int, mode: str) -> pd.Series:
    cur = current_best_pred(df, pivot, month)
    temporal = temporal_ridge_v1_pred(df, pivot, month)
    cluster = cluster_v2_pred(df, pivot, month, 30, 0.22, (0.97, 1.03), 1.0)
    analog = analog_v2_pred(df, pivot, month, 0.18, 1.0, (0.97, 1.03))
    rolling = base.rolling_median_pred(pivot, month, 3)
    feats = trajectory_features_v2(df, pivot, month)
    pred = cur.copy()
    if mode == "selector":
        pred.loc[feats["regime"].isin(["growing", "declining", "high_stable"])] = temporal.loc[feats["regime"].isin(["growing", "declining", "high_stable"])]
        pred.loc[feats["regime"].isin(["volatile", "low_volatile"])] = cluster.loc[feats["regime"].isin(["volatile", "low_volatile"])]
        pred.loc[feats["regime"].isin(["october_spike", "october_drop"])] = 0.80 * cur.loc[feats["regime"].isin(["october_spike", "october_drop"])] + 0.20 * rolling.loc[feats["regime"].isin(["october_spike", "october_drop"])]
        return base.clip_series(0.70 * pred + 0.30 * cur, index=pivot.index)
    if mode == "blend":
        pred = 0.55 * cur + 0.25 * temporal + 0.15 * cluster + 0.05 * analog
        outlier = feats["regime"].isin(["october_spike", "october_drop"])
        pred.loc[outlier] = 0.85 * cur.loc[outlier] + 0.15 * rolling.loc[outlier]
        return base.clip_series(pred, index=pivot.index)
    # temporal mode
    stable = feats["regime"].isin(["stable", "growing", "declining", "high_stable"])
    pred.loc[stable] = 0.45 * temporal.loc[stable] + 0.55 * cur.loc[stable]
    return base.clip_series(pred, index=pivot.index)


def residual_v2_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int, kind: str, beta: float, clip_bounds=(0.98, 1.02)) -> pd.Series:
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.compose import ColumnTransformer
    except Exception:
        return current_best_pred(df, pivot, month)

    rows = []
    for t in range(7, month):
        feats = trajectory_features_v2(df, pivot, t)
        pred = current_best_pred(df, pivot, t)
        part = feats[[base.REGION_COL, base.AREA_COL, "rto_decile", "volatility_bucket", "trend_bucket", "outlier_bucket", "volatility", "trend", "signed_outlier"]].copy()
        part["target"] = np.log(base.safe_ratio(pivot[t], pred, 1.0)).clip(-0.20, 0.20)
        rows.append(part)
    train = pd.concat(rows)
    current = trajectory_features_v2(df, pivot, month)
    x_cols = [base.REGION_COL, base.AREA_COL, "rto_decile", "volatility_bucket", "trend_bucket", "outlier_bucket", "volatility", "trend", "signed_outlier"]
    cat_cols = [base.REGION_COL, base.AREA_COL, "rto_decile", "volatility_bucket", "trend_bucket", "outlier_bucket"]
    num_cols = ["volatility", "trend", "signed_outlier"]
    if kind == "hgb":
        # HGB без one-hot категорий: используем числовые bucket-коды.
        x_train = train[x_cols].copy()
        x_test = current[x_cols].copy()
        for col in [base.REGION_COL, base.AREA_COL]:
            mapping = {v: i for i, v in enumerate(pd.concat([x_train[col], x_test[col]]).astype(str).unique())}
            x_train[col] = x_train[col].astype(str).map(mapping)
            x_test[col] = x_test[col].astype(str).map(mapping)
        model = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.035, l2_regularization=3.0, random_state=RANDOM_SEED)
        model.fit(x_train, train["target"])
        raw = pd.Series(model.predict(x_test), index=pivot.index)
    elif kind == "extra":
        x_train = train[x_cols].copy()
        x_test = current[x_cols].copy()
        for col in [base.REGION_COL, base.AREA_COL]:
            mapping = {v: i for i, v in enumerate(pd.concat([x_train[col], x_test[col]]).astype(str).unique())}
            x_train[col] = x_train[col].astype(str).map(mapping)
            x_test[col] = x_test[col].astype(str).map(mapping)
        model = ExtraTreesRegressor(n_estimators=160, min_samples_leaf=120, random_state=RANDOM_SEED, n_jobs=-1)
        model.fit(x_train, train["target"])
        raw = pd.Series(model.predict(x_test), index=pivot.index)
    else:
        pre = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=100), cat_cols),
                ("num", StandardScaler(), num_cols),
            ],
            sparse_threshold=0.3,
        )
        model = make_pipeline(pre, Ridge(alpha=50.0, random_state=RANDOM_SEED))
        model.fit(train[x_cols], train["target"])
        raw = pd.Series(model.predict(current[x_cols]), index=pivot.index)
    raw = raw - raw.median()
    mult = np.exp(beta * raw).clip(*clip_bounds)
    cur = current_best_pred(df, pivot, month)
    return base.clip_series(cur * mult, index=pivot.index)


def hybrid_cluster_temporal_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int, mode: str) -> pd.Series:
    cur = current_best_pred(df, pivot, month)
    temporal = temporal_v2_pred(df, pivot, month, "ridge", 0.22, (0.97, 1.03), 1.0)
    cluster = cluster_v2_pred(df, pivot, month, 30, 0.22, (0.97, 1.03), 1.0)
    if mode == "simple":
        raw = 0.50 * cluster + 0.50 * temporal
        return base.clip_series(0.80 * raw + 0.20 * cur, index=pivot.index)
    feats = trajectory_features_v2(df, pivot, month)
    if mode == "regime":
        raw = regime_selector_pred(df, pivot, month, "selector")
        return base.clip_series(0.85 * raw + 0.15 * cur, index=pivot.index)
    pred = cur.copy()
    low = feats["rto_decile"] <= 2
    high = feats["rto_decile"] >= 8
    pred.loc[low] = 0.60 * cur.loc[low] + 0.25 * cluster.loc[low] + 0.15 * temporal.loc[low]
    pred.loc[high] = 0.55 * cur.loc[high] + 0.35 * temporal.loc[high] + 0.10 * cluster.loc[high]
    mid = ~(low | high)
    pred.loc[mid] = 0.60 * cur.loc[mid] + 0.20 * cluster.loc[mid] + 0.20 * temporal.loc[mid]
    return base.clip_series(pred, index=pivot.index)


def current_best_4digits_submission(pred: pd.Series, path: Path) -> bool:
    sub = pd.DataFrame({ID_COL: pred.index, "rto": np.round(pred.astype(float).to_numpy(), 4)}).sort_values(ID_COL).reset_index(drop=True)
    sub.to_csv(path, index=False, encoding="utf-8")
    if path.stat().st_size >= 1_000_000:
        path.unlink(missing_ok=True)
        return False
    validate_saved_csv(path)
    return True


def build_experiments(df: pd.DataFrame, pivot: pd.DataFrame) -> list[Experiment]:
    return [
        Experiment("cluster_blend_v1", "cluster_blend_v1", "confirmed", "reference", lambda m: current_best_pred(df, pivot, m), "", "Текущий best 95.91: blend ratio_shrink и cluster."),
        Experiment("temporal_ridge_ratio_v1", "temporal_ridge_ratio_v1", "confirmed", "reference", lambda m: temporal_ridge_v1_pred(df, pivot, m), "", "Второй best-like файл 95.91: temporal ridge."),
        Experiment("cluster_path_k20_v2", "cluster_path_v2", "trajectory_clustering_v2", "safe", lambda m: cluster_v2_pred(df, pivot, m, 20, 0.24, (0.97, 1.03), 0.55), "test_cluster_path_k20_v2.csv", "Кластерная поправка по траектории k=20, усиленная относительно v1."),
        Experiment("cluster_ratio_k30_v2", "cluster_ratio_v2", "trajectory_clustering_v2", "moderate", lambda m: cluster_v2_pred(df, pivot, m, 30, 0.28, (0.97, 1.03), 0.65), "test_cluster_ratio_k30_v2.csv", "Кластерная поправка k=30, больше вес cluster signal."),
        Experiment("cluster_pca_k20_v2", "cluster_pca_v2", "trajectory_clustering_v2", "moderate", lambda m: cluster_v2_pred(df, pivot, m, 20, 0.30, (0.95, 1.05), 0.50), "test_cluster_pca_k20_v2.csv", "PCA-like proxy: более широкий clip для cluster trajectory."),
        Experiment("cluster_selector_v2", "cluster_selector_v2", "trajectory_clustering_v2", "moderate", lambda m: segment_temporal_pred(df, pivot, m, "volatility_bucket"), "test_cluster_selector_v2.csv", "Selector по volatility bucket выбирает current/cluster/temporal/rolling."),
        Experiment("cluster_temporal_blend_v2", "cluster_temporal_blend_v2", "trajectory_clustering_v2", "safe", lambda m: hybrid_cluster_temporal_pred(df, pivot, m, "simple"), "test_cluster_temporal_blend_v2.csv", "Равная смесь cluster v2 и temporal v2 с safety blend к current best."),
        Experiment("temporal_ridge_v2", "temporal_ridge_v2", "temporal_models_v2", "moderate", lambda m: temporal_v2_pred(df, pivot, m, "ridge", 0.24, (0.97, 1.03), 0.80), "test_temporal_ridge_v2.csv", "Temporal ridge v2: больше beta, но blend с current best."),
        Experiment("temporal_huber_v2", "temporal_huber_v2", "temporal_models_v2", "moderate", lambda m: temporal_v2_pred(df, pivot, m, "ridge", 0.30, (0.95, 1.05), 0.55), "test_temporal_huber_v2.csv", "Robust linear proxy: более широкий temporal сигнал, ослабленный blend."),
        Experiment("temporal_hgb_v2", "temporal_hgb_v2", "temporal_models_v2", "exploratory", lambda m: temporal_v2_pred(df, pivot, m, "hgb", 0.16, (0.97, 1.03), 0.70), "test_temporal_hgb_v2.csv", "HistGradientBoosting по траекторным признакам."),
        Experiment("temporal_extratrees_v2", "temporal_extratrees_v2", "temporal_models_v2", "exploratory", lambda m: temporal_v2_pred(df, pivot, m, "extra", 0.16, (0.97, 1.03), 0.65), "test_temporal_extratrees_v2.csv", "ExtraTrees по temporal features."),
        Experiment("temporal_logratio_blend_v2", "temporal_logratio_blend_v2", "temporal_models_v2", "safe", lambda m: 0.65 * current_best_pred(df, pivot, m) + 0.35 * temporal_v2_pred(df, pivot, m, "ridge", 0.22, (0.98, 1.02), 1.0), "test_temporal_logratio_blend_v2.csv", "Лог-ratio temporal correction с мягким blend."),
        Experiment("analog_path_v2", "analog_path_v2", "analog_v2", "moderate", lambda m: analog_v2_pred(df, pivot, m, 0.24, 0.55, (0.97, 1.03)), "test_analog_path_v2.csv", "Analog v2 по похожим траекториям, k/bucket proxy, blend 55%."),
        Experiment("analog_ratio_v2", "analog_ratio_v2", "analog_v2", "safe", lambda m: analog_v2_pred(df, pivot, m, 0.18, 0.45, (0.98, 1.02)), "test_analog_ratio_v2.csv", "Более мягкий analog по ratio path."),
        Experiment("analog_segmented_v2", "analog_segmented_v2", "analog_v2", "moderate", lambda m: 0.70 * current_best_pred(df, pivot, m) + 0.30 * segment_temporal_pred(df, pivot, m, "area_x_volatility"), "test_analog_segmented_v2.csv", "Segmented analog/selector по area x volatility."),
        Experiment("segment_temporal_rto_decile_v1", "segment_temporal", "segment_temporal", "moderate", lambda m: segment_temporal_pred(df, pivot, m, "rto_decile"), "test_segment_temporal_rto_decile_v1.csv", "Segment-specific temporal model по decile РТО."),
        Experiment("segment_temporal_cluster_v1", "segment_temporal", "segment_temporal", "moderate", lambda m: segment_temporal_pred(df, pivot, m, "trend_bucket"), "test_segment_temporal_cluster_v1.csv", "Cluster/regime proxy: segment temporal по trend bucket."),
        Experiment("segment_temporal_volatility_v1", "segment_temporal", "segment_temporal", "moderate", lambda m: segment_temporal_pred(df, pivot, m, "volatility_bucket"), "test_segment_temporal_volatility_v1.csv", "Segment-specific temporal model по volatility bucket."),
        Experiment("mape_decile_strategy_v1", "mape_decile_strategy", "mape_aware", "moderate", lambda m: mape_decile_strategy_pred(df, pivot, m), "test_mape_decile_strategy_v1.csv", "MAPE-aware decile strategy: разные веса temporal/cluster/rolling по decile РТО."),
        Experiment("low_rto_temporal_blend_v1", "low_rto_temporal", "mape_aware", "exploratory", lambda m: 0.75 * current_best_pred(df, pivot, m) + 0.25 * base.low_rto_decile_pred(df, pivot, m), "test_low_rto_temporal_blend_v1.csv", "Нижние decile РТО частично тянутся к rolling median."),
        Experiment("decile_cluster_temporal_v1", "decile_cluster_temporal", "mape_aware", "moderate", lambda m: hybrid_cluster_temporal_pred(df, pivot, m, "decile"), "test_decile_cluster_temporal_v1.csv", "Decile-specific веса cluster/temporal/current best."),
        Experiment("regime_selector_v1", "regime_selector", "regime_detection", "moderate", lambda m: regime_selector_pred(df, pivot, m, "selector"), "test_regime_selector_v1.csv", "Regime detection: stable/growing/declining/volatile/outlier."),
        Experiment("regime_blend_v1", "regime_blend", "regime_detection", "safe", lambda m: regime_selector_pred(df, pivot, m, "blend"), "test_regime_blend_v1.csv", "Regime blend: разные веса по режимам."),
        Experiment("regime_temporal_v1", "regime_temporal", "regime_detection", "moderate", lambda m: regime_selector_pred(df, pivot, m, "temporal"), "test_regime_temporal_v1.csv", "Temporal применяется только в стабильных режимах."),
        Experiment("oof_residual_ridge_v2", "oof_residual_ridge_v2", "residual_v2", "safe", lambda m: residual_v2_pred(df, pivot, m, "ridge", 0.12, (0.98, 1.02)), "test_oof_residual_ridge_v2.csv", "Log residual correction v2: Ridge, сильный shrink."),
        Experiment("oof_residual_hgb_v2", "oof_residual_hgb_v2", "residual_v2", "exploratory", lambda m: residual_v2_pred(df, pivot, m, "hgb", 0.10, (0.98, 1.02)), "test_oof_residual_hgb_v2.csv", "Log residual correction v2: HGB."),
        Experiment("oof_residual_cluster_v2", "oof_residual_cluster_v2", "residual_v2", "moderate", lambda m: residual_v2_pred(df, pivot, m, "extra", 0.10, (0.98, 1.02)), "test_oof_residual_cluster_v2.csv", "Log residual correction v2: ExtraTrees cluster-like."),
        Experiment("hybrid_cluster_temporal_v1", "hybrid_cluster_temporal", "hybrid", "safe", lambda m: hybrid_cluster_temporal_pred(df, pivot, m, "simple"), "test_hybrid_cluster_temporal_v1.csv", "Hybrid: cluster v2 x temporal v2."),
        Experiment("hybrid_regime_cluster_temporal_v1", "hybrid_regime_cluster_temporal", "hybrid", "moderate", lambda m: hybrid_cluster_temporal_pred(df, pivot, m, "regime"), "test_hybrid_regime_cluster_temporal_v1.csv", "Hybrid: regime selector выбирает cluster/temporal."),
        Experiment("hybrid_decile_cluster_temporal_v1", "hybrid_decile_cluster_temporal", "hybrid", "moderate", lambda m: hybrid_cluster_temporal_pred(df, pivot, m, "decile"), "test_hybrid_decile_cluster_temporal_v1.csv", "Hybrid: decile-specific cluster/temporal weights."),
    ]


def ape(y_true: pd.Series, pred: pd.Series) -> pd.Series:
    return ((pred - y_true).abs() / y_true.abs().clip(lower=1e-8)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def score_experiment(exp: Experiment, df: pd.DataFrame, pivot: pd.DataFrame, current_test: pd.Series) -> dict:
    scores = {}
    best_scores = {}
    for month in FOLDS:
        pred = exp.pred_func(month).reindex(pivot.index)
        scores[month] = mape_percent(pivot[month], pred)
        best_scores[month] = mape_percent(pivot[month], current_best_pred(df, pivot, month))
    test_pred = exp.pred_func(11).reindex(current_test.index)
    rel = ((test_pred - current_test) / current_test).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    abs_rel = rel.abs()
    weighted = scores[8] * 0.1 + scores[9] * 0.2 + scores[10] * 0.7
    fold10_delta = scores[10] - best_scores[10]
    safe = weighted + 0.5 * max(0.0, fold10_delta) + 14.0 * abs_rel.mean() + 7.0 * (abs_rel > 0.01).mean()
    exploratory = weighted + 0.25 * max(0.0, fold10_delta) + 2.5 * abs_rel.mean() + 1.5 * (abs_rel > 0.03).mean()
    return {
        "experiment_name": exp.experiment_name,
        "model_name": exp.model_name,
        "hypothesis_group": exp.hypothesis_group,
        "candidate_class": exp.candidate_class,
        "fold8_mape": scores[8],
        "fold9_mape": scores[9],
        "fold10_mape": scores[10],
        "mean_mape": float(np.mean(list(scores.values()))),
        "weighted_mape_127": weighted,
        "safe_risk_score": safe,
        "exploratory_score": exploratory,
        "delta_vs_current_best_fold10": fold10_delta,
        "mean_abs_delta_vs_current_best": float((test_pred - current_test).abs().mean()),
        "mean_rel_delta_vs_current_best": float(rel.mean()),
        "mean_abs_relative_delta_vs_current_best": float(abs_rel.mean()),
        "max_rel_delta_vs_current_best": float(abs_rel.max()),
        "share_abs_delta_vs_current_best_gt_0p1pct": float((abs_rel > 0.001).mean()),
        "share_abs_delta_vs_current_best_gt_0p3pct": float((abs_rel > 0.003).mean()),
        "share_abs_delta_vs_current_best_gt_1pct": float((abs_rel > 0.010).mean()),
        "share_abs_delta_vs_current_best_gt_3pct": float((abs_rel > 0.030).mean()),
        "share_changed_stores": float((abs_rel > 0).mean()),
        "expected_direction": exp.expected_direction,
        "generated_submission": "",
        "comment": exp.comment,
    }


def enrich_lb(results: pd.DataFrame) -> pd.DataFrame:
    lb = load_leaderboard()
    lb["lb_score"] = pd.to_numeric(lb["lb_score"], errors="coerce")
    lb["lb_mape"] = pd.to_numeric(lb["lb_mape"], errors="coerce")
    results["lb_score"] = np.nan
    results["lb_mape"] = np.nan
    results["lb_verdict"] = ""
    for idx, row in results.iterrows():
        hit = lb[lb["model_name"].astype(str).eq(str(row["model_name"]))]
        if len(hit):
            latest = hit.iloc[-1]
            results.loc[idx, "lb_score"] = latest["lb_score"]
            results.loc[idx, "lb_mape"] = latest["lb_mape"]
            results.loc[idx, "lb_verdict"] = latest["verdict"]
    results["delta_lb_vs_best"] = pd.to_numeric(results["lb_score"], errors="coerce") - 95.91
    return results


def duplicate_prediction(pred: pd.Series, existing: list[Path]) -> bool:
    rounded = np.round(pred.sort_index().to_numpy(dtype=float), 2)
    for path in existing:
        try:
            other = pd.read_csv(path).sort_values(ID_COL)["rto"].to_numpy(dtype=float)
        except Exception:
            continue
        if len(other) == len(rounded) and np.array_equal(np.round(other, 2), rounded):
            return True
    return False


def choose_candidates(results: pd.DataFrame) -> list[str]:
    pool = results[results["lb_verdict"].astype(str).eq("") | results["lb_verdict"].isna()].copy()
    pool = pool[(pool["max_rel_delta_vs_current_best"] <= 0.08) & (pool["share_abs_delta_vs_current_best_gt_3pct"] <= 0.35)]
    plan = [
        ("trajectory_clustering_v2", "safe_risk_score", 3),
        ("temporal_models_v2", "exploratory_score", 3),
        ("analog_v2", "exploratory_score", 2),
        ("segment_temporal", "exploratory_score", 2),
        ("mape_aware", "exploratory_score", 2),
        ("regime_detection", "exploratory_score", 2),
        ("residual_v2", "safe_risk_score", 1),
        ("hybrid", "exploratory_score", 2),
    ]
    selected = []
    for group, score, n in plan:
        part = pool[pool["hypothesis_group"].eq(group)].sort_values([score, "fold10_mape"])
        selected.extend(part.head(n)["experiment_name"].tolist())
    return list(dict.fromkeys(selected))[:15]


def best_model_comparison(df: pd.DataFrame, pivot: pd.DataFrame, current_test: pd.Series) -> pd.DataFrame:
    rows = []
    model_funcs = {
        "baseline_last_month": lambda m: base.baseline_pred(pivot, m),
        "ratio_shrink_b0p06": lambda m: ratio_best_pred(df, pivot, m),
        "cluster_blend_v1": lambda m: cluster_blend_v1_pred(df, pivot, m),
        "temporal_ridge_ratio_v1": lambda m: temporal_ridge_v1_pred(df, pivot, m),
    }
    for month in FOLDS:
        y = pivot[month]
        baseline = model_funcs["baseline_last_month"](month)
        temporal = model_funcs["temporal_ridge_ratio_v1"](month)
        cluster = model_funcs["cluster_blend_v1"](month)
        for name, fn in model_funcs.items():
            pred = fn(month)
            a = ape(y, pred)
            rel = ((pred - cluster) / cluster).replace([np.inf, -np.inf], np.nan).fillna(0.0).abs()
            rows.append(
                {
                    "scope": f"fold{month}",
                    "model_name": name,
                    "mape": float(a.mean() * 100),
                    "mae": float((pred - y).abs().mean()),
                    "median_ape": float(a.median()),
                    "p90_ape": float(a.quantile(0.90)),
                    "p95_ape": float(a.quantile(0.95)),
                    "p99_ape": float(a.quantile(0.99)),
                    "mean_signed_error": float((pred / y - 1.0).mean()),
                    "median_pred_div_baseline": float((pred / baseline).median()),
                    "mean_delta_vs_current_best": float((pred - cluster).mean()),
                    "max_delta_vs_current_best": float((pred - cluster).abs().max()),
                    "share_delta_gt_0p1pct": float((rel > 0.001).mean()),
                    "share_delta_gt_0p3pct": float((rel > 0.003).mean()),
                    "share_delta_gt_1pct": float((rel > 0.010).mean()),
                    "share_delta_gt_3pct": float((rel > 0.030).mean()),
                    "share_temporal_better_baseline": float((ape(y, temporal) < ape(y, baseline)).mean()),
                    "share_cluster_better_baseline": float((ape(y, cluster) < ape(y, baseline)).mean()),
                    "share_temporal_cluster_same_direction": float((np.sign(temporal - baseline) == np.sign(cluster - baseline)).mean()),
                }
            )
    baseline_test = base.baseline_pred(pivot, 11)
    for name, pred in {
        "baseline_last_month": baseline_test,
        "ratio_shrink_b0p06": ratio_best_pred(df, pivot, 11),
        "cluster_blend_v1": cluster_blend_v1_pred(df, pivot, 11),
        "temporal_ridge_ratio_v1": temporal_ridge_v1_pred(df, pivot, 11),
    }.items():
        rel = ((pred - current_test) / current_test).replace([np.inf, -np.inf], np.nan).fillna(0.0).abs()
        rows.append(
            {
                "scope": "test",
                "model_name": name,
                "mape": np.nan,
                "mae": np.nan,
                "median_ape": np.nan,
                "p90_ape": np.nan,
                "p95_ape": np.nan,
                "p99_ape": np.nan,
                "mean_signed_error": np.nan,
                "median_pred_div_baseline": float((pred / baseline_test).median()),
                "mean_delta_vs_current_best": float((pred - current_test).mean()),
                "max_delta_vs_current_best": float((pred - current_test).abs().max()),
                "share_delta_gt_0p1pct": float((rel > 0.001).mean()),
                "share_delta_gt_0p3pct": float((rel > 0.003).mean()),
                "share_delta_gt_1pct": float((rel > 0.010).mean()),
                "share_delta_gt_3pct": float((rel > 0.030).mean()),
                "share_temporal_better_baseline": np.nan,
                "share_cluster_better_baseline": np.nan,
                "share_temporal_cluster_same_direction": float((np.sign(temporal_ridge_v1_pred(df, pivot, 11) - baseline_test) == np.sign(current_test - baseline_test)).mean()),
            }
        )
    return pd.DataFrame(rows)


def best_model_segment_report(df: pd.DataFrame, pivot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for month in FOLDS:
        feats = trajectory_features_v2(df, pivot, month)
        y = pivot[month]
        cluster = cluster_blend_v1_pred(df, pivot, month)
        temporal = temporal_ridge_v1_pred(df, pivot, month)
        for seg in ["rto_decile", "volatility_bucket", "trend_bucket", "outlier_bucket", base.REGION_COL, base.AREA_COL, base.ALCOHOL_COL, base.CASH_COL, "regime", "region_x_rto_decile"]:
            min_n = 300 if seg == "region_x_rto_decile" else 100
            for value, idx in feats.groupby(seg, dropna=False).groups.items():
                if len(idx) < min_n:
                    continue
                idx = list(idx)
                cluster_mape = float(ape(y.loc[idx], cluster.loc[idx]).mean() * 100)
                temporal_mape = float(ape(y.loc[idx], temporal.loc[idx]).mean() * 100)
                rows.append(
                    {
                        "fold": month,
                        "segment_type": seg,
                        "segment_value": str(value),
                        "n": len(idx),
                        "cluster_mape": cluster_mape,
                        "temporal_mape": temporal_mape,
                        "winner": "temporal" if temporal_mape < cluster_mape else "cluster",
                        "temporal_minus_cluster_mape": temporal_mape - cluster_mape,
                    }
                )
    return pd.DataFrame(rows)


def write_best_model_comparison_reports(comp: pd.DataFrame, seg: pd.DataFrame) -> None:
    comp.to_csv(REPORTS_DIR / "best_model_comparison.csv", index=False, encoding="utf-8")
    temporal_wins = seg.sort_values("temporal_minus_cluster_mape").head(12)
    cluster_wins = seg.sort_values("temporal_minus_cluster_mape", ascending=False).head(12)
    lines = [
        "# Сравнение лучших моделей",
        "",
        "`cluster_blend_v1` и `temporal_ridge_ratio_v1` оба получили LB `95.91`. Current best выбран `cluster_blend_v1`, потому что он более консервативен: меньше среднее и максимальное отклонение от предыдущего best, при равном LB.",
        "",
        "## Метрики по fold и test",
        "",
        "```text",
        comp.to_string(index=False),
        "```",
        "",
        "## Где temporal_ridge лучше cluster_blend",
        "",
        "```text",
        temporal_wins.to_string(index=False),
        "```",
        "",
        "## Где cluster_blend лучше temporal_ridge",
        "",
        "```text",
        cluster_wins.to_string(index=False),
        "```",
        "",
        "## Вывод",
        "",
        "- Temporal и cluster улучшают пересекающиеся, но не полностью одинаковые группы магазинов.",
        "- Temporal сильнее на части трендовых и режимных сегментов, но может быть агрессивнее по max delta.",
        "- Cluster blend недокорректирует часть трендовых магазинов, зато безопаснее как current best.",
        "- Есть смысл развивать гибриды: regime/decile selector и cluster-specific temporal beta.",
    ]
    (REPORTS_DIR / "best_model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_family_reports(results: pd.DataFrame) -> None:
    reports = {
        "cluster_analysis_v2.md": ("trajectory_clustering_v2", "Кластеризация траекторий v2"),
        "temporal_model_analysis.md": ("temporal_models_v2", "Temporal models v2"),
        "analog_analysis.md": ("analog_v2", "Analog forecasting v2"),
        "regime_analysis.md": ("regime_detection", "Regime detection"),
        "hybrid_analysis.md": ("hybrid", "Hybrid cluster-temporal"),
    }
    for filename, (group, title) in reports.items():
        part = results[results["hypothesis_group"].eq(group)].sort_values("exploratory_score")
        lines = [
            f"# {title}",
            "",
            "```text",
            part[["experiment_name", "candidate_class", "fold10_mape", "weighted_mape_127", "safe_risk_score", "exploratory_score", "mean_abs_relative_delta_vs_current_best", "max_rel_delta_vs_current_best", "generated_submission", "comment"]].to_string(index=False),
            "```",
            "",
            "Вывод: сохранялись только кандидаты с понятной гипотезой и контролируемым отклонением от current best; слишком близкие дубли и слишком резкие варианты отфильтрованы.",
        ]
        (REPORTS_DIR / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_main_reports(results: pd.DataFrame, generated: pd.DataFrame) -> None:
    best = load_best_submission()
    safe = results.sort_values("safe_risk_score").head(20)
    exploratory = results.sort_values("exploratory_score").head(20)
    safe_rec = generated[generated["candidate_class"].eq("safe")].sort_values("safe_risk_score")
    moderate_rec = generated[generated["candidate_class"].eq("moderate")].sort_values("exploratory_score")
    exp_rec = generated[generated["candidate_class"].eq("exploratory")].sort_values("exploratory_score")
    lines = [
        "# Отчет по экспериментам",
        "",
        "## Краткий вывод",
        "",
        "Текущий подтвержденный потолок: `95.91`. `ratio_shrink` дал прирост до `95.88`, но дальше уперся. Новый сигнал leaderboard: траектория магазина и кластеризация по динамике полезнее простых микропоправок.",
        "",
        f"Current best: `{best['filename']}`, score `{best['lb_score']:.2f}`. `test.csv` восстановлен как копия этого файла.",
        "",
        "## Проверенные семейства",
        "",
        "- trajectory clustering v2;",
        "- temporal models v2;",
        "- analog forecasting v2;",
        "- segment-specific temporal;",
        "- MAPE-aware decile strategy;",
        "- regime detection;",
        "- residual correction v2;",
        "- hybrid cluster-temporal.",
        "",
        "## Топ-10 safe candidates",
        "",
        "```text",
        safe[["experiment_name", "hypothesis_group", "candidate_class", "fold10_mape", "weighted_mape_127", "safe_risk_score", "generated_submission", "mean_abs_relative_delta_vs_current_best", "max_rel_delta_vs_current_best"]].head(10).to_string(index=False),
        "```",
        "",
        "## Топ-10 moderate/exploratory candidates",
        "",
        "```text",
        exploratory[["experiment_name", "hypothesis_group", "candidate_class", "fold10_mape", "weighted_mape_127", "exploratory_score", "generated_submission", "mean_abs_relative_delta_vs_current_best", "max_rel_delta_vs_current_best"]].head(10).to_string(index=False),
        "```",
    ]
    (REPORTS_DIR / "experiments.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rec = [
        "# Рекомендованные сабмиты",
        "",
        f"Current best: `{best['filename']}`, score `{best['lb_score']:.2f}`. После неудачной отправки восстановить: `python scripts/restore_best_submission.py`.",
        "",
        "## Safe candidates",
        "",
        "```text",
        safe_rec[["generated_submission", "hypothesis_group", "safe_risk_score", "weighted_mape_127", "mean_abs_relative_delta_vs_current_best", "max_rel_delta_vs_current_best", "share_changed_stores"]].to_string(index=False),
        "```",
        "",
        "## Moderate candidates",
        "",
        "```text",
        moderate_rec[["generated_submission", "hypothesis_group", "exploratory_score", "weighted_mape_127", "mean_abs_relative_delta_vs_current_best", "max_rel_delta_vs_current_best", "share_changed_stores"]].head(8).to_string(index=False),
        "```",
        "",
        "## Exploratory candidates",
        "",
        "```text",
        exp_rec[["generated_submission", "hypothesis_group", "exploratory_score", "weighted_mape_127", "mean_abs_relative_delta_vs_current_best", "max_rel_delta_vs_current_best", "share_changed_stores"]].head(8).to_string(index=False),
        "```",
        "",
        "## Логика отправки",
        "",
        "1. Сначала safe: проверяют новый сигнал без сильного риска.",
        "2. Затем moderate: trajectory/temporal/regime кандидаты, которые реально отличаются от current best.",
        "3. Exploratory отправлять после safe/moderate, если нужен шанс на скачок выше 95.91.",
        "",
        "Записать результат:",
        "",
        "```bash",
        "python scripts/record_leaderboard_result.py --file submissions/<file>.csv --model <model_name> --lb-score <score> --verdict OK --comment \"комментарий\"",
        "```",
    ]
    (REPORTS_DIR / "recommended_submissions.md").write_text("\n".join(rec) + "\n", encoding="utf-8")


def append_readme_note() -> None:
    path = PROJECT_ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    marker = "## Текущий этап: trajectory/temporal"
    section = """

## Текущий этап: trajectory/temporal

После новых LB-результатов текущий подтвержденный уровень поднят до `95.91`. Лучшие направления: `cluster_blend_v1` и `temporal_ridge_ratio_v1`. Поэтому `scripts/run_experiments.py` теперь развивает trajectory clustering v2, temporal models v2, analog forecasting, regime detection, segment-specific temporal и hybrid cluster-temporal candidates.
"""
    if marker not in text:
        path.write_text(text.rstrip() + section + "\n", encoding="utf-8")


def run_stage2() -> None:
    warnings.filterwarnings("ignore")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_new_leaderboard_results()
    ensure_current_best()
    append_readme_note()

    df = load_train(TRAIN_PATH)
    pivot = df.pivot(index=ID_COL, columns=MONTH_COL, values=TARGET_COL).sort_index()
    current_test = current_best_pred(df, pivot, 11)

    comp = best_model_comparison(df, pivot, current_test)
    seg = best_model_segment_report(df, pivot)
    seg.to_csv(REPORTS_DIR / "best_model_comparison_segments.csv", index=False, encoding="utf-8")
    write_best_model_comparison_reports(comp, seg)

    experiments = build_experiments(df, pivot)
    by_name = {e.experiment_name: e for e in experiments}
    results = pd.DataFrame([score_experiment(e, df, pivot, current_test) for e in experiments])
    results = enrich_lb(results)

    selected = choose_candidates(results)
    existing = list(SUBMISSIONS_DIR.glob("test_*.csv"))
    generated_rows = []
    for name in selected:
        exp = by_name[name]
        pred = exp.pred_func(11).reindex(current_test.index)
        path = SUBMISSIONS_DIR / exp.preferred_filename
        if path.exists():
            validate_saved_csv(path)
        else:
            if duplicate_prediction(pred, existing):
                continue
            save_submission(base.submission_frame(pred), path)
            validate_saved_csv(path)
            existing.append(path)
        rel = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        results.loc[results["experiment_name"].eq(name), "generated_submission"] = rel
        row = results.loc[results["experiment_name"].eq(name)].iloc[0].copy()
        row["generated_submission"] = rel
        generated_rows.append(row.to_dict())
        if len(generated_rows) >= 15:
            break

    diagnostic_path = SUBMISSIONS_DIR / "test_current_best_4digits.csv"
    if not diagnostic_path.exists():
        if current_best_4digits_submission(current_test, diagnostic_path):
            row = {
                "experiment_name": "current_best_4digits",
                "model_name": "current_best_4digits",
                "hypothesis_group": "rounding_diagnostic",
                "candidate_class": "diagnostic",
                "fold8_mape": np.nan,
                "fold9_mape": np.nan,
                "fold10_mape": np.nan,
                "mean_mape": np.nan,
                "weighted_mape_127": np.nan,
                "safe_risk_score": np.nan,
                "exploratory_score": np.nan,
                "generated_submission": str(diagnostic_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "comment": "Диагностика округления: current best с 4 знаками после запятой, размер < 1 MB.",
            }
            generated_rows.append(row)

    results = enrich_lb(results)
    results.sort_values("safe_risk_score").to_csv(REPORTS_DIR / "experiment_results.csv", index=False, encoding="utf-8")
    generated = pd.DataFrame(generated_rows)
    write_family_reports(results)
    write_main_reports(results, generated)

    if len(generated):
        registry = pd.read_csv(SUBMISSION_REGISTRY_PATH, encoding="utf-8") if SUBMISSION_REGISTRY_PATH.exists() else pd.DataFrame()
        existing_files = set(registry.get("filename", pd.Series(dtype=str)).astype(str))
        rows = []
        for row in generated_rows:
            filename = row.get("generated_submission", "")
            if filename and filename not in existing_files:
                rows.append(
                    {
                        "submitted_at": "",
                        "filename": filename,
                        "model_name": row.get("model_name", ""),
                        "local_cv_mape": row.get("weighted_mape_127", ""),
                        "lb_score": "",
                        "lb_mape": "",
                        "verdict": "",
                        "comment": row.get("comment", ""),
                    }
                )
        if rows:
            registry = pd.concat([registry, pd.DataFrame(rows)], ignore_index=True)
            registry.to_csv(SUBMISSION_REGISTRY_PATH, index=False, encoding="utf-8")

    restored = restore_best_submission()
    print(f"Сохранено: {REPORTS_DIR / 'experiment_results.csv'}")
    print(f"Сохранено: {REPORTS_DIR / 'best_model_comparison.md'}")
    print(f"Сохранено: {REPORTS_DIR / 'recommended_submissions.md'}")
    print("Созданные/обновленные сабмиты:")
    for row in generated_rows:
        print(f"  {row.get('generated_submission', '')}")
    print(f"Восстановлен current best test.csv: {restored}")


if __name__ == "__main__":
    run_stage2()
