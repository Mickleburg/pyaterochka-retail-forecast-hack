from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .config import ID_COL, MONTH_COL, REPORTS_DIR, SUBMISSIONS_DIR, TARGET_COL, TRAIN_PATH, RANDOM_SEED
from .features import load_train
from .metrics import mape_percent
from .registry import (
    BEST_SUBMISSION_PATH,
    LEADERBOARD_RESULTS_PATH,
    SUBMISSION_REGISTRY_PATH,
    load_best_submission,
    load_leaderboard,
    restore_best_submission,
    save_best_submission,
)
from .submit import save_submission, validate_saved_csv


FOLDS = [8, 9, 10]
WEIGHT_127 = {8: 0.1, 9: 0.2, 10: 0.7}
WEIGHT_235 = {8: 0.2, 9: 0.3, 10: 0.5}

REGION_COL = "Регион"
AREA_COL = "Торговая площадь, категориальный"
SETTLEMENT_COL = "Населенный пункт"
CASH_COL = "Количество касс"
ALCOHOL_COL = "Флаг алкогольной лицензии"
OPEN_COL = "Дата открытия, категориальный"

CURRENT_BEST_FILE = "submissions/test_ratio_shrink_b0p06_c97_103.csv"
CURRENT_BEST_MODEL = "ratio_shrink_b0p06_c97_103"

PREDICTION_MODELS = [
    "baseline",
    "current_best",
    "rolling_mean_2",
    "rolling_mean_3",
    "rolling_median_3",
    "damped_add",
    "damped_ratio",
    "outlier_smoothing",
    "segment_region",
    "segment_area",
    "ml_ratio",
    "analog_knn",
]


@dataclass
class Experiment:
    experiment_name: str
    model_name: str
    hypothesis_group: str
    pred_func: callable
    comment: str
    preferred_filename: str
    category: str
    expected_direction: str = "mixed"
    force_generate: bool = False


def ensure_current_best() -> None:
    best = {
        "filename": CURRENT_BEST_FILE,
        "model_name": CURRENT_BEST_MODEL,
        "lb_score": 95.88,
        "lb_mape": 4.12,
        "verdict": "OK",
        "is_confirmed_by_leaderboard": True,
    }
    current = load_best_submission()
    if float(current.get("lb_score", 0.0)) < best["lb_score"] or current.get("filename") != best["filename"]:
        save_best_submission(best)
    else:
        save_best_submission(current)


def ensure_known_leaderboard_rows() -> None:
    rows = [
        ("submissions/test_ratio_shrink_b0p06_c97_103.csv", "ratio_shrink_b0p06_c97_103", 95.88, "OK", "ratio_shrink beta 0.06, текущий лучший подтвержденный уровень"),
        ("submissions/test_ratio_shrink_b0p07_c98_102.csv", "ratio_shrink_b0p07_c98_102", 95.88, "OK", "ratio_shrink beta 0.07, тот же LB score; менее консервативен"),
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


def clean_leaderboard_comments() -> None:
    replacements = {
        "official_mean_october_baseline": "официальный бейзлайн, проверка формата",
        "catboost_mape": "первый кандидат catboost/fallback",
        "catboost_log": "первый кандидат catboost log/fallback",
        "baseline_last_month": "сильный бейзлайн: РТО предыдущего месяца",
        "ensemble_conservative_v1": "такой же результат, как у baseline/current best на тот момент",
        "ensemble_conservative_v2": "такой же результат, как у baseline/current best на тот момент",
        "last_month_mult_101": "множитель 1.010, хуже текущего лучшего",
        "last_month_mult_102": "множитель 1.020, хуже текущего лучшего",
        "last_month_mult_1015": "CE в Контесте; локальный формат был проверен отдельно",
        "last_month_mult_0995": "множитель 0.995, немного хуже текущего лучшего",
        "last_month_mult_09975": "множитель 0.9975, на уровне baseline",
        "last_month_mult_10025": "множитель 1.0025, немного хуже текущего лучшего",
        "ratio_shrink_b0p05_c97_103": "ratio_shrink beta 0.05, подтвержденное улучшение до 95.87",
        "residual_centered_v1": "centered residual хуже текущего лучшего",
        "segment_alcohol_s0p05_k500_c98_102": "сегментная микропоправка alcohol, на уровне baseline",
        "segment_alcohol_s0p05_k500_c97_103": "сегментная микропоправка alcohol, на уровне baseline",
        "segment_area_shrink_v1": "сегментная микропоправка area, на уровне baseline",
        "segment_blend_shrink_v1": "сегментная микропоправка blend, на уровне baseline",
        "segment_region_shrink_v1": "сегментная микропоправка region, на уровне baseline",
        "ratio_shrink_b0p06_c97_103": "ratio_shrink beta 0.06, текущий лучший подтвержденный уровень",
        "ratio_shrink_b0p07_c98_102": "ratio_shrink beta 0.07, тот же LB score; менее консервативен",
    }
    lb = load_leaderboard()
    for idx, row in lb.iterrows():
        model = str(row["model_name"])
        if model in replacements:
            lb.loc[idx, "comment"] = replacements[model]
    lb.to_csv(LEADERBOARD_RESULTS_PATH, index=False, encoding="utf-8")


def as_series(values, index) -> pd.Series:
    return pd.Series(values, index=index).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def clip_series(values, index=None) -> pd.Series:
    out = pd.Series(values, index=index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.clip(lower=0.0)


def submission_frame(pred: pd.Series) -> pd.DataFrame:
    return (
        pd.DataFrame({ID_COL: pred.index, "rto": np.round(pred.astype(float).to_numpy(), 2)})
        .sort_values(ID_COL)
        .reset_index(drop=True)
    )


def baseline_pred(pivot: pd.DataFrame, month: int) -> pd.Series:
    return pivot[month - 1].copy()


def safe_ratio(num: pd.Series, den: pd.Series, fill: float = 1.0) -> pd.Series:
    return (num / den).replace([np.inf, -np.inf], np.nan).fillna(fill)


def static_at(df: pd.DataFrame, month: int) -> pd.DataFrame:
    return df[df[MONTH_COL] == month][[ID_COL, REGION_COL, AREA_COL, SETTLEMENT_COL, CASH_COL, ALCOHOL_COL, OPEN_COL]].set_index(ID_COL)


def rolling_mean_pred(pivot: pd.DataFrame, month: int, window: int) -> pd.Series:
    return clip_series(pivot[list(range(month - window, month))].mean(axis=1), index=pivot.index)


def rolling_median_pred(pivot: pd.DataFrame, month: int, window: int) -> pd.Series:
    return clip_series(pivot[list(range(month - window, month))].median(axis=1), index=pivot.index)


def damped_add_pred(pivot: pd.DataFrame, month: int, alpha: float = 0.12) -> pd.Series:
    base = baseline_pred(pivot, month)
    return clip_series(base + alpha * (pivot[month - 1] - pivot[month - 2]), index=base.index)


def damped_ratio_pred(pivot: pd.DataFrame, month: int, alpha: float = 0.10, clip_bounds=(0.97, 1.03)) -> pd.Series:
    base = baseline_pred(pivot, month)
    recent = safe_ratio(pivot[month - 1], pivot[month - 2], 1.0)
    ratio = (1.0 + alpha * (recent - 1.0)).clip(*clip_bounds)
    return clip_series(base * ratio, index=base.index)


def outlier_smooth_pred(pivot: pd.DataFrame, month: int, mode: str = "symmetric", threshold: float = 0.12, blend: float = 0.90) -> pd.Series:
    base = baseline_pred(pivot, month)
    ref = pivot[[month - 4, month - 3, month - 2]].median(axis=1)
    deviation = safe_ratio(base, ref, 1.0) - 1.0
    if mode == "high":
        mask = deviation > threshold
    elif mode == "low":
        mask = deviation < -threshold
    else:
        mask = deviation.abs() > threshold
    pred = base.copy()
    pred.loc[mask] = blend * base.loc[mask] + (1.0 - blend) * ref.loc[mask]
    return clip_series(pred, index=base.index)


def dynamic_features(df: pd.DataFrame, pivot: pd.DataFrame, month: int) -> pd.DataFrame:
    out = static_at(df, month - 1).reindex(pivot.index)
    out["rto_lag_1"] = pivot[month - 1]
    out["rto_lag_2"] = pivot[month - 2]
    out["rto_lag_3"] = pivot[month - 3]
    out["rto_lag_4"] = pivot[month - 4]
    out["rto_mean_3"] = pivot[[month - 3, month - 2, month - 1]].mean(axis=1)
    out["store_mean"] = pivot[list(range(1, month))].mean(axis=1)
    out["ratio_10_9"] = safe_ratio(pivot[month - 1], pivot[month - 2], 1.0)
    out["ratio_9_8"] = safe_ratio(pivot[month - 2], pivot[month - 3], 1.0)
    out["ratio_8_7"] = safe_ratio(pivot[month - 3], pivot[month - 4], 1.0)
    ratio_cols = ["ratio_8_7", "ratio_9_8", "ratio_10_9"]
    out["volatility"] = out[ratio_cols].std(axis=1)
    out["trend"] = out["ratio_10_9"] - 1.0
    ref = pivot[[month - 4, month - 3, month - 2]].median(axis=1)
    out["outlier_score"] = (safe_ratio(pivot[month - 1], ref, 1.0) - 1.0).abs()
    out["signed_outlier"] = safe_ratio(pivot[month - 1], ref, 1.0) - 1.0
    out["rto_decile"] = pd.qcut(out["rto_lag_1"], 10, labels=False, duplicates="drop").astype("float").fillna(-1).astype(int)
    out["mean_rto_decile"] = pd.qcut(out["store_mean"], 10, labels=False, duplicates="drop").astype("float").fillna(-1).astype(int)
    out["volatility_bucket"] = pd.qcut(out["volatility"], 5, labels=False, duplicates="drop").astype("float").fillna(-1).astype(int)
    out["trend_bucket"] = pd.cut(out["trend"], [-np.inf, -0.05, -0.015, 0.015, 0.05, np.inf], labels=False).astype("float").fillna(-1).astype(int)
    out["outlier_bucket"] = pd.cut(out["outlier_score"], [-np.inf, 0.05, 0.10, 0.15, np.inf], labels=False).astype("float").fillna(-1).astype(int)
    for col in ["ratio_10_9", "ratio_9_8", "ratio_8_7"]:
        out[f"{col}_bucket"] = pd.cut(out[col] - 1.0, [-np.inf, -0.05, -0.015, 0.015, 0.05, np.inf], labels=False).astype("float").fillna(-1).astype(int)
    out["region_x_rto_decile"] = out[REGION_COL].astype(str) + "__" + out["rto_decile"].astype(str)
    out["area_x_volatility"] = out[AREA_COL].astype(str) + "__" + out["volatility_bucket"].astype(str)
    out["alcohol_x_rto_decile"] = out[ALCOHOL_COL].astype(str) + "__" + out["rto_decile"].astype(str)
    out["region_x_trend"] = out[REGION_COL].astype(str) + "__" + out["trend_bucket"].astype(str)
    out["area_x_trend"] = out[AREA_COL].astype(str) + "__" + out["trend_bucket"].astype(str)
    return out


def segment_multiplier(
    df: pd.DataFrame,
    pivot: pd.DataFrame,
    month: int,
    group_cols: list[str],
    shrink_weight: float,
    k: int,
    clip_bounds: tuple[float, float],
) -> pd.Series:
    rows = []
    for t in range(2, month):
        feats = static_at(df, t - 1).reindex(pivot.index)
        growth = safe_ratio(pivot[t], pivot[t - 1], 1.0)
        part = feats[group_cols].copy()
        part["growth"] = growth
        rows.append(part)
    hist = pd.concat(rows, ignore_index=True)
    hist = hist[hist["growth"].between(0.65, 1.45)]
    global_growth = float(hist["growth"].median())
    stats = hist.groupby(group_cols, dropna=False)["growth"].agg(["median", "count"]).reset_index()
    stats["w"] = stats["count"] / (stats["count"] + k)
    stats["mult"] = 1.0 + shrink_weight * stats["w"] * (stats["median"] - global_growth)
    stats["mult"] = stats["mult"].clip(*clip_bounds)
    current = static_at(df, month - 1).reindex(pivot.index)
    merged = current[group_cols].reset_index().merge(stats[group_cols + ["mult"]], on=group_cols, how="left").set_index(ID_COL)
    return merged["mult"].reindex(pivot.index).fillna(1.0)


def segment_pred(
    df: pd.DataFrame,
    pivot: pd.DataFrame,
    month: int,
    group_cols: list[str],
    shrink_weight: float = 0.05,
    k: int = 500,
    clip_bounds: tuple[float, float] = (0.97, 1.03),
) -> pd.Series:
    base = baseline_pred(pivot, month)
    return clip_series(base * segment_multiplier(df, pivot, month, group_cols, shrink_weight, k, clip_bounds), index=base.index)


def raw_ratio_signal(df: pd.DataFrame, pivot: pd.DataFrame, month: int) -> pd.Series:
    region_area = segment_multiplier(df, pivot, month, [REGION_COL, AREA_COL], 0.30, 300, (0.95, 1.05))
    recent = safe_ratio(pivot[month - 1], pivot[month - 2], 1.0).clip(0.85, 1.15)
    return 0.85 * region_area + 0.15 * recent


def ratio_shrink_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int, beta: float, clip_bounds: tuple[float, float]) -> pd.Series:
    base = baseline_pred(pivot, month)
    ratio = (1.0 + beta * (raw_ratio_signal(df, pivot, month) - 1.0)).clip(*clip_bounds)
    return clip_series(base * ratio, index=base.index)


def current_best_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int) -> pd.Series:
    return ratio_shrink_pred(df, pivot, month, 0.06, (0.97, 1.03))


def ml_ratio_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int, beta: float = 0.08, clip_bounds=(0.97, 1.03)) -> pd.Series:
    rows = []
    for t in range(5, month):
        feats = dynamic_features(df, pivot, t)
        feats["target_ratio"] = safe_ratio(pivot[t], pivot[t - 1], 1.0)
        rows.append(feats)
    hist = pd.concat(rows)
    hist = hist[hist["target_ratio"].between(0.65, 1.45)]
    global_ratio = float(hist["target_ratio"].median())
    current = dynamic_features(df, pivot, month)
    pieces = []
    for cols, weight, k in [
        ([REGION_COL], 0.20, 300),
        ([AREA_COL], 0.18, 300),
        (["rto_decile"], 0.18, 200),
        (["volatility_bucket"], 0.14, 200),
        ([REGION_COL, "rto_decile"], 0.15, 500),
        ([AREA_COL, "volatility_bucket"], 0.15, 500),
    ]:
        stats = hist.groupby(cols, dropna=False)["target_ratio"].agg(["median", "count"]).reset_index()
        stats["w"] = stats["count"] / (stats["count"] + k)
        stats["ratio"] = stats["w"] * stats["median"] + (1.0 - stats["w"]) * global_ratio
        merged = current[cols].reset_index().merge(stats[cols + ["ratio"]], on=cols, how="left").set_index(ID_COL)
        pieces.append(weight * merged["ratio"].reindex(pivot.index).fillna(global_ratio))
    raw = sum(pieces)
    raw = raw / raw.median()
    final_ratio = (1.0 + beta * (raw - 1.0)).clip(*clip_bounds)
    return clip_series(baseline_pred(pivot, month) * final_ratio, index=pivot.index)


def analog_knn_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int, beta: float = 0.12, clip_bounds=(0.97, 1.03)) -> pd.Series:
    rows = []
    for t in range(5, month):
        feats = dynamic_features(df, pivot, t)
        feats["next_ratio"] = safe_ratio(pivot[t], pivot[t - 1], 1.0)
        rows.append(feats[[REGION_COL, AREA_COL, "rto_decile", "volatility_bucket", "trend_bucket", "next_ratio"]])
    hist = pd.concat(rows)
    hist = hist[hist["next_ratio"].between(0.65, 1.45)]
    global_ratio = float(hist["next_ratio"].median())
    stats = hist.groupby([REGION_COL, AREA_COL, "rto_decile", "volatility_bucket", "trend_bucket"], dropna=False)["next_ratio"].agg(["median", "count"]).reset_index()
    stats["w"] = stats["count"] / (stats["count"] + 120)
    stats["ratio"] = stats["w"] * stats["median"] + (1.0 - stats["w"]) * global_ratio
    cur = dynamic_features(df, pivot, month)
    cols = [REGION_COL, AREA_COL, "rto_decile", "volatility_bucket", "trend_bucket"]
    merged = cur[cols].reset_index().merge(stats[cols + ["ratio"]], on=cols, how="left").set_index(ID_COL)
    raw = merged["ratio"].reindex(pivot.index).fillna(global_ratio)
    ratio = (1.0 + beta * (raw - 1.0)).clip(*clip_bounds)
    return clip_series(baseline_pred(pivot, month) * ratio, index=pivot.index)


def candidate_predictions(df: pd.DataFrame, pivot: pd.DataFrame, month: int) -> dict[str, pd.Series]:
    return {
        "baseline": baseline_pred(pivot, month),
        "current_best": current_best_pred(df, pivot, month),
        "rolling_mean_2": rolling_mean_pred(pivot, month, 2),
        "rolling_mean_3": rolling_mean_pred(pivot, month, 3),
        "rolling_median_3": rolling_median_pred(pivot, month, 3),
        "damped_add": damped_add_pred(pivot, month, 0.12),
        "damped_ratio": damped_ratio_pred(pivot, month, 0.10, (0.97, 1.03)),
        "outlier_smoothing": outlier_smooth_pred(pivot, month, "symmetric", 0.12, 0.90),
        "segment_region": segment_pred(df, pivot, month, [REGION_COL], 0.08, 500, (0.98, 1.02)),
        "segment_area": segment_pred(df, pivot, month, [AREA_COL], 0.08, 500, (0.98, 1.02)),
        "ml_ratio": ml_ratio_pred(df, pivot, month, 0.08, (0.97, 1.03)),
        "analog_knn": analog_knn_pred(df, pivot, month, 0.12, (0.97, 1.03)),
    }


def ape(y_true: pd.Series, pred: pd.Series) -> pd.Series:
    return ((pred - y_true).abs() / y_true.abs().clip(lower=1e-8)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_oof_predictions(df: pd.DataFrame, pivot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for month in FOLDS:
        feats = dynamic_features(df, pivot, month)
        preds = candidate_predictions(df, pivot, month)
        fold = pd.DataFrame({ID_COL: pivot.index, "valid_month": month, "true_rto": pivot[month], "rto_lag_1": pivot[month - 1], "rto_lag_2": pivot[month - 2], "rto_lag_3": pivot[month - 3], "rto_lag_4": pivot[month - 4]}).set_index(ID_COL)
        for name, pred in preds.items():
            fold[f"{name}_pred"] = pred.reindex(fold.index)
            fold[f"{name}_ape"] = ape(fold["true_rto"], fold[f"{name}_pred"])
        fold["signed_error_ratio"] = fold["current_best_pred"] / fold["true_rto"] - 1.0
        keep = [
            REGION_COL,
            AREA_COL,
            SETTLEMENT_COL,
            CASH_COL,
            ALCOHOL_COL,
            OPEN_COL,
            "rto_decile",
            "mean_rto_decile",
            "volatility_bucket",
            "trend_bucket",
            "outlier_bucket",
            "ratio_10_9_bucket",
            "ratio_9_8_bucket",
            "ratio_8_7_bucket",
            "region_x_rto_decile",
            "area_x_volatility",
            "alcohol_x_rto_decile",
            "region_x_trend",
            "area_x_trend",
            "volatility",
            "trend",
            "signed_outlier",
        ]
        fold = fold.join(feats[keep])
        rows.append(fold.reset_index())
    return pd.concat(rows, ignore_index=True)


def build_segment_error_mining(oof: pd.DataFrame) -> pd.DataFrame:
    segment_cols = [
        REGION_COL,
        SETTLEMENT_COL,
        AREA_COL,
        CASH_COL,
        ALCOHOL_COL,
        OPEN_COL,
        "rto_decile",
        "mean_rto_decile",
        "volatility_bucket",
        "trend_bucket",
        "outlier_bucket",
        "ratio_10_9_bucket",
        "ratio_9_8_bucket",
        "ratio_8_7_bucket",
        "region_x_rto_decile",
        "area_x_volatility",
        "alcohol_x_rto_decile",
        "region_x_trend",
        "area_x_trend",
    ]
    rows = []
    for seg in segment_cols:
        min_n = 300 if "_x_" in seg else 100
        grouped = oof.groupby([seg, "valid_month"], dropna=False)
        per_fold = []
        for (value, fold), part in grouped:
            if len(part) < max(30, min_n // 3):
                continue
            item = {"segment_type": seg, "segment_value": str(value), "valid_month": fold, "n": len(part)}
            for model in PREDICTION_MODELS:
                item[f"{model}_mape"] = float(part[f"{model}_ape"].mean() * 100.0)
            item["current_best_bias"] = float((part["current_best_pred"] / part["true_rto"] - 1.0).mean())
            item["median_pred_div_true"] = float((part["current_best_pred"] / part["true_rto"]).median())
            per_fold.append(item)
        if not per_fold:
            continue
        pf = pd.DataFrame(per_fold)
        for value, part in pf.groupby("segment_value", dropna=False):
            if part["n"].sum() < min_n:
                continue
            agg = {"segment_type": seg, "segment_value": str(value), "n": int(part["n"].sum()), "fold_count": int(part["valid_month"].nunique())}
            for model in PREDICTION_MODELS:
                agg[f"{model}_mape"] = float(np.average(part[f"{model}_mape"], weights=part["n"]))
            mapes = {model: agg[f"{model}_mape"] for model in PREDICTION_MODELS}
            best_model = min(mapes, key=mapes.get)
            agg["best_model_in_segment"] = best_model
            agg["best_model_mape"] = mapes[best_model]
            agg["improvement_best_vs_current"] = agg["current_best_mape"] - agg["best_model_mape"]
            agg["signed_bias_current_best"] = float(np.average(part["current_best_bias"], weights=part["n"]))
            agg["median_pred_div_true"] = float(np.average(part["median_pred_div_true"], weights=part["n"]))
            signs = np.sign(part["current_best_bias"])
            agg["same_bias_sign_folds"] = int(max((signs > 0).sum(), (signs < 0).sum()))
            agg["stability_score"] = float(agg["improvement_best_vs_current"] * (agg["same_bias_sign_folds"] / max(1, agg["fold_count"])))
            rows.append(agg)
    return pd.DataFrame(rows).sort_values(["stability_score", "improvement_best_vs_current"], ascending=False)


def history_bias_frame(df: pd.DataFrame, pivot: pd.DataFrame, month: int) -> pd.DataFrame:
    rows = []
    start = 5
    for t in range(start, month):
        feats = dynamic_features(df, pivot, t)
        pred = current_best_pred(df, pivot, t)
        part = feats.copy()
        part["bias_ratio"] = safe_ratio(pivot[t], pred, 1.0)
        part["target"] = pivot[t]
        part["pred"] = pred
        rows.append(part)
    hist = pd.concat(rows)
    return hist[hist["bias_ratio"].between(0.70, 1.30)]


def segment_calibrated_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int, segment_cols: list[str], shrink: float, k: int, clip_bounds: tuple[float, float]) -> pd.Series:
    base = current_best_pred(df, pivot, month)
    hist = history_bias_frame(df, pivot, month)
    current = dynamic_features(df, pivot, month)
    stats = hist.groupby(segment_cols, dropna=False)["bias_ratio"].agg(["median", "count"]).reset_index()
    stats["w"] = stats["count"] / (stats["count"] + k)
    stats["multiplier"] = 1.0 + shrink * stats["w"] * (stats["median"] - 1.0)
    stats["multiplier"] = stats["multiplier"].clip(*clip_bounds)
    merged = current[segment_cols].reset_index().merge(stats[segment_cols + ["multiplier"]], on=segment_cols, how="left").set_index(ID_COL)
    mult = merged["multiplier"].reindex(pivot.index).fillna(1.0)
    return clip_series(base * mult, index=pivot.index)


def selector_by_segment_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int, segment_cols: list[str], weight: float, min_gain: float = 0.03) -> pd.Series:
    base = current_best_pred(df, pivot, month)
    rows = []
    for t in range(6, month):
        feats = dynamic_features(df, pivot, t)
        preds = candidate_predictions(df, pivot, t)
        part = feats[segment_cols].copy()
        for model, pred in preds.items():
            part[f"{model}_ape"] = ape(pivot[t], pred)
        rows.append(part)
    hist = pd.concat(rows)
    current = dynamic_features(df, pivot, month)
    current_preds = candidate_predictions(df, pivot, month)
    stats_rows = []
    for key, part in hist.groupby(segment_cols, dropna=False):
        if len(part) < 120:
            continue
        mapes = {model: float(part[f"{model}_ape"].mean() * 100.0) for model in PREDICTION_MODELS}
        best_model = min(mapes, key=mapes.get)
        gain = mapes["current_best"] - mapes[best_model]
        if gain < min_gain:
            best_model = "current_best"
        key_tuple = key if isinstance(key, tuple) else (key,)
        stats_rows.append(dict(zip(segment_cols, key_tuple), best_model=best_model, gain=gain, n=len(part)))
    if not stats_rows:
        return base
    stats = pd.DataFrame(stats_rows)
    merged = current[segment_cols].reset_index().merge(stats[segment_cols + ["best_model", "gain"]], on=segment_cols, how="left").set_index(ID_COL)
    selected = base.copy()
    for model in PREDICTION_MODELS:
        mask = merged["best_model"].eq(model).reindex(pivot.index).fillna(False)
        if mask.any():
            selected.loc[mask] = current_preds[model].loc[mask]
    return clip_series(weight * selected + (1.0 - weight) * base, index=pivot.index)


def optimize_weights(y: pd.Series, preds: pd.DataFrame, min_current_weight: float = 0.5, trials: int = 300) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_SEED)
    cols = list(preds.columns)
    cur_idx = cols.index("current_best")
    best_w = np.zeros(len(cols))
    best_w[cur_idx] = 1.0
    best_score = mape_percent(y, preds @ best_w)
    for _ in range(trials):
        w = rng.dirichlet(np.ones(len(cols)))
        w[cur_idx] = max(w[cur_idx], min_current_weight)
        w = w / w.sum()
        score = mape_percent(y, preds @ w) + 0.02 * float(np.sum((w - best_w) ** 2))
        if score < best_score:
            best_score = score
            best_w = w
    return best_w


def weighted_oof_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int, bucket_col: str | None, min_current_weight: float) -> tuple[pd.Series, pd.DataFrame]:
    models = ["baseline", "current_best", "rolling_mean_3", "outlier_smoothing", "damped_ratio", "segment_area", "ml_ratio", "analog_knn"]
    hist_rows = []
    for t in range(7, month):
        feats = dynamic_features(df, pivot, t)
        preds = candidate_predictions(df, pivot, t)
        part = pd.DataFrame({ID_COL: pivot.index, "target": pivot[t]}).set_index(ID_COL)
        for model in models:
            part[model] = preds[model]
        if bucket_col:
            part[bucket_col] = feats[bucket_col]
        hist_rows.append(part)
    hist = pd.concat(hist_rows)
    test_preds_all = candidate_predictions(df, pivot, month)
    test_preds = pd.DataFrame({model: test_preds_all[model] for model in models})
    best = test_preds_all["current_best"].copy()
    weight_rows = []
    if not bucket_col:
        w = optimize_weights(hist["target"], hist[models], min_current_weight)
        pred = test_preds @ w
        weight_rows.append({"bucket_type": "global", "bucket_value": "all", **dict(zip(models, w))})
        return clip_series(pred, index=pivot.index), pd.DataFrame(weight_rows)
    current = dynamic_features(df, pivot, month)
    pred = best.copy()
    for bucket_value, train_part in hist.groupby(bucket_col, dropna=False):
        if len(train_part) < 300:
            continue
        w = optimize_weights(train_part["target"], train_part[models], min_current_weight, trials=180)
        mask = current[bucket_col].eq(bucket_value)
        if mask.any():
            pred.loc[mask] = (test_preds.loc[mask, models] @ w).to_numpy()
        weight_rows.append({"bucket_type": bucket_col, "bucket_value": str(bucket_value), "n": len(train_part), **dict(zip(models, w))})
    return clip_series(pred, index=pivot.index), pd.DataFrame(weight_rows)


def temporal_features(pivot: pd.DataFrame, month: int) -> pd.DataFrame:
    rows = {}
    base = pivot[month - 1]
    for lag in range(1, 7):
        rows[f"norm_lag_{lag}"] = safe_ratio(pivot[month - lag], base, 1.0)
    for lag in range(1, 6):
        rows[f"ratio_lag_{lag}"] = safe_ratio(pivot[month - lag], pivot[month - lag - 1], 1.0).clip(0.5, 1.8)
    vals3 = np.log1p(pivot[[month - 3, month - 2, month - 1]])
    x3 = np.array([-2.0, -1.0, 0.0])
    rows["slope3"] = ((vals3 - vals3.mean(axis=1).to_numpy()[:, None]) @ x3) / float((x3**2).sum())
    vals6 = np.log1p(pivot[list(range(month - 6, month))])
    x6 = np.arange(6, dtype=float) - 5
    rows["slope6"] = ((vals6 - vals6.mean(axis=1).to_numpy()[:, None]) @ x6) / float((x6**2).sum())
    rows["volatility"] = pd.DataFrame({k: v for k, v in rows.items() if k.startswith("ratio_lag_")}).std(axis=1)
    rows["last_to_mean3"] = safe_ratio(base, pivot[[month - 4, month - 3, month - 2]].mean(axis=1), 1.0)
    frame = pd.DataFrame(rows, index=pivot.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame


def temporal_model_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int, model_kind: str, beta: float, clip_bounds: tuple[float, float]) -> pd.Series:
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
        from sklearn.linear_model import HuberRegressor, Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return current_best_pred(df, pivot, month)

    xs, ys = [], []
    for t in range(7, month):
        x = temporal_features(pivot, t)
        y = safe_ratio(pivot[t], pivot[t - 1], 1.0).clip(0.70, 1.30)
        xs.append(x)
        ys.append(y)
    x_train = pd.concat(xs)
    y_train = pd.concat(ys)
    if model_kind == "hgb":
        model = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.04, l2_regularization=2.0, random_state=RANDOM_SEED)
    elif model_kind == "extra":
        model = ExtraTreesRegressor(n_estimators=120, min_samples_leaf=80, random_state=RANDOM_SEED, n_jobs=-1)
    else:
        model = make_pipeline(StandardScaler(), HuberRegressor(alpha=0.02, epsilon=1.2, max_iter=300))
    model.fit(x_train, y_train)
    raw = pd.Series(model.predict(temporal_features(pivot, month)), index=pivot.index)
    raw = raw / raw.median()
    ratio = (1.0 + beta * (raw - 1.0)).clip(*clip_bounds)
    return clip_series(baseline_pred(pivot, month) * ratio, index=pivot.index)


def cluster_ratio_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int, n_clusters: int, beta: float, clip_bounds: tuple[float, float]) -> pd.Series:
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
    except Exception:
        return current_best_pred(df, pivot, month)
    hist_parts = []
    for t in range(7, month):
        x = temporal_features(pivot, t)
        y = safe_ratio(pivot[t], pivot[t - 1], 1.0).clip(0.70, 1.30)
        hist_parts.append((x, y))
    x_hist = pd.concat([p[0] for p in hist_parts])
    y_hist = pd.concat([p[1] for p in hist_parts])
    model = make_pipeline(StandardScaler(), KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10))
    labels = model.fit_predict(x_hist)
    stats = pd.DataFrame({"label": labels, "ratio": y_hist.to_numpy()}).groupby("label")["ratio"].agg(["median", "count"])
    global_ratio = float(y_hist.median())
    current_labels = model.predict(temporal_features(pivot, month))
    raw = pd.Series(current_labels, index=pivot.index).map(stats["median"]).fillna(global_ratio)
    raw = raw / raw.median()
    ratio = (1.0 + beta * (raw - 1.0)).clip(*clip_bounds)
    return clip_series(baseline_pred(pivot, month) * ratio, index=pivot.index)


def low_rto_decile_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int) -> pd.Series:
    best = current_best_pred(df, pivot, month)
    median3 = rolling_median_pred(pivot, month, 3)
    feats = dynamic_features(df, pivot, month)
    pred = best.copy()
    low = feats["rto_decile"] <= 1
    mid = feats["rto_decile"].between(2, 4)
    pred.loc[low] = 0.75 * best.loc[low] + 0.25 * median3.loc[low]
    pred.loc[mid] = 0.90 * best.loc[mid] + 0.10 * median3.loc[mid]
    return clip_series(pred, index=pivot.index)


def decile_beta_ratio_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int) -> pd.Series:
    base = baseline_pred(pivot, month)
    raw = raw_ratio_signal(df, pivot, month)
    feats = dynamic_features(df, pivot, month)
    beta = pd.Series(0.06, index=pivot.index)
    beta.loc[feats["rto_decile"] <= 1] = 0.03
    beta.loc[feats["rto_decile"] >= 8] = 0.08
    ratio = (1.0 + beta * (raw - 1.0)).clip(0.97, 1.03)
    return clip_series(base * ratio, index=pivot.index)


def build_experiments(df: pd.DataFrame, pivot: pd.DataFrame) -> list[Experiment]:
    exps: list[Experiment] = []
    add = exps.append
    add(Experiment("baseline_last_month", "baseline_last_month", "sent", lambda m: baseline_pred(pivot, m), "РТО предыдущего месяца.", "", "reference"))
    add(Experiment("ratio_shrink_b0p06_c97_103", "ratio_shrink", "sent", lambda m: current_best_pred(df, pivot, m), "Текущий лучший подтвержденный ratio_shrink.", "", "reference"))

    segment_specs = [
        ("rto_decile", ["rto_decile"], "test_segment_calibrated_rto_decile_v1.csv"),
        ("volatility", ["volatility_bucket"], "test_segment_calibrated_volatility_v1.csv"),
        ("trend", ["trend_bucket"], "test_segment_calibrated_trend_v1.csv"),
        ("region_rto", ["region_x_rto_decile"], "test_segment_calibrated_region_rto_v1.csv"),
        ("area_vol", ["area_x_volatility"], "test_segment_calibrated_area_vol_v1.csv"),
    ]
    for name, cols, filename in segment_specs:
        add(Experiment(f"segment_calibrated_{name}_s0p20_c98_102", "segment_calibrated", "segment_calibration", lambda m, cols=cols: segment_calibrated_pred(df, pivot, m, cols, 0.20, 250, (0.98, 1.02)), f"Сегментная калибровка bias по {name}, shrink=0.20, clip=(0.98,1.02).", filename, "safe"))
        add(Experiment(f"exploratory_segment_{name}_s0p30_c98_102", "segment_calibrated", "segment_calibration", lambda m, cols=cols: segment_calibrated_pred(df, pivot, m, cols, 0.30, 180, (0.98, 1.02)), f"Более заметная segment calibration по {name}, shrink=0.30.", "", "exploratory"))

    selector_specs = [
        ("rto_decile", ["rto_decile"], "test_selector_by_rto_decile_v1.csv"),
        ("volatility", ["volatility_bucket"], "test_selector_by_volatility_v1.csv"),
        ("outlier", ["outlier_bucket"], "test_selector_by_outlier_v1.csv"),
        ("region_rto", ["region_x_rto_decile"], "test_selector_by_region_rto_v1.csv"),
    ]
    for name, cols, filename in selector_specs:
        add(Experiment(f"selector_by_{name}_w0p25", "segment_selector", "selector", lambda m, cols=cols: selector_by_segment_pred(df, pivot, m, cols, 0.25, 0.02), f"Безопасный selector по сегменту {name}: альтернативная модель получает вес 25%, остальное current best.", filename, "safe"))
        add(Experiment(f"selector_by_{name}_w0p50", "segment_selector", "selector", lambda m, cols=cols: selector_by_segment_pred(df, pivot, m, cols, 0.50, 0.02), f"Selector выбирает лучшую базовую модель по сегменту {name} и смешивает 50/50 с current best.", filename, "moderate"))
        add(Experiment(f"exploratory_selector_{name}_w0p75", "segment_selector", "selector", lambda m, cols=cols: selector_by_segment_pred(df, pivot, m, cols, 0.75, 0.04), f"Более смелый selector по сегменту {name}, safety blend 75/25.", "", "exploratory"))

    add(Experiment("weighted_oof_global_v1", "weighted_mixture", "weighted_mixture", lambda m: weighted_oof_pred(df, pivot, m, None, 0.60)[0], "OOF-оптимизация глобальных весов с ограничением current_best >= 0.60.", "test_weighted_oof_global_v1.csv", "safe"))
    add(Experiment("weighted_oof_rto_decile_v1", "weighted_mixture", "weighted_mixture", lambda m: weighted_oof_pred(df, pivot, m, "rto_decile", 0.50)[0], "OOF-веса по decile РТО, current_best >= 0.50.", "test_weighted_oof_rto_decile_v1.csv", "moderate"))
    add(Experiment("weighted_oof_volatility_v1", "weighted_mixture", "weighted_mixture", lambda m: weighted_oof_pred(df, pivot, m, "volatility_bucket", 0.50)[0], "OOF-веса по bucket волатильности.", "test_weighted_oof_volatility_v1.csv", "moderate"))
    add(Experiment("weighted_oof_trend_v1", "weighted_mixture", "weighted_mixture", lambda m: weighted_oof_pred(df, pivot, m, "trend_bucket", 0.50)[0], "OOF-веса по bucket тренда.", "test_weighted_oof_trend_v1.csv", "moderate"))

    add(Experiment("temporal_ridge_ratio_v1", "temporal_pattern", "temporal_model", lambda m: temporal_model_pred(df, pivot, m, "ridge", 0.18, (0.97, 1.03)), "Temporal pattern model: robust linear model по нормализованной траектории.", "test_temporal_ridge_ratio_v1.csv", "moderate"))
    add(Experiment("temporal_hgb_ratio_v1", "temporal_pattern", "temporal_model", lambda m: temporal_model_pred(df, pivot, m, "hgb", 0.12, (0.97, 1.03)), "Temporal pattern model: HistGradientBoosting по форме траектории.", "test_temporal_hgb_ratio_v1.csv", "exploratory"))
    add(Experiment("temporal_ensemble_ratio_v1", "temporal_pattern", "temporal_model", lambda m: 0.50 * temporal_model_pred(df, pivot, m, "ridge", 0.18, (0.97, 1.03)) + 0.50 * temporal_model_pred(df, pivot, m, "hgb", 0.12, (0.97, 1.03)), "Ансамбль temporal ridge и HGB.", "test_temporal_ensemble_ratio_v1.csv", "exploratory"))

    add(Experiment("cluster_ratio_k10_v1", "trajectory_cluster", "clustering", lambda m: cluster_ratio_pred(df, pivot, m, 10, 0.18, (0.97, 1.03)), "Кластеры магазинов по временной траектории, k=10.", "test_cluster_ratio_k10_v1.csv", "moderate"))
    add(Experiment("cluster_ratio_k20_v1", "trajectory_cluster", "clustering", lambda m: cluster_ratio_pred(df, pivot, m, 20, 0.18, (0.97, 1.03)), "Кластеры магазинов по временной траектории, k=20.", "test_cluster_ratio_k20_v1.csv", "exploratory"))
    add(Experiment("cluster_blend_v1", "trajectory_cluster", "clustering", lambda m: 0.70 * current_best_pred(df, pivot, m) + 0.30 * cluster_ratio_pred(df, pivot, m, 20, 0.18, (0.97, 1.03)), "Blend current best с кластерным прогнозом.", "test_cluster_blend_v1.csv", "moderate"))

    add(Experiment("october_high_rollback_safe_v1", "october_outlier", "outlier_specific", lambda m: outlier_smooth_pred(pivot, m, "high", 0.12, 0.95), "Безопасный High October rollback: высокий октябрь слегка откатывается к rolling median.", "test_october_high_rollback_v1.csv", "moderate", "down"))
    add(Experiment("october_low_recovery_safe_v1", "october_outlier", "outlier_specific", lambda m: outlier_smooth_pred(pivot, m, "low", 0.12, 0.95), "Безопасный Low October recovery: низкий октябрь слегка возвращается к rolling median.", "test_october_low_recovery_v1.csv", "moderate", "up"))
    add(Experiment("october_asymmetric_safe_v1", "october_outlier", "outlier_specific", lambda m: 0.5 * outlier_smooth_pred(pivot, m, "high", 0.12, 0.93) + 0.5 * outlier_smooth_pred(pivot, m, "low", 0.16, 0.97), "Безопасная асимметричная обработка октябрьских выбросов.", "test_october_asymmetric_v1.csv", "exploratory", "mixed"))
    add(Experiment("october_high_rollback_raw", "october_outlier", "outlier_specific", lambda m: outlier_smooth_pred(pivot, m, "high", 0.10, 0.80), "Исходный High October rollback оказался слишком резким; оставлен только для оценки риска.", "", "exploratory", "down"))
    add(Experiment("october_low_recovery_raw", "october_outlier", "outlier_specific", lambda m: outlier_smooth_pred(pivot, m, "low", 0.10, 0.80), "Исходный Low October recovery оказался слишком резким; оставлен только для оценки риска.", "", "exploratory", "up"))

    add(Experiment("mape_low_rto_decile_v1", "mape_aware", "low_rto", lambda m: low_rto_decile_pred(df, pivot, m), "MAPE-aware поправка: нижние decile сильнее тянутся к rolling median.", "test_mape_low_rto_decile_v1.csv", "exploratory"))
    add(Experiment("decile_beta_ratio_v1", "mape_aware", "low_rto", lambda m: decile_beta_ratio_pred(df, pivot, m), "Разная beta ratio_shrink по decile РТО.", "test_decile_beta_ratio_v1.csv", "moderate"))

    add(Experiment("exploratory_segment_bias_v1", "exploratory_segment", "exploratory", lambda m: segment_calibrated_pred(df, pivot, m, ["region_x_trend"], 0.35, 120, (0.97, 1.03)), "Исследовательская segment bias calibration по region x trend.", "test_exploratory_segment_bias_v1.csv", "exploratory"))
    add(Experiment("exploratory_outlier_v1", "exploratory_outlier", "exploratory", lambda m: outlier_smooth_pred(pivot, m, "symmetric", 0.08, 0.75), "Исследовательская более сильная outlier correction.", "test_exploratory_outlier_v1.csv", "exploratory"))
    add(Experiment("exploratory_selector_v1", "exploratory_selector", "exploratory", lambda m: selector_by_segment_pred(df, pivot, m, ["region_x_rto_decile"], 0.75, 0.03), "Исследовательский selector по region x rto_decile.", "test_exploratory_selector_v1.csv", "exploratory"))
    return exps


def score_experiment(exp: Experiment, df: pd.DataFrame, pivot: pd.DataFrame, baseline_test: pd.Series, current_best_test: pd.Series) -> dict:
    scores = {}
    current_scores = {}
    for month in FOLDS:
        pred = exp.pred_func(month).reindex(pivot.index)
        scores[month] = mape_percent(pivot[month], pred)
        current_scores[month] = mape_percent(pivot[month], current_best_pred(df, pivot, month))
    test_pred = exp.pred_func(11).reindex(current_best_test.index)
    rel = ((test_pred - current_best_test) / current_best_test).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    abs_rel = rel.abs()
    changed = abs_rel > 0
    fold10_delta = scores[10] - current_scores[10]
    weighted = scores[8] * 0.1 + scores[9] * 0.2 + scores[10] * 0.7
    safe = weighted + 0.6 * max(0.0, fold10_delta) + 18.0 * abs_rel.mean() + 9.0 * (abs_rel > 0.003).mean() + 5.0 * (abs_rel > 0.01).mean()
    exploratory = weighted + 0.35 * max(0.0, fold10_delta) + 4.0 * abs_rel.mean() + 2.0 * (abs_rel > 0.03).mean()
    return {
        "experiment_name": exp.experiment_name,
        "model_name": exp.model_name,
        "hypothesis_group": exp.hypothesis_group,
        "category": exp.category,
        "expected_direction": exp.expected_direction,
        "fold8_mape": scores[8],
        "fold9_mape": scores[9],
        "fold10_mape": scores[10],
        "mean_mape": float(np.mean(list(scores.values()))),
        "weighted_mape_235": scores[8] * 0.2 + scores[9] * 0.3 + scores[10] * 0.5,
        "weighted_mape_127": weighted,
        "delta_vs_current_best_fold10": fold10_delta,
        "safe_risk_score": safe,
        "exploratory_score": exploratory,
        "mean_abs_delta_vs_current_best": float((test_pred - current_best_test).abs().mean()),
        "mean_rel_delta_vs_current_best": float(rel.mean()),
        "mean_abs_relative_delta_vs_current_best": float(abs_rel.mean()),
        "max_rel_delta_vs_current_best": float(abs_rel.max()),
        "share_abs_delta_vs_current_best_gt_0p1pct": float((abs_rel > 0.001).mean()),
        "share_abs_delta_vs_current_best_gt_0p3pct": float((abs_rel > 0.003).mean()),
        "share_abs_delta_vs_current_best_gt_1pct": float((abs_rel > 0.010).mean()),
        "share_abs_delta_vs_current_best_gt_3pct": float((abs_rel > 0.030).mean()),
        "share_changed_stores": float(changed.mean()),
        "generated_submission": "",
        "comment": exp.comment,
    }


def enrich_with_lb(results: pd.DataFrame) -> pd.DataFrame:
    lb = load_leaderboard()
    lb["lb_score"] = pd.to_numeric(lb["lb_score"], errors="coerce")
    lb["lb_mape"] = pd.to_numeric(lb["lb_mape"], errors="coerce")
    results["lb_score"] = np.nan
    results["lb_mape"] = np.nan
    results["lb_verdict"] = ""
    aliases = {"ratio_shrink_b0p06_c97_103": "ratio_shrink_b0p06_c97_103"}
    for idx, row in results.iterrows():
        names = {row["experiment_name"], aliases.get(row["experiment_name"], row["experiment_name"])}
        hit = lb[lb["model_name"].astype(str).isin(names)]
        if len(hit):
            latest = hit.iloc[-1]
            results.loc[idx, "lb_score"] = latest["lb_score"]
            results.loc[idx, "lb_mape"] = latest["lb_mape"]
            results.loc[idx, "lb_verdict"] = latest["verdict"]
    results["delta_lb_vs_best"] = pd.to_numeric(results["lb_score"], errors="coerce") - 95.88
    return results


def is_duplicate_prediction(pred: pd.Series, existing_files: list[Path]) -> bool:
    rounded = np.round(pred.sort_index().to_numpy(dtype=float), 2)
    for path in existing_files:
        if not path.exists():
            continue
        try:
            other = pd.read_csv(path).sort_values(ID_COL)["rto"].to_numpy(dtype=float)
        except Exception:
            continue
        if len(other) == len(rounded) and np.array_equal(np.round(other, 2), rounded):
            return True
    return False


def choose_candidates(results: pd.DataFrame) -> list[str]:
    sent = set(results.loc[results["lb_verdict"].astype(str) != "", "experiment_name"])
    pool = results[~results["experiment_name"].isin(sent)].copy()
    pool = pool[(pool["max_rel_delta_vs_current_best"] <= 0.06) & (pool["share_abs_delta_vs_current_best_gt_3pct"] <= 0.25)]
    selected: list[str] = []
    plan = [
        ("segment_calibration", "safe_risk_score", 3),
        ("selector", "exploratory_score", 2),
        ("weighted_mixture", "safe_risk_score", 2),
        ("temporal_model", "exploratory_score", 2),
        ("clustering", "exploratory_score", 1),
        ("outlier_specific", "exploratory_score", 2),
        ("low_rto", "exploratory_score", 1),
        ("exploratory", "exploratory_score", 2),
    ]
    for group, score_col, limit in plan:
        part = pool[pool["hypothesis_group"] == group].sort_values([score_col, "fold10_mape"])
        selected.extend(part.head(limit)["experiment_name"].tolist())
    return list(dict.fromkeys(selected))[:12]


def write_segment_report(segment_mining: pd.DataFrame) -> None:
    over = segment_mining.sort_values("signed_bias_current_best", ascending=False).head(15)
    under = segment_mining.sort_values("signed_bias_current_best").head(15)
    winners = segment_mining[segment_mining["improvement_best_vs_current"] > 0].sort_values("improvement_best_vs_current", ascending=False).head(20)
    lines = [
        "# Segment error mining",
        "",
        "Отчет построен по OOF-прогнозам fold 8, 9, 10. Малые сегменты отфильтрованы, для комбинаций используется более высокий порог размера.",
        "",
        "## Сегменты, где current_best систематически завышает",
        "",
        "```text",
        over[["segment_type", "segment_value", "n", "current_best_mape", "signed_bias_current_best", "same_bias_sign_folds", "best_model_in_segment", "improvement_best_vs_current"]].to_string(index=False),
        "```",
        "",
        "## Сегменты, где current_best систематически занижает",
        "",
        "```text",
        under[["segment_type", "segment_value", "n", "current_best_mape", "signed_bias_current_best", "same_bias_sign_folds", "best_model_in_segment", "improvement_best_vs_current"]].to_string(index=False),
        "```",
        "",
        "## Где current_best проигрывает другим моделям",
        "",
        "```text",
        winners[["segment_type", "segment_value", "n", "current_best_mape", "best_model_in_segment", "best_model_mape", "improvement_best_vs_current", "stability_score"]].to_string(index=False),
        "```",
        "",
        "## Вывод",
        "",
        "- Сильные средние выводы по всем магазинам почти исчерпаны: прирост current_best над baseline мал, но стабилен.",
        "- Адресные поправки логично искать в сегментах по уровню РТО, волатильности, тренду, площади и комбинациях регион x decile.",
        "- На части сегментов current_best систематически завышает или занижает прогноз, поэтому segment calibration и selector имеют смысл как следующий класс гипотез.",
        "- Для маленьких групп выводы не используются: риск случайного fold-шума выше ожидаемого выигрыша.",
    ]
    (REPORTS_DIR / "segment_error_mining.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_mixture_weights(df: pd.DataFrame, pivot: pd.DataFrame) -> None:
    rows = []
    for name, bucket, min_w in [
        ("global", None, 0.60),
        ("rto_decile", "rto_decile", 0.50),
        ("volatility", "volatility_bucket", 0.50),
        ("trend", "trend_bucket", 0.50),
    ]:
        _, w = weighted_oof_pred(df, pivot, 11, bucket, min_w)
        w.insert(0, "mixture_name", name)
        rows.append(w)
    weights = pd.concat(rows, ignore_index=True)
    lines = [
        "# Веса mixture of experts",
        "",
        "Веса подбирались на исторических переходах с ограничением на минимальный вес `current_best`. Это не попытка угадать leaderboard, а способ проверить, есть ли устойчивые локальные преимущества у альтернативных прогнозов.",
        "",
        "```text",
        weights.head(80).to_string(index=False),
        "```",
    ]
    (REPORTS_DIR / "mixture_weights.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cluster_report(df: pd.DataFrame, pivot: pd.DataFrame) -> None:
    rows = []
    for k in [5, 10, 20, 40]:
        for month in FOLDS:
            pred = cluster_ratio_pred(df, pivot, month, k, 0.18, (0.97, 1.03))
            rows.append({"k": k, "fold": month, "mape": mape_percent(pivot[month], pred), "current_best_mape": mape_percent(pivot[month], current_best_pred(df, pivot, month))})
    table = pd.DataFrame(rows)
    lines = [
        "# Анализ кластеров траекторий",
        "",
        "Кластеры строились по нормализованной временной траектории и ratio-признакам. Прогноз использует медианный следующий ratio кластера со shrink к 1.",
        "",
        "```text",
        table.to_string(index=False),
        "```",
        "",
        "Вывод: кластеризация дает интерпретируемый исследовательский сигнал, но локально не выглядит надежнее current_best. Поэтому сохранен только ограниченный blend/кандидат, а не агрессивный кластерный прогноз.",
    ]
    (REPORTS_DIR / "cluster_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reports(results: pd.DataFrame, generated: pd.DataFrame, segment_mining: pd.DataFrame) -> None:
    best = load_best_submission()
    safe_top = results.sort_values("safe_risk_score").head(20)
    exp_top = results.sort_values("exploratory_score").head(20)
    safe_rec = generated[generated["category"].isin(["safe", "moderate"])].sort_values("safe_risk_score")
    exp_rec = generated[generated["category"].eq("exploratory")].sort_values("exploratory_score")
    lines = [
        "# Отчет по экспериментам",
        "",
        "## Краткий вывод",
        "",
        "Текущий подтвержденный потолок: `95.88`. Простые микропоправки вокруг `ratio_shrink` почти исчерпаны: большинство близких blend/selector/mixture кандидатов дают 95.87-95.88. Дальше работаем через поиск систематических ошибок и адресные corrections.",
        "",
        f"Текущий best: `{best['filename']}`, score `{best['lb_score']:.2f}`, LB MAPE `{best['lb_mape']:.2f}`. `test.csv` должен оставаться копией этого файла.",
        "",
        "## Что пробовали на этом этапе",
        "",
        "- `segment calibration`: множитель к current_best по сегментному bias с сильным shrink к 1.",
        "- `selector/gating`: выбор альтернативной модели внутри сегмента, затем safety blend с current_best.",
        "- `weighted mixture`: OOF-оптимизация весов глобально и по bucket'ам.",
        "- `temporal pattern model`: Ridge/Huber и HGB по форме временной траектории магазина.",
        "- `clustering`: KMeans по нормализованным траекториям и ratio-признакам.",
        "- `outlier-specific correction`: отдельная обработка магазинов, где октябрь похож на аномалию.",
        "- `low-RTO MAPE-aware`: отдельная логика для нижних decile РТО, где MAPE особенно чувствителен.",
        "",
        "## Новая логика риска",
        "",
        "Теперь есть два рейтинга. `safe_risk_score` сильно штрафует отклонение от current_best. `exploratory_score` мягче относится к отличиям и нужен, чтобы не получать только почти идентичные копии.",
        "",
        "```text",
        "safe_risk_score = weighted_mape_127 + штраф за fold10 + сильный штраф за отклонения >0.3% и >1%",
        "exploratory_score = weighted_mape_127 + меньший штраф за отклонение + контроль fold10 и доли изменений >3%",
        "```",
        "",
        "## Топ по safe_risk_score",
        "",
        "```text",
        safe_top[["experiment_name", "hypothesis_group", "fold10_mape", "weighted_mape_127", "safe_risk_score", "exploratory_score", "generated_submission", "mean_abs_relative_delta_vs_current_best", "max_rel_delta_vs_current_best", "share_abs_delta_vs_current_best_gt_1pct"]].to_string(index=False),
        "```",
        "",
        "## Топ по exploratory_score",
        "",
        "```text",
        exp_top[["experiment_name", "hypothesis_group", "fold10_mape", "weighted_mape_127", "safe_risk_score", "exploratory_score", "generated_submission", "mean_abs_relative_delta_vs_current_best", "max_rel_delta_vs_current_best", "share_abs_delta_vs_current_best_gt_3pct"]].to_string(index=False),
        "```",
        "",
        "## Главные выводы segment mining",
        "",
        "```text",
        segment_mining.head(20)[["segment_type", "segment_value", "n", "current_best_mape", "best_model_in_segment", "improvement_best_vs_current", "signed_bias_current_best", "stability_score"]].to_string(index=False),
        "```",
    ]
    (REPORTS_DIR / "experiments.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rec_lines = [
        "# Рекомендованные сабмиты",
        "",
        f"Текущий лучший подтвержденный файл: `{best['filename']}`, score `{best['lb_score']:.2f}`. Его можно восстановить командой `python scripts/restore_best_submission.py`.",
        "",
        "## Safe candidates",
        "",
    ]
    if len(safe_rec):
        rec_lines.extend(["```text", safe_rec[["generated_submission", "hypothesis_group", "safe_risk_score", "weighted_mape_127", "mean_abs_relative_delta_vs_current_best", "max_rel_delta_vs_current_best", "share_changed_stores"]].head(6).to_string(index=False), "```"])
    else:
        rec_lines.append("Безопасные кандидаты не прошли фильтры генерации.")
    rec_lines.extend(["", "## Exploratory candidates", ""])
    if len(exp_rec):
        rec_lines.extend(["```text", exp_rec[["generated_submission", "hypothesis_group", "exploratory_score", "weighted_mape_127", "mean_abs_relative_delta_vs_current_best", "max_rel_delta_vs_current_best", "share_changed_stores"]].head(6).to_string(index=False), "```"])
    else:
        rec_lines.append("Исследовательские кандидаты не прошли фильтры генерации.")
    rec_lines.extend(
        [
            "",
            "## Как отправлять",
            "",
            "1. Сначала отправлять safe candidates: они ближе к current best и меньше рискуют потерять качество.",
            "2. Если safe-кандидаты не двигают score, отправлять exploratory candidates: они меняют более адресные группы магазинов и могут сдвинуть плато 95.88.",
            "3. После каждого результата обязательно записывать LB в реестр:",
            "",
            "```bash",
            "python scripts/record_leaderboard_result.py --file submissions/<file>.csv --model <model_name> --lb-score <score> --verdict OK --comment \"комментарий\"",
            "```",
            "",
            "4. Если новый файл хуже, восстановить best:",
            "",
            "```bash",
            "python scripts/restore_best_submission.py",
            "```",
        ]
    )
    (REPORTS_DIR / "recommended_submissions.md").write_text("\n".join(rec_lines) + "\n", encoding="utf-8")


def update_submission_space_note() -> None:
    path = REPORTS_DIR / "submission_space_analysis.md"
    text = ""
    if path.exists():
        text = path.read_text(encoding="utf-8")
    addition = """

## Вывод после выхода на плато 95.88

Почти идентичные файлы дают тот же score, поэтому очередные микроскопические blend вокруг `ratio_shrink` уже малоинформативны. Чтобы сдвинуться выше 95.88, нужны адресные изменения групп магазинов: сегменты с устойчивым bias, октябрьские outlier, низкие decile РТО и траекторные кластеры.

Глобальный рост вреден: множители `1.010` и `1.020` заметно ухудшили LB. Residual-модель тоже вредна. `ratio_shrink` остается полезной базой, но дальнейший поиск должен использовать его как current best, а не как единственный источник новых файлов.
"""
    if "Вывод после выхода на плато 95.88" not in text:
        path.write_text(text.rstrip() + addition + "\n", encoding="utf-8")


def run_error_analysis() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_train(TRAIN_PATH)
    pivot = df.pivot(index=ID_COL, columns=MONTH_COL, values=TARGET_COL).sort_index()
    oof = build_oof_predictions(df, pivot)
    oof.to_csv(REPORTS_DIR / "oof_predictions.csv", index=False, encoding="utf-8")
    mining = build_segment_error_mining(oof)
    mining.to_csv(REPORTS_DIR / "segment_error_mining.csv", index=False, encoding="utf-8")
    write_segment_report(mining)
    # Совместимость со старым именем отчета.
    (REPORTS_DIR / "error_analysis.md").write_text((REPORTS_DIR / "segment_error_mining.md").read_text(encoding="utf-8"), encoding="utf-8")
    mining.to_csv(REPORTS_DIR / "error_slices.csv", index=False, encoding="utf-8")
    print(f"Сохранено: {REPORTS_DIR / 'oof_predictions.csv'}")
    print(f"Сохранено: {REPORTS_DIR / 'segment_error_mining.csv'}")
    print(f"Сохранено: {REPORTS_DIR / 'segment_error_mining.md'}")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_known_leaderboard_rows()
    clean_leaderboard_comments()
    ensure_current_best()

    df = load_train(TRAIN_PATH)
    pivot = df.pivot(index=ID_COL, columns=MONTH_COL, values=TARGET_COL).sort_index()
    baseline_test = baseline_pred(pivot, 11)
    current_test = current_best_pred(df, pivot, 11)

    oof = build_oof_predictions(df, pivot)
    oof.to_csv(REPORTS_DIR / "oof_predictions.csv", index=False, encoding="utf-8")
    segment_mining = build_segment_error_mining(oof)
    segment_mining.to_csv(REPORTS_DIR / "segment_error_mining.csv", index=False, encoding="utf-8")
    write_segment_report(segment_mining)
    (REPORTS_DIR / "error_analysis.md").write_text((REPORTS_DIR / "segment_error_mining.md").read_text(encoding="utf-8"), encoding="utf-8")
    segment_mining.to_csv(REPORTS_DIR / "error_slices.csv", index=False, encoding="utf-8")

    exps = build_experiments(df, pivot)
    exp_by_name = {exp.experiment_name: exp for exp in exps}
    results = pd.DataFrame([score_experiment(exp, df, pivot, baseline_test, current_test) for exp in exps])
    results = enrich_with_lb(results)

    selected = choose_candidates(results)
    existing = list(SUBMISSIONS_DIR.glob("test_*.csv"))
    generated_rows = []
    used = set()
    for name in selected:
        exp = exp_by_name[name]
        pred = exp.pred_func(11).reindex(current_test.index)
        filename = exp.preferred_filename or f"test_{exp.experiment_name}.csv"
        if filename in used:
            continue
        used.add(filename)
        path = SUBMISSIONS_DIR / filename
        if path.exists():
            validate_saved_csv(path)
        else:
            if is_duplicate_prediction(pred, existing):
                continue
            save_submission(submission_frame(pred), path)
            validate_saved_csv(path)
            existing.append(path)
        rel_path = str(path.relative_to(TRAIN_PATH.parents[1])).replace("\\", "/")
        results.loc[results["experiment_name"] == name, "generated_submission"] = rel_path
        row = results.loc[results["experiment_name"] == name].iloc[0].copy()
        row["generated_submission"] = rel_path
        generated_rows.append(row.to_dict())
        if len(generated_rows) >= 12:
            break

    results = enrich_with_lb(results)
    results.sort_values("safe_risk_score").to_csv(REPORTS_DIR / "experiment_results.csv", index=False, encoding="utf-8")
    generated = pd.DataFrame(generated_rows)
    write_mixture_weights(df, pivot)
    write_cluster_report(df, pivot)
    write_reports(results, generated, segment_mining)
    update_submission_space_note()

    registry_rows = []
    if len(generated):
        registry = pd.read_csv(SUBMISSION_REGISTRY_PATH, encoding="utf-8") if SUBMISSION_REGISTRY_PATH.exists() else pd.DataFrame()
        existing_files = set(registry.get("filename", pd.Series(dtype=str)).astype(str))
        for row in generated_rows:
            if row["generated_submission"] not in existing_files:
                registry_rows.append(
                    {
                        "submitted_at": "",
                        "filename": row["generated_submission"],
                        "model_name": row["model_name"],
                        "local_cv_mape": row["weighted_mape_127"],
                        "lb_score": "",
                        "lb_mape": "",
                        "verdict": "",
                        "comment": row["comment"],
                    }
                )
        if registry_rows:
            registry = pd.concat([registry, pd.DataFrame(registry_rows)], ignore_index=True)
            registry.to_csv(SUBMISSION_REGISTRY_PATH, index=False, encoding="utf-8")

    restored = restore_best_submission()
    print(f"Сохранено: {REPORTS_DIR / 'experiment_results.csv'}")
    print(f"Сохранено: {REPORTS_DIR / 'oof_predictions.csv'}")
    print(f"Сохранено: {REPORTS_DIR / 'segment_error_mining.csv'}")
    print(f"Сохранено: {REPORTS_DIR / 'experiments.md'}")
    print(f"Сохранено: {REPORTS_DIR / 'recommended_submissions.md'}")
    print("Созданные новые сабмиты:")
    for row in generated_rows:
        print(f"  {row['generated_submission']}")
    print(f"Восстановлен лучший подтвержденный test.csv: {restored}")
