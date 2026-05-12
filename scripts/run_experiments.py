from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ID_COL, MONTH_COL, REPORTS_DIR, SUBMISSIONS_DIR, TARGET_COL, TRAIN_PATH
from src.features import load_train
from src.metrics import mape_percent
from src.registry import SUBMISSION_REGISTRY_PATH, load_best_submission, load_leaderboard, restore_best_submission
from src.submit import save_submission


FOLDS = [8, 9, 10]
WEIGHT_235 = {8: 0.2, 9: 0.3, 10: 0.5}
WEIGHT_127 = {8: 0.1, 9: 0.2, 10: 0.7}

REGION_COL = "Регион"
AREA_COL = "Торговая площадь, категориальный"
CASH_COL = "Количество касс"
ALCOHOL_COL = "Флаг алкогольной лицензии"
OPEN_COL = "Дата открытия, категориальный"

SEGMENT_CACHE: dict[tuple, pd.Series] = {}
RAW_RATIO_CACHE: dict[int, pd.Series] = {}


@dataclass
class Experiment:
    experiment_name: str
    model_name: str
    pred_func: callable
    comment: str
    preferred_filename: str = ""
    force_generate: bool = False


def safe_suffix(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def clip_series(values, index=None) -> pd.Series:
    out = pd.Series(values, index=index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.clip(lower=0.0)


def submission_frame(pred: pd.Series) -> pd.DataFrame:
    return (
        pd.DataFrame({ID_COL: pred.index, "rto": np.round(pred.to_numpy(dtype=float), 2)})
        .sort_values(ID_COL)
        .reset_index(drop=True)
    )


def baseline_pred(pivot: pd.DataFrame, month: int) -> pd.Series:
    return pivot[month - 1].copy()


def segment_multiplier(
    df: pd.DataFrame,
    pivot: pd.DataFrame,
    month: int,
    group_cols: list[str],
    shrink_weight: float,
    k: int,
    clip_bounds: tuple[float, float],
) -> pd.Series:
    key = (month, tuple(group_cols), float(shrink_weight), int(k), tuple(clip_bounds))
    if key in SEGMENT_CACHE:
        return SEGMENT_CACHE[key].copy()

    hist = df[df[MONTH_COL] < month].sort_values([ID_COL, MONTH_COL]).copy()
    hist["prev_rto"] = hist.groupby(ID_COL)[TARGET_COL].shift(1)
    hist["growth"] = hist[TARGET_COL] / hist["prev_rto"]
    hist = hist.replace([np.inf, -np.inf], np.nan).dropna(subset=["growth"])
    hist = hist[(hist["prev_rto"] > 0) & (hist["growth"].between(0.70, 1.35))]
    global_growth = float(hist["growth"].median()) if len(hist) else 1.0

    stats = hist.groupby(group_cols, dropna=False)["growth"].agg(["median", "count"]).reset_index()
    stats["size_w"] = stats["count"] / (stats["count"] + k)
    stats["mult"] = 1.0 + shrink_weight * stats["size_w"] * (stats["median"] - global_growth)
    stats["mult"] = stats["mult"].clip(*clip_bounds)

    current = df[df[MONTH_COL] == month - 1][[ID_COL] + group_cols]
    merged = current.merge(stats[group_cols + ["mult"]], on=group_cols, how="left").set_index(ID_COL)
    result = merged["mult"].reindex(pivot.index).fillna(1.0)
    SEGMENT_CACHE[key] = result.copy()
    return result


def segment_pred(
    df: pd.DataFrame,
    pivot: pd.DataFrame,
    month: int,
    group_cols: list[str],
    shrink_weight: float,
    k: int,
    clip_bounds: tuple[float, float],
) -> pd.Series:
    base = baseline_pred(pivot, month)
    mult = segment_multiplier(df, pivot, month, group_cols, shrink_weight, k, clip_bounds)
    return clip_series(base * mult, index=base.index)


def segment_blend_multiplier(
    df: pd.DataFrame,
    pivot: pd.DataFrame,
    month: int,
    shrink_weight: float = 0.05,
    k: int = 500,
    clip_bounds: tuple[float, float] = (0.97, 1.03),
) -> pd.Series:
    region = segment_multiplier(df, pivot, month, [REGION_COL], shrink_weight, k, clip_bounds)
    area = segment_multiplier(df, pivot, month, [AREA_COL], shrink_weight, k, clip_bounds)
    alcohol = segment_multiplier(df, pivot, month, [ALCOHOL_COL], shrink_weight, k, clip_bounds)
    return 0.45 * region + 0.35 * area + 0.20 * alcohol


def segment_blend_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int) -> pd.Series:
    base = baseline_pred(pivot, month)
    return clip_series(base * segment_blend_multiplier(df, pivot, month), index=base.index)


def raw_ratio_signal(df: pd.DataFrame, pivot: pd.DataFrame, month: int) -> pd.Series:
    if month in RAW_RATIO_CACHE:
        return RAW_RATIO_CACHE[month].copy()
    region_area = segment_multiplier(df, pivot, month, [REGION_COL, AREA_COL], 0.30, 300, (0.95, 1.05))
    recent = (pivot[month - 1] / pivot[month - 2]).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.85, 1.15)
    result = 0.85 * region_area + 0.15 * recent
    RAW_RATIO_CACHE[month] = result.copy()
    return result


def ratio_shrink_ratio(
    df: pd.DataFrame,
    pivot: pd.DataFrame,
    month: int,
    beta: float,
    clip_bounds: tuple[float, float],
) -> pd.Series:
    ratio = 1.0 + beta * (raw_ratio_signal(df, pivot, month) - 1.0)
    return ratio.clip(*clip_bounds)


def ratio_shrink_pred(
    df: pd.DataFrame,
    pivot: pd.DataFrame,
    month: int,
    beta: float,
    clip_bounds: tuple[float, float],
) -> pd.Series:
    base = baseline_pred(pivot, month)
    return clip_series(base * ratio_shrink_ratio(df, pivot, month, beta, clip_bounds), index=base.index)


def current_best_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int) -> pd.Series:
    return ratio_shrink_pred(df, pivot, month, 0.05, (0.97, 1.03))


def outlier_smooth_pred(pivot: pd.DataFrame, month: int, mode: str, threshold: float, blend: float) -> pd.Series:
    base = baseline_pred(pivot, month)
    prev_mean = pivot[[month - 4, month - 3, month - 2]].mean(axis=1)
    deviation = (base / prev_mean - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if mode == "rollback":
        mask = deviation > threshold
    elif mode == "symmetric":
        mask = deviation.abs() > threshold
    else:
        raise ValueError(f"Unknown outlier mode: {mode}")
    pred = base.copy()
    pred.loc[mask] = blend * base.loc[mask] + (1.0 - blend) * prev_mean.loc[mask]
    return clip_series(pred, index=base.index)


def residual_centered_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int, limit: float = 0.01) -> pd.Series:
    base = baseline_pred(pivot, month)
    hist = df[df[MONTH_COL] < month].sort_values([ID_COL, MONTH_COL]).copy()
    hist["prev_rto"] = hist.groupby(ID_COL)[TARGET_COL].shift(1)
    hist["rel_resid"] = hist[TARGET_COL] / hist["prev_rto"] - 1.0
    hist = hist.replace([np.inf, -np.inf], np.nan).dropna(subset=["rel_resid"])
    hist = hist[hist["rel_resid"].between(-0.20, 0.20)]
    stats = hist.groupby(REGION_COL, dropna=False)["rel_resid"].agg(["median", "count"]).reset_index()
    stats["w"] = stats["count"] / (stats["count"] + 300)
    stats["corr"] = stats["w"] * stats["median"]
    current = df[df[MONTH_COL] == month - 1][[ID_COL, REGION_COL]]
    corr = current.merge(stats[[REGION_COL, "corr"]], on=REGION_COL, how="left").set_index(ID_COL)["corr"].reindex(pivot.index).fillna(0.0)
    corr = (corr - corr.median()).clip(-limit, limit)
    return clip_series(base * (1.0 + corr), index=base.index)


def growth_features(pivot: pd.DataFrame, month: int) -> pd.DataFrame:
    ratios = pd.DataFrame(
        {
            "ratio_8_7": pivot[month - 3] / pivot[month - 4],
            "ratio_9_8": pivot[month - 2] / pivot[month - 3],
            "ratio_10_9": pivot[month - 1] / pivot[month - 2],
        }
    ).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    ratios["volatility"] = ratios[["ratio_8_7", "ratio_9_8", "ratio_10_9"]].std(axis=1)
    ratios["same_direction"] = np.sign(ratios["ratio_10_9"] - 1.0) == np.sign(ratios["ratio_9_8"] - 1.0)
    prev_mean = pivot[[month - 4, month - 3, month - 2]].mean(axis=1)
    ratios["october_outlier_score"] = (pivot[month - 1] / prev_mean - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).abs()
    return ratios


def gated_ratio_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int, gate: str, threshold: float) -> pd.Series:
    base = baseline_pred(pivot, month)
    best = current_best_pred(df, pivot, month)
    ratio = ratio_shrink_ratio(df, pivot, month, 0.05, (0.97, 1.03))
    feats = growth_features(pivot, month)

    if gate == "stable":
        mask = feats["volatility"] < threshold
    elif gate == "unstable":
        mask = feats["volatility"] >= threshold
    elif gate == "smallcorr":
        mask = (ratio - 1.0).abs() < threshold
    elif gate == "direction":
        mask = feats["same_direction"]
    elif gate == "outlier":
        mask = feats["october_outlier_score"] > threshold
    else:
        raise ValueError(f"Unknown gate: {gate}")

    pred = base.copy()
    pred.loc[mask] = best.loc[mask]
    return clip_series(pred, index=base.index)


def ratio_segment_prior_pred(
    df: pd.DataFrame,
    pivot: pd.DataFrame,
    month: int,
    beta_model: float,
    beta_segment: float,
    segment_kind: str,
    clip_bounds: tuple[float, float],
) -> pd.Series:
    base = baseline_pred(pivot, month)
    model_ratio = raw_ratio_signal(df, pivot, month)
    if segment_kind == "region":
        segment_ratio = segment_multiplier(df, pivot, month, [REGION_COL], 0.05, 500, (0.97, 1.03))
    elif segment_kind == "area":
        segment_ratio = segment_multiplier(df, pivot, month, [AREA_COL], 0.05, 500, (0.97, 1.03))
    elif segment_kind == "alcohol":
        segment_ratio = segment_multiplier(df, pivot, month, [ALCOHOL_COL], 0.05, 500, (0.97, 1.03))
    elif segment_kind == "blend":
        segment_ratio = segment_blend_multiplier(df, pivot, month, 0.05, 500, (0.97, 1.03))
    else:
        raise ValueError(f"Unknown segment kind: {segment_kind}")
    final_ratio = 1.0 + beta_model * (model_ratio - 1.0) + beta_segment * (segment_ratio - 1.0)
    return clip_series(base * final_ratio.clip(*clip_bounds), index=base.index)


def categorical_svd_probe(df: pd.DataFrame) -> str:
    string_cols = [col for col in df.columns if df[col].dtype == "object"]
    return (
        "В данных есть категориальные признаки: "
        + ", ".join(f"`{col}`" for col in string_cols)
        + ". Свободного текста нет, поэтому Word2Vec/NLP не выглядит оправданным. "
        "Для микрокоррекций уже используются сегментные признаки; отдельный embedding-сабмит не генерировался."
    )


def build_experiments(df: pd.DataFrame, pivot: pd.DataFrame) -> list[Experiment]:
    experiments: list[Experiment] = [
        Experiment(
            "baseline_last_month",
            "baseline_last_month",
            lambda month: baseline_pred(pivot, month),
            "Базовый прогноз: РТО предыдущего месяца.",
        ),
        Experiment(
            "ratio_shrink_b0p05_c97_103",
            "ratio_shrink_model",
            lambda month: current_best_pred(df, pivot, month),
            "Текущий лучший подтвержденный ratio_shrink: beta=0.05, clip=(0.97, 1.03).",
        ),
    ]

    for c in [0.995, 0.9975, 1.0025, 1.010, 1.015, 1.020]:
        name = str(c).rstrip("0").rstrip(".").replace(".", "")
        experiments.append(
            Experiment(
                f"last_month_mult_{name}",
                "last_month_multiplier",
                lambda month, c=c: baseline_pred(pivot, month) * c,
                f"Исторически отправленный глобальный множитель {c}.",
            )
        )

    experiments.extend(
        [
            Experiment(
                "residual_centered_v1",
                "residual_centered",
                lambda month: residual_centered_pred(df, pivot, month, 0.01),
                "Исторически отправленная centered residual поправка.",
            ),
            Experiment(
                "segment_alcohol_s0p05_k500_c97_103",
                "segment_shrink",
                lambda month: segment_pred(df, pivot, month, [ALCOHOL_COL], 0.05, 500, (0.97, 1.03)),
                "Исторически отправленная микропоправка по флагу алкоголя, clip=(0.97, 1.03).",
            ),
            Experiment(
                "segment_alcohol_s0p05_k500_c98_102",
                "segment_shrink",
                lambda month: segment_pred(df, pivot, month, [ALCOHOL_COL], 0.05, 500, (0.98, 1.02)),
                "Исторически отправленная микропоправка по флагу алкоголя, clip=(0.98, 1.02).",
            ),
            Experiment(
                "segment_area_shrink_v1",
                "segment_shrink",
                lambda month: segment_pred(df, pivot, month, [AREA_COL], 0.05, 500, (0.97, 1.03)),
                "Исторически отправленная микропоправка по категории площади.",
            ),
            Experiment(
                "segment_blend_shrink_v1",
                "segment_blend_shrink",
                lambda month: segment_blend_pred(df, pivot, month),
                "Исторически отправленная смесь сегментных микропоправок.",
            ),
            Experiment(
                "segment_region_shrink_v1",
                "segment_shrink",
                lambda month: segment_pred(df, pivot, month, [REGION_COL], 0.05, 500, (0.97, 1.03)),
                "Исторически отправленная микропоправка по региону.",
            ),
        ]
    )

    for beta in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]:
        for clip_bounds in [(0.995, 1.005), (0.990, 1.010), (0.985, 1.015), (0.980, 1.020), (0.970, 1.030)]:
            clip_name = f"c{int(clip_bounds[0] * 1000)}_{int(clip_bounds[1] * 1000)}"
            name = f"ratio_shrink_b{safe_suffix(beta)}_{clip_name}"
            preferred = {
                "ratio_shrink_b0p03_c990_1010": "test_ratio_shrink_b0p03_c99_101.csv",
                "ratio_shrink_b0p04_c970_1030": "test_ratio_shrink_b0p04_c97_103.csv",
                "ratio_shrink_b0p06_c970_1030": "test_ratio_shrink_b0p06_c97_103.csv",
                "ratio_shrink_b0p05_c990_1010": "test_ratio_shrink_b0p05_c99_101.csv",
                "ratio_shrink_b0p07_c980_1020": "test_ratio_shrink_b0p07_c98_102.csv",
            }.get(name, "")
            experiments.append(
                Experiment(
                    name,
                    "ratio_shrink_model",
                    lambda month, beta=beta, clip_bounds=clip_bounds: ratio_shrink_pred(df, pivot, month, beta, clip_bounds),
                    f"Тонкая настройка ratio_shrink: beta={beta}, clip={clip_bounds}.",
                    preferred,
                )
            )

    experiments.extend(
        [
            Experiment(
                "best_ratio_blend_baseline_90_10",
                "best_ratio_blend",
                lambda month: 0.90 * current_best_pred(df, pivot, month) + 0.10 * baseline_pred(pivot, month),
                "Смесь: 90% current best ratio_shrink + 10% baseline_last_month.",
                "test_best_ratio_blend_baseline_90_10.csv",
            ),
            Experiment(
                "best_ratio_blend_baseline_95_05",
                "best_ratio_blend",
                lambda month: 0.95 * current_best_pred(df, pivot, month) + 0.05 * baseline_pred(pivot, month),
                "Смесь: 95% current best ratio_shrink + 5% baseline_last_month.",
                "test_best_ratio_blend_baseline_95_05.csv",
            ),
            Experiment(
                "best_ratio_blend_area_98_02",
                "best_ratio_blend",
                lambda month: 0.98 * current_best_pred(df, pivot, month)
                + 0.02 * segment_pred(df, pivot, month, [AREA_COL], 0.05, 500, (0.97, 1.03)),
                "Смесь: 98% current best + 2% segment area shrink.",
                "test_best_ratio_blend_area_98_02.csv",
            ),
            Experiment(
                "best_ratio_blend_segment_98_02",
                "best_ratio_blend",
                lambda month: 0.98 * current_best_pred(df, pivot, month) + 0.02 * segment_blend_pred(df, pivot, month),
                "Смесь: 98% current best + 2% segment blend shrink.",
                "test_best_ratio_blend_segment_98_02.csv",
            ),
            Experiment(
                "best_ratio_blend_alcohol_99_01",
                "best_ratio_blend",
                lambda month: 0.99 * current_best_pred(df, pivot, month)
                + 0.01 * segment_pred(df, pivot, month, [ALCOHOL_COL], 0.05, 500, (0.97, 1.03)),
                "Смесь: 99% current best + 1% segment alcohol shrink.",
            ),
        ]
    )

    for threshold in [0.03, 0.05, 0.07, 0.10]:
        experiments.append(
            Experiment(
                f"ratio_gated_stable_t{safe_suffix(threshold)}",
                "ratio_gated",
                lambda month, threshold=threshold: gated_ratio_pred(df, pivot, month, "stable", threshold),
                f"Применять ratio_shrink только для стабильных магазинов: volatility < {threshold}.",
                "test_ratio_gated_stable_v1.csv" if threshold == 0.05 else "",
            )
        )
        experiments.append(
            Experiment(
                f"ratio_gated_unstable_t{safe_suffix(threshold)}",
                "ratio_gated",
                lambda month, threshold=threshold: gated_ratio_pred(df, pivot, month, "unstable", threshold),
                f"Применять ratio_shrink только для нестабильных магазинов: volatility >= {threshold}.",
            )
        )

    for threshold in [0.005, 0.010, 0.015, 0.020]:
        experiments.append(
            Experiment(
                f"ratio_gated_smallcorr_t{safe_suffix(threshold)}",
                "ratio_gated",
                lambda month, threshold=threshold: gated_ratio_pred(df, pivot, month, "smallcorr", threshold),
                f"Применять ratio_shrink только если абсолютная ratio-поправка меньше {threshold}.",
                "test_ratio_gated_smallcorr_v1.csv" if threshold == 0.010 else "",
            )
        )

    experiments.append(
        Experiment(
            "ratio_gated_direction",
            "ratio_gated",
            lambda month: gated_ratio_pred(df, pivot, month, "direction", 0.0),
            "Применять ratio_shrink только если направление двух последних growth совпадает.",
        )
    )

    for threshold in [0.05, 0.08, 0.10, 0.12]:
        experiments.append(
            Experiment(
                f"ratio_gated_outlier_t{safe_suffix(threshold)}",
                "ratio_gated",
                lambda month, threshold=threshold: gated_ratio_pred(df, pivot, month, "outlier", threshold),
                f"Применять ratio_shrink только для октябрьских outlier-магазинов: score > {threshold}.",
                "test_ratio_gated_outlier_v1.csv" if threshold == 0.08 else "",
            )
        )

    for threshold in [1.10, 1.12, 1.15, 1.18]:
        for blend in [0.90, 0.93, 0.95, 0.97]:
            name = f"outlier_rollback_t{safe_suffix(threshold)}_b{safe_suffix(blend)}"
            experiments.append(
                Experiment(
                    name,
                    "outlier_rollback",
                    lambda month, threshold=threshold, blend=blend: outlier_smooth_pred(pivot, month, "rollback", threshold - 1.0, blend),
                    f"Откат октябрьского выброса: threshold={threshold}, blend={blend}.",
                    "test_outlier_rollback_t1p15_b0p95.csv" if name == "outlier_rollback_t1p15_b0p95" else "",
                )
            )

    for threshold in [0.10, 0.12, 0.15]:
        for blend in [0.93, 0.95, 0.97]:
            name = f"outlier_smoothing_t{safe_suffix(threshold)}_b{safe_suffix(blend)}"
            experiments.append(
                Experiment(
                    name,
                    "outlier_smoothing",
                    lambda month, threshold=threshold, blend=blend: outlier_smooth_pred(pivot, month, "symmetric", threshold, blend),
                    f"Симметричное сглаживание outlier: threshold={threshold}, blend={blend}.",
                    "test_outlier_smoothing_t0p12_b0p95.csv" if name == "outlier_smoothing_t0p12_b0p95" else "",
                )
            )

    for beta_model in [0.03, 0.05, 0.07]:
        for beta_segment in [0.01, 0.02, 0.03]:
            for clip_bounds in [(0.99, 1.01), (0.98, 1.02), (0.97, 1.03)]:
                for segment_kind in ["region", "area", "alcohol", "blend"]:
                    name = (
                        f"ratio_segment_prior_m{safe_suffix(beta_model)}_s{safe_suffix(beta_segment)}_"
                        f"{segment_kind}_c{int(clip_bounds[0] * 100)}_{int(clip_bounds[1] * 100)}"
                    )
                    preferred = ""
                    if beta_model == 0.03 and beta_segment == 0.01 and segment_kind == "blend" and clip_bounds == (0.99, 1.01):
                        preferred = "test_ratio_segment_prior_v1.csv"
                    if beta_model == 0.05 and beta_segment == 0.02 and segment_kind == "area" and clip_bounds == (0.98, 1.02):
                        preferred = "test_ratio_segment_prior_v2.csv"
                    experiments.append(
                        Experiment(
                            name,
                            "ratio_segment_prior",
                            lambda month, beta_model=beta_model, beta_segment=beta_segment, segment_kind=segment_kind, clip_bounds=clip_bounds: ratio_segment_prior_pred(
                                df, pivot, month, beta_model, beta_segment, segment_kind, clip_bounds
                            ),
                            f"Ratio shrink с сегментным prior: beta_model={beta_model}, beta_segment={beta_segment}, segment={segment_kind}, clip={clip_bounds}.",
                            preferred,
                        )
                    )

    return experiments


def score_experiment(exp: Experiment, df: pd.DataFrame, pivot: pd.DataFrame, baseline_test: pd.Series, current_best_test: pd.Series) -> dict:
    scores = {}
    current_scores = {}
    baseline_scores = {}
    for month in FOLDS:
        pred = exp.pred_func(month).reindex(pivot.index)
        scores[month] = mape_percent(pivot[month], pred)
        current_scores[month] = mape_percent(pivot[month], current_best_pred(df, pivot, month))
        baseline_scores[month] = mape_percent(pivot[month], baseline_pred(pivot, month))

    test_pred = exp.pred_func(11).reindex(current_best_test.index)
    abs_delta_baseline = (test_pred - baseline_test).abs()
    abs_delta_current = (test_pred - current_best_test).abs()
    rel_delta_current = ((test_pred - current_best_test) / current_best_test).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    abs_rel_current = rel_delta_current.abs()

    return {
        "experiment_name": exp.experiment_name,
        "model_name": exp.model_name,
        "fold8_mape": scores[8],
        "fold9_mape": scores[9],
        "fold10_mape": scores[10],
        "mean_mape": float(np.mean(list(scores.values()))),
        "weighted_mape_235": scores[8] * 0.2 + scores[9] * 0.3 + scores[10] * 0.5,
        "weighted_mape_127": scores[8] * 0.1 + scores[9] * 0.2 + scores[10] * 0.7,
        "current_best_fold10_mape": current_scores[10],
        "baseline_fold10_mape": baseline_scores[10],
        "delta_vs_baseline_last_month_fold10": scores[10] - baseline_scores[10],
        "delta_vs_current_best_fold10": scores[10] - current_scores[10],
        "mean_abs_delta_vs_baseline_last_month": float(abs_delta_baseline.mean()),
        "mean_abs_delta_vs_current_best": float(abs_delta_current.mean()),
        "mean_rel_delta_vs_current_best": float(rel_delta_current.mean()),
        "mean_abs_relative_delta_vs_current_best": float(abs_rel_current.mean()),
        "max_rel_delta_vs_current_best": float(abs_rel_current.max()),
        "share_abs_delta_vs_current_best_gt_0p1pct": float((abs_rel_current > 0.001).mean()),
        "share_abs_delta_vs_current_best_gt_0p3pct": float((abs_rel_current > 0.003).mean()),
        "share_abs_delta_vs_current_best_gt_1pct": float((abs_rel_current > 0.010).mean()),
        "generated_submission": "",
        "comment": exp.comment,
    }


def enrich_with_leaderboard(results: pd.DataFrame) -> pd.DataFrame:
    lb = load_leaderboard().copy()
    lb["filename"] = lb["filename"].astype(str).str.replace("\\", "/", regex=False)
    lb["lb_score"] = pd.to_numeric(lb["lb_score"], errors="coerce")
    lb["lb_mape"] = pd.to_numeric(lb["lb_mape"], errors="coerce")

    results["lb_score"] = np.nan
    results["lb_mape"] = np.nan
    results["lb_verdict"] = ""
    for idx, row in results.iterrows():
        exp_name = row["experiment_name"]
        aliases = {
            "ratio_shrink_b0p05_c970_1030": "ratio_shrink_b0p05_c97_103",
        }
        model_names = {exp_name, aliases.get(exp_name, exp_name)}
        if exp_name == "baseline_last_month":
            model_names.add("baseline_last_month")
        match = lb[lb["model_name"].astype(str).isin(model_names)]
        if len(match):
            latest = match.iloc[-1]
            results.loc[idx, "lb_score"] = latest["lb_score"]
            results.loc[idx, "lb_mape"] = latest["lb_mape"]
            results.loc[idx, "lb_verdict"] = latest["verdict"]
    best_score = 95.87
    results["delta_lb_vs_best"] = pd.to_numeric(results["lb_score"], errors="coerce") - best_score
    return results


def add_risk_score(results: pd.DataFrame) -> pd.DataFrame:
    current_best = results.loc[results["experiment_name"] == "ratio_shrink_b0p05_c970_1030"]
    if len(current_best) == 0:
        current_best = results.loc[results["experiment_name"] == "ratio_shrink_b0p05_c97_103"]
    current_best_fold10 = float(current_best.iloc[0]["fold10_mape"])

    aggressive_global_penalty = np.where(
        (results["model_name"] == "last_month_multiplier") & (results["mean_rel_delta_vs_current_best"].abs() > 0.002),
        0.25 + 20.0 * results["mean_rel_delta_vs_current_best"].abs(),
        0.0,
    )
    results["risk_score"] = (
        results["weighted_mape_127"]
        + 0.5 * (results["fold10_mape"] - current_best_fold10).clip(lower=0.0)
        + 20.0 * results["mean_abs_relative_delta_vs_current_best"]
        + 10.0 * results["share_abs_delta_vs_current_best_gt_0p3pct"]
        + aggressive_global_penalty
    )
    return results


def filename_for(exp: Experiment, used: set[str]) -> str:
    if exp.preferred_filename and exp.preferred_filename not in used:
        return exp.preferred_filename
    return f"test_{exp.experiment_name}.csv"


def choose_candidates(results: pd.DataFrame) -> list[str]:
    already_sent = set(results.loc[results["lb_verdict"].astype(str) != "", "experiment_name"])
    pool = results[
        (results["experiment_name"] != "baseline_last_month")
        & (~results["experiment_name"].isin(already_sent))
        & (results["max_rel_delta_vs_current_best"] <= 0.006)
        & (results["share_abs_delta_vs_current_best_gt_1pct"] == 0)
        & (results["mean_abs_relative_delta_vs_current_best"] > 1e-7)
    ].copy()

    priority = [
        "ratio_shrink_b0p03_c990_1010",
        "ratio_shrink_b0p04_c970_1030",
        "ratio_shrink_b0p06_c970_1030",
        "ratio_shrink_b0p05_c990_1010",
        "ratio_shrink_b0p07_c980_1020",
        "best_ratio_blend_baseline_95_05",
        "best_ratio_blend_area_98_02",
        "ratio_gated_smallcorr_t0p01",
        "ratio_gated_stable_t0p05",
        "outlier_rollback_t1p15_b0p95",
        "outlier_smoothing_t0p12_b0p95",
        "ratio_segment_prior_m0p03_s0p01_blend_c99_101",
        "ratio_segment_prior_m0p05_s0p02_area_c98_102",
    ]

    selected = [name for name in priority if name in set(pool["experiment_name"])]
    for model_name in ["ratio_shrink_model", "best_ratio_blend", "ratio_gated", "outlier_rollback", "outlier_smoothing", "ratio_segment_prior"]:
        part = pool[pool["model_name"] == model_name].sort_values(["risk_score", "max_rel_delta_vs_current_best"])
        selected.extend(part.head(2)["experiment_name"].tolist())
    return list(dict.fromkeys(selected))[:8]


def write_reports(results: pd.DataFrame, generated: pd.DataFrame, embedding_note: str) -> None:
    best = load_best_submission()
    lb = load_leaderboard()
    top = results.sort_values("risk_score").head(20)
    rec = generated.sort_values("risk_score") if len(generated) else generated

    experiments_md = [
        "# Отчет по экспериментам",
        "",
        "## Краткий вывод",
        "",
        "Новый лучший подтвержденный сабмит: `submissions/test_ratio_shrink_b0p05_c97_103.csv`, score `95.87`, LB MAPE `4.13`. Улучшение относительно `baseline_last_month` всего `+0.01`, поэтому дальнейший поиск должен быть очень осторожным.",
        "",
        "Глобальные множители ухудшают качество: `0.995` дал 95.85, `0.9975` дал 95.86, `1.0025` дал 95.85, `1.010` дал 95.73, `1.020` дал 95.40. Это говорит, что простое смещение всех прогнозов не работает.",
        "",
        "`ratio_shrink` стал основной линией, потому что он почти не отходит от `baseline_last_month`, но дает слабую индивидуальную поправку и единственный подтвердил улучшение до 95.87.",
        "",
        "`residual_centered_v1` развивать агрессивно не стоит: score 95.79 заметно хуже. Сегментные поправки по region/area/blend/alcohol дали 95.86: они безопасны как микросигнал, но сами по себе не улучшают.",
        "",
        "## Согласование локальной валидации с leaderboard",
        "",
        "Локальная валидация недостаточно хорошо предсказывает LB для глобальных множителей: положительные множители выглядели неплохо на fold10, но ухудшили leaderboard. Поэтому главный критерий теперь: близость к current best + микроскопическое локальное улучшение + отсутствие деградации на fold10.",
        "",
        "Новая формула риска:",
        "",
        "```text",
        "risk_score = weighted_mape_127",
        "             + 0.5 * max(0, fold10_mape - current_best_fold10_mape)",
        "             + 20 * mean_abs_relative_delta_vs_current_best",
        "             + 10 * share_abs_delta_vs_current_best_gt_0p3pct",
        "             + penalty_for_aggressive_global_shift",
        "```",
        "",
        "Эта метрика специально штрафует даже локально неплохие варианты, если они слишком далеко уходят от подтвержденного current best.",
        "",
        "## Категориальные признаки / embeddings",
        "",
        embedding_note,
        "",
        "## Leaderboard-результаты",
        "",
        "```text",
        lb[["filename", "model_name", "lb_score", "lb_mape", "verdict", "comment"]].to_string(index=False),
        "```",
        "",
        "## Топ экспериментов по risk_score",
        "",
        "```text",
        top[[
            "experiment_name",
            "model_name",
            "fold10_mape",
            "weighted_mape_127",
            "risk_score",
            "generated_submission",
            "mean_abs_relative_delta_vs_current_best",
            "max_rel_delta_vs_current_best",
            "share_abs_delta_vs_current_best_gt_0p3pct",
            "lb_score",
            "lb_verdict",
        ]].to_string(index=False),
        "```",
    ]
    (REPORTS_DIR / "experiments.md").write_text("\n".join(experiments_md) + "\n", encoding="utf-8")

    recommended_md = [
        "# Рекомендованные сабмиты",
        "",
        f"Текущий лучший подтвержденный сабмит: `{best['filename']}`, score `{best['lb_score']:.2f}`, LB MAPE `{best['lb_mape']:.2f}`.",
        "",
        "## Новые кандидаты",
        "",
    ]
    if len(rec):
        recommended_md.append("```text")
        recommended_md.append(
            rec[[
                "generated_submission",
                "model_name",
                "weighted_mape_127",
                "risk_score",
                "mean_abs_relative_delta_vs_current_best",
                "max_rel_delta_vs_current_best",
                "share_abs_delta_vs_current_best_gt_0p3pct",
            ]].to_string(index=False)
        )
        recommended_md.append("```")
    else:
        recommended_md.append("Новые кандидаты не созданы: все варианты отфильтрованы как рискованные.")

    recommended_md.extend(
        [
            "",
            "## Что отправлять дальше",
            "",
        ]
    )
    for i, (_, row) in enumerate(rec.head(3).iterrows(), start=1):
        recommended_md.append(f"{i}. `{row['generated_submission']}`")
        recommended_md.append(f"   - идея: {row['comment']}")
        recommended_md.append(
            f"   - среднее абсолютное относительное отличие от current best: {row['mean_abs_relative_delta_vs_current_best']:.6f}; "
            f"максимальное: {row['max_rel_delta_vs_current_best']:.6f}; risk_score: {row['risk_score']:.6f}"
        )
        recommended_md.append("   - риск приемлемый: кандидат остается очень близко к подтвержденному ratio_shrink.")
    recommended_md.extend(
        [
            "",
            "Не стоит повторно отправлять глобальные множители `1.010`, `1.020` и `residual_centered_v1`: leaderboard уже показал ухудшение.",
            "",
            "После каждого результата LB нужно записать его в реестр:",
            "",
            "```bash",
            "python scripts/record_leaderboard_result.py --file submissions/<file>.csv --model <model_name> --lb-score <score> --verdict OK --comment \"комментарий\"",
            "```",
            "",
            "Восстановить текущий лучший `test.csv`:",
            "",
            "```bash",
            "python scripts/restore_best_submission.py",
            "```",
        ]
    )
    (REPORTS_DIR / "recommended_submissions.md").write_text("\n".join(recommended_md) + "\n", encoding="utf-8")


def main() -> None:
    df = load_train(TRAIN_PATH)
    pivot = df.pivot(index=ID_COL, columns=MONTH_COL, values=TARGET_COL).sort_index()
    baseline_test = baseline_pred(pivot, 11)
    current_best_test = current_best_pred(df, pivot, 11)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Восстанавливаем файл, который уже был отправлен, если его нет локально.
    alcohol_c97 = SUBMISSIONS_DIR / "test_segment_alcohol_s0p05_k500_c97_103.csv"
    if not alcohol_c97.exists():
        pred = segment_pred(df, pivot, 11, [ALCOHOL_COL], 0.05, 500, (0.97, 1.03))
        save_submission(submission_frame(pred), alcohol_c97)

    experiments = build_experiments(df, pivot)
    exp_by_name = {exp.experiment_name: exp for exp in experiments}
    results = pd.DataFrame([score_experiment(exp, df, pivot, baseline_test, current_best_test) for exp in experiments])
    results = add_risk_score(enrich_with_leaderboard(results))

    selected = choose_candidates(results)
    generated_rows = []
    used_filenames: set[str] = set()
    for name in selected:
        exp = exp_by_name[name]
        row = results.loc[results["experiment_name"] == name].iloc[0].copy()
        filename = filename_for(exp, used_filenames)
        used_filenames.add(filename)
        path = SUBMISSIONS_DIR / filename
        save_submission(submission_frame(exp.pred_func(11).reindex(current_best_test.index)), path)
        rel_path = str(path.relative_to(ROOT)).replace("\\", "/")
        results.loc[results["experiment_name"] == name, "generated_submission"] = rel_path
        row["generated_submission"] = rel_path
        generated_rows.append(row.to_dict())

    results = add_risk_score(enrich_with_leaderboard(results))
    results.sort_values("risk_score").to_csv(REPORTS_DIR / "experiment_results.csv", index=False, encoding="utf-8")

    generated = pd.DataFrame(generated_rows)
    write_reports(results.sort_values("risk_score"), generated, categorical_svd_probe(df))

    registry = load_leaderboard().copy()
    registry_rows = []
    for row in generated_rows:
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
    print(f"Сохранен отчет: {REPORTS_DIR / 'experiment_results.csv'}")
    print(f"Сохранен отчет: {REPORTS_DIR / 'experiments.md'}")
    print(f"Сохранен отчет: {REPORTS_DIR / 'recommended_submissions.md'}")
    print("Созданные сабмиты:")
    for row in generated_rows:
        print(f"  {row['generated_submission']}")
    print(f"Восстановлен лучший подтвержденный test.csv: {restored}")


if __name__ == "__main__":
    main()
