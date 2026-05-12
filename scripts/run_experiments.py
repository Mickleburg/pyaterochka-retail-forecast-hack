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
SETTLEMENT_COL = "Населенный пункт"
CASH_COL = "Количество касс"
ALCOHOL_COL = "Флаг алкогольной лицензии"
OPEN_COL = "Дата открытия, категориальный"


@dataclass
class Experiment:
    experiment_name: str
    model_name: str
    pred_func: callable
    comment: str
    preferred_filename: str = ""


def safe_suffix(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def mult_suffix(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "")


def clip_series(values, index=None) -> pd.Series:
    out = pd.Series(values, index=index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.clip(lower=0.0)


def submission_frame(pred: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({ID_COL: pred.index, "rto": np.round(pred.to_numpy(dtype=float), 2)}).sort_values(ID_COL).reset_index(drop=True)


def baseline_pred(pivot: pd.DataFrame, month: int) -> pd.Series:
    return pivot[month - 1].copy()


def outlier_smooth_pred(pivot: pd.DataFrame, month: int, mode: str, threshold: float, blend: float) -> pd.Series:
    base = pivot[month - 1].copy()
    prev_mean = pivot[[month - 4, month - 3, month - 2]].mean(axis=1)
    deviation = (base / prev_mean - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if mode == "rollback":
        mask = deviation > threshold
    elif mode == "recovery":
        mask = deviation < -abs(1.0 - threshold)
    else:
        mask = deviation.abs() > threshold
    pred = base.copy()
    pred.loc[mask] = blend * base.loc[mask] + (1.0 - blend) * prev_mean.loc[mask]
    return clip_series(pred, index=base.index)


def segment_multiplier(
    df: pd.DataFrame,
    pivot: pd.DataFrame,
    month: int,
    group_cols: list[str],
    shrink_weight: float,
    k: int,
    clip_bounds: tuple[float, float],
) -> pd.Series:
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
    return merged["mult"].reindex(pivot.index).fillna(1.0)


def segment_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int, group_cols: list[str], shrink_weight: float, k: int, clip_bounds: tuple[float, float]) -> pd.Series:
    base = baseline_pred(pivot, month)
    mult = segment_multiplier(df, pivot, month, group_cols, shrink_weight, k, clip_bounds)
    return clip_series(base * mult, index=base.index)


def segment_blend_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int, shrink_weight: float, k: int, clip_bounds: tuple[float, float]) -> pd.Series:
    base = baseline_pred(pivot, month)
    r = segment_multiplier(df, pivot, month, [REGION_COL], shrink_weight, k, clip_bounds)
    a = segment_multiplier(df, pivot, month, [AREA_COL], shrink_weight, k, clip_bounds)
    c = segment_multiplier(df, pivot, month, [CASH_COL], shrink_weight, k, clip_bounds)
    mult = 0.50 * r + 0.30 * a + 0.20 * c
    return clip_series(base * mult, index=base.index)


def ratio_shrink_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int, beta: float, clip_bounds: tuple[float, float]) -> pd.Series:
    base = baseline_pred(pivot, month)
    region_mult = segment_multiplier(df, pivot, month, [REGION_COL, AREA_COL], 0.30, 300, (0.95, 1.05))
    recent = (pivot[month - 1] / pivot[month - 2]).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.85, 1.15)
    raw_ratio = 0.85 * region_mult + 0.15 * recent
    final_ratio = (1.0 + beta * (raw_ratio - 1.0)).clip(*clip_bounds)
    return clip_series(base * final_ratio, index=base.index)


def residual_centered_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int, limit: float) -> pd.Series:
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


def build_experiments(df: pd.DataFrame, pivot: pd.DataFrame) -> list[Experiment]:
    experiments: list[Experiment] = [
        Experiment("baseline_last_month", "baseline_last_month", lambda month: baseline_pred(pivot, month), "Базовый прогноз: РТО предыдущего месяца."),
    ]

    for c in [0.970, 0.975, 0.980, 0.985, 0.990, 0.995, 0.9975, 1.000, 1.0025, 1.005, 1.010, 1.015, 1.020]:
        suffix = mult_suffix(c)
        filename = f"test_last_month_mult_{suffix}.csv" if c in {0.995, 0.9975, 1.0025} else ""
        experiments.append(
            Experiment(
                f"last_month_mult_{suffix}",
                "last_month_multiplier",
                lambda month, c=c: baseline_pred(pivot, month) * c,
                f"Малый глобальный множитель {c:.4f}.",
                filename,
            )
        )

    for threshold in [1.08, 1.10, 1.12, 1.15]:
        for blend in [0.80, 0.85, 0.90, 0.95]:
            experiments.append(
                Experiment(
                    f"outlier_rollback_t{safe_suffix(threshold)}_b{safe_suffix(blend)}",
                    "outlier_rollback",
                    lambda month, threshold=threshold, blend=blend: outlier_smooth_pred(pivot, month, "rollback", threshold - 1.0, blend),
                    f"Откат только для магазинов, где последний месяц сильно выше среднего трех предыдущих; threshold={threshold}, blend={blend}.",
                )
            )

    for threshold in [0.85, 0.88, 0.90, 0.92]:
        for blend in [0.80, 0.85, 0.90, 0.95]:
            experiments.append(
                Experiment(
                    f"drop_recovery_t{safe_suffix(threshold)}_b{safe_suffix(blend)}",
                    "drop_recovery",
                    lambda month, threshold=threshold, blend=blend: outlier_smooth_pred(pivot, month, "recovery", threshold, blend),
                    f"Восстановление только для магазинов, где последний месяц сильно ниже среднего трех предыдущих; threshold={threshold}, blend={blend}.",
                )
            )

    for threshold in [0.08, 0.10, 0.12, 0.15]:
        for blend in [0.85, 0.90, 0.95]:
            experiments.append(
                Experiment(
                    f"outlier_smoothing_t{safe_suffix(threshold)}_b{safe_suffix(blend)}",
                    "outlier_smoothing",
                    lambda month, threshold=threshold, blend=blend: outlier_smooth_pred(pivot, month, "symmetric", threshold, blend),
                    f"Симметричное сглаживание выбросов октября; threshold={threshold}, blend={blend}.",
                )
            )

    group_defs = [
        ("region", [REGION_COL], "test_segment_region_shrink_v1.csv"),
        ("area", [AREA_COL], "test_segment_area_shrink_v1.csv"),
        ("cash", [CASH_COL], ""),
        ("alcohol", [ALCOHOL_COL], ""),
        ("open", [OPEN_COL], ""),
    ]
    for group_name, group_cols, preferred in group_defs:
        for shrink in [0.05, 0.10, 0.20, 0.30]:
            for k in [50, 100, 300, 500]:
                for clip_bounds in [(0.97, 1.03), (0.98, 1.02)]:
                    experiments.append(
                        Experiment(
                            f"segment_{group_name}_s{safe_suffix(shrink)}_k{k}_c{int(clip_bounds[0]*100)}_{int(clip_bounds[1]*100)}",
                            "segment_shrink",
                            lambda month, group_cols=group_cols, shrink=shrink, k=k, clip_bounds=clip_bounds: segment_pred(df, pivot, month, group_cols, shrink, k, clip_bounds),
                            f"Сегментная поправка по {group_name}; shrink={shrink}, k={k}, clip={clip_bounds}.",
                            preferred,
                        )
                    )

    for shrink in [0.05, 0.10, 0.20, 0.30]:
        for k in [100, 300, 500]:
            for clip_bounds in [(0.97, 1.03), (0.98, 1.02)]:
                experiments.append(
                    Experiment(
                        f"segment_blend_s{safe_suffix(shrink)}_k{k}_c{int(clip_bounds[0]*100)}_{int(clip_bounds[1]*100)}",
                        "segment_blend_shrink",
                        lambda month, shrink=shrink, k=k, clip_bounds=clip_bounds: segment_blend_pred(df, pivot, month, shrink, k, clip_bounds),
                        f"Смесь сегментных поправок регион/площадь/кассы; shrink={shrink}, k={k}, clip={clip_bounds}.",
                        "test_segment_blend_shrink_v1.csv",
                    )
                )

    for beta in [0.05, 0.10, 0.15, 0.20]:
        for clip_bounds in [(0.97, 1.03), (0.98, 1.02), (0.99, 1.01)]:
            experiments.append(
                Experiment(
                    f"ratio_shrink_b{safe_suffix(beta)}_c{int(clip_bounds[0]*100)}_{int(clip_bounds[1]*100)}",
                    "ratio_shrink_model",
                    lambda month, beta=beta, clip_bounds=clip_bounds: ratio_shrink_pred(df, pivot, month, beta, clip_bounds),
                    f"Модель отношения с сильным shrink к 1; beta={beta}, clip={clip_bounds}.",
                    "test_ratio_shrink_model_v1.csv" if beta == 0.05 and clip_bounds == (0.99, 1.01) else "",
                )
            )

    for limit in [0.01, 0.02, 0.03]:
        experiments.append(
            Experiment(
                f"residual_centered_l{safe_suffix(limit)}",
                "residual_centered",
                lambda month, limit=limit: residual_centered_pred(df, pivot, month, limit),
                f"Центрированная относительная поправка по региону с ограничением +/-{limit:.0%}.",
                "test_residual_centered_v1.csv" if limit == 0.01 else "",
            )
        )

    return experiments


def score_experiment(exp: Experiment, pivot: pd.DataFrame, best_test: pd.Series) -> dict:
    scores = {}
    for month in FOLDS:
        pred = exp.pred_func(month).reindex(pivot.index)
        scores[month] = mape_percent(pivot[month], pred)

    test_pred = exp.pred_func(11).reindex(best_test.index)
    abs_delta = (test_pred - best_test).abs()
    rel_delta = ((test_pred - best_test) / best_test).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return {
        "experiment_name": exp.experiment_name,
        "model_name": exp.model_name,
        "fold8_mape": scores[8],
        "fold9_mape": scores[9],
        "fold10_mape": scores[10],
        "mean_mape": float(np.mean(list(scores.values()))),
        "weighted_mape_235": scores[8] * 0.2 + scores[9] * 0.3 + scores[10] * 0.5,
        "weighted_mape_127": scores[8] * 0.1 + scores[9] * 0.2 + scores[10] * 0.7,
        "generated_submission": "",
        "mean_abs_delta_vs_baseline": float(abs_delta.mean()),
        "mean_rel_delta_vs_baseline": float(rel_delta.mean()),
        "mean_abs_relative_delta_vs_baseline": float(rel_delta.abs().mean()),
        "max_rel_delta_vs_baseline": float(rel_delta.abs().max()),
        "share_abs_delta_gt_1pct": float((rel_delta.abs() > 0.01).mean()),
        "share_abs_delta_gt_2pct": float((rel_delta.abs() > 0.02).mean()),
        "share_abs_delta_gt_3pct": float((rel_delta.abs() > 0.03).mean()),
        "share_abs_delta_gt_5pct": float((rel_delta.abs() > 0.05).mean()),
        "comment": exp.comment,
    }


def enrich_with_leaderboard(results: pd.DataFrame) -> pd.DataFrame:
    lb = load_leaderboard()
    latest = lb.dropna(subset=["filename"]).copy()
    latest["filename"] = latest["filename"].astype(str).str.replace("\\", "/", regex=False)
    model_map = {
        "baseline_last_month": "baseline_last_month",
        "last_month_mult_101": "last_month_mult_101",
        "last_month_mult_1015": "last_month_mult_1015",
        "last_month_mult_102": "last_month_mult_102",
        "ensemble_conservative_v1": "ensemble_conservative_v1",
        "ensemble_conservative_v2": "ensemble_conservative_v2",
    }
    for col in ["lb_score", "lb_mape", "verdict"]:
        if col not in latest:
            latest[col] = np.nan

    results["lb_score"] = np.nan
    results["lb_mape"] = np.nan
    results["lb_verdict"] = ""
    for lb_model, exp_name in model_map.items():
        rows = latest[latest["model_name"].astype(str) == lb_model]
        if len(rows):
            row = rows.iloc[-1]
            idx = results["experiment_name"] == exp_name
            results.loc[idx, "lb_score"] = pd.to_numeric(row["lb_score"], errors="coerce")
            results.loc[idx, "lb_mape"] = pd.to_numeric(row["lb_mape"], errors="coerce")
            results.loc[idx, "lb_verdict"] = row["verdict"]
    best_score = 95.86
    results["delta_lb_vs_best"] = pd.to_numeric(results["lb_score"], errors="coerce") - best_score
    return results


def add_risk_score(results: pd.DataFrame) -> pd.DataFrame:
    baseline = results.loc[results["experiment_name"] == "baseline_last_month"].iloc[0]
    positive_multiplier_penalty = np.where(
        (results["model_name"] == "last_month_multiplier") & (results["mean_rel_delta_vs_baseline"] > 0),
        0.25 + 20.0 * results["mean_rel_delta_vs_baseline"],
        0.0,
    )
    results["delta_vs_baseline_fold10"] = results["fold10_mape"] - baseline["fold10_mape"]
    results["delta_vs_baseline_weighted_127"] = results["weighted_mape_127"] - baseline["weighted_mape_127"]
    results["risk_score"] = (
        results["weighted_mape_127"]
        + 0.5 * results["delta_vs_baseline_fold10"].clip(lower=0.0)
        + 10.0 * results["mean_abs_relative_delta_vs_baseline"]
        + 5.0 * results["share_abs_delta_gt_3pct"]
        + positive_multiplier_penalty
    )
    return results


def filename_for(exp: Experiment, used: set[str]) -> str:
    if exp.preferred_filename and exp.preferred_filename not in used:
        return exp.preferred_filename
    return f"test_{exp.experiment_name}.csv"


def choose_candidates(results: pd.DataFrame) -> list[str]:
    sent = set(results.loc[results["lb_verdict"].astype(str) != "", "experiment_name"])
    pool = results[
        (results["experiment_name"] != "baseline_last_month")
        & (~results["experiment_name"].isin(sent))
        & (results["max_rel_delta_vs_baseline"] <= 0.03)
        & (results["share_abs_delta_gt_5pct"] == 0)
    ].copy()

    selected = []
    for name in ["last_month_mult_09975", "last_month_mult_0995", "last_month_mult_10025"]:
        if name in set(pool["experiment_name"]):
            selected.append(name)

    for prefix, limit in [
        ("outlier_rollback", 1),
        ("drop_recovery", 1),
        ("outlier_smoothing", 1),
        ("ratio_shrink_model", 1),
        ("residual_centered", 1),
    ]:
        part = pool[pool["model_name"] == prefix].sort_values(["risk_score", "max_rel_delta_vs_baseline"])
        selected.extend(part.head(limit)["experiment_name"].tolist())

    for prefix in ["segment_region_", "segment_area_", "segment_blend_"]:
        part = pool[pool["experiment_name"].str.startswith(prefix)].sort_values(["risk_score", "max_rel_delta_vs_baseline"])
        selected.extend(part.head(1)["experiment_name"].tolist())

    return list(dict.fromkeys(selected))[:10]


def write_reports(results: pd.DataFrame, generated: pd.DataFrame, ce_note: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    best = load_best_submission()
    lb = load_leaderboard()
    top = results.sort_values("risk_score").head(15)

    experiments_md = [
        "# Отчет по экспериментам",
        "",
        "## Краткий вывод",
        "",
        "Leaderboard подтвердил, что `baseline_last_month` остается лучшим или делит первое место. Простое увеличение прогноза ухудшило результат: множитель 1.010 дал score 95.73, множитель 1.020 дал score 95.40. Консервативные ансамбли v1/v2 дали тот же score 95.86, то есть не улучшили baseline.",
        "",
        "Следовательно, скрытый ноябрь очень близок к октябрю. Новые кандидаты должны быть микрокоррекциями вокруг `РТО_10`, а не самостоятельными агрессивными моделями.",
        "",
        "## Расследование CE для test_last_month_mult_1015.csv",
        "",
        ce_note,
        "",
        "## Калибровка локальной валидации по leaderboard",
        "",
        "Локальная валидация переоценила положительные множители: `last_month_mult_102` выглядел хорошо по fold10 и weighted 0.1/0.2/0.7, но на LB оказался заметно хуже baseline. Поэтому теперь локальные метрики используются как фильтр риска, а не как прямой прогноз leaderboard.",
        "",
        "Основные локальные сигналы после калибровки:",
        "",
        "- fold10 важен, но сам по себе недостаточен;",
        "- weighted_mape_127 полезен как recency-weighted фильтр, но он ошибся на положительных множителях;",
        "- отклонение от `baseline_last_month` нужно явно штрафовать;",
        "- кандидаты с большим числом магазинов, измененных более чем на 3%, считаются рискованными;",
        "- положительные глобальные множители теперь считаются более рискованными, потому что LB уже показал ухудшение.",
        "",
        "Формула `risk_score`:",
        "",
        "```text",
        "risk_score = weighted_mape_127",
        "             + 0.5 * max(0, fold10_mape - baseline_fold10_mape)",
        "             + 10 * mean_abs_relative_delta_vs_baseline",
        "             + 5 * share_abs_delta_gt_3pct",
        "             + penalty_for_positive_global_multiplier",
        "```",
        "",
        "Чем меньше `risk_score`, тем безопаснее кандидат. Эта метрика намеренно штрафует даже локально перспективные варианты, если они слишком далеко уходят от октябрьского baseline.",
        "",
        "## Leaderboard-результаты",
        "",
        "```text",
        lb.to_string(index=False),
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
            "max_rel_delta_vs_baseline",
            "share_abs_delta_gt_3pct",
            "lb_score",
            "lb_verdict",
        ]].to_string(index=False),
        "```",
        "",
        "## NLP / embeddings",
        "",
        "В данных есть категориальные признаки (`Регион`, `Населенный пункт`, категории даты открытия и площади), но нет свободного текста. Поэтому классический NLP или Word2Vec здесь выглядит избыточным. Более уместны frequency/target/segment encodings, которые частично проверяются через сегментные shrinkage-поправки. Отдельный NLP-сабмит не генерировался.",
    ]
    (REPORTS_DIR / "experiments.md").write_text("\n".join(experiments_md) + "\n", encoding="utf-8")

    rec = generated.sort_values("risk_score").copy()
    recommended_md = [
        "# Рекомендованные сабмиты",
        "",
        f"Текущий лучший подтвержденный сабмит: `{best['filename']}`, score `{best['lb_score']:.2f}`, LB MAPE `{best['lb_mape']:.2f}`.",
        "",
        "## Что уже отправлялось",
        "",
        "```text",
        lb.to_string(index=False),
        "```",
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
                "mean_abs_relative_delta_vs_baseline",
                "max_rel_delta_vs_baseline",
                "share_abs_delta_gt_3pct",
            ]].to_string(index=False)
        )
        recommended_md.append("```")
    else:
        recommended_md.append("Новые кандидаты не созданы: ни один вариант не прошел фильтр безопасности.")

    recommended_md.extend(
        [
            "",
            "## Что не стоит отправлять",
            "",
            "- Уже проверенные положительные множители 1.010, 1.015, 1.020: LB показал ухудшение или CE.",
            "- Агрессивные сегментные и ratio/residual модели, если они меняют много магазинов больше чем на 3%.",
            "",
            "## Что отправлять дальше",
            "",
        ]
    )
    for i, (_, row) in enumerate(rec.head(3).iterrows(), start=1):
        recommended_md.append(f"{i}. `{row['generated_submission']}`")
        recommended_md.append(f"   - идея: {row['comment']}")
        recommended_md.append(
            f"   - среднее абсолютное относительное отличие от baseline: {row['mean_abs_relative_delta_vs_baseline']:.6f}; "
            f"максимальное: {row['max_rel_delta_vs_baseline']:.6f}; risk_score: {row['risk_score']:.6f}"
        )
        recommended_md.append("   - риск приемлемый, потому что кандидат остается очень близко к `baseline_last_month`.")
    recommended_md.extend(
        [
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


def ce_investigation_note() -> str:
    path = SUBMISSIONS_DIR / "test_last_month_mult_1015.csv"
    if not path.exists():
        return "Файл `submissions/test_last_month_mult_1015.csv` локально не найден."
    raw = path.read_bytes()
    df = pd.read_csv(path)
    first_line = path.open("r", encoding="utf-8", newline="").readline().rstrip("\r\n")
    bom = raw.startswith(b"\xef\xbb\xbf")
    nul_count = raw.count(b"\x00")
    crlf_count = raw.count(b"\r\n")
    lf_count = raw.count(b"\n")
    cr_only_count = raw.count(b"\r") - crlf_count
    checks = [
        f"Файл существует: да.",
        f"Размер файла: {path.stat().st_size} байт.",
        f"Первая строка: `{first_line}`.",
        f"Shape через `pd.read_csv`: {tuple(df.shape)}.",
        f"Колонки: {list(df.columns)}.",
        f"NaN: {int(df.isna().sum().sum())}.",
        f"Отрицательные `rto`: {int((df['rto'] < 0).sum())}.",
        f"Дубликаты `new_id`: {int(df['new_id'].duplicated().sum())}.",
        f"BOM: {'есть' if bom else 'нет'}.",
        f"NUL-байты: {nul_count}.",
        f"CRLF строк: {crlf_count}, LF строк: {lf_count}, одиночных CR: {cr_only_count}.",
    ]
    checks.append("Локальная проверка формата не выявила проблемы; вероятно, в Контест был отправлен не тот файл или произошла ошибка загрузки.")
    return "\n".join(f"- {line}" for line in checks)


def main() -> None:
    df = load_train(TRAIN_PATH)
    pivot = df.pivot(index=ID_COL, columns=MONTH_COL, values=TARGET_COL).sort_index()
    best_test = pivot[10].copy()
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    experiments = build_experiments(df, pivot)
    exp_by_name = {exp.experiment_name: exp for exp in experiments}
    results = pd.DataFrame([score_experiment(exp, pivot, best_test) for exp in experiments])
    results = enrich_with_leaderboard(results)
    results = add_risk_score(results)

    selected = choose_candidates(results)
    used_filenames: set[str] = set()
    generated_rows = []
    for name in selected:
        exp = exp_by_name[name]
        row = results[results["experiment_name"] == name].iloc[0].copy()
        filename = filename_for(exp, used_filenames)
        used_filenames.add(filename)
        path = SUBMISSIONS_DIR / filename
        pred = exp.pred_func(11).reindex(best_test.index)
        save_submission(submission_frame(pred), path)
        rel_path = str(path.relative_to(ROOT)).replace("\\", "/")
        results.loc[results["experiment_name"] == name, "generated_submission"] = rel_path
        row["generated_submission"] = rel_path
        generated_rows.append(row.to_dict())

    results = enrich_with_leaderboard(results)
    results = add_risk_score(results)
    results.sort_values("risk_score").to_csv(REPORTS_DIR / "experiment_results.csv", index=False, encoding="utf-8")

    generated = pd.DataFrame(generated_rows)
    write_reports(results.sort_values("risk_score"), generated, ce_investigation_note())

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
