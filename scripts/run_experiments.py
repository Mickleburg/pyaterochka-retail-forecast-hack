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
from src.registry import (
    KNOWN_LEADERBOARD_ROWS,
    SUBMISSION_REGISTRY_PATH,
    load_best_submission,
    restore_best_submission,
)
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


def clip_non_negative(values) -> pd.Series:
    return pd.Series(values).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)


def make_submission_frame(pred: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame({ID_COL: pred.index.to_numpy(), "rto": np.round(pred.to_numpy(dtype=float), 2)})
    return out.sort_values(ID_COL).reset_index(drop=True)


def group_growth_ratio(df: pd.DataFrame, month: int, group_cols: list[str], k: int) -> pd.DataFrame:
    hist = df[df[MONTH_COL] < month].sort_values([ID_COL, MONTH_COL]).copy()
    hist["prev_rto"] = hist.groupby(ID_COL)[TARGET_COL].shift(1)
    hist["growth_ratio"] = hist[TARGET_COL] / hist["prev_rto"]
    hist = hist.replace([np.inf, -np.inf], np.nan).dropna(subset=["growth_ratio"])
    hist = hist[(hist["prev_rto"] > 0) & (hist["growth_ratio"].between(0.5, 1.8))]

    global_ratio = float(hist["growth_ratio"].median()) if len(hist) else 1.0
    stats = hist.groupby(group_cols, dropna=False)["growth_ratio"].agg(["median", "count"]).reset_index()
    stats["weight"] = stats["count"] / (stats["count"] + k)
    stats["shrunk_ratio"] = stats["weight"] * stats["median"] + (1.0 - stats["weight"]) * global_ratio
    return stats[group_cols + ["shrunk_ratio"]], global_ratio


def group_growth_pred(
    df: pd.DataFrame,
    pivot: pd.DataFrame,
    month: int,
    group_cols: list[str],
    k: int,
    ratio_clip: tuple[float, float] = (0.85, 1.15),
) -> pd.Series:
    base = pivot[month - 1].copy()
    stats, global_ratio = group_growth_ratio(df, month, group_cols, k)
    current = df[df[MONTH_COL] == month - 1][[ID_COL] + group_cols].copy()
    merged = current.merge(stats, how="left", on=group_cols).set_index(ID_COL)
    ratio = merged["shrunk_ratio"].reindex(base.index).fillna(global_ratio).clip(*ratio_clip)
    return clip_non_negative(base * ratio)


def blend_group_growth_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int, k: int) -> pd.Series:
    region = group_growth_pred(df, pivot, month, [REGION_COL], k)
    area = group_growth_pred(df, pivot, month, [AREA_COL], k)
    cash = group_growth_pred(df, pivot, month, [CASH_COL], k)
    base = pivot[month - 1]
    return clip_non_negative(0.70 * base + 0.15 * region + 0.10 * area + 0.05 * cash)


def ratio_model_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int, clip: tuple[float, float]) -> pd.Series:
    base = pivot[month - 1].copy()
    group = group_growth_pred(df, pivot, month, [REGION_COL, AREA_COL], 50, ratio_clip=(0.80, 1.20))
    group_ratio = (group / base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    recent_ratio = (pivot[month - 1] / pivot[month - 2]).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.80, 1.20)
    ratio = (0.85 * group_ratio + 0.15 * recent_ratio).clip(*clip)
    return clip_non_negative(base * ratio)


def residual_model_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int, clip: tuple[float, float]) -> pd.Series:
    base = pivot[month - 1].copy()
    hist = df[df[MONTH_COL] < month].sort_values([ID_COL, MONTH_COL]).copy()
    hist["prev_rto"] = hist.groupby(ID_COL)[TARGET_COL].shift(1)
    hist["relative_residual"] = hist[TARGET_COL] / hist["prev_rto"] - 1.0
    hist = hist.replace([np.inf, -np.inf], np.nan).dropna(subset=["relative_residual"])
    hist = hist[hist["relative_residual"].between(-0.30, 0.30)]
    global_resid = float(hist["relative_residual"].median()) if len(hist) else 0.0
    stats = hist.groupby(REGION_COL, dropna=False)["relative_residual"].agg(["median", "count"]).reset_index()
    stats["w"] = stats["count"] / (stats["count"] + 100)
    stats["shrunk_resid"] = stats["w"] * stats["median"] + (1.0 - stats["w"]) * global_resid
    current = df[df[MONTH_COL] == month - 1][[ID_COL, REGION_COL]].copy()
    merged = current.merge(stats[[REGION_COL, "shrunk_resid"]], how="left", on=REGION_COL).set_index(ID_COL)
    resid = merged["shrunk_resid"].reindex(base.index).fillna(global_resid).clip(clip[0] - 1.0, clip[1] - 1.0)
    return clip_non_negative(base * (1.0 + resid))


def build_experiments(df: pd.DataFrame, pivot: pd.DataFrame) -> list[Experiment]:
    experiments: list[Experiment] = [
        Experiment("baseline_last_month", "baseline_last_month", lambda month: pivot[month - 1], "confirmed best baseline"),
    ]

    for c in [0.980, 0.985, 0.990, 0.995, 1.000, 1.005, 1.010, 1.015, 1.020]:
        suffix = str(c).replace(".", "")
        experiments.append(
            Experiment(
                f"last_month_mult_{suffix}",
                "last_month_multiplier",
                lambda month, c=c: pivot[month - 1] * c,
                f"pred = rto_lag_1 * {c:.3f}",
            )
        )

    for alpha in [-0.50, -0.25, -0.10, 0.00, 0.10, 0.25, 0.50]:
        suffix = str(alpha).replace("-", "m").replace(".", "p")
        experiments.append(
            Experiment(
                f"damped_trend_add_{suffix}",
                "damped_trend_add",
                lambda month, alpha=alpha: clip_non_negative(
                    pivot[month - 1] + alpha * (pivot[month - 1] - pivot[month - 2])
                ),
                f"additive damped trend alpha={alpha}",
            )
        )
        experiments.append(
            Experiment(
                f"damped_trend_ratio_{suffix}",
                "damped_trend_ratio",
                lambda month, alpha=alpha: clip_non_negative(
                    pivot[month - 1]
                    * (1.0 + alpha * ((pivot[month - 1] / pivot[month - 2]).replace([np.inf, -np.inf], np.nan).fillna(1.0) - 1.0))
                ),
                f"ratio damped trend alpha={alpha}",
            )
        )

    for group_name, group_cols in [
        ("region", [REGION_COL]),
        ("area", [AREA_COL]),
        ("settlement", [SETTLEMENT_COL]),
        ("cash", [CASH_COL]),
        ("alcohol", [ALCOHOL_COL]),
        ("open", [OPEN_COL]),
    ]:
        for k in [10, 30, 50, 100, 300]:
            experiments.append(
                Experiment(
                    f"group_growth_{group_name}_k{k}",
                    "group_growth",
                    lambda month, group_cols=group_cols, k=k: group_growth_pred(df, pivot, month, group_cols, k),
                    f"group median growth by {group_name}, shrink k={k}",
                )
            )

    for k in [10, 30, 50, 100, 300]:
        experiments.append(
            Experiment(
                f"group_growth_blend_k{k}",
                "group_growth_blend",
                lambda month, k=k: blend_group_growth_pred(df, pivot, month, k),
                f"blend of region/area/cash group growth, shrink k={k}",
            )
        )

    for clip in [(0.80, 1.20), (0.90, 1.10), (0.95, 1.05)]:
        suffix = f"{int(clip[0] * 100)}_{int(clip[1] * 100)}"
        experiments.append(
            Experiment(
                f"ratio_model_freq_clip_{suffix}",
                "ratio_model_freq_fallback",
                lambda month, clip=clip: ratio_model_pred(df, pivot, month, clip),
                f"ratio model fallback with clip {clip}",
            )
        )
        experiments.append(
            Experiment(
                f"residual_model_region_clip_{suffix}",
                "residual_model_region_fallback",
                lambda month, clip=clip: residual_model_pred(df, pivot, month, clip),
                f"relative residual by region with clip {clip}",
            )
        )

    return experiments


def score_experiment(exp: Experiment, pivot: pd.DataFrame) -> dict:
    fold_scores = {}
    for month in FOLDS:
        pred = exp.pred_func(month).reindex(pivot.index)
        fold_scores[month] = mape_percent(pivot[month], pred)
    return {
        "experiment_name": exp.experiment_name,
        "model_name": exp.model_name,
        "fold8_mape": fold_scores[8],
        "fold9_mape": fold_scores[9],
        "fold10_mape": fold_scores[10],
        "mean_mape": float(np.mean(list(fold_scores.values()))),
        "weighted_mape_235": sum(fold_scores[m] * WEIGHT_235[m] for m in FOLDS),
        "weighted_mape_127": sum(fold_scores[m] * WEIGHT_127[m] for m in FOLDS),
        "generated_submission": "",
        "mean_relative_delta_vs_best": np.nan,
        "max_relative_delta_vs_best": np.nan,
        "comment": exp.comment,
    }


def choose_candidates(results: pd.DataFrame) -> list[str]:
    baseline = results.loc[results["experiment_name"] == "baseline_last_month"].iloc[0]
    eligible = results[
        (results["experiment_name"] != "baseline_last_month")
        & (results["fold10_mape"] <= baseline["fold10_mape"])
        & (results["weighted_mape_127"] <= baseline["weighted_mape_127"])
    ].copy()

    selected: list[str] = []
    for model_name, limit in [
        ("last_month_multiplier", 3),
        ("damped_trend_add", 1),
        ("damped_trend_ratio", 1),
        ("group_growth", 2),
        ("group_growth_blend", 1),
        ("ratio_model_freq_fallback", 2),
        ("residual_model_region_fallback", 1),
    ]:
        part = eligible[eligible["model_name"] == model_name].sort_values(["weighted_mape_127", "fold10_mape"])
        selected.extend(part.head(limit)["experiment_name"].tolist())

    selected = list(dict.fromkeys(selected))
    return selected


def safe_filename(experiment_name: str) -> str:
    mapping = {
        "ratio_model_freq_clip_80_120": "test_ratio_model_cb_v1.csv",
        "ratio_model_freq_clip_95_105": "test_ratio_model_cb_clipped_v1.csv",
    }
    if experiment_name in mapping:
        return mapping[experiment_name]
    if experiment_name.startswith("last_month_mult_"):
        suffix = experiment_name.replace("last_month_mult_", "")
        return f"test_last_month_mult_{suffix}.csv"
    if experiment_name.startswith("group_growth_region"):
        return "test_group_growth_region_v1.csv"
    if experiment_name.startswith("group_growth_area"):
        return "test_group_growth_area_v1.csv"
    if experiment_name.startswith("group_growth_blend"):
        return "test_group_growth_blend_v1.csv"
    if experiment_name.startswith("damped_trend_"):
        return f"test_{experiment_name}.csv"
    if experiment_name.startswith("residual_model_"):
        return f"test_{experiment_name}.csv"
    return f"test_{experiment_name}.csv"


def write_reports(results: pd.DataFrame, generated: pd.DataFrame, recommendations: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    best = load_best_submission()

    experiment_md = [
        "# Experiment Analysis",
        "",
        "Leaderboard already shows that `baseline_last_month` is very strong: the confirmed score is 95.86, which corresponds to LB MAPE 4.14.",
        "",
        "The first CatBoost/fallback candidates scored slightly worse than last-month baseline. The likely reason is that the hidden November target is dominated by store-level continuity: the latest observed RTO contains most of the signal, while model-based corrections can overfit fold-specific seasonality or smooth away useful store-level information.",
        "",
        "Local CV is useful for rejecting risky changes, but it is not perfectly aligned with the leaderboard. The mean of folds 8/9/10 can be pulled by month 9, while fold 10 is the closest available proxy for November. Weighted folds are therefore tracked separately:",
        "",
        "- mean folds 8,9,10: broad stability check;",
        "- fold 10 only: closest chronological proxy;",
        "- weighted 0.2/0.3/0.5: moderate recency bias;",
        "- weighted 0.1/0.2/0.7: strong recency bias.",
        "",
        "Conclusion: improvements should be small and controlled. The best candidates should remain very close to `rto_month_10`, with small multiplicative or shrinkage corrections.",
        "",
        "Top experiments by weighted_mape_127:",
        "",
        "```text",
        results.sort_values("weighted_mape_127").head(15).to_string(index=False),
        "```",
    ]
    (REPORTS_DIR / "experiments.md").write_text("\n".join(experiment_md) + "\n", encoding="utf-8")

    rec_md = [
        "# Recommended Submissions",
        "",
        f"Current confirmed best: `{best['filename']}` with score `{best['lb_score']:.2f}`.",
        "",
        "New candidates were generated only when they did not look worse than `baseline_last_month` on both fold 10 and weighted_mape_127, except conservative ensemble aliases that are intentionally almost identical to the best baseline.",
        "",
        "Recommended next submissions:",
        "",
    ]
    for idx, row in recommendations.head(3).iterrows():
        rec_md.append(f"{idx + 1}. `{row['generated_submission']}`")
        rec_md.append(
            f"   - weighted_mape_127={row['weighted_mape_127']:.6f}, fold10={row['fold10_mape']:.6f}, "
            f"max_delta_vs_best={row['max_relative_delta_vs_best']:.6f}"
        )
        rec_md.append(f"   - why: {row['comment']}")
    rec_md.extend(
        [
            "",
            "After every new LB result, record it:",
            "",
            "```bash",
            "python scripts/record_leaderboard_result.py --file <submission> --model <model> --lb-score <score> --verdict OK --comment \"<note>\"",
            "```",
            "",
            "If a new candidate is worse, restore the current best:",
            "",
            "```bash",
            "python scripts/restore_best_submission.py",
            "```",
        ]
    )
    (REPORTS_DIR / "recommended_submissions.md").write_text("\n".join(rec_md) + "\n", encoding="utf-8")


def main() -> None:
    df = load_train(TRAIN_PATH)
    pivot = df.pivot(index=ID_COL, columns=MONTH_COL, values=TARGET_COL).sort_index()
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    experiments = build_experiments(df, pivot)
    rows = [score_experiment(exp, pivot) for exp in experiments]
    results = pd.DataFrame(rows)
    baseline = results.loc[results["experiment_name"] == "baseline_last_month"].iloc[0]

    exp_by_name = {exp.experiment_name: exp for exp in experiments}
    selected = choose_candidates(results)

    generated_rows = []
    best_test = pivot[10]

    for name in selected:
        exp = exp_by_name[name]
        test_pred = exp.pred_func(11).reindex(pivot.index)
        rel_delta = ((test_pred - best_test) / best_test).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if rel_delta.abs().max() > 0.25:
            continue
        filename = safe_filename(name)
        path = SUBMISSIONS_DIR / filename
        save_submission(make_submission_frame(test_pred), path)
        idx = results["experiment_name"] == name
        results.loc[idx, "generated_submission"] = str(path.relative_to(ROOT)).replace("\\", "/")
        results.loc[idx, "mean_relative_delta_vs_best"] = float(rel_delta.mean())
        results.loc[idx, "max_relative_delta_vs_best"] = float(rel_delta.abs().max())
        generated_rows.append(results.loc[idx].iloc[0].to_dict())

    generated = pd.DataFrame(generated_rows)

    # Conservative ensemble files are required by the workflow and intentionally stay near the best baseline.
    generated_lookup = {row["experiment_name"]: row for row in generated_rows}
    ratio_source = generated.iloc[0] if not generated.empty else baseline
    ratio_pred = exp_by_name.get(str(ratio_source["experiment_name"]), exp_by_name["baseline_last_month"]).pred_func(11).reindex(pivot.index)
    ensemble_defs = [
        ("ensemble_conservative_v1", "ensemble_conservative", 0.95 * best_test + 0.05 * ratio_pred, "0.95 baseline + 0.05 best local candidate"),
        ("ensemble_conservative_v2", "ensemble_conservative", 0.98 * best_test + 0.02 * ratio_pred, "0.98 baseline + 0.02 best local candidate"),
    ]
    for name, model_name, pred, comment in ensemble_defs:
        rel_delta = ((pred - best_test) / best_test).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        path = SUBMISSIONS_DIR / f"test_{name}.csv"
        save_submission(make_submission_frame(pred), path)
        scores = []
        source_exp = exp_by_name.get(str(ratio_source["experiment_name"]), exp_by_name["baseline_last_month"])
        for month in FOLDS:
            fold_pred = 0.95 * pivot[month - 1] + 0.05 * source_exp.pred_func(month).reindex(pivot.index)
            if name.endswith("v2"):
                fold_pred = 0.98 * pivot[month - 1] + 0.02 * source_exp.pred_func(month).reindex(pivot.index)
            scores.append(mape_percent(pivot[month], fold_pred))
        row = {
            "experiment_name": name,
            "model_name": model_name,
            "fold8_mape": scores[0],
            "fold9_mape": scores[1],
            "fold10_mape": scores[2],
            "mean_mape": float(np.mean(scores)),
            "weighted_mape_235": scores[0] * 0.2 + scores[1] * 0.3 + scores[2] * 0.5,
            "weighted_mape_127": scores[0] * 0.1 + scores[1] * 0.2 + scores[2] * 0.7,
            "generated_submission": str(path.relative_to(ROOT)).replace("\\", "/"),
            "mean_relative_delta_vs_best": float(rel_delta.mean()),
            "max_relative_delta_vs_best": float(rel_delta.abs().max()),
            "comment": comment,
        }
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
        generated_rows.append(row)

    results.sort_values(["weighted_mape_127", "fold10_mape"]).to_csv(REPORTS_DIR / "experiment_results.csv", index=False)

    generated = pd.DataFrame(generated_rows)
    recommendations = generated.sort_values(["weighted_mape_127", "fold10_mape"]).head(5) if not generated.empty else generated
    write_reports(results.sort_values(["weighted_mape_127", "fold10_mape"]), generated, recommendations)

    registry_rows = KNOWN_LEADERBOARD_ROWS.copy()
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
    pd.DataFrame(registry_rows).to_csv(SUBMISSION_REGISTRY_PATH, index=False)

    restored = restore_best_submission()
    print(f"Saved {REPORTS_DIR / 'experiment_results.csv'}")
    print(f"Saved {REPORTS_DIR / 'experiments.md'}")
    print(f"Saved {REPORTS_DIR / 'recommended_submissions.md'}")
    print("Generated submissions:")
    for row in generated_rows:
        print(f"  {row['generated_submission']}")
    print(f"Restored best confirmed test.csv: {restored}")


if __name__ == "__main__":
    main()
