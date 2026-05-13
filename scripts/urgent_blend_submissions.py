from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import stage2_experiments as st2
from src.config import ID_COL, MONTH_COL, REPORTS_DIR, SUBMISSIONS_DIR, TARGET_COL, TRAIN_PATH, PROJECT_ROOT
from src.features import load_train
from src.metrics import mape_percent
from src.registry import (
    LEADERBOARD_RESULTS_PATH,
    SUBMISSION_REGISTRY_PATH,
    load_best_submission,
    load_leaderboard,
    restore_best_submission,
    save_best_submission,
)
from src.submit import save_submission, validate_saved_csv


A_FILE = SUBMISSIONS_DIR / "test_cluster_temporal_blend_v2.csv"
KNOWN_FILES = {
    "A_cluster_temporal": A_FILE,
    "B_decile": SUBMISSIONS_DIR / "test_decile_cluster_temporal_v1.csv",
    "C_huber": SUBMISSIONS_DIR / "test_temporal_huber_v2.csv",
    "D_logratio": SUBMISSIONS_DIR / "test_temporal_logratio_blend_v2.csv",
    "E_prevbest": SUBMISSIONS_DIR / "test_cluster_blend_v1.csv",
    "F_old_temporal": SUBMISSIONS_DIR / "test_temporal_ridge_ratio_v1.csv",
    "G_ridge_v2": SUBMISSIONS_DIR / "test_temporal_ridge_v2.csv",
}


@dataclass
class Candidate:
    filename: str
    candidate_type: str
    pred: pd.Series
    comment: str
    priority_hint: int


def update_leaderboard_and_best() -> None:
    rows = [
        ("submissions/test_temporal_ridge_v2.csv", "temporal_ridge_v2", 95.90, "OK", "чистый temporal ridge v2 хуже нового best"),
        ("submissions/test_temporal_huber_v2.csv", "temporal_huber_v2", 95.91, "OK", "temporal huber/proxy на уровне предыдущего best"),
        ("submissions/test_cluster_temporal_blend_v2.csv", "cluster_temporal_blend_v2", 95.93, "OK", "новый best: cluster x temporal blend"),
        ("submissions/test_decile_cluster_temporal_v1.csv", "decile_cluster_temporal_v1", 95.92, "OK", "близкое направление: decile cluster x temporal"),
        ("submissions/test_temporal_logratio_blend_v2.csv", "temporal_logratio_blend_v2", 95.91, "OK", "temporal logratio blend на уровне предыдущего best"),
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
    save_best_submission(
        {
            "filename": "submissions/test_cluster_temporal_blend_v2.csv",
            "model_name": "cluster_temporal_blend_v2",
            "lb_score": 95.93,
            "lb_mape": 4.07,
            "verdict": "OK",
            "is_confirmed_by_leaderboard": True,
        }
    )


def read_submission(path: Path) -> pd.Series:
    validate_saved_csv(path)
    frame = pd.read_csv(path).sort_values(ID_COL)
    return pd.Series(frame["rto"].to_numpy(dtype=float), index=frame[ID_COL].to_numpy(), name=path.name)


def submission_from_series(pred: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({ID_COL: pred.index, "rto": np.round(pred.to_numpy(dtype=float), 2)}).sort_values(ID_COL).reset_index(drop=True)


def deviation_stats(pred: pd.Series, best: pd.Series) -> dict:
    pred = pred.reindex(best.index)
    rel = ((pred - best) / best).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    abs_rel = rel.abs()
    return {
        "mean_abs_delta_vs_A": float((pred - best).abs().mean()),
        "mean_rel_delta_vs_A": float(rel.mean()),
        "mean_abs_rel_delta_vs_A": float(abs_rel.mean()),
        "max_rel_delta_vs_A": float(abs_rel.max()),
        "share_delta_gt_0p1pct": float((abs_rel > 0.001).mean()),
        "share_delta_gt_0p3pct": float((abs_rel > 0.003).mean()),
        "share_delta_gt_1pct": float((abs_rel > 0.010).mean()),
        "corr_with_A": float(pred.corr(best)),
        "share_changed_stores": float((abs_rel > 0).mean()),
    }


def save_candidate(candidate: Candidate, best: pd.Series) -> dict | None:
    path = SUBMISSIONS_DIR / candidate.filename
    pred = candidate.pred.reindex(best.index)
    rounded = np.round(pred.to_numpy(dtype=float), 2)
    a_rounded = np.round(best.to_numpy(dtype=float), 2)
    if np.array_equal(rounded, a_rounded):
        return None
    stats = deviation_stats(pred, best)
    if stats["max_rel_delta_vs_A"] < 0.00001:
        return None
    save_submission(submission_from_series(pred), path)
    validate_saved_csv(path)
    return {
        "filename": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "type": candidate.candidate_type,
        "comment": candidate.comment,
        "priority_hint": candidate.priority_hint,
        **stats,
    }


def fast_blend_candidates(preds: dict[str, pd.Series]) -> list[Candidate]:
    a = preds["A_cluster_temporal"]
    specs = [
        ("test_blend_cluster_temporal_decile_90_10.csv", "blend", 0.90, "B_decile", "90% A + 10% decile_cluster_temporal", 1),
        ("test_blend_cluster_temporal_decile_80_20.csv", "blend", 0.80, "B_decile", "80% A + 20% decile_cluster_temporal", 2),
        ("test_blend_cluster_temporal_decile_70_30.csv", "blend", 0.70, "B_decile", "70% A + 30% decile_cluster_temporal", 3),
        ("test_blend_cluster_temporal_decile_50_50.csv", "blend", 0.50, "B_decile", "50% A + 50% decile_cluster_temporal", 6),
        ("test_blend_cluster_temporal_huber_90_10.csv", "blend", 0.90, "C_huber", "90% A + 10% temporal_huber", 7),
        ("test_blend_cluster_temporal_logratio_90_10.csv", "blend", 0.90, "D_logratio", "90% A + 10% temporal_logratio", 8),
        ("test_blend_cluster_temporal_prevbest_95_05.csv", "blend", 0.95, "E_prevbest", "95% A + 5% previous cluster best", 9),
        ("test_blend_cluster_temporal_prevbest_90_10.csv", "blend", 0.90, "E_prevbest", "90% A + 10% previous cluster best", 10),
        ("test_blend_cluster_temporal_oldtemporal_90_10.csv", "blend", 0.90, "F_old_temporal", "90% A + 10% old temporal ridge", 11),
        ("test_blend_cluster_temporal_oldtemporal_80_20.csv", "blend", 0.80, "F_old_temporal", "80% A + 20% old temporal ridge", 12),
    ]
    out = []
    for filename, typ, aw, other, comment, priority in specs:
        pred = aw * a + (1.0 - aw) * preds[other].reindex(a.index)
        out.append(Candidate(filename, typ, pred, comment, priority))
    return out


def pred_A(df: pd.DataFrame, pivot: pd.DataFrame, month: int) -> pd.Series:
    return st2.hybrid_cluster_temporal_pred(df, pivot, month, "simple")


def pred_B(df: pd.DataFrame, pivot: pd.DataFrame, month: int) -> pd.Series:
    return st2.hybrid_cluster_temporal_pred(df, pivot, month, "decile")


def pred_T_huber(df: pd.DataFrame, pivot: pd.DataFrame, month: int) -> pd.Series:
    return st2.temporal_v2_pred(df, pivot, month, "ridge", 0.30, (0.95, 1.05), 0.55)


def pred_T_ridge(df: pd.DataFrame, pivot: pd.DataFrame, month: int) -> pd.Series:
    return st2.temporal_v2_pred(df, pivot, month, "ridge", 0.24, (0.97, 1.03), 0.80)


def pred_T_logratio(df: pd.DataFrame, pivot: pd.DataFrame, month: int) -> pd.Series:
    cur = st2.current_best_pred(df, pivot, month)
    temp = st2.temporal_v2_pred(df, pivot, month, "ridge", 0.22, (0.98, 1.02), 1.0)
    return 0.65 * cur + 0.35 * temp


def segment_features(df: pd.DataFrame, pivot: pd.DataFrame, month: int) -> pd.DataFrame:
    feats = st2.trajectory_features_v2(df, pivot, month)
    feats["rto_vol"] = feats["rto_decile"].astype(str) + "__" + feats["volatility_bucket"].astype(str)
    return feats


def local_score(pred_func, df: pd.DataFrame, pivot: pd.DataFrame) -> tuple[float, float, float, float]:
    scores = []
    for month in [8, 9, 10]:
        scores.append(mape_percent(pivot[month], pred_func(month)))
    weighted = scores[0] * 0.1 + scores[1] * 0.2 + scores[2] * 0.7
    return scores[0], scores[1], scores[2], weighted


def selector_ab_candidates(df: pd.DataFrame, pivot: pd.DataFrame) -> list[Candidate]:
    rows = []
    for segment in ["rto_decile", "volatility_bucket", "trend_bucket", "regime", "rto_vol"]:
        fold_rows = []
        for month in [8, 9, 10]:
            feats = segment_features(df, pivot, month)
            a = pred_A(df, pivot, month)
            b = pred_B(df, pivot, month)
            y = pivot[month]
            for value, idx in feats.groupby(segment, dropna=False).groups.items():
                if len(idx) < (300 if segment == "rto_vol" else 120):
                    continue
                idx = list(idx)
                a_mape = float(((a.loc[idx] - y.loc[idx]).abs() / y.loc[idx]).mean() * 100)
                b_mape = float(((b.loc[idx] - y.loc[idx]).abs() / y.loc[idx]).mean() * 100)
                fold_rows.append({"segment": segment, "value": value, "month": month, "n": len(idx), "gain_b": a_mape - b_mape})
        if not fold_rows:
            continue
        fr = pd.DataFrame(fold_rows)
        winners = []
        for value, part in fr.groupby("value"):
            if (part["gain_b"] > 0).sum() >= 2 and np.average(part["gain_b"], weights=part["n"]) > 0.005:
                winners.append(value)
        if not winners:
            continue
        for w in [0.50]:
            a_test = pred_A(df, pivot, 11)
            b_test = pred_B(df, pivot, 11)
            feats_test = segment_features(df, pivot, 11)
            selected = a_test.copy()
            mask = feats_test[segment].isin(winners)
            selected.loc[mask] = b_test.loc[mask]
            final = w * selected + (1.0 - w) * a_test
            f8, f9, f10, weighted = local_score(lambda m, segment=segment, winners=winners, w=w: selector_ab_pred(df, pivot, m, segment, winners, w), df, pivot)
            rows.append(
                {
                    "segment": segment,
                    "pred": final,
                    "fold10": f10,
                    "weighted": weighted,
                    "comment": f"A/B selector по {segment}: B применяется в {len(winners)} сегментах, safety weight={w}",
                }
            )
    rows = sorted(rows, key=lambda x: (x["weighted"], x["fold10"]))[:3]
    name_map = {
        "rto_decile": "test_selector_A_B_rto_decile_w50.csv",
        "volatility_bucket": "test_selector_A_B_volatility_w50.csv",
        "trend_bucket": "test_selector_A_B_trend_w50.csv",
        "regime": "test_selector_A_B_regime_w50.csv",
        "rto_vol": "test_selector_A_B_rto_vol_w50.csv",
    }
    return [
        Candidate(name_map[row["segment"]], "selector", row["pred"], f"{row['comment']}; local weighted={row['weighted']:.4f}, fold10={row['fold10']:.4f}", 20 + i)
        for i, row in enumerate(rows, start=1)
    ]


def selector_ab_pred(df: pd.DataFrame, pivot: pd.DataFrame, month: int, segment: str, winners: list, w: float) -> pd.Series:
    a = pred_A(df, pivot, month)
    b = pred_B(df, pivot, month)
    feats = segment_features(df, pivot, month)
    selected = a.copy()
    mask = feats[segment].isin(winners)
    selected.loc[mask] = b.loc[mask]
    return w * selected + (1.0 - w) * a


def temporal_override_candidates(df: pd.DataFrame, pivot: pd.DataFrame) -> list[Candidate]:
    specs = [
        ("trend_bucket", [4], pred_T_huber, "test_temporal_override_trend_v1.csv", "override high trend bucket через temporal_huber"),
        ("regime", ["october_spike"], pred_T_huber, "test_temporal_override_spike_v1.csv", "override october_spike через temporal_huber"),
        ("region_x_rto_decile", ["Москва г__9"], pred_T_ridge, "test_temporal_override_region_rto_v1.csv", "override Москва г x high rto через temporal_ridge"),
    ]
    out = []
    for segment, values, t_func, filename, comment in specs:
        def make_pred(month, segment=segment, values=values, t_func=t_func):
            a = pred_A(df, pivot, month)
            t = t_func(df, pivot, month)
            feats = segment_features(df, pivot, month)
            final = a.copy()
            mask = feats[segment].isin(values)
            final.loc[mask] = 0.50 * a.loc[mask] + 0.50 * t.loc[mask]
            return final
        f8, f9, f10, weighted = local_score(make_pred, df, pivot)
        if f10 <= mape_percent(pivot[10], pred_A(df, pivot, 10)) + 0.08:
            out.append(Candidate(filename, "override", make_pred(11), f"{comment}; local weighted={weighted:.4f}, fold10={f10:.4f}", 30 + len(out)))
    return out[:3]


def v3_candidates(df: pd.DataFrame, pivot: pd.DataFrame) -> list[Candidate]:
    a = pred_A(df, pivot, 11)
    b = pred_B(df, pivot, 11)
    huber = pred_T_huber(df, pivot, 11)
    old = read_submission(KNOWN_FILES["E_prevbest"]).reindex(a.index)
    return [
        Candidate("test_cluster_temporal_blend_v3_more_temporal.csv", "v3", 0.85 * a + 0.15 * huber, "v3: немного больше temporal_huber поверх A", 40),
        Candidate("test_cluster_temporal_blend_v3_less_temporal.csv", "v3", 0.90 * a + 0.10 * old, "v3: чуть меньше temporal, ближе к previous cluster best", 41),
        Candidate("test_cluster_temporal_blend_v3_decile_weighted.csv", "v3", 0.75 * a + 0.25 * b, "v3: decile-weighted blend A/B", 42),
    ]


def urgent_comparison(preds: dict[str, pd.Series], df: pd.DataFrame, pivot: pd.DataFrame) -> pd.DataFrame:
    best = preds["A_cluster_temporal"]
    rows = []
    for name, pred in preds.items():
        stats = deviation_stats(pred.reindex(best.index), best)
        rows.append({"name": name, "filename": str(KNOWN_FILES[name].relative_to(PROJECT_ROOT)).replace("\\", "/"), **stats})
    comp = pd.DataFrame(rows)
    # Сегментное отличие B от A.
    feats = segment_features(df, pivot, 11)
    b = preds["B_decile"].reindex(best.index)
    rel = ((b - best) / best).abs()
    seg_rows = []
    for seg in ["rto_decile", "volatility_bucket", "trend_bucket", "regime"]:
        for value, idx in feats.groupby(seg, dropna=False).groups.items():
            if len(idx) < 100:
                continue
            idx = list(idx)
            seg_rows.append({"segment_type": seg, "segment_value": str(value), "n": len(idx), "mean_abs_rel_B_vs_A": float(rel.loc[idx].mean()), "max_abs_rel_B_vs_A": float(rel.loc[idx].max())})
    seg = pd.DataFrame(seg_rows).sort_values("mean_abs_rel_B_vs_A", ascending=False)
    comp.to_csv(REPORTS_DIR / "urgent_best_comparison.csv", index=False, encoding="utf-8")
    lines = [
        "# Срочное сравнение best-like сабмитов",
        "",
        "База A: `submissions/test_cluster_temporal_blend_v2.csv`, подтвержденный score `95.93`.",
        "",
        "## Отличия файлов от A",
        "",
        "```text",
        comp.to_string(index=False),
        "```",
        "",
        "## Где decile_cluster_temporal сильнее отличается от A",
        "",
        "```text",
        seg.head(30).to_string(index=False),
        "```",
        "",
        "## Вывод",
        "",
        "- `decile_cluster_temporal_v1` близок к A и уже подтвердил `95.92`, поэтому A/B blends и A/B selector имеют самый высокий приоритет.",
        "- Чистый temporal_ridge_v2 получил `95.90`, поэтому temporal нужно применять только как небольшой blend или override на выбранных сегментах.",
        "- Blends с previous best нужны как safety-контроль, но не являются главным направлением.",
    ]
    (REPORTS_DIR / "urgent_best_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return comp


def write_reports(rows: pd.DataFrame) -> None:
    rows = rows.sort_values(["priority", "safe_risk"])
    rows.to_csv(REPORTS_DIR / "urgent_blend_results.csv", index=False, encoding="utf-8")
    short = rows.head(8)
    lines = [
        "# Срочные рекомендации",
        "",
        "Текущий best: `submissions/test_cluster_temporal_blend_v2.csv`, score `95.93`, LB MAPE `4.07`.",
        "",
        "`decile_cluster_temporal_v1` подтвердил близкое направление (`95.92`), поэтому главный приоритет: blends и selectors между A и B. Чистый temporal_ridge_v2 ухудшил LB до `95.90`, поэтому temporal используется только дозированно.",
        "",
        "## Все urgent candidates",
        "",
        "```text",
        rows[["filename", "type", "mean_rel_delta_vs_A", "max_rel_delta_vs_A", "weighted_mape_127", "fold10_mape", "safe_risk", "priority", "comment"]].to_string(index=False),
        "```",
        "",
        "## Short-list в порядке отправки",
        "",
        "```text",
        short[["priority", "filename", "type", "mean_abs_rel_delta_vs_A", "max_rel_delta_vs_A", "safe_risk", "comment"]].to_string(index=False),
        "```",
        "",
        "## Что не развиваем сейчас",
        "",
        "- Чистый `temporal_ridge_v2`: LB `95.90`, хуже best.",
        "- `october_high_rollback`: LB `95.83`, направление вредное.",
        "- Aggressive residual и глобальные множители: ранее ухудшали LB.",
        "- Долгие AutoML/тяжелые модели: времени мало, urgent batch должен быть быстрым.",
    ]
    (REPORTS_DIR / "urgent_recommendations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Кратко обновляем основные отчеты, не перетирая всю историю.
    for path in [REPORTS_DIR / "experiments.md", REPORTS_DIR / "recommended_submissions.md"]:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        marker = "\n## Срочный этап после best 95.93\n"
        section = marker + "\n".join(lines[2:24]) + "\n"
        if marker in text:
            text = text.split(marker)[0].rstrip() + section
        else:
            text = text.rstrip() + "\n" + section
        path.write_text(text, encoding="utf-8")


def add_registry_rows(rows: pd.DataFrame) -> None:
    registry = pd.read_csv(SUBMISSION_REGISTRY_PATH, encoding="utf-8") if SUBMISSION_REGISTRY_PATH.exists() else pd.DataFrame()
    existing = set(registry.get("filename", pd.Series(dtype=str)).astype(str))
    new_rows = []
    for _, row in rows.iterrows():
        if row["filename"] not in existing:
            new_rows.append(
                {
                    "submitted_at": "",
                    "filename": row["filename"],
                    "model_name": Path(row["filename"]).stem.replace("test_", ""),
                    "local_cv_mape": row.get("weighted_mape_127", ""),
                    "lb_score": "",
                    "lb_mape": "",
                    "verdict": "",
                    "comment": row["comment"],
                }
            )
    if new_rows:
        registry = pd.concat([registry, pd.DataFrame(new_rows)], ignore_index=True)
        registry.to_csv(SUBMISSION_REGISTRY_PATH, index=False, encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    update_leaderboard_and_best()

    preds = {name: read_submission(path) for name, path in KNOWN_FILES.items()}
    best = preds["A_cluster_temporal"]

    df = load_train(TRAIN_PATH)
    pivot = df.pivot(index=ID_COL, columns=MONTH_COL, values=TARGET_COL).sort_index()
    urgent_comparison(preds, df, pivot)

    candidates = []
    candidates.extend(fast_blend_candidates(preds))
    candidates.extend(selector_ab_candidates(df, pivot))
    candidates.extend(temporal_override_candidates(df, pivot))
    candidates.extend(v3_candidates(df, pivot))

    keep_order = [
        "test_blend_cluster_temporal_decile_90_10.csv",
        "test_blend_cluster_temporal_decile_80_20.csv",
        "test_blend_cluster_temporal_decile_70_30.csv",
        "test_blend_cluster_temporal_decile_50_50.csv",
        "test_selector_A_B_regime_w50.csv",
        "test_selector_A_B_volatility_w50.csv",
        "test_blend_cluster_temporal_huber_90_10.csv",
        "test_blend_cluster_temporal_logratio_90_10.csv",
        "test_temporal_override_trend_v1.csv",
        "test_cluster_temporal_blend_v3_decile_weighted.csv",
    ]
    order = {name: idx + 1 for idx, name in enumerate(keep_order)}
    candidates = [cand for cand in candidates if cand.filename in order]
    candidates.sort(key=lambda cand: order[cand.filename])

    saved = []
    for cand in candidates:
        row = save_candidate(cand, best)
        if row is None:
            continue
        # Быстрые локальные метрики для кандидата, если можем построить по OOF не всегда нужны.
        row["fold10_mape"] = np.nan
        row["weighted_mape_127"] = np.nan
        row["safe_risk"] = row["mean_abs_rel_delta_vs_A"] + 0.5 * row["share_delta_gt_1pct"] + 0.1 * row["max_rel_delta_vs_A"]
        row["priority"] = order[cand.filename]
        saved.append(row)

    rows = pd.DataFrame(saved)
    if len(rows):
        # Поднимаем A/B blends наверх, затем selector/override/v3.
        rows = rows.sort_values(["priority", "safe_risk"]).reset_index(drop=True)
        rows["priority"] = np.arange(1, len(rows) + 1)
        write_reports(rows)
        add_registry_rows(rows)

    destination = restore_best_submission()
    validate_saved_csv(destination)
    print(f"Обновлен best: {load_best_submission()}")
    print(f"Восстановлен test.csv: {destination}")
    if len(rows):
        print("Созданы urgent candidates:")
        for _, row in rows.iterrows():
            print(f"  {row['filename']}")


if __name__ == "__main__":
    main()
