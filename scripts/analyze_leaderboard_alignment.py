from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import REPORTS_DIR


EXPERIMENT_RESULTS_PATH = REPORTS_DIR / "experiment_results.csv"
LEADERBOARD_RESULTS_PATH = REPORTS_DIR / "leaderboard_results.csv"
ALIGNMENT_REPORT_PATH = REPORTS_DIR / "leaderboard_alignment.md"


def normalize_experiment_name(name: str) -> str:
    aliases = {
        "ratio_shrink_b0p05_c97_103": "ratio_shrink_b0p05_c970_1030",
        "ratio_shrink_b0p06_c97_103": "ratio_shrink_b0p06_c97_103",
        "ratio_shrink_b0p07_c98_102": "ratio_shrink_b0p07_c98_102",
    }
    return aliases.get(name, name)


def build_alignment_frame() -> pd.DataFrame:
    exp = pd.read_csv(EXPERIMENT_RESULTS_PATH)
    if "risk_score" not in exp.columns:
        exp["risk_score"] = exp.get("safe_risk_score")
    lb = pd.read_csv(LEADERBOARD_RESULTS_PATH, encoding="utf-8")
    lb = lb[lb["verdict"].astype(str).eq("OK")].copy()
    lb["lb_mape"] = pd.to_numeric(lb["lb_mape"], errors="coerce")
    lb["experiment_name"] = lb["model_name"].astype(str).map(normalize_experiment_name)
    merged = exp.merge(
        lb[["experiment_name", "filename", "lb_score", "lb_mape", "verdict"]],
        on="experiment_name",
        how="inner",
        suffixes=("", "_leaderboard"),
    )
    return merged.dropna(subset=["lb_mape"])


def corr_or_na(df: pd.DataFrame, col: str) -> float:
    if col not in df or len(df) < 3 or df[col].nunique(dropna=True) < 2 or df["lb_mape"].nunique(dropna=True) < 2:
        return float("nan")
    return float(df[col].corr(df["lb_mape"]))


def main() -> None:
    merged = build_alignment_frame()
    metrics = [
        "mean_mape",
        "fold10_mape",
        "weighted_mape_127",
        "safe_risk_score",
        "exploratory_score",
        "risk_score",
        "mean_rel_delta_vs_current_best",
        "max_rel_delta_vs_current_best",
    ]
    corr_df = pd.DataFrame(
        [{"metric": metric, "corr_with_lb_mape": corr_or_na(merged, metric)} for metric in metrics if metric in merged]
    )
    display_cols = [
        "experiment_name",
        "model_name",
        "fold10_mape",
        "weighted_mape_127",
        "safe_risk_score",
        "exploratory_score",
        "mean_rel_delta_vs_current_best",
        "max_rel_delta_vs_current_best",
        "lb_score",
        "lb_mape",
    ]
    display_cols = [col for col in display_cols if col in merged.columns]

    lines = [
        "# Согласование локальной валидации с leaderboard",
        "",
        f"Число OK-сабмитов, для которых удалось сопоставить локальные метрики и LB: {len(merged)}.",
        "",
        "Корреляции считаются с `lb_mape`, то есть меньше лучше. Данных мало, поэтому это диагностика, а не статистически надежный вывод.",
        "",
        "```text",
        corr_df.to_string(index=False),
        "```",
        "",
        "## Осторожный вывод",
        "",
        "- Локальная валидация недостаточно надежно предсказывает LB для глобальных множителей.",
        "- Положительные глобальные множители были переоценены локально и ухудшили LB.",
        "- Близость к текущему best остается отдельным критерием отбора, но теперь нужен и exploratory-рейтинг, чтобы не получать только копии.",
        "- Основной фильтр: не ухудшать fold10, сохранять понятную гипотезу и контролировать долю магазинов с заметным отклонением.",
        "",
        "## Сопоставленные строки",
        "",
        "```text",
        merged[display_cols].sort_values("lb_mape").to_string(index=False),
        "```",
    ]
    ALIGNMENT_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    experiments_path = REPORTS_DIR / "experiments.md"
    if experiments_path.exists():
        text = experiments_path.read_text(encoding="utf-8")
        marker = "\n## Согласование local CV с LB\n"
        section = marker + "\n".join(lines[2:]) + "\n"
        if marker in text:
            text = text.split(marker)[0].rstrip() + section
        else:
            text = text.rstrip() + "\n" + section
        experiments_path.write_text(text, encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
