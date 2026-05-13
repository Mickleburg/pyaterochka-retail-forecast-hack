from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ID_COL, REPORTS_DIR


LEADERBOARD_RESULTS_PATH = REPORTS_DIR / "leaderboard_results.csv"
REPORT_PATH = REPORTS_DIR / "submission_space_analysis.md"


def load_sent_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    lb = pd.read_csv(LEADERBOARD_RESULTS_PATH, encoding="utf-8")
    rows = []
    preds = []
    for _, row in lb.iterrows():
        path = ROOT / str(row["filename"])
        if not path.exists() or str(row["verdict"]) != "OK":
            continue
        sub = pd.read_csv(path).sort_values(ID_COL).reset_index(drop=True)
        col = Path(row["filename"]).stem
        preds.append(sub[["rto"]].rename(columns={"rto": col}))
        rows.append({**row.to_dict(), "column": col})
    meta = pd.DataFrame(rows)
    matrix = pd.concat(preds, axis=1) if preds else pd.DataFrame()
    return meta, matrix


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    meta, matrix = load_sent_predictions()
    best_col = meta.sort_values("lb_score", ascending=False).iloc[0]["column"]
    best_pred = matrix[best_col]

    rows = []
    for _, row in meta.iterrows():
        col = row["column"]
        diff = matrix[col] - best_pred
        rel = diff / best_pred
        rows.append(
            {
                "filename": row["filename"],
                "model_name": row["model_name"],
                "lb_score": row["lb_score"],
                "mean_rel_delta_vs_best": float(rel.mean()),
                "mean_abs_rel_delta_vs_best": float(rel.abs().mean()),
                "max_abs_rel_delta_vs_best": float(rel.abs().max()),
                "corr_with_best": float(matrix[col].corr(best_pred)),
            }
        )
    summary = pd.DataFrame(rows).sort_values("lb_score", ascending=False)

    corr = matrix.corr()
    near_duplicates = []
    cols = list(matrix.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            rel = ((matrix[left] - matrix[right]) / matrix[right]).abs()
            if rel.max() < 1e-6:
                near_duplicates.append((left, right, float(rel.max())))

    lines = [
        "# Анализ пространства сабмитов",
        "",
        "Этот отчет использует только уже отправленные предсказания и их LB score. Скрытые ответы не восстанавливаются; анализ нужен только для диагностики безопасных направлений.",
        "",
        "## Сравнение с текущим лучшим",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Корреляции предсказаний",
        "",
        "```text",
        corr.round(6).to_string(),
        "```",
        "",
        "## Практически идентичные файлы",
        "",
        "```text",
        pd.DataFrame(near_duplicates, columns=["left", "right", "max_abs_rel_delta"]).to_string(index=False),
        "```",
        "",
        "## Вывод",
        "",
        "- Глобальное увеличение прогнозов явно ухудшало LB: 1.010 и 1.020 дали заметную просадку.",
        "- Микросегментные поправки чаще всего нейтральны: region/area/blend/alcohol держатся на уровне 95.86.",
        "- Прирост дали только очень слабые индивидуальные ratio-поправки (`b0p06`, `b0p07`).",
        "- Усиливать ratio_shrink можно только осторожно; перспективнее искать selector/gating для небольшого подмножества магазинов, чем двигать все прогнозы целиком.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Сохранено: {REPORT_PATH}")


if __name__ == "__main__":
    main()
