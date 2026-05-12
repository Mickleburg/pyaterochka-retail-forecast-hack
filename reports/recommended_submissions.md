# Рекомендованные сабмиты

Текущий лучший подтвержденный сабмит: `submissions/test_ratio_shrink_b0p05_c97_103.csv`, score `95.87`, LB MAPE `4.13`.

## Новые кандидаты

```text
                                generated_submission          model_name  weighted_mape_127  risk_score  mean_abs_relative_delta_vs_current_best  max_rel_delta_vs_current_best  share_abs_delta_vs_current_best_gt_0p3pct
    submissions/test_best_ratio_blend_area_98_02.csv    best_ratio_blend           6.778269    6.778498                                 0.000009                       0.000030                                        0.0
submissions/test_best_ratio_blend_baseline_95_05.csv    best_ratio_blend           6.778360    6.778913                                 0.000022                       0.000063                                        0.0
     submissions/test_ratio_shrink_b0p06_c97_103.csv  ratio_shrink_model           6.777418    6.779197                                 0.000089                       0.000253                                        0.0
     submissions/test_ratio_shrink_b0p07_c98_102.csv  ratio_shrink_model           6.776673    6.780231                                 0.000178                       0.000506                                        0.0
     submissions/test_ratio_shrink_b0p04_c97_103.csv  ratio_shrink_model           6.778935    6.781152                                 0.000089                       0.000253                                        0.0
     submissions/test_ratio_shrink_b0p03_c99_101.csv  ratio_shrink_model           6.779713    6.784156                                 0.000178                       0.000506                                        0.0
         submissions/test_ratio_segment_prior_v1.csv ratio_segment_prior           6.779750    6.784206                                 0.000178                       0.000504                                        0.0
          submissions/test_ratio_gated_stable_v1.csv         ratio_gated           6.783271    6.792135                                 0.000324                       0.001265                                        0.0
```

## Что отправлять дальше

1. `submissions/test_best_ratio_blend_area_98_02.csv`
   - идея: Смесь: 98% current best + 2% segment area shrink.
   - среднее абсолютное относительное отличие от current best: 0.000009; максимальное: 0.000030; risk_score: 6.778498
   - риск приемлемый: кандидат остается очень близко к подтвержденному ratio_shrink.
2. `submissions/test_best_ratio_blend_baseline_95_05.csv`
   - идея: Смесь: 95% current best ratio_shrink + 5% baseline_last_month.
   - среднее абсолютное относительное отличие от current best: 0.000022; максимальное: 0.000063; risk_score: 6.778913
   - риск приемлемый: кандидат остается очень близко к подтвержденному ratio_shrink.
3. `submissions/test_ratio_shrink_b0p06_c97_103.csv`
   - идея: Тонкая настройка ratio_shrink: beta=0.06, clip=(0.97, 1.03).
   - среднее абсолютное относительное отличие от current best: 0.000089; максимальное: 0.000253; risk_score: 6.779197
   - риск приемлемый: кандидат остается очень близко к подтвержденному ratio_shrink.

Не стоит повторно отправлять глобальные множители `1.010`, `1.020` и `residual_centered_v1`: leaderboard уже показал ухудшение.

После каждого результата LB нужно записать его в реестр:

```bash
python scripts/record_leaderboard_result.py --file submissions/<file>.csv --model <model_name> --lb-score <score> --verdict OK --comment "комментарий"
```

Восстановить текущий лучший `test.csv`:

```bash
python scripts/restore_best_submission.py
```
