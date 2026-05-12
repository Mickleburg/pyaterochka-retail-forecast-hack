# Рекомендованные сабмиты

Текущий лучший подтвержденный сабмит: `submissions/test_baseline_last_month.csv`, score `95.86`, LB MAPE `4.14`.

## Что уже отправлялось

```text
 submitted_at                                            filename                     model_name  local_cv_mape  lb_score  lb_mape verdict                                           comment
          NaN submissions/test_official_mean_october_baseline.csv official_mean_october_baseline            NaN     54.34    45.66      OK            официальный бейзлайн, проверка формата
          NaN                  submissions/test_catboost_mape.csv                  catboost_mape            NaN     95.80     4.20      OK                 первый кандидат catboost/fallback
          NaN                   submissions/test_catboost_log.csv                   catboost_log            NaN     95.80     4.20      OK             первый кандидат catboost log/fallback
          NaN            submissions/test_baseline_last_month.csv            baseline_last_month            NaN     95.86     4.14      OK             текущее лучшее подтвержденное решение
          NaN       submissions/test_ensemble_conservative_v1.csv       ensemble_conservative_v1            NaN     95.86     4.14      OK        такой же результат, как у текущего лучшего
          NaN       submissions/test_ensemble_conservative_v2.csv       ensemble_conservative_v2            NaN     95.86     4.14      OK        такой же результат, как у текущего лучшего
          NaN            submissions/test_last_month_mult_101.csv            last_month_mult_101            NaN     95.73     4.27      OK            множитель 1.010, хуже текущего лучшего
          NaN            submissions/test_last_month_mult_102.csv            last_month_mult_102            NaN     95.40     4.60      OK            множитель 1.020, хуже текущего лучшего
          NaN           submissions/test_last_month_mult_1015.csv           last_month_mult_1015            NaN      0.00   100.00      CE CE в Контесте; локальный формат проверен отдельно
```

## Новые кандидаты

```text
                           generated_submission            model_name  weighted_mape_127  risk_score  mean_abs_relative_delta_vs_baseline  max_rel_delta_vs_baseline  share_abs_delta_gt_3pct
submissions/test_ratio_shrink_b0p05_c97_103.csv    ratio_shrink_model           6.778170    6.782620                             0.000445                   0.001267                      0.0
    submissions/test_segment_area_shrink_v1.csv        segment_shrink           6.783316    6.784340                             0.000076                   0.000351                      0.0
   submissions/test_segment_blend_shrink_v1.csv  segment_blend_shrink           6.786279    6.788402                             0.000086                   0.000371                      0.0
  submissions/test_segment_region_shrink_v1.csv        segment_shrink           6.789414    6.793432                             0.000160                   0.000500                      0.0
     submissions/test_last_month_mult_09975.csv last_month_multiplier           6.838395    6.927696                             0.002500                   0.002500                      0.0
      submissions/test_residual_centered_v1.csv     residual_centered           6.909653    6.998037                             0.003411                   0.010000                      0.0
     submissions/test_last_month_mult_10025.csv last_month_multiplier           6.731614    7.056614                             0.002500                   0.002500                      0.0
      submissions/test_last_month_mult_0995.csv last_month_multiplier           6.900025    7.080809                             0.005000                   0.005000                      0.0
```

## Что не стоит отправлять

- Уже проверенные положительные множители 1.010, 1.015, 1.020: LB показал ухудшение или CE.
- Агрессивные сегментные и ratio/residual модели, если они меняют много магазинов больше чем на 3%.

## Что отправлять дальше

1. `submissions/test_ratio_shrink_b0p05_c97_103.csv`
   - идея: Модель отношения с сильным shrink к 1; beta=0.05, clip=(0.97, 1.03).
   - среднее абсолютное относительное отличие от baseline: 0.000445; максимальное: 0.001267; risk_score: 6.782620
   - риск приемлемый, потому что кандидат остается очень близко к `baseline_last_month`.
2. `submissions/test_segment_area_shrink_v1.csv`
   - идея: Сегментная поправка по area; shrink=0.05, k=500, clip=(0.97, 1.03).
   - среднее абсолютное относительное отличие от baseline: 0.000076; максимальное: 0.000351; risk_score: 6.784340
   - риск приемлемый, потому что кандидат остается очень близко к `baseline_last_month`.
3. `submissions/test_segment_blend_shrink_v1.csv`
   - идея: Смесь сегментных поправок регион/площадь/кассы; shrink=0.05, k=500, clip=(0.97, 1.03).
   - среднее абсолютное относительное отличие от baseline: 0.000086; максимальное: 0.000371; risk_score: 6.788402
   - риск приемлемый, потому что кандидат остается очень близко к `baseline_last_month`.

После каждого результата LB нужно записать его в реестр:

```bash
python scripts/record_leaderboard_result.py --file submissions/<file>.csv --model <model_name> --lb-score <score> --verdict OK --comment "комментарий"
```

Восстановить текущий лучший `test.csv`:

```bash
python scripts/restore_best_submission.py
```
