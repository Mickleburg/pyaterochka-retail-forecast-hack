# Отчет по экспериментам

## Краткий вывод

Новый лучший подтвержденный сабмит: `submissions/test_ratio_shrink_b0p05_c97_103.csv`, score `95.87`, LB MAPE `4.13`. Улучшение относительно `baseline_last_month` всего `+0.01`, поэтому дальнейший поиск должен быть очень осторожным.

Глобальные множители ухудшают качество: `0.995` дал 95.85, `0.9975` дал 95.86, `1.0025` дал 95.85, `1.010` дал 95.73, `1.020` дал 95.40. Это говорит, что простое смещение всех прогнозов не работает.

`ratio_shrink` стал основной линией, потому что он почти не отходит от `baseline_last_month`, но дает слабую индивидуальную поправку и единственный подтвердил улучшение до 95.87.

`residual_centered_v1` развивать агрессивно не стоит: score 95.79 заметно хуже. Сегментные поправки по region/area/blend/alcohol дали 95.86: они безопасны как микросигнал, но сами по себе не улучшают.

## Согласование локальной валидации с leaderboard

Локальная валидация недостаточно хорошо предсказывает LB для глобальных множителей: положительные множители выглядели неплохо на fold10, но ухудшили leaderboard. Поэтому главный критерий теперь: близость к current best + микроскопическое локальное улучшение + отсутствие деградации на fold10.

Новая формула риска:

```text
risk_score = weighted_mape_127
             + 0.5 * max(0, fold10_mape - current_best_fold10_mape)
             + 20 * mean_abs_relative_delta_vs_current_best
             + 10 * share_abs_delta_vs_current_best_gt_0p3pct
             + penalty_for_aggressive_global_shift
```

Эта метрика специально штрафует даже локально неплохие варианты, если они слишком далеко уходят от подтвержденного current best.

## Категориальные признаки / embeddings

В данных есть категориальные признаки: . Свободного текста нет, поэтому Word2Vec/NLP не выглядит оправданным. Для микрокоррекций уже используются сегментные признаки; отдельный embedding-сабмит не генерировался.

## Leaderboard-результаты

```text
                                               filename                         model_name  lb_score  lb_mape verdict                                              comment
    submissions/test_official_mean_october_baseline.csv     official_mean_october_baseline     54.34    45.66      OK               официальный бейзлайн, проверка формата
                     submissions/test_catboost_mape.csv                      catboost_mape     95.80     4.20      OK                    первый кандидат catboost/fallback
                      submissions/test_catboost_log.csv                       catboost_log     95.80     4.20      OK                первый кандидат catboost log/fallback
               submissions/test_baseline_last_month.csv                baseline_last_month     95.86     4.14      OK                текущее лучшее подтвержденное решение
          submissions/test_ensemble_conservative_v1.csv           ensemble_conservative_v1     95.86     4.14      OK           такой же результат, как у текущего лучшего
          submissions/test_ensemble_conservative_v2.csv           ensemble_conservative_v2     95.86     4.14      OK           такой же результат, как у текущего лучшего
               submissions/test_last_month_mult_101.csv                last_month_mult_101     95.73     4.27      OK               множитель 1.010, хуже текущего лучшего
               submissions/test_last_month_mult_102.csv                last_month_mult_102     95.40     4.60      OK               множитель 1.020, хуже текущего лучшего
              submissions/test_last_month_mult_1015.csv               last_month_mult_1015      0.00   100.00      CE    CE в Контесте; локальный формат проверен отдельно
              submissions/test_last_month_mult_0995.csv               last_month_mult_0995     95.85     4.15      OK       множитель 0.995, немного хуже текущего лучшего
             submissions/test_last_month_mult_09975.csv              last_month_mult_09975     95.86     4.14      OK                 множитель 0.9975, на уровне baseline
             submissions/test_last_month_mult_10025.csv              last_month_mult_10025     95.85     4.15      OK      множитель 1.0025, немного хуже текущего лучшего
        submissions/test_ratio_shrink_b0p05_c97_103.csv         ratio_shrink_b0p05_c97_103     95.87     4.13      OK             новый лучший подтвержденный ratio_shrink
              submissions/test_residual_centered_v1.csv               residual_centered_v1     95.79     4.21      OK              centered residual хуже текущего лучшего
submissions/test_segment_alcohol_s0p05_k500_c98_102.csv segment_alcohol_s0p05_k500_c98_102     95.86     4.14      OK сегментная микропоправка alcohol, на уровне baseline
            submissions/test_segment_area_shrink_v1.csv             segment_area_shrink_v1     95.86     4.14      OK    сегментная микропоправка area, на уровне baseline
           submissions/test_segment_blend_shrink_v1.csv            segment_blend_shrink_v1     95.86     4.14      OK   сегментная микропоправка blend, на уровне baseline
          submissions/test_segment_region_shrink_v1.csv           segment_region_shrink_v1     95.86     4.14      OK  сегментная микропоправка region, на уровне baseline
submissions/test_segment_alcohol_s0p05_k500_c97_103.csv segment_alcohol_s0p05_k500_c97_103     95.86     4.14      OK сегментная микропоправка alcohol, на уровне baseline
```

## Топ экспериментов по risk_score

```text
                                experiment_name          model_name  fold10_mape  weighted_mape_127  risk_score generated_submission  mean_abs_relative_delta_vs_current_best  max_rel_delta_vs_current_best  share_abs_delta_vs_current_best_gt_0p3pct  lb_score lb_verdict
                     ratio_shrink_b0p05_c97_103  ratio_shrink_model     6.361144           6.778170    6.778170                                                  0.000000e+00                   0.000000e+00                                        0.0     95.87         OK
                   ratio_shrink_b0p05_c980_1020  ratio_shrink_model     6.361144           6.778170    6.778170                                                  0.000000e+00                   0.000000e+00                                        0.0       NaN           
                   ratio_shrink_b0p05_c995_1004  ratio_shrink_model     6.361144           6.778170    6.778170                                                  0.000000e+00                   0.000000e+00                                        0.0       NaN           
                   ratio_shrink_b0p05_c970_1030  ratio_shrink_model     6.361144           6.778170    6.778170                                                  0.000000e+00                   0.000000e+00                                        0.0     95.87         OK
                   ratio_shrink_b0p05_c985_1014  ratio_shrink_model     6.361144           6.778170    6.778170                                                  0.000000e+00                   0.000000e+00                                        0.0       NaN           
                   ratio_shrink_b0p05_c990_1010  ratio_shrink_model     6.361144           6.778170    6.778170                                                  0.000000e+00                   0.000000e+00                                        0.0       NaN           
                   ratio_gated_smallcorr_t0p015         ratio_gated     6.361144           6.778170    6.778170                                                  0.000000e+00                   0.000000e+00                                        0.0       NaN           
                    ratio_gated_smallcorr_t0p02         ratio_gated     6.361144           6.778170    6.778170                                                  0.000000e+00                   0.000000e+00                                        0.0       NaN           
                   ratio_gated_smallcorr_t0p005         ratio_gated     6.361144           6.778170    6.778170                                                  0.000000e+00                   0.000000e+00                                        0.0       NaN           
                    ratio_gated_smallcorr_t0p01         ratio_gated     6.361144           6.778170    6.778170                                                  0.000000e+00                   0.000000e+00                                        0.0       NaN           
ratio_segment_prior_m0p05_s0p01_alcohol_c97_103 ratio_segment_prior     6.361144           6.778170    6.778172                                                  6.143068e-08                   5.130955e-07                                        0.0       NaN           
ratio_segment_prior_m0p05_s0p01_alcohol_c98_102 ratio_segment_prior     6.361144           6.778170    6.778172                                                  6.143068e-08                   5.130955e-07                                        0.0       NaN           
ratio_segment_prior_m0p05_s0p01_alcohol_c99_101 ratio_segment_prior     6.361144           6.778170    6.778172                                                  6.143068e-08                   5.130955e-07                                        0.0       NaN           
ratio_segment_prior_m0p05_s0p02_alcohol_c99_101 ratio_segment_prior     6.361144           6.778171    6.778173                                                  1.228614e-07                   1.026191e-06                                        0.0       NaN           
ratio_segment_prior_m0p05_s0p02_alcohol_c98_102 ratio_segment_prior     6.361144           6.778171    6.778173                                                  1.228614e-07                   1.026191e-06                                        0.0       NaN           
ratio_segment_prior_m0p05_s0p02_alcohol_c97_103 ratio_segment_prior     6.361144           6.778171    6.778173                                                  1.228614e-07                   1.026191e-06                                        0.0       NaN           
ratio_segment_prior_m0p05_s0p03_alcohol_c98_102 ratio_segment_prior     6.361143           6.778171    6.778175                                                  1.842920e-07                   1.539286e-06                                        0.0       NaN           
ratio_segment_prior_m0p05_s0p03_alcohol_c99_101 ratio_segment_prior     6.361143           6.778171    6.778175                                                  1.842920e-07                   1.539286e-06                                        0.0       NaN           
ratio_segment_prior_m0p05_s0p03_alcohol_c97_103 ratio_segment_prior     6.361143           6.778171    6.778175                                                  1.842920e-07                   1.539286e-06                                        0.0       NaN           
   ratio_segment_prior_m0p05_s0p01_area_c99_101 ratio_segment_prior     6.361149           6.778182    6.778199                                                  7.549000e-07                   3.513457e-06                                        0.0       NaN           
```

## Расчет согласования local CV с LB
Число OK-сабмитов, для которых удалось сопоставить локальные метрики и LB: 13.

Корреляции считаются с `lb_mape`, то есть меньше лучше. Данных пока мало, поэтому выводы нужно считать осторожными.

```text
                        metric  corr_with_lb_mape
                     mean_mape           0.974040
                   fold10_mape          -0.884454
             weighted_mape_127          -0.737096
                    risk_score           0.663310
mean_rel_delta_vs_current_best           0.926057
 max_rel_delta_vs_current_best           0.923964
```

## Осторожный вывод

- Локальная валидация недостаточно надежно предсказывает LB для глобальных множителей.
- Положительные глобальные множители были переоценены локально и ухудшили LB.
- Близость к текущему best стала отдельным критерием отбора.
- Основной фильтр теперь: кандидат должен быть очень близок к `test_ratio_shrink_b0p05_c97_103`, не ухудшать fold10 и не иметь большой доли магазинов с заметным отклонением.

## Сопоставленные строки

```text
                   experiment_name            model_name  fold10_mape  weighted_mape_127  risk_score  mean_rel_delta_vs_current_best  max_rel_delta_vs_current_best  lb_score  lb_mape
      ratio_shrink_b0p05_c970_1030    ratio_shrink_model     6.361144           6.778170    6.778170                        0.000000                       0.000000     95.87     4.13
               baseline_last_month   baseline_last_month     6.365661           6.782111    6.793266                       -0.000282                       0.001265     95.86     4.14
segment_alcohol_s0p05_k500_c98_102        segment_shrink     6.365631           6.782155    6.793304                       -0.000282                       0.001316     95.86     4.14
segment_alcohol_s0p05_k500_c97_103        segment_shrink     6.365631           6.782155    6.793304                       -0.000282                       0.001316     95.86     4.14
            segment_area_shrink_v1        segment_shrink     6.366198           6.783316    6.794857                       -0.000281                       0.001483     95.86     4.14
           segment_blend_shrink_v1  segment_blend_shrink     6.368011           6.785808    6.798351                       -0.000283                       0.001413     95.86     4.14
          segment_region_shrink_v1        segment_shrink     6.370503           6.789414    6.803624                       -0.000285                       0.001581     95.86     4.14
             last_month_mult_09975 last_month_multiplier     6.494262           6.838395   10.438633                       -0.002781                       0.003762     95.86     4.14
             last_month_mult_10025 last_month_multiplier     6.242213           6.731614    7.693173                        0.002218                       0.003747     95.85     4.15
              last_month_mult_0995 last_month_multiplier     6.627229           6.900025   17.494272                       -0.005280                       0.006259     95.85     4.15
              residual_centered_v1     residual_centered     6.474205           6.909653   11.731256                       -0.000877                       0.011100     95.79     4.21
               last_month_mult_101 last_month_multiplier     5.903976           6.613995   17.252622                        0.009716                       0.011256     95.73     4.27
               last_month_mult_102 last_month_multiplier     5.545709           6.546991   17.585505                        0.019713                       0.021269     95.40     4.60
```
