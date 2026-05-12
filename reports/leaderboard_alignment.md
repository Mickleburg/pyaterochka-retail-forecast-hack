# Согласование локальной валидации с leaderboard

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
