# Согласование локальной валидации с leaderboard

Число OK-сабмитов, для которых удалось сопоставить локальные метрики и LB: 2.

Корреляции считаются с `lb_mape`, то есть меньше лучше. Данных мало, поэтому это диагностика, а не статистически надежный вывод.

```text
                        metric  corr_with_lb_mape
                     mean_mape                NaN
                   fold10_mape                NaN
             weighted_mape_127                NaN
               safe_risk_score                NaN
             exploratory_score                NaN
                    risk_score                NaN
mean_rel_delta_vs_current_best                NaN
 max_rel_delta_vs_current_best                NaN
```

## Осторожный вывод

- Локальная валидация недостаточно надежно предсказывает LB для глобальных множителей.
- Положительные глобальные множители были переоценены локально и ухудшили LB.
- Близость к текущему best остается отдельным критерием отбора, но теперь нужен и exploratory-рейтинг, чтобы не получать только копии.
- Основной фильтр: не ухудшать fold10, сохранять понятную гипотезу и контролировать долю магазинов с заметным отклонением.

## Сопоставленные строки

```text
           experiment_name          model_name  fold10_mape  weighted_mape_127  safe_risk_score  exploratory_score  mean_rel_delta_vs_current_best  max_rel_delta_vs_current_best  lb_score  lb_mape
ratio_shrink_b0p06_c97_103        ratio_shrink     6.360338           6.777467         6.777467           6.777467                        0.000000                       0.000000     95.88     4.12
       baseline_last_month baseline_last_month     6.365661           6.782111         6.794908           6.786108                       -0.000338                       0.001517     95.86     4.14
```
