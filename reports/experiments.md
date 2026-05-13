# Отчет по экспериментам

## Краткий вывод

Текущий подтвержденный потолок: `95.91`. `ratio_shrink` дал прирост до `95.88`, но дальше уперся. Новый сигнал leaderboard: траектория магазина и кластеризация по динамике полезнее простых микропоправок.

Current best: `submissions/test_cluster_blend_v1.csv`, score `95.91`. `test.csv` восстановлен как копия этого файла.

## Проверенные семейства

- trajectory clustering v2;
- temporal models v2;
- analog forecasting v2;
- segment-specific temporal;
- MAPE-aware decile strategy;
- regime detection;
- residual correction v2;
- hybrid cluster-temporal.

## Топ-10 safe candidates

```text
                  experiment_name   hypothesis_group candidate_class  fold10_mape  weighted_mape_127  safe_risk_score                            generated_submission  mean_abs_relative_delta_vs_current_best  max_rel_delta_vs_current_best
       temporal_logratio_blend_v2 temporal_models_v2            safe     6.292976           6.732098         6.744010 submissions/test_temporal_logratio_blend_v2.csv                                 0.000851                       0.007423
       decile_cluster_temporal_v1         mape_aware        moderate     6.301982           6.742752         6.767498 submissions/test_decile_cluster_temporal_v1.csv                                 0.001380                       0.012148
hybrid_decile_cluster_temporal_v1             hybrid        moderate     6.301982           6.742752         6.767498                                                                                 0.001380                       0.012148
          temporal_ridge_ratio_v1          confirmed       reference     6.164473           6.640786         6.768092                                                                                 0.002084                       0.028637
               regime_temporal_v1   regime_detection        moderate     6.338564           6.763292         6.768139         submissions/test_regime_temporal_v1.csv                                 0.000346                       0.009232
                temporal_huber_v2 temporal_models_v2        moderate     6.186513           6.657451         6.770586          submissions/test_temporal_huber_v2.csv                                 0.001848                       0.026736
                temporal_ridge_v2 temporal_models_v2        moderate     6.162670           6.639990         6.785455          submissions/test_temporal_ridge_v2.csv                                 0.002144                       0.023961
           temporal_extratrees_v2 temporal_models_v2     exploratory     6.315536           6.762252         6.792092                                                                                 0.001671                       0.012679
                 cluster_blend_v1          confirmed       reference     6.379679           6.794212         6.794212                                                                                 0.000000                       0.000000
            oof_residual_ridge_v2        residual_v2            safe     6.333702           6.782783         6.808431      submissions/test_oof_residual_ridge_v2.csv                                 0.001759                       0.016441
```

## Топ-10 moderate/exploratory candidates

```text
                  experiment_name         hypothesis_group candidate_class  fold10_mape  weighted_mape_127  exploratory_score                            generated_submission  mean_abs_relative_delta_vs_current_best  max_rel_delta_vs_current_best
                temporal_ridge_v2       temporal_models_v2        moderate     6.162670           6.639990           6.645350          submissions/test_temporal_ridge_v2.csv                                 0.002144                       0.023961
          temporal_ridge_ratio_v1                confirmed       reference     6.164473           6.640786           6.645996                                                                                 0.002084                       0.028637
                temporal_huber_v2       temporal_models_v2        moderate     6.186513           6.657451           6.662070          submissions/test_temporal_huber_v2.csv                                 0.001848                       0.026736
        cluster_temporal_blend_v2 trajectory_clustering_v2            safe     6.254825           6.713988           6.720675  submissions/test_cluster_temporal_blend_v2.csv                                 0.002675                       0.022346
       hybrid_cluster_temporal_v1                   hybrid            safe     6.254825           6.713988           6.720675                                                                                 0.002675                       0.022346
       temporal_logratio_blend_v2       temporal_models_v2            safe     6.292976           6.732098           6.734225 submissions/test_temporal_logratio_blend_v2.csv                                 0.000851                       0.007423
       decile_cluster_temporal_v1               mape_aware        moderate     6.301982           6.742752           6.746201 submissions/test_decile_cluster_temporal_v1.csv                                 0.001380                       0.012148
hybrid_decile_cluster_temporal_v1                   hybrid        moderate     6.301982           6.742752           6.746201                                                                                 0.001380                       0.012148
          mape_decile_strategy_v1               mape_aware        moderate     6.309862           6.744676           6.754307                                                                                 0.001669                       0.584355
               regime_temporal_v1         regime_detection        moderate     6.338564           6.763292           6.764157         submissions/test_regime_temporal_v1.csv                                 0.000346                       0.009232
```

## Согласование local CV с LB
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
        experiment_name              model_name  fold10_mape  weighted_mape_127  safe_risk_score  exploratory_score  mean_rel_delta_vs_current_best  max_rel_delta_vs_current_best  lb_score  lb_mape
temporal_ridge_ratio_v1 temporal_ridge_ratio_v1     6.164473           6.640786         6.768092           6.645996                       -0.000488                       0.028637     95.91     4.09
       cluster_blend_v1        cluster_blend_v1     6.379679           6.794212         6.794212           6.794212                        0.000000                       0.000000     95.91     4.09
```
