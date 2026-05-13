# Hybrid cluster-temporal

```text
                  experiment_name candidate_class  fold10_mape  weighted_mape_127  safe_risk_score  exploratory_score  mean_abs_relative_delta_vs_current_best  max_rel_delta_vs_current_best generated_submission                                            comment
       hybrid_cluster_temporal_v1            safe     6.254825           6.713988         6.881485           6.720675                                 0.002675                       0.022346                                       Hybrid: cluster v2 x temporal v2.
hybrid_decile_cluster_temporal_v1        moderate     6.301982           6.742752         6.767498           6.746201                                 0.001380                       0.012148                       Hybrid: decile-specific cluster/temporal weights.
hybrid_regime_cluster_temporal_v1        moderate     6.632466           6.955732         8.045359           7.040983                                 0.004019                       0.417495                      Hybrid: regime selector выбирает cluster/temporal.
```

Вывод: сохранялись только кандидаты с понятной гипотезой и контролируемым отклонением от current best; слишком близкие дубли и слишком резкие варианты отфильтрованы.
