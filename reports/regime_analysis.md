# Regime detection

```text
   experiment_name candidate_class  fold10_mape  weighted_mape_127  safe_risk_score  exploratory_score  mean_abs_relative_delta_vs_current_best  max_rel_delta_vs_current_best                    generated_submission                                                      comment
regime_temporal_v1        moderate     6.338564           6.763292         6.768139           6.764157                                 0.000346                       0.009232 submissions/test_regime_temporal_v1.csv            Temporal применяется только в стабильных режимах.
regime_selector_v1        moderate     6.687194           6.991881         8.367358           7.099063                                 0.004729                       0.491171                                         Regime detection: stable/growing/declining/volatile/outlier.
   regime_blend_v1            safe     6.724947           7.017861         8.404793           7.137546                                 0.004587                       0.526254                                                                Regime blend: разные веса по режимам.
```

Вывод: сохранялись только кандидаты с понятной гипотезой и контролируемым отклонением от current best; слишком близкие дубли и слишком резкие варианты отфильтрованы.
