# Temporal models v2

```text
           experiment_name candidate_class  fold10_mape  weighted_mape_127  safe_risk_score  exploratory_score  mean_abs_relative_delta_vs_current_best  max_rel_delta_vs_current_best                            generated_submission                                                                comment
         temporal_ridge_v2        moderate     6.162670           6.639990         6.785455           6.645350                                 0.002144                       0.023961          submissions/test_temporal_ridge_v2.csv               Temporal ridge v2: больше beta, но blend с current best.
         temporal_huber_v2        moderate     6.186513           6.657451         6.770586           6.662070                                 0.001848                       0.026736          submissions/test_temporal_huber_v2.csv Robust linear proxy: более широкий temporal сигнал, ослабленный blend.
temporal_logratio_blend_v2            safe     6.292976           6.732098         6.744010           6.734225                                 0.000851                       0.007423 submissions/test_temporal_logratio_blend_v2.csv                          Лог-ratio temporal correction с мягким blend.
    temporal_extratrees_v2     exploratory     6.315536           6.762252         6.792092           6.766429                                 0.001671                       0.012679                                                                                       ExtraTrees по temporal features.
           temporal_hgb_v2     exploratory     6.331913           6.775452         6.835537           6.780907                                 0.002182                       0.024290                                                                         HistGradientBoosting по траекторным признакам.
```

Вывод: сохранялись только кандидаты с понятной гипотезой и контролируемым отклонением от current best; слишком близкие дубли и слишком резкие варианты отфильтрованы.
