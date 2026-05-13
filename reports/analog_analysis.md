# Analog forecasting v2

```text
    experiment_name candidate_class  fold10_mape  weighted_mape_127  safe_risk_score  exploratory_score  mean_abs_relative_delta_vs_current_best  max_rel_delta_vs_current_best                     generated_submission                                                      comment
analog_segmented_v2        moderate     6.370195           6.789471         6.813713           6.791192                                 0.000689                       0.020428 submissions/test_analog_segmented_v2.csv              Segmented analog/selector по area x volatility.
    analog_ratio_v2            safe     6.367283           6.800599         6.810888           6.802436                                 0.000735                       0.005734     submissions/test_analog_ratio_v2.csv                           Более мягкий analog по ratio path.
     analog_path_v2        moderate     6.362433           6.807442         6.822556           6.810141                                 0.001080                       0.009479                                          Analog v2 по похожим траекториям, k/bucket proxy, blend 55%.
```

Вывод: сохранялись только кандидаты с понятной гипотезой и контролируемым отклонением от current best; слишком близкие дубли и слишком резкие варианты отфильтрованы.
