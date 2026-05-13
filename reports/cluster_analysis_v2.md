# Кластеризация траекторий v2

```text
          experiment_name candidate_class  fold10_mape  weighted_mape_127  safe_risk_score  exploratory_score  mean_abs_relative_delta_vs_current_best  max_rel_delta_vs_current_best                           generated_submission                                                                  comment
cluster_temporal_blend_v2            safe     6.254825           6.713988         6.881485           6.720675                                 0.002675                       0.022346 submissions/test_cluster_temporal_blend_v2.csv     Равная смесь cluster v2 и temporal v2 с safety blend к current best.
     cluster_ratio_k30_v2        moderate     6.342043           6.786442         8.042901           6.797919                                 0.004591                       0.026211                                                                    Кластерная поправка k=30, больше вес cluster signal.
      cluster_selector_v2        moderate     6.374657           6.792990         7.426226           6.804758                                 0.002641                       0.068092                                                Selector по volatility bucket выбирает current/cluster/temporal/rolling.
      cluster_path_k20_v2            safe     6.413489           6.824470         6.896170           6.837675                                 0.001901                       0.012113       submissions/test_cluster_path_k20_v2.csv       Кластерная поправка по траектории k=20, усиленная относительно v1.
       cluster_pca_k20_v2        moderate     6.434307           6.840501         6.928682           6.859994                                 0.002335                       0.020629        submissions/test_cluster_pca_k20_v2.csv               PCA-like proxy: более широкий clip для cluster trajectory.
```

Вывод: сохранялись только кандидаты с понятной гипотезой и контролируемым отклонением от current best; слишком близкие дубли и слишком резкие варианты отфильтрованы.
