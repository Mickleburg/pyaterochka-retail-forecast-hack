# Рекомендованные сабмиты

Current best: `submissions/test_cluster_blend_v1.csv`, score `95.91`. После неудачной отправки восстановить: `python scripts/restore_best_submission.py`.

## Safe candidates

```text
                           generated_submission         hypothesis_group  safe_risk_score  weighted_mape_127  mean_abs_relative_delta_vs_current_best  max_rel_delta_vs_current_best  share_changed_stores
submissions/test_temporal_logratio_blend_v2.csv       temporal_models_v2         6.744010           6.732098                                 0.000851                       0.007423              1.000000
     submissions/test_oof_residual_ridge_v2.csv              residual_v2         6.808431           6.782783                                 0.001759                       0.016441              0.999951
           submissions/test_analog_ratio_v2.csv                analog_v2         6.810888           6.800599                                 0.000735                       0.005734              1.000000
 submissions/test_cluster_temporal_blend_v2.csv trajectory_clustering_v2         6.881485           6.713988                                 0.002675                       0.022346              1.000000
       submissions/test_cluster_path_k20_v2.csv trajectory_clustering_v2         6.896170           6.824470                                 0.001901                       0.012113              1.000000
```

## Moderate candidates

```text
                               generated_submission         hypothesis_group  exploratory_score  weighted_mape_127  mean_abs_relative_delta_vs_current_best  max_rel_delta_vs_current_best  share_changed_stores
             submissions/test_temporal_ridge_v2.csv       temporal_models_v2           6.645350           6.639990                                 0.002144                       0.023961              1.000000
             submissions/test_temporal_huber_v2.csv       temporal_models_v2           6.662070           6.657451                                 0.001848                       0.026736              1.000000
    submissions/test_decile_cluster_temporal_v1.csv               mape_aware           6.746201           6.742752                                 0.001380                       0.012148              1.000000
            submissions/test_regime_temporal_v1.csv         regime_detection           6.764157           6.763292                                 0.000346                       0.009232              0.447296
           submissions/test_analog_segmented_v2.csv                analog_v2           6.791192           6.789471                                 0.000689                       0.020428              0.877759
submissions/test_segment_temporal_volatility_v1.csv         segment_temporal           6.804758           6.792990                                 0.002641                       0.068092              0.800000
submissions/test_segment_temporal_rto_decile_v1.csv         segment_temporal           6.822636           6.820284                                 0.000941                       0.018614              0.699976
            submissions/test_cluster_pca_k20_v2.csv trajectory_clustering_v2           6.859994           6.840501                                 0.002335                       0.020629              1.000000
```

## Exploratory candidates

```text
Empty DataFrame
Columns: [generated_submission, hypothesis_group, exploratory_score, weighted_mape_127, mean_abs_relative_delta_vs_current_best, max_rel_delta_vs_current_best, share_changed_stores]
Index: []
```

## Логика отправки

1. Сначала safe: проверяют новый сигнал без сильного риска.
2. Затем moderate: trajectory/temporal/regime кандидаты, которые реально отличаются от current best.
3. Exploratory отправлять после safe/moderate, если нужен шанс на скачок выше 95.91.

Записать результат:

```bash
python scripts/record_leaderboard_result.py --file submissions/<file>.csv --model <model_name> --lb-score <score> --verdict OK --comment "комментарий"
```
