# Рекомендованные сабмиты

Текущий лучший подтвержденный файл: `submissions/test_ratio_shrink_b0p06_c97_103.csv`, score `95.88`. Его можно восстановить командой `python scripts/restore_best_submission.py`.

## Safe candidates

```text
                                 generated_submission    hypothesis_group  safe_risk_score  weighted_mape_127  mean_abs_relative_delta_vs_current_best  max_rel_delta_vs_current_best  share_changed_stores
            submissions/test_decile_beta_ratio_v1.csv             low_rto         6.776698           6.775090                                 0.000089                       0.000738              0.400000
          submissions/test_weighted_oof_global_v1.csv    weighted_mixture         6.891072           6.785762                                 0.000757                       0.040697              1.000000
submissions/test_segment_calibrated_region_rto_v1.csv segment_calibration         7.054844           6.809963                                 0.000981                       0.004360              1.000000
                submissions/test_cluster_blend_v1.csv          clustering         7.261613           6.794212                                 0.000995                       0.009062              1.000000
       submissions/test_selector_by_rto_decile_v1.csv            selector         7.399667           6.752065                                 0.000909                       0.053199              0.499976
     submissions/test_segment_calibrated_trend_v1.csv segment_calibration         7.552839           6.829344                                 0.001945                       0.003275              1.000000
```

## Exploratory candidates

```text
                            generated_submission hypothesis_group  exploratory_score  weighted_mape_127  mean_abs_relative_delta_vs_current_best  max_rel_delta_vs_current_best  share_changed_stores
 submissions/test_temporal_ensemble_ratio_v1.csv   temporal_model           6.728374           6.720591                                 0.001946                       0.029619                   1.0
submissions/test_exploratory_segment_bias_v1.csv      exploratory           6.867223           6.846699                                 0.005131                       0.017414                   1.0
```

## Как отправлять

1. Сначала отправлять safe candidates: они ближе к current best и меньше рискуют потерять качество.
2. Если safe-кандидаты не двигают score, отправлять exploratory candidates: они меняют более адресные группы магазинов и могут сдвинуть плато 95.88.
3. После каждого результата обязательно записывать LB в реестр:

```bash
python scripts/record_leaderboard_result.py --file submissions/<file>.csv --model <model_name> --lb-score <score> --verdict OK --comment "комментарий"
```

4. Если новый файл хуже, восстановить best:

```bash
python scripts/restore_best_submission.py
```
