# Срочные рекомендации

Текущий best: `submissions/test_cluster_temporal_blend_v2.csv`, score `95.93`, LB MAPE `4.07`.

Главный вывод urgent-этапа: развиваем не чистый temporal, а связку `cluster_temporal_blend_v2` + `decile_cluster_temporal_v1`, потому что A дал `95.93`, B дал `95.92`, а чистый `temporal_ridge_v2` снизился до `95.90`.

## Urgent Candidates

```text
 priority filename                                                       type      mean_rel_delta_vs_A max_rel_delta_vs_A safe_risk
 1        submissions/test_blend_cluster_temporal_decile_90_10.csv       blend     0.000098            0.001077           0.000243
 2        submissions/test_blend_cluster_temporal_decile_80_20.csv       blend     0.000195            0.002153           0.000485
 3        submissions/test_blend_cluster_temporal_decile_70_30.csv       blend     0.000293            0.003230           0.000728
 4        submissions/test_blend_cluster_temporal_decile_50_50.csv       blend     0.000488            0.005383           0.001213
 5        submissions/test_selector_A_B_regime_w50.csv                   selector  0.000061            0.005136           0.000622
 6        submissions/test_selector_A_B_volatility_w50.csv               selector  0.000067            0.005221           0.000630
 7        submissions/test_blend_cluster_temporal_huber_90_10.csv        blend     0.000168            0.003004           0.000528
 8        submissions/test_blend_cluster_temporal_logratio_90_10.csv     blend     0.000182            0.001600           0.000385
 9        submissions/test_temporal_override_trend_v1.csv                override  0.000302            0.007877           0.001221
 10       submissions/test_cluster_temporal_blend_v3_decile_weighted.csv v3        0.000244            0.002691           0.000607
```

## Short-list Отправки

1. `submissions/test_blend_cluster_temporal_decile_90_10.csv` — самый осторожный A/B blend.
2. `submissions/test_blend_cluster_temporal_decile_80_20.csv` — чуть сильнее добавляет подтвержденный decile-сигнал.
3. `submissions/test_blend_cluster_temporal_decile_70_30.csv` — умеренная проверка A/B.
4. `submissions/test_selector_A_B_regime_w50.csv` — применяет decile-кандидат только в выбранных режимах.
5. `submissions/test_selector_A_B_volatility_w50.csv` — проверяет сегменты волатильности.
6. `submissions/test_blend_cluster_temporal_huber_90_10.csv` — осторожно добавляет temporal_huber.
7. `submissions/test_blend_cluster_temporal_logratio_90_10.csv` — осторожно добавляет temporal_logratio.
8. `submissions/test_cluster_temporal_blend_v3_decile_weighted.csv` — v3 как A/B blend с весом 25% decile.

## Что Не Развиваем Сейчас

- Чистый `temporal_ridge_v2`: LB `95.90`, хуже best.
- `october_high_rollback`: LB `95.83`, направление вредное.
- Aggressive residual и глобальные множители: ранее ухудшали LB.
- Долгие AutoML/тяжелые модели: времени мало, urgent batch должен быть быстрым.

