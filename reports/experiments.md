# Experiment Analysis

Leaderboard already shows that `baseline_last_month` is very strong: the confirmed score is 95.86, which corresponds to LB MAPE 4.14.

The first CatBoost/fallback candidates scored slightly worse than last-month baseline. The likely reason is that the hidden November target is dominated by store-level continuity: the latest observed RTO contains most of the signal, while model-based corrections can overfit fold-specific seasonality or smooth away useful store-level information.

Local CV is useful for rejecting risky changes, but it is not perfectly aligned with the leaderboard. The mean of folds 8/9/10 can be pulled by month 9, while fold 10 is the closest available proxy for November. Weighted folds are therefore tracked separately:

- mean folds 8,9,10: broad stability check;
- fold 10 only: closest chronological proxy;
- weighted 0.2/0.3/0.5: moderate recency bias;
- weighted 0.1/0.2/0.7: strong recency bias.

Conclusion: improvements should be small and controlled. The best candidates should remain very close to `rto_month_10`, with small multiplicative or shrinkage corrections.

Top experiments by weighted_mape_127:

```text
         experiment_name            model_name  fold8_mape  fold9_mape  fold10_mape  mean_mape  weighted_mape_235  weighted_mape_127                          generated_submission  mean_relative_delta_vs_best  max_relative_delta_vs_best                                              comment
     last_month_mult_102 last_month_multiplier    6.532848   10.058549     5.545709   7.379035           7.096989           6.546991      submissions/test_last_month_mult_102.csv                       0.0200                      0.0200                             pred = rto_lag_1 * 1.020
    last_month_mult_1015 last_month_multiplier    6.156635    9.770745     5.711058   7.212813           7.018080           6.567553     submissions/test_last_month_mult_1015.csv                       0.0150                      0.0150                             pred = rto_lag_1 * 1.015
     last_month_mult_101 last_month_multiplier    5.805231    9.503443     5.903976   7.070883           6.964067           6.613995      submissions/test_last_month_mult_101.csv                       0.0100                      0.0100                             pred = rto_lag_1 * 1.010
    damped_trend_add_0p1      damped_trend_add    5.485743    8.751364     6.260846   6.832651           6.852980           6.681439                                                                        NaN                         NaN                      additive damped trend alpha=0.1
    last_month_mult_1005 last_month_multiplier    5.484560    9.257209     6.123618   6.955129           6.935884           6.686430                                                                        NaN                         NaN                             pred = rto_lag_1 * 1.005
  damped_trend_ratio_0p1    damped_trend_ratio    5.530253    8.810772     6.270821   6.870615           6.884693           6.704754                                                                        NaN                         NaN                         ratio damped trend alpha=0.1
   damped_trend_add_0p25      damped_trend_add    6.012057    8.388624     6.363668   6.921450           6.900833           6.733498                                                                        NaN                         NaN                     additive damped trend alpha=0.25
 damped_trend_ratio_0p25    damped_trend_ratio    6.137067    8.524861     6.337851   6.999926           6.953797           6.755174                                                                        NaN                         NaN                        ratio damped trend alpha=0.25
ensemble_conservative_v1 ensemble_conservative    5.252057    9.075280     6.315664   6.881000           6.930827           6.761227 submissions/test_ensemble_conservative_v1.csv                       0.0010                      0.0010            0.95 baseline + 0.05 best local candidate
ensemble_conservative_v2 ensemble_conservative    5.219259    9.049135     6.345576   6.871323           6.931380           6.773656 submissions/test_ensemble_conservative_v2.csv                       0.0004                      0.0004            0.98 baseline + 0.02 best local candidate
     baseline_last_month   baseline_last_month    5.197672    9.031903     6.365661   6.865079           6.931936           6.782111                                                                        NaN                         NaN                              confirmed best baseline
      last_month_mult_10 last_month_multiplier    5.197672    9.031903     6.365661   6.865079           6.931936           6.782111                                                                        NaN                         NaN                             pred = rto_lag_1 * 1.000
    damped_trend_add_0p0      damped_trend_add    5.197672    9.031903     6.365661   6.865079           6.931936           6.782111                                                                        NaN                         NaN                      additive damped trend alpha=0.0
  damped_trend_ratio_0p0    damped_trend_ratio    5.197672    9.031903     6.365661   6.865079           6.931936           6.782111                                                                        NaN                         NaN                         ratio damped trend alpha=0.0
 group_growth_blend_k300    group_growth_blend    5.741245    9.343466     6.204533   7.096415           7.053555           6.785991                                                                        NaN                         NaN blend of region/area/cash group growth, shrink k=300
```
