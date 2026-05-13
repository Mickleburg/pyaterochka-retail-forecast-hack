# Отчет по экспериментам

## Краткий вывод

Текущий подтвержденный потолок: `95.88`. Простые микропоправки вокруг `ratio_shrink` почти исчерпаны: большинство близких blend/selector/mixture кандидатов дают 95.87-95.88. Дальше работаем через поиск систематических ошибок и адресные corrections.

Текущий best: `submissions/test_ratio_shrink_b0p06_c97_103.csv`, score `95.88`, LB MAPE `4.12`. `test.csv` должен оставаться копией этого файла.

## Что пробовали на этом этапе

- `segment calibration`: множитель к current_best по сегментному bias с сильным shrink к 1.
- `selector/gating`: выбор альтернативной модели внутри сегмента, затем safety blend с current_best.
- `weighted mixture`: OOF-оптимизация весов глобально и по bucket'ам.
- `temporal pattern model`: Ridge/Huber и HGB по форме временной траектории магазина.
- `clustering`: KMeans по нормализованным траекториям и ratio-признакам.
- `outlier-specific correction`: отдельная обработка магазинов, где октябрь похож на аномалию.
- `low-RTO MAPE-aware`: отдельная логика для нижних decile РТО, где MAPE особенно чувствителен.

## Новая логика риска

Теперь есть два рейтинга. `safe_risk_score` сильно штрафует отклонение от current_best. `exploratory_score` мягче относится к отличиям и нужен, чтобы не получать только почти идентичные копии.

```text
safe_risk_score = weighted_mape_127 + штраф за fold10 + сильный штраф за отклонения >0.3% и >1%
exploratory_score = weighted_mape_127 + меньший штраф за отклонение + контроль fold10 и доли изменений >3%
```

## Топ по safe_risk_score

```text
                             experiment_name    hypothesis_group  fold10_mape  weighted_mape_127  safe_risk_score  exploratory_score                                  generated_submission  mean_abs_relative_delta_vs_current_best  max_rel_delta_vs_current_best  share_abs_delta_vs_current_best_gt_1pct
                        decile_beta_ratio_v1             low_rto     6.356880           6.775090         6.776698           6.775448             submissions/test_decile_beta_ratio_v1.csv                                 0.000089                       0.000738                                 0.000000
                  ratio_shrink_b0p06_c97_103                sent     6.360338           6.777467         6.777467           6.777467                                                                                       0.000000                       0.000000                                 0.000000
                         baseline_last_month                sent     6.365661           6.782111         6.794908           6.786108                                                                                       0.000534                       0.001517                                 0.000000
                      weighted_oof_global_v1    weighted_mixture     6.360338           6.785762         6.891072           6.788887           submissions/test_weighted_oof_global_v1.csv                                 0.000757                       0.040697                                 0.000437
 segment_calibrated_region_rto_s0p20_c98_102 segment_calibration     6.368667           6.809963         7.054844           6.816804 submissions/test_segment_calibrated_region_rto_v1.csv                                 0.000981                       0.004360                                 0.000000
                            cluster_blend_v1          clustering     6.379679           6.794212         7.261613           6.804962                 submissions/test_cluster_blend_v1.csv                                 0.000995                       0.009062                                 0.000000
                selector_by_rto_decile_w0p25            selector     6.314717           6.752065         7.399667           6.757059        submissions/test_selector_by_rto_decile_v1.csv                                 0.000909                       0.053199                                 0.006452
                   selector_by_outlier_w0p25            selector     6.327261           6.802649         7.484403           6.809994                                                                                       0.001060                       0.118251                                 0.006888
      segment_calibrated_trend_s0p20_c98_102 segment_calibration     6.362533           6.829344         7.552839           6.837892      submissions/test_segment_calibrated_trend_v1.csv                                 0.001945                       0.003275                                 0.000000
                selector_by_region_rto_w0p25            selector     6.317084           6.752798         7.646126           6.759524                                                                                       0.001124                       0.171964                                 0.013873
 segment_calibrated_rto_decile_s0p20_c98_102 segment_calibration     6.360660           6.813573         7.743150           6.820167 submissions/test_segment_calibrated_rto_decile_v1.csv                                 0.001620                       0.003649                                 0.000000
                selector_by_volatility_w0p25            selector     6.310956           6.765501         7.941404           6.772747                                                                                       0.001036                       0.118251                                 0.015086
                  weighted_oof_rto_decile_v1    weighted_mixture     6.352694           6.779513         8.094447           6.792022                                                                                       0.001478                       0.324206                                 0.021392
               october_high_rollback_safe_v1    outlier_specific     6.381627           6.771780         8.128858           6.785719         submissions/test_october_high_rollback_v1.csv                                 0.001549                       0.041416                                 0.024739
   segment_calibrated_area_vol_s0p20_c98_102 segment_calibration     6.349931           6.811649         8.180269           6.818916                                                                                       0.001817                       0.003420                                 0.000000
                  temporal_ensemble_ratio_v1      temporal_model     6.251480           6.720591         8.353093           6.728374       submissions/test_temporal_ensemble_ratio_v1.csv                                 0.001946                       0.029619                                 0.012321
                     temporal_ridge_ratio_v1      temporal_model     6.182051           6.672905         8.425639           6.681783          submissions/test_temporal_ridge_ratio_v1.csv                                 0.002050                       0.031477                                 0.018967
       exploratory_selector_volatility_w0p75            selector     6.245006           6.768997         8.473359           6.791510                                                                                       0.001893                       0.354754                                 0.044870
exploratory_segment_region_rto_s0p30_c98_102 segment_calibration     6.374621           6.834887         8.490126           6.846560                                                                                       0.001668                       0.007072                                 0.000000
 segment_calibrated_volatility_s0p20_c98_102 segment_calibration     6.349388           6.807927         8.641030           6.815283                                                                                       0.001839                       0.003345                                 0.000000
```

## Топ по exploratory_score

```text
                      experiment_name hypothesis_group  fold10_mape  weighted_mape_127  safe_risk_score  exploratory_score                            generated_submission  mean_abs_relative_delta_vs_current_best  max_rel_delta_vs_current_best  share_abs_delta_vs_current_best_gt_3pct
              temporal_ridge_ratio_v1   temporal_model     6.182051           6.672905         8.425639           6.681783    submissions/test_temporal_ridge_ratio_v1.csv                                 0.002050                       0.031477                                 0.000340
           temporal_ensemble_ratio_v1   temporal_model     6.251480           6.720591         8.353093           6.728374 submissions/test_temporal_ensemble_ratio_v1.csv                                 0.001946                       0.029619                                 0.000000
         selector_by_rto_decile_w0p50         selector     6.273920           6.731671         9.008588           6.745540                                                                                 0.001818                       0.106398                                 0.003299
exploratory_selector_rto_decile_w0p75         selector     6.252698           6.724841         9.842083           6.748652                                                                                 0.002727                       0.159597                                 0.006452
         selector_by_region_rto_w0p50         selector     6.283430           6.736538         9.181357           6.756297                                                                                 0.002247                       0.343928                                 0.005384
         selector_by_rto_decile_w0p25         selector     6.314717           6.752065         7.399667           6.757059  submissions/test_selector_by_rto_decile_v1.csv                                 0.000909                       0.053199                                 0.000679
         selector_by_region_rto_w0p25         selector     6.317084           6.752798         7.646126           6.759524                                                                                 0.001124                       0.171964                                 0.001116
              exploratory_selector_v1      exploratory     6.262093           6.730560         9.873966           6.771256                                                                                 0.003262                       0.515893                                 0.013825
exploratory_selector_region_rto_w0p75         selector     6.264970           6.732749         9.772837           6.772229                                                                                 0.003151                       0.515893                                 0.013437
         selector_by_volatility_w0p25         selector     6.310956           6.765501         7.941404           6.772747                                                                                 0.001036                       0.118251                                 0.001552
                 decile_beta_ratio_v1          low_rto     6.356880           6.775090         6.776698           6.775448       submissions/test_decile_beta_ratio_v1.csv                                 0.000089                       0.000738                                 0.000000
           ratio_shrink_b0p06_c97_103             sent     6.360338           6.777467         6.777467           6.777467                                                                                 0.000000                       0.000000                                 0.000000
                temporal_hgb_ratio_v1   temporal_model     6.330339           6.775469         9.435233           6.784955                                                                                 0.002323                       0.030727                                 0.000097
         selector_by_volatility_w0p50         selector     6.273232           6.763215         8.855550           6.785179                                                                                 0.002071                       0.236503                                 0.006840
        october_high_rollback_safe_v1 outlier_specific     6.381627           6.771780         8.128858           6.785719   submissions/test_october_high_rollback_v1.csv                                 0.001549                       0.041416                                 0.000146
                  baseline_last_month             sent     6.365661           6.782111         6.794908           6.786108                                                                                 0.000534                       0.001517                                 0.000000
               weighted_oof_global_v1 weighted_mixture     6.360338           6.785762         6.891072           6.788887     submissions/test_weighted_oof_global_v1.csv                                 0.000757                       0.040697                                 0.000049
exploratory_selector_volatility_w0p75         selector     6.245006           6.768997         8.473359           6.791510                                                                                 0.001893                       0.354754                                 0.007470
           weighted_oof_rto_decile_v1 weighted_mixture     6.352694           6.779513         8.094447           6.792022                                                                                 0.001478                       0.324206                                 0.003299
           weighted_oof_volatility_v1 weighted_mixture     6.355371           6.775155        10.968432           6.802389                                                                                 0.003898                       0.131976                                 0.005821
```

## Главные выводы segment mining

```text
       segment_type         segment_value    n  current_best_mape best_model_in_segment  improvement_best_vs_current  signed_bias_current_best  stability_score
     region_x_trend Краснодарский край__2  327          12.693413        rolling_mean_3                     3.204295                  0.113511         3.204295
   Населенный пункт           Геленджик г  105          38.778600            damped_add                     2.629906                  0.375015         2.629906
   Населенный пункт               Анапа г  114          32.675860            damped_add                     2.451995                  0.323215         2.451995
     region_x_trend Краснодарский край__0  674          17.439657            damped_add                     2.089179                  0.129142         2.089179
     region_x_trend  Ленинградская обл__0  502          13.965300            damped_add                     1.390648                  0.119507         1.390648
   Населенный пункт             Можайск г  108          15.803433            damped_add                     1.167423                  0.151352         1.167423
     region_x_trend Краснодарский край__3  438           8.741277        rolling_mean_3                     2.293197                  0.064908         1.146598
     region_x_trend     Московская обл__0 1784          11.315324            damped_add                     1.091186                  0.090973         1.091186
     region_x_trend Краснодарский край__1  393          11.326049        rolling_mean_3                     1.673215                  0.083286         0.836607
     region_x_trend           Москва г__4 1006           6.585541            damped_add                     0.805881                 -0.058059         0.805881
     region_x_trend   Владимирская обл__0  336           9.449027            damped_add                     0.757678                  0.072436         0.757678
    Количество касс                    17  237          11.433596            damped_add                     0.693885                  0.053701         0.693885
     region_x_trend     Московская обл__4  704           7.729800        rolling_mean_3                     1.287199                  0.030877         0.643599
     region_x_trend     Ростовская обл__3  380           4.506201        rolling_mean_3                     0.583901                  0.027010         0.583901
   Населенный пункт         Стерлитамак г  120           4.490109        rolling_mean_3                     0.785489                  0.003693         0.523660
  area_x_volatility            Большой__4 1979          11.260058            damped_add                     0.509986                  0.063458         0.509986
   Населенный пункт             Энгельс г  168           6.234490      rolling_median_3                     0.697290                  0.014360         0.464860
     region_x_trend     Татарстан Респ__4  379           6.223579        rolling_mean_3                     0.919938                  0.029742         0.459969
  area_x_volatility          Маленький__4 2594          10.912147            damped_add                     0.458556                  0.078137         0.458556
region_x_rto_decile  Ленинградская обл__9  370          11.981779            damped_add                     0.455931                  0.104847         0.455931
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
           experiment_name          model_name  fold10_mape  weighted_mape_127  safe_risk_score  exploratory_score  mean_rel_delta_vs_current_best  max_rel_delta_vs_current_best  lb_score  lb_mape
ratio_shrink_b0p06_c97_103        ratio_shrink     6.360338           6.777467         6.777467           6.777467                        0.000000                       0.000000     95.88     4.12
       baseline_last_month baseline_last_month     6.365661           6.782111         6.794908           6.786108                       -0.000338                       0.001517     95.86     4.14
```
