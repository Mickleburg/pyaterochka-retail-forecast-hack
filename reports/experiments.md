# Отчет по экспериментам

## Краткий вывод

Leaderboard подтвердил, что `baseline_last_month` остается лучшим или делит первое место. Простое увеличение прогноза ухудшило результат: множитель 1.010 дал score 95.73, множитель 1.020 дал score 95.40. Консервативные ансамбли v1/v2 дали тот же score 95.86, то есть не улучшили baseline.

Следовательно, скрытый ноябрь очень близок к октябрю. Новые кандидаты должны быть микрокоррекциями вокруг `РТО_10`, а не самостоятельными агрессивными моделями.

## Расследование CE для test_last_month_mult_1015.csv

- Файл существует: да.
- Размер файла: 378669 байт.
- Первая строка: `new_id,rto`.
- Shape через `pd.read_csv`: (20615, 2).
- Колонки: ['new_id', 'rto'].
- NaN: 0.
- Отрицательные `rto`: 0.
- Дубликаты `new_id`: 0.
- BOM: нет.
- NUL-байты: 0.
- CRLF строк: 20616, LF строк: 20616, одиночных CR: 0.
- Локальная проверка формата не выявила проблемы; вероятно, в Контест был отправлен не тот файл или произошла ошибка загрузки.

## Калибровка локальной валидации по leaderboard

Локальная валидация переоценила положительные множители: `last_month_mult_102` выглядел хорошо по fold10 и weighted 0.1/0.2/0.7, но на LB оказался заметно хуже baseline. Поэтому теперь локальные метрики используются как фильтр риска, а не как прямой прогноз leaderboard.

Основные локальные сигналы после калибровки:

- fold10 важен, но сам по себе недостаточен;
- weighted_mape_127 полезен как recency-weighted фильтр, но он ошибся на положительных множителях;
- отклонение от `baseline_last_month` нужно явно штрафовать;
- кандидаты с большим числом магазинов, измененных более чем на 3%, считаются рискованными;
- положительные глобальные множители теперь считаются более рискованными, потому что LB уже показал ухудшение.

Формула `risk_score`:

```text
risk_score = weighted_mape_127
             + 0.5 * max(0, fold10_mape - baseline_fold10_mape)
             + 10 * mean_abs_relative_delta_vs_baseline
             + 5 * share_abs_delta_gt_3pct
             + penalty_for_positive_global_multiplier
```

Чем меньше `risk_score`, тем безопаснее кандидат. Эта метрика намеренно штрафует даже локально перспективные варианты, если они слишком далеко уходят от октябрьского baseline.

## Leaderboard-результаты

```text
 submitted_at                                            filename                     model_name  local_cv_mape  lb_score  lb_mape verdict                                           comment
          NaN submissions/test_official_mean_october_baseline.csv official_mean_october_baseline            NaN     54.34    45.66      OK            официальный бейзлайн, проверка формата
          NaN                  submissions/test_catboost_mape.csv                  catboost_mape            NaN     95.80     4.20      OK                 первый кандидат catboost/fallback
          NaN                   submissions/test_catboost_log.csv                   catboost_log            NaN     95.80     4.20      OK             первый кандидат catboost log/fallback
          NaN            submissions/test_baseline_last_month.csv            baseline_last_month            NaN     95.86     4.14      OK             текущее лучшее подтвержденное решение
          NaN       submissions/test_ensemble_conservative_v1.csv       ensemble_conservative_v1            NaN     95.86     4.14      OK        такой же результат, как у текущего лучшего
          NaN       submissions/test_ensemble_conservative_v2.csv       ensemble_conservative_v2            NaN     95.86     4.14      OK        такой же результат, как у текущего лучшего
          NaN            submissions/test_last_month_mult_101.csv            last_month_mult_101            NaN     95.73     4.27      OK            множитель 1.010, хуже текущего лучшего
          NaN            submissions/test_last_month_mult_102.csv            last_month_mult_102            NaN     95.40     4.60      OK            множитель 1.020, хуже текущего лучшего
          NaN           submissions/test_last_month_mult_1015.csv           last_month_mult_1015            NaN      0.00   100.00      CE CE в Контесте; локальный формат проверен отдельно
```

## Топ экспериментов по risk_score

```text
                   experiment_name            model_name  fold10_mape  weighted_mape_127  risk_score generated_submission  max_rel_delta_vs_baseline  share_abs_delta_gt_3pct  lb_score lb_verdict
      outlier_rollback_t1p15_b0p95      outlier_rollback     6.374613           6.770237    6.780457                                        0.030602                 0.000049       NaN           
               baseline_last_month   baseline_last_month     6.365661           6.782111    6.782111                                        0.000000                 0.000000     95.86         OK
                 last_month_mult_1 last_month_multiplier     6.365661           6.782111    6.782111                                        0.000000                 0.000000       NaN           
segment_alcohol_s0p05_k500_c98_102        segment_shrink     6.365631           6.782155    6.782216                                        0.000051                 0.000000       NaN           
segment_alcohol_s0p05_k500_c97_103        segment_shrink     6.365631           6.782155    6.782216                                        0.000051                 0.000000       NaN           
segment_alcohol_s0p05_k300_c98_102        segment_shrink     6.365635           6.782155    6.782217                                        0.000052                 0.000000       NaN           
segment_alcohol_s0p05_k300_c97_103        segment_shrink     6.365635           6.782155    6.782217                                        0.000052                 0.000000       NaN           
segment_alcohol_s0p05_k100_c98_102        segment_shrink     6.365638           6.782155    6.782217                                        0.000053                 0.000000       NaN           
segment_alcohol_s0p05_k100_c97_103        segment_shrink     6.365638           6.782155    6.782217                                        0.000053                 0.000000       NaN           
 segment_alcohol_s0p05_k50_c97_103        segment_shrink     6.365639           6.782155    6.782217                                        0.000053                 0.000000       NaN           
 segment_alcohol_s0p05_k50_c98_102        segment_shrink     6.365639           6.782155    6.782217                                        0.000053                 0.000000       NaN           
 segment_alcohol_s0p1_k500_c97_103        segment_shrink     6.365602           6.782199    6.782322                                        0.000102                 0.000000       NaN           
 segment_alcohol_s0p1_k500_c98_102        segment_shrink     6.365602           6.782199    6.782322                                        0.000102                 0.000000       NaN           
 segment_alcohol_s0p1_k300_c97_103        segment_shrink     6.365609           6.782199    6.782324                                        0.000104                 0.000000       NaN           
 segment_alcohol_s0p1_k300_c98_102        segment_shrink     6.365609           6.782199    6.782324                                        0.000104                 0.000000       NaN           
```

## NLP / embeddings

В данных есть категориальные признаки (`Регион`, `Населенный пункт`, категории даты открытия и площади), но нет свободного текста. Поэтому классический NLP или Word2Vec здесь выглядит избыточным. Более уместны frequency/target/segment encodings, которые частично проверяются через сегментные shrinkage-поправки. Отдельный NLP-сабмит не генерировался.
