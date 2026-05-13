# Pyaterochka Retail Forecast Hack

## Описание проекта

Репозиторий содержит воспроизводимое кодовое решение для соревнования X5 / Пятерочка: прогноз РТО магазинов на месяц 11, то есть на ноябрь. Основной фокус сейчас - безопасная генерация сабмитов, учет leaderboard-результатов и аккуратные эксперименты вокруг сильного baseline последнего месяца.

Jupyter Notebook пока намеренно не делается. Он будет подготовлен позже, когда будет выбран лучший сабмит.

## Данные

Файл данных:

```text
data/train.csv
```

Содержимое:

- 206150 строк
- 20615 магазинов
- месяцы 1..10
- идентификатор магазина: `new_id`
- временной столбец: `Месяц`
- целевой столбец: `РТО`

Каждая строка соответствует одному магазину в одном месяце.

## Формат Сабмита

Файл для Контеста должен содержать две колонки:

```text
new_id,rto
```

Требования:

- сабмит для Контеста должен быть **С заголовком** `new_id,rto`
- pandas index сохранять нельзя
- строк данных должно быть 20615 без учета заголовка
- ровно две колонки: `new_id`, `rto`
- разделитель: запятая
- кодировка: UTF-8
- без NaN
- без отрицательных `rto`
- `new_id` уникальны
- размер файла меньше 1 МБ

Перед отправкой обязательно проверьте файл:

```bash
python scripts/check_submission.py submissions/test_baseline_last_month.csv
```

## Архитектура Проекта

```text
data/
  train.csv
src/
  config.py
  features.py
  metrics.py
  models.py
  registry.py
  submit.py
  validation.py
scripts/
  analyze_leaderboard_alignment.py
  analyze_submission_space.py
  check_submission.py
  error_analysis.py
  list_submissions.py
  make_baseline_submission.py
  make_submission.py
  record_leaderboard_result.py
  restore_best_submission.py
  run_experiments.py
  validate.py
reports/
  best_submission.json
  error_analysis.md
  error_slices.csv
  experiment_results.csv
  experiments.md
  leaderboard_alignment.md
  leaderboard_results.csv
  recommended_submissions.md
  submission_registry.csv
  submission_space_analysis.md
submissions/
```

Назначение:

- `src/features.py` - leakage-free лаги, rolling-признаки через `shift(1)`, динамика, календарные признаки.
- `src/models.py` - baseline-модели, CatBoost MAPE, CatBoost log, fallback при отсутствии CatBoost.
- `src/submit.py` - генерация прогнозов, официальный baseline, сохранение и строгая проверка CSV.
- `src/registry.py` - учет leaderboard-результатов и best-known сабмита.
- `src/validation.py` - time-based validation по месяцам 8, 9, 10.
- `scripts/run_experiments.py` - консервативные эксперименты вокруг `baseline_last_month`.
- `reports/` - отчеты, реестр результатов и рекомендации.
- `submissions/` - готовые CSV-файлы.

## Установка Окружения

Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Если CatBoost недоступен, пайплайн не ломается и использует fallback. Для полноценного ML-эксперимента CatBoost желательно установить.

## Проверка Формата

```bash
python scripts/check_submission.py test.csv
python scripts/check_submission.py submissions/test_baseline_last_month.csv
```

Checker печатает размер файла, первую строку, колонки, shape, dtypes, количество NaN, отрицательных `rto`, дубликатов `new_id`, head/tail и verdict.

## Генерация Сабмитов

Официальный baseline организаторов:

```bash
python scripts/make_baseline_submission.py --official --make-test-copy
```

Baseline последнего месяца:

```bash
python scripts/make_baseline_submission.py
```

ML-сабмиты:

```bash
python scripts/make_submission.py --model catboost_mape
python scripts/make_submission.py --model catboost_log
```

По умолчанию `make_submission.py` не перезаписывает `test.csv`. Если передать `--make-test-copy`, скрипт покажет предупреждение о текущем лучшем подтвержденном сабмите. Для экспериментов можно использовать защиту:

```bash
python scripts/make_submission.py --model catboost_mape --make-test-copy --restore-best-after
```

## Учёт Leaderboard-Результатов

Известные результаты хранятся в:

```text
reports/leaderboard_results.csv
reports/submission_registry.csv
reports/best_submission.json
```

Записать новый результат:

```bash
python scripts/record_leaderboard_result.py --file submissions/test_candidate.csv --model candidate_name --lb-score 95.90 --verdict OK --comment "краткий комментарий"
```

Если `verdict == OK` и score лучше текущего best-known, `reports/best_submission.json` обновится автоматически. `test.csv` при записи результата не трогается.

Посмотреть список сабмитов:

```bash
python scripts/list_submissions.py
```

## Восстановление Лучшего Сабмита

В зачёт идет последнее отправленное решение, поэтому после неудачного эксперимента нужно быстро вернуть лучший подтвержденный файл:

```bash
python scripts/restore_best_submission.py
```

Команда копирует файл из `reports/best_submission.json` в корень проекта как `test.csv` и проверяет формат.

## Эксперименты

Запустить текущий набор консервативных экспериментов:

```bash
python scripts/run_experiments.py
```

Скрипт сохраняет:

- `reports/experiment_results.csv`
- `reports/experiments.md`
- `reports/recommended_submissions.md`
- новые CSV в `submissions/`

Дополнительная аналитика:

```bash
python scripts/error_analysis.py
python scripts/analyze_submission_space.py
python scripts/analyze_leaderboard_alignment.py
```

Эксперименты строятся вокруг текущего best `ratio_shrink_b0p06_c97_103`, потому что плато 95.88 уже почти исчерпало простые микропоправки. `scripts/run_experiments.py` делает OOF-прогнозы, segment error mining, segment calibration, selector/gating, weighted mixture, temporal pattern model, clustering, outlier-specific и low-RTO MAPE-aware кандидатов. Новые кандидаты не копируются в `test.csv` автоматически; после run script текущий best-known восстанавливается обратно.

Перед отправкой нового решения проверьте, не хуже ли оно ожидаемо: последнее отправленное решение идет в зачёт.

## Правила Ведения Экспериментов

- Новый кандидат сначала сохраняется только в `submissions/`.
- `test.csv` не перезаписывается автоматически.
- Перед отправкой кандидат проверяется командой `python scripts/check_submission.py <path>`.
- После отправки в Контест результат записывается в реестр через `scripts/record_leaderboard_result.py`.
- Если кандидат хуже текущего лучшего, нужно восстановить лучший `test.csv` командой `python scripts/restore_best_submission.py`.
- Не стоит отправлять агрессивные варианты, которые заметно уходят от `baseline_last_month`: leaderboard уже показал, что ноябрь очень близок к октябрю.

## Текущий Лучший Подтверждённый Результат

```text
submissions/test_ratio_shrink_b0p06_c97_103.csv
score = 95.88
LB MAPE = 4.12
```

Уже проверенные выводы leaderboard:

- `submissions/test_ratio_shrink_b0p05_c97_103.csv` впервые улучшил baseline: score `95.87`.
- `submissions/test_ratio_shrink_b0p06_c97_103.csv` и `submissions/test_ratio_shrink_b0p07_c98_102.csv` подняли уровень до `95.88`; текущим best выбран более консервативный `b0p06`.
- `submissions/test_baseline_last_month.csv` остается очень сильным baseline: score `95.86`.
- `submissions/test_ensemble_conservative_v1.csv` и `submissions/test_ensemble_conservative_v2.csv` повторили score `95.86`, но не улучшили baseline.
- `submissions/test_last_month_mult_0995.csv` получил score `95.85`.
- `submissions/test_last_month_mult_09975.csv` получил score `95.86`.
- `submissions/test_last_month_mult_10025.csv` получил score `95.85`.
- `submissions/test_last_month_mult_101.csv` получил score `95.73`, то есть множитель `1.010` хуже baseline.
- `submissions/test_last_month_mult_102.csv` получил score `95.40`, то есть множитель `1.020` заметно хуже baseline.
- `submissions/test_last_month_mult_1015.csv` получил CE; локальная проверка формата не выявила проблему.
- `submissions/test_residual_centered_v1.csv` получил score `95.79`, поэтому residual-поправку пока не стоит развивать агрессивно.
- Сегментные микропоправки по region/area/blend/alcohol дали `95.86`: они безопасны как слабый сигнал, но сами по себе не улучшили best.

Для восстановления:

```bash
python scripts/restore_best_submission.py
```
