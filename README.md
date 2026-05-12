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
  check_submission.py
  list_submissions.py
  make_baseline_submission.py
  make_submission.py
  record_leaderboard_result.py
  restore_best_submission.py
  run_experiments.py
  validate.py
reports/
  best_submission.json
  experiment_results.csv
  experiments.md
  leaderboard_results.csv
  recommended_submissions.md
  submission_registry.csv
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
python scripts/record_leaderboard_result.py --file submissions/test_candidate.csv --model candidate_name --lb-score 95.90 --verdict OK --comment "short note"
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

Эксперименты строятся вокруг `baseline_last_month`, потому что leaderboard показывает, что простой прогноз `РТО месяца 10` очень силен. Новые кандидаты не копируются в `test.csv` автоматически; после run script текущий best-known восстанавливается обратно.

Перед отправкой нового решения проверьте, не хуже ли оно ожидаемо: последнее отправленное решение идет в зачёт.

## Текущий Лучший Подтверждённый Результат

```text
submissions/test_baseline_last_month.csv
score = 95.86
LB MAPE = 4.14
```

Для восстановления:

```bash
python scripts/restore_best_submission.py
```
