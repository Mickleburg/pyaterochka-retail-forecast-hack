# Pyaterochka Retail Forecast Hack

## Описание проекта

Кодовое решение для соревнования X5 / Пятерочка: прогноз РТО магазинов на месяц 11, то есть на ноябрь. Цель текущего этапа - воспроизводимые скрипты для валидации по времени и генерации корректных CSV-сабмитов для Яндекс.Контеста.

Jupyter Notebook пока намеренно не делается. Он будет подготовлен позже, когда будет выбран лучший сабмит.

## Данные

Основной файл данных:

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

Итоговый файл для Контеста называется `test.csv` и содержит две колонки:

```text
new_id,rto
```

Требования:

- CSV должен быть **С заголовком** `new_id,rto`
- 20615 строк данных без учета заголовка
- ровно две колонки: `new_id`, `rto`
- без pandas index
- разделитель: запятая
- кодировка: UTF-8
- без NaN
- без отрицательных `rto`
- `new_id` уникальны
- размер файла меньше 1 МБ

**Файл для Контеста должен быть С заголовком. Не используйте `--no-header` для отправки.**

## Архитектура Проекта

```text
data/
  train.csv
src/
  config.py
  features.py
  metrics.py
  models.py
  submit.py
  validation.py
scripts/
  check_submission.py
  make_baseline_submission.py
  make_submission.py
  validate.py
reports/
submissions/
```

Назначение основных файлов:

- `src/features.py` - лаги РТО, rolling-признаки через `shift(1)`, динамика, календарные признаки, строки месяца 11.
- `src/models.py` - baseline-модели, CatBoost MAPE, CatBoost log, fallback при отсутствии CatBoost.
- `src/submit.py` - генерация прогнозов, официальный baseline, сохранение и строгая проверка CSV.
- `src/validation.py` - time-based validation по месяцам 8, 9, 10.
- `scripts/make_baseline_submission.py` - официальный mean October baseline и baseline по последнему месяцу.
- `scripts/make_submission.py` - генерация ML-сабмитов.
- `scripts/check_submission.py` - проверка готового CSV перед отправкой.
- `reports/` - отчеты валидации.
- `submissions/` - готовые CSV-файлы.

## Установка

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

Зависимости включают `pandas`, `numpy`, `scikit-learn`, `catboost`. Если CatBoost не установлен, код не ломает пайплайн и использует fallback, но для основного ML-сабмита рекомендуется установить CatBoost.

## Быстрый Старт

Сначала сгенерируйте официальный baseline организаторов и скопируйте его в корень как `test.csv`:

```bash
python scripts/make_baseline_submission.py --official --make-test-copy
python scripts/check_submission.py test.csv
```

Этот файл нужен для первой контрольной отправки формата.

## Валидация

```bash
python scripts/validate.py
```

Используется time-based validation:

- train months `<= 7`, valid month `8`
- train months `<= 8`, valid month `9`
- train months `<= 9`, valid month `10`

Метрика: MAPE в процентах. Результаты сохраняются в:

```text
reports/validation_results.csv
```

## Генерация ML-Сабмитов

```bash
python scripts/make_submission.py --model catboost_mape --make-test-copy
python scripts/make_submission.py --model catboost_log --make-test-copy
```

Также можно сгенерировать baseline последнего месяца:

```bash
python scripts/make_baseline_submission.py
```

Все contest-файлы по умолчанию сохраняются с заголовком. Флаг `--no-header` в `make_submission.py` оставлен только для отладки и не должен использоваться для Контеста.

## Рекомендованный Порядок Отправки

1. Сначала отправить `submissions/test_official_mean_october_baseline.csv`, чтобы проверить формат. Ожидается OK и около 54.34 балла.
2. Потом отправлять `submissions/test_baseline_last_month.csv`, CatBoost-сабмиты или ensemble.

## Воспроизводимость

В проекте фиксируется:

```text
RANDOM_SEED = 2026
```

Случайный `train_test_split` не используется. Валидация строится только по времени.
