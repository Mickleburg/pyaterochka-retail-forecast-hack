# X5 / Pyaterochka RTO Forecast

Кодовое решение для прогноза РТО магазинов на следующий месяц, то есть на месяц 11. На этом этапе ноутбук намеренно не делается: сначала фиксируем воспроизводимые скрипты, валидацию по времени и корректные CSV для Яндекс.Контеста.

## Установка

```bash
pip install -r requirements.txt
```

Основная модель использует CatBoost. Если `catboost` недоступен в окружении, скрипты автоматически используют простой детерминированный fallback, чтобы можно было проверить пайплайн и формат сабмита.

## Валидация

```bash
python scripts/validate.py
```

Валидация не использует случайный `train_test_split`. Фолды:

- train months `<= 7`, valid month `8`
- train months `<= 8`, valid month `9`
- train months `<= 9`, valid month `10`

Метрика: MAPE в процентах. Результаты сохраняются в `reports/validation_results.csv`.

## Генерация сабмитов

```bash
python scripts/make_baseline_submission.py
python scripts/make_submission.py --model catboost_mape
python scripts/make_submission.py --model catboost_log
python scripts/make_submission.py --model catboost_mape --make-test-copy
```

Файлы сохраняются в `submissions/`:

- `submissions/test_baseline_last_month.csv`
- `submissions/test_catboost_mape.csv`
- `submissions/test_catboost_log.csv`

При флаге `--make-test-copy` выбранный файл дополнительно копируется в корень проекта как `test.csv`. Именно `test.csv` нужно загружать в Контест, если требуется файл с таким именем.

По умолчанию CSV пишется без заголовка, с двумя колонками в порядке `new_id,rto`, 20615 строк, `rto` округлён до 2 знаков.

## Структура

```text
data/train.csv
src/
  config.py
  features.py
  metrics.py
  models.py
  submit.py
  validation.py
scripts/
  make_baseline_submission.py
  make_submission.py
  validate.py
reports/
submissions/
```
