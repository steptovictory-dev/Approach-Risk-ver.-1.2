# Approach Risk 1.2 – Altitude Forecasting Pipeline

Воспроизводимый конвейер прогноза высоты на заходе: PostgreSQL, SQL, классические модели, PyTorch и MLflow

Статус: сквозные запуски LFPG, EGLL и EDDF выполнены 14.10.2026. PostgreSQL, четыре модели каждого аэропорта и MLflow проверены. Анализ ошибок и условной неопределенности выполнен отдельно для всех трех аэропортов

[Финальное ревью воспроизводимости](docs/FINAL_REVIEW.md): проверены чистое окружение, восстановление моделей, 14 тестов и выполнение обоих ноутбуков. Получение архивов с нуля и повторное обучение не проверялись

## Постановка

Первый этап использует сохраненные сезонные данные LFPG 09L из Approach Risk 1.1. Цель – последующая наблюдаемая высота через 27–33 секунды. Модели учат поправку к прогнозу постоянной вертикальной скорости. Это исследовательская регрессия высоты; эксплуатационная применимость не подтверждена

Список дней и train/validation/test переносится из завершенного исследования без повторного отбора. Каждый заход получает равный суммарный вес. Преобразования обучаются только на train; параметры и остановка PyTorch выбираются по validation. Test используется для итоговой оценки после фиксации вариантов

Фактический горизонт входит в признаки исходного эксперимента. Для онлайн-прогноза строго на 30 секунд потребуется отдельная постановка и проверка

## Обязательный результат

- Идемпотентная загрузка в PostgreSQL и содержательные SQL-проверки
- Физический baseline, Ridge, Polynomial Ridge и PyTorch MLP на одинаковом split
- Реальные MLflow runs с параметрами, метриками, идентификаторами данных и артефактами
- Таблицы, графики, выполненный ноутбук и проверенные команды воспроизведения
- Перенос конвейера на EGLL и EDDF после рабочего запуска LFPG

Исходные данные, модели, окружение и все чекпоинты остаются локальными. Новые зависимости изолируются от завершенных проектов 1.0 и 1.1

## Проверенный результат LFPG

| Модель | MAE, ft |
| --- | ---: |
| Физический baseline | 49.35 |
| Ridge | 20.03 |
| Polynomial Ridge | 17.16 |
| PyTorch MLP | 18.86 |

Каждый из 55 тестовых заходов имеет равный вклад. MLP уступила Polynomial Ridge. Четыре настоящих MLflow runs сохранены локально. [Отчет](reports/experiment_summary.md), [выполненный ноутбук](notebooks/01_lfpg_pipeline.ipynb)

## Повторить на имеющихся локальных данных

Python 3.12.14. Установить requirements-lock.txt в отдельное окружение. PostgreSQL 17.11 используется локально на 127.0.0.1:55432. DATABASE_URL хранится в игнорируемом .env

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

Для просмотра сохраненных результатов без нового обучения:

```powershell
.\.venv\Scripts\python.exe scripts/verify_reproduction.py
```

Команда требует имеющихся локальных данных, моделей и работающего PostgreSQL. Она сверяет версии lock-файла, хеши исходных Parquet, метрики восстановленных моделей, выполняет тесты и оба ноутбука. Она не получает исходные архивы с нуля и не обучает модели

[Подготовка PostgreSQL, запуск, остановка и устройство загрузчика](docs/POSTGRESQL.md)

[SQL JOIN: перенос split и проверка количества строк](docs/SQL_JOIN.md)

[GROUP BY и CTE: агрегация MAE по заходам](docs/EVENT_MAE.md)

```powershell
.\.venv\Scripts\python.exe scripts/audit_source.py
.\.venv\Scripts\python.exe scripts/database.py
.\.venv\Scripts\python.exe scripts/query_reports.py
.\.venv\Scripts\python.exe scripts/train.py
.\.venv\Scripts\python.exe scripts/query_reports.py --sql-file join_checks.sql
.\.venv\Scripts\python.exe scripts/event_mae_report.py
.\.venv\Scripts\python.exe scripts/review_preprocessing.py
.\.venv\Scripts\python.exe scripts/analyze_errors.py
.\.venv\Scripts\python.exe scripts/uncertainty.py
.\.venv\Scripts\python.exe scripts/model_artifacts.py
.\.venv\Scripts\python.exe scripts/build_notebook.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Первый аудит ожидает завершенный локальный проект 1.1 рядом с 1.2. Это пока инструкция повторения на имеющихся данных, проверка получения данных с нуля еще не выполнена. Синтетические примеры не используются для заявленных метрик

Независимые проверки весов и bootstrap без локальных данных и базы:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_preprocessing.py tests/test_uncertainty.py
```

Полный набор тестов требует локальных данных и PostgreSQL. `model_artifacts.py` импортирует классические модели LFPG из 1.1 с указанием происхождения; это отдельная процедура для текущего архивного запуска, не этап нового обучения

## Отдельные эксперименты EGLL и EDDF

[Выполненный ноутбук трех аэропортов](notebooks/02_three_airports.ipynb)

| Модель | LFPG MAE, ft | EGLL MAE, ft | EDDF MAE, ft |
| --- | ---: | ---: | ---: |
| Baseline | 49.35 | 55.95 | 72.60 |
| Ridge | 20.03 | 16.34 | 22.23 |
| Polynomial Ridge | 17.16 | 15.45 | 20.87 |
| PyTorch MLP | 18.86 | 14.52 | 20.56 |

Все модели обучены отдельно по аэропортам. В EGLL MLP улучшает среднюю MAE каждого из трех тестовых дней. В EDDF общий выигрыш MLP определяется майским днем; в августе и октябре Polynomial Ridge лучше. Парные интервалы по бортам условны на выбранных днях и не доказывают устойчивость преимущества на других днях

Загрузка и обучение принимают `--airport EGLL` или `--airport EDDF`. Результаты сохраняются в `reports/EGLL` и `reports/EDDF`, данные и модели – в соответствующих локальных подкаталогах. LFPG сохраняет прежние пути. Общая база различает аэропорты в первичных и внешних ключах

```powershell
.\.venv\Scripts\python.exe scripts/audit_source.py --airport EGLL
.\.venv\Scripts\python.exe scripts/database.py --airport EGLL
.\.venv\Scripts\python.exe scripts/train.py --airport EGLL
.\.venv\Scripts\python.exe scripts/audit_source.py --airport EDDF
.\.venv\Scripts\python.exe scripts/database.py --airport EDDF
.\.venv\Scripts\python.exe scripts/train.py --airport EDDF
.\.venv\Scripts\python.exe scripts/verify_airport_models.py
.\.venv\Scripts\python.exe scripts/analyze_airport_errors.py
.\.venv\Scripts\python.exe scripts/analyze_flight_phase.py
.\.venv\Scripts\python.exe scripts/build_airport_comparison.py
```

Каждый запуск обучения использует зафиксированный split своего аэропорта и выбирает параметры только по validation. Повторный запуск создает новые MLflow runs; он не нужен для просмотра готовых результатов. Итоговое сравнение не является проверкой переноса модели между аэропортами

Диагностика по высоте и удалению включена в ноутбук трех аэропортов. Отчеты `reports/phase_errors.csv` и `reports/phase_paired_errors.csv` содержат MAE с равным вкладом заходов внутри интервала, размеры групп и парные разницы MLP относительно Polynomial Ridge. Состав групп меняется; редкие группы от 3000 ft содержат только 4 захода EGLL и 6 EDDF. Анализ описательный, без нового подбора моделей по test
