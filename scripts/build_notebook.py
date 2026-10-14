"""Create and execute a compact presentation of the verified LFPG run"""
import nbformat
from nbclient import NotebookClient
from audit_source import ROOT
from notebook_runtime import ProjectKernelManager

cells = [
    nbformat.v4.new_markdown_cell('# Approach Risk 1.2 – Altitude Forecasting Pipeline\n\n'
        'LFPG 09L: PostgreSQL → SQL → модели → MLflow. Реальный запуск на локальных данных 2025 года'),
    nbformat.v4.new_markdown_cell('## Задача и протокол\n\n'
        'Прогнозируем последующую наблюдаемую высоту через 27–33 секунды. Физический baseline '
        'продолжает текущую вертикальную скорость; модели учат поправку. Девять train-дней, '
        'три validation-дня и три test-дня перенесены из 1.1. Каждый заход имеет равный вес. '
        'Scaler обучается на train; параметры моделей и early stopping выбираются по validation'),
    nbformat.v4.new_code_cell('# Загружаем сохраненные результаты без повторного обучения\n'
        'from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\n'
        'root = Path.cwd()\ndisplay(pd.read_csv(root / "reports/split_inventory.csv"))'),
    nbformat.v4.new_markdown_cell('## PostgreSQL и качество\n\n'
        'Первая загрузка: 15 931 строка. Повторная: ноль новых строк. Извлечение сохранило '
        'все значения. Ключ наблюдения: airport + event_id + origin_time. SQL JOIN присоединяет '
        'фиксированный split дня; CTE считает ошибки с равным вкладом заходов'),
    nbformat.v4.new_code_cell('# Показываем состав данных из SQL и проверку физических соотношений\n'
        'display(pd.read_csv(root / "reports/sql/01.csv").query("airport == \'LFPG\'"))\n'
        'display(pd.read_csv(root / "reports/sql/07.csv").query("airport == \'LFPG\'"))'),
    nbformat.v4.new_markdown_cell('## Итоговое сравнение\n\n'
        'Test содержит 55 заходов. MAE сначала усредняется внутри захода, затем между заходами. '
        'RMSE – квадратный корень из среднего по заходам MSE. Все варианты зафиксированы до '
        'вычисления test-метрик; новый подбор по test не проводится'),
    nbformat.v4.new_code_cell('# Показываем измеренные метрики и график сравнения\n'
        'display(pd.read_csv(root / "reports/model_metrics.csv"))\n'
        'display(Image(filename=str(root / "reports/model_comparison.png")))'),
    nbformat.v4.new_markdown_cell('## PyTorch\n\n'
        'MLP с двумя скрытыми слоями и ReLU учит residual. DataLoader перемешивает только train. '
        'Взвешенный MSE сохраняет равный вклад заходов. Adam, seed 20261014, ширина 16 или 32, '
        'до 300 эпох; остановка после 30 эпох без улучшения validation MAE'),
    nbformat.v4.new_code_cell('# Проверяем выбор параметров по валидации без обращения к тестовой выборке\n'
        'display(pd.read_csv(root / "reports/validation_search.csv"))\n'
        'display(Image(filename=str(root / "reports/mlp_validation.png")))'),
    nbformat.v4.new_markdown_cell('## Интерпретация и ограничения\n\n'
        'MLP улучшила физический baseline, но уступила Polynomial Ridge на этом тесте. '
        'Это результат ограниченного эксперимента, без доказательства общего превосходства модели. '
        'Данные относятся к одной полосе и отобранным погожим дням. Между split повторяются борта. '
        'Фактический горизонт известен ретроспективно и входит в признаки. '
        'Строгий онлайн-горизонт 30 секунд, go-around и эксплуатационная безопасность не проверены')
]
notebook = nbformat.v4.new_notebook(cells=cells)
method_cells = [
    nbformat.v4.new_markdown_cell('## Структура данных и соединение\n\n'
        '`study_days` содержит одну запись на аэропорт и день; `examples` – пары наблюдений. '
        'Первичный ключ пары: airport + event_id + origin_time. Внешний ключ связывает ее с днем. '
        'JOIN по airport и day переносит фиксированный split. Проверяем отдельно потери строк '
        'и множественные соответствия. Загрузка через временную таблицу и COPY выполняется '
        'одной транзакцией; конфликт данных или ошибка итоговой сверки отменяют изменения\n\n'
        '```sql\nSELECT e.event_id, e.origin_time, d.season, d.split\n'
        'FROM altitude.examples e\nJOIN altitude.study_days d USING (airport, day);\n```'),
    nbformat.v4.new_code_cell('# Показываем проверку соединения и размеры выборок\n'
        'display(pd.read_csv(root / "reports/sql/join_01.csv"))\n'
        'display(pd.read_csv(root / "reports/sql/join_03.csv"))'),
    nbformat.v4.new_markdown_cell('## Признаки и доступность во времени\n\n'
        'Определения сверены с causal_features в scripts/lfpg_training.py и '
        'research/lfpg-seasonal-learning/scripts/prepare_examples.py исходного проекта 1.1\n\n'
        '| Признак | Смысл | Единицы | Источник и доступность |\n'
        '| --- | --- | --- | --- |\n'
        '| height_ft | Высота с поправкой QNH над высотой аэродрома | ft | Текущая барометрическая высота, QNH на момент прогноза, высота аэродрома |\n'
        '| vertical_rate_fps | Барометрическая вертикальная скорость с фактором QNH | ft/s | Текущее сообщение; геометрическая скорость исключена |\n'
        '| past_height_rate_fps | Изменение барометрической высоты за предшествующий интервал с фактором текущего QNH | ft/s | Текущая и последняя подходящая прошлая точка |\n'
        '| speed_knots | Путевая скорость | kt | Текущее наблюдение; это не приборная скорость |\n'
        '| distance_m | Расстояние до порога | m | Текущее положение и геометрия полосы |\n'
        '| lateral_m | Поперечное отклонение от оси | m | Текущее положение и геометрия полосы |\n'
        '| heading_error_deg | Разница путевого угла и направления полосы в диапазоне [-180, 180) | градусы | Текущий track и геометрия; это не курс носа самолета |\n'
        '| past_gap_s | Интервал до прошлой точки | s | История 5–30 секунд |\n'
        '| horizon_s | Интервал до выбранного будущего наблюдения | s | Ретроспективный выбор цели, 27–33 секунды |\n\n'
        'QNH применяется как исследовательское преобразование барометрической высоты; '
        'height_ft не является радиовысотой. Причинность восьми текущих/исторических признаков '
        'относится к построению примеров и предполагаемой доступности METAR. '
        'Сохраненный Parquet сам по себе не доказывает историческое время публикации METAR\n\n'
        'horizon_s входит также в физический baseline. Строгий онлайн-прогноз на 30 секунд '
        'требует отдельного определения цели и эксперимента. Простая замена всех значений '
        'горизонта на 30 в этих примерах не повторяет исходную постановку'),
    nbformat.v4.new_markdown_cell('## Взвешенная предобработка\n\n'
        'Для захода с n_i парами вес каждой пары пропорционален 1/n_i. '
        'Среднее и дисперсия StandardScaler вычисляются с этими весами только по train. '
        'Преобразование z = (x − mu_train) / sigma_train применяется к validation и test '
        'без обновления параметров. Это выравнивает вклад заходов, но не дней и сезонов\n\n'
        'Проверка ниже отдельно вычисляет взвешенные моменты, сверяет сохраненный scaler MLP '
        'и проверяет неизменность параметров после transform. Повторное обучение моделей '
        'и новый выбор параметров не выполняются'),
    nbformat.v4.new_code_cell('# Проверяем предобработку и показываем параметры обучающей выборки\n'
        'import sys, json\nsys.path.insert(0, str(root / "scripts"))\n'
        'from review_preprocessing import run as review_preprocessing\n'
        'review_preprocessing()\n'
        'display(pd.read_csv(root / "reports/scaler_train.csv"))\n'
        'display(Image(filename=str(root / "reports/scaled_train_features.png")))'),
    nbformat.v4.new_markdown_cell('Гистограммы используют только train и веса заходов. '
        'StandardScaler меняет масштаб, но не делает распределения нормальными. '
        'Наличие нескольких групп на графиках требует отдельного анализа структуры выборки; '
        'по этим графикам не вводились новые правила удаления наблюдений')
]
notebook.cells[5:5] = method_cells
cells_for_aggregation = [
    nbformat.v4.new_markdown_cell('## Усреднение ошибки по заходам\n\n'
        'GROUP BY сначала вычисляет MAE каждого захода; CTE хранит промежуточный результат '
        'в пределах запроса. Затем ошибки усредняются между заходами. При усреднении по строкам '
        'вклад захода пропорционален числу пар. На тесте baseline: 49.35 ft по заходам '
        'и 51.58 ft по строкам. Количество пар зависит от плотности и пригодности наблюдений, '
        'поэтому не равно длительности захода'),
    nbformat.v4.new_code_cell('# Показываем независимый SQL-расчет и распределение числа пар\n'
        'display(pd.read_csv(root / "reports/sql/event_mae_01.csv"))\n'
        'display(Image(filename=str(root / "reports/event_pair_distribution.png")))')
]
result_index = next(i for i, cell in enumerate(notebook.cells)
                    if cell.cell_type == 'markdown' and cell.source.startswith('## Итоговое сравнение'))
notebook.cells[result_index:result_index] = cells_for_aggregation
error_cells = [
    nbformat.v4.new_markdown_cell('## Анализ ошибок фиксированных моделей\n\n'
        'Использованы сохраненная MLP из 1.2 и классические pooled-модели из 1.1. '
        'Классические модели первого запуска 1.2 не были сохранены. '
        'Для моделей 1.1 проверены набор признаков, выбранные alpha и совпадение агрегатных '
        'MAE/RMSE с результатами 1.2; совпадение агрегатов не доказывает идентичность '
        'каждого отдельного предсказания первого запуска. Хеши артефактов зафиксированы. '
        'Обучение, выбор параметров и корректировка моделей не выполнялись\n\n'
        'Bias определяется как прогноз минус наблюдаемая высота: положительное значение '
        'означает завышение. Средняя bias и RMSE сохраняют равный вклад заходов. '
        'P90 относится к распределению MAE отдельных заходов, не к ошибкам отдельных точек'),
    nbformat.v4.new_code_cell('# Показываем сезонные ошибки и парное сравнение одних и тех же заходов\n'
        'display(pd.read_csv(root / "reports/errors_by_season.csv"))\n'
        'display(pd.read_csv(root / "reports/paired_event_errors.csv"))\n'
        'display(Image(filename=str(root / "reports/error_analysis.png")))'),
    nbformat.v4.new_markdown_cell('Polynomial Ridge улучшила baseline на 52 из 55 заходов, '
        'MLP – на 51. MLP уступила Polynomial Ridge на 42 заходах и улучшила ее на 13; '
        'средняя разница MAE составила +1.70 ft. Осенью Ridge имеет меньшую среднюю '
        'MAE, чем Polynomial Ridge и MLP. Общее превосходство Polynomial Ridge не '
        'означает превосходства на каждом заходе или в каждой подгруппе\n\n'
        'Летом средняя bias Polynomial Ridge равна +9.22 ft, MLP +10.62 ft; '
        'весной соответственно −2.97 и −11.59 ft, осенью −2.17 и −3.07 ft. '
        'Каждому сезону соответствует один тестовый день. Сезон, конкретный день и '
        'состав заходов смешаны: эти наблюдения не доказывают причинный сезонный эффект. '
        'Доверительные интервалы и статистическая значимость пока не оценены. '
        'Точки одного захода зависимы, а борта могут повторяться между днями; '
        'независимый bootstrap строк не подходит для обоснования выводов')
]
notebook.cells[-1:-1] = error_cells
uncertainty_cells = [
    nbformat.v4.new_markdown_cell('## Неопределенность парной разницы ошибок\n\n'
        'В тесте 55 заходов, 53 борта и три дня. Два борта встречаются в нескольких днях. '
        'Выполнен парный кластерный bootstrap по icao24: выбираем 53 борта с возвращением, '
        'сохраняем все заходы каждого выбранного борта вместе и считаем среднюю разницу '
        'MAE моделей по заходам. Число заходов в повторе может меняться. '
        '10 000 повторов, seed 20261014; границы – процентили 2.5% и 97.5%\n\n'
        'Отрицательная разница означает меньшую ошибку первой модели. '
        'Это исследовательские интервалы условно на наблюдаемых днях, а не оценка '
        'переноса на новые дни. Общая зависимость внутри дня не моделируется. '
        'При трех тестовых днях надежную неопределенность по дням установить нельзя'),
    nbformat.v4.new_code_cell('# Показываем состав кластеров и интервалы парных разниц\n'
        'display(pd.read_csv(root / "reports/test_cluster_inventory.csv"))\n'
        'display(pd.read_csv(root / "reports/paired_uncertainty.csv"))\n'
        'display(Image(filename=str(root / "reports/paired_uncertainty.png")))'),
    nbformat.v4.new_markdown_cell('Средняя разница MLP − Polynomial Ridge равна +1.70 ft, '
        'интервал +0.83…+2.65 ft. В рамках выбранной схемы повторной выборки преимущество '
        'Polynomial Ridge сохраняется. Это не подтверждающий тест общей научной гипотезы: '
        'сравнения исследовательские, выполнены после просмотра test, поправка на '
        'множественные сравнения не применялась. Не учтены неопределенность обучающей '
        'выборки и вариативность seed MLP. Для обобщения нужны дополнительные независимые '
        'дни и заранее фиксированная схема оценки')
]
notebook.cells[-1:-1] = uncertainty_cells
torch_cells = [
    nbformat.v4.new_markdown_cell('## PyTorch: устройство и восстановление модели\n\n'
        'Выбранная MLP: 9 → 16 → 16 → 1, ReLU между линейными слоями, 449 параметров. '
        'TensorDataset хранит тройки признаков, residual и веса. DataLoader перемешивает '
        'train и формирует пакеты до 256 примеров, без дополнительных процессов\n\n'
        'Цель обучения – residual / s, где s = 79.87898 ft – взвешенное среднеквадратическое '
        'значение residual на train, без вычитания среднего. Это нормировка масштаба цели, '
        'не StandardScaler для цели. Прогноз высоты равен baseline + s × выход сети\n\n'
        'Функция потерь пакета: mean(w × (prediction − target)^2). Веса нормированы '
        'по всей обучающей выборке, поэтому при равномерной выборке строк градиент '
        'соответствует взвешенной цели в среднем. Малый последний пакет и последовательное '
        'обновление параметров не эквивалентны одному полному градиентному шагу\n\n'
        'zero_grad очищает градиенты, backward вычисляет производные, Adam обновляет веса '
        '(learning rate 0.001, weight decay 0.0001). Для validation выполняются eval '
        'и no_grad. train_weighted_mse в истории усредняет потери пакетов, вычисленные '
        'при меняющихся параметрах внутри эпохи; это не точная train-loss финальной '
        'модели эпохи. Ее нельзя напрямую сопоставлять с validation MAE в ft\n\n'
        'Ширина и лучшая эпоха выбираются по validation. Test не участвует в остановке. '
        'Из двух ширин выбрана 16; лучшие веса относятся к эпохе 145, остановка – 175. '
        'Один seed не характеризует вариативность нейросетевого обучения'),
    nbformat.v4.new_code_cell('# Показываем выбор эпохи и проверку восстановленных моделей\n'
        'display(pd.read_csv(root / "reports/mlp_selection.csv"))\n'
        'display(pd.read_csv(root / "reports/artifact_verification.csv"))'),
    nbformat.v4.new_markdown_cell('Артефакт MLP содержит state_dict, ширину, порядок признаков, '
        'средние и масштабы признаков, масштаб residual. Он пригоден для восстановления '
        'прогнозов, но не для точного продолжения обучения: состояние optimizer и генератора '
        'случайных чисел не сохранено\n\n'
        'Классические pipeline сохранены в 1.2 как импорт из 1.1. Для них проверены '
        'validation-метрики, итоговые MAE/RMSE и неизменность прогнозов после сохранения. '
        'Это не восстановленные артефакты первого запуска 1.2. Будущие запуски train.py '
        'сохраняют выбранные pipeline непосредственно при обучении и добавляют их в MLflow')
]
notebook.cells[-1:-1] = torch_cells
notebook.cells.append(nbformat.v4.new_markdown_cell('## Итог LFPG\n\n'
    'Физический baseline улучшен обучаемыми поправками. Polynomial Ridge имеет меньшую '
    'среднюю MAE, чем MLP на текущем тесте. Условный bootstrap по бортам поддерживает '
    'это направление разницы, но не характеризует перенос на новые дни. '
    'Воспроизведение сохраненных прогнозов отделено от повторного обучения. '
    'Для строгого онлайн-горизонта 30 секунд требуется новая заранее определенная '
    'постановка; текущий ретроспективный горизонт не заменяется молча'))
NotebookClient(notebook, timeout=120, kernel_name='python3', kernel_manager_class=ProjectKernelManager,
               resources={'metadata': {'path': str(ROOT)}}).execute()
folder = ROOT/'notebooks'
folder.mkdir(exist_ok=True)
nbformat.write(notebook, folder/'01_lfpg_pipeline.ipynb')
print('Executed notebook saved')
