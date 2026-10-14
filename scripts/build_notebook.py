"""Create and execute a compact presentation of the verified LFPG run"""
import nbformat
from nbclient import NotebookClient
from audit_source import ROOT

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
        'display(pd.read_csv(root / "reports/sql/01.csv"))\n'
        'display(pd.read_csv(root / "reports/sql/07.csv"))'),
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
notebook.cells[-1:-1] = cells_for_aggregation
NotebookClient(notebook, timeout=120, kernel_name='python3',
               resources={'metadata': {'path': str(ROOT)}}).execute()
folder = ROOT/'notebooks'
folder.mkdir(exist_ok=True)
nbformat.write(notebook, folder/'01_lfpg_pipeline.ipynb')
print('Executed notebook saved')
