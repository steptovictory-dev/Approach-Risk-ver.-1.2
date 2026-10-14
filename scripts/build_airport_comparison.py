"""Выполненный ноутбук независимых экспериментов трех аэропортов"""
import os
import json
import pandas as pd
import numpy as np
import nbformat
from nbclient import NotebookClient
from audit_source import ROOT
from airport_paths import paths
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'data/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run():
    frames = []
    for airport in ['LFPG','EGLL','EDDF']:
        folder = paths(airport)
        result = pd.read_csv(folder['reports']/'model_metrics.csv')
        frames.append(result.assign(airport=airport))
        if airport != 'LFPG':
            # Сверяем классические результаты с завершенным исследованием 1.1
            source = ROOT.parent/f'Approach Risk ver. 1.1/research/{airport.lower()}-seasonal-learning/reports/csv/final_table.csv'
            reference = pd.read_csv(source,index_col=0)
            for name in ['Ridge','Polynomial Ridge','Constant vertical rate']:
                actual = result.set_index('model').loc[name,'macro_mae_ft']
                if not np.isclose(actual,reference.loc[name,'all'],atol=1e-6):
                    raise ValueError(f'Не совпал результат 1.1: {airport}, {name}')
    combined = pd.concat(frames,ignore_index=True)
    combined.to_csv(ROOT/'reports/three_airports_metrics.csv',index=False)
    fig,axes = plt.subplots(1,3,figsize=(12,4),sharey=True)
    order = ['Constant vertical rate','Ridge','Polynomial Ridge','PyTorch MLP']
    for ax,airport in zip(axes,['LFPG','EGLL','EDDF']):
        group = combined.loc[combined.airport.eq(airport)].set_index('model').loc[order]
        ax.bar(range(4),group.macro_mae_ft,color=['#8193a5','#45a88a','#23876c','#ee806b'])
        ax.set_xticks(range(4),['Baseline','Ridge','Poly Ridge','MLP'],rotation=35,ha='right')
        ax.set_title(airport)
    axes[0].set_ylabel('Средняя MAE заходов, ft')
    fig.tight_layout()
    fig.savefig(ROOT/'reports/three_airports_comparison.png',dpi=180)
    plt.close(fig)
    cells = [nbformat.v4.new_markdown_cell('# Сравнение независимых экспериментов LFPG, EGLL и EDDF\n\n'
        'Каждая модель обучена отдельно для своего аэропорта. Зафиксированы исходные сезонные '
        'дни 1.1: девять train, три validation и три test. Горизонт 27–33 секунды, '
        'поправка к физическому baseline, равный вклад заходов. Перенос обученной модели '
        'между аэропортами здесь не исследуется'),
        nbformat.v4.new_code_cell('# Загружаем сохраненные результаты без повторного обучения\n'
            'from pathlib import Path\nimport pandas as pd\nimport json\n'
            'from IPython.display import display, Image\nroot = Path.cwd()\n'
            'display(pd.read_csv(root / "reports/three_airports_metrics.csv"))\n'
            'display(Image(filename=str(root / "reports/three_airports_comparison.png")))')]
    for airport in ['EGLL','EDDF']:
        cells += [nbformat.v4.new_markdown_cell(f'## {airport}: выборка, загрузка и выбор модели\n\n'
            'Данные проходят те же проверки причинности и согласованности. Дополнительное поле '
            'airport_elevation_ft сохранено в PostgreSQL как метаданные. Повторная загрузка '
            'не добавляет строк. Все модели этого аэропорта оценены на одинаковом test. '
            'Классические MAE сверены с 1.1'),
            nbformat.v4.new_code_cell(f'# Показываем инвентаризацию, проверку импорта и выбор по validation\n'
                f'display(pd.read_csv(root / "reports/{airport}/split_inventory.csv"))\n'
                f'print(json.loads((root / "reports/{airport}/ingestion.json").read_text()))\n'
                f'display(pd.read_csv(root / "reports/{airport}/validation_search.csv"))\n'
                f'display(Image(filename=str(root / "reports/{airport}/mlp_validation.png")))')]
        cells += [nbformat.v4.new_markdown_cell(f'## {airport}: парный анализ ошибок и неопределенности\n\n'
            'Все сравнения выполнены на сохраненных моделях без переобучения. '
            'MAE, bias и MSE сначала усредняются внутри захода. Bias = прогноз минус наблюдение; '
            'положительный знак означает завышение высоты. P90 – 90-й процентиль MAE заходов. '
            'Сезон здесь представлен одним днем и не отделен от состава этого дня\n\n'
            'Парный bootstrap выбирает борта с возвращением, сохраняя все заходы выбранного '
            'борта вместе. 10 000 повторов, seed 20261014, границы 2.5% и 97.5%. '
            'Интервалы условны на трех наблюдаемых днях: общая зависимость внутри дня '
            'и неопределенность обучения не моделируются. Один seed MLP, без поправки '
            'на множественные сравнения; анализ выполнен после просмотра test'),
            nbformat.v4.new_code_cell('# Показываем ошибки по дням, парные разницы и интервалы\n'
                f'display(pd.read_csv(root / "reports/{airport}/errors_by_day.csv"))\n'
                f'display(pd.read_csv(root / "reports/{airport}/paired_uncertainty.csv"))\n'
                f'daily = pd.read_csv(root / "reports/{airport}/paired_errors_by_day.csv")\n'
                'display(daily.loc[daily.model.eq("PyTorch MLP") & daily.reference.eq("Polynomial Ridge")])\n'
                f'display(Image(filename=str(root / "reports/{airport}/error_analysis.png")))')]
        interpretation = ('MLP улучшила Polynomial Ridge на 777 из 1136 заходов; средняя разница '
            '−0.93 ft, условный интервал −1.13…−0.77 ft. Средние разницы на всех трех днях '
            'отрицательны: −1.70, −0.43, −0.65 ft. MLP в среднем занижает высоту на каждом '
            'из этих дней. Вывод ограничен выбранной выборкой и не подтверждает перенос '
            'на другие условия') if airport == 'EGLL' else (
            'MLP улучшила Polynomial Ridge на 294 из 540 заходов; средняя разница −0.31 ft, '
            'условный интервал −0.56…−0.06 ft. По дням знак меняется: май −1.40 ft, '
            'август +0.36 ft, октябрь +0.37 ft. Общий выигрыш определяется майским днем; '
            'преимущество не сохраняется во всех тестовых днях. MLP в среднем завышает '
            'высоту на всех трех днях. Интервал по бортам не устраняет зависимость '
            'результата от конкретного дня')
        cells += [nbformat.v4.new_markdown_cell(interpretation)]
    cells += [nbformat.v4.new_markdown_cell('## Ограничения сравнения\n\n'
        'Размер и состав выборок различаются; абсолютная MAE не определяет сложность аэропорта. '
        'На сезон приходится один тестовый день, борта могут повторяться между split, '
        'фактический горизонт известен ретроспективно. Для MLP использован один seed. '
        'Парные интервалы оценены отдельно для каждого аэропорта и условны на наблюдаемых днях. '
        'Новые параметры по test не подбирались. Эксплуатационная применимость не установлена')]
    notebook = nbformat.v4.new_notebook(cells=cells)
    NotebookClient(notebook,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(notebook,ROOT/'notebooks/02_three_airports.ipynb')
    print(combined.to_string(index=False))


if __name__ == '__main__':
    run()
