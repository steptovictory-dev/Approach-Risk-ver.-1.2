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
    cells += [nbformat.v4.new_markdown_cell('## Ограничения сравнения\n\n'
        'Размер и состав выборок различаются; абсолютная MAE не определяет сложность аэропорта. '
        'На сезон приходится один тестовый день, борта могут повторяться между split, '
        'фактический горизонт известен ретроспективно. Для MLP использован один seed. '
        'Сезонный анализ ошибок и интервалы LFPG не переносятся автоматически на EGLL/EDDF. '
        'Новые параметры по test не подбирались. Эксплуатационная применимость не установлена')]
    notebook = nbformat.v4.new_notebook(cells=cells)
    NotebookClient(notebook,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(notebook,ROOT/'notebooks/02_three_airports.ipynb')
    print(combined.to_string(index=False))


if __name__ == '__main__':
    run()
