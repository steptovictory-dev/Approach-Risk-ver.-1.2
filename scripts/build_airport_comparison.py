"""Выполненный ноутбук независимых экспериментов трех аэропортов"""
import os
import json
import pandas as pd
import numpy as np
import nbformat
from nbclient import NotebookClient
from audit_source import ROOT
from airport_paths import paths
from notebook_runtime import ProjectKernelManager
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
    cells += [nbformat.v4.new_markdown_cell('## Ошибки по высоте и удалению до порога\n\n'
        'Использованы сохраненные модели и заранее заданные для этой описательной диагностики интервалы. '
        'MAE сначала усредняется внутри захода и интервала, затем между представленными заходами. '
        'Один заход может присутствовать в нескольких интервалах; состав выборки меняется. '
        'Средние по интервалам нельзя складывать для восстановления общей MAE. '
        'Высота является барометрическим прокси, а не радиовысотой. Границы не являются эксплуатационными ограничениями'),
        nbformat.v4.new_code_cell('# Показываем ошибки, размеры групп и парные разницы фиксированных моделей\n'
            'display(pd.read_csv(root / "reports/phase_errors.csv"))\n'
            'display(pd.read_csv(root / "reports/phase_paired_errors.csv"))\n'
            'display(Image(filename=str(root / "reports/phase_mae.png")))\n'
            'display(Image(filename=str(root / "reports/phase_delta.png")))'),
        nbformat.v4.new_markdown_cell('В LFPG наибольший проигрыш MLP относительно Polynomial Ridge '
            'наблюдается в интервалах 2000–3000 ft и от 12 km. В EGLL средняя разница отрицательна '
            'во всех интервалах с большим числом заходов. В EDDF знак зависит от интервала. '
            'Группы от 3000 ft содержат лишь 4 захода EGLL и 6 EDDF; эти крайние значения '
            'не подтверждают устойчивую закономерность. Ниже 500 ft и ближе 2 km наблюдений нет. '
            'Интервалы неопределенности здесь не рассчитаны; анализ описательный, после просмотра test, '
            'без нового выбора параметров или признаков')]
    cells += [nbformat.v4.new_markdown_cell('## Распределение ошибок и вклад худших заходов\n\n'
        'Медиана, P90 и P95 рассчитаны по MAE отдельных заходов с равным весом. '
        'Квантили используют линейную интерполяцию. Число худших заходов равно ceil(0.05 × N): '
        '3 из 55 LFPG, 57 из 1136 EGLL, 27 из 540 EDDF. Поэтому фактическая доля '
        'может немного превышать 5%. Концентрация – доля суммы MAE заходов, '
        'не сумма ошибок всех прогнозных точек. Для каждой модели худшая группа определяется отдельно\n\n'
        'В парной таблице дополнительно сравнивается один общий набор худших заходов, '
        'выбранный по MAE Polynomial Ridge. Это зависимая от модели сравнения группа, '
        'а не независимая проверка преимущества MLP. Ниже диагонали на графике MLP лучше'),
        nbformat.v4.new_code_cell('# Показываем квантили, концентрацию ошибки и парное сравнение заходов\n'
            'display(pd.read_csv(root / "reports/error_tails.csv"))\n'
            'display(pd.read_csv(root / "reports/error_tails_paired.csv"))\n'
            'display(Image(filename=str(root / "reports/error_tails.png")))'),
        nbformat.v4.new_markdown_cell('В LFPG Polynomial Ridge лучше MLP по медиане, P90 и P95. '
            'В EGLL MLP уменьшает медиану с 14.00 до 13.11 ft и P90 с 22.49 до 21.77 ft; '
            'P95 меняется мало: 25.88 против 25.77 ft. В EDDF медианы практически совпадают '
            '(18.735 и 18.737 ft), P95 уменьшается с 38.76 до 37.85 ft. '
            'Квантили MAE моделей и медиана парных разниц отвечают на разные вопросы\n\n'
            'Худшие группы обучаемых моделей дают около 10–12% суммы MAE, для Ridge EDDF '
            'около 13%. На общей группе худших заходов Polynomial Ridge MLP уменьшает '
            'среднюю MAE в EGLL на 3.76 ft и EDDF на 3.07 ft; на остальных заходах '
            'средние разницы также отрицательны: −0.78 и −0.17 ft. В LFPG обе разницы '
            'положительны. Наблюдаемая средняя разница не определяется только этой худшей группой. '
            'Анализ выполнен после просмотра test, без интервалов неопределенности. '
            'P95 LFPG основан на 55 заходах и особенно чувствителен к составу выборки. '
            'Хвост MAE заходов не описывает хвост мгновенных ошибок высоты')]
    cells += [nbformat.v4.new_markdown_cell('## Чувствительность выбранной MLP к seed\n\n'
        'Для каждого аэропорта повторены пять запусков с seed 20261014–20261018. '
        'Архитектура зафиксирована по исходному выбору на validation: LFPG 9 → 16 → 16 → 1, '
        'EGLL и EDDF 9 → 32 → 32 → 1. Сохраняются исходные признаки, split, '
        'оптимизатор, веса заходов и правило ранней остановки. Меняются инициализация '
        'весов и порядок minibatch. Лучший epoch каждого запуска выбирается только по validation\n\n'
        'Все 15 обучений завершены перед новой оценкой test. Все seed представлены в отчетах; '
        'лучший по test не выбирается, исходные модели не заменены, ансамбль не строится. '
        'Исходный seed воспроизводит исходную MAE каждого аэропорта. Разброс между seed '
        'не является доверительным интервалом и не учитывает неопределенность новых дней. '
        'Это дополнительная диагностика после просмотра исходного test'),
        nbformat.v4.new_code_cell('# Показываем все запуски, разброс и разницы по тестовым дням\n'
            'display(pd.read_csv(root / "reports/seed_stability_runs.csv"))\n'
            'display(pd.read_csv(root / "reports/seed_stability_metrics.csv"))\n'
            'display(pd.read_csv(root / "reports/seed_stability_summary.csv"))\n'
            'seed_days = pd.read_csv(root / "reports/seed_stability_by_day.csv")\n'
            'display(seed_days)\n'
            '# Сопоставляем диапазоны пяти seed внутри каждого фиксированного дня\n'
            'display(seed_days.groupby(["airport", "day"]).agg(\n'
            '    mean_delta_ft=("delta_vs_polynomial_ft", "mean"),\n'
            '    min_delta_ft=("delta_vs_polynomial_ft", "min"),\n'
            '    max_delta_ft=("delta_vs_polynomial_ft", "max"),\n'
            '    seeds_better=("delta_vs_polynomial_ft", lambda values: int((values < 0).sum()))))\n'
            'display(Image(filename=str(root / "reports/seed_stability.png")))')]
    cells += [nbformat.v4.new_markdown_cell('Все пять seed сохраняют знак общей разницы: '
        'в LFPG MLP хуже Polynomial Ridge на 1.70–3.61 ft, в EGLL лучше на 0.65–0.98 ft, '
        'в EDDF лучше на 0.26–0.50 ft. Средняя MAE MLP между seed составляет '
        '19.45 ± 0.77 ft LFPG, 14.60 ± 0.15 ft EGLL и 20.50 ± 0.10 ft EDDF; '
        'после знака ± указано выборочное стандартное отклонение пяти запусков\n\n'
        'В EGLL разница отрицательна на каждом из трех дней при каждом seed. '
        'В EDDF майская разница отрицательна для всех seed, а в августе и октябре '
        'знак зависит от seed. В LFPG весенняя и летняя разницы положительны '
        'при всех seed, осенняя меняет знак. Устойчивость общего среднего '
        'не означает устойчивость каждого дня или перенос на новые дни')]
    cells += [nbformat.v4.new_markdown_cell('## Ограничения сравнения\n\n'
        'Размер и состав выборок различаются; абсолютная MAE не определяет сложность аэропорта. '
        'На сезон приходится один тестовый день, борта могут повторяться между split, '
        'фактический горизонт известен ретроспективно. Основные результаты относятся к исходному seed; '
        'дополнительная проверка пяти seed сохраняет те же дни и архитектуры. '
        'Парные интервалы оценены отдельно для каждого аэропорта и условны на наблюдаемых днях. '
        'Новые параметры по test не подбирались. Эксплуатационная применимость не установлена')]
    cells += [nbformat.v4.new_markdown_cell('## Совокупные исследовательские выводы\n\n'
        'Обучаемая поправка уменьшает среднюю ошибку относительно продолжения вертикальной '
        'скорости во всех трех исследованных аэропортах. Универсального преимущества MLP '
        'над Polynomial Ridge не установлено: LFPG поддерживает Polynomial Ridge, '
        'EGLL поддерживает MLP на трех наблюдаемых днях, EDDF показывает смену знака '
        'разницы по дням. Эти выводы относятся к выбранным погожим дням и полосам\n\n'
        'Следующая научная проверка должна заранее определить доступный на момент прогноза '
        'горизонт и независимые дни оценки. Текущие test-результаты не используются для '
        'нового подбора признаков или параметров. Go-around и эксплуатационная безопасность '
        'не оцениваются в этой работе'),
        nbformat.v4.new_code_cell('# Показываем проверку соответствия отчетов исходным MLflow runs\n'
            'display(pd.read_csv(root / "reports/tracking_review.csv"))')]
    notebook = nbformat.v4.new_notebook(cells=cells)
    NotebookClient(notebook,timeout=120,kernel_name='python3',kernel_manager_class=ProjectKernelManager,
                   resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(notebook,ROOT/'notebooks/02_three_airports.ipynb')
    print(combined.to_string(index=False))


if __name__ == '__main__':
    run()
