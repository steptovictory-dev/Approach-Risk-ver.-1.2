"""Сравнение способов усреднения ошибки физического baseline"""
import os
from audit_source import ROOT
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'data/matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from query_reports import run


def main():
    run('event_mae.sql', 'event_mae_')
    summary = pd.read_csv(ROOT/'reports/sql/event_mae_01.csv')
    sizes = pd.read_csv(ROOT/'reports/sql/event_mae_02.csv')
    frame = pd.read_parquet(ROOT/'data/postgres_examples.parquet')
    # Независимо сверяем результат SQL с расчетом по извлеченным данным
    for row in summary.itertuples(index=False):
        part = frame.loc[frame.split.eq(row.split)]
        errors = (part.target_height_ft-part.baseline_ft).abs()
        macro = errors.groupby(part.event_id).mean().mean()
        if not np.isclose(macro, row.event_macro_mae_ft) or not np.isclose(errors.mean(), row.row_mae_ft):
            raise ValueError('Расчеты SQL и Python расходятся')
    labels = {'train': 'Обучение', 'validation': 'Валидация', 'test': 'Тест'}
    colors = {'train': '#45a88a', 'validation': '#8193a5', 'test': '#ee806b'}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)
    bins = np.arange(0, sizes.pairs.max()+11, 10)
    for ax, split in zip(axes, ['train','validation','test']):
        group = sizes.loc[sizes.split.eq(split)]
        # Частоты из SQL используются как веса гистограммы
        ax.hist(group.pairs, bins=bins, weights=group.events, color=colors[split])
        ax.set_title(labels[split])
        ax.set_xlabel('Пар наблюдений на заход')
    axes[0].set_ylabel('Число заходов')
    fig.tight_layout()
    fig.savefig(ROOT/'reports/event_pair_distribution.png', dpi=180)
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
