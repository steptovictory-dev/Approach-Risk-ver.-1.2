"""Парный кластерный bootstrap ошибок на наблюдаемых тестовых днях"""
import os
import json
import numpy as np
import pandas as pd
from audit_source import ROOT
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'data/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 20261014
REPETITIONS = 10000


def clustered_bootstrap(values, clusters, repetitions=REPETITIONS, seed=SEED):
    # Сохраняем все заходы выбранного борта вместе, не разрывая парность моделей
    groups = pd.DataFrame({'value':values,'cluster':clusters}).groupby('cluster').value.agg(['sum','count'])
    generator = np.random.default_rng(seed)
    indexes = generator.integers(0,len(groups),size=(repetitions,len(groups)))
    sums = groups['sum'].to_numpy()[indexes].sum(axis=1)
    counts = groups['count'].to_numpy()[indexes].sum(axis=1)
    return sums/counts


def run():
    source = pd.read_parquet(ROOT/'data/postgres_examples.parquet')
    test = source.loc[source.split.eq('test')]
    metadata = test[['event_id','icao24','day','season']].drop_duplicates()
    if metadata.event_id.duplicated().any():
        raise ValueError('Неоднозначные метаданные заходов')
    errors = pd.read_csv(ROOT/'data/event_errors.csv')
    matrix = errors.pivot(index='event_id',columns='model',values='mae_ft')
    meta = metadata.set_index('event_id').loc[matrix.index]
    rows = []
    for model, reference in [('Polynomial Ridge','Constant vertical rate'),
                             ('PyTorch MLP','Constant vertical rate'),
                             ('PyTorch MLP','Polynomial Ridge'),
                             ('Ridge','Polynomial Ridge')]:
        delta = matrix[model]-matrix[reference]
        draws = clustered_bootstrap(delta.to_numpy(),meta.icao24.to_numpy())
        lower,upper = np.quantile(draws,[.025,.975])
        rows.append({'model':model,'reference':reference,'mean_delta_mae_ft':delta.mean(),
                     'lower_95_ft':lower,'upper_95_ft':upper,'clusters':meta.icao24.nunique(),
                     'events':len(matrix),'repetitions':REPETITIONS,'seed':SEED})
    result = pd.DataFrame(rows)
    result.to_csv(ROOT/'reports/paired_uncertainty.csv',index=False)
    inventory = metadata.groupby(['day','season']).agg(events=('event_id','size'),aircraft=('icao24','nunique')).reset_index()
    inventory.to_csv(ROOT/'reports/test_cluster_inventory.csv',index=False)
    cross = int(metadata.groupby('icao24').day.nunique().gt(1).sum())
    report = {'method':'paired aircraft-cluster percentile bootstrap',
              'test_days':metadata.day.nunique(),'aircraft':metadata.icao24.nunique(),
              'aircraft_on_multiple_test_days':cross,'events':len(metadata),
              'repetitions':REPETITIONS,'seed':SEED,
              'scope':'conditional on the three observed test days; no day-level generalization',
              'limitations':['shared day dependence not modeled','one fitted MLP seed',
                             'training uncertainty not modeled','no multiplicity correction',
                             'exploratory comparisons after test inspection']}
    (ROOT/'reports/uncertainty_method.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    fig,ax = plt.subplots(figsize=(9,4))
    for index,row in result.iterrows():
        ax.plot([row.lower_95_ft,row.upper_95_ft],[index,index],color='#45a88a',linewidth=3)
        ax.plot(row.mean_delta_mae_ft,index,'o',color='#ee806b')
    ax.set_yticks(range(len(result)),[f'{r.model} − {r.reference}' for r in result.itertuples()])
    ax.axvline(0,color='#8193a5',linestyle='--')
    ax.set_xlabel('Разница MAE заходов, ft; ниже нуля – лучше модели слева')
    ax.set_title('95% интервалы по бортам\nУсловно на трех тестовых днях')
    fig.tight_layout()
    fig.savefig(ROOT/'reports/paired_uncertainty.png',dpi=180)
    plt.close(fig)
    print(inventory.to_string(index=False))
    print(result.to_string(index=False))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    run()
