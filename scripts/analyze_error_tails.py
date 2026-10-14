"""Распределение MAE заходов и вклад крупных ошибок фиксированных моделей"""
import os
import json
import math
import numpy as np
import pandas as pd
from audit_source import ROOT, digest
from airport_paths import paths
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'data/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def tail_summary(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Нужны конечные неотрицательные MAE заходов')
    count = math.ceil(.05*len(values))
    total = values.sum()
    return {'events':len(values), 'macro_mae_ft':values.mean(),
            'median_event_mae_ft':np.median(values),
            'p90_event_mae_ft':np.quantile(values,.9,method='linear'),
            'p95_event_mae_ft':np.quantile(values,.95,method='linear'),
            'worst_events':count, 'worst_fraction':count/len(values),
            'worst_share_of_sum_event_mae':np.sort(values)[-count:].sum()/total if total else 0.0}


def run():
    order = ['Constant vertical rate','Ridge','Polynomial Ridge','PyTorch MLP']
    summaries, paired, sources = [], [], {}
    fig,axes = plt.subplots(3,2,figsize=(13,11))
    for row,airport in enumerate(['LFPG','EGLL','EDDF']):
        folder = paths(airport)
        source = folder['data']/'event_errors.csv'
        events = pd.read_csv(source)
        frame = pd.read_parquet(folder['data']/'postgres_examples.parquet')
        expected = set(frame.loc[frame.split.eq('test'),'event_id'])
        reference = pd.read_csv(folder['reports']/'model_metrics.csv').set_index('model')
        if set(events.model) != set(order) or events.duplicated(['event_id','model']).any():
            raise ValueError('Неверный набор моделей или повторные ошибки захода')
        for name,color in zip(order,['#8193a5','#45a88a','#23876c','#ee806b']):
            part = events.loc[events.model.eq(name)]
            if set(part.event_id) != expected:
                raise ValueError('Набор заходов отличается от test')
            summary = tail_summary(part.mae_ft)
            if not np.isclose(summary['macro_mae_ft'],reference.loc[name,'macro_mae_ft'],atol=1e-5,rtol=1e-7):
                raise ValueError('Средняя MAE отличается от сохраненного отчета')
            summaries.append({'airport':airport,'model':name,**summary})
            values = np.sort(part.mae_ft.to_numpy())[::-1]
            cumulative = np.cumsum(values)/values.sum() if values.sum() else np.zeros(len(values))
            axes[row,0].plot(np.arange(1,len(values)+1)/len(values)*100,cumulative*100,color=color,label=name)
        matrix = events.pivot(index='event_id',columns='model',values='mae_ft')
        delta = matrix['PyTorch MLP']-matrix['Polynomial Ridge']
        # Худшие заходы определяет фиксированная модель сравнения, набор общий для обеих моделей
        count = math.ceil(.05*len(matrix))
        ranked = matrix.reset_index().sort_values(['Polynomial Ridge','event_id'],ascending=[False,True])
        worst_ids = ranked.iloc[:count].event_id
        worst = matrix.loc[worst_ids]
        rest = matrix.drop(index=worst_ids)
        paired.append({'airport':airport,'events':len(matrix),'mean_delta_mae_ft':delta.mean(),
                       'median_delta_mae_ft':delta.median(),
                       'better_events':int((delta < -1e-6).sum()),
                       'worse_events':int((delta > 1e-6).sum()),
                       'tied_events':int((abs(delta)<=1e-6).sum()),
                       'reference_worst_events':count,
                       'reference_worst_poly_mae_ft':worst['Polynomial Ridge'].mean(),
                       'reference_worst_mlp_mae_ft':worst['PyTorch MLP'].mean(),
                       'reference_worst_mean_delta_ft':(worst['PyTorch MLP']-worst['Polynomial Ridge']).mean(),
                       'remaining_mean_delta_ft':(rest['PyTorch MLP']-rest['Polynomial Ridge']).mean()})
        axes[row,0].set_title(f'{airport}: концентрация ошибки')
        axes[row,0].set_xlabel('Доля заходов от худших к лучшим, %')
        axes[row,0].set_ylabel('Доля суммы MAE заходов, %')
        axes[row,0].axvline(count/len(matrix)*100,color='#8193a5',linestyle='--')
        axes[row,0].set_xlim(0,100)
        axes[row,0].set_ylim(0,100)
        axes[row,1].scatter(matrix['Polynomial Ridge'],matrix['PyTorch MLP'],s=15,alpha=.45,color='#45a88a')
        limit = matrix[['Polynomial Ridge','PyTorch MLP']].to_numpy().max()*1.05
        axes[row,1].plot([0,limit],[0,limit],color='#8193a5',linestyle='--')
        axes[row,1].set_xlim(0,limit)
        axes[row,1].set_ylim(0,limit)
        axes[row,1].set_title(f'{airport}: один заход – одна точка')
        axes[row,1].set_xlabel('MAE Polynomial Ridge, ft')
        axes[row,1].set_ylabel('MAE PyTorch MLP, ft')
        sources[airport] = {'event_errors_sha256':digest(source),
                            'test_source_sha256':digest(folder['data']/'postgres_examples.parquet')}
    axes[0,0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT/'reports/error_tails.png',dpi=160)
    plt.close(fig)
    pd.DataFrame(summaries).to_csv(ROOT/'reports/error_tails.csv',index=False)
    pd.DataFrame(paired).to_csv(ROOT/'reports/error_tails_paired.csv',index=False)
    method = {'unit':'one event MAE, equal event weights','quantiles':'numpy linear interpolation',
              'worst_count':'ceil(0.05 * number of events)',
              'concentration':'sum of worst event MAEs / sum of all event MAEs, separately per model',
              'paired_subset':'same events ranked by Polynomial Ridge MAE, event_id breaks ties',
              'new_training':False,'hypothesis_testing':False,'sources':sources,
              'limitations':['descriptive analysis after test inspection','model-specific worst cohorts differ',
                             'reference-selected worst cohort is not an independent group',
                             'LFPG has only 55 test events; upper quantiles are fragile',
                             'event MAE tails are not instantaneous altitude-error tails']}
    (ROOT/'reports/error_tails_method.json').write_text(json.dumps(method,indent=2),encoding='utf-8')
    print(pd.DataFrame(summaries).to_string(index=False))
    print(pd.DataFrame(paired).to_string(index=False))


if __name__ == '__main__':
    run()
