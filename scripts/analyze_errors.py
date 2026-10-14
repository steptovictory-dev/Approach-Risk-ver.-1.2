"""Анализ ошибок фиксированных моделей без повторного обучения"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from audit_source import ROOT, FEATURES, digest
from train import event_metrics
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'data/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run():
    test = pd.read_parquet(ROOT/'data/postgres_examples.parquet').query("split == 'test'").copy()
    source = ROOT.parent/'Approach Risk ver. 1.1/research/lfpg-seasonal-learning/artifacts/models.joblib'
    bundle = joblib.load(source)
    if bundle['features'] != FEATURES:
        raise ValueError('Набор признаков сохраненных моделей отличается')
    saved = torch.load(ROOT/'artifacts/mlp.pt', map_location='cpu', weights_only=True)
    width = saved['width']
    network = nn.Sequential(nn.Linear(len(FEATURES),width),nn.ReLU(),
                            nn.Linear(width,width),nn.ReLU(),nn.Linear(width,1))
    network.load_state_dict(saved['state_dict'])
    network.eval()
    x = (test[FEATURES].to_numpy()-np.array(saved['scaler_mean']))/np.array(saved['scaler_scale'])
    with torch.no_grad():
        neural = test.baseline_ft.to_numpy()+network(torch.tensor(x,dtype=torch.float32)).squeeze(1).numpy()*saved['target_scale']
    predictions = {'Constant vertical rate': test.baseline_ft.to_numpy(), 'PyTorch MLP': neural}
    search = pd.read_csv(ROOT/'reports/validation_search.csv')
    for name in ['Ridge','Polynomial Ridge']:
        selected = search.loc[search.model.eq(name)].sort_values('validation_macro_mae_ft').iloc[0]
        if bundle['parameters']['pooled',name]['alpha'] != selected.parameter:
            raise ValueError('Параметры исходной модели не совпадают с выбором 1.2')
        predictions[name] = test.baseline_ft.to_numpy()+bundle['models']['pooled',name].predict(test[FEATURES])
    reference = pd.read_csv(ROOT/'reports/model_metrics.csv').set_index('model')
    records = []
    for name, prediction in predictions.items():
        # Совпадение итоговых метрик проверяем до интерпретации ошибок
        measured = event_metrics(test,prediction)
        for metric,value in measured.items():
            if not np.isclose(value,reference.loc[name,metric],rtol=1e-8,atol=1e-6):
                raise ValueError(f'Не совпала метрика {name}: {metric}')
        errors = prediction-test.target_height_ft.to_numpy()
        rows = test[['event_id','day','season']].copy()
        rows['absolute'] = abs(errors)
        rows['signed'] = errors
        rows['squared'] = errors**2
        events = rows.groupby(['event_id','day','season']).agg(
            mae_ft=('absolute','mean'),bias_ft=('signed','mean'),mse_ft=('squared','mean'),pairs=('absolute','size')).reset_index()
        events['model'] = name
        records.append(events)
    events = pd.concat(records,ignore_index=True)
    # Подробные ошибки с идентификаторами сохраняем только локально
    events.to_csv(ROOT/'data/event_errors.csv',index=False)
    summaries = []
    for (name,season),group in events.groupby(['model','season']):
        summaries.append({'model':name,'season':season,'events':len(group),
                          'macro_mae_ft':group.mae_ft.mean(),'macro_bias_ft':group.bias_ft.mean(),
                          'macro_rmse_ft':np.sqrt(group.mse_ft.mean()),
                          'median_event_mae_ft':group.mae_ft.median(),
                          'p90_event_mae_ft':group.mae_ft.quantile(.9)})
    summary = pd.DataFrame(summaries)
    summary.to_csv(ROOT/'reports/errors_by_season.csv',index=False)
    order = ['Constant vertical rate','Ridge','Polynomial Ridge','PyTorch MLP']
    pivot = events.pivot(index='event_id',columns='model',values='mae_ft')
    paired = []
    for name in order[1:]:
        for baseline in ['Constant vertical rate','Polynomial Ridge']:
            if name == baseline:
                continue
            delta = pivot[name]-pivot[baseline]
            paired.append({'model':name,'reference':baseline,'events':len(delta),
                           'mean_delta_mae_ft':delta.mean(),'median_delta_mae_ft':delta.median(),
                           'better_events':int((delta < -1e-6).sum()),'worse_events':int((delta > 1e-6).sum())})
    pd.DataFrame(paired).to_csv(ROOT/'reports/paired_event_errors.csv',index=False)
    provenance = {'classical_models_source':str(source.relative_to(ROOT.parent)),
                  'classical_models_sha256':digest(source),'mlp_sha256':digest(ROOT/'artifacts/mlp.pt'),
                  'test_data_sha256':digest(ROOT/'data/postgres_examples.parquet'),
                  'aggregate_metrics_match':True,'retrained':False}
    (ROOT/'reports/error_analysis_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    colors = ['#8193a5','#45a88a','#23876c','#ee806b']
    fig,axes = plt.subplots(1,2,figsize=(12,4))
    for name,color in zip(order,colors):
        group = summary.set_index(['model','season']).loc[name].reindex(['spring','summer','autumn'])
        axes[0].plot(['Весна','Лето','Осень'],group.macro_mae_ft,marker='o',label=name,color=color)
        values = np.sort(events.loc[events.model.eq(name),'mae_ft'])
        axes[1].step(values,np.arange(1,len(values)+1)/len(values),where='post',label=name,color=color)
    axes[0].set_ylabel('Средняя MAE заходов, ft')
    axes[1].set_xlabel('MAE отдельного захода, ft')
    axes[1].set_ylabel('Доля заходов с MAE не выше значения')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT/'reports/error_analysis.png',dpi=180)
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    run()
