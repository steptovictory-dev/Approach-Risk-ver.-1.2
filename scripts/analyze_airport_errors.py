"""Парный анализ фиксированных моделей EGLL и EDDF"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from airport_paths import paths
from audit_source import ROOT, FEATURES, digest
from model_artifacts import load_mlp
from train import event_metrics
from uncertainty import clustered_bootstrap, REPETITIONS, SEED
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'data/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def analyze(airport):
    folders = paths(airport)
    frame = pd.read_parquet(folders['data']/'postgres_examples.parquet')
    test = frame.loc[frame.split.eq('test')].copy()
    metadata = test[['event_id','icao24','day','season']].drop_duplicates()
    if metadata.event_id.duplicated().any():
        raise ValueError('Неоднозначные метаданные заходов')
    predictions = {'Constant vertical rate':test.baseline_ft.to_numpy()}
    for name,filename in [('Ridge','ridge.joblib'),('Polynomial Ridge','polynomial_ridge.joblib')]:
        saved = joblib.load(folders['artifacts']/filename)
        predictions[name] = test.baseline_ft.to_numpy()+saved['model'].predict(test[FEATURES])
    _,_,predict = load_mlp(folders['artifacts']/'mlp.pt')
    predictions['PyTorch MLP'] = predict(test)
    reference = pd.read_csv(folders['reports']/'model_metrics.csv').set_index('model')
    records = []
    for name,prediction in predictions.items():
        for key,value in event_metrics(test,prediction).items():
            if not np.isclose(value,reference.loc[name,key],atol=1e-5,rtol=1e-7):
                raise ValueError('Исходные метрики не воспроизведены')
        errors = prediction-test.target_height_ft.to_numpy()
        rows = test[['event_id','day','season','icao24']].copy()
        rows['absolute'] = abs(errors)
        rows['signed'] = errors
        rows['squared'] = errors**2
        events = rows.groupby(['event_id','day','season','icao24']).agg(
            mae_ft=('absolute','mean'),bias_ft=('signed','mean'),mse_ft=('squared','mean')).reset_index()
        records.append(events.assign(model=name))
    events = pd.concat(records,ignore_index=True)
    # Подробные ошибки с идентификаторами остаются в локальной папке данных
    events.to_csv(folders['data']/'event_errors.csv',index=False)
    summaries = []
    for (name,day,season),group in events.groupby(['model','day','season']):
        summaries.append({'model':name,'day':day,'season':season,'events':len(group),
                          'macro_mae_ft':group.mae_ft.mean(),'macro_bias_ft':group.bias_ft.mean(),
                          'macro_rmse_ft':np.sqrt(group.mse_ft.mean()),
                          'p90_event_mae_ft':group.mae_ft.quantile(.9)})
    summary = pd.DataFrame(summaries)
    summary.to_csv(folders['reports']/'errors_by_day.csv',index=False)
    matrix = events.pivot(index='event_id',columns='model',values='mae_ft')
    meta = metadata.set_index('event_id').loc[matrix.index]
    paired, daily = [], []
    for model,baseline in [('Polynomial Ridge','Constant vertical rate'),('PyTorch MLP','Constant vertical rate'),
                           ('PyTorch MLP','Polynomial Ridge'),('Ridge','Polynomial Ridge')]:
        delta = matrix[model]-matrix[baseline]
        draws = clustered_bootstrap(delta.to_numpy(),meta.icao24.to_numpy())
        lower,upper = np.quantile(draws,[.025,.975])
        paired.append({'model':model,'reference':baseline,'events':len(delta),'aircraft':meta.icao24.nunique(),
                       'mean_delta_mae_ft':delta.mean(),'lower_95_ft':lower,'upper_95_ft':upper,
                       'better_events':int((delta < -1e-6).sum()),'worse_events':int((delta > 1e-6).sum()),
                       'tied_events':int((abs(delta)<=1e-6).sum())})
        for day in sorted(meta.day.unique()):
            part = delta.loc[meta.day.eq(day)]
            daily.append({'day':day,'model':model,'reference':baseline,'events':len(part),
                          'mean_delta_mae_ft':part.mean(),'better_events':int((part < -1e-6).sum())})
    paired = pd.DataFrame(paired)
    paired.to_csv(folders['reports']/'paired_uncertainty.csv',index=False)
    pd.DataFrame(daily).to_csv(folders['reports']/'paired_errors_by_day.csv',index=False)
    report = {'airport':airport,'test_days':meta.day.nunique(),'events':len(meta),
              'aircraft':meta.icao24.nunique(),'aircraft_on_multiple_days':int(meta.groupby('icao24').day.nunique().gt(1).sum()),
              'method':'paired aircraft-cluster percentile bootstrap','repetitions':REPETITIONS,'seed':SEED,
              'conditional_on_observed_days':True,'day_dependence_modeled':False,'retrained':False,
              'artifact_sha256':{f:digest(folders['artifacts']/f) for f in ['ridge.joblib','polynomial_ridge.joblib','mlp.pt']}}
    (folders['reports']/'error_analysis_method.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    fig,axes = plt.subplots(1,2,figsize=(12,4))
    order = list(predictions)
    for name,color in zip(order,['#8193a5','#45a88a','#23876c','#ee806b']):
        group = summary.loc[summary.model.eq(name)].sort_values('day')
        axes[0].plot(group.day,group.macro_mae_ft,marker='o',label=name,color=color)
        values = np.sort(events.loc[events.model.eq(name),'mae_ft'])
        axes[1].step(values,np.arange(1,len(values)+1)/len(values),where='post',color=color,label=name)
    axes[0].set_ylabel('Средняя MAE заходов, ft')
    axes[0].tick_params(axis='x',rotation=20)
    axes[1].set_xlabel('MAE отдельного захода, ft')
    axes[1].set_ylabel('Доля заходов с MAE не выше значения')
    axes[1].legend(fontsize=8)
    fig.suptitle(airport)
    fig.tight_layout()
    fig.savefig(folders['reports']/'error_analysis.png',dpi=160)
    plt.close(fig)
    print(airport)
    print(paired.to_string(index=False))
    print(pd.DataFrame(daily).query("model == 'PyTorch MLP' and reference == 'Polynomial Ridge'").to_string(index=False))


if __name__ == '__main__':
    for airport in ['EGLL','EDDF']:
        analyze(airport)
