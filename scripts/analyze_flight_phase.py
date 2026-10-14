"""Описательная диагностика ошибок по высоте и удалению до порога"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from audit_source import ROOT, FEATURES, digest
from airport_paths import paths
from model_artifacts import load_mlp
from train import event_metrics
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'data/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INTERVALS = {
    'height_ft': ([-np.inf,500,1000,1500,2000,3000,np.inf],
                  ['<500','500–1000','1000–1500','1500–2000','2000–3000','≥3000']),
    'distance_m': ([-np.inf,2000,4000,6000,8000,12000,np.inf],
                   ['<2','2–4','4–6','6–8','8–12','≥12'])}


def summarize_bins(frame, prediction, feature):
    boundaries,labels = INTERVALS[feature]
    groups = pd.cut(frame[feature],boundaries,right=False,labels=labels)
    errors = prediction-frame.target_height_ft.to_numpy()
    records = frame[['event_id']].copy()
    records['bin'] = groups
    records['absolute'] = abs(errors)
    records['signed'] = errors
    # Один заход имеет одинаковый вклад внутри каждого интервала, где он представлен
    return records.groupby(['event_id','bin'],observed=True).agg(
        mae_ft=('absolute','mean'),bias_ft=('signed','mean'),pairs=('absolute','size')).reset_index()


def run():
    rows, paired = [], []
    order = ['Constant vertical rate','Ridge','Polynomial Ridge','PyTorch MLP']
    for airport in ['LFPG','EGLL','EDDF']:
        folder = paths(airport)
        frame = pd.read_parquet(folder['data']/'postgres_examples.parquet')
        test = frame.loc[frame.split.eq('test')].copy()
        predictions = {'Constant vertical rate':test.baseline_ft.to_numpy()}
        for name,filename in [('Ridge','ridge.joblib'),('Polynomial Ridge','polynomial_ridge.joblib')]:
            bundle = joblib.load(folder['artifacts']/filename)
            predictions[name] = test.baseline_ft.to_numpy()+bundle['model'].predict(test[FEATURES])
        _,_,predict = load_mlp(folder['artifacts']/'mlp.pt')
        predictions['PyTorch MLP'] = predict(test)
        reference = pd.read_csv(folder['reports']/'model_metrics.csv').set_index('model')
        for name,prediction in predictions.items():
            for key,value in event_metrics(test,prediction).items():
                if not np.isclose(value,reference.loc[name,key],atol=1e-5,rtol=1e-7):
                    raise ValueError('Метрики фиксированных моделей не совпали')
        for feature in INTERVALS:
            events = []
            for name,prediction in predictions.items():
                part = summarize_bins(test,prediction,feature)
                if part.pairs.sum() != len(test):
                    raise ValueError('Интервалы не покрывают все наблюдения')
                events.append(part.assign(model=name))
                for label,group in part.groupby('bin',observed=True):
                    rows.append({'airport':airport,'feature':feature,'bin':str(label),'model':name,
                                 'events':len(group),'pairs':int(group.pairs.sum()),
                                 'macro_mae_ft':group.mae_ft.mean(),'macro_bias_ft':group.bias_ft.mean()})
            combined = pd.concat(events,ignore_index=True)
            combined.to_csv(folder['data']/f'phase_errors_{feature}.csv',index=False)
            matrix = combined.pivot(index=['event_id','bin'],columns='model',values='mae_ft')
            delta = matrix['PyTorch MLP']-matrix['Polynomial Ridge']
            for label,group in delta.groupby(level='bin',observed=True):
                paired.append({'airport':airport,'feature':feature,'bin':str(label),'events':len(group),
                               'delta_mae_ft':group.mean(),'better_events':int((group < -1e-6).sum()),
                               'worse_events':int((group > 1e-6).sum())})
    result = pd.DataFrame(rows)
    differences = pd.DataFrame(paired)
    result.to_csv(ROOT/'reports/phase_errors.csv',index=False)
    differences.to_csv(ROOT/'reports/phase_paired_errors.csv',index=False)
    for mode in ['mae','delta']:
        fig,axes = plt.subplots(3,2,figsize=(13,11))
        for row,airport in enumerate(['LFPG','EGLL','EDDF']):
            for col,feature in enumerate(INTERVALS):
                ax = axes[row,col]
                labels = INTERVALS[feature][1]
                if mode == 'mae':
                    for name,color in zip(order,['#8193a5','#45a88a','#23876c','#ee806b']):
                        group = result.loc[result.airport.eq(airport)&result.feature.eq(feature)&result.model.eq(name)].set_index('bin').reindex(labels)
                        ax.plot(range(len(labels)),group.macro_mae_ft,marker='o',color=color,label=name)
                else:
                    group = differences.loc[differences.airport.eq(airport)&differences.feature.eq(feature)].set_index('bin').reindex(labels)
                    ax.plot(range(len(labels)),group.delta_mae_ft,marker='o',color='#ee806b')
                    ax.axhline(0,color='#8193a5',linestyle='--')
                ax.set_title(airport)
                counts = differences.loc[differences.airport.eq(airport)&differences.feature.eq(feature)].set_index('bin').reindex(labels)
                for position,label in enumerate(labels):
                    count = counts.loc[label,'events']
                    if pd.notna(count) and count < 10:
                        ax.text(position,0.96,f'n={int(count)}',transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=9)
                ax.set_xticks(range(len(labels)),labels,rotation=25,ha='right')
                ax.set_xlabel('Высота-прокси на момент прогноза, ft' if feature=='height_ft' else 'Удаление до порога на момент прогноза, km')
                ax.set_ylabel('Средняя MAE заходов, ft' if mode=='mae' else 'MAE MLP − MAE Polynomial Ridge, ft')
        if mode=='mae':
            axes[0,1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(ROOT/f'reports/phase_{mode}.png',dpi=160)
        plt.close(fig)
    provenance = {'interval_rule':'fixed descriptive bins, left-closed/right-open',
                  'height_edges_ft':[500,1000,1500,2000,3000],'distance_edges_m':[2000,4000,6000,8000,12000],
                  'unit':'mean within event and bin, then mean across events represented in bin',
                  'models_frozen':True,'hypothesis_testing':False,
                  'sparse_group_annotation':'n < 10 marked on plots; all observations retained',
                  'limitations':['cohort varies by bin','same events can appear in several bins',
                                 'barometric height proxy, not radio altitude','not operational gates'],
                  'input_sha256':{a:digest(paths(a)['data']/'postgres_examples.parquet') for a in ['LFPG','EGLL','EDDF']}}
    (ROOT/'reports/phase_method.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(differences.to_string(index=False))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run()
