"""Пять фиксированных seed для выбранных архитектур MLP"""
import os
import json
import argparse
import subprocess
import importlib.metadata
import numpy as np
import pandas as pd
import torch
from audit_source import ROOT, FEATURES, digest
from airport_paths import paths
from train import mlp_fit, event_metrics
from model_artifacts import load_mlp
import mlflow
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'data/matplotlib'))
import matplotlib.pyplot as plt

PROTOCOL = ROOT/'reports/seed_stability_protocol.json'


def freeze_protocol():
    if PROTOCOL.exists():
        raise ValueError('Постановка уже зафиксирована')
    airports = {}
    for airport in ['LFPG','EGLL','EDDF']:
        folder = paths(airport)
        _,saved,_ = load_mlp(folder['artifacts']/'mlp.pt')
        airports[airport] = {'width':saved['width'],
                             'dataset_sha256':digest(folder['data']/'postgres_examples.parquet'),
                             'original_mlp_sha256':digest(folder['artifacts']/'mlp.pt'),
                             'reference_metrics_sha256':digest(folder['reports']/'model_metrics.csv')}
    value = {'seeds':[20261014,20261015,20261016,20261017,20261018],
             'airports':airports,'features':FEATURES,'threads':2,
             'architecture_selection':'original validation-selected width, fixed across seeds',
             'training':'same mlp_fit, Adam lr .001, weight decay .0001, batch 256, max 300, patience 30',
             'checkpoint_selection':'lowest validation event MAE, separately per seed',
             'test_use':'evaluation of every seed after all 15 fits, no seed selection',
             'scope':'post-test descriptive sensitivity, fixed observed days; not independent validation'}
    PROTOCOL.write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps(value,indent=2))


def load_protocol():
    value = json.loads(PROTOCOL.read_text())
    if value['features'] != FEATURES or len(set(value['seeds'])) != 5:
        raise ValueError('Неверная постановка эксперимента')
    for airport,config in value['airports'].items():
        folder = paths(airport)
        for key,path in [('dataset_sha256',folder['data']/'postgres_examples.parquet'),
                         ('original_mlp_sha256',folder['artifacts']/'mlp.pt'),
                         ('reference_metrics_sha256',folder['reports']/'model_metrics.csv')]:
            if digest(path) != config[key]:
                raise ValueError('Изменился фиксированный источник эксперимента')
    return value


def artifact_path(airport,seed):
    return ROOT/f'artifacts/seed_stability/{airport}/mlp_{seed}.pt'


def train_all():
    protocol = load_protocol()
    torch.set_num_threads(protocol['threads'])
    # Повторное обучение требует явного удаления отдельного каталога пользователем
    if any(artifact_path(a,s).exists() for a in protocol['airports'] for s in protocol['seeds']):
        raise ValueError('Артефакты этапа уже существуют, используйте --verify')
    sha = subprocess.run(['git','-c',f'safe.directory={ROOT.as_posix()}','rev-parse','HEAD'],
                         cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    versions = {p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','torch','mlflow']}
    source_hashes = {name:digest(ROOT/'scripts'/name) for name in ['train.py','seed_stability.py','preprocessing.py']}
    mlflow.set_tracking_uri('sqlite:///'+(ROOT/'mlflow.db').as_posix())
    mlflow.set_experiment('MLP fixed-architecture seed sensitivity')
    rows = []
    for airport,config in protocol['airports'].items():
        frame = pd.read_parquet(paths(airport)['data']/'postgres_examples.parquet')
        train = frame.loc[frame.split.eq('train')].copy()
        validation = frame.loc[frame.split.eq('validation')].copy()
        for seed in protocol['seeds']:
            print(f'TRAIN {airport} seed={seed}',flush=True)
            with mlflow.start_run(run_name=f'{airport} seed {seed}') as active:
                mlflow.log_params({'airport':airport,'seed':seed,'width':config['width'],
                                   'git_commit':sha,'protocol_sha256':digest(PROTOCOL),
                                   'dataset_sha256':config['dataset_sha256'],
                                   'package_versions':json.dumps(versions),'threads':protocol['threads'],
                                   'source_sha256':json.dumps(source_hashes),'working_tree_changes':True})
                _,score,history,model,scaler,scale = mlp_fit(train,validation,config['width'],seed=seed)
                path = artifact_path(airport,seed)
                path.parent.mkdir(parents=True,exist_ok=True)
                torch.save({'state_dict':model.state_dict(),'width':config['width'],'features':FEATURES,
                            'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),
                            'target_scale':scale,'seed':seed,'protocol_sha256':digest(PROTOCOL)},path)
                history_path = path.with_suffix('.csv')
                pd.DataFrame(history).to_csv(history_path,index=False)
                best_epoch = min(history,key=lambda item:item['validation_macro_mae_ft'])['epoch']
                mlflow.log_metrics({'validation_macro_mae_ft':score,'best_epoch':best_epoch,
                                    'stopped_epoch':history[-1]['epoch']})
                mlflow.log_artifact(str(path))
                mlflow.log_artifact(str(history_path))
                mlflow.log_artifact(str(PROTOCOL))
                for filename in source_hashes:
                    mlflow.log_artifact(str(ROOT/'scripts'/filename),artifact_path='training_source')
                rows.append({'airport':airport,'seed':seed,'width':config['width'],
                             'validation_macro_mae_ft':score,'best_epoch':best_epoch,
                             'stopped_epoch':history[-1]['epoch'],'run_id':active.info.run_id,
                             'artifact_sha256':digest(path)})
                print(f'DONE {airport} seed={seed} epoch={best_epoch}',flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'reports/seed_stability_runs.csv',index=False)
    evaluate(log_metrics=True)


def evaluate(log_metrics=False):
    protocol = load_protocol()
    torch.set_num_threads(protocol['threads'])
    runs = pd.read_csv(ROOT/'reports/seed_stability_runs.csv')
    expected = {(a,s) for a in protocol['airports'] for s in protocol['seeds']}
    if len(runs)!=15 or set(zip(runs.airport,runs.seed)) != expected:
        raise ValueError('Нужны все 15 запусков')
    all_rows,days = [], []
    mlflow.set_tracking_uri('sqlite:///'+(ROOT/'mlflow.db').as_posix())
    for airport,config in protocol['airports'].items():
        folder = paths(airport)
        frame = pd.read_parquet(folder['data']/'postgres_examples.parquet')
        test = frame.loc[frame.split.eq('test')].copy()
        validation = frame.loc[frame.split.eq('validation')].copy()
        reference = pd.read_csv(folder['reports']/'model_metrics.csv').set_index('model')
        daily_path = folder['reports']/('errors_by_season.csv' if airport=='LFPG' else 'errors_by_day.csv')
        daily_reference = pd.read_csv(daily_path)
        for seed in protocol['seeds']:
            row = runs.loc[runs.airport.eq(airport)&runs.seed.eq(seed)].iloc[0]
            path = artifact_path(airport,seed)
            if digest(path)!=row.artifact_sha256:
                raise ValueError('Изменился артефакт seed')
            _,saved,predict = load_mlp(path)
            if saved['protocol_sha256']!=digest(PROTOCOL) or saved['seed']!=seed or saved['width']!=config['width']:
                raise ValueError('Артефакт не соответствует постановке')
            if not np.isclose(event_metrics(validation,predict(validation))['macro_mae_ft'],
                              row.validation_macro_mae_ft,atol=1e-5,rtol=1e-7):
                raise ValueError('Не воспроизвелась validation MAE')
            prediction = predict(test)
            scores = event_metrics(test,prediction)
            # Исходный seed проверяет сохранение процедуры обучения
            if seed==20261014 and not np.isclose(scores['macro_mae_ft'],reference.loc['PyTorch MLP','macro_mae_ft'],atol=1e-5,rtol=1e-7):
                raise ValueError('Исходный seed не воспроизвел исходную MAE')
            all_rows.append({'airport':airport,'seed':seed,**scores,
                             'delta_vs_polynomial_ft':scores['macro_mae_ft']-reference.loc['Polynomial Ridge','macro_mae_ft']})
            for day,part in test.groupby('day'):
                key = 'season' if airport=='LFPG' else 'day'
                value = part.season.iloc[0] if airport=='LFPG' else day
                base = daily_reference.loc[daily_reference.model.eq('Polynomial Ridge')&daily_reference[key].eq(value),'macro_mae_ft'].item()
                score = event_metrics(part,predict(part))['macro_mae_ft']
                days.append({'airport':airport,'seed':seed,'day':day,'events':part.event_id.nunique(),
                             'macro_mae_ft':score,'polynomial_macro_mae_ft':base,'delta_vs_polynomial_ft':score-base})
            if log_metrics:
                with mlflow.start_run(run_id=row.run_id):
                    mlflow.log_metrics(scores)
                    mlflow.log_metric('delta_vs_polynomial_ft',all_rows[-1]['delta_vs_polynomial_ft'])
    results = pd.DataFrame(all_rows)
    daily = pd.DataFrame(days)
    results.to_csv(ROOT/'reports/seed_stability_metrics.csv',index=False)
    daily.to_csv(ROOT/'reports/seed_stability_by_day.csv',index=False)
    summary = results.groupby('airport').agg(seeds=('seed','size'),mean_mae_ft=('macro_mae_ft','mean'),
        sd_mae_ft=('macro_mae_ft','std'),min_mae_ft=('macro_mae_ft','min'),max_mae_ft=('macro_mae_ft','max'),
        mean_delta_ft=('delta_vs_polynomial_ft','mean'),min_delta_ft=('delta_vs_polynomial_ft','min'),
        max_delta_ft=('delta_vs_polynomial_ft','max')).reset_index()
    summary['seeds_better_than_polynomial'] = summary.airport.map(results.groupby('airport').delta_vs_polynomial_ft.apply(lambda x:int((x<0).sum())))
    summary.to_csv(ROOT/'reports/seed_stability_summary.csv',index=False)
    fig,axes = plt.subplots(1,3,figsize=(13,4))
    for ax,airport in zip(axes,protocol['airports']):
        group = daily.loc[daily.airport.eq(airport)]
        labels = sorted(group.day.unique())
        for seed,color in zip(protocol['seeds'],['#ee806b','#45a88a','#23876c','#bc6255','#8193a5']):
            part = group.loc[group.seed.eq(seed)].set_index('day').loc[labels]
            ax.plot(range(3),part.delta_vs_polynomial_ft,marker='o',label=str(seed),alpha=.8,color=color)
        ax.axhline(0,color='#8193a5',linestyle='--')
        ax.set_xticks(range(3),labels,rotation=25,ha='right')
        ax.set_title(airport)
        ax.set_ylabel('MAE MLP − MAE Polynomial Ridge, ft')
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT/'reports/seed_stability.png',dpi=160)
    plt.close(fig)
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--freeze',action='store_true')
    group.add_argument('--verify',action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze_protocol()
    elif args.verify:
        evaluate()
    else:
        train_all()
