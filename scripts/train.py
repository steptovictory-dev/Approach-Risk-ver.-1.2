"""Compare residual models using validation selection and event weights"""
from pathlib import Path
import copy
import json
import subprocess
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import mlflow
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_source import FEATURES, ROOT, digest

SEED = 20261014


def event_metrics(frame, prediction):
    # Average within each event first, then give every event equal weight
    errors = pd.DataFrame({'event': frame.event_id.to_numpy(),
                           'absolute': np.abs(prediction-frame.target_height_ft.to_numpy()),
                           'squared': (prediction-frame.target_height_ft.to_numpy())**2})
    means = errors.groupby('event').mean()
    return {'macro_mae_ft': float(means.absolute.mean()),
            'macro_rmse_ft': float(np.sqrt(means.squared.mean()))}


def weights(frame):
    value = 1 / frame.groupby('event_id').event_id.transform('size').to_numpy()
    return value / value.mean()


def mlp_fit(train, validation, width):
    # Fit all transformations only on train
    torch.manual_seed(SEED)
    scaler = StandardScaler().fit(train[FEATURES], sample_weight=weights(train))
    target_scale = max(float(np.sqrt(np.average(train.residual_ft.to_numpy()**2,
                                               weights=weights(train)))), 1.0)
    x = torch.tensor(scaler.transform(train[FEATURES]), dtype=torch.float32)
    y = torch.tensor(train.residual_ft.to_numpy()/target_scale, dtype=torch.float32)
    w = torch.tensor(weights(train), dtype=torch.float32)
    vx = torch.tensor(scaler.transform(validation[FEATURES]), dtype=torch.float32)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(TensorDataset(x, y, w), batch_size=256, shuffle=True,
                        generator=generator)
    model = nn.Sequential(nn.Linear(len(FEATURES), width), nn.ReLU(),
                          nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    best, state, stale, history = float('inf'), None, 0, []
    for epoch in range(1, 301):
        model.train()
        total = 0.0
        for bx, by, bw in loader:
            optimizer.zero_grad()
            # Normalized event weights preserve equal event contribution in expectation
            loss = (((model(bx).squeeze(1)-by)**2)*bw).mean()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())*len(bx)
        model.eval()
        with torch.no_grad():
            prediction = validation.baseline_ft.to_numpy()+model(vx).squeeze(1).numpy()*target_scale
        score = event_metrics(validation, prediction)['macro_mae_ft']
        history.append({'epoch': epoch, 'train_weighted_mse': total/len(train),
                        'validation_macro_mae_ft': score, 'width': width})
        if score < best:
            best, state, stale = score, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= 30:
            break
    model.load_state_dict(state)
    model.eval()
    def predict(frame):
        with torch.no_grad():
            values = torch.tensor(scaler.transform(frame[FEATURES]), dtype=torch.float32)
            return frame.baseline_ft.to_numpy()+model(values).squeeze(1).numpy()*target_scale
    return predict, best, history, model, scaler, target_scale


def run():
    np.random.seed(SEED)
    torch.set_num_threads(2)
    path = ROOT / 'data/postgres_examples.parquet'
    if not path.exists():
        raise RuntimeError('Run PostgreSQL ingestion and extraction before training')
    frame = pd.read_parquet(path)
    train = frame.loc[frame.split.eq('train')].copy()
    validation = frame.loc[frame.split.eq('validation')].copy()
    test = frame.loc[frame.split.eq('test')].copy()
    if frame.groupby(['airport', 'event_id']).split.nunique().ne(1).any():
        raise ValueError('An event crosses splits')
    reports, artifacts = ROOT/'reports', ROOT/'artifacts'
    reports.mkdir(exist_ok=True)
    artifacts.mkdir(exist_ok=True)
    models = [('Constant vertical rate', lambda f: f.baseline_ft.to_numpy(), {})]
    search = []
    for family in ['Ridge', 'Polynomial Ridge']:
        candidates = []
        for alpha in [1, 10, 100]:
            steps = [('scale', StandardScaler())]
            if family == 'Polynomial Ridge':
                steps += [('poly', PolynomialFeatures(2, include_bias=False)),
                          ('expanded_scale', StandardScaler())]
            steps += [('regressor', Ridge(alpha=alpha, solver='svd'))]
            model = Pipeline(steps)
            params = {'scale__sample_weight': weights(train),
                      'regressor__sample_weight': weights(train)}
            if family == 'Polynomial Ridge':
                params['expanded_scale__sample_weight'] = weights(train)
            model.fit(train[FEATURES], train.residual_ft, **params)
            predict = lambda f, m=model: f.baseline_ft.to_numpy()+m.predict(f[FEATURES])
            score = event_metrics(validation, predict(validation))['macro_mae_ft']
            candidates.append((score, alpha, predict))
            search.append({'model': family, 'parameter': alpha, 'validation_macro_mae_ft': score})
        _, alpha, predict = min(candidates, key=lambda item: item[0])
        models.append((family, predict, {'alpha': alpha}))
    candidates, histories = [], []
    for width in [16, 32]:
        predict, score, history, model, scaler, scale = mlp_fit(train, validation, width)
        candidates.append((score, width, predict, model, scaler, scale))
        histories.extend(history)
        search.append({'model': 'PyTorch MLP', 'parameter': width,
                       'validation_macro_mae_ft': score})
    _, width, predict, model, scaler, scale = min(candidates, key=lambda item: item[0])
    torch.save({'state_dict': model.state_dict(), 'width': width, 'features': FEATURES,
                'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(),
                'target_scale': scale}, artifacts/'mlp.pt')
    models.append(('PyTorch MLP', predict, {'width': width, 'learning_rate': .001,
                                         'max_epochs': 300, 'patience': 30}))
    # All candidates are frozen before the first test evaluation
    pd.DataFrame(search).to_csv(reports/'validation_search.csv', index=False)
    pd.DataFrame(histories).to_csv(reports/'mlp_history.csv', index=False)
    mlflow.set_tracking_uri('sqlite:///'+(ROOT/'mlflow.db').as_posix())
    mlflow.set_experiment('LFPG altitude forecasting')
    sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                         text=True).stdout.strip() or 'uncommitted'
    rows = []
    for name, predict, params in models:
        with mlflow.start_run(run_name=name) as active:
            mlflow.log_params({**params, 'seed': SEED, 'airport': 'LFPG',
                               'train_rows': len(train), 'validation_rows': len(validation),
                               'test_rows': len(test), 'dataset_sha256': digest(path),
                               'git_commit': sha, 'features': json.dumps(FEATURES)})
            prediction = predict(test)
            scores = event_metrics(test, prediction)
            val = event_metrics(validation, predict(validation))
            mlflow.log_metrics({**scores, 'validation_macro_mae_ft': val['macro_mae_ft']})
            mlflow.log_artifact(str(reports/'source_audit.json'))
            mlflow.log_artifact(str(reports/'validation_search.csv'))
            if name == 'PyTorch MLP':
                mlflow.log_artifact(str(artifacts/'mlp.pt'))
                mlflow.log_artifact(str(reports/'mlp_history.csv'))
            rows.append({'model': name, **scores, 'run_id': active.info.run_id})
    result = pd.DataFrame(rows)
    result.to_csv(reports/'model_metrics.csv', index=False)
    plt.figure(figsize=(9, 4))
    plt.barh(result.model, result.macro_mae_ft, color=['#8193a5','#45a88a','#45a88a','#ee806b'])
    plt.xlabel('Mean event MAE, ft')
    plt.tight_layout()
    plt.savefig(reports/'model_comparison.png', dpi=180)
    plt.close()
    plt.figure(figsize=(8, 4))
    for size, group in pd.DataFrame(histories).groupby('width'):
        plt.plot(group.epoch, group.validation_macro_mae_ft, label=f'Width {size}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation mean event MAE, ft')
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports/'mlp_validation.png', dpi=180)
    plt.close()
    print(result.to_string(index=False))


if __name__ == '__main__':
    run()
