"""Проверка предобработки и сохраненного scaler без обучения моделей"""
import os
import json
import numpy as np
import pandas as pd
import torch
from audit_source import FEATURES, ROOT
from preprocessing import fit_scaler, event_weights
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'data/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run():
    frame = pd.read_parquet(ROOT/'data/postgres_examples.parquet')
    train = frame.loc[frame.split.eq('train')]
    w = event_weights(train)
    scaler = fit_scaler(train)
    z = scaler.transform(train[FEATURES])
    mean = np.average(train[FEATURES], axis=0, weights=w)
    variance = np.average((train[FEATURES].to_numpy()-mean)**2, axis=0, weights=w)
    saved = torch.load(ROOT/'artifacts/mlp.pt', map_location='cpu', weights_only=True)
    checks = {
        'weighted_mean_matches': bool(np.allclose(scaler.mean_, mean)),
        'weighted_variance_matches': bool(np.allclose(scaler.var_, variance)),
        'weighted_scaled_mean_zero': bool(np.allclose(np.average(z, axis=0, weights=w), 0, atol=1e-10)),
        'weighted_scaled_variance_one': bool(np.allclose(np.average(z*z, axis=0, weights=w), 1)),
        'saved_mlp_mean_matches': bool(np.allclose(saved['scaler_mean'], scaler.mean_)),
        'saved_mlp_scale_matches': bool(np.allclose(saved['scaler_scale'], scaler.scale_)),
    }
    original_mean, original_scale = scaler.mean_.copy(), scaler.scale_.copy()
    for split in ['validation', 'test']:
        scaler.transform(frame.loc[frame.split.eq(split), FEATURES])
    checks['transform_preserves_parameters'] = bool(np.array_equal(original_mean, scaler.mean_) and
                                                  np.array_equal(original_scale, scaler.scale_))
    if not all(checks.values()):
        raise ValueError(checks)
    pd.DataFrame({'feature': FEATURES, 'weighted_mean': mean, 'weighted_std': scaler.scale_,
                  'row_mean': train[FEATURES].mean().to_numpy()}).to_csv(ROOT/'reports/scaler_train.csv', index=False)
    (ROOT/'reports/preprocessing_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    # Показываем распределения только train, не выбирая признаки по тестовым результатам
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))
    for index, ax in enumerate(axes.flat):
        ax.hist(z[:,index], bins=35, weights=w, density=True, color='#45a88a')
        ax.set_title(FEATURES[index])
        ax.set_xlabel('Значение после масштабирования')
    fig.tight_layout()
    fig.savefig(ROOT/'reports/scaled_train_features.png', dpi=160)
    plt.close(fig)
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    run()
