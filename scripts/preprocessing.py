"""Взвешенное масштабирование признаков только по обучающей выборке"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from audit_source import FEATURES


def event_weights(frame):
    # Нормируем веса так, чтобы средний вес строки был равен единице
    counts = frame.groupby(['airport', 'event_id']).event_id.transform('size').to_numpy()
    weights = 1 / counts
    return weights / weights.mean()


def fit_scaler(train):
    # Явно запрещаем попадание validation и test в обучение преобразования
    if train.empty or not train.split.eq('train').all():
        raise ValueError('Для обучения scaler допустимы только строки train')
    if not np.isfinite(train[FEATURES].to_numpy()).all():
        raise ValueError('Признаки должны быть конечными числами')
    return StandardScaler().fit(train[FEATURES], sample_weight=event_weights(train))
