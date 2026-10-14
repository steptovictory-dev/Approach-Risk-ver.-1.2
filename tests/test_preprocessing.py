"""Проверки границ обучения и весов преобразования"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from preprocessing import fit_scaler, event_weights
from audit_source import FEATURES


def sample():
    frame = pd.DataFrame({feature: [0.0, 0.0, 12.0] for feature in FEATURES})
    return frame.assign(airport='LFPG', event_id=['a','a','b'], split='train')


def test_scaler_equal_event_contribution():
    frame = sample()
    scaler = fit_scaler(frame)
    assert np.allclose(scaler.mean_, 6.0)
    assert np.allclose(scaler.var_, 36.0)
    totals = pd.Series(event_weights(frame)).groupby(frame.event_id).sum()
    assert np.allclose(totals, totals.iloc[0])


def test_scaler_rejects_validation_and_transform_does_not_fit():
    train = sample()
    validation = train.assign(split='validation')
    with pytest.raises(ValueError, match='train'):
        fit_scaler(pd.concat([train, validation]))
    scaler = fit_scaler(train)
    before = scaler.mean_.copy()
    scaler.transform(validation[FEATURES]*1000)
    assert np.array_equal(before, scaler.mean_)
