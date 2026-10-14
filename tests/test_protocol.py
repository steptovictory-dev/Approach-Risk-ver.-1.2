"""Tests for event weighting and the frozen experimental protocol"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from train import event_metrics, weights
from audit_source import ROOT


def test_dense_event_does_not_dominate_metric():
    # У одного захода десять наблюдений, но оба захода имеют одинаковый вклад
    frame = pd.DataFrame({'event_id': ['a']*10+['b'], 'target_height_ft': [0.0]*11})
    prediction = np.array([2.0]*10+[10.0])
    assert event_metrics(frame, prediction)['macro_mae_ft'] == 6.0
    totals = pd.Series(weights(frame)).groupby(frame.event_id).sum()
    assert np.allclose(totals, totals.iloc[0])


def test_frozen_split_and_source_roundtrip():
    source = pd.read_parquet(ROOT/'data/lfpg_examples.parquet')
    extracted = pd.read_parquet(ROOT/'data/postgres_examples.parquet')
    keys = ['day', 'event_id', 'origin_time']
    pd.testing.assert_frame_equal(source[extracted.columns].sort_values(keys).reset_index(drop=True),
                                  extracted.sort_values(keys).reset_index(drop=True), check_dtype=False)
    assert extracted.groupby('event_id').split.nunique().eq(1).all()
    assert extracted.loc[extracted.split.eq('test')].event_id.nunique() == 55


def test_public_text_style():
    # Проверяем файлы проекта, исключая сторонние библиотеки и локальные исходные данные
    for directory in ['scripts','tests','docs','sql']:
        for path in (ROOT/directory).rglob('*'):
            if path.suffix in ['.py','.md','.sql']:
                text = path.read_text(encoding='utf-8')
                assert chr(0x451) not in text and chr(0x401) not in text
                assert chr(0x2014) not in text
