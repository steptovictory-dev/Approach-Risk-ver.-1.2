"""Проверка границ интервалов и равного вклада заходов"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_flight_phase import summarize_bins


def test_phase_aggregation_preserves_event_weights_and_bin_edges():
    # Точка на границе 500 входит в следующий интервал
    frame = pd.DataFrame({'event_id':['a','a','b','b'],'height_ft':[400,450,480,500],
                          'target_height_ft':[0.,0.,0.,0.]})
    result = summarize_bins(frame,np.array([2.,2.,10.,6.]),'height_ft')
    lower = result.loc[result.bin.eq('<500')]
    assert lower.mae_ft.mean() == 6.
    assert result.pairs.sum() == len(frame)
    assert result.loc[result.bin.eq('500–1000'),'mae_ft'].iloc[0] == 6.
