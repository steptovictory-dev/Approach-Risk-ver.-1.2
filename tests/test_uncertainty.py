"""Проверки парной повторной выборки целых кластеров"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from uncertainty import clustered_bootstrap


def test_clusters_keep_all_events_and_are_reproducible():
    # У первого борта два захода, у второго один: допустимы только три средних
    values = np.array([0.,12.,30.])
    labels = np.array(['a','a','b'])
    draws = clustered_bootstrap(values,labels,repetitions=100,seed=7)
    assert set(np.round(draws,8)).issubset({6.,14.,30.})
    assert np.array_equal(draws,clustered_bootstrap(values,labels,repetitions=100,seed=7))


def test_constant_paired_difference_stays_constant():
    draws = clustered_bootstrap(np.array([2.,2.,2.]),np.array(['a','a','b']),repetitions=20)
    assert np.allclose(draws,2.)
