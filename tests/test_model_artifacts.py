"""Проверка восстановленной нейросети без повторного обучения"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from model_artifacts import load_mlp
from audit_source import ROOT


def test_mlp_inference_is_batch_independent():
    network, saved, predict = load_mlp(ROOT/'artifacts/mlp.pt')
    frame = pd.read_parquet(ROOT/'data/postgres_examples.parquet').head(10)
    combined = predict(frame)
    individual = np.array([predict(frame.iloc[[index]])[0] for index in range(len(frame))])
    # Допускаем малую погрешность арифметики float32 при смене размера пакета
    assert np.allclose(combined,individual,atol=1e-3,rtol=0)
    assert not network.training
    assert sum(parameter.numel() for parameter in network.parameters()) == 449
