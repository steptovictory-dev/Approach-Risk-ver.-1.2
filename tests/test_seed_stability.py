"""Проверяем воспроизводимость обучения и влияние seed на состояние модели"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train import mlp_fit
from audit_source import FEATURES


def test_seed_controls_training_and_does_not_change_train_scaler():
    torch.set_num_threads(2)
    rng = np.random.default_rng(42)
    train = pd.DataFrame(rng.normal(size=(24,len(FEATURES))),columns=FEATURES)
    train = train.assign(airport='LFPG',event_id=np.repeat(['a','b','c','d'],6),
                         split='train',baseline_ft=1000.,residual_ft=3.)
    train['target_height_ft'] = train.baseline_ft+train.residual_ft
    validation = train.assign(split='validation').copy()
    # Одинаковый seed должен повторять обучение, другой меняет веса при том же scaler
    first = mlp_fit(train,validation,4,seed=71)
    repeated = mlp_fit(train,validation,4,seed=71)
    changed = mlp_fit(train,validation,4,seed=72)
    assert np.array_equal(first[0](validation),repeated[0](validation))
    assert first[2] == repeated[2]
    assert any(not torch.equal(value,changed[3].state_dict()[key])
               for key,value in first[3].state_dict().items())
    assert np.array_equal(first[4].mean_,changed[4].mean_)
    assert first[5] == changed[5]
