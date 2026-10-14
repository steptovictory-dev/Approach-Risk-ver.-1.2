"""Проверка сохраненных моделей EGLL и EDDF без обучения"""
import joblib
import numpy as np
import pandas as pd
from airport_paths import paths
from audit_source import FEATURES, digest
from model_artifacts import load_mlp
from train import event_metrics


def run():
    records = []
    for airport in ['EGLL','EDDF']:
        folders = paths(airport)
        frame = pd.read_parquet(folders['data']/'postgres_examples.parquet')
        test = frame.loc[frame.split.eq('test')]
        reference = pd.read_csv(folders['reports']/'model_metrics.csv').set_index('model')
        for name,filename in [('Ridge','ridge.joblib'),('Polynomial Ridge','polynomial_ridge.joblib'),('PyTorch MLP','mlp.pt')]:
            path = folders['artifacts']/filename
            if name == 'PyTorch MLP':
                _,_,predict = load_mlp(path)
                prediction = predict(test)
            else:
                saved = joblib.load(path)
                if saved['features'] != FEATURES or saved['origin'] != 'trained_in_1.2':
                    raise ValueError('Неверное происхождение или признаки артефакта')
                prediction = test.baseline_ft.to_numpy()+saved['model'].predict(test[FEATURES])
            # Сверяем фиксированные результаты, не принимая новых решений по test
            scores = event_metrics(test,prediction)
            for key,value in scores.items():
                if not np.isclose(value,reference.loc[name,key],atol=1e-5,rtol=1e-7):
                    raise ValueError(f'Не совпала метрика восстановленной модели: {airport}, {name}, {key}')
            records.append({'airport':airport,'model':name,'sha256':digest(path),
                            'metrics_match':True,'origin':'trained_in_1.2',**scores})
    destination = paths('LFPG')['reports']/'airport_artifact_verification.csv'
    pd.DataFrame(records).to_csv(destination,index=False)
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == '__main__':
    run()
