"""Сверка всех повторных запусков с артефактами и MLflow без обучения"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from audit_source import ROOT, digest
from seed_stability import load_protocol, artifact_path, evaluate


def run():
    protocol = load_protocol()
    runs = pd.read_csv(ROOT/'reports/seed_stability_runs.csv')
    before = pd.read_csv(ROOT/'reports/seed_stability_metrics.csv').set_index(['airport','seed'])
    daily_before = pd.read_csv(ROOT/'reports/seed_stability_by_day.csv').set_index(['airport','seed','day'])
    evaluate()
    after = pd.read_csv(ROOT/'reports/seed_stability_metrics.csv').set_index(['airport','seed'])
    daily_after = pd.read_csv(ROOT/'reports/seed_stability_by_day.csv').set_index(['airport','seed','day'])
    for original,restored in [(before,after),(daily_before,daily_after)]:
        if not original.index.equals(restored.index) or not np.allclose(original,restored,atol=1e-5,rtol=1e-7):
            raise ValueError('Метрики повторных запусков не восстановились')
    client = MlflowClient(tracking_uri='sqlite:///'+(ROOT/'mlflow.db').as_posix())
    records = []
    for row in runs.itertuples(index=False):
        saved = client.get_run(row.run_id)
        if saved.info.status!='FINISHED' or saved.data.params['airport']!=row.airport:
            raise ValueError('Неверный статус или аэропорт повторного запуска')
        if int(saved.data.params['seed'])!=row.seed or int(saved.data.params['width'])!=row.width:
            raise ValueError('Параметры MLflow не совпадают')
        if saved.data.params['dataset_sha256']!=protocol['airports'][row.airport]['dataset_sha256']:
            raise ValueError('Хеш данных MLflow не совпадает')
        if saved.data.metrics['best_epoch']!=row.best_epoch or saved.data.metrics['stopped_epoch']!=row.stopped_epoch:
            raise ValueError('История выбора состояния MLflow не совпадает')
        if saved.data.params['protocol_sha256']!=digest(ROOT/'reports/seed_stability_protocol.json'):
            raise ValueError('Изменилась постановка запуска')
        scores = after.loc[(row.airport,row.seed)]
        for key in scores.index:
            if not np.isclose(saved.data.metrics[key],scores[key],atol=1e-5,rtol=1e-7):
                raise ValueError('Метрики MLflow не совпадают')
        if not np.isclose(saved.data.metrics['validation_macro_mae_ft'],row.validation_macro_mae_ft,atol=1e-5):
            raise ValueError('Validation MLflow не совпала')
        destination = ROOT/'data/seed_tracking_review'/row.run_id
        destination.mkdir(parents=True,exist_ok=True)
        logged = client.download_artifacts(row.run_id,artifact_path(row.airport,row.seed).name,str(destination))
        if digest(Path(logged))!=row.artifact_sha256:
            raise ValueError('Архивная модель MLflow отличается от локальной')
        for filename,sha in json.loads(saved.data.params['source_sha256']).items():
            logged = client.download_artifacts(row.run_id,f'training_source/{filename}',str(destination))
            if digest(Path(logged))!=sha:
                raise ValueError('Исходники MLflow не совпадают с хешами обучения')
        records.append({'airport':row.airport,'seed':row.seed,'run_id':row.run_id,
                        'status':saved.info.status,'restored_metrics_match':True,
                        'logged_model_hash_matches':True,'logged_training_source_hashes_match':True,
                        'dataset_sha256':protocol['airports'][row.airport]['dataset_sha256']})
    pd.DataFrame(records).to_csv(ROOT/'reports/seed_stability_review.csv',index=False)
    print(f'Проверены {len(records)} моделей и MLflow runs')


if __name__=='__main__':
    run()
