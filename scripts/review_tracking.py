"""Проверка MLflow и добавление явно помеченных сведений ревью"""
import json
import subprocess
import importlib.metadata
import pandas as pd
from mlflow import MlflowClient
from audit_source import ROOT, digest
from airport_paths import paths


def run():
    client = MlflowClient(tracking_uri='sqlite:///'+(ROOT/'mlflow.db').as_posix())
    packages = ['numpy','pandas','pyarrow','scikit-learn','torch','mlflow','psycopg','matplotlib']
    versions = {package:importlib.metadata.version(package) for package in packages}
    records = []
    for airport in ['LFPG','EGLL','EDDF']:
        folders = paths(airport)
        results = pd.read_csv(folders['reports']/'model_metrics.csv')
        source = json.loads((folders['reports']/'source_audit.json').read_text())
        for row in results.itertuples(index=False):
            existing = client.get_run(row.run_id)
            expected = {'macro_mae_ft':row.macro_mae_ft,'macro_rmse_ft':row.macro_rmse_ft}
            if existing.info.status != 'FINISHED' or existing.data.params['airport'] != airport:
                raise ValueError('Неверный статус или аэропорт MLflow run')
            for key,value in expected.items():
                if abs(existing.data.metrics[key]-value) > 1e-6:
                    raise ValueError('Метрики отчета и MLflow расходятся')
            sha = existing.data.params['git_commit']
            git = subprocess.run(['git','-c',f'safe.directory={ROOT.as_posix()}',
                                  'cat-file','-e',sha+'^{commit}'],cwd=ROOT,capture_output=True)
            if git.returncode:
                raise ValueError('SHA запуска отсутствует в локальной истории')
            # Новые сведения отделены от исходных параметров обучения
            current_hash = digest(folders['data']/'postgres_examples.parquet')
            review = {'run_id':row.run_id,'airport':airport,'model':row.model,
                      'review_type':'posthoc verification; original metrics and parameters preserved',
                      'original_git_commit':sha,'original_dataset_sha256':existing.data.params['dataset_sha256'],
                      'current_dataset_sha256':current_hash,
                      'dataset_hash_matches':current_hash==existing.data.params['dataset_sha256'],
                      'split_source_sha256':source['split_sha256'],
                      'environment_versions_at_review':versions,
                      'environment_note':'current snapshot; not an independent historical runtime capture',
                      'lockfile_sha256':digest(ROOT/'requirements-lock.txt')}
            filename = folders['reports']/f'tracking_review_{row.run_id}.json'
            filename.write_text(json.dumps(review,indent=2),encoding='utf-8')
            client.log_artifact(row.run_id,str(filename),artifact_path='posthoc_review')
            client.log_artifact(row.run_id,str(ROOT/'requirements-lock.txt'),artifact_path='posthoc_review')
            client.set_tag(row.run_id,'posthoc_review','verified reports, git commit and current data hash')
            # LFPG classical artifacts have a different origin and are labeled explicitly
            if row.model in ['Ridge','Polynomial Ridge']:
                artifact = 'ridge.joblib' if row.model == 'Ridge' else 'polynomial_ridge.joblib'
                if airport == 'LFPG':
                    client.log_artifact(row.run_id,str(folders['artifacts']/artifact),artifact_path='posthoc_import_from_1_1')
                    client.set_tag(row.run_id,'classical_artifact_origin','imported from 1.1 after run; not original 1.2 artifact')
            records.append({'airport':airport,'model':row.model,'run_id':row.run_id,'metrics_match':True,
                            'git_commit_exists':True,'dataset_hash_matches':review['dataset_hash_matches'],
                            'artifact_root_entries':len(client.list_artifacts(row.run_id))})
    pd.DataFrame(records).to_csv(ROOT/'reports/tracking_review.csv',index=False)
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == '__main__':
    run()
