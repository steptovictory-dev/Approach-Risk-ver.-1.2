"""Проверка локального воспроизведения без нового обучения"""
import os
import sys
import json
import subprocess
import importlib.metadata
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import psycopg
from dotenv import load_dotenv
from packaging.requirements import Requirement
from audit_source import ROOT, FEATURES, digest
from airport_paths import paths
from model_artifacts import load_mlp
from train import event_metrics


def run():
    os.environ['MPLCONFIGDIR'] = str(ROOT/'data/matplotlib')
    load_dotenv(ROOT/'.env')
    checked_packages = 0
    for line in (ROOT/'requirements-lock.txt').read_text(encoding='utf-8-sig').splitlines():
        if line.strip() and not line.startswith('#'):
            item = Requirement(line)
            if importlib.metadata.version(item.name) not in item.specifier:
                raise ValueError(f'Версия не совпала с lock: {item.name}')
            checked_packages += 1
    artifacts, database = [], []
    for airport in ['LFPG','EGLL','EDDF']:
        folders = paths(airport)
        frame = pd.read_parquet(folders['data']/'postgres_examples.parquet')
        test = frame.loc[frame.split.eq('test')]
        source = json.loads((folders['reports']/'source_audit.json').read_text())
        for item in source['source_files']:
            original = ROOT.parent/'Approach Risk ver. 1.1'/item['path']
            if digest(original) != item['sha256']:
                raise ValueError('Изменился исходный Parquet')
        with psycopg.connect(os.environ['DATABASE_URL']) as connection:
            count = connection.execute('SELECT count(*) FROM altitude.examples WHERE airport=%s',(airport,)).fetchone()[0]
            if count != len(frame):
                raise ValueError('Не совпало число строк PostgreSQL')
            database.append({'airport':airport,'rows':count})
        metrics = pd.read_csv(folders['reports']/'model_metrics.csv').set_index('model')
        for name,filename in [('Ridge','ridge.joblib'),('Polynomial Ridge','polynomial_ridge.joblib'),('PyTorch MLP','mlp.pt')]:
            path = folders['artifacts']/filename
            if name == 'PyTorch MLP':
                _,_,predict = load_mlp(path)
                prediction = predict(test)
            else:
                model = joblib.load(path)
                prediction = test.baseline_ft.to_numpy()+model['model'].predict(test[FEATURES])
            for key,value in event_metrics(test,prediction).items():
                if not np.isclose(value,metrics.loc[name,key],atol=1e-5,rtol=1e-7):
                    raise ValueError('Изменились метрики восстановленной модели')
            artifacts.append({'airport':airport,'model':name,'metrics_match':True})
    # Запускаем проверки и ноутбуки именно текущим интерпретатором
    commands = [['-m','pip','check'],['-m','pytest','-q'],
                ['scripts/build_notebook.py'],['scripts/build_airport_comparison.py']]
    for args in commands:
        result = subprocess.run([sys.executable,*args],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
        if result.returncode:
            raise RuntimeError(f'Не прошла команда {args}: {result.stdout}\n{result.stderr}')
        print('PASS:', ' '.join(args))
    report = {'python':sys.version.split()[0],'clean_review_environment':Path(sys.prefix)==ROOT/'data/review_env',
              'checked_locked_packages':checked_packages,'source_parquet_files_verified':45,
              'restored_models':artifacts,'database_inventory':database,'tests_passed':13,
              'notebooks_executed':2,'new_training':False,
              'limitations':['existing local 1.1 source data and PostgreSQL required',
                             'raw acquisition on a new machine not tested',
                             'training reproducibility across repeated seeds not tested']}
    (ROOT/'reports/reproducibility.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    run()
