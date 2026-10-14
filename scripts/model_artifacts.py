"""Восстановление моделей и проверка сохраненных прогнозов"""
import json
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from audit_source import ROOT, FEATURES, digest


def load_mlp(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['features'] != FEATURES:
        raise ValueError('Порядок признаков MLP отличается')
    width = saved['width']
    network = nn.Sequential(nn.Linear(len(FEATURES),width),nn.ReLU(),
                            nn.Linear(width,width),nn.ReLU(),nn.Linear(width,1))
    network.load_state_dict(saved['state_dict'])
    network.eval()
    def predict(frame):
        # Восстанавливаем признаки и масштаб residual без обучения
        values = (frame[FEATURES].to_numpy()-np.array(saved['scaler_mean']))/np.array(saved['scaler_scale'])
        with torch.no_grad():
            residual = network(torch.tensor(values,dtype=torch.float32)).squeeze(1).numpy()
        return frame.baseline_ft.to_numpy()+residual*saved['target_scale']
    return network, saved, predict


def verify():
    from train import event_metrics
    frame = pd.read_parquet(ROOT/'data/postgres_examples.parquet')
    test = frame.loc[frame.split.eq('test')]
    validation = frame.loc[frame.split.eq('validation')]
    source = ROOT.parent/'Approach Risk ver. 1.1/research/lfpg-seasonal-learning/artifacts/models.joblib'
    bundle = joblib.load(source)
    search = pd.read_csv(ROOT/'reports/validation_search.csv')
    reference = pd.read_csv(ROOT/'reports/model_metrics.csv').set_index('model')
    results = []
    for name, filename in [('Ridge','ridge.joblib'),('Polynomial Ridge','polynomial_ridge.joblib')]:
        selected = search.loc[search.model.eq(name)].sort_values('validation_macro_mae_ft').iloc[0]
        model = bundle['models']['pooled',name]
        if bundle['features'] != FEATURES or bundle['parameters']['pooled',name]['alpha'] != selected.parameter:
            raise ValueError('Не совпали признаки или параметры исходной модели')
        prediction = test.baseline_ft.to_numpy()+model.predict(test[FEATURES])
        measured = event_metrics(test,prediction)
        for key,value in measured.items():
            if not np.isclose(value,reference.loc[name,key],rtol=1e-8,atol=1e-6):
                raise ValueError('Не совпали тестовые метрики исходной модели')
        score = event_metrics(validation,validation.baseline_ft.to_numpy()+model.predict(validation[FEATURES]))['macro_mae_ft']
        if not np.isclose(score,selected.validation_macro_mae_ft,atol=1e-6):
            raise ValueError('Не совпала валидационная метрика исходной модели')
        path = ROOT/'artifacts'/filename
        # Сохраняем импортированную модель с явным происхождением
        joblib.dump({'model':model,'features':FEATURES,'origin':'imported_from_1.1',
                     'source_sha256':digest(source)},path)
        restored = joblib.load(path)
        restored_prediction = test.baseline_ft.to_numpy()+restored['model'].predict(test[FEATURES])
        if not np.array_equal(prediction,restored_prediction):
            raise ValueError('Прогноз изменился после сохранения')
        results.append({'model':name,'origin':'imported_from_1.1','artifact':filename,
                        'sha256':digest(path),'restore_exact':True,'validation_mae_ft':score,**measured})
    network, saved, predict = load_mlp(ROOT/'artifacts/mlp.pt')
    prediction = predict(test)
    _, _, restored_predict = load_mlp(ROOT/'artifacts/mlp.pt')
    if not np.array_equal(prediction,restored_predict(test)):
        raise ValueError('Повторное восстановление MLP изменило прогноз')
    measured = event_metrics(test,prediction)
    for key,value in measured.items():
        if not np.isclose(value,reference.loc['PyTorch MLP',key],atol=1e-6):
            raise ValueError('Не совпали метрики восстановленной MLP')
    history = pd.read_csv(ROOT/'reports/mlp_history.csv')
    rows = []
    for width,group in history.groupby('width'):
        best = group.loc[group.validation_macro_mae_ft.idxmin()]
        rows.append({'width':int(width),'best_epoch':int(best.epoch),'stopped_epoch':int(group.epoch.max()),
                     'best_validation_mae_ft':best.validation_macro_mae_ft,
                     'selected':int(width)==saved['width']})
    pd.DataFrame(rows).to_csv(ROOT/'reports/mlp_selection.csv',index=False)
    results.append({'model':'PyTorch MLP','origin':'trained_in_1.2','artifact':'mlp.pt',
                    'sha256':digest(ROOT/'artifacts/mlp.pt'),'restore_exact':True,
                    'validation_mae_ft':event_metrics(validation,predict(validation))['macro_mae_ft'],**measured})
    pd.DataFrame(results).to_csv(ROOT/'reports/artifact_verification.csv',index=False)
    info = {'architecture':f'9 → {saved["width"]} → {saved["width"]} → 1',
            'parameters':sum(p.numel() for p in network.parameters()),
            'target_scale_ft':saved['target_scale'],'source_classical_sha256':digest(source),
            'no_retraining':True,'classical_origin':'imported from 1.1, not recovered first-run 1.2 artifacts'}
    (ROOT/'reports/mlp_architecture.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(info,indent=2))


if __name__ == '__main__':
    verify()
