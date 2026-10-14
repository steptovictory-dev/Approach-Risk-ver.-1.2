"""Проверки реальной транзакции на существующей локальной базе LFPG"""
import os
from pathlib import Path
import sys
import pandas as pd
import psycopg
import pytest
from dotenv import load_dotenv
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import database
from audit_source import ROOT


def test_conflict_rolls_back_new_day(monkeypatch):
    load_dotenv(ROOT/'.env')
    uri = os.environ.get('DATABASE_URL')
    if not uri:
        pytest.skip('Для интеграционного теста нужен локальный DATABASE_URL')
    frame = pd.read_parquet(ROOT/'data/lfpg_examples.parquet')
    days = pd.read_csv(ROOT/'data/selected_days.csv')
    # Добавляем день внутри транзакции, затем провоцируем конфликт существующей строки
    marker = '2099-01-01'
    with psycopg.connect(uri) as connection:
        before = connection.execute('SELECT count(*) FROM altitude.examples').fetchone()[0]
        assert connection.execute('SELECT count(*) FROM altitude.study_days WHERE day=%s',
                                  (marker,)).fetchone()[0] == 0
    extra = pd.DataFrame([{'day': marker, 'season': 'spring', 'split': 'train'}])
    changed_days = pd.concat([extra, days], ignore_index=True)
    changed_frame = frame.copy()
    changed_frame.loc[0, 'height_ft'] += 1
    monkeypatch.setattr(database.pd, 'read_parquet', lambda path: changed_frame)
    monkeypatch.setattr(database.pd, 'read_csv', lambda path: changed_days)
    with pytest.raises(ValueError, match='Existing observations differ'):
        database.run()
    # После ошибки новый день и любые изменения наблюдений отсутствуют
    with psycopg.connect(uri) as connection:
        assert connection.execute('SELECT count(*) FROM altitude.examples').fetchone()[0] == before
        assert connection.execute('SELECT count(*) FROM altitude.study_days WHERE day=%s',
                                  (marker,)).fetchone()[0] == 0


def test_roundtrip_failure_rolls_back_new_day(monkeypatch):
    load_dotenv(ROOT/'.env')
    uri = os.environ.get('DATABASE_URL')
    if not uri:
        pytest.skip('Для интеграционного теста нужен локальный DATABASE_URL')
    days = pd.read_csv(ROOT/'data/selected_days.csv')
    marker = '2099-01-02'
    with psycopg.connect(uri) as connection:
        assert connection.execute('SELECT count(*) FROM altitude.study_days WHERE day=%s',
                                  (marker,)).fetchone()[0] == 0
    extra = pd.DataFrame([{'day': marker, 'season': 'spring', 'split': 'train'}])
    monkeypatch.setattr(database.pd, 'read_csv', lambda path: pd.concat([extra, days], ignore_index=True))
    # Имитируем отказ итоговой сверки, чтобы проверить границу транзакции
    def fail_comparison(*args, **kwargs):
        raise AssertionError('Injected roundtrip failure')
    monkeypatch.setattr(database.pd.testing, 'assert_frame_equal', fail_comparison)
    with pytest.raises(AssertionError, match='Injected roundtrip failure'):
        database.run()
    with psycopg.connect(uri) as connection:
        assert connection.execute('SELECT count(*) FROM altitude.study_days WHERE day=%s',
                                  (marker,)).fetchone()[0] == 0
