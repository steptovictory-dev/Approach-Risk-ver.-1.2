"""Validate the frozen LFPG source and record its provenance"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['height_ft', 'vertical_rate_fps', 'past_height_rate_fps',
            'speed_knots', 'distance_m', 'lateral_m', 'heading_error_deg',
            'past_gap_s', 'horizon_s']


def digest(path):
    # Вычисляем хеши потоком, без загрузки исходных архивов в память
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit(source):
    study = source / 'research/lfpg-seasonal-learning'
    split_path = study / 'reports/csv/selected_days.csv'
    days = pd.read_csv(split_path)
    if len(days) != 15 or days.day.duplicated().any():
        raise ValueError('Expected 15 distinct frozen days')
    frames, files = [], []
    for row in days.itertuples(index=False):
        path = study / 'data' / row.day / 'examples.parquet'
        frame = pd.read_parquet(path)
        if len(frame) != row.examples or frame.event_id.nunique() != row.approaches:
            raise ValueError(f'Inventory mismatch: {row.day}')
        if not frame.day.eq(row.day).all():
            raise ValueError(f'Day mismatch: {row.day}')
        frames.append(frame.assign(season=row.season, split=row.split, airport='LFPG'))
        files.append({'path': str(path.relative_to(source)), 'sha256': digest(path)})
    frame = pd.concat(frames, ignore_index=True)
    # Сохраняем исходное время в секундах Unix для явного преобразования в PostgreSQL
    numeric = FEATURES + ['target_height_ft', 'baseline_ft', 'residual_ft',
                          'origin_time', 'target_time', 'origin_metar_time',
                          'origin_metar_available_time', 'boundary_time']
    checks = {
        'no_nulls': not frame.isna().any().any(),
        'finite_numeric': bool(np.isfinite(frame[numeric].to_numpy()).all()),
        'unique_pair': not frame.duplicated(['airport', 'event_id', 'origin_time']).any(),
        'event_single_split': bool(frame.groupby('event_id').split.nunique().eq(1).all()),
        'event_single_day': bool(frame.groupby('event_id').day.nunique().eq(1).all()),
        'metar_available': bool(frame.origin_metar_available_time.le(frame.origin_time).all()),
        'target_before_boundary': bool(frame.target_time.lt(frame.boundary_time).all()),
        'horizon_range': bool(frame.horizon_s.between(27, 33).all()),
        'horizon_matches_time': bool(np.allclose(frame.target_time-frame.origin_time, frame.horizon_s)),
        'residual_matches_target': bool(np.allclose(frame.target_height_ft-frame.baseline_ft, frame.residual_ft)),
        'baseline_matches_physics': bool(np.allclose(frame.height_ft+frame.vertical_rate_fps*frame.horizon_s, frame.baseline_ft)),
    }
    if not all(checks.values()):
        raise ValueError(f'Source checks failed: {checks}')
    # Сохраняем локально только проверенные примеры, исходные файлы не изменяем
    local = ROOT / 'data'
    local.mkdir(exist_ok=True)
    frame.to_parquet(local / 'lfpg_examples.parquet', index=False)
    days.to_csv(local / 'selected_days.csv', index=False)
    report = {'airport': 'LFPG', 'rows': len(frame), 'events': frame.event_id.nunique(),
              'features': FEATURES, 'checks': checks, 'source_files': files,
              'split_sha256': digest(split_path),
              'staged_sha256': digest(local / 'lfpg_examples.parquet')}
    reports = ROOT / 'reports'
    reports.mkdir(exist_ok=True)
    (reports / 'source_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    summary = frame.groupby(['season', 'split']).agg(rows=('event_id', 'size'),
                                                     events=('event_id', 'nunique'))
    summary.to_csv(reports / 'split_inventory.csv')
    print(summary.to_string())
    print(f'PASS: {len(frame)} rows, {frame.event_id.nunique()} events')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT.parent / 'Approach Risk ver. 1.1')
    audit(parser.parse_args().source)
