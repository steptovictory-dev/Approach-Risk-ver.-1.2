"""Проверки изоляции аэропортов и полноты извлечения"""
import sys
from pathlib import Path
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from airport_paths import paths


@pytest.mark.parametrize('airport,rows,events',[('EGLL',166508,5601),('EDDF',96714,2634)])
def test_airport_roundtrip_and_frozen_split(airport,rows,events):
    folders = paths(airport)
    source = pd.read_parquet(folders['data']/f'{airport.lower()}_examples.parquet')
    extracted = pd.read_parquet(folders['data']/'postgres_examples.parquet')
    keys = ['day','event_id','origin_time']
    pd.testing.assert_frame_equal(source[extracted.columns].sort_values(keys).reset_index(drop=True),
                                  extracted.sort_values(keys).reset_index(drop=True),check_dtype=False)
    assert len(extracted) == rows
    assert extracted.event_id.nunique() == events
    assert extracted.airport.eq(airport).all()
    assert extracted.groupby('event_id').split.nunique().eq(1).all()


def test_airport_folders_are_distinct():
    for kind in ['data','reports','artifacts']:
        assert len({paths(a)[kind] for a in ['LFPG','EGLL','EDDF']}) == 3
