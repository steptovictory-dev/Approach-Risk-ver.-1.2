"""Idempotent PostgreSQL ingestion and SQL feature extraction"""
import json
import os
import numpy as np
import pandas as pd
import psycopg
from psycopg import sql
from dotenv import load_dotenv
from audit_source import ROOT


def run():
    load_dotenv(ROOT/'.env')
    uri = os.environ.get('DATABASE_URL')
    if not uri:
        raise RuntimeError('Set DATABASE_URL in the local .env')
    frame = pd.read_parquet(ROOT/'data/lfpg_examples.parquet')
    days = pd.read_csv(ROOT/'data/selected_days.csv')
    # Reject incomplete or conflicting source records before opening a transaction
    if frame.isna().any().any() or frame.duplicated(['airport','event_id','origin_time']).any():
        raise ValueError('Source contains nulls or duplicate keys')
    if not np.isfinite(frame.select_dtypes('number').to_numpy()).all():
        raise ValueError('Source contains non-finite values')
    columns = [c for c in frame.columns if c not in ['season', 'split']]
    with psycopg.connect(uri) as connection:
        with connection.cursor() as cursor:
            cursor.execute((ROOT/'sql/01_schema.sql').read_text(encoding='utf-8'))
            # Immutable day metadata: a changed split must fail rather than overwrite
            for row in days.itertuples(index=False):
                cursor.execute('INSERT INTO altitude.study_days VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING',
                               ('LFPG', row.day, row.season, row.split))
                cursor.execute('SELECT season,split FROM altitude.study_days WHERE airport=%s AND day=%s',
                               ('LFPG', row.day))
                if cursor.fetchone() != (row.season, row.split):
                    raise ValueError('Existing split differs from frozen source')
            cursor.execute('CREATE TEMP TABLE incoming (LIKE altitude.examples INCLUDING DEFAULTS) ON COMMIT DROP')
            with cursor.copy(sql.SQL('COPY incoming ({}) FROM STDIN').format(
                    sql.SQL(',').join(map(sql.Identifier, columns)))) as copy:
                for row in frame[columns].itertuples(index=False, name=None):
                    copy.write_row(row)
            # Compare all columns for existing keys; silently accepting changed data is unsafe
            cursor.execute('''SELECT count(*) FROM incoming i JOIN altitude.examples e
                USING (airport,event_id,origin_time) WHERE to_jsonb(i) IS DISTINCT FROM to_jsonb(e)''')
            if cursor.fetchone()[0]:
                raise ValueError('Existing observations differ from incoming source')
            cursor.execute('SELECT count(*) FROM altitude.examples')
            before = cursor.fetchone()[0]
            cursor.execute('INSERT INTO altitude.examples SELECT * FROM incoming ON CONFLICT DO NOTHING')
            cursor.execute('SELECT count(*) FROM altitude.examples')
            after = cursor.fetchone()[0]
            cursor.execute('''SELECT e.*,d.season,d.split FROM altitude.examples e
                JOIN altitude.study_days d USING (airport,day)
                WHERE e.airport=%s ORDER BY e.day,e.event_id,e.origin_time''', ('LFPG',))
            extracted = pd.DataFrame(cursor.fetchall(), columns=[d.name for d in cursor.description])
    # Normalize SQL dates back to the original day format
    extracted['day'] = extracted.day.astype(str)
    expected = frame.sort_values(['day','event_id','origin_time']).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected[extracted.columns], extracted, check_dtype=False)
    extracted.to_parquet(ROOT/'data/postgres_examples.parquet', index=False)
    log = {'read': len(frame), 'inserted': after-before, 'already_present': len(frame)-(after-before),
           'database_rows': after, 'extracted': len(extracted), 'roundtrip_exact': True}
    (ROOT/'reports/ingestion.json').write_text(json.dumps(log, indent=2), encoding='utf-8')
    print(json.dumps(log))


if __name__ == '__main__':
    run()
