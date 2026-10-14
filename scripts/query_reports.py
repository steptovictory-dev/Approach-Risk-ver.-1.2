"""Execute analytical SQL and export numbered result tables"""
import os
import argparse
import pandas as pd
import psycopg
import sqlparse
from dotenv import load_dotenv
from audit_source import ROOT


def run(sql_file='analytics.sql', prefix=''):
    load_dotenv(ROOT/'.env')
    folder = ROOT/'reports/sql'
    folder.mkdir(parents=True, exist_ok=True)
    statements = sqlparse.split((ROOT/'sql'/sql_file).read_text(encoding='utf-8'))
    with psycopg.connect(os.environ['DATABASE_URL']) as connection:
        for index, statement in enumerate(statements, 1):
            # Читаем каждый запрос из проверенного SQL-файла проекта
            with connection.cursor() as cursor:
                cursor.execute(statement)
                result = pd.DataFrame(cursor.fetchall(), columns=[d.name for d in cursor.description])
                result.to_csv(folder/f'{prefix}{index:02d}.csv', index=False)
                print(f'Query {index}: {len(result)} result rows')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sql-file', choices=['analytics.sql', 'join_checks.sql', 'event_mae.sql'], default='analytics.sql')
    args = parser.parse_args()
    prefix = {'analytics.sql': '', 'join_checks.sql': 'join_', 'event_mae.sql': 'event_mae_'}[args.sql_file]
    run(args.sql_file, prefix)
