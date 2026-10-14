"""Execute analytical SQL and export numbered result tables"""
import os
import pandas as pd
import psycopg
import sqlparse
from dotenv import load_dotenv
from audit_source import ROOT


def run():
    load_dotenv(ROOT/'.env')
    folder = ROOT/'reports/sql'
    folder.mkdir(parents=True, exist_ok=True)
    statements = sqlparse.split((ROOT/'sql/analytics.sql').read_text(encoding='utf-8'))
    with psycopg.connect(os.environ['DATABASE_URL']) as connection:
        for index, statement in enumerate(statements, 1):
            # Читаем каждый запрос из проверенного SQL-файла проекта
            with connection.cursor() as cursor:
                cursor.execute(statement)
                result = pd.DataFrame(cursor.fetchall(), columns=[d.name for d in cursor.description])
                result.to_csv(folder/f'{index:02d}.csv', index=False)
                print(f'Query {index}: {len(result)} result rows')


if __name__ == '__main__':
    run()
