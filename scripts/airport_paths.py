"""Раздельные пути данных и результатов каждого аэропорта"""
from audit_source import ROOT


def paths(airport):
    if airport not in ['LFPG','EGLL','EDDF']:
        raise ValueError('Неизвестный аэропорт')
    # Сохраняем существующие пути LFPG и изолируем новые исследования
    suffix = () if airport == 'LFPG' else (airport,)
    return {name: ROOT.joinpath(name,*suffix) for name in ['data','reports','artifacts']}
