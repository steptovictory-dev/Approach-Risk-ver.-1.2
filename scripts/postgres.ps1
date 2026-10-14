param(
    [ValidateSet('start', 'stop', 'status')]
    [string]$Action = 'status'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$pgCtl = Join-Path $projectRoot 'data\postgres-runtime\pgsql\bin\pg_ctl.exe'
$cluster = Join-Path $projectRoot 'data\postgres-cluster'
$logPath = Join-Path $projectRoot 'data\postgres-server.log'
# Проверяем локальные пути перед управлением сервером проекта
if (!(Test-Path -LiteralPath $pgCtl) -or !(Test-Path -LiteralPath (Join-Path $cluster 'PG_VERSION'))) {
    throw 'Локальный PostgreSQL не подготовлен: см. docs/POSTGRESQL.md'
}
if ($Action -eq 'start') {
    # Сервер принимает подключения только с этого компьютера
    & $pgCtl -D $cluster -l $logPath -o '-p 55432 -h 127.0.0.1' -w start
} elseif ($Action -eq 'stop') {
    # Корректно завершаем подключения и сохраняем состояние базы
    & $pgCtl -D $cluster -m fast -w stop
} else {
    & $pgCtl -D $cluster status
}
if ($LASTEXITCODE -ne 0) { throw "pg_ctl завершился с кодом $LASTEXITCODE" }
