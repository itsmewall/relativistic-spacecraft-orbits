# scripts/run_mission.ps1
param(
    [string]$Yaml = "src/relorbit_py/mission.yaml"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

Write-Host "==> Preparando Missão Espacial..." -ForegroundColor Cyan

# Garante que o C++ está atualizado
& python -m pip install -e .

# Roda o script de missão
Write-Host "==> Executando Mission Runner com $Yaml" -ForegroundColor Yellow
& python src/relorbit_py/run_mission.py

Write-Host "`n==> SUCESSO. Verifique a pasta out/missions para os resultados." -ForegroundColor Green