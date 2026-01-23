# scripts/run_all.ps1
# Reprodutibilidade: um comando que instala (editable) e roda validação+plots gerando out/report.json
# Uso:
#   powershell -ExecutionPolicy Bypass -File .\scripts\run_all.ps1
# Opcional:
#   powershell -ExecutionPolicy Bypass -File .\scripts\run_all.ps1 -OutDir out -Cases .\src\relorbit_py\cases.yaml

param(
  [string]$OutDir = "out",
  [string]$Cases  = ".\src\relorbit_py\cases.yaml"
)

$ErrorActionPreference = "Stop"

# Executar a partir da raiz do repo
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

Write-Host "==> Repo root: $repoRoot"
Write-Host "==> OutDir:    $OutDir"
Write-Host "==> Cases:     $Cases"
Write-Host "==> Format:    $Format"

function Get-PythonCmd {
  if (Get-Command py -ErrorAction SilentlyContinue) {
    return @{ exe = "py"; args = @("-3") }
  }
  if (Get-Command python -ErrorAction SilentlyContinue) {
    return @{ exe = "python"; args = @() }
  }
  throw "Python não encontrado. Instale Python 3 e garanta 'py' ou 'python' no PATH."
}

$pyCmd  = Get-PythonCmd
$pyExe  = $pyCmd.exe
$pyArgs = $pyCmd.args

Write-Host "==> Python: $pyExe $($pyArgs -join ' ')"

# 1) Instala em modo editable
Write-Host "`n==> [1/2] pip install -e ."
& $pyExe @pyArgs -m pip install -e .
if ($LASTEXITCODE -ne 0) { throw "Falha no pip install -e ." }

# 2) Roda validação + plots e gera out/report.json
Write-Host "`n==> [2/2] python -m relorbit_py.validate --plots --out $OutDir --cases $Cases"
& $pyExe @pyArgs -m relorbit_py.validate --plots --out $OutDir --cases $Cases
if ($LASTEXITCODE -ne 0) { throw "Falha na validação." }


# 3) Roda a animação 
Write-Host "`n==> [3/3] python -m relorbit_py.animate --out $OutDir --cases $Cases --format $Format"
& $pyExe @pyArgs -m relorbit_py.animate --out $OutDir --cases $Cases --format $Format
if ($LASTEXITCODE -ne 0) { throw "Falha na animação." }

Write-Host "`n==> OK. Gerados:"
Write-Host "    - $OutDir\report.json"
Write-Host "    - $OutDir\plots\*"
