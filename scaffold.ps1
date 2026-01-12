# scaffold.ps1
# Cria a estrutura híbrida (C++ engine + Python) com pastas e arquivos vazios.
# Uso (PowerShell):  ./scaffold.ps1
# Dica: execute na pasta onde você quer criar o repositório.

$ErrorActionPreference = "Stop"

$Repo = "relativistic-bh-orbits-mission-sim"

$Dirs = @(
  "$Repo",
  "$Repo/src_cpp/include/relorbit/solvers",
  "$Repo/src_cpp/include/relorbit/models",
  "$Repo/src_cpp/lib",
  "$Repo/src_cpp/bindings",
  "$Repo/src_cpp/tests_cpp",
  "$Repo/src/relorbit_py",
  "$Repo/tests",
  "$Repo/docs/figures",
  "$Repo/docs/reports"
)

$Files = @(
  "$Repo/README.md",
  "$Repo/LICENSE",
  "$Repo/pyproject.toml",
  "$Repo/validation_cases.yaml",
  "$Repo/CMakeLists.txt",

  "$Repo/src_cpp/include/relorbit/units.hpp",
  "$Repo/src_cpp/include/relorbit/types.hpp",
  "$Repo/src_cpp/include/relorbit/analysis.hpp",
  "$Repo/src_cpp/include/relorbit/api.hpp",
  "$Repo/src_cpp/include/relorbit/solvers/rk4.hpp",
  "$Repo/src_cpp/include/relorbit/solvers/rk45.hpp",
  "$Repo/src_cpp/include/relorbit/models/newton.hpp",
  "$Repo/src_cpp/include/relorbit/models/schwarzschild.hpp",

  "$Repo/src_cpp/lib/units.cpp",
  "$Repo/src_cpp/lib/analysis.cpp",
  "$Repo/src_cpp/lib/api.cpp",
  "$Repo/src_cpp/bindings/pybind_module.cpp",

  "$Repo/src_cpp/tests_cpp/test_units.cpp",
  "$Repo/src_cpp/tests_cpp/test_newton.cpp",
  "$Repo/src_cpp/tests_cpp/test_schwarzschild.cpp",
  "$Repo/src_cpp/tests_cpp/test_isco.cpp",
  "$Repo/src_cpp/tests_cpp/test_convergence.cpp",

  "$Repo/src/relorbit_py/__init__.py",
  "$Repo/src/relorbit_py/units.py",
  "$Repo/src/relorbit_py/cases.py",
  "$Repo/src/relorbit_py/validate.py",
  "$Repo/src/relorbit_py/plots.py",
  "$Repo/src/relorbit_py/mission.py",
  "$Repo/src/relorbit_py/report.py",
  "$Repo/src/relorbit_py/datasets.py",

  "$Repo/tests/test_pipeline.py",
  "$Repo/tests/test_validation_cases.py",
  "$Repo/tests/test_figures_smoke.py"
)

function New-EmptyFileIfMissing([string]$Path) {
  if (-not (Test-Path $Path)) {
    New-Item -ItemType File -Path $Path -Force | Out-Null
  }
}

# Cria diretórios
foreach ($d in $Dirs) {
  if (-not (Test-Path $d)) {
    New-Item -ItemType Directory -Path $d -Force | Out-Null
  }
}

# Cria arquivos vazios (sem sobrescrever existentes)
foreach ($f in $Files) {
  New-EmptyFileIfMissing $f
}

Write-Host "OK: Estrutura criada em .\$Repo"
Write-Host "Próximo passo: preencher README.md, pyproject.toml, CMakeLists.txt e começar pelo módulo C++ (pybind_module.cpp)."
