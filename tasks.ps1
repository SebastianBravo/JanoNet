# Equivalente de Makefile para Windows PowerShell.
# Uso:  .\tasks.ps1 setup | test | test-all | lint | format
# Las tareas de preprocess, train, eval y app se añaden en sus fases.
param([Parameter(Position = 0)][string]$Task = "help")

$ErrorActionPreference = "Stop"
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }

switch ($Task) {
    "setup"    { & $py -m pip install -e ".[preprocess,train,dev]" }
    "test"     { & $py -m pytest -q }
    "test-all" { & $py -m pytest -q -rs }
    "lint"     { & $py -m ruff check src tests }
    "format"   { & $py -m ruff format src tests }
    default    { Write-Host "Tareas: setup, test, test-all, lint, format" }
}
exit $LASTEXITCODE
