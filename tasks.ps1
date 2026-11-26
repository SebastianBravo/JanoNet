# Equivalente de Makefile para Windows PowerShell.
# Uso:  .\tasks.ps1 setup | test | test-all | lint | format | sources | faces | split | cache | favc
# Las tareas de train, eval y app se añaden en sus fases.
param([Parameter(Position = 0)][string]$Task = "help")

$ErrorActionPreference = "Stop"
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }

switch ($Task) {
    "setup"    { & $py -m pip install -e ".[preprocess,train,dev]" }
    "test"     { & $py -m pytest -q -m "not slow" }
    "test-all" { & $py -m pytest -q -rs }
    "lint"     { & $py -m ruff check src tests scripts docs }
    "format"   { & $py -m ruff format src tests scripts docs }
    "sources"  { & $py scripts/01_build_manifest.py sources }
    "faces"    { & $py scripts/02_preprocess.py faces }
    "split"    { & $py scripts/01_build_manifest.py split --check }
    "cache"    { & $py scripts/02_preprocess.py cache }
    "favc"     {
        & $py scripts/01_build_manifest.py sources --datasets fakeavceleb --out outputs/fakeavceleb_sources.csv
        if ($LASTEXITCODE -eq 0) { & $py scripts/02_preprocess.py faces --videos outputs/fakeavceleb_sources.csv --out manifests/local/fakeavceleb_preprocessing.csv }
        if ($LASTEXITCODE -eq 0) { & $py scripts/01_build_manifest.py split --check --write-local }
    }
    default    { Write-Host "Tareas: setup, test, test-all, lint, format, sources, faces, split, cache, favc" }
}
exit $LASTEXITCODE
