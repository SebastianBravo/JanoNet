# Atajos para Linux/macOS. En Windows: tasks.ps1 (mismos nombres).
# La tarea app se añade en la Fase 6.
PY ?= python

.PHONY: setup test test-all lint format sources faces split cache favc train predict eval

setup:
	$(PY) -m pip install -e ".[preprocess,train,dev]"

test:
	$(PY) -m pytest -q -m "not slow"

# Incluye paridad contra el repo de desarrollo y checkpoints si están definidos
# JANONET_DEV_REPO y JANONET_CHECKPOINTS.
test-all:
	$(PY) -m pytest -q -rs

lint:
	$(PY) -m ruff check src tests scripts docs

format:
	$(PY) -m ruff format src tests scripts docs

# Datos (ver DATA.md). Rutas por variables de entorno JANONET_*.
sources:
	$(PY) scripts/01_build_manifest.py sources

faces:
	$(PY) scripts/02_preprocess.py faces

split:
	$(PY) scripts/01_build_manifest.py split --check

cache:
	$(PY) scripts/02_preprocess.py cache

# Regenera localmente las filas de FakeAVCeleb (fuera de git por sus términos de uso).
favc:
	$(PY) scripts/01_build_manifest.py sources --datasets fakeavceleb --out outputs/fakeavceleb_sources.csv
	$(PY) scripts/02_preprocess.py faces --videos outputs/fakeavceleb_sources.csv --out manifests/local/fakeavceleb_preprocessing.csv
	$(PY) scripts/01_build_manifest.py split --check --write-local

# Entrenamiento (Fase 4). make train MODEL=rs_ca_tr
MODEL ?= rs_ca_tr
train:
	$(PY) scripts/03_train.py --model $(MODEL) --run-dir runs/$(MODEL)

# Evaluación (Fase 3). MODELS=u100 rs_ca rs_ca_tr ensemble
MODELS ?= u100 rs_ca rs_ca_tr ensemble
predict:
	$(PY) scripts/04_predict_split.py --models $(MODELS) --split test --out outputs/predictions_test.csv

eval:
	$(PY) scripts/05_evaluate.py --out outputs/evaluation --compare
