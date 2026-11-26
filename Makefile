# Atajos para Linux/macOS. En Windows: tasks.ps1 (mismos nombres).
# Las tareas de preprocess, train, eval y app se añaden en sus fases.
PY ?= python

.PHONY: setup test test-all lint format

setup:
	$(PY) -m pip install -e ".[preprocess,train,dev]"

test:
	$(PY) -m pytest -q

# Incluye paridad contra el repo de desarrollo y checkpoints si están definidos
# JANONET_DEV_REPO y JANONET_CHECKPOINTS.
test-all:
	$(PY) -m pytest -q -rs

lint:
	$(PY) -m ruff check src tests

format:
	$(PY) -m ruff format src tests
