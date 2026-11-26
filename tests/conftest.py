"""Fixtures compartidas.

Variables de entorno opcionales (los tests que las necesitan se saltan si faltan):

- ``JANONET_CHECKPOINTS``: una o varias carpetas (separadas por ``os.pathsep``) donde
  buscar ``<source_run>/best.ckpt`` (formato del repositorio de desarrollo) o
  ``<name>.pt`` (pesos publicados). ``weights/`` se busca siempre.
- ``JANONET_DEV_REPO``: raíz del repositorio de desarrollo, para las pruebas de
  paridad contra el código original.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml

from janonet import paths

MODELS = ("u100", "rs_ca", "rs_ca_tr", "e6")


def model_cfg(name: str) -> dict:
    with open(paths.configs_dir() / "models" / f"{name}.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def find_checkpoint(name: str) -> Path | None:
    run = model_cfg(name)["meta"]["source_run"]
    dirs = [paths.weights_dir()]
    env = os.environ.get("JANONET_CHECKPOINTS")
    if env:
        dirs += [Path(d) for d in env.split(os.pathsep) if d]
    for d in dirs:
        for candidate in (d / f"{name}.pt", d / run / "best.ckpt"):
            if candidate.is_file():
                return candidate
    return None


@pytest.fixture
def checkpoint_for():
    def _get(name: str) -> Path:
        ckpt = find_checkpoint(name)
        if ckpt is None:
            pytest.skip(f"no hay checkpoint de {name} (define JANONET_CHECKPOINTS)")
        return ckpt

    return _get


@pytest.fixture(scope="session")
def dev_repo() -> Path:
    env = os.environ.get("JANONET_DEV_REPO")
    if not env or not (Path(env) / "src" / "tfm").is_dir():
        pytest.skip("JANONET_DEV_REPO no apunta al repositorio de desarrollo")
    root = Path(env)
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    return root
