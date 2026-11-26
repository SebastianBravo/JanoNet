"""Rutas del proyecto. Nada absoluto: todo sale de la raíz del repo o de variables de entorno.

Variables de entorno reconocidas (todas opcionales):

- ``JANONET_ROOT``     raíz del repositorio (por defecto, la que contiene ``configs/``).
- ``JANONET_WEIGHTS``  carpeta de pesos (por defecto ``<raíz>/weights``).
- ``JANONET_DATA``     carpeta con los datasets originales (Fase 2).
- ``JANONET_CACHE``    carpeta de la caché de 32 cuadros (Fase 2).
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    env = os.environ.get("JANONET_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # src/janonet/paths.py -> raíz del repo (instalación editable).
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "configs").is_dir():
        return candidate
    # Instalación no editable: se busca hacia arriba desde el directorio actual.
    for parent in [Path.cwd(), *Path.cwd().parents]:
        if (parent / "configs" / "models").is_dir():
            return parent
    raise FileNotFoundError("No se encontró la raíz de JanoNet (carpeta con configs/). Define JANONET_ROOT.")


def configs_dir() -> Path:
    return project_root() / "configs"


def weights_dir() -> Path:
    env = os.environ.get("JANONET_WEIGHTS")
    return Path(env).expanduser() if env else project_root() / "weights"


def srm_stats_path() -> Path:
    return configs_dir() / "srm_stats.json"
