"""Lectura de configs YAML y resolución de rutas por variable de entorno."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from janonet import paths


@cache
def load_yaml(name: str) -> dict[str, Any]:
    """``load_yaml("data")`` lee ``configs/data.yaml``."""
    with open(paths.configs_dir() / f"{name}.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def data_root() -> Path:
    env = os.environ.get("JANONET_DATA")
    return Path(env).expanduser() if env else paths.project_root() / "data"


def resolve_location(spec: dict[str, str]) -> Path:
    """``{env: VAR, default_dir: x}`` -> ``$VAR`` o ``<JANONET_DATA>/x``."""
    env = os.environ.get(spec["env"])
    return Path(env).expanduser() if env else data_root() / spec["default_dir"]


def dataset_root(key: str) -> Path:
    return resolve_location(load_yaml("data")["roots"][key])


def output_dir(key: str) -> Path:
    return resolve_location(load_yaml("data")["outputs"][key])
