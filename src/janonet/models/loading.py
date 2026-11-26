"""Carga de configs y pesos.

Acepta dos formatos:

- checkpoint de entrenamiento (el del repositorio de desarrollo y el de
  ``janonet.train``): un ``dict`` con la clave ``"model"``;
- pesos publicados (Fase 5): el ``state_dict`` directamente.

La carga es siempre ``strict=True``: una clave sobrante o faltante es un error, no un
aviso.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from janonet import paths
from janonet.models.janonet import JanoNet, build_model

MODEL_NAMES = ("u100", "rs_ca", "rs_ca_tr", "e6")


def model_config_path(name_or_path: str | Path) -> Path:
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p
    candidate = paths.configs_dir() / "models" / f"{name_or_path}.yaml"
    if not candidate.exists():
        raise FileNotFoundError(f"no existe la config de modelo {name_or_path!r} ({candidate})")
    return candidate


def load_config(name_or_path: str | Path) -> dict[str, Any]:
    with open(model_config_path(name_or_path), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_state_dict_from_checkpoint(path: str | Path) -> dict[str, torch.Tensor]:
    obj = torch.load(Path(path), map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj
    raise ValueError(f"{path}: formato de checkpoint no reconocido")


def load_model(name_or_path: str | Path, checkpoint: str | Path, device: str | torch.device = "cpu") -> JanoNet:
    """Construye el modelo de la config y carga sus pesos con ``strict=True``."""
    cfg = load_config(name_or_path)
    model = build_model(cfg["model"], pretrained=False)
    model.load_state_dict(load_state_dict_from_checkpoint(checkpoint), strict=True)
    return model.to(device).eval()
