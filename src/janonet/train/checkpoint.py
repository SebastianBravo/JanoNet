"""Checkpoints atómicos y rotados, y parada limpia por señal.

- ``last.ckpt`` (el último), ``prev.ckpt`` (el anterior) y ``best.ckpt`` (mejor AP
  REAL en validación). Si ``last`` no carga, se usa ``prev``.
- Escritura: ``torch.save`` a ``.tmp`` en el mismo directorio, ``fsync`` y
  ``os.replace`` (atómico en el mismo volumen): un corte deja el anterior intacto.
- El checkpoint lleva todo lo necesario para reanudar bit a bit: modelo, optimizador,
  escalador de AMP, época, pasos, muestras consumidas en la época, estado del sampler y
  de los generadores aleatorios, y un ``config_hash`` que impide reanudar con otra
  receta o con otros datos.
- SIGINT/SIGTERM solo marcan una bandera; el bucle guarda y sale al terminar el
  micro-lote en curso (``torch.save`` no es reentrante).

Origen: ``dataset_analysis/src/tfm/train/{checkpoint,signals}.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

FORMAT_VERSION = 1
LAST, PREV, BEST = "last.ckpt", "prev.ckpt", "best.ckpt"


class CheckpointError(RuntimeError):
    pass


def config_hash(payload: dict[str, Any]) -> str:
    """Hash estable de la receta experimental (modelo + claves experimentales de train + datos)."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def capture_rng_state() -> dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_atomic(state: dict[str, Any], path: str | os.PathLike[str]) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        with open(tmp, "wb") as fh:
            torch.save(state, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return dest


def load_checkpoint(path: str | os.PathLike[str], map_location: Any = "cpu") -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise CheckpointError(f"no existe {p}")
    try:
        return torch.load(p, map_location=map_location, weights_only=False)
    except Exception as exc:  # noqa: BLE001
        raise CheckpointError(f"{p} no se pudo cargar: {type(exc).__name__}: {exc}") from exc


class CheckpointManager:
    """Rotación ``last``/``prev``/``best`` en ``run_dir`` y verificación de compatibilidad."""

    def __init__(self, run_dir: str | os.PathLike[str], cfg_hash: str):
        self.run_dir = Path(run_dir)
        self.cfg_hash = cfg_hash

    def save(self, state: dict[str, Any], is_best: bool = False) -> None:
        state = {
            **state,
            "format_version": FORMAT_VERSION,
            "config_hash": self.cfg_hash,
            # El Trainer pasa el estado del inicio del ciclo de acumulación si guarda a
            # mitad de uno (al reanudar se rebobina a ese inicio).
            "rng_state": state.get("rng_state") or capture_rng_state(),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        last, prev = self.run_dir / LAST, self.run_dir / PREV
        if last.exists():
            os.replace(last, prev)
        save_atomic(state, last)
        if is_best:
            save_atomic(state, self.run_dir / BEST)

    def load_latest(self, map_location: Any = "cpu") -> tuple[dict[str, Any], Path] | None:
        for name in (LAST, PREV):
            p = self.run_dir / name
            if not p.exists():
                continue
            try:
                state = load_checkpoint(p, map_location)
            except CheckpointError as exc:
                print(f"[train] AVISO: {exc}; se prueba el anterior", flush=True)
                continue
            if state.get("config_hash") != self.cfg_hash:
                raise CheckpointError(
                    f"{p} es de otra receta (config_hash {state.get('config_hash')} != {self.cfg_hash}). "
                    "Usa otro --run-dir o --resume never."
                )
            return state, p
        return None


class StopSignal:
    """Bandera de parada para SIGINT/SIGTERM (y SIGBREAK en Windows)."""

    def __init__(self) -> None:
        self.should_stop = False

    def install(self) -> StopSignal:
        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is not None:
                try:
                    signal.signal(sig, self._handler)
                except (ValueError, OSError):  # fuera del hilo principal
                    pass
        return self

    def _handler(self, signum, frame) -> None:  # noqa: ARG002
        if not self.should_stop:
            print(f"\n[train] señal {signum}: se guarda y se sale al terminar el micro-lote", flush=True)
        self.should_stop = True

    def request_stop(self) -> None:
        self.should_stop = True
