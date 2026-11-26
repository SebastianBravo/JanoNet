"""Dataset de clips desde la caché, sampler reanudable y muestreo estratificado.

- ``ClipDataset`` devuelve ``(clip uint8 (16, 3, 224, 224), sample_id, y)``. En
  ``eval`` el clip es el central (cuadros 8..23); en ``train`` la ventana se desplaza
  de forma determinista con ``derive_seed(semilla, época, sample_id)``. Sin aumentos de
  datos. La conversión a float [0, 1] se hace después, en el dispositivo.
- ``ResumableSampler``: permutación por época a partir de ``(semilla, época)``, que se
  puede retomar desde cualquier índice sin releer nada.
- ``stratified_sample_ids``: muestra reproducible estratificada por ``(dataset, label)``.

Origen: ``dataset_analysis/src/tfm/data/{dataset,sampler,samples}.py``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from janonet.data import cache


class ClipDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        cache_dir: str | os.PathLike[str],
        split_mode: str = "train",
        base_seed: int = 20260910,
        temporal_stride: int = 1,
        clip_len: int = cache.CLIP_LEN,
        epoch: int = 0,
    ):
        if split_mode not in ("train", "eval"):
            raise ValueError(f"split_mode debe ser 'train' o 'eval', llegó {split_mode!r}")
        if temporal_stride not in (1, 2):
            raise ValueError(f"temporal_stride debe ser 1 o 2, llegó {temporal_stride}")
        self.cache_dir = Path(cache_dir)
        self.split_mode = split_mode
        self.base_seed = int(base_seed)
        self.temporal_stride = int(temporal_stride)
        self.clip_len = int(clip_len)
        self.sample_ids = frame["sample_id"].to_numpy(dtype=np.int64, copy=True)
        self.labels = frame["y"].to_numpy(dtype=np.int64, copy=True)
        self.max_offset = cache.max_window_offset(cache.N_FRAMES, self.clip_len, self.temporal_stride)
        self.center_offset = cache.center_window_offset(cache.N_FRAMES, self.clip_len, self.temporal_stride)
        # Memoria compartida: con persistent_workers los workers conservan su copia del
        # Dataset, y un entero normal nunca les llegaría al cambiar de época.
        self._epoch = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._epoch[0] = int(epoch)

    @property
    def epoch(self) -> int:
        return int(self._epoch[0])

    def set_epoch(self, epoch: int) -> None:
        self._epoch[0] = int(epoch)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def window_offset(self, sample_id: int) -> int:
        if self.split_mode == "eval":
            return self.center_offset
        return cache.train_window_offset(self.base_seed, self.epoch, sample_id, self.max_offset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        sample_id = int(self.sample_ids[index])
        win = cache.read_window(
            cache.bin_path(self.cache_dir, sample_id),
            self.window_offset(sample_id),
            self.clip_len,
            self.temporal_stride,
        )
        clip = torch.from_numpy(np.ascontiguousarray(win.transpose(0, 3, 1, 2)))
        return clip, sample_id, int(self.labels[index])


def to_float01(clip_uint8: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    """uint8 -> float32 en [0, 1], ya en el dispositivo (se transfiere uint8: 4x menos datos)."""
    return clip_uint8.to(device, non_blocking=True).to(torch.float32).div_(255.0)


class ResumableSampler(Sampler[int]):
    """Permutación reproducible por época, reanudable desde ``start_index``."""

    def __init__(
        self, n_samples: int, base_seed: int = 20260910, epoch: int = 0, shuffle: bool = True, start_index: int = 0
    ):
        if n_samples <= 0:
            raise ValueError(f"n_samples debe ser positivo, llegó {n_samples}")
        self.n_samples = int(n_samples)
        self.base_seed = int(base_seed)
        self.epoch = int(epoch)
        self.shuffle = bool(shuffle)
        self.start_index = int(start_index)

    def permutation(self) -> np.ndarray:
        if not self.shuffle:
            return np.arange(self.n_samples, dtype=np.int64)
        seed = cache.derive_seed(self.base_seed, self.epoch) % (2**63)
        return np.random.default_rng(seed).permutation(self.n_samples).astype(np.int64, copy=False)

    def __iter__(self) -> Iterator[int]:
        for idx in self.permutation()[self.start_index :]:
            yield int(idx)

    def __len__(self) -> int:
        return self.n_samples - self.start_index

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.start_index = 0

    def set_start_index(self, start_index: int) -> None:
        if not 0 <= start_index <= self.n_samples:
            raise ValueError(f"start_index={start_index} fuera de [0, {self.n_samples}]")
        self.start_index = int(start_index)

    def state_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "base_seed": self.base_seed,
            "epoch": self.epoch,
            "shuffle": self.shuffle,
            "start_index": self.start_index,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["n_samples"]) != self.n_samples:
            raise ValueError(f"el checkpoint tiene n_samples={state['n_samples']}, el dataset {self.n_samples}")
        self.base_seed = int(state["base_seed"])
        self.epoch = int(state["epoch"])
        self.shuffle = bool(state["shuffle"])
        self.start_index = int(state["start_index"])


def worker_init_fn(worker_id: int) -> None:
    """Semilla de cada worker a partir de ``(semilla, época, worker_id)`` (higiene: el dataset no usa RNG)."""
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    ds = info.dataset
    seed = cache.derive_seed(int(getattr(ds, "base_seed", 0)), int(getattr(ds, "epoch", 0)), worker_id) % (2**31 - 1)
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stratified_sample_ids(
    df: pd.DataFrame, n: int, seed: int, strata: tuple[str, ...] = ("dataset", "label")
) -> list[int]:
    """Muestra estratificada reproducible (reparto proporcional, resto mayor, >= 1 por estrato)."""
    if n <= 0:
        return []
    if n >= len(df):
        return sorted(int(s) for s in df["sample_id"])
    groups = df.groupby(list(strata), observed=True)
    sizes = groups.size().sort_index()
    exact = sizes / sizes.sum() * n
    alloc = exact.astype(int).clip(upper=sizes)
    alloc[alloc == 0] = 1
    alloc = alloc.clip(upper=sizes)
    rest = int(n - alloc.sum())
    if rest > 0:
        for key in (exact - alloc).sort_values(ascending=False).index:
            if rest == 0:
                break
            if alloc[key] < sizes[key]:
                alloc[key] += 1
                rest -= 1
    elif rest < 0:
        for key in (exact - alloc).sort_values(ascending=True).index:
            if rest == 0:
                break
            if alloc[key] > 1:
                alloc[key] -= 1
                rest += 1
    chosen: list[int] = []
    for key, group in groups:
        k = int(alloc[key])
        if k > 0:
            chosen.extend(int(s) for s in group.sample(n=min(k, len(group)), random_state=seed)["sample_id"])
    return sorted(chosen)
