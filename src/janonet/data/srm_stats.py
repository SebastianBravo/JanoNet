"""Estadísticas de normalización del residuo SRM (media y desviación por canal) sobre train.

Muestra: 2000 videos de train estratificados por ``(dataset, label)`` (semilla 20260812)
x 4 cuadros repartidos en la ventana central de 16 (índices 0, 5, 10, 15). Muchos videos
y pocos cuadros por video, porque los cuadros consecutivos de un recorte facial son casi
el mismo dato. Se calcula dos veces con semillas distintas (20260812 y 1337) y se
reporta la divergencia como control del tamaño de muestra.

Se acumula en float64 con sumas y sumas de cuadrados del residuo **float32** (las
estadísticas del estudio se calcularon así; el redondeo a float16 de
``residual_precision`` se aplica después, al usarlas).

Origen: ``dataset_analysis/scripts/03b_compute_srm_stats.py``. El fichero del estudio es
``configs/srm_stats.json``; recalcularlo solo hace falta para otro conjunto de datos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from janonet.data import cache
from janonet.data.clip_dataset import stratified_sample_ids
from janonet.models import srm as SRM


def frame_indices(n_frames_per_video: int, clip_len: int = cache.CLIP_LEN) -> list[int]:
    if n_frames_per_video >= clip_len:
        return list(range(clip_len))
    return [int(round(i)) for i in np.linspace(0, clip_len - 1, n_frames_per_video)]


def accumulate_stats(
    sample_ids: list[int],
    cache_dir: Path,
    n_frames_per_video: int = 4,
    batch_videos: int = 32,
    padding_mode: str = SRM.DEFAULT_SRM_PADDING_MODE,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray, int]:
    weight = SRM.srm_filter_bank(torch.float32).to(device)
    offset = cache.center_window_offset()
    idxs = frame_indices(n_frames_per_video)
    s = torch.zeros(SRM.SRM_N_KERNELS, dtype=torch.float64, device=device)
    s2 = torch.zeros(SRM.SRM_N_KERNELS, dtype=torch.float64, device=device)
    n_px = 0
    batch: list[np.ndarray] = []

    def flush() -> None:
        nonlocal n_px, batch
        if not batch:
            return
        t = torch.from_numpy(np.stack(batch)).to(device)
        t = t.reshape(-1, *t.shape[2:]).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
        r = SRM.apply_srm(t, weight, padding_mode).double()
        s.add_(r.sum(dim=(0, 2, 3)))
        s2.add_((r * r).sum(dim=(0, 2, 3)))
        n_px += r.shape[0] * r.shape[2] * r.shape[3]
        batch = []

    for sid in sample_ids:
        win = cache.read_window(cache.bin_path(cache_dir, sid), offset, cache.CLIP_LEN, 1)
        batch.append(np.ascontiguousarray(win[idxs]))
        if len(batch) >= batch_videos:
            flush()
    flush()
    mean = (s / n_px).cpu().numpy()
    std = np.sqrt(np.clip((s2 / n_px).cpu().numpy() - mean**2, 1e-20, None))
    return mean, std, n_px


def compute(
    train: pd.DataFrame,
    cache_dir: Path,
    n_videos: int = 2000,
    n_frames_per_video: int = 4,
    seed: int = 20260812,
    seed_check: int = 1337,
    device: str = "cpu",
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for tag, sd in (("main", seed), ("check", seed_check)):
        ids = stratified_sample_ids(train, n_videos, sd)
        mean, std, n_px = accumulate_stats(ids, cache_dir, n_frames_per_video, device=device)
        out[tag] = {"mean": mean, "std": std, "n_px": n_px, "ids": ids}
    a, b = out["main"], out["check"]
    return {
        "mean": [float(v) for v in a["mean"]],
        "std": [float(v) for v in a["std"]],
        "n_kernels": SRM.SRM_N_KERNELS,
        "input_range": "[0,1] (uint8/255), sin normalizacion ImageNet",
        "padding_mode": SRM.DEFAULT_SRM_PADDING_MODE,
        "n_videos": n_videos,
        "n_frames": n_videos * len(frame_indices(n_frames_per_video)),
        "n_frames_per_video": len(frame_indices(n_frames_per_video)),
        "frame_indices_in_center_window": frame_indices(n_frames_per_video),
        "n_pixels_per_channel": int(a["n_px"]),
        "seed": seed,
        "split": "train",
        "convergence_check": {
            "seed_check": seed_check,
            "overlap_videos": len(set(a["ids"]) & set(b["ids"])),
            "max_std_rel_diff": float(np.max(np.abs(a["std"] - b["std"]) / a["std"])),
            "max_mean_shift_in_std": float(np.max(np.abs(a["mean"] - b["mean"]) / a["std"])),
            "mean_check": [float(v) for v in b["mean"]],
            "std_check": [float(v) for v in b["std"]],
        },
    }
