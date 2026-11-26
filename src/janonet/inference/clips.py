"""Inferencia sobre clips de la caché (o clips ya extraídos de un video).

El clip que entra a la red es el **central**: cuadros 8..23 de los 32 cacheados,
pasados a float [0, 1] (``uint8 / 255``). La red normaliza cada rama por su cuenta.
El puntaje oficial es ``prob_mean`` (módulo de media) o ``prob_video`` (Transformer).

La inferencia corre en float32 (CPU o GPU). Las predicciones del estudio se calcularon
con AMP (float16) en una V100: hay diferencias pequeñas, y en float16 la sigmoide
satura a 1,0 a partir de ``|logit| ≥ 8,3`` (156 videos de prueba de U100 tenían
``prob_mean = 1,0``), cosa que en float32 no ocurre.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import torch

from janonet.data import cache
from janonet.models.janonet import JanoNet


def clip_to_tensor(clip_uint8: np.ndarray) -> torch.Tensor:
    """``(T, H, W, 3)`` uint8 RGB -> ``(1, T, 3, H, W)`` float32 en [0, 1]."""
    x = torch.from_numpy(np.ascontiguousarray(clip_uint8.transpose(0, 3, 1, 2)))
    return x.to(torch.float32).div_(255.0).unsqueeze(0)


@torch.no_grad()
def score_clips(model: JanoNet, clips: torch.Tensor) -> dict[str, np.ndarray]:
    """``(B, T, 3, H, W)`` en [0, 1] -> puntaje oficial, logit de video (o None) y logits por cuadro."""
    out = model(clips)
    frame_logits = out["frame_logits"].float()
    if out["mode"] == "transformer":
        video_logit = out["video_logits"].float()
        score = torch.sigmoid(video_logit)
    else:
        video_logit = None
        score = torch.sigmoid(frame_logits).mean(dim=1)
    return {
        "score": score.cpu().numpy().astype(np.float32),
        "video_logit": None if video_logit is None else video_logit.cpu().numpy(),
        "frame_logits": frame_logits.cpu().numpy(),
    }


def _batches(sample_ids: list[int], size: int) -> Iterator[list[int]]:
    for i in range(0, len(sample_ids), size):
        yield sample_ids[i : i + size]


def predict_cached(
    model: JanoNet,
    sample_ids: Iterable[int],
    cache_dir: str | Path,
    batch_size: int = 4,
    device: str | torch.device = "cpu",
) -> dict[str, np.ndarray]:
    """Puntaje oficial de cada ``sample_id`` a partir de su ``.bin`` (clip central)."""
    model = model.to(device).eval()
    ids = [int(s) for s in sample_ids]
    off = cache.center_window_offset()
    scores, logits = [], []
    for chunk in _batches(ids, batch_size):
        clips = np.stack([cache.read_window(cache.bin_path(cache_dir, s), off, cache.CLIP_LEN) for s in chunk])
        x = torch.from_numpy(np.ascontiguousarray(clips.transpose(0, 1, 4, 2, 3))).to(device)
        res = score_clips(model, x.to(torch.float32).div_(255.0))
        scores.append(res["score"])
        logits.append(
            res["video_logit"] if res["video_logit"] is not None else np.full(len(chunk), np.nan, dtype=np.float32)
        )
    return {"sample_id": np.array(ids), "score": np.concatenate(scores), "video_logit": np.concatenate(logits)}
