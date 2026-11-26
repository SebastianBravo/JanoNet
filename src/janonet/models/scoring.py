"""Puntaje oficial por video y su logit.

Regla del documento: las configuraciones con Transformer usan la salida por video
(``prob_video = sigmoid(video_logits)``); las de módulo de media, la media de las 16
probabilidades por cuadro (``prob_mean``). Origen:
``dataset_analysis/src/tfm/eval/aggregate.py:40-44`` (``OFFICIAL_BY_TEMPORAL``).

El **logit oficial** es ``logit(clip(score, eps, 1 - eps))`` con ``eps = 1e-6``, en
float64. Es la definición con la que se calculó el ensamble del documento
(``export_resultados/scripts/lib_export.py:43,370-372``). Con ``eps = 1e-6`` el logit
queda acotado a ±13,8155: un ``prob_mean`` que saturó a 1,0 en fp16 durante la
evaluación original da exactamente +13,8155.

Las puntuaciones son **puntajes de falsedad** (FAKE = 1), no probabilidades
calibradas.
"""

from __future__ import annotations

import numpy as np
import torch

LOGIT_EPS = 1e-6

OFFICIAL_SCORE = {"mean": "prob_mean", "transformer": "prob_video"}


def official_score(out: dict[str, torch.Tensor]) -> torch.Tensor:
    """Puntaje oficial ``(B,)`` a partir de la salida del modelo."""
    if out["mode"] == "transformer":
        return torch.sigmoid(out["video_logits"].float())
    return torch.sigmoid(out["frame_logits"].float()).mean(dim=1)


def logit(p, eps: float = LOGIT_EPS) -> np.ndarray:
    """Logit acotado, en float64."""
    q = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(q / (1 - q))


def sigmoid(x) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))
