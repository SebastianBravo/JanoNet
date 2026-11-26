"""Ensamble sin entrenamiento: media de los logits oficiales de sus miembros.

``logit_ens = mean_i logit(clip(score_i, 1e-6, 1 - 1e-6))`` y
``score_ens = sigmoid(logit_ens)``. Con U100 + RS-ca-tr reproduce el AP REAL de
prueba del documento (0,897; fig. 11). Ver ``configs/models/ensemble.yaml``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from janonet.models.scoring import LOGIT_EPS, logit, sigmoid


def ensemble_logit(member_scores: Sequence[np.ndarray], eps: float = LOGIT_EPS) -> np.ndarray:
    """Media de los logits oficiales. ``member_scores``: puntajes oficiales en [0, 1]."""
    if len(member_scores) < 2:
        raise ValueError("un ensamble necesita al menos dos miembros")
    stacked = np.stack([logit(s, eps) for s in member_scores])
    return stacked.mean(axis=0)


def ensemble_score(member_scores: Sequence[np.ndarray], eps: float = LOGIT_EPS) -> np.ndarray:
    return sigmoid(ensemble_logit(member_scores, eps))
