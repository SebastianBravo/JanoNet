"""Ensamble sin entrenamiento: media de los logits oficiales de sus miembros.

``logit_ens = (logit(clip(s_1)) + ... + logit(clip(s_n))) / n`` con
``clip(s) = clip(s, 1e-6, 1 - 1e-6)``, y ``score_ens = 1 / (1 + exp(-logit_ens))``.

La aritmética se hace **en la precisión de los puntajes de entrada** (float32 en las
predicciones del estudio y en la inferencia), como en los scripts del documento
(``analisis3.py:109-113``, ``analisis4.py:3-6``): así el ensamble U100 + RS-ca-tr
reproduce exactamente sus cifras (AP REAL de prueba 0,897; fig. 11). Ver
``configs/models/ensemble.yaml``.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import reduce

import numpy as np

from janonet.models.scoring import LOGIT_EPS


def _logit_same_dtype(p: np.ndarray, eps: float = LOGIT_EPS) -> np.ndarray:
    q = np.clip(np.asarray(p), eps, 1 - eps)
    return np.log(q / (1 - q))


def ensemble_logit(member_scores: Sequence[np.ndarray], eps: float = LOGIT_EPS) -> np.ndarray:
    """Media de los logits oficiales. ``member_scores``: puntajes oficiales en [0, 1]."""
    if len(member_scores) < 2:
        raise ValueError("un ensamble necesita al menos dos miembros")
    logits = [_logit_same_dtype(s, eps) for s in member_scores]
    return reduce(lambda a, b: a + b, logits) / len(logits)


def ensemble_score(member_scores: Sequence[np.ndarray], eps: float = LOGIT_EPS) -> np.ndarray:
    return 1 / (1 + np.exp(-ensemble_logit(member_scores, eps)))
