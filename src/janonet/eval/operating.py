"""Métricas, umbrales y puntos de operación.

Convención: FAKE = 1 es la clase positiva del puntaje (puntaje alto = falso). La
clase REAL se evalúa como positiva invirtiendo el puntaje (``1 - s``).

Régimen productivo con prevalencia de falsos ``pi`` (0,02 por defecto):

    fricción        = 1 + (1 - pi)/pi · FPR/TPR      (alertas por deepfake detectado)
    precisión FAKE  = pi·TPR / (pi·TPR + (1 - pi)·FPR)
    alertas / 1000  = 1000 · (pi·TPR + (1 - pi)·FPR)

Umbrales, siempre elegidos en validación:

- **A**: máximo F1 de la clase REAL, sobre la rejilla de 2001 cuantiles del puntaje.
- **B**: el menor umbral con fricción ≤ 5 a ``pi = 0,02``, **sin** factor de deriva.
- Escalera: el criterio de B con fricción objetivo 3, 4, 5, 6, 8, 10, 15 y 20, y el
  mayor umbral con recall FAKE ≥ 0,80, 0,85, 0,90 y 0,95.

Origen: ``analisis_documento_JanoNet/scripts/analisis2.py:11-47,70-72`` (scripts que
generaron las cifras del documento v12). Se conserva la aritmética exacta para que
las cifras coincidan.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta
from sklearn.metrics import average_precision_score, roc_auc_score

PI = 0.02
FRICTION_TARGET = 5.0
FRICTION_LADDER = (3, 4, 5, 6, 8, 10, 15, 20)
RECALL_LADDER = (0.80, 0.85, 0.90, 0.95)
PREVALENCES = (0.005, 0.01, 0.02, 0.05, 0.10)


def friction(tpr: float, fpr: float, pi: float = PI) -> float:
    return 1 + (1 - pi) / pi * fpr / tpr if tpr > 0 else np.inf


def threshold_friction(sv: np.ndarray, yv: np.ndarray, target: float = FRICTION_TARGET, pi: float = PI) -> float:
    """Menor umbral de validación con fricción ≤ ``target`` (si ninguno lo cumple, el mayor)."""
    n_fake, n_real = (yv == 1).sum(), (yv == 0).sum()
    o = np.argsort(-sv)
    s, y = sv[o], yv[o]
    tp, fp = np.cumsum(y == 1), np.cumsum(y == 0)
    last = np.r_[s[1:] != s[:-1], True]  # el umbral s[i] incluye todos los >= s[i]
    tp, fp, th = tp[last], fp[last], s[last]
    tpr, fpr = tp / n_fake, fp / n_real
    f = np.where(tpr > 0, 1 + (1 - pi) / pi * fpr / np.maximum(tpr, 1e-12), np.inf)
    ok = np.where(f <= target)[0]
    return th[ok].min() if len(ok) else th.max()


def threshold_recall(sv: np.ndarray, yv: np.ndarray, target: float) -> float:
    """Mayor umbral de validación con recall FAKE ≥ ``target``."""
    o = np.sort(sv[yv == 1])[::-1]
    return o[int(np.ceil(target * len(o))) - 1]


def threshold_max_f1_real(sv: np.ndarray, yv: np.ndarray, n_quantiles: int = 2001) -> float:
    """Umbral A: máximo F1 de REAL (se predice REAL si ``s < t``) sobre una rejilla de cuantiles."""
    best = (-1.0, None)
    for t in np.unique(np.quantile(sv, np.linspace(0, 1, n_quantiles))):
        p = sv >= t
        tn = ((~p) & (yv == 0)).sum()
        fn = ((~p) & (yv == 1)).sum()
        fp = (p & (yv == 0)).sum()
        pr = tn / (tn + fn) if tn + fn else 0
        rr = tn / (tn + fp)
        f1 = 2 * pr * rr / (pr + rr) if pr + rr else 0
        if f1 > best[0]:
            best = (f1, t)
    return best[1]


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def production_metrics(s: np.ndarray, y: np.ndarray, t: float, pi: float = PI) -> dict[str, float]:
    """Régimen productivo en el umbral ``t`` (TPR y FPR medidos; precisión y F1 reponderados a ``pi``)."""
    p = s >= t
    tp, fn = (p & (y == 1)).sum(), ((~p) & (y == 1)).sum()
    fp, tn = (p & (y == 0)).sum(), ((~p) & (y == 0)).sum()
    tpr, fpr = tp / (tp + fn), fp / (fp + tn)
    pf = pi * tpr / (pi * tpr + (1 - pi) * fpr) if tpr > 0 else 0
    f1f = 2 * pf * tpr / (pf + tpr) if pf + tpr else 0
    tnr = 1 - fpr
    pr = (1 - pi) * tnr / ((1 - pi) * tnr + pi * (1 - tpr))
    f1r = 2 * pr * tnr / (pr + tnr)
    return dict(
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        tpr=tpr,
        fpr=fpr,
        prec_fake=pf,
        f1_fake=f1f,
        f1_real=f1r,
        fric=friction(tpr, fpr, pi),
        alertas=1000 * (pi * tpr + (1 - pi) * fpr),
    )


def standard_metrics(s: np.ndarray, y: np.ndarray, t: float) -> dict[str, float]:
    """Régimen estándar (prevalencia natural del split) en el umbral ``t``, más AP REAL y AUC."""
    p = s >= t
    tp, fn = (p & (y == 1)).sum(), ((~p) & (y == 1)).sum()
    fp, tn = (p & (y == 0)).sum(), ((~p) & (y == 0)).sum()
    pr, rr = tn / (tn + fn), tn / (tn + fp)
    pf, rf = tp / (tp + fp), tp / (tp + fn)
    return dict(
        acc=(tp + tn) / len(y),
        p_real=pr,
        r_real=rr,
        f1_real=2 * pr * rr / (pr + rr),
        p_fake=pf,
        r_fake=rf,
        f1_fake=2 * pf * rf / (pf + rf),
        ap_real=average_precision_score(1 - y, 1 - s),
        auc=roc_auc_score(y, s),
    )


@dataclass(frozen=True)
class Thresholds:
    A: float
    B: float


def select_thresholds(sv: np.ndarray, yv: np.ndarray) -> Thresholds:
    return Thresholds(A=threshold_max_f1_real(sv, yv), B=threshold_friction(sv, yv, FRICTION_TARGET))


def operating_ladder(
    sv: np.ndarray, yv: np.ndarray, pi: float = PI, frictions=FRICTION_LADDER, recalls=RECALL_LADDER
) -> list[dict[str, float]]:
    """Umbrales de la escalera elegidos en validación: ``[{tipo, objetivo, umbral}]``."""
    out = [dict(tipo="fric", objetivo=float(f), umbral=float(threshold_friction(sv, yv, f, pi))) for f in frictions]
    out += [dict(tipo="recall", objetivo=float(r), umbral=float(threshold_recall(sv, yv, r))) for r in recalls]
    return out
