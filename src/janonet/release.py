"""Fichas de publicación de cada modelo: umbrales, escalera de operación y métricas.

Cada modelo publicado lleva un JSON (``configs/release/<modelo>.json``) con:

- la config del modelo y su puntaje oficial;
- las estadísticas SRM con que se normaliza (si tiene rama SRM);
- los umbrales A (máximo F1 REAL) y B (fricción ≤ 5 a π = 0,02), elegidos en validación;
- la **escalera de operación** para cada prevalencia π ∈ {0,005, 0,01, 0,02, 0,05, 0,10}:
  fricción objetivo 3, 4, 5, 6, 8, 10, 15 y 20 (umbral elegido en validación a esa π) y
  recall objetivo 0,80, 0,85, 0,90 y 0,95; cada punto con su TPR, FPR y fricción en
  validación y, como contexto, en prueba;
- las métricas de prueba del documento en los dos regímenes;
- el fichero de pesos, su tamaño y SHA256 (los modelos, no el ensamble).

Todo se calcula con ``janonet.eval`` sobre las predicciones de referencia del estudio,
así que coincide con las tablas del documento.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from janonet import __version__
from janonet.eval.operating import (
    FRICTION_LADDER,
    PREVALENCES,
    RECALL_LADDER,
    production_metrics,
    standard_metrics,
    threshold_friction,
    threshold_recall,
)
from janonet.eval.study import Study

DEFINITIONS = {
    "score": "puntaje de falsedad en [0, 1] (FAKE = 1); no es una probabilidad calibrada",
    "friction": "1 + (1 - pi)/pi * FPR/TPR: alertas por deepfake detectado",
    "threshold_A": "máximo F1 de la clase REAL en validación (rejilla de 2001 cuantiles)",
    "threshold_B": "menor umbral de validación con fricción <= 5 a pi = 0,02 (sin factor de deriva)",
    "ladder_friction": "menor umbral de validación con fricción <= objetivo a la pi indicada",
    "ladder_recall": "mayor umbral de validación con recall FAKE >= objetivo (no depende de pi)",
    "test_metrics": "contexto: medidas en prueba con el umbral elegido en validación",
}


def _pm(s: np.ndarray, y: np.ndarray, t: float, pi: float) -> dict[str, float]:
    m = production_metrics(s, y, t, pi)
    return {k: float(m[k]) for k in ("tpr", "fpr", "fric", "prec_fake", "alertas")}


def operating_points(
    st: Study, cfg: str, prevalences=PREVALENCES, frictions=FRICTION_LADDER, recalls=RECALL_LADDER
) -> dict[str, list[dict[str, Any]]]:
    sv, s = st.val_scores[cfg], st.test_scores[cfg]
    out: dict[str, list[dict[str, Any]]] = {}
    for pi in prevalences:
        pts = []
        for f in frictions:
            t = float(threshold_friction(sv, st.val_y, f, pi))
            pts.append(
                {
                    "tipo": "fric",
                    "objetivo": float(f),
                    "umbral": t,
                    "val": _pm(sv, st.val_y, t, pi),
                    "test": _pm(s, st.test_y, t, pi),
                }
            )
        for r in recalls:
            t = float(threshold_recall(sv, st.val_y, r))
            pts.append(
                {
                    "tipo": "recall",
                    "objetivo": float(r),
                    "umbral": t,
                    "val": _pm(sv, st.val_y, t, pi),
                    "test": _pm(s, st.test_y, t, pi),
                }
            )
        out[f"{pi:g}"] = pts
    return out


def test_metrics(st: Study, cfg: str) -> dict[str, Any]:
    s, y = st.test_scores[cfg], st.test_y
    th = st.thresholds[cfg]
    std = {k: float(v) for k, v in standard_metrics(s, y, th["A"]).items()}
    prod = _pm(s, y, th["B"], 0.02)
    return {
        "standard_at_A": std,
        "production_at_B_pi_0.02": prod,
        "n_test": int(len(y)),
        "n_test_real": int((y == 0).sum()),
        "n_test_fake": int((y == 1).sum()),
    }


def model_card(
    st: Study,
    cfg: str,
    name: str,
    model_config: dict[str, Any],
    srm_stats: dict[str, Any] | None,
    weights: dict[str, Any] | None,
) -> dict[str, Any]:
    th = st.thresholds[cfg]
    return {
        "name": name,
        "study_config": cfg,
        "janonet_version": __version__,
        "definitions": DEFINITIONS,
        "config": model_config,
        "srm_stats": srm_stats,
        "thresholds": {"A": float(th["A"]), "B": float(th["B"]), "B_pi": 0.02, "B_friction_target": 5.0},
        "operating_points": operating_points(st, cfg),
        "test_metrics": test_metrics(st, cfg),
        "weights": weights,
    }
