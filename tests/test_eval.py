"""Unidades de la evaluación: fricción, umbrales y métricas productivas."""

from __future__ import annotations

import numpy as np
import pytest

from janonet.eval import operating as O


def test_friccion_a_dos_por_ciento():
    # 1 + 49 · FPR/TPR
    assert O.friction(0.9, 0.09) == pytest.approx(1 + 49 * 0.1)
    assert O.friction(0.0, 0.1) == np.inf


def test_umbral_B_es_el_menor_con_friccion_en_objetivo():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(900, int), np.zeros(100, int)]
    s = np.r_[rng.beta(5, 2, 900), rng.beta(2, 5, 100)].astype(np.float32)
    t = O.threshold_friction(s, y, 5.0)
    m = O.production_metrics(s, y, t)
    assert m["fric"] <= 5.0
    lower = np.unique(s)[np.unique(s) < t]
    assert len(lower) == 0 or O.production_metrics(s, y, lower.max())["fric"] > 5.0


def test_umbral_de_recall():
    y = np.r_[np.ones(10, int), np.zeros(5, int)]
    s = np.r_[np.linspace(0.1, 1.0, 10), np.full(5, 0.05)]
    t = O.threshold_recall(s, y, 0.80)
    assert O.production_metrics(s, y, t)["tpr"] >= 0.80
    assert t == pytest.approx(0.3)


def test_metricas_productivas_reponderan_a_pi():
    y = np.r_[np.ones(100, int), np.zeros(100, int)]
    s = np.r_[np.full(90, 0.9), np.full(10, 0.1), np.full(5, 0.9), np.full(95, 0.1)]
    m = O.production_metrics(s, y, 0.5, pi=0.02)
    assert (m["tpr"], m["fpr"]) == (0.9, 0.05)
    assert m["prec_fake"] == pytest.approx(0.02 * 0.9 / (0.02 * 0.9 + 0.98 * 0.05))
    assert m["fric"] == pytest.approx(1 / m["prec_fake"])
    assert m["alertas"] == pytest.approx(1000 * (0.02 * 0.9 + 0.98 * 0.05))


def test_clopper_pearson_extremos():
    assert O.clopper_pearson(0, 10)[0] == 0.0
    assert O.clopper_pearson(10, 10)[1] == 1.0
    lo, hi = O.clopper_pearson(5, 10)
    assert lo < 0.5 < hi


def test_escalera_incluye_15_y_20():
    rng = np.random.default_rng(1)
    y = np.r_[np.ones(500, int), np.zeros(200, int)]
    s = np.r_[rng.random(500) * 0.5 + 0.5, rng.random(200)].astype(np.float32)
    lad = O.operating_ladder(s, y)
    assert [r["objetivo"] for r in lad if r["tipo"] == "fric"] == [3, 4, 5, 6, 8, 10, 15, 20]
    assert [r["objetivo"] for r in lad if r["tipo"] == "recall"] == [0.80, 0.85, 0.90, 0.95]
