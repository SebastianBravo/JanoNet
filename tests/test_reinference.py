"""Reinferencia con el pipeline nuevo frente a las predicciones guardadas (CPU, float32).

Tolerancia medida sobre 353 videos de prueba (200 al azar + los 156 en que U100 saturaba,
``outputs/reinfer_report.json``): diferencia máxima 0,020 (E6), 0,018 (RS-ca-tr),
0,012 (U100), 0,004 (RS-ca), 0,008 (ensamble), y ninguna decisión distinta en el umbral B.
Aquí se comprueban 12 videos para que corra en un par de minutos. En esta muestra un
video de RS-ca-tr cambia de decisión: su puntaje guardado está a menos de 0,02 del
umbral B, así que el test solo exige que los cambios ocurran dentro de esa franja.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from janonet.data import manifest as M
from janonet.eval import reference as R
from janonet.inference.clips import predict_cached
from janonet.models.loading import load_config, load_model

from .conftest import MODELS

pytestmark = [pytest.mark.slow, pytest.mark.parity, pytest.mark.checkpoints]

MAX_ABS = {"u100": 0.015, "rs_ca": 0.01, "rs_ca_tr": 0.02, "e6": 0.025}


@pytest.fixture(scope="module")
def cache_dir() -> Path:
    env = os.environ.get("JANONET_CACHE")
    if not env or not Path(env).is_dir():
        pytest.skip("JANONET_CACHE no apunta a la caché de 32 cuadros")
    return Path(env)


@pytest.mark.parametrize("name", MODELS)
def test_reinferencia_cerca_de_lo_guardado(name, cache_dir, checkpoint_for):
    s = M.read_samples(include_local=False)
    ids = np.sort(np.random.default_rng(0).choice(s.loc[s["split"] == "test", "sample_id"], 12, replace=False))
    model = load_model(name, checkpoint_for(name))
    new = predict_cached(model, ids, cache_dir)["score"]
    ref = R.load_predictions(include_local=False)
    cfg = load_config(name)["meta"]["study_config"]
    ref = ref[(ref["config"] == cfg) & (ref["split"] == "test")].set_index("sample_id").loc[ids, "score"].to_numpy()
    th = R.document_thresholds()[cfg]["B"]
    assert np.abs(new - ref).max() <= MAX_ABS[name]
    # Una decisión solo puede cambiar si el puntaje guardado está a menos de la
    # tolerancia del umbral B.
    flips = (new >= th) != (ref >= th)
    assert (np.abs(ref[flips] - th) <= MAX_ABS[name]).all()
    print(
        f"{name}: máx |dif| {np.abs(new - ref).max():.4f}; decisiones distintas en B: {int(flips.sum())}",
        [(float(r), float(n)) for r, n in zip(ref[flips], new[flips], strict=True)],
        f"umbral {th}",
    )
