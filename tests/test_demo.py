"""Lógica de la demo: elección del punto de operación y predicción de un video nuevo."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from janonet import paths
from janonet.eval import reference as R
from janonet.release import load_card, pick_operating_point


@pytest.mark.parametrize("name", ["u100", "rs_ca_tr", "rs_ca", "ensemble"])
def test_puntos_de_operacion_de_cada_objetivo(name):
    card = load_card(name)
    assert pick_operating_point(card, 0.02, "compromise")["umbral"] == card["thresholds"]["B"]
    for pi in (0.005, 0.01, 0.02, 0.05, 0.1):
        high = pick_operating_point(card, pi, "max_detection")
        assert high["tipo"] == "recall" and high["objetivo"] == 0.95
        thresholds = [pick_operating_point(card, pi, "friction", f)["umbral"] for f in (3, 4, 5, 6, 8, 10)]
        assert thresholds == sorted(thresholds, reverse=True)  # más presupuesto -> umbral más bajo
        for r in (0.80, 0.85, 0.90, 0.95):
            assert pick_operating_point(card, pi, "recall", r)["val"]["tpr"] >= r
    with pytest.raises(ValueError):
        pick_operating_point(card, 0.02, "friction", 7)


@pytest.fixture(scope="module")
def models():
    pytest.importorskip("retinaface")
    from janonet.models.loading import load_model

    out = {}
    for m in ("u100", "rs_ca_tr", "rs_ca"):
        w = paths.weights_dir() / f"{m}.pt"
        if not w.exists():
            pytest.skip("faltan los pesos (scripts/download_weights.py)")
        out[m] = load_model(m, w)
    return out


@pytest.mark.slow
def test_video_sin_rostro_no_es_procesable(models, tmp_path):
    from janonet.inference.video import predict_video

    p = tmp_path / "ruido.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 240))
    rng = np.random.default_rng(0)
    for _ in range(30):
        w.write(cv2.GaussianBlur(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8), (15, 15), 0))
    w.release()
    r = predict_video(p, models)
    assert not r.processable and r.reasons and not r.scores


@pytest.mark.slow
def test_ejemplo_reproduce_los_puntajes_del_estudio(models):
    """El ejemplo 100_M121 (prueba, sample_id 5875) con el pipeline completo desde el video original."""
    from janonet.inference.video import predict_video

    f = Path(paths.project_root()) / "examples" / "deeperforensics_end_to_end_100_M121.mp4"
    r = predict_video(f, models)
    assert r.processable and r.clip.shape == (16, 224, 224, 3)
    ref = R.load_predictions()
    ref = ref[(ref["split"] == "test") & (ref["sample_id"] == 5875)].set_index("config")["score"]
    for m, c in (("u100", "U100"), ("rs_ca_tr", "RS-ca-tr"), ("rs_ca", "RS-ca")):
        assert abs(r.scores[m] - ref[c]) <= 0.02
    assert r.srm_raw.shape == (3, 224, 224) and r.srm_norm.shape == (3, 224, 224)
