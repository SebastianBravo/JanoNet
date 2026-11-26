"""Paridad del preprocesamiento contra la caché del estudio (20 videos).

Para cada video: se vuelve a procesar desde el original con ``preprocess_video``, se
decodifican los 32 cuadros de la ventana y se comparan con el ``.bin`` del estudio;
el clip central (cuadros 8..23) es el que entra a la red. También se comparan las
magnitudes del manifiesto (cuadros procesados, tamaños de recorte, reglas de limpieza).

Necesita los videos originales (raíces de ``configs/data.yaml``) y
``JANONET_REFERENCE_CACHE`` (la caché del estudio). Lento: RetinaFace en CPU
(~50 min los 20 videos). Escribe el detalle en ``outputs/parity_preprocess.json``.

Tolerancia documentada
----------------------
Con el entorno fijado (opencv-python 4.10) salen idénticos bit a bit 19 de 20 videos.
El restante (``sample_id`` 8079, original FF++ 718) difiere en un máximo de 6 niveles
en 3 564 píxeles de 4,8 M (0,07 %), en los cuadros 20–26 de la ventana. La causa,
aislada reprocesando ese video: OpenCV 4.10 y 4.11 decodifican el video de origen de
forma ligeramente distinta a la 4.12 con la que se generó la caché, RetinaFace devuelve
una caja desplazada 1 px en un cuadro y ese recorte se propaga por el GOP de mp4v. Con
OpenCV 4.12 (mismo código, TensorFlow en CPU o GPU) el video sale idéntico. OpenCV 4.12
exige numpy >= 2, incompatible con TensorFlow 2.10. Las magnitudes del manifiesto son
idénticas en los 20.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from janonet import paths
from janonet.data import cache, sources
from janonet.data import manifest as M

pytestmark = [pytest.mark.slow, pytest.mark.parity]

#: Cuántos videos de cada (dataset, etiqueta). 20 en total.
STRATA = {
    ("dfdc", "REAL"): 3,
    ("dfdc", "FAKE"): 3,
    ("FaceForensics++", "FAKE"): 4,
    ("deeper-forensics", "FAKE"): 3,
    ("deeper-forensics", "REAL"): 2,
    ("FakeAVCeleb", "FAKE"): 3,
    ("FakeAVCeleb", "REAL"): 2,
}
SEED = 20261125
#: Tolerancia documentada (ver docstring): lo medido en el peor de los 20 videos.
MAX_ABS_DIFF = 6
MAX_FRAC_PIXELS_DIFF = 0.001
COMPARED_FIELDS = [
    "num_frames",
    "processed_frames",
    "original_face_width",
    "original_face_height",
    "square_face_width",
    "square_face_height",
    "resize_action",
    "checked_frames",
    "valid_face_frames",
    "missing_face_frames",
    "too_many_faces_frames",
    "meets_the_clean_rules",
]
_REPORT: dict[str, dict] = {}


def parity_selection() -> pd.DataFrame:
    s = M.read_samples()
    # Sin las filas locales de FakeAVCeleb, sus 5 videos simplemente no se seleccionan.
    parts = [
        sub.sample(n, random_state=SEED)
        for (d, lab), n in STRATA.items()
        if len(sub := s[(s.dataset == d) & (s.label == lab)]) >= n
    ]
    return pd.concat(parts).reset_index(drop=True)


SELECTION = parity_selection()


@pytest.fixture(scope="module")
def reference_cache() -> Path:
    env = os.environ.get("JANONET_REFERENCE_CACHE")
    if not env or not Path(env).is_dir():
        pytest.skip("JANONET_REFERENCE_CACHE no apunta a la caché del estudio")
    return Path(env)


@pytest.fixture(scope="module")
def detector():
    pytest.importorskip("retinaface")
    from janonet.preprocess.detect import FaceDetector

    return FaceDetector(0.9)


@pytest.fixture(scope="module", autouse=True)
def write_report():
    yield
    if _REPORT:
        out = paths.project_root() / "outputs" / "parity_preprocess.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(_REPORT, indent=1, ensure_ascii=False), encoding="utf-8")


@pytest.mark.parametrize("row", SELECTION.to_dict("records"), ids=lambda r: f"{r['sample_id']}")
def test_recorte_y_clip_identicos_a_la_cache(row, reference_cache, detector, tmp_path):
    from janonet.preprocess.video import preprocess_video

    try:
        src = sources.resolve_video(row["video_id"])
    except FileNotFoundError:
        pytest.skip(f"video original no disponible: {row['video_id']}")
    res = preprocess_video(src, tmp_path / "v.mp4", detector=detector)
    pre = M.read_preprocessing().set_index("video_id").loc[row["video_id"]]

    _, ref = cache.read_bin(cache.bin_path(reference_cache, row["sample_id"]), verify_crc=True)
    new = cache.decode_window(tmp_path / "v.mp4", int(row["window_start"]))
    diff = np.abs(new.astype(np.int16) - ref.astype(np.int16))
    off = cache.center_window_offset()
    _REPORT[str(row["sample_id"])] = {
        "video_id": row["video_id"],
        "dataset": row["dataset"],
        "label": row["label"],
        "resize_action": row["resize_action"],
        "square_face_width": int(row["square_face_width"]),
        "frames_identical_32": bool((diff == 0).all()),
        "max_abs_diff": int(diff.max()),
        "n_pixels_diff": int((diff > 0).sum()),
        "clip16_identical": bool(np.array_equal(new[off : off + 16], ref[off : off + 16])),
        "fields": {f: [getattr(res, f), pre[f].item() if hasattr(pre[f], "item") else pre[f]] for f in COMPARED_FIELDS},
    }
    for f in COMPARED_FIELDS:
        assert getattr(res, f) == pre[f], f"{f}: {getattr(res, f)} != {pre[f]}"
    frac = float((diff > 0).mean())
    assert (
        diff.max() <= MAX_ABS_DIFF and frac <= MAX_FRAC_PIXELS_DIFF
    ), f"diferencia máxima {diff.max()} en {int((diff > 0).sum())} píxeles ({100 * frac:.3f} %)"
