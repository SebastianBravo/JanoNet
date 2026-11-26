"""Inferencia sobre un video nuevo, con el mismo camino que los datos de entrenamiento.

1. ``preprocess_video``: RetinaFace cada 5 cuadros, reglas de limpieza, recorte cuadrado
   y resize a 224, reescrito en mp4v (la red se entrenó con cuadros que pasaron por ese
   códec).
2. Si el video no cumple las reglas de limpieza (sin rostro, varios rostros, pose no
   frontal) o tiene menos de 32 cuadros útiles, **no se puntúa**: hay que pedir una nueva
   captura.
3. Se decodifican los 32 cuadros centrados (``window_start = (n - 32) // 2``) y el clip
   central de 16 (cuadros 8..23) entra a cada modelo; el ensamble es la media de logits.

Devuelve también lo necesario para mostrar el paso a paso: el cuadro original con la
detección y los 5 puntos, el recorte nativo, el clip y el residuo SRM de un cuadro.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from janonet.data import cache
from janonet.inference.clips import score_clips
from janonet.models.ensemble import ensemble_score
from janonet.models.janonet import JanoNet
from janonet.preprocess.geometry import make_square_crop_from_box
from janonet.preprocess.video import PreprocessResult, load_preprocess_config, preprocess_video

REASONS_ES = {
    "No frames could be evaluated.": "no se pudo evaluar ningún cuadro",
    "Too many frames with >=2 faces.": "aparece más de un rostro en demasiados cuadros",
    "Face not reliably detected.": "no se detecta un rostro de forma estable",
    "No valid single-face frames available.": "no hay cuadros con un único rostro",
    "Face not frontal enough most of the time.": "el rostro no está de frente la mayor parte del tiempo",
    "No processed frames were produced.": "no se obtuvo ningún recorte",
}


def reasons_es(message: str) -> list[str]:
    return [es for en, es in REASONS_ES.items() if en in message]


@dataclass
class VideoResult:
    processable: bool
    reasons: list[str]
    pre: PreprocessResult
    timings: dict[str, float] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    clip: np.ndarray | None = None  # (16, 224, 224, 3) uint8 RGB
    clip_frame_indices: list[int] = field(default_factory=list)  # índices en el video original
    frame_index: int | None = None  # cuadro representativo (original)
    clip_position: int | None = None  # su posición dentro del clip (0..15)
    frame_rgb: np.ndarray | None = None
    detection: dict[str, Any] | None = None  # caja y 5 puntos del cuadro representativo
    square_box: tuple[int, int, int, int] | None = None
    native_crop_rgb: np.ndarray | None = None
    srm_raw: np.ndarray | None = None  # (3, 224, 224)
    srm_norm: np.ndarray | None = None  # (3, 224, 224)


def _read_frame(path: str | Path, k: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    try:
        for _ in range(k + 1):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"no se pudo leer el cuadro {k}")
        return frame
    finally:
        cap.release()


def predict_video(
    path: str | Path,
    models: dict[str, JanoNet],
    detector=None,
    ensemble: tuple[str, str] | None = ("u100", "rs_ca_tr"),
    srm_model: str | None = "rs_ca_tr",
    progress=None,
) -> VideoResult:
    """Procesa un video y puntúa el clip central con cada modelo de ``models``."""
    cfg = load_preprocess_config()
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        processed = Path(tmp) / "face.mp4"
        pre = preprocess_video(path, processed, detector=detector, cfg=cfg, keep_detections=True, progress=progress)
        timings = {"preprocesamiento": time.perf_counter() - t0}
        reasons = reasons_es(pre.clean_rules_message) if not pre.meets_the_clean_rules else []
        if pre.meets_the_clean_rules and pre.processed_frames < cache.N_FRAMES:
            reasons.append(f"el video es demasiado corto ({pre.processed_frames} cuadros útiles; hacen falta 32)")
        if reasons:
            return VideoResult(False, reasons, pre, timings)
        ws = cache.window_start_for(pre.processed_frames)
        frames32 = cache.decode_window(processed, ws)

    off = cache.center_window_offset()
    clip = frames32[off : off + cache.CLIP_LEN]
    dets = pre.detections
    # El video procesado empieza en el primer cuadro con exactamente un rostro.
    first = next(d["frame"] for d in dets if len(d["faces"]) == 1)
    idx = [first + ws + off + t for t in range(cache.CLIP_LEN)]

    t1 = time.perf_counter()
    x = torch.from_numpy(np.ascontiguousarray(clip.transpose(0, 3, 1, 2))).float().div_(255.0).unsqueeze(0)
    scores = {}
    for name, model in models.items():
        scores[name] = float(score_clips(model, x)["score"][0])
    if ensemble and all(m in scores for m in ensemble):
        e = ensemble_score([np.array([scores[m]], dtype=np.float32) for m in ensemble])
        scores["ensemble"] = float(e[0])
    timings["modelos"] = time.perf_counter() - t1

    # Cuadro representativo: el de detección más cercano al centro del clip.
    center = (idx[0] + idx[-1]) / 2
    cands = [d for d in dets if idx[0] <= d["frame"] <= idx[-1] and len(d["faces"]) == 1]
    if not cands:
        cands = [d for d in dets if d["frame"] <= idx[-1] and len(d["faces"]) == 1][-1:]
    det = min(cands, key=lambda d: (abs(d["frame"] - center), d["frame"]))
    k = det["frame"] if idx[0] <= det["frame"] <= idx[-1] else idx[len(idx) // 2]
    frame = _read_frame(path, k)
    box = np.asarray(det["faces"][0]["box"], dtype=np.float32)
    x1, y1, x2, y2 = make_square_crop_from_box(
        box, frame.shape[1], frame.shape[0], cfg["crop"]["margin"], cfg["crop"]["center_shift_y"]
    )
    t = idx.index(k)

    res = VideoResult(
        True,
        [],
        pre,
        timings,
        scores,
        clip,
        idx,
        k,
        t,
        frame[:, :, ::-1].copy(),
        det["faces"][0],
        (x1, y1, x2, y2),
        frame[y1:y2, x1:x2, ::-1].copy(),
    )
    if srm_model and srm_model in models and models[srm_model].srm_residual is not None:
        layer = models[srm_model].srm_residual
        with torch.no_grad():
            xf = x[0, t : t + 1]
            res.srm_raw = layer.raw(xf)[0].numpy()
            res.srm_norm = layer(xf)[0].numpy()
    res.timings["total"] = time.perf_counter() - t0
    return res
