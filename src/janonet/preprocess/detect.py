"""Detector de rostros RetinaFace (paquete ``retina-face`` sobre TensorFlow 2.10).

TensorFlow se importa al construir el detector, no al importar ``janonet``: quien solo
evalúa o entrena con la caché no necesita el extra ``[preprocess]``.

Origen: ``deepfake_detection/code/preprocess_v2.py:353-412``.
"""

from __future__ import annotations

import os

import numpy as np

#: Orden de puntos devuelto (el del detector original): left_eye, right_eye, nose,
#: mouth_left, mouth_right, con los nombres de RetinaFace.
LANDMARK_ORDER = ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")


class FaceDetector:
    def __init__(self, score_threshold: float = 0.9):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        try:
            import tensorflow as tf
            from retinaface import RetinaFace
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise ImportError(
                'Falta el extra de preprocesamiento: pip install -e ".[preprocess]" '
                "(en macOS con Apple Silicon, además: pip install --no-deps retina-face==0.0.17)"
            ) from exc
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:  # noqa: BLE001
                pass
        self._retinaface = RetinaFace
        self.score_threshold = score_threshold

    def detect_faces(
        self, frame_bgr: np.ndarray, min_face_size: int = 60
    ) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """``[(caja x1y1x2y2, puntos 5x2, score)]`` de los rostros que superan umbral y tamaño."""
        resp = self._retinaface.detect_faces(frame_bgr, threshold=self.score_threshold)
        results = []
        if not isinstance(resp, dict) or not resp:
            return results
        for identity in resp.values():
            score = float(identity.get("score", 0.0))
            if score < self.score_threshold:
                continue
            area, lmk = identity.get("facial_area"), identity.get("landmarks")
            if area is None or lmk is None:
                continue
            x1, y1, x2, y2 = area
            if min(x2 - x1, y2 - y1) < min_face_size:
                continue
            points = np.array([lmk[k] for k in LANDMARK_ORDER], dtype=np.float32)
            results.append((np.array([x1, y1, x2, y2], dtype=np.float32), points, score))
        return results
