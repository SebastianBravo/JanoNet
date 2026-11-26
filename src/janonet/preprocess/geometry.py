"""Geometría facial: recorte cuadrado, suavizado de puntos y criterio de frontalidad.

Origen: ``deepfake_detection/code/preprocess_v2.py:120-121,239-277,298-346``.
"""

from __future__ import annotations

import numpy as np


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def smooth_landmarks(prev: np.ndarray | None, curr: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    """EMA de los 5 puntos. Solo alimenta el criterio de frontalidad."""
    if prev is None:
        return curr
    return alpha * prev + (1.0 - alpha) * curr


def make_square_crop_from_box(
    box: np.ndarray, img_w: int, img_h: int, margin: float = 0.0, center_shift_y: float = 0.05
) -> tuple[int, int, int, int]:
    """Caja cuadrada de lado ``max(w, h) * (1 + 2*margin)`` centrada en la caja.

    El centro baja ``center_shift_y * lado`` y la caja se recorta a los límites de la
    imagen (en el borde puede dejar de ser cuadrada).
    """
    x1, y1, x2, y2 = box
    face_w = max(0.0, x2 - x1)
    face_h = max(0.0, y2 - y1)
    side = max(face_w, face_h) * (1.0 + 2.0 * margin)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0 + center_shift_y * side
    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = int(round(cx + side / 2.0))
    y2 = int(round(cy + side / 2.0))
    x1 = clamp(x1, 0, img_w - 1)
    y1 = clamp(y1, 0, img_h - 1)
    x2 = clamp(x2, x1 + 1, img_w)
    y2 = clamp(y2, y1 + 1, img_h)
    return x1, y1, x2, y2


def frontal_score(landmarks: np.ndarray) -> float:
    """Puntaje de pose con 5 puntos (menor = más frontal).

    Orden de puntos: ojo 0, ojo 1, nariz, comisura 0, comisura 1 (tal como los
    devuelve ``FaceDetector``). Combina el descentramiento horizontal de la nariz
    respecto a ojos y boca y la inclinación de ambas líneas, normalizados por la
    distancia entre ojos.
    """
    left_eye, right_eye, nose, mouth_left, mouth_right = landmarks
    eye_mid = (left_eye + right_eye) / 2.0
    mouth_mid = (mouth_left + mouth_right) / 2.0
    inter_eye = np.linalg.norm(right_eye - left_eye)
    if inter_eye < 1e-6:
        return 999.0
    nose_eye = abs(nose[0] - eye_mid[0]) / inter_eye
    nose_mouth = abs(nose[0] - mouth_mid[0]) / inter_eye
    eye_tilt = abs(left_eye[1] - right_eye[1]) / inter_eye
    mouth_tilt = abs(mouth_left[1] - mouth_right[1]) / inter_eye
    return float(0.45 * nose_eye + 0.35 * nose_mouth + 0.10 * eye_tilt + 0.10 * mouth_tilt)


def is_frontal(landmarks: np.ndarray, threshold: float) -> bool:
    return frontal_score(landmarks) <= threshold
