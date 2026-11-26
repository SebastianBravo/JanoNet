"""Preprocesamiento de un video: detección, limpieza, recorte cuadrado y redimensionado.

Recorre el video cuadro a cuadro (``cv2.VideoCapture.read``):

1. Detecta con RetinaFace en el cuadro 0, cada ``detect_every`` cuadros y en todos los
   cuadros mientras no haya una detección válida (exactamente un rostro).
2. En los cuadros revisados cuenta rostros; si hay exactamente uno, actualiza la
   última caja y los puntos (estos con EMA, solo para la frontalidad).
3. Cada cuadro posterior a la primera detección válida se recorta con la **caja sin
   suavizar de la última detección válida** (cuadrada, margen 0, centro desplazado
   0,05·lado hacia abajo), se redimensiona a 224x224 con interpolación bicúbica y se
   escribe en un mp4 con códec ``mp4v``.
4. Al final aplica las reglas de limpieza por video.

La caché y la red leen ese mp4 (es decir, ven la pérdida del códec mp4v). Por eso la
inferencia sobre un video nuevo pasa por el mismo camino.

Origen: ``deepfake_detection/code/preprocess_v2.py:467-770`` (``process_video``).
"""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from janonet import config
from janonet.preprocess.geometry import is_frontal, make_square_crop_from_box, smooth_landmarks

INTERPOLATIONS = {"cubic": cv2.INTER_CUBIC, "area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR}


@dataclass
class PreprocessResult:
    """Mismas magnitudes que el manifiesto del estudio (sin rutas ni parámetros)."""

    num_frames: int
    processed_frames: int
    fps: float
    video_width: int
    video_height: int
    faces_in_video: int
    nonfrontal_fraction: float
    face_detection_percentage: float
    original_face_width: int
    original_face_height: int
    square_face_width: int
    square_face_height: int
    resize_action: str
    percentage_resize_action: float
    meets_the_clean_rules: bool
    clean_rules_message: str
    checked_frames: int
    valid_face_frames: int
    missing_face_frames: int
    too_many_faces_frames: int
    max_faces_in_frame: int
    avg_faces_per_checked_frame: float
    too_many_faces_fraction: float
    valid_face_fraction: float
    #: Detecciones de cada cuadro revisado: {frame, faces: [{box, landmarks, score}]}.
    detections: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def as_row(self) -> dict[str, Any]:
        row = asdict(self)
        row.pop("detections")
        return row


def load_preprocess_config() -> dict[str, Any]:
    return config.load_yaml("preprocess")


def evaluate_cleaning_rules(c: dict[str, int], rules: dict[str, Any]) -> tuple[bool, str, dict[str, float]]:
    """Reglas por video (``preprocess_v2.py:684-727``). Devuelve ``(cumple, mensaje, fracciones)``."""
    checked, valid = c["checked_frames"], c["valid_face_frames"]
    msg, ok = "", True
    if checked == 0:
        ok, msg = False, "No frames could be evaluated."
    too_many = c["too_many_faces_frames"] / checked if checked > 0 else 0.0
    valid_frac = valid / checked if checked > 0 else 0.0
    if too_many > rules["max_too_many_faces_fraction"]:
        ok = False
        msg += (
            f" Too many frames with >=2 faces. fraction={too_many:.3f}, "
            f"count={c['too_many_faces_frames']}/{checked}."
        )
    if c["missing_face_frames"] > rules["max_missing_face_frames"] or valid_frac < rules["min_valid_fraction"]:
        ok = False
        missing = c["missing_face_frames"]
        msg += f" Face not reliably detected. valid_fraction={valid_frac:.3f}, missing={missing}/{checked}."
    if valid == 0:
        ok = False
        msg += " No valid single-face frames available."
    nonfrontal = c["nonfrontal_frames"] / valid if valid > 0 else 0.0
    if nonfrontal > rules["max_nonfrontal_fraction"]:
        ok = False
        msg += (
            f" Face not frontal enough most of the time. nonfrontal_fraction={nonfrontal:.3f}, "
            f"threshold={rules['max_nonfrontal_fraction']:.3f}."
        )
    if c["processed_frames"] == 0:
        ok = False
        msg += " No processed frames were produced."
    return (
        ok,
        msg.strip(),
        {"too_many_faces_fraction": too_many, "valid_face_fraction": valid_frac, "nonfrontal_fraction": nonfrontal},
    )


def preprocess_video(
    input_path: str | Path,
    output_path: str | Path,
    detector=None,
    cfg: dict[str, Any] | None = None,
    keep_detections: bool = False,
    progress=None,
) -> PreprocessResult:
    """Procesa un video y escribe el video de rostro 224x224 en ``output_path``.

    ``progress(cuadro, total)``, si se da, se llama tras cada detección (para la demo).
    """
    from janonet.preprocess.detect import FaceDetector

    cfg = cfg or load_preprocess_config()
    det_cfg, fr_cfg, crop_cfg = cfg["detector"], cfg["frontal"], cfg["crop"]
    size = int(cfg["resize"]["size"])
    interp = INTERPOLATIONS[cfg["resize"]["interpolation"]]
    if crop_cfg.get("align_faces"):
        raise NotImplementedError("align_faces=true no se usó en el estudio y no está portado")
    detector = detector or FaceDetector(det_cfg["score_threshold"])

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"no se pudo abrir el video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output_path.with_name(output_path.stem + ".partial.mp4")
    writer = None

    prev_lmk = last_lmk = last_box = None
    first_face = first_square = None
    resize_action, pct_resize = "", 0.0
    c = dict(
        checked_frames=0,
        valid_face_frames=0,
        nonfrontal_frames=0,
        too_many_faces_frames=0,
        missing_face_frames=0,
        processed_frames=0,
    )
    max_faces, total_faces = 0, 0
    detections: list[dict[str, Any]] = []
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % int(det_cfg["detect_every"]) == 0 or last_lmk is None:
                found = detector.detect_faces(frame, min_face_size=int(det_cfg["min_face_size"]))
                c["checked_frames"] += 1
                if progress is not None:
                    progress(frame_idx, num_frames)
                total_faces += len(found)
                max_faces = max(max_faces, len(found))
                if keep_detections:
                    detections.append(
                        {
                            "frame": frame_idx,
                            "faces": [
                                {"box": b.tolist(), "landmarks": pts.tolist(), "score": s} for b, pts, s in found
                            ],
                        }
                    )
                if len(found) == 0:
                    c["missing_face_frames"] += 1
                elif len(found) > 1:
                    c["too_many_faces_frames"] += 1
                else:
                    box, lmk, _ = found[0]
                    lmk = smooth_landmarks(prev_lmk, lmk, alpha=float(fr_cfg["landmark_smoothing_alpha"]))
                    prev_lmk = lmk.copy()
                    last_lmk = lmk.copy()
                    last_box = box.copy()
                    c["valid_face_frames"] += 1
                    if not is_frontal(lmk, float(fr_cfg["threshold"])):
                        c["nonfrontal_frames"] += 1

            if last_lmk is None or last_box is None:
                frame_idx += 1
                continue

            x1, y1, x2, y2 = make_square_crop_from_box(
                last_box,
                img_w=frame.shape[1],
                img_h=frame.shape[0],
                margin=float(crop_cfg["margin"]),
                center_shift_y=float(crop_cfg["center_shift_y"]),
            )
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                frame_idx += 1
                continue
            if first_face is None:
                first_face = (
                    int(round(max(0.0, last_box[2] - last_box[0]))),
                    int(round(max(0.0, last_box[3] - last_box[1]))),
                )
            if first_square is None:
                first_square = (int(round(x2 - x1)), int(round(y2 - y1)))
            if resize_action == "":
                pre_w = crop.shape[1]
                resize_action = "upsample" if size > pre_w else "downsample" if size < pre_w else "same"
                pct_resize = (size - pre_w) / pre_w * 100.0 if pre_w > 0 else 0.0

            crop = cv2.resize(crop, (size, size), interpolation=interp)
            if writer is None:
                writer = cv2.VideoWriter(
                    str(tmp_output), cv2.VideoWriter_fourcc(*cfg["video"]["codec"]), fps, (size, size)
                )
                if not writer.isOpened():
                    raise RuntimeError(f"no se pudo abrir el escritor de video: {tmp_output}")
            writer.write(crop)
            c["processed_frames"] += 1
            frame_idx += 1
    finally:
        if writer is not None:
            writer.release()
        cap.release()

    if tmp_output.exists():
        shutil.move(str(tmp_output), str(output_path))

    ok, msg, fr = evaluate_cleaning_rules(c, cfg["cleaning"])
    checked = c["checked_frames"]
    return PreprocessResult(
        num_frames=num_frames,
        processed_frames=c["processed_frames"],
        fps=float(fps),
        video_width=width,
        video_height=height,
        faces_in_video=max_faces,
        nonfrontal_fraction=fr["nonfrontal_fraction"],
        face_detection_percentage=fr["valid_face_fraction"] * 100.0,
        original_face_width=first_face[0] if first_face else 0,
        original_face_height=first_face[1] if first_face else 0,
        square_face_width=first_square[0] if first_square else 0,
        square_face_height=first_square[1] if first_square else 0,
        resize_action=resize_action,
        percentage_resize_action=pct_resize,
        meets_the_clean_rules=ok,
        clean_rules_message=msg,
        checked_frames=checked,
        valid_face_frames=c["valid_face_frames"],
        missing_face_frames=c["missing_face_frames"],
        too_many_faces_frames=c["too_many_faces_frames"],
        max_faces_in_frame=max_faces,
        avg_faces_per_checked_frame=(total_faces / checked) if checked else 0.0,
        too_many_faces_fraction=fr["too_many_faces_fraction"],
        valid_face_fraction=fr["valid_face_fraction"],
        detections=detections,
    )


def crop_box_for_frame(detections: list[dict[str, Any]], frame_idx: int) -> dict[str, Any] | None:
    """Detección vigente para un cuadro: la última con exactamente un rostro en un cuadro <= ``frame_idx``."""
    valid = [d for d in detections if d["frame"] <= frame_idx and len(d["faces"]) == 1]
    return valid[-1] if valid else None


def to_numpy_box(d: dict[str, Any]) -> np.ndarray:
    return np.asarray(d["faces"][0]["box"], dtype=np.float32)
