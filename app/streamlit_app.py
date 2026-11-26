"""Demo de JanoNet en Streamlit (CPU).

    streamlit run app/streamlit_app.py

Necesita los pesos en ``weights/`` (``python scripts/download_weights.py``) y el extra
``[app]`` (incluye RetinaFace). Muestra el paso a paso del preprocesamiento y de la rama
SRM, los puntajes de U100, RS-ca-tr, RS-ca y el ensamble, y la decisión de cada modelo en
el punto de operación que corresponde al objetivo del negocio y a la prevalencia elegidos.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

from janonet import paths
from janonet.inference.video import predict_video
from janonet.models.loading import load_config, load_model
from janonet.release import load_card, pick_operating_point

MODELS = ("u100", "rs_ca_tr", "rs_ca")
SHOWN = ("u100", "rs_ca_tr", "rs_ca", "ensemble")
PREVALENCES = {"0,5 %": 0.005, "1 %": 0.01, "2 %": 0.02, "5 %": 0.05, "10 %": 0.1}
OBJECTIVES = {
    "Máxima detección": "max_detection",
    "Presupuesto de fricción": "friction",
    "Recall mínimo": "recall",
    "Compromiso": "compromise",
}
P_SCALE = 99.0

st.set_page_config(page_title="JanoNet", page_icon="🎭", layout="wide")


# ── Recursos cacheados ───────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Cargando modelos…")
def get_models() -> dict:
    torch.set_num_threads(max(1, (os.cpu_count() or 2)))
    out = {}
    for m in MODELS:
        w = paths.weights_dir() / f"{m}.pt"
        if not w.exists():
            return {}
        out[m] = load_model(m, w, device="cpu")
    return out


@st.cache_resource(show_spinner="Cargando el detector de rostros (RetinaFace)…")
def get_detector():
    from janonet.preprocess.detect import FaceDetector

    return FaceDetector(0.9)


@st.cache_data(show_spinner=False)
def get_cards() -> dict:
    return {m: load_card(m) for m in SHOWN}


@st.cache_data(show_spinner=False)
def get_examples() -> list[dict]:
    meta = json.loads((paths.project_root() / "examples" / "examples.json").read_text(encoding="utf-8"))["examples"]
    return [e for e in meta if (paths.project_root() / "examples" / e["file"]).exists()]


def cache_dir() -> Path:
    env = os.environ.get("JANONET_DEMO_CACHE")
    return Path(env) if env else paths.project_root() / "outputs" / "demo_cache"


def analyze(video_bytes: bytes, name: str) -> dict:
    """Procesa el video una sola vez por contenido; el resultado se guarda en disco por hash.

    RetinaFace en CPU es lo lento (≈ 2 s por detección en 720p): un video de 15–20 s tarda
    unos minutos la primera vez y nada las siguientes.
    """
    key = hashlib.sha256(video_bytes).hexdigest()
    f = cache_dir() / f"{key}.pkl"
    if f.exists():
        try:
            with open(f, "rb") as fh:
                res = pickle.load(fh)
            res["timings"] = {**res["timings"], "desde_cache": True}
            return res
        except Exception:  # noqa: BLE001 - una caché corrupta se recalcula
            pass
    bar = st.progress(0.0, text="Detectando el rostro cuadro a cuadro (RetinaFace, CPU)…")

    def progress(i: int, n: int) -> None:
        if n > 0:
            bar.progress(min(i / n, 1.0), text=f"Detectando el rostro: cuadro {i} de {n} (RetinaFace, CPU)…")

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / ("video" + Path(name).suffix)
        p.write_bytes(video_bytes)
        res = predict_video(p, get_models(), detector=get_detector(), progress=progress).__dict__
    bar.empty()
    f.parent.mkdir(parents=True, exist_ok=True)
    with open(f, "wb") as fh:
        pickle.dump(res, fh)
    return res


# ── Dibujo ───────────────────────────────────────────────────────────────────


def draw_detection(frame: np.ndarray, det: dict, square: tuple[int, int, int, int]) -> np.ndarray:
    img = np.ascontiguousarray(frame.copy())
    g = max(1, round(min(img.shape[:2]) / 360))
    x1, y1, x2, y2 = square
    cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), (0, 200, 255), g, cv2.LINE_AA)
    b = [int(round(v)) for v in det["box"]]
    cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), g, cv2.LINE_AA)
    for x, y in det["landmarks"]:
        cv2.circle(img, (int(round(x)), int(round(y))), 2 * g + 1, (255, 0, 0), -1, cv2.LINE_AA)
    return img


def strip(clip: np.ndarray, gap: int = 4) -> np.ndarray:
    n, h, w, _ = clip.shape
    out = np.full((h, n * w + (n - 1) * gap, 3), 255, np.uint8)
    for i in range(n):
        out[:, i * (w + gap) : i * (w + gap) + w] = clip[i]
    return out


def gray(a: np.ndarray, scale: float) -> np.ndarray:
    return (np.clip(np.abs(a) / scale, 0, 1) * 255).round().astype(np.uint8)


# ── Interfaz ─────────────────────────────────────────────────────────────────

st.title("JanoNet · detección de deepfakes de tipo face swap")
st.caption(
    "Ramas RGB y SRM con atención cruzada, evaluadas bajo prevalencia realista de deepfakes. "
    "Los puntajes son **puntajes de falsedad**, no probabilidades calibradas: solo se comparan con "
    "el umbral del punto de operación. Uso de investigación; no tomar decisiones sin revisión humana."
)

models = get_models()
if not models:
    st.error("Faltan los pesos. Ejecuta `python scripts/download_weights.py` y recarga la página.")
    st.stop()
cards = get_cards()

with st.sidebar:
    st.header("Video")
    examples = get_examples()
    source = st.radio("Origen", ["Ejemplo", "Subir un video"], index=0)
    chosen = None
    if source == "Ejemplo":
        labels = {f"{e['file']} ({e['label']})": e for e in examples}
        chosen = labels[st.selectbox("Ejemplo", list(labels))] if labels else None
        upload = None
    else:
        upload = st.file_uploader("Video (mp4, avi, mov, mkv, webm)", type=["mp4", "avi", "mov", "mkv", "webm"])

    st.header("Punto de operación")
    pi_label = st.select_slider("Prevalencia esperada de deepfakes", list(PREVALENCES), value="2 %")
    pi = PREVALENCES[pi_label]
    obj_label = st.radio("Objetivo del negocio", list(OBJECTIVES), index=3)
    objective = OBJECTIVES[obj_label]
    value = None
    if objective == "friction":
        value = st.select_slider("Fricción máxima (alertas por deepfake detectado)", [3, 4, 5, 6, 8, 10], value=5)
    elif objective == "recall":
        value = st.select_slider("Recall mínimo de deepfakes", [0.80, 0.85, 0.90, 0.95], value=0.90)
    st.caption(
        {
            "max_detection": "Detectar al menos el 95 % de los deepfakes de validación.",
            "friction": "El umbral más bajo cuya fricción en validación no supera el presupuesto.",
            "recall": "El umbral más alto que detecta al menos ese recall en validación.",
            "compromise": "Fricción ≤ 5 en validación a la prevalencia elegida (umbral B si π = 2 %).",
        }[objective]
    )

if source == "Ejemplo" and chosen is not None:
    video_bytes = (paths.project_root() / "examples" / chosen["file"]).read_bytes()
    name = chosen["file"]
elif source == "Subir un video" and upload is not None:
    video_bytes, name, chosen = upload.getvalue(), upload.name, None
else:
    st.info("Elige un ejemplo o sube un video con un solo rostro de frente.")
    st.stop()

left, right = st.columns([1, 2])
with left:
    st.video(video_bytes)
    if chosen and chosen.get("citation"):
        st.info(f"**Fuente:** {chosen['note']}\n\n**Cita:** {chosen['citation']}\n\n{chosen.get('terms') or ''}")

with right:
    r = analyze(video_bytes, name)

    if not r["processable"]:
        st.error("**No procesable: solicitar recaptura.**")
        st.write("Motivo: " + "; ".join(r["reasons"]) + ".")
        st.caption("El sistema solo puntúa videos con un único rostro, visible y de frente durante la captura.")
        st.caption(f"Tiempo: {r['timings'].get('preprocesamiento', 0):.1f} s")
        st.stop()

    rows = []
    st.subheader("Puntajes y decisión")
    cards_cols = st.columns(2)
    for i, m in enumerate(SHOWN):
        p = pick_operating_point(cards[m], pi, objective, value)
        s = r["scores"][m]
        alert = s >= p["umbral"]
        name_m = load_config(m)["meta"]["display_name"]
        cards_cols[i % 2].metric(
            name_m,
            f"{s:.4f}",
            ("FALSO · alerta" if alert else "real") + f" (umbral {p['umbral']:.4f})",
            delta_color="inverse" if alert else "normal",
        )
        rows.append(
            {
                "Modelo": name_m,
                "Puntaje": round(s, 4),
                "Umbral": round(p["umbral"], 4),
                "Decisión": "FALSO" if alert else "real",
                "Recall": f"{100 * p['test']['tpr']:.1f} %",
                "FPR": f"{100 * p['test']['fpr']:.1f} %",
                "Fricción": round(p["test"]["fric"], 2),
                "Alertas/1000": round(p["test"]["alertas"], 1),
            }
        )
    st.caption(
        f"Prevalencia {pi_label}, objetivo «{obj_label}». Recall, FPR, fricción y alertas son los **medidos en "
        "prueba** con el umbral elegido en validación (contexto, no una garantía para este video). "
        f"Tiempo en CPU: {r['timings']['total']:.1f} s ({r['timings']['preprocesamiento']:.1f} s de "
        f"preprocesamiento, {r['timings']['modelos']:.1f} s de modelos)"
        + (" — resultado reutilizado de la caché de la demo." if r["timings"].get("desde_cache") else ".")
    )

st.markdown("**Punto de operación por modelo** (recall, FPR, fricción y alertas esperados, medidos en prueba)")
st.dataframe(pd.DataFrame(rows).set_index("Modelo"), use_container_width=True)

st.subheader("Paso a paso")
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    st.image(
        draw_detection(r["frame_rgb"], r["detection"], r["square_box"]),
        caption=f"1 · Cuadro {r['frame_index']}: detección RetinaFace (verde), 5 puntos faciales (rojo) y "
        "recorte cuadrado de margen 0 (azul)",
        use_column_width=True,
    )
with c2:
    h, w = r["native_crop_rgb"].shape[:2]
    st.image(r["native_crop_rgb"], caption=f"2 · Recorte nativo ({w} × {h} px)", use_column_width=True)
with c3:
    pre = r["pre"]
    st.image(
        r["clip"][r["clip_position"]],
        caption=f"3 · Redimensionado a 224 × 224 ({pre.resize_action}, "
        f"{pre.percentage_resize_action:+.0f} %), tal como entra a la red",
        use_column_width=True,
    )
st.image(
    strip(r["clip"]),
    caption=f"4 · Clip central de 16 cuadros (cuadros {r['clip_frame_indices'][0]}–"
    f"{r['clip_frame_indices'][-1]} del video)",
    use_column_width=True,
)

if r["srm_raw"] is not None:
    s_raw = float(np.percentile(np.abs(r["srm_raw"]), P_SCALE))
    s_norm = float(np.percentile(np.abs(r["srm_norm"]), P_SCALE))
    cols = st.columns(6)
    names = ["K1 · 3 × 3", "K2 · 5 × 5", "K3 · 1 × 3 horizontal"]
    for i in range(3):
        cols[i].image(gray(r["srm_raw"][i], s_raw), caption=f"5 · Filtro {names[i]}", use_column_width=True)
        cols[3 + i].image(
            gray(r["srm_norm"][i], s_norm), caption=f"6 · Residuo normalizado, canal {i + 1}", use_column_width=True
        )
    st.caption(
        f"Rama SRM del cuadro {r['frame_index']}: respuesta de los 3 filtros fijos (sumada sobre R, G y B) y residuo "
        f"normalizado con las estadísticas de entrenamiento, que es lo que entra a la ResNet-18. Se muestra |x| en "
        f"gris con una sola escala por grupo (percentil {P_SCALE:g}): filtros {s_raw:.4f}, residuo {s_norm:.2f} "
        "desviaciones típicas."
    )
