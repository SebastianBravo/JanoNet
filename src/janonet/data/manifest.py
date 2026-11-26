"""Manifiestos sin medios y tabla maestra.

Dos tablas:

- **preprocessing**: los 140 423 videos procesados, con el resultado del
  preprocesamiento y de las reglas de limpieza. ``manifest_order`` es el orden de
  filas del estudio (necesario para reproducir los ``group_id``).
- **samples**: los 94 234 videos limpios con su partición por identidad,
  ``sample_id`` (clave de todas las predicciones) y columnas derivadas.

Cada tabla se guarda en dos partes:

- ``manifests/*.csv.gz`` (en git): DFDC, FaceForensics++ y DeeperForensics.
- ``manifests/local/fakeavceleb_*.csv.gz`` (**fuera de git**): FakeAVCeleb. Sus
  términos de uso prohíben distribuir datos derivados del dataset, así que esas filas
  las regenera cada usuario desde su propia descarga (DATA.md §3). Los lectores unen
  las dos partes si la local existe.

Sin las filas de FakeAVCeleb los ``sample_id`` y la partición publicados siguen
siendo los del estudio; lo que no se puede es **reconstruir** la partición, porque el
reparto se optimiza sobre los cuatro datasets a la vez.

Origen de la tabla maestra: ``dataset_analysis/src/tfm/data/samples.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from janonet import config, paths
from janonet.data import sources
from janonet.data.identity import build_identity_graph
from janonet.data.split import assign_splits

PREPROCESSING_FILE = "preprocessing.csv.gz"
SAMPLES_FILE = "samples.csv.gz"

#: Resultado del preprocesamiento por video (mismas columnas que el manifiesto que
#: escribía preprocess_v2.py, sin rutas ni parámetros constantes).
PREPROCESSING_COLUMNS = [
    "manifest_order",
    "video_id",
    "dataset",
    "label",
    "method",
    "dfdc_part",
    "dfdc_original",
    "num_frames",
    "processed_frames",
    "fps",
    "video_width",
    "video_height",
    "faces_in_video",
    "nonfrontal_fraction",
    "face_detection_percentage",
    "original_face_width",
    "original_face_height",
    "square_face_width",
    "square_face_height",
    "resize_action",
    "percentage_resize_action",
    "meets_the_clean_rules",
    "clean_rules_message",
    "checked_frames",
    "valid_face_frames",
    "missing_face_frames",
    "too_many_faces_frames",
    "max_faces_in_frame",
    "avg_faces_per_checked_frame",
    "too_many_faces_fraction",
    "valid_face_fraction",
]

SAMPLES_COLUMNS = [
    "sample_id",
    "video_id",
    "dataset",
    "method",
    "label",
    "y",
    "split",
    "group_id",
    "identity_kind",
    "df_donor_actor",
    "num_frames",
    "processed_frames",
    "fps",
    "video_width",
    "video_height",
    "resize_action",
    "percentage_resize_action",
    "original_face_width",
    "original_face_height",
    "square_face_width",
    "square_face_height",
    "nonfrontal_fraction",
    "valid_face_fraction",
    "face_detection_percentage",
    "resize_bucket",
    "duration_stride1_s",
    "duration_stride2_s",
    "duration_bucket",
    "window_start",
]

CACHE_N_FRAMES = 32
CLIP_LEN = 16


#: Dataset cuyas filas no se versionan (términos de uso).
LOCAL_ONLY_DATASET = "FakeAVCeleb"
LOCAL_PREFIX = "fakeavceleb_"

#: Videos por dataset en el estudio, para detectar manifiestos incompletos.
EXPECTED_FAKEAVCELEB = {"preprocessing": 10_209, "samples": 5_787}

_PRE_DTYPES = {"dfdc_part": "string", "dfdc_original": "string", "clean_rules_message": "string"}
_SAMPLES_DTYPES = {"df_donor_actor": "string", "group_id": "string"}


def manifests_dir() -> Path:
    return paths.project_root() / "manifests"


def local_dir() -> Path:
    return manifests_dir() / "local"


def _read(path: Path, dtypes: dict[str, str]) -> pd.DataFrame:
    return pd.read_csv(path, dtype=dtypes, float_precision="round_trip")


def _with_local(public: Path, dtypes: dict[str, str], sort_col: str, include_local: bool) -> pd.DataFrame:
    df = _read(public, dtypes)
    if include_local:
        for name in (LOCAL_PREFIX + public.name, LOCAL_PREFIX + public.name.removesuffix(".gz")):
            local = local_dir() / name
            if local.exists():
                df = pd.concat([df, _read(local, dtypes)], ignore_index=True)
                break
    return df.sort_values(sort_col, kind="mergesort", ignore_index=True)


def read_preprocessing(path: str | Path | None = None, include_local: bool = True) -> pd.DataFrame:
    """Tabla de preprocesamiento. Sin ``path``, une la parte pública y la local."""
    if path:
        return _read(Path(path), _PRE_DTYPES)
    return _with_local(manifests_dir() / PREPROCESSING_FILE, _PRE_DTYPES, "manifest_order", include_local)


def read_samples(path: str | Path | None = None, include_local: bool = True) -> pd.DataFrame:
    """Tabla maestra. Sin ``path``, une la parte pública y la local."""
    if path:
        return _read(Path(path), _SAMPLES_DTYPES)
    return _with_local(manifests_dir() / SAMPLES_FILE, _SAMPLES_DTYPES, "sample_id", include_local)


def has_fakeavceleb(df: pd.DataFrame, table: str) -> bool:
    """¿Están todas las filas de FakeAVCeleb? ``table`` es 'preprocessing' o 'samples'."""
    return int((df["dataset"] == LOCAL_ONLY_DATASET).sum()) == EXPECTED_FAKEAVCELEB[table]


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # mtime fijo en el gzip para que dos exportaciones idénticas den el mismo fichero.
    df.to_csv(path, index=False, lineterminator="\n", compression={"method": "gzip", "mtime": 0})
    return path


def write_split_manifests(pre: pd.DataFrame | None, samples: pd.DataFrame | None, out_dir: Path) -> list[Path]:
    """Escribe cada tabla en su parte pública (``out_dir``) y local (``out_dir/local``)."""
    written = []
    for df, name in ((pre, PREPROCESSING_FILE), (samples, SAMPLES_FILE)):
        if df is None:
            continue
        local = df["dataset"] == LOCAL_ONLY_DATASET
        written.append(write_csv(df[~local], out_dir / name))
        if local.any():
            written.append(write_csv(df[local], out_dir / "local" / (LOCAL_PREFIX + name)))
    return written


# ─────────────────────────────────────────────────────────────────────────────
# Partición
# ─────────────────────────────────────────────────────────────────────────────


def dfdc_meta_from_preprocessing(pre: pd.DataFrame) -> pd.DataFrame:
    d = pre[pre["dataset"] == "dfdc"]
    return pd.DataFrame(
        {
            "filename": d["video_id"].str.split("/", n=1).str[1].to_numpy(),
            "label": d["label"].to_numpy(),
            "original": d["dfdc_original"].astype(object).where(d["dfdc_original"].notna(), None).to_numpy(),
            "part": d["dfdc_part"].to_numpy(),
        }
    )


def build_split(pre: pd.DataFrame, split_cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Grafo de identidades + partición sobre el pool limpio, en el orden ``manifest_order``."""
    cfg = split_cfg or config.load_yaml("data")["split"]
    if not has_fakeavceleb(pre, "preprocessing"):
        raise RuntimeError(
            "faltan las filas de FakeAVCeleb: la partición se optimiza sobre los cuatro datasets a la vez y "
            "no se puede reconstruir sin ellas. Regenéralas desde tu descarga (DATA.md §3)."
        )
    pool = pre[pre[cfg["pool_column"]].astype(bool)].sort_values("manifest_order", kind="mergesort")
    pool = pool.reset_index(drop=True)
    grouped = build_identity_graph(pool, dfdc_meta_from_preprocessing(pre), cfg["dfdc_granularity"])
    return assign_splits(
        grouped,
        cfg["fractions"],
        seed=cfg["seed"],
        force_dfd_blocks_apart=cfg["force_dfd_blocks_apart"],
        max_passes=cfg.get("refine_max_passes", 40),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tabla maestra
# ─────────────────────────────────────────────────────────────────────────────


def _bucket_labels(edges: list[float]) -> list[str]:
    def fmt(v: float, decimals: bool) -> str:
        if v == float("inf"):
            return "inf"
        return f"{v:.2f}" if decimals else f"{int(v)}"

    decimals = any(float(e) != int(e) for e in edges if e != float("inf"))
    return [
        f"({fmt(edges[i], decimals)}, {fmt(edges[i + 1], decimals)}{')' if edges[i + 1] == float('inf') else ']'}"
        for i in range(len(edges) - 1)
    ]


def build_samples(
    pre: pd.DataFrame, split_frame: pd.DataFrame, samples_cfg: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Une preprocesamiento y partición y deriva las columnas de la tabla maestra."""
    cfg = samples_cfg or config.load_yaml("data")["samples"]
    keep = ["video_id", "group_id", "identity_kind", "df_donor_actor", "split"]
    df = split_frame[keep].merge(pre, on="video_id", how="inner", validate="one_to_one")
    if len(df) != len(split_frame):
        raise RuntimeError("el join entre partición y preprocesamiento cambió el número de filas")

    # sample_id: orden de la ruta del video procesado en el desarrollo (estable).
    df["_key"] = df["video_id"].map(sources.legacy_sort_key)
    df = df.sort_values("_key", kind="mergesort", ignore_index=True).drop(columns="_key")
    df.insert(0, "sample_id", np.arange(len(df), dtype=np.int32))
    df["y"] = (df["label"] == "FAKE").astype("int8")

    r_edges = cfg["resize_bucket_edges"]
    df["resize_bucket"] = pd.cut(
        df["percentage_resize_action"], bins=r_edges, labels=_bucket_labels(r_edges), right=True
    ).astype("string")
    if df["resize_bucket"].isna().any():
        raise RuntimeError("percentage_resize_action fuera de los cortes")

    if (df["fps"] <= 0).any():
        raise RuntimeError("fps no positivo")
    df["duration_stride1_s"] = (CLIP_LEN - 1) / df["fps"]
    df["duration_stride2_s"] = 2 * (CLIP_LEN - 1) / df["fps"]
    d_edges = [float(e) for e in cfg["duration_bucket_edges"]]
    df["duration_bucket"] = pd.cut(
        df["duration_stride1_s"], bins=d_edges, labels=_bucket_labels(d_edges), right=True
    ).astype("string")

    if (df["processed_frames"] < CACHE_N_FRAMES).any():
        raise RuntimeError(f"hay videos limpios con menos de {CACHE_N_FRAMES} cuadros")
    df["window_start"] = ((df["processed_frames"] - CACHE_N_FRAMES) // 2).astype("int32")
    return df[SAMPLES_COLUMNS]
