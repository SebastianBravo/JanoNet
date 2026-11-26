"""Fuentes de video: identificadores, etiquetas, métodos y localización en disco.

``video_id``
------------
Identificador canónico de cada video, relativo a la raíz de su dataset y con ``/``:

==================================================================  ================  =====
``video_id``                                                        ``dataset``        label
==================================================================  ================  =====
``dfdc/<fichero>.mp4``                                              dfdc              metadata
``ffpp/manipulated_sequences/<Método>/c23/videos/<t>_<s>.mp4``      FaceForensics++   FAKE
``ffpp/original_sequences/youtube/c23/videos/<t>.mp4``              deeper-forensics  REAL
``deeperforensics/manipulated_videos/<variante>/<t>_<actor>.mp4``   deeper-forensics  FAKE
``fakeavceleb/<tipo>/<etnia>/<género>/<idNNNNN>/<fichero>.mp4``     FakeAVCeleb       tipo
==================================================================  ================  =====

Los originales de FF++ llevan ``dataset = deeper-forensics`` porque así entraron al
estudio: son los videos objetivo de DeeperForensics-1.0 (el desarrollo los leyó de
``deeper_forensics_processed/REAL``). Se conserva la etiqueta para que todas las cifras
por dataset coincidan con el documento.

Origen: ``dataset_analysis/code/add_labels.py`` (etiquetas),
``dataset_unzip/code/unzip_*.py`` (organización de las fuentes) y el notebook
``code/notebook_analisis_identidades.ipynb`` (metadata de DFDC).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

import pandas as pd

from janonet import config

#: Prefijo del video_id -> clave de raíz en configs/data.yaml.
PREFIXES = ("dfdc", "ffpp", "deeperforensics", "fakeavceleb")

DATASET_DFDC = "dfdc"
DATASET_FFPP = "FaceForensics++"
DATASET_DF = "deeper-forensics"
DATASET_FAVC = "FakeAVCeleb"

FAVC_TYPE_TO_LABEL = {"FakeVideo-RealAudio": "FAKE", "RealVideo-RealAudio": "REAL"}


def prefix_of(video_id: str) -> str:
    prefix = video_id.split("/", 1)[0]
    if prefix not in PREFIXES:
        raise ValueError(f"video_id con prefijo desconocido: {video_id!r}")
    return prefix


def dataset_of(video_id: str) -> str:
    prefix = prefix_of(video_id)
    if prefix == "dfdc":
        return DATASET_DFDC
    if prefix == "fakeavceleb":
        return DATASET_FAVC
    if prefix == "deeperforensics" or video_id.startswith("ffpp/original_sequences/"):
        return DATASET_DF
    return DATASET_FFPP


def method_of(video_id: str) -> str:
    """Método de manipulación legible desde la ruta (DFDC y FakeAVCeleb: ver ``fakeavceleb_methods``)."""
    parts = video_id.split("/")
    prefix = parts[0]
    if prefix == "ffpp":
        return "original" if parts[1] == "original_sequences" else parts[2]
    if prefix == "deeperforensics":
        return parts[2]
    if prefix == "fakeavceleb":
        return "original" if parts[1] == "RealVideo-RealAudio" else "unknown"
    return "unknown"  # DFDC no publica el método por video


def label_from_path(video_id: str) -> str | None:
    """Etiqueta deducible de la ruta. DFDC devuelve ``None`` (sale de su metadata)."""
    parts = video_id.split("/")
    prefix = parts[0]
    if prefix == "ffpp":
        return "REAL" if parts[1] == "original_sequences" else "FAKE"
    if prefix == "deeperforensics":
        return "FAKE"
    if prefix == "fakeavceleb":
        return FAVC_TYPE_TO_LABEL[parts[1]]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Orden heredado del desarrollo
# ─────────────────────────────────────────────────────────────────────────────
# En el desarrollo, `sample_id` se asignó ordenando por la ruta absoluta del video
# procesado (Windows, separador "\") y las filas del manifiesto seguían el orden de
# la ruta absoluta del video de entrada. Todas las rutas compartían el mismo prefijo
# por dataset, así que ordenar por el resto de la ruta da el mismo orden. Estas dos
# funciones reconstruyen ese resto a partir del video_id, para que una reconstrucción
# desde cero asigne los mismos sample_id y los mismos group_id.

_LEGACY_PROCESSED = {
    "dfdc": "dfdc_train_all_processed_test",
    "ffpp_manip": "face_forensics_processed",
    "ffpp_orig": "deeper_forensics_processed\\REAL",
    "df": "deeper_forensics_processed\\FAKE",
    "favc": "FakeAVCeleb_processed",
}
_LEGACY_INPUT = {  # (orden del manifiesto consolidado, carpeta de entrada)
    "df": (0, "deeper-forensics_unzipped\\FAKE"),
    "ffpp_orig": (0, "deeper-forensics_unzipped\\REAL"),
    "dfdc": (1, "dfdc_train_all_unzipped\\videos"),
    "ffpp_manip": (2, "ff++\\manipulated_sequences"),
    "favc": (3, "FakeAVCeleb_unzipped"),
}
_FAVC_LEGACY_TOP = {"FakeVideo-RealAudio": "FAKE", "RealVideo-RealAudio": "REAL"}


def _legacy_parts(video_id: str) -> tuple[str, list[str]]:
    parts = video_id.split("/")
    prefix = parts[0]
    if prefix == "dfdc":
        return "dfdc", parts[1:]
    if prefix == "ffpp":
        if parts[1] == "original_sequences":
            return "ffpp_orig", parts[1:]
        return "ffpp_manip", parts[2:]
    if prefix == "deeperforensics":
        return "df", parts[2:]
    return "favc", [_FAVC_LEGACY_TOP[parts[1]], *parts[2:]]


def legacy_sort_key(video_id: str) -> str:
    """Clave cuyo orden reproduce el ``sample_id`` del desarrollo."""
    kind, rest = _legacy_parts(video_id)
    return "\\".join([_LEGACY_PROCESSED[kind], *rest])


def legacy_manifest_order_key(video_id: str) -> tuple[int, str]:
    """Clave cuyo orden reproduce el orden de filas del manifiesto consolidado."""
    kind, rest = _legacy_parts(video_id)
    order, folder = _LEGACY_INPUT[kind]
    if kind == "df":
        return order, "\\".join([folder, *rest])
    return order, "\\".join([folder, *rest])


# ─────────────────────────────────────────────────────────────────────────────
# Metadata de DFDC y FakeAVCeleb
# ─────────────────────────────────────────────────────────────────────────────


def load_dfdc_metadata(metadata_files: Iterable[Path]) -> pd.DataFrame:
    """Lee los ``metadata.json`` de las partes de DFDC.

    La parte se toma de ``part_<NN>`` en el nombre del fichero o de su carpeta
    (``dfdc_train_part_NN/metadata.json`` en la descarga oficial).
    Devuelve ``filename, label, original, part``.
    """
    rows = []
    for path in sorted(Path(p) for p in metadata_files):
        m = re.search(r"part_(\d+)", path.name) or re.search(r"part_(\d+)", path.parent.name)
        part = m.group(1) if m else path.stem
        entries = json.loads(path.read_text(encoding="utf-8"))
        for fname, info in entries.items():
            rows.append({"filename": fname, "label": info.get("label"), "original": info.get("original"), "part": part})
    df = pd.DataFrame(rows)
    if df["filename"].duplicated().any():
        raise ValueError("nombres de fichero de DFDC repetidos entre partes")
    return df


def find_dfdc_metadata(root: Path) -> list[Path]:
    """Busca los JSON de metadata de DFDC bajo ``root`` (layout oficial o aplanado)."""
    return sorted(p for p in root.rglob("*.json") if "metadata" in p.name)


def load_fakeavceleb_metadata(meta_csv: Path) -> pd.DataFrame:
    """``meta_data.csv`` de FakeAVCeleb v1.2 indexado por ``video_id``.

    La décima columna (sin nombre) es la carpeta ``FakeAVCeleb/<tipo>/<etnia>/<género>/<id>``.
    """
    md = pd.read_csv(meta_csv)
    md = md.rename(columns={md.columns[9]: "dir", "path": "filename"})
    sub = md["dir"].str.split("/", n=1).str[1]
    md["video_id"] = "fakeavceleb/" + sub + "/" + md["filename"]
    md = md[md["type"].isin(FAVC_TYPE_TO_LABEL)]
    return md[["video_id", "method", "category", "type", "race", "gender"]].drop_duplicates("video_id")


# ─────────────────────────────────────────────────────────────────────────────
# Localización en disco
# ─────────────────────────────────────────────────────────────────────────────

#: Alternativas de layout aceptadas además del canónico (relativas a la raíz).
#: Corresponden a la organización usada en el desarrollo (dataset_unzip).
_ALT_LAYOUTS = {
    "deeperforensics": [("manipulated_videos/", "FAKE/")],
    "fakeavceleb": [("FakeVideo-RealAudio/", "FAKE/"), ("RealVideo-RealAudio/", "REAL/")],
}


def candidate_paths(video_id: str) -> list[Path]:
    """Rutas posibles del video original, de la canónica a las alternativas."""
    prefix, rel = video_id.split("/", 1)
    root = config.dataset_root(prefix)
    out = [root / PurePosixPath(rel)]
    if prefix == "dfdc":
        out.append(root / "videos" / rel)
    if prefix == "ffpp" and rel.startswith("original_sequences/"):
        # Los originales también pueden estar donde los dejó DeeperForensics.
        out.append(config.dataset_root("deeperforensics") / "REAL" / PurePosixPath(rel))
    for canonical, alt in _ALT_LAYOUTS.get(prefix, []):
        if rel.startswith(canonical):
            out.append(root / PurePosixPath(alt + rel[len(canonical) :]))
    return out


def resolve_video(video_id: str, dfdc_index: dict[str, Path] | None = None) -> Path:
    """Ruta del video original. Para DFDC acepta un índice por nombre de fichero."""
    for p in candidate_paths(video_id):
        if p.is_file():
            return p
    if dfdc_index is not None and prefix_of(video_id) == "dfdc":
        name = video_id.split("/", 1)[1]
        if name in dfdc_index:
            return dfdc_index[name]
    raise FileNotFoundError(f"no se encontró {video_id} (buscado en {[str(p) for p in candidate_paths(video_id)]})")


def index_dfdc_videos(root: Path) -> dict[str, Path]:
    """Índice ``nombre -> ruta`` de todos los mp4 de DFDC (layout oficial por partes o aplanado)."""
    return {p.name: p for p in root.rglob("*.mp4")}
