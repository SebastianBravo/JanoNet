"""Grafo de identidades (union-find) y verificación de fuga entre particiones.

Unidad de identidad por dataset:

- **DFDC**: la parte del dataset (``dfdc_train_part_NN``), granularidad ``part`` (la
  que entrenó). La alternativa ``original`` agrupa por el video original del metadata.
- **FakeAVCeleb**: el ``idNNNNN`` de la ruta (REAL y FAKE del mismo id van juntos).
- **FF++ (5 métodos) y DeeperForensics**: el id de 3 dígitos del video objetivo; un
  video manipulado ``TTT_SSS`` une los nodos ``TTT`` y ``SSS`` (contiene dos caras).
- **FF++/DeepFakeDetection**: componente conexa de actores ``NN_MM__...``.

Los actores donantes de DeeperForensics (``M101``, ``W101``, ...) **no** entran al
grafo: unirlos colapsaría FF++ y DeeperForensics en 9 grupos (uno con el 78,8 % de los
videos). Es una limitación documentada y medida por ``leakage_report``.

Los ``group_id`` son los nombres de nodo raíz y dependen del orden de las uniones: para
reproducir los del estudio hay que recorrer las filas en el orden del manifiesto.

Origen: ``dataset_analysis/code/notebook_analisis_identidades.ipynb``, celdas 8 y 14.
"""

from __future__ import annotations

import re

import pandas as pd


class UnionFind:
    """Union-find con compresión de camino. La raíz de una unión es la de ``b``."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _stem(video_id: str) -> str:
    name = video_id.rsplit("/", 1)[-1]
    return name[:-4] if name.lower().endswith(".mp4") else name


def build_identity_graph(
    frame: pd.DataFrame, dfdc_meta: pd.DataFrame | None, dfdc_granularity: str = "part"
) -> pd.DataFrame:
    """Añade ``identity_node``, ``identity_kind``, ``df_donor_actor`` y ``group_id``.

    ``frame`` necesita ``video_id``, ``dataset`` y ``label``; ``dfdc_meta`` (de
    ``sources.load_dfdc_metadata``) es obligatorio si hay videos de DFDC.
    """
    if dfdc_granularity not in ("part", "original"):
        raise ValueError(f"dfdc_granularity debe ser 'part' u 'original', llegó {dfdc_granularity!r}")
    meta = dfdc_meta.set_index("filename") if dfdc_meta is not None else None
    uf = UnionFind()
    nodes, donors, kinds = [], [], []

    for video_id, dataset, label in zip(frame["video_id"], frame["dataset"], frame["label"], strict=True):
        parts = video_id.split("/")
        stem = _stem(video_id)
        donor = None
        if dataset == "dfdc":
            info = meta.loc[parts[-1]]
            if dfdc_granularity == "part":
                node, kind = f"DFDC::part{info['part']}", "dfdc_part"
            else:
                base = info["original"] if info["label"] == "FAKE" and isinstance(info["original"], str) else parts[-1]
                node, kind = f"DFDC::{base}", "dfdc_original"
        elif dataset == "FakeAVCeleb":
            pid = next(seg for seg in parts if re.fullmatch(r"id\d+", seg))
            node, kind = f"FAVC::{pid}", "favc_id"
        elif dataset == "deeper-forensics":
            if label == "REAL":
                node = f"FFID::{stem}"
            else:
                target, donor = stem.split("_", 1)
                node = f"FFID::{target}"
            kind = "ffpp_video_id"
        elif dataset == "FaceForensics++":
            method = parts[2]
            if method == "DeepFakeDetection":
                target, source = stem.split("__")[0].split("_")
                node = f"DFDACT::{target}"
                uf.union(node, f"DFDACT::{source}")
                kind = "ffpp_dfd_actor"
            else:
                target, source = stem.split("_")
                node = f"FFID::{target}"
                uf.union(node, f"FFID::{source}")
                kind = "ffpp_video_id"
        else:
            raise ValueError(f"dataset no contemplado: {dataset}")
        uf.find(node)
        nodes.append(node)
        donors.append(donor)
        kinds.append(kind)

    out = frame.copy()
    out["identity_node"] = nodes
    out["identity_kind"] = kinds
    out["df_donor_actor"] = donors
    # Las raíces se resuelven al final, con todas las aristas aplicadas.
    out["group_id"] = [uf.find(n) for n in nodes]
    return out


def identity_occurrences(frame: pd.DataFrame, dfdc_meta: pd.DataFrame | None) -> pd.DataFrame:
    """Todas las identidades visibles por video (incluidas las que el grafo ignora)."""
    meta = dfdc_meta.set_index("filename") if dfdc_meta is not None else None
    rows = []
    for idx, video_id, dataset, label in zip(
        frame.index, frame["video_id"], frame["dataset"], frame["label"], strict=True
    ):
        parts = video_id.split("/")
        stem = _stem(video_id)
        if dataset == "dfdc":
            info = meta.loc[parts[-1]]
            base = info["original"] if info["label"] == "FAKE" and isinstance(info["original"], str) else parts[-1]
            rows += [(idx, "dfdc_original", base), (idx, "dfdc_part", info["part"])]
        elif dataset == "FakeAVCeleb":
            rows.append((idx, "favc_id", next(seg for seg in parts if re.fullmatch(r"id\d+", seg))))
        elif dataset == "deeper-forensics":
            if label == "REAL":
                rows.append((idx, "ffpp_video_id", stem))
            else:
                target, actor = stem.split("_", 1)
                rows += [(idx, "ffpp_video_id", target), (idx, "df_donor_actor", actor)]
        elif dataset == "FaceForensics++":
            if parts[2] == "DeepFakeDetection":
                target, source = stem.split("__")[0].split("_")
                rows += [(idx, "ffpp_dfd_actor", target), (idx, "ffpp_dfd_actor", source)]
            else:
                target, source = stem.split("_")
                rows += [(idx, "ffpp_video_id", target), (idx, "ffpp_video_id", source)]
    return pd.DataFrame(rows, columns=["row", "id_type", "id_value"])


def leakage_report(frame: pd.DataFrame, dfdc_meta: pd.DataFrame | None) -> pd.DataFrame:
    """Por tipo de identidad: cuántas hay y cuántas cruzan particiones."""
    occ = identity_occurrences(frame, dfdc_meta)
    occ["split"] = occ["row"].map(frame["split"])
    n = occ.groupby(["id_type", "id_value"])["split"].nunique()
    rep = n.groupby("id_type").agg(identities="size", cross_splits=lambda s: int((s > 1).sum()))
    rep["pct_leak"] = (100 * rep["cross_splits"] / rep["identities"]).round(2)
    return rep
