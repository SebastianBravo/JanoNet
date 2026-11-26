"""Predicciones de referencia del estudio y tablas del documento.

- ``reference/predictions.csv.gz`` (en git) y ``reference/local/fakeavceleb_predictions.csv.gz``
  (fuera de git): puntaje oficial por ``sample_id`` de las 13 configuraciones del
  documento en validación y prueba (semilla 20260910).
- ``reference/document/``: los CSV de resultados que generaron las tablas y figuras
  del documento v12 (agregados, sin datos por video).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from janonet import paths
from janonet.data import manifest as M


def reference_dir() -> Path:
    return paths.project_root() / "reference"


def load_predictions(include_local: bool = True) -> pd.DataFrame:
    """``config, split, sample_id, score`` (parte pública + local si existe)."""
    df = pd.read_csv(reference_dir() / "predictions.csv.gz", float_precision="round_trip")
    local = reference_dir() / "local" / "fakeavceleb_predictions.csv.gz"
    if include_local and local.exists():
        df = pd.concat([df, pd.read_csv(local, float_precision="round_trip")], ignore_index=True)
    return df


def samples_with_study_method() -> pd.DataFrame:
    """Tabla maestra con ``method_study`` (método de FakeAVCeleb con la regla del documento)."""
    s = M.read_samples()
    f = M.local_dir() / "fakeavceleb_method_study.csv.gz"
    if f.exists():
        s = s.merge(pd.read_csv(f), on="video_id", how="left")
    return s


def complete() -> bool:
    """¿Están las filas de FakeAVCeleb (manifiesto y predicciones)?"""
    return (
        M.has_fakeavceleb(M.read_samples(), "samples")
        and (reference_dir() / "local" / "fakeavceleb_predictions.csv.gz").exists()
    )


def document_table(name: str) -> pd.DataFrame:
    p = reference_dir() / "document" / name
    if name == "r4_ci.csv":
        return pd.read_csv(p, header=[0, 1], index_col=[0, 1, 2, 3], float_precision="round_trip")
    return pd.read_csv(p, float_precision="round_trip")


def document_thresholds() -> dict[str, dict[str, float]]:
    return json.loads((reference_dir() / "document" / "r2_umbrales.json").read_text(encoding="utf-8"))


#: Claves de cada tabla del documento y columnas que el CSV original redondeó a float32.
TABLE_KEYS = {
    "r2_global.csv": (["cfg"], ["thA", "thB"]),
    "r2_escalera.csv": (["cfg", "tipo", "objetivo"], []),
    "r2_prevalencia.csv": (["cfg", "pi"], []),
    "r3_boot.csv": (["boot", "a", "b", "metrica"], []),
    "r3_subsets.csv": (["sub", "cfg"], []),
    "r3_macro.csv": (["cfg"], []),
    "r3_demografia.csv": (["cfg", "var", "grupo"], []),
    "r3_fps.csv": (["fps", "cfg"], []),
    "r3_errores.csv": (["a", "b"], []),
    "r4_point.csv": (["pi", "a", "b"], []),
    "r5_segmentos.csv": (["tipo", "seg", "cfg"], []),
}


def compare_table(new: pd.DataFrame, name: str) -> dict[str, float]:
    """Máxima diferencia absoluta por columna numérica frente a la tabla del documento.

    Devuelve también ``_missing`` (filas del documento ausentes en ``new``). Las columnas
    que el CSV original guardó como float32 se comparan tras redondear a float32.
    """
    keys, f32 = TABLE_KEYS[name]
    ref = document_table(name)
    new = new.reset_index() if keys[0] not in new.columns else new
    for k in keys:
        ref[k] = ref[k].astype(str).replace("nan", "")
        new = new.assign(**{k: new[k].astype(str).replace("nan", "")})
    m = ref.merge(new, on=keys, how="left", suffixes=("_ref", ""), indicator=True)
    out: dict[str, float] = {"_missing": float((m["_merge"] != "both").sum()), "_rows": float(len(ref))}
    for c in ref.columns:
        if c in keys or not pd.api.types.is_numeric_dtype(ref[c]):
            continue
        a, b = m[c + "_ref"].to_numpy(float), m[c].to_numpy(float)
        if c in f32:
            a, b = a.astype("float32").astype(float), b.astype("float32").astype(float)
        both_nan = pd.isna(a) & pd.isna(b)
        diff = abs(a - b)
        diff[both_nan] = 0.0
        out[c] = float(diff.max()) if len(diff) else 0.0
    return out


def compare_r4_ci(q: pd.DataFrame) -> float:
    ref = document_table("r4_ci.csv")
    q = q.copy()
    q.index = pd.MultiIndex.from_tuples([(b, str(float(p)), a, bb) for b, p, a, bb in q.index])
    ref.index = pd.MultiIndex.from_tuples([(b, str(float(p)), a, bb) for b, p, a, bb in ref.index])
    q.columns = pd.MultiIndex.from_tuples([(m, str(float(t))) for m, t in q.columns])
    ref.columns = pd.MultiIndex.from_tuples([(m, str(float(t))) for m, t in ref.columns])
    return float((q.loc[ref.index, ref.columns] - ref).abs().max().max())
