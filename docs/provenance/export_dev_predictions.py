"""Exporta las predicciones por video que usó el documento como referencia de paridad.

Herramienta de procedencia. Lee ``export_resultados/predicciones_por_video.parquet``
del desarrollo (la tabla ``pv.parquet`` de los scripts del documento), se queda con la
semilla 20260910, las 13 configuraciones del documento y los splits de validación y
prueba, y escribe el puntaje oficial (``score_oficial``) por ``sample_id``:

- ``reference/predictions.csv.gz`` (en git): DFDC, FaceForensics++ y DeeperForensics;
- ``reference/local/fakeavceleb_predictions.csv.gz`` (fuera de git);
- ``manifests/local/fakeavceleb_method_study.csv.gz``: el método de FakeAVCeleb con la
  regla del documento (ver ``sources.fakeavceleb_method_study``), verificado contra
  la columna ``favc_metodo`` de la tabla.

Uso::

    python docs/provenance/export_dev_predictions.py --dev-repo <ruta> --favc-meta <meta_data.csv>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from janonet import paths
from janonet.data import manifest as M
from janonet.data import sources
from janonet.eval.study import STUDY_CONFIGS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dev-repo", type=Path, required=True)
    ap.add_argument("--favc-meta", type=Path, required=True)
    args = ap.parse_args()

    cols = ["config", "seed", "split", "video_id", "label", "dataset", "score_oficial", "score_fuente", "favc_metodo"]
    pv = pd.read_parquet(args.dev_repo / "export_resultados/predicciones_por_video.parquet", columns=cols)
    pv = pv[(pv["seed"] == 20260910) & pv["config"].isin(STUDY_CONFIGS) & pv["split"].isin(["val", "test"])]
    samples = M.read_samples()
    s = samples.set_index("sample_id")
    # En la tabla del desarrollo, video_id es el sample_id.
    assert (s.loc[pv["video_id"], "y"].to_numpy() == pv["label"].to_numpy()).all(), "etiquetas no coinciden"
    assert (s.loc[pv["video_id"], "split"].to_numpy() == pv["split"].to_numpy()).all(), "particiones no coinciden"
    print(pv.groupby("config")["score_fuente"].unique().to_string())

    pred = pd.DataFrame(
        {
            "config": pv["config"],
            "split": pv["split"],
            "sample_id": pv["video_id"].astype(int),
            "score": pv["score_oficial"].astype("float64"),
        }
    )
    pred = pred.sort_values(["config", "split", "sample_id"], ignore_index=True)
    local = pv["dataset"].to_numpy() == M.LOCAL_ONLY_DATASET
    ref = paths.project_root() / "reference"
    print("->", M.write_csv(pred[~local], ref / "predictions.csv.gz"))
    print("->", M.write_csv(pred[local], ref / "local" / "fakeavceleb_predictions.csv.gz"))

    # Método de FakeAVCeleb con la regla del documento, para todos sus videos limpios.
    favc = samples[samples["dataset"] == M.LOCAL_ONLY_DATASET]
    ms = sources.fakeavceleb_method_study(args.favc_meta, favc["video_id"])
    chk = pv[local & (pv["label"] == 1).to_numpy()].drop_duplicates("video_id")
    vid = s.loc[chk["video_id"], "video_id"].to_numpy()
    assert (ms.loc[vid].to_numpy() == chk["favc_metodo"].to_numpy()).all(), "la regla del documento no se reproduce"
    out = pd.DataFrame({"video_id": ms.index, "method_study": ms.to_numpy()})
    print("->", M.write_csv(out, M.local_dir() / "fakeavceleb_method_study.csv.gz"))
    diff = favc.set_index("video_id")["method"].reindex(out["video_id"]).to_numpy() != out["method_study"].to_numpy()
    fakes = favc.set_index("video_id")["label"].reindex(out["video_id"]).to_numpy() == "FAKE"
    print(f"videos de FakeAVCeleb con método distinto al del documento: {int((diff & fakes).sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
