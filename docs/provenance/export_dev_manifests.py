"""Exporta los manifiestos de ``manifests/`` desde los artefactos del desarrollo.

Herramienta de procedencia, no parte del pipeline: documenta cómo se generaron
``manifests/preprocessing.csv.gz`` y ``manifests/samples.csv.gz`` a partir de lo que
realmente corrió, y verifica que el código de ``janonet`` reconstruye exactamente lo
mismo (orden de filas, ``group_id``, partición y ``sample_id``).

Entradas (rutas por argumento):

- ``--dev-repo``: raíz de ``dataset_analysis`` (usa
  ``data/with_labels/manifest_all_datasets_with_labels.csv``,
  ``analysis_outputs/identity_split/identity_split_dfdc_part.csv``,
  ``data/training/samples.parquet`` y ``metadata/*.json`` de DFDC).
- ``--favc-meta``: ``meta_data.csv`` de FakeAVCeleb v1.2 (para el método por video).

Uso::

    python docs/provenance/export_dev_manifests.py --dev-repo <ruta> --favc-meta <ruta>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path, PureWindowsPath

import pandas as pd

from janonet.data import manifest as M
from janonet.data import sources

_PROCESSED_ROOTS = {
    "dfdc_train_all_processed_test": "dfdc",
    "face_forensics_processed": "ffpp_manip",
    "deeper_forensics_processed": "df",
    "FakeAVCeleb_processed": "favc",
}


def video_id_from_dev_output(output_video: str) -> str:
    parts = PureWindowsPath(output_video).parts
    i = next(i for i, p in enumerate(parts) if p in _PROCESSED_ROOTS)
    kind, rest = _PROCESSED_ROOTS[parts[i]], list(parts[i + 1 :])
    if kind == "dfdc":
        return "dfdc/" + "/".join(rest)
    if kind == "ffpp_manip":
        return "ffpp/manipulated_sequences/" + "/".join(rest)
    if kind == "df":
        if rest[0] == "REAL":
            return "ffpp/" + "/".join(rest[1:])
        return "deeperforensics/manipulated_videos/" + "/".join(rest[1:])
    top = {"FAKE": "FakeVideo-RealAudio", "REAL": "RealVideo-RealAudio"}[rest[0]]
    return "fakeavceleb/" + "/".join([top, *rest[1:]])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dev-repo", type=Path, required=True)
    ap.add_argument("--favc-meta", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=M.manifests_dir())
    args = ap.parse_args()
    dev = args.dev_repo

    man = pd.read_csv(dev / "data/with_labels/manifest_all_datasets_with_labels.csv", low_memory=False)
    man.insert(0, "manifest_order", range(len(man)))
    man["video_id"] = man["output_video"].map(video_id_from_dev_output)
    assert man["video_id"].is_unique

    # Etiqueta y dataset: los del manifiesto; deben coincidir con los deducidos del video_id.
    assert (man["video_id"].map(sources.dataset_of) == man["dataset"]).all(), "dataset no coincide"
    from_path = man["video_id"].map(sources.label_from_path)
    assert ((from_path == man["label"]) | from_path.isna()).all(), "etiqueta no coincide"

    # DFDC: parte y original desde su metadata.
    meta = sources.load_dfdc_metadata(sorted((dev / "metadata").glob("*metadata.json")))
    fn = man["video_id"].str.split("/", n=1).str[1]
    is_dfdc = man["dataset"] == "dfdc"
    m = meta.set_index("filename")
    man["dfdc_part"] = pd.Series(pd.NA, index=man.index, dtype="string")
    man["dfdc_original"] = pd.Series(pd.NA, index=man.index, dtype="string")
    man.loc[is_dfdc, "dfdc_part"] = fn[is_dfdc].map(m["part"]).values
    man.loc[is_dfdc, "dfdc_original"] = fn[is_dfdc].map(m["original"]).values
    assert (fn[is_dfdc].map(m["label"]) == man.loc[is_dfdc, "label"]).all(), "etiqueta DFDC no coincide"

    # Método de manipulación.
    man["method"] = man["video_id"].map(sources.method_of)
    favc = sources.load_fakeavceleb_metadata(args.favc_meta).set_index("video_id")
    is_favc_fake = (man["dataset"] == "FakeAVCeleb") & (man["label"] == "FAKE")
    man.loc[is_favc_fake, "method"] = man.loc[is_favc_fake, "video_id"].map(favc["method"]).fillna("unknown").values

    # Verificación: el orden heredado reproduce manifest_order.
    order = sorted(range(len(man)), key=lambda i: sources.legacy_manifest_order_key(man["video_id"].iat[i]))
    assert order == list(range(len(man))), "legacy_manifest_order_key no reproduce el orden del manifiesto"

    pre = man[M.PREPROCESSING_COLUMNS].copy()
    pre["meets_the_clean_rules"] = pre["meets_the_clean_rules"].astype(bool)

    # Tabla maestra congelada del estudio.
    s = pd.read_parquet(dev / "data/training/samples.parquet")
    s["video_id"] = s["output_video"].map(video_id_from_dev_output)
    s = s.merge(man[["video_id", "method"]], on="video_id", how="left", validate="one_to_one")
    s["resize_bucket"] = s["resize_bucket"].astype("string")
    s["duration_bucket"] = s["duration_bucket"].astype("string")
    samples = s[M.SAMPLES_COLUMNS].sort_values("sample_id", ignore_index=True)

    # Verificación: janonet reconstruye partición y tabla maestra desde preprocessing.
    rebuilt = M.build_samples(pre, M.build_split(pre))
    for col in M.SAMPLES_COLUMNS:
        a, b = rebuilt[col], samples[col]
        if a.dtype.kind == "f":
            same = (a.to_numpy() == b.to_numpy()).all()
        else:
            same = (a.astype("string").fillna("<NA>") == b.astype("string").fillna("<NA>")).all()
        print(f"  {col:<28} {'idéntica' if same else 'DIFIERE'}")
        assert same, f"la reconstrucción difiere en {col}"

    # FakeAVCeleb va a manifests/local/ (fuera de git): sus términos de uso prohíben
    # distribuir datos derivados del dataset.
    for p in M.write_split_manifests(pre, samples, args.out):
        print("->", p)
    print(f"{len(pre):,} y {len(samples):,} videos en total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
