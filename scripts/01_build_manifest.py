"""Inventario de fuentes y construcción de la partición por identidad.

Dos etapas (el preprocesamiento facial va entre ambas, ver DATA.md):

    # 1) inventario: video_id, dataset, etiqueta y método de cada video de las fuentes
    python scripts/01_build_manifest.py sources [--datasets dfdc ffpp deeperforensics fakeavceleb]
                                                [--out outputs/sources.csv]

    # 3) partición: grafo de identidades + split + tabla maestra, desde el resultado
    #    del preprocesamiento (por defecto, los manifiestos de manifests/, parte pública
    #    + parte local de FakeAVCeleb)
    python scripts/01_build_manifest.py split [--preprocessing <csv>] [--out-dir outputs/manifests]
                                              [--check] [--write-local]

``--check`` compara la tabla maestra resultante con la de ``manifests/`` columna a
columna, en las filas que existan en ambas. ``--write-local`` (tras un ``--check``
correcto) guarda las filas de FakeAVCeleb de la tabla maestra en ``manifests/local/``.

Regenerar las filas de FakeAVCeleb (fuera de git por sus términos de uso):

    python scripts/01_build_manifest.py sources --datasets fakeavceleb --out outputs/fakeavceleb_sources.csv
    python scripts/02_preprocess.py faces --videos outputs/fakeavceleb_sources.csv \\
        --out manifests/local/fakeavceleb_preprocessing.csv
    python scripts/01_build_manifest.py split --check --write-local
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from janonet import config
from janonet.data import manifest as M
from janonet.data import sources
from janonet.data.identity import leakage_report


def _scan(prefixes: list[str]) -> list[str]:
    cfg = config.load_yaml("data")["subsets"]
    ids: list[str] = []
    if "dfdc" in prefixes:
        ids += [f"dfdc/{p.name}" for p in sorted(sources.index_dfdc_videos(config.dataset_root("dfdc")).values())]
    if "ffpp" in prefixes:
        ffpp, comp = config.dataset_root("ffpp"), cfg["ffpp_compression"]
        for method in cfg["ffpp_methods"]:
            for p in sorted((ffpp / "manipulated_sequences" / method / comp / "videos").glob("*.mp4")):
                ids.append(f"ffpp/manipulated_sequences/{method}/{comp}/videos/{p.name}")
        for p in sorted((ffpp / "original_sequences" / "youtube" / comp / "videos").glob("*.mp4")):
            ids.append(f"ffpp/original_sequences/youtube/{comp}/videos/{p.name}")
    if "deeperforensics" in prefixes:
        df_root = config.dataset_root("deeperforensics")
        for variant in cfg["deeperforensics_variants"]:
            for p in sorted(q for q in df_root.rglob("*.mp4") if q.parent.name == variant):
                ids.append(f"deeperforensics/manipulated_videos/{variant}/{p.name}")
    if "fakeavceleb" in prefixes:
        favc_root = config.dataset_root("fakeavceleb")
        for vtype, label in cfg["fakeavceleb_types"].items():
            # Layout de la descarga v1.2 (<tipo>/...) o el reorganizado en FAKE/ y REAL/.
            base = favc_root / vtype if (favc_root / vtype).is_dir() else favc_root / label
            ids += [f"fakeavceleb/{vtype}/" + p.relative_to(base).as_posix() for p in sorted(base.rglob("*.mp4"))]
    return ids


def stage_sources(prefixes: list[str], out: Path) -> int:
    inv = pd.DataFrame({"video_id": _scan(prefixes)}).drop_duplicates("video_id")
    inv["dataset"] = inv["video_id"].map(sources.dataset_of)
    inv["label"] = inv["video_id"].map(sources.label_from_path)
    inv["method"] = inv["video_id"].map(sources.method_of)
    inv["dfdc_part"] = pd.Series(pd.NA, index=inv.index, dtype="string")
    inv["dfdc_original"] = pd.Series(pd.NA, index=inv.index, dtype="string")
    if "dfdc" in prefixes:
        m = sources.load_dfdc_metadata(sources.find_dfdc_metadata(config.dataset_root("dfdc"))).set_index("filename")
        is_dfdc = inv["dataset"] == "dfdc"
        names = inv.loc[is_dfdc, "video_id"].str.split("/", n=1).str[1]
        inv.loc[is_dfdc, "label"] = names.map(m["label"]).values
        inv.loc[is_dfdc, "dfdc_part"] = names.map(m["part"]).values
        inv.loc[is_dfdc, "dfdc_original"] = names.map(m["original"]).values
    if "fakeavceleb" in prefixes:
        favc_meta = next(iter(sorted(config.dataset_root("fakeavceleb").rglob("meta_data.csv"))), None)
        if favc_meta is None:
            print("aviso: sin meta_data.csv de FakeAVCeleb; el método de sus falsos queda como 'unknown'")
        else:
            fm = sources.load_fakeavceleb_metadata(favc_meta).set_index("video_id")
            sel = (inv["dataset"] == "FakeAVCeleb") & (inv["label"] == "FAKE")
            inv.loc[sel, "method"] = inv.loc[sel, "video_id"].map(fm["method"]).fillna("unknown").values

    # Orden de filas del estudio (de él dependen los nombres de los group_id). Con un
    # inventario parcial, la posición se calcula junto a las filas ya publicadas.
    known = M.read_preprocessing(include_local=False)["video_id"]
    universe = pd.Index(pd.concat([known, inv["video_id"]]).unique())
    keys = universe.map(sources.legacy_manifest_order_key).tolist()
    order = pd.Series(np.arange(len(universe)), index=universe[sorted(range(len(universe)), key=keys.__getitem__)])
    inv["manifest_order"] = inv["video_id"].map(order).to_numpy()
    inv = inv.sort_values("manifest_order", ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    inv.to_csv(out, index=False)
    print(inv.groupby(["dataset", "label"]).size().to_string())
    print(f"{len(inv):,} videos -> {out}")
    return 0


def _differences(new: pd.DataFrame, ref: pd.DataFrame) -> list[str]:
    both = new.merge(ref[["sample_id", "video_id"]], on="video_id", how="inner", suffixes=("", "_ref"))
    ref = ref.set_index("video_id").loc[both["video_id"]].reset_index()
    new = new.set_index("video_id").loc[both["video_id"]].reset_index()
    diff = []
    for c in M.SAMPLES_COLUMNS:
        a, b = new[c], ref[c]
        same = (
            np.array_equal(a.to_numpy(float), b.to_numpy(float), equal_nan=True)
            if a.dtype.kind == "f"
            else (a.astype("string").fillna("<NA>") == b.astype("string").fillna("<NA>")).all()
        )
        if not same:
            diff.append(c)
    print(f"comparadas {len(both):,} filas con manifests/")
    return diff


def stage_split(preprocessing: Path | None, out_dir: Path, check: bool, write_local: bool) -> int:
    pre = M.read_preprocessing(preprocessing)
    split_frame = M.build_split(pre)
    samples = M.build_samples(pre, split_frame)
    print(samples.groupby(["split", "label"]).size().unstack().to_string())
    print(leakage_report(split_frame, M.dfdc_meta_from_preprocessing(pre)).to_string())
    if check or write_local:
        diff = _differences(samples, M.read_samples())
        print("idéntica a manifests/" if not diff else f"DIFIERE en {diff}")
        if diff:
            return 1
    for p in M.write_split_manifests(None, samples, out_dir):
        print("->", p)
    if write_local:
        local = samples[samples["dataset"] == M.LOCAL_ONLY_DATASET]
        print("->", M.write_csv(local, M.local_dir() / (M.LOCAL_PREFIX + M.SAMPLES_FILE)))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Inventario de fuentes y partición por identidad")
    sub = ap.add_subparsers(dest="stage", required=True)
    s1 = sub.add_parser("sources")
    s1.add_argument("--datasets", nargs="+", default=list(sources.PREFIXES), choices=sources.PREFIXES)
    s1.add_argument("--out", type=Path, default=Path("outputs/sources.csv"))
    s2 = sub.add_parser("split")
    s2.add_argument("--preprocessing", type=Path, default=None)
    s2.add_argument("--out-dir", type=Path, default=Path("outputs/manifests"))
    s2.add_argument("--check", action="store_true")
    s2.add_argument("--write-local", action="store_true")
    args = ap.parse_args()
    if args.stage == "sources":
        return stage_sources(args.datasets, args.out)
    return stage_split(args.preprocessing, args.out_dir, args.check, args.write_local)


if __name__ == "__main__":
    sys.exit(main())
