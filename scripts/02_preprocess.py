"""Preprocesamiento facial y caché de 32 cuadros.

    # 2) rostros: RetinaFace, limpieza, recorte y resize -> videos de 224x224 (mp4v)
    #    y una fila de resultado por video (reanudable: salta los ya hechos)
    python scripts/02_preprocess.py faces --videos outputs/sources.csv --out outputs/preprocessing.csv

    # 4) caché: 32 cuadros centrados de cada video limpio -> <cache>/<sid//1000>/<sid>.bin
    python scripts/02_preprocess.py cache [--samples manifests/samples.csv.gz] [--workers 4]

Rutas: los videos originales se buscan en las raíces de ``configs/data.yaml``; los
procesados van a ``$JANONET_PROCESSED/<video_id>`` y la caché a ``$JANONET_CACHE``.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath

import pandas as pd

from janonet import config
from janonet.data import cache, sources
from janonet.data import manifest as M


def processed_path(video_id: str) -> Path:
    return config.output_dir("processed") / PurePosixPath(video_id)


def stage_faces(videos: Path, out: Path, limit: int | None) -> int:
    from janonet.preprocess.detect import FaceDetector
    from janonet.preprocess.video import load_preprocess_config, preprocess_video

    cfg = load_preprocess_config()
    todo = pd.read_csv(videos, dtype={"dfdc_part": "string", "dfdc_original": "string"})
    done = set(pd.read_csv(out, usecols=["video_id"])["video_id"]) if out.exists() else set()
    todo = todo[~todo["video_id"].isin(done)]
    if limit:
        todo = todo.head(limit)
    dfdc_index = None
    detector = FaceDetector(cfg["detector"]["score_threshold"])
    cols = [c for c in M.PREPROCESSING_COLUMNS]
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        if new_file:
            w.writeheader()
        for i, row in enumerate(todo.itertuples(index=False), 1):
            t0 = time.perf_counter()
            try:
                src = sources.resolve_video(row.video_id, dfdc_index)
            except FileNotFoundError:
                if dfdc_index is None and sources.prefix_of(row.video_id) == "dfdc":
                    dfdc_index = sources.index_dfdc_videos(config.dataset_root("dfdc"))
                    src = sources.resolve_video(row.video_id, dfdc_index)
                else:
                    raise
            res = preprocess_video(src, processed_path(row.video_id), detector=detector, cfg=cfg)
            w.writerow({**row._asdict(), **res.as_row()})
            fh.flush()
            print(
                f"[{i}/{len(todo)}] {row.video_id}  limpio={res.meets_the_clean_rules}  "
                f"{time.perf_counter() - t0:.1f}s"
            )
    return 0


def _cache_one(task: tuple[int, str, int, str]) -> tuple[int, str]:
    sample_id, video_id, window_start, cache_dir = task
    try:
        frames = cache.decode_window(processed_path(video_id), window_start)
        cache.write_bin(cache.bin_path(cache_dir, sample_id), frames, sample_id, window_start)
        return sample_id, "ok"
    except Exception as exc:  # noqa: BLE001 - se registra y se sigue
        return sample_id, f"{type(exc).__name__}: {exc}"


def stage_cache(samples_path: Path | None, workers: int, limit: int | None) -> int:
    samples = M.read_samples(samples_path)
    cache_dir = config.output_dir("cache")
    pending = [
        (int(r.sample_id), r.video_id, int(r.window_start), str(cache_dir))
        for r in samples.itertuples()
        if not cache.bin_path(cache_dir, r.sample_id).exists()
    ]
    if limit:
        pending = pending[:limit]
    print(f"{len(pending):,} videos pendientes -> {cache_dir}")
    fails = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (sid, status) in enumerate(ex.map(_cache_one, pending, chunksize=8), 1):
            if status != "ok":
                fails += 1
                print(f"  fallo {sid}: {status}")
            if i % 1000 == 0:
                print(f"  {i:,}/{len(pending):,}")
    print(f"listo: {len(pending) - fails:,} escritos, {fails} fallos")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Preprocesamiento facial y caché")
    sub = ap.add_subparsers(dest="stage", required=True)
    f = sub.add_parser("faces")
    f.add_argument("--videos", type=Path, default=Path("outputs/sources.csv"))
    f.add_argument("--out", type=Path, default=Path("outputs/preprocessing.csv"))
    f.add_argument("--limit", type=int, default=None)
    c = sub.add_parser("cache")
    c.add_argument("--samples", type=Path, default=None)
    c.add_argument("--workers", type=int, default=4)
    c.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.stage == "faces":
        return stage_faces(args.videos, args.out, args.limit)
    return stage_cache(args.samples, args.workers, args.limit)


if __name__ == "__main__":
    sys.exit(main())
