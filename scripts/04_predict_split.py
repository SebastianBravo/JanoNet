"""Predicciones por video de un split desde la caché de 32 cuadros.

    python scripts/04_predict_split.py --models u100 rs_ca rs_ca_tr --split test \\
        [--limit 200 --seed 20261126] [--sample-ids ids.txt] [--device cpu] \\
        [--checkpoint u100=ruta/al/checkpoint] --out outputs/predictions_test.csv

Sin ``--checkpoint``, cada modelo se busca en ``weights/<nombre>.pt``. Escribe
``config, split, sample_id, score, video_logit`` (``config`` = nombre del modelo en el
documento) y, si se pide el ensamble (``--models ... ensemble``), también sus filas.
La salida sirve como entrada de ``05_evaluate.py --predictions``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from janonet import config, paths
from janonet.data import manifest as M
from janonet.inference.clips import predict_cached
from janonet.models.ensemble import ensemble_score
from janonet.models.loading import load_config, load_model


def main() -> int:
    ap = argparse.ArgumentParser(description="Predicciones por video desde la caché")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--checkpoint", action="append", default=[], help="nombre=ruta (repetible)")
    ap.add_argument("--sample-ids", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20261126)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("outputs/predictions.csv"))
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    samples = M.read_samples()
    ids = np.sort(samples.loc[samples["split"] == args.split, "sample_id"].to_numpy())
    if args.sample_ids:
        ids = np.array(sorted(int(x) for x in args.sample_ids.read_text().split()))
    elif args.limit:
        ids = np.sort(np.random.default_rng(args.seed).choice(ids, args.limit, replace=False))
    ckpts = dict(c.split("=", 1) for c in args.checkpoint)
    cache_dir = config.output_dir("cache")

    rows = []
    member_scores = {}
    for name in [m for m in args.models if m != "ensemble"]:
        cfg = load_config(name)
        ckpt = Path(ckpts.get(name, paths.weights_dir() / f"{name}.pt"))
        model = load_model(name, ckpt, device=args.device)
        t0 = time.perf_counter()
        res = predict_cached(model, ids, cache_dir, args.batch_size, args.device)
        dt = time.perf_counter() - t0
        print(f"{name}: {len(ids)} videos en {dt:.0f} s ({dt / len(ids):.2f} s/video, {args.device})")
        member_scores[name] = res["score"]
        rows.append(
            pd.DataFrame(
                {
                    "config": cfg["meta"]["study_config"],
                    "split": args.split,
                    "sample_id": ids,
                    "score": res["score"],
                    "video_logit": res["video_logit"],
                }
            )
        )
        del model
    if "ensemble" in args.models:
        ens = load_config("ensemble")["ensemble"]
        scores = ensemble_score([member_scores[m] for m in ens["members"]], ens["eps"])
        rows.append(
            pd.DataFrame(
                {"config": "ENS", "split": args.split, "sample_id": ids, "score": scores, "video_logit": np.nan}
            )
        )
    out = pd.concat(rows, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"{len(out):,} filas -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
