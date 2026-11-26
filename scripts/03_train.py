"""Entrena un modelo de JanoNet desde la caché de 32 cuadros.

    python scripts/03_train.py --model rs_ca_tr --run-dir runs/rs_ca_tr
    python scripts/03_train.py --model u100 --run-dir runs/u100_smoke --subset 64 --epochs 1 --device cpu

La receta (hiperparámetros, semilla, AMP, acumulación) sale de la sección ``train`` de
``configs/models/<modelo>.yaml``. ``--epochs``, ``--batch-size``/``--accum-steps``,
``--num-workers`` y ``--subset`` la sobrescriben solo para pruebas.

Reanudable: con ``--resume auto`` (por defecto) continúa desde ``last.ckpt`` si existe y
es de la misma receta. SIGINT/SIGTERM guardan y salen limpio.

Salidas en ``--run-dir``: ``config.yaml`` (config efectiva), ``run_manifest.json``,
``metrics_epoch.csv``, ``val_scores_epochNN.csv``, ``last/prev/best.ckpt`` y
``step_losses.csv``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from janonet import __version__, config
from janonet.data import cache
from janonet.data import manifest as M
from janonet.data.clip_dataset import ClipDataset, ResumableSampler, stratified_sample_ids, worker_init_fn
from janonet.models import build_model
from janonet.models.loading import load_config
from janonet.train.checkpoint import CheckpointManager, StopSignal, config_hash
from janonet.train.loop import TrainConfig, Trainer, evaluate, make_grad_scaler, pos_weight_from_counts


def main() -> int:
    ap = argparse.ArgumentParser(description="Entrenamiento de JanoNet")
    ap.add_argument("--model", required=True, help="u100 | rs_ca | rs_ca_tr | e6 | ruta a un YAML")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--accum-steps", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument(
        "--subset", type=int, default=None, help="N videos de train (y N/4 de val), estratificados; solo para pruebas"
    )
    ap.add_argument("--train-eval-subset", type=int, default=None)
    ap.add_argument("--resume", choices=["auto", "never"], default="auto")
    ap.add_argument(
        "--loss-check",
        action="store_true",
        help="medir la pérdida del train efectivo (modo eval) antes y después; para corridas cortas",
    )
    args = ap.parse_args()

    full_cfg = load_config(args.model)
    tcfg = TrainConfig.from_dict(full_cfg["train"])
    for k in ("epochs", "batch_size", "accum_steps", "num_workers", "train_eval_subset"):
        if getattr(args, k) is not None:
            setattr(tcfg, k, getattr(args, k))
    device = torch.device(args.device)
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Datos ──────────────────────────────────────────────────────────────
    samples = M.read_samples()
    if not M.has_fakeavceleb(samples, "samples"):
        print(
            "[train] AVISO: faltan las filas de FakeAVCeleb (manifests/local/): se entrena sin ellas, "
            "así que el modelo no es el del estudio."
        )
    train_df = samples[samples["split"] == "train"]
    val_df = samples[samples["split"] == "val"]
    if args.subset:
        train_df = train_df[train_df["sample_id"].isin(stratified_sample_ids(train_df, args.subset, tcfg.seed))]
        val_df = val_df[val_df["sample_id"].isin(stratified_sample_ids(val_df, max(args.subset // 4, 8), tcfg.seed))]
    cache_dir = config.output_dir("cache")
    missing = [s for s in pd.concat([train_df, val_df])["sample_id"] if not cache.bin_path(cache_dir, s).exists()]
    if missing:
        print(
            f"[train] faltan {len(missing):,} .bin en {cache_dir} (ej. {missing[:3]}): ejecuta "
            "02_preprocess.py cache",
            file=sys.stderr,
        )
        return 1

    ds_train = ClipDataset(train_df, cache_dir, "train", base_seed=tcfg.seed, temporal_stride=tcfg.temporal_stride)
    ds_val = ClipDataset(val_df, cache_dir, "eval", base_seed=tcfg.seed, temporal_stride=tcfg.temporal_stride)
    sampler = ResumableSampler(len(ds_train), base_seed=tcfg.seed)
    kw = tcfg.loader_kwargs()
    # Cada cargador con su propio generador: crear un iterador no consume así el RNG
    # global (el del dropout), y reanudar no desplaza las máscaras.
    kw["generator"] = torch.Generator().manual_seed(tcfg.seed)
    dl_train = DataLoader(
        ds_train, sampler=sampler, drop_last=True, worker_init_fn=worker_init_fn if tcfg.num_workers > 0 else None, **kw
    )
    dl_val = DataLoader(ds_val, shuffle=False, drop_last=False, **kw)
    dl_train_eval = None
    if tcfg.train_eval_subset:
        ev = train_df
        if tcfg.train_eval_subset < len(train_df):
            ev = train_df[
                train_df["sample_id"].isin(stratified_sample_ids(train_df, tcfg.train_eval_subset, tcfg.seed))
            ]
        dl_train_eval = DataLoader(ClipDataset(ev, cache_dir, "eval", base_seed=tcfg.seed), shuffle=False, **kw)

    # ── Modelo, pérdida y optimizador ──────────────────────────────────────
    torch.manual_seed(tcfg.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = tcfg.cudnn_benchmark and not tcfg.deterministic
        torch.backends.cudnn.deterministic = tcfg.deterministic
    model = build_model(full_cfg["model"]).to(device)
    w = pos_weight_from_counts(train_df["y"].to_numpy(), tcfg.pos_weight_alpha)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([w], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=tcfg.lr, weight_decay=tcfg.weight_decay
    )
    scaler = make_grad_scaler(tcfg.amp, device)

    h = config_hash(
        {
            "model": full_cfg["model"],
            "train": tcfg.experimental(),
            "train_ids": sorted(int(s) for s in train_df["sample_id"]),
        }
    )
    ckpt = CheckpointManager(run_dir, h)
    stop = StopSignal().install()
    trainer = Trainer(
        cfg=tcfg,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        train_loader=dl_train,
        val_loader=dl_val,
        dataset=ds_train,
        sampler=sampler,
        ckpt=ckpt,
        run_dir=run_dir,
        scaler=scaler,
        stop=stop,
        train_eval_loader=dl_train_eval,
    )
    resumed = trainer.resume() if args.resume == "auto" else False

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(
            {**full_cfg, "train": {**full_cfg["train"], **vars(tcfg)}}, fh, sort_keys=False, allow_unicode=True
        )
    manifest = {
        "model": full_cfg["meta"]["name"],
        "config_hash": h,
        "resumed": resumed,
        "janonet_version": __version__,
        "start_time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_train_real": int((train_df["y"] == 0).sum()),
        "n_train_fake": int((train_df["y"] == 1).sum()),
        "pos_weight": w,
        "trainable_params": n_train,
        "command_line": " ".join(sys.argv),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(
        f"[train] {manifest['model']} en {device}: {len(train_df):,} train / {len(val_df):,} val, "
        f"pos_weight {w:.6f}, {n_train:,} parámetros entrenables, batch {tcfg.batch_size} x {tcfg.accum_steps}"
    )

    # Pérdida del train efectivo en modo eval (clip central), antes y después: la
    # comprobación de que el entrenamiento hace bajar la pérdida en las corridas cortas.
    dl_loss = None
    if args.loss_check:
        dl_loss = DataLoader(ClipDataset(train_df, cache_dir, "eval", base_seed=tcfg.seed), shuffle=False, **kw)
        before = evaluate(model, dl_loss, device, tcfg.amp, criterion)["loss"]
        print(f"[train] pérdida en train (modo eval) antes: {before:.4f}")

    t0 = time.perf_counter()
    res = trainer.fit()
    pd.DataFrame({"step_loss": trainer.step_losses}).to_csv(run_dir / "step_losses.csv", index=False)
    if dl_loss is not None:
        after = evaluate(model, dl_loss, device, tcfg.amp, criterion)["loss"]
        print(f"[train] pérdida en train (modo eval) después: {after:.4f}")
        manifest.update(loss_check_before=before, loss_check_after=after)
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(
        f"[train] {'interrumpido' if res['interrupted'] else 'terminado'} en {(time.perf_counter() - t0) / 60:.1f} "
        f"min; mejor val_ap_real {res['best_metric']:.4f} (época {res['best_epoch']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
