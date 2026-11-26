"""Bucle de entrenamiento end-to-end, reanudable.

Receta de los modelos finales (``configs/models/*.yaml``, sección ``train``):

- AdamW sobre los parámetros entrenables, ``lr = 1e-4``, ``weight_decay = 1e-4``, sin
  programador de tasa de aprendizaje.
- ``BCEWithLogitsLoss(pos_weight = alpha · n_REAL / n_FAKE)`` con los conteos del
  train efectivo (``alpha = 1``: las dos clases pesan lo mismo). FAKE = 1.
- Pérdida según el módulo temporal: ``mean`` = media de las 16 BCE por cuadro;
  ``transformer`` = una BCE sobre el logit de video.
- AMP float16 con ``GradScaler`` en GPU; acumulación de gradiente (``accum_steps``).
- Una época = una pasada por train con una permutación determinista y ventana de 16
  cuadros desplazada de forma determinista; ``drop_last``.
- Al final de cada época: puntaje oficial en validación (clip central) y AP REAL. El
  ``best.ckpt`` es la época con mayor AP REAL de validación.

Sin MLflow: las métricas van a ``metrics_epoch.csv`` y los puntajes de validación de
cada época a ``val_scores_epochNN.csv``.

Origen: ``dataset_analysis/src/tfm/train/loop.py`` y ``scripts/04_train.py``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

from janonet.data.clip_dataset import to_float01
from janonet.train.checkpoint import CheckpointManager, StopSignal, capture_rng_state, restore_rng_state

PRIMARY_METRIC = "val_ap_real"

#: Claves de ``train`` que definen el resultado (entran en el hash de reanudación).
EXPERIMENTAL_KEYS = (
    "epochs",
    "batch_size",
    "accum_steps",
    "lr",
    "weight_decay",
    "amp",
    "seed",
    "temporal_stride",
    "pos_weight_alpha",
)


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 16
    accum_steps: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-4
    amp: bool = True
    seed: int = 20260910
    temporal_stride: int = 1
    pos_weight_alpha: float = 1.0
    num_workers: int = 4
    prefetch_factor: int = 4
    persistent_workers: bool = True
    pin_memory: bool = True
    cudnn_benchmark: bool = True
    deterministic: bool = False
    checkpoint_every_n_steps: int = 500
    log_every_n_steps: int = 50
    train_eval_subset: int = 9000

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainConfig:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def experimental(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in EXPERIMENTAL_KEYS}

    def loader_kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            kw.update(persistent_workers=self.persistent_workers, prefetch_factor=self.prefetch_factor)
        return kw


def pos_weight_from_counts(y: np.ndarray, alpha: float = 1.0) -> float:
    """``alpha · n_REAL / n_FAKE`` sobre el train efectivo (FAKE = 1)."""
    n_fake, n_real = int((y == 1).sum()), int((y == 0).sum())
    if n_fake == 0 or n_real == 0:
        raise ValueError(f"train degenerado: n_FAKE={n_fake}, n_REAL={n_real}")
    return alpha * n_real / n_fake


def compute_loss(out: dict[str, Any], y: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    if out["mode"] == "mean":
        return criterion(out["frame_logits"], y.unsqueeze(1).expand_as(out["frame_logits"]))
    return criterion(out["video_logits"], y)


def video_scores(out: dict[str, Any]) -> torch.Tensor:
    """Puntaje oficial en float32 (la sigmoide en float32: en float16 satura a 1,0 desde |logit| ≥ 8,3)."""
    if out["mode"] == "mean":
        return torch.sigmoid(out["frame_logits"].float()).mean(dim=1)
    return torch.sigmoid(out["video_logits"].float())


def ap_real(y: np.ndarray, score_fake: np.ndarray) -> float:
    y_real = 1 - np.asarray(y).astype(int)
    if y_real.sum() in (0, len(y_real)):
        return float("nan")
    return float(average_precision_score(y_real, 1.0 - np.asarray(score_fake)))


def make_grad_scaler(enabled: bool, device: torch.device):
    """``torch.cuda.amp.GradScaler``: la única que existe en el torch 2.1.2 de los modelos."""
    if device.type != "cuda" or not enabled:
        return None
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        return torch.cuda.amp.GradScaler(enabled=True)


def rewind_to_cycle(samples_consumed: int, micro_step: int, batch_size: int, accum_steps: int) -> tuple[int, int]:
    """Lleva muestras y ``micro_step`` al inicio del ciclo de acumulación en curso (al reanudar)."""
    if samples_consumed % batch_size:
        raise ValueError("samples_consumed no es múltiplo de batch_size (el cargador usa drop_last)")
    extra = (samples_consumed // batch_size) % accum_steps
    return samples_consumed - extra * batch_size, micro_step - extra


@torch.no_grad()
def evaluate(
    model: nn.Module, loader, device: torch.device, amp: bool, criterion: nn.Module | None = None
) -> dict[str, Any]:
    """Puntajes oficiales de validación (clip central). Prevalencia natural, sin reponderar."""
    model.eval()
    ids, ys, scores, losses = [], [], [], []
    for clip_u8, sample_id, y in loader:
        x = to_float01(clip_u8, device)
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            out = model(x)
            if criterion is not None:
                losses.append(float(compute_loss(out, y.to(device).float(), criterion)))
        ids.append(sample_id.numpy())
        ys.append(y.numpy())
        scores.append(video_scores(out).cpu().numpy())
    y = np.concatenate(ys)
    s = np.concatenate(scores)
    return {
        "sample_id": np.concatenate(ids),
        "y": y,
        "score": s,
        "ap_real": ap_real(y, s),
        "loss": float(np.mean(losses)) if losses else float("nan"),
    }


class Trainer:
    def __init__(
        self,
        *,
        cfg: TrainConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        train_loader,
        val_loader,
        dataset,
        sampler,
        ckpt: CheckpointManager,
        run_dir: Path,
        scaler=None,
        stop: StopSignal | None = None,
        train_eval_loader=None,
    ):
        self.cfg, self.model, self.optimizer, self.criterion = cfg, model, optimizer, criterion
        self.device, self.train_loader, self.val_loader = device, train_loader, val_loader
        self.train_eval_loader = train_eval_loader
        self.dataset, self.sampler, self.ckpt, self.scaler = dataset, sampler, ckpt, scaler
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.stop = stop or StopSignal()
        self.epoch = self.global_step = self.micro_step = self.samples_consumed = 0
        self.best_metric, self.best_epoch = -math.inf, -1
        self.history: list[dict[str, float]] = []
        self.step_losses: list[float] = []
        self._cycle_rng: dict[str, Any] | None = None

    # ── Reanudación ────────────────────────────────────────────────────────
    def resume(self) -> bool:
        found = self.ckpt.load_latest(map_location=self.device)
        if found is None:
            return False
        state, path = found
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scaler is not None and state.get("scaler"):
            self.scaler.load_state_dict(state["scaler"])
        restore_rng_state(state.get("rng_state"))
        self.epoch, self.global_step = int(state["epoch"]), int(state["global_step"])
        self.best_metric, self.best_epoch = float(state["best_metric"]), int(state["best_epoch"])
        self.history = list(state.get("history", []))
        self.samples_consumed, self.micro_step = rewind_to_cycle(
            int(state["samples_consumed_in_epoch"]), int(state["micro_step"]), self.cfg.batch_size, self.cfg.accum_steps
        )
        self.sampler.load_state_dict(state["sampler_state"])
        self.sampler.set_start_index(self.samples_consumed)
        print(
            f"[train] reanudado desde {path.name}: época {self.epoch}, paso {self.global_step}, "
            f"{self.samples_consumed:,} muestras de la época",
            flush=True,
        )
        return True

    def _state(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "micro_step": self.micro_step,
            "samples_consumed_in_epoch": self.samples_consumed,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "sampler_state": self.sampler.state_dict(),
            "history": self.history,
            # A mitad de un ciclo de acumulación se guarda el RNG de su inicio, que es
            # adonde se rebobina al reanudar: así el dropout repite las mismas máscaras.
            "rng_state": self._cycle_rng if self.micro_step % self.cfg.accum_steps else capture_rng_state(),
        }

    # ── Una época ──────────────────────────────────────────────────────────
    def train_one_epoch(self) -> dict[str, float]:
        self.model.train()
        self.dataset.set_epoch(self.epoch)
        if self.samples_consumed == 0:
            self.sampler.set_epoch(self.epoch)
        else:
            self.sampler.epoch = self.epoch
            self.sampler.set_start_index(self.samples_consumed)
        use_amp = self.cfg.amp and self.device.type == "cuda"
        total, n = 0.0, 0
        t0 = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        for clip_u8, _sid, y in self.train_loader:
            if self.micro_step % self.cfg.accum_steps == 0:
                self._cycle_rng = capture_rng_state()
            x = to_float01(clip_u8, self.device)
            target = y.to(self.device, non_blocking=True).float()
            with torch.autocast("cuda", enabled=use_amp):
                loss = compute_loss(self.model(x), target, self.criterion)
            scaled = loss / self.cfg.accum_steps
            if self.scaler is not None:
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()
            total += float(loss.item())
            n += 1
            self.step_losses.append(float(loss.item()))
            self.micro_step += 1
            self.samples_consumed += int(clip_u8.shape[0])
            if self.micro_step % self.cfg.accum_steps == 0:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                if self.cfg.log_every_n_steps and self.global_step % self.cfg.log_every_n_steps == 0:
                    print(f"[train] paso {self.global_step}: loss {loss.item():.4f}", flush=True)
                if self.cfg.checkpoint_every_n_steps and self.global_step % self.cfg.checkpoint_every_n_steps == 0:
                    self.ckpt.save(self._state())
            if self.stop.should_stop:
                self.ckpt.save(self._state())
                print(f"[train] parada limpia en época {self.epoch}, paso {self.global_step}", flush=True)
                break
        peak = torch.cuda.max_memory_allocated(self.device) / 1024**2 if self.device.type == "cuda" else 0.0
        return {
            "train_loss": total / max(n, 1),
            "epoch_seconds": time.perf_counter() - t0,
            "steps": float(n),
            "vram_peak_mb": float(peak),
        }

    # ── Ciclo completo ─────────────────────────────────────────────────────
    def fit(self) -> dict[str, Any]:
        while self.epoch < self.cfg.epochs:
            metrics = self.train_one_epoch()
            if self.stop.should_stop:
                return self._result(interrupted=True)
            self.samples_consumed = self.micro_step = 0
            res = evaluate(self.model, self.val_loader, self.device, self.cfg.amp)
            metrics.update({"val_ap_real": res["ap_real"], "val_n": float(len(res["y"]))})
            if self.train_eval_loader is not None:
                tr = evaluate(self.model, self.train_eval_loader, self.device, self.cfg.amp)
                metrics["train_ap_real"] = tr["ap_real"]
            pd.DataFrame({"sample_id": res["sample_id"], "y": res["y"], "score": res["score"]}).to_csv(
                self.run_dir / f"val_scores_epoch{self.epoch:02d}.csv", index=False
            )
            is_best = res["ap_real"] > self.best_metric
            if is_best:
                self.best_metric, self.best_epoch = res["ap_real"], self.epoch
            self.history.append({"epoch": self.epoch, **metrics})
            self.epoch += 1
            self.ckpt.save(self._state(), is_best=is_best)
            pd.DataFrame(self.history).to_csv(self.run_dir / "metrics_epoch.csv", index=False)
            vram = f", {metrics['vram_peak_mb'] / 1024:.1f} GB" if metrics["vram_peak_mb"] else ""
            print(
                f"[train] época {self.epoch - 1}: loss {metrics['train_loss']:.4f}  "
                f"{PRIMARY_METRIC} {res['ap_real']:.4f}{'  <- mejor' if is_best else ''}  "
                f"({metrics['epoch_seconds'] / 60:.1f} min{vram})",
                flush=True,
            )
        return self._result(interrupted=False)

    def _result(self, interrupted: bool) -> dict[str, Any]:
        return {
            "interrupted": interrupted,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "history": self.history,
        }
