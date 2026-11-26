"""Agregación temporal de los 16 cuadros: ``mean`` | ``transformer``.

- ``mean``: la cabeza lineal se aplica a cada cuadro; la pérdida es la media de las
  16 BCE y el puntaje oficial del video es la **media de las 16 probabilidades**
  (``prob_mean``). No hay logit de video.
- ``transformer``: los 16 tokens entran a un encoder (pre-norm) y sale **un logit
  por video**; el puntaje oficial es ``sigmoid(video_logits)`` (``prob_video``). Los
  logits por cuadro son diagnóstico.

Origen: ``dataset_analysis/src/tfm/models/temporal.py`` (ver PROVENANCE.md). El modo
``residual_diff`` del repositorio de desarrollo (experimento descartado) no se porta.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

TEMPORAL_MODES = ("mean", "transformer")
POS_ENCODINGS = ("learned", "sinusoidal")
TRANSFORMER_POOLS = ("mean", "cls")


class TemporalOutput(dict):
    """``frame_logits`` ``(B, T)``; ``video_logits`` ``(B,)`` o ``None``; ``mode``."""


class MeanTemporal(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.in_features = in_features
        self.head = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> TemporalOutput:
        if x.dim() != 3:
            raise ValueError(f"se esperaba (B, T, D), llegó {tuple(x.shape)}")
        return TemporalOutput(frame_logits=self.head(x).squeeze(-1), video_logits=None, mode="mean")


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]


class TransformerTemporal(nn.Module):
    """Encoder pre-norm sobre los tokens temporales; un logit por video."""

    def __init__(
        self,
        in_features: int,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 8,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        pos_encoding: str = "learned",
        pool: str = "mean",
        max_len: int = 64,
    ):
        super().__init__()
        if pos_encoding not in POS_ENCODINGS:
            raise ValueError(f"pos_encoding debe ser uno de {POS_ENCODINGS}, llegó {pos_encoding!r}")
        if pool not in TRANSFORMER_POOLS:
            raise ValueError(f"pool debe ser uno de {TRANSFORMER_POOLS}, llegó {pool!r}")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} debe ser divisible por n_heads={n_heads}")
        self.in_features = in_features
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pos_encoding_kind = pos_encoding
        self.pool = pool
        self.input_proj = nn.Identity() if in_features == d_model else nn.Linear(in_features, d_model)
        self.pos = (
            LearnedPositionalEncoding(d_model, max_len)
            if pos_encoding == "learned"
            else SinusoidalPositionalEncoding(d_model, max_len)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if pool == "cls" else None
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward or 4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        # Con norm_first=True no hay camino de tensores anidados; se desactiva explícito.
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> TemporalOutput:
        if x.dim() != 3:
            raise ValueError(f"se esperaba (B, T, D), llegó {tuple(x.shape)}")
        h = self.pos(self.input_proj(x))
        if self.cls_token is not None:
            h = torch.cat([self.cls_token.expand(h.shape[0], -1, -1), h], dim=1)
        h = self.encoder(h)
        summary, per_frame = (h[:, 0], h[:, 1:]) if self.cls_token is not None else (h.mean(dim=1), h)
        return TemporalOutput(
            frame_logits=self.head(per_frame).squeeze(-1),
            video_logits=self.head(summary).squeeze(-1),
            mode="transformer",
        )


def build_temporal(mode: str, in_features: int, **kwargs) -> nn.Module:
    if mode not in TEMPORAL_MODES:
        raise ValueError(f"temporal debe ser uno de {TEMPORAL_MODES}, llegó {mode!r}")
    if mode == "mean":
        extra = set(kwargs) - {"max_len"}
        if extra:
            raise ValueError(f"temporal='mean' no tiene hiperparámetros; sobran {sorted(extra)}")
        return MeanTemporal(in_features)
    return TransformerTemporal(in_features, **kwargs)
