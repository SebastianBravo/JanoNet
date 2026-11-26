"""Fusión de las rejillas RGB y SRM: ``none`` | ``concat`` | ``cross_attention``.

Todas operan sobre rejillas ``(N, C, 7, 7)`` antes del pooling global y devuelven una
rejilla ``(N, D, 7, 7)``.

Origen: ``dataset_analysis/src/tfm/models/fusion.py`` (ver PROVENANCE.md).
"""

from __future__ import annotations

import torch
import torch.nn as nn

FUSION_MODES = ("none", "concat", "cross_attention")


def _to_tokens(grid: torch.Tensor) -> torch.Tensor:
    """``(N, C, H, W)`` -> ``(N, H*W, C)``."""
    return grid.flatten(2).transpose(1, 2)


def _to_grid(tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """``(N, H*W, C)`` -> ``(N, C, H, W)``."""
    n, _, c = tokens.shape
    return tokens.transpose(1, 2).reshape(n, c, h, w)


class NoFusion(nn.Module):
    """Una sola rama: deja pasar su rejilla."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, grids: dict[str, torch.Tensor]) -> torch.Tensor:
        if len(grids) != 1:
            raise ValueError(f"fusion='none' espera una rama, llegaron {sorted(grids)}")
        return next(iter(grids.values()))


class ConcatFusion(nn.Module):
    """Concatena las dos rejillas por canales."""

    def __init__(self, rgb_channels: int, srm_channels: int):
        super().__init__()
        self.rgb_channels = rgb_channels
        self.srm_channels = srm_channels
        self.out_channels = rgb_channels + srm_channels

    def forward(self, grids: dict[str, torch.Tensor]) -> torch.Tensor:
        rgb, srm = grids["rgb"], grids["srm"]
        if rgb.shape[-2:] != srm.shape[-2:]:
            raise ValueError(f"rejillas distintas: rgb {tuple(rgb.shape[-2:])}, srm {tuple(srm.shape[-2:])}")
        return torch.cat([rgb, srm], dim=1)


class CrossAttentionFusion(nn.Module):
    """Atención cruzada bidireccional sobre los 49 tokens espaciales.

    Ambas ramas se proyectan con una conv 1x1 a ``d_model``; RGB atiende a SRM y SRM
    atiende a RGB, cada una con conexión residual y LayerNorm; la salida concatena las
    dos (``2 * d_model`` canales).
    """

    def __init__(
        self, rgb_channels: int, srm_channels: int, d_model: int = 256, n_heads: int = 8, dropout: float = 0.0
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} debe ser divisible por n_heads={n_heads}")
        self.rgb_channels = rgb_channels
        self.srm_channels = srm_channels
        self.d_model = d_model
        self.n_heads = n_heads
        self.out_channels = 2 * d_model
        self.proj_rgb = nn.Conv2d(rgb_channels, d_model, kernel_size=1)
        self.proj_srm = nn.Conv2d(srm_channels, d_model, kernel_size=1)
        self.attn_rgb_from_srm = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_srm_from_rgb = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_rgb = nn.LayerNorm(d_model)
        self.norm_srm = nn.LayerNorm(d_model)

    def forward(self, grids: dict[str, torch.Tensor]) -> torch.Tensor:
        rgb, srm = grids["rgb"], grids["srm"]
        if rgb.shape[-2:] != srm.shape[-2:]:
            raise ValueError(f"rejillas distintas: rgb {tuple(rgb.shape[-2:])}, srm {tuple(srm.shape[-2:])}")
        h, w = rgb.shape[-2:]
        t_rgb = _to_tokens(self.proj_rgb(rgb))
        t_srm = _to_tokens(self.proj_srm(srm))
        cross_rgb, _ = self.attn_rgb_from_srm(query=t_rgb, key=t_srm, value=t_srm, need_weights=False)
        cross_srm, _ = self.attn_srm_from_rgb(query=t_srm, key=t_rgb, value=t_rgb, need_weights=False)
        t_rgb = self.norm_rgb(t_rgb + cross_rgb)
        t_srm = self.norm_srm(t_srm + cross_srm)
        return _to_grid(torch.cat([t_rgb, t_srm], dim=-1), h, w)


def build_fusion(
    mode: str,
    rgb_channels: int | None,
    srm_channels: int | None,
    d_model: int = 256,
    n_heads: int = 8,
    dropout: float = 0.0,
) -> nn.Module:
    if mode not in FUSION_MODES:
        raise ValueError(f"fusion debe ser uno de {FUSION_MODES}, llegó {mode!r}")
    if mode == "none":
        channels = rgb_channels if rgb_channels is not None else srm_channels
        if channels is None:
            raise ValueError("fusion='none' necesita al menos una rama")
        return NoFusion(channels)
    if rgb_channels is None or srm_channels is None:
        raise ValueError(f"fusion={mode!r} necesita las dos ramas")
    if mode == "concat":
        return ConcatFusion(rgb_channels, srm_channels)
    return CrossAttentionFusion(rgb_channels, srm_channels, d_model, n_heads, dropout)
