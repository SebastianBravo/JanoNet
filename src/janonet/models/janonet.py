"""Modelo completo: ramas -> fusión -> pooling global -> temporal -> cabeza.

La entrada es el clip en float [0, 1] con forma ``(B, T, 3, 224, 224)``. Cada rama
normaliza por su cuenta: la RGB con ImageNet, la SRM con sus propias estadísticas.

Los nombres de los atributos (``rgb_backbone``, ``srm_residual``, ``srm_backbone``,
``fusion``, ``temporal``, ``pool``) son los del modelo que se entrenó
(``DeepfakeDetector`` en el repositorio de desarrollo): de ellos dependen las claves
del ``state_dict`` y, por tanto, que los checkpoints carguen con ``strict=True``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from janonet import paths
from janonet.models.fusion import FUSION_MODES, build_fusion
from janonet.models.rgb import RGBBackbone, normalize_imagenet
from janonet.models.srm import (
    DEFAULT_SRM_PADDING_MODE,
    SRMBackbone,
    SRMResidual,
    check_padding_mode,
    load_srm_stats,
)
from janonet.models.temporal import TEMPORAL_MODES, build_temporal

BRANCHES = ("rgb", "srm")


class ConfigError(ValueError):
    """La config no describe un modelo construible."""


def validate_config(cfg: dict[str, Any]) -> None:
    branches = cfg.get("branches")
    if not branches or not isinstance(branches, (list, tuple)):
        raise ConfigError(f"'branches' debe ser una lista con 'rgb', 'srm' o ambas; llegó {branches!r}")
    unknown = [b for b in branches if b not in BRANCHES]
    if unknown:
        raise ConfigError(f"ramas desconocidas: {unknown}")
    if len(set(branches)) != len(branches):
        raise ConfigError(f"ramas repetidas: {list(branches)}")
    fusion = cfg.get("fusion", "none")
    if fusion not in FUSION_MODES:
        raise ConfigError(f"fusión desconocida {fusion!r}; válidas: {list(FUSION_MODES)}")
    temporal = cfg.get("temporal", "mean")
    if temporal not in TEMPORAL_MODES:
        raise ConfigError(f"modo temporal desconocido {temporal!r}; válidos: {list(TEMPORAL_MODES)}")
    if fusion != "none" and set(branches) != set(BRANCHES):
        raise ConfigError(f"fusion={fusion!r} necesita las dos ramas; branches={list(branches)}")
    if fusion == "none" and len(branches) != 1:
        raise ConfigError("dos ramas sin fusión: usa 'concat' o 'cross_attention'")
    if temporal == "mean" and cfg.get("transformer"):
        raise ConfigError("hiperparámetros de transformer con temporal='mean'")


class JanoNet(nn.Module):
    """Detector de video. ``forward`` devuelve ``frame_logits`` y ``video_logits``."""

    def __init__(
        self,
        branches: list[str],
        rgb_backbone: RGBBackbone | None,
        srm_residual: SRMResidual | None,
        srm_backbone: SRMBackbone | None,
        fusion: nn.Module,
        temporal: nn.Module,
    ):
        super().__init__()
        self.branches = list(branches)
        self.rgb_backbone = rgb_backbone
        self.srm_residual = srm_residual
        self.srm_backbone = srm_backbone
        self.fusion = fusion
        self.temporal = temporal
        self.pool = nn.AdaptiveAvgPool2d(1)

    def fused_grid(self, clip01: torch.Tensor) -> torch.Tensor:
        """``(B, T, 3, H, W)`` en [0, 1] -> rejilla fusionada ``(B*T, D, 7, 7)``."""
        if clip01.dim() != 5:
            raise ValueError(f"se esperaba (B, T, 3, H, W) en [0, 1], llegó {tuple(clip01.shape)}")
        b, t = clip01.shape[:2]
        x = clip01.reshape(b * t, *clip01.shape[2:])
        grids: dict[str, torch.Tensor] = {}
        if "rgb" in self.branches:
            grids["rgb"] = self.rgb_backbone(normalize_imagenet(x))
        if "srm" in self.branches:
            grids["srm"] = self.srm_backbone(self.srm_residual(x))
        return self.fusion(grids)

    def forward(self, clip01: torch.Tensor) -> dict[str, torch.Tensor]:
        b, t = clip01.shape[:2]
        feats = self.pool(self.fused_grid(clip01)).flatten(1)
        return self.temporal(feats.reshape(b, t, -1))

    @property
    def temporal_mode(self) -> str:
        return "transformer" if hasattr(self.temporal, "encoder") else "mean"


def build_model(
    cfg: dict[str, Any], srm_stats_path: str | Path | None = None, pretrained: bool | None = None
) -> JanoNet:
    """Construye el modelo desde la sección de modelo de una config.

    ``pretrained=None`` respeta la config (ImageNet para entrenar); ``False`` evita
    descargar pesos de ImageNet cuando se va a cargar un checkpoint.
    """
    validate_config(cfg)
    branches = list(cfg["branches"])
    rgb_cfg = dict(cfg.get("rgb") or {})
    srm_cfg = dict(cfg.get("srm") or {})
    rgb_backbone = srm_residual = srm_backbone = None
    rgb_channels = srm_channels = None

    if "rgb" in branches:
        rgb_backbone = RGBBackbone(
            pretrained=rgb_cfg.get("pretrained", True) if pretrained is None else pretrained,
            freeze_until=rgb_cfg.get("freeze_until"),
        )
        rgb_channels = rgb_backbone.out_channels

    if "srm" in branches:
        padding_mode = srm_cfg.get("padding_mode", DEFAULT_SRM_PADDING_MODE)
        stats = load_srm_stats(srm_stats_path or paths.srm_stats_path())
        check_padding_mode(padding_mode, stats)
        srm_residual = SRMResidual(mean=stats["mean"], std=stats["std"], padding_mode=padding_mode)
        srm_backbone = SRMBackbone(
            stem_stride=srm_cfg.get("stem_stride", 2),
            use_maxpool=srm_cfg.get("use_maxpool", True),
            grid_adjust=srm_cfg.get("grid_adjust", "stride"),
            pretrained=srm_cfg.get("pretrained", True) if pretrained is None else pretrained,
        )
        srm_channels = srm_backbone.out_channels

    fusion = build_fusion(cfg.get("fusion", "none"), rgb_channels, srm_channels, **dict(cfg.get("fusion_params") or {}))
    temporal = build_temporal(cfg.get("temporal", "mean"), fusion.out_channels, **dict(cfg.get("transformer") or {}))
    return JanoNet(branches, rgb_backbone, srm_residual, srm_backbone, fusion, temporal)
