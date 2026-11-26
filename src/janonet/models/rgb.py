"""Rama RGB: EfficientNet-B0 sin clasificador.

Entrada ``(N, 3, 224, 224)`` normalizada con las estadísticas de ImageNet; salida
``(N, 1280, 7, 7)``, la rejilla espacial **sin** pooling global (la atención cruzada
necesita los 49 tokens).

``requires_grad=False`` congela los pesos pero no las estadísticas de BatchNorm, así
que ``train()`` fuerza ``eval()`` en los bloques congelados. Solo importa para E6
(``freeze_until: features.7``); U100, RS-ca y RS-ca-tr entrenan el 100 %.

Origen: ``dataset_analysis/src/tfm/models/rgb_backbone.py`` (ver PROVENANCE.md).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

RGB_OUT_CHANNELS = 1280
N_FEATURE_BLOCKS = 9

#: Estadísticas de ImageNet para normalizar la entrada de la rama RGB.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_imagenet(x01: torch.Tensor) -> torch.Tensor:
    """``x01`` en [0, 1] con canales en ``dim=-3`` -> normalizado con ImageNet."""
    m = torch.as_tensor(IMAGENET_MEAN, dtype=x01.dtype, device=x01.device).reshape(-1, 1, 1)
    s = torch.as_tensor(IMAGENET_STD, dtype=x01.dtype, device=x01.device).reshape(-1, 1, 1)
    return (x01 - m) / s


def _parse_freeze_point(freeze_until: str) -> tuple[int, int | None]:
    """``"features.7"`` -> ``(7, None)``; ``"features.6.2"`` -> ``(6, 2)``."""
    text = str(freeze_until).strip().removeprefix("features.")
    parts = text.split(".")
    try:
        if len(parts) not in (1, 2):
            raise ValueError
        idx = int(parts[0])
        sub = int(parts[1]) if len(parts) == 2 else None
    except ValueError as exc:
        raise ValueError(f"freeze_until debe ser 'features.<n>' o 'features.<n>.<m>', llegó {freeze_until!r}") from exc
    if not 0 <= idx < N_FEATURE_BLOCKS:
        raise ValueError(f"freeze_until={freeze_until!r} fuera de rango (0..{N_FEATURE_BLOCKS - 1})")
    return idx, sub


class RGBBackbone(nn.Module):
    """EfficientNet-B0 recortado a ``features``.

    ``freeze_until`` nombra el último módulo congelado, inclusive. ``None`` = 100 %
    entrenable.
    """

    def __init__(self, pretrained: bool = True, freeze_until: str | None = None):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b0(weights=weights)
        self.features = model.features
        self.out_channels = RGB_OUT_CHANNELS
        self.frozen_module_names: list[str] = []
        self.freeze_until = freeze_until
        if freeze_until is not None:
            self._freeze(*_parse_freeze_point(freeze_until))

    def _freeze(self, up_to: int, up_to_sub: int | None = None) -> None:
        if up_to_sub is not None:
            children = list(self.features[up_to].named_children())
            kinds = {type(c).__name__ for _, c in children}
            if not kinds <= {"MBConv", "FusedMBConv"}:
                raise ValueError(f"features.{up_to} no es una secuencia de MBConv ({sorted(kinds)})")
            if not 0 <= up_to_sub < len(children):
                raise ValueError(f"features.{up_to} tiene {len(children)} MBConv, llegó {up_to_sub}")
        for name, block in self.features.named_children():
            i = int(name)
            if i < up_to or (i == up_to and up_to_sub is None):
                for p in block.parameters():
                    p.requires_grad_(False)
                self.frozen_module_names.append(f"features.{name}")
            elif i == up_to:
                for sub_name, sub in block.named_children():
                    if int(sub_name) <= up_to_sub:
                        for p in sub.parameters():
                            p.requires_grad_(False)
                        self.frozen_module_names.append(f"features.{name}.{sub_name}")

    def train(self, mode: bool = True) -> RGBBackbone:
        super().train(mode)
        if mode:
            for name in self.frozen_module_names:
                self.get_submodule(name).eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"se esperaba (N, 3, H, W), llegó {tuple(x.shape)}")
        return self.features(x)

    def trainable_fraction(self) -> float:
        total = sum(p.numel() for p in self.features.parameters())
        trainable = sum(p.numel() for p in self.features.parameters() if p.requires_grad)
        return trainable / total

    def extra_repr(self) -> str:
        return f"out_channels={self.out_channels}, freeze_until={self.freeze_until!r}"
