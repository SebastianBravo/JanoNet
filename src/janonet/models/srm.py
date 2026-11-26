"""Rama SRM: residuo de 3 kernels fijos + ResNet-18.

Residuo
-------
Tres kernels de Spatial Rich Models (Fridrich y Kodovský; usados en visión por
RGB-N, Zhou et al., CVPR 2018): primer orden 3x3, segundo orden 5x5 y direccional
horizontal 1x3, embebidos en 5x5 y divididos por su factor. Cada canal de salida
aplica su kernel a R, G y B **y los suma** (no es depthwise). Relleno
``replicate`` de 2 px. La entrada es el cuadro en [0, 1] (uint8/255), **sin**
normalizar con ImageNet. La salida se normaliza con ``(r - mean) / std`` por canal,
con las estadísticas de ``configs/srm_stats.json`` (calculadas sobre train y válidas
solo para el relleno con que se calcularon).

Backbone
--------
ResNet-18 sin ``avgpool``/``fc``. Con ``stem_stride=2`` y ``use_maxpool=True`` (los
valores que corrieron) la rejilla de salida ya es 7x7 y ``post_adjust`` es la
identidad. El mecanismo de ajuste se conserva para no cambiar la semántica de la
config, y se verifica con un forward de prueba en el constructor.

Origen: ``dataset_analysis/src/tfm/data/srm.py`` y
``dataset_analysis/src/tfm/models/srm_backbone.py`` (ver PROVENANCE.md).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

SRM_KERNEL_SIZE = 5
SRM_N_KERNELS = 3
SRM_PADDING = SRM_KERNEL_SIZE // 2

#: K1 — primer orden, 3x3 embebido en 5x5. Divisor 4.
K1 = (
    (0, 0, 0, 0, 0),
    (0, -1, 2, -1, 0),
    (0, 2, -4, 2, 0),
    (0, -1, 2, -1, 0),
    (0, 0, 0, 0, 0),
)
#: K2 — segundo orden, 5x5 completo. Divisor 12.
K2 = (
    (-1, 2, -2, 2, -1),
    (2, -6, 8, -6, 2),
    (-2, 8, -12, 8, -2),
    (2, -6, 8, -6, 2),
    (-1, 2, -2, 2, -1),
)
#: K3 — direccional horizontal 1x3 embebido en 5x5. Divisor 2.
K3 = (
    (0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0),
    (0, 1, -2, 1, 0),
    (0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0),
)
DIVISORS = (4.0, 12.0, 2.0)

SRM_PADDING_MODES = ("replicate", "zeros", "reflect")
DEFAULT_SRM_PADDING_MODE = "replicate"


def srm_kernels() -> np.ndarray:
    """Los 3 kernels ya divididos por su factor. ``(3, 5, 5)``, float64."""
    stack = np.stack([np.asarray(k, dtype=np.float64) for k in (K1, K2, K3)])
    return stack / np.asarray(DIVISORS, dtype=np.float64).reshape(-1, 1, 1)


def srm_filter_bank(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Pesos para ``conv2d``: ``(3, 3, 5, 5)`` = (salida, entrada RGB, kH, kW)."""
    k = torch.from_numpy(srm_kernels()).to(dtype)
    return k.unsqueeze(1).repeat(1, 3, 1, 1).contiguous()


def apply_srm(x: torch.Tensor, weight: torch.Tensor, padding_mode: str = DEFAULT_SRM_PADDING_MODE) -> torch.Tensor:
    """Respuesta cruda del banco SRM. ``x`` en [0, 1], ``(N, 3, H, W)`` -> ``(N, 3, H, W)``."""
    if x.dim() != 4 or x.shape[1] != 3:
        raise ValueError(f"se esperaba (N, 3, H, W), llegó {tuple(x.shape)}")
    if padding_mode not in SRM_PADDING_MODES:
        raise ValueError(f"padding_mode debe ser uno de {SRM_PADDING_MODES}, llegó {padding_mode!r}")
    w = weight.to(device=x.device, dtype=x.dtype)
    if padding_mode == "zeros":
        return F.conv2d(x, w, bias=None, padding=SRM_PADDING)
    return F.conv2d(F.pad(x, (SRM_PADDING,) * 4, mode=padding_mode), w, bias=None, padding=0)


def load_srm_stats(path: str | Path) -> dict[str, Any]:
    """Lee y valida ``srm_stats.json``."""
    p = Path(path)
    stats = json.loads(p.read_text(encoding="utf-8"))
    for key in ("mean", "std"):
        if len(stats.get(key, [])) != SRM_N_KERNELS:
            raise ValueError(f"{p}: '{key}' debe tener {SRM_N_KERNELS} valores")
    if any(s <= 0 for s in stats["std"]):
        raise ValueError(f"{p}: hay std no positivas: {stats['std']}")
    if "padding_mode" not in stats:
        raise ValueError(f"{p}: falta 'padding_mode'")
    return stats


RESIDUAL_PRECISIONS = ("float32", "float16")


class SRMResidual(nn.Module):
    """Capa fija: residuo SRM normalizado.

    Pesos, media y std van como *buffers* no persistentes: no se entrenan y **no**
    están en el ``state_dict`` (igual que en el modelo que se entrenó), así que la
    carga con ``strict=True`` no los espera.

    ``residual_precision="float16"`` redondea la entrada, los pesos y la respuesta del
    banco a float16, que es lo que hacía la convolución bajo AMP con la que se
    entrenaron y evaluaron los modelos finales. La rama SRM es muy sensible a ese
    redondeo (el residuo vive en ~1e-2): calculado en float32, RS-ca-tr se aleja hasta
    0,26 de sus puntajes publicados y E6 hasta 0,76; con el redondeo, menos de 0,01.
    El redondeo es idempotente, así que bajo AMP en GPU no cambia nada.
    """

    def __init__(
        self,
        mean: tuple[float, ...] | list[float] | None = None,
        std: tuple[float, ...] | list[float] | None = None,
        dtype: torch.dtype = torch.float32,
        padding_mode: str = DEFAULT_SRM_PADDING_MODE,
        residual_precision: str = "float32",
    ):
        super().__init__()
        if padding_mode not in SRM_PADDING_MODES:
            raise ValueError(f"padding_mode debe ser uno de {SRM_PADDING_MODES}, llegó {padding_mode!r}")
        if residual_precision not in RESIDUAL_PRECISIONS:
            raise ValueError(f"residual_precision debe ser uno de {RESIDUAL_PRECISIONS}, llegó {residual_precision!r}")
        if (mean is None) != (std is None):
            raise ValueError("mean y std deben darse juntos o ninguno de los dos")
        self.padding_mode = padding_mode
        self.residual_precision = residual_precision
        self.register_buffer("weight", srm_filter_bank(dtype), persistent=False)
        self.is_normalized = mean is not None
        if self.is_normalized:
            m = torch.as_tensor(mean, dtype=dtype).reshape(1, SRM_N_KERNELS, 1, 1)
            s = torch.as_tensor(std, dtype=dtype).reshape(1, SRM_N_KERNELS, 1, 1)
            if torch.any(s <= 0):
                raise ValueError(f"std debe ser positiva, llegó {std}")
        else:
            m = torch.zeros(1, SRM_N_KERNELS, 1, 1, dtype=dtype)
            s = torch.ones(1, SRM_N_KERNELS, 1, 1, dtype=dtype)
        self.register_buffer("mean", m, persistent=False)
        self.register_buffer("std", s, persistent=False)

    @classmethod
    def from_stats_file(cls, path: str | Path, **kwargs: Any) -> SRMResidual:
        stats = load_srm_stats(path)
        kwargs.setdefault("padding_mode", stats["padding_mode"])
        return cls(mean=stats["mean"], std=stats["std"], **kwargs)

    def raw(self, x: torch.Tensor) -> torch.Tensor:
        """Respuesta cruda de los 3 kernels (sin normalizar)."""
        if self.residual_precision == "float16":
            dt = x.dtype
            r = apply_srm(x.half().to(dt), self.weight.half().to(dt), self.padding_mode)
            return r.half().to(dt)
        return apply_srm(x, self.weight, self.padding_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.raw(x) - self.mean) / self.std

    def extra_repr(self) -> str:
        return (
            f"n_kernels={SRM_N_KERNELS}, normalized={self.is_normalized}, padding_mode={self.padding_mode!r}, "
            f"residual_precision={self.residual_precision!r}"
        )


def check_padding_mode(config_padding_mode: str, stats: dict) -> None:
    """Aborta si el relleno de la config no es el de ``srm_stats.json``."""
    stored = stats.get("padding_mode")
    if stored != config_padding_mode:
        raise ValueError(
            f"padding_mode incoherente: config={config_padding_mode!r}, srm_stats.json={stored!r}. "
            "Las estadísticas solo valen para el relleno con que se calcularon."
        )


REQUIRED_GRID = 7
SRM_OUT_CHANNELS = 512
GRID_ADJUST_MODES = ("stride", "pool")


class SRMBackbone(nn.Module):
    """ResNet-18 sobre el residuo SRM normalizado: ``(N, 3, 224, 224)`` -> ``(N, 512, 7, 7)``."""

    def __init__(
        self,
        stem_stride: int = 2,
        use_maxpool: bool = True,
        grid_adjust: str = "stride",
        pretrained: bool = True,
        input_size: int = 224,
    ):
        super().__init__()
        if grid_adjust not in GRID_ADJUST_MODES:
            raise ValueError(f"grid_adjust debe ser uno de {GRID_ADJUST_MODES}, llegó {grid_adjust!r}")
        if stem_stride < 1:
            raise ValueError(f"stem_stride debe ser >= 1, llegó {stem_stride}")
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = resnet18(weights=weights)
        self.stem_stride = stem_stride
        self.use_maxpool = use_maxpool
        self.grid_adjust = grid_adjust
        self.out_channels = SRM_OUT_CHANNELS
        r.conv1.stride = (stem_stride, stem_stride)
        self.conv1, self.bn1, self.relu = r.conv1, r.bn1, r.relu
        self.maxpool = r.maxpool if use_maxpool else nn.Identity()
        self.layer1, self.layer2 = r.layer1, r.layer2
        self.layer3, self.layer4 = r.layer3, r.layer4
        self.post_adjust: nn.Module = nn.Identity()
        self._enforce_grid(input_size)

    def _grid_size(self, input_size: int) -> int:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            out = self._forward_body(torch.zeros(1, 3, input_size, input_size))
        self.train(was_training)
        return int(out.shape[-1])

    def _enforce_grid(self, input_size: int) -> None:
        actual = self._grid_size(input_size)
        if actual == REQUIRED_GRID:
            return
        if actual < REQUIRED_GRID:
            raise ValueError(f"el stem deja una rejilla {actual}x{actual} < {REQUIRED_GRID}x{REQUIRED_GRID}")
        factor, exact = divmod(actual, REQUIRED_GRID)
        if self.grid_adjust == "stride" and exact == 0:
            _set_layer_stride(self.layer1, factor)
        else:
            self.post_adjust = nn.AdaptiveAvgPool2d(REQUIRED_GRID)
        final = self._grid_size(input_size)
        if final != REQUIRED_GRID:
            raise ValueError(f"no se pudo llevar la rejilla a 7x7 (quedó en {final}x{final})")

    def _forward_body(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.post_adjust(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"se esperaba (N, C, H, W), llegó {tuple(x.shape)}")
        return self._forward_body(x)

    def extra_repr(self) -> str:
        return (
            f"out_channels={self.out_channels}, stem_stride={self.stem_stride}, "
            f"use_maxpool={self.use_maxpool}, grid_adjust={self.grid_adjust!r}"
        )


def _set_layer_stride(layer: nn.Sequential, stride: int) -> None:
    block = layer[0]
    block.conv1.stride = (stride, stride)
    if block.downsample is not None:
        block.downsample[0].stride = (stride, stride)
        return
    block.downsample = nn.Sequential(
        nn.Conv2d(block.conv1.in_channels, block.conv2.out_channels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(block.conv2.out_channels),
    )
