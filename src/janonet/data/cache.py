"""Caché de 32 cuadros por video en contenedores ``.bin``.

Formato idéntico al del desarrollo (así las cachés son intercambiables): cabecera de
32 bytes little-endian ``<4sHHHHBBHIIII>``

    magic 'TFMF' | versión 1 | n_frames 32 | alto 224 | ancho 224 | canales 3 |
    dtype 0 (uint8) | stride 1 | window_start | sample_id | crc32 del payload | 0

seguida de ``32 x 224 x 224 x 3`` bytes ``(cuadro, H, W, C)`` en RGB, sin compresión.
Ruta: ``<cache>/<sample_id // 1000>/<sample_id>.bin``.

Los 32 cuadros se decodifican del video procesado (mp4v) empezando en
``window_start = (processed_frames - 32) // 2``. El clip que entra a la red son 16 de
ellos: en evaluación, los cuadros 8..23 (ventana central).

Origen: ``dataset_analysis/src/tfm/data/binstore.py`` y ``dataset.py``.
"""

from __future__ import annotations

import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = b"TFMF"
FORMAT_VERSION = 1
N_FRAMES = 32
HEIGHT = WIDTH = 224
CHANNELS = 3
CLIP_LEN = 16
HEADER = struct.Struct("<4sHHHHBBHIIII")
HEADER_SIZE = HEADER.size
FRAME_NBYTES = HEIGHT * WIDTH * CHANNELS
PAYLOAD_NBYTES = N_FRAMES * FRAME_NBYTES
assert HEADER_SIZE == 32


class CacheError(RuntimeError):
    """Un fichero de caché no cumple el formato."""


@dataclass(frozen=True)
class BinHeader:
    sample_id: int
    window_start: int
    crc32: int
    n_frames: int = N_FRAMES
    height: int = HEIGHT
    width: int = WIDTH
    channels: int = CHANNELS
    dtype: int = 0
    stride: int = 1
    version: int = FORMAT_VERSION

    def pack(self) -> bytes:
        return HEADER.pack(
            MAGIC,
            self.version,
            self.n_frames,
            self.height,
            self.width,
            self.channels,
            self.dtype,
            self.stride,
            self.window_start,
            self.sample_id,
            self.crc32,
            0,
        )

    @classmethod
    def unpack(cls, raw: bytes) -> BinHeader:
        if len(raw) < HEADER_SIZE:
            raise CacheError("cabecera truncada")
        (magic, version, n, h, w, c, dtype, stride, ws, sid, crc, _) = HEADER.unpack(raw[:HEADER_SIZE])
        if magic != MAGIC:
            raise CacheError(f"magic inválido {magic!r}")
        return cls(sid, ws, crc, n, h, w, c, dtype, stride, version)


def bin_path(cache_dir: str | os.PathLike[str], sample_id: int) -> Path:
    return Path(cache_dir) / str(int(sample_id) // 1000) / f"{int(sample_id)}.bin"


def write_bin(path: str | os.PathLike[str], frames: np.ndarray, sample_id: int, window_start: int) -> Path:
    """Escritura atómica (``.tmp`` + ``os.replace``)."""
    if frames.dtype != np.uint8 or frames.shape != (N_FRAMES, HEIGHT, WIDTH, CHANNELS):
        raise CacheError(
            f"se esperaban {N_FRAMES}x{HEIGHT}x{WIDTH}x{CHANNELS} uint8, llegó {frames.shape} {frames.dtype}"
        )
    payload = np.ascontiguousarray(frames).tobytes()
    header = BinHeader(int(sample_id), int(window_start), zlib.crc32(payload) & 0xFFFFFFFF).pack()
    final = Path(path)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(final.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(header)
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, final)
    return final


def read_bin(path: str | os.PathLike[str], verify_crc: bool = False) -> tuple[BinHeader, np.ndarray]:
    with open(path, "rb") as fh:
        header = BinHeader.unpack(fh.read(HEADER_SIZE))
        payload = fh.read(PAYLOAD_NBYTES)
    if len(payload) != PAYLOAD_NBYTES:
        raise CacheError("payload truncado")
    if verify_crc and (zlib.crc32(payload) & 0xFFFFFFFF) != header.crc32:
        raise CacheError("CRC32 no coincide")
    return header, np.frombuffer(payload, dtype=np.uint8).reshape(N_FRAMES, HEIGHT, WIDTH, CHANNELS)


def read_window(path: str | os.PathLike[str], start: int, count: int = CLIP_LEN, stride: int = 1) -> np.ndarray:
    """``count`` cuadros desde ``start`` dentro de los 32 cacheados."""
    last = start + (count - 1) * stride
    if start < 0 or count <= 0 or last >= N_FRAMES:
        raise CacheError(f"ventana [{start}, {last}] fuera de los {N_FRAMES} cuadros")
    with open(path, "rb") as fh:
        if stride == 1:
            fh.seek(HEADER_SIZE + start * FRAME_NBYTES)
            raw = fh.read(count * FRAME_NBYTES)
            return np.frombuffer(raw, dtype=np.uint8).reshape(count, HEIGHT, WIDTH, CHANNELS)
        out = np.empty((count, HEIGHT, WIDTH, CHANNELS), dtype=np.uint8)
        for i in range(count):
            fh.seek(HEADER_SIZE + (start + i * stride) * FRAME_NBYTES)
            out[i] = np.frombuffer(fh.read(FRAME_NBYTES), dtype=np.uint8).reshape(HEIGHT, WIDTH, CHANNELS)
        return out


def decode_window(video_path: str | os.PathLike[str], window_start: int, n_frames: int = N_FRAMES) -> np.ndarray:
    """Decodifica ``n_frames`` consecutivos desde ``window_start`` y los devuelve en RGB.

    Recorre con ``grab()`` y solo decodifica los cuadros pedidos. No usa
    ``CAP_PROP_POS_FRAMES``: con mp4v el *seek* cae en el keyframe anterior y devuelve
    cuadros que no son los pedidos.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise CacheError(f"no se pudo abrir {video_path}")
    try:
        out = np.empty((n_frames, HEIGHT, WIDTH, CHANNELS), dtype=np.uint8)
        got, idx = 0, 0
        while idx < window_start + n_frames:
            if not cap.grab():
                raise CacheError(f"fin de video en el cuadro {idx}")
            if idx >= window_start:
                ok, bgr = cap.retrieve()
                if not ok or bgr.shape != (HEIGHT, WIDTH, CHANNELS):
                    raise CacheError(f"cuadro {idx} ilegible o con forma {getattr(bgr, 'shape', None)}")
                out[got] = bgr[:, :, ::-1]
                got += 1
            idx += 1
        return out
    finally:
        cap.release()


def window_start_for(processed_frames: int) -> int:
    return max(0, (int(processed_frames) - N_FRAMES) // 2)


# ── Ventana del clip ─────────────────────────────────────────────────────────

_M64 = 0xFFFFFFFFFFFFFFFF


def _mix64(x: int) -> int:
    """Mezclador splitmix64 (determinista entre procesos, a diferencia de ``hash()``)."""
    x &= _M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _M64
    return (x ^ (x >> 31)) & _M64


def derive_seed(base_seed: int, *parts: int) -> int:
    h = _mix64(int(base_seed) ^ 0x9E3779B97F4A7C15)
    for p in parts:
        h = _mix64(h ^ _mix64(int(p) & _M64))
    return h


def max_window_offset(n_cached: int = N_FRAMES, clip_len: int = CLIP_LEN, stride: int = 1) -> int:
    span = 1 + (clip_len - 1) * stride
    if span > n_cached:
        raise ValueError(f"un clip de {clip_len} con stride {stride} no cabe en {n_cached} cuadros")
    return n_cached - span


def center_window_offset(n_cached: int = N_FRAMES, clip_len: int = CLIP_LEN, stride: int = 1) -> int:
    """Ventana central de evaluación. Con 32/16/stride 1 da 8."""
    return max_window_offset(n_cached, clip_len, stride) // 2


def train_window_offset(base_seed: int, epoch: int, sample_id: int, max_offset: int) -> int:
    """Offset de entrenamiento determinista a partir de ``(semilla, época, sample_id)``."""
    if max_offset <= 0:
        return 0
    return derive_seed(base_seed, epoch, sample_id) % (max_offset + 1)
