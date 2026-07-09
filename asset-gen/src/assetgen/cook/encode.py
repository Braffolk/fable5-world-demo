"""Raster/record payload codecs + their decode twins.

Encodings (LAC1 `enc` field):
  1 = quant16 + 2D-delta + compress   (height, waterY)
  2 = u8 planes + compress            (landcover classId/vegDensity/soil — RLE left to the codec)
  3 = records + compress              (tree instance SoA)

Compression is INSIDE the payload (default raw-zlib "deflate", decodable in the browser by
DecompressionStream('deflate')), so cooked files are self-contained and content-hashable.
Every encoder has a decode twin used by `assetgen verify` for bit-exact round-trip checks.
"""
from __future__ import annotations

import zlib

import numpy as np

from ..config import EncodeConfig


def _compress(cfg: EncodeConfig, raw: bytes) -> bytes:
    if cfg.codec == "deflate":
        return zlib.compress(raw, cfg.deflate_level)
    if cfg.codec == "zstd":
        import zstandard

        return zstandard.ZstdCompressor(level=cfg.zstd_level).compress(raw)
    raise ValueError(f"unknown codec {cfg.codec}")


def _decompress(cfg: EncodeConfig, payload: bytes) -> bytes:
    if cfg.codec == "deflate":
        return zlib.decompress(payload)
    if cfg.codec == "zstd":
        import zstandard

        return zstandard.ZstdDecompressor().decompress(payload)
    raise ValueError(f"unknown codec {cfg.codec}")


# --- enc 1: quantized u16 raster with 2D delta filter ---------------------------------


def delta2d(q: np.ndarray) -> np.ndarray:
    """Wrapping u16 delta: each texel predicts from its left neighbor; first column
    predicts from the texel above; (0,0) stays raw. Residuals of smooth terrain are
    near-zero -> deflate/zstd crush them."""
    d = q.astype(np.uint16).copy()
    d[:, 1:] -= q[:, :-1]
    d[1:, 0] -= q[:-1, 0]
    return d


def undelta2d(d: np.ndarray) -> np.ndarray:
    q = np.cumsum(d.astype(np.uint64), axis=1, dtype=np.uint64)
    first_col = np.cumsum(d[:, 0].astype(np.uint64), dtype=np.uint64)
    q += (first_col - d[:, 0])[:, None]
    return (q & 0xFFFF).astype(np.uint16)


def encode_quant16(cfg: EncodeConfig, arr: np.ndarray, qscale: float) -> tuple[bytes, float]:
    """f32/f64 raster -> (payload, qoffset). Values clip into [qoffset, qoffset + 65535*qscale]."""
    finite = arr[np.isfinite(arr)]
    qoffset = float(np.floor(finite.min() - 1.0)) if finite.size else 0.0
    q = np.round((np.nan_to_num(arr, nan=qoffset) - qoffset) / qscale)
    q = np.clip(q, 0, 65535).astype(np.uint16)
    return _compress(cfg, delta2d(q).tobytes()), qoffset


def decode_quant16(
    cfg: EncodeConfig, payload: bytes, res: int, qoffset: float, qscale: float
) -> np.ndarray:
    d = np.frombuffer(_decompress(cfg, payload), dtype=np.uint16).reshape(res, res)
    return undelta2d(d).astype(np.float32) * qscale + qoffset


# --- enc 2: stacked u8 planes ----------------------------------------------------------


def encode_u8_planes(cfg: EncodeConfig, planes: list[np.ndarray]) -> bytes:
    for p in planes:
        assert p.dtype == np.uint8 and p.shape == planes[0].shape
    return _compress(cfg, b"".join(p.tobytes() for p in planes))


def decode_u8_planes(cfg: EncodeConfig, payload: bytes, res: int, nplanes: int) -> list[np.ndarray]:
    raw = _decompress(cfg, payload)
    assert len(raw) == res * res * nplanes
    return [
        np.frombuffer(raw[i * res * res : (i + 1) * res * res], dtype=np.uint8).reshape(res, res)
        for i in range(nplanes)
    ]


# --- enc 3: record SoA ------------------------------------------------------------------


def encode_records(cfg: EncodeConfig, columns: list[np.ndarray]) -> bytes:
    """Struct-of-arrays: each column is a 1D typed array; all must share length."""
    n = len(columns[0]) if columns else 0
    for c in columns:
        assert c.ndim == 1 and len(c) == n
    return _compress(cfg, b"".join(c.tobytes() for c in columns))


def decode_records(cfg: EncodeConfig, payload: bytes, count: int, dtypes: list[str]) -> list[np.ndarray]:
    raw = _decompress(cfg, payload)
    out, off = [], 0
    for dt in dtypes:
        size = count * np.dtype(dt).itemsize
        out.append(np.frombuffer(raw[off : off + size], dtype=dt))
        off += size
    assert off == len(raw)
    return out
