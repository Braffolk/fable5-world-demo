"""Deterministic world-coordinate BLAKE2b noise for agriculture v2 (laas-micro-prf1).

Keyed by integer world keys so every draw is identical regardless of storage
chunking, worker count, batch order, or crop window. No periodicity, wrap,
reflection, or clamp is applied. Same person key as the raised-bog nucleation
field so the whole microtopography family shares one keyed-noise contract.
"""
from __future__ import annotations

import hashlib

import numpy as np

PERSON = b"laas-micro-prf1"
_SCALE = 1.0 / float(1 << 64)


def _key(ints: tuple[int, ...]) -> bytes:
    return b"".join(int(v).to_bytes(8, "little", signed=True) for v in ints)


def uniform(*ints: int) -> float:
    """Scalar uniform in [0, 1) keyed by an integer tuple."""
    digest = hashlib.blake2b(_key(ints), digest_size=8, person=PERSON).digest()
    return int.from_bytes(digest, "little") * _SCALE


def streams(ints: tuple[int, ...], count: int) -> np.ndarray:
    """`count` independent uniforms in [0, 1) from one keyed hash (count <= 8)."""
    if not 1 <= count <= 8:
        raise ValueError("streams supports 1..8 draws per key")
    digest = hashlib.blake2b(_key(ints), digest_size=64, person=PERSON).digest()
    words = np.frombuffer(digest, dtype="<u8", count=8).astype(np.float64)
    return (words * _SCALE)[:count]


def uniform_cells(cols_int: np.ndarray, rows_int: np.ndarray, stream: int = 0) -> np.ndarray:
    """Vectorized per-cell uniform keyed by (col_int, row_int, stream).

    ``cols_int``/``rows_int`` are broadcast integer arrays of matched shape.
    """
    cols = np.asarray(cols_int, dtype=np.int64)
    rows = np.asarray(rows_int, dtype=np.int64)
    if cols.shape != rows.shape:
        raise ValueError("cols/rows must share a shape")
    flat_c = cols.reshape(-1)
    flat_r = rows.reshape(-1)
    out = np.empty(flat_c.shape[0], dtype=np.float64)
    for i in range(flat_c.shape[0]):
        key = _key((int(flat_c[i]), int(flat_r[i]), int(stream)))
        out[i] = int.from_bytes(
            hashlib.blake2b(key, digest_size=8, person=PERSON).digest(), "little"
        ) * _SCALE
    return out.reshape(cols.shape)


def normal_from_two(u1: float, u2: float) -> float:
    """One standard normal via Box-Muller from two independent uniforms."""
    u1 = min(max(u1, 1e-12), 1.0 - 1e-12)
    return float(np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2))
