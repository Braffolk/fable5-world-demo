"""Deterministic world-coordinate BLAKE2b nucleation noise (laas-micro-prf1).

Keyed by integer world (E, N) cell coordinates so the field is identical regardless
of storage chunking, worker count, batch order, or crop window. No periodicity, wrap,
reflection, or clamp is ever applied.
"""
from __future__ import annotations

import hashlib

import numpy as np

PERSON = b"laas-micro-prf1"


def world_uniform(east_int: np.ndarray, north_int: np.ndarray) -> np.ndarray:
    """Per-cell uniform value in [0, 1) keyed by integer world coordinates.

    ``east_int`` and ``north_int`` are broadcast integer arrays of matched shape.
    Each cell hashes its own (E, N) with BLAKE2b(person=laas-micro-prf1); the first
    8 bytes become a uint64 mapped to [0, 1). Vectorized over unique coordinate pairs.
    """
    east = np.asarray(east_int, dtype=np.int64)
    north = np.asarray(north_int, dtype=np.int64)
    if east.shape != north.shape:
        raise ValueError("east/north coordinate arrays must share a shape")
    flat_e = east.reshape(-1)
    flat_n = north.reshape(-1)
    out = np.empty(flat_e.shape[0], dtype=np.float64)
    scale = 1.0 / float(1 << 64)
    for index in range(flat_e.shape[0]):
        key = int(flat_e[index]).to_bytes(8, "little", signed=True) + int(
            flat_n[index]
        ).to_bytes(8, "little", signed=True)
        digest = hashlib.blake2b(key, digest_size=8, person=PERSON).digest()
        out[index] = int.from_bytes(digest, "little") * scale
    return out.reshape(east.shape)


def world_uniform_grid(
    east_min: int, north_max: int, width: int, height: int, cell_m: int
) -> np.ndarray:
    """Uniform nucleation field over an integer world grid, rows north-to-south.

    Cell (row, col) is centered at world east = east_min + col*cell_m, north =
    north_max - row*cell_m (integer cell coordinates). Independent of any chunking.
    """
    cols = east_min + np.arange(width, dtype=np.int64) * cell_m
    rows = north_max - np.arange(height, dtype=np.int64) * cell_m
    east = np.broadcast_to(cols[None, :], (height, width))
    north = np.broadcast_to(rows[:, None], (height, width))
    return world_uniform(east, north)
