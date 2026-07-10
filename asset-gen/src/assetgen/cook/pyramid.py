"""Coarse-rung (LOD >= 1) derivation for the 2 m-family raster layers.

Unlike height (whose coarse rungs re-read the source mosaic at bigger texels), the 2 m
layers reduce the already-cooked finer rung: each LOD k chunk covers 4x4 LOD k-1 chunks
and each texel the 4x4 finer texels underneath. Reducers are per-plane: MAJORITY for
class ids (ties break to the HIGHER id, so water/sea/wetland win mixed coastline texels
— the far shell tints by class), plain or cover-weighted MEAN for densities/heights,
and majority-wet for water levels. Assembly takes the 4-texel apron strip from the
east/south finer neighbors where they exist and edge-replicates where they don't.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ..grid import ChunkId

ReadFiner = Callable[[ChunkId], list[np.ndarray] | None]


def assemble_finer(
    read_finer: ReadFiner, coarse: ChunkId, n: int, fills: list[float]
) -> list[np.ndarray] | None:
    """Mosaic the up-to-4x4 LOD k-1 chunks under `coarse` into (4n+4, 4n+4) planes.

    n = real texels per finer chunk (res - 1); the +4 strip is the coarse apron's worth
    of finer texels, pasted from the east/south/corner neighbors. Missing interior
    chunks keep `fills`; missing strip neighbors edge-replicate. Returns None when no
    interior finer chunk exists at all (coarse chunk carries no data -> omit).
    """
    first = None
    tiles: dict[tuple[int, int], list[np.ndarray]] = {}
    for dz in range(5):
        for dx in range(5):
            planes = read_finer(ChunkId(4 * coarse.cx + dx, 4 * coarse.cz + dz, coarse.lod - 1))
            if planes is not None:
                tiles[dx, dz] = planes
                first = first if first is not None else planes
    if first is None or not any(dx < 4 and dz < 4 for dx, dz in tiles):
        return None

    big = [np.full((4 * n + 4, 4 * n + 4), fill, dtype=p.dtype) for fill, p in zip(fills, first)]
    for (dx, dz), planes in tiles.items():
        if dx == 4 or dz == 4:
            continue
        for b, p in zip(big, planes):
            b[dz * n : (dz + 1) * n, dx * n : (dx + 1) * n] = p[:n, :n]
    for b in big:  # apron default: replicate the last real row/col…
        b[:, 4 * n :] = b[:, 4 * n - 1 : 4 * n]
        b[4 * n :, :] = b[4 * n - 1 : 4 * n, :]
    for (dx, dz), planes in tiles.items():  # …then paste real neighbors over it
        if dx < 4 and dz < 4:
            continue
        h, w = (4 if dz == 4 else n), (4 if dx == 4 else n)
        for b, p in zip(big, planes):
            b[dz * n : dz * n + h, dx * n : dx * n + w] = p[:h, :w]
    return big


def blocks16(plane: np.ndarray) -> np.ndarray:
    """(4m, 4m) raster -> (m, m, 16) view: each coarse texel's 4x4 finer block."""
    m = plane.shape[0] // 4
    return plane.reshape(m, 4, m, 4).transpose(0, 2, 1, 3).reshape(m, m, 16)


def majority_u8(blocks: np.ndarray) -> np.ndarray:
    """Majority value per block; ties break to the HIGHER value (see module doc)."""
    values = np.unique(blocks)[::-1]  # descending -> argmax's first-max = highest value
    counts = np.stack([(blocks == v).sum(axis=-1) for v in values])
    return values[counts.argmax(axis=0)].astype(np.uint8)


def mean_u8(blocks: np.ndarray) -> np.ndarray:
    return np.rint(blocks.astype(np.float32).mean(axis=-1)).astype(np.uint8)


def weighted_mean_u8(value_blocks: np.ndarray, weight_blocks: np.ndarray) -> np.ndarray:
    """Weight-averaged value per block, 0 where all weights are 0 (e.g. canopy height
    weighted by cover, so treeless finer texels don't drag the mean toward 0)."""
    w = weight_blocks.astype(np.float32)
    wsum = w.sum(axis=-1)
    v = (value_blocks.astype(np.float32) * w).sum(axis=-1)
    out = np.divide(v, wsum, out=np.zeros_like(wsum), where=wsum > 0)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def wet_majority(blocks: np.ndarray) -> np.ndarray:
    """Water reduction: block is wet iff >= half its finer texels are wet (finite);
    level = mean of the wet levels. Ties lean wet — coastlines must read as water."""
    wet = np.isfinite(blocks)
    nwet = wet.sum(axis=-1)
    majority = nwet * 2 >= blocks.shape[-1]
    level = np.divide(
        np.nansum(blocks, axis=-1), nwet, out=np.zeros_like(nwet, dtype=np.float64),
        where=nwet > 0,
    )
    return np.where(majority & (nwet > 0), level, np.nan).astype(np.float32)
