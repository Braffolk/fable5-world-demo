"""Exact geometry for signed physical height LODs.

The shared WorldGrid remains nonnegative and unchanged. Height format 2 extends it
with LOD -2/-1 using integer 1/32 m units, which exactly represent every sample
center from 0.0625 m through the established coarse rungs.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import GridConfig

HEIGHT_LOD_MIN = -2
HEIGHT_LOD_MAX = 4
UNITS_PER_METER = 32


@dataclass(frozen=True, order=True)
class HeightChunkId:
    lod: int
    cx: int
    cz: int


@dataclass(frozen=True)
class HeroCoverage:
    parent: HeightChunkId
    published_fine: tuple[HeightChunkId, ...]
    transient_support: tuple[HeightChunkId, ...]
    authority_lod0: HeightChunkId


def _check_lod(lod: int) -> None:
    if not HEIGHT_LOD_MIN <= lod <= HEIGHT_LOD_MAX:
        raise ValueError(f"height LOD {lod} outside {HEIGHT_LOD_MIN}..{HEIGHT_LOD_MAX}")


def texel_units(lod: int) -> int:
    _check_lod(lod)
    if lod >= 0:
        return UNITS_PER_METER * 4**lod
    divisor = 4 ** (-lod)
    if UNITS_PER_METER % divisor:
        raise ValueError(f"LOD {lod} is not representable in 1/{UNITS_PER_METER} m units")
    return UNITS_PER_METER // divisor


def footprint_units(grid: GridConfig, lod: int) -> int:
    if (grid.chunk_m, grid.chunk_res, grid.lod_step) != (2048, 2048, 4):
        raise ValueError("signed height geometry requires the frozen (2048,2048,4) lattice")
    return grid.chunk_m * texel_units(lod)


def texel_m(lod: int) -> float:
    return texel_units(lod) / UNITS_PER_METER


def footprint_m(grid: GridConfig, lod: int) -> float:
    return footprint_units(grid, lod) / UNITS_PER_METER


def chunk_origin_en_units(grid: GridConfig, chunk: HeightChunkId) -> tuple[int, int]:
    footprint = footprint_units(grid, chunk.lod)
    return (
        grid.anchor_e * UNITS_PER_METER + chunk.cx * footprint,
        grid.anchor_n * UNITS_PER_METER - chunk.cz * footprint,
    )


def sample_center_en_units(
    grid: GridConfig,
    chunk: HeightChunkId,
    col: int,
    row: int,
) -> tuple[int, int]:
    if not 0 <= col <= grid.chunk_res or not 0 <= row <= grid.chunk_res:
        raise IndexError(f"sample {(col, row)} outside 0..{grid.chunk_res}")
    texel = texel_units(chunk.lod)
    if texel % 2:
        raise AssertionError("height sample center is not integral in geometry units")
    origin_e, origin_n = chunk_origin_en_units(grid, chunk)
    half = texel // 2
    return origin_e + col * texel + half, origin_n - row * texel - half


def parent_of(chunk: HeightChunkId) -> HeightChunkId:
    if chunk.lod >= HEIGHT_LOD_MAX:
        raise ValueError(f"LOD {chunk.lod} has no supported parent")
    return HeightChunkId(chunk.lod + 1, chunk.cx // 4, chunk.cz // 4)


def children_of(parent: HeightChunkId) -> tuple[HeightChunkId, ...]:
    if parent.lod <= HEIGHT_LOD_MIN:
        raise ValueError(f"LOD {parent.lod} has no supported children")
    lod = parent.lod - 1
    return tuple(
        HeightChunkId(lod, parent.cx * 4 + dx, parent.cz * 4 + dz)
        for dz in range(4)
        for dx in range(4)
    )


def plan_hero(parent_cx: int, parent_cz: int) -> HeroCoverage:
    """One published -1 parent, its 16 children, and decoded apron support.

    The support set is the east child column, south child row, and southeast
    corner. These nine transient chunks supply the four fine samples required by
    each outer parent apron sample after quantization/decode.
    """
    parent = HeightChunkId(-1, parent_cx, parent_cz)
    children = children_of(parent)
    fine_cx0 = parent_cx * 4
    fine_cz0 = parent_cz * 4
    support = tuple(
        [HeightChunkId(-2, fine_cx0 + 4, fine_cz0 + dz) for dz in range(4)]
        + [HeightChunkId(-2, fine_cx0 + dx, fine_cz0 + 4) for dx in range(4)]
        + [HeightChunkId(-2, fine_cx0 + 4, fine_cz0 + 4)]
    )
    return HeroCoverage(parent, children, support, parent_of(parent))
