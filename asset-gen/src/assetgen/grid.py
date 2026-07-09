"""World chunk-grid math. Pure functions, no IO — the one place coordinate conventions live.

Conventions (frozen; mirrored in the shipped manifest):
- CRS: L-EST97 / EPSG:3301, planar meters. E = easting, N = northing.
- Game axes: gameX = E - ANCHOR_E (east is +X); gameZ = ANCHOR_N - N (south is +Z);
  gameY = EH2000 height in meters. Raster row order == +Z, so row 0 is the chunk's
  northern edge — matching the renderer's heights[row*res+col] row=z convention.
- LOD k: texel = lod_step^k meters, chunk footprint = chunk_m * lod_step^k meters —
  a quadtree with 4x4 branching; every LOD's payload raster is (chunk_res+1)^2.
- Payload raster of chunk (cx, cz, lod): (chunk_res+1) x (chunk_res+1) texels of size t.
  Texel (col i, row j) is the source pixel CENTERED at
      E = e_min + (i + 0.5) * t,   N = n_max - (j + 0.5) * t.
  The extra far row/column (i or j == chunk_res) is an apron duplicating the east/south
  neighbor's first pixels, so bilinear sampling never crosses a fetch boundary.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import GridConfig


@dataclass(frozen=True)
class ChunkId:
    cx: int
    cz: int
    lod: int


def lod_texel(grid: GridConfig, lod: int) -> int:
    """Texel size in meters at this LOD."""
    return grid.lod_step**lod


def lod_footprint(grid: GridConfig, lod: int) -> int:
    """Chunk footprint (edge length) in meters at this LOD."""
    return grid.chunk_m * grid.lod_step**lod


def game_from_en(grid: GridConfig, e: float, n: float) -> tuple[float, float]:
    return e - grid.anchor_e, grid.anchor_n - n


def en_from_game(grid: GridConfig, x: float, z: float) -> tuple[float, float]:
    return x + grid.anchor_e, grid.anchor_n - z


def chunk_id_for_en(grid: GridConfig, e: float, n: float, lod: int) -> ChunkId:
    f = lod_footprint(grid, lod)
    x, z = game_from_en(grid, e, n)
    return ChunkId(int(x // f), int(z // f), lod)


def chunk_bounds_en(grid: GridConfig, c: ChunkId) -> tuple[int, int, int, int]:
    """Exact chunk footprint (e_min, n_min, e_max, n_max) — apron NOT included."""
    f = lod_footprint(grid, c.lod)
    e_min = grid.anchor_e + c.cx * f
    n_max = grid.anchor_n - c.cz * f
    return e_min, n_max - f, e_min + f, n_max


def chunk_raster_window_en(grid: GridConfig, c: ChunkId) -> tuple[int, int, int, int, int]:
    """Source-mosaic window the payload raster copies, apron INCLUDED.

    Returns (e_min, n_min, e_max, n_max, texel): (chunk_res+1) texels per axis,
    spanning [e_min, e_max) x (n_min, n_max] with e_max - e_min == footprint + texel.
    """
    t = lod_texel(grid, c.lod)
    e_min, n_min, e_max, n_max = chunk_bounds_en(grid, c)
    return e_min, n_min - t, e_max + t, n_max, t


def chunks_covering_bbox_en(
    grid: GridConfig, bbox_en: tuple[float, float, float, float], lod: int
) -> list[ChunkId]:
    """All chunks whose footprint intersects the (min_e, min_n, max_e, max_n) bbox.

    Row-major, north-to-south then west-to-east — deterministic cook order.
    """
    min_e, min_n, max_e, max_n = bbox_en
    f = lod_footprint(grid, lod)
    x0, z0 = game_from_en(grid, min_e, max_n)  # NW corner -> smallest x and z
    x1, z1 = game_from_en(grid, max_e, min_n)  # SE corner -> largest x and z
    cx0, cx1 = int(x0 // f), int((x1 - 1e-9) // f)
    cz0, cz1 = int(z0 // f), int((z1 - 1e-9) // f)
    return [ChunkId(cx, cz, lod) for cz in range(cz0, cz1 + 1) for cx in range(cx0, cx1 + 1)]


def snap_bbox_to_chunks_en(
    grid: GridConfig, bbox_en: tuple[float, float, float, float], lod: int = 0
) -> tuple[int, int, int, int]:
    """Smallest chunk-aligned bbox containing bbox_en (at the given LOD's footprint)."""
    ids = chunks_covering_bbox_en(grid, bbox_en, lod)
    bounds = [chunk_bounds_en(grid, c) for c in ids]
    return (
        min(b[0] for b in bounds),
        min(b[1] for b in bounds),
        max(b[2] for b in bounds),
        max(b[3] for b in bounds),
    )
