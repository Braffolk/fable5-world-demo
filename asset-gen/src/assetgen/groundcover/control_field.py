"""Ground-cover control-field v1.

This first carrier turns the already cooked, spatially correlated understory
community field into the generic two-type contract consumed by the O(1) raycast
lane. It deliberately does not claim to be the final >=10 native-species facies
cook: type ids are structural cover classes, not unverified species names.
"""
from __future__ import annotations

from enum import IntEnum

import numpy as np
from scipy import ndimage

from ..config import BaseConfig
from ..cook.chunkio import ChunkMeta, read_chunk, write_chunk
from ..cook.encode import decode_u8_planes, encode_u8_planes
from ..cook.height_cook import chunk_path
from ..grid import ChunkId, chunk_bounds_en, chunks_covering_bbox_en
from ..process.fieldnoise import value_noise_global
from ..process.landcover import BIOME_TEXEL, veg_density
from ..process.suitability import cell_wetness


class GroundCoverId(IntEnum):
    """Stable runtime ids. Values 0..63 fit the procedural body-id contract."""

    # GRASS=0 is a migration law: legacy procedural body ids have zero low bits,
    # so carrying today's grass through the new contract is payload-identical.
    GRASS = 0
    MOSS = 1
    SEDGE = 2
    LICHEN = 3
    FORB = 4
    DWARF_SHRUB = 5
    BARE = 63


# Existing cooked community id -> (dominant cover, subdominant cover, sub fraction).
# These are functional forms only. Species enter later through the native-species baker.
_COMMUNITY_MIX: dict[int, tuple[GroundCoverId, GroundCoverId, int]] = {
    0: (GroundCoverId.BARE, GroundCoverId.BARE, 0),
    1: (GroundCoverId.LICHEN, GroundCoverId.DWARF_SHRUB, 118),
    2: (GroundCoverId.MOSS, GroundCoverId.DWARF_SHRUB, 108),
    3: (GroundCoverId.FORB, GroundCoverId.MOSS, 74),
    4: (GroundCoverId.FORB, GroundCoverId.GRASS, 82),
    5: (GroundCoverId.MOSS, GroundCoverId.SEDGE, 88),
    6: (GroundCoverId.SEDGE, GroundCoverId.MOSS, 96),
    7: (GroundCoverId.GRASS, GroundCoverId.FORB, 86),
    8: (GroundCoverId.FORB, GroundCoverId.GRASS, 92),
    9: (GroundCoverId.GRASS, GroundCoverId.FORB, 70),
}


def _grid_coords(window_en: tuple[float, float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    e_min, n_min, e_max, n_max, texel = window_en
    rows = round((n_max - n_min) / texel)
    cols = round((e_max - e_min) / texel)
    east = e_min + (np.arange(cols, dtype=np.float64)[None, :] + 0.5) * texel
    north = n_max - (np.arange(rows, dtype=np.float64)[:, None] + 0.5) * texel
    return np.broadcast_to(east, (rows, cols)), np.broadcast_to(north, (rows, cols))


def _clump_ids(east: np.ndarray, north: np.ndarray) -> np.ndarray:
    """World-seamless, domain-warped patch identity (not a runtime hash)."""
    warp_x = (value_noise_global(east, north, 46.0, 0x41A7) - 0.5) * 18.0
    warp_n = (value_noise_global(east, north, 53.0, 0x72D3) - 0.5) * 18.0
    cx = np.floor((east + warp_x) / 11.0).astype(np.int64)
    cn = np.floor((north + warp_n) / 11.0).astype(np.int64)
    # uint64 arithmetic gives a deterministic 16-bit id without signed overflow.
    h = (cx.astype(np.uint64) * np.uint64(0x9E3779B185EBCA87)) ^ (
        cn.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)
    )
    h ^= h >> np.uint64(29)
    h *= np.uint64(0x165667B19E3779F9)
    h ^= h >> np.uint64(32)
    return (h & np.uint64(0xFFFF)).astype(np.uint16)


def derive_control_planes(
    community: np.ndarray,
    density: np.ndarray,
    soil_type: np.ndarray,
    tex_core: np.ndarray,
    canopy_cover: np.ndarray,
    window_en: tuple[float, float, float, float, float],
) -> list[np.ndarray]:
    """Return the eight u8 planes in the manifest's frozen order."""
    arrays = (community, density, soil_type, tex_core, canopy_cover)
    if len({a.shape for a in arrays}) != 1:
        raise ValueError("groundcover inputs must share one 2 m lattice")
    if community.dtype != np.uint8 or density.dtype != np.uint8:
        raise ValueError("groundcover community/density must be u8")

    type_a_lut = np.zeros(256, dtype=np.uint8)
    type_b_lut = np.zeros(256, dtype=np.uint8)
    blend_lut = np.zeros(256, dtype=np.uint8)
    for cid, (a, b, blend) in _COMMUNITY_MIX.items():
        type_a_lut[cid] = int(a)
        type_b_lut[cid] = int(b)
        blend_lut[cid] = blend
    unknown = ~np.isin(community, np.fromiter(_COMMUNITY_MIX, dtype=np.uint8))
    if unknown.any():
        ids = np.unique(community[unknown]).tolist()
        raise ValueError(f"groundcover has unmapped community ids {ids}")

    type_a = type_a_lut[community]
    type_b = type_b_lut[community]
    blend = blend_lut[community]
    vigor = density.copy()
    blend[vigor == 0] = 0
    type_a[vigor == 0] = int(GroundCoverId.BARE)
    type_b[vigor == 0] = int(GroundCoverId.BARE)

    wet = cell_wetness(soil_type, tex_core)
    wet = np.where(np.isin(community, [5, 6]), np.maximum(wet, 0.9), wet)
    moisture = np.rint(np.clip(wet, 0.0, 1.0) * 255.0).astype(np.uint8)

    canopy = canopy_cover.astype(np.float32) / 255.0
    mask = canopy_cover > 0
    if mask.any():
        distance_m = ndimage.distance_transform_edt(~mask) * float(window_en[4])
        proximity = np.maximum(canopy, np.exp(-distance_m / 12.0) * 0.82)
    else:
        proximity = canopy
    canopy_proximity = np.rint(np.clip(proximity, 0.0, 1.0) * 255.0).astype(np.uint8)

    east, north = _grid_coords(window_en)
    clump = _clump_ids(east, north)
    clump_lo = (clump & np.uint16(0xFF)).astype(np.uint8)
    clump_hi = (clump >> np.uint16(8)).astype(np.uint8)
    return [type_a, type_b, clump_lo, clump_hi, blend, vigor, moisture, canopy_proximity]


def _window_2m(base: BaseConfig, chunk: ChunkId) -> tuple[float, float, float, float, float]:
    e_min, n_min, e_max, n_max = chunk_bounds_en(base.grid, chunk)
    return e_min, n_min - BIOME_TEXEL, e_max + BIOME_TEXEL, n_max, BIOME_TEXEL


def _read_planes(base: BaseConfig, layer: str, chunk: ChunkId, count: int) -> list[np.ndarray] | None:
    path = chunk_path(layer, chunk)
    if not path.exists():
        return None
    meta, payload = read_chunk(path)
    return decode_u8_planes(base.encode, payload, meta.res, count)


def cook_groundcover(base: BaseConfig, bbox_en, log=print) -> None:
    """Cook LOD0 control chunks from already-cooked ecology plus CHM evidence."""
    chunks = chunks_covering_bbox_en(base.grid, bbox_en, 0)
    cooked = 0
    for i, chunk in enumerate(chunks):
        dest = chunk_path("groundcover", chunk)
        if dest.exists():
            cooked += 1
            continue
        under = _read_planes(base, "understory", chunk, 2)
        soil = _read_planes(base, "soil", chunk, 5)
        if under is None or soil is None:
            raise FileNotFoundError("groundcover needs understory + soil cooked first")
        window = _window_2m(base, chunk)
        canopy = veg_density(window)
        planes = derive_control_planes(under[0], under[1], soil[0], soil[1], canopy, window)
        bounds = chunk_bounds_en(base.grid, chunk)
        meta = ChunkMeta(
            layer="groundcover",
            lod=0,
            enc=2,
            cx=chunk.cx,
            cz=chunk.cz,
            res=planes[0].shape[0],
            count=0,
            origin_e=bounds[0],
            origin_n=bounds[3],
            qoffset=0.0,
            qscale=0.0,
        )
        write_chunk(dest, meta, encode_u8_planes(base.encode, planes))
        cooked += 1
        if (i + 1) % 8 == 0 or i + 1 == len(chunks):
            log(f"  groundcover [{i + 1}/{len(chunks)}]")
    log(f"  groundcover: {cooked}/{len(chunks)} control chunks")
