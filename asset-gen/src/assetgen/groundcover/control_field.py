"""Ground-cover control-field v2.

The frozen type bytes carry six functional response classes. The appended v2
profile bytes and root-reach closure mask carry the exact, provenance-backed
native palette independently, so two species of one functional class never
collapse to the same procedural identity.
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


class GroundCoverProfileId(IntEnum):
    """Exact zero-based native palette shared with the runtime atlas."""

    AGROSTIS_CAPILLARIS = 0
    AVENELLA_FLEXUOSA = 1
    CALAMAGROSTIS_CANESCENS = 2
    CAREX_CESPITOSA = 3
    ERIOPHORUM_VAGINATUM = 4
    SPHAGNUM_CAPILLIFOLIUM = 5
    PLEUROZIUM_SCHREBERI = 6
    CLADONIA_RANGIFERINA = 7
    OXALIS_ACETOSELLA = 8
    MAIANTHEMUM_BIFOLIUM = 9
    VACCINIUM_MYRTILLUS = 10
    CALLUNA_VULGARIS = 11


# Existing cooked community id -> (dominant cover, subdominant cover, sub fraction).
# These functional bytes are the frozen v1 carrier. Exact native species are selected
# independently in the appended v2 profile bytes below.
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


def _clump_unit(clump: np.ndarray, salt: int) -> np.ndarray:
    """Stable decorrelated [0,1] variate for an already-coherent clump."""
    h = clump.astype(np.uint32) ^ np.uint32(salt)
    h ^= h >> np.uint32(7)
    h *= np.uint32(0x45D9F3B)
    h ^= h >> np.uint32(11)
    return (h & np.uint32(0xFFFF)).astype(np.float32) / np.float32(0xFFFF)


def _facies_latents(
    east: np.ndarray,
    north: np.ndarray,
    clump: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Two world-seamless correlated truncation fields for exact-species patches.

    Broad value-noise fields prevent per-cell species salt-and-pepper. A small
    decorrelated clump term lets the existing domain-warped 11 m patch identity
    sharpen local facies without making chunk identity or worker order an input.
    """
    broad_a = value_noise_global(east, north, 41.0, 0x2D51)
    fine_a = value_noise_global(east, north, 17.0, 0x63B9)
    broad_b = value_noise_global(east, north, 47.0, 0x78C5)
    fine_b = value_noise_global(east, north, 19.0, 0x19E7)
    a = 0.58 * broad_a + 0.30 * fine_a + 0.12 * _clump_unit(clump, 0xA531)
    b = 0.58 * broad_b + 0.30 * fine_b + 0.12 * _clump_unit(clump, 0xC2B7)
    return a.astype(np.float32), b.astype(np.float32)


def _select_profiles(
    cover: np.ndarray,
    community: np.ndarray,
    vigor: np.ndarray,
    wetness: np.ndarray,
    tex_core: np.ndarray,
    canopy_cover: np.ndarray,
    facies_a: np.ndarray,
    facies_b: np.ndarray,
) -> np.ndarray:
    """Choose an exact native profile inside each frozen functional class.

    Community remains the strongest authority. The continuous terms only split
    ecologically compatible members of that class into coherent facies:

    - open mesic grass / dry acidic forest grass / wet reed-grass;
    - minerotrophic sedge / raised-bog cotton-grass;
    - wet Sphagnum / mineral-soil feather moss;
    - wood sorrel / may lily, and bilberry / heather.

    Density gates live cover and also distinguishes sparse poor-ground facies;
    it never thins runtime geometry or alters the frozen v1 type bytes.
    """
    profile = np.full(cover.shape, 0xFF, dtype=np.uint8)
    live = (vigor > 0) & (cover != int(GroundCoverId.BARE))
    vigor_f = vigor.astype(np.float32) / 255.0
    canopy = canopy_cover.astype(np.float32) / 255.0
    peat = np.isin(tex_core, [9, 10, 11, 12, 13, 14]).astype(np.float32)
    sand = np.isin(tex_core, [1, 2, 3]).astype(np.float32)

    grass = live & (cover == int(GroundCoverId.GRASS))
    profile[grass] = int(GroundCoverProfileId.AGROSTIS_CAPILLARIS)
    calamagrostis_score = (
        0.58 * wetness
        + 0.22 * peat
        + 0.12 * vigor_f
        + 0.16 * facies_b
        + 0.22 * (community == 8)
        + 0.08 * (community == 4)
        - 0.32 * (community == 7)
    )
    calamagrostis = grass & (calamagrostis_score >= 0.73)
    profile[calamagrostis] = int(GroundCoverProfileId.CALAMAGROSTIS_CANESCENS)
    avenella_score = (
        0.50 * (1.0 - wetness)
        + 0.22 * sand
        + 0.14 * canopy
        + 0.14 * (1.0 - vigor_f)
        + 0.16 * facies_a
        + 0.18 * (community == 9)
        - 0.38 * (community == 7)
        - 0.35 * (community == 8)
    )
    acid_grassland = (community == 7) & (sand > 0) & (wetness <= 0.40)
    avenella = (
        grass
        & ~calamagrostis
        & ((community == 9) | acid_grassland)
        & (avenella_score >= 0.74)
    )
    profile[avenella] = int(GroundCoverProfileId.AVENELLA_FLEXUOSA)

    moss = live & (cover == int(GroundCoverId.MOSS))
    profile[moss] = int(GroundCoverProfileId.PLEUROZIUM_SCHREBERI)
    sphagnum_score = (
        0.62 * wetness
        + 0.24 * peat
        + 0.30 * np.isin(community, [5, 6])
        + 0.10 * facies_b
    )
    profile[moss & (sphagnum_score >= 0.72)] = int(GroundCoverProfileId.SPHAGNUM_CAPILLIFOLIUM)

    sedge = live & (cover == int(GroundCoverId.SEDGE))
    profile[sedge] = int(GroundCoverProfileId.CAREX_CESPITOSA)
    eriophorum_score = (
        0.30 * (community == 5)
        + 0.30 * peat
        + 0.22 * wetness
        + 0.16 * facies_a
        - 0.15 * (community == 6)
    )
    profile[sedge & (eriophorum_score >= 0.55)] = int(GroundCoverProfileId.ERIOPHORUM_VAGINATUM)

    lichen = live & (cover == int(GroundCoverId.LICHEN))
    profile[lichen] = int(GroundCoverProfileId.CLADONIA_RANGIFERINA)

    forb = live & (cover == int(GroundCoverId.FORB))
    profile[forb] = int(GroundCoverProfileId.OXALIS_ACETOSELLA)
    maianthemum_score = (
        0.32 * facies_b
        + 0.22 * canopy
        + 0.14 * (1.0 - vigor_f)
        + 0.18 * (community == 3)
        + 0.16 * np.isin(community, [8, 9])
        - 0.28 * (community == 7)
        - 0.12 * (community == 4)
    )
    profile[forb & (maianthemum_score >= 0.61)] = int(GroundCoverProfileId.MAIANTHEMUM_BIFOLIUM)

    shrub = live & (cover == int(GroundCoverId.DWARF_SHRUB))
    profile[shrub] = int(GroundCoverProfileId.VACCINIUM_MYRTILLUS)
    calluna_score = (
        0.30 * (community == 1)
        + 0.28 * (community == 5)
        + 0.18 * (1.0 - wetness)
        + 0.18 * (1.0 - canopy)
        + 0.10 * peat
        + 0.16 * facies_a
        - 0.30 * (community == 2)
    )
    calluna = shrub & ((community == 1) | (calluna_score >= 0.58))
    profile[calluna] = int(GroundCoverProfileId.CALLUNA_VULGARIS)
    return profile


def derive_control_planes(
    community: np.ndarray,
    density: np.ndarray,
    soil_type: np.ndarray,
    tex_core: np.ndarray,
    canopy_cover: np.ndarray,
    window_en: tuple[float, float, float, float, float],
) -> list[np.ndarray]:
    """Return the twelve u8 planes in the manifest's versioned order.

    The first eight v1 planes remain byte-for-byte frozen. v2 appends a
    conservative 4 m query-closure mask plus exact profile A/B ids. At chunk
    rims, where this cook invocation cannot see the adjacent chunk's interior,
    the mask is deliberately all twelve native profiles: bounded over-query is
    correct; silently omitting a possible overhanging root is not.
    """
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
    facies_a, facies_b = _facies_latents(east, north, clump)
    profile_a = _select_profiles(
        type_a, community, vigor, wet, tex_core, canopy_cover, facies_a, facies_b
    )
    profile_b = _select_profiles(
        type_b, community, vigor, wet, tex_core, canopy_cover, facies_a, facies_b
    )

    # O's profile pair is not necessarily the pair at a profile root up to 4 m
    # away. Encode the complete bounded candidate union once at cook time. The
    # 2 m carrier uses a 5x5 maximum filter (radius two texels = 4 m).
    candidate_mask = np.zeros(community.shape, dtype=np.uint16)
    live = vigor > 0
    for profile_id in range(len(GroundCoverProfileId)):
        present = live & ((profile_a == profile_id) | (profile_b == profile_id))
        nearby = ndimage.maximum_filter(present, size=5, mode="constant", cval=0) > 0
        candidate_mask |= nearby.astype(np.uint16) << np.uint16(profile_id)
    # The source ecology chunk has only its traditional one-sample shared apron;
    # two closure samples are required. All-types at the narrow rim is the
    # deterministic seam-safe fallback until the neighbor is assembled.
    candidate_mask[:2, :] = np.uint16(0x0FFF)
    candidate_mask[-2:, :] = np.uint16(0x0FFF)
    candidate_mask[:, :2] = np.uint16(0x0FFF)
    candidate_mask[:, -2:] = np.uint16(0x0FFF)
    candidate_mask[~live] = np.uint16(0)
    candidate_mask_lo = (candidate_mask & np.uint16(0xFF)).astype(np.uint8)
    candidate_mask_hi = (candidate_mask >> np.uint16(8)).astype(np.uint8)

    return [
        type_a, type_b, clump_lo, clump_hi,
        blend, vigor, moisture, canopy_proximity,
        candidate_mask_lo, candidate_mask_hi, profile_a, profile_b,
    ]


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
