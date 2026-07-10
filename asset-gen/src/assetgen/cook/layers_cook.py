"""Vector-derived layer cooks (biome / water / soil) — 2 m/texel rasters at LOD0 chunks,
plus their coarse rungs (and the CHM-derived canopy layer) reduced via cook.pyramid."""
from __future__ import annotations

import numpy as np

from ..config import DATA_IN, BaseConfig
from ..grid import ChunkId, chunk_bounds_en, chunks_covering_bbox_en, lod_footprint
from ..process.landcover import BIOME_TEXEL, load_rules, rasterize_classes, veg_density
from ..process.mosaic import RasterStack, dem_sources
from ..process.soil import check_unknown_budget, rasterize_soil, unmapped_textures, unmapped_types
from ..process.water import rasterize_water
from .chunkio import ChunkMeta, read_chunk, write_chunk
from .encode import (
    decode_quant16,
    decode_u8_planes,
    encode_quant16,
    encode_records,
    encode_u8_planes,
)
from .height_cook import chunk_path
from .pyramid import assemble_finer, blocks16, majority_u8, mean_u8, weighted_mean_u8, wet_majority

CANOPY_MIN_M = 2.0  # CHM >= this counts as canopy — same threshold as veg_density


def _window_2m(base: BaseConfig, c: ChunkId) -> tuple[float, float, float, float, float]:
    e_min, n_min, e_max, n_max = chunk_bounds_en(base.grid, c)
    return e_min, n_min - BIOME_TEXEL, e_max + BIOME_TEXEL, n_max, BIOME_TEXEL


def _res_2m(base: BaseConfig) -> int:
    return base.grid.chunk_m // int(BIOME_TEXEL) + 1


def _meta(base: BaseConfig, layer: str, c: ChunkId, enc: int, qoffset=0.0, qscale=0.0) -> ChunkMeta:
    b = chunk_bounds_en(base.grid, c)
    return ChunkMeta(
        layer=layer, lod=c.lod, enc=enc, cx=c.cx, cz=c.cz, res=_res_2m(base), count=0,
        origin_e=b[0], origin_n=b[3], qoffset=qoffset, qscale=qscale,
    )


def cook_biome(base: BaseConfig, bbox_en, log=print) -> None:
    rules = load_rules()
    chunks = chunks_covering_bbox_en(base.grid, bbox_en, 0)
    for i, c in enumerate(chunks):
        dest = chunk_path("biome", c)
        if dest.exists():
            continue
        window = _window_2m(base, c)
        class_plane = rasterize_classes(rules, window)
        density_plane = veg_density(window)
        payload = encode_u8_planes(base.encode, [class_plane, density_plane])
        write_chunk(dest, _meta(base, "biome", c, enc=2), payload)
        if (i + 1) % 16 == 0 or i + 1 == len(chunks):
            log(f"  biome [{i + 1}/{len(chunks)}]")

    # coarse rungs: majority classId + mean vegDensity over the finer rung
    n = _res_2m(base) - 1
    for lod in (l for l in base.grid.lods if l >= 1):
        done = 0
        for c in chunks_covering_bbox_en(base.grid, bbox_en, lod):
            dest = chunk_path("biome", c)
            if dest.exists():
                done += 1
                continue
            big = assemble_finer(
                lambda f: _read_planes(base, "biome", f, 2), c, n, fills=[0, 0]
            )
            if big is None:
                continue  # no cooked finer data at all under this chunk
            cls = majority_u8(blocks16(big[0]))
            dens = mean_u8(blocks16(big[1]))
            payload = encode_u8_planes(base.encode, [cls, dens])
            write_chunk(dest, _meta(base, "biome", c, enc=2), payload)
            done += 1
        log(f"  biome lod{lod}: {done} chunks")


def cook_water(base: BaseConfig, bbox_en, log=print) -> None:
    stack = RasterStack(dem_sources(DATA_IN))
    chunks = chunks_covering_bbox_en(base.grid, bbox_en, 0)
    n_wet = 0
    for i, c in enumerate(chunks):
        dest = chunk_path("water", c)
        if dest.exists():
            n_wet += 1
            continue
        water = rasterize_water(_window_2m(base, c), stack)
        if water is None or not np.isfinite(water).any():
            continue  # all-dry chunk: omitted, manifest absence is authoritative
        qscale = base.encode.water_qscale
        # wet levels quantize normally; NaN (dry) lands on the reserved q=0 (qoffset)
        # value — client rule: q == 0 means dry, substitute own bed height - 2
        payload, qoffset = encode_quant16(base.encode, water.astype(np.float64), qscale)
        out = decode_quant16(base.encode, payload, water.shape[0], qoffset, qscale)
        wet = np.isfinite(water)
        assert float(np.max(np.abs(out[wet] - water[wet]))) <= qscale * 0.5 + 1e-3
        write_chunk(
            dest, _meta(base, "water", c, enc=1, qoffset=qoffset, qscale=qscale), payload
        )
        n_wet += 1
        if (i + 1) % 16 == 0 or i + 1 == len(chunks):
            log(f"  water [{i + 1}/{len(chunks)}] ({n_wet} wet)")
    log(f"  water: {n_wet}/{len(chunks)} chunks carry water")

    # LOD1: texel wet iff >= half its 16 finer texels are wet; level = their mean
    def read_wet(c: ChunkId) -> list[np.ndarray] | None:
        p = chunk_path("water", c)
        if not p.exists():
            return None  # absent chunk = all dry
        meta, payload = read_chunk(p)
        w = decode_quant16(base.encode, payload, meta.res, meta.qoffset, meta.qscale)
        w[w <= meta.qoffset + meta.qscale * 0.5] = np.nan  # reserved q=0 = dry
        return [w]

    n = _res_2m(base) - 1
    n_wet1 = 0
    for c in chunks_covering_bbox_en(base.grid, bbox_en, 1):
        dest = chunk_path("water", c)
        if dest.exists():
            n_wet1 += 1
            continue
        big = assemble_finer(read_wet, c, n, fills=[np.nan])
        if big is None:
            continue
        water = wet_majority(blocks16(big[0]))
        if not np.isfinite(water).any():
            continue  # majority-dry everywhere: omitted like LOD0
        qscale = base.encode.water_qscale
        payload, qoffset = encode_quant16(base.encode, water.astype(np.float64), qscale)
        out = decode_quant16(base.encode, payload, water.shape[0], qoffset, qscale)
        wet = np.isfinite(water)
        assert float(np.max(np.abs(out[wet] - water[wet]))) <= qscale * 0.5 + 1e-3
        write_chunk(
            dest, _meta(base, "water", c, enc=1, qoffset=qoffset, qscale=qscale), payload
        )
        n_wet1 += 1
    log(f"  water lod1: {n_wet1} wet chunks")


def cook_canopy(base: BaseConfig, bbox_en, log=print) -> None:
    """Far-forest canopy planes [heightM, cover] from the CHM — LODs 1..4 only (near
    canopy derives from tree records, so LOD0 is deliberately not cooked)."""
    chm_paths = sorted((DATA_IN / "chm").glob("*.tif"))
    if not chm_paths:
        raise FileNotFoundError("no CHM rasters — run `assetgen fetch --only elevation`")
    stack = RasterStack(chm_paths)
    n = _res_2m(base) - 1

    # LOD1 straight from the CHM at 2 m: 16 samples per 8 m texel
    done = 0
    for c in chunks_covering_bbox_en(base.grid, bbox_en, 1):
        dest = chunk_path("canopy", c)
        if dest.exists():
            done += 1
            continue
        e_min, n_min, e_max, n_max = chunk_bounds_en(base.grid, c)
        t = lod_footprint(base.grid, 1) // n  # coarse texel (8 m) = apron width
        chm = np.nan_to_num(
            stack.read_window(e_min, n_min - t, e_max + t, n_max, BIOME_TEXEL), nan=0.0
        )
        canopy = (chm >= CANOPY_MIN_M).astype(np.uint8)
        height = weighted_mean_u8(blocks16(np.clip(chm, 0, 255)), blocks16(canopy))
        cover = np.rint(blocks16(canopy).mean(axis=-1) * 255).astype(np.uint8)
        if not cover.any():
            continue  # no canopy (or no CHM coverage): absent chunk = no canopy data
        payload = encode_u8_planes(base.encode, [height, cover])
        write_chunk(dest, _meta(base, "canopy", c, enc=2), payload)
        done += 1
    log(f"  canopy lod1: {done} chunks")

    # LODs 2..4 reduce the previous rung (cover-weighted height, mean cover)
    for lod in (l for l in base.grid.lods if l >= 2):
        done = 0
        for c in chunks_covering_bbox_en(base.grid, bbox_en, lod):
            dest = chunk_path("canopy", c)
            if dest.exists():
                done += 1
                continue
            big = assemble_finer(
                lambda f: _read_planes(base, "canopy", f, 2), c, n, fills=[0, 0]
            )
            if big is None:
                continue
            hb, cb = blocks16(big[0]), blocks16(big[1])
            height, cover = weighted_mean_u8(hb, cb), mean_u8(cb)
            if not cover.any():
                continue
            payload = encode_u8_planes(base.encode, [height, cover])
            write_chunk(dest, _meta(base, "canopy", c, enc=2), payload)
            done += 1
        log(f"  canopy lod{lod}: {done} chunks")


def cook_soil(base: BaseConfig, bbox_en, log=print) -> None:
    chunks = chunks_covering_bbox_en(base.grid, bbox_en, 0)
    for i, c in enumerate(chunks):
        dest = chunk_path("soil", c)
        if dest.exists():
            continue
        planes = rasterize_soil(_window_2m(base, c))
        payload = encode_u8_planes(base.encode, planes)
        write_chunk(dest, _meta(base, "soil", c, enc=2), payload)
        if (i + 1) % 16 == 0 or i + 1 == len(chunks):
            log(f"  soil [{i + 1}/{len(chunks)}]")
    check_unknown_budget(len(chunks) * _res_2m(base) ** 2, log=log)
    if unmapped_types or unmapped_textures:
        log("  soil: unparseable codes present but under budget (censused above)")


def _read_planes(base: BaseConfig, layer: str, c, nplanes: int) -> list | None:
    p = chunk_path(layer, c)
    if not p.exists():
        return None
    meta, payload = read_chunk(p)
    return decode_u8_planes(base.encode, payload, meta.res, nplanes)


def _slope_2m(base: BaseConfig, c, res_2m: int) -> np.ndarray:
    """Slope (deg) on the 2 m grid from the cooked LOD0 height chunk."""
    from ..process.suitability import slope_deg_from_height

    p = chunk_path("height", c)
    if not p.exists():
        return np.zeros((res_2m, res_2m), dtype=np.float32)
    meta, payload = read_chunk(p)
    h = decode_quant16(base.encode, payload, meta.res, meta.qoffset, meta.qscale)
    return slope_deg_from_height(h, res_2m)


def cook_understory(base: BaseConfig, bbox_en, log=print) -> None:
    from ..process.fieldnoise import soften_field
    from ..process.suitability import (
        cell_richness,
        cell_wetness,
        community_targets,
        understory_suitability,
    )
    from ..process.understory import rasterize_understory, unmapped_site_types

    tw, tf = community_targets()
    res2 = _res_2m(base)
    chunks = chunks_covering_bbox_en(base.grid, bbox_en, 0)
    for i, c in enumerate(chunks):
        dest = chunk_path("understory", c)
        if dest.exists():
            continue
        biome = _read_planes(base, "biome", c, 2)
        soil = _read_planes(base, "soil", c, 5)
        if biome is None or soil is None:
            raise FileNotFoundError("understory needs biome + soil cooked first")
        window = _window_2m(base, c)
        community, density = rasterize_understory(window, biome[0], biome[1], soil[4])
        # ecological suitability: cut density by slope + soil wetness/richness + stoniness
        slope = _slope_2m(base, c, res2)
        wet = cell_wetness(soil[0], soil[1])
        rich = cell_richness(soil[4], soil[1])
        suit = understory_suitability(community, tw, tf, slope, wet, rich, soil[3])
        density = np.clip(density.astype(np.float32) * suit, 0, 255).astype(np.uint8)
        # fuzz the polygon edges + break up flat interiors (seamless across chunks)
        community, density = soften_field(community, density, window, seed=1101)
        payload = encode_u8_planes(base.encode, [community, density])
        write_chunk(dest, _meta(base, "understory", c, enc=2), payload)
        if (i + 1) % 16 == 0 or i + 1 == len(chunks):
            log(f"  understory [{i + 1}/{len(chunks)}]")
    if unmapped_site_types:
        log(f"  understory: unmapped site types {dict(sorted(unmapped_site_types.items(), key=lambda kv: -kv[1]))}")


def cook_debris(base: BaseConfig, bbox_en, log=print) -> None:
    from ..process.debris import rasterize_debris
    from ..process.fieldnoise import soften_field
    from ..process.suitability import debris_suitability

    res2 = _res_2m(base)
    chunks = chunks_covering_bbox_en(base.grid, bbox_en, 0)
    for i, c in enumerate(chunks):
        dest = chunk_path("debris", c)
        if dest.exists():
            continue
        biome = _read_planes(base, "biome", c, 2)
        soil = _read_planes(base, "soil", c, 5)
        if biome is None or soil is None:
            raise FileNotFoundError("debris needs biome + soil cooked first")
        window = _window_2m(base, c)
        surface, density = rasterize_debris(window, biome[0], soil[3])
        slope = _slope_2m(base, c, res2)
        suit = debris_suitability(surface, slope, soil[3])
        density = np.clip(density.astype(np.float32) * suit, 0, 255).astype(np.uint8)
        surface, density = soften_field(surface, density, window, seed=2202)
        payload = encode_u8_planes(base.encode, [surface, density])
        write_chunk(dest, _meta(base, "debris", c, enc=2), payload)
        if (i + 1) % 16 == 0 or i + 1 == len(chunks):
            log(f"  debris [{i + 1}/{len(chunks)}]")


def cook_boulders(base: BaseConfig, bbox_en, log=print) -> None:
    from ..process.boulders import boulders_for_chunk

    chunks = chunks_covering_bbox_en(base.grid, bbox_en, 0)
    total = 0
    for i, c in enumerate(chunks):
        dest = chunk_path("boulders", c)
        if dest.exists():
            continue
        cols = boulders_for_chunk(chunk_bounds_en(base.grid, c), base.grid.chunk_m)
        if len(cols) == 0:
            continue  # absent chunk = no mapped boulders
        columns = [cols.x, cols.z, cols.kind, cols.size, cols.variant]
        payload = encode_records(base.encode, columns)
        b = chunk_bounds_en(base.grid, c)
        meta = ChunkMeta(
            layer="boulders", lod=0, enc=3, cx=c.cx, cz=c.cz, res=0, count=len(cols),
            origin_e=b[0], origin_n=b[3], qoffset=0.0, qscale=base.grid.chunk_m / 65535.0,
        )
        write_chunk(dest, meta, payload)
        total += len(cols)
    log(f"  boulders: {total} mapped boulders across AOI")
