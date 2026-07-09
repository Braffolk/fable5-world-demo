"""Vector-derived layer cooks (biome / water / soil) — 2 m/texel rasters at LOD0 chunks."""
from __future__ import annotations

import numpy as np

from ..config import DATA_IN, BaseConfig
from ..grid import ChunkId, chunk_bounds_en, chunks_covering_bbox_en
from ..process.landcover import BIOME_TEXEL, load_rules, rasterize_classes, veg_density
from ..process.mosaic import RasterStack, dem_sources
from ..process.soil import check_unknown_budget, rasterize_soil, unmapped_textures, unmapped_types
from ..process.water import rasterize_water
from .chunkio import ChunkMeta, write_chunk
from .encode import decode_quant16, encode_quant16, encode_u8_planes
from .height_cook import chunk_path


def _window_2m(base: BaseConfig, c: ChunkId) -> tuple[float, float, float, float, float]:
    e_min, n_min, e_max, n_max = chunk_bounds_en(base.grid, c)
    return e_min, n_min - BIOME_TEXEL, e_max + BIOME_TEXEL, n_max, BIOME_TEXEL


def _res_2m(base: BaseConfig) -> int:
    return base.grid.chunk_m // int(BIOME_TEXEL) + 1


def _meta(base: BaseConfig, layer: str, c: ChunkId, enc: int, qoffset=0.0, qscale=0.0) -> ChunkMeta:
    b = chunk_bounds_en(base.grid, c)
    return ChunkMeta(
        layer=layer, lod=0, enc=enc, cx=c.cx, cz=c.cz, res=_res_2m(base), count=0,
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
