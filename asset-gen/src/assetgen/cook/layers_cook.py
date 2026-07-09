"""Vector-derived layer cooks (biome / water / soil) — 2 m/texel rasters at LOD0 chunks."""
from __future__ import annotations

import numpy as np

from ..config import DATA_IN, BaseConfig
from ..grid import ChunkId, chunk_bounds_en, chunks_covering_bbox_en
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


def _read_planes(base: BaseConfig, layer: str, c, nplanes: int) -> list | None:
    p = chunk_path(layer, c)
    if not p.exists():
        return None
    meta, payload = read_chunk(p)
    return decode_u8_planes(base.encode, payload, meta.res, nplanes)


def cook_understory(base: BaseConfig, bbox_en, log=print) -> None:
    from ..process.understory import rasterize_understory, unmapped_site_types

    chunks = chunks_covering_bbox_en(base.grid, bbox_en, 0)
    for i, c in enumerate(chunks):
        dest = chunk_path("understory", c)
        if dest.exists():
            continue
        biome = _read_planes(base, "biome", c, 2)
        soil = _read_planes(base, "soil", c, 5)
        if biome is None or soil is None:
            raise FileNotFoundError("understory needs biome + soil cooked first")
        planes = rasterize_understory(_window_2m(base, c), biome[0], biome[1], soil[4])
        payload = encode_u8_planes(base.encode, planes)
        write_chunk(dest, _meta(base, "understory", c, enc=2), payload)
        if (i + 1) % 16 == 0 or i + 1 == len(chunks):
            log(f"  understory [{i + 1}/{len(chunks)}]")
    if unmapped_site_types:
        log(f"  understory: unmapped site types {dict(sorted(unmapped_site_types.items(), key=lambda kv: -kv[1]))}")


def cook_debris(base: BaseConfig, bbox_en, log=print) -> None:
    from ..process.debris import rasterize_debris

    chunks = chunks_covering_bbox_en(base.grid, bbox_en, 0)
    for i, c in enumerate(chunks):
        dest = chunk_path("debris", c)
        if dest.exists():
            continue
        biome = _read_planes(base, "biome", c, 2)
        soil = _read_planes(base, "soil", c, 5)
        if biome is None or soil is None:
            raise FileNotFoundError("debris needs biome + soil cooked first")
        planes = rasterize_debris(_window_2m(base, c), biome[0], soil[3])
        payload = encode_u8_planes(base.encode, planes)
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
