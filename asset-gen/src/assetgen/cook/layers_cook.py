"""Vector-derived layer cooks (biome / water / soil) — 2 m/texel rasters at LOD0 chunks,
plus their coarse rungs (and the CHM-derived canopy layer) reduced via cook.pyramid."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..config import DATA_IN, BaseConfig
from ..grid import ChunkId, chunk_bounds_en, chunks_covering_bbox_en, lod_footprint
from ..process.landcover import BIOME_TEXEL, load_rules, rasterize_classes, veg_density
from ..process.mosaic import RasterStack, dem_sources
from ..process.soil import check_unknown_budget, rasterize_soil, unmapped_textures, unmapped_types
from ..process.water import bed_depth_field, rasterize_water
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


# --- country-wide coarse biome floor ---------------------------------------------------
# cook_biome REDUCES its coarse rungs from a fine LOD0 cooked only over the 16 km pilot,
# so beyond the pilot the far terrain has NO biome and renders as bare soil (#108). This
# mirrors the height country-floor: rasterize the whole-country ETAK landcover DIRECTLY at
# the coarse texel for the finest floor rung, then reduce the coarser rungs from it.
#
# vegDensity (biome plane 1 = bio.y) is the load-bearing far-colour plane — grassW/forestW
# scale by it, so "far = dirt" is really "far vegDensity = 0". The honest whole-country
# analog of the pilot's CHM canopy-cover fraction is the ETAK forest/shrub coverage
# fraction (forest = full canopy, shrub = half): both mean "share of ground under tree
# canopy", so near (CHM) and far (ETAK) stay consistent. Canopy HEIGHT has no whole-country
# source (CHM is pilot-only), so the canopy layer (treetop tint) stays pilot-only.
_BIOME_SUPERSAMPLE = 4  # sub-texels per coarse texel — MUST be 4 for blocks16's 4x4 blocks
_CANOPY_COVER = {"forest": 255, "shrub": 128}  # class -> vegDensity weight (canopy fraction)

_floor_base: BaseConfig | None = None
_floor_rules = None


def _biome_coarse_window(base: BaseConfig, c: ChunkId, sub: int):
    """Super-sampled ETAK window for a coarse biome chunk: `sub` sub-texels per coarse
    texel, one-coarse-texel east/south apron (mirrors _window_2m). Rasterizing at sub of
    the coarse texel and 4x4-block-reducing gives a smooth cover fraction, not a binary."""
    t = BIOME_TEXEL * base.grid.lod_step ** c.lod  # coarse biome texel (m)
    e_min, n_min, e_max, n_max = chunk_bounds_en(base.grid, c)
    return e_min, n_min - t, e_max + t, n_max, t / sub


def _biome_coarse_planes(base: BaseConfig, rules, c: ChunkId):
    """Rasterize one coarse biome chunk straight from ETAK -> (classId, vegDensity) 1025²
    u8 planes, or None where no ETAK polygon covers the chunk (open Baltic / off-country)."""
    window = _biome_coarse_window(base, c, _BIOME_SUPERSAMPLE)
    class_sub = rasterize_classes(rules, window)
    if not class_sub.any():
        return None
    veg_sub = np.zeros_like(class_sub)
    for name, weight in _CANOPY_COVER.items():
        veg_sub[class_sub == rules.palette[name]] = weight
    cls = majority_u8(blocks16(class_sub))      # dominant land-cover class per coarse texel
    dens = mean_u8(blocks16(veg_sub))           # canopy-cover fraction * 255 per coarse texel
    return cls, dens


def _init_floor_worker(base: BaseConfig, rules) -> None:
    global _floor_base, _floor_rules
    _floor_base, _floor_rules = base, rules


def _cook_biome_floor_one(c: ChunkId) -> int:
    """Worker: cook one coarse chunk straight from ETAK, OVERWRITING any existing chunk.
    Returns 0 where the chunk carries no ETAK land-cover (open Baltic / off-country).

    Overwrite, NOT gap-fill: the pilot's coarse rungs are REDUCED from a 16 km fine cook
    that fills only ~1/4 of a 32 km lod2 chunk, so the pilot's own lod2/3/4 are ~80% none
    (bare soil) over their footprint — preserving them would keep the dirt bug right around
    the pilot. The complete ETAK-direct rasterize replaces them; the near view is unaffected
    (it streams the finer, untouched lod0/lod1)."""
    base, rules = _floor_base, _floor_rules
    assert base is not None and rules is not None
    planes = _biome_coarse_planes(base, rules, c)
    if planes is None:
        return 0
    dest = chunk_path("biome", c)
    write_chunk(dest, _meta(base, "biome", c, enc=2), encode_u8_planes(base.encode, list(planes)))
    return dest.stat().st_size


def cook_biome_floor(base: BaseConfig, bbox_en, lods, workers: int = 6, log=print) -> None:
    """Whole-country coarse biome floor. Finest floor rung = direct ETAK rasterize over the
    whole country (overwriting the pilot's incomplete reduced rungs); coarser rungs reduce
    from it. Run AFTER the pilot cook — it leaves lod0/lod1 alone and completes lod2+."""
    rules = load_rules()
    n = _res_2m(base) - 1
    lods = sorted(lods)
    base_lod = lods[0]

    chunks = chunks_covering_bbox_en(base.grid, bbox_en, base_lod)
    log(f"  biome floor lod{base_lod}: {len(chunks)} chunks direct from ETAK ({workers} workers)")
    cooked = total = 0
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_floor_worker, initargs=(base, rules)
    ) as ex:
        for i, size in enumerate(ex.map(_cook_biome_floor_one, chunks, chunksize=2)):
            if size:
                cooked += 1
                total += size
            if (i + 1) % 8 == 0 or i + 1 == len(chunks):
                log(f"    [{i + 1}/{len(chunks)}] {cooked} cooked ({total / 1e6:.1f} MB)")
    log(f"  biome floor lod{base_lod}: {cooked}/{len(chunks)} chunks carry land-cover")

    for lod in lods[1:]:  # coarser rungs: majority classId + mean vegDensity, OVERWRITE
        done = 0
        for c in chunks_covering_bbox_en(base.grid, bbox_en, lod):
            big = assemble_finer(lambda f: _read_planes(base, "biome", f, 2), c, n, fills=[0, 0])
            if big is None:
                continue
            cls = majority_u8(blocks16(big[0]))
            dens = mean_u8(blocks16(big[1]))
            write_chunk(
                chunk_path("biome", c), _meta(base, "biome", c, enc=2),
                encode_u8_planes(base.encode, [cls, dens]),
            )
            done += 1
        log(f"  biome floor lod{lod}: {done} chunks reduced")


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


WATERBED_FLAG = 1  # ChunkMeta.flags bit: this height chunk has its bed carved under water


def cook_waterbed(base: BaseConfig, bbox_en, log=print) -> None:
    """POST-PASS (#104 depth + #114 shoreline): carve a submerged bed into the LOD0 height
    chunks under the water mask (height − depth) and emit the anti-aliased shore-coverage
    layer `watercover`.

    Run AFTER height + water are cooked. Touches ONLY under-water texels — so generated
    scatter (trees from nDSM, understory/debris/boulders from landcover; none read the DTM
    bed for COUNTS) is unaffected, and dry texels keep their DECODED height bit-for-bit
    (a shifted qoffset re-quantizes them to the identical value). Idempotent: carved chunks
    are tagged in the header (flags bit) and skipped on re-run, so the output is
    byte-deterministic. Pilot LOD0 only — near-camera water is where the flat surface
    z-fights the terrain; coarse far water is sub-pixel and its height stays pure DTM.
    """
    stack = RasterStack(dem_sources(DATA_IN))
    chunks = chunks_covering_bbox_en(base.grid, bbox_en, 0)
    n_carved = n_cov = 0
    for i, c in enumerate(chunks):
        hp = chunk_path("height", c)
        if not hp.exists():
            continue
        hmeta, hpayload = read_chunk(hp)
        if hmeta.flags & WATERBED_FLAG:
            n_carved += 1
            continue  # already carved — idempotent re-run
        depth2, cov2 = bed_depth_field(_window_2m(base, c), stack)

        if cov2.any():  # coverage (#114): emit wherever any water polygon touches the chunk
            cov = np.rint(np.clip(cov2, 0.0, 1.0) * 255.0).astype(np.uint8)
            write_chunk(chunk_path("watercover", c), _meta(base, "watercover", c, enc=2),
                        encode_u8_planes(base.encode, [cov]))
            n_cov += 1

        if float(depth2.max()) <= 0.0:
            continue  # coverage-only (e.g. sea, deferred) — nothing to carve

        # carve: subtract the 2 m depth (nearest-upsampled to the 1 m height grid) under water
        h = decode_quant16(base.encode, hpayload, hmeta.res, hmeta.qoffset, hmeta.qscale)
        d1 = np.repeat(np.repeat(depth2, 2, axis=0), 2, axis=1)[: hmeta.res, : hmeta.res]
        carved = (h - d1).astype(np.float64)
        qscale = hmeta.qscale
        payload, qoffset = encode_quant16(base.encode, carved, qscale)
        out = decode_quant16(base.encode, payload, hmeta.res, qoffset, qscale)
        err = float(np.max(np.abs(out - carved)))
        if err > qscale * 0.5 + 1e-3:
            raise AssertionError(f"waterbed round-trip {err} m on {c}")
        b = chunk_bounds_en(base.grid, c)
        write_chunk(hp, ChunkMeta(
            layer="height", lod=0, enc=1, cx=c.cx, cz=c.cz, res=hmeta.res, count=0,
            origin_e=b[0], origin_n=b[3], qoffset=qoffset, qscale=qscale, flags=WATERBED_FLAG,
        ), payload)
        n_carved += 1
        if (i + 1) % 16 == 0 or i + 1 == len(chunks):
            log(f"  waterbed [{i + 1}/{len(chunks)}] ({n_carved} carved, {n_cov} coverage)")

    # coverage LOD1: mean fraction per 4×4 block (mirrors biome's coarse-rung reduction)
    n = _res_2m(base) - 1
    done = 0
    for c in chunks_covering_bbox_en(base.grid, bbox_en, 1):
        big = assemble_finer(lambda f: _read_planes(base, "watercover", f, 1), c, n, fills=[0])
        if big is None:
            continue
        cov = mean_u8(blocks16(big[0]))
        if not cov.any():
            continue
        write_chunk(chunk_path("watercover", c), _meta(base, "watercover", c, enc=2),
                    encode_u8_planes(base.encode, [cov]))
        done += 1
    log(f"  waterbed: {n_carved} height chunks carved, {n_cov} coverage LOD0 + {done} LOD1")


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
