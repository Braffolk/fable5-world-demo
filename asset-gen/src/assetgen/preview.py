"""QA preview PNGs from cooked chunks — the human eyeball gate before any S3 sync."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .config import DATA_OUT, DATA_WORK, BaseConfig
from .cook.chunkio import read_chunk
from .cook.encode import decode_quant16
from .grid import chunks_covering_bbox_en

PREVIEW_DIR = DATA_OUT / "preview"


def _assemble_height(base: BaseConfig, bbox_en, lod: int) -> tuple[np.ndarray, int] | None:
    """Mosaic decoded height chunks (apron dropped) into one array. Returns (arr, texel)."""
    ids = chunks_covering_bbox_en(base.grid, bbox_en, lod)
    res = base.grid.chunk_res
    xs = sorted({c.cx for c in ids})
    zs = sorted({c.cz for c in ids})
    full = np.full((len(zs) * res, len(xs) * res), np.nan, dtype=np.float32)
    found = 0
    for c in ids:
        p = DATA_WORK / "chunks" / "height" / str(lod) / f"{c.cx}_{c.cz}.lac"
        if not p.exists():
            continue
        meta, payload = read_chunk(p)
        arr = decode_quant16(base.encode, payload, meta.res, meta.qoffset, meta.qscale)
        r, q = zs.index(c.cz) * res, xs.index(c.cx) * res
        full[r : r + res, q : q + res] = arr[:res, :res]
        found += 1
    if not found:
        return None
    return full, base.grid.lod_step**lod


def hillshade(arr: np.ndarray, texel: float, az_deg=315.0, alt_deg=45.0) -> np.ndarray:
    dz_dy, dz_dx = np.gradient(np.nan_to_num(arr, nan=0.0), texel)
    slope = np.arctan(np.hypot(dz_dx, dz_dy))
    aspect = np.arctan2(-dz_dx, dz_dy)
    az, alt = np.radians(az_deg), np.radians(alt_deg)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return (np.clip(shade, 0, 1) * 255).astype(np.uint8)


CLASS_COLORS = {
    0: (30, 30, 30), 1: (34, 102, 51), 2: (96, 140, 70), 3: (140, 180, 90),
    4: (168, 160, 130), 5: (222, 205, 150), 6: (196, 176, 105), 7: (170, 150, 120),
    8: (150, 110, 140), 9: (110, 130, 120), 10: (90, 70, 60), 11: (60, 120, 190),
    12: (80, 150, 210), 13: (40, 90, 160),
}


def _assemble_u8(base: BaseConfig, layer: str, bbox_en, plane: int, nplanes: int) -> np.ndarray | None:
    from .cook.encode import decode_u8_planes

    ids = chunks_covering_bbox_en(base.grid, bbox_en, 0)
    res = base.grid.chunk_m // 2  # 2 m layers, apron dropped
    xs = sorted({c.cx for c in ids})
    zs = sorted({c.cz for c in ids})
    full = np.zeros((len(zs) * res, len(xs) * res), dtype=np.uint8)
    found = 0
    for c in ids:
        p = DATA_WORK / "chunks" / layer / "0" / f"{c.cx}_{c.cz}.lac"
        if not p.exists():
            continue
        meta, payload = read_chunk(p)
        arr = decode_u8_planes(base.encode, payload, meta.res, nplanes)[plane]
        r, q = zs.index(c.cz) * res, xs.index(c.cx) * res
        full[r : r + res, q : q + res] = arr[:res, :res]
        found += 1
    return full if found else None


def _hash_palette(ids: np.ndarray) -> np.ndarray:
    """Deterministic distinct-ish color per id (for soil-type style many-class rasters)."""
    h = (ids.astype(np.uint32) * 2654435761) & 0xFFFFFFFF
    rgb = np.stack([(h >> s) & 0xFF for s in (0, 8, 16)], axis=-1).astype(np.uint8)
    rgb[ids == 0] = (25, 25, 25)
    return rgb


def preview_layers(base: BaseConfig, aoi_name: str, bbox_en, log=print) -> list[Path]:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    classes = _assemble_u8(base, "biome", bbox_en, 0, 2)
    if classes is not None:
        rgb = np.zeros((*classes.shape, 3), dtype=np.uint8)
        for cid, color in CLASS_COLORS.items():
            rgb[classes == cid] = color
        p = PREVIEW_DIR / f"{aoi_name}-landcover.png"
        Image.fromarray(rgb[::2, ::2], "RGB").save(p, optimize=True)
        written.append(p)
        log(f"  {p.name}: classes present {sorted(int(v) for v in np.unique(classes))}")
    density = _assemble_u8(base, "biome", bbox_en, 1, 2)
    if density is not None:
        p = PREVIEW_DIR / f"{aoi_name}-vegdensity.png"
        Image.fromarray(density[::2, ::2], "L").save(p, optimize=True)
        written.append(p)
    soil = _assemble_u8(base, "soil", bbox_en, 0, 5)
    if soil is not None:
        p = PREVIEW_DIR / f"{aoi_name}-soiltypes.png"
        Image.fromarray(_hash_palette(soil)[::2, ::2], "RGB").save(p, optimize=True)
        written.append(p)
        log(f"  {p.name}: {len(np.unique(soil))} distinct soil types in AOI")
    got = _assemble_height(base, bbox_en, 0)
    wet = _assemble_wet_mask(base, bbox_en)
    if wet is not None and got is not None:
        img = hillshade(got[0], 1.0)[::2, ::2][: wet.shape[0], : wet.shape[1]].copy()
        rgb = np.stack([img, img, img], axis=-1)
        rgb[wet] = (60, 130, 200)
        p = PREVIEW_DIR / f"{aoi_name}-water.png"
        Image.fromarray(rgb[::2, ::2], "RGB").save(p, optimize=True)
        written.append(p)
        log(f"  {p.name}: {wet.mean():.2%} of AOI is wet")
    return written


SPECIES_COLORS = {
    0: (20, 80, 40), 1: (10, 60, 30), 2: (30, 90, 60), 3: (40, 110, 70), 4: (60, 120, 50),
    5: (150, 200, 90), 6: (120, 170, 110), 7: (200, 210, 120), 8: (180, 200, 130),
    9: (150, 180, 100), 10: (100, 150, 90), 11: (200, 150, 60), 12: (180, 170, 90),
    13: (210, 140, 70), 14: (190, 160, 100), 15: (185, 165, 110), 16: (200, 180, 120),
    17: (170, 200, 150), 18: (220, 120, 90), 19: (210, 190, 200), 20: (90, 80, 70),
}


def preview_trees(base: BaseConfig, aoi_name: str, bbox_en, log=print) -> list[Path]:
    from .cook.encode import decode_records

    ids = chunks_covering_bbox_en(base.grid, bbox_en, 0)
    xs = sorted({c.cx for c in ids})
    zs = sorted({c.cz for c in ids})
    scale_px = 4  # 2048 m -> 512 px per chunk
    side = base.grid.chunk_m // scale_px
    img = np.full((len(zs) * side, len(xs) * side, 3), 18, dtype=np.uint8)
    total = 0
    import collections

    hist = collections.Counter()
    for c in ids:
        p = DATA_WORK / "chunks" / "trees" / "0" / f"{c.cx}_{c.cz}.lac"
        if not p.exists():
            continue
        meta, payload = read_chunk(p)
        if meta.count == 0:
            continue
        x, z, sp, sc, _ = decode_records(base.encode, payload, meta.count, ["u2", "u2", "u1", "u1", "u1"])
        px = (x.astype(np.float32) / 65535 * side).astype(int) + xs.index(c.cx) * side
        pz = (z.astype(np.float32) / 65535 * side).astype(int) + zs.index(c.cz) * side
        for cid in np.unique(sp):
            m = sp == cid
            img[np.clip(pz[m], 0, img.shape[0] - 1), np.clip(px[m], 0, img.shape[1] - 1)] = SPECIES_COLORS.get(int(cid), (255, 0, 255))
            hist[int(cid)] += int(m.sum())
        total += meta.count
    if not total:
        return []
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    p = PREVIEW_DIR / f"{aoi_name}-trees.png"
    Image.fromarray(img, "RGB").save(p, optimize=True)
    top = ", ".join(f"{i}:{n * 100 // total}%" for i, n in hist.most_common(8))
    log(f"  {p.name}: {total} trees; top species ids {top}")
    return [p]


def _assemble_wet_mask(base: BaseConfig, bbox_en) -> np.ndarray | None:
    """Wet texels across the AOI: quantized value > 0 (q == 0 is the reserved dry code)."""
    import zlib

    ids = chunks_covering_bbox_en(base.grid, bbox_en, 0)
    res = base.grid.chunk_m // 2
    xs = sorted({c.cx for c in ids})
    zs = sorted({c.cz for c in ids})
    full = np.zeros((len(zs) * res, len(xs) * res), dtype=bool)
    found = 0
    for c in ids:
        p = DATA_WORK / "chunks" / "water" / "0" / f"{c.cx}_{c.cz}.lac"
        if not p.exists():
            continue
        meta, payload = read_chunk(p)
        arr = decode_quant16(base.encode, payload, meta.res, meta.qoffset, meta.qscale)
        wet = arr > meta.qoffset + meta.qscale * 0.5
        r, q = zs.index(c.cz) * res, xs.index(c.cx) * res
        full[r : r + res, q : q + res] = wet[:res, :res]
        found += 1
    return full if found else None


def debug_composite(base: BaseConfig, aoi_name: str, bbox_en, scale_px: int = 1, log=print) -> list[Path]:
    """One fused debug image at `scale_px` meters/pixel: hillshade terrain, tinted by land
    cover, water in blue, trees as species-colored dots. scale_px=1 => true 1 m = 1 px."""
    from .cook.encode import decode_records

    got = _assemble_height(base, bbox_en, 0)
    if got is None:
        log("  (no height chunks; run cook first)")
        return []
    height, _ = got  # 1 m grid, apron-trimmed mosaic
    H, W = height.shape

    shade = hillshade(height, 1.0).astype(np.float32) / 255.0
    rgb = np.zeros((H, W, 3), dtype=np.float32)

    classes = _assemble_u8(base, "biome", bbox_en, 0, 2)  # 2 m grid
    if classes is not None:
        cls_full = np.repeat(np.repeat(classes, 2, 0), 2, 1)[:H, :W]
        tint = np.zeros((H, W, 3), dtype=np.float32)
        for cid, color in CLASS_COLORS.items():
            tint[cls_full == cid] = color
        rgb = tint * (0.45 + 0.55 * shade[..., None])
    else:
        rgb = np.stack([shade * 200] * 3, -1)

    wet = _assemble_wet_mask(base, bbox_en)  # 2 m grid
    if wet is not None:
        wet_full = np.repeat(np.repeat(wet, 2, 0), 2, 1)[:H, :W]
        rgb[wet_full] = np.array([50, 110, 190]) * (0.6 + 0.4 * shade[wet_full][..., None])

    img = rgb.astype(np.uint8)

    # stamp trees: species-colored, size-scaled dot per crown (chunk-local -> AOI pixel)
    ids = chunks_covering_bbox_en(base.grid, bbox_en, 0)
    xs = sorted({c.cx for c in ids})
    zs = sorted({c.cz for c in ids})
    n_trees = 0
    for c in ids:
        p = DATA_WORK / "chunks" / "trees" / "0" / f"{c.cx}_{c.cz}.lac"
        if not p.exists():
            continue
        meta, payload = read_chunk(p)
        if meta.count == 0:
            continue
        x, z, sp, sc, _ = decode_records(base.encode, payload, meta.count, ["u2", "u2", "u1", "u1", "u1"])
        px = (x.astype(np.float32) / 65535 * base.grid.chunk_m + xs.index(c.cx) * base.grid.chunk_m).astype(int)
        pz = (z.astype(np.float32) / 65535 * base.grid.chunk_m + zs.index(c.cz) * base.grid.chunk_m).astype(int)
        ok = (px >= 0) & (px < W) & (pz >= 0) & (pz < H)
        px, pz, sp = px[ok], pz[ok], sp[ok]
        for cid in np.unique(sp):
            m = sp == cid
            img[pz[m], px[m]] = SPECIES_COLORS.get(int(cid), (255, 0, 255))
        n_trees += int(ok.sum())

    if scale_px > 1:
        img = img[::scale_px, ::scale_px]
    while max(img.shape[:2]) > 16384:
        img = img[::2, ::2]
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    p = PREVIEW_DIR / f"{aoi_name}-composite-{img.shape[1]}x{img.shape[0]}.png"
    Image.fromarray(img, "RGB").save(p, optimize=True)
    log(f"  {p.name}: {W}x{H} m fused ({n_trees} trees) at {p.name.split('-')[-1]}")
    return [p]


def preview_height(base: BaseConfig, aoi_name: str, bbox_en, log=print) -> list[Path]:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for lod in base.grid.lods:
        got = _assemble_height(base, bbox_en, lod)
        if got is None:
            continue
        arr, texel = got
        img = hillshade(arr, texel)
        # keep previews manageable
        while max(img.shape) > 8192:
            img = img[::2, ::2]
        p = PREVIEW_DIR / f"{aoi_name}-height-lod{lod}-hillshade.png"
        Image.fromarray(img, "L").save(p, optimize=True)
        finite = arr[np.isfinite(arr)]
        log(
            f"  {p.name}: {img.shape[1]}x{img.shape[0]}, "
            f"height range {finite.min():.1f}..{finite.max():.1f} m"
        )
        written.append(p)
    return written
