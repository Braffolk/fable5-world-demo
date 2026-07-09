"""Water-surface (waterY) chunk rasterization @ 2 m, u16-quantized by the cooker.

Renderer semantics (frozen contract): wet texel = water-surface elevation (m EH2000);
dry texel = bed height - 2.0 m sentinel, so the water clipmap dives under terrain.

Levels: sea = 0.0; standing water = per-polygon flat level (median LiDAR DTM inside the
polygon — laser returns off water are already flat, and we read the FULL polygon footprint
so lakes spanning chunk borders get one consistent level); rivers = lightly smoothed DTM
inside the river polygon (the LiDAR water surface itself).
"""
from __future__ import annotations

import numpy as np
import rasterio.features
import rasterio.transform

from ..config import DATA_IN
from .etak_read import etak_gpkg, read_layer_window
from .mosaic import RasterStack, dem_sources

DRY_SENTINEL_DROP = 2.0
_level_cache: dict[int, float] = {}


def _polygon_level(stack: RasterStack, etak_id: int, geom) -> float:
    """Flat standing-water level from the whole polygon's DTM footprint (cached per id)."""
    if etak_id in _level_cache:
        return _level_cache[etak_id]
    minx, miny, maxx, maxy = geom.bounds
    # cap the read at ~1024^2 texels; big lakes are flat anyway so coarser sampling is fine
    texel = max(2.0, (maxx - minx) / 1024.0, (maxy - miny) / 1024.0)
    e0, n0 = np.floor(minx / texel) * texel, np.floor(miny / texel) * texel
    e1, n1 = np.ceil(maxx / texel) * texel, np.ceil(maxy / texel) * texel
    dtm = stack.read_window(e0, n0, e1, n1, texel)
    transform = rasterio.transform.from_origin(e0, n1, texel, texel)
    mask = rasterio.features.rasterize(
        [(geom, 1)], out_shape=dtm.shape, transform=transform
    ).astype(bool)
    inside = dtm[mask & np.isfinite(dtm)]
    level = float(np.median(inside)) if inside.size else 0.0
    _level_cache[etak_id] = level
    return level


def _median3x3(arr: np.ndarray) -> np.ndarray:
    pads = np.pad(arr, 1, mode="edge")
    stack9 = np.stack([pads[dy : dy + arr.shape[0], dx : dx + arr.shape[1]]
                       for dy in range(3) for dx in range(3)])
    return np.median(stack9, axis=0)


def rasterize_water(
    window_en: tuple[float, float, float, float, float], stack: RasterStack
) -> np.ndarray | None:
    """waterY float raster for a chunk window, or None when the chunk has no water at all."""
    e_min, n_min, e_max, n_max, t = window_en
    cols = round((e_max - e_min) / t)
    rows = round((n_max - n_min) / t)
    transform = rasterio.transform.from_origin(e_min, n_max, t, t)
    bbox = (e_min, n_min, e_max, n_max)
    gpkg = etak_gpkg()

    sea_geoms, _ = read_layer_window(gpkg, "E_201_meri_a", bbox)
    still_geoms, still_cols = read_layer_window(gpkg, "E_202_seisuveekogu_a", bbox, fields=["etak_id"])
    flow_geoms, _ = read_layer_window(gpkg, "E_203_vooluveekogu_a", bbox)
    if len(sea_geoms) + len(still_geoms) + len(flow_geoms) == 0:
        return None

    dtm = np.nan_to_num(stack.read_window(e_min, n_min, e_max, n_max, t), nan=0.0)
    # dry texels stay NaN -> the cooker encodes them as the reserved q=0 value and the
    # CLIENT reconstructs bed-2 from its own height layer (shipping bed-2 here would
    # duplicate the terrain signal and bloat every chunk ~15x)
    water = np.full_like(dtm, np.nan)

    if len(flow_geoms):
        flow_mask = rasterio.features.rasterize(
            [(g, 1) for g in flow_geoms if g is not None],
            out_shape=(rows, cols), transform=transform,
        ).astype(bool)
        smoothed = _median3x3(dtm)
        water[flow_mask] = smoothed[flow_mask]

    for i, geom in enumerate(still_geoms):
        if geom is None:
            continue
        level = _polygon_level(stack, int(still_cols["etak_id"][i]), geom)
        mask = rasterio.features.rasterize(
            [(geom, 1)], out_shape=(rows, cols), transform=transform
        ).astype(bool)
        water[mask] = level

    if len(sea_geoms):
        sea_mask = rasterio.features.rasterize(
            [(g, 1) for g in sea_geoms if g is not None],
            out_shape=(rows, cols), transform=transform,
        ).astype(bool)
        water[sea_mask] = 0.0
    return water
