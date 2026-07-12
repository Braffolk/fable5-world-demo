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
import shapely
from scipy import ndimage

from ..config import DATA_IN
from .etak_read import etak_gpkg, read_layer_window
from .mosaic import RasterStack, dem_sources

DRY_SENTINEL_DROP = 2.0
_level_cache: dict[int, float] = {}

# --- submerged-bed depth model (#104) --------------------------------------------------
# Depth is a DETERMINISTIC physics function of REAL measured channel width + REAL ETAK
# type — never faked, no noise. Rivers: Leopold-Maddock hydraulic geometry gives mean
# depth d ∝ w^0.8 (from d ∝ Q^0.4, w ∝ Q^0.5); we carry a thalweg factor 1.5 for the
# max carve. Width comes from the REAL polygon footprint (2·max shore-distance) since the
# E_203 area layer stores no width; the flow-type coefficient comes from the nearest ETAK
# centreline (E_203_..._j `tyyp`). Lakes: an explicit shore-shelf wedge (not claimed as
# measured bathymetry) — depth ramps to a per-type plateau over a ~10 m shelf.
RIVER_K = 0.20            # base hydraulic coefficient (natural channels)
RIVER_THALWEG = 1.5      # mean-depth → max/thalweg-depth factor for the carve
RIVER_DMAX_CAP = 2.5     # m — guard against over-deep wide reaches / mill ponds
LAKE_SHELF_M = 10.0      # m — shore-shelf ramp width
# E_203_..._j `tyyp` code → hydraulic-coefficient multiplier on RIVER_K.
RIVER_TYPE_MULT = {10: 1.05, 30: 1.0, 40: 0.9, 50: 0.75}  # Jõgi / Oja / Peakraav / Kraav
# E_202 `tyyp` code → shore-shelf plateau depth (m).
LAKE_TYPE_DMAX = {10: 2.75, 20: 2.75, 30: 2.75, 40: 1.5, 50: 1.0, 60: 1.0, 999: 1.0}
# (Järv / Paisjärv / Tehisjärv=2.75 ; Laugas=1.5 ; Biotiik / Tiik / Muu=1.0)
COVERAGE_SUPERSAMPLE = 8  # sub-texels per axis per texel for the anti-aliased shore α (#114)


def _codes(cols: dict, key: str, n: int) -> list:
    arr = cols.get(key)
    if arr is None or len(arr) != n:
        return [None] * n
    return list(arr)


def bed_depth_field(
    window_en: tuple[float, float, float, float, float], stack: RasterStack
) -> tuple[np.ndarray, np.ndarray]:
    """(depth, coverage) rasters for a chunk window at the window's texel size.

    depth[i,j]  = submerged-bed depth in meters BELOW the water surface, 0 on dry texels.
                  The wet extent is rasterized from the SAME ETAK polygons as
                  `rasterize_water` (E_203_..._a rivers + E_202 lakes + E_201 sea), so
                  depth>0 exactly under the rendered-water mask. The carver subtracts this
                  from the cooked DTM height.
    coverage[i,j] = anti-aliased water-area fraction ∈ [0,1] (#114), from an 8× supersampled
                  rasterization of the same polygons, box-downsampled to the window texel.

    Sea (E_201) is included in `coverage` but NOT carved (depth stays 0) — sea/big-lake
    bathymetry needs external survey data and is deferred.
    """
    e_min, n_min, e_max, n_max, t = window_en
    cols = round((e_max - e_min) / t)
    rows = round((n_max - n_min) / t)
    transform = rasterio.transform.from_origin(e_min, n_max, t, t)
    bbox = (e_min, n_min, e_max, n_max)
    gpkg = etak_gpkg()

    sea_geoms, _ = read_layer_window(gpkg, "E_201_meri_a", bbox)
    lake_geoms, lake_cols = read_layer_window(gpkg, "E_202_seisuveekogu_a", bbox, fields=["tyyp"])
    riv_geoms, _ = read_layer_window(gpkg, "E_203_vooluveekogu_a", bbox)
    line_geoms, line_cols = read_layer_window(gpkg, "E_203_vooluveekogu_j", bbox, fields=["tyyp"])

    depth = np.zeros((rows, cols), dtype=np.float32)

    # --- rivers: measured-width hydraulic parabola, per connected reach -----------------
    rg = [g for g in riv_geoms if g is not None]
    if rg:
        m_riv = rasterio.features.rasterize(
            [(g, 1) for g in rg], out_shape=(rows, cols), transform=transform
        ).astype(bool)
        if m_riv.any():
            d_shore = ndimage.distance_transform_edt(m_riv) * t  # m to nearest bank
            lbl, nlab = ndimage.label(m_riv)
            idx = np.arange(1, nlab + 1)
            max_d = np.atleast_1d(ndimage.maximum(d_shore, labels=lbl, index=idx))
            cents = ndimage.center_of_mass(m_riv, labels=lbl, index=idx)  # list of (row,col)
            # nearest ETAK centreline → flow-type multiplier (measured area layer has none)
            lg = [(g, ty) for g, ty in zip(line_geoms, _codes(line_cols, "tyyp", len(line_geoms))) if g is not None]
            tree = shapely.STRtree([g for g, _ in lg]) if lg else None
            ltypes = [ty for _, ty in lg]
            dmax_map = np.zeros(nlab + 1, dtype=np.float64)
            r_map = np.ones(nlab + 1, dtype=np.float64)
            for k in range(1, nlab + 1):
                w = 2.0 * float(max_d[k - 1])
                if w <= 0.0:
                    continue
                mult = 1.0
                if tree is not None:
                    r, c = cents[k - 1]
                    p = shapely.Point(e_min + (c + 0.5) * t, n_max - (r + 0.5) * t)
                    mult = RIVER_TYPE_MULT.get(int(ltypes[int(tree.nearest(p))] or 0), 1.0)
                dmax_map[k] = min(RIVER_K * mult * w**0.8 * RIVER_THALWEG, RIVER_DMAX_CAP)
                r_map[k] = max(w / 2.0, 1e-6)
            s = np.clip(d_shore / r_map[lbl], 0.0, 1.0)
            d_riv = dmax_map[lbl] * (2.0 * s - s * s)
            depth = np.where(m_riv, np.maximum(depth, d_riv), depth).astype(np.float32)

    # --- lakes: per-type shore-shelf wedge ----------------------------------------------
    lake_shapes = [
        (g, float(LAKE_TYPE_DMAX.get(int(ty) if ty is not None else -1, 1.0)))
        for g, ty in zip(lake_geoms, _codes(lake_cols, "tyyp", len(lake_geoms)))
        if g is not None
    ]
    if lake_shapes:
        dmax_r = rasterio.features.rasterize(
            lake_shapes, out_shape=(rows, cols), transform=transform, dtype="float32"
        )
        m_lake = dmax_r > 0.0
        if m_lake.any():
            d_shore = ndimage.distance_transform_edt(m_lake) * t
            s = np.clip(d_shore / LAKE_SHELF_M, 0.0, 1.0)
            d_lake = dmax_r * (2.0 * s - s * s)
            depth = np.where(m_lake, np.maximum(depth, d_lake), depth).astype(np.float32)

    # --- anti-aliased shore coverage α (#114): 8× supersample → box-downsample -----------
    union = rg + [g for g in lake_geoms if g is not None] + [g for g in sea_geoms if g is not None]
    if union:
        ss = COVERAGE_SUPERSAMPLE
        st = rasterio.transform.from_origin(e_min, n_max, t / ss, t / ss)
        fine = rasterio.features.rasterize(
            [(g, 1) for g in union], out_shape=(rows * ss, cols * ss), transform=st, dtype="uint8"
        ).astype(np.float32)
        coverage = fine.reshape(rows, ss, cols, ss).mean(axis=(1, 3)).astype(np.float32)
    else:
        coverage = np.zeros((rows, cols), dtype=np.float32)

    return depth, coverage


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
