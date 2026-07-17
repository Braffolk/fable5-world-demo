"""World-coordinate parcel geometry, canonical lattice, base height, and masks.

Everything here is a pure function of world coordinates + source geometry, so any
crop of the domain yields byte-identical samples (the v1 partition-invariance fix).
The parcel-id and boundary-distance rasters are built once over a fixed, integer
aligned super-bbox and sampled by nearest cell, independent of the query window.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyogrio.raw
import rasterio.features
import shapely
from affine import Affine

from ....config import ASSET_GEN_ROOT, load_base
from ....height_geom import (
    UNITS_PER_METER,
    HeightChunkId,
    sample_center_en_units,
)
from ...repair.baseline import SampleGrid, bilinear_baseline

_HALF = 0.5
_RASTER_PITCH_M = 0.5


def canonical_axes(bbox_en: tuple[float, float, float, float], texel_m: float):
    """Chunk-aligned cell centers: east ascending, north descending.

    Uses the same half-texel-offset convention as the LOD chunk lattice so samples
    from overlapping windows coincide exactly (enables real partition-invariance).
    """
    e_min, n_min, e_max, n_max = bbox_en
    half = texel_m * _HALF
    ncol = int(round((e_max - e_min) / texel_m)) + 1
    nrow = int(round((n_max - n_min) / texel_m)) + 1
    east = e_min + half + np.arange(ncol) * texel_m
    north = n_max - half - np.arange(nrow) * texel_m
    return east, north


# --------------------------------------------------------------------------- base
def _corrected_source(source_path: str, lod0_cx: int, lod0_cz: int):
    grid = load_base().grid
    height = np.load(ASSET_GEN_ROOT / Path(source_path), allow_pickle=False)
    if height.shape != (grid.chunk_res, grid.chunk_res):
        raise ValueError(f"corrected LOD0 source has unexpected shape {height.shape}")
    chunk = HeightChunkId(0, lod0_cx, lod0_cz)
    ce_u, cn_u = sample_center_en_units(grid, chunk, 0, 0)
    source_grid = SampleGrid(
        center_e=ce_u / UNITS_PER_METER,
        center_n=cn_u / UNITS_PER_METER,
        texel_m=1.0,
        rows=grid.chunk_res,
        cols=grid.chunk_res,
    )
    return np.asarray(height, dtype=np.float64), source_grid


def sample_base(source_path: str, lod0_cx: int, lod0_cz: int, east, north) -> np.ndarray:
    """Bilinear corrected-DTM parent height on the given canonical centers."""
    source, source_grid = _corrected_source(source_path, lod0_cx, lod0_cz)
    target = SampleGrid(float(east[0]), float(north[0]), float(east[1] - east[0]),
                        int(north.size), int(east.size))
    tile = bilinear_baseline(source, source_grid=source_grid, target_grid=target)
    if not tile.valid.all():
        raise ValueError("agriculture v2 base leaves pinned corrected-DTM support")
    return np.asarray(tile.height, dtype=np.float64)


# ------------------------------------------------------------------------- parcels
@dataclass(frozen=True)
class ParcelInfo:
    etak_id: int
    theta_rad: float          # principal (row) axis, east-of-grid-x
    cx: float
    cy: float
    along_half_m: float       # half extent projected on the principal axis


class ParcelField:
    """Rasterized arable-parcel identity + boundary distance over a fixed super-bbox."""

    def __init__(self, etak_path: Path, arable_layer: str,
                 super_bbox: tuple[int, int, int, int]) -> None:
        e0, n0, e1, n1 = (int(v) for v in super_bbox)
        self.pitch = _RASTER_PITCH_M
        self.e0, self.n1 = e0, n1
        self.width = int(round((e1 - e0) / self.pitch))
        self.height = int(round((n1 - n0) / self.pitch))
        self.transform = Affine(self.pitch, 0.0, e0, 0.0, -self.pitch, n1)

        meta, _fids, wkbs, vals = pyogrio.raw.read(
            etak_path, layer=arable_layer, bbox=(e0, n0, e1, n1),
            columns=["etak_id"], return_fids=True,
        )
        if str(meta.get("crs")) != "EPSG:3301":
            raise ValueError("ETAK arable layer is not EPSG:3301")
        self.infos: dict[int, ParcelInfo] = {}
        shapes = []
        if wkbs is not None:
            for i, wkb in enumerate(wkbs):
                pid = int(vals[0][i])
                geom = shapely.force_2d(shapely.from_wkb(bytes(wkb)))
                shapes.append((shapely.geometry.mapping(geom), pid))
                self.infos[pid] = self._info(pid, geom)
        self.id_raster = rasterio.features.rasterize(
            shapes, out_shape=(self.height, self.width), transform=self.transform,
            fill=0, dtype="int32", all_touched=False,
        ) if shapes else np.zeros((self.height, self.width), dtype=np.int32)

        # Boundary distance (metres) for interior samples; 0 outside any parcel.
        from scipy import ndimage
        inside = self.id_raster > 0
        self.dist_raster = ndimage.distance_transform_edt(
            inside, sampling=self.pitch
        ).astype(np.float32)

    @staticmethod
    def _info(pid: int, geom: shapely.Geometry) -> ParcelInfo:
        poly = geom if geom.geom_type == "Polygon" else max(geom.geoms, key=lambda g: g.area)
        xy = np.asarray(poly.exterior.coords, dtype=np.float64)
        mean = xy.mean(axis=0)
        _u, _s, vt = np.linalg.svd(xy - mean, full_matrices=False)
        axis = vt[0]
        theta = float(np.arctan2(axis[1], axis[0]))
        c = geom.centroid
        proj = (xy - mean) @ axis
        return ParcelInfo(pid, theta, float(c.x), float(c.y),
                          float(0.5 * (proj.max() - proj.min())))

    def _cell(self, east: np.ndarray, north: np.ndarray):
        col = np.clip(((east - self.e0) / self.pitch).astype(np.int64), 0, self.width - 1)
        row = np.clip(((self.n1 - north) / self.pitch).astype(np.int64), 0, self.height - 1)
        return row, col

    def sample(self, east2d: np.ndarray, north2d: np.ndarray):
        row, col = self._cell(east2d, north2d)
        return self.id_raster[row, col], self.dist_raster[row, col]


# --------------------------------------------------------------------------- masks
_HARD_AREA_LAYERS = {
    "building": "E_404_maaalune_hoone_ka",
    "yard": "E_302_ou_a",
    "forest": "E_305_puittaimestik_a",
    "wetland": "E_306_margala_a",
}


def load_hard_and_ditch(etak_path: Path, bbox: tuple[int, int, int, int],
                        east: np.ndarray, north: np.ndarray):
    """Rasterize hard-surface areas and ditch/watercourse lines over the query grid.

    Returns (hard_mask, ditch_mask) as bool arrays on the query lattice. Ditches are
    watercourse lines buffered by their attribute width and preserved as typed
    structure (added relief is zeroed there; the ditch itself is not touched here).
    """
    texel = float(east[1] - east[0])
    x0 = float(east[0]) - 0.5 * texel
    y0 = float(north[0]) + 0.5 * texel
    transform = Affine(texel, 0.0, x0, 0.0, -texel, y0)
    shape = (int(north.size), int(east.size))
    hard_shapes = []
    for layer in _HARD_AREA_LAYERS.values():
        try:
            _m, _f, wkbs, _v = pyogrio.raw.read(
                etak_path, layer=layer, bbox=bbox, columns=[], return_fids=True)
        except Exception:
            wkbs = None
        if wkbs is not None:
            for wkb in wkbs:
                hard_shapes.append(shapely.force_2d(shapely.from_wkb(bytes(wkb))))
    ditch_shapes = []
    try:
        _m, _f, wkbs, attrs = pyogrio.raw.read(
            etak_path, layer="E_203_vooluveekogu_j", bbox=bbox,
            columns=["laius"], return_fids=True)
    except Exception:
        wkbs = None
        attrs = None
    if wkbs is not None:
        for i, wkb in enumerate(wkbs):
            line = shapely.force_2d(shapely.from_wkb(bytes(wkb)))
            width = 1.0
            if attrs is not None and attrs[0][i] is not None:
                try:
                    width = max(float(attrs[0][i]), 0.5)
                except (TypeError, ValueError):
                    width = 1.0
            ditch_shapes.append(line.buffer(0.5 * width + 0.5))

    def _burn(shapes) -> np.ndarray:
        if not shapes:
            return np.zeros(shape, dtype=bool)
        burned = rasterio.features.rasterize(
            ((shapely.geometry.mapping(g), 1) for g in shapes),
            out_shape=shape, transform=transform, fill=0, dtype="uint8",
            all_touched=True,
        )
        return burned.astype(bool)

    return _burn(hard_shapes), _burn(ditch_shapes)
