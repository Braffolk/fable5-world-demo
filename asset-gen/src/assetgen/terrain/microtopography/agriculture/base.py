"""Pinned corrected-DTM evaluation on the canonical height sample lattice."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

from ....config import ASSET_GEN_ROOT, load_base
from ....height_geom import (
    UNITS_PER_METER,
    HeightChunkId,
    chunk_origin_en_units,
    sample_center_en_units,
    texel_units,
)
from ...repair.baseline import SampleGrid, bilinear_baseline
from .model import CultivatedConditions


@dataclass(frozen=True)
class SurfaceGrid:
    eastings: np.ndarray
    northings: np.ndarray
    height: np.ndarray


@dataclass(frozen=True)
class SourceSlope:
    median_slope_deg: float
    downhill_aspect_deg_east_of_grid_north: float


def _source(conditions: CultivatedConditions) -> tuple[np.ndarray, SampleGrid]:
    grid = load_base().grid
    path = ASSET_GEN_ROOT / Path(conditions.corrected_base_path)
    height = np.load(path, allow_pickle=False)
    if height.shape != (grid.chunk_res, grid.chunk_res):
        raise ValueError(f"corrected LOD0 source has unexpected shape {height.shape}")
    chunk = HeightChunkId(
        0, conditions.corrected_base_lod0_cx, conditions.corrected_base_lod0_cz
    )
    center_e_u, center_n_u = sample_center_en_units(grid, chunk, 0, 0)
    source_grid = SampleGrid(
        center_e=center_e_u / UNITS_PER_METER,
        center_n=center_n_u / UNITS_PER_METER,
        texel_m=1.0,
        rows=grid.chunk_res,
        cols=grid.chunk_res,
    )
    return np.asarray(height, dtype=np.float64), source_grid


def fine_payload_base(conditions: CultivatedConditions) -> SurfaceGrid:
    """Return exact 2049 canonical LOD -2 centers including east/south apron."""
    grid = load_base().grid
    chunk = HeightChunkId(conditions.fine_lod, conditions.fine_cx, conditions.fine_cz)
    origin_e_u, origin_n_u = chunk_origin_en_units(grid, chunk)
    step_u = texel_units(conditions.fine_lod)
    east = (
        origin_e_u + step_u // 2 + np.arange(2049) * step_u
    ) / UNITS_PER_METER
    north = (
        origin_n_u - step_u // 2 - np.arange(2049) * step_u
    ) / UNITS_PER_METER
    source, source_grid = _source(conditions)
    target = SampleGrid(float(east[0]), float(north[0]), conditions.texel_m, 2049, 2049)
    tile = bilinear_baseline(source, source_grid=source_grid, target_grid=target)
    if not tile.valid.all():
        raise ValueError("canonical fine payload leaves pinned corrected-DTM support")
    return SurfaceGrid(east, north, np.asarray(tile.height, dtype=np.float32))


def process_halo_base(
    conditions: CultivatedConditions,
    *,
    texel_m: float,
) -> SurfaceGrid:
    """Evaluate a real exterior DTM halo on north-up process-cell centers."""
    grid = load_base().grid
    chunk = HeightChunkId(conditions.fine_lod, conditions.fine_cx, conditions.fine_cz)
    origin_e_u, origin_n_u = chunk_origin_en_units(grid, chunk)
    e_min = origin_e_u / UNITS_PER_METER
    n_max = origin_n_u / UNITS_PER_METER
    halo_cells = int(np.ceil(conditions.flow_context_halo_m / texel_m))
    core_cells = int(round(conditions.size_m / texel_m))
    side = core_cells + 2 * halo_cells
    east = e_min + (np.arange(side) - halo_cells + 0.5) * texel_m
    north = n_max - (np.arange(side) - halo_cells + 0.5) * texel_m
    source, source_grid = _source(conditions)
    target = SampleGrid(float(east[0]), float(north[0]), texel_m, side, side)
    tile = bilinear_baseline(source, source_grid=source_grid, target_grid=target)
    if not tile.valid.all():
        raise ValueError("hydrology halo leaves pinned corrected-DTM support")
    return SurfaceGrid(east, north, np.asarray(tile.height, dtype=np.float32))


def rgb_conditioning_crop(conditions: CultivatedConditions) -> np.ndarray:
    """Read the exact 128 m evidence crop used to observe operation orientation."""
    grid = load_base().grid
    chunk = HeightChunkId(conditions.fine_lod, conditions.fine_cx, conditions.fine_cz)
    origin_e_u, origin_n_u = chunk_origin_en_units(grid, chunk)
    e_min = origin_e_u / UNITS_PER_METER
    n_max = origin_n_u / UNITS_PER_METER
    path = ASSET_GEN_ROOT / conditions.rgb_orthophoto_path
    with rasterio.open(path) as dataset:
        window = from_bounds(
            e_min,
            n_max - conditions.size_m,
            e_min + conditions.size_m,
            n_max,
            dataset.transform,
        )
        rgb = dataset.read(indexes=(1, 2, 3), window=window)
    if rgb.shape != (3, 640, 640):
        raise ValueError(f"unexpected 0.20 m RGB conditioning crop shape {rgb.shape}")
    return np.moveaxis(rgb, 0, -1)


def corrected_source_slope(conditions: CultivatedConditions) -> SourceSlope:
    """Measure the selected owned core on the native corrected 1 m observations."""
    grid = load_base().grid
    source, _ = _source(conditions)
    fine = HeightChunkId(conditions.fine_lod, conditions.fine_cx, conditions.fine_cz)
    fine_e_u, fine_n_u = chunk_origin_en_units(grid, fine)
    lod0 = HeightChunkId(
        0, conditions.corrected_base_lod0_cx, conditions.corrected_base_lod0_cz
    )
    lod0_e_u, lod0_n_u = chunk_origin_en_units(grid, lod0)
    col0 = (fine_e_u - lod0_e_u) // UNITS_PER_METER
    row0 = (lod0_n_u - fine_n_u) // UNITS_PER_METER
    side = int(round(conditions.size_m))
    core = source[row0 : row0 + side, col0 : col0 + side]
    if core.shape != (side, side):
        raise ValueError("selected R0 source core leaves corrected LOD0 support")
    south_gradient, east_gradient = np.gradient(core, 1.0)
    slope = np.degrees(np.arctan(np.hypot(east_gradient, south_gradient)))
    downhill_east = float(np.mean(-east_gradient))
    downhill_north = float(np.mean(south_gradient))
    aspect = float(np.degrees(np.arctan2(downhill_east, downhill_north)) % 360.0)
    return SourceSlope(float(np.median(slope)), aspect)
