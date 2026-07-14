"""Exact cell-overlap resampling for the four frozen Moore output phases."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion

from .archive import SourceGrid


SOURCE_CELL_M = 0.01
FINE_CELL_M = 0.0625
PHASES_M = ((0.0, 0.0), (0.005, 0.0), (0.0, 0.005), (0.005, 0.005))


@dataclass(frozen=True)
class AreaGrid:
    values: np.ndarray
    valid: np.ndarray
    x_bounds_m: np.ndarray
    y_bounds_m: np.ndarray
    role: str

    @property
    def x_centers_m(self) -> np.ndarray:
        return 0.5 * (self.x_bounds_m[:-1] + self.x_bounds_m[1:])

    @property
    def y_centers_m(self) -> np.ndarray:
        return 0.5 * (self.y_bounds_m[:-1] + self.y_bounds_m[1:])

    @property
    def cell_m(self) -> float:
        return float(self.x_bounds_m[1] - self.x_bounds_m[0])


@dataclass(frozen=True)
class FinePhase:
    phase_index: int
    offset_xy_m: tuple[float, float]
    height: AreaGrid
    finite_overlap_fraction: np.ndarray
    overlap_valid: np.ndarray
    target_valid: np.ndarray
    boundary_or_incomplete: np.ndarray


def _target_bounds(source_bounds: np.ndarray, offset_m: float) -> np.ndarray:
    source_origin = float(source_bounds[0])
    lattice_origin = source_origin + offset_m
    first = int(np.floor((source_bounds[0] - lattice_origin) / FINE_CELL_M + 1e-12))
    stop = int(np.ceil((source_bounds[-1] - lattice_origin) / FINE_CELL_M - 1e-12))
    return lattice_origin + np.arange(first, stop + 1, dtype=np.float64) * FINE_CELL_M


def overlap_lengths(target_bounds: np.ndarray, source_bounds: np.ndarray) -> np.ndarray:
    """Dense exact interval-overlap matrix, target rows by source columns."""
    left = np.maximum(target_bounds[:-1, None], source_bounds[None, :-1])
    right = np.minimum(target_bounds[1:, None], source_bounds[None, 1:])
    return np.maximum(right - left, 0.0)


def materialize_phase(source: SourceGrid, phase_index: int) -> FinePhase:
    if phase_index < 0 or phase_index >= len(PHASES_M):
        raise ValueError("invalid Moore phase index")
    offset_x, offset_y = PHASES_M[phase_index]
    source_x_bounds = source.x_centers_m[0] - 0.005 + np.arange(
        source.x_centers_m.size + 1, dtype=np.float64
    ) * SOURCE_CELL_M
    source_y_bounds = source.y_centers_m[0] - 0.005 + np.arange(
        source.y_centers_m.size + 1, dtype=np.float64
    ) * SOURCE_CELL_M
    target_x_bounds = _target_bounds(source_x_bounds, offset_x)
    target_y_bounds = _target_bounds(source_y_bounds, offset_y)
    wx = overlap_lengths(target_x_bounds, source_x_bounds)
    wy = overlap_lengths(target_y_bounds, source_y_bounds)
    finite = np.isfinite(source.z_m)
    source_values = np.where(finite, source.z_m, 0.0)
    finite_area = wy @ finite.astype(np.float64) @ wx.T
    weighted_height = wy @ source_values @ wx.T
    cell_area = FINE_CELL_M * FINE_CELL_M
    fraction = np.clip(finite_area / cell_area, 0.0, 1.0)
    overlap_valid = fraction >= (0.95 - 1e-12)
    values = np.full(finite_area.shape, np.nan, dtype=np.float64)
    np.divide(weighted_height, finite_area, out=values, where=overlap_valid)
    target_valid = binary_erosion(
        overlap_valid,
        structure=np.ones((3, 3), dtype=np.bool_),
        border_value=0,
    )
    values[~target_valid] = np.nan
    boundary = ~target_valid
    height = AreaGrid(
        values=values,
        valid=target_valid,
        x_bounds_m=target_x_bounds,
        y_bounds_m=target_y_bounds,
        role="H_phase_exact_source_cell_overlap",
    )
    return FinePhase(
        phase_index=phase_index,
        offset_xy_m=(offset_x, offset_y),
        height=height,
        finite_overlap_fraction=fraction,
        overlap_valid=overlap_valid,
        target_valid=target_valid,
        boundary_or_incomplete=boundary,
    )


def materialize_four_phases(source: SourceGrid) -> tuple[FinePhase, ...]:
    return tuple(materialize_phase(source, index) for index in range(4))


def common_overlay(
    grids: tuple[AreaGrid, AreaGrid, AreaGrid, AreaGrid]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Overlay all phase boundaries without interpolation and intersect validity."""
    # All phase boundaries are integral multiples of 0.0025 m from one
    # arbitrary source origin. Integerizing relative to that origin preserves
    # the exact 0.005/0.0625 geometry without quantizing the source datum.
    x_reference = min(float(grid.x_bounds_m[0]) for grid in grids)
    y_reference = min(float(grid.y_bounds_m[0]) for grid in grids)
    x_units = np.unique(
        np.concatenate(
            [np.rint((grid.x_bounds_m - x_reference) / 0.0025).astype(np.int64) for grid in grids]
        )
    )
    y_units = np.unique(
        np.concatenate(
            [np.rint((grid.y_bounds_m - y_reference) / 0.0025).astype(np.int64) for grid in grids]
        )
    )
    x_bounds = x_reference + x_units.astype(np.float64) * 0.0025
    y_bounds = y_reference + y_units.astype(np.float64) * 0.0025
    x_mid = 0.5 * (x_bounds[:-1] + x_bounds[1:])
    y_mid = 0.5 * (y_bounds[:-1] + y_bounds[1:])
    x_indices: list[np.ndarray] = []
    y_indices: list[np.ndarray] = []
    common = np.ones((y_mid.size, x_mid.size), dtype=np.bool_)
    for grid in grids:
        xi = np.searchsorted(grid.x_bounds_m, x_mid, side="right") - 1
        yi = np.searchsorted(grid.y_bounds_m, y_mid, side="right") - 1
        x_inside = (xi >= 0) & (xi < grid.values.shape[1])
        y_inside = (yi >= 0) & (yi < grid.values.shape[0])
        clipped_x = np.clip(xi, 0, grid.values.shape[1] - 1)
        clipped_y = np.clip(yi, 0, grid.values.shape[0] - 1)
        common &= y_inside[:, None] & x_inside[None, :]
        common &= grid.valid[clipped_y[:, None], clipped_x[None, :]]
        x_indices.append(xi)
        y_indices.append(yi)
    return x_bounds, y_bounds, common, tuple(x_indices), tuple(y_indices)


def complete_common_cells(
    grid: AreaGrid,
    overlay_x_bounds: np.ndarray,
    overlay_y_bounds: np.ndarray,
    common: np.ndarray,
) -> np.ndarray:
    """Cells whose complete area lies in the four-phase common support."""
    x_mid = 0.5 * (overlay_x_bounds[:-1] + overlay_x_bounds[1:])
    y_mid = 0.5 * (overlay_y_bounds[:-1] + overlay_y_bounds[1:])
    xi = np.searchsorted(grid.x_bounds_m, x_mid, side="right") - 1
    yi = np.searchsorted(grid.y_bounds_m, y_mid, side="right") - 1
    inside_x = (xi >= 0) & (xi < grid.values.shape[1])
    inside_y = (yi >= 0) & (yi < grid.values.shape[0])
    dx = np.diff(overlay_x_bounds)
    dy = np.diff(overlay_y_bounds)
    areas = dy[:, None] * dx[None, :]
    accumulated = np.zeros(grid.values.shape, dtype=np.float64)
    oy, ox = np.nonzero(common & inside_y[:, None] & inside_x[None, :])
    np.add.at(accumulated, (yi[oy], xi[ox]), areas[oy, ox])
    expected = grid.cell_m * grid.cell_m
    return grid.valid & np.isclose(accumulated, expected, rtol=0, atol=1e-10)
