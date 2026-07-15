"""Exact-overlap source resampling and no-padding F4/R4 analysis bands."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .contract import ANALYSIS_CELL_M, F4_COEFFICIENTS


_F4 = np.asarray(F4_COEFFICIENTS, dtype=np.float64) / 64.0


@dataclass(frozen=True)
class RegularGrid:
    values: np.ndarray
    valid: np.ndarray
    x_bounds_m: np.ndarray
    y_bounds_m: np.ndarray
    role: str

    @property
    def cell_m(self) -> float:
        return float(self.x_bounds_m[1] - self.x_bounds_m[0])

    @property
    def x_centers_m(self) -> np.ndarray:
        return 0.5 * (self.x_bounds_m[:-1] + self.x_bounds_m[1:])

    @property
    def y_centers_m(self) -> np.ndarray:
        return 0.5 * (self.y_bounds_m[:-1] + self.y_bounds_m[1:])


@dataclass(frozen=True)
class PhaseBands:
    phase_xy_m: tuple[float, float]
    h2: RegularGrid
    a1: RegularGrid
    a0: RegularGrid
    b1: RegularGrid
    b2: RegularGrid


@dataclass(frozen=True)
class GridIntegral:
    area_m2: float
    complete_cells: int
    sum_area_value: float
    sum_area_squared: float
    minimum: float | None
    maximum: float | None


@dataclass(frozen=True)
class CommonPhaseIntegral:
    area_m2: float
    phase_sum_area_squared: tuple[float, float, float, float]
    pair_sum_area_squared_difference: tuple[float, float, float, float, float, float]


def _target_bounds(
    minimum_m: float,
    maximum_m: float,
    lattice_origin_m: float,
) -> np.ndarray:
    first = math.ceil((minimum_m - lattice_origin_m) / ANALYSIS_CELL_M - 1e-11)
    stop = math.floor((maximum_m - lattice_origin_m) / ANALYSIS_CELL_M + 1e-11)
    if stop <= first:
        return np.empty(0, dtype=np.float64)
    return lattice_origin_m + np.arange(first, stop + 1, dtype=np.float64) * ANALYSIS_CELL_M


def _overlap_indices(
    target_bounds: np.ndarray,
    source_bounds: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    result: list[tuple[np.ndarray, np.ndarray]] = []
    source_min = float(source_bounds[0])
    source_step = float(source_bounds[1] - source_bounds[0])
    source_count = source_bounds.size - 1
    for left, right in zip(target_bounds[:-1], target_bounds[1:], strict=True):
        first = max(0, int(math.floor((left - source_min) / source_step + 1e-12)))
        stop = min(source_count, int(math.ceil((right - source_min) / source_step - 1e-12)))
        indices = np.arange(first, stop, dtype=np.int64)
        lengths = np.maximum(
            np.minimum(right, source_bounds[indices + 1])
            - np.maximum(left, source_bounds[indices]),
            0.0,
        )
        keep = lengths > 0.0
        result.append((indices[keep], lengths[keep]))
    return result


def exact_overlap_resample(
    source_values: np.ndarray,
    source_valid: np.ndarray,
    source_x_bounds_m: np.ndarray,
    source_y_bounds_m: np.ndarray,
    *,
    lattice_origin_xy_m: tuple[float, float],
) -> RegularGrid:
    """Resample finite source cells as constant areas without support invention."""
    values = np.asarray(source_values, dtype=np.float64)
    valid = np.asarray(source_valid, dtype=np.bool_)
    if values.ndim != 2 or values.shape != valid.shape:
        raise ValueError("source values and validity must be aligned 2D arrays")
    if source_x_bounds_m.size != values.shape[1] + 1 or source_y_bounds_m.size != values.shape[0] + 1:
        raise ValueError("source bounds do not match source values")
    if not np.all(np.diff(source_x_bounds_m) > 0) or not np.all(np.diff(source_y_bounds_m) > 0):
        raise ValueError("source bounds must increase")

    target_x = _target_bounds(
        float(source_x_bounds_m[0]), float(source_x_bounds_m[-1]), lattice_origin_xy_m[0]
    )
    target_y = _target_bounds(
        float(source_y_bounds_m[0]), float(source_y_bounds_m[-1]), lattice_origin_xy_m[1]
    )
    if target_x.size < 2 or target_y.size < 2:
        return RegularGrid(
            np.empty((0, 0), dtype=np.float64),
            np.empty((0, 0), dtype=np.bool_),
            target_x,
            target_y,
            "H2_exact_source_cell_overlap",
        )

    x_overlap = _overlap_indices(target_x, source_x_bounds_m)
    y_overlap = _overlap_indices(target_y, source_y_bounds_m)
    horizontal_height = np.zeros((values.shape[0], len(x_overlap)), dtype=np.float64)
    horizontal_finite = np.zeros_like(horizontal_height)
    finite_values = np.where(valid, values, 0.0)
    for column, (indices, lengths) in enumerate(x_overlap):
        horizontal_height[:, column] = finite_values[:, indices] @ lengths
        horizontal_finite[:, column] = valid[:, indices].astype(np.float64) @ lengths

    output = np.full((len(y_overlap), len(x_overlap)), np.nan, dtype=np.float64)
    finite_area = np.zeros_like(output)
    for row, (indices, lengths) in enumerate(y_overlap):
        output[row] = lengths @ horizontal_height[indices]
        finite_area[row] = lengths @ horizontal_finite[indices]
    cell_area = ANALYSIS_CELL_M * ANALYSIS_CELL_M
    complete = np.isclose(finite_area, cell_area, rtol=0.0, atol=cell_area * 1e-10)
    np.divide(output, finite_area, out=output, where=complete)
    output[~complete] = np.nan
    return RegularGrid(output, complete, target_x, target_y, "H2_exact_source_cell_overlap")


def filter4(grid: RegularGrid, role: str) -> RegularGrid:
    """Apply normative float64 east-then-north F4 with complete support only."""
    ny, nx = grid.values.shape
    out_y = ny // 4
    out_x = nx // 4
    px = np.arange(out_x, dtype=np.int64)
    py = np.arange(out_y, dtype=np.int64)
    good_x = np.flatnonzero((4 * px - 3 >= 0) & (4 * px + 6 < nx))
    good_y = np.flatnonzero((4 * py - 3 >= 0) & (4 * py + 6 < ny))
    east = np.zeros((ny, out_x), dtype=np.float64)
    east_valid = np.zeros((ny, out_x), dtype=np.bool_)
    east_valid[:, good_x] = True
    for coefficient, k in zip(_F4, range(10), strict=True):
        columns = 4 * px[good_x] - 3 + k
        selected_valid = grid.valid[:, columns]
        east[:, good_x] += coefficient * np.where(selected_valid, grid.values[:, columns], 0.0)
        east_valid[:, good_x] &= selected_valid
    values = np.zeros((out_y, out_x), dtype=np.float64)
    valid = np.zeros((out_y, out_x), dtype=np.bool_)
    valid[good_y] = True
    for coefficient, k in zip(_F4, range(10), strict=True):
        rows = 4 * py[good_y] - 3 + k
        selected_valid = east_valid[rows]
        values[good_y] += coefficient * np.where(selected_valid, east[rows], 0.0)
        valid[good_y] &= selected_valid
    values[~valid] = np.nan
    cell_m = grid.cell_m * 4.0
    return RegularGrid(
        values,
        valid,
        grid.x_bounds_m[0] + np.arange(out_x + 1, dtype=np.float64) * cell_m,
        grid.y_bounds_m[0] + np.arange(out_y + 1, dtype=np.float64) * cell_m,
        role,
    )


def _keys_weights(t: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            -0.5 * t + t * t - 0.5 * t * t * t,
            1.0 - 2.5 * t * t + 1.5 * t * t * t,
            0.5 * t + 2.0 * t * t - 1.5 * t * t * t,
            -0.5 * t * t + 0.5 * t * t * t,
        ),
        axis=1,
    )


def reconstruct4(coarse: RegularGrid, target: RegularGrid, role: str) -> RegularGrid:
    """Reconstruct with separable Keys cubic a=-0.5 and no padded context."""
    if coarse.values.size == 0 or target.values.size == 0:
        return RegularGrid(
            np.full(target.values.shape, np.nan, dtype=np.float64),
            np.zeros(target.values.shape, dtype=np.bool_),
            target.x_bounds_m.copy(),
            target.y_bounds_m.copy(),
            role,
        )
    ux = (target.x_centers_m - coarse.x_centers_m[0]) / coarse.cell_m
    uy = (target.y_centers_m - coarse.y_centers_m[0]) / coarse.cell_m
    base_x = np.floor(ux + 0.5).astype(np.int64)
    base_y = np.floor(uy + 0.5).astype(np.int64)
    tx = ux - base_x
    ty = uy - base_y
    wx = _keys_weights(tx)
    wy = _keys_weights(ty)
    x_ok = (base_x - 1 >= 0) & (base_x + 2 < coarse.values.shape[1])
    y_ok = (base_y - 1 >= 0) & (base_y + 2 < coarse.values.shape[0])
    east = np.zeros((coarse.values.shape[0], target.values.shape[1]), dtype=np.float64)
    east_valid = np.zeros_like(east, dtype=np.bool_)
    good_x = np.flatnonzero(x_ok)
    east_valid[:, good_x] = True
    for slot, delta in enumerate((-1, 0, 1, 2)):
        indices = base_x[good_x] + delta
        selected = coarse.valid[:, indices]
        east[:, good_x] += wx[good_x, slot] * np.where(selected, coarse.values[:, indices], 0.0)
        east_valid[:, good_x] &= selected
    values = np.zeros(target.values.shape, dtype=np.float64)
    valid = np.zeros(target.values.shape, dtype=np.bool_)
    good_y = np.flatnonzero(y_ok)
    valid[good_y] = True
    for slot, delta in enumerate((-1, 0, 1, 2)):
        indices = base_y[good_y] + delta
        selected = east_valid[indices]
        values[good_y] += wy[good_y, slot, None] * np.where(selected, east[indices], 0.0)
        valid[good_y] &= selected
    valid &= target.valid
    values[~valid] = np.nan
    return RegularGrid(values, valid, target.x_bounds_m.copy(), target.y_bounds_m.copy(), role)


def analyze_phase(h2: RegularGrid, phase_xy_m: tuple[float, float]) -> PhaseBands:
    a1 = filter4(h2, "A1=F4(H2)")
    a0 = filter4(a1, "A0=F4(A1)")
    r4_a1 = reconstruct4(a1, h2, "R4(A1)")
    r4_a0 = reconstruct4(a0, a1, "R4(A0)")
    b2_valid = h2.valid & r4_a1.valid
    b1_valid = a1.valid & r4_a0.valid
    b2 = np.where(b2_valid, h2.values - r4_a1.values, np.nan)
    b1 = np.where(b1_valid, a1.values - r4_a0.values, np.nan)
    return PhaseBands(
        phase_xy_m,
        h2,
        a1,
        a0,
        RegularGrid(b1, b1_valid, a1.x_bounds_m.copy(), a1.y_bounds_m.copy(), "B1"),
        RegularGrid(b2, b2_valid, h2.x_bounds_m.copy(), h2.y_bounds_m.copy(), "B2"),
    )


def integrate_grid(
    grid: RegularGrid,
    core_bounds: tuple[float, float, float, float],
) -> GridIntegral:
    """Integrate a constant-cell grid over its exact intersection with a core."""
    min_x, min_y, max_x, max_y = core_bounds
    x_lengths = np.maximum(
        np.minimum(grid.x_bounds_m[1:], max_x) - np.maximum(grid.x_bounds_m[:-1], min_x),
        0.0,
    )
    y_lengths = np.maximum(
        np.minimum(grid.y_bounds_m[1:], max_y) - np.maximum(grid.y_bounds_m[:-1], min_y),
        0.0,
    )
    full_x = np.isclose(x_lengths, grid.cell_m, rtol=0.0, atol=1e-10)
    full_y = np.isclose(y_lengths, grid.cell_m, rtol=0.0, atol=1e-10)
    total_area = 0.0
    sum_value = 0.0
    sum_squared = 0.0
    complete_cells = 0
    minimum: float | None = None
    maximum: float | None = None
    for row, dy in enumerate(y_lengths):
        if dy <= 0.0:
            continue
        keep = grid.valid[row] & (x_lengths > 0.0)
        if not np.any(keep):
            continue
        area = dy * x_lengths[keep]
        values = grid.values[row, keep]
        total_area += float(np.sum(area, dtype=np.float64))
        sum_value += float(np.sum(area * values, dtype=np.float64))
        sum_squared += float(np.sum(area * values * values, dtype=np.float64))
        complete_cells += int(np.count_nonzero(keep & full_x)) if full_y[row] else 0
        row_min = float(np.min(values))
        row_max = float(np.max(values))
        minimum = row_min if minimum is None else min(minimum, row_min)
        maximum = row_max if maximum is None else max(maximum, row_max)
    return GridIntegral(total_area, complete_cells, sum_value, sum_squared, minimum, maximum)


def _overlay_bounds(
    grids: tuple[RegularGrid, RegularGrid, RegularGrid, RegularGrid],
    axis: str,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    arrays = []
    for grid in grids:
        bounds = grid.x_bounds_m if axis == "x" else grid.y_bounds_m
        arrays.append(bounds[(bounds > minimum) & (bounds < maximum)])
    return np.unique(np.concatenate((np.asarray([minimum, maximum]), *arrays)))


def integrate_common_phases(
    grids: tuple[RegularGrid, RegularGrid, RegularGrid, RegularGrid],
    core_bounds: tuple[float, float, float, float],
) -> CommonPhaseIntegral:
    """Compare four phases by exact cell-overlap integration on common support."""
    min_x, min_y, max_x, max_y = core_bounds
    x_bounds = _overlay_bounds(grids, "x", min_x, max_x)
    y_bounds = _overlay_bounds(grids, "y", min_y, max_y)
    x_mid = 0.5 * (x_bounds[:-1] + x_bounds[1:])
    y_mid = 0.5 * (y_bounds[:-1] + y_bounds[1:])
    dx = np.diff(x_bounds)
    dy = np.diff(y_bounds)
    x_indices = tuple(
        np.searchsorted(grid.x_bounds_m, x_mid, side="right") - 1 for grid in grids
    )
    y_indices = tuple(
        np.searchsorted(grid.y_bounds_m, y_mid, side="right") - 1 for grid in grids
    )
    x_inside = tuple(
        (indices >= 0) & (indices < grid.values.shape[1])
        for indices, grid in zip(x_indices, grids, strict=True)
    )
    y_inside = tuple(
        (indices >= 0) & (indices < grid.values.shape[0])
        for indices, grid in zip(y_indices, grids, strict=True)
    )
    phase_energy = np.zeros(4, dtype=np.float64)
    pair_energy = np.zeros(6, dtype=np.float64)
    total_area = 0.0
    for overlay_row, row_height in enumerate(dy):
        if not all(inside[overlay_row] for inside in y_inside):
            continue
        values = []
        common = np.ones(x_mid.size, dtype=np.bool_)
        for phase, grid in enumerate(grids):
            common &= x_inside[phase]
            clipped_x = np.clip(x_indices[phase], 0, grid.values.shape[1] - 1)
            source_row = y_indices[phase][overlay_row]
            common &= grid.valid[source_row, clipped_x]
            values.append(grid.values[source_row, clipped_x])
        if not np.any(common):
            continue
        area = row_height * dx[common]
        total_area += float(np.sum(area, dtype=np.float64))
        for phase in range(4):
            phase_values = values[phase][common]
            phase_energy[phase] += np.sum(area * phase_values * phase_values, dtype=np.float64)
        pair = 0
        for left in range(4):
            for right in range(left + 1, 4):
                difference = values[left][common] - values[right][common]
                pair_energy[pair] += np.sum(area * difference * difference, dtype=np.float64)
                pair += 1
    return CommonPhaseIntegral(
        total_area,
        tuple(float(value) for value in phase_energy),
        tuple(float(value) for value in pair_energy),
    )
