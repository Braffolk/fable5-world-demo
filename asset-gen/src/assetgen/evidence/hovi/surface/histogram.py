"""Bounded streaming vertical histograms for merged Hovi point observations."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import laspy
import numpy as np

from .model import GridSpec


@dataclass(frozen=True)
class VerticalHistogram:
    grid: GridSpec
    z_origin_m: float
    z_bin_m: float
    counts: np.ndarray
    decoded_points: int
    included_points: int


@dataclass(frozen=True)
class WindowedVerticalHistogram:
    grid: GridSpec
    reference_z_m: np.ndarray
    half_window_m: float
    z_bin_m: float
    counts: np.ndarray
    below_window_count: np.ndarray
    above_window_count: np.ndarray
    decoded_points: int
    referenced_points: int
    included_points: int


@dataclass(frozen=True)
class SelectedModeFootprint:
    selected_point_count: np.ndarray
    quadrant_count: np.ndarray
    subcell_count: np.ndarray
    decoded_points: int
    candidate_points: int
    included_points: int


def grid_from_bounds(
    resolution_m: float,
    min_x_m: float,
    min_y_m: float,
    max_x_m: float,
    max_y_m: float,
) -> GridSpec:
    origin_ix = math.floor(min_x_m / resolution_m)
    origin_iy = math.floor(min_y_m / resolution_m)
    max_ix = math.floor(max_x_m / resolution_m)
    max_iy = math.floor(max_y_m / resolution_m)
    return GridSpec(
        resolution_m=resolution_m,
        origin_x_m=origin_ix * resolution_m,
        origin_y_m=origin_iy * resolution_m,
        width=max_ix - origin_ix + 1,
        height=max_iy - origin_iy + 1,
    )


def cell_indices(grid: GridSpec, x_m: np.ndarray, y_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ix = np.floor((x_m - grid.origin_x_m) / grid.resolution_m).astype(np.int64)
    iy = np.floor((y_m - grid.origin_y_m) / grid.resolution_m).astype(np.int64)
    inside = (ix >= 0) & (ix < grid.width) & (iy >= 0) & (iy < grid.height)
    return iy * grid.width + ix, inside


def map_grid_centers(child: GridSpec, parent: GridSpec) -> np.ndarray:
    x = child.origin_x_m + (np.arange(child.width, dtype=np.float64) + 0.5) * child.resolution_m
    y = child.origin_y_m + (np.arange(child.height, dtype=np.float64) + 0.5) * child.resolution_m
    ix = np.floor((x - parent.origin_x_m) / parent.resolution_m).astype(np.int64)
    iy = np.floor((y - parent.origin_y_m) / parent.resolution_m).astype(np.int64)
    parent_index = iy[:, None] * parent.width + ix[None, :]
    valid = (
        (ix[None, :] >= 0)
        & (ix[None, :] < parent.width)
        & (iy[:, None] >= 0)
        & (iy[:, None] < parent.height)
    )
    parent_index[~valid] = -1
    return parent_index


def _add_unique(flat_counts: np.ndarray, keys: np.ndarray) -> None:
    if keys.size == 0:
        return
    unique, increments = np.unique(keys, return_counts=True)
    current = flat_counts[unique].astype(np.uint64)
    updated = current + increments.astype(np.uint64)
    if np.any(updated > np.iinfo(flat_counts.dtype).max):
        raise OverflowError("Hovi vertical histogram count overflow")
    flat_counts[unique] = updated.astype(flat_counts.dtype)


def _add_cell_counts(target: np.ndarray, cells: np.ndarray) -> None:
    if cells.size == 0:
        return
    unique, increments = np.unique(cells, return_counts=True)
    target[unique] += increments.astype(target.dtype)


def accumulate_structural_histogram(
    laz_path: Path,
    *,
    xy_resolution_m: float,
    z_bin_m: float,
    chunk_points: int,
    memory_limit_bytes: int,
    log: Callable[[str], None] = print,
) -> VerticalHistogram:
    """Accumulate all finite observations; no vertical rank or point class is selected."""
    with laspy.open(laz_path) as reader:
        header = reader.header
        mins = np.asarray(header.mins, dtype=np.float64)
        maxs = np.asarray(header.maxs, dtype=np.float64)
        grid = grid_from_bounds(xy_resolution_m, mins[0], mins[1], maxs[0], maxs[1])
        z_origin_bin = math.floor(float(mins[2]) / z_bin_m)
        z_max_bin = math.floor(float(maxs[2]) / z_bin_m)
        z_bins = z_max_bin - z_origin_bin + 1
        required = grid.cell_count * z_bins * np.dtype(np.uint32).itemsize
        if required > memory_limit_bytes:
            raise ValueError(
                f"Hovi structural histogram requires {required / (1 << 30):.2f} GiB, "
                "above the frozen memory limit"
            )
        counts = np.zeros((grid.cell_count, z_bins), dtype=np.uint32)
        flat_counts = counts.ravel()
        decoded = 0
        included = 0
        for chunk_index, points in enumerate(reader.chunk_iterator(chunk_points), start=1):
            x = np.asarray(points.x, dtype=np.float64)
            y = np.asarray(points.y, dtype=np.float64)
            z = np.asarray(points.z, dtype=np.float64)
            decoded += z.size
            finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            cells, inside = cell_indices(grid, x, y)
            z_index = np.floor(z / z_bin_m).astype(np.int64) - z_origin_bin
            valid = finite & inside & (z_index >= 0) & (z_index < z_bins)
            keys = cells[valid] * z_bins + z_index[valid]
            _add_unique(flat_counts, keys)
            included += int(np.count_nonzero(valid))
            if chunk_index == 1 or chunk_index % 25 == 0:
                log(f"Hovi structural modes: {decoded:,}/{int(header.point_count):,} points")
        if decoded != int(header.point_count) or included != decoded:
            raise ValueError(
                f"Hovi structural histogram excluded observations: decoded={decoded}, "
                f"included={included}, header={int(header.point_count)}"
            )
        return VerticalHistogram(
            grid=grid,
            z_origin_m=z_origin_bin * z_bin_m,
            z_bin_m=z_bin_m,
            counts=counts,
            decoded_points=decoded,
            included_points=included,
        )


def aggregate_histogram(source: VerticalHistogram, resolution_m: float) -> VerticalHistogram:
    parent = grid_from_bounds(
        resolution_m,
        source.grid.origin_x_m,
        source.grid.origin_y_m,
        source.grid.upper_x_m - 0.5 * source.grid.resolution_m,
        source.grid.upper_y_m - 0.5 * source.grid.resolution_m,
    )
    mapping = map_grid_centers(source.grid, parent).ravel()
    output = np.zeros((parent.cell_count, source.counts.shape[1]), dtype=np.uint32)
    for parent_cell in range(parent.cell_count):
        children = np.flatnonzero(mapping == parent_cell)
        if children.size:
            summed = source.counts[children].sum(axis=0, dtype=np.uint64)
            if np.any(summed > np.iinfo(np.uint32).max):
                raise OverflowError("Hovi aggregate vertical histogram count overflow")
            output[parent_cell] = summed.astype(np.uint32)
    return VerticalHistogram(
        grid=parent,
        z_origin_m=source.z_origin_m,
        z_bin_m=source.z_bin_m,
        counts=output,
        decoded_points=source.decoded_points,
        included_points=source.included_points,
    )


def accumulate_tangent_sheared_histogram(
    laz_path: Path,
    *,
    grid: GridSpec,
    slope_x: np.ndarray,
    slope_y: np.ndarray,
    z_bin_m: float,
    maximum_slope: float,
    chunk_points: int,
    memory_limit_bytes: int,
    stage_name: str,
    log: Callable[[str], None] = print,
) -> VerticalHistogram:
    """Accumulate full-height modes after projecting observations to cell centers."""
    tangent_x = np.asarray(slope_x, dtype=np.float32).reshape(-1)
    tangent_y = np.asarray(slope_y, dtype=np.float32).reshape(-1)
    if tangent_x.size != grid.cell_count or tangent_y.size != grid.cell_count:
        raise ValueError("Hovi tangent-sheared slopes do not match their grid")
    with laspy.open(laz_path) as reader:
        header = reader.header
        allowance = maximum_slope * grid.resolution_m / np.sqrt(2.0)
        z_origin_bin = math.floor((float(header.mins[2]) - allowance) / z_bin_m)
        z_max_bin = math.floor((float(header.maxs[2]) + allowance) / z_bin_m)
        z_bins = z_max_bin - z_origin_bin + 1
        required = grid.cell_count * z_bins * np.dtype(np.uint32).itemsize
        if required > memory_limit_bytes:
            raise ValueError(
                f"Hovi {stage_name} tangent-sheared histogram requires "
                f"{required / (1 << 30):.2f} GiB, above the frozen memory limit"
            )
        counts = np.zeros((grid.cell_count, z_bins), dtype=np.uint32)
        flat_counts = counts.ravel()
        decoded = included = 0
        for chunk_index, points in enumerate(reader.chunk_iterator(chunk_points), start=1):
            x = np.asarray(points.x, dtype=np.float64)
            y = np.asarray(points.y, dtype=np.float64)
            z = np.asarray(points.z, dtype=np.float64)
            decoded += z.size
            finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            cells, inside = cell_indices(grid, x, y)
            safe_cells = np.clip(cells, 0, grid.cell_count - 1)
            valid = (
                finite
                & inside
                & np.isfinite(tangent_x[safe_cells])
                & np.isfinite(tangent_y[safe_cells])
            )
            selected_cells = cells[valid]
            center_x = (
                grid.origin_x_m
                + (selected_cells % grid.width + 0.5) * grid.resolution_m
            )
            center_y = (
                grid.origin_y_m
                + (selected_cells // grid.width + 0.5) * grid.resolution_m
            )
            centered_z = (
                z[valid]
                - tangent_x[selected_cells] * (x[valid] - center_x)
                - tangent_y[selected_cells] * (y[valid] - center_y)
            )
            z_index = np.floor(centered_z / z_bin_m).astype(np.int64) - z_origin_bin
            valid_bin = (z_index >= 0) & (z_index < z_bins)
            keys = selected_cells[valid_bin] * z_bins + z_index[valid_bin]
            _add_unique(flat_counts, keys)
            included += int(np.count_nonzero(valid_bin))
            if chunk_index == 1 or chunk_index % 25 == 0:
                log(
                    f"Hovi {stage_name} tangent-sheared modes: "
                    f"{decoded:,}/{int(header.point_count):,} points"
                )
        if decoded != int(header.point_count):
            raise ValueError(
                f"Hovi {stage_name} tangent-sheared decoded count differs from LAS header"
            )
        return VerticalHistogram(
            grid=grid,
            z_origin_m=z_origin_bin * z_bin_m,
            z_bin_m=z_bin_m,
            counts=counts,
            decoded_points=decoded,
            included_points=included,
        )


def accumulate_windowed_histogram(
    laz_path: Path,
    *,
    grid: GridSpec,
    reference_z_m: np.ndarray,
    slope_x: np.ndarray,
    slope_y: np.ndarray,
    half_window_m: float,
    z_bin_m: float,
    chunk_points: int,
    memory_limit_bytes: int,
    log: Callable[[str], None] = print,
) -> WindowedVerticalHistogram:
    """Accumulate modes around a selected structural sheet, preserving unsupported cells."""
    reference = np.asarray(reference_z_m, dtype=np.float32).reshape(-1)
    tangent_x = np.asarray(slope_x, dtype=np.float32).reshape(-1)
    tangent_y = np.asarray(slope_y, dtype=np.float32).reshape(-1)
    if reference.size != grid.cell_count:
        raise ValueError("Hovi fine reference does not match the final grid")
    if tangent_x.size != grid.cell_count or tangent_y.size != grid.cell_count:
        raise ValueError("Hovi fine tangent does not match the final grid")
    z_bins = int(round(2.0 * half_window_m / z_bin_m)) + 1
    required = grid.cell_count * z_bins * np.dtype(np.uint32).itemsize
    if required > memory_limit_bytes:
        raise ValueError(
            f"Hovi fine histogram requires {required / (1 << 30):.2f} GiB, "
            "above the frozen memory limit"
        )
    counts = np.zeros((grid.cell_count, z_bins), dtype=np.uint32)
    below = np.zeros(grid.cell_count, dtype=np.uint64)
    above = np.zeros(grid.cell_count, dtype=np.uint64)
    flat_counts = counts.ravel()
    decoded = referenced = included = 0
    with laspy.open(laz_path) as reader:
        header_count = int(reader.header.point_count)
        for chunk_index, points in enumerate(reader.chunk_iterator(chunk_points), start=1):
            x = np.asarray(points.x, dtype=np.float64)
            y = np.asarray(points.y, dtype=np.float64)
            z = np.asarray(points.z, dtype=np.float64)
            decoded += z.size
            finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            cells, inside = cell_indices(grid, x, y)
            valid_cell = finite & inside
            valid_cell &= np.isfinite(reference[np.clip(cells, 0, reference.size - 1)])
            valid_cell &= np.isfinite(tangent_x[np.clip(cells, 0, reference.size - 1)])
            valid_cell &= np.isfinite(tangent_y[np.clip(cells, 0, reference.size - 1)])
            selected_cells = cells[valid_cell]
            selected_z = z[valid_cell]
            selected_x = x[valid_cell]
            selected_y = y[valid_cell]
            center_x = (
                grid.origin_x_m
                + (selected_cells % grid.width + 0.5) * grid.resolution_m
            )
            center_y = (
                grid.origin_y_m
                + (selected_cells // grid.width + 0.5) * grid.resolution_m
            )
            tangent_reference = (
                reference[selected_cells]
                + tangent_x[selected_cells] * (selected_x - center_x)
                + tangent_y[selected_cells] * (selected_y - center_y)
            )
            delta = selected_z - tangent_reference
            referenced += selected_cells.size
            below_mask = delta < -half_window_m
            above_mask = delta > half_window_m + 0.5 * z_bin_m
            _add_cell_counts(below, selected_cells[below_mask])
            _add_cell_counts(above, selected_cells[above_mask])
            within = ~(below_mask | above_mask)
            window_cells = selected_cells[within]
            z_index = np.floor(
                (delta[within] + half_window_m) / z_bin_m
            ).astype(np.int64)
            valid_bin = (z_index >= 0) & (z_index < z_bins)
            keys = window_cells[valid_bin] * z_bins + z_index[valid_bin]
            _add_unique(flat_counts, keys)
            included += int(np.count_nonzero(valid_bin))
            if chunk_index == 1 or chunk_index % 25 == 0:
                log(f"Hovi fine modes: {decoded:,}/{header_count:,} points")
        if decoded != header_count:
            raise ValueError("Hovi fine histogram decoded count differs from LAS header")
        if int(counts.sum(dtype=np.uint64) + below.sum() + above.sum()) != referenced:
            raise ValueError("Hovi fine histogram does not conserve referenced observations")
    return WindowedVerticalHistogram(
        grid=grid,
        reference_z_m=reference.reshape(grid.height, grid.width),
        half_window_m=half_window_m,
        z_bin_m=z_bin_m,
        counts=counts,
        below_window_count=below.reshape(grid.height, grid.width),
        above_window_count=above.reshape(grid.height, grid.width),
        decoded_points=decoded,
        referenced_points=referenced,
        included_points=included,
    )


def accumulate_selected_mode_footprint(
    laz_path: Path,
    *,
    grid: GridSpec,
    selected_center_z_m: np.ndarray,
    slope_x: np.ndarray,
    slope_y: np.ndarray,
    half_band_m: float,
    subcell_axis: int,
    chunk_points: int,
    log: Callable[[str], None] = print,
) -> SelectedModeFootprint:
    """Measure within-cell XY support for the already selected tangent-mode sheet."""
    if subcell_axis <= 0 or subcell_axis * subcell_axis > 16:
        raise ValueError("Hovi footprint subcell axis must fit a uint16 occupancy mask")
    selected = np.asarray(selected_center_z_m, dtype=np.float32).reshape(-1)
    tangent_x = np.asarray(slope_x, dtype=np.float32).reshape(-1)
    tangent_y = np.asarray(slope_y, dtype=np.float32).reshape(-1)
    if selected.size != grid.cell_count or tangent_x.size != grid.cell_count or tangent_y.size != grid.cell_count:
        raise ValueError("Hovi selected footprint inputs do not match the final grid")
    selected_count = np.zeros(grid.cell_count, dtype=np.uint32)
    quadrant_bits = np.zeros(grid.cell_count, dtype=np.uint8)
    subcell_bits = np.zeros(grid.cell_count, dtype=np.uint16)
    decoded = candidate_points = included = 0
    with laspy.open(laz_path) as reader:
        header_count = int(reader.header.point_count)
        for chunk_index, points in enumerate(reader.chunk_iterator(chunk_points), start=1):
            x = np.asarray(points.x, dtype=np.float64)
            y = np.asarray(points.y, dtype=np.float64)
            z = np.asarray(points.z, dtype=np.float64)
            decoded += z.size
            finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            cells, inside = cell_indices(grid, x, y)
            safe_cells = np.clip(cells, 0, selected.size - 1)
            candidate = (
                finite
                & inside
                & np.isfinite(selected[safe_cells])
                & np.isfinite(tangent_x[safe_cells])
                & np.isfinite(tangent_y[safe_cells])
            )
            candidate_cells = cells[candidate]
            candidate_x = x[candidate]
            candidate_y = y[candidate]
            candidate_z = z[candidate]
            candidate_points += candidate_cells.size
            lower_x = grid.origin_x_m + (candidate_cells % grid.width) * grid.resolution_m
            lower_y = grid.origin_y_m + (candidate_cells // grid.width) * grid.resolution_m
            local_x = np.clip((candidate_x - lower_x) / grid.resolution_m, 0.0, 1.0 - 1e-12)
            local_y = np.clip((candidate_y - lower_y) / grid.resolution_m, 0.0, 1.0 - 1e-12)
            expected = (
                selected[candidate_cells]
                + tangent_x[candidate_cells] * ((local_x - 0.5) * grid.resolution_m)
                + tangent_y[candidate_cells] * ((local_y - 0.5) * grid.resolution_m)
            )
            accepted = np.abs(candidate_z - expected) <= half_band_m
            accepted_cells = candidate_cells[accepted]
            accepted_x = local_x[accepted]
            accepted_y = local_y[accepted]
            _add_cell_counts(selected_count, accepted_cells)
            quadrant = (accepted_y >= 0.5).astype(np.uint8) * 2 + (accepted_x >= 0.5).astype(np.uint8)
            np.bitwise_or.at(quadrant_bits, accepted_cells, np.left_shift(np.uint8(1), quadrant))
            sub_x = np.floor(accepted_x * subcell_axis).astype(np.uint16)
            sub_y = np.floor(accepted_y * subcell_axis).astype(np.uint16)
            subcell = sub_y * subcell_axis + sub_x
            np.bitwise_or.at(
                subcell_bits,
                accepted_cells,
                np.left_shift(np.uint16(1), subcell),
            )
            included += accepted_cells.size
            if chunk_index == 1 or chunk_index % 25 == 0:
                log(f"Hovi selected-mode footprint: {decoded:,}/{header_count:,} points")
        if decoded != header_count:
            raise ValueError("Hovi selected footprint decoded count differs from LAS header")
    quadrant_count = np.fromiter(
        (int(value).bit_count() for value in quadrant_bits),
        dtype=np.uint8,
        count=grid.cell_count,
    )
    subcell_count = np.fromiter(
        (int(value).bit_count() for value in subcell_bits),
        dtype=np.uint8,
        count=grid.cell_count,
    )
    return SelectedModeFootprint(
        selected_point_count=selected_count.reshape(grid.height, grid.width),
        quadrant_count=quadrant_count.reshape(grid.height, grid.width),
        subcell_count=subcell_count.reshape(grid.height, grid.width),
        decoded_points=decoded,
        candidate_points=candidate_points,
        included_points=included,
    )
