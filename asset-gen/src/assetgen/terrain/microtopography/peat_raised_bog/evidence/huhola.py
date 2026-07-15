"""Pinned 0.50 m HuHoLa classification over measured Moore support."""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion


SOURCE_CELL_M = 0.01
CLASS_CELL_M = 0.50
SOURCE_CELLS_PER_CLASS_CELL = 50
FILL_THRESHOLD_M = 0.04
CONNECTIVITY = np.ones((3, 3), dtype=np.bool_)
NEIGHBORS = tuple(
    (dy, dx)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if dy != 0 or dx != 0
)

CLASS_LAWN = 0
CLASS_HOLLOW = 1
CLASS_HUMMOCK = 2
CLASS_NAMES = {
    CLASS_LAWN: "lawn",
    CLASS_HOLLOW: "hollow",
    CLASS_HUMMOCK: "hummock",
}


@dataclass(frozen=True)
class ClassifiedPlot:
    height_m: np.ndarray
    valid: np.ndarray
    interior_valid: np.ndarray
    hollow_fill_m: np.ndarray
    hummock_fill_m: np.ndarray
    hhdh_m: np.ndarray
    classes: np.ndarray
    x_bounds_m: np.ndarray
    y_bounds_m: np.ndarray


def _aggregate_complete_50cm(
    source_height_m: np.ndarray,
    source_valid: np.ndarray,
    source_x_centers_m: np.ndarray,
    source_y_centers_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Area-mean exact 50x50 source cells; partial or nonfinite cells get no credit."""
    height = np.asarray(source_height_m, dtype=np.float64)
    valid = np.asarray(source_valid, dtype=np.bool_)
    x = np.asarray(source_x_centers_m, dtype=np.float64)
    y = np.asarray(source_y_centers_m, dtype=np.float64)
    if height.ndim != 2 or height.shape != valid.shape or height.shape != (y.size, x.size):
        raise ValueError("Moore source arrays do not form one rectilinear grid")
    if x.size < 2 or y.size < 2:
        raise ValueError("Moore source grid is too small")
    if not np.allclose(np.diff(x), SOURCE_CELL_M, rtol=0, atol=2e-12):
        raise ValueError("Moore x spacing changed")
    if not np.allclose(np.diff(y), SOURCE_CELL_M, rtol=0, atol=2e-12):
        raise ValueError("Moore y spacing changed")
    ny = height.shape[0] // SOURCE_CELLS_PER_CLASS_CELL
    nx = height.shape[1] // SOURCE_CELLS_PER_CLASS_CELL
    if ny == 0 or nx == 0:
        raise ValueError("Moore source grid cannot contain a complete 0.50 m cell")
    trimmed_height = height[
        : ny * SOURCE_CELLS_PER_CLASS_CELL,
        : nx * SOURCE_CELLS_PER_CLASS_CELL,
    ]
    trimmed_valid = valid[
        : ny * SOURCE_CELLS_PER_CLASS_CELL,
        : nx * SOURCE_CELLS_PER_CLASS_CELL,
    ]
    blocks = trimmed_height.reshape(
        ny,
        SOURCE_CELLS_PER_CLASS_CELL,
        nx,
        SOURCE_CELLS_PER_CLASS_CELL,
    )
    support = trimmed_valid.reshape(
        ny,
        SOURCE_CELLS_PER_CLASS_CELL,
        nx,
        SOURCE_CELLS_PER_CLASS_CELL,
    )
    complete = np.all(support, axis=(1, 3))
    values = np.mean(np.where(support, blocks, 0.0), axis=(1, 3), dtype=np.float64)
    values[~complete] = np.nan
    x_origin = float(x[0] - 0.5 * SOURCE_CELL_M)
    y_origin = float(y[0] - 0.5 * SOURCE_CELL_M)
    x_bounds = x_origin + np.arange(nx + 1, dtype=np.float64) * CLASS_CELL_M
    y_bounds = y_origin + np.arange(ny + 1, dtype=np.float64) * CLASS_CELL_M
    return values, complete, x_bounds, y_bounds


def _is_support_boundary(valid: np.ndarray, row: int, column: int) -> bool:
    ny, nx = valid.shape
    for dy, dx in NEIGHBORS:
        rr, cc = row + dy, column + dx
        if rr < 0 or rr >= ny or cc < 0 or cc >= nx or not valid[rr, cc]:
            return True
    return False


def _priority_fill(height_m: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Deterministic 8-neighbor priority flood, equivalent to fill without flat fixing."""
    filled = np.asarray(height_m, dtype=np.float64).copy()
    measured = np.asarray(valid, dtype=np.bool_)
    visited = np.zeros(measured.shape, dtype=np.bool_)
    queue: list[tuple[float, int, int]] = []
    for row, column in zip(*np.nonzero(measured), strict=True):
        if _is_support_boundary(measured, int(row), int(column)):
            visited[row, column] = True
            queue.append((float(filled[row, column]), int(row), int(column)))
    heapq.heapify(queue)
    while queue:
        spill, row, column = heapq.heappop(queue)
        for dy, dx in NEIGHBORS:
            rr, cc = row + dy, column + dx
            if (
                rr < 0
                or rr >= measured.shape[0]
                or cc < 0
                or cc >= measured.shape[1]
                or not measured[rr, cc]
                or visited[rr, cc]
            ):
                continue
            visited[rr, cc] = True
            filled[rr, cc] = max(float(filled[rr, cc]), spill)
            heapq.heappush(queue, (float(filled[rr, cc]), rr, cc))
    if np.any(measured & ~visited):
        raise AssertionError("priority flood did not visit complete support")
    filled[~measured] = np.nan
    return filled


def classify_plot(
    source_height_m: np.ndarray,
    source_valid: np.ndarray,
    source_x_centers_m: np.ndarray,
    source_y_centers_m: np.ndarray,
) -> ClassifiedPlot:
    height, valid, x_bounds, y_bounds = _aggregate_complete_50cm(
        source_height_m,
        source_valid,
        source_x_centers_m,
        source_y_centers_m,
    )
    if not np.any(valid):
        empty = np.full(height.shape, np.nan, dtype=np.float64)
        return ClassifiedPlot(
            height,
            valid,
            np.zeros_like(valid),
            empty.copy(),
            empty.copy(),
            empty.copy(),
            np.full(height.shape, -1, dtype=np.int8),
            x_bounds,
            y_bounds,
        )
    filled = _priority_fill(height, valid)
    inverted = np.nanmax(height[valid]) - height
    inverted_filled = _priority_fill(inverted, valid)
    hollow_fill = filled - height
    hummock_fill = inverted_filled - inverted
    hhdh = hollow_fill - hummock_fill
    classes = np.full(height.shape, -1, dtype=np.int8)
    classes[valid] = CLASS_LAWN
    classes[valid & (hhdh > FILL_THRESHOLD_M)] = CLASS_HOLLOW
    classes[valid & (hhdh < -FILL_THRESHOLD_M)] = CLASS_HUMMOCK
    # HuHoLa's explicit nested-feature rule overrides the threshold classes.
    both = valid & (hollow_fill > 0.0) & (hummock_fill > 0.0)
    classes[both & (hollow_fill > hummock_fill)] = CLASS_HOLLOW
    classes[both & (hummock_fill >= hollow_fill)] = CLASS_HUMMOCK
    interior = binary_erosion(valid, structure=CONNECTIVITY, border_value=0, iterations=1)
    for values in (hollow_fill, hummock_fill, hhdh):
        values[~valid] = np.nan
    return ClassifiedPlot(
        height,
        valid,
        interior,
        hollow_fill,
        hummock_fill,
        hhdh,
        classes,
        x_bounds,
        y_bounds,
    )
