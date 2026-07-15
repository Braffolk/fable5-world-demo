"""Evidence-grid drainage domains; this module does not synthesize terrain."""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class DrainageDomain:
    upstream: np.ndarray
    outlet: np.ndarray
    collar: np.ndarray
    solve_domain: np.ndarray
    filled_height: np.ndarray
    fill_depth: np.ndarray
    edge_leak: np.ndarray


_NEIGHBORS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _seed(
    heap: list[tuple[float, int, int, int, int]],
    labels: np.ndarray,
    filled: np.ndarray,
    valid: np.ndarray,
    mask: np.ndarray,
    label: int,
    priority: int,
) -> None:
    rows, cols = np.nonzero(mask & valid & (labels < 0))
    for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
        labels[row, col] = label
        heapq.heappush(
            heap,
            (float(filled[row, col]), priority, row, col, label),
        )


def delineate_drainage_domain(
    height: np.ndarray,
    valid: np.ndarray,
    target_outlet: np.ndarray,
    other_outlet: np.ndarray,
    *,
    collar_cells: int,
) -> DrainageDomain:
    """Partition a bounded evidence grid by priority-flood outlet ownership.

    Target outlet cells have precedence only on the same cell. Outer-boundary and
    non-target water seeds compete by spill elevation, so a target basin that
    reaches the evidence edge is detected rather than accepted as complete.
    """
    if height.ndim != 2 or any(
        array.shape != height.shape
        for array in (valid, target_outlet, other_outlet)
    ):
        raise ValueError("drainage arrays must be same-shape two-dimensional grids")
    if collar_cells < 1:
        raise ValueError("drainage collar must contain at least one evidence cell")
    valid = valid.astype(bool, copy=False) & np.isfinite(height)
    target_outlet = target_outlet.astype(bool, copy=False) & valid
    other_outlet = other_outlet.astype(bool, copy=False) & valid & ~target_outlet
    if not target_outlet.any():
        raise ValueError("target outlet has no valid evidence cells")

    labels = np.full(height.shape, -1, dtype=np.int8)
    filled = np.asarray(height, dtype=np.float64).copy()
    heap: list[tuple[float, int, int, int, int]] = []
    _seed(heap, labels, filled, valid, target_outlet, 1, 0)
    _seed(heap, labels, filled, valid, other_outlet, 0, 1)
    boundary = np.zeros(height.shape, dtype=bool)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    _seed(heap, labels, filled, valid, boundary, 0, 2)

    rows, cols = height.shape
    while heap:
        level, _priority, row, col, label = heapq.heappop(heap)
        for drow, dcol in _NEIGHBORS:
            next_row = row + drow
            next_col = col + dcol
            if not (0 <= next_row < rows and 0 <= next_col < cols):
                continue
            if not valid[next_row, next_col] or labels[next_row, next_col] >= 0:
                continue
            labels[next_row, next_col] = label
            next_level = max(level, float(filled[next_row, next_col]))
            filled[next_row, next_col] = next_level
            heapq.heappush(
                heap,
                (next_level, 3, next_row, next_col, label),
            )

    upstream = (labels == 1) & valid & ~target_outlet
    edge_leak = upstream & boundary
    collar = ndimage.binary_dilation(
        upstream | target_outlet,
        iterations=collar_cells,
        border_value=0,
    ) & valid & ~upstream & ~target_outlet
    solve_domain = upstream | target_outlet | collar
    fill_depth = np.zeros(height.shape, dtype=np.float32)
    fill_depth[valid] = (filled[valid] - height[valid]).astype(np.float32)
    filled[~valid] = np.nan
    return DrainageDomain(
        upstream=upstream,
        outlet=target_outlet,
        collar=collar,
        solve_domain=solve_domain,
        filled_height=filled.astype(np.float32),
        fill_depth=fill_depth,
        edge_leak=edge_leak,
    )
