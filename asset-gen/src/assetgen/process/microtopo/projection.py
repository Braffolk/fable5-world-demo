"""AOI-wide smooth projection of measured residuals into the LOD0 mean nullspace."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import ndimage
from scipy.interpolate import CubicSpline
from scipy.linalg import solve


@lru_cache(maxsize=8)
def _cell_mean_operator(cells: int, factor: int) -> np.ndarray:
    """Map coarse knot values to exact means of a C2 cubic reconstruction."""
    if cells < 4 or factor < 2:
        raise ValueError("smooth projection requires at least four cells and factor >= 2")
    knots = np.arange(cells, dtype=np.float64) + 0.5
    samples = (np.arange(cells * factor, dtype=np.float64) + 0.5) / factor
    basis = CubicSpline(knots, np.eye(cells), axis=0)(samples)
    operator = basis.reshape(cells, factor, cells).mean(axis=1)
    operator.setflags(write=False)
    return operator


@dataclass(frozen=True)
class SmoothMeanNullProjector:
    """One immutable AOI solution evaluated identically by every chunk crop."""

    coefficients: np.ndarray
    soft_distance_cells: np.ndarray
    factor: int
    zero_fringe_cells: float
    taper_cells: float

    def taper_window(
        self, row0: int, col0: int, cell_rows: int, cell_cols: int
    ) -> np.ndarray:
        """C2 fine-sample taper derived from one AOI-wide mask distance field."""
        if np.isinf(self.soft_distance_cells).all():
            return np.ones(
                (cell_rows * self.factor, cell_cols * self.factor), dtype=np.float64
            )
        fy = row0 + (np.arange(cell_rows * self.factor) + 0.5) / self.factor - 0.5
        fx = col0 + (np.arange(cell_cols * self.factor) + 0.5) / self.factor - 0.5
        distance = ndimage.map_coordinates(
            self.soft_distance_cells,
            np.meshgrid(fy, fx, indexing="ij"),
            order=1,
            mode="nearest",
            prefilter=False,
        )
        u = np.clip(
            (distance - self.zero_fringe_cells) / self.taper_cells, 0.0, 1.0
        )
        return u**3 * (u * (u * 6.0 - 15.0) + 10.0)

    def _correction_window(
        self, row0: int, col0: int, cell_rows: int, cell_cols: int
    ) -> np.ndarray:
        rows, cols = self.coefficients.shape
        if not (
            0 <= row0 <= rows - cell_rows
            and 0 <= col0 <= cols - cell_cols
        ):
            raise ValueError("projection window lies outside the planned AOI")
        knots_x = np.arange(cols, dtype=np.float64) + 0.5
        fine_x = col0 + (np.arange(cell_cols * self.factor) + 0.5) / self.factor
        along_x = CubicSpline(knots_x, self.coefficients, axis=1)(fine_x)
        knots_y = np.arange(rows, dtype=np.float64) + 0.5
        fine_y = row0 + (np.arange(cell_rows * self.factor) + 0.5) / self.factor
        return CubicSpline(knots_y, along_x, axis=0)(fine_y)

    def project_window(
        self,
        residual: np.ndarray,
        allowed: np.ndarray,
        *,
        row0: int,
        col0: int,
    ) -> np.ndarray:
        """Project a complete-cell window and zero every rejected/hard sample."""
        detail = np.asarray(residual, dtype=np.float64)
        soft = np.asarray(allowed, dtype=bool)
        if detail.shape != soft.shape or any(size % self.factor for size in detail.shape):
            raise ValueError("projection window must contain complete aligned cells")
        cell_rows = detail.shape[0] // self.factor
        cell_cols = detail.shape[1] // self.factor
        taper = self.taper_window(row0, col0, cell_rows, cell_cols)
        correction = self._correction_window(row0, col0, cell_rows, cell_cols)
        projected = np.where(soft, taper * (detail - correction), 0.0)
        blocks = projected.reshape(
            cell_rows, self.factor, cell_cols, self.factor
        )
        means = blocks.mean(axis=(1, 3))
        transition = (
            taper.reshape(cell_rows, self.factor, cell_cols, self.factor)
            .min(axis=(1, 3)) < 1.0
        ) & (
            taper.reshape(cell_rows, self.factor, cell_cols, self.factor)
            .max(axis=(1, 3)) > 0.0
        )
        if np.any(transition):
            u = (np.arange(self.factor, dtype=np.float64) + 0.5) / self.factor
            bubble_1d = u**3 * (1.0 - u) ** 3
            bubble_1d /= np.mean(bubble_1d)
            bubble = bubble_1d[:, None] * bubble_1d[None, :]
            correction_blocks = means[:, None, :, None] * bubble[None, :, None, :]
            correction_blocks *= transition[:, None, :, None]
            blocks -= correction_blocks
            projected = blocks.reshape(detail.shape)
            means = blocks.mean(axis=(1, 3))
        if float(np.max(np.abs(means))) >= 1e-10:
            raise AssertionError("smooth projection did not satisfy exact cell means")
        if np.any(projected[~soft] != 0.0):
            raise AssertionError("smooth projection changed a hard sample")
        return projected


def build_smooth_mean_null_projector(
    residual_cell_means: np.ndarray,
    fully_soft_cells: np.ndarray,
    *,
    factor: int = 16,
    zero_fringe_cells: float = 1.0,
    taper_cells: float = 2.0,
) -> SmoothMeanNullProjector:
    """Solve one global C2 correction and fail closed around every mask boundary.

    Cells containing any hard sample are rejected in full. The rejection is
    dilated into soft terrain so the sparse mixed-boundary solve can be deferred
    without reintroducing the tiny-support division or touching hard samples.
    """
    means = np.asarray(residual_cell_means, dtype=np.float64)
    fully_soft = np.asarray(fully_soft_cells, dtype=bool)
    if means.ndim != 2 or means.shape != fully_soft.shape or not np.isfinite(means).all():
        raise ValueError("projection plan arrays must be equal finite 2D fields")
    if zero_fringe_cells <= 0 or taper_cells <= 0:
        raise ValueError("projection fringe and taper must be positive")
    if fully_soft.all():
        distance = np.full(means.shape, np.inf, dtype=np.float64)
    else:
        distance = ndimage.distance_transform_edt(fully_soft).astype(np.float64)
    target = means
    ty = _cell_mean_operator(means.shape[0], factor)
    tx = _cell_mean_operator(means.shape[1], factor)
    intermediate = solve(ty, target, assume_a="gen", check_finite=False)
    coefficients = solve(
        tx, intermediate.T, assume_a="gen", check_finite=False
    ).T
    return SmoothMeanNullProjector(
        coefficients=np.ascontiguousarray(coefficients),
        soft_distance_cells=np.ascontiguousarray(distance),
        factor=factor,
        zero_fringe_cells=float(zero_fringe_cells),
        taper_cells=float(taper_cells),
    )
