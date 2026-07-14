"""World-aligned reconstruction of a 1 m DTM onto the 0.25 m authority grid."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SampleGrid:
    """North-up sample-center grid; array rows run from north to south."""

    center_e: float
    center_n: float
    texel_m: float
    rows: int
    cols: int

    def __post_init__(self) -> None:
        if (
            not np.isfinite((self.center_e, self.center_n, self.texel_m)).all()
            or self.texel_m <= 0.0
            or self.rows < 1
            or self.cols < 1
        ):
            raise ValueError("invalid sample grid")

    def eastings(self) -> np.ndarray:
        return self.center_e + np.arange(self.cols, dtype=np.float64) * self.texel_m

    def northings(self) -> np.ndarray:
        return self.center_n - np.arange(self.rows, dtype=np.float64) * self.texel_m


@dataclass(frozen=True)
class BaselineTile:
    height: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        height = np.array(self.height, dtype=np.float64, copy=True)
        valid = np.array(self.valid, dtype=np.bool_, copy=True)
        if height.ndim != 2 or valid.shape != height.shape:
            raise ValueError("baseline height/valid arrays must be matching 2D rasters")
        if not np.isfinite(height[valid]).all() or np.isfinite(height[~valid]).any():
            raise ValueError("baseline invalid samples must be NaN and valid samples finite")
        height.flags.writeable = False
        valid.flags.writeable = False
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "valid", valid)


def bilinear_baseline(
    source_height: np.ndarray,
    *,
    source_grid: SampleGrid,
    target_grid: SampleGrid,
) -> BaselineTile:
    """Evaluate a real-support 1 m source on a target sample-center grid.

    The source and target may use any positive spacing, although production uses
    1 m and 0.25 m. The caller supplies the complete real source halo. A target
    whose four-point stencil leaves that halo is rejected. A stencil containing
    any nonfinite source observation emits explicit invalid/NaN; it is never
    nearest-filled, clamped, wrapped, or reflected.
    """
    source = np.asarray(source_height)
    if source.ndim != 2 or source.shape != (source_grid.rows, source_grid.cols):
        raise ValueError("source_height does not match source_grid")
    source = np.asarray(source, dtype=np.float64)

    source_x = (target_grid.eastings() - source_grid.center_e) / source_grid.texel_m
    source_y = (source_grid.center_n - target_grid.northings()) / source_grid.texel_m
    col0 = np.floor(source_x).astype(np.int64)
    row0 = np.floor(source_y).astype(np.int64)
    if (
        col0.min(initial=0) < 0
        or row0.min(initial=0) < 0
        or col0.max(initial=-1) + 1 >= source_grid.cols
        or row0.max(initial=-1) + 1 >= source_grid.rows
    ):
        raise ValueError("target grid leaves declared real source support")

    tx = source_x - col0
    ty = source_y - row0
    northwest = source[row0[:, None], col0[None, :]]
    northeast = source[row0[:, None], (col0 + 1)[None, :]]
    southwest = source[(row0 + 1)[:, None], col0[None, :]]
    southeast = source[(row0 + 1)[:, None], (col0 + 1)[None, :]]
    valid = (
        np.isfinite(northwest)
        & np.isfinite(northeast)
        & np.isfinite(southwest)
        & np.isfinite(southeast)
    )
    wx = tx[None, :]
    wy = ty[:, None]
    with np.errstate(invalid="ignore"):
        result = (
            northwest * (1.0 - wx) * (1.0 - wy)
            + northeast * wx * (1.0 - wy)
            + southwest * (1.0 - wx) * wy
            + southeast * wx * wy
        )
    result[~valid] = np.nan
    return BaselineTile(result, valid)
