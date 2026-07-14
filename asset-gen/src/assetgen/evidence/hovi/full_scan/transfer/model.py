"""Same-definition common-grid observables for Hovi transfer audit."""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np


HOVI_RESOLUTIONS_M = (0.025, 0.05, 0.0625, 0.125, 0.25, 1.0)
_HOVI_TICK_M = Fraction(1, 40)
_HOVI_X_TICKS = (-271, 1491)
_HOVI_Y_TICKS = (-244, 1351)
_FIXED_PHASE_OVERHEAD_BYTES = 8 << 20
_ARTIFACT_SERIALIZATION_WORK_BYTES = 32 << 20
_EXPANDED_BATCH_BYTES_PER_POINT = 256


def _ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def resolution_tag(resolution_m: float) -> str:
    return f"{resolution_m:g}".replace(".", "p") + "m"


@dataclass(frozen=True)
class GridLevel:
    resolution_m: float
    width: int
    height: int

    @property
    def cells(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class CommonGridPlan:
    origin_x_m: float
    origin_y_m: float
    upper_x_m: float
    upper_y_m: float
    resolutions_m: tuple[float, ...]

    def __post_init__(self) -> None:
        values = (
            self.origin_x_m,
            self.origin_y_m,
            self.upper_x_m,
            self.upper_y_m,
            *self.resolutions_m,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Hovi comparison grid contains a non-finite value")
        if self.upper_x_m <= self.origin_x_m or self.upper_y_m <= self.origin_y_m:
            raise ValueError("Hovi comparison AOI must have positive area")
        if not self.resolutions_m or any(value <= 0 for value in self.resolutions_m):
            raise ValueError("Hovi comparison resolutions must be positive")
        if len(set(self.resolutions_m)) != len(self.resolutions_m):
            raise ValueError("Hovi comparison resolutions must be unique")

    @property
    def levels(self) -> tuple[GridLevel, ...]:
        width = Fraction(str(self.upper_x_m)) - Fraction(str(self.origin_x_m))
        height = Fraction(str(self.upper_y_m)) - Fraction(str(self.origin_y_m))
        return tuple(
            GridLevel(
                resolution_m=resolution,
                width=_ceil_fraction(width / Fraction(str(resolution))),
                height=_ceil_fraction(height / Fraction(str(resolution))),
            )
            for resolution in self.resolutions_m
        )

    @property
    def bytes_per_source(self) -> int:
        bytes_per_cell = np.dtype(np.uint64).itemsize + np.dtype(np.float32).itemsize
        return sum(level.cells * bytes_per_cell for level in self.levels)

    @property
    def largest_level_cells(self) -> int:
        return max(level.cells for level in self.levels)

    @property
    def total_level_cells(self) -> int:
        return sum(level.cells for level in self.levels)

    def document(self) -> dict[str, Any]:
        return {
            "coordinate_role": "numeric_source_coordinates_only",
            "origin_xy_m": [self.origin_x_m, self.origin_y_m],
            "aoi_upper_exclusive_xy_m": [self.upper_x_m, self.upper_y_m],
            "cell_policy": "half-open nominal cells; final edge cell may be AOI-clipped",
            "levels": [
                {
                    "resolution_m": level.resolution_m,
                    "shape_yx": [level.height, level.width],
                    "grid_upper_exclusive_xy_m": [
                        self.origin_x_m + level.width * level.resolution_m,
                        self.origin_y_m + level.height * level.resolution_m,
                    ],
                }
                for level in self.levels
            ],
        }


def frozen_hovi_grid_plan() -> CommonGridPlan:
    return CommonGridPlan(
        origin_x_m=float(_HOVI_X_TICKS[0] * _HOVI_TICK_M),
        origin_y_m=float(_HOVI_Y_TICKS[0] * _HOVI_TICK_M),
        upper_x_m=float(_HOVI_X_TICKS[1] * _HOVI_TICK_M),
        upper_y_m=float(_HOVI_Y_TICKS[1] * _HOVI_TICK_M),
        resolutions_m=HOVI_RESOLUTIONS_M,
    )


@dataclass(frozen=True)
class ComparisonMemoryBound:
    accumulation_bytes: int
    phased_metrics_bytes: int
    array_transfer_bytes: int
    artifact_serialization_bytes: int

    @property
    def peak_bytes(self) -> int:
        return max(
            self.accumulation_bytes,
            self.phased_metrics_bytes,
            self.array_transfer_bytes,
            self.artifact_serialization_bytes,
        )


def comparison_memory_bound(
    plan: CommonGridPlan,
    *,
    batch_points: int,
) -> ComparisonMemoryBound:
    """Conservatively bound every explicit comparison phase before allocation."""
    if not isinstance(batch_points, int) or batch_points <= 0:
        raise ValueError("Hovi transfer memory bound requires a positive batch size")
    resident_observables = 2 * plan.bytes_per_source
    largest = plan.largest_level_cells
    cells = plan.total_level_cells
    # Metrics retain three occupancy masks. The remaining 37 bytes/cell cover the
    # largest float64 work array, indexed operands, finite masks, quantile copies,
    # and NumPy temporaries while only one distribution is live at a time.
    phased_metric_work = largest * 40
    return ComparisonMemoryBound(
        accumulation_bytes=(
            resident_observables
            + batch_points * _EXPANDED_BATCH_BYTES_PER_POINT
            + largest * np.dtype(np.int64).itemsize
            + _FIXED_PHASE_OVERHEAD_BYTES
        ),
        phased_metrics_bytes=(
            resident_observables
            + phased_metric_work
            + _FIXED_PHASE_OVERHEAD_BYTES
        ),
        array_transfer_bytes=(
            resident_observables
            + 2 * cells * np.dtype(np.bool_).itemsize
            + largest * np.dtype(np.bool_).itemsize
            + _FIXED_PHASE_OVERHEAD_BYTES
        ),
        artifact_serialization_bytes=(
            resident_observables
            + 2 * cells * np.dtype(np.bool_).itemsize
            + _ARTIFACT_SERIALIZATION_WORK_BYTES
            + _FIXED_PHASE_OVERHEAD_BYTES
        ),
    )


@dataclass
class LevelObservables:
    spec: GridLevel
    point_count: np.ndarray
    nearest_in_cell_center_distance_m: np.ndarray


class MultiscaleObservables:
    """Accumulate only count, occupancy, and in-cell center proximity."""

    def __init__(self, plan: CommonGridPlan):
        self.plan = plan
        self.levels = tuple(
            LevelObservables(
                spec=level,
                point_count=np.zeros((level.height, level.width), dtype=np.uint64),
                nearest_in_cell_center_distance_m=np.full(
                    (level.height, level.width), np.inf, dtype=np.float32
                ),
            )
            for level in plan.levels
        )
        self.input_points = 0
        self.included_points = 0
        self.outside_aoi_points = 0
        self._arrays_transferred = False

    def update(self, x_m: np.ndarray, y_m: np.ndarray) -> None:
        if self._arrays_transferred:
            raise RuntimeError("Hovi comparison arrays were already transferred")
        x = np.asarray(x_m, dtype=np.float64)
        y = np.asarray(y_m, dtype=np.float64)
        if x.ndim != 1 or y.shape != x.shape:
            raise ValueError("Hovi comparison XY batches must be same-length vectors")
        if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
            raise ValueError("Hovi comparison cannot bin non-finite XY observations")
        self.input_points += int(x.size)
        inside = (
            (x >= self.plan.origin_x_m)
            & (x < self.plan.upper_x_m)
            & (y >= self.plan.origin_y_m)
            & (y < self.plan.upper_y_m)
        )
        outside = int(x.size - np.count_nonzero(inside))
        self.outside_aoi_points += outside
        if outside:
            x = x[inside]
            y = y[inside]
        self.included_points += int(x.size)
        for level in self.levels:
            resolution = level.spec.resolution_m
            ix = np.floor((x - self.plan.origin_x_m) / resolution).astype(np.int64)
            iy = np.floor((y - self.plan.origin_y_m) / resolution).astype(np.int64)
            if (
                np.any(ix < 0)
                or np.any(ix >= level.spec.width)
                or np.any(iy < 0)
                or np.any(iy >= level.spec.height)
            ):
                raise ValueError("Hovi common-grid indexing escaped its declared shape")
            flat = iy * level.spec.width + ix
            counts = np.bincount(flat, minlength=level.spec.cells)
            np.add(
                level.point_count.ravel(),
                counts,
                out=level.point_count.ravel(),
                dtype=np.uint64,
                casting="unsafe",
            )
            center_x = self.plan.origin_x_m + (ix + 0.5) * resolution
            center_y = self.plan.origin_y_m + (iy + 0.5) * resolution
            distance = np.hypot(x - center_x, y - center_y).astype(np.float32)
            np.minimum.at(level.nearest_in_cell_center_distance_m.ravel(), flat, distance)

    def validate(self) -> None:
        for level in self.levels:
            if int(level.point_count.sum(dtype=np.uint64)) != self.included_points:
                raise ValueError("Hovi common-grid level does not conserve included points")
            occupied = level.point_count > 0
            if np.any(np.isfinite(level.nearest_in_cell_center_distance_m) != occupied):
                raise ValueError("Hovi center-proximity availability differs from occupancy")

    def arrays(self, prefix: str) -> dict[str, np.ndarray]:
        self.validate()
        output: dict[str, np.ndarray] = {}
        for level in self.levels:
            tag = resolution_tag(level.spec.resolution_m)
            nearest = level.nearest_in_cell_center_distance_m.copy()
            nearest[~np.isfinite(nearest)] = np.nan
            output[f"{prefix}_point_count__{tag}"] = level.point_count
            output[f"{prefix}_occupied__{tag}"] = level.point_count > 0
            output[f"{prefix}_nearest_in_cell_center_distance_m__{tag}"] = nearest
        return output

    def take_arrays(self, prefix: str) -> dict[str, np.ndarray]:
        """Transfer accumulator storage into the final product without grid copies."""
        if self._arrays_transferred:
            raise RuntimeError("Hovi comparison arrays were already transferred")
        self.validate()
        output: dict[str, np.ndarray] = {}
        for level in self.levels:
            tag = resolution_tag(level.spec.resolution_m)
            nearest = level.nearest_in_cell_center_distance_m
            unavailable = np.isfinite(nearest)
            np.logical_not(unavailable, out=unavailable)
            nearest[unavailable] = np.nan
            del unavailable
            output[f"{prefix}_point_count__{tag}"] = level.point_count
            output[f"{prefix}_occupied__{tag}"] = level.point_count > 0
            output[f"{prefix}_nearest_in_cell_center_distance_m__{tag}"] = nearest
        self._arrays_transferred = True
        return output


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite_mask = np.isfinite(finite)
    if not np.all(finite_mask):
        finite = finite[finite_mask]
    if not finite.size:
        return {"count": 0, "min": None, "p05": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "median": float(np.median(finite)),
        "p95": float(np.quantile(finite, 0.95)),
        "max": float(np.max(finite)),
    }


def comparison_metrics(
    full: MultiscaleObservables,
    thinned: MultiscaleObservables,
) -> dict[str, Any]:
    if full.plan != thinned.plan:
        raise ValueError("Hovi comparison inputs were not accumulated on the same grid")
    full.validate()
    thinned.validate()
    levels: list[dict[str, Any]] = []
    for full_level, thin_level in zip(full.levels, thinned.levels, strict=True):
        full_count = full_level.point_count
        thin_count = thin_level.point_count
        full_occupied = full_count > 0
        thin_occupied = thin_count > 0
        joint = full_occupied & thin_occupied
        both_occupied = int(np.count_nonzero(joint))
        full_only_occupied = int(np.count_nonzero(full_occupied & ~thin_occupied))
        thinned_only_occupied = int(np.count_nonzero(~full_occupied & thin_occupied))
        both_empty = int(np.count_nonzero(~full_occupied & ~thin_occupied))
        full_metrics = _source_level_metrics(full_level, full_occupied)
        thinned_metrics = _source_level_metrics(thin_level, thin_occupied)

        count_delta = thin_count.astype(np.float64)
        count_delta -= full_count
        count_delta_distribution = _distribution(count_delta)
        del count_delta

        ratio = thin_count[full_occupied].astype(np.float64)
        ratio /= full_count[full_occupied]
        ratio_distribution = _distribution(ratio)
        del ratio

        center_delta = thin_level.nearest_in_cell_center_distance_m[joint].astype(
            np.float64
        )
        center_delta -= full_level.nearest_in_cell_center_distance_m[joint]
        center_delta_distribution = _distribution(center_delta)
        del center_delta
        levels.append(
            {
                "resolution_m": full_level.spec.resolution_m,
                "shape_yx": [full_level.spec.height, full_level.spec.width],
                "full": full_metrics,
                "thinned": thinned_metrics,
                "paired_numeric_cells": {
                    "both_occupied": both_occupied,
                    "full_only_occupied": full_only_occupied,
                    "thinned_only_occupied": thinned_only_occupied,
                    "both_empty": both_empty,
                    "point_count_thinned_minus_full": count_delta_distribution,
                    "point_count_thinned_over_full_where_full_occupied": ratio_distribution,
                    "center_distance_thinned_minus_full_m_where_both_occupied": (
                        center_delta_distribution
                    ),
                },
            }
        )
        del full_occupied, thin_occupied, joint
    return {
        "schema_version": "hovi-full-vs-thinned-common-grid-metrics/1.0.0",
        "observable_boundary": {
            "included": [
                "raw_point_count",
                "occupancy",
                "minimum_planar_point_to_own_cell_center_distance_m",
            ],
            "point_record_definition": (
                "decoded finite numeric XY observation assigned to the common AOI; "
                "full-scan invalid/direction nonpoint streams are excluded"
            ),
            "occupancy_definition": "point_count > 0",
            "center_distance_definition": (
                "minimum planar distance from a point to the center of its own nominal "
                "half-open grid cell"
            ),
            "frame_equivalence_claim": False,
            "point_correspondence_claim": False,
            "surface_claim": False,
            "thinning_algorithm_claim": False,
            "scan_or_view_transfer_claim": False,
        },
        "grid": full.plan.document(),
        "source_totals": {
            "full": _source_totals(full),
            "thinned": _source_totals(thinned),
        },
        "levels": levels,
    }


def _source_totals(value: MultiscaleObservables) -> dict[str, int]:
    return {
        "input_points": value.input_points,
        "included_points": value.included_points,
        "outside_aoi_points": value.outside_aoi_points,
    }


def _source_level_metrics(
    level: LevelObservables,
    occupied: np.ndarray | None = None,
) -> dict[str, Any]:
    if occupied is None:
        occupied = level.point_count > 0
    return {
        "point_count_sum": int(level.point_count.sum(dtype=np.uint64)),
        "occupied_cells": int(np.count_nonzero(occupied)),
        "occupied_fraction": float(np.mean(occupied)),
        "point_count_per_cell": _distribution(level.point_count.astype(np.float64)),
        "nearest_in_cell_center_distance_m": _distribution(
            level.nearest_in_cell_center_distance_m[occupied]
        ),
    }
