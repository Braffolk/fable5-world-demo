"""Streaming, semantics-neutral observed-support inventory for merged Hovi TLS."""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import laspy
import numpy as np

SUPPORT_RESOLUTIONS_M = (0.025, 0.05, 0.0625, 0.125, 0.25, 1.0)
_MAX_SUPPORT_ARRAY_BYTES = 1 << 30


def resolution_tag(resolution_m: float) -> str:
    return f"{resolution_m:g}".replace(".", "p") + "m"


def _grid_shape(
    resolution_m: float, min_x_m: float, min_y_m: float, max_x_m: float, max_y_m: float
) -> tuple[int, int, int, int]:
    origin_ix = math.floor(min_x_m / resolution_m)
    origin_iy = math.floor(min_y_m / resolution_m)
    max_ix = math.floor(max_x_m / resolution_m)
    max_iy = math.floor(max_y_m / resolution_m)
    return origin_ix, origin_iy, max_ix - origin_ix + 1, max_iy - origin_iy + 1


def _json_dtype(dtype: np.dtype) -> list[list[Any]]:
    output: list[list[Any]] = []
    for item in dtype.descr:
        converted = list(item)
        if len(converted) == 3 and isinstance(converted[2], tuple):
            converted[2] = list(converted[2])
        output.append(converted)
    return output


def _vlr_record(vlr: Any) -> dict[str, Any]:
    payload = vlr.record_data_bytes()
    return {
        "user_id": vlr.user_id,
        "record_id": int(vlr.record_id),
        "description": vlr.description,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _header_record(header: Any) -> dict[str, Any]:
    try:
        crs = header.parse_crs()
        crs_wkt = crs.to_wkt() if crs is not None else None
        crs_parse_error = None
    except Exception as error:  # Malformed source CRS is inventory evidence.
        crs_wkt = None
        crs_parse_error = {"type": type(error).__name__, "message": str(error)}
    return {
        "las_version": str(header.version),
        "point_format": int(header.point_format.id),
        "point_record_bytes": int(header.point_format.size),
        "extra_bytes": int(header.point_format.num_extra_bytes),
        "point_count": int(header.point_count),
        "points_by_return": [int(value) for value in header.number_of_points_by_return],
        "scales": [float(value) for value in header.scales],
        "offsets": [float(value) for value in header.offsets],
        "declared_min_xyz": [float(value) for value in header.mins],
        "declared_max_xyz": [float(value) for value in header.maxs],
        "global_encoding": int(header.global_encoding.value),
        "system_identifier": header.system_identifier,
        "generating_software": header.generating_software,
        "creation_date": header.creation_date.isoformat() if header.creation_date else None,
        "dimension_names": list(header.point_format.dimension_names),
        "point_record_dtype": _json_dtype(header.point_format.dtype()),
        "crs_wkt": crs_wkt,
        "crs_parse_error": crs_parse_error,
        "vlrs": [_vlr_record(vlr) for vlr in header.vlrs],
        "evlrs": [_vlr_record(vlr) for vlr in (header.evlrs or [])],
    }


@dataclass
class SupportGrid:
    resolution_m: float
    origin_x_m: float
    origin_y_m: float
    width: int
    height: int
    point_count: np.ndarray
    nearest_in_cell_center_distance_m: np.ndarray

    @classmethod
    def from_bounds(
        cls,
        resolution_m: float,
        min_x_m: float,
        min_y_m: float,
        max_x_m: float,
        max_y_m: float,
    ) -> "SupportGrid":
        origin_ix, origin_iy, width, height = _grid_shape(
            resolution_m, min_x_m, min_y_m, max_x_m, max_y_m
        )
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid Hovi support grid at {resolution_m:g} m")
        return cls(
            resolution_m=resolution_m,
            origin_x_m=origin_ix * resolution_m,
            origin_y_m=origin_iy * resolution_m,
            width=width,
            height=height,
            point_count=np.zeros((height, width), dtype=np.uint64),
            nearest_in_cell_center_distance_m=np.full(
                (height, width), np.inf, dtype=np.float32
            ),
        )

    @property
    def upper_x_m(self) -> float:
        return self.origin_x_m + self.width * self.resolution_m

    @property
    def upper_y_m(self) -> float:
        return self.origin_y_m + self.height * self.resolution_m

    def update(self, x_m: np.ndarray, y_m: np.ndarray) -> int:
        ix = np.floor((x_m - self.origin_x_m) / self.resolution_m).astype(np.int64)
        iy = np.floor((y_m - self.origin_y_m) / self.resolution_m).astype(np.int64)
        inside = (ix >= 0) & (ix < self.width) & (iy >= 0) & (iy < self.height)
        if not np.all(inside):
            ix = ix[inside]
            iy = iy[inside]
            x_m = x_m[inside]
            y_m = y_m[inside]
        flat = iy * self.width + ix
        counts = np.bincount(flat, minlength=self.point_count.size)
        self.point_count.ravel()[:] += counts.astype(np.uint64, copy=False)

        center_x = self.origin_x_m + (ix + 0.5) * self.resolution_m
        center_y = self.origin_y_m + (iy + 0.5) * self.resolution_m
        distance = np.hypot(x_m - center_x, y_m - center_y)
        np.minimum.at(
            self.nearest_in_cell_center_distance_m.ravel(), flat, distance.astype(np.float32)
        )
        return int(np.count_nonzero(~inside)) if inside.size else 0

    def metadata(self) -> dict[str, Any]:
        occupied = self.point_count > 0
        nearest = self.nearest_in_cell_center_distance_m[occupied]
        return {
            "resolution_m": self.resolution_m,
            "origin_xy_m": [self.origin_x_m, self.origin_y_m],
            "upper_exclusive_xy_m": [self.upper_x_m, self.upper_y_m],
            "shape_yx": [self.height, self.width],
            "total_cells": int(self.point_count.size),
            "occupied_cells": int(np.count_nonzero(occupied)),
            "occupied_fraction": float(np.mean(occupied)),
            "point_count_sum": int(self.point_count.sum(dtype=np.uint64)),
            "max_points_per_cell": int(self.point_count.max(initial=0)),
            "nearest_in_cell_center_distance_m": {
                "available_cells": int(nearest.size),
                "min": float(nearest.min()) if nearest.size else None,
                "median": float(np.median(nearest)) if nearest.size else None,
                "p95": float(np.quantile(nearest, 0.95)) if nearest.size else None,
                "max": float(nearest.max()) if nearest.size else None,
            },
        }


@dataclass(frozen=True)
class SupportInventory:
    header: dict[str, Any]
    point_counts: dict[str, int]
    observed_extents: dict[str, list[float] | None]
    invalids: dict[str, int]
    categorical_counts: dict[str, dict[str, int]]
    grids: tuple[SupportGrid, ...]

    def arrays(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {
            "support_resolutions_m": np.asarray(
                [grid.resolution_m for grid in self.grids], dtype=np.float64
            )
        }
        for grid in self.grids:
            tag = resolution_tag(grid.resolution_m)
            nearest = grid.nearest_in_cell_center_distance_m.copy()
            nearest[~np.isfinite(nearest)] = np.nan
            arrays[f"point_count__{tag}"] = grid.point_count
            arrays[f"nearest_in_cell_center_distance_m__{tag}"] = nearest
        return arrays


def _update_category(counter: Counter[int], values: np.ndarray) -> None:
    unique, counts = np.unique(values, return_counts=True)
    counter.update(
        {int(value): int(count) for value, count in zip(unique, counts, strict=True)}
    )


def inventory_observed_support(
    path: Path,
    *,
    resolutions_m: Iterable[float] = SUPPORT_RESOLUTIONS_M,
    chunk_points: int = 1_000_000,
    log=print,
) -> SupportInventory:
    """Inventory raw observations without identifying or extracting a ground surface."""
    if chunk_points <= 0:
        raise ValueError("Hovi chunk_points must be positive")
    resolutions = tuple(float(value) for value in resolutions_m)
    if resolutions != SUPPORT_RESOLUTIONS_M:
        raise ValueError(f"Hovi support resolutions are frozen to {SUPPORT_RESOLUTIONS_M}")

    with laspy.open(path) as reader:
        header = reader.header
        declared_min = np.asarray(header.mins, dtype=np.float64)
        declared_max = np.asarray(header.maxs, dtype=np.float64)
        if (
            declared_min.shape != (3,)
            or declared_max.shape != (3,)
            or not np.all(np.isfinite(declared_min))
            or not np.all(np.isfinite(declared_max))
            or np.any(declared_max < declared_min)
        ):
            raise ValueError("Hovi LAS header has invalid declared XYZ extents")
        shapes = tuple(
            _grid_shape(
                resolution,
                float(declared_min[0]),
                float(declared_min[1]),
                float(declared_max[0]),
                float(declared_max[1]),
            )
            for resolution in resolutions
        )
        support_bytes = sum(width * height * (8 + 4) for _, _, width, height in shapes)
        if support_bytes > _MAX_SUPPORT_ARRAY_BYTES:
            raise ValueError(
                f"Hovi support grids require {support_bytes / (1 << 30):.2f} GiB, above the "
                f"{_MAX_SUPPORT_ARRAY_BYTES / (1 << 30):.0f} GiB inventory limit"
            )
        grids = tuple(
            SupportGrid.from_bounds(
                resolution,
                float(declared_min[0]),
                float(declared_min[1]),
                float(declared_max[0]),
                float(declared_max[1]),
            )
            for resolution in resolutions
        )

        dimension_names = set(header.point_format.dimension_names)
        category_names = tuple(
            name
            for name in ("classification", "return_number", "number_of_returns")
            if name in dimension_names
        )
        category_counts = {name: Counter() for name in category_names}
        decoded_points = 0
        finite_xyz_points = 0
        support_points = 0
        nonfinite_x = nonfinite_y = nonfinite_z = nonfinite_any_xyz = 0
        outside_declared_xyz = 0
        invalid_return_relation = 0
        observed_min = np.full(3, np.inf, dtype=np.float64)
        observed_max = np.full(3, -np.inf, dtype=np.float64)
        scale = np.abs(np.asarray(header.scales, dtype=np.float64))
        tolerance = np.maximum(scale * 0.51, np.finfo(np.float64).eps)

        for chunk_index, points in enumerate(reader.chunk_iterator(chunk_points), start=1):
            count = len(points)
            decoded_points += count
            x_m = np.asarray(points.x, dtype=np.float64)
            y_m = np.asarray(points.y, dtype=np.float64)
            z_m = np.asarray(points.z, dtype=np.float64)
            finite_x = np.isfinite(x_m)
            finite_y = np.isfinite(y_m)
            finite_z = np.isfinite(z_m)
            finite_xyz = finite_x & finite_y & finite_z
            nonfinite_x += int(count - np.count_nonzero(finite_x))
            nonfinite_y += int(count - np.count_nonzero(finite_y))
            nonfinite_z += int(count - np.count_nonzero(finite_z))
            nonfinite_any_xyz += int(count - np.count_nonzero(finite_xyz))
            if np.any(finite_xyz):
                xyz = np.column_stack((x_m[finite_xyz], y_m[finite_xyz], z_m[finite_xyz]))
                observed_min = np.minimum(observed_min, xyz.min(axis=0))
                observed_max = np.maximum(observed_max, xyz.max(axis=0))
                finite_xyz_points += int(xyz.shape[0])
                within_header = np.all(
                    (xyz >= declared_min - tolerance) & (xyz <= declared_max + tolerance), axis=1
                )
                outside_declared_xyz += int(xyz.shape[0] - np.count_nonzero(within_header))
                supported_xy = xyz[within_header, :2]
                support_points += int(supported_xy.shape[0])
                for grid in grids:
                    leaked = grid.update(supported_xy[:, 0], supported_xy[:, 1])
                    if leaked:
                        raise ValueError(
                            f"Hovi support grid rejected {leaked} header-valid points at "
                            f"{grid.resolution_m:g} m"
                        )
            for name, counter in category_counts.items():
                _update_category(counter, np.asarray(points[name]))
            if "return_number" in dimension_names and "number_of_returns" in dimension_names:
                returns = np.asarray(points["return_number"])
                totals = np.asarray(points["number_of_returns"])
                invalid_return_relation += int(
                    np.count_nonzero((returns == 0) | (totals == 0) | (returns > totals))
                )
            log(f"Hovi TLS: decoded chunk {chunk_index} ({decoded_points:,} points total)")

        if decoded_points != int(header.point_count):
            raise ValueError(
                f"Hovi decoded point count {decoded_points} differs from header "
                f"{int(header.point_count)}"
            )
        for grid in grids:
            if int(grid.point_count.sum(dtype=np.uint64)) != support_points:
                raise ValueError(
                    f"Hovi {grid.resolution_m:g} m support count does not conserve points"
                )

        observed_extents = {
            "finite_min_xyz": observed_min.tolist() if finite_xyz_points else None,
            "finite_max_xyz": observed_max.tolist() if finite_xyz_points else None,
        }
        return SupportInventory(
            header=_header_record(header),
            point_counts={
                "header": int(header.point_count),
                "decoded": decoded_points,
                "finite_xyz": finite_xyz_points,
                "included_in_support_grids": support_points,
            },
            observed_extents=observed_extents,
            invalids={
                "nonfinite_x": nonfinite_x,
                "nonfinite_y": nonfinite_y,
                "nonfinite_z": nonfinite_z,
                "nonfinite_any_xyz": nonfinite_any_xyz,
                "outside_declared_xyz": outside_declared_xyz,
                "invalid_return_relation": invalid_return_relation,
            },
            categorical_counts={
                name: {str(key): value for key, value in sorted(counter.items())}
                for name, counter in category_counts.items()
            },
            grids=grids,
        )
