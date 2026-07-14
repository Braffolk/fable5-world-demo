"""Typed compact forms for one cultivated surface, without raster noise fields."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import CultivatedConditions

_PHI = 1.6180339887498948
_SQRT2 = 1.4142135623730951
_SQRT3 = 1.7320508075688772
_SQRT5 = 2.23606797749979


def _fraction(value: float | np.ndarray) -> float | np.ndarray:
    return value - np.floor(value)


def _wrapped(value: np.ndarray, period: float) -> np.ndarray:
    return value - np.floor(value / period + 0.5) * period


def _compact_bump(distance_ratio: np.ndarray) -> np.ndarray:
    core = np.maximum(1.0 - np.square(distance_ratio), 0.0)
    return np.square(core)


def _coordinates(
    east: np.ndarray,
    north: np.ndarray,
    conditions: CultivatedConditions,
    heading_deg_east_of_grid_north: float,
) -> tuple[np.ndarray, np.ndarray]:
    angle = np.deg2rad(heading_deg_east_of_grid_north)
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    de = (np.asarray(east, dtype=np.float64) - conditions.parcel_anchor_e_m).astype(
        np.float32
    )
    dn = (np.asarray(north, dtype=np.float64) - conditions.parcel_anchor_n_m).astype(
        np.float32
    )
    along = dn[:, None] * cosine + de[None, :] * sine
    cross = de[None, :] * cosine - dn[:, None] * sine
    return along, cross


@dataclass(frozen=True)
class OperationForms:
    rows: np.ndarray
    tracks: np.ndarray
    track_mask: np.ndarray


def evaluate_operation_forms(
    east: np.ndarray,
    north: np.ndarray,
    conditions: CultivatedConditions,
) -> OperationForms:
    along, cross = _coordinates(
        east, north, conditions, conditions.row_heading_deg_east_of_grid_north
    )
    operated_cross = cross - conditions.track_curvature_per_m * np.square(along)

    row_phase = _wrapped(operated_cross, conditions.row_spacing_m)
    ridge = _compact_bump(np.abs(row_phase) / (0.31 * conditions.row_spacing_m))
    furrow_distance = np.abs(np.abs(row_phase) - 0.5 * conditions.row_spacing_m)
    furrow = _compact_bump(furrow_distance / (0.36 * conditions.row_spacing_m))
    rows = conditions.row_relief_m * (ridge - 0.72 * furrow)

    pass_phase = _wrapped(operated_cross, conditions.track_pass_spacing_m)
    track_phase = np.abs(pass_phase) - 0.5 * conditions.track_gauge_m
    half_width = 0.5 * conditions.track_width_m
    rut = _compact_bump(np.abs(track_phase) / half_width)
    rim_distance = np.abs(np.abs(track_phase) - 1.18 * half_width)
    rim = _compact_bump(rim_distance / (0.32 * half_width))
    tread_phase = _wrapped(
        along + 0.42 * pass_phase, conditions.tread_spacing_m
    )
    tread = _compact_bump(
        np.abs(tread_phase) / (0.30 * conditions.tread_spacing_m)
    )
    tracks = (
        -conditions.track_depth_m * rut * (0.78 + 0.22 * tread)
        + 0.18 * conditions.track_depth_m * rim
    )
    rows *= 1.0 - 0.84 * rut
    return OperationForms(
        np.asarray(rows, dtype=np.float32),
        np.asarray(tracks, dtype=np.float32),
        np.asarray(rut, dtype=np.float32),
    )


@dataclass(frozen=True)
class ClodForms:
    height: np.ndarray
    candidate_count: int
    retained_count: int
    subtype_counts: tuple[int, int, int]
    diameter_range_m: tuple[float, float]
    height_range_m: tuple[float, float]


def evaluate_clods(
    east: np.ndarray,
    north: np.ndarray,
    track_mask: np.ndarray,
    conditions: CultivatedConditions,
) -> ClodForms:
    """Rasterize an explicit low-discrepancy inventory of resolved large clods."""
    east = np.asarray(east, dtype=np.float64)
    north = np.asarray(north, dtype=np.float64)
    result = np.zeros((north.size, east.size), dtype=np.float32)
    radius_max = 0.5 * conditions.clod_diameter_max_m
    cell = conditions.clod_cell_m
    ix0 = int(np.floor((east[0] - radius_max) / cell))
    ix1 = int(np.floor((east[-1] + radius_max) / cell))
    north_low = min(float(north[0]), float(north[-1]))
    north_high = max(float(north[0]), float(north[-1]))
    north_ascending = north[::-1]
    iy0 = int(np.floor((north_low - radius_max) / cell))
    iy1 = int(np.floor((north_high + radius_max) / cell))
    heading = np.deg2rad(90.0 - conditions.row_heading_deg_east_of_grid_north)
    retained = 0
    subtype_counts = [0, 0, 0]
    observed_diameters: list[float] = []
    observed_heights: list[float] = []

    for iy in range(iy0, iy1 + 1):
        for ix in range(ix0, ix1 + 1):
            selector = float(_fraction((ix + 0.5) * _PHI + (iy + 0.5) * _SQRT2))
            if selector >= conditions.clod_occupancy:
                continue
            offset_e = float(_fraction(ix * _SQRT3 + iy / _PHI) - 0.5) * 0.72
            offset_n = float(_fraction(ix / _SQRT5 + iy * _PHI) - 0.5) * 0.72
            center_e = (ix + 0.5 + offset_e) * cell
            center_n = (iy + 0.5 + offset_n) * cell
            major_q = float(_fraction(ix * _SQRT2 + iy * _SQRT5))
            minor_q = float(_fraction(ix * _PHI - iy / _SQRT3))
            height_q = float(_fraction(ix / _SQRT2 + iy * _SQRT3))
            diameter = conditions.clod_diameter_min_m + major_q * (
                conditions.clod_diameter_max_m - conditions.clod_diameter_min_m
            )
            minor_diameter = max(
                conditions.clod_diameter_min_m,
                diameter * (0.58 + 0.34 * minor_q),
            )
            height = conditions.clod_height_min_m + height_q * (
                conditions.clod_height_max_m - conditions.clod_height_min_m
            )
            subtype = int(np.floor(float(_fraction(ix / _PHI + iy * _SQRT5)) * 3.0))
            subtype = min(subtype, 2)
            exponent = (2.0, 3.6, 1.55)[subtype]
            angle = heading + (float(_fraction(ix * _SQRT5 - iy * _SQRT2)) - 0.5) * 1.15
            cosine, sine = float(np.cos(angle)), float(np.sin(angle))
            half_major = 0.5 * diameter
            half_minor = 0.5 * minor_diameter
            col0 = max(0, int(np.searchsorted(east, center_e - half_major) - 1))
            col1 = min(east.size, int(np.searchsorted(east, center_e + half_major) + 1))
            asc0 = max(
                0,
                int(np.searchsorted(north_ascending, center_n - half_major) - 1),
            )
            asc1 = min(
                north.size,
                int(np.searchsorted(north_ascending, center_n + half_major) + 1),
            )
            row0 = north.size - asc1
            row1 = north.size - asc0
            if row0 >= row1 or col0 >= col1:
                continue
            de = east[col0:col1][None, :] - center_e
            dn = north[row0:row1][:, None] - center_n
            local_x = (de * cosine + dn * sine) / half_major
            local_y = (-de * sine + dn * cosine) / half_minor
            radius = np.power(
                np.power(np.abs(local_x), exponent)
                + np.power(np.abs(local_y), exponent),
                1.0 / exponent,
            )
            cap = height * np.square(np.maximum(1.0 - np.square(radius), 0.0))
            view = result[row0:row1, col0:col1]
            np.maximum(view, cap.astype(np.float32), out=view)
            retained += 1
            subtype_counts[subtype] += 1
            observed_diameters.append(diameter)
            observed_heights.append(height)

    result *= 1.0 - 0.78 * np.asarray(track_mask, dtype=np.float32)
    candidate_count = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
    if retained == 0:
        diameter_range = (0.0, 0.0)
        height_range = (0.0, 0.0)
    else:
        diameter_range = (min(observed_diameters), max(observed_diameters))
        height_range = (min(observed_heights), max(observed_heights))
    return ClodForms(
        result,
        candidate_count,
        retained,
        tuple(subtype_counts),
        diameter_range,
        height_range,
    )
