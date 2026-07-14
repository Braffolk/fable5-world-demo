"""Deterministic numerical primitives for qualified flowing-water profiles.

This module never decides whether a raw point is water evidence and never selects
an epoch. It consumes already qualified station observations and fails closed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely

from .model import (
    AbstainedReach,
    FlowOrientation,
    FlowProfile,
    WaterAbstention,
    WaterSamples,
)


@dataclass(frozen=True)
class RobustLocation:
    value: float
    scale: float
    mad: float
    iterations: int


class ProfileRejected(ValueError):
    """Raised when qualified observations cannot support a publishable profile."""

    def __init__(self, reason: WaterAbstention, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def huber_location(
    values: np.ndarray,
    *,
    cutoff: float = 1.345,
    min_scale_m: float = 0.02,
    tolerance_m: float = 1e-8,
    max_iterations: int = 50,
) -> RobustLocation:
    """Canonical deterministic Huber location used by the repair specification."""
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
        raise ProfileRejected(
            WaterAbstention.ROBUST_FIT_FAILED, "Huber input must be nonempty finite 1D data"
        )
    location = float(np.median(samples))
    mad = float(np.median(np.abs(samples - location)))
    scale = max(1.4826 * mad, min_scale_m)
    for iteration in range(1, max_iterations + 1):
        normalized = np.abs((samples - location) / scale)
        weights = np.minimum(1.0, cutoff / np.maximum(normalized, 1e-300))
        updated = float(np.dot(weights, samples) / weights.sum(dtype=np.float64))
        if abs(updated - location) < tolerance_m:
            return RobustLocation(updated, scale, mad, iteration)
        location = updated
    raise ProfileRejected(
        WaterAbstention.ROBUST_FIT_FAILED,
        f"Huber location did not converge in {max_iterations} iterations",
    )


def station_locations(
    point_station_m: np.ndarray,
    point_y: np.ndarray,
    station_m: np.ndarray,
    *,
    interval_m: float = 3.0,
    minimum_samples: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Robust-locate qualified points in a centered along-channel interval."""
    point_s = np.asarray(point_station_m, dtype=np.float64)
    point_z = np.asarray(point_y, dtype=np.float64)
    stations = np.asarray(station_m, dtype=np.float64)
    if (
        point_s.ndim != 1
        or point_z.shape != point_s.shape
        or stations.ndim != 1
        or not np.isfinite(point_s).all()
        or not np.isfinite(point_z).all()
        or not np.isfinite(stations).all()
        or interval_m <= 0.0
        or minimum_samples < 1
    ):
        raise ValueError("invalid station-location inputs")
    order = np.argsort(point_s, kind="stable")
    ordered_s, ordered_z = point_s[order], point_z[order]
    locations = np.full(stations.shape, np.nan, dtype=np.float64)
    counts = np.zeros(stations.shape, dtype=np.int32)
    half = interval_m / 2.0
    for index, station in enumerate(stations):
        lo = int(np.searchsorted(ordered_s, station - half, side="left"))
        hi = int(np.searchsorted(ordered_s, station + half, side="right"))
        counts[index] = hi - lo
        if hi - lo >= minimum_samples:
            locations[index] = huber_location(ordered_z[lo:hi]).value
    return locations, counts


def hampel_accept(
    values: np.ndarray,
    station_m: np.ndarray,
    *,
    window_m: float = 11.0,
    threshold: float = 3.0,
) -> np.ndarray:
    """Return the deterministic finite/Hampel acceptance mask."""
    samples = np.asarray(values, dtype=np.float64)
    stations = np.asarray(station_m, dtype=np.float64)
    if samples.shape != stations.shape or samples.ndim != 1 or window_m <= 0.0:
        raise ValueError("invalid Hampel inputs")
    finite = np.isfinite(samples)
    accepted = finite.copy()
    finite_indices = np.flatnonzero(finite)
    half = window_m / 2.0
    for index in finite_indices:
        local = finite & (np.abs(stations - stations[index]) <= half)
        neighborhood = samples[local]
        if neighborhood.size < 3:
            accepted[index] = False
            continue
        median = float(np.median(neighborhood))
        mad = float(np.median(np.abs(neighborhood - median)))
        if abs(samples[index] - median) > threshold * 1.4826 * mad:
            accepted[index] = False
    return accepted


def longest_missing_span_m(station_m: np.ndarray, accepted: np.ndarray) -> float:
    """Physical length of the longest run of unsupported station cells.

    Edge gaps are measured from the reach endpoint to the nearest accepted station.
    """
    stations = np.asarray(station_m, dtype=np.float64)
    mask = np.asarray(accepted, dtype=bool)
    if stations.ndim != 1 or mask.shape != stations.shape or stations.size < 2:
        raise ValueError("invalid missing-span inputs")
    valid = np.flatnonzero(mask)
    if valid.size < 2:
        return float("inf")
    spacing = np.diff(stations)
    if not np.allclose(spacing, spacing[0], rtol=0.0, atol=1e-9):
        raise ValueError("missing-span measurement requires regular stations")
    step = spacing[0]
    spans = np.concatenate(
        (
            [stations[valid[0]] - stations[0]],
            np.maximum(0.0, stations[valid[1:]] - stations[valid[:-1]] - step),
            [stations[-1] - stations[valid[-1]]],
        )
    )
    return float(np.max(spans))


def _endpoint_location(values: np.ndarray, station_m: np.ndarray, start: bool) -> RobustLocation:
    edge = station_m[0] + 21.0 if start else station_m[-1] - 21.0
    mask = station_m <= edge if start else station_m >= edge
    finite = np.isfinite(values) & mask
    if np.count_nonzero(finite) < 4:
        raise ProfileRejected(
            WaterAbstention.ENDPOINT_UNSUPPORTED,
            "fewer than four qualified observations in a 21 m endpoint interval",
        )
    return huber_location(values[finite])


def infer_orientation(
    values: np.ndarray, station_m: np.ndarray
) -> tuple[FlowOrientation, RobustLocation, RobustLocation]:
    """Infer downstream direction only when endpoint evidence clears the spec gate."""
    samples = np.asarray(values, dtype=np.float64)
    stations = np.asarray(station_m, dtype=np.float64)
    start = _endpoint_location(samples, stations, True)
    end = _endpoint_location(samples, stations, False)
    endpoint_mad = max(start.mad, end.mad)
    if abs(start.value - end.value) <= max(0.05, 2.0 * endpoint_mad):
        return FlowOrientation.UNKNOWN, start, end
    orientation = FlowOrientation.FORWARD if start.value > end.value else FlowOrientation.REVERSE
    return orientation, start, end


def isotonic_nonincreasing(values: np.ndarray) -> np.ndarray:
    """Unweighted deterministic PAVA, merging the first violating adjacent pools."""
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("PAVA input must be finite 1D data")
    means: list[float] = []
    counts: list[int] = []
    for sample in samples:
        means.append(float(sample))
        counts.append(1)
        while len(means) >= 2 and means[-2] < means[-1]:
            count = counts[-2] + counts[-1]
            mean = (means[-2] * counts[-2] + means[-1] * counts[-1]) / count
            means[-2:] = [mean]
            counts[-2:] = [count]
    return np.repeat(np.asarray(means, dtype=np.float64), counts)


def gaussian_positive(
    values: np.ndarray, *, sigma_samples: float = 4.0, truncate: float = 4.0
) -> np.ndarray:
    """Positive normalized Gaussian convolution with nearest-end extension."""
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1 or not np.isfinite(samples).all() or sigma_samples <= 0.0:
        raise ValueError("invalid Gaussian inputs")
    radius = int(np.ceil(truncate * sigma_samples))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_samples) ** 2)
    kernel /= kernel.sum(dtype=np.float64)
    padded = np.pad(samples, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def fit_flow_profile(
    *,
    reach_id: str,
    epoch_id: str,
    source_artifact_sha256: str,
    station_m: np.ndarray,
    observation_y: np.ndarray,
    maximum_missing_span_m: float = 20.0,
) -> FlowProfile | AbstainedReach:
    """Fit one already qualified, unbranched, discontinuity-free reach segment."""
    stations = np.asarray(station_m, dtype=np.float64)
    observations = np.asarray(observation_y, dtype=np.float64)
    try:
        if (
            stations.ndim != 1
            or observations.shape != stations.shape
            or stations.size < 2
            or not np.isfinite(stations).all()
            or not np.allclose(np.diff(stations), 1.0, rtol=0.0, atol=1e-9)
        ):
            raise ProfileRejected(
                WaterAbstention.PROFILE_NONFINITE,
                "flow profiles require aligned finite 1 m stations and matching observations",
            )
        accepted = hampel_accept(observations, stations)
        if np.count_nonzero(accepted) < 4:
            raise ProfileRejected(
                WaterAbstention.WATER_SUPPORT_INSUFFICIENT,
                "fewer than four station observations remain after Hampel rejection",
            )
        span = longest_missing_span_m(stations, accepted)
        if not np.isfinite(span) or span > maximum_missing_span_m:
            raise ProfileRejected(
                WaterAbstention.MISSING_SPAN,
                f"longest unsupported station span is {span} m",
            )
        filtered = np.where(accepted, observations, np.nan)
        orientation, _, _ = infer_orientation(filtered, stations)
        accepted_indices = np.flatnonzero(accepted)
        filled = np.interp(stations, stations[accepted_indices], observations[accepted_indices])
        if orientation is FlowOrientation.FORWARD:
            constrained = isotonic_nonincreasing(filled)
        elif orientation is FlowOrientation.REVERSE:
            constrained = isotonic_nonincreasing(filled[::-1])[::-1]
        else:
            constrained = filled
        water_y = gaussian_positive(constrained)
        return FlowProfile(
            reach_id=reach_id,
            epoch_id=epoch_id,
            station_m=stations,
            observation_y=observations,
            accepted=accepted,
            water_y=water_y,
            orientation=orientation,
            longest_missing_span_m=span,
            source_artifact_sha256=source_artifact_sha256,
        )
    except ProfileRejected as error:
        return AbstainedReach(reach_id, error.reason, error.detail)


def unknown_bathymetry_depth(distance_inside_m: np.ndarray) -> np.ndarray:
    """Zero-confidence rendering envelope; never a measured bed estimate."""
    distance = np.asarray(distance_inside_m, dtype=np.float64)
    if not np.isfinite(distance).all():
        raise ValueError("distance_inside_m must be finite")
    u = np.clip(distance, 0.0, 1.0)
    return 0.10 * (6.0 * u**5 - 15.0 * u**4 + 10.0 * u**3)


def evaluate_flowing_water(
    *,
    easting: np.ndarray,
    northing: np.ndarray,
    water_polygon,
    centerline,
    profile: FlowProfile,
    endpoint_extension_m: float = 0.0,
) -> WaterSamples:
    """Evaluate a qualified profile and conservative bed at paired world points.

    ``water_polygon`` owns wet membership. The centerline supplies only station
    distance. Wet points outside the profile's explicit station interval plus a
    caller-declared evidence-edge extension remain unresolved with NaN heights.
    Callers must not publish those samples or replace them with raw DTM.
    """
    east = np.asarray(easting, dtype=np.float64)
    north = np.asarray(northing, dtype=np.float64)
    if (
        east.ndim != 1
        or north.shape != east.shape
        or not np.isfinite(east).all()
        or not np.isfinite(north).all()
        or not np.isfinite(endpoint_extension_m)
        or endpoint_extension_m < 0.0
    ):
        raise ValueError("invalid flowing-water sample coordinates")
    if water_polygon is None or water_polygon.is_empty or centerline is None or centerline.is_empty:
        raise ValueError("water polygon and centerline must be nonempty")
    if centerline.geom_type != "LineString" or not centerline.is_simple:
        raise ValueError("flowing-water evaluation requires one simple unbranched LineString")

    points = shapely.points(east, north)
    wet = np.asarray(shapely.covers(water_polygon, points), dtype=bool)
    station = np.asarray(shapely.line_locate_point(centerline, points), dtype=np.float64)
    lower = profile.station_m[0] - endpoint_extension_m
    upper = profile.station_m[-1] + endpoint_extension_m
    supported = wet & (station >= lower) & (station <= upper)
    water_y = np.full(east.shape, np.nan, dtype=np.float64)
    bed_y = np.full(east.shape, np.nan, dtype=np.float64)
    if np.any(supported):
        water_y[supported] = np.interp(
            station[supported], profile.station_m, profile.water_y
        )
        distance = np.asarray(
            shapely.distance(points[supported], water_polygon.boundary), dtype=np.float64
        )
        bed_y[supported] = water_y[supported] - unknown_bathymetry_depth(distance)
    return WaterSamples(wet, supported, water_y, bed_y)
