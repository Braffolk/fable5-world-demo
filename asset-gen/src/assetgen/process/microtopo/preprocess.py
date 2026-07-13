"""Bounded-memory, fail-closed preparation of LUKE Lapinjarvi TLS exemplars."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator, griddata
from scipy.ndimage import (
    binary_closing,
    convolve,
    distance_transform_edt,
    label,
    median_filter,
)

from .model import GroundSurface, PatchBank

LAPINJARVI_ROLES = {
    "k11": "calibration",
    "k32": "calibration",
    "k36": "calibration",
    "k19": "holdout",
}

PointChunk = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class GroundExtractionConfig:
    texel_m: float = 0.0625
    crop_radius_m: float = 8.0
    support_cell_m: float = 0.25
    reservoir_points: int = 64
    min_support_points_per_view: int = 8
    min_support_views: int = 2
    min_cluster_points_per_view: int = 2
    max_view_disagreement_m: float = 0.08
    seed_radius_m: float = 0.85
    seed_height_tolerance_m: float = 0.35
    max_connected_gap_cells: int = 2
    max_connected_slope: float = 0.80
    connected_step_slack_m: float = 0.05
    min_cell_points: int = 2
    below_support_m: float = 0.04
    above_support_m: float = 0.10
    support_outlier_m: float = 0.14
    max_fill_hole_m: float = 0.125
    fine_reservoir_points: int = 24
    fine_min_points_per_view: int = 2
    fine_min_cluster_points_per_view: int = 1
    fine_max_view_disagreement_m: float = 0.04
    fine_sheet_window_cells: int = 5
    fine_sheet_outlier_m: float = 0.06
    min_connected_support_fraction: float = 0.45
    min_measured_fraction: float = 0.55
    min_largest_component_fraction: float = 0.95
    max_vertical_span_m: float = 3.0
    max_abs_elevation_m: float = 3.0
    max_fine_step_m: float = 0.25
    max_p99_fine_slope: float = 1.5
    gate_patch_cells: int = 64
    gate_patch_stride: int = 16
    min_useful_direct_fraction: float = 0.70
    min_useful_patch_windows: int = 4


def iter_laz_chunks(path: Path, chunk_points: int = 2_000_000) -> Iterator[PointChunk]:
    """Yield bounded LAZ chunks, including LUKE's 1..5 scan identifier."""
    try:
        import laspy  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("LAZ preparation requires the optional laspy[lazrs] dependency") from exc
    with laspy.open(path) as reader:
        for chunk in reader.chunk_iterator(chunk_points):
            yield (
                np.asarray(chunk.x, dtype=np.float64),
                np.asarray(chunk.y, dtype=np.float64),
                np.asarray(chunk.z, dtype=np.float64),
                np.asarray(chunk.user_data, dtype=np.uint8),
            )


def load_laz_xyz(path: Path) -> Iterator[PointChunk]:
    """Compatibility name; unlike the old implementation this returns a stream."""
    return iter_laz_chunks(path)


def _sample_hash(x: np.ndarray, y: np.ndarray, z: np.ndarray, view: np.ndarray) -> np.ndarray:
    qx = np.rint(x * 10_000).astype(np.int64).view(np.uint64)
    qy = np.rint(y * 10_000).astype(np.int64).view(np.uint64)
    qz = np.rint(z * 10_000).astype(np.int64).view(np.uint64)
    value = qx * np.uint64(0x9E3779B185EBCA87)
    value ^= qy * np.uint64(0xC2B2AE3D27D4EB4F)
    value ^= qz * np.uint64(0x165667B19E3779F9)
    value ^= view.astype(np.uint64) * np.uint64(0x85EBCA77C2B2AE63)
    value ^= value >> np.uint64(30)
    value *= np.uint64(0xBF58476D1CE4E5B9)
    value ^= value >> np.uint64(27)
    value *= np.uint64(0x94D049BB133111EB)
    return value ^ (value >> np.uint64(31))


class _CellReservoir:
    """Deterministic lowest-hash reservoir with bounded memory per raster cell."""

    def __init__(self, groups: int, capacity: int):
        self.capacity = capacity
        self.keys = np.full((groups, capacity), np.iinfo(np.uint64).max, dtype=np.uint64)
        self.values = np.full((groups, capacity), np.nan, dtype=np.float64)
        self.count = np.zeros(groups, dtype=np.int64)

    def update(self, group: np.ndarray, value: np.ndarray, key: np.ndarray) -> None:
        if not group.size:
            return
        order = np.argsort(group, kind="stable")
        group, value, key = group[order], value[order], key[order]
        starts = np.flatnonzero(np.r_[True, group[1:] != group[:-1]])
        ends = np.r_[starts[1:], group.size]
        for start, end in zip(starts, ends, strict=True):
            g = int(group[start])
            self.count[g] += end - start
            existing = self.keys[g] != np.iinfo(np.uint64).max
            keys = np.r_[self.keys[g, existing], key[start:end]]
            values = np.r_[self.values[g, existing], value[start:end]]
            if keys.size > self.capacity:
                selected = np.argpartition(keys, self.capacity - 1)[: self.capacity]
                keys, values = keys[selected], values[selected]
            rank = np.argsort(keys, kind="stable")
            n = len(rank)
            self.keys[g] = np.iinfo(np.uint64).max
            self.values[g] = np.nan
            self.keys[g, :n] = keys[rank]
            self.values[g, :n] = values[rank]

    def quantile(self, q: float, min_total: int) -> np.ndarray:
        result = np.full(len(self.count), np.nan, dtype=np.float64)
        for g in np.flatnonzero(self.count >= min_total):
            values = self.values[g, np.isfinite(self.values[g])]
            if values.size:
                result[g] = np.quantile(values, q, method="linear")
        return result


def _lowest_multiview_clusters(
    reservoir: _CellReservoir,
    cells: int,
    config: GroundExtractionConfig,
    *,
    min_points_per_view: int | None = None,
    min_cluster_points_per_view: int | None = None,
    max_view_disagreement_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the lowest tight height cluster represented by multiple scanner views."""
    area = cells * cells
    min_points = (
        config.min_support_points_per_view
        if min_points_per_view is None else min_points_per_view
    )
    min_cluster = (
        config.min_cluster_points_per_view
        if min_cluster_points_per_view is None else min_cluster_points_per_view
    )
    disagreement = (
        config.max_view_disagreement_m
        if max_view_disagreement_m is None else max_view_disagreement_m
    )
    elevation = np.full(area, np.nan, dtype=np.float64)
    views_used = np.zeros(area, dtype=np.uint8)
    spread = np.full(area, np.inf, dtype=np.float64)
    for cell in range(area):
        values: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for view in range(5):
            group = view * area + cell
            if reservoir.count[group] < min_points:
                continue
            sampled = reservoir.values[group, np.isfinite(reservoir.values[group])]
            if sampled.size:
                values.append(sampled)
                labels.append(np.full(sampled.size, view, dtype=np.uint8))
        if len(values) < config.min_support_views:
            continue
        z = np.concatenate(values)
        view_id = np.concatenate(labels)
        order = np.argsort(z, kind="stable")
        z, view_id = z[order], view_id[order]
        end = 0
        for start in range(z.size):
            end = max(end, start + 1)
            while end < z.size and z[end] - z[start] <= disagreement:
                end += 1
            window_views, counts = np.unique(view_id[start:end], return_counts=True)
            represented = window_views[counts >= min_cluster]
            if represented.size >= config.min_support_views:
                window_z = z[start:end]
                window_view = view_id[start:end]
                view_medians = np.asarray([
                    np.median(window_z[window_view == represented_view])
                    for represented_view in represented
                ])
                elevation[cell] = float(np.median(view_medians))
                views_used[cell] = represented.size
                spread[cell] = float(np.ptp(view_medians))
                break
    shape = (cells, cells)
    return elevation.reshape(shape), views_used.reshape(shape), spread.reshape(shape)


def _center_connected_support(
    candidate: np.ndarray,
    candidate_views: np.ndarray,
    candidate_spread: np.ndarray,
    config: GroundExtractionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only low-surface candidates reachable from the known ground origin."""
    cells = candidate.shape[0]
    cell_m = config.support_cell_m
    coords = (np.arange(cells) + 0.5) * cell_m - config.crop_radius_m
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    present = np.isfinite(candidate)
    provisional, _ = _nearest_fill(candidate)
    local_median = median_filter(provisional, size=3, mode="nearest")
    neighborhood_count = convolve(present.astype(np.uint8), np.ones((3, 3)), mode="constant")
    plausible = (
        present
        & (np.abs(candidate - local_median) <= config.support_outlier_m)
        & (neighborhood_count >= 4)
    )
    seed = (
        plausible
        & (xx * xx + yy * yy <= config.seed_radius_m**2)
        & (np.abs(candidate) <= config.seed_height_tolerance_m)
    )
    if not seed.any():
        raise ValueError("no multi-view ground support near LUKE's known (0,0,0) origin")

    accepted = np.zeros_like(plausible)
    queue: list[tuple[float, int, int]] = []
    for sy, sx in np.argwhere(seed):
        accepted[sy, sx] = True
        heapq.heappush(queue, (float(xx[sy, sx] ** 2 + yy[sy, sx] ** 2), int(sy), int(sx)))
    gap = config.max_connected_gap_cells
    offsets = [
        (dy, dx)
        for dy in range(-gap, gap + 1)
        for dx in range(-gap, gap + 1)
        if (dy or dx) and dy * dy + dx * dx <= gap * gap
    ]
    while queue:
        _, y0, x0 = heapq.heappop(queue)
        for dy, dx in offsets:
            y1, x1 = y0 + dy, x0 + dx
            if not (0 <= y1 < cells and 0 <= x1 < cells):
                continue
            if accepted[y1, x1] or not plausible[y1, x1]:
                continue
            distance_m = np.hypot(dy, dx) * cell_m
            allowed_step = (
                config.connected_step_slack_m + config.max_connected_slope * distance_m
            )
            if abs(candidate[y1, x1] - candidate[y0, x0]) > allowed_step:
                continue
            accepted[y1, x1] = True
            heapq.heappush(
                queue, (float(xx[y1, x1] ** 2 + yy[y1, x1] ** 2), y1, x1)
            )

    connected = np.where(accepted, candidate, np.nan)
    accepted_confidence = np.zeros_like(candidate)
    accepted_confidence[accepted] = (
        np.minimum(candidate_views[accepted] / config.min_support_views, 1.0)
        * np.exp(-((candidate_spread[accepted] / config.max_view_disagreement_m) ** 2))
    )
    return connected, accepted_confidence


def _crop_chunk(chunk: PointChunk, radius: float) -> PointChunk:
    x, y, z, view = chunk
    valid = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        & (x * x + y * y <= radius * radius)
        & (view >= 1) & (view <= 5)
    )
    return x[valid], y[valid], z[valid], view[valid]


def _cell_index(x: np.ndarray, y: np.ndarray, origin: float, cell_m: float, cells: int) -> np.ndarray:
    ix = np.floor((x - origin) / cell_m).astype(np.int64)
    iy = np.floor((y - origin) / cell_m).astype(np.int64)
    if np.any((ix < 0) | (iy < 0) | (ix >= cells) | (iy >= cells)):
        raise ValueError("cropped point fell outside extraction grid")
    return iy * cells + ix


def _nearest_fill(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(values)
    if not valid.any():
        raise ValueError("no multi-view TLS support survived ground extraction")
    distance, indices = distance_transform_edt(~valid, return_indices=True)
    return values[tuple(indices)], distance


def _fill_small_holes(
    values: np.ndarray, valid: np.ndarray, max_cells: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.where(valid, values, np.nan)
    filled, distance = _nearest_fill(source)
    _, nearest_indices = distance_transform_edt(~valid, return_indices=True)
    holes = (~valid) & (distance <= max_cells)
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64) * 0.25
    for _ in range(max(4, max_cells * 4)):
        relaxed = convolve(filled, kernel, mode="nearest")
        filled[holes] = relaxed[holes]
    filled[~(valid | holes)] = np.nan
    return filled, distance, nearest_indices


def _ground_gate_metrics(
    elevation: np.ndarray,
    measured: np.ndarray,
    confidence: np.ndarray,
    support_valid: np.ndarray,
    connected_support: np.ndarray,
    config: GroundExtractionConfig,
    source_id: str,
) -> dict[str, float | int]:
    fine_cells = elevation.shape[0]
    fine_axis = (np.arange(fine_cells) + 0.5) * config.texel_m - config.crop_radius_m
    fine_inside = fine_axis[:, None] ** 2 + fine_axis[None, :] ** 2 <= config.crop_radius_m**2
    support_cells = support_valid.shape[0]
    support_axis = (
        (np.arange(support_cells) + 0.5) * config.support_cell_m - config.crop_radius_m
    )
    support_inside = (
        support_axis[:, None] ** 2 + support_axis[None, :] ** 2 <= config.crop_radius_m**2
    )
    connected_fraction = float(
        np.count_nonzero(support_valid & support_inside) / max(1, support_inside.sum())
    )
    measured_fraction = float(measured.sum() / max(1, fine_inside.sum()))

    closed = binary_closing(measured, structure=np.ones((3, 3))) & fine_inside
    components, count = label(closed, structure=np.ones((3, 3)))
    component_sizes = np.bincount(components.ravel())[1:] if count else np.empty(0, dtype=int)
    largest_fraction = float(component_sizes.max() / max(1, closed.sum())) if count else 0.0

    z = elevation[measured]
    if not z.size:
        raise ValueError("ground extraction produced no measured elevation")
    p01, p99 = np.percentile(z, [1, 99])
    steps: list[np.ndarray] = []
    horizontal = measured[:, 1:] & measured[:, :-1]
    vertical = measured[1:, :] & measured[:-1, :]
    if horizontal.any():
        steps.append(np.abs(np.diff(elevation, axis=1))[horizontal])
    if vertical.any():
        steps.append(np.abs(np.diff(elevation, axis=0))[vertical])
    all_steps = np.concatenate(steps) if steps else np.asarray([np.inf])

    useful = 0
    p = config.gate_patch_cells
    valid_surface = np.isfinite(elevation)
    for y0 in range(0, fine_cells - p + 1, config.gate_patch_stride):
        for x0 in range(0, fine_cells - p + 1, config.gate_patch_stride):
            sl = np.s_[y0 : y0 + p, x0 : x0 + p]
            if (
                np.mean(valid_surface[sl]) >= 0.995
                and np.mean(measured[sl]) >= config.min_useful_direct_fraction
                and np.mean(confidence[sl]) >= 0.35
            ):
                useful += 1
    metrics: dict[str, float | int] = {
        "connected_support_fraction": connected_fraction,
        "measured_fraction": measured_fraction,
        "largest_component_fraction": largest_fraction,
        "vertical_p01_m": float(p01),
        "vertical_p99_m": float(p99),
        "vertical_span_p01_p99_m": float(p99 - p01),
        "max_abs_elevation_m": float(np.max(np.abs(z))),
        "max_fine_step_m": float(np.max(all_steps)),
        "p99_fine_slope": float(np.percentile(all_steps / config.texel_m, 99)),
        "useful_patch_windows": useful,
        "connected_support_min_m": float(np.nanmin(connected_support)),
        "connected_support_max_m": float(np.nanmax(connected_support)),
    }
    failures: list[str] = []
    checks = (
        (connected_fraction >= config.min_connected_support_fraction, "connected support"),
        (measured_fraction >= config.min_measured_fraction, "measured coverage"),
        (largest_fraction >= config.min_largest_component_fraction, "largest component"),
        (p99 - p01 <= config.max_vertical_span_m, "vertical span"),
        (np.max(np.abs(z)) <= config.max_abs_elevation_m, "absolute elevation"),
        (np.max(all_steps) <= config.max_fine_step_m, "fine step"),
        (
            np.percentile(all_steps / config.texel_m, 99) <= config.max_p99_fine_slope,
            "p99 fine slope",
        ),
        (useful >= config.min_useful_patch_windows, "useful patch windows"),
    )
    failures.extend(name for passed, name in checks if not passed)
    if failures:
        raise ValueError(
            f"ground extraction hard gates failed for {source_id} ("
            + ", ".join(failures)
            + "): "
            + repr(metrics)
        )
    return metrics


def _extract_from_chunks(
    chunk_factory: Callable[[], Iterator[PointChunk]],
    *,
    source_id: str,
    config: GroundExtractionConfig,
    source_provenance: dict | None,
) -> GroundSurface:
    if source_id not in LAPINJARVI_ROLES:
        raise ValueError(f"unsupported Lapinjarvi source_id {source_id!r}")
    diameter = 2.0 * config.crop_radius_m
    fine_cells = int(round(diameter / config.texel_m))
    support_cells = int(round(diameter / config.support_cell_m))
    if not np.isclose(fine_cells * config.texel_m, diameter):
        raise ValueError("crop diameter must be an integer number of fine cells")
    if not np.isclose(support_cells * config.support_cell_m, diameter):
        raise ValueError("crop diameter must be an integer number of support cells")
    origin = -config.crop_radius_m
    support_area = support_cells * support_cells
    support_samples = _CellReservoir(5 * support_area, config.reservoir_points)
    input_points = cropped_points = 0

    # Pass 1: an independently sampled low envelope for each retained scanner view.
    for raw in chunk_factory():
        input_points += len(raw[0])
        x, y, z, view = _crop_chunk(raw, config.crop_radius_m)
        cropped_points += len(x)
        flat = _cell_index(x, y, origin, config.support_cell_m, support_cells)
        group = (view.astype(np.int64) - 1) * support_area + flat
        support_samples.update(group, z, _sample_hash(x, y, z, view))
    if cropped_points < 100:
        raise ValueError("too few valid multi-view TLS points inside the safe crop")
    candidate, candidate_views, candidate_spread = _lowest_multiview_clusters(
        support_samples, support_cells, config
    )
    connected_support, connected_confidence = _center_connected_support(
        candidate, candidate_views, candidate_spread, config
    )
    support_valid = np.isfinite(connected_support)
    support_coords = np.argwhere(support_valid)
    if len(support_coords) < 3:
        raise ValueError("connected ground manifold has fewer than three support cells")
    grid_y, grid_x = np.mgrid[:support_cells, :support_cells]
    # Linear interpolation is permitted only inside the accepted manifold's hull and
    # within a bounded distance of actual connected support.
    support_interpolated = griddata(
        support_coords[:, ::-1], connected_support[support_valid],
        (grid_x, grid_y), method="linear", fill_value=np.nan,
    )
    support_distance, nearest_indices = distance_transform_edt(
        ~support_valid, return_indices=True
    )
    support_evidence = (
        np.isfinite(support_interpolated)
        & (support_distance <= config.max_connected_gap_cells)
    )
    support_clean = np.where(support_valid, connected_support, support_interpolated)
    support_confidence = np.where(
        support_valid,
        connected_confidence,
        connected_confidence[tuple(nearest_indices)]
        * np.clip(
            1.0 - support_distance / (config.max_connected_gap_cells + 1.0), 0.0, 1.0
        ),
    )
    support_confidence[~support_evidence] = 0.0
    support_centers = (
        (np.arange(support_cells, dtype=np.float64) + 0.5) * config.support_cell_m
        + origin
    )
    support_height_at = RegularGridInterpolator(
        (support_centers, support_centers), support_clean,
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    support_confidence_at = RegularGridInterpolator(
        (support_centers, support_centers), support_confidence,
        method="linear", bounds_error=False, fill_value=0.0,
    )

    # Pass 2: retain only returns close to supported ground and rasterize robust medians.
    fine_area = fine_cells * fine_cells
    fine_samples = _CellReservoir(5 * fine_area, config.fine_reservoir_points)
    fine_count = np.zeros(fine_area, dtype=np.int64)
    ground_like_points = 0
    for raw in chunk_factory():
        x, y, z, view = _crop_chunk(raw, config.crop_radius_m)
        query = np.column_stack((y, x))
        point_support = support_height_at(query)
        point_support_confidence = support_confidence_at(query)
        dz = z - point_support
        keep = (
            np.isfinite(point_support)
            & (point_support_confidence > 0)
            & (dz >= -config.below_support_m)
            & (dz <= config.above_support_m)
        )
        x, y, z, view = x[keep], y[keep], z[keep], view[keep]
        ground_like_points += len(x)
        flat = _cell_index(x, y, origin, config.texel_m, fine_cells)
        group = (view.astype(np.int64) - 1) * fine_area + flat
        fine_samples.update(group, z, _sample_hash(x, y, z, view))
        np.add.at(fine_count, flat, 1)

    surface, fine_views, fine_spread = _lowest_multiview_clusters(
        fine_samples,
        fine_cells,
        config,
        min_points_per_view=config.fine_min_points_per_view,
        min_cluster_points_per_view=config.fine_min_cluster_points_per_view,
        max_view_disagreement_m=config.fine_max_view_disagreement_m,
    )
    preliminary = np.isfinite(surface)
    preliminary_fill, _ = _nearest_fill(surface)
    local_sheet = median_filter(
        preliminary_fill, size=config.fine_sheet_window_cells, mode="nearest"
    )
    sheet_consistent = preliminary & (
        np.abs(surface - local_sheet) <= config.fine_sheet_outlier_m
    )
    surface[~sheet_consistent] = np.nan
    # A remaining discontinuity above the existing hard maximum is ambiguous object
    # geometry, not a usable heightfield observation. Reject both incident cells.
    reject = np.zeros_like(sheet_consistent)
    dx = np.abs(np.diff(surface, axis=1))
    bad_x = sheet_consistent[:, 1:] & sheet_consistent[:, :-1] & (dx > config.max_fine_step_m)
    reject[:, 1:] |= bad_x
    reject[:, :-1] |= bad_x
    dy = np.abs(np.diff(surface, axis=0))
    bad_y = sheet_consistent[1:, :] & sheet_consistent[:-1, :] & (dy > config.max_fine_step_m)
    reject[1:, :] |= bad_y
    reject[:-1, :] |= bad_y
    surface[reject] = np.nan
    point_count = fine_count.reshape(fine_cells, fine_cells)
    measured = np.isfinite(surface)
    max_fill_cells = int(np.floor(config.max_fill_hole_m / config.texel_m + 1e-9))
    elevation, fill_distance, fill_nearest = _fill_small_holes(
        surface, measured, max_fill_cells
    )

    centers = (np.arange(fine_cells, dtype=np.float64) + 0.5) * config.texel_m
    radial_centers = (centers[:, None] - config.crop_radius_m) ** 2 + (
        centers[None, :] - config.crop_radius_m
    ) ** 2 <= config.crop_radius_m**2
    fine_local = centers - config.crop_radius_m
    fine_y, fine_x = np.meshgrid(fine_local, fine_local, indexing="ij")
    support_score_fine = support_confidence_at(
        np.column_stack((fine_y.ravel(), fine_x.ravel()))
    ).reshape(fine_cells, fine_cells)
    density_score = np.minimum(point_count / max(1, config.min_cell_points * 4), 1.0)
    fine_view_score = np.minimum(fine_views / config.min_support_views, 1.0)
    fine_agreement_score = np.exp(
        -((fine_spread / config.fine_max_view_disagreement_m) ** 2)
    )
    direct_confidence = density_score * fine_view_score * fine_agreement_score * support_score_fine
    confidence = np.where(
        measured,
        direct_confidence,
        direct_confidence[tuple(fill_nearest)]
        * np.clip(1.0 - fill_distance / (max_fill_cells + 1.0), 0.0, 1.0),
    )
    confidence[~np.isfinite(elevation)] = 0.0
    measured &= radial_centers
    confidence *= radial_centers
    elevation[~radial_centers] = np.nan
    gates = _ground_gate_metrics(
        elevation, measured, confidence, support_valid, connected_support, config, source_id
    )
    provenance = {
        "dataset": "LUKE Lapinjarvi Forest TLS 2016",
        "license": "CC-BY-4.0",
        "source_id": source_id,
        "role": LAPINJARVI_ROLES[source_id],
        "input_points": int(input_points),
        "cropped_points": int(cropped_points),
        "ground_like_points": int(ground_like_points),
        "sampling": "deterministic lowest-hash reservoir per raster cell and scanner view",
        "candidate_support_fraction": float(np.isfinite(candidate).mean()),
        "connected_support_fraction": gates["connected_support_fraction"],
        "measured_fraction_in_crop": float(measured.sum() / max(1, radial_centers.sum())),
        "hard_gates": gates,
        "config": vars(config),
    }
    if source_provenance:
        provenance["source"] = dict(source_provenance)
    return GroundSurface(
        elevation_m=elevation,
        measured=measured,
        confidence=confidence,
        origin_x_m=origin,
        origin_y_m=origin,
        texel_m=config.texel_m,
        source_id=source_id,
        role=LAPINJARVI_ROLES[source_id],
        provenance=provenance,
    )


def extract_ground_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    scan_id: np.ndarray,
    source_id: str,
    config: GroundExtractionConfig = GroundExtractionConfig(),
    source_provenance: dict | None = None,
) -> GroundSurface:
    """Array entry point for tests/small captures; production LAZ uses two streamed passes."""
    arrays = tuple(np.asarray(v) for v in (x, y, z, scan_id))
    if any(v.ndim != 1 or v.shape != arrays[0].shape for v in arrays):
        raise ValueError("x, y, z, scan_id must be equal-length one-dimensional arrays")
    return _extract_from_chunks(
        lambda: iter((arrays,)), source_id=source_id, config=config,
        source_provenance=source_provenance,
    )


def prepare_lapinjarvi_laz(
    path: Path,
    *,
    source_id: str,
    sha256: str,
    config: GroundExtractionConfig = GroundExtractionConfig(),
) -> GroundSurface:
    return _extract_from_chunks(
        lambda: iter_laz_chunks(path), source_id=source_id, config=config,
        source_provenance={"path": str(path), "sha256": sha256},
    )


prepare_lapinjavi_laz = prepare_lapinjarvi_laz


def _smooth_measured_trend(elevation: np.ndarray, factor: int) -> np.ndarray:
    """Cubic interpolation of measured 1 m block means, with no authority projection."""
    h, w = elevation.shape
    if h % factor or w % factor:
        raise ValueError("surface dimensions must be divisible by coarse_factor")
    valid = np.isfinite(elevation)
    safe = np.where(valid, elevation, 0.0)
    values = safe.reshape(h // factor, factor, w // factor, factor).sum(axis=(1, 3))
    weights = valid.reshape(h // factor, factor, w // factor, factor).sum(axis=(1, 3))
    coarse = np.divide(values, weights, out=np.full_like(values, np.nan), where=weights > 0)
    coarse_filled, _ = _nearest_fill(coarse)
    coarse_y = (np.arange(coarse.shape[0]) + 0.5) * factor
    coarse_x = (np.arange(coarse.shape[1]) + 0.5) * factor
    fine_y = np.arange(h) + 0.5
    fine_x = np.arange(w) + 0.5
    spline = RectBivariateSpline(coarse_y, coarse_x, coarse_filled, kx=3, ky=3, s=0)
    return spline(fine_y, fine_x)


def build_residual_bank(
    surfaces: Iterable[GroundSurface],
    *,
    patch_cells: int = 64,
    overlap_cells: int = 16,
    stride_cells: int = 16,
    coarse_factor: int = 16,
    min_valid_fraction: float = 0.995,
    min_direct_fraction: float = 0.70,
    min_mean_confidence: float = 0.35,
    max_fill_distance_cells: float = 2.0,
    require_approved: bool = True,
) -> PatchBank:
    """Build unscaled measured residual patches; holdouts can never enter calibration."""
    items = list(surfaces)
    if not items:
        raise ValueError("no ground surfaces")
    if any(s.role != "calibration" for s in items):
        raise ValueError("holdout or unknown-role surface cannot enter the calibration bank")
    if require_approved and any(s.qa_status != "approved" for s in items):
        raise ValueError("all calibration ground surfaces require explicit QA approval")
    texel = items[0].texel_m
    if any(not np.isclose(s.texel_m, texel) for s in items):
        raise ValueError("all ground surfaces must use the same texel size")
    if patch_cells % coarse_factor or stride_cells <= 0:
        raise ValueError("patch_cells must be coarse-factor aligned and stride positive")

    patches: list[np.ndarray] = []
    quality: list[float] = []
    source_index: list[int] = []
    source_ids = tuple(s.source_id for s in items)
    rejected = 0
    for source_i, surface in enumerate(items):
        elevation = np.asarray(surface.elevation_m, dtype=np.float64)
        residual = elevation - _smooth_measured_trend(elevation, coarse_factor)
        valid = np.isfinite(residual)
        fill_distance, fill_nearest = distance_transform_edt(
            ~valid, return_distances=True, return_indices=True
        )
        residual_filled = residual[tuple(fill_nearest)]
        h, w = elevation.shape
        for y0 in range(0, h - patch_cells + 1, stride_cells):
            for x0 in range(0, w - patch_cells + 1, stride_cells):
                sl = np.s_[y0 : y0 + patch_cells, x0 : x0 + patch_cells]
                valid_fraction = float(np.mean(valid[sl]))
                direct_fraction = float(np.mean(surface.measured[sl]))
                mean_conf = float(np.mean(surface.confidence[sl]))
                candidate = residual_filled[sl]
                if (
                    valid_fraction < min_valid_fraction
                    or direct_fraction < min_direct_fraction
                    or mean_conf < min_mean_confidence
                    or np.max(fill_distance[sl]) > max_fill_distance_cells
                ):
                    rejected += 1
                    continue
                patches.append(candidate.copy())
                quality.append(valid_fraction * direct_fraction * mean_conf)
                source_index.append(source_i)
    if not patches:
        raise ValueError("no residual patches passed coverage/confidence gates")
    missing_sources = [
        source_ids[i] for i in range(len(source_ids)) if i not in source_index
    ]
    if missing_sources:
        raise ValueError(
            "calibration source produced no accepted residual patches: "
            + ", ".join(missing_sources)
        )
    bank = PatchBank(
        patches_m=np.stack(patches),
        quality=np.asarray(quality),
        source_index=np.asarray(source_index),
        source_ids=source_ids,
        texel_m=texel,
        overlap_cells=overlap_cells,
        provenance={
            "method": "measured TLS residual below cubic interpolation of 1 m block means",
            "authorityProjection": "deferred to cook; absent from exemplar bank",
            "patch_cells": patch_cells,
            "stride_cells": stride_cells,
            "coarse_factor": coarse_factor,
            "min_valid_fraction": min_valid_fraction,
            "min_direct_fraction": min_direct_fraction,
            "min_mean_confidence": min_mean_confidence,
            "max_fill_distance_cells": max_fill_distance_cells,
            "accepted_patches": len(patches),
            "acceptedBySource": {
                source_ids[i]: int(np.count_nonzero(np.asarray(source_index) == i))
                for i in range(len(source_ids))
            },
            "rejected_windows": rejected,
            "sources": [s.provenance for s in items],
        },
    )
    bank.validate()
    return bank
