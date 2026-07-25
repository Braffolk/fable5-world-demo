#!/usr/bin/env python3
"""Cook-side Candidate-H angular continuity gate.

This analysis never pairs adjacent direction charts at the same address.  A
filtered source atom is converted to its representative world point, that point
is reprojected to the target direction's top plane, and the target atom is read
at the corresponding unwrapped nearest texel centre.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image, ImageDraw
from scipy import ndimage


ANALYSIS_VERSION = "candidate-h-angular-continuity-v1"
MISS = np.uint32(0xFFFFFFFF)
SCALES = (4, 16)
DISTANCES_M = (2.0, 5.0, 10.0)
FOV_Y_DEGREES = 55.0
REVIEW_HEIGHT_PX = 1440
THRESHOLDS = {
    "coverage_p95": 0.08,
    "coverage_p99": 0.20,
    "premul_rgb_p95": 0.06,
    "premul_rgb_p99": 0.15,
    "transverse_px_p95": 0.75,
    "transverse_px_p99": 1.5,
    "connected_exceedance_fraction": 0.01,
}


@dataclass(frozen=True)
class Slice:
    direction: np.ndarray
    depth_min: float
    depth_max: float


@dataclass
class Profile:
    raw: bytes
    path: Path
    stored_width: int
    stored_height: int
    interior_width: int
    interior_height: int
    atlas_columns: int
    atlas_rows: int
    gutter: int
    top_h: float
    origin_x: float
    origin_z: float
    size_x: float
    size_z: float
    slices: list[Slice]
    texels: np.ndarray
    owners: np.ndarray
    vertices: np.ndarray
    triangles: np.ndarray
    bounds: np.ndarray


@dataclass
class AtomSet:
    coverage: np.ndarray
    depth: np.ndarray
    premul_rgb: np.ndarray


def u32(raw: bytes, offset: int) -> int:
    return struct.unpack_from("<I", raw, offset)[0]


def f32(raw: bytes, offset: int) -> float:
    return struct.unpack_from("<f", raw, offset)[0]


def load_profile(path: Path) -> Profile:
    raw = path.read_bytes()
    if raw[:4] != b"GCRP" or u32(raw, 4) != 4:
        raise ValueError("Candidate H gate requires GCRP/v4")
    stored_width = u32(raw, 12)
    stored_height = u32(raw, 16)
    atlas_columns = u32(raw, 20)
    atlas_rows = u32(raw, 24)
    slice_count = u32(raw, 28)
    if u32(raw, 32) != 8 or u32(raw, 76) != 128:
        raise ValueError("non-canonical GCRP/v4 layout")
    payload_offset = u32(raw, 36)
    interior_width = u32(raw, 44)
    interior_height = u32(raw, 48)
    gutter = u32(raw, 52)
    owner_offset = u32(raw, 80)
    vertex_offset = u32(raw, 84)
    triangle_offset = u32(raw, 88)
    vertex_count = u32(raw, 92)
    triangle_count = u32(raw, 96)
    atlas_width = stored_width * atlas_columns
    atlas_height = stored_height * atlas_rows
    texel_count = atlas_width * atlas_height
    slices: list[Slice] = []
    for index in range(slice_count):
        base = 128 + index * 64
        direction = np.array([f32(raw, base), f32(raw, base + 4), f32(raw, base + 8)], dtype=np.float64)
        slices.append(Slice(direction, f32(raw, base + 12), f32(raw, base + 16)))
    return Profile(
        raw=raw,
        path=path,
        stored_width=stored_width,
        stored_height=stored_height,
        interior_width=interior_width,
        interior_height=interior_height,
        atlas_columns=atlas_columns,
        atlas_rows=atlas_rows,
        gutter=gutter,
        top_h=f32(raw, 56),
        origin_x=f32(raw, 60),
        origin_z=f32(raw, 64),
        size_x=f32(raw, 68),
        size_z=f32(raw, 72),
        slices=slices,
        texels=np.frombuffer(raw, dtype="<u2", count=texel_count * 4, offset=payload_offset).reshape(atlas_height, atlas_width, 4),
        owners=np.frombuffer(raw, dtype="<u4", count=texel_count, offset=owner_offset).reshape(atlas_height, atlas_width),
        vertices=np.frombuffer(raw, dtype="<u4", count=vertex_count * 4, offset=vertex_offset).reshape(vertex_count, 4),
        triangles=np.frombuffer(raw, dtype="<u4", count=triangle_count * 4, offset=triangle_offset).reshape(triangle_count, 4),
        bounds=np.array([f32(raw, offset) for offset in (100, 104, 108, 112, 116, 120)], dtype=np.float64),
    )


def slice_interior(profile: Profile, slice_index: int) -> tuple[np.ndarray, np.ndarray]:
    column = slice_index % profile.atlas_columns
    row = slice_index // profile.atlas_columns
    x0 = column * profile.stored_width + profile.gutter
    y0 = row * profile.stored_height + profile.gutter
    return (
        profile.texels[y0 : y0 + profile.interior_height, x0 : x0 + profile.interior_width],
        profile.owners[y0 : y0 + profile.interior_height, x0 : x0 + profile.interior_width],
    )


def lo16(value: np.ndarray) -> np.ndarray:
    return (value & np.uint32(0xFFFF)).astype(np.float64)


def hi16(value: np.ndarray) -> np.ndarray:
    return (value >> np.uint32(16)).astype(np.float64)


def decode_slice_base(profile: Profile, slice_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geometry, owners = slice_interior(profile, slice_index)
    covered = (owners != MISS) & (geometry[..., 3] != 0)
    slice_data = profile.slices[slice_index]
    depth = np.full(covered.shape, np.nan, dtype=np.float32)
    depth[covered] = (
        slice_data.depth_min
        + geometry[..., 0][covered].astype(np.float64) / 65535.0
        * (slice_data.depth_max - slice_data.depth_min)
    ).astype(np.float32)
    rgb = np.zeros((*covered.shape, 3), dtype=np.float32)
    flat_indices = np.flatnonzero(covered.ravel())
    if flat_indices.size == 0:
        return covered, depth, rgb

    y_index, x_index = np.divmod(flat_indices, profile.interior_width)
    tokens = owners.ravel()[flat_indices]
    triangle_ids = (tokens & np.uint32(0x3FFFFF)).astype(np.int64)
    copy_x = (((tokens >> np.uint32(22)) & np.uint32(0x1F)).astype(np.int32) - 16).astype(np.float64)
    copy_z = (((tokens >> np.uint32(27)) & np.uint32(0x1F)).astype(np.int32) - 16).astype(np.float64)
    vertex_ids = profile.triangles[triangle_ids, :3].astype(np.int64)
    records = profile.vertices[vertex_ids]
    unit = 1.0 / 65535.0
    span = profile.bounds[3:] - profile.bounds[:3]

    positions = np.empty((flat_indices.size, 3, 3), dtype=np.float64)
    positions[..., 0] = profile.bounds[0] + lo16(records[..., 0]) * unit * span[0] + copy_x[:, None] * profile.size_x
    positions[..., 1] = profile.bounds[1] + hi16(records[..., 0]) * unit * span[1]
    positions[..., 2] = profile.bounds[2] + lo16(records[..., 1]) * unit * span[2] + copy_z[:, None] * profile.size_z
    vertex_rgb = np.empty((flat_indices.size, 3, 3), dtype=np.float64)
    vertex_rgb[..., 0] = hi16(records[..., 1]) * unit
    vertex_rgb[..., 1] = lo16(records[..., 2]) * unit
    vertex_rgb[..., 2] = hi16(records[..., 2]) * unit

    origin = np.empty((flat_indices.size, 3), dtype=np.float64)
    origin[:, 0] = profile.origin_x + (x_index + 0.5) / profile.interior_width * profile.size_x
    origin[:, 1] = profile.top_h
    origin[:, 2] = profile.origin_z + (1.0 - (y_index + 0.5) / profile.interior_height) * profile.size_z
    a = positions[:, 0] - origin
    e1 = positions[:, 1] - positions[:, 0]
    e2 = positions[:, 2] - positions[:, 0]
    direction = slice_data.direction
    pvec = np.cross(np.broadcast_to(direction, e2.shape), e2)
    determinant = np.einsum("ij,ij->i", e1, pvec)
    weights = np.full((flat_indices.size, 3), 1.0 / 3.0, dtype=np.float64)
    valid = np.abs(determinant) > 1e-12
    if np.any(valid):
        tvec = -a[valid]
        inverse = 1.0 / determinant[valid]
        wb = np.einsum("ij,ij->i", tvec, pvec[valid]) * inverse
        qvec = np.cross(tvec, e1[valid])
        wc = np.einsum("ij,j->i", qvec, direction) * inverse
        wa = 1.0 - wb - wc
        w = np.maximum(np.stack((wa, wb, wc), axis=1), 0.0)
        sums = np.sum(w, axis=1)
        good = sums > 1e-12
        w[good] /= sums[good, None]
        w[~good] = 1.0 / 3.0
        weights[valid] = w
    hit_rgb = np.einsum("ni,nic->nc", weights, vertex_rgb)
    rgb.reshape(-1, 3)[flat_indices] = np.clip(hit_rgb, 0.0, 1.0).astype(np.float32)
    return covered, depth, rgb


def toroidal_nanmedian(values: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(values, ((radius, radius), (radius, radius)), mode="wrap")
    windows = sliding_window_view(padded, (2 * radius + 1, 2 * radius + 1))
    output = np.empty(values.shape, dtype=np.float32)
    chunk_rows = 24 if radius <= 4 else 3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for y0 in range(0, values.shape[0], chunk_rows):
            y1 = min(values.shape[0], y0 + chunk_rows)
            output[y0:y1] = np.nanmedian(windows[y0:y1], axis=(-2, -1)).astype(np.float32)
    return output


def build_atoms(profile: Profile) -> tuple[dict[int, AtomSet], list[np.ndarray]]:
    regular_count = len(profile.slices) - 1
    shape = (regular_count, profile.interior_height, profile.interior_width)
    atoms = {
        scale: AtomSet(
            coverage=np.empty(shape, dtype=np.float32),
            depth=np.empty(shape, dtype=np.float32),
            premul_rgb=np.empty((*shape, 3), dtype=np.float32),
        )
        for scale in SCALES
    }
    near_samples: list[np.ndarray] = []
    for slice_index in range(regular_count):
        covered, depth, rgb = decode_slice_base(profile, slice_index)
        hit_rgb = rgb[covered]
        if hit_rgb.size:
            near_samples.append(hit_rgb[:: max(1, hit_rgb.shape[0] // 2048)])
        for scale in SCALES:
            size = 2 * scale + 1
            coverage = ndimage.uniform_filter(covered.astype(np.float32), size=size, mode="wrap")
            coverage = np.rint(np.clip(coverage, 0.0, 1.0) * 255.0).astype(np.float32) / 255.0
            premul = np.empty((*covered.shape, 3), dtype=np.float32)
            for channel in range(3):
                premul[..., channel] = ndimage.uniform_filter(
                    rgb[..., channel] * covered,
                    size=size,
                    mode="wrap",
                )
            median = toroidal_nanmedian(depth, scale)
            slice_data = profile.slices[slice_index]
            normalised = (median - slice_data.depth_min) / (slice_data.depth_max - slice_data.depth_min)
            code = np.rint(np.clip(normalised, 0.0, 1.0) * 1023.0)
            quantised = slice_data.depth_min + code / 1023.0 * (slice_data.depth_max - slice_data.depth_min)
            quantised[~np.isfinite(median)] = np.nan
            atoms[scale].coverage[slice_index] = coverage
            atoms[scale].depth[slice_index] = quantised.astype(np.float32)
            atoms[scale].premul_rgb[slice_index] = premul
        print(f"[candidate-h] filtered slice {slice_index + 1}/{regular_count}", flush=True)
    return atoms, near_samples


def fit_palette(atoms: dict[int, AtomSet], near_samples: list[np.ndarray]) -> np.ndarray:
    samples = list(near_samples)
    for scale in SCALES:
        atom = atoms[scale]
        flat_a = atom.coverage.reshape(-1)
        flat_p = atom.premul_rgb.reshape(-1, 3)
        valid = np.flatnonzero(flat_a > 1.0 / 255.0)
        if valid.size:
            chosen = valid[:: max(1, valid.size // 80_000)][:80_000]
            samples.append(flat_p[chosen] / flat_a[chosen, None])
    sample = np.clip(np.concatenate(samples, axis=0), 0.0, 1.0).astype(np.float32)
    if sample.shape[0] > 240_000:
        rng = np.random.default_rng(0xCADA)
        sample = sample[rng.choice(sample.shape[0], 240_000, replace=False)]
    luminance = sample @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    centers = [sample[int(np.argmax(luminance))]]
    distance = np.sum((sample - centers[0]) ** 2, axis=1)
    for _ in range(15):
        index = int(np.argmax(distance))
        centers.append(sample[index])
        distance = np.minimum(distance, np.sum((sample - centers[-1]) ** 2, axis=1))
    palette = np.stack(centers).astype(np.float32)
    for _ in range(24):
        labels = nearest_class(sample, palette)
        updated = palette.copy()
        for index in range(16):
            selected = sample[labels == index]
            if selected.size:
                updated[index] = np.mean(selected, axis=0)
        if float(np.max(np.abs(updated - palette))) < 1e-6:
            palette = updated
            break
        palette = updated
    return np.clip(palette, 0.0, 1.0)


def nearest_class(colours: np.ndarray, palette: np.ndarray) -> np.ndarray:
    flat = colours.reshape(-1, 3)
    result = np.empty(flat.shape[0], dtype=np.uint8)
    for start in range(0, flat.shape[0], 65_536):
        end = min(flat.shape[0], start + 65_536)
        distance = np.sum((flat[start:end, None, :] - palette[None, :, :]) ** 2, axis=2)
        result[start:end] = np.argmin(distance, axis=1).astype(np.uint8)
    return result.reshape(colours.shape[:-1])


def direction(elevation_degrees: float, azimuth_index: float) -> np.ndarray:
    elevation = math.radians(elevation_degrees)
    azimuth = azimuth_index * math.tau / 16.0
    horizontal = math.cos(elevation)
    return np.array([horizontal * math.cos(azimuth), -math.sin(elevation), horizontal * math.sin(azimuth)], dtype=np.float64)


def lattice_map(profile: Profile) -> dict[tuple[int, int], int]:
    elevations = (15.0, 35.0, 55.0, 75.0)
    result: dict[tuple[int, int], int] = {}
    for index, slice_data in enumerate(profile.slices[:-1]):
        value = slice_data.direction
        elevation = math.degrees(math.asin(-float(value[1])))
        row = min(range(4), key=lambda candidate: abs(elevation - elevations[candidate]))
        azimuth = int(round((math.atan2(float(value[2]), float(value[0])) % math.tau) / math.tau * 16.0)) % 16
        result[(row, azimuth)] = index
    if len(result) != 64:
        raise ValueError(f"expected accepted 16x4 regular lattice, got {len(result)} cells")
    return result


def quantiles(values: list[np.ndarray]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "maximum": 0.0}
    merged = np.concatenate(values).astype(np.float64, copy=False)
    if merged.size == 0:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "maximum": 0.0}
    return {
        "count": int(merged.size),
        "mean": float(np.mean(merged)),
        "p50": float(np.quantile(merged, 0.50)),
        "p95": float(np.quantile(merged, 0.95)),
        "p99": float(np.quantile(merged, 0.99)),
        "maximum": float(np.max(merged)),
    }


def largest_periodic_component(mask: np.ndarray) -> int:
    if not np.any(mask):
        return 0
    tiled = np.tile(mask, (3, 3))
    labels, _ = ndimage.label(tiled, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8))
    height, width = mask.shape
    center = labels[height : 2 * height, width : 2 * width]
    counts = np.bincount(center.ravel())
    return int(np.max(counts[1:])) if counts.size > 1 else 0


class Group:
    def __init__(self, name: str, shape: tuple[int, int]) -> None:
        self.name = name
        self.coverage: list[np.ndarray] = []
        self.rgb: list[np.ndarray] = []
        self.transverse_m: list[np.ndarray] = []
        self.longitudinal_m: list[np.ndarray] = []
        self.delta_tau: list[np.ndarray] = []
        self.class_change: list[np.ndarray] = []
        self.max_connected_fraction = 0.0
        self.max_connected_pair = ""
        self.maps = {
            "coverage": np.zeros(shape, dtype=np.float32),
            "rgb": np.zeros(shape, dtype=np.float32),
            "transverse_px_2m": np.zeros(shape, dtype=np.float32),
            "class_change": np.zeros(shape, dtype=np.float32),
        }


def pixel_footprint(distance: float) -> float:
    return 2.0 * distance * math.tan(math.radians(FOV_Y_DEGREES) * 0.5) / REVIEW_HEIGHT_PX


def compare_pair(
    profile: Profile,
    atom: AtomSet,
    palette: np.ndarray,
    source_slice: int,
    target_slice: int,
    live_direction: np.ndarray,
    pair_name: str,
    group: Group,
) -> None:
    height, width = profile.interior_height, profile.interior_width
    y_index, x_index = np.indices((height, width))
    qx = profile.origin_x + (x_index + 0.5) / width * profile.size_x
    qz = profile.origin_z + (1.0 - (y_index + 0.5) / height) * profile.size_z
    source_a = atom.coverage[source_slice]
    source_depth = atom.depth[source_slice]
    source_valid = (source_a > 1.0 / 255.0) & np.isfinite(source_depth)
    source_direction = profile.slices[source_slice].direction
    target_direction = profile.slices[target_slice].direction
    pa_x = qx + source_depth * source_direction[0]
    pa_y = profile.top_h + source_depth * source_direction[1]
    pa_z = qz + source_depth * source_direction[2]
    target_parameter = (pa_y - profile.top_h) / target_direction[1]
    qb_x = pa_x - target_parameter * target_direction[0]
    qb_z = pa_z - target_parameter * target_direction[2]

    normal_x = (qb_x - profile.origin_x) / profile.size_x
    normal_z = (qb_z - profile.origin_z) / profile.size_z
    tile_x = np.floor(normal_x)
    tile_z = np.floor(normal_z)
    phase_x = normal_x - tile_x
    phase_z = normal_z - tile_z
    target_x = np.floor(phase_x * width).astype(np.int64) % width
    target_y = np.floor((1.0 - phase_z) * height).astype(np.int64) % height
    qb_center_x = profile.origin_x + (tile_x + (target_x + 0.5) / width) * profile.size_x
    qb_center_z = profile.origin_z + (tile_z + 1.0 - (target_y + 0.5) / height) * profile.size_z

    target_a = atom.coverage[target_slice, target_y, target_x]
    target_depth = atom.depth[target_slice, target_y, target_x]
    target_valid = (target_a > 1.0 / 255.0) & np.isfinite(target_depth)
    both = source_valid & target_valid

    source_conditional = np.zeros((height, width, 3), dtype=np.float32)
    source_conditional[source_valid] = atom.premul_rgb[source_slice][source_valid] / source_a[source_valid, None]
    target_premul_raw = atom.premul_rgb[target_slice, target_y, target_x]
    target_conditional = np.zeros((height, width, 3), dtype=np.float32)
    target_conditional[target_valid] = target_premul_raw[target_valid] / target_a[target_valid, None]
    source_class = nearest_class(source_conditional, palette)
    target_class = nearest_class(target_conditional, palette)
    source_compact = source_a[..., None] * palette[source_class]
    target_compact = target_a[..., None] * palette[target_class]

    delta_a = np.abs(source_a - target_a)
    delta_rgb = np.max(np.abs(source_compact - target_compact), axis=2)
    pb_x = qb_center_x + target_depth * target_direction[0]
    pb_y = profile.top_h + target_depth * target_direction[1]
    pb_z = qb_center_z + target_depth * target_direction[2]
    delta = np.stack((pb_x - pa_x, pb_y - pa_y, pb_z - pa_z), axis=2)
    longitudinal_signed = np.einsum("ijk,k->ij", delta, live_direction)
    transverse = np.sqrt(np.maximum(0.0, np.sum(delta * delta, axis=2) - longitudinal_signed * longitudinal_signed))
    longitudinal = np.abs(longitudinal_signed)
    delta_tau = np.abs(target_depth - source_depth)
    class_change = (source_class != target_class).astype(np.float32)

    group.coverage.append(delta_a[source_valid].astype(np.float32))
    group.rgb.append(delta_rgb[source_valid].astype(np.float32))
    group.transverse_m.append(transverse[both].astype(np.float32))
    group.longitudinal_m.append(longitudinal[both].astype(np.float32))
    group.delta_tau.append(delta_tau[both].astype(np.float32))
    group.class_change.append(class_change[both].astype(np.float32))

    transverse_px_2m = transverse / pixel_footprint(2.0)
    exceed = source_valid & (
        (delta_a > THRESHOLDS["coverage_p99"])
        | (delta_rgb > THRESHOLDS["premul_rgb_p99"])
        | (both & (transverse_px_2m > THRESHOLDS["transverse_px_p99"]))
    )
    support_count = int(np.count_nonzero(source_valid))
    component = largest_periodic_component(exceed)
    fraction = component / support_count if support_count else 0.0
    if fraction > group.max_connected_fraction:
        group.max_connected_fraction = fraction
        group.max_connected_pair = pair_name

    group.maps["coverage"] = np.maximum(group.maps["coverage"], np.where(source_valid, delta_a, 0.0))
    group.maps["rgb"] = np.maximum(group.maps["rgb"], np.where(source_valid, delta_rgb, 0.0))
    group.maps["transverse_px_2m"] = np.maximum(group.maps["transverse_px_2m"], np.where(both, transverse_px_2m, 0.0))
    group.maps["class_change"] = np.maximum(group.maps["class_change"], np.where(both, class_change, 0.0))


def finish_group(group: Group) -> dict[str, Any]:
    coverage = quantiles(group.coverage)
    rgb = quantiles(group.rgb)
    transverse_m = quantiles(group.transverse_m)
    transverse_px = {
        f"{distance:g}m": {
            key: (value / pixel_footprint(distance) if key not in ("count",) else value)
            for key, value in transverse_m.items()
        }
        for distance in DISTANCES_M
    }
    longitudinal = quantiles(group.longitudinal_m)
    delta_tau = quantiles(group.delta_tau)
    class_change = quantiles(group.class_change)
    checks: dict[str, bool] = {
        "coverageP95": float(coverage["p95"]) <= THRESHOLDS["coverage_p95"],
        "coverageP99": float(coverage["p99"]) <= THRESHOLDS["coverage_p99"],
        "rgbP95": float(rgb["p95"]) <= THRESHOLDS["premul_rgb_p95"],
        "rgbP99": float(rgb["p99"]) <= THRESHOLDS["premul_rgb_p99"],
        "connectedRegion": group.max_connected_fraction < THRESHOLDS["connected_exceedance_fraction"],
    }
    for distance in DISTANCES_M:
        value = transverse_px[f"{distance:g}m"]
        checks[f"transverseP95At{distance:g}m"] = float(value["p95"]) <= THRESHOLDS["transverse_px_p95"]
        checks[f"transverseP99At{distance:g}m"] = float(value["p99"]) <= THRESHOLDS["transverse_px_p99"]
    return {
        "name": group.name,
        "green": all(checks.values()),
        "checks": checks,
        "coverageDelta": coverage,
        "premultipliedRgbMaxChannelDelta": rgb,
        "transverseDisplacementMetres": transverse_m,
        "transverseDisplacementPixels": transverse_px,
        "longitudinalDisplacementMetres": longitudinal,
        "depthParameterDelta": delta_tau,
        "classChangeRateSecondary": class_change,
        "largestConnectedExceedance": {
            "fractionOfDirectedSourceSupport": group.max_connected_fraction,
            "pair": group.max_connected_pair,
        },
    }


def heat_colour(values: np.ndarray, limit: float) -> np.ndarray:
    t = np.clip(values / max(limit, 1e-9), 0.0, 1.0)
    rgb = np.empty((*values.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.rint(255.0 * t).astype(np.uint8)
    rgb[..., 1] = np.rint(255.0 * np.minimum(t * 2.0, 1.0) * (1.0 - 0.55 * t)).astype(np.uint8)
    rgb[..., 2] = np.rint(30.0 * (1.0 - t)).astype(np.uint8)
    return rgb


def write_group_png(path: Path, title: str, maps: dict[str, np.ndarray]) -> None:
    panels = [
        ("|delta A| / 0.20", heat_colour(maps["coverage"], 0.20)),
        ("premul RGB / 0.15", heat_colour(maps["rgb"], 0.15)),
        ("transverse px@2m / 1.5", heat_colour(maps["transverse_px_2m"], 1.5)),
        ("palette class change", heat_colour(maps["class_change"], 1.0)),
    ]
    scale = 2
    panel_width = panels[0][1].shape[1] * scale
    panel_height = panels[0][1].shape[0] * scale
    header = 46
    image = Image.new("RGB", (panel_width * len(panels), panel_height + header), (18, 18, 18))
    draw = ImageDraw.Draw(image)
    draw.text((8, 4), title, fill=(255, 255, 255))
    for index, (label, array) in enumerate(panels):
        panel = Image.fromarray(array, mode="RGB").resize((panel_width, panel_height), Image.Resampling.NEAREST)
        image.paste(panel, (index * panel_width, header))
        draw.text((index * panel_width + 8, 24), label, fill=(230, 230, 230))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="src/assets/groundcover/calamagrostis-canescens.gcrp")
    parser.add_argument("--threshold-note", default="docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-24/GRASSPROFILE2-CANDIDATE-H-ANGULAR-CONTINUITY-GATE.md")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source_path = (root / args.source).resolve()
    threshold_path = (root / args.threshold_note).resolve()
    script_path = Path(__file__).resolve()
    source_bytes = source_path.read_bytes()
    source_sha = sha256_bytes(source_bytes)
    script_sha = sha256_bytes(script_path.read_bytes())
    threshold_sha = sha256_bytes(threshold_path.read_bytes())
    recipe = {
        "analysisVersion": ANALYSIS_VERSION,
        "sourceSha256": source_sha,
        "analyzerSha256": script_sha,
        "thresholdNoteSha256": threshold_sha,
        "scales": SCALES,
        "kernel": "unit toroidal square radius sigma",
        "depthStatistic": "covered-sample median quantised to 10-bit per slice",
        "coverageQuantisation": "round UNORM8",
        "palette": "deterministic community-global 16-class RGB k-means including near and both atom scales",
        "thresholds": THRESHOLDS,
        "distancesMetres": DISTANCES_M,
        "fovYDegrees": FOV_Y_DEGREES,
        "reviewHeightPixels": REVIEW_HEIGHT_PX,
    }
    recipe_sha = sha256_bytes(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode())
    output = root / "data/work/groundcover-candidate-h-angular-continuity" / source_sha[:16] / recipe_sha[:16]
    qa = output / "qa"
    output.mkdir(parents=True, exist_ok=True)
    profile = load_profile(source_path)
    if profile.interior_width != 256 or profile.interior_height != 256 or len(profile.slices) != 65:
        raise ValueError("Candidate H gate expected accepted 256-square, 64-direction-plus-pole profile")
    atoms, near_samples = build_atoms(profile)
    palette = fit_palette(atoms, near_samples)
    mapping = lattice_map(profile)
    group_order = ["elevation-25", "elevation-45", "elevation-65", "azimuth-15", "azimuth-35", "azimuth-55", "azimuth-75"]
    all_results: dict[str, Any] = {}
    image_records: list[dict[str, Any]] = []
    image_number = 1
    elevations = (15.0, 35.0, 55.0, 75.0)

    for scale in SCALES:
        groups = {name: Group(name, (profile.interior_height, profile.interior_width)) for name in group_order}
        atom = atoms[scale]
        for low_row, boundary in enumerate((25.0, 45.0, 65.0)):
            high_row = low_row + 1
            group = groups[f"elevation-{int(boundary)}"]
            for azimuth in range(16):
                a = mapping[(low_row, azimuth)]
                b = mapping[(high_row, azimuth)]
                live = direction(boundary, float(azimuth))
                compare_pair(profile, atom, palette, a, b, live, f"row{low_row}-to-{high_row}-az{azimuth}", group)
                compare_pair(profile, atom, palette, b, a, live, f"row{high_row}-to-{low_row}-az{azimuth}", group)
        for row, elevation in enumerate(elevations):
            group = groups[f"azimuth-{int(elevation)}"]
            for azimuth in range(16):
                next_azimuth = (azimuth + 1) % 16
                a = mapping[(row, azimuth)]
                b = mapping[(row, next_azimuth)]
                live = direction(elevation, azimuth + 0.5)
                compare_pair(profile, atom, palette, a, b, live, f"az{azimuth}-to-{next_azimuth}-row{row}", group)
                compare_pair(profile, atom, palette, b, a, live, f"az{next_azimuth}-to-{azimuth}-row{row}", group)

        scale_results: dict[str, Any] = {}
        for name in group_order:
            group = groups[name]
            scale_results[name] = finish_group(group)
            filename = f"{image_number:03d}-{name}-sigma{scale}.png"
            path = qa / filename
            write_group_png(path, f"Candidate H {name}, sigma={scale} texels (worst over directed pairs)", group.maps)
            data = path.read_bytes()
            image_records.append({
                "number": image_number,
                "file": filename,
                "sha256": sha256_bytes(data),
                "dimensions": [2048, 558],
                "interpretation": "Four worst-over-pairs phase maps: coverage delta, compact premultiplied-RGB delta, transverse displacement in pixels at 2 m, and secondary palette-class change. Black is zero; red reaches or exceeds the labelled p99 point threshold.",
                "group": name,
                "scaleTexels": scale,
            })
            image_number += 1
        all_results[f"sigma{scale}"] = scale_results

    every_group = [result for scale_result in all_results.values() for result in scale_result.values()]
    green = all(bool(result["green"]) for result in every_group)
    report = {
        "recipe": {**recipe, "recipeSha256": recipe_sha},
        "source": {
            "path": str(source_path.relative_to(root)),
            "sha256": source_sha,
            "bytes": len(source_bytes),
            "dimensions": [profile.interior_width, profile.interior_height],
            "slices": len(profile.slices),
            "topH": profile.top_h,
            "tile": [profile.origin_x, profile.origin_z, profile.size_x, profile.size_z],
        },
        "method": {
            "pairing": "source filtered representative world point -> exact target-direction top-plane reprojection -> target nearest unwrapped texel centre",
            "targetAddress": "q_b=P_a.xz-((P_a.y-H)/c_b.y)c_b.xz",
            "forbidden": "same-address source/target comparison is never used",
            "directions": "both directions for every 25/45/65-degree aligned row pair and every azimuth midpoint in all four regular rows",
        },
        "paletteLinearRgb": palette.tolist(),
        "results": all_results,
        "decision": {
            "green": green,
            "verdict": "GREEN" if green else "RED",
            "rule": "one RED primary metric or connected-region violation blocks Candidate H before runtime",
        },
    }
    report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (output / "report.json").write_text(report_text)
    report_sha = sha256_bytes(report_text.encode())
    qa_index = {
        "schema": "laas-candidate-h-angular-continuity-qa/v1",
        "sourceSha256": source_sha,
        "recipeSha256": recipe_sha,
        "analyzerSha256": script_sha,
        "thresholdNoteSha256": threshold_sha,
        "reportSha256": report_sha,
        "images": image_records,
    }
    qa_index_text = json.dumps(qa_index, indent=2, sort_keys=True) + "\n"
    (qa / "index.json").write_text(qa_index_text)
    qa_index_sha = sha256_bytes(qa_index_text.encode())

    summary = [
        "# Candidate H angular-continuity gate",
        "",
        f"- Verdict: **{'GREEN' if green else 'RED'}**",
        f"- Source SHA-256: `{source_sha}`",
        f"- Recipe SHA-256: `{recipe_sha}`",
        f"- Analyzer SHA-256: `{script_sha}`",
        f"- Threshold-note SHA-256: `{threshold_sha}`",
        f"- Report SHA-256: `{report_sha}`",
        f"- QA index SHA-256: `{qa_index_sha}`",
        "- Pairing: exact filtered representative world point reprojection; equal addresses were never compared.",
        "",
        "| Scale | Boundary family | A p95/p99 | RGB p95/p99 | transverse px@2m p95/p99 | max connected | Verdict |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for scale in SCALES:
        for name in group_order:
            value = all_results[f"sigma{scale}"][name]
            summary.append(
                f"| {scale} | {name} | {value['coverageDelta']['p95']:.4f}/{value['coverageDelta']['p99']:.4f} "
                f"| {value['premultipliedRgbMaxChannelDelta']['p95']:.4f}/{value['premultipliedRgbMaxChannelDelta']['p99']:.4f} "
                f"| {value['transverseDisplacementPixels']['2m']['p95']:.3f}/{value['transverseDisplacementPixels']['2m']['p99']:.3f} "
                f"| {100.0 * value['largestConnectedExceedance']['fractionOfDirectedSourceSupport']:.3f}% "
                f"| {'GREEN' if value['green'] else 'RED'} |"
            )
    summary.extend([
        "",
        "Frozen thresholds: |delta A| p95<=0.08/p99<=0.20; premultiplied RGB p95<=0.06/p99<=0.15; transverse p95<=0.75 px/p99<=1.5 px at 2 m, 5 m, and 10 m; no connected exceedance region >=1%.",
        "",
        "Exact command:",
        "",
        "```text",
        "uv run --project asset-gen python tools/groundcover-bake/analyze_candidate_h_angular_continuity.py",
        "```",
        "",
    ])
    (output / "SUMMARY.md").write_text("\n".join(summary))
    root_index = root / "data/work/groundcover-candidate-h-angular-continuity/index.json"
    root_index.parent.mkdir(parents=True, exist_ok=True)
    root_index.write_text(json.dumps({
        "current": str((output / "report.json").relative_to(root)),
        "sourceSha256": source_sha,
        "recipeSha256": recipe_sha,
        "analyzerSha256": script_sha,
        "reportSha256": report_sha,
        "qaIndexSha256": qa_index_sha,
        "verdict": "GREEN" if green else "RED",
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "verdict": "GREEN" if green else "RED",
        "output": str(output.relative_to(root)),
        "reportSha256": report_sha,
        "groups": {
            scale: {name: value["green"] for name, value in group_values.items()}
            for scale, group_values in all_results.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
