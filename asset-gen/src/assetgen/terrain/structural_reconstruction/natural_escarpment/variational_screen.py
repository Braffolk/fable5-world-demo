"""Two-dimensional constrained reconstruction of mapped terrain breaks.

The rejected station-wise screen built independent one-dimensional profiles and
then pasted them into nearest-station strips.  This screen instead treats the
structural correction as one residual surface.  Measured low/high shoulders
provide a soft target; a screened membrane solve couples both map dimensions.
The residual is exactly zero at hard exclusions, the finite feature endpoints,
and the reconstruction-corridor boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage, sparse
from scipy.sparse.linalg import cg
import shapely

from ....config import DATA_WORK
from .screen import (
    _bound,
    _canonical_bytes,
    _grid_coords,
    _identity,
    _imagery_support,
    _line_frame,
    _load_line,
    _orient_normals,
    _sample_grid,
    _site_source,
)

_CONFIG_SCHEMA = "laas.natural-escarpment-variational-config/1"
_MANIFEST_SCHEMA = "laas.natural-escarpment-variational/1"
_ARTIFACT_ROOT = DATA_WORK / "terrain" / "natural-escarpment-variational" / "sha256"


def _read_config(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != _CONFIG_SCHEMA:
        raise ValueError("unsupported variational escarpment config")
    if document.get("attempt") not in {1, 2} or document.get("taevaskoda_used") is not False:
        raise ValueError("Development attempt/sentinel boundary differs")
    if [row.get("site_id") for row in document["sites"]] != ["development_a", "development_c"]:
        raise ValueError("Development site order differs")
    if [row.get("target_etak_id") for row in document["sites"]] != [1826743, 9688719]:
        raise ValueError("Development ETAK identities differ")
    return document


def _smootherstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


def _profile_evidence(
    *,
    height: np.ndarray,
    bbox: tuple[float, float, float, float],
    stations: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    imagery_support: np.ndarray,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    radius = float(config["profile_radius_m"])
    step = float(config["profile_step_m"])
    offsets = np.arange(-radius, radius + step * 0.5, step, dtype=np.float64)
    xx = points[:, 0, None] + normals[:, 0, None] * offsets[None, :]
    yy = points[:, 1, None] + normals[:, 1, None] * offsets[None, :]
    profiles = _sample_grid(height, bbox, xx.ravel(), yy.ravel(), 3).reshape(xx.shape)
    smooth = ndimage.gaussian_filter1d(
        profiles,
        sigma=float(config["profile_measurement_sigma_m"]) / step,
        axis=1,
        mode="nearest",
    )
    gradient = np.gradient(smooth, step, axis=1)
    search = np.abs(offsets) <= float(config["break_search_radius_m"])
    first = int(np.flatnonzero(search)[0])
    peak_index = np.argmax(gradient[:, search], axis=1) + first
    peak_gradient = gradient[np.arange(len(points)), peak_index]
    center = offsets[peak_index]

    fraction = float(config["shoulder_gradient_fraction"])
    minimum_half = float(config["minimum_half_width_m"])
    maximum_half = float(config["maximum_half_width_m"])
    foot = np.empty(len(points), dtype=np.float64)
    crest = np.empty(len(points), dtype=np.float64)
    for index, peak in enumerate(peak_index):
        threshold = max(float(peak_gradient[index]) * fraction, 0.02)
        left = peak
        while left > 0 and center[index] - offsets[left] < maximum_half and gradient[index, left] > threshold:
            left -= 1
        right = peak
        while right + 1 < len(offsets) and offsets[right] - center[index] < maximum_half and gradient[index, right] > threshold:
            right += 1
        foot[index] = min(float(offsets[left]), center[index] - minimum_half)
        crest[index] = max(float(offsets[right]), center[index] + minimum_half)

    along_sigma = float(config["along_line_evidence_sigma_m"]) / max(float(np.median(np.diff(stations))), 1e-9)
    center = ndimage.gaussian_filter1d(center, along_sigma, mode="nearest")
    foot = ndimage.gaussian_filter1d(foot, along_sigma, mode="nearest")
    crest = ndimage.gaussian_filter1d(crest, along_sigma, mode="nearest")
    foot = np.minimum(foot, center - minimum_half)
    crest = np.maximum(crest, center + minimum_half)

    def sample_profile(distance: np.ndarray) -> np.ndarray:
        column = np.clip(np.rint((distance + radius) / step).astype(np.int64), 0, len(offsets) - 1)
        return smooth[np.arange(len(points)), column]

    low_height = ndimage.gaussian_filter1d(sample_profile(foot), along_sigma, mode="nearest")
    high_height = ndimage.gaussian_filter1d(sample_profile(crest), along_sigma, mode="nearest")
    relief = high_height - low_height
    valid = (
        (relief >= float(config["minimum_relief_m"]))
        & (relief <= float(config["maximum_relief_m"]))
        & (peak_gradient >= float(config["minimum_peak_gradient"]))
        & (imagery_support >= float(config["minimum_imagery_support"]))
    )
    # Confidence is continuous so isolated weak stations fade instead of making
    # a new along-line boundary.  Invalid runs still become exact zero anchors.
    confidence = ndimage.gaussian_filter1d(valid.astype(np.float64), max(along_sigma * 0.5, 1.0), mode="nearest")
    confidence *= np.clip(imagery_support, 0.0, 1.0)
    return {
        "stations": stations,
        "foot": foot,
        "center": center,
        "crest": crest,
        "low_height": low_height,
        "high_height": high_height,
        "relief": relief,
        "peak_gradient": peak_gradient,
        "imagery_support": imagery_support,
        "valid": valid,
        "confidence": confidence,
    }


def _query_line_frame(
    x: np.ndarray,
    y: np.ndarray,
    line,
    orientation_stations: np.ndarray,
    orientation_normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xx, yy = np.meshgrid(x, y)
    query = shapely.points(xx.ravel(), yy.ravel())
    station = shapely.line_locate_point(line, query)
    projected = shapely.line_interpolate_point(line, station)
    px = shapely.get_x(projected)
    py = shapely.get_y(projected)
    tangent_radius = 0.5
    before = shapely.line_interpolate_point(line, np.maximum(station - tangent_radius, 0.0))
    after = shapely.line_interpolate_point(line, np.minimum(station + tangent_radius, float(line.length)))
    tangent = np.column_stack([
        shapely.get_x(after) - shapely.get_x(before),
        shapely.get_y(after) - shapely.get_y(before),
    ])
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normals = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    evidence_normal = np.column_stack([
        np.interp(station, orientation_stations, orientation_normals[:, 0]),
        np.interp(station, orientation_stations, orientation_normals[:, 1]),
    ])
    flip = np.sum(normals * evidence_normal, axis=1) < 0.0
    normals[flip] *= -1.0
    delta = np.column_stack([xx.ravel() - px, yy.ravel() - py])
    signed = np.sum(delta * normals, axis=1)
    distance = np.linalg.norm(delta, axis=1)
    return station.reshape(xx.shape), signed.reshape(xx.shape), distance.reshape(xx.shape)


def _interp_field(station_grid: np.ndarray, evidence: dict[str, np.ndarray], name: str) -> np.ndarray:
    return np.interp(station_grid.ravel(), evidence["stations"], evidence[name]).reshape(station_grid.shape)


def _eligible_masks(
    arrays: dict[str, np.ndarray],
    source_bbox: tuple[float, float, float, float],
    xx: np.ndarray,
    yy: np.ndarray,
    signed_distance: np.ndarray,
) -> dict[str, np.ndarray]:
    def nearest(name: str) -> np.ndarray:
        return _sample_grid(
            arrays[name].astype(np.float64), source_bbox, xx.ravel(), yy.ravel(), 0
        ).reshape(xx.shape) >= 0.5

    masks = {
        name: nearest(name)
        for name in ("valid", "water", "object", "non_heightfield", "protected_structure", "unknown", "target_feature")
    }
    material = (
        (arrays["soil_feature_index"] >= 0)
        & (arrays["geology_lithology_code"] > 0)
        & (arrays["geology_genesis_code"] > 0)
    )
    masks["material_supported"] = _sample_grid(
        material.astype(np.float64), source_bbox, xx.ravel(), yy.ravel(), 0
    ).reshape(xx.shape) >= 0.5
    target = masks["target_feature"]
    protected_allowed = target | (np.abs(signed_distance) <= 1.5)
    masks["eligible"] = (
        masks["valid"]
        & ~masks["water"]
        & ~masks["object"]
        & ~masks["non_heightfield"]
        & ~masks["unknown"]
        & masks["material_supported"]
        & (~masks["protected_structure"] | protected_allowed)
    )
    masks["forbidden"] = ~masks["eligible"]
    return masks


def _structural_target(
    *,
    c0: np.ndarray,
    station_grid: np.ndarray,
    signed_distance: np.ndarray,
    line_length: float,
    evidence: dict[str, np.ndarray],
    site_config: dict[str, Any],
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    foot = _interp_field(station_grid, evidence, "foot")
    crest = _interp_field(station_grid, evidence, "crest")
    low = _interp_field(station_grid, evidence, "low_height")
    high = _interp_field(station_grid, evidence, "high_height")
    confidence = _interp_field(station_grid, evidence, "confidence")
    normalized = (signed_distance - foot) / np.maximum(crest - foot, 1e-6)
    profile = site_config["profile"]
    phase = _smootherstep((normalized - float(profile["toe_fraction"])) / (
        float(profile["shoulder_fraction"]) - float(profile["toe_fraction"])
    ))
    template = low + phase * (high - low)

    endpoint_distance = np.minimum(station_grid, line_length - station_grid)
    endpoint = _smootherstep(
        (endpoint_distance - float(config["endpoint_zero_m"]))
        / float(config["endpoint_taper_m"])
    )
    outer_distance = np.maximum(foot - signed_distance, signed_distance - crest)
    boundary = 1.0 - _smootherstep(outer_distance / float(config["outer_match_band_m"]))
    boundary[outer_distance <= 0.0] = 1.0
    confidence = _smootherstep(
        (confidence - float(config["minimum_solve_confidence"]))
        / max(1.0 - float(config["minimum_solve_confidence"]), 1e-9)
    )
    target = (
        float(site_config["structural_strength"])
        * (template - c0)
        * endpoint
        * boundary
        * confidence
    )
    outer_limit = np.maximum(np.abs(foot), np.abs(crest)) + float(config["outer_match_band_m"])
    return target, {
        "foot": foot,
        "crest": crest,
        "endpoint_distance": endpoint_distance,
        "endpoint_taper": endpoint,
        "boundary_taper": boundary,
        "confidence": confidence,
        "outer_limit": outer_limit,
    }


def _solve_residual(
    target: np.ndarray,
    active: np.ndarray,
    target_weight: np.ndarray,
    membrane_weight: float,
    tolerance: float,
    maximum_iterations: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    rows, cols = target.shape
    index = np.full((rows, cols), -1, dtype=np.int64)
    index[active] = np.arange(np.count_nonzero(active), dtype=np.int64)
    count = int(np.count_nonzero(active))
    if count == 0:
        raise ValueError("variational corridor contains no free cells")
    center = index[active]
    diagonal = target_weight[active].astype(np.float64) + 4.0 * membrane_weight
    matrix_rows = [center]
    matrix_cols = [center]
    matrix_data = [diagonal]
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted = np.full_like(index, -1)
        source_rows = slice(max(0, -dr), min(rows, rows - dr))
        source_cols = slice(max(0, -dc), min(cols, cols - dc))
        target_rows = slice(max(0, dr), min(rows, rows + dr))
        target_cols = slice(max(0, dc), min(cols, cols + dc))
        shifted[target_rows, target_cols] = index[source_rows, source_cols]
        neighbor = shifted[active]
        present = neighbor >= 0
        matrix_rows.append(center[present])
        matrix_cols.append(neighbor[present])
        matrix_data.append(np.full(np.count_nonzero(present), -membrane_weight, dtype=np.float64))
    matrix = sparse.csr_matrix(
        (np.concatenate(matrix_data), (np.concatenate(matrix_rows), np.concatenate(matrix_cols))),
        shape=(count, count),
    )
    rhs = target_weight[active] * target[active]
    solution, info = cg(matrix, rhs, rtol=tolerance, atol=0.0, maxiter=maximum_iterations)
    if info != 0:
        raise RuntimeError(f"variational solve did not converge: cg info={info}")
    residual = np.zeros_like(target, dtype=np.float64)
    residual[active] = solution
    equation = matrix @ solution - rhs
    return residual, {
        "free_cells": count,
        "matrix_nnz": int(matrix.nnz),
        "cg_info": int(info),
        "equation_max_abs": float(np.max(np.abs(equation))),
        "equation_rms": float(np.sqrt(np.mean(equation * equation))),
    }


def _hillshade(height: np.ndarray, pitch: float) -> np.ndarray:
    gy, gx = np.gradient(height, pitch)
    normal = np.stack([-gx, gy, np.ones_like(height)], axis=-1)
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-9)
    light = np.asarray([-0.55, 0.42, 0.72], dtype=np.float64)
    light /= np.linalg.norm(light)
    shade = np.clip(np.sum(normal * light, axis=-1), 0.0, 1.0)
    return np.clip(0.18 + 0.82 * shade, 0.0, 1.0)


def _relief_image(height: np.ndarray, pitch: float, lo: float, hi: float) -> np.ndarray:
    value = np.clip((height - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    color = np.stack([
        0.18 + 0.60 * value,
        0.32 + 0.52 * value,
        0.20 + 0.42 * (1.0 - value),
    ], axis=-1)
    color *= _hillshade(height, pitch)[..., None]
    return (np.clip(color, 0.0, 1.0) * 255.0).astype(np.uint8)


def _profile_plot(
    c0: np.ndarray,
    c1: np.ndarray,
    station_grid: np.ndarray,
    signed_distance: np.ndarray,
    evidence: dict[str, np.ndarray],
    width: int = 1200,
    height: int = 420,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 247, dtype=np.uint8)
    stations = evidence["stations"]
    selected = [0.25, 0.50, 0.75]
    colors = [
        (np.asarray([39, 86, 170], dtype=np.uint8), np.asarray([220, 72, 44], dtype=np.uint8)),
        (np.asarray([48, 128, 76], dtype=np.uint8), np.asarray([190, 58, 142], dtype=np.uint8)),
        (np.asarray([110, 92, 36], dtype=np.uint8), np.asarray([38, 142, 158], dtype=np.uint8)),
    ]
    profiles = []
    for fraction, pair in zip(selected, colors, strict=True):
        target_station = float(stations[-1]) * fraction
        band = np.abs(station_grid - target_station) <= 0.2
        columns = np.flatnonzero(band.ravel())
        if len(columns) < 8:
            continue
        distance = signed_distance.ravel()[columns]
        order = np.argsort(distance)
        profiles.append((distance[order], c0.ravel()[columns][order], c1.ravel()[columns][order], pair))
    if not profiles:
        return canvas
    all_distance = np.concatenate([row[0] for row in profiles])
    all_height = np.concatenate([np.concatenate([row[1], row[2]]) for row in profiles])
    xmin, xmax = np.percentile(all_distance, [1, 99])
    ymin, ymax = np.percentile(all_height, [1, 99])
    pad = max((ymax - ymin) * 0.08, 0.1)
    ymin -= pad
    ymax += pad
    for distance, before, after, pair in profiles:
        inside = (distance >= xmin) & (distance <= xmax)
        px = np.rint((distance[inside] - xmin) / max(xmax - xmin, 1e-9) * (width - 1)).astype(int)
        for values, color in ((before[inside], pair[0]), (after[inside], pair[1])):
            py = np.rint((1.0 - (values - ymin) / max(ymax - ymin, 1e-9)) * (height - 1)).astype(int)
            py = np.clip(py, 0, height - 1)
            for index in range(len(px) - 1):
                steps = max(abs(px[index + 1] - px[index]), abs(py[index + 1] - py[index]), 1)
                xs = np.linspace(px[index], px[index + 1], steps + 1).astype(int)
                ys = np.linspace(py[index], py[index + 1], steps + 1).astype(int)
                canvas[ys, xs] = color
    return canvas


def _save_qa(
    qa: Path,
    site_id: str,
    c0: np.ndarray,
    c1: np.ndarray,
    residual: np.ndarray,
    active: np.ndarray,
    masks: dict[str, np.ndarray],
    station_grid: np.ndarray,
    signed_distance: np.ndarray,
    evidence: dict[str, np.ndarray],
    pitch: float,
) -> list[dict[str, Any]]:
    qa.mkdir(parents=True, exist_ok=True)
    lo, hi = np.percentile(c0, [1.0, 99.0])
    before_after = np.concatenate([
        _relief_image(c0, pitch, float(lo), float(hi)),
        _relief_image(c1, pitch, float(lo), float(hi)),
    ], axis=1)
    scale = max(float(np.percentile(np.abs(residual[active]), 99.5)), 0.05)
    signed = np.clip(residual / scale * 0.5 + 0.5, 0.0, 1.0)
    residual_rgb = np.stack([signed, 1.0 - 2.0 * np.abs(signed - 0.5), 1.0 - signed], axis=-1)
    ownership = np.stack([active, masks["eligible"], masks["target_feature"]], axis=-1).astype(np.float64)
    structure = np.concatenate([(residual_rgb * 255).astype(np.uint8), (ownership * 255).astype(np.uint8)], axis=1)
    profiles = _profile_plot(c0, c1, station_grid, signed_distance, evidence)
    names = [
        (f"01-{site_id}-before-after-shaded.png", before_after, "measured C0 left; global 2D variational C1 right; common shaded-relief scale"),
        (f"02-{site_id}-residual-ownership.png", structure, "signed residual left; free/eligible/ETAK target ownership right"),
        (f"03-{site_id}-cross-profiles.png", profiles, "three along-line cross-sections; muted lines are C0 and vivid lines are C1"),
    ]
    rows = []
    for name, pixels, interpretation in names:
        path = qa / name
        Image.fromarray(pixels).save(path, optimize=True)
        rows.append({**_identity(path), "dimensions": [int(pixels.shape[1]), int(pixels.shape[0])], "interpretation": interpretation})
    return rows


def _boundary_jump(residual: np.ndarray, active: np.ndarray) -> float:
    boundary_values = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbor = np.roll(active, (dr, dc), axis=(0, 1))
        boundary_values.append(np.abs(residual[active & ~neighbor]))
    values = np.concatenate([value for value in boundary_values if value.size])
    return float(np.max(values)) if values.size else 0.0


def _run_site(
    config: dict[str, Any],
    bundle: dict[str, Any],
    closure: dict[str, Any],
    row: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    site_id = row["site_id"]
    site, source_path, source_bbox = _site_source(bundle, closure, site_id)
    with np.load(source_path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    source_height = arrays["height"].astype(np.float64)
    line, line_record = _load_line(_bound(site["etak_window"]), int(row["target_etak_id"]))
    stations, points, normals = _line_frame(line, float(config["evidence_station_step_m"]))
    normals = _orient_normals(points, normals, source_height, source_bbox)
    imagery = _imagery_support(points, normals, _bound(site["orthophoto"]["rgb"]), _bound(site["orthophoto"]["cir"]))
    evidence = _profile_evidence(
        height=source_height,
        bbox=source_bbox,
        stations=stations,
        points=points,
        normals=normals,
        imagery_support=imagery,
        config=config,
    )
    bbox = tuple(float(value) for value in row["output_bbox_en"])
    pitch = float(config["output_pitch_m"])
    x, y = _grid_coords(bbox, pitch)
    xx, yy = np.meshgrid(x, y)
    c0 = _sample_grid(source_height, source_bbox, xx.ravel(), yy.ravel(), 3).reshape(xx.shape)
    station_grid, signed_distance, _ = _query_line_frame(
        x, y, line, stations, normals
    )
    masks = _eligible_masks(arrays, source_bbox, xx, yy, signed_distance)
    target, target_state = _structural_target(
        c0=c0,
        station_grid=station_grid,
        signed_distance=signed_distance,
        line_length=float(line.length),
        evidence=evidence,
        site_config=row,
        config=config,
    )
    clearance = ndimage.distance_transform_edt(masks["eligible"]) * pitch
    clearance_taper = _smootherstep(
        (clearance - float(config["fixed_boundary_collar_m"]))
        / float(config["hard_mask_taper_m"])
    )
    target *= clearance_taper
    corridor = (
        (signed_distance > target_state["foot"] - float(config["outer_match_band_m"]) + float(config["fixed_boundary_collar_m"]))
        & (signed_distance < target_state["crest"] + float(config["outer_match_band_m"]) - float(config["fixed_boundary_collar_m"]))
    )
    active = (
        masks["eligible"]
        & corridor
        & (target_state["endpoint_distance"] > float(config["endpoint_zero_m"]))
        & (target_state["confidence"] > 0.0)
        & (clearance > float(config["fixed_boundary_collar_m"]))
    )
    # Keep a complete fixed-cell ring around every free component.  This makes
    # all transaction boundaries exact C0 rather than merely near-zero.
    active &= ndimage.binary_erosion(masks["eligible"], iterations=1, border_value=0)
    zero_anchor = 1.0 - (
        target_state["boundary_taper"]
        * target_state["endpoint_taper"]
        * clearance_taper
    )
    target_weight = (
        float(config["target_weight_min"])
        + float(config["target_weight_evidence"]) * target_state["confidence"]
        + float(config["boundary_anchor_weight"]) * zero_anchor
    )
    residual, solve = _solve_residual(
        target,
        active,
        target_weight,
        float(config["membrane_weight"]),
        float(config["solver_relative_tolerance"]),
        int(config["solver_maximum_iterations"]),
    )
    c1 = c0 + residual
    site_dir = output / "sites" / site_id
    site_dir.mkdir(parents=True)
    np.savez_compressed(
        site_dir / "surface.npz",
        c0=c0.astype(np.float32),
        c1=c1.astype(np.float32),
        residual=residual.astype(np.float32),
        active=active.astype(np.uint8),
        eligible=masks["eligible"].astype(np.uint8),
        target_feature=masks["target_feature"].astype(np.uint8),
    )
    qa = _save_qa(
        output / "qa", site_id, c0, c1, residual, active, masks,
        station_grid, signed_distance, evidence, pitch,
    )
    endpoint_zone = target_state["endpoint_distance"] <= float(config["endpoint_zero_m"])
    output_boundary = np.zeros_like(active)
    output_boundary[[0, -1], :] = True
    output_boundary[:, [0, -1]] = True
    changed = np.abs(residual) > 1e-9
    metrics = {
        "line_length_m": float(line.length),
        "station_count": int(len(stations)),
        "valid_station_fraction": float(np.mean(evidence["valid"])),
        "imagery_support_median": float(np.median(evidence["imagery_support"])),
        "relief_median_m": float(np.median(evidence["relief"][evidence["valid"]])) if np.any(evidence["valid"]) else 0.0,
        "active_cells": int(np.count_nonzero(active)),
        "active_area_m2": float(np.count_nonzero(active) * pitch * pitch),
        "changed_cells": int(np.count_nonzero(changed)),
        "residual_abs_p50_m": float(np.percentile(np.abs(residual[active]), 50)),
        "residual_abs_p99_m": float(np.percentile(np.abs(residual[active]), 99)),
        "residual_min_m": float(np.min(residual)),
        "residual_max_m": float(np.max(residual)),
        "forbidden_max_abs_residual_m": float(np.max(np.abs(residual[masks["forbidden"]]))),
        "water_max_abs_residual_m": float(np.max(np.abs(residual[masks["water"]]))) if np.any(masks["water"]) else 0.0,
        "endpoint_zero_max_abs_residual_m": float(np.max(np.abs(residual[endpoint_zone]))),
        "output_boundary_max_abs_residual_m": float(np.max(np.abs(residual[output_boundary]))),
        "active_boundary_jump_max_m": _boundary_jump(residual, active),
        "finite_c1": bool(np.all(np.isfinite(c1))),
        "solve": solve,
    }
    return {
        "site_id": site_id,
        "target_etak_id": int(row["target_etak_id"]),
        "target_record": line_record,
        "profile": row["profile"],
        "source_domain": _identity(source_path),
        "etak": site["etak_window"],
        "soil": site["soil_window"],
        "geology": site["geology_window"],
        "rgb": site["orthophoto"]["rgb"],
        "cir": site["orthophoto"]["cir"],
        "bbox_en": list(bbox),
        "pitch_m": pitch,
        "shape": list(c0.shape),
        "metrics": metrics,
        "surface": _identity(site_dir / "surface.npz"),
        "qa": qa,
    }


def run(config_path: Path) -> Path:
    config = _read_config(config_path)
    bundle_path = _bound(config["condition_bundle"])
    closure_path = _bound(config["development_a_closure"])
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    if "solver_authorized_condition_authority" not in _canonical_bytes(closure).decode("ascii"):
        raise ValueError("Development A closure is not authorized")
    implementation = [
        _identity(Path(__file__)),
        _identity(Path(__file__).with_name("screen.py")),
    ]
    recipe = {
        "config": _identity(config_path),
        "condition_bundle": config["condition_bundle"],
        "development_a_closure": config["development_a_closure"],
        "implementation": implementation,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    artifact_id = hashlib.sha256(_canonical_bytes(recipe)).hexdigest()
    final = _ARTIFACT_ROOT / artifact_id
    if final.exists():
        return final
    _ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{artifact_id}.", dir=_ARTIFACT_ROOT))
    try:
        sites = [_run_site(config, bundle, closure, row, staging) for row in config["sites"]]
        failures = []
        for site in sites:
            metrics = site["metrics"]
            if metrics["valid_station_fraction"] < 0.35:
                failures.append(f"{site['site_id']}: insufficient evidence-qualified line coverage")
            if metrics["active_area_m2"] < 100.0 or metrics["residual_abs_p99_m"] < 0.03:
                failures.append(f"{site['site_id']}: structural correction is not materially present")
            if not metrics["finite_c1"]:
                failures.append(f"{site['site_id']}: non-finite C1")
            if max(
                metrics["forbidden_max_abs_residual_m"],
                metrics["water_max_abs_residual_m"],
                metrics["endpoint_zero_max_abs_residual_m"],
                metrics["output_boundary_max_abs_residual_m"],
            ) > 1e-12:
                failures.append(f"{site['site_id']}: exact fixed-cell constraint differs")
            if max(abs(metrics["residual_min_m"]), abs(metrics["residual_max_m"])) > float(config["maximum_residual_abs_m"]):
                failures.append(f"{site['site_id']}: structural correction exceeds bounded envelope")
            if metrics["active_boundary_jump_max_m"] > float(config["maximum_active_boundary_jump_m"]):
                failures.append(f"{site['site_id']}: corridor boundary does not taper continuously")
        manifest = {
            "schema_version": _MANIFEST_SCHEMA,
            "artifact_id": artifact_id,
            "decision": "development_float_candidate_pass" if not failures else "development_float_candidate_rejected",
            "failures": failures,
            "scope": "single-valued mapped shoulder/scarp/bank-break macro reconstruction; global 2D residual surface",
            "not_claimed": ["fine rock morphology", "vertical face", "undercut", "production", "packing", "runtime"],
            "taevaskoda_opened": False,
            "recipe": recipe,
            "sites": sites,
        }
        (staging / "manifest.json").write_bytes(_canonical_bytes(manifest) + b"\n")
        images = [image for site in sites for image in site["qa"]]
        (staging / "qa" / "index.json").write_bytes(_canonical_bytes({
            "schema_version": "laas.qa-index/1",
            "artifact_id": artifact_id,
            "recipe_sha256": hashlib.sha256(_canonical_bytes(recipe)).hexdigest(),
            "images": images,
        }) + b"\n")
        os.replace(staging, final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
