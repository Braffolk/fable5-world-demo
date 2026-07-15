"""Bounded structural-only natural-escarpment research screen.

The screen reconstructs a mapped, single-valued bank transition in line-normal
coordinates. It does not simulate erosion or add unresolved morphology. Stable
shoulders are estimated from accepted C0 on either side of the mapped feature;
RGB/CIR edges qualify the structural evidence but never supply height.
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
import rasterio
from PIL import Image
from scipy import ndimage
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree
from shapely import wkb

from ....config import ASSET_GEN_ROOT, DATA_WORK

_CONFIG_SCHEMA = "laas.natural-escarpment-screen-config/1"
_MANIFEST_SCHEMA = "laas.natural-escarpment-screen/1"
_ARTIFACT_ROOT = DATA_WORK / "terrain" / "natural-escarpment-screen" / "sha256"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_path(path: Path) -> str:
    return str(path.resolve().relative_to(ASSET_GEN_ROOT.parent))


def _identity(path: Path) -> dict[str, Any]:
    return {"path": _repo_path(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _bound(identity: dict[str, Any]) -> Path:
    path = ASSET_GEN_ROOT.parent / identity["path"]
    if not path.is_file() or path.stat().st_size != identity["bytes"] or _sha256_file(path) != identity["sha256"]:
        raise ValueError(f"bound input differs: {path}")
    return path


def _read_config(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != _CONFIG_SCHEMA:
        raise ValueError("unsupported natural-escarpment config")
    if document.get("attempt") not in {1, 2} or document.get("taevaskoda_used") is not False:
        raise ValueError("Development config attempt/sentinel boundary differs")
    if [row.get("site_id") for row in document["sites"]] != ["development_a", "development_c"]:
        raise ValueError("Development site order differs")
    if [row.get("target_etak_id") for row in document["sites"]] != [1826743, 9688719]:
        raise ValueError("Development ETAK identities differ")
    return document


def _load_line(etak_path: Path, target_id: int):
    document = json.loads(etak_path.read_text(encoding="utf-8"))
    rows = [row for row in document["records"] if row["attributes"].get("etak_id") == target_id]
    if len(rows) != 1:
        raise ValueError(f"ETAK target {target_id} is not unique")
    geometry = wkb.loads(bytes.fromhex(rows[0]["geometry"]["wkb_hex"]))
    if geometry.geom_type != "LineString" or not geometry.is_valid or geometry.length < 32.0:
        raise ValueError("target is not a usable natural-escarpment line")
    attrs = rows[0]["attributes"]
    if attrs.get("kaldaastang") != 10 or attrs.get("tyyp") not in {10, 20}:
        raise ValueError("target is not typed as a bank/natural escarpment")
    return geometry, rows[0]


def _grid_coords(bbox: tuple[float, float, float, float], pitch: float) -> tuple[np.ndarray, np.ndarray]:
    width = int(round((bbox[2] - bbox[0]) / pitch))
    height = int(round((bbox[3] - bbox[1]) / pitch))
    x = bbox[0] + (np.arange(width, dtype=np.float64) + 0.5) * pitch
    y = bbox[3] - (np.arange(height, dtype=np.float64) + 0.5) * pitch
    return x, y


def _sample_grid(array: np.ndarray, bbox: tuple[float, float, float, float], x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    rows = bbox[3] - y - 0.5
    cols = x - bbox[0] - 0.5
    return ndimage.map_coordinates(array, [rows, cols], order=order, mode="nearest", prefilter=order > 1)


def _sample_raster(path: Path, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    with rasterio.open(path) as source:
        rows, cols = rasterio.transform.rowcol(source.transform, x, y, op=lambda v: v)
        coords = np.vstack([np.asarray(rows), np.asarray(cols)])
        channels = []
        for band in range(1, source.count + 1):
            data = source.read(band)
            channels.append(ndimage.map_coordinates(data, coords, order=1, mode="nearest").astype(np.float64) / 255.0)
    return np.stack(channels, axis=-1)


def _line_frame(line, step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stations = np.arange(0.0, line.length + step * 0.5, step, dtype=np.float64)
    stations[-1] = line.length
    points = np.asarray([[line.interpolate(float(s)).x, line.interpolate(float(s)).y] for s in stations])
    tangents = np.gradient(points, stations, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    return stations, points, normals


def _orient_normals(
    points: np.ndarray,
    normals: np.ndarray,
    height: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> np.ndarray:
    plus = _sample_grid(height, bbox, points[:, 0] + normals[:, 0] * 8.0, points[:, 1] + normals[:, 1] * 8.0, 1)
    minus = _sample_grid(height, bbox, points[:, 0] - normals[:, 0] * 8.0, points[:, 1] - normals[:, 1] * 8.0, 1)
    if float(np.nanmedian(plus - minus)) < 0.0:
        normals = -normals
    return normals


def _imagery_support(
    points: np.ndarray,
    normals: np.ndarray,
    rgb_path: Path,
    cir_path: Path,
) -> np.ndarray:
    offsets = np.arange(-6.0, 6.01, 0.5, dtype=np.float64)
    xx = points[:, 0, None] + normals[:, 0, None] * offsets[None, :]
    yy = points[:, 1, None] + normals[:, 1, None] * offsets[None, :]
    rgb = _sample_raster(rgb_path, xx.ravel(), yy.ravel()).reshape(*xx.shape, 3)
    cir = _sample_raster(cir_path, xx.ravel(), yy.ravel()).reshape(*xx.shape, 3)
    rgb_luma = rgb @ np.asarray([0.2126, 0.7152, 0.0722])
    # CIR is NIR/R/G in the bound Maa-amet product. Use an edge magnitude only;
    # it never becomes elevation or determines the high/low side.
    nir = cir[..., 0]
    edge = np.maximum(np.abs(np.gradient(rgb_luma, offsets, axis=1)), np.abs(np.gradient(nir, offsets, axis=1)))
    local = np.max(edge[:, np.abs(offsets) <= 2.0], axis=1)
    background = np.median(edge[:, np.abs(offsets) >= 3.0], axis=1) + 1e-4
    ratio = local / background
    return np.clip((ratio - 0.75) / 2.25, 0.0, 1.0)


def _profile_model(
    *,
    height: np.ndarray,
    bbox: tuple[float, float, float, float],
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
    smooth = ndimage.gaussian_filter1d(profiles, sigma=1.0 / step, axis=1, mode="nearest")
    gradient = np.gradient(smooth, step, axis=1)
    search = np.abs(offsets) <= float(config["break_search_radius_m"])
    peak_idx = np.argmax(gradient[:, search], axis=1) + np.flatnonzero(search)[0]
    peak = gradient[np.arange(len(points)), peak_idx]
    center = offsets[peak_idx]
    fraction = float(config["shoulder_gradient_fraction"])
    min_half = float(config["minimum_half_width_m"])
    max_half = float(config["maximum_half_width_m"])
    foot = np.empty(len(points), dtype=np.float64)
    crest = np.empty(len(points), dtype=np.float64)
    for index, pidx in enumerate(peak_idx):
        threshold = max(peak[index] * fraction, 0.02)
        left = pidx
        while left > 0 and center[index] - offsets[left] < max_half and gradient[index, left] > threshold:
            left -= 1
        right = pidx
        while right + 1 < len(offsets) and offsets[right] - center[index] < max_half and gradient[index, right] > threshold:
            right += 1
        foot[index] = min(offsets[left], center[index] - min_half)
        crest[index] = max(offsets[right], center[index] + min_half)
    station_sigma = max(1.0, float(config["station_smoothing_m"]) / 2.355)
    center = ndimage.gaussian_filter1d(center, station_sigma, mode="nearest")
    foot = ndimage.gaussian_filter1d(foot, station_sigma, mode="nearest")
    crest = ndimage.gaussian_filter1d(crest, station_sigma, mode="nearest")
    foot = np.minimum(foot, center - min_half)
    crest = np.maximum(crest, center + min_half)

    def sample_at(distance: np.ndarray) -> np.ndarray:
        indices = np.clip(np.rint((distance + radius) / step).astype(int), 0, len(offsets) - 1)
        return smooth[np.arange(len(points)), indices]

    foot_h = ndimage.gaussian_filter1d(sample_at(foot), station_sigma, mode="nearest")
    center_h = ndimage.gaussian_filter1d(sample_at(center), station_sigma, mode="nearest")
    crest_h = ndimage.gaussian_filter1d(sample_at(crest), station_sigma, mode="nearest")
    outer_low_h = ndimage.gaussian_filter1d(sample_at(foot - float(config["outer_match_band_m"])), station_sigma, mode="nearest")
    outer_high_h = ndimage.gaussian_filter1d(sample_at(crest + float(config["outer_match_band_m"])), station_sigma, mode="nearest")
    relief = crest_h - foot_h
    valid = (
        (relief >= float(config["minimum_relief_m"]))
        & (relief <= float(config["maximum_relief_m"]))
        & (peak > 0.05)
        & (imagery_support >= float(config["minimum_imagery_support"]))
    )
    return {
        "foot": foot,
        "center": center,
        "crest": crest,
        "foot_h": foot_h,
        "center_h": np.clip(center_h, foot_h, crest_h),
        "crest_h": crest_h,
        "outer_low_h": np.minimum(outer_low_h, foot_h),
        "outer_high_h": np.maximum(outer_high_h, crest_h),
        "peak_gradient": peak,
        "relief": relief,
        "imagery_support": imagery_support,
        "valid": valid,
    }


def _reconstruct(
    *,
    x: np.ndarray,
    y: np.ndarray,
    c0: np.ndarray,
    masks: dict[str, np.ndarray],
    frame_points: np.ndarray,
    frame_normals: np.ndarray,
    model: dict[str, np.ndarray],
    outer_match_band: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    xx, yy = np.meshgrid(x, y)
    query = np.column_stack([xx.ravel(), yy.ravel()])
    tree = cKDTree(frame_points)
    line_distance, nearest = tree.query(query, workers=-1)
    delta = query - frame_points[nearest]
    distance = np.sum(delta * frame_normals[nearest], axis=1)
    c1 = c0.ravel().copy()
    reconstructed = c0.ravel().copy()
    active = np.zeros(c1.shape, dtype=bool)
    for station in np.unique(nearest):
        take = np.flatnonzero(nearest == station)
        if not model["valid"][station]:
            continue
        foot = model["foot"][station]
        center = model["center"][station]
        crest = model["crest"][station]
        knots_x = np.asarray([foot - outer_match_band, foot, center, crest, crest + outer_match_band])
        left = model["outer_low_h"][station]
        right = model["outer_high_h"][station]
        knots_y = np.maximum.accumulate(np.asarray([left, model["foot_h"][station], model["center_h"][station], model["crest_h"][station], right]))
        interpolator = PchipInterpolator(knots_x, knots_y, extrapolate=False)
        # Signed distance alone describes an infinite tangent line. The true
        # Euclidean cap is required at the finite ETAK endpoints.
        inside = (
            (distance[take] >= knots_x[0])
            & (distance[take] <= knots_x[-1])
            & (line_distance[take] <= max(abs(knots_x[0]), abs(knots_x[-1])))
        )
        values = interpolator(distance[take][inside])
        reconstructed[take[inside]] = values
        active[take[inside]] = True
    target = masks["target_feature"].ravel()
    protected_allowed = target | (np.abs(distance) <= 1.5)
    eligible = (
        masks["valid"].ravel()
        & ~masks["water"].ravel()
        & ~masks["object"].ravel()
        & ~masks["non_heightfield"].ravel()
        & ~masks["unknown"].ravel()
        & masks["material_supported"].ravel()
        & (~masks["protected_structure"].ravel() | protected_allowed)
    )
    apply = active & eligible & np.isfinite(reconstructed)
    c1[apply] = reconstructed[apply]
    return c1.reshape(c0.shape), {
        "signed_distance": distance.reshape(c0.shape),
        "active": active.reshape(c0.shape),
        "eligible": eligible.reshape(c0.shape),
        "applied": apply.reshape(c0.shape),
    }


def _normalize(array: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    finite = array[np.isfinite(array)]
    lo = float(np.percentile(finite, 1.0)) if lo is None else lo
    hi = float(np.percentile(finite, 99.0)) if hi is None else hi
    return np.clip((array - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def _color_height(height: np.ndarray, lo: float, hi: float) -> np.ndarray:
    value = _normalize(height, lo, hi)
    r = np.clip(1.6 * value, 0.0, 1.0)
    g = np.clip(1.8 - 2.0 * np.abs(value - 0.55), 0.0, 1.0)
    b = np.clip(1.4 * (1.0 - value), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def _panel(*images: np.ndarray) -> np.ndarray:
    return np.concatenate(images, axis=1)


def _save_qa(
    qa: Path,
    site_id: str,
    c0: np.ndarray,
    c1: np.ndarray,
    residual: np.ndarray,
    state: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    frame_station: np.ndarray,
    model: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    qa.mkdir(parents=True, exist_ok=True)
    lo = float(np.percentile(c0, 1.0)); hi = float(np.percentile(c0, 99.0))
    before_after = _panel(_color_height(c0, lo, hi), _color_height(c1, lo, hi))
    scale = max(float(np.percentile(np.abs(residual), 99.5)), 0.05)
    signed = np.clip(residual / scale * 0.5 + 0.5, 0.0, 1.0)
    residual_rgb = np.stack([signed, 1.0 - np.abs(signed - 0.5) * 2.0, 1.0 - signed], axis=-1)
    applied_rgb = np.stack([state["applied"], state["eligible"], masks["target_feature"]], axis=-1).astype(np.float64)
    structure = _panel((residual_rgb * 255).astype(np.uint8), (applied_rgb * 255).astype(np.uint8))

    width = 1024
    height = 360
    profile = np.full((height, width, 3), 248, dtype=np.uint8)
    s = np.linspace(0, frame_station[-1], len(model["relief"]))
    series = [
        (model["relief"], np.asarray([32, 92, 180], dtype=np.uint8)),
        (model["crest"] - model["foot"], np.asarray([209, 88, 31], dtype=np.uint8)),
        (model["imagery_support"] * max(float(np.max(model["relief"])), 1.0), np.asarray([38, 142, 72], dtype=np.uint8)),
    ]
    ymax = max(float(np.max(values)) for values, _ in series) * 1.08
    for values, color in series:
        px = np.clip(np.rint(s / max(s[-1], 1e-9) * (width - 1)).astype(int), 0, width - 1)
        py = np.clip(np.rint((1.0 - values / max(ymax, 1e-9)) * (height - 1)).astype(int), 0, height - 1)
        for i in range(len(px) - 1):
            steps = max(abs(px[i + 1] - px[i]), abs(py[i + 1] - py[i]), 1)
            xs = np.linspace(px[i], px[i + 1], steps + 1).astype(int)
            ys = np.linspace(py[i], py[i + 1], steps + 1).astype(int)
            profile[ys, xs] = color

    names = [
        (f"01-{site_id}-before-after.png", before_after, "C0 left; structural C1 right; identical height color scale"),
        (f"02-{site_id}-residual-eligibility.png", structure, "signed residual left; applied/eligible/target ownership right"),
        (f"03-{site_id}-station-profile.png", profile, "station diagnostics: relief blue, transition width orange, imagery support green"),
    ]
    rows = []
    for name, pixels, interpretation in names:
        path = qa / name
        Image.fromarray(pixels).save(path, optimize=True)
        rows.append({**_identity(path), "dimensions": [int(pixels.shape[1]), int(pixels.shape[0])], "interpretation": interpretation})
    return rows


def _site_source(bundle: dict[str, Any], closure: dict[str, Any], site_id: str) -> tuple[dict[str, Any], Path, tuple[float, float, float, float]]:
    site = next(row for row in bundle["sites"] if row["site_id"] == site_id)
    if site_id == "development_a":
        source_identity = closure["outputs"]["canonical_domain"]
        bbox = tuple(float(v) for v in closure["recipe"]["canonical_bbox_en"])
    else:
        source_identity = site["arrays"]
        bbox = tuple(float(v) for v in site["bbox_en"])
    return site, _bound(source_identity), bbox


def _run_site(config: dict[str, Any], bundle: dict[str, Any], closure: dict[str, Any], row: dict[str, Any], output: Path) -> dict[str, Any]:
    site_id = row["site_id"]
    site, source_path, source_bbox = _site_source(bundle, closure, site_id)
    with np.load(source_path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    line, line_record = _load_line(_bound(site["etak_window"]), int(row["target_etak_id"]))
    rgb_path = _bound(site["orthophoto"]["rgb"])
    cir_path = _bound(site["orthophoto"]["cir"])
    stations, points, normals = _line_frame(line, float(config["station_step_m"]))
    normals = _orient_normals(points, normals, arrays["height"].astype(np.float64), source_bbox)
    image_support = _imagery_support(points, normals, rgb_path, cir_path)
    model = _profile_model(
        height=arrays["height"].astype(np.float64), bbox=source_bbox, points=points,
        normals=normals, imagery_support=image_support, config=config,
    )
    bbox = tuple(float(v) for v in row["output_bbox_en"])
    pitch = float(config["output_pitch_m"])
    x, y = _grid_coords(bbox, pitch)
    xx, yy = np.meshgrid(x, y)
    c0 = _sample_grid(arrays["height"].astype(np.float64), source_bbox, xx.ravel(), yy.ravel(), 3).reshape(xx.shape)
    nearest = lambda name: _sample_grid(arrays[name].astype(np.float64), source_bbox, xx.ravel(), yy.ravel(), 0).reshape(xx.shape) >= 0.5
    material = (arrays["soil_feature_index"] >= 0) & (arrays["geology_lithology_code"] > 0) & (arrays["geology_genesis_code"] > 0)
    masks = {
        name: nearest(name) for name in (
            "valid", "water", "object", "non_heightfield", "protected_structure", "unknown", "target_feature"
        )
    }
    masks["material_supported"] = _sample_grid(material.astype(np.float64), source_bbox, xx.ravel(), yy.ravel(), 0).reshape(xx.shape) >= 0.5
    c1, state = _reconstruct(
        x=x, y=y, c0=c0, masks=masks, frame_points=points,
        frame_normals=normals, model=model, outer_match_band=float(config["outer_match_band_m"]),
    )
    residual = c1 - c0
    site_dir = output / "sites" / site_id
    site_dir.mkdir(parents=True)
    np.savez_compressed(
        site_dir / "surface.npz", c0=c0.astype(np.float32), c1=c1.astype(np.float32),
        residual=residual.astype(np.float32), applied=state["applied"].astype(np.uint8),
        eligible=state["eligible"].astype(np.uint8), target_feature=masks["target_feature"].astype(np.uint8),
    )
    qa = _save_qa(output / "qa", site_id, c0, c1, residual, state, masks, stations, model)
    applied = state["applied"]
    forbidden = masks["water"] | masks["object"] | masks["non_heightfield"] | (masks["protected_structure"] & ~masks["target_feature"] & (np.abs(state["signed_distance"]) > 1.5))
    metrics = {
        "line_length_m": float(line.length),
        "line_bounds_en": [float(v) for v in line.bounds],
        "station_count": int(len(stations)),
        "valid_station_fraction": float(np.mean(model["valid"])),
        "imagery_support_median": float(np.median(model["imagery_support"])),
        "relief_median_m": float(np.median(model["relief"][model["valid"]])) if np.any(model["valid"]) else 0.0,
        "transition_width_median_m": float(np.median((model["crest"] - model["foot"])[model["valid"]])) if np.any(model["valid"]) else 0.0,
        "applied_cells": int(np.count_nonzero(applied)),
        "applied_area_m2": float(np.count_nonzero(applied) * pitch * pitch),
        "residual_abs_p50_m": float(np.percentile(np.abs(residual[applied]), 50)) if np.any(applied) else 0.0,
        "residual_abs_p99_m": float(np.percentile(np.abs(residual[applied]), 99)) if np.any(applied) else 0.0,
        "residual_min_m": float(np.min(residual)),
        "residual_max_m": float(np.max(residual)),
        "forbidden_max_abs_residual_m": float(np.max(np.abs(residual[forbidden]))) if np.any(forbidden) else 0.0,
        "water_max_abs_residual_m": float(np.max(np.abs(residual[masks["water"]]))) if np.any(masks["water"]) else 0.0,
        "target_applied_fraction": float(np.mean(applied[masks["target_feature"]])) if np.any(masks["target_feature"]) else 0.0,
        "adjacent_structural_coverage": float(np.mean(applied[(np.abs(state["signed_distance"]) <= 4.0) & state["eligible"]])) if np.any((np.abs(state["signed_distance"]) <= 4.0) & state["eligible"]) else 0.0,
    }
    return {
        "site_id": site_id,
        "target_etak_id": int(row["target_etak_id"]),
        "target_record": line_record,
        "source_domain": _identity(source_path),
        "etak": site["etak_window"],
        "soil": site["soil_window"],
        "geology": site["geology_window"],
        "rgb": site["orthophoto"]["rgb"],
        "cir": site["orthophoto"]["cir"],
        "bbox_en": list(bbox), "pitch_m": pitch, "shape": list(c0.shape),
        "metrics": metrics, "surface": _identity(site_dir / "surface.npz"), "qa": qa,
    }


def run(config_path: Path) -> Path:
    config = _read_config(config_path)
    bundle_path = _bound(config["condition_bundle"])
    closure_path = _bound(config["development_a_closure"])
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    if closure.get("decision") != "solver_authorized_condition_authority" and closure.get("status") != "solver_authorized_condition_authority":
        # Current immutable closure records the authority in its gate/status map.
        if "solver_authorized_condition_authority" not in _canonical_bytes(closure).decode("ascii"):
            raise ValueError("Development A closure is not authorized")
    implementation = [_identity(Path(__file__)), _identity(Path(__file__).with_name("__main__.py"))]
    recipe = {
        "config": _identity(config_path), "condition_bundle": config["condition_bundle"],
        "development_a_closure": config["development_a_closure"], "implementation": implementation,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
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
            if metrics["applied_area_m2"] < 100.0 or metrics["adjacent_structural_coverage"] < 0.35:
                failures.append(f"{site['site_id']}: structural reconstruction is not materially present")
            if metrics["forbidden_max_abs_residual_m"] > 1e-9 or metrics["water_max_abs_residual_m"] > 1e-9:
                failures.append(f"{site['site_id']}: hard-mask leakage")
            if metrics["residual_abs_p99_m"] < 0.03:
                failures.append(f"{site['site_id']}: candidate only resamples C0")
            if max(abs(metrics["residual_min_m"]), abs(metrics["residual_max_m"])) > float(config.get("maximum_residual_abs_m", 3.0)):
                failures.append(f"{site['site_id']}: structural correction envelope is catastrophic")
        manifest = {
            "schema_version": _MANIFEST_SCHEMA,
            "artifact_id": artifact_id,
            "decision": "development_float_candidate_pass" if not failures else "development_float_candidate_rejected",
            "failures": failures,
            "scope": "single-valued mapped natural escarpment/bank break; structural macro only",
            "not_claimed": ["fine rock morphology", "vertical face", "undercut", "production", "packing", "runtime"],
            "taevaskoda_opened": False,
            "recipe": recipe,
            "sites": sites,
        }
        (staging / "manifest.json").write_bytes(_canonical_bytes(manifest) + b"\n")
        qa_rows = [item for site in sites for item in site["qa"]]
        (staging / "qa" / "index.json").write_bytes(_canonical_bytes({"schema_version": "laas.qa-index/1", "images": qa_rows}) + b"\n")
        os.replace(staging, final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.config))
