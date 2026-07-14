"""Conservative R0 negative-form diagnostics for exact OPARA dense epoch 438.

This module measures connected depressions which survive several defensible
reconstruction masks and baseline choices.  It does not identify erosion truth,
authorize source-height transfer, or expose an input to terrain synthesis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.spatial import cKDTree

from ...config import DATA_IN, DATA_WORK


_RETENTION_ID = "db8163ca878efc5ff98c0f68474724678f524c7b18ac03422d93005e7be0502e"
_ARCHIVE_SHA256 = "02e6897902ae9d94028ef62c56df482167a18d7d5fd5ae149a1a031ea6e32bd7"
_ROOT = "B_sample_plot/III_plot_1_processed/sfm_timelapse/2021-07-21/timelapse"
_DENSE_MEMBER = f"{_ROOT}/dense/dense_438.ply"
_PRECISION_MEMBER = f"{_ROOT}/ptPrecision/pt_prec_index_438.txt"
_M3C2_MEMBER = f"{_ROOT}/m3c2/m3c2_0-to-438.txt"
_DENSE_SHA256 = "cab24fdecb50fb74a477483fd9e7772c50c9f5372321633540042fa906ca6224"
_PRECISION_SHA256 = "6c5f5ebebe2545a1a7e068ad4adf800779b15b2dc989c1bc61c798c1c533e20e"
_SCHEMA = "opara-1038-negative-form-diagnostic/1.0.0"
_ALGORITHM = "direct-support-consensus-negative-forms/1.1.0"

_PLY_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("nx", "<f4"),
        ("ny", "<f4"),
        ("nz", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("class", "u1"),
        ("confidence", "u1"),
    ]
)


@dataclass(frozen=True)
class Variant:
    name: str
    minimum_confidence: int
    maximum_precision_z_m: float
    maximum_cell_spread_m: float
    dark_quantile: float
    baseline_radius_m: float
    minimum_depth_m: float


_VARIANTS = (
    Variant("permissive", 3, 0.0050, 0.012, 0.01, 0.10, 0.003),
    Variant("nominal", 4, 0.0040, 0.008, 0.02, 0.15, 0.005),
    Variant("strict", 5, 0.0030, 0.006, 0.05, 0.20, 0.008),
)

_CELL_M = 0.005
_BOUNDARY_EROSION_M = 0.5
_FOOTPRINT_QUANTILE = 0.001
_MINIMUM_DIRECT_POINTS = 4
_MINIMUM_COMPONENT_AREA_M2 = 0.0004
_MINIMUM_COMPONENT_LENGTH_M = 0.08
_MINIMUM_COMPONENT_ASPECT = 4.0
_MAXIMUM_SLOPE_AXIS_DIFFERENCE_DEG = 30.0
_PRECISION_MAX_DISTANCE_M = 0.05


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _archive() -> Path:
    root = DATA_IN / "evidence" / "opara_1038" / _RETENTION_ID
    manifest = json.loads((root / "retained.json").read_bytes())
    path = root / str(manifest["archive"]["path"])
    if (
        manifest.get("status") != "complete"
        or manifest.get("archive", {}).get("sha256") != _ARCHIVE_SHA256
        or not path.is_file()
    ):
        raise ValueError("retained OPARA archive identity changed")
    return path


def _extract_exact(archive: Path, root: Path) -> tuple[Path, Path]:
    bsdtar = shutil.which("bsdtar")
    if bsdtar is None:
        raise RuntimeError("OPARA diagnostic requires bsdtar")
    subprocess.run(
        [bsdtar, "-xf", archive, "-C", root, _DENSE_MEMBER, _PRECISION_MEMBER],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    dense = root / _DENSE_MEMBER
    precision = root / _PRECISION_MEMBER
    if _sha256_file(dense) != _DENSE_SHA256:
        raise ValueError("dense_438 payload identity changed")
    if _sha256_file(precision) != _PRECISION_SHA256:
        raise ValueError("pt_prec_index_438 payload identity changed")
    return dense, precision


def _read_ply(path: Path) -> np.memmap:
    with path.open("rb") as source:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = source.readline()
            if not line or len(header) > 4096:
                raise ValueError("dense_438 has an invalid PLY header")
            header += line
    expected = (
        "ply\nformat binary_little_endian 1.0\n"
        "element vertex 2260817     \n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uchar class\nproperty uchar confidence\nend_header\n"
    ).encode()
    if header != expected:
        raise ValueError("dense_438 PLY schema changed")
    count = 2_260_817
    if path.stat().st_size != len(header) + count * _PLY_DTYPE.itemsize:
        raise ValueError("dense_438 PLY byte count changed")
    return np.memmap(path, dtype=_PLY_DTYPE, mode="r", offset=len(header), shape=(count,))


def _read_precision(path: Path) -> np.ndarray:
    rows = np.genfromtxt(path, delimiter="\t", names=True, encoding="utf-8-sig")
    expected = (
        "Xm",
        "Ym",
        "Zm",
        "sXmm",
        "sYmm",
        "sZmm",
        "covXXm2",
        "covXYm2",
        "covXZm2",
        "covYYm2",
        "covYZm2",
        "covZZm2",
    )
    if rows.dtype.names != expected or len(rows) != 10_052:
        raise ValueError("pt_prec_index_438 schema changed")
    return rows


def _deduplicate(records: np.ndarray) -> tuple[np.ndarray, int]:
    xyz = np.column_stack((records["x"], records["y"], records["z"]))
    _, first = np.unique(xyz, axis=0, return_index=True)
    first.sort()
    return records[first], len(records) - len(first)


def _aggregate(records: np.ndarray, precision: np.ndarray) -> dict[str, Any]:
    bounds = {}
    for axis in ("x", "y"):
        bounds[axis] = np.quantile(
            records[axis], [_FOOTPRINT_QUANTILE, 1.0 - _FOOTPRINT_QUANTILE]
        )
    x0, x1 = bounds["x"] + np.array([_BOUNDARY_EROSION_M, -_BOUNDARY_EROSION_M])
    y0, y1 = bounds["y"] + np.array([_BOUNDARY_EROSION_M, -_BOUNDARY_EROSION_M])
    if x1 <= x0 or y1 <= y0:
        raise ValueError("0.5 m erosion removes the OPARA footprint")
    width = int(np.floor((x1 - x0) / _CELL_M))
    height = int(np.floor((y1 - y0) / _CELL_M))
    x1 = x0 + width * _CELL_M
    y1 = y0 + height * _CELL_M

    inside = (
        (records["x"] >= x0)
        & (records["x"] < x1)
        & (records["y"] >= y0)
        & (records["y"] < y1)
    )
    points = records[inside]
    ix = np.floor((points["x"] - x0) / _CELL_M).astype(np.int32)
    iy = np.floor((points["y"] - y0) / _CELL_M).astype(np.int32)
    flat = iy * width + ix
    cells = height * width
    count = np.bincount(flat, minlength=cells).astype(np.uint16)

    def mean(name: str) -> np.ndarray:
        total = np.bincount(flat, weights=points[name], minlength=cells)
        return np.divide(total, count, out=np.full(cells, np.nan), where=count > 0)

    z_min = np.full(cells, np.inf)
    z_max = np.full(cells, -np.inf)
    np.minimum.at(z_min, flat, points["z"])
    np.maximum.at(z_max, flat, points["z"])
    direct = count >= _MINIMUM_DIRECT_POINTS
    spread = z_max - z_min
    spread[~direct] = np.nan

    centers_x = x0 + (np.arange(width) + 0.5) * _CELL_M
    centers_y = y0 + (np.arange(height) + 0.5) * _CELL_M
    xx, yy = np.meshgrid(centers_x, centers_y)
    precision_tree = cKDTree(np.column_stack((precision["Xm"], precision["Ym"])))
    precision_distance, precision_index = precision_tree.query(
        np.column_stack((xx.ravel(), yy.ravel())), workers=-1
    )
    precision_z = precision["sZmm"][precision_index] / 1000.0
    precision_z[precision_distance > _PRECISION_MAX_DISTANCE_M] = np.nan

    return {
        "shape": (height, width),
        "bounds": (float(x0), float(y0), float(x1), float(y1)),
        "source_bounds": {
            "x": [float(bounds["x"][0]), float(bounds["x"][1])],
            "y": [float(bounds["y"][0]), float(bounds["y"][1])],
        },
        "input_point_count": int(len(points)),
        "count": count.reshape(height, width),
        "direct": direct.reshape(height, width),
        "z": mean("z").reshape(height, width),
        "spread": spread.reshape(height, width),
        "nz": mean("nz").reshape(height, width),
        "confidence": mean("confidence").reshape(height, width),
        "rgb": np.stack(
            (mean("red"), mean("green"), mean("blue")), axis=-1
        ).reshape(height, width, 3),
        "precision_z": precision_z.reshape(height, width),
        "precision_distance": precision_distance.reshape(height, width),
    }


def _nearest_fill(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if not np.any(valid):
        raise ValueError("no valid OPARA cells survive")
    nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return values[tuple(nearest)]


def _robust_base(z: np.ndarray, valid: np.ndarray, radius_m: float) -> np.ndarray:
    filled = _nearest_fill(z, valid)
    radius_cells = max(2, int(round(radius_m / _CELL_M)))
    # A 65th-percentile envelope bridges narrow negative incisions without using maxima.
    size = 2 * radius_cells + 1
    envelope = ndimage.percentile_filter(filled, percentile=65, size=size, mode="nearest")
    return ndimage.gaussian_filter(envelope, sigma=radius_cells / 3.0, mode="nearest")


def _slope_axis(grid: dict[str, Any]) -> tuple[float, float]:
    valid = grid["direct"] & np.isfinite(grid["z"])
    yy, xx = np.nonzero(valid)
    x0, y0, _, _ = grid["bounds"]
    x = x0 + (xx + 0.5) * _CELL_M
    y = y0 + (yy + 0.5) * _CELL_M
    design = np.column_stack((x, y, np.ones(len(x))))
    z = grid["z"][valid]
    weights = np.ones(len(z))
    coefficients = np.zeros(3)
    for _ in range(6):
        coefficients = np.linalg.lstsq(
            design * weights[:, None], z * weights, rcond=None
        )[0]
        residual = z - design @ coefficients
        scale = max(1e-6, 1.4826 * float(np.median(np.abs(residual - np.median(residual)))))
        weights = np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(residual), 1e-12))
    gradient = coefficients[:2]
    slope_percent = float(np.hypot(*gradient) * 100.0)
    axis_degrees = float(np.degrees(np.arctan2(gradient[1], gradient[0])) % 180.0)
    return axis_degrees, slope_percent


def _axis_difference_degrees(a: float, b: float) -> float:
    difference = abs(a - b) % 180.0
    return min(difference, 180.0 - difference)


def _filter_rill_like(
    binary: np.ndarray,
    bounds: tuple[float, float, float, float],
    slope_axis_degrees: float,
) -> np.ndarray:
    labels, component_count = ndimage.label(
        binary, structure=np.ones((3, 3), dtype=np.uint8)
    )
    keep = np.zeros(component_count + 1, dtype=bool)
    x0, y0, _, _ = bounds
    for label_id in range(1, component_count + 1):
        yy, xx = np.nonzero(labels == label_id)
        area = len(xx) * _CELL_M**2
        if area < _MINIMUM_COMPONENT_AREA_M2:
            continue
        coordinates = np.column_stack(
            (x0 + (xx + 0.5) * _CELL_M, y0 + (yy + 0.5) * _CELL_M)
        )
        centered = coordinates - coordinates.mean(axis=0)
        covariance = centered.T @ centered / max(1, len(centered) - 1)
        _, eigenvectors = np.linalg.eigh(covariance)
        direction = eigenvectors[:, 1]
        along = centered @ direction
        length = float(np.ptp(along) + _CELL_M)
        width = float(area / max(length, _CELL_M))
        orientation = float(np.degrees(np.arctan2(direction[1], direction[0])) % 180.0)
        keep[label_id] = (
            length >= _MINIMUM_COMPONENT_LENGTH_M
            and length / max(width, _CELL_M) >= _MINIMUM_COMPONENT_ASPECT
            and _axis_difference_degrees(orientation, slope_axis_degrees)
            <= _MAXIMUM_SLOPE_AXIS_DIFFERENCE_DEG
        )
    return keep[labels]


def _initial_masks(grid: dict[str, Any], variant: Variant) -> dict[str, np.ndarray]:
    direct = grid["direct"]
    rgb = grid["rgb"]
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    threshold = float(np.nanquantile(luminance[direct], variant.dark_quantile))
    dark = direct & (luminance <= threshold)
    confidence = direct & (
        (grid["confidence"] < variant.minimum_confidence)
        | ~np.isfinite(grid["precision_z"])
        | (grid["precision_z"] > variant.maximum_precision_z_m)
    )
    ambiguity = direct & (
        (grid["spread"] > variant.maximum_cell_spread_m)
        | (grid["nz"] < 0.55)
    )
    unsupported = ~direct
    preliminary = ~(unsupported | dark | confidence | ambiguity)
    preliminary_base = _robust_base(grid["z"], preliminary, variant.baseline_radius_m)
    positive = preliminary & (
        (grid["z"] - preliminary_base)
        > max(0.008, 2.5 * float(np.nanmedian(grid["spread"][preliminary])))
    )
    valid = preliminary & ~positive
    return {
        "unsupported": unsupported,
        "dark": dark,
        "confidence": confidence,
        "ambiguity": ambiguity,
        "positive": positive,
        "valid": valid,
    }


def _variant_result(
    grid: dict[str, Any], variant: Variant, slope_axis_degrees: float
) -> dict[str, Any]:
    masks = _initial_masks(grid, variant)
    base = _robust_base(grid["z"], masks["valid"], variant.baseline_radius_m)
    depth = base - grid["z"]
    negative = masks["valid"] & (depth >= variant.minimum_depth_m)
    negative = _filter_rill_like(negative, grid["bounds"], slope_axis_degrees)
    return {"masks": masks, "base": base, "depth": depth, "negative": negative}


def _thin(binary: np.ndarray) -> np.ndarray:
    skeleton = binary.astype(bool).copy()
    while True:
        changed = False
        for first in (True, False):
            p = np.pad(skeleton, 1)
            n = [
                p[:-2, 1:-1],
                p[:-2, 2:],
                p[1:-1, 2:],
                p[2:, 2:],
                p[2:, 1:-1],
                p[2:, :-2],
                p[1:-1, :-2],
                p[:-2, :-2],
            ]
            neighbors = sum(n)
            transitions = sum((~n[index]) & n[(index + 1) % 8] for index in range(8))
            if first:
                product_a = n[0] & n[2] & n[4]
                product_b = n[2] & n[4] & n[6]
            else:
                product_a = n[0] & n[2] & n[6]
                product_b = n[0] & n[4] & n[6]
            remove = skeleton & (neighbors >= 2) & (neighbors <= 6)
            remove &= transitions == 1
            remove &= ~product_a & ~product_b
            if np.any(remove):
                skeleton[remove] = False
                changed = True
        if not changed:
            return skeleton


def _form_metrics(
    core: np.ndarray, depth: np.ndarray, bounds: tuple[float, float, float, float]
) -> list[dict[str, Any]]:
    labels, form_count = ndimage.label(core, structure=np.ones((3, 3), dtype=np.uint8))
    forms = []
    x0, y0, _, _ = bounds
    for label_id in range(1, form_count + 1):
        yy, xx = np.nonzero(labels == label_id)
        if not len(xx):
            continue
        coordinates = np.column_stack(
            (x0 + (xx + 0.5) * _CELL_M, y0 + (yy + 0.5) * _CELL_M)
        )
        centered = coordinates - coordinates.mean(axis=0)
        covariance = centered.T @ centered / max(1, len(centered) - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        along = centered @ direction
        across = centered @ np.array([-direction[1], direction[0]])
        length = float(np.ptp(along) + _CELL_M)
        width = float(len(xx) * _CELL_M**2 / max(length, _CELL_M))
        orientation = float(np.degrees(np.arctan2(direction[1], direction[0])) % 180.0)
        component = labels == label_id
        skeleton = _thin(component)
        neighbors = ndimage.convolve(
            skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant"
        ) - skeleton
        branch_clusters, branch_count = ndimage.label(skeleton & (neighbors >= 3))
        del branch_clusters
        values = depth[component]
        forms.append(
            {
                "id": label_id,
                "cell_count": int(len(xx)),
                "area_m2": float(len(xx) * _CELL_M**2),
                "centroid_xy_m": [float(value) for value in coordinates.mean(axis=0)],
                "length_m": length,
                "equivalent_width_m": width,
                "orientation_deg_ccw_from_positive_x": orientation,
                "depth_m": {
                    "median": float(np.median(values)),
                    "p95": float(np.percentile(values, 95)),
                    "maximum": float(np.max(values)),
                },
                "branch_point_clusters": int(branch_count),
            }
        )
    forms.sort(key=lambda item: (-item["length_m"], item["id"]))
    for index, form in enumerate(forms, start=1):
        form["id"] = index
    if len(forms) > 1:
        centroids = np.array([form["centroid_xy_m"] for form in forms])
        distances, _ = cKDTree(centroids).query(centroids, k=2)
        for form, spacing in zip(forms, distances[:, 1], strict=True):
            form["nearest_centroid_spacing_m"] = float(spacing)
    else:
        for form in forms:
            form["nearest_centroid_spacing_m"] = None
    return forms


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "minimum": None, "median": None, "p95": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def _scale_rgb(rgb: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(rgb, nan=0.0), 0, 255).astype(np.uint8)


def _label(image: Image.Image, title: str, lines: list[str]) -> Image.Image:
    pad = 12
    font = ImageFont.load_default()
    line_height = 14
    header = 26 + len(lines) * line_height
    canvas = Image.new("RGB", (image.width, image.height + header), "white")
    canvas.paste(image, (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 7), title, fill="black", font=font)
    for index, line in enumerate(lines):
        draw.text((pad, 23 + index * line_height), line, fill="#333333", font=font)
    return canvas


def _render_pngs(
    root: Path,
    grid: dict[str, Any],
    variants: dict[str, dict[str, Any]],
    core: np.ndarray,
    forms: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    scale = 3
    rgb = _scale_rgb(grid["rgb"])
    rgb[~grid["direct"]] = (255, 0, 255)
    source = Image.fromarray(np.flipud(rgb), "RGB").resize(
        (rgb.shape[1] * scale, rgb.shape[0] * scale), Image.Resampling.NEAREST
    )
    source = _label(
        source,
        "01 SOURCE: exact dense_438 direct-support RGB",
        ["magenta = <4 source points/cell; footprint already eroded 0.5 m"],
    )

    masks = variants["nominal"]["masks"]
    mask_rgb = np.zeros((*core.shape, 3), dtype=np.uint8)
    mask_rgb[masks["valid"]] = (230, 230, 230)
    mask_rgb[masks["unsupported"]] = (20, 20, 20)
    mask_rgb[masks["confidence"]] = (255, 174, 0)
    mask_rgb[masks["dark"]] = (30, 80, 220)
    mask_rgb[masks["ambiguity"]] = (220, 40, 40)
    mask_rgb[masks["positive"]] = (190, 50, 190)
    mask_image = Image.fromarray(np.flipud(mask_rgb), "RGB").resize(
        (core.shape[1] * scale, core.shape[0] * scale), Image.Resampling.NEAREST
    )
    mask_image = _label(
        mask_image,
        "02 MASK: nominal conservative exclusions",
        ["gray valid; black unsupported; orange confidence; blue dark/wet; red multisheet/edge; purple positive"],
    )

    nominal_depth = variants["nominal"]["depth"]
    valid_depth = np.where(masks["valid"], nominal_depth, np.nan)
    maximum = max(0.001, float(np.nanpercentile(valid_depth, 99)))
    normalized = np.clip(valid_depth / maximum, 0, 1)
    depth_rgb = np.zeros((*core.shape, 3), dtype=np.uint8)
    depth_rgb[..., 0] = np.nan_to_num(normalized * 255).astype(np.uint8)
    depth_rgb[..., 1] = np.nan_to_num(np.sqrt(normalized) * 180).astype(np.uint8)
    depth_rgb[..., 2] = np.nan_to_num((1.0 - normalized) * 70).astype(np.uint8)
    depth_rgb[~masks["valid"]] = (18, 18, 18)
    depth_rgb[core] = (255, 255, 255)
    depth_image = Image.fromarray(np.flipud(depth_rgb), "RGB").resize(
        (core.shape[1] * scale, core.shape[0] * scale), Image.Resampling.NEAREST
    )
    depth_image = _label(
        depth_image,
        "03 DEPTH: nominal robust-base depression and all-variant core",
        [f"color scale 0..{maximum * 1000:.1f} mm; white = survives permissive, nominal, and strict variants"],
    )

    geometry = Image.fromarray(np.flipud(rgb // 2), "RGB")
    draw = ImageDraw.Draw(geometry)
    x0, y0, _, _ = grid["bounds"]
    for form in forms:
        cx, cy = form["centroid_xy_m"]
        px = (cx - x0) / _CELL_M
        py = core.shape[0] - 1 - (cy - y0) / _CELL_M
        angle = np.radians(form["orientation_deg_ccw_from_positive_x"])
        half = form["length_m"] / (2 * _CELL_M)
        dx, dy = np.cos(angle) * half, -np.sin(angle) * half
        draw.line((px - dx, py - dy, px + dx, py + dy), fill=(255, 255, 0), width=1)
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(255, 0, 0))
        draw.text((px + 3, py + 1), str(form["id"]), fill="white")
    geometry = geometry.resize(
        (core.shape[1] * scale, core.shape[0] * scale), Image.Resampling.NEAREST
    )
    geometry = _label(
        geometry,
        "04 FORMS: consensus centroids and PCA orientation/length",
        ["R0 diagnostic only; lines are measured axes, not erosion labels or source-height truth"],
    )

    images = [
        ("01-source-direct-rgb.png", source, "direct-support source RGB after fixed boundary erosion"),
        ("02-nominal-exclusion-mask.png", mask_image, "nominal exclusion ownership by final precedence"),
        ("03-consensus-negative-depth.png", depth_image, "nominal depression depth with all-variant consensus core"),
        ("04-consensus-form-geometry.png", geometry, "consensus-form centroid, orientation, and length overlay"),
    ]
    records = []
    for name, image, interpretation in images:
        path = root / name
        image.save(path, format="PNG", optimize=False, compress_level=9)
        records.append(
            {
                "path": name,
                "sha256": _sha256_file(path),
                "width_px": image.width,
                "height_px": image.height,
                "interpretation": interpretation,
            }
        )
    return records


def _summarize_forms(forms: list[dict[str, Any]]) -> dict[str, Any]:
    spacing = [
        form["nearest_centroid_spacing_m"]
        for form in forms
        if form["nearest_centroid_spacing_m"] is not None
    ]
    return {
        "count": len(forms),
        "length_m": _distribution([form["length_m"] for form in forms]),
        "equivalent_width_m": _distribution([form["equivalent_width_m"] for form in forms]),
        "median_depth_m": _distribution([form["depth_m"]["median"] for form in forms]),
        "p95_depth_m": _distribution([form["depth_m"]["p95"] for form in forms]),
        "orientation_deg_ccw_from_positive_x": _distribution(
            [form["orientation_deg_ccw_from_positive_x"] for form in forms]
        ),
        "nearest_centroid_spacing_m": _distribution(spacing),
        "branch_point_clusters": _distribution(
            [float(form["branch_point_clusters"]) for form in forms]
        ),
    }


def build_diagnostic(output_root: Path | None = None) -> Path:
    archive = _archive()
    recipe = {
        "schema_version": _SCHEMA,
        "algorithm": _ALGORITHM,
        "source": {
            "archive_sha256": _ARCHIVE_SHA256,
            "dense_member": _DENSE_MEMBER,
            "dense_sha256": _DENSE_SHA256,
            "precision_member": _PRECISION_MEMBER,
            "precision_sha256": _PRECISION_SHA256,
            "m3c2_member_excluded": _M3C2_MEMBER,
            "m3c2_exclusion_reason": "comparison references absent dense epoch 0",
        },
        "grid": {
            "cell_m": _CELL_M,
            "footprint_quantile_each_tail": _FOOTPRINT_QUANTILE,
            "boundary_erosion_m": _BOUNDARY_EROSION_M,
            "minimum_direct_points_per_cell": _MINIMUM_DIRECT_POINTS,
            "no_interpolation_for_measurement": True,
        },
        "component_limits": {
            "minimum_area_m2": _MINIMUM_COMPONENT_AREA_M2,
            "minimum_length_m": _MINIMUM_COMPONENT_LENGTH_M,
            "minimum_length_to_equivalent_width": _MINIMUM_COMPONENT_ASPECT,
            "maximum_slope_axis_difference_deg": _MAXIMUM_SLOPE_AXIS_DIFFERENCE_DEG,
        },
        "variants": [variant.__dict__ for variant in _VARIANTS],
    }
    artifact_id = hashlib.sha256(_canonical_json(recipe)).hexdigest()
    root = output_root or (
        DATA_WORK
        / "microtopography"
        / "opara_1038"
        / "negative_form_diagnostic"
        / "sha256"
        / artifact_id
    )
    index_path = root / "index.json"
    if index_path.exists():
        return index_path

    with tempfile.TemporaryDirectory(prefix="opara-1038-negative-forms-") as temporary:
        dense_path, precision_path = _extract_exact(archive, Path(temporary))
        raw = _read_ply(dense_path)
        if np.any(raw["class"] != 0):
            raise ValueError("dense_438 class channel is no longer constant zero")
        records, duplicate_count = _deduplicate(raw)
        precision = _read_precision(precision_path)
        grid = _aggregate(records, precision)
        slope_axis_degrees, slope_percent = _slope_axis(grid)
        variants = {
            variant.name: _variant_result(grid, variant, slope_axis_degrees)
            for variant in _VARIANTS
        }
        core = np.logical_and.reduce(
            [variants[variant.name]["negative"] for variant in _VARIANTS]
        )
        core = _filter_rill_like(core, grid["bounds"], slope_axis_degrees)
        forms = _form_metrics(core, variants["nominal"]["depth"], grid["bounds"])

        sensitivity = {}
        for variant in _VARIANTS:
            result = variants[variant.name]
            variant_forms = _form_metrics(
                result["negative"], result["depth"], grid["bounds"]
            )
            sensitivity[variant.name] = {
                "form_summary": _summarize_forms(variant_forms),
                "valid_cell_count": int(np.count_nonzero(result["masks"]["valid"])),
                "negative_cell_count": int(np.count_nonzero(result["negative"])),
                "mask_cell_counts": {
                    name: int(np.count_nonzero(mask))
                    for name, mask in result["masks"].items()
                    if name != "valid"
                },
            }

        root.mkdir(parents=True, exist_ok=False)
        images = _render_pngs(root, grid, variants, core, forms)
        summaries = _summarize_forms(forms)
        index = {
            **recipe,
            "artifact_id": artifact_id,
            "status": "diagnostic_forms_survived" if forms else "no_conservative_forms_survived",
            "claim_boundary": {
                "evidence_rank": "R0",
                "interpretation": "connected negative-form diagnostic only",
                "not_claimed": [
                    "erosion truth",
                    "independent geometry truth",
                    "R1 weak supervision",
                    "Estonia transfer",
                    "source patch or height transfer",
                ],
            },
            "source_observations": {
                "raw_vertex_count": int(len(raw)),
                "exact_duplicate_xyz_removed": int(duplicate_count),
                "unique_vertex_count": int(len(records)),
                "constant_class_value": 0,
                "cropped_direct_point_count": grid["input_point_count"],
                "robust_source_bounds_xy_m": grid["source_bounds"],
                "eroded_grid_bounds_xy_m": list(grid["bounds"]),
                "eroded_grid_shape_yx": list(grid["shape"]),
                "robust_slope_percent": slope_percent,
                "robust_slope_axis_deg_ccw_from_positive_x": slope_axis_degrees,
            },
            "consensus": {
                "definition": "pixelwise intersection of all three retained variant form masks",
                "confidence": "low scientific confidence; high procedural conservatism within retained R0 source",
                "form_summary": summaries,
                "forms": forms,
            },
            "sensitivity": sensitivity,
            "qa_images": images,
        }
        encoded = _canonical_json(index)
        temporary_index = index_path.with_suffix(".json.part")
        with temporary_index.open("xb") as target:
            target.write(encoded)
            target.flush()
            os.fsync(target.fileno())
        temporary_index.replace(index_path)
    return index_path


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the bounded OPARA dense_438 R0 negative-form diagnostic."
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(build_diagnostic(args.output_root))


if __name__ == "__main__":
    _main()
