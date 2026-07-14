"""Labeled QA for Biala Gora coastal-process calibration evidence."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

_PANEL_W = 640
_PANEL_H = 600
_LABEL_H = 84


def _sequential(value: np.ndarray) -> np.ndarray:
    value = np.nan_to_num(np.clip(value, 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0)
    stops = np.asarray(
        ((7, 18, 30), (23, 89, 113), (38, 157, 127), (190, 217, 88), (250, 225, 77)),
        dtype=np.float32,
    )
    position = value * (len(stops) - 1)
    lower = np.floor(position).astype(np.intp)
    upper = np.minimum(lower + 1, len(stops) - 1)
    weight = (position - lower)[..., None]
    return np.rint(stops[lower] * (1 - weight) + stops[upper] * weight).astype(np.uint8)


def _diverging(value: np.ndarray) -> np.ndarray:
    value = np.nan_to_num(np.clip(value, -1.0, 1.0), nan=0.0, posinf=1.0, neginf=-1.0)
    low = np.asarray((35, 90, 149), dtype=np.float32)
    middle = np.asarray((240, 237, 218), dtype=np.float32)
    high = np.asarray((180, 54, 45), dtype=np.float32)
    weight = np.abs(value)[..., None]
    end = np.where((value >= 0)[..., None], high, low)
    return np.rint(middle * (1 - weight) + end * weight).astype(np.uint8)


def _nearest_fill(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if not np.any(valid):
        return np.zeros(values.shape, dtype=np.float64)
    indices = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return values[tuple(indices)]


def _panel(rgb: np.ndarray, title: str, detail: str, *, photo: bool = False) -> Image.Image:
    canvas = Image.new("RGB", (_PANEL_W, _PANEL_H + _LABEL_H), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill="black")
    draw.text((10, 31), detail, fill=(50, 50, 50))
    image = Image.fromarray(np.flipud(rgb), mode="RGB")
    image.thumbnail(
        (_PANEL_W - 20, _PANEL_H - 20),
        Image.Resampling.BILINEAR if photo else Image.Resampling.NEAREST,
    )
    canvas.paste(image, ((_PANEL_W - image.width) // 2, _LABEL_H))
    return canvas


def _pair(path: Path, left: Image.Image, right: Image.Image, footer: str) -> None:
    footer_h = 42
    canvas = Image.new("RGB", (left.width + right.width, left.height + footer_h), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    ImageDraw.Draw(canvas).text((10, left.height + 12), footer, fill=(100, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=True)


def render_candidate_qa(
    arrays: dict[str, np.ndarray], qa_dir: Path, *, cell_m: float
) -> tuple[dict[str, Any], ...]:
    all_count = arrays["all_return_count"]
    ground_count = arrays["vendor_class2_count"]
    rgb_count = arrays["rgb_return_count"]
    rgb = arrays["rgb_mean_u8"].copy()
    rgb[rgb_count == 0] = (104, 18, 43)
    observed = all_count > 0
    ground_fraction = np.divide(
        ground_count,
        all_count,
        out=np.zeros(all_count.shape, dtype=np.float64),
        where=observed,
    )
    ground_rgb = _sequential(ground_fraction)
    ground_rgb[~observed] = (104, 18, 43)
    path1 = qa_dir / "01_rgb_and_vendor_ground_fraction.png"
    _pair(
        path1,
        _panel(rgb, "Mean source RGB", "magenta = no nonzero RGB return", photo=True),
        _panel(ground_rgb, "Vendor class-2 fraction", "0..1 of all returns; magenta = unobserved"),
        "CALIBRATION ONLY / VENDOR CLASS 2 IS NOT AUTHORITATIVE GROUND / NORTH UP",
    )

    ground = arrays["vendor_class2_z_mean_m"].astype(np.float64)
    valid = np.isfinite(ground)
    filled = _nearest_fill(ground, valid)
    gy, gx = np.gradient(filled, cell_m)
    normal_z = np.ones(filled.shape, dtype=np.float64)
    length = np.sqrt(gx * gx + gy * gy + normal_z * normal_z)
    light = (-0.45 * gx - 0.55 * gy + 0.70 * normal_z) / length
    hillshade = np.clip(0.5 + 0.5 * light, 0.0, 1.0)
    shade_rgb = np.repeat(np.rint(255 * hillshade)[..., None].astype(np.uint8), 3, axis=2)
    shade_rgb[~valid] = (104, 18, 43)
    sigma_cells = 10.0 / cell_m
    background = ndimage.gaussian_filter(filled, sigma=sigma_cells, mode="nearest")
    relief = filled - background
    relief_scale = max(float(np.quantile(np.abs(relief[valid]), 0.98)), 0.01)
    relief_rgb = _diverging(relief / relief_scale)
    relief_rgb[~valid] = (104, 18, 43)
    path2 = qa_dir / "02_vendor_ground_process_forms.png"
    _pair(
        path2,
        _panel(shade_rgb, "Vendor class-2 hillshade", "nearest fill only drives shading; unsupported cells remain magenta"),
        _panel(relief_rgb, "Display-only local relief", f"z minus 10 m Gaussian background; +/-{relief_scale:.2f} m"),
        "UNQUALIFIED MORPHOLOGY CANDIDATE / DISPLAY DERIVATIVES ARE NOT TARGET ARRAYS",
    )

    all_range = arrays["all_return_z_range_m"].astype(np.float64)
    all_valid = np.isfinite(all_range)
    all_scale = max(float(np.quantile(all_range[all_valid], 0.99)), 0.01)
    all_rgb = _sequential(all_range / all_scale)
    all_rgb[~all_valid] = (104, 18, 43)
    ground_range = arrays["vendor_class2_z_range_m"].astype(np.float64)
    ground_valid = np.isfinite(ground_range)
    ground_scale = max(float(np.quantile(ground_range[ground_valid], 0.99)), 0.01)
    range_rgb = _sequential(ground_range / ground_scale)
    range_rgb[~ground_valid] = (104, 18, 43)
    path3 = qa_dir / "03_vertical_multiplicity.png"
    _pair(
        path3,
        _panel(all_rgb, "All-return vertical range", f"0..p99 {all_scale:.2f} m per {cell_m:g} m cell"),
        _panel(range_rgb, "Vendor class-2 vertical range", f"0..p99 {ground_scale:.2f} m per {cell_m:g} m cell"),
        "VERTICAL MULTIPLICITY FLAGS VEGETATION / CLIFF / CLASSIFICATION RISK; NOT HEIGHTFIELD TRUTH",
    )
    return (
        {
            "path": path1,
            "interpretation": "Source RGB and vendor class-2 fraction expose land cover and the spatial behavior of the supplied classification without promoting it to authoritative ground.",
        },
        {
            "path": path2,
            "interpretation": "A display-only hillshade and 10 m-background residual reveal coastal cliff, slope-failure, beach, and surface-form candidates where vendor class 2 exists. Unsupported cells remain explicit.",
        },
        {
            "path": path3,
            "interpretation": "Within-cell vertical ranges expose multi-surface, vegetation, steep-face, and classification ambiguity relevant to heightfield validity.",
        },
    )
