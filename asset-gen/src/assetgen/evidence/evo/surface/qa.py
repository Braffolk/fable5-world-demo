"""Labeled visual diagnostics for the Evo surface candidate."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

_PANEL = 600
_LABEL = 86
_INVALID = np.asarray((104, 25, 42), dtype=np.uint8)


def _sequential(value: np.ndarray) -> np.ndarray:
    value = np.nan_to_num(np.clip(value, 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0)
    stops = np.asarray(
        ((5, 18, 38), (17, 93, 112), (41, 160, 132), (190, 219, 91), (250, 224, 71)),
        dtype=np.float32,
    )
    position = value * (len(stops) - 1)
    lower = np.floor(position).astype(np.intp)
    upper = np.minimum(lower + 1, len(stops) - 1)
    weight = (position - lower)[..., None]
    return np.rint(stops[lower] * (1 - weight) + stops[upper] * weight).astype(np.uint8)


def _diverging(value: np.ndarray) -> np.ndarray:
    value = np.nan_to_num(np.clip(value, -1.0, 1.0), nan=0.0, posinf=1.0, neginf=-1.0)
    low = np.asarray((39, 91, 156), dtype=np.float32)
    middle = np.asarray((239, 237, 220), dtype=np.float32)
    high = np.asarray((184, 56, 43), dtype=np.float32)
    weight = np.abs(value)[..., None]
    end = np.where((value >= 0)[..., None], high, low)
    return np.rint(middle * (1 - weight) + end * weight).astype(np.uint8)


def _scale(values: np.ndarray, valid: np.ndarray, quantile: float = 0.98) -> float:
    observed = np.abs(values[valid])
    return max(float(np.quantile(observed, quantile)), 1e-6) if observed.size else 1.0


def _affine_residual(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    y, x = np.indices(values.shape, dtype=np.float64)
    design = np.column_stack((np.ones(np.count_nonzero(valid)), x[valid], y[valid]))
    coefficients, *_ = np.linalg.lstsq(design, values[valid], rcond=None)
    return values - (coefficients[0] + coefficients[1] * x + coefficients[2] * y)


def _panel(rgb: np.ndarray, title: str, detail: str) -> Image.Image:
    canvas = Image.new("RGB", (_PANEL, _PANEL + _LABEL), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill="black")
    draw.text((10, 30), detail, fill=(45, 45, 45))
    image = Image.fromarray(np.flipud(rgb), mode="RGB").resize(
        (_PANEL - 20, _PANEL - 20), Image.Resampling.NEAREST
    )
    canvas.paste(image, (10, _LABEL))
    return canvas


def _pair(path: Path, left: Image.Image, right: Image.Image, footer: str) -> None:
    footer_h = 42
    canvas = Image.new("RGB", (left.width + right.width, left.height + footer_h), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    ImageDraw.Draw(canvas).text((10, left.height + 12), footer, fill=(90, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=True)


def _hillshade(height: np.ndarray, valid: np.ndarray, cell_m: float) -> np.ndarray:
    filled = height.copy()
    if not np.all(valid):
        median = float(np.nanmedian(filled[valid])) if np.any(valid) else 0.0
        filled[~valid] = median
    dz_dy, dz_dx = np.gradient(filled, cell_m)
    azimuth = np.deg2rad(315.0)
    altitude = np.deg2rad(38.0)
    slope = np.arctan(np.hypot(dz_dx, dz_dy))
    aspect = np.arctan2(dz_dy, -dz_dx)
    light = np.sin(altitude) * np.cos(slope) + np.cos(altitude) * np.sin(slope) * np.cos(
        azimuth - aspect
    )
    gray = np.rint(255.0 * np.clip((light + 0.15) / 1.15, 0.0, 1.0)).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    neighborhood = valid.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            neighborhood &= np.roll(np.roll(valid, dy, axis=0), dx, axis=1)
    neighborhood[[0, -1], :] = False
    neighborhood[:, [0, -1]] = False
    rgb[~neighborhood] = _INVALID
    return rgb


def render_surface_qa(
    arrays: dict[str, np.ndarray], qa_dir: Path, cell_m: float
) -> tuple[dict[str, Any], ...]:
    height = arrays["height_m_f64"]
    valid = arrays["heightfield_valid"]
    observed = arrays["valid_observed"]

    observed_height = height[observed]
    low = float(np.nanquantile(observed_height, 0.01))
    high = float(np.nanquantile(observed_height, 0.99))
    height_rgb = _sequential((height - low) / max(high - low, 1e-6))
    height_rgb[~observed] = _INVALID
    residual = _affine_residual(height, observed)
    residual_scale = _scale(residual, observed)
    residual_rgb = _diverging(residual / residual_scale)
    residual_rgb[~observed] = _INVALID
    path1 = qa_dir / "01_candidate_height.png"
    _pair(
        path1,
        _panel(height_rgb, "Direct candidate height", f"p01..p99 {low:.3f}..{high:.3f} m"),
        _panel(residual_rgb, "Detrended candidate height", f"display limits +/-{residual_scale:.3f} m"),
        "SOURCE-NORMALIZED FLOOR-SHEET CANDIDATE / RED = NO DIRECT ESTIMATE / NORTH UP",
    )

    hillshade = _hillshade(height, valid, cell_m)
    slope = arrays["local_plane_slope"]
    slope_ceiling = max(float(np.nanquantile(slope[observed], 0.99)), 1e-6)
    slope_rgb = _sequential(slope / slope_ceiling)
    slope_rgb[~valid] = _INVALID
    path2 = qa_dir / "02_candidate_hillshade.png"
    _pair(
        path2,
        _panel(hillshade, "Candidate hillshade", "315 deg azimuth; red lacks a valid 3x3 neighborhood"),
        _panel(slope_rgb, "Local fitted slope", f"0..p99 {slope_ceiling:.3f} rise/run"),
        "NO HOLE FILLING OR INTERPOLATION / HILLSHADE USES ONLY VALID 3x3 OUTPUT NEIGHBORHOODS",
    )

    views = arrays["view_group_count"].astype(np.float64)
    view_ceiling = max(float(np.nanmax(views)), 1.0)
    view_rgb = _sequential(views / view_ceiling)
    view_rgb[~observed] = _INVALID
    nearest = arrays["nearest_support_m"].astype(np.float64)
    nearest_rgb = _sequential(nearest / (cell_m / np.sqrt(2.0)))
    nearest_rgb[~np.isfinite(nearest)] = _INVALID
    path3 = qa_dir / "03_direct_support.png"
    _pair(
        path3,
        _panel(view_rgb, "Source-ID groups", f"0..{view_ceiling:.0f}; three IDs are a redundancy gate only"),
        _panel(nearest_rgb, "Nearest accepted return", f"0..cell corner {cell_m / np.sqrt(2.0):.4f} m"),
        "DIRECT H=0 / TREEID=0 / LAST-RETURN SUPPORT ONLY / PHYSICAL VIEW COUNT UNKNOWN",
    )

    ambiguity = arrays["ambiguity_flags"]
    palette = np.asarray(
        (
            (30, 128, 95),
            (104, 25, 42),
            (215, 125, 38),
            (196, 78, 44),
            (123, 72, 141),
            (70, 80, 105),
        ),
        dtype=np.uint8,
    )
    category = np.zeros(ambiguity.shape, dtype=np.uint8)
    category[(ambiguity & 1) != 0] = 1
    category[(ambiguity & 2) != 0] = 2
    category[(ambiguity & 4) != 0] = 3
    category[(ambiguity & 8) != 0] = 4
    category[(ambiguity & 16) != 0] = 5
    ambiguity_rgb = palette[category]
    fraction = arrays["inlier_fraction"].astype(np.float64)
    fraction_rgb = _sequential(fraction)
    fraction_rgb[~observed] = _INVALID
    path4 = qa_dir / "04_ambiguity_and_abstention.png"
    _pair(
        path4,
        _panel(ambiguity_rgb, "Geometric abstention", "green valid; red none; orange views; rust bracket; purple rank; gray consensus"),
        _panel(fraction_rgb, "Plane-consensus fraction", "accepted within 0.02 m of first local plane"),
        "SEMANTIC CONFIDENCE REMAINS UNKNOWN EVERYWHERE / NOT TARGET TRUTH / NOT SYNTHESIS AUTHORITY",
    )

    return (
        {"path": path1, "interpretation": "Direct 6.25 cm candidate height and its display-only affine residual. Red cells have no direct candidate; no gaps are filled."},
        {"path": path2, "interpretation": "Hillshade and fitted local slope for geometrically valid cells. Hillshade suppresses every pixel lacking a fully valid 3 by 3 neighborhood."},
        {"path": path3, "interpretation": "Unique source-ID support and nearest accepted return distance. Three IDs provide a redundancy gate; their mapping to independent physical stations is unavailable."},
        {"path": path4, "interpretation": "Dominant geometric abstention reason and robust plane-consensus fraction. Stable-organic, low vegetation, roots, and deadwood remain semantically unresolved."},
    )
