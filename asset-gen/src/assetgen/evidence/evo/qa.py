"""Small, explicitly unqualified QA views for Evo raw candidate evidence."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

_PANEL = 600
_LABEL = 86


def _sequential(value: np.ndarray) -> np.ndarray:
    value = np.nan_to_num(np.clip(value, 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0)
    stops = np.asarray(
        ((6, 16, 31), (14, 87, 111), (35, 158, 131), (183, 222, 91), (252, 226, 74)),
        dtype=np.float32,
    )
    position = value * (len(stops) - 1)
    lower = np.floor(position).astype(np.intp)
    upper = np.minimum(lower + 1, len(stops) - 1)
    weight = (position - lower)[..., None]
    return np.rint(stops[lower] * (1 - weight) + stops[upper] * weight).astype(np.uint8)


def _diverging(value: np.ndarray) -> np.ndarray:
    value = np.nan_to_num(
        np.clip(value, -1.0, 1.0), nan=0.0, posinf=1.0, neginf=-1.0
    )
    low = np.asarray((36, 91, 150), dtype=np.float32)
    middle = np.asarray((241, 238, 220), dtype=np.float32)
    high = np.asarray((180, 55, 46), dtype=np.float32)
    weight = np.abs(value)[..., None]
    end = np.where((value >= 0)[..., None], high, low)
    return np.rint(middle * (1 - weight) + end * weight).astype(np.uint8)


def _valid_scale(values: np.ndarray, valid: np.ndarray, quantile: float = 0.99) -> float:
    observed = np.abs(values[valid])
    if not observed.size:
        return 1.0
    return max(float(np.quantile(observed, quantile)), 1e-6)


def _affine_residual(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    y, x = np.indices(values.shape, dtype=np.float64)
    design = np.column_stack((np.ones(np.count_nonzero(valid)), x[valid], y[valid]))
    coefficients, *_ = np.linalg.lstsq(design, values[valid], rcond=None)
    plane = coefficients[0] + coefficients[1] * x + coefficients[2] * y
    return values - plane


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


def render_candidate_qa(arrays: dict[str, np.ndarray], qa_dir: Path) -> tuple[dict[str, Any], ...]:
    all_count = arrays["all_return_count"]
    unassigned_count = arrays["unassigned_return_count"]
    occupied = all_count > 0
    logged = np.log1p(all_count.astype(np.float64))
    count_ceiling = max(float(np.quantile(logged[occupied], 0.99)), 1.0)
    count_rgb = _sequential(logged / count_ceiling)
    fraction = np.divide(
        unassigned_count,
        all_count,
        out=np.zeros(all_count.shape, dtype=np.float64),
        where=occupied,
    )
    fraction_rgb = _sequential(fraction)
    fraction_rgb[~occupied] = (92, 20, 41)
    path1 = qa_dir / "01_raw_observation_support.png"
    _pair(
        path1,
        _panel(count_rgb, "All-return count", f"log scale; p99 log ceiling {count_ceiling:.3g}"),
        _panel(fraction_rgb, "treeid=0 fraction", "red = unobserved; zero means assigned tree returns only"),
        "RAW CANDIDATE / UNQUALIFIED / NOT GROUND CLASSIFICATION / NO SYNTHESIS AUTHORIZATION",
    )

    reference = arrays["source_ground_reference_z_mean_m"].astype(np.float64)
    reference_valid = np.isfinite(reference)
    reference_residual = _affine_residual(reference, reference_valid)
    reference_scale = _valid_scale(reference_residual, reference_valid, 0.98)
    reference_rgb = _diverging(reference_residual / reference_scale)
    reference_rgb[~reference_valid] = (92, 20, 41)
    spread = arrays["source_ground_reference_z_range_m"].astype(np.float64)
    spread_valid = np.isfinite(spread)
    spread_ceiling = max(float(np.quantile(spread[spread_valid], 0.99)), 0.01)
    spread_rgb = _sequential(spread / spread_ceiling)
    spread_rgb[~spread_valid] = (92, 20, 41)
    path2 = qa_dir / "02_source_height_reference.png"
    _pair(
        path2,
        _panel(reference_rgb, "Mean source z - h", f"affine residual; color limits +/-{reference_scale:.3f} m"),
        _panel(spread_rgb, "Within-cell source z - h range", f"0..p99 {spread_ceiling:.3f} m"),
        "SOURCE-PROVIDED HEIGHT REFERENCE / NOT INDEPENDENTLY MEASURED FLOOR / NORTH UP",
    )

    lower = arrays["unassigned_lower_observed_z_m"].astype(np.float64)
    lower_valid = np.isfinite(lower)
    lower_residual = _affine_residual(lower, lower_valid)
    lower_scale = _valid_scale(lower_residual, lower_valid, 0.98)
    lower_rgb = _diverging(lower_residual / lower_scale)
    lower_rgb[~lower_valid] = (92, 20, 41)
    difference = lower - reference
    difference_valid = lower_valid & reference_valid
    difference_scale = _valid_scale(difference, difference_valid, 0.98)
    difference_rgb = _diverging(difference / difference_scale)
    difference_rgb[~difference_valid] = (92, 20, 41)
    path3 = qa_dir / "03_unassigned_lower_envelope.png"
    _pair(
        path3,
        _panel(lower_rgb, "treeid=0 lower observed z", f"affine residual; color limits +/-{lower_scale:.3f} m"),
        _panel(difference_rgb, "Lower z minus mean source z-h", f"color limits +/-{difference_scale:.3f} m"),
        "LOWEST UNASSIGNED RETURN IS NOT A GROUND LABEL OR TARGET / NORTH UP",
    )

    return (
        {
            "path": path1,
            "interpretation": "All-return observation density and the fraction carrying source treeid=0. Empty cells are explicit. treeid=0 mixes floor, understory, and other unassigned returns.",
        },
        {
            "path": path2,
            "interpretation": "The source-provided z-h reference after removing one display-only affine plane, plus its within-cell range. It diagnoses the supplied normalization and is not independent floor truth.",
        },
        {
            "path": path3,
            "interpretation": "The raw lower envelope of treeid=0 returns and its difference from the source z-h reference. Lowest-return selection is shown only as unqualified candidate evidence.",
        },
    )
