"""Labeled visual diagnostics for conservative Hovi candidate surfaces."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .model import SurfaceCandidate

_WIDTH = 1100
_MARGIN = 22
_TITLE_H = 52


def _palette(values: np.ndarray, stops: tuple[tuple[int, int, int], ...]) -> np.ndarray:
    value = np.nan_to_num(
        values.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0
    )
    value = np.clip(value, 0.0, 1.0)
    colors = np.asarray(stops, dtype=np.float32)
    position = value * (len(colors) - 1)
    lower = np.floor(position).astype(np.intp)
    upper = np.minimum(lower + 1, len(colors) - 1)
    weight = (position - lower)[..., None]
    return np.rint(colors[lower] * (1.0 - weight) + colors[upper] * weight).astype(np.uint8)


def _fit_image(rgb: np.ndarray, title: str, output: Path) -> None:
    source = Image.fromarray(np.flipud(rgb), mode="RGB")
    scale = min((_WIDTH - 2 * _MARGIN) / source.width, 900 / source.height)
    size = (
        max(1, int(round(source.width * scale))),
        max(1, int(round(source.height * scale))),
    )
    source = source.resize(size, Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (_WIDTH, _TITLE_H + size[1] + _MARGIN), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((_MARGIN, 16), title, fill="black")
    left = (_WIDTH - size[0]) // 2
    canvas.paste(source, (left, _TITLE_H))
    draw.rectangle((left - 1, _TITLE_H - 1, left + size[0], _TITLE_H + size[1]), outline="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def _nearest_fill(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros(values.shape, dtype=np.float32)
    indices = ndimage.distance_transform_edt(~finite, return_distances=False, return_indices=True)
    return values[tuple(indices)].astype(np.float32)


def _height_hillshade(candidate: SurfaceCandidate) -> np.ndarray:
    height = candidate.absolute_local_z_m
    direct = candidate.direct_mask & np.isfinite(height)
    valid = ndimage.binary_erosion(
        direct, structure=ndimage.generate_binary_structure(2, 2), border_value=0
    )
    filled = _nearest_fill(height)
    values = height[valid]
    low, high = (np.quantile(values, (0.01, 0.99)) if values.size else (0.0, 1.0))
    normalized = np.clip((filled - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    rgb = _palette(
        normalized,
        ((28, 42, 31), (54, 97, 57), (132, 143, 74), (184, 139, 84), (224, 211, 174)),
    ).astype(np.float32)
    dy, dx = np.gradient(filled, candidate.grid.resolution_m)
    normal = np.dstack((-dx, -dy, np.ones_like(filled)))
    normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-12)
    light = np.asarray((-0.45, 0.55, 0.70), dtype=np.float32)
    light /= np.linalg.norm(light)
    shade = np.clip(np.sum(normal * light, axis=2), 0.0, 1.0)
    rgb *= (0.45 + 0.70 * shade)[..., None]
    rgb[~valid] = (42, 42, 42)
    rgb[candidate.inferred_mask] = 0.65 * rgb[candidate.inferred_mask] + 0.35 * np.asarray(
        (40, 190, 210), dtype=np.float32
    )
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _selection_score(candidate: SurfaceCandidate) -> np.ndarray:
    rgb = _palette(
        candidate.selection_score,
        ((62, 20, 35), (180, 55, 45), (224, 170, 64), (74, 167, 113), (224, 240, 210)),
    )
    rgb[candidate.unsupported_mask] = (25, 25, 25)
    rgb[~candidate.analysis_core_mask] = (93, 85, 107)
    return rgb


def _status(candidate: SurfaceCandidate) -> np.ndarray:
    rgb = np.full((*candidate.direct_mask.shape, 3), (28, 28, 31), dtype=np.uint8)
    rgb[candidate.unsupported_mask] = (35, 35, 35)
    rgb[candidate.direct_mask] = (62, 164, 91)
    rgb[candidate.inferred_mask] = (41, 178, 205)
    rgb[candidate.above_sheet_competitor_mask] = (191, 50, 53)
    rgb[candidate.ambiguity_mask] = (237, 146, 46)
    rgb[candidate.topology_rejected_mask] = (193, 66, 178)
    rgb[~candidate.analysis_core_mask] = (93, 85, 107)
    return rgb


def _interpolation(candidate: SurfaceCandidate) -> np.ndarray:
    rgb = np.full((*candidate.direct_mask.shape, 3), (32, 42, 50), dtype=np.uint8)
    rgb[candidate.direct_mask] = (72, 92, 84)
    rgb[candidate.interpolation_mask] = (40, 210, 225)
    rgb[candidate.ambiguity_mask] = (225, 136, 44)
    rgb[~candidate.analysis_core_mask] = (32, 42, 50)
    return rgb


def _relief(candidate: SurfaceCandidate) -> np.ndarray:
    valid = candidate.direct_mask & np.isfinite(candidate.absolute_local_z_m) & np.isfinite(
        candidate.structural_reference_z_m
    )
    relief = candidate.absolute_local_z_m - candidate.structural_reference_z_m
    scale = float(np.quantile(np.abs(relief[valid]), 0.98)) if np.any(valid) else 0.20
    scale = max(scale, 0.04)
    normalized = np.clip(0.5 + 0.5 * relief / scale, 0.0, 1.0)
    rgb = _palette(
        normalized,
        ((30, 71, 121), (105, 158, 190), (226, 225, 208), (196, 118, 72), (113, 42, 43)),
    )
    rgb[~valid] = (32, 32, 32)
    return rgb


def _analysis_domain(candidate: SurfaceCandidate) -> np.ndarray:
    rgb = np.full((*candidate.direct_mask.shape, 3), (22, 22, 24), dtype=np.uint8)
    rgb[candidate.nominal_layout_mask] = (86, 73, 104)
    rgb[candidate.pre_erosion_domain_mask] = (50, 92, 132)
    internal_invalid = candidate.analysis_core_mask & ~candidate.tangent_reference_valid_mask
    internal_valid = candidate.analysis_core_mask & candidate.tangent_reference_valid_mask
    rgb[internal_invalid] = (190, 66, 52)
    rgb[internal_valid] = (56, 164, 105)
    return rgb


def render_surface_qa(
    candidate: SurfaceCandidate,
    qa_dir: Path,
    *,
    disposition: str,
) -> tuple[tuple[Path, str], ...]:
    abstention = disposition == "abstention"
    prefix = (
        "ABSTENTION DIAGNOSTIC; NO SURFACE IS AUTHORIZED. "
        if abstention
        else "UNQUALIFIED Q1 PREVIEW ONLY; UNUSABLE AS A SURFACE OR SYNTHESIS TARGET. "
    )
    title_prefix = "ABSTENTION" if abstention else "UNUSABLE Q1 PREVIEW"
    records = (
        (
            "01_direct_height_hillshade.png",
            _height_hillshade(candidate),
            prefix + "Direct-only absolute local-z hillshade. The displayed mask is eroded one cell around every NaN or inferred cell; nearest fill is used only behind the hidden derivative stencil, never displayed as geometry.",
        ),
        (
            "02_selection_score.png",
            _selection_score(candidate),
            prefix + "Per-cell geometric selection score from mode support, prominence, one-sidedness, dominance, and tangent-residual neighbor agreement. It is not confidence or qualification probability.",
        ),
        (
            "03_ambiguity_above_sheet_unsupported.png",
            _status(candidate),
            prefix + "Status classes: green direct, cyan inferred, orange ambiguous competing sheets, red neutral above-sheet/multi-surface competition, magenta disconnected direct candidates relabeled unresolved, dark unsupported, gray-violet outside the eroded analysis core.",
        ),
        (
            "04_interpolation.png",
            _interpolation(candidate),
            prefix + "Interpolation audit. Cyan is limited to tiny enclosed holes passing a local-plane residual gate; direct observed-mode cells are never modified.",
        ),
        (
            "05_local_relief_from_structural_sheet.png",
            _relief(candidate),
            prefix + "Direct-only signed local relief relative to the selected 0.25 m structural tangent sheet, not a decorative high-pass or synthesized displacement.",
        ),
        (
            "06_analysis_domain_and_tangent_support.png",
            _analysis_domain(candidate),
            prefix + "Analysis-domain stages: violet nominal layout, blue observed outer domain before halo erosion, green analysis-core cells with a valid structural tangent, red internal tangent-invalid cells retained as unsupported holes in the denominator, black outside.",
        ),
    )
    output = []
    for filename, rgb, interpretation in records:
        path = qa_dir / filename
        _fit_image(
            rgb,
            f"HY_SPRUCE4 - {title_prefix} - {filename.removesuffix('.png').replace('_', ' ')} - north up",
            path,
        )
        output.append((path, interpretation))
    return tuple(output)


def render_sparse_photo_qa(photo_paths: tuple[Path, ...], qa_dir: Path) -> tuple[Path, str]:
    if len(photo_paths) != 4:
        raise ValueError("Hovi sparse photo QA requires the four retained quadrat photographs")
    panel_w, panel_h = 520, 390
    canvas = Image.new("RGB", (2 * panel_w, 2 * panel_h + 54), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (16, 16),
        "HY_SPRUCE4 sparse unregistered semantic QA only - not dense labels or geometry",
        fill="black",
    )
    for index, path in enumerate(photo_paths):
        with Image.open(path) as source:
            image = source.convert("RGB")
        image.thumbnail((panel_w - 16, panel_h - 40), Image.Resampling.LANCZOS)
        left = (index % 2) * panel_w + (panel_w - image.width) // 2
        top = 54 + (index // 2) * panel_h + 28
        canvas.paste(image, (left, top))
        draw.text(((index % 2) * panel_w + 8, 54 + (index // 2) * panel_h + 8), path.stem, fill="black")
    output = qa_dir / "06_sparse_quadrat_photo_qa.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)
    return (
        output,
        "Four retained 1 m quadrat photographs shown only to verify that moss/litter, roots/twigs, and vascular vegetation coexist. They are not spatially registered and do not label TLS cells.",
    )
