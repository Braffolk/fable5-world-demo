"""Packing-independent numeric and visual diagnostics for the frozen screen."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter, label
from scipy.stats import wasserstein_distance

from .operators import Reconstruction, SurfaceArrays


RELIEF_SIGMA_CELLS = 8.0
RELIEF_TRUNCATE_SIGMA = 4.0
GRADIENT_EROSION_CELLS = 1
MIN_SPECTRAL_SQUARE_CELLS = 64


def _target_support(truth: SurfaceArrays) -> np.ndarray:
    return (
        np.isfinite(truth.elevation_m)
        & np.isfinite(truth.confidence)
        & (truth.confidence > 0)
    )


def _finite_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = np.asarray(values)[mask & np.isfinite(values)]
    if not selected.size:
        raise ValueError("diagnostic has no supported samples")
    return selected


def _relief(
    height: np.ndarray,
    support: np.ndarray,
    sigma: float = RELIEF_SIGMA_CELLS,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian residual only where the complete truncated kernel is supported."""
    margin = int(np.ceil(sigma * RELIEF_TRUNCATE_SIGMA))
    interior = binary_erosion(support, iterations=margin, border_value=0)
    weight = gaussian_filter(
        support.astype(np.float64), sigma=sigma, mode="constant", cval=0.0,
        truncate=RELIEF_TRUNCATE_SIGMA,
    )
    smooth = np.divide(
        gaussian_filter(
            np.where(support, height, 0.0), sigma=sigma, mode="constant", cval=0.0,
            truncate=RELIEF_TRUNCATE_SIGMA,
        ),
        weight,
        out=np.full_like(height, np.nan, dtype=np.float64),
        where=weight > 0,
    )
    valid = interior & (weight >= 1.0 - 1e-12) & np.isfinite(height)
    return np.where(valid, height - smooth, np.nan), valid


def _component_areas(values: np.ndarray, support: np.ndarray) -> np.ndarray:
    sample = _finite_values(values, support)
    sigma = float(np.std(sample))
    if sigma <= 0:
        return np.asarray([0.0])
    areas: list[float] = []
    for sign in (-1.0, 1.0):
        components, count = label(support & (sign * values > 0.75 * sigma))
        if count:
            areas.extend(np.bincount(components.ravel())[1:].astype(float).tolist())
    return np.asarray(areas or [0.0])


def _largest_supported_square(mask: np.ndarray) -> tuple[slice, slice]:
    """Find a deterministic all-supported square for boundary-safe Fourier QA."""
    integral = np.pad(mask.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    for size in range(min(mask.shape), MIN_SPECTRAL_SQUARE_CELLS - 1, -1):
        sums = (
            integral[size:, size:]
            - integral[:-size, size:]
            - integral[size:, :-size]
            + integral[:-size, :-size]
        )
        matches = np.argwhere(sums == size * size)
        if matches.size:
            y0, x0 = (int(value) for value in matches[0])
            return slice(y0, y0 + size), slice(x0, x0 + size)
    raise ValueError(
        f"no all-supported {MIN_SPECTRAL_SQUARE_CELLS}-cell square for spectral QA"
    )


def _support_safe_axis_spectral_fraction(
    values: np.ndarray, support: np.ndarray
) -> tuple[float, int]:
    ys, xs = _largest_supported_square(support)
    crop = np.asarray(values[ys, xs], dtype=np.float64)
    yy, xx = np.mgrid[: crop.shape[0], : crop.shape[1]]
    design = np.column_stack((np.ones(crop.size), xx.ravel(), yy.ravel()))
    coefficients, *_ = np.linalg.lstsq(design, crop.ravel(), rcond=None)
    detrended = crop - (design @ coefficients).reshape(crop.shape)
    window = np.hanning(crop.shape[0])[:, None] * np.hanning(crop.shape[1])[None, :]
    spectrum = np.abs(np.fft.rfft2(detrended * window)) ** 2
    axial = np.r_[spectrum[0, 1:], spectrum[1:, 0]]
    return float(np.sum(axial) / max(1e-12, np.sum(spectrum))), crop.shape[0]


def assignment_trace(reconstruction: Reconstruction) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for row, atom_index in enumerate(reconstruction.atom_index):
        record: dict[str, Any] = {
            "target_patch_origin_low_yx": reconstruction.patch_origin_low[row].tolist(),
            "target_patch_origin_local_m_yx": (
                reconstruction.patch_origin_low[row].astype(float) - 8.0
            ).tolist(),
            "selected_atom_index": int(atom_index),
            "coefficient": None,
            "match_error": None,
            "selected_source_id": None,
            "selected_source_patch_origin_low_yx": None,
        }
        if atom_index >= 0:
            record.update(
                {
                    "coefficient": float(reconstruction.coefficient[row]),
                    "match_error": float(reconstruction.match_error[row]),
                    "selected_source_id": reconstruction.atom_source[int(atom_index)],
                    "selected_source_patch_origin_low_yx": reconstruction.atom_origin_low[
                        int(atom_index)
                    ].tolist(),
                    "selected_source_patch_origin_local_m_yx": (
                        reconstruction.atom_origin_low[int(atom_index)].astype(float) - 8.0
                    ).tolist(),
                }
            )
        trace.append(record)
    return trace


def numeric_diagnostics(
    truth: SurfaceArrays,
    reconstruction: Reconstruction,
    *,
    stride_hr: int,
) -> dict[str, Any]:
    target_support = _target_support(truth)
    common_supported = target_support & reconstruction.support
    common_direct = truth.measured & common_supported
    coverage = float(np.count_nonzero(common_direct) / max(1, np.count_nonzero(truth.measured)))
    truth_relief, truth_relief_valid = _relief(truth.elevation_m, common_supported)
    output_relief, output_relief_valid = _relief(reconstruction.elevation_m, common_supported)
    relief_valid = truth_relief_valid & output_relief_valid
    tr = _finite_values(truth_relief, relief_valid)
    rr = _finite_values(output_relief, relief_valid)
    truth_rms = float(np.sqrt(np.mean(tr * tr)))
    output_rms = float(np.sqrt(np.mean(rr * rr)))

    # Central-gradient samples are one-cell eroded, so neither operand can read a
    # void substitute or a support boundary.
    gradient_valid = binary_erosion(
        common_direct, iterations=GRADIENT_EROSION_CELLS, border_value=0
    )
    safe_truth = np.where(common_direct, truth.elevation_m, 0.0)
    safe_output = np.where(common_direct, reconstruction.elevation_m, 0.0)
    gy_t, gx_t = np.gradient(safe_truth)
    gy_r, gx_r = np.gradient(safe_output)
    grad_t = np.hypot(gx_t, gy_t)[gradient_valid]
    grad_r = np.hypot(gx_r, gy_r)[gradient_valid]
    if not grad_t.size:
        raise ValueError("no eroded common support for gradient QA")

    areas_t = _component_areas(truth_relief, relief_valid)
    areas_r = _component_areas(output_relief, relief_valid)
    area_scale = max(1.0, float(np.median(areas_t)))
    selected = reconstruction.atom_index[reconstruction.atom_index >= 0]
    atom_counts = Counter(int(value) for value in selected)
    source_counts = Counter(reconstruction.atom_source[index] for index in selected)
    total = max(1, len(selected))
    atom_prob = np.asarray(list(atom_counts.values()), dtype=float) / total
    source_prob = np.asarray(list(source_counts.values()), dtype=float) / total
    effective_atoms = float(1.0 / np.sum(atom_prob * atom_prob)) if atom_prob.size else 0.0

    dy = np.abs(np.diff(reconstruction.elevation_m, axis=0))
    dx = np.abs(np.diff(reconstruction.elevation_m, axis=1))
    valid_dy = common_supported[1:, :] & common_supported[:-1, :]
    valid_dx = common_supported[:, 1:] & common_supported[:, :-1]
    seam_rows = np.arange(stride_hr, reconstruction.elevation_m.shape[0], stride_hr) - 1
    seam_cols = np.arange(stride_hr, reconstruction.elevation_m.shape[1], stride_hr) - 1
    seam_values = np.r_[
        dy[seam_rows, :][valid_dy[seam_rows, :]],
        dx[:, seam_cols][valid_dx[:, seam_cols]],
    ]
    all_steps = np.r_[dy[valid_dy], dx[valid_dx]]
    seam_ratio = (
        float(np.median(seam_values) / max(1e-12, np.median(all_steps)))
        if seam_values.size else float("inf")
    )
    axial_fraction, spectral_square_cells = _support_safe_axis_spectral_fraction(
        output_relief, relief_valid
    )
    neighbor_pairs = relief_valid[:, 1:] & relief_valid[:, :-1]
    neighbor_correlation = (
        float(
            np.corrcoef(
                output_relief[:, 1:][neighbor_pairs],
                output_relief[:, :-1][neighbor_pairs],
            )[0, 1]
        )
        if np.count_nonzero(neighbor_pairs) >= 16 else float("nan")
    )
    return {
        "direct_truth_coverage": coverage,
        "fine_relief_rms_ratio": output_rms / max(1e-12, truth_rms),
        "gradient_p95_ratio": float(
            np.percentile(grad_r, 95) / max(1e-12, np.percentile(grad_t, 95))
        ),
        "excursion_component_wasserstein_median_units": float(
            wasserstein_distance(areas_t, areas_r) / area_scale
        ),
        "effective_atom_count": effective_atoms,
        "max_atom_fraction": float(atom_prob.max()) if atom_prob.size else 1.0,
        "source_count": len(source_counts),
        "max_source_fraction": float(source_prob.max()) if source_prob.size else 1.0,
        "seam_step_ratio": seam_ratio,
        "axis_spectral_energy_fraction": axial_fraction,
        "spectral_all_supported_square_cells": spectral_square_cells,
        "fine_relief_neighbor_correlation": neighbor_correlation,
        "assigned_patch_count": int(len(selected)),
        "gradient_support_cells": int(np.count_nonzero(gradient_valid)),
        "relief_support_cells": int(np.count_nonzero(relief_valid)),
    }


def gate_failures(metrics: dict[str, Any], gates: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for name, bounds in gates["numeric_ranges"].items():
        value = float(metrics[name])
        if not np.isfinite(value) or value < float(bounds[0]) or value > float(bounds[1]):
            failures.append(f"{name}={value!r} outside [{bounds[0]}, {bounds[1]}]")
    return failures


def _assignment_panels(
    reconstruction: Reconstruction,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    origins = reconstruction.patch_origin_low
    ys = sorted(set(int(value) for value in origins[:, 0]))
    xs = sorted(set(int(value) for value in origins[:, 1]))
    source_ids = sorted(set(reconstruction.atom_source))
    source_code = {source_id: index + 1 for index, source_id in enumerate(source_ids)}
    sources = np.full((len(ys), len(xs)), np.nan)
    coordinates = np.full((len(ys), len(xs)), np.nan)
    for row, atom_index in enumerate(reconstruction.atom_index):
        y0, x0 = (int(value) for value in origins[row])
        if atom_index < 0:
            continue
        source = reconstruction.atom_source[int(atom_index)]
        ay, ax = reconstruction.atom_origin_low[int(atom_index)]
        sources[ys.index(y0), xs.index(x0)] = source_code[source]
        coordinates[ys.index(y0), xs.index(x0)] = int(ay) * 16 + int(ax)
    return sources, coordinates, source_code


def write_visual_panel(
    path: Path,
    truth: SurfaceArrays,
    reconstruction: Reconstruction,
    title: str,
) -> None:
    """Write float-height and assignment diagnostics without packing or rendering."""
    from PIL import Image, ImageDraw

    target_support = _target_support(truth)
    common_supported = target_support & reconstruction.support
    masked_truth = np.where(common_supported, truth.elevation_m, np.nan)
    masked_output = np.where(common_supported, reconstruction.elevation_m, np.nan)
    truth_relief, truth_relief_valid = _relief(masked_truth, common_supported)
    output_relief, output_relief_valid = _relief(masked_output, common_supported)
    relief_valid = truth_relief_valid & output_relief_valid
    source_panel, coordinate_panel, source_code = _assignment_panels(reconstruction)
    source_legend = ",".join(
        f"{code}={source_id}" for source_id, code in source_code.items()
    )
    panels = (
        (masked_truth, "01 target height on common support", False),
        (masked_output, "02 reconstruction on common support", False),
        (np.where(common_supported, masked_output - masked_truth, np.nan), "03 reconstruction - target", True),
        (np.where(relief_valid, truth_relief, np.nan), "04 target fine relief (kernel-safe)", True),
        (np.where(relief_valid, output_relief, np.nan), "05 reconstructed fine relief", True),
        (np.where(common_supported, reconstruction.confidence, np.nan), "06 common-support confidence", False),
        (source_panel, f"07 selected source ({source_legend})", False),
        (coordinate_panel, "08 selected exemplar y*16+x", False),
    )
    tile = 320
    label_height = 34
    title_height = 44
    canvas = Image.new("RGB", (tile * 4, title_height + (tile + label_height) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 12), title, fill="black")
    for index, (array, label_text, diverging) in enumerate(panels):
        finite = array[np.isfinite(array)]
        if not finite.size:
            rgb = np.zeros((*array.shape, 3), dtype=np.uint8)
        elif diverging:
            limit = max(1e-9, float(np.percentile(np.abs(finite), 99)))
            normalized = np.clip(array / limit, -1.0, 1.0)
            rgb = np.stack(
                (
                    np.where(normalized >= 0, 255, 255 * (1 + normalized)),
                    255 * (1 - np.abs(normalized)),
                    np.where(normalized <= 0, 255, 255 * (1 - normalized)),
                ), axis=-1,
            )
        else:
            lo, hi = np.percentile(finite, [1, 99])
            normalized = np.clip((array - lo) / max(1e-9, hi - lo), 0.0, 1.0)
            rgb = np.stack(
                (255 * normalized, 255 * np.sqrt(normalized), 255 * (1 - normalized)),
                axis=-1,
            )
        rgb = np.where(np.isfinite(array)[..., None], rgb, 32)
        rgb = np.asarray(np.clip(rgb, 0, 255), dtype=np.uint8)
        panel = Image.fromarray(np.flipud(rgb)).resize(
            (tile, tile), Image.Resampling.NEAREST
        )
        x = (index % 4) * tile
        y = title_height + (index // 4) * (tile + label_height)
        canvas.paste(panel, (x, y))
        draw.text((x + 8, y + tile + 10), label_text, fill="black")
    canvas.save(path, format="PNG", optimize=True)
