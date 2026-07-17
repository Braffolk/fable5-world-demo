"""v4 gate evaluation: v3.0 carried-over gates + the three v3.1 mire-scale gates.

The carried-over gates (safety_exactness, parent_deviation, amplitude_envelope, topology,
pool_coupling) are evaluated EXACTLY as v3.0 via ``gates.evaluate`` on the output core+halo
window; their thresholds are unchanged. The v3.0 window ``anti_corrugation`` gate is
REPLACED by the three mire-scale gates v3.1 froze (128 m blocks over the whole mire, using
the unmodified orientation/skeleton/spacing machinery from ``gates`` and ``network``).
Thresholds are read from config; no tuning. Every number is reported honestly.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import gates, network


def _doubled_angle_vectors(ridge_mask: np.ndarray) -> np.ndarray:
    """Per-cell exp(2j*theta) of the string tangent, same construction as
    :func:`gates._orientation_resultant` (gaussian-smoothed ridge field gradient normal).
    """
    field = ndimage.gaussian_filter(ridge_mask.astype(np.float64), 2.0)
    gy, gx = np.gradient(field)
    angle = np.arctan2(gy, gx) + np.pi / 2.0
    return np.exp(2.0j * angle)


def _spacing_peaks(skeleton: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance-transform spacing peaks: return (peak_mask, 2*distance) — same estimator
    as :func:`network.analyze_topology` but keeping peak LOCATIONS."""
    distance = ndimage.distance_transform_edt(~skeleton)
    local_max = ndimage.maximum_filter(distance, size=3)
    peaks = (distance == local_max) & (distance > 0.5)
    return peaks, 2.0 * distance


def compute_mire_scale_gates(
    whole_labels: np.ndarray,
    whole_region: np.ndarray,
    process_pitch: float,
    cfg: dict,
) -> tuple[dict, dict]:
    """The three v3.1 mire-scale anti-corrugation gates on the whole-mire label field.

    ``whole_region`` is the whole-mire authority (mire minus hard) on the process grid;
    ridges outside it are not counted. All thresholds come from ``cfg``. Returns the
    serializable gate dict AND a companion dict of intermediate fields (skeleton,
    spacing_field, patterned_cell, per-block spacing) for QA (no second skeletonize).
    """
    block = int(cfg["block_size_cells"])
    min_ridge = int(cfg["min_ridge_cells"])
    r_dom_max = float(cfg["block_orientation_resultant_R_dom_max"])
    spacing_cv_min = float(cfg["pooled_spacing_cv_min"])
    r_split = float(cfg["regime_split_R_int"])
    min_blocks = int(cfg["min_blocks_per_regime"])
    min_frac = float(cfg["min_fraction_per_regime"])

    ridge_mask = (whole_labels == network.RIDGE) & whole_region
    rows, cols = ridge_mask.shape
    nby, nbx = rows // block, cols // block  # discard high-edge remainder strip

    vectors = _doubled_angle_vectors(ridge_mask)

    theta_unit: list[complex] = []  # unit vector exp(2j*theta_b) per patterned block
    r_int_values: list[float] = []
    patterned_cell = np.zeros(ridge_mask.shape, dtype=bool)
    patterned_blocks = 0
    for by in range(nby):
        for bx in range(nbx):
            r0, c0 = by * block, bx * block
            sub = ridge_mask[r0 : r0 + block, c0 : c0 + block]
            n = int(sub.sum())
            if n < min_ridge:
                continue
            patterned_blocks += 1
            patterned_cell[r0 : r0 + block, c0 : c0 + block] = True
            vsub = vectors[r0 : r0 + block, c0 : c0 + block][sub]
            if vsub.size >= 8:
                mean_vec = complex(np.mean(vsub))
                r_int = abs(mean_vec)
                r_int_values.append(r_int)
                if r_int > 0.0:
                    theta_unit.append(mean_vec / r_int)
            else:
                r_int_values.append(0.0)

    # gate A: dispersion of block orientations.
    r_dom = float(abs(np.mean(theta_unit))) if theta_unit else 1.0

    # gate B: pooled spacing CV over peaks that fall inside a patterned block.
    skeleton = network.skeletonize(ridge_mask)
    peaks, spacing_field = _spacing_peaks(skeleton)
    keep = peaks & patterned_cell
    spacing = spacing_field[keep] * process_pitch
    spacing_cv = float(spacing.std() / spacing.mean()) if spacing.size > 1 else 0.0

    # Per-128 m-block mean measured spacing map (QA "scale bands"; shows variation).
    block_spacing = np.full((nby, nbx), np.nan, dtype=np.float64)
    for by in range(nby):
        for bx in range(nbx):
            r0, c0 = by * block, bx * block
            bk = keep[r0 : r0 + block, c0 : c0 + block]
            if bk.any():
                block_spacing[by, bx] = float(
                    np.mean(spacing_field[r0 : r0 + block, c0 : c0 + block][bk]) * process_pitch
                )

    # gate C: two regimes present.
    r_int_arr = np.asarray(r_int_values, dtype=np.float64)
    labyrinth = int((r_int_arr < r_split).sum())
    aligned = int((r_int_arr >= r_split).sum())
    total = max(patterned_blocks, 1)
    lab_ok = labyrinth >= min_blocks and labyrinth / total >= min_frac
    ali_ok = aligned >= min_blocks and aligned / total >= min_frac

    fields = {
        "skeleton": skeleton,
        "spacing_field": spacing_field,
        "peaks_kept": keep,
        "patterned_cell": patterned_cell,
        "block_spacing": block_spacing,
        "block_size": block,
    }
    report = {
        "block_grid": {
            "block_size_cells": block,
            "block_size_m": block * process_pitch,
            "blocks_scanned": nby * nbx,
            "patterned_blocks": patterned_blocks,
            "min_ridge_cells": min_ridge,
        },
        "gate_a_orientation_domain_dispersion": {
            "R_dom": r_dom,
            "R_dom_max": r_dom_max,
            "circular_variance": 1.0 - r_dom,
            "passed": r_dom <= r_dom_max,
        },
        "gate_b_pooled_spacing_cv": {
            "pooled_spacing_cv": spacing_cv,
            "pooled_spacing_cv_min": spacing_cv_min,
            "spacing_mean_m": float(spacing.mean()) if spacing.size else 0.0,
            "spacing_std_m": float(spacing.std()) if spacing.size else 0.0,
            "spacing_samples": int(spacing.size),
            "passed": spacing_cv >= spacing_cv_min,
        },
        "gate_c_two_regimes_present": {
            "labyrinth_blocks": labyrinth,
            "aligned_blocks": aligned,
            "patterned_blocks": patterned_blocks,
            "regime_split_R_int": r_split,
            "min_blocks_per_regime": min_blocks,
            "min_fraction_per_regime": min_frac,
            "labyrinth_fraction": labyrinth / total,
            "aligned_fraction": aligned / total,
            "passed": bool(lab_ok and ali_ok),
        },
    }
    return report, fields


def evaluate_v4(
    *,
    proc_labels: np.ndarray,
    proc_region: np.ndarray,
    topology: network.Topology,
    process_pitch: float,
    deviation_1m: np.ndarray,
    authority_1m: np.ndarray,
    relief_025m: np.ndarray,
    hard_residual_cells: int,
    open_water_residual_cells: int,
    pool_residual_cells: int,
    crop_identity_exact: bool,
    sealed_pixels_opened: int,
    window_thresholds: dict,
    whole_labels: np.ndarray,
    whole_region: np.ndarray,
    mire_scale_cfg: dict,
) -> tuple[dict, dict]:
    """Full v4 gate report + QA field companion. ``all_passed`` requires the five
    carried-over v3.0 gates (evaluated on the window, EXCLUDING the retired window
    anti_corrugation) AND the three mire-scale gates to pass."""
    window = gates.evaluate(
        proc_labels=proc_labels,
        proc_region=proc_region,
        topology=topology,
        process_pitch=process_pitch,
        deviation_1m=deviation_1m,
        authority_1m=authority_1m,
        relief_025m=relief_025m,
        hard_residual_cells=hard_residual_cells,
        open_water_residual_cells=open_water_residual_cells,
        pool_residual_cells=pool_residual_cells,
        crop_identity_exact=crop_identity_exact,
        sealed_pixels_opened=sealed_pixels_opened,
        thresholds=window_thresholds,
    )
    # The v3.0 window anti_corrugation gate is RETIRED for v4 (replaced by mire-scale
    # gates); retain its numbers as a window diagnostic but drop it from the pass set.
    window_anti_corrugation = window.pop("anti_corrugation")
    window.pop("all_passed", None)

    mire_scale, fields = compute_mire_scale_gates(
        whole_labels, whole_region, process_pitch, mire_scale_cfg
    )

    carried_passed = all(row["passed"] for row in window.values())
    mire_passed = (
        mire_scale["gate_a_orientation_domain_dispersion"]["passed"]
        and mire_scale["gate_b_pooled_spacing_cv"]["passed"]
        and mire_scale["gate_c_two_regimes_present"]["passed"]
    )
    report = {
        "carried_over_v3_window_gates": window,
        "window_anti_corrugation_diagnostic_retired": window_anti_corrugation,
        "mire_scale_anti_corrugation": mire_scale,
        "carried_over_passed": bool(carried_passed),
        "mire_scale_passed": bool(mire_passed),
        "all_passed": bool(carried_passed and mire_passed),
    }
    return report, fields
