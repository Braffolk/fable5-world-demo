"""Honest evaluation of every frozen v3 gate. No tuning; report exactly what happens."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import network


def _orientation_resultant(ridge_mask: np.ndarray, region: np.ndarray) -> float:
    """Global orientation coherence R in [0,1]; ~1 = mire-wide aligned (bad)."""
    field = ndimage.gaussian_filter(ridge_mask.astype(np.float64), 2.0)
    gy, gx = np.gradient(field)
    # String tangent angle = gradient normal rotated 90 deg; use doubled angle.
    angle = np.arctan2(gy, gx) + np.pi / 2.0
    weight = ridge_mask & region
    if weight.sum() < 8:
        return 0.0
    vec = np.exp(2.0j * angle[weight])
    return float(np.abs(np.mean(vec)))


def evaluate(
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
    thresholds: dict,
) -> dict:
    ridge_mask = proc_labels == network.RIDGE
    patterned = proc_region & (proc_labels != network.LAWN)
    patterned_area_km2 = float(patterned.sum()) * process_pitch**2 / 1.0e6

    spacing = topology.spacing_samples * process_pitch
    spacing_cv = float(spacing.std() / spacing.mean()) if spacing.size > 1 else 0.0
    orientation_r = _orientation_resultant(ridge_mask, proc_region)

    junctions_per_km2 = topology.junction_count / patterned_area_km2 if patterned_area_km2 else 0.0
    terms_per_km2 = topology.termination_count / patterned_area_km2 if patterned_area_km2 else 0.0

    coupling, coupled_n, perim_n = network.pool_coupling(
        proc_labels, radius_cells=float(thresholds["pool_coupling_radius_m"]) / process_pitch
    )

    # Parent deviation diagnostic.
    dev = np.abs(deviation_1m)
    form_pattern = authority_1m & (dev > 0.005)
    form_free = authority_1m & ~form_pattern
    dev_unpat_p95 = float(np.percentile(dev[form_free], 95.0)) if form_free.any() else 0.0
    dev_pat_p99 = float(np.percentile(dev[form_pattern], 99.0)) if form_pattern.any() else 0.0
    dev_p50 = float(np.percentile(dev[authority_1m], 50.0)) if authority_1m.any() else 0.0
    dev_max = float(dev[authority_1m].max()) if authority_1m.any() else 0.0

    # Amplitude: 2-16 m band RMS of the carved relief (0.25 m), and core |relief| p99.
    r = relief_025m
    band = ndimage.gaussian_filter(r, 2.0 / 0.25) - ndimage.gaussian_filter(r, 16.0 / 0.25)
    nz = r != 0.0
    band_rms = float(np.sqrt(np.mean(band[nz] ** 2))) if nz.any() else 0.0
    relief_p99 = float(np.percentile(np.abs(r[nz]), 99.0)) if nz.any() else 0.0

    g = thresholds
    gates = {
        "anti_corrugation": {
            "spacing_cv": spacing_cv,
            "spacing_cv_min": g["string_spacing_cv_min"],
            "spacing_mean_m": float(spacing.mean()) if spacing.size else 0.0,
            "orientation_resultant_length": orientation_r,
            "orientation_resultant_length_max": g["global_orientation_resultant_length_max"],
            "passed": spacing_cv >= g["string_spacing_cv_min"]
            and orientation_r <= g["global_orientation_resultant_length_max"],
        },
        "topology": {
            "patterned_area_km2": patterned_area_km2,
            "junction_count": topology.junction_count,
            "termination_count": topology.termination_count,
            "branch_count": topology.branch_count,
            "merge_count": topology.merge_count,
            "junctions_per_km2": junctions_per_km2,
            "terminations_per_km2": terms_per_km2,
            "junctions_per_km2_min": g["branch_merge_junctions_per_km2_min"],
            "terminations_per_km2_min": g["terminations_per_km2_min"],
            "passed": (
                junctions_per_km2 >= g["branch_merge_junctions_per_km2_min"]
                and terms_per_km2 >= g["terminations_per_km2_min"]
                and topology.branch_count >= g["branch_min_count"]
                and topology.merge_count >= g["merge_min_count"]
            ),
        },
        "pool_coupling": {
            "coupling_fraction": coupling,
            "coupled_perimeter_cells": coupled_n,
            "pool_perimeter_cells": perim_n,
            "coupling_min": g["pool_perimeter_adjacent_fraction_min"],
            "passed": coupling >= g["pool_perimeter_adjacent_fraction_min"],
        },
        "parent_deviation": {
            "unpatterned_p95_m": dev_unpat_p95,
            "unpatterned_p95_max_m": g["unpatterned_form_free_p95_max_m"],
            "patterned_p99_m": dev_pat_p99,
            "patterned_p99_max_m": g["patterned_area_p99_max_m"],
            "authority_p50_m": dev_p50,
            "authority_max_m": dev_max,
            "patterned_area_fraction": float(form_pattern.sum() / max(authority_1m.sum(), 1)),
            "passed": dev_unpat_p95 <= g["unpatterned_form_free_p95_max_m"]
            and dev_pat_p99 <= g["patterned_area_p99_max_m"],
        },
        "amplitude_envelope": {
            "band_2_to_16m_rms_m": band_rms,
            "band_rms_min_m": g["b1_band_2_to_16m_rms_min_m"],
            "band_rms_max_m": g["b1_band_2_to_16m_rms_max_m"],
            "core_abs_relief_p99_m": relief_p99,
            "core_abs_relief_p99_max_m": g["core_abs_relief_p99_max_m"],
            "passed": g["b1_band_2_to_16m_rms_min_m"] <= band_rms <= g["b1_band_2_to_16m_rms_max_m"]
            and relief_p99 <= g["core_abs_relief_p99_max_m"],
        },
        "safety_exactness": {
            "hard_residual_cells": hard_residual_cells,
            "open_water_residual_cells": open_water_residual_cells,
            "pool_residual_cells": pool_residual_cells,
            "crop_identity_exact": crop_identity_exact,
            "sealed_pixels_opened": sealed_pixels_opened,
            "passed": hard_residual_cells == 0
            and open_water_residual_cells == 0
            and pool_residual_cells == 0
            and crop_identity_exact
            and sealed_pixels_opened == 0,
        },
    }
    gates["all_passed"] = all(row["passed"] for row in gates.values())
    return gates
