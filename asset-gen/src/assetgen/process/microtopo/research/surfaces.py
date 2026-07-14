"""Fail-closed Hovi gate and unchanged Evo R2 surface adapters."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .bundle import FrozenBundle, sha256_file
from .contracts import ResearchSurfaceEvidence, SemanticResult


def _partition(
    shape: tuple[int, int],
    origin_xy_m: tuple[float, float],
    plot_origin_xy_m: tuple[float, float],
    cell_m: float,
    contract_id: str,
    *,
    analysis_halo_m: float,
    model_halo_m: float | None,
) -> np.ndarray:
    yy, xx = np.indices(shape)
    local_x = origin_xy_m[0] + (xx + 0.5) * cell_m - plot_origin_xy_m[0]
    local_y = origin_xy_m[1] + (yy + 0.5) * cell_m - plot_origin_xy_m[1]
    bx = np.floor(local_x / 8.0).astype(np.int64)
    by = np.floor(local_y / 8.0).astype(np.int64)
    out = np.empty(shape, dtype=np.uint8)
    for y in np.unique(by):
        for x in np.unique(bx):
            mask = (bx == x) & (by == y)
            if not np.any(mask):
                continue
            digest = hashlib.blake2b(
                f"{contract_id},{x},{y}".encode(), digest_size=32
            ).digest()
            value = digest[0] % 10
            out[mask] = 0 if value <= 5 else (1 if value <= 7 else 2)
    complete_halo_m = None if model_halo_m is None else max(analysis_halo_m, model_halo_m)
    if complete_halo_m is None:
        out[:] = 255
        return out
    x_local = np.mod(local_x, 8.0)
    y_local = np.mod(local_y, 8.0)
    halo_crosses = (
        (x_local < complete_halo_m)
        | (x_local > 8.0 - complete_halo_m)
        | (y_local < complete_halo_m)
        | (y_local > 8.0 - complete_halo_m)
    )
    out[halo_crosses] = 255
    return out


def failed_hovi_surface(bundle: FrozenBundle, semantic: SemanticResult) -> ResearchSurfaceEvidence:
    """Represent the prerequisite failure without inventing a selected Hovi sheet."""
    if semantic.audit_passed:
        raise ValueError("successful semantics require the disjoint-view reconstruction path")
    shape = (64, 64)
    nan = np.full(shape, np.nan, dtype=np.float32)
    zero = np.zeros(shape, dtype=np.float32)
    surface = ResearchSurfaceEvidence(
        tier="R0_raw",
        source_id="hovi.hyytiala.2019.HY_SPRUCE4",
        cell_m=0.0625,
        origin_xy_m=(-6.775, -6.1),
        source_index=np.zeros(shape, dtype=np.uint32),
        height_m=np.full(shape, np.nan, dtype=np.float64),
        direct_observed=np.zeros(shape, dtype=np.bool_),
        direct_support_distance_m=nan.copy(),
        footprint_m=nan.copy(),
        view_count=nan.copy(),
        interpolation_distance_m=nan.copy(),
        semantic_probabilities=np.full((*shape, 6), np.nan, dtype=np.float32),
        semantic_probability_available=np.zeros((*shape, 6), dtype=np.bool_),
        p_semantic_subclass_unknown=np.ones(shape, dtype=np.float32),
        heightfield_valid_probability=np.zeros(shape, dtype=np.float32),
        provisional_surface_probability=np.zeros(shape, dtype=np.float32),
        p_unknown=np.ones(shape, dtype=np.float32),
        dynamic_water_probability=zero.copy(),
        forbidden=np.zeros(shape, dtype=np.bool_),
        group_a_support=nan.copy(),
        group_b_support=nan.copy(),
        physical_view_count_available=np.zeros(shape, dtype=np.bool_),
        redundancy_group_count=np.zeros(shape, dtype=np.uint8),
        redundancy_group_count_available=np.zeros(shape, dtype=np.bool_),
        disagreement_m=np.full(shape, np.nan, dtype=np.float32),
        effective_support_radius_m=np.full(shape, np.nan, dtype=np.float32),
        epistemic_uncertainty_m=nan.copy(),
        repeatability_uncertainty_m=nan.copy(),
        joint_direct_support=np.zeros(shape, dtype=np.bool_),
        unique_geometry_support=np.zeros(shape, dtype=np.bool_),
        band_a_m=nan.copy(),
        band_b_m=nan.copy(),
        local_disagreement_energy_m2=nan.copy(),
        local_confidence=zero.copy(),
        view_confidence=zero.copy(),
        weak_geometry_weight=zero.copy(),
        area_confidence=0.0,
        cross_signal_m2=0.0,
        cross_noise_m2=0.0,
        band_confidence=0.0,
        b1_eligible=np.zeros(shape, dtype=np.bool_),
        b2_eligible=np.zeros(shape, dtype=np.bool_),
        partition=_partition(
            shape,
            (-6.775, -6.1),
            (-6.775, -6.1),
            0.0625,
            bundle.contract["contract_id"],
            analysis_halo_m=1.0,
            model_halo_m=None,
        ),
        unknown_reason=np.full(shape, 1, dtype=np.uint8),
    )
    surface.validate()
    return surface


def _evo_manifest(bundle: FrozenBundle, build_id: str) -> Path:
    return next(
        path
        for path, _ in bundle.inputs
        if f"evo/surface/sha256/{build_id}" in path.as_posix()
    )


def adapt_evo(bundle: FrozenBundle, build_id: str, plot: str) -> ResearchSurfaceEvidence:
    """Map the frozen Evo candidate fields without changing its estimator or authority."""
    manifest_path = _evo_manifest(bundle, build_id)
    manifest = json.loads(manifest_path.read_bytes())
    artifact_row = manifest["artifacts"]["candidate_npz"]
    path = manifest_path.parent / artifact_row["path"]
    if sha256_file(path) != artifact_row["sha256"]:
        raise ValueError(f"Evo {plot} candidate changed")
    with np.load(path) as source:
        observed = source["valid_observed"].astype(np.bool_)
        valid = observed & source["heightfield_valid"].astype(np.bool_)
        height = np.where(valid, source["height_m_f64"], np.nan)
        support = source["nearest_support_m"].astype(np.float32)
        disagreement = source["local_plane_residual_rms_m"].astype(np.float32)
        shape = height.shape
        surface = ResearchSurfaceEvidence(
            tier="R2_research_audit",
            source_id=f"evo.2024.{plot}",
            cell_m=0.0625,
            origin_xy_m=(0.0, 0.0),
            source_index=source["source_point_ids_or_reconstructable_index"].astype(np.uint32),
            height_m=height,
            direct_observed=valid,
            direct_support_distance_m=support,
            footprint_m=source["effective_footprint_m"].astype(np.float32),
            view_count=np.full(shape, np.nan, dtype=np.float32),
            interpolation_distance_m=source["interpolation_distance_m"].astype(np.float32),
            semantic_probabilities=np.full((*shape, 6), np.nan, dtype=np.float32),
            semantic_probability_available=np.zeros((*shape, 6), dtype=np.bool_),
            p_semantic_subclass_unknown=np.ones(shape, dtype=np.float32),
            heightfield_valid_probability=valid.astype(np.float32),
            provisional_surface_probability=np.zeros(shape, dtype=np.float32),
            p_unknown=np.ones(shape, dtype=np.float32),
            dynamic_water_probability=source["water_or_dynamic"].astype(np.float32),
            forbidden=source["water_or_dynamic"].astype(np.bool_) | source["human_invalid"].astype(np.bool_),
            group_a_support=np.full(shape, np.nan, dtype=np.float32),
            group_b_support=np.full(shape, np.nan, dtype=np.float32),
            physical_view_count_available=np.zeros(shape, dtype=np.bool_),
            redundancy_group_count=source["view_group_count"].astype(np.uint8),
            redundancy_group_count_available=observed.copy(),
            disagreement_m=disagreement,
            effective_support_radius_m=np.maximum(support, source["effective_footprint_m"]),
            epistemic_uncertainty_m=np.full(shape, np.nan, dtype=np.float32),
            repeatability_uncertainty_m=disagreement,
            joint_direct_support=np.zeros(shape, dtype=np.bool_),
            unique_geometry_support=np.zeros(shape, dtype=np.bool_),
            band_a_m=np.full(shape, np.nan, dtype=np.float32),
            band_b_m=np.full(shape, np.nan, dtype=np.float32),
            local_disagreement_energy_m2=np.full(shape, np.nan, dtype=np.float32),
            local_confidence=np.zeros(shape, dtype=np.float32),
            view_confidence=np.zeros(shape, dtype=np.float32),
            weak_geometry_weight=np.zeros(shape, dtype=np.float32),
            area_confidence=0.0,
            cross_signal_m2=0.0,
            cross_noise_m2=0.0,
            band_confidence=0.0,
            b1_eligible=np.zeros(shape, dtype=np.bool_),
            b2_eligible=np.zeros(shape, dtype=np.bool_),
            partition=_partition(
                shape,
                (0.0, 0.0),
                (0.0, 0.0),
                0.0625,
                bundle.contract["contract_id"],
                analysis_halo_m=1.0,
                model_halo_m=None,
            ),
            unknown_reason=np.full(shape, 2, dtype=np.uint8),
        )
    surface.validate()
    return surface


def save_surface(path: Path, surface: ResearchSurfaceEvidence) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as target:
        np.savez_compressed(
            target,
            height_m_f64=surface.height_m,
            source_index_u32=surface.source_index,
            direct_observed=surface.direct_observed,
            direct_support_distance_m_f32=surface.direct_support_distance_m,
            footprint_m_f32=surface.footprint_m,
            physical_view_count_f32=surface.view_count,
            physical_view_count_available=surface.physical_view_count_available,
            redundancy_group_count_u8=surface.redundancy_group_count,
            redundancy_group_count_available=surface.redundancy_group_count_available,
            interpolation_distance_m_f32=surface.interpolation_distance_m,
            semantic_probabilities_f32=surface.semantic_probabilities,
            semantic_probability_available=surface.semantic_probability_available,
            p_semantic_subclass_unknown_f32=surface.p_semantic_subclass_unknown,
            heightfield_valid_probability_f32=surface.heightfield_valid_probability,
            provisional_surface_probability_f32=surface.provisional_surface_probability,
            p_unknown_f32=surface.p_unknown,
            dynamic_water_probability_f32=surface.dynamic_water_probability,
            forbidden=surface.forbidden,
            group_a_support_f32=surface.group_a_support,
            group_b_support_f32=surface.group_b_support,
            disagreement_m_f32=surface.disagreement_m,
            effective_support_radius_m_f32=surface.effective_support_radius_m,
            epistemic_uncertainty_m_f32=surface.epistemic_uncertainty_m,
            repeatability_uncertainty_m_f32=surface.repeatability_uncertainty_m,
            M_C=surface.joint_direct_support,
            G_C=surface.unique_geometry_support,
            b_A_m_f32=surface.band_a_m,
            b_B_m_f32=surface.band_b_m,
            d_i_m2_f32=surface.local_disagreement_energy_m2,
            c_local_i_f32=surface.local_confidence,
            c_view_i_f32=surface.view_confidence,
            w_i_f32=surface.weak_geometry_weight,
            b1_eligible=surface.b1_eligible,
            b2_eligible=surface.b2_eligible,
            partition_u8=surface.partition,
            unknown_reason_u8=surface.unknown_reason,
            cell_m_f64=np.asarray(surface.cell_m),
            origin_xy_m_f64=np.asarray(surface.origin_xy_m),
        )
    return {
        "path": f"surfaces/{path.name}",
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "source_id": surface.source_id,
        "tier": surface.tier,
        "shape_yx": list(surface.height_m.shape),
        "direct_cells": int(surface.direct_observed.sum()),
        "b1_eligible_cells": int(surface.b1_eligible.sum()),
        "b2_eligible_cells": int(surface.b2_eligible.sum()),
        "c_area": surface.area_confidence,
        "S_C_m2": surface.cross_signal_m2,
        "N_C_m2": surface.cross_noise_m2,
        "c_band": surface.band_confidence,
        "weighted_effective_area_m2": float(surface.weak_geometry_weight.sum() * surface.cell_m**2),
        "p_semantic_subclass_unknown": 1.0,
        "total_surface_error": None,
    }
