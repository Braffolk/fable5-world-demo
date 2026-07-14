"""Typed scientific contracts for the forest weak-evidence gate."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


RADII_M = (0.0625, 0.125, 0.25, 0.5, 1.0)
FEATURE_NAMES = tuple(
    f"{name}_r{radius:g}m"
    for radius in RADII_M
    for name in (
        "linearity",
        "planarity",
        "scattering",
        "normal_verticality",
        "support_count",
        "nearest_spacing",
    )
) + ("height_above_lower_envelope_plane",)


@dataclass(frozen=True)
class SemanticMethod:
    learning_rate: float = 0.05
    max_iter: int = 300
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 64
    l2_regularization: float = 1.0
    random_state: int = 20260714
    bootstrap_resamples: int = 10_000
    forbidden_ratio_max: float = 0.02
    block_m: float = 8.0
    descriptor_cell_m: float = 0.03125


@dataclass(frozen=True)
class SemanticResult:
    threshold: float | None
    audit_passed: bool
    train_feature_envelope: np.ndarray
    model_path: Path
    calibration: dict
    audits: tuple[dict, dict]


@dataclass(frozen=True)
class ResearchSurfaceEvidence:
    tier: Literal["R0_raw", "R1_weak_surface", "R2_research_audit"]
    source_id: str
    cell_m: float
    origin_xy_m: tuple[float, float]
    source_index: np.ndarray
    height_m: np.ndarray
    direct_observed: np.ndarray
    direct_support_distance_m: np.ndarray
    footprint_m: np.ndarray
    view_count: np.ndarray
    interpolation_distance_m: np.ndarray
    semantic_probabilities: np.ndarray
    semantic_probability_available: np.ndarray
    p_semantic_subclass_unknown: np.ndarray
    heightfield_valid_probability: np.ndarray
    provisional_surface_probability: np.ndarray
    p_unknown: np.ndarray
    dynamic_water_probability: np.ndarray
    forbidden: np.ndarray
    group_a_support: np.ndarray
    group_b_support: np.ndarray
    physical_view_count_available: np.ndarray
    redundancy_group_count: np.ndarray
    redundancy_group_count_available: np.ndarray
    disagreement_m: np.ndarray
    effective_support_radius_m: np.ndarray
    epistemic_uncertainty_m: np.ndarray
    repeatability_uncertainty_m: np.ndarray
    joint_direct_support: np.ndarray
    unique_geometry_support: np.ndarray
    band_a_m: np.ndarray
    band_b_m: np.ndarray
    local_disagreement_energy_m2: np.ndarray
    local_confidence: np.ndarray
    view_confidence: np.ndarray
    weak_geometry_weight: np.ndarray
    area_confidence: float
    cross_signal_m2: float
    cross_noise_m2: float
    band_confidence: float
    b1_eligible: np.ndarray
    b2_eligible: np.ndarray
    partition: np.ndarray
    unknown_reason: np.ndarray

    def validate(self) -> None:
        shape = self.height_m.shape
        arrays = (
            self.source_index,
            self.direct_observed,
            self.direct_support_distance_m,
            self.footprint_m,
            self.view_count,
            self.interpolation_distance_m,
            self.p_semantic_subclass_unknown,
            self.heightfield_valid_probability,
            self.provisional_surface_probability,
            self.p_unknown,
            self.dynamic_water_probability,
            self.forbidden,
            self.group_a_support,
            self.group_b_support,
            self.physical_view_count_available,
            self.redundancy_group_count,
            self.redundancy_group_count_available,
            self.disagreement_m,
            self.effective_support_radius_m,
            self.epistemic_uncertainty_m,
            self.repeatability_uncertainty_m,
            self.joint_direct_support,
            self.unique_geometry_support,
            self.band_a_m,
            self.band_b_m,
            self.local_disagreement_energy_m2,
            self.local_confidence,
            self.view_confidence,
            self.weak_geometry_weight,
            self.b1_eligible,
            self.b2_eligible,
            self.partition,
            self.unknown_reason,
        )
        if self.height_m.ndim != 2 or any(value.shape != shape for value in arrays):
            raise ValueError(f"research surface fields disagree for {self.source_id}")
        if self.semantic_probabilities.shape != (*shape, 6):
            raise ValueError("research surface requires six separate semantic probabilities")
        if self.semantic_probability_available.shape != (*shape, 6):
            raise ValueError("semantic availability must accompany every probability channel")
        if np.any(~self.semantic_probability_available & np.isfinite(self.semantic_probabilities)):
            raise ValueError("unavailable semantic probabilities must be explicit NaN")
        if np.any(~self.physical_view_count_available & np.isfinite(self.view_count)):
            raise ValueError("unknown physical view counts must be explicit NaN")
        if np.any(
            ~self.physical_view_count_available
            & (np.isfinite(self.group_a_support) | np.isfinite(self.group_b_support))
        ):
            raise ValueError("unknown physical view-group support must be explicit NaN")
        finite_height = np.isfinite(self.height_m)
        if np.any(finite_height != self.direct_observed):
            raise ValueError("finite weak-surface height must equal direct support")
        if np.any(
            (self.b1_eligible | self.b2_eligible)
            & (self.forbidden | (self.weak_geometry_weight <= 0))
        ):
            raise ValueError("zero-weight or forbidden cells cannot be research-trainable")
        if np.any((self.weak_geometry_weight < 0) | (self.weak_geometry_weight > 1)):
            raise ValueError("weak geometry weight must be a fraction")
        if not np.allclose(self.p_unknown, 1.0 - self.weak_geometry_weight):
            raise ValueError("p_unknown must be the complement of weak geometry weight")
