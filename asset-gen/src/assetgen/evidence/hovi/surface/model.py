"""Frozen contracts for the Hovi merged-cloud candidate-surface extractor."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntFlag
from typing import Any

import numpy as np


class Reason(IntFlag):
    DIRECT_VERTICAL_MODE = 1 << 0
    BROAD_TANGENT_WINDOW = 1 << 1
    NEIGHBOR_CONSISTENT = 1 << 2
    AMBIGUOUS_COMPETITOR = 1 << 3
    ABOVE_SHEET_COMPETITOR = 1 << 4
    UNSUPPORTED = 1 << 5
    INTERPOLATED_SMALL_HOLE = 1 << 6
    LOW_SELECTION_SCORE = 1 << 7
    OUTSIDE_ANALYSIS_CORE = 1 << 8
    SUBSTANTIAL_UNDERBURDEN = 1 << 9
    INSUFFICIENT_XY_FOOTPRINT = 1 << 10
    INVALID_TANGENT_REFERENCE = 1 << 11
    DISCONNECTED_DIRECT_CANDIDATE = 1 << 12


@dataclass(frozen=True)
class SurfaceConfig:
    context_resolution_m: float = 1.0
    structural_resolution_m: float = 0.25
    final_resolution_m: float = 0.0625
    structural_z_bin_m: float = 0.02
    final_z_bin_m: float = 0.01
    final_half_window_m: float = 0.60
    mode_count: int = 5
    structural_mode_radius_bins: int = 2
    final_mode_radius_bins: int = 2
    context_min_mode_support: int = 120
    structural_min_mode_support: int = 24
    final_min_mode_support: int = 6
    context_underburden_clearance_m: float = 0.30
    structural_underburden_clearance_m: float = 0.10
    final_underburden_clearance_m: float = 0.06
    context_neighbor_step_m: float = 0.45
    structural_neighbor_step_m: float = 0.18
    final_neighbor_step_m: float = 0.10
    structural_parent_tolerance_m: float = 0.50
    final_parent_tolerance_m: float = 0.42
    ambiguity_quality_ratio: float = 0.72
    ambiguity_min_separation_m: float = 0.035
    max_selected_underburden_fraction: float = 0.28
    min_direct_selection_score: float = 0.55
    minimum_footprint_quadrants: int = 3
    minimum_footprint_subcells: int = 6
    footprint_subcell_axis: int = 4
    footprint_registration_allowance_m: float = 0.006
    tangent_radius_cells: int = 2
    tangent_minimum_cells: int = 6
    context_tangent_max_residual_m: float = 0.08
    structural_tangent_max_residual_m: float = 0.04
    tangent_max_slope: float = 2.5
    analysis_core_erosion_m: float = 1.0
    topology_minimum_component_cells: int = 8
    topology_structural_anchor_residual_m: float = 0.12
    topology_minimum_anchor_fraction: float = 0.50
    max_interpolation_component_cells: int = 4
    interpolation_ring_min_cells: int = 8
    interpolation_plane_max_residual_m: float = 0.025
    minimum_context_component_coverage: float = 0.60
    minimum_analysis_direct_coverage: float = 0.55
    maximum_analysis_ambiguity_fraction: float = 0.35
    histogram_chunk_points: int = 2_000_000
    histogram_memory_limit_bytes: int = 1 << 30

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GridSpec:
    resolution_m: float
    origin_x_m: float
    origin_y_m: float
    width: int
    height: int

    @property
    def cell_count(self) -> int:
        return self.width * self.height

    @property
    def upper_x_m(self) -> float:
        return self.origin_x_m + self.width * self.resolution_m

    @property
    def upper_y_m(self) -> float:
        return self.origin_y_m + self.height * self.resolution_m

    def as_record(self) -> dict[str, Any]:
        return {
            "resolution_m": self.resolution_m,
            "origin_xy_m": [self.origin_x_m, self.origin_y_m],
            "upper_exclusive_xy_m": [self.upper_x_m, self.upper_y_m],
            "shape_yx": [self.height, self.width],
        }


@dataclass(frozen=True)
class ModeEvidence:
    height_m: np.ndarray
    support_count: np.ndarray
    prominence: np.ndarray
    underburden_fraction: np.ndarray
    quality: np.ndarray


@dataclass(frozen=True)
class TangentPrediction:
    child_center_z_m: np.ndarray
    child_slope_x: np.ndarray
    child_slope_y: np.ndarray
    child_valid_mask: np.ndarray
    parent_slope_x: np.ndarray
    parent_slope_y: np.ndarray
    parent_valid_mask: np.ndarray
    parent_fit_residual_m: np.ndarray


@dataclass(frozen=True)
class SheetSelection:
    selected_height_m: np.ndarray
    selected_mode_index: np.ndarray
    selection_score: np.ndarray
    direct_mask: np.ndarray
    ambiguity_mask: np.ndarray
    above_sheet_competitor_mask: np.ndarray
    unsupported_mask: np.ndarray
    neighbor_consistent_mask: np.ndarray
    broad_window_consistent_mask: np.ndarray
    selected_support_count: np.ndarray
    selected_underburden_fraction: np.ndarray
    competing_mode_ratio: np.ndarray
    component_diagnostics: dict[str, Any] | None = None


@dataclass(frozen=True)
class SurfaceCandidate:
    grid: GridSpec
    absolute_local_z_m: np.ndarray
    mode_candidate_z_m: np.ndarray
    selection_score: np.ndarray
    reason_bits: np.ndarray
    direct_mask: np.ndarray
    inferred_mask: np.ndarray
    ambiguity_mask: np.ndarray
    above_sheet_competitor_mask: np.ndarray
    unsupported_mask: np.ndarray
    interpolation_mask: np.ndarray
    analysis_core_mask: np.ndarray
    nominal_layout_mask: np.ndarray
    observed_fine_mask: np.ndarray
    tangent_window_observed_mask: np.ndarray
    tangent_reference_valid_mask: np.ndarray
    pre_erosion_domain_mask: np.ndarray
    topology_rejected_mask: np.ndarray
    selected_support_count: np.ndarray
    selected_underburden_fraction: np.ndarray
    competing_mode_ratio: np.ndarray
    footprint_quadrant_count: np.ndarray
    footprint_subcell_count: np.ndarray
    structural_reference_z_m: np.ndarray
