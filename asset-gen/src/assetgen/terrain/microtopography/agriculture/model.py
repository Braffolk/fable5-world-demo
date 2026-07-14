"""Explicit condition contract for the cultivated agricultural R0 prototype."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CultivatedConditions:
    state: str
    condition_status: str
    fine_lod: int
    fine_cx: int
    fine_cz: int
    corrected_base_path: str
    corrected_base_lod0_cx: int
    corrected_base_lod0_cz: int
    corrected_base_sha256: str
    rgb_orthophoto_path: str
    rgb_orthophoto_sha256: str
    cir_orthophoto_path: str
    cir_orthophoto_sha256: str
    parcel_etak_id: int
    parcel_boundary_clearance_m: float
    soil_distribution: dict[str, float]
    substrate_evidence_status: str
    hard_mask_intersections: dict[str, bool]
    size_m: float
    texel_m: float
    parcel_anchor_e_m: float
    parcel_anchor_n_m: float
    row_heading_deg_east_of_grid_north: float
    row_spacing_m: float
    row_relief_m: float
    track_pass_spacing_m: float
    track_gauge_m: float
    track_curvature_per_m: float
    track_width_m: float
    track_depth_m: float
    tread_spacing_m: float
    clod_cell_m: float
    clod_occupancy: float
    clod_diameter_min_m: float
    clod_diameter_max_m: float
    clod_height_min_m: float
    clod_height_max_m: float
    rainfall_event_mm: float
    runoff_coefficient: float
    track_runoff_multiplier: float
    flow_context_halo_m: float
    rill_contributing_area_m2: float
    rill_depth_max_m: float
    rill_width_min_m: float
    rill_width_max_m: float
    sediment_detachment_fraction: float
    sediment_deposition_fraction: float
    sediment_deposition_max_m: float

    def __post_init__(self) -> None:
        if self.state != "orientation_observed_state_unknown":
            raise ValueError(
                "R0 owns only observed row orientation; actual management state is unknown"
            )
        if self.condition_status != "orientation_observed_forms_development_hypothesis":
            raise ValueError("R0 must distinguish observed orientation from assumed forms")
        numeric = np.asarray(
            [value for value in asdict(self).values() if isinstance(value, (int, float))],
            dtype=np.float64,
        )
        if not np.isfinite(numeric).all():
            raise ValueError("agriculture conditions contain a nonfinite value")
        if self.fine_lod != -2 or self.size_m != 128.0 or self.texel_m != 0.0625:
            raise ValueError("R0 is bounded to one 128 m tile at 0.0625 m")
        if self.size_m / self.texel_m != 2048:
            raise ValueError("tile extent must contain exactly 2048 intervals")
        if self.row_spacing_m < 0.2 or self.tread_spacing_m < 0.125:
            raise ValueError("oriented forms may not fall below the 0.125 m Nyquist wavelength")
        if self.clod_diameter_min_m < 0.125:
            raise ValueError("clods below the 0.125 m resolved tail are forbidden")
        if not self.clod_diameter_min_m <= self.clod_diameter_max_m:
            raise ValueError("invalid clod diameter interval")
        if not 0.0 <= self.clod_occupancy <= 1.0:
            raise ValueError("clod occupancy must be in [0, 1]")
        if self.parcel_boundary_clearance_m < self.size_m * 0.49:
            raise ValueError("R0 tile is not sufficiently interior to its observed parcel")
        if set(self.soil_distribution) != {"LP", "E2I"}:
            raise ValueError("R0 must preserve the full observed LP/E2I soil distribution")
        if abs(sum(self.soil_distribution.values()) - 1.0) > 1e-6:
            raise ValueError("soil fractions must sum to one")
        if self.substrate_evidence_status != "raw_1_200k_codes_uninterpreted_not_guessed":
            raise ValueError("low-authority substrate codes may not be guessed")
        expected_masks = {"water", "building", "paved_road", "forest"}
        if set(self.hard_mask_intersections) != expected_masks:
            raise ValueError("hard-mask evidence fields differ from the frozen R0 contract")
        if any(self.hard_mask_intersections.values()):
            raise ValueError("this bounded R0 tile was selected to have no hard-mask overlap")
        for digest in (
            self.corrected_base_sha256,
            self.rgb_orthophoto_sha256,
            self.cir_orthophoto_sha256,
        ):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("source SHA-256 is malformed")
        if min(
            self.row_relief_m,
            self.track_pass_spacing_m,
            self.track_gauge_m,
            self.track_width_m,
            self.track_depth_m,
            self.clod_cell_m,
            self.clod_height_min_m,
            self.rainfall_event_mm,
            self.flow_context_halo_m,
            self.rill_contributing_area_m2,
            self.rill_depth_max_m,
            self.rill_width_min_m,
            self.sediment_deposition_max_m,
        ) <= 0.0:
            raise ValueError("positive agriculture condition is non-positive")
        if self.rill_width_min_m > self.rill_width_max_m:
            raise ValueError("invalid rill width interval")
        if not 0.0 <= self.runoff_coefficient <= 1.0:
            raise ValueError("runoff coefficient must be in [0, 1]")
        if not 0.0 <= self.sediment_detachment_fraction <= 1.0:
            raise ValueError("sediment detachment fraction must be in [0, 1]")
        if not 0.0 <= self.sediment_deposition_fraction <= 1.0:
            raise ValueError("sediment deposition fraction must be in [0, 1]")

    @property
    def side(self) -> int:
        return int(round(self.size_m / self.texel_m)) + 1

    def canonical(self) -> dict[str, object]:
        return asdict(self)


def load_conditions(path: Path) -> CultivatedConditions:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != "laas.agriculture-cultivated-r0.conditions/1":
        raise ValueError("unsupported agriculture condition schema")
    conditions = payload.get("conditions")
    if not isinstance(conditions, dict):
        raise ValueError("conditions must be a JSON object")
    expected = set(CultivatedConditions.__dataclass_fields__)
    supplied = set(conditions)
    if supplied != expected:
        raise ValueError(
            f"condition fields differ; missing={sorted(expected - supplied)}, "
            f"extra={sorted(supplied - expected)}"
        )
    return CultivatedConditions(**conditions)
