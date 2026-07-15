"""Strict physical inputs and configuration for the erodible-slope solver."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ProcessConfig:
    schema_version: str
    process_texel_m: float
    fine_texel_m: float
    rainfall_depth_m: float
    event_duration_s: float
    manning_n_s_m13: float
    water_density_kg_m3: float
    gravity_m_s2: float
    transport_kg_per_joule: float
    deposit_bulk_density_kg_m3: float
    rill_area_threshold_m2: float
    rill_width_min_m: float
    rill_width_max_m: float
    rill_depth_max_m: float
    headcut_relief_max_m: float
    headcut_slope_break_min: float
    seep_relief_max_m: float
    toe_relief_max_m: float
    streamline_step_m: float
    streamline_max_length_m: float
    min_slope: float
    deposition_slope_ceiling: float
    mass_balance_relative_tolerance: float
    seed: int
    material_rules: dict[str, dict[str, float]]

    def __post_init__(self) -> None:
        if self.schema_version != "laas.erodible-slope-process-config/1":
            raise ValueError("unsupported erodible-slope process config")
        numeric = np.asarray(
            [
                self.process_texel_m,
                self.fine_texel_m,
                self.rainfall_depth_m,
                self.event_duration_s,
                self.manning_n_s_m13,
                self.water_density_kg_m3,
                self.gravity_m_s2,
                self.transport_kg_per_joule,
                self.deposit_bulk_density_kg_m3,
                self.rill_area_threshold_m2,
                self.rill_width_min_m,
                self.rill_width_max_m,
                self.rill_depth_max_m,
                self.headcut_relief_max_m,
                self.headcut_slope_break_min,
                self.seep_relief_max_m,
                self.toe_relief_max_m,
                self.streamline_step_m,
                self.streamline_max_length_m,
                self.min_slope,
                self.deposition_slope_ceiling,
                self.mass_balance_relative_tolerance,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(numeric).all() or np.any(numeric <= 0.0):
            raise ValueError("physical process coefficients must be finite and positive")
        if self.process_texel_m < self.fine_texel_m:
            raise ValueError("process grid may not be finer than the absolute master")
        if self.rill_width_min_m > self.rill_width_max_m:
            raise ValueError("rill width interval is reversed")
        if self.streamline_step_m > self.process_texel_m:
            raise ValueError("streamline integration step exceeds process texel")
        if self.deposition_slope_ceiling >= 1.0:
            raise ValueError("deposition slope ceiling must be a dimensionless gradient")
        required = {
            "bulk_density_kg_m3",
            "critical_shear_pa",
            "detachment_kg_m2_s_pa",
            "runoff_fraction",
            "seep_flux_m_s",
            "max_erodible_depth_m",
        }
        for name, rule in self.material_rules.items():
            if set(rule) != required:
                raise ValueError(f"material rule {name} fields differ from contract")
            values = np.asarray(list(rule.values()), dtype=np.float64)
            if not np.isfinite(values).all() or np.any(values < 0.0):
                raise ValueError(f"material rule {name} has invalid values")
            if rule["runoff_fraction"] > 1.0:
                raise ValueError(f"material rule {name} runoff fraction exceeds one")


@dataclass(frozen=True)
class SlopeDomain:
    site_id: str
    bbox_en: tuple[float, float, float, float]
    texel_m: float
    height_m: np.ndarray
    valid: np.ndarray
    solve_domain: np.ndarray
    upstream_domain: np.ndarray
    outlet: np.ndarray
    collar: np.ndarray
    routing_barrier: np.ndarray
    hard_exclusion: np.ndarray
    unknown: np.ndarray
    vegetation_cover: np.ndarray
    seep_likelihood: np.ndarray
    upstream_water_m3: np.ndarray
    material_rule: np.ndarray
    material_rule_names: tuple[str, ...]
    source_identity: dict[str, Any]

    def __post_init__(self) -> None:
        arrays = (
            self.height_m,
            self.valid,
            self.solve_domain,
            self.upstream_domain,
            self.outlet,
            self.collar,
            self.routing_barrier,
            self.hard_exclusion,
            self.unknown,
            self.vegetation_cover,
            self.seep_likelihood,
            self.upstream_water_m3,
            self.material_rule,
        )
        shape = self.height_m.shape
        if len(shape) != 2 or min(shape) < 4:
            raise ValueError("slope domain must be a nontrivial two-dimensional raster")
        if any(value.shape != shape for value in arrays):
            raise ValueError("slope-domain arrays differ in shape")
        if not np.isfinite(self.height_m[self.valid]).all():
            raise ValueError("valid slope-domain height contains nonfinite values")
        if not np.isfinite(self.vegetation_cover).all():
            raise ValueError("vegetation evidence contains nonfinite values")
        if np.any((self.vegetation_cover < 0.0) | (self.vegetation_cover > 1.0)):
            raise ValueError("vegetation evidence must be normalized to [0, 1]")
        if not np.isfinite(self.seep_likelihood).all() or np.any(
            (self.seep_likelihood < 0.0) | (self.seep_likelihood > 1.0)
        ):
            raise ValueError("localized seep likelihood must be finite in [0, 1]")
        if not np.isfinite(self.upstream_water_m3).all() or np.any(
            self.upstream_water_m3 < 0.0
        ):
            raise ValueError("upstream water boundary must be finite and nonnegative")
        masks = (
            self.valid,
            self.solve_domain,
            self.upstream_domain,
            self.outlet,
            self.collar,
            self.routing_barrier,
            self.hard_exclusion,
            self.unknown,
        )
        if any(value.dtype != np.bool_ for value in masks):
            raise TypeError("slope-domain masks must be boolean")
        if not np.any(self.solve_domain):
            raise ValueError("physical solve domain is empty")
        if not np.any(self.outlet & self.solve_domain):
            raise ValueError("physical solve domain has no real outlet")
        if np.any(self.outlet & ~self.solve_domain):
            raise ValueError("outlets must lie in the physical solve domain")
        if np.any(self.upstream_water_m3 > 0.0) and not np.all(
            self.upstream_domain[self.upstream_water_m3 > 0.0]
        ):
            raise ValueError("upstream water enters outside the declared upstream boundary")
        if np.any(self.material_rule >= len(self.material_rule_names)):
            raise ValueError("material rule index is outside its bound inventory")
        e0, n0, e1, n1 = self.bbox_en
        expected_cols = int(round((e1 - e0) / self.texel_m))
        expected_rows = int(round((n1 - n0) / self.texel_m))
        if shape != (expected_rows, expected_cols):
            raise ValueError("domain bbox, texel, and raster shape disagree")
        if self.texel_m <= 0.0:
            raise ValueError("domain texel must be positive")
        boundary = np.zeros(shape, dtype=bool)
        boundary[[0, -1], :] = True
        boundary[:, [0, -1]] = True
        edge_active = self.active & boundary
        illegal_edge = edge_active & ~self.outlet & ~self.upstream_domain
        if np.any(illegal_edge):
            raise ValueError(
                "solve domain touches storage edge outside real outlet/upstream boundary"
            )

    @property
    def active(self) -> np.ndarray:
        """Routing support, including real water outlets but excluding the collar."""
        return (
            self.solve_domain
            & ~self.collar
            & self.valid
            & (~self.routing_barrier | self.outlet)
        )

    @property
    def form_active(self) -> np.ndarray:
        """Dry material support that may own detachment and visible relief."""
        return (
            self.active
            & ~self.unknown
            & ~self.hard_exclusion
            & ~self.outlet
            & (self.material_rule >= 0)
        )

    @property
    def hydrologic_source_active(self) -> np.ndarray:
        """Known dry material that contributes event water, even when protected."""
        return (
            self.active
            & ~self.unknown
            & ~self.outlet
            & (self.material_rule >= 0)
        )


def load_process_config(path: Path) -> ProcessConfig:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = set(ProcessConfig.__dataclass_fields__)
    if set(document) != expected:
        raise ValueError(
            f"process config fields differ: missing={sorted(expected - set(document))}, "
            f"extra={sorted(set(document) - expected)}"
        )
    return ProcessConfig(**document)
