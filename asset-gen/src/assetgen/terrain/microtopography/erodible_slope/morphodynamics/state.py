"""Physical contracts and extensive state for the nested slope solve.

Accepted terrain is an inclusive sample grid.  Morphology lives on the finite
volume cells between those samples; no mutable height array is authoritative.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MorphodynamicsConfig:
    event_duration_s: float
    rainfall_depth_m: float
    material_seed: int
    material_halo_m: float
    sand_correlation_m: float
    till_correlation_m: float
    sand_log_std: float
    till_log_std: float
    material_factor_min: float
    material_factor_max: float
    maturity_growth_s: float
    maturity_heal_s: float
    maturity_exponent: float
    maturity_floor: float
    transport_kg_per_joule: float
    deposit_bulk_density_kg_m3: float
    water_density_kg_m3: float
    gravity_m_s2: float
    manning_n_s_m13: float
    deposition_slope_ceiling: float
    repose_sand_gradient: float
    repose_till_gradient: float
    sediment_relative_tolerance: float

    def __post_init__(self) -> None:
        positive = np.asarray(
            [
                self.event_duration_s,
                self.rainfall_depth_m,
                self.material_halo_m,
                self.sand_correlation_m,
                self.till_correlation_m,
                self.sand_log_std,
                self.till_log_std,
                self.maturity_growth_s,
                self.maturity_heal_s,
                self.maturity_exponent,
                self.transport_kg_per_joule,
                self.deposit_bulk_density_kg_m3,
                self.water_density_kg_m3,
                self.gravity_m_s2,
                self.manning_n_s_m13,
                self.deposition_slope_ceiling,
                self.repose_sand_gradient,
                self.repose_till_gradient,
                self.sediment_relative_tolerance,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(positive).all() or np.any(positive <= 0.0):
            raise ValueError("physical morphodynamic values must be finite and positive")
        if not 0.0 < self.material_factor_min < self.material_factor_max:
            raise ValueError("material-factor bounds are invalid")
        if not 0.0 < self.maturity_floor <= 1.0:
            raise ValueError("maturity floor must lie in (0, 1]")
        if self.deposition_slope_ceiling >= 1.0:
            raise ValueError("deposition slope ceiling must be a dimensionless gradient")
        if isinstance(self.material_seed, bool) or not isinstance(self.material_seed, int):
            raise TypeError("material seed must be an integer")


@dataclass(frozen=True)
class FixedNestedRecipe:
    """Recipe-bound nested domain; output does not determine the solve canvas."""

    fine_canvas_bbox_en: tuple[float, float, float, float]
    fine_texel_m: float
    influence_guard_m: float
    output_windows: tuple[tuple[slice, slice], ...]
    expected_node_shape: tuple[int, int]
    maximum_rss_bytes: int
    maximum_workspace_bytes: int

    def __post_init__(self) -> None:
        e0, n0, e1, n1 = self.fine_canvas_bbox_en
        intervals = (
            int(round((n1 - n0) / self.fine_texel_m)),
            int(round((e1 - e0) / self.fine_texel_m)),
        )
        if self.expected_node_shape != (intervals[0] + 1, intervals[1] + 1):
            raise ValueError("fixed fine bbox and inclusive node shape disagree")
        if self.influence_guard_m <= 0.0:
            raise ValueError("nested solve requires a positive physical guard")
        if self.maximum_rss_bytes <= 0 or self.maximum_workspace_bytes <= 0:
            raise ValueError("memory and workspace ceilings must be positive")
        guard = int(round(self.influence_guard_m / self.fine_texel_m))
        if not np.isclose(guard * self.fine_texel_m, self.influence_guard_m):
            raise ValueError("influence guard is not aligned to the fine lattice")
        for rows, cols in self.output_windows:
            starts = (rows.start, cols.start)
            stops = (rows.stop, cols.stop)
            if None in starts or None in stops:
                raise ValueError("output windows must have explicit bounds")
            if starts[0] < guard or starts[1] < guard:
                raise ValueError("output window violates the north/west influence guard")
            if stops[0] > self.expected_node_shape[0] - guard:
                raise ValueError("output window violates the south influence guard")
            if stops[1] > self.expected_node_shape[1] - guard:
                raise ValueError("output window violates the east influence guard")


@dataclass(frozen=True)
class FineAuthority:
    """Exact accepted C0 and fine-lattice authority supplied by integration."""

    recipe: FixedNestedRecipe
    c0_node_m: np.ndarray
    valid_node: np.ndarray
    routing_node: np.ndarray
    form_active_node: np.ndarray
    hydrologic_source_active_node: np.ndarray
    outlet_node: np.ndarray
    interface_outflow_node: np.ndarray
    vegetation_node: np.ndarray
    seep_node: np.ndarray
    material_rule_node: np.ndarray
    material_rule_names: tuple[str, ...]

    def __post_init__(self) -> None:
        shape = self.recipe.expected_node_shape
        arrays = (
            self.c0_node_m,
            self.valid_node,
            self.routing_node,
            self.form_active_node,
            self.hydrologic_source_active_node,
            self.outlet_node,
            self.interface_outflow_node,
            self.vegetation_node,
            self.seep_node,
            self.material_rule_node,
        )
        if any(value.shape != shape for value in arrays):
            raise ValueError("fine-authority arrays differ from the inclusive node shape")
        for value in (
            self.valid_node,
            self.routing_node,
            self.form_active_node,
            self.hydrologic_source_active_node,
            self.outlet_node,
            self.interface_outflow_node,
        ):
            if value.dtype != np.bool_:
                raise TypeError("fine-authority masks must be boolean")
        if not np.isfinite(self.c0_node_m[self.valid_node]).all():
            raise ValueError("accepted C0 contains nonfinite valid samples")
        if self.c0_node_m.dtype != np.float64:
            raise TypeError("accepted C0 must remain float64 through morphodynamics")
        if np.any(self.form_active_node & ~self.valid_node):
            raise ValueError("form-active nodes must be valid")
        boundary = np.zeros(shape, dtype=bool)
        boundary[[0, -1], :] = True
        boundary[:, [0, -1]] = True
        if np.any(self.form_active_node & boundary):
            raise ValueError("fine canvas outer node ring must be non-morphological")
        if np.any(self.hydrologic_source_active_node & ~self.routing_node):
            raise ValueError("hydrologic sources must be routable")
        if np.any(self.outlet_node & self.interface_outflow_node):
            raise ValueError("real outlets and parent/fine interfaces must be distinct")
        if np.any(self.material_rule_node >= len(self.material_rule_names)):
            raise ValueError("material rule index is outside its inventory")
        if not np.isfinite(self.vegetation_node).all() or np.any(
            (self.vegetation_node < 0.0) | (self.vegetation_node > 1.0)
        ):
            raise ValueError("fine vegetation must lie in [0, 1]")
        if not np.isfinite(self.seep_node).all() or np.any(
            (self.seep_node < 0.0) | (self.seep_node > 1.0)
        ):
            raise ValueError("fine seep evidence must lie in [0, 1]")

    @property
    def cell_shape(self) -> tuple[int, int]:
        rows, cols = self.c0_node_m.shape
        return rows - 1, cols - 1

    @staticmethod
    def _all_four(mask: np.ndarray) -> np.ndarray:
        return mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]

    @staticmethod
    def _any_four(mask: np.ndarray) -> np.ndarray:
        return mask[:-1, :-1] | mask[1:, :-1] | mask[:-1, 1:] | mask[1:, 1:]

    @property
    def routing_cell(self) -> np.ndarray:
        return self._all_four(self.routing_node)

    @property
    def form_active_cell(self) -> np.ndarray:
        # Requiring four active descendants guarantees every excluded node is C0.
        return self._all_four(self.form_active_node)

    @property
    def hydrologic_source_active_cell(self) -> np.ndarray:
        return self._all_four(self.hydrologic_source_active_node)

    @property
    def outlet_cell(self) -> np.ndarray:
        return self.routing_cell & self._any_four(self.outlet_node)

    @property
    def interface_outflow_cell(self) -> np.ndarray:
        return self.routing_cell & self._any_four(self.interface_outflow_node)

    def cell_average(self, values: np.ndarray) -> np.ndarray:
        if values.shape != self.c0_node_m.shape:
            raise ValueError("node field has the wrong shape")
        return 0.25 * (
            values[:-1, :-1]
            + values[1:, :-1]
            + values[:-1, 1:]
            + values[1:, 1:]
        )


@dataclass(frozen=True)
class NestedBoundaryFlux:
    """Parent-organization flux entering the fixed fine control volume."""

    water_inflow_m3_s: np.ndarray
    sediment_inflow_kg_s: np.ndarray
    parent_exported_water_m3_s: float
    parent_exported_sediment_kg_s: float

    def validate(self, cell_shape: tuple[int, int]) -> None:
        if self.water_inflow_m3_s.shape != cell_shape:
            raise ValueError("nested water inflow has the wrong cell shape")
        if self.sediment_inflow_kg_s.shape != cell_shape:
            raise ValueError("nested sediment inflow has the wrong cell shape")
        if not np.isfinite(self.water_inflow_m3_s).all() or np.any(
            self.water_inflow_m3_s < 0.0
        ):
            raise ValueError("nested water inflow must be finite and nonnegative")
        if not np.isfinite(self.sediment_inflow_kg_s).all() or np.any(
            self.sediment_inflow_kg_s < 0.0
        ):
            raise ValueError("nested sediment inflow must be finite and nonnegative")
        if self.parent_exported_water_m3_s < 0.0:
            raise ValueError("parent water export is negative")
        if self.parent_exported_sediment_kg_s < 0.0:
            raise ValueError("parent sediment export is negative")
        if not np.isfinite(
            [self.parent_exported_water_m3_s, self.parent_exported_sediment_kg_s]
        ).all():
            raise ValueError("parent transfer accounting must be finite")


@dataclass
class MassLedger:
    imported_sediment_kg: float = 0.0
    exported_sediment_kg: float = 0.0
    thermally_relocated_kg: float = 0.0
    maximum_step_error_kg: float = 0.0


@dataclass
class MorphodynamicState:
    """Extensive solid inventories on control cells."""

    eroded_substrate_kg: np.ndarray
    deposited_colluvium_kg: np.ndarray
    suspended_kg: np.ndarray
    ponded_sediment_kg: np.ndarray
    maturity: np.ndarray
    substrate_limit_kg: np.ndarray
    ledger: MassLedger = field(default_factory=MassLedger)

    def validate(self, authority: FineAuthority) -> None:
        shape = authority.cell_shape
        arrays = (
            self.eroded_substrate_kg,
            self.deposited_colluvium_kg,
            self.suspended_kg,
            self.maturity,
            self.substrate_limit_kg,
        )
        if any(value.shape != shape for value in arrays):
            raise ValueError("morphodynamic inventories differ from the control grid")
        if any(not np.isfinite(value).all() for value in arrays):
            raise ValueError("morphodynamic state contains nonfinite values")
        if any(np.any(value < 0.0) for value in arrays):
            raise ValueError("morphodynamic inventories are negative")
        if self.ponded_sediment_kg.ndim != 1:
            raise ValueError("ponded sediment must be a one-dimensional reservoir inventory")
        if not np.isfinite(self.ponded_sediment_kg).all() or np.any(
            self.ponded_sediment_kg < 0.0
        ):
            raise ValueError("ponded sediment inventory is invalid")
        if np.any(self.eroded_substrate_kg > self.substrate_limit_kg + 1e-8):
            raise ValueError("substrate erosion exceeds the physical inventory")
        if np.any(self.maturity > 1.0):
            raise ValueError("maturity is outside [0, 1]")
        inactive = ~authority.form_active_cell
        if any(
            np.any(value[inactive] != 0.0)
            for value in (
                self.eroded_substrate_kg,
                self.deposited_colluvium_kg,
                self.suspended_kg,
                self.maturity,
                self.substrate_limit_kg,
            )
        ):
            raise ValueError("excluded descendants own morphology state")

    def sediment_residual_kg(self) -> float:
        return float(
            np.sum(self.eroded_substrate_kg, dtype=np.float64)
            + self.ledger.imported_sediment_kg
            - np.sum(self.deposited_colluvium_kg, dtype=np.float64)
            - np.sum(self.suspended_kg, dtype=np.float64)
            - np.sum(self.ponded_sediment_kg, dtype=np.float64)
            - self.ledger.exported_sediment_kg
        )


def allocate_array(
    workspace: Path | None,
    name: str,
    shape: tuple[int, int],
    dtype: np.dtype | type,
    *,
    fill: float = 0.0,
) -> np.ndarray:
    if workspace is None:
        result = np.empty(shape, dtype=dtype)
    else:
        workspace.mkdir(parents=True, exist_ok=True)
        result = np.memmap(workspace / f"{name}.bin", mode="w+", dtype=dtype, shape=shape)
    result.fill(fill)
    return result


def initial_state(
    authority: FineAuthority,
    bulk_density_kg_m3: np.ndarray,
    maximum_erodible_depth_m: np.ndarray,
    reservoir_count: int,
    *,
    workspace: Path | None,
) -> MorphodynamicState:
    shape = authority.cell_shape
    if reservoir_count < 0:
        raise ValueError("reservoir count is negative")
    if bulk_density_kg_m3.shape != shape or maximum_erodible_depth_m.shape != shape:
        raise ValueError("material inventory fields differ from the control grid")
    state = MorphodynamicState(
        eroded_substrate_kg=allocate_array(workspace, "eroded-substrate-kg", shape, np.float64),
        deposited_colluvium_kg=allocate_array(workspace, "deposited-colluvium-kg", shape, np.float64),
        suspended_kg=allocate_array(workspace, "suspended-kg", shape, np.float64),
        ponded_sediment_kg=np.zeros(reservoir_count, dtype=np.float64),
        maturity=allocate_array(workspace, "maturity", shape, np.float32),
        substrate_limit_kg=allocate_array(workspace, "substrate-limit-kg", shape, np.float64),
    )
    cell_area = authority.recipe.fine_texel_m**2
    state.substrate_limit_kg[...] = (
        bulk_density_kg_m3.astype(np.float64)
        * maximum_erodible_depth_m.astype(np.float64)
        * cell_area
    )
    state.substrate_limit_kg[~authority.form_active_cell] = 0.0
    state.validate(authority)
    return state


def compose_height_nodes(
    authority: FineAuthority,
    state: MorphodynamicState,
    substrate_density_kg_m3: np.ndarray,
    deposit_density_kg_m3: float,
    *,
    output: np.ndarray | None = None,
    tile_rows: int = 256,
) -> np.ndarray:
    """Derive C1 from C0 and extensive inventories without mutating C0."""
    state.validate(authority)
    shape = authority.c0_node_m.shape
    if substrate_density_kg_m3.shape != authority.cell_shape:
        raise ValueError("substrate density differs from the control grid")
    result = np.empty(shape, dtype=np.float64) if output is None else output
    if result.shape != shape:
        raise ValueError("height output has the wrong inclusive node shape")
    result[...] = authority.c0_node_m
    cell_area = authority.recipe.fine_texel_m**2
    for start in range(0, authority.cell_shape[0], tile_rows):
        stop = min(authority.cell_shape[0], start + tile_rows)
        eroded_volume = state.eroded_substrate_kg[start:stop] / np.maximum(
            substrate_density_kg_m3[start:stop].astype(np.float64), 1.0
        )
        deposited_volume = state.deposited_colluvium_kg[start:stop] / deposit_density_kg_m3
        dz = (deposited_volume - eroded_volume) * (0.25 / cell_area)
        result[start:stop, :-1] += dz
        result[start + 1 : stop + 1, :-1] += dz
        result[start:stop, 1:] += dz
        result[start + 1 : stop + 1, 1:] += dz
    # This assignment is the non-negotiable descendant-preservation gate.
    result[~authority.form_active_node] = authority.c0_node_m[~authority.form_active_node]
    return result


def assert_refinement_partition(parent_kg: np.ndarray, children_kg: np.ndarray) -> None:
    """Gate exact extensive quartering when an AMR organization layer is used."""
    expected = np.repeat(np.repeat(parent_kg * 0.25, 2, axis=0), 2, axis=1)
    if children_kg.shape != expected.shape or not np.array_equal(children_kg, expected):
        raise RuntimeError("refinement did not split extensive kilograms by exact quarters")
