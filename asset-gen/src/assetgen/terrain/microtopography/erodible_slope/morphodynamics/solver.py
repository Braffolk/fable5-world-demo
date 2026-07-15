"""One organization prepass and one authoritative fine morphodynamic event."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .hydrology import HydrologyResult, OrganizationHierarchy, solve_hydrology
from .material_field import MaterialFields, build_material_fields
from .relaxation import RelaxationEvent, relax_colluvium
from .sediment import SedimentEvent, advance_sediment_event, update_maturity
from .state import (
    FineAuthority,
    FixedNestedRecipe,
    MorphodynamicState,
    MorphodynamicsConfig,
    NestedBoundaryFlux,
    compose_height_nodes,
    initial_state,
)
from ..solver.model import ProcessConfig


@dataclass(frozen=True)
class MorphodynamicsSummary:
    node_shape: tuple[int, int]
    cell_shape: tuple[int, int]
    eroded_kg: float
    deposited_kg: float
    imported_kg: float
    exported_kg: float
    suspended_kg: float
    ponded_sediment_kg: float
    thermally_relocated_kg: float
    maximum_abs_relief_m: float
    solid_ledger_error_kg: float
    volume_identity_error_m3: float
    source_water_m3_s: float
    exported_water_m3_s: float
    trapped_water_m3_s: float


@dataclass(frozen=True)
class MorphodynamicsResult:
    authority: FineAuthority
    state: MorphodynamicState
    material: MaterialFields
    hydrology: HydrologyResult
    c1_node_m: np.ndarray
    sediment_event: SedimentEvent
    relaxation_event: RelaxationEvent
    summary: MorphodynamicsSummary

    def output(self, index: int = 0) -> np.ndarray:
        rows, cols = self.authority.recipe.output_windows[index]
        return self.c1_node_m[rows, cols]


def _require_parent_identity(
    canonical: NestedBoundaryFlux,
    control: NestedBoundaryFlux,
    shape: tuple[int, int],
    tolerance: float,
) -> NestedBoundaryFlux:
    canonical.validate(shape)
    control.validate(shape)
    if not np.array_equal(canonical.water_inflow_m3_s, control.water_inflow_m3_s):
        raise RuntimeError("canonical/control fine-interface water differs")
    if not np.array_equal(canonical.sediment_inflow_kg_s, control.sediment_inflow_kg_s):
        raise RuntimeError("canonical/control fine-interface sediment differs")
    if canonical.parent_exported_water_m3_s != control.parent_exported_water_m3_s:
        raise RuntimeError("canonical/control parent water accounting differs")
    if canonical.parent_exported_sediment_kg_s != control.parent_exported_sediment_kg_s:
        raise RuntimeError("canonical/control parent sediment accounting differs")
    water_transfer = float(np.sum(canonical.water_inflow_m3_s, dtype=np.float64))
    sediment_transfer = float(np.sum(canonical.sediment_inflow_kg_s, dtype=np.float64))
    if abs(water_transfer - canonical.parent_exported_water_m3_s) > max(
        1e-12, water_transfer * tolerance
    ):
        raise RuntimeError("parent/fine water transfer identity differs")
    if abs(sediment_transfer - canonical.parent_exported_sediment_kg_s) > max(
        1e-12, sediment_transfer * tolerance
    ):
        raise RuntimeError("parent/fine sediment transfer identity differs")
    return canonical


def _require_organization_identity(
    canonical: OrganizationHierarchy,
    control: OrganizationHierarchy,
    shape: tuple[int, int],
) -> OrganizationHierarchy:
    if canonical.source_role != "canonical" or control.source_role != "control":
        raise ValueError("fine organizations are not canonical/control bound")
    for value in (
        canonical.filled_level_m,
        canonical.flood_parent,
        canonical.drain_rank,
        canonical.connected,
        canonical.reservoir_owner,
        control.filled_level_m,
        control.flood_parent,
        control.drain_rank,
        control.connected,
        control.reservoir_owner,
    ):
        if value.shape != shape:
            raise ValueError("fine organization differs from the control grid")
    comparisons = (
        (canonical.filled_level_m, control.filled_level_m, "filled level"),
        (canonical.flood_parent, control.flood_parent, "flood parent"),
        (canonical.drain_rank, control.drain_rank, "drain rank"),
        (canonical.connected, control.connected, "connectivity"),
        (canonical.reservoir_owner, control.reservoir_owner, "reservoir owner"),
        (
            canonical.reservoir_capacity_m3,
            control.reservoir_capacity_m3,
            "reservoir capacity",
        ),
    )
    for left, right, role in comparisons:
        if not np.array_equal(left, right):
            raise RuntimeError(f"canonical/control fine organization {role} differs")
    return canonical


def _unique_array_bytes(values: tuple[np.ndarray, ...]) -> int:
    seen: set[int] = set()
    total = 0
    for value in values:
        identity = id(value)
        if identity not in seen:
            total += value.nbytes
            seen.add(identity)
    return total


def _require_resource_budget(
    authority: FineAuthority,
    canonical_boundary: NestedBoundaryFlux,
    control_boundary: NestedBoundaryFlux,
    canonical_organization: OrganizationHierarchy,
    control_organization: OrganizationHierarchy,
    config: MorphodynamicsConfig,
) -> None:
    """Admit the solve against explicit per-phase dtype/liveness bounds."""
    recipe = authority.recipe
    cells = authority.cell_shape[0] * authority.cell_shape[1]
    nodes = authority.c0_node_m.size
    margin = int(round(config.material_halo_m / recipe.fine_texel_m))
    expanded = (authority.cell_shape[0] + 2 * margin) * (
        authority.cell_shape[1] + 2 * margin
    )
    inputs = _unique_array_bytes(
        (
            authority.c0_node_m,
            authority.valid_node,
            authority.routing_node,
            authority.form_active_node,
            authority.hydrologic_source_active_node,
            authority.outlet_node,
            authority.interface_outflow_node,
            authority.vegetation_node,
            authority.seep_node,
            authority.material_rule_node,
            canonical_boundary.water_inflow_m3_s,
            canonical_boundary.sediment_inflow_kg_s,
            control_boundary.water_inflow_m3_s,
            control_boundary.sediment_inflow_kg_s,
            canonical_organization.filled_level_m,
            canonical_organization.flood_parent,
            canonical_organization.drain_rank,
            canonical_organization.connected,
            canonical_organization.reservoir_owner,
            canonical_organization.reservoir_capacity_m3,
            control_organization.filled_level_m,
            control_organization.flood_parent,
            control_organization.drain_rank,
            control_organization.connected,
            control_organization.reservoir_owner,
            control_organization.reservoir_capacity_m3,
        )
    )
    material_live = 12 * expanded + 32 * cells
    state_live = 36 * cells + 8 * canonical_organization.reservoir_capacity_m3.size
    hydrology_live = 16 * nodes + 88 * cells
    maturity_live = 4 * cells + min(cells, 128 * authority.cell_shape[1]) * 56
    relaxation_live = 16 * nodes + 32 * cells + min(
        cells, 128 * authority.cell_shape[1]
    ) * 32
    material_tile = (256 + 2 * margin) ** 2 * 40
    phase_bytes = {
        "material": inputs + material_live + material_tile,
        "hydrology": inputs + 32 * cells + state_live + hydrology_live,
        "maturity": inputs + 32 * cells + state_live + hydrology_live + maturity_live,
        "relaxation": inputs
        + 32 * cells
        + state_live
        + hydrology_live
        + relaxation_live,
        "final_composition": inputs
        + 32 * cells
        + state_live
        + hydrology_live
        + 8 * nodes,
    }
    peak_phase, peak_bytes = max(phase_bytes.items(), key=lambda value: value[1])
    # Every named work array remains an immutable phase artifact until the
    # workspace transaction ends. Count its dtype bytes independently of RSS.
    workspace_bytes = (
        12 * expanded
        + (32 + 36 + 4 + 60 + 32) * cells
        + 16 * nodes
        + 8 * canonical_organization.reservoir_capacity_m3.size
    )
    if peak_bytes > recipe.maximum_rss_bytes:
        raise MemoryError(
            f"morphodynamic {peak_phase} phase needs {peak_bytes} live bytes, "
            f"above the {recipe.maximum_rss_bytes} byte RSS ceiling; phases={phase_bytes}"
        )
    if workspace_bytes > recipe.maximum_workspace_bytes:
        raise MemoryError(
            f"morphodynamic phases need {workspace_bytes} workspace bytes, above "
            f"the {recipe.maximum_workspace_bytes} byte ceiling"
        )


def solve_morphodynamics(
    authority: FineAuthority,
    process: ProcessConfig,
    canonical_boundary: NestedBoundaryFlux,
    control_boundary: NestedBoundaryFlux,
    canonical_organization: OrganizationHierarchy,
    control_organization: OrganizationHierarchy,
    *,
    workspace: Path | None,
    config: MorphodynamicsConfig,
) -> MorphodynamicsResult:
    """Solve one recipe-bound event without packing, publishing, or runtime work."""
    if config is None:
        raise TypeError("a hash-bound morphodynamics config is required")
    if process.fine_texel_m != authority.recipe.fine_texel_m:
        raise ValueError("process and fixed fine lattice differ")
    if process.event_duration_s != config.event_duration_s:
        raise ValueError("process and morphodynamic event duration differ")
    if process.rainfall_depth_m != config.rainfall_depth_m:
        raise ValueError("process and morphodynamic rainfall differ")
    workspace = Path(workspace) if workspace is not None else None
    _require_resource_budget(
        authority,
        canonical_boundary,
        control_boundary,
        canonical_organization,
        control_organization,
        config,
    )
    boundary = _require_parent_identity(
        canonical_boundary,
        control_boundary,
        authority.cell_shape,
        config.sediment_relative_tolerance,
    )
    hierarchy = _require_organization_identity(
        canonical_organization,
        control_organization,
        authority.cell_shape,
    )
    material = build_material_fields(authority, process, config, workspace=workspace)
    state = initial_state(
        authority,
        material.bulk_density_kg_m3,
        material.maximum_erodible_depth_m,
        hierarchy.reservoir_capacity_m3.size,
        workspace=workspace,
    )
    hydrology = solve_hydrology(
        authority,
        hierarchy,
        state,
        material,
        boundary,
        config,
        workspace=workspace,
    )
    event_detachment_factor = update_maturity(
        authority,
        state,
        material,
        hydrology,
        config,
        workspace=workspace,
    )
    sediment = advance_sediment_event(
        authority,
        state,
        material,
        hydrology,
        boundary,
        config,
        event_detachment_factor,
    )
    relaxation = relax_colluvium(
        authority,
        state,
        material,
        config,
        workspace=workspace,
    )
    state.validate(authority)
    c1 = compose_height_nodes(
        authority,
        state,
        material.bulk_density_kg_m3,
        config.deposit_bulk_density_kg_m3,
    )
    excluded = ~authority.form_active_node
    if not np.array_equal(c1[excluded], authority.c0_node_m[excluded]):
        raise RuntimeError("hard/unknown/water/protected descendants changed from C0")
    solid_error = state.sediment_residual_kg()
    solid_scale = (
        float(np.sum(state.eroded_substrate_kg, dtype=np.float64))
        + state.ledger.imported_sediment_kg
    )
    if abs(solid_error) > max(
        1e-8, solid_scale * config.sediment_relative_tolerance
    ):
        raise RuntimeError(f"final solid ledger differs by {solid_error} kg")
    cell_area = authority.recipe.fine_texel_m**2
    height_volume = 0.0
    for start in range(0, authority.cell_shape[0], 256):
        stop = min(authority.cell_shape[0], start + 256)
        delta = c1[start : stop + 1] - authority.c0_node_m[start : stop + 1]
        control_delta = 0.25 * (
            delta[:-1, :-1]
            + delta[1:, :-1]
            + delta[:-1, 1:]
            + delta[1:, 1:]
        )
        height_volume += float(np.sum(control_delta, dtype=np.float64) * cell_area)
    inventory_volume = 0.0
    for start in range(0, authority.cell_shape[0], 256):
        stop = min(authority.cell_shape[0], start + 256)
        inventory_volume += float(
            np.sum(
                state.deposited_colluvium_kg[start:stop]
                / config.deposit_bulk_density_kg_m3
                - state.eroded_substrate_kg[start:stop]
                / np.maximum(
                    material.bulk_density_kg_m3[start:stop].astype(np.float64),
                    1.0,
                ),
                dtype=np.float64,
            )
        )
    volume_error = height_volume - inventory_volume
    if abs(volume_error) > max(1e-10, abs(inventory_volume) * 2e-8):
        raise RuntimeError(f"C1-C0 and extensive solid volume differ by {volume_error} m3")
    summary = MorphodynamicsSummary(
        node_shape=authority.c0_node_m.shape,
        cell_shape=authority.cell_shape,
        eroded_kg=float(np.sum(state.eroded_substrate_kg, dtype=np.float64)),
        deposited_kg=float(np.sum(state.deposited_colluvium_kg, dtype=np.float64)),
        imported_kg=state.ledger.imported_sediment_kg,
        exported_kg=state.ledger.exported_sediment_kg,
        suspended_kg=float(np.sum(state.suspended_kg, dtype=np.float64)),
        ponded_sediment_kg=float(
            np.sum(state.ponded_sediment_kg, dtype=np.float64)
        ),
        thermally_relocated_kg=state.ledger.thermally_relocated_kg,
        maximum_abs_relief_m=float(np.max(np.abs(c1 - authority.c0_node_m))),
        solid_ledger_error_kg=solid_error,
        volume_identity_error_m3=volume_error,
        source_water_m3_s=hydrology.source_water_m3_s,
        exported_water_m3_s=hydrology.exported_water_m3_s,
        trapped_water_m3_s=hydrology.trapped_water_m3_s,
    )
    return MorphodynamicsResult(
        authority=authority,
        state=state,
        material=material,
        hydrology=hydrology,
        c1_node_m=c1,
        sediment_event=sediment,
        relaxation_event=relaxation,
        summary=summary,
    )
