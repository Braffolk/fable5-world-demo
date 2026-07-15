"""Recipe-owned assembly of the exact Development A morphodynamic inputs."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from .hydrology import (
    OrganizationHierarchy,
    ParentOrganizationHierarchy,
    bind_parent_organization_to_fine,
    build_authorized_parent_organization,
)
from .recipe import FrozenMorphodynamicsRecipe, load_frozen_recipe
from .state import FineAuthority, NestedBoundaryFlux
from .structural_base import (
    CANONICAL_BBOX_EN,
    CONTROL_BBOX_EN,
    FINE_CANVAS_BBOX_EN,
    FINE_CANVAS_SHAPE,
    StructuralFineCanvas,
    load_development_a_structural_base,
)
from ..solver.load import load_crop_evaluation_plan
from ..solver.model import CropEvaluation, ProcessConfig, SlopeDomain, load_process_config


@dataclass(frozen=True)
class BoundDevelopmentA:
    recipe: FrozenMorphodynamicsRecipe
    structural_base: StructuralFineCanvas
    process: ProcessConfig
    evaluations: tuple[CropEvaluation, ...]
    authority: FineAuthority
    canonical_domain: SlopeDomain
    control_domain: SlopeDomain
    canonical_boundary: NestedBoundaryFlux
    control_boundary: NestedBoundaryFlux
    canonical_organization: OrganizationHierarchy
    control_organization: OrganizationHierarchy
    semantic_sha256: str


def _node_coordinates(
    domain: SlopeDomain,
    structural: StructuralFineCanvas,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    e0, _, _, n1 = structural.bbox_en
    de0, _, _, dn1 = domain.bbox_en
    texel = structural.texel_m
    east = e0 + (np.arange(FINE_CANVAS_SHAPE[1], dtype=np.float64) + 0.5) * texel
    north = n1 - (np.arange(FINE_CANVAS_SHAPE[0], dtype=np.float64) + 0.5) * texel
    fractional_cols = (east - (de0 + 0.5 * domain.texel_m)) / domain.texel_m
    fractional_rows = ((dn1 - 0.5 * domain.texel_m) - north) / domain.texel_m
    nearest_cols = np.floor((east - de0) / domain.texel_m).astype(np.int32)
    nearest_rows = np.floor((dn1 - north) / domain.texel_m).astype(np.int32)
    if (
        nearest_rows.min() < 0
        or nearest_cols.min() < 0
        or nearest_rows.max() >= domain.height_m.shape[0]
        or nearest_cols.max() >= domain.height_m.shape[1]
        or fractional_rows.min() < 0.0
        or fractional_cols.min() < 0.0
        or fractional_rows.max() > domain.height_m.shape[0] - 1
        or fractional_cols.max() > domain.height_m.shape[1] - 1
    ):
        raise ValueError("fine canvas lacks exact 1 m condition support")
    return fractional_rows, fractional_cols, nearest_rows, nearest_cols


def _nearest_nodes(
    values: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    return np.asarray(values[rows[:, None], cols[None, :]]).copy()


def _linear_nodes(
    values: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    result = np.empty(FINE_CANVAS_SHAPE, dtype=np.float32)
    for start in range(0, result.shape[0], 256):
        stop = min(start + 256, result.shape[0])
        row_coordinates = np.broadcast_to(
            rows[start:stop, None], (stop - start, result.shape[1])
        )
        col_coordinates = np.broadcast_to(
            cols[None, :], (stop - start, result.shape[1])
        )
        result[start:stop] = ndimage.map_coordinates(
            values,
            (row_coordinates, col_coordinates),
            order=1,
            mode="nearest",
            prefilter=False,
        ).astype(np.float32)
    return result


def _suppress_material_interfaces(
    form_active: np.ndarray,
    material_rule: np.ndarray,
) -> None:
    northwest = material_rule[:-1, :-1]
    heterogeneous = (
        (material_rule[1:, :-1] != northwest)
        | (material_rule[:-1, 1:] != northwest)
        | (material_rule[1:, 1:] != northwest)
    )
    if not np.any(heterogeneous):
        return
    forbidden_node = np.zeros(form_active.shape, dtype=bool)
    forbidden_node[:-1, :-1] |= heterogeneous
    forbidden_node[1:, :-1] |= heterogeneous
    forbidden_node[:-1, 1:] |= heterogeneous
    forbidden_node[1:, 1:] |= heterogeneous
    form_active[forbidden_node] = False


def _fine_authority(
    domain: SlopeDomain,
    structural: StructuralFineCanvas,
    recipe: FrozenMorphodynamicsRecipe,
) -> FineAuthority:
    fractional_rows, fractional_cols, rows, cols = _node_coordinates(
        domain, structural
    )
    valid = _nearest_nodes(domain.valid, rows, cols).astype(bool)
    routing = _nearest_nodes(domain.active, rows, cols).astype(bool)
    form_active = _nearest_nodes(domain.form_active, rows, cols).astype(bool)
    hydrologic_source = _nearest_nodes(
        domain.hydrologic_source_active, rows, cols
    ).astype(bool)
    outlet = _nearest_nodes(domain.outlet, rows, cols).astype(bool) & routing
    material_rule = _nearest_nodes(domain.material_rule, rows, cols).astype(np.int16)
    form_active &= ~structural.forbidden_morphology
    form_active &= ~structural.unknown_bathymetry
    hydrologic_source &= ~structural.unknown_bathymetry
    _suppress_material_interfaces(form_active, material_rule)

    boundary = np.zeros(FINE_CANVAS_SHAPE, dtype=bool)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    interface = boundary & routing & ~outlet
    form_active[interface] = False

    arrays = (
        valid,
        routing,
        form_active,
        hydrologic_source,
        outlet,
        interface,
        material_rule,
    )
    for value in arrays:
        value.flags.writeable = False
    vegetation = _linear_nodes(
        domain.vegetation_cover, fractional_rows, fractional_cols
    )
    seep = _linear_nodes(domain.seep_likelihood, fractional_rows, fractional_cols)
    vegetation.flags.writeable = False
    seep.flags.writeable = False
    return FineAuthority(
        recipe=recipe.fixed_nested,
        c0_node_m=structural.c0_height_m,
        valid_node=valid,
        routing_node=routing,
        form_active_node=form_active,
        hydrologic_source_active_node=hydrologic_source,
        outlet_node=outlet,
        interface_outflow_node=interface,
        vegetation_node=vegetation,
        seep_node=seep,
        material_rule_node=material_rule,
        material_rule_names=domain.material_rule_names,
    )


def _require_authority_identity(
    canonical: FineAuthority,
    control: FineAuthority,
) -> FineAuthority:
    if canonical.material_rule_names != control.material_rule_names:
        raise RuntimeError("canonical/control fine material inventories differ")
    names = (
        "valid_node",
        "routing_node",
        "form_active_node",
        "hydrologic_source_active_node",
        "outlet_node",
        "interface_outflow_node",
        "vegetation_node",
        "seep_node",
        "material_rule_node",
    )
    for name in names:
        if not np.array_equal(getattr(canonical, name), getattr(control, name)):
            raise RuntimeError(f"canonical/control fine authority {name} differs")
    return canonical


def _parent_source_water(
    domain: SlopeDomain,
    process: ProcessConfig,
) -> np.ndarray:
    runoff = np.zeros(domain.height_m.shape, dtype=np.float64)
    seep_flux = np.zeros(domain.height_m.shape, dtype=np.float64)
    for index, name in enumerate(domain.material_rule_names):
        if name not in process.material_rules:
            raise ValueError(f"parent domain names unbound material rule {name}")
        selected = domain.material_rule == index
        runoff[selected] = process.material_rules[name]["runoff_fraction"]
        seep_flux[selected] = process.material_rules[name]["seep_flux_m_s"]
    source = (
        (process.rainfall_depth_m / process.event_duration_s)
        * runoff
        * np.clip(1.0 - 0.78 * domain.vegetation_cover, 0.08, 1.0)
        + seep_flux * domain.seep_likelihood
    ) * domain.texel_m**2
    source[~domain.hydrologic_source_active] = 0.0
    source += domain.upstream_water_m3 / process.event_duration_s
    return source


def _inside_fine_parent_cells(domain: SlopeDomain) -> np.ndarray:
    e0, n0, e1, n1 = domain.bbox_en
    fe0, fn0, fe1, fn1 = FINE_CANVAS_BBOX_EN
    east = e0 + (np.arange(domain.height_m.shape[1]) + 0.5) * domain.texel_m
    north = n1 - (np.arange(domain.height_m.shape[0]) + 0.5) * domain.texel_m
    return (
        (north[:, None] >= fn0)
        & (north[:, None] < fn1)
        & (east[None, :] >= fe0)
        & (east[None, :] < fe1)
    )


def _fine_interface_index(
    domain: SlopeDomain,
    source_index: int,
    target_index: int,
    cell_shape: tuple[int, int],
) -> tuple[int, int]:
    rows, cols = domain.height_m.shape
    source_row, source_col = divmod(source_index, cols)
    target_row, target_col = divmod(target_index, cols)
    e0, _, _, n1 = domain.bbox_en
    fe0, _, fe1, fn1 = FINE_CANVAS_BBOX_EN
    source_e = e0 + (source_col + 0.5) * domain.texel_m
    source_n = n1 - (source_row + 0.5) * domain.texel_m
    target_e = e0 + (target_col + 0.5) * domain.texel_m
    target_n = n1 - (target_row + 0.5) * domain.texel_m
    fine = (fe1 - fe0) / cell_shape[1]
    row = int(np.floor((fn1 - target_n) / fine))
    col = int(np.floor((target_e - fe0) / fine))
    if source_n >= fn1:
        row = 0
    elif source_n < FINE_CANVAS_BBOX_EN[1]:
        row = cell_shape[0] - 1
    if source_e < fe0:
        col = 0
    elif source_e >= fe1:
        col = cell_shape[1] - 1
    return (
        int(np.clip(row, 0, cell_shape[0] - 1)),
        int(np.clip(col, 0, cell_shape[1] - 1)),
    )


def _derive_parent_boundary(
    domain: SlopeDomain,
    parent: ParentOrganizationHierarchy,
    authority: FineAuthority,
    process: ProcessConfig,
) -> NestedBoundaryFlux:
    inside = _inside_fine_parent_cells(domain)
    source = _parent_source_water(domain, process)
    source[inside] = 0.0
    accumulated = source.ravel().copy()
    inflow = np.zeros(authority.cell_shape, dtype=np.float64)
    active = domain.active.ravel()
    connected = parent.connected.ravel()
    rank = parent.drain_rank.ravel()
    order = np.flatnonzero(active & connected & ~inside.ravel())
    order = order[np.argsort(rank[order])[::-1]]
    for index in order:
        target = int(parent.flood_parent.ravel()[index])
        if target < 0:
            continue
        if inside.ravel()[target]:
            row, col = _fine_interface_index(
                domain, int(index), target, authority.cell_shape
            )
            if not authority.routing_cell[row, col]:
                raise RuntimeError("parent water enters outside fine routing support")
            inflow[row, col] += accumulated[index]
        elif active[target]:
            accumulated[target] += accumulated[index]
    inflow.flags.writeable = False
    sediment = np.zeros(authority.cell_shape, dtype=np.float64)
    sediment.flags.writeable = False
    return NestedBoundaryFlux(
        water_inflow_m3_s=inflow,
        sediment_inflow_kg_s=sediment,
        parent_exported_water_m3_s=float(np.sum(inflow, dtype=np.float64)),
        # The parent pass organizes water only; no unmodelled exterior erosion
        # is converted into a synthetic sediment boundary condition.
        parent_exported_sediment_kg_s=0.0,
    )


def _require_nested_identity(
    canonical_boundary: NestedBoundaryFlux,
    control_boundary: NestedBoundaryFlux,
    canonical_organization: OrganizationHierarchy,
    control_organization: OrganizationHierarchy,
) -> None:
    boundary_pairs = (
        (
            canonical_boundary.water_inflow_m3_s,
            control_boundary.water_inflow_m3_s,
            "water boundary",
        ),
        (
            canonical_boundary.sediment_inflow_kg_s,
            control_boundary.sediment_inflow_kg_s,
            "sediment boundary",
        ),
    )
    for canonical, control, role in boundary_pairs:
        if not np.array_equal(canonical, control):
            raise RuntimeError(f"canonical/control {role} differs")
    if (
        canonical_boundary.parent_exported_water_m3_s
        != control_boundary.parent_exported_water_m3_s
        or canonical_boundary.parent_exported_sediment_kg_s
        != control_boundary.parent_exported_sediment_kg_s
    ):
        raise RuntimeError("canonical/control parent transfer accounting differs")
    organization_pairs = (
        (
            canonical_organization.filled_level_m,
            control_organization.filled_level_m,
            "filled level",
        ),
        (
            canonical_organization.flood_parent,
            control_organization.flood_parent,
            "flood parent",
        ),
        (
            canonical_organization.drain_rank,
            control_organization.drain_rank,
            "drain rank",
        ),
        (
            canonical_organization.connected,
            control_organization.connected,
            "connectivity",
        ),
        (
            canonical_organization.reservoir_owner,
            control_organization.reservoir_owner,
            "reservoir owner",
        ),
        (
            canonical_organization.reservoir_capacity_m3,
            control_organization.reservoir_capacity_m3,
            "reservoir capacity",
        ),
    )
    for canonical, control, role in organization_pairs:
        if not np.array_equal(canonical, control):
            raise RuntimeError(f"canonical/control fine organization {role} differs")


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    dtype = array.dtype.newbyteorder("<")
    if array.dtype != dtype:
        array = array.astype(dtype, copy=False)
    digest = hashlib.sha256()
    digest.update(dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _semantic_sha256(
    recipe: FrozenMorphodynamicsRecipe,
    authority: FineAuthority,
    canonical_boundary: NestedBoundaryFlux,
    control_boundary: NestedBoundaryFlux,
    canonical_organization: OrganizationHierarchy,
    control_organization: OrganizationHierarchy,
) -> str:
    arrays: dict[str, np.ndarray] = {
        "c0": authority.c0_node_m,
        "valid": authority.valid_node,
        "routing": authority.routing_node,
        "form_active": authority.form_active_node,
        "hydrologic_source": authority.hydrologic_source_active_node,
        "outlet": authority.outlet_node,
        "interface": authority.interface_outflow_node,
        "vegetation": authority.vegetation_node,
        "seep": authority.seep_node,
        "material_rule": authority.material_rule_node,
        "canonical_water_boundary": canonical_boundary.water_inflow_m3_s,
        "canonical_sediment_boundary": canonical_boundary.sediment_inflow_kg_s,
        "control_water_boundary": control_boundary.water_inflow_m3_s,
        "control_sediment_boundary": control_boundary.sediment_inflow_kg_s,
    }
    for role, organization in (
        ("canonical", canonical_organization),
        ("control", control_organization),
    ):
        arrays[f"{role}_filled"] = organization.filled_level_m
        arrays[f"{role}_parent"] = organization.flood_parent
        arrays[f"{role}_rank"] = organization.drain_rank
        arrays[f"{role}_connected"] = organization.connected
        arrays[f"{role}_reservoir_owner"] = organization.reservoir_owner
        arrays[f"{role}_reservoir_capacity"] = organization.reservoir_capacity_m3
    document = {
        "schema": "laas.erodible-slope-bound-development-a/1",
        "recipeSha256": recipe.sha256,
        "c0Sha256": recipe.c0_sha256,
        "spatialSha256": recipe.spatial_sha256,
        "arrays": {name: _array_sha256(value) for name, value in sorted(arrays.items())},
        "materialRuleNames": list(authority.material_rule_names),
        "parentSedimentPolicy": "zero-no-exterior-morphodynamics",
    }
    payload = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def load_bound_development_a(
    recipe_path: Path,
    *,
    expected_recipe_sha256: str,
) -> BoundDevelopmentA:
    """Load every production input solely through one caller-bound recipe."""
    structural = load_development_a_structural_base()
    recipe = load_frozen_recipe(
        recipe_path,
        expected_sha256=expected_recipe_sha256,
        structural_base=structural,
    )
    process = load_process_config(recipe.bindings.process_config.repository_path)
    evaluations, _, evaluation_sha256, canonical_domain, control_domain = (
        load_crop_evaluation_plan(
            recipe.bindings.evaluation_plan.repository_path,
            condition_bundle_path=recipe.bindings.condition_bundle.repository_path,
            config=process,
        )
    )
    if evaluation_sha256 != recipe.bindings.evaluation_plan.sha256:
        raise ValueError("evaluation loader returned another identity")
    for role, domain, expected_bbox in (
        ("canonical", canonical_domain, CANONICAL_BBOX_EN),
        ("control", control_domain, CONTROL_BBOX_EN),
    ):
        if domain.texel_m != 1.0 or tuple(domain.bbox_en) != expected_bbox:
            raise ValueError(f"{role} domain differs from the exact 1 m contract")
        closure = domain.source_identity.get("domain_closure", {})
        if (
            closure.get("path") != recipe.bindings.domain_closure.relative_path
            or closure.get("sha256") != recipe.bindings.domain_closure.sha256
        ):
            raise ValueError(f"{role} domain names another closure authority")
        structural.require_domain(role, tuple(domain.bbox_en))
    expected_crops = {
        tuple(evaluation.bbox_en) for evaluation in evaluations
    }
    if expected_crops != {
        (680448.0, 6444416.0, 680576.0, 6444544.0),
        (680576.0, 6444416.0, 680704.0, 6444544.0),
    }:
        raise ValueError("evaluation output windows differ from the fixed recipe")

    canonical_authority = _fine_authority(canonical_domain, structural, recipe)
    control_authority = _fine_authority(control_domain, structural, recipe)
    authority = _require_authority_identity(canonical_authority, control_authority)
    canonical_parent = build_authorized_parent_organization(
        canonical_domain, role="canonical"
    )
    control_parent = build_authorized_parent_organization(control_domain, role="control")
    canonical_organization = bind_parent_organization_to_fine(
        canonical_parent, authority
    )
    control_organization = bind_parent_organization_to_fine(control_parent, authority)
    canonical_boundary = _derive_parent_boundary(
        canonical_domain, canonical_parent, authority, process
    )
    control_boundary = _derive_parent_boundary(
        control_domain, control_parent, authority, process
    )
    _require_nested_identity(
        canonical_boundary,
        control_boundary,
        canonical_organization,
        control_organization,
    )
    semantic_sha256 = _semantic_sha256(
        recipe,
        authority,
        canonical_boundary,
        control_boundary,
        canonical_organization,
        control_organization,
    )
    return BoundDevelopmentA(
        recipe=recipe,
        structural_base=structural,
        process=process,
        evaluations=evaluations,
        authority=authority,
        canonical_domain=canonical_domain,
        control_domain=control_domain,
        canonical_boundary=canonical_boundary,
        control_boundary=control_boundary,
        canonical_organization=canonical_organization,
        control_organization=control_organization,
        semantic_sha256=semantic_sha256,
    )
