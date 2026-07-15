"""Priority-flood organization and continuous fine-grid water routing."""
from __future__ import annotations

from dataclasses import dataclass
import heapq
from pathlib import Path

import numpy as np
from scipy import ndimage

from .material_field import MaterialFields
from .state import (
    FineAuthority,
    MorphodynamicState,
    MorphodynamicsConfig,
    NestedBoundaryFlux,
    allocate_array,
    compose_height_nodes,
)
from .structural_base import CANONICAL_BBOX_EN, CONTROL_BBOX_EN
from ..solver.model import SlopeDomain


_DIRECTIONS = np.asarray(
    [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)],
    dtype=np.int8,
)


@dataclass(frozen=True)
class ParentOrganizationHierarchy:
    role: str
    bbox_en: tuple[float, float, float, float]
    texel_m: float
    filled_level_m: np.ndarray
    flood_parent: np.ndarray
    drain_rank: np.ndarray
    connected: np.ndarray
    reservoir_owner: np.ndarray


@dataclass(frozen=True)
class OrganizationHierarchy:
    filled_level_m: np.ndarray
    flood_parent: np.ndarray
    drain_rank: np.ndarray
    connected: np.ndarray
    reservoir_owner: np.ndarray
    reservoir_capacity_m3: np.ndarray
    source_role: str


@dataclass(frozen=True)
class HydrologyResult:
    target_a: np.ndarray
    target_b: np.ndarray
    weight_b: np.ndarray
    drop_a_m: np.ndarray
    drop_b_m: np.ndarray
    branch_length_a_m: np.ndarray
    branch_length_b_m: np.ndarray
    discharge_m3_s: np.ndarray
    slope: np.ndarray
    shear_pa: np.ndarray
    descending_order: np.ndarray
    ponded_water_m3: np.ndarray
    reservoir_owner: np.ndarray
    drain_rank: np.ndarray
    source_water_m3_s: float
    exported_water_m3_s: float
    trapped_water_m3_s: float
    maximum_cell_travel_s: float


def _priority_flood_parent(
    domain: SlopeDomain,
    *,
    role: str,
) -> ParentOrganizationHierarchy:
    """Build one immutable hierarchy on an already authorized parent domain."""
    height = np.asarray(domain.height_m, dtype=np.float64)
    active = domain.active
    exits = domain.outlet
    shape = height.shape
    size = height.size
    filled = np.full(shape, np.inf, dtype=np.float64)
    parent = np.full(shape, -1, dtype=np.int32)
    rank = np.full(shape, -1, dtype=np.int32)
    visited = np.zeros(shape, dtype=bool)
    queue: list[tuple[float, int]] = []
    for index in np.flatnonzero(exits & active):
        filled.ravel()[index] = height.ravel()[index]
        heapq.heappush(queue, (float(height.ravel()[index]), int(index)))
        visited.ravel()[index] = True
    if not queue:
        raise ValueError("authorized parent domain has no real outlet")
    cursor = 0
    rows, cols = shape
    while queue:
        level, index = heapq.heappop(queue)
        rank.ravel()[index] = cursor
        cursor += 1
        row, col = divmod(index, cols)
        for dr_raw, dc_raw in _DIRECTIONS:
            rr, cc = row + int(dr_raw), col + int(dc_raw)
            if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                continue
            child = rr * cols + cc
            if visited.ravel()[child] or not active.ravel()[child]:
                continue
            visited.ravel()[child] = True
            child_level = max(float(height.ravel()[child]), level)
            filled.ravel()[child] = child_level
            parent.ravel()[child] = index
            heapq.heappush(queue, (child_level, child))
    # Closed parent components remain disconnected reservoirs; no storage edge
    # is silently promoted to an outlet.
    filled[active & ~visited] = height[active & ~visited]
    if np.any(parent.ravel() >= size):
        raise RuntimeError("priority-flood parent is outside the control grid")
    depressed = active & visited & (filled > height)
    components, component_count = ndimage.label(
        depressed,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    owner = np.full(shape, -1, dtype=np.int64)
    e0, _, _, n1 = domain.bbox_en
    for label in range(1, component_count + 1):
        members = np.flatnonzero(components.ravel() == label)
        member_rows, member_cols = np.unravel_index(members, shape)
        east_index = np.floor(e0 + (member_cols + 0.5) * domain.texel_m).astype(np.int64)
        north_index = np.floor(n1 - (member_rows + 0.5) * domain.texel_m).astype(np.int64)
        # The world-coordinate owner survives canonical/control index changes.
        world_owner = int(np.min(north_index * 10_000_000 + east_index))
        owner.ravel()[members] = world_owner
    return ParentOrganizationHierarchy(
        role=role,
        bbox_en=domain.bbox_en,
        texel_m=domain.texel_m,
        filled_level_m=filled.astype(np.float32),
        flood_parent=parent,
        drain_rank=rank,
        connected=visited,
        reservoir_owner=owner,
    )


def build_authorized_parent_organization(
    domain: SlopeDomain,
    *,
    role: str,
) -> ParentOrganizationHierarchy:
    """Build the production hierarchy only on canonical/control 1 m C0."""
    expected = {"canonical": CANONICAL_BBOX_EN, "control": CONTROL_BBOX_EN}
    if role not in expected or tuple(domain.bbox_en) != expected[role]:
        raise ValueError("parent organization role/bbox is not authorized")
    if domain.texel_m != 1.0:
        raise ValueError("production organization must be built on the 1 m parent")
    return _priority_flood_parent(domain, role=role)


def _parent_fractional_coordinates(
    parent: ParentOrganizationHierarchy,
    authority: FineAuthority,
) -> tuple[np.ndarray, np.ndarray]:
    """Map fine finite-volume controls to 1 m parent sample centers.

    `prolong_structural_4x` places the accepted fine samples at bbox edge plus
    0.5*dx (the parent-center phases start at -3/8). Morphology controls lie
    between adjacent accepted samples, so their project-specific phase is bbox
    edge plus 1.0*dx, not the conventional edge plus 0.5*dx cell phase.
    """
    e0, _, _, n1 = authority.recipe.fine_canvas_bbox_en
    pe0, _, _, pn1 = parent.bbox_en
    fine = authority.recipe.fine_texel_m
    east = e0 + (np.arange(authority.cell_shape[1], dtype=np.float64) + 1.0) * fine
    north = n1 - (np.arange(authority.cell_shape[0], dtype=np.float64) + 1.0) * fine
    cols = (east - (pe0 + 0.5 * parent.texel_m)) / parent.texel_m
    rows = ((pn1 - 0.5 * parent.texel_m) - north) / parent.texel_m
    if (
        rows.min() < 0.0
        or cols.min() < 0.0
        or rows.max() > parent.filled_level_m.shape[0] - 1
        or cols.max() > parent.filled_level_m.shape[1] - 1
    ):
        raise ValueError("fixed fine canvas lacks parent interpolation support")
    return rows, cols


def bind_parent_organization_to_fine(
    parent: ParentOrganizationHierarchy,
    authority: FineAuthority,
) -> OrganizationHierarchy:
    """Transfer one parent hierarchy to the inclusive fine control lattice."""
    rows, cols = _parent_fractional_coordinates(parent, authority)
    nearest_rows = np.rint(rows).astype(np.int32)
    nearest_cols = np.rint(cols).astype(np.int32)
    shape = authority.cell_shape
    routing = authority.routing_cell
    filled = np.empty(shape, dtype=np.float32)
    connected = np.zeros(shape, dtype=bool)
    owner_world = np.full(shape, -1, dtype=np.int64)
    flood_parent = np.full(authority.cell_shape, -1, dtype=np.int32)
    rank = np.full(shape, -1, dtype=np.int64)
    rank_stride = 2 * (shape[0] + shape[1]) + 1
    projection_offset = shape[0] + shape[1]
    parent_cols = parent.filled_level_m.shape[1]
    fine_col = np.arange(shape[1], dtype=np.int32)[None, :]
    block_rows = 128
    for start in range(0, shape[0], block_rows):
        stop = min(shape[0], start + block_rows)
        coordinate_block = np.meshgrid(rows[start:stop], cols, indexing="ij")
        filled_block = ndimage.map_coordinates(
            parent.filled_level_m,
            coordinate_block,
            order=1,
            mode="nearest",
            prefilter=False,
        ).astype(np.float32)
        c0 = authority.c0_node_m
        fine_height = 0.25 * (
            c0[start:stop, :-1]
            + c0[start + 1 : stop + 1, :-1]
            + c0[start:stop, 1:]
            + c0[start + 1 : stop + 1, 1:]
        )
        filled[start:stop] = np.maximum(filled_block, fine_height).astype(np.float32)
        parent_row = nearest_rows[start:stop, None]
        parent_flat = parent_row * parent_cols + nearest_cols[None, :]
        connected_block = parent.connected.ravel()[parent_flat] & routing[start:stop]
        connected[start:stop] = connected_block
        owner_block = parent.reservoir_owner.ravel()[parent_flat].astype(np.int64)
        owner_block[~connected_block] = -1
        owner_world[start:stop] = owner_block
        parent_target = parent.flood_parent.ravel()[parent_flat]
        target_row = np.where(parent_target >= 0, parent_target // parent_cols, parent_row)
        target_col = np.where(parent_target >= 0, parent_target % parent_cols, nearest_cols[None, :])
        direction_row = np.sign(target_row - parent_row).astype(np.int8)
        direction_col = np.sign(target_col - nearest_cols[None, :]).astype(np.int8)
        fine_row = np.arange(start, stop, dtype=np.int32)[:, None]
        fallback_row = fine_row + direction_row
        fallback_col = fine_col + direction_col
        inside = (
            (fallback_row >= 0)
            & (fallback_row < shape[0])
            & (fallback_col >= 0)
            & (fallback_col < shape[1])
        )
        candidate = fallback_row * shape[1] + fallback_col
        candidate_safe = np.clip(candidate, 0, routing.size - 1)
        accepted = (
            inside
            & connected_block
            & ((direction_row != 0) | (direction_col != 0))
            & routing.ravel()[candidate_safe]
        )
        flood_parent[start:stop][accepted] = candidate[accepted]
        parent_rank = parent.drain_rank.ravel()[parent_flat].astype(np.int64)
        projection = (
            fine_row.astype(np.int64) * direction_row
            + fine_col.astype(np.int64) * direction_col
        )
        local_rank = projection_offset - projection
        rank_block = parent_rank * rank_stride + local_rank
        rank_block[~connected_block] = -1
        rank[start:stop] = rank_block
    fallback_indices = np.flatnonzero(flood_parent.ravel() >= 0)
    if np.any(
        rank.ravel()[flood_parent.ravel()[fallback_indices]]
        >= rank.ravel()[fallback_indices]
    ):
        raise RuntimeError("transferred flood-parent direction does not decrease fine rank")

    components: list[tuple[int, np.ndarray]] = []
    for world_owner in np.unique(owner_world[owner_world >= 0]):
        labels, count = ndimage.label(
            owner_world == world_owner,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        for label in range(1, count + 1):
            members = np.flatnonzero(labels.ravel() == label)
            components.append((int(members[0]), members))
    components.sort(key=lambda value: value[0])
    capacities = np.zeros(len(components), dtype=np.float64)
    owner_dense = np.full(authority.cell_shape, -1, dtype=np.int32)
    cell_area = authority.recipe.fine_texel_m**2
    for dense, (_, members) in enumerate(components):
        owner_dense.ravel()[members] = dense
    for start in range(0, shape[0], block_rows):
        stop = min(shape[0], start + block_rows)
        c0 = authority.c0_node_m
        fine_height = 0.25 * (
            c0[start:stop, :-1]
            + c0[start + 1 : stop + 1, :-1]
            + c0[start:stop, 1:]
            + c0[start + 1 : stop + 1, 1:]
        )
        dense = owner_dense[start:stop]
        selected = dense >= 0
        if np.any(selected):
            depth = np.maximum(filled[start:stop].astype(np.float64) - fine_height, 0.0)
            capacities += np.bincount(
                dense[selected],
                weights=depth[selected] * cell_area,
                minlength=capacities.size,
            )
    return OrganizationHierarchy(
        filled_level_m=filled,
        flood_parent=flood_parent,
        drain_rank=rank,
        connected=connected,
        reservoir_owner=owner_dense,
        reservoir_capacity_m3=capacities,
        source_role=parent.role,
    )


def _neighbor(
    indices: np.ndarray,
    directions: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.unravel_index(indices, shape)
    delta = _DIRECTIONS[directions]
    rr = rows + delta[:, 0]
    cc = cols + delta[:, 1]
    inside = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
    target = np.full(indices.shape, -1, dtype=np.int32)
    target[inside] = (rr[inside] * shape[1] + cc[inside]).astype(np.int32)
    return target, inside


def _advances_hierarchy(
    source: np.ndarray,
    target: np.ndarray,
    inside: np.ndarray,
    hierarchy: OrganizationHierarchy,
) -> np.ndarray:
    accepted = inside.copy()
    level = hierarchy.filled_level_m.ravel()
    rank = hierarchy.drain_rank.ravel()
    connected = hierarchy.connected.ravel()
    accepted[inside] &= connected[target[inside]]
    lower_level = np.zeros(source.shape, dtype=bool)
    equal_lower_rank = np.zeros(source.shape, dtype=bool)
    lower_level[inside] = level[target[inside]] < level[source[inside]]
    equal_lower_rank[inside] = (
        (level[target[inside]] == level[source[inside]])
        & (rank[target[inside]] < rank[source[inside]])
    )
    return accepted & (lower_level | equal_lower_rank)


def _build_targets(
    height: np.ndarray,
    authority: FineAuthority,
    hierarchy: OrganizationHierarchy,
    *,
    workspace: Path | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Route along the continuous facet direction, with a non-erosive flood fallback."""
    shape = height.shape
    texel = authority.recipe.fine_texel_m
    active = authority.routing_cell
    exits = authority.outlet_cell | authority.interface_outflow_cell
    target_a = allocate_array(workspace, "route-target-a", shape, np.int32, fill=-1)
    target_b = allocate_array(workspace, "route-target-b", shape, np.int32, fill=-1)
    weight_b = allocate_array(workspace, "route-weight-b", shape, np.float32)
    drop_a = allocate_array(workspace, "route-drop-a-m", shape, np.float32)
    drop_b = allocate_array(workspace, "route-drop-b-m", shape, np.float32)
    length_a = allocate_array(workspace, "route-length-a-m", shape, np.float32)
    length_b = allocate_array(workspace, "route-length-b-m", shape, np.float32)
    d_south, d_east = np.gradient(height.astype(np.float64), texel)
    downhill_east, downhill_south = -d_east, -d_south
    slope = np.hypot(downhill_east, downhill_south).astype(np.float32)
    slope[~active] = 0.0

    source = np.flatnonzero(active).astype(np.int32)
    flat_height = height.ravel().astype(np.float64)
    # Evaluate the planar descent vector in each adjacent-neighbor triangle.
    # Blocks bound all temporary vectors; only the compact winning branches
    # survive. This is triangular-facet D-infinity, not an eight-way lookup or
    # a smoothed discharge field.
    for block_start in range(0, source.size, 524_288):
        block = source[block_start : block_start + 524_288]
        count = block.size
        best = np.zeros(count, dtype=np.float64)
        best_a = np.full(count, -1, dtype=np.int32)
        best_b = np.full(count, -1, dtype=np.int32)
        best_wb = np.zeros(count, dtype=np.float32)
        z0 = flat_height[block]
        for direction in range(8):
            next_direction = (direction + 1) % 8
            da = np.full(count, direction, dtype=np.int8)
            db = np.full(count, next_direction, dtype=np.int8)
            ca, inside_a = _neighbor(block, da, shape)
            cb, inside_b = _neighbor(block, db, shape)
            valid_a = _advances_hierarchy(block, ca, inside_a, hierarchy)
            valid_b = _advances_hierarchy(block, cb, inside_b, hierarchy)
            dr_a, dc_a = (int(value) for value in _DIRECTIONS[direction])
            dr_b, dc_b = (int(value) for value in _DIRECTIONS[next_direction])
            distance_a = texel * (np.sqrt(2.0) if dr_a and dc_a else 1.0)
            distance_b = texel * (np.sqrt(2.0) if dr_b and dc_b else 1.0)
            edge_a = np.zeros(count, dtype=np.float64)
            edge_b = np.zeros(count, dtype=np.float64)
            edge_a[valid_a] = np.maximum(
                (z0[valid_a] - flat_height[ca[valid_a]]) / distance_a, 0.0
            )
            edge_b[valid_b] = np.maximum(
                (z0[valid_b] - flat_height[cb[valid_b]]) / distance_b, 0.0
            )
            facet_slope = np.maximum(edge_a, edge_b)
            facet_a = np.where(edge_a >= edge_b, ca, cb).astype(np.int32)
            facet_b = np.full(count, -1, dtype=np.int32)
            facet_wb = np.zeros(count, dtype=np.float32)
            both = valid_a & valid_b
            if np.any(both):
                determinant = dc_a * dr_b - dc_b * dr_a
                dz_a = flat_height[ca[both]] - z0[both]
                dz_b = flat_height[cb[both]] - z0[both]
                grad_east = (dz_a * dr_b - dz_b * dr_a) / (texel * determinant)
                grad_south = (dc_a * dz_b - dc_b * dz_a) / (texel * determinant)
                down_east, down_south = -grad_east, -grad_south
                alpha = (down_east * dr_b - down_south * dc_b) / determinant
                beta = (dc_a * down_south - dr_a * down_east) / determinant
                inside_facet = (alpha >= 0.0) & (beta >= 0.0)
                positions = np.flatnonzero(both)[inside_facet]
                if positions.size:
                    plane_slope = np.hypot(down_east[inside_facet], down_south[inside_facet])
                    positive = plane_slope > 0.0
                    positions = positions[positive]
                    if positions.size:
                        aa = alpha[inside_facet][positive]
                        bb = beta[inside_facet][positive]
                        facet_slope[positions] = plane_slope[positive]
                        facet_a[positions] = ca[positions]
                        facet_b[positions] = cb[positions]
                        facet_wb[positions] = (bb / (aa + bb)).astype(np.float32)
            better = facet_slope > best
            best[better] = facet_slope[better]
            best_a[better] = facet_a[better]
            best_b[better] = facet_b[better]
            best_wb[better] = facet_wb[better]
        routed = best > 0.0
        target_a.ravel()[block[routed]] = best_a[routed]
        target_b.ravel()[block[routed]] = best_b[routed]
        weight_b.ravel()[block[routed]] = best_wb[routed]

    # If the instantaneous gradient cannot advance the immutable hierarchy,
    # drain by the priority-flood parent. It carries water but has zero stream
    # power when the actual branch is flat/uphill.
    continuous = target_a.ravel()[source] >= 0
    fallback = ~continuous & hierarchy.connected.ravel()[source] & ~exits.ravel()[source]
    parent = hierarchy.flood_parent.ravel()[source[fallback]]
    has_parent = parent >= 0
    fallback_source = source[fallback][has_parent]
    target_a.ravel()[fallback_source] = parent[has_parent]

    for target, drop, length in (
        (target_a, drop_a, length_a),
        (target_b, drop_b, length_b),
    ):
        selected = target.ravel() >= 0
        indices = np.flatnonzero(selected)
        destinations = target.ravel()[indices]
        drop.ravel()[indices] = np.maximum(
            flat_height[indices] - flat_height[destinations], 0.0
        ).astype(np.float32)
        dr = np.abs(indices // shape[1] - destinations // shape[1])
        dc = np.abs(indices % shape[1] - destinations % shape[1])
        length.ravel()[indices] = np.where((dr > 0) & (dc > 0), texel * np.sqrt(2.0), texel)
    target_a[exits] = -2
    target_b[exits] = -1
    weight_b[exits] = 0.0
    drop_a[exits] = 0.0
    drop_b[exits] = 0.0
    length_a[exits] = 0.0
    length_b[exits] = 0.0
    return target_a, target_b, weight_b, drop_a, drop_b, length_a, length_b, slope


def solve_hydrology(
    authority: FineAuthority,
    hierarchy: OrganizationHierarchy,
    state: MorphodynamicState,
    material: MaterialFields,
    boundary: NestedBoundaryFlux,
    config: MorphodynamicsConfig,
    *,
    workspace: Path | None,
) -> HydrologyResult:
    state.validate(authority)
    boundary.validate(authority.cell_shape)
    height_node = allocate_array(
        workspace,
        "hydrology-height-node",
        authority.c0_node_m.shape,
        np.float64,
    )
    compose_height_nodes(
        authority,
        state,
        material.bulk_density_kg_m3,
        config.deposit_bulk_density_kg_m3,
        output=height_node,
    )
    height = allocate_array(
        workspace,
        "hydrology-height-cell",
        authority.cell_shape,
        np.float32,
    )
    for start in range(0, authority.cell_shape[0], 256):
        stop = min(authority.cell_shape[0], start + 256)
        height[start:stop] = 0.25 * (
            height_node[start:stop, :-1]
            + height_node[start + 1 : stop + 1, :-1]
            + height_node[start:stop, 1:]
            + height_node[start + 1 : stop + 1, 1:]
        )
    (
        target_a,
        target_b,
        weight_b,
        drop_a,
        drop_b,
        length_a,
        length_b,
        slope,
    ) = _build_targets(height, authority, hierarchy, workspace=workspace)
    active = authority.routing_cell
    exits = authority.outlet_cell | authority.interface_outflow_cell
    cell_area = authority.recipe.fine_texel_m**2
    vegetation = authority.cell_average(authority.vegetation_node)
    seep = authority.cell_average(authority.seep_node)
    source = (
        (config.rainfall_depth_m / config.event_duration_s)
        * material.runoff_fraction
        * np.clip(1.0 - 0.78 * vegetation, 0.08, 1.0)
        + material.seep_flux_m_s * seep
    ) * cell_area
    source = source.astype(np.float64)
    source[~authority.hydrologic_source_active_cell] = 0.0
    source += boundary.water_inflow_m3_s
    if np.any((boundary.water_inflow_m3_s > 0.0) & ~active):
        raise ValueError("parent water enters outside fine routing support")

    discharge = allocate_array(workspace, "water-discharge-m3-s", authority.cell_shape, np.float64)
    discharge.fill(0.0)
    ponded = allocate_array(workspace, "ponded-water-m3", authority.cell_shape, np.float64)
    order = np.flatnonzero(active.ravel())
    order = order[np.argsort(hierarchy.drain_rank.ravel()[order])[::-1]].astype(np.int32)
    ta, tb = target_a.ravel(), target_b.ravel()
    wb = weight_b.ravel().astype(np.float64)
    q = source.ravel().copy()
    exported = 0.0
    trapped = 0.0
    reservoir_remaining = hierarchy.reservoir_capacity_m3.astype(np.float64, copy=True)
    reservoir_owner = hierarchy.reservoir_owner.ravel()
    for index in order:
        owner = int(reservoir_owner[index])
        retained = (
            min(q[index] * config.event_duration_s, reservoir_remaining[owner])
            if owner >= 0
            else 0.0
        )
        if owner >= 0:
            reservoir_remaining[owner] -= retained
        ponded.ravel()[index] = retained
        trapped += retained / config.event_duration_s
        moving = q[index] - retained / config.event_duration_s
        discharge.ravel()[index] = moving
        if exits.ravel()[index]:
            exported += moving
            continue
        if ta[index] >= 0:
            q[ta[index]] += moving * (1.0 - wb[index])
        if tb[index] >= 0:
            q[tb[index]] += moving * wb[index]
        if ta[index] < 0 and tb[index] < 0:
            trapped += moving
    source_total = float(np.sum(source, dtype=np.float64))
    error = source_total - exported - trapped
    if abs(error) > max(1e-10, source_total * 2e-8):
        raise RuntimeError(f"fine water ledger differs by {error} m3/s")

    unit_q = discharge / authority.recipe.fine_texel_m
    hydraulic_slope = np.maximum(slope, 1e-6)
    depth = np.where(
        active & (unit_q > 0.0),
        (unit_q * config.manning_n_s_m13 / np.sqrt(hydraulic_slope)) ** (3.0 / 5.0),
        0.0,
    )
    shear = (config.water_density_kg_m3 * config.gravity_m_s2 * depth * slope).astype(np.float32)
    velocity = np.where(
        depth > 0.0,
        np.power(depth, 2.0 / 3.0) * np.sqrt(hydraulic_slope) / config.manning_n_s_m13,
        0.0,
    )
    lengths = np.maximum(length_a, length_b)
    travel = np.divide(
        lengths,
        velocity,
        out=np.full(authority.cell_shape, np.inf, dtype=np.float64),
        where=velocity > 0.0,
    )
    finite_travel = travel[active & np.isfinite(travel)]
    return HydrologyResult(
        target_a=target_a,
        target_b=target_b,
        weight_b=weight_b,
        drop_a_m=drop_a,
        drop_b_m=drop_b,
        branch_length_a_m=length_a,
        branch_length_b_m=length_b,
        discharge_m3_s=discharge,
        slope=slope,
        shear_pa=shear,
        descending_order=order,
        ponded_water_m3=ponded,
        reservoir_owner=hierarchy.reservoir_owner,
        drain_rank=hierarchy.drain_rank,
        source_water_m3_s=source_total,
        exported_water_m3_s=exported,
        trapped_water_m3_s=trapped,
        maximum_cell_travel_s=float(np.max(finite_travel, initial=0.0)),
    )
