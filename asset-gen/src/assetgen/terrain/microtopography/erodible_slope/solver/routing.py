"""Continuous-direction whole-domain routing with explicit physical boundaries."""
from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np

from .model import SlopeDomain

_DIRECTIONS = np.asarray(
    [
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
    ],
    dtype=np.int8,
)


@dataclass(frozen=True)
class RoutingResult:
    downhill_east: np.ndarray
    downhill_south: np.ndarray
    slope: np.ndarray
    target_a: np.ndarray
    target_b: np.ndarray
    weight_a: np.ndarray
    weight_b: np.ndarray
    water_m3: np.ndarray
    contributing_area_m2: np.ndarray
    exported_water_m3: float
    trapped_water_m3: float
    source_water_m3: float
    ponded_water_m3: np.ndarray
    spilled_water_m3: np.ndarray
    depression_capacity_m3: np.ndarray
    closed_depression_cells: int


def _gradient(height: np.ndarray, texel_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Return downhill east/south components in world-axis units."""
    d_south, d_east = np.gradient(np.asarray(height, dtype=np.float64), texel_m)
    return -d_east, -d_south


def _neighbor_indices(
    rows: np.ndarray,
    cols: np.ndarray,
    direction: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta = _DIRECTIONS[direction]
    rr = rows + delta[:, 0]
    cc = cols + delta[:, 1]
    inside = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
    return rr, cc, inside


def _depression_spills(
    domain: SlopeDomain,
    target_a: np.ndarray,
    target_b: np.ndarray,
    weight_a: np.ndarray,
    weight_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Derive finite reservoir capacities and an outlet-rooted spill tree.

    Membership and saddle levels come from immutable measured height. This
    function never constructs or returns a modified/filled terrain surface.
    """
    height = domain.height_m.ravel()
    active = domain.active.ravel()
    outlet = domain.outlet.ravel()
    primary = np.where(weight_b > weight_a, target_b, target_a).ravel()
    labels = np.full(height.size, -3, dtype=np.int64)
    order = np.flatnonzero(active)
    order = order[np.argsort(height[order])]
    for index in order:
        if outlet[index]:
            labels[index] = -2
        elif primary[index] >= 0:
            labels[index] = labels[primary[index]]
        else:
            labels[index] = index
    sinks = np.unique(labels[labels >= 0])
    spill_target = np.full(height.size, -1, dtype=np.int64)
    capacity = np.zeros(height.size, dtype=np.float64)
    spill_order = np.full(height.size, np.inf, dtype=np.float64)
    spill_depth = np.full(height.size, -1, dtype=np.int32)
    if not sinks.size:
        return spill_target, capacity, spill_order, spill_depth, 0

    rows, cols = domain.height_m.shape
    label_grid = labels.reshape((rows, cols))
    source_parts: list[np.ndarray] = []
    destination_parts: list[np.ndarray] = []
    saddle_parts: list[np.ndarray] = []
    for dr_raw, dc_raw in _DIRECTIONS:
        dr, dc = int(dr_raw), int(dc_raw)
        source_rows = slice(max(0, -dr), min(rows, rows - dr))
        source_cols = slice(max(0, -dc), min(cols, cols - dc))
        target_rows = slice(max(0, dr), min(rows, rows + dr))
        target_cols = slice(max(0, dc), min(cols, cols + dc))
        source_label = label_grid[source_rows, source_cols]
        destination_label = label_grid[target_rows, target_cols]
        accepted = (
            (source_label >= 0)
            & (source_label != destination_label)
            & ((destination_label >= 0) | (destination_label == -2))
        )
        if not np.any(accepted):
            continue
        source_parts.append(source_label[accepted])
        destination_parts.append(destination_label[accepted])
        saddle_parts.append(
            np.maximum(
                domain.height_m[source_rows, source_cols][accepted],
                domain.height_m[target_rows, target_cols][accepted],
            )
        )

    spill_elevation = np.full(height.size, np.nan, dtype=np.float64)
    if source_parts:
        source = np.concatenate(source_parts)
        destination = np.concatenate(destination_parts)
        saddle = np.concatenate(saddle_parts)
        sink_to_node = {int(value): index for index, value in enumerate(sinks)}
        outlet_node = len(sinks)
        edge_minimum: dict[tuple[int, int], float] = {}
        for source_sink, destination_sink, edge_saddle in zip(
            source, destination, saddle, strict=True
        ):
            source_node = sink_to_node[int(source_sink)]
            destination_node = (
                outlet_node
                if destination_sink == -2
                else sink_to_node[int(destination_sink)]
            )
            if source_node == destination_node:
                continue
            edge = (
                min(source_node, destination_node),
                max(source_node, destination_node),
            )
            edge_minimum[edge] = min(
                edge_minimum.get(edge, np.inf), float(edge_saddle)
            )
        adjacency: list[list[tuple[int, float]]] = [
            [] for _ in range(outlet_node + 1)
        ]
        for (left, right), edge_saddle in edge_minimum.items():
            adjacency[left].append((right, edge_saddle))
            adjacency[right].append((left, edge_saddle))
        minimax = np.full(outlet_node + 1, np.inf, dtype=np.float64)
        parent = np.full(outlet_node + 1, -1, dtype=np.int64)
        parent_saddle = np.full(outlet_node + 1, np.nan, dtype=np.float64)
        minimax[outlet_node] = -np.inf
        queue: list[tuple[float, int]] = [(-np.inf, outlet_node)]
        while queue:
            cost, node = heapq.heappop(queue)
            if cost != minimax[node]:
                continue
            for neighbor, edge_saddle in adjacency[node]:
                candidate = max(cost, edge_saddle)
                if candidate < minimax[neighbor]:
                    minimax[neighbor] = candidate
                    parent[neighbor] = node
                    parent_saddle[neighbor] = edge_saddle
                    heapq.heappush(queue, (candidate, neighbor))
        node_depth = np.full(outlet_node + 1, -1, dtype=np.int32)
        node_depth[outlet_node] = 0
        for start_node in range(outlet_node):
            path: list[int] = []
            node = start_node
            while node >= 0 and node_depth[node] < 0:
                path.append(node)
                node = int(parent[node])
            depth = int(node_depth[node]) if node >= 0 else -1
            for path_node in reversed(path):
                depth = depth + 1 if depth >= 0 else -1
                node_depth[path_node] = depth
        for node, sink in enumerate(sinks):
            if parent[node] < 0:
                continue
            destination_node = int(parent[node])
            spill_target[sink] = (
                -2 if destination_node == outlet_node else sinks[destination_node]
            )
            spill_elevation[sink] = parent_saddle[node]
            spill_order[sink] = minimax[node]
            spill_depth[sink] = node_depth[node]

    basin_cells = np.flatnonzero(labels >= 0)
    bounded = np.zeros(height.size, dtype=bool)
    bounded[basin_cells] = np.isfinite(spill_elevation[labels[basin_cells]])
    storage_depth = np.zeros(height.size, dtype=np.float64)
    storage_depth[bounded] = np.maximum(
        spill_elevation[labels[bounded]] - height[bounded], 0.0
    )
    capacity_by_sink = np.bincount(
        labels[bounded],
        weights=storage_depth[bounded] * domain.texel_m**2,
        minlength=height.size,
    )
    capacity[sinks] = capacity_by_sink[sinks]
    closed = int(np.count_nonzero(spill_target[sinks] == -1))
    return spill_target, capacity, spill_order, spill_depth, closed


def route_continuous(
    domain: SlopeDomain,
    *,
    local_water_depth_m: np.ndarray,
) -> RoutingResult:
    """Route water between the two facets bracketing the continuous gradient angle.

    Cells first route to strictly lower active neighbors. Strict sinks retain
    finite event water below immutable-height basin saddles and spill overflow
    through an outlet-rooted basin tree; no terrain is filled or breached.
    """
    shape = domain.height_m.shape
    if local_water_depth_m.shape != shape:
        raise ValueError("local water field differs from physical domain")
    if not np.isfinite(local_water_depth_m).all() or np.any(local_water_depth_m < 0):
        raise ValueError("local water depth must be finite and nonnegative")

    active = domain.active
    east, south = _gradient(domain.height_m, domain.texel_m)
    east = np.where(active, east, 0.0)
    south = np.where(active, south, 0.0)
    slope = np.hypot(east, south)
    angle = np.mod(np.arctan2(south, east), 2.0 * np.pi)
    sector = angle / (np.pi / 4.0)
    direction_a = np.floor(sector).astype(np.int8) % 8
    fraction_b = sector - np.floor(sector)
    direction_b = (direction_a + 1) % 8

    flat_indices = np.flatnonzero(active)
    rows, cols = np.unravel_index(flat_indices, shape)
    rr_a, cc_a, inside_a = _neighbor_indices(rows, cols, direction_a[active], shape)
    rr_b, cc_b, inside_b = _neighbor_indices(rows, cols, direction_b[active], shape)
    target_a = np.full(domain.height_m.size, -1, dtype=np.int64)
    target_b = np.full(domain.height_m.size, -1, dtype=np.int64)
    weight_a = np.zeros(domain.height_m.size, dtype=np.float64)
    weight_b = np.zeros(domain.height_m.size, dtype=np.float64)
    source_height = domain.height_m[active]

    valid_a = inside_a.copy()
    valid_b = inside_b.copy()
    if np.any(inside_a):
        valid_a[inside_a] &= active[rr_a[inside_a], cc_a[inside_a]]
        valid_a[inside_a] &= (
            domain.height_m[rr_a[inside_a], cc_a[inside_a]]
            < source_height[inside_a]
        )
    if np.any(inside_b):
        valid_b[inside_b] &= active[rr_b[inside_b], cc_b[inside_b]]
        valid_b[inside_b] &= (
            domain.height_m[rr_b[inside_b], cc_b[inside_b]]
            < source_height[inside_b]
        )
    raw_a = (1.0 - fraction_b[active]) * valid_a
    raw_b = fraction_b[active] * valid_b
    total = raw_a + raw_b

    # When the continuous direction brackets no lower facet, select the steepest
    # strictly lower neighbor. This resolves facet degeneracy without filling pits.
    unresolved = total <= 0.0
    if np.any(unresolved):
        unresolved_rows = rows[unresolved]
        unresolved_cols = cols[unresolved]
        best_drop = np.zeros(unresolved_rows.size, dtype=np.float64)
        best_direction = np.full(unresolved_rows.size, -1, dtype=np.int8)
        for direction, (dr, dc) in enumerate(_DIRECTIONS):
            rr = unresolved_rows + int(dr)
            cc = unresolved_cols + int(dc)
            inside = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
            candidates = np.zeros_like(inside)
            candidates[inside] = active[rr[inside], cc[inside]]
            distance = domain.texel_m * (np.sqrt(2.0) if dr and dc else 1.0)
            drop = np.zeros_like(best_drop)
            drop[candidates] = (
                domain.height_m[unresolved_rows[candidates], unresolved_cols[candidates]]
                - domain.height_m[rr[candidates], cc[candidates]]
            ) / distance
            better = drop > best_drop
            best_drop[better] = drop[better]
            best_direction[better] = direction
        resolved = best_direction >= 0
        unresolved_positions = np.flatnonzero(unresolved)
        positions = unresolved_positions[resolved]
        selected = _DIRECTIONS[best_direction[resolved]]
        rr_a[positions] = rows[positions] + selected[:, 0]
        cc_a[positions] = cols[positions] + selected[:, 1]
        raw_a[positions] = 1.0
        raw_b[positions] = 0.0
        total[positions] = 1.0

    routable = total > 0.0
    raw_a[routable] /= total[routable]
    raw_b[routable] /= total[routable]
    a_used = raw_a > 0.0
    b_used = raw_b > 0.0
    target_a[flat_indices[a_used]] = np.ravel_multi_index(
        (rr_a[a_used], cc_a[a_used]), shape
    )
    target_b[flat_indices[b_used]] = np.ravel_multi_index(
        (rr_b[b_used], cc_b[b_used]), shape
    )
    weight_a[flat_indices] = raw_a
    weight_b[flat_indices] = raw_b
    (
        spill_target,
        depression_capacity,
        spill_order,
        spill_depth,
        closed_depressions,
    ) = _depression_spills(
        domain,
        target_a.reshape(shape),
        target_b.reshape(shape),
        weight_a.reshape(shape),
        weight_b.reshape(shape),
    )

    cell_area = domain.texel_m**2
    local_water = np.where(active, local_water_depth_m * cell_area, 0.0)
    water = local_water.ravel().astype(np.float64, copy=True)
    water += domain.upstream_water_m3.ravel()
    contributing_area = np.where(active, cell_area, 0.0).ravel()
    order = flat_indices[np.argsort(domain.height_m.ravel()[flat_indices])[::-1]]
    exported = 0.0
    trapped = 0.0
    ponded = np.zeros(domain.height_m.size, dtype=np.float64)
    spilled = np.zeros(domain.height_m.size, dtype=np.float64)
    outlet_flat = domain.outlet.ravel()
    for index in order:
        volume = water[index]
        area = contributing_area[index]
        if outlet_flat[index]:
            exported += volume
            continue
        ta = target_a[index]
        tb = target_b[index]
        wa = weight_a[index]
        wb = weight_b[index]
        if ta < 0 and tb < 0:
            continue
        if ta >= 0:
            water[ta] += volume * wa
            contributing_area[ta] += area * wa
        if tb >= 0:
            water[tb] += volume * wb
            contributing_area[tb] += area * wb

    sink_indices = np.flatnonzero(
        active.ravel()
        & ~outlet_flat
        & (target_a < 0)
        & (target_b < 0)
    )
    basin_order = sorted(
        sink_indices,
        key=lambda index: (spill_order[index], spill_depth[index]),
        reverse=True,
    )
    for index in basin_order:
        volume = water[index]
        area = contributing_area[index]
        destination = spill_target[index]
        retained = (
            min(volume, depression_capacity[index])
            if destination != -1
            else volume
        )
        overflow = volume - retained
        ponded[index] = retained
        spilled[index] = overflow
        trapped += retained
        if destination == -2:
            exported += overflow
        elif destination >= 0:
            water[destination] += overflow
            contributing_area[destination] += area

    source_water = float(np.sum(local_water) + np.sum(domain.upstream_water_m3))
    accounting_error = source_water - exported - trapped
    if abs(accounting_error) > max(1e-10, source_water * 1e-12):
        raise RuntimeError(f"water routing is not conservative: {accounting_error} m3")
    return RoutingResult(
        downhill_east=east,
        downhill_south=south,
        slope=slope,
        target_a=target_a.reshape(shape),
        target_b=target_b.reshape(shape),
        weight_a=weight_a.reshape(shape),
        weight_b=weight_b.reshape(shape),
        water_m3=water.reshape(shape),
        contributing_area_m2=contributing_area.reshape(shape),
        exported_water_m3=exported,
        trapped_water_m3=trapped,
        source_water_m3=source_water,
        ponded_water_m3=ponded.reshape(shape),
        spilled_water_m3=spilled.reshape(shape),
        depression_capacity_m3=depression_capacity.reshape(shape),
        closed_depression_cells=closed_depressions,
    )
