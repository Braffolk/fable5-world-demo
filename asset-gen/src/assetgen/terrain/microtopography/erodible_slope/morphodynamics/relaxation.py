"""Internal, blockwise mass-conservative colluvium relaxation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .material_field import MaterialFields
from .state import (
    FineAuthority,
    MorphodynamicState,
    MorphodynamicsConfig,
    allocate_array,
    compose_height_nodes,
)


@dataclass(frozen=True)
class RelaxationEvent:
    relocated_kg: float
    mass_error_kg: float


def _pair_blocks(shape: tuple[int, int], dr: int, dc: int, block_rows: int = 128):
    rows, cols = shape
    row_start, row_stop = max(0, -dr), min(rows, rows - dr)
    col_start, col_stop = max(0, -dc), min(cols, cols - dc)
    for start in range(row_start, row_stop, block_rows):
        stop = min(row_stop, start + block_rows)
        rr = np.arange(start, stop, dtype=np.int32)[:, None]
        cc = np.arange(col_start, col_stop, dtype=np.int32)[None, :]
        left = np.broadcast_to(rr * cols + cc, (stop - start, col_stop - col_start)).ravel()
        right = left + dr * cols + dc
        yield left, right


def _demand(
    left: np.ndarray,
    right: np.ndarray,
    height: np.ndarray,
    repose: np.ndarray,
    eligible: np.ndarray,
    distance_m: float,
    density_kg_m3: float,
    cell_area_m2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    difference = height[left] - height[right]
    high = np.where(difference >= 0.0, left, right)
    low = np.where(difference >= 0.0, right, left)
    threshold = np.minimum(repose[left], repose[right]) * distance_m
    excess = np.maximum(np.abs(difference) - threshold, 0.0)
    demand = 0.5 * excess * density_kg_m3 * cell_area_m2
    demand[~(eligible[high] & eligible[low])] = 0.0
    return high, low, demand


def relax_colluvium(
    authority: FineAuthority,
    state: MorphodynamicState,
    material: MaterialFields,
    config: MorphodynamicsConfig,
    *,
    workspace: Path | None,
) -> RelaxationEvent:
    """Move existing deposits to repose with two recomputed streaming passes."""
    before = float(np.sum(state.deposited_colluvium_kg, dtype=np.float64))
    if before <= 0.0:
        return RelaxationEvent(0.0, 0.0)
    node_height = allocate_array(
        workspace,
        "relax-height-node",
        authority.c0_node_m.shape,
        np.float64,
    )
    compose_height_nodes(
        authority,
        state,
        material.bulk_density_kg_m3,
        config.deposit_bulk_density_kg_m3,
        output=node_height,
    )
    shape = authority.cell_shape
    height = allocate_array(workspace, "relax-height-cell", shape, np.float64)
    for start in range(0, shape[0], 256):
        stop = min(shape[0], start + 256)
        height[start:stop] = 0.25 * (
            node_height[start:stop, :-1]
            + node_height[start + 1 : stop + 1, :-1]
            + node_height[start:stop, 1:]
            + node_height[start + 1 : stop + 1, 1:]
        )
    flat_height = height.ravel()
    repose = material.repose_gradient.ravel().astype(np.float64)
    eligible = authority.form_active_cell.ravel()
    available = state.deposited_colluvium_kg.ravel()
    outgoing_demand = allocate_array(workspace, "relax-demand", shape, np.float64)
    outgoing = allocate_array(workspace, "relax-outgoing", shape, np.float64)
    incoming = allocate_array(workspace, "relax-incoming", shape, np.float64)
    cell_area = authority.recipe.fine_texel_m**2
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    for dr, dc in directions:
        distance = authority.recipe.fine_texel_m * (np.sqrt(2.0) if dr and dc else 1.0)
        for left, right in _pair_blocks(shape, dr, dc):
            high, _, demand = _demand(
                left,
                right,
                flat_height,
                repose,
                eligible,
                distance,
                config.deposit_bulk_density_kg_m3,
                cell_area,
            )
            np.add.at(outgoing_demand.ravel(), high, demand)
    for dr, dc in directions:
        distance = authority.recipe.fine_texel_m * (np.sqrt(2.0) if dr and dc else 1.0)
        for left, right in _pair_blocks(shape, dr, dc):
            high, low, demand = _demand(
                left,
                right,
                flat_height,
                repose,
                eligible,
                distance,
                config.deposit_bulk_density_kg_m3,
                cell_area,
            )
            scale = np.minimum(
                1.0,
                np.divide(
                    available[high],
                    outgoing_demand.ravel()[high],
                    out=np.ones(demand.shape, dtype=np.float64),
                    where=outgoing_demand.ravel()[high] > 0.0,
                ),
            )
            moved = demand * scale
            np.add.at(outgoing.ravel(), high, moved)
            np.add.at(incoming.ravel(), low, moved)
    available += incoming.ravel() - outgoing.ravel()
    np.maximum(available, 0.0, out=available)
    moved_out = float(np.sum(outgoing, dtype=np.float64))
    moved_in = float(np.sum(incoming, dtype=np.float64))
    error = moved_out - moved_in
    tolerance = max(1e-8, moved_out * config.sediment_relative_tolerance)
    if abs(error) > tolerance:
        raise RuntimeError(f"colluvium relaxation ledger differs by {error} kg")
    after = float(np.sum(state.deposited_colluvium_kg, dtype=np.float64))
    if abs(after - before) > max(1e-8, before * config.sediment_relative_tolerance):
        raise RuntimeError("colluvium relaxation changed total solid inventory")
    state.ledger.thermally_relocated_kg += moved_out
    state.validate(authority)
    return RelaxationEvent(moved_out, error)
