"""Post-gate connectivity audit for direct Hovi preview cells."""
from __future__ import annotations

from typing import Any

import numpy as np

from .model import GridSpec


def audit_direct_topology(
    direct_candidate_mask: np.ndarray,
    selected_height_m: np.ndarray,
    structural_reference_m: np.ndarray,
    slope_x: np.ndarray,
    slope_y: np.ndarray,
    grid: GridSpec,
    *,
    edge_residual_limit_m: float,
    minimum_component_cells: int,
    structural_anchor_residual_m: float,
    minimum_anchor_fraction: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Reject islands supported only by candidates that failed the direct-cell gates."""
    direct = np.asarray(direct_candidate_mask, dtype=bool)
    height = np.asarray(selected_height_m, dtype=np.float64)
    reference = np.asarray(structural_reference_m, dtype=np.float64)
    tangent_x = np.asarray(slope_x, dtype=np.float64)
    tangent_y = np.asarray(slope_y, dtype=np.float64)
    expected_shape = (grid.height, grid.width)
    if any(
        value.shape != expected_shape
        for value in (direct, height, reference, tangent_x, tangent_y)
    ):
        raise ValueError("Hovi topology audit inputs do not match the final grid")
    direct &= (
        np.isfinite(height)
        & np.isfinite(reference)
        & np.isfinite(tangent_x)
        & np.isfinite(tangent_y)
    )

    def neighbors(row: int, column: int):
        if row > 0:
            yield row - 1, column
        if row + 1 < grid.height:
            yield row + 1, column
        if column > 0:
            yield row, column - 1
        if column + 1 < grid.width:
            yield row, column + 1

    def compatible(row: int, column: int, other_row: int, other_column: int) -> bool:
        dx = (other_column - column) * grid.resolution_m
        dy = (other_row - row) * grid.resolution_m
        expected_dz = (
            0.5 * (tangent_x[row, column] + tangent_x[other_row, other_column]) * dx
            + 0.5 * (tangent_y[row, column] + tangent_y[other_row, other_column]) * dy
        )
        residual = height[other_row, other_column] - height[row, column] - expected_dz
        return abs(float(residual)) <= edge_residual_limit_m

    visited = np.zeros(expected_shape, dtype=bool)
    components: list[np.ndarray] = []
    for start_row, start_column in zip(*np.nonzero(direct), strict=True):
        if visited[start_row, start_column]:
            continue
        visited[start_row, start_column] = True
        stack = [(int(start_row), int(start_column))]
        members: list[int] = []
        while stack:
            row, column = stack.pop()
            members.append(row * grid.width + column)
            for other_row, other_column in neighbors(row, column):
                if (
                    visited[other_row, other_column]
                    or not direct[other_row, other_column]
                    or not compatible(row, column, other_row, other_column)
                ):
                    continue
                visited[other_row, other_column] = True
                stack.append((other_row, other_column))
        components.append(np.asarray(members, dtype=np.int64))
    components.sort(key=lambda item: item.size, reverse=True)

    accepted = np.zeros(expected_shape, dtype=bool)
    component_records = []
    for index, members in enumerate(components):
        rows = members // grid.width
        columns = members % grid.width
        anchored = (
            np.abs(height[rows, columns] - reference[rows, columns])
            <= structural_anchor_residual_m
        )
        anchor_fraction = float(np.mean(anchored))
        keep = index == 0 or (
            members.size >= minimum_component_cells
            and anchor_fraction >= minimum_anchor_fraction
        )
        if keep:
            accepted[rows, columns] = True
        component_records.append(
            {
                "rank": index + 1,
                "cells": int(members.size),
                "area_m2": float(members.size * grid.resolution_m**2),
                "structural_anchor_fraction": anchor_fraction,
                "accepted_as_direct": keep,
                "basis": "dominant_component" if index == 0 else "structural_anchor",
            }
        )
    rejected = direct & ~accepted
    diagnostics = {
        "policy": "tangent_compatible_direct_components_with_structural_anchor",
        "candidate_direct_cells": int(np.count_nonzero(direct)),
        "accepted_direct_cells": int(np.count_nonzero(accepted)),
        "rejected_unresolved_cells": int(np.count_nonzero(rejected)),
        "component_count": len(components),
        "edge_residual_limit_m": edge_residual_limit_m,
        "minimum_component_cells": minimum_component_cells,
        "structural_anchor_residual_m": structural_anchor_residual_m,
        "minimum_anchor_fraction": minimum_anchor_fraction,
        "components": component_records[:64],
    }
    return accepted, rejected, diagnostics
