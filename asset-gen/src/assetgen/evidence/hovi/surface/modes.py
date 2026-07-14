"""Vertical-mode evidence and conservative multiscale 2.5D sheet selection."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from .histogram import map_grid_centers
from .model import (
    GridSpec,
    ModeEvidence,
    SheetSelection,
    SurfaceConfig,
    TangentPrediction,
)


def _window_sum(values: np.ndarray, radius: int) -> np.ndarray:
    output = values.astype(np.uint64, copy=True)
    for offset in range(1, radius + 1):
        output[:, offset:] += values[:, :-offset]
        output[:, :-offset] += values[:, offset:]
    return output


def extract_vertical_modes(
    counts: np.ndarray,
    *,
    z_bin_m: float,
    z_base_m: float | np.ndarray,
    mode_count: int,
    mode_radius_bins: int,
    min_mode_support: int,
    underburden_clearance_m: float,
    block_cells: int = 768,
) -> ModeEvidence:
    """Extract compact vertical modes; absolute elevation rank is never a score term."""
    histogram = np.asarray(counts)
    if histogram.ndim != 2 or not np.issubdtype(histogram.dtype, np.integer):
        raise ValueError("Hovi vertical modes require a cell-by-height integer histogram")
    cell_count, z_bins = histogram.shape
    if mode_count <= 0 or mode_count >= z_bins:
        raise ValueError("Hovi vertical mode_count is invalid")
    base = np.asarray(z_base_m, dtype=np.float64)
    if base.ndim == 0:
        base = np.full(cell_count, float(base), dtype=np.float64)
    else:
        base = base.reshape(-1)
    if base.size != cell_count:
        raise ValueError("Hovi per-cell vertical origin does not match histogram")

    height = np.full((cell_count, mode_count), np.nan, dtype=np.float32)
    support_out = np.zeros((cell_count, mode_count), dtype=np.uint32)
    prominence_out = np.zeros((cell_count, mode_count), dtype=np.float32)
    underburden_out = np.ones((cell_count, mode_count), dtype=np.float32)
    quality_out = np.zeros((cell_count, mode_count), dtype=np.float32)
    broad_radius = max(mode_radius_bins * 4, mode_radius_bins + 2)
    clearance_bins = max(1, int(np.ceil(underburden_clearance_m / z_bin_m)))
    bin_axis = np.arange(z_bins, dtype=np.int64)

    for start in range(0, cell_count, block_cells):
        stop = min(start + block_cells, cell_count)
        raw = histogram[start:stop]
        support = _window_sum(raw, mode_radius_bins)
        broad = _window_sum(raw, broad_radius)
        prefix = np.cumsum(raw, axis=1, dtype=np.uint64)
        below_index = bin_axis - clearance_bins
        below = np.zeros_like(support)
        valid_below = below_index >= 0
        below[:, valid_below] = prefix[:, below_index[valid_below]]
        underburden = below / np.maximum(below + support, 1)
        prominence = support / np.maximum(broad, 1)

        peaks = np.zeros(support.shape, dtype=bool)
        peaks[:, 1:-1] = (
            (support[:, 1:-1] > support[:, :-2])
            & (support[:, 1:-1] >= support[:, 2:])
        )
        valid = peaks & (support >= min_mode_support) & (prominence >= 0.08)
        quality = (
            np.log1p(support.astype(np.float64))
            * np.sqrt(np.clip(prominence, 0.0, 1.0))
            * np.square(1.0 - np.clip(underburden, 0.0, 1.0))
        )
        quality[~valid] = -np.inf
        selected_bins = np.argpartition(quality, -mode_count, axis=1)[:, -mode_count:]
        selected_quality = np.take_along_axis(quality, selected_bins, axis=1)
        order = np.argsort(selected_quality, axis=1)[:, ::-1]
        selected_bins = np.take_along_axis(selected_bins, order, axis=1)
        selected_quality = np.take_along_axis(selected_quality, order, axis=1)

        rows = np.arange(stop - start, dtype=np.int64)[:, None]
        selected_support = np.take_along_axis(support, selected_bins, axis=1)
        numerator = np.zeros(selected_support.shape, dtype=np.float64)
        denominator = np.zeros(selected_support.shape, dtype=np.uint64)
        for offset in range(-mode_radius_bins, mode_radius_bins + 1):
            bins = np.clip(selected_bins + offset, 0, z_bins - 1)
            weights = raw[rows, bins]
            numerator += weights * (bins + 0.5)
            denominator += weights
        centroid = numerator / np.maximum(denominator, 1)
        selected_prominence = np.take_along_axis(prominence, selected_bins, axis=1)
        selected_underburden = np.take_along_axis(underburden, selected_bins, axis=1)
        valid_selected = np.isfinite(selected_quality) & (denominator > 0)
        selected_height = base[start:stop, None] + centroid * z_bin_m

        height[start:stop][valid_selected] = selected_height[valid_selected].astype(np.float32)
        support_out[start:stop][valid_selected] = selected_support[valid_selected].astype(np.uint32)
        prominence_out[start:stop][valid_selected] = selected_prominence[valid_selected].astype(
            np.float32
        )
        underburden_out[start:stop][valid_selected] = selected_underburden[
            valid_selected
        ].astype(np.float32)
        quality_out[start:stop][valid_selected] = selected_quality[valid_selected].astype(
            np.float32
        )

    return ModeEvidence(
        height_m=height,
        support_count=support_out,
        prominence=prominence_out,
        underburden_fraction=underburden_out,
        quality=quality_out,
    )


def _normalized_quality(modes: ModeEvidence) -> np.ndarray:
    quality = np.where(
        np.isfinite(modes.height_m) & np.isfinite(modes.quality) & (modes.quality > 0.0),
        modes.quality,
        0.0,
    ).astype(np.float64)
    maximum = np.max(quality, axis=1, initial=0.0)
    return np.divide(
        quality,
        maximum[:, None],
        out=np.zeros_like(quality),
        where=maximum[:, None] > 0.0,
    )


def discover_context_component(
    modes: ModeEvidence,
    grid: GridSpec,
    observed_cell_mask: np.ndarray,
    *,
    config: SurfaceConfig,
    max_neighbor_step_m: float,
    continuation_slope_x: np.ndarray | None = None,
    continuation_slope_y: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select one mode or null per cell, then retain one compatible component."""
    heights = modes.height_m
    valid = np.isfinite(heights) & (modes.quality > 0.0)
    cells, mode_count = heights.shape
    if cells != grid.cell_count:
        raise ValueError("Hovi context modes do not match their grid")
    if continuation_slope_x is None or continuation_slope_y is None:
        if continuation_slope_x is not None or continuation_slope_y is not None:
            raise ValueError("Hovi context continuation requires both tangent slopes")
        slope_x = slope_y = None
        continuation_kind = "raw_bootstrap"
    else:
        slope_x = np.asarray(continuation_slope_x, dtype=np.float64).reshape(-1)
        slope_y = np.asarray(continuation_slope_y, dtype=np.float64).reshape(-1)
        if slope_x.size != cells or slope_y.size != cells:
            raise ValueError("Hovi context continuation slopes do not match its grid")
        valid &= np.isfinite(slope_x[:, None]) & np.isfinite(slope_y[:, None])
        continuation_kind = "symmetric_same_sheet_tangent_edge_residual"
    normalized = _normalized_quality(modes)
    support_strength = 1.0 - np.exp(
        -modes.support_count.astype(np.float64)
        / max(config.context_min_mode_support, 1)
    )
    unary = (
        0.45 * normalized
        + 0.20 * support_strength
        + 0.15 * modes.prominence
        + 0.20 * (1.0 - np.clip(modes.underburden_fraction, 0.0, 1.0))
    )
    unary[~valid] = -np.inf
    observed_count = int(np.count_nonzero(observed_cell_mask))
    if not np.any(valid):
        return np.zeros_like(valid), {
            "status": "no_seed",
            "observed_cells": observed_count,
            "ranked_components": [],
        }

    def neighbors(cell: int):
        row, column = divmod(cell, grid.width)
        if row > 0:
            yield cell - grid.width
        if row + 1 < grid.height:
            yield cell + grid.width
        if column > 0:
            yield cell - 1
        if column + 1 < grid.width:
            yield cell + 1

    def edge_residual(
        cell: int, mode_indexes: np.ndarray, neighbor: int, neighbor_mode: int
    ) -> np.ndarray:
        actual = heights[cell, mode_indexes] - heights[neighbor, neighbor_mode]
        if slope_x is None or slope_y is None:
            return actual
        row, column = divmod(cell, grid.width)
        neighbor_row, neighbor_column = divmod(neighbor, grid.width)
        dx = (column - neighbor_column) * grid.resolution_m
        dy = (row - neighbor_row) * grid.resolution_m
        expected = (
            0.5 * (slope_x[cell] + slope_x[neighbor]) * dx
            + 0.5 * (slope_y[cell] + slope_y[neighbor]) * dy
        )
        return actual - expected

    def choose_mode(
        cell: int, adjacent_assignments: list[tuple[int, int]]
    ) -> tuple[int, float, str]:
        available = valid[cell].copy()
        if adjacent_assignments:
            all_modes = np.arange(mode_count, dtype=np.intp)
            differences = np.column_stack(
                [
                    np.abs(edge_residual(cell, all_modes, neighbor, neighbor_mode))
                    for neighbor, neighbor_mode in adjacent_assignments
                ]
            )
            required_neighbors = len(adjacent_assignments) // 2 + 1
            available &= (
                np.count_nonzero(differences <= max_neighbor_step_m, axis=1)
                >= required_neighbors
            )
        indexes = np.flatnonzero(available)
        if indexes.size == 0:
            return -1, 0.0, "discontinuity"
        if adjacent_assignments:
            residual_ratio = np.column_stack(
                [
                    edge_residual(cell, indexes, neighbor, neighbor_mode)
                    / max_neighbor_step_m
                    for neighbor, neighbor_mode in adjacent_assignments
                ]
            )
            agreement = 1.0 - np.mean(
                np.minimum(np.square(residual_ratio), 1.0),
                axis=1,
            )
            scores = 0.72 * unary[cell, indexes] + 0.28 * agreement
        else:
            scores = unary[cell, indexes]
        order = np.argsort(scores)[::-1]
        selected_mode = int(indexes[order[0]])
        selected_score = float(scores[order[0]])
        for position in order[1:]:
            competitor_mode = int(indexes[position])
            separation = abs(
                float(heights[cell, competitor_mode] - heights[cell, selected_mode])
            )
            if separation < config.ambiguity_min_separation_m:
                continue
            ratio = float(scores[position] / max(selected_score, 1e-12))
            if ratio >= config.ambiguity_quality_ratio:
                return -1, ratio, "competition"
        return selected_mode, selected_score, "selected"

    assigned = np.full(cells, -1, dtype=np.int16)
    status = np.full(cells, "unsupported", dtype="U13")
    for cell in range(cells):
        assigned[cell], _, status[cell] = choose_mode(cell, [])
    for _ in range(8):
        updated = np.full(cells, -1, dtype=np.int16)
        updated_status = np.full(cells, "unsupported", dtype="U13")
        for cell in range(cells):
            adjacent = [
                (other, int(assigned[other]))
                for other in neighbors(cell)
                if assigned[other] >= 0
            ]
            updated[cell], _, updated_status[cell] = choose_mode(cell, adjacent)
        if np.array_equal(updated, assigned):
            status = updated_status
            break
        assigned = updated
        status = updated_status

    visited = np.zeros(cells, dtype=bool)
    ranked_components: list[tuple[float, int, float, int, np.ndarray]] = []
    component_id = 0
    for start in np.flatnonzero(assigned >= 0):
        if visited[start]:
            continue
        component_id += 1
        stack = [int(start)]
        visited[start] = True
        members: list[int] = []
        while stack:
            cell = stack.pop()
            members.append(cell)
            for neighbor in neighbors(cell):
                if visited[neighbor] or assigned[neighbor] < 0:
                    continue
                residual = float(
                    edge_residual(
                        cell,
                        np.asarray([assigned[cell]], dtype=np.intp),
                        neighbor,
                        int(assigned[neighbor]),
                    )[0]
                )
                if abs(residual) <= max_neighbor_step_m:
                    visited[neighbor] = True
                    stack.append(neighbor)
        member_array = np.asarray(members, dtype=np.int64)
        mean_quality = float(np.mean(unary[member_array, assigned[member_array]]))
        score = member_array.size * (1.0 + 0.20 * mean_quality)
        ranked_components.append(
            (score, int(member_array.size), mean_quality, component_id, member_array)
        )
    ranked_components.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    if not ranked_components:
        return np.zeros_like(valid), {
            "status": "no_single_valued_component",
            "observed_cells": observed_count,
            "ranked_components": [],
        }
    _, coverage, mean_quality, winning_component, winning_cells = ranked_components[0]
    mask = np.zeros_like(valid)
    mask[winning_cells, assigned[winning_cells]] = True
    competition_count = int(np.count_nonzero(status == "competition"))
    discontinuity_count = int(np.count_nonzero(status == "discontinuity"))
    diagnostics = {
        "status": "selected",
        "observed_cells": observed_count,
        "selection_policy": "one_label_or_null_robust_neighbor_energy_then_component",
        "continuation_quantity": continuation_kind,
        "selected_component": winning_component,
        "selected_cells": int(coverage),
        "selected_coverage_fraction": coverage / max(observed_count, 1),
        "selected_mean_normalized_quality": mean_quality,
        "competition_abstentions": competition_count,
        "discontinuity_abstentions": discontinuity_count,
        "ranked_components": [
            {
                "component": int(component),
                "cell_count": int(component_coverage),
                "coverage_fraction": component_coverage / max(observed_count, 1),
                "score": float(score),
                "mean_normalized_quality": float(component_quality),
            }
            for score, component_coverage, component_quality, component, _ in ranked_components[:8]
        ],
    }
    return mask, diagnostics


def _shift(array: np.ndarray, dy: int, dx: int, fill: float) -> np.ndarray:
    output = np.full(array.shape, fill, dtype=array.dtype)
    source_y = slice(max(0, -dy), min(array.shape[0], array.shape[0] - dy))
    source_x = slice(max(0, -dx), min(array.shape[1], array.shape[1] - dx))
    target_y = slice(max(0, dy), min(array.shape[0], array.shape[0] + dy))
    target_x = slice(max(0, dx), min(array.shape[1], array.shape[1] + dx))
    output[target_y, target_x] = array[source_y, source_x]
    return output


def select_conditioned_sheet(
    modes: ModeEvidence,
    grid: GridSpec,
    config: SurfaceConfig,
    *,
    min_mode_support: int,
    neighbor_step_m: float,
    parent_height_m: np.ndarray | None = None,
    parent_tolerance_m: float | None = None,
    continuation_slope_x: np.ndarray | None = None,
    continuation_slope_y: np.ndarray | None = None,
    candidate_mask: np.ndarray | None = None,
    iterations: int = 5,
) -> SheetSelection:
    heights = modes.height_m.astype(np.float64)
    cells, mode_count = heights.shape
    valid = np.isfinite(heights) & (modes.support_count >= min_mode_support)
    if candidate_mask is not None:
        valid &= candidate_mask
    parent = None
    if parent_height_m is not None:
        if parent_tolerance_m is None:
            raise ValueError("Hovi parent-conditioned selection requires a tolerance")
        parent = np.asarray(parent_height_m, dtype=np.float64).reshape(-1)
        valid &= np.isfinite(parent[:, None])
        valid &= np.abs(heights - parent[:, None]) <= parent_tolerance_m
    if continuation_slope_x is None or continuation_slope_y is None:
        if continuation_slope_x is not None or continuation_slope_y is not None:
            raise ValueError("Hovi conditioned selection requires both tangent slopes")
        tangent_x = tangent_y = None
    else:
        tangent_x = np.asarray(continuation_slope_x, dtype=np.float64).reshape(-1)
        tangent_y = np.asarray(continuation_slope_y, dtype=np.float64).reshape(-1)
        if tangent_x.size != cells or tangent_y.size != cells:
            raise ValueError("Hovi continuation slopes do not match its grid")
        valid &= np.isfinite(tangent_x[:, None]) & np.isfinite(tangent_y[:, None])

    normalized = _normalized_quality(modes)
    support_strength = 1.0 - np.exp(
        -modes.support_count.astype(np.float64) / max(min_mode_support, 1)
    )
    opacity = 1.0 - np.clip(modes.underburden_fraction, 0.0, 1.0)
    unary = (
        0.35 * normalized
        + 0.25 * support_strength
        + 0.20 * modes.prominence
        + 0.20 * opacity
    )
    unary[~valid] = -np.inf
    selected = np.argmax(unary, axis=1).astype(np.int16)
    has_candidate = np.any(valid, axis=1)
    selected[~has_candidate] = -1
    for _ in range(iterations):
        chosen = np.full(cells, np.nan, dtype=np.float64)
        chosen[has_candidate] = heights[np.flatnonzero(has_candidate), selected[has_candidate]]
        chosen_grid = chosen.reshape(grid.height, grid.width)
        neighbor_score = np.zeros((cells, mode_count), dtype=np.float64)
        neighbor_count = np.zeros((cells, mode_count), dtype=np.uint8)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = _shift(chosen_grid, dy, dx, np.nan).reshape(-1)
            available = np.isfinite(neighbor)
            if tangent_x is None or tangent_y is None:
                expected = 0.0
            else:
                neighbor_slope_x = _shift(
                    tangent_x.reshape(grid.height, grid.width), dy, dx, np.nan
                ).reshape(-1)
                neighbor_slope_y = _shift(
                    tangent_y.reshape(grid.height, grid.width), dy, dx, np.nan
                ).reshape(-1)
                expected = (
                    0.5 * (tangent_x + neighbor_slope_x) * dx * grid.resolution_m
                    + 0.5 * (tangent_y + neighbor_slope_y) * dy * grid.resolution_m
                )
                available &= np.isfinite(expected)
            residual = heights - neighbor[:, None] - np.asarray(expected)[..., None]
            agreement = 1.0 - np.minimum(
                np.square(residual / neighbor_step_m), 1.0
            )
            neighbor_score += np.where(available[:, None], agreement, 0.0)
            neighbor_count += available[:, None]
        neighbor_mean = np.divide(
            neighbor_score,
            neighbor_count,
            out=np.zeros_like(neighbor_score),
            where=neighbor_count > 0,
        )
        score = 0.76 * unary + 0.24 * neighbor_mean
        score[~valid] = -np.inf
        selected = np.argmax(score, axis=1).astype(np.int16)
        has_candidate = np.any(valid, axis=1)
        selected[~has_candidate] = -1

    selected_height = np.full(cells, np.nan, dtype=np.float32)
    selected_support = np.zeros(cells, dtype=np.uint32)
    selected_underburden = np.ones(cells, dtype=np.float32)
    selected_prominence = np.zeros(cells, dtype=np.float32)
    selected_unary = np.zeros(cells, dtype=np.float64)
    chosen_cells = np.flatnonzero(has_candidate)
    chosen_modes = selected[has_candidate]
    selected_height[has_candidate] = heights[chosen_cells, chosen_modes]
    selected_support[has_candidate] = modes.support_count[chosen_cells, chosen_modes]
    selected_underburden[has_candidate] = modes.underburden_fraction[chosen_cells, chosen_modes]
    selected_prominence[has_candidate] = modes.prominence[chosen_cells, chosen_modes]
    selected_unary[has_candidate] = unary[chosen_cells, chosen_modes]

    competitor_ratio = np.zeros(cells, dtype=np.float64)
    above_sheet_competitor = np.zeros(cells, dtype=bool)
    for mode in range(mode_count):
        other = valid[:, mode] & has_candidate & (mode != selected)
        ratio = np.divide(
            unary[:, mode],
            selected_unary,
            out=np.zeros(cells, dtype=np.float64),
            where=other & (selected_unary > 0.0),
        )
        separation = np.abs(heights[:, mode] - selected_height)
        qualifying = other & (separation >= config.ambiguity_min_separation_m)
        competitor_ratio = np.maximum(
            competitor_ratio, np.where(qualifying, ratio, 0.0)
        )
        above_sheet_competitor |= (
            other
            & (heights[:, mode] > selected_height + 0.08)
            & (modes.support_count[:, mode] >= np.maximum(8, 0.20 * selected_support))
        )
    ambiguity = (
        has_candidate
        & (competitor_ratio >= config.ambiguity_quality_ratio)
    )
    substantial_underburden = (
        has_candidate
        & (selected_underburden > config.max_selected_underburden_fraction)
    )
    ambiguity |= substantial_underburden

    chosen_absolute = np.full(cells, np.nan, dtype=np.float64)
    chosen_absolute[has_candidate] = heights[chosen_cells, chosen_modes]
    chosen_grid = chosen_absolute.reshape(grid.height, grid.width)
    neighbor_agreement_sum = np.zeros(cells, dtype=np.float64)
    neighbor_available = np.zeros(cells, dtype=np.uint8)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbor = _shift(chosen_grid, dy, dx, np.nan).reshape(-1)
        available = np.isfinite(neighbor) & has_candidate
        if tangent_x is None or tangent_y is None:
            expected = 0.0
        else:
            neighbor_slope_x = _shift(
                tangent_x.reshape(grid.height, grid.width), dy, dx, np.nan
            ).reshape(-1)
            neighbor_slope_y = _shift(
                tangent_y.reshape(grid.height, grid.width), dy, dx, np.nan
            ).reshape(-1)
            expected = (
                0.5 * (tangent_x + neighbor_slope_x) * dx * grid.resolution_m
                + 0.5 * (tangent_y + neighbor_slope_y) * dy * grid.resolution_m
            )
            available &= np.isfinite(expected)
        residual = chosen_absolute - neighbor - np.asarray(expected)
        neighbor_agreement_sum += np.where(
            available,
            1.0
            - np.minimum(
                np.square(residual / neighbor_step_m), 1.0
            ),
            0.0,
        )
        neighbor_available += available
    neighbor_agreement = np.divide(
        neighbor_agreement_sum,
        neighbor_available,
        out=np.zeros(cells, dtype=np.float64),
        where=neighbor_available > 0,
    )
    neighbor_consistent = has_candidate & (neighbor_agreement >= 0.55)
    if parent is None:
        broad_window_consistent = has_candidate.copy()
    else:
        broad_window_consistent = has_candidate & (
            np.abs(selected_height - parent) <= parent_tolerance_m
        )
    dominance = 1.0 - np.clip(competitor_ratio, 0.0, 1.0)
    support_score = 1.0 - np.exp(
        -selected_support.astype(np.float64) / max(min_mode_support, 1)
    )
    selection_score = (
        0.25 * support_score
        + 0.20 * selected_prominence
        + 0.20 * (1.0 - selected_underburden)
        + 0.20 * dominance
        + 0.15 * neighbor_agreement
    )
    selection_score[~has_candidate] = 0.0
    direct = (
        has_candidate
        & ~ambiguity
        & neighbor_consistent
        & broad_window_consistent
        & (selection_score >= config.min_direct_selection_score)
    )
    return SheetSelection(
        selected_height_m=selected_height.reshape(grid.height, grid.width),
        selected_mode_index=selected.reshape(grid.height, grid.width),
        selection_score=selection_score.astype(np.float32).reshape(grid.height, grid.width),
        direct_mask=direct.reshape(grid.height, grid.width),
        ambiguity_mask=ambiguity.reshape(grid.height, grid.width),
        above_sheet_competitor_mask=above_sheet_competitor.reshape(grid.height, grid.width),
        unsupported_mask=(~has_candidate).reshape(grid.height, grid.width),
        neighbor_consistent_mask=neighbor_consistent.reshape(grid.height, grid.width),
        broad_window_consistent_mask=broad_window_consistent.reshape(grid.height, grid.width),
        selected_support_count=selected_support.reshape(grid.height, grid.width),
        selected_underburden_fraction=selected_underburden.reshape(grid.height, grid.width),
        competing_mode_ratio=competitor_ratio.astype(np.float32).reshape(
            grid.height, grid.width
        ),
    )


def predict_same_sheet_tangent(
    parent_grid: GridSpec,
    child_grid: GridSpec,
    parent_height_m: np.ndarray,
    parent_direct_mask: np.ndarray,
    *,
    prior_slope_x: np.ndarray | None = None,
    prior_slope_y: np.ndarray | None = None,
    neighbor_step_m: float,
    radius_cells: int,
    minimum_cells: int,
    maximum_residual_m: float,
    maximum_slope: float,
) -> TangentPrediction:
    """Anchor robust local tangents to observed parent heights without crossing steps."""
    height = np.asarray(parent_height_m, dtype=np.float64)
    direct = np.asarray(parent_direct_mask, dtype=bool) & np.isfinite(height)
    if height.shape != (parent_grid.height, parent_grid.width) or direct.shape != height.shape:
        raise ValueError("Hovi tangent inputs do not match the parent grid")
    if prior_slope_x is None or prior_slope_y is None:
        if prior_slope_x is not None or prior_slope_y is not None:
            raise ValueError("Hovi tangent fitting requires both prior slopes")
        prior_x = prior_y = None
        edge_limit = neighbor_step_m + maximum_slope * parent_grid.resolution_m
    else:
        prior_x = np.asarray(prior_slope_x, dtype=np.float64)
        prior_y = np.asarray(prior_slope_y, dtype=np.float64)
        if prior_x.shape != height.shape or prior_y.shape != height.shape:
            raise ValueError("Hovi prior tangent slopes do not match the parent grid")
        direct &= np.isfinite(prior_x) & np.isfinite(prior_y)
        edge_limit = neighbor_step_m
    slope_x = np.full(height.shape, np.nan, dtype=np.float32)
    slope_y = np.full(height.shape, np.nan, dtype=np.float32)
    residual_out = np.full(height.shape, np.nan, dtype=np.float32)
    valid_out = np.zeros(height.shape, dtype=bool)

    for center_row, center_column in zip(*np.nonzero(direct), strict=True):
        row_low = max(0, center_row - radius_cells)
        row_high = min(parent_grid.height, center_row + radius_cells + 1)
        column_low = max(0, center_column - radius_cells)
        column_high = min(parent_grid.width, center_column + radius_cells + 1)
        visited = {(int(center_row), int(center_column))}
        stack = [(int(center_row), int(center_column))]
        members: list[tuple[int, int]] = []
        while stack:
            row, column = stack.pop()
            members.append((row, column))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                other_row = row + dy
                other_column = column + dx
                other = (other_row, other_column)
                if (
                    other in visited
                    or other_row < row_low
                    or other_row >= row_high
                    or other_column < column_low
                    or other_column >= column_high
                    or not direct[other]
                ):
                    continue
                actual_dz = float(height[other] - height[row, column])
                if prior_x is None or prior_y is None:
                    edge_residual = actual_dz
                else:
                    dx_m = (other_column - column) * parent_grid.resolution_m
                    dy_m = (other_row - row) * parent_grid.resolution_m
                    expected_dz = (
                        0.5 * (prior_x[other] + prior_x[row, column]) * dx_m
                        + 0.5 * (prior_y[other] + prior_y[row, column]) * dy_m
                    )
                    edge_residual = actual_dz - expected_dz
                if abs(edge_residual) <= edge_limit:
                    visited.add(other)
                    stack.append(other)
        if len(members) < minimum_cells:
            continue
        rows = np.asarray([item[0] for item in members], dtype=np.float64)
        columns = np.asarray([item[1] for item in members], dtype=np.float64)
        design = np.column_stack(
            (
                (columns - center_column) * parent_grid.resolution_m,
                (rows - center_row) * parent_grid.resolution_m,
            )
        )
        response = height[
            np.asarray(rows, dtype=np.intp), np.asarray(columns, dtype=np.intp)
        ] - height[center_row, center_column]
        if np.linalg.matrix_rank(design) < 2:
            continue
        weights = np.ones(response.size, dtype=np.float64)
        coefficients = np.zeros(2, dtype=np.float64)
        for _ in range(4):
            weighted_design = design * np.sqrt(weights[:, None])
            weighted_response = response * np.sqrt(weights)
            coefficients, _, rank, _ = np.linalg.lstsq(
                weighted_design, weighted_response, rcond=None
            )
            if rank < 2:
                break
            residual = response - design @ coefficients
            scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
            huber = max(maximum_residual_m * 0.5, 1.5 * float(scale), 1e-4)
            absolute = np.abs(residual)
            weights = np.minimum(1.0, huber / np.maximum(absolute, 1e-12))
        residual = response - design @ coefficients
        inlier = np.abs(residual) <= maximum_residual_m
        if np.count_nonzero(inlier) < minimum_cells or np.linalg.matrix_rank(design[inlier]) < 2:
            continue
        coefficients, _, _, _ = np.linalg.lstsq(design[inlier], response[inlier], rcond=None)
        residual = response[inlier] - design[inlier] @ coefficients
        residual_p95 = float(np.quantile(np.abs(residual), 0.95))
        if residual_p95 > maximum_residual_m or np.linalg.norm(coefficients) > maximum_slope:
            continue
        slope_x[center_row, center_column] = coefficients[0]
        slope_y[center_row, center_column] = coefficients[1]
        residual_out[center_row, center_column] = residual_p95
        valid_out[center_row, center_column] = True

    mapping = map_grid_centers(child_grid, parent_grid)
    child_reference = np.full(mapping.shape, np.nan, dtype=np.float32)
    child_slope_x = np.full(mapping.shape, np.nan, dtype=np.float32)
    child_slope_y = np.full(mapping.shape, np.nan, dtype=np.float32)
    child_valid = np.zeros(mapping.shape, dtype=bool)
    valid_mapping = mapping >= 0
    parent_rows = np.zeros(mapping.shape, dtype=np.intp)
    parent_columns = np.zeros(mapping.shape, dtype=np.intp)
    parent_rows[valid_mapping] = mapping[valid_mapping] // parent_grid.width
    parent_columns[valid_mapping] = mapping[valid_mapping] % parent_grid.width
    valid_mapping &= valid_out[parent_rows, parent_columns]
    child_rows, child_columns = np.indices(mapping.shape)
    child_x = child_grid.origin_x_m + (child_columns + 0.5) * child_grid.resolution_m
    child_y = child_grid.origin_y_m + (child_rows + 0.5) * child_grid.resolution_m
    parent_x = parent_grid.origin_x_m + (parent_columns + 0.5) * parent_grid.resolution_m
    parent_y = parent_grid.origin_y_m + (parent_rows + 0.5) * parent_grid.resolution_m
    predicted = (
        height[parent_rows, parent_columns]
        + slope_x[parent_rows, parent_columns] * (child_x - parent_x)
        + slope_y[parent_rows, parent_columns] * (child_y - parent_y)
    )
    child_reference[valid_mapping] = predicted[valid_mapping]
    child_slope_x[valid_mapping] = slope_x[parent_rows, parent_columns][valid_mapping]
    child_slope_y[valid_mapping] = slope_y[parent_rows, parent_columns][valid_mapping]
    child_valid[valid_mapping] = True
    return TangentPrediction(
        child_center_z_m=child_reference,
        child_slope_x=child_slope_x,
        child_slope_y=child_slope_y,
        child_valid_mask=child_valid,
        parent_slope_x=slope_x,
        parent_slope_y=slope_y,
        parent_valid_mask=valid_out,
        parent_fit_residual_m=residual_out,
    )


def interpolate_small_supported_holes(
    selected_height_m: np.ndarray,
    direct_mask: np.ndarray,
    selection_score: np.ndarray,
    eligible_hole_mask: np.ndarray,
    ambiguity_mask: np.ndarray,
    above_sheet_competitor_mask: np.ndarray,
    analysis_core_mask: np.ndarray,
    grid: GridSpec,
    config: SurfaceConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer only fully enclosed tiny holes; direct observations are never altered."""
    output = np.asarray(selected_height_m, dtype=np.float32).copy()
    interpolation = np.zeros(output.shape, dtype=bool)
    holes = eligible_hole_mask & ~direct_mask
    labels, count = ndimage.label(holes, structure=ndimage.generate_binary_structure(2, 1))
    for label_id in range(1, count + 1):
        component = labels == label_id
        size = int(np.count_nonzero(component))
        if size == 0 or size > config.max_interpolation_component_cells:
            continue
        component_rows, component_columns = np.nonzero(component)
        if (
            np.any(component_rows == 0)
            or np.any(component_columns == 0)
            or np.any(component_rows == grid.height - 1)
            or np.any(component_columns == grid.width - 1)
        ):
            continue
        boundary = ndimage.binary_dilation(
            component, structure=ndimage.generate_binary_structure(2, 2)
        ) & ~component
        guarded_boundary = (
            direct_mask
            & (selection_score >= 0.65)
            & ~ambiguity_mask
            & ~above_sheet_competitor_mask
            & analysis_core_mask
        )
        if not np.all(guarded_boundary[boundary]):
            continue
        ring = boundary
        rows, columns = np.nonzero(ring)
        if rows.size < config.interpolation_ring_min_cells:
            continue
        x = grid.origin_x_m + (columns + 0.5) * grid.resolution_m
        y = grid.origin_y_m + (rows + 0.5) * grid.resolution_m
        design = np.column_stack((x, y, np.ones(x.size)))
        coefficients, _, rank, _ = np.linalg.lstsq(design, output[ring], rcond=None)
        if rank < 3:
            continue
        residual = np.abs(design @ coefficients - output[ring])
        if float(residual.max(initial=0.0)) > config.interpolation_plane_max_residual_m:
            continue
        hole_rows, hole_columns = component_rows, component_columns
        hole_x = grid.origin_x_m + (component_columns + 0.5) * grid.resolution_m
        hole_y = grid.origin_y_m + (component_rows + 0.5) * grid.resolution_m
        output[component] = (
            coefficients[0] * hole_x + coefficients[1] * hole_y + coefficients[2]
        ).astype(np.float32)
        interpolation[component] = True
    return output, interpolation
