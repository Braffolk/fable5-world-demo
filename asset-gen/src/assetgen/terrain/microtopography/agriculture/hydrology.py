"""Bounded flow-connected rill incision and sediment deposition for R0."""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .base import process_halo_base
from .forms import evaluate_operation_forms
from .model import CultivatedConditions

_NEIGHBORS = (
    (-1, -1, 2.0**0.5),
    (-1, 0, 1.0),
    (-1, 1, 2.0**0.5),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (1, -1, 2.0**0.5),
    (1, 0, 1.0),
    (1, 1, 2.0**0.5),
)


def _compact_bump(distance_ratio: np.ndarray) -> np.ndarray:
    core = np.maximum(1.0 - np.square(distance_ratio), 0.0)
    return np.square(core)


def _priority_flood_receivers(
    raw_height: np.ndarray,
    texel_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a depression-conditioned surface and one boundary-draining tree."""
    height = np.asarray(raw_height, dtype=np.float64)
    rows, cols = height.shape
    filled = np.full_like(height, np.nan)
    receiver = np.full(height.size, -1, dtype=np.int64)
    receiver_distance = np.zeros(height.size, dtype=np.float32)
    visited = np.zeros(height.shape, dtype=bool)
    heap: list[tuple[float, int]] = []

    boundary = np.zeros(height.shape, dtype=bool)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    for flat in np.flatnonzero(boundary):
        row, col = divmod(int(flat), cols)
        visited[row, col] = True
        filled[row, col] = height[row, col]
        heapq.heappush(heap, (float(height[row, col]), int(flat)))

    epsilon = 1e-8
    while heap:
        current_height, flat = heapq.heappop(heap)
        row, col = divmod(flat, cols)
        for dr, dc, distance_cells in _NEIGHBORS:
            rr, cc = row + dr, col + dc
            if rr < 0 or rr >= rows or cc < 0 or cc >= cols or visited[rr, cc]:
                continue
            visited[rr, cc] = True
            child = rr * cols + cc
            conditioned = max(
                float(height[rr, cc]),
                current_height + epsilon * distance_cells,
            )
            filled[rr, cc] = conditioned
            receiver[child] = flat
            receiver_distance[child] = texel_m * distance_cells
            heapq.heappush(heap, (conditioned, child))
    if not visited.all() or not np.isfinite(filled).all():
        raise AssertionError("priority flood failed to connect its process domain")
    return filled, receiver, receiver_distance


@dataclass(frozen=True)
class HydrologyForms:
    incision: np.ndarray
    deposition: np.ndarray
    flow_area_coarse: np.ndarray
    channel_coarse: np.ndarray
    process_texel_m: float
    generated_sediment_m3: float
    deposited_sediment_m3: float
    exported_sediment_m3: float
    outlet_count: int
    channel_cells: int


def _sample_payload_from_halo(
    field: np.ndarray,
    *,
    factor: int,
    halo_cells: int,
    core_cells: int,
) -> np.ndarray:
    fine = ndimage.zoom(
        np.asarray(field, dtype=np.float32),
        factor,
        order=1,
        mode="nearest",
        grid_mode=True,
        prefilter=False,
    )
    if fine.shape[0] != fine.shape[1]:
        raise AssertionError("upsampled hydrology field is not square")
    start = halo_cells * factor
    stop = start + core_cells * factor + 1
    payload = fine[start:stop, start:stop]
    expected = core_cells * factor + 1
    if payload.shape != (expected, expected):
        raise AssertionError("hydrology halo cannot supply the canonical payload apron")
    return payload


def synthesize_hydrology(conditions: CultivatedConditions) -> HydrologyForms:
    process_texel = 0.25
    factor = int(round(process_texel / conditions.texel_m))
    core_cells = int(round(conditions.size_m / process_texel))
    halo_cells = int(np.ceil(conditions.flow_context_halo_m / process_texel))
    side = core_cells + 2 * halo_cells
    surface = process_halo_base(conditions, texel_m=process_texel)
    if surface.height.shape != (side, side):
        raise AssertionError("corrected-DTM hydrology halo shape differs from contract")
    base = surface.height
    operations = evaluate_operation_forms(surface.eastings, surface.northings, conditions)
    guide = ndimage.gaussian_filter(
        base.astype(np.float64)
        + operations.tracks.astype(np.float64),
        sigma=1.0,
        mode="nearest",
    )
    filled, receiver, receiver_distance = _priority_flood_receivers(
        guide, process_texel
    )
    order = np.argsort(filled.ravel(), kind="stable")[::-1]
    cell_area = process_texel * process_texel
    flow_area = np.full(filled.size, cell_area, dtype=np.float64)
    rain_excess = (
        conditions.rainfall_event_mm
        * 0.001
        * conditions.runoff_coefficient
        * cell_area
        * (
            1.0
            + (conditions.track_runoff_multiplier - 1.0)
            * operations.track_mask.ravel().astype(np.float64)
        )
    )
    runoff = np.asarray(rain_excess, dtype=np.float64)
    for flat in order:
        target = receiver[flat]
        if target >= 0:
            flow_area[target] += flow_area[flat]
            runoff[target] += runoff[flat]

    flat_filled = filled.ravel()
    valid_receiver = receiver >= 0
    slope = np.zeros_like(flow_area)
    slope[valid_receiver] = np.maximum(
        (flat_filled[valid_receiver] - flat_filled[receiver[valid_receiver]])
        / receiver_distance[valid_receiver],
        0.0,
    )
    gy, gx = np.gradient(base.astype(np.float64), process_texel)
    macroscopic_slope = max(float(np.median(np.hypot(gx, gy))), 0.002)
    effective_slope = np.maximum(slope, 0.25 * macroscopic_slope)
    threshold = conditions.rill_contributing_area_m2
    active = flow_area >= threshold
    active_values = flow_area[active]
    scale_area = (
        float(np.quantile(active_values, 0.98)) if active_values.size else threshold
    )
    area_term = np.sqrt(
        np.clip((flow_area - threshold) / max(scale_area - threshold, threshold), 0.0, 1.0)
    )
    slope_term = np.clip(effective_slope / (1.5 * macroscopic_slope), 0.15, 1.0)
    center_depth = conditions.rill_depth_max_m * area_term * slope_term
    channel = active.reshape(filled.shape)
    center_depth_2d = center_depth.reshape(filled.shape)
    width = conditions.rill_width_min_m + (
        conditions.rill_width_max_m - conditions.rill_width_min_m
    ) * area_term.reshape(filled.shape)
    if channel.any():
        distance, indexes = ndimage.distance_transform_edt(
            ~channel,
            sampling=process_texel,
            return_indices=True,
        )
        nearest_depth = center_depth_2d[tuple(indexes)]
        nearest_width = width[tuple(indexes)]
        incision = -nearest_depth * _compact_bump(
            distance / np.maximum(nearest_width, process_texel)
        )
    else:
        incision = np.zeros_like(filled)

    erosion_volume = (
        np.maximum(-incision.ravel(), 0.0)
        * cell_area
        * conditions.sediment_detachment_fraction
    )
    load = np.zeros_like(flow_area)
    deposition_volume = np.zeros_like(flow_area)
    runoff_scale = max(float(np.quantile(runoff, 0.98)), 1e-12)
    for flat in order:
        load[flat] += erosion_volume[flat]
        target = receiver[flat]
        capacity_depth = conditions.sediment_deposition_max_m * (
            0.18
            + 0.82
            * np.sqrt(np.clip(runoff[flat] / runoff_scale, 0.0, 1.0))
            * np.sqrt(np.clip(effective_slope[flat] / macroscopic_slope, 0.0, 1.0))
        )
        capacity_volume = capacity_depth * cell_area
        excess = max(load[flat] - capacity_volume, 0.0)
        deposited = min(
            excess * conditions.sediment_deposition_fraction,
            conditions.sediment_deposition_max_m * cell_area,
        )
        deposition_volume[flat] = deposited
        remaining = load[flat] - deposited
        if target >= 0:
            load[target] += remaining
        else:
            load[flat] = remaining

    outlets = receiver < 0
    exported = float(load[outlets].sum())
    generated = float(erosion_volume.sum())
    deposited = float(deposition_volume.sum())
    deposition = (deposition_volume / cell_area).reshape(filled.shape)
    core = np.s_[halo_cells : halo_cells + core_cells, halo_cells : halo_cells + core_cells]
    return HydrologyForms(
        incision=np.asarray(
            _sample_payload_from_halo(
                incision,
                factor=factor,
                halo_cells=halo_cells,
                core_cells=core_cells,
            ),
            dtype=np.float32,
        ),
        deposition=np.asarray(
            _sample_payload_from_halo(
                deposition,
                factor=factor,
                halo_cells=halo_cells,
                core_cells=core_cells,
            ),
            dtype=np.float32,
        ),
        flow_area_coarse=np.asarray(flow_area.reshape(filled.shape)[core], dtype=np.float32),
        channel_coarse=np.asarray(channel[core], dtype=np.bool_),
        process_texel_m=process_texel,
        generated_sediment_m3=generated,
        deposited_sediment_m3=deposited,
        exported_sediment_m3=exported,
        outlet_count=int(outlets.sum()),
        channel_cells=int(channel[core].sum()),
    )
