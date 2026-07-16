"""Deterministic water-table, vegetation-state, and peat feedback dynamics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from ....repair.prolong import prolong_structural_4x


_BUBBLE_1D = np.asarray(
    [u**3 * (1.0 - u) ** 3 for u in (0.125, 0.375, 0.625, 0.875)],
    dtype=np.float64,
)
_BUBBLE_1D /= np.mean(_BUBBLE_1D)
_BUBBLE_2D = _BUBBLE_1D[:, None] * _BUBBLE_1D[None, :]


@dataclass(frozen=True)
class Result:
    structural_fine_m: np.ndarray
    final_fine_m: np.ndarray
    delta_fine_m: np.ndarray
    water_depth_1m: np.ndarray
    hummock_1m: np.ndarray
    lawn_1m: np.ndarray
    hollow_1m: np.ndarray
    peat_state_1m: np.ndarray
    directional_eligibility_1m: np.ndarray
    process_state_05m: np.ndarray
    process_hummock_05m: np.ndarray
    process_lawn_05m: np.ndarray
    process_hollow_05m: np.ndarray
    trace: np.ndarray
    initial_process_state_05m: np.ndarray


def _masked_laplacian(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
    total = np.zeros_like(value, dtype=np.float64)
    count = np.zeros_like(value, dtype=np.float64)
    selected = mask[1:, :] & mask[:-1, :]
    total[1:, :][selected] += value[:-1, :][selected]
    count[1:, :][selected] += 1.0
    total[:-1, :][selected] += value[1:, :][selected]
    count[:-1, :][selected] += 1.0
    selected = mask[:, 1:] & mask[:, :-1]
    total[:, 1:][selected] += value[:, :-1][selected]
    count[:, 1:][selected] += 1.0
    total[:, :-1][selected] += value[:, 1:][selected]
    count[:, :-1][selected] += 1.0
    average = np.divide(total, count, out=value.copy(), where=count > 0.0)
    return np.where(mask, average - value, 0.0)


def _simplex(
    hummock: np.ndarray,
    lawn: np.ndarray,
    hollow: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hummock = np.where(mask, np.maximum(hummock, 1.0e-8), 0.0)
    lawn = np.where(mask, np.maximum(lawn, 1.0e-8), 0.0)
    hollow = np.where(mask, np.maximum(hollow, 1.0e-8), 0.0)
    total = hummock + lawn + hollow
    return (
        np.divide(hummock, total, out=np.zeros_like(total), where=mask),
        np.divide(lawn, total, out=np.zeros_like(total), where=mask),
        np.divide(hollow, total, out=np.zeros_like(total), where=mask),
    )


def _fitness(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hummock = np.exp(-0.5 * ((depth - 0.70) / 0.18) ** 2)
    lawn = 1.12 * np.exp(-0.5 * ((depth - 0.50) / 0.18) ** 2)
    hollow = np.exp(-0.5 * ((depth - 0.29) / 0.17) ** 2)
    return hummock, lawn, hollow


def _coarse_dynamics(
    height: np.ndarray,
    authority: np.ndarray,
    drain_mask: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, ...]:
    model = config["model"]
    broad = ndimage.gaussian_filter(height, float(model["broad_relief_sigma_m"]))
    samples = broad[authority]
    centered = broad - float(np.median(samples))
    scale = max(float(np.percentile(np.abs(centered[authority]), 95.0)), 1.0e-6)
    relief = np.clip(centered / scale, -2.0, 2.0)
    interior = ndimage.distance_transform_edt(authority)
    boundary_drain = np.exp(-interior / float(model["boundary_drain_length_m"]))
    if np.any(drain_mask):
        infrastructure_drain = np.exp(
            -ndimage.distance_transform_edt(~drain_mask) / 18.0
        )
    else:
        infrastructure_drain = np.zeros_like(height)
    target_depth = np.clip(
        0.49 + 0.11 * relief + 0.17 * boundary_drain + 0.11 * infrastructure_drain,
        0.16,
        0.86,
    )
    residual = height - ndimage.gaussian_filter(height, 4.0)
    residual_scale = max(float(np.percentile(np.abs(residual[authority]), 95.0)), 1.0e-6)
    peat = np.where(authority, np.clip(residual / residual_scale, -1.0, 1.0) * 0.12, 0.0)
    depth = np.where(authority, target_depth + 0.08 * peat, 0.0)
    fit_h, fit_l, fit_o = _fitness(depth)
    hummock, lawn, hollow = _simplex(fit_h, fit_l, fit_o, authority)
    trace: list[list[float]] = []
    dt = float(model["coarse_dt"])
    for step in range(int(model["coarse_steps"])):
        depth_target = np.clip(
            target_depth + 0.16 * peat + 0.035 * hummock - 0.025 * hollow,
            0.10,
            0.92,
        )
        depth += dt * (
            0.46 * _masked_laplacian(depth, authority)
            + 0.10 * (depth_target - depth)
        )
        depth = np.where(authority, np.clip(depth, 0.08, 0.95), 0.0)

        fit_h, fit_l, fit_o = _fitness(depth)
        fit_h += 0.34 * hummock - 0.11 * hollow
        fit_l += 0.17 * (hummock + hollow)
        fit_o += 0.30 * hollow - 0.10 * hummock
        mean_fit = hummock * fit_h + lawn * fit_l + hollow * fit_o
        hummock += dt * (
            0.43 * hummock * (fit_h - mean_fit)
            + 0.10 * _masked_laplacian(hummock, authority)
        )
        lawn += dt * (
            0.43 * lawn * (fit_l - mean_fit)
            + 0.18 * _masked_laplacian(lawn, authority)
        )
        hollow += dt * (
            0.43 * hollow * (fit_o - mean_fit)
            + 0.08 * _masked_laplacian(hollow, authority)
        )
        hummock, lawn, hollow = _simplex(hummock, lawn, hollow, authority)

        accretion = 0.78 * hummock + 0.24 * lawn - 0.69 * hollow
        decomposition = 0.31 * np.clip(0.45 - depth, 0.0, None)
        peat += dt * (
            0.025 * (accretion - decomposition - 0.34 * peat)
            + 0.13 * _masked_laplacian(peat, authority)
        )
        peat = np.where(authority, np.clip(peat, -1.0, 1.0), 0.0)
        if step % 10 == 0 or step == int(model["coarse_steps"]) - 1:
            trace.append(
                [
                    float(step),
                    float(np.mean(depth[authority])),
                    float(np.mean(hummock[authority])),
                    float(np.mean(lawn[authority])),
                    float(np.mean(hollow[authority])),
                    float(np.std(peat[authority])),
                ]
            )

    gy, gx = np.gradient(broad)
    slope = np.hypot(gx, gy)
    start = float(model["directional_slope_start"])
    full = float(model["directional_slope_full"])
    eligibility = np.clip((slope - start) / (full - start), 0.0, 1.0)
    eligibility *= np.clip(
        interior / float(model["directional_interior_min_m"]), 0.0, 1.0
    )
    eligibility = np.where(authority, eligibility, 0.0)
    return depth, hummock, lawn, hollow, peat, eligibility, broad, np.asarray(trace)


def _process_dynamics(
    height: np.ndarray,
    authority: np.ndarray,
    depth: np.ndarray,
    hummock: np.ndarray,
    lawn: np.ndarray,
    hollow: np.ndarray,
    eligibility: np.ndarray,
    broad: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, ...]:
    model = config["model"]
    zoom = 2
    process_mask = np.repeat(np.repeat(authority, zoom, axis=0), zoom, axis=1)
    height_05 = ndimage.zoom(height, zoom, order=3, mode="nearest", prefilter=True)
    residual = height_05 - ndimage.gaussian_filter(height_05, 6.0)
    residual_scale = max(
        float(np.percentile(np.abs(residual[process_mask]), 95.0)), 1.0e-6
    )
    state = np.where(process_mask, np.clip(residual / residual_scale, -1.0, 1.0), 0.0)
    initial = state.copy()
    depth_05 = ndimage.zoom(depth, zoom, order=3, mode="nearest", prefilter=True)
    h = ndimage.zoom(hummock, zoom, order=3, mode="nearest", prefilter=True)
    l = ndimage.zoom(lawn, zoom, order=3, mode="nearest", prefilter=True)
    o = ndimage.zoom(hollow, zoom, order=3, mode="nearest", prefilter=True)
    h, l, o = _simplex(h, l, o, process_mask)
    eligibility_05 = ndimage.zoom(
        eligibility, zoom, order=3, mode="nearest", prefilter=True
    )
    broad_05 = ndimage.zoom(broad, zoom, order=3, mode="nearest", prefilter=True)
    broad_gy, broad_gx = np.gradient(broad_05, 0.5)
    norm = np.maximum(np.hypot(broad_gx, broad_gy), 1.0e-9)
    tangent_x = -broad_gy / norm
    tangent_y = broad_gx / norm
    dt = float(model["process_dt"])

    for _ in range(int(model["process_steps"])):
        local_depth = np.clip(depth_05 + 0.115 * state, 0.06, 0.96)
        fit_h, fit_l, fit_o = _fitness(local_depth)
        fit_h += 0.31 * h - 0.13 * o + 0.13 * np.maximum(state, 0.0)
        fit_l += 0.19 * (h + o) - 0.06 * np.abs(state)
        fit_o += 0.29 * o - 0.12 * h + 0.13 * np.maximum(-state, 0.0)
        mean_fit = h * fit_h + l * fit_l + o * fit_o
        h += dt * (0.68 * h * (fit_h - mean_fit) + 0.08 * _masked_laplacian(h, process_mask))
        l += dt * (0.68 * l * (fit_l - mean_fit) + 0.15 * _masked_laplacian(l, process_mask))
        o += dt * (0.68 * o * (fit_o - mean_fit) + 0.07 * _masked_laplacian(o, process_mask))
        h, l, o = _simplex(h, l, o, process_mask)

        gy, gx = np.gradient(state, 0.5)
        flux_tangent = tangent_x * gx + tangent_y * gy
        flux_x = eligibility_05 * tangent_x * flux_tangent
        flux_y = eligibility_05 * tangent_y * flux_tangent
        directional = np.gradient(flux_x, 0.5, axis=1) + np.gradient(
            flux_y, 0.5, axis=0
        )
        reaction = 0.54 * (state - state**3) + 0.44 * (h - o)
        state += dt * (
            reaction
            + 0.34 * _masked_laplacian(state, process_mask)
            + 0.075 * directional
        )
        state = np.where(process_mask, np.clip(state, -1.35, 1.35), 0.0)
    return state, h, l, o, initial


def simulate(
    height: np.ndarray,
    authority: np.ndarray,
    drain_mask: np.ndarray,
    config: dict[str, Any],
) -> Result:
    if height.shape != authority.shape or height.shape != drain_mask.shape:
        raise ValueError("height and condition masks must share the 1 m whole-mire grid")
    if not np.isfinite(height).all():
        raise ValueError("measured carrier contains nonfinite height")
    (
        depth,
        hummock,
        lawn,
        hollow,
        peat,
        eligibility,
        broad,
        trace,
    ) = _coarse_dynamics(height, authority, drain_mask, config)
    state, process_h, process_l, process_o, initial = _process_dynamics(
        height,
        authority,
        depth,
        hummock,
        lawn,
        hollow,
        eligibility,
        broad,
        config,
    )

    support = np.pad(height, 2, mode="reflect")
    structural = prolong_structural_4x(
        support,
        parent_rows=(2, height.shape[0] + 2),
        parent_cols=(2, height.shape[1] + 2),
    )
    process_fine = ndimage.zoom(state, 2, order=3, mode="nearest", prefilter=True)
    process_fine = np.where(
        np.repeat(np.repeat(authority, 4, axis=0), 4, axis=1), process_fine, 0.0
    )
    raw_delta = float(config["model"]["peat_relief_amplitude_m"]) * np.tanh(process_fine)
    blocks = raw_delta.reshape(height.shape[0], 4, height.shape[1], 4)
    for _ in range(2):
        block_mean = np.mean(blocks, axis=(1, 3), dtype=np.float64)
        blocks -= block_mean[:, None, :, None] * _BUBBLE_2D[None, :, None, :]
    limit = float(config["model"]["peat_relief_limit_m"])
    if float(np.max(np.abs(raw_delta))) > limit:
        raw_delta *= limit / float(np.max(np.abs(raw_delta)))
        blocks = raw_delta.reshape(height.shape[0], 4, height.shape[1], 4)
        for _ in range(2):
            block_mean = np.mean(blocks, axis=(1, 3), dtype=np.float64)
            blocks -= block_mean[:, None, :, None] * _BUBBLE_2D[None, :, None, :]
    final = structural + raw_delta
    final_blocks = final.reshape(height.shape[0], 4, height.shape[1], 4)
    for _ in range(2):
        closure = height - np.mean(final_blocks, axis=(1, 3), dtype=np.float64)
        final_blocks += closure[:, None, :, None] * _BUBBLE_2D[None, :, None, :]
    raw_delta = final - structural
    return Result(
        structural_fine_m=structural,
        final_fine_m=final,
        delta_fine_m=raw_delta,
        water_depth_1m=depth,
        hummock_1m=hummock,
        lawn_1m=lawn,
        hollow_1m=hollow,
        peat_state_1m=peat,
        directional_eligibility_1m=eligibility,
        process_state_05m=state,
        process_hummock_05m=process_h,
        process_lawn_05m=process_l,
        process_hollow_05m=process_o,
        trace=trace,
        initial_process_state_05m=initial,
    )
