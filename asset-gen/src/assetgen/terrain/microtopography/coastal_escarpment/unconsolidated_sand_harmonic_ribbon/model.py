"""Macro reconstruction and connected erosion in an injective harmonic ribbon."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.interpolate import griddata
from scipy.signal import find_peaks

from ..unconsolidated_sand_curvilinear_strip.evidence import (
    Evidence,
    MASTER_PITCH_M,
    SOLVE_PITCH_M,
)
from ..unconsolidated_sand_curvilinear_strip.model import (
    StripModel,
    _pchip_eval,
)
from .domain import HarmonicRibbon, _sample_grid


@dataclass(frozen=True)
class Result:
    master_m: np.ndarray
    macro_master_m: np.ndarray
    micro_master_m: np.ndarray
    solve_absolute_m: np.ndarray
    solve_delta_m: np.ndarray
    macro_correction_m: np.ndarray
    erosion_m: np.ndarray
    deposition_m: np.ndarray
    measured_solve_m: np.ndarray
    control_u: np.ndarray
    control_v: np.ndarray
    control_height_m: np.ndarray
    measured_residual_uv_m: np.ndarray
    valley_strength_uv: np.ndarray
    path_field_uv: np.ndarray
    deposition_uv: np.ndarray
    path_count: int
    source_band_p95_m: float


def _smoothstep(value: np.ndarray) -> np.ndarray:
    x = np.clip(value, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _control_fields(
    evidence: Evidence,
    strip: StripModel,
    ribbon: HarmonicRibbon,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = int(config["u_samples"])
    control_u = np.linspace(0.0, 1.0, count)
    control_v = np.empty((count, ribbon.chain_v.shape[1]), dtype=np.float64)
    control_h = np.empty_like(control_v)
    spacing_m = float(strip.station_m[-1]) / max(count - 1, 1)
    sigma = float(config["along_u_smoothing_m"]) / spacing_m
    for index in range(ribbon.chain_v.shape[1]):
        source_u = ribbon.chain_u[:, index]
        order = np.argsort(source_u)
        source_u = source_u[order]
        source_v = ribbon.chain_v[:, index][order]
        source_h = strip.chain_height_m[:, index][order]
        unique, unique_index = np.unique(source_u, return_index=True)
        control_v[:, index] = np.interp(control_u, unique, source_v[unique_index])
        control_h[:, index] = np.interp(control_u, unique, source_h[unique_index])
        control_v[:, index] = ndimage.gaussian_filter1d(control_v[:, index], sigma, mode="nearest")
        control_h[:, index] = ndimage.gaussian_filter1d(control_h[:, index], sigma, mode="nearest")
    control_v[:, 0], control_v[:, -1] = 0.0, 1.0
    separation = float(config["minimum_control_v_separation"])
    for index in range(1, control_v.shape[1]):
        control_v[:, index] = np.maximum(
            control_v[:, index], control_v[:, index - 1] + separation
        )
    for index in range(control_v.shape[1] - 2, -1, -1):
        control_v[:, index] = np.minimum(
            control_v[:, index], control_v[:, index + 1] - separation
        )
    if np.any(np.diff(control_v, axis=1) < separation - 1.0e-9):
        raise ValueError("joint control interpolation cannot preserve chain order")
    # Re-anchor the two topological boundaries to accepted C0 after smoothing.
    apron_c0 = _sample_grid(evidence.c0_solve_m, ribbon.chains_xy[:, 0])
    crest_c0 = _sample_grid(evidence.c0_solve_m, ribbon.chains_xy[:, -1])
    for index, values in ((0, apron_c0), (-1, crest_c0)):
        source_u = ribbon.chain_u[:, index]
        order = np.argsort(source_u)
        unique, unique_index = np.unique(source_u[order], return_index=True)
        ordered_values = values[order]
        control_h[:, index] = np.interp(control_u, unique, ordered_values[unique_index])
        control_h[:, index] = ndimage.gaussian_filter1d(control_h[:, index], sigma, mode="nearest")
    return control_u, control_v, control_h


def _evaluate_macro(
    evidence: Evidence,
    ribbon: HarmonicRibbon,
    control_u: np.ndarray,
    control_v: np.ndarray,
    control_h: np.ndarray,
    config: dict,
) -> np.ndarray:
    flat_u = np.nan_to_num(ribbon.u, nan=0.0).ravel()
    knots = np.stack(
        [np.interp(flat_u, control_u, control_v[:, index]) for index in range(control_v.shape[1])],
        axis=-1,
    ).reshape((*ribbon.u.shape, control_v.shape[1]))
    heights = np.stack(
        [np.interp(flat_u, control_u, control_h[:, index]) for index in range(control_h.shape[1])],
        axis=-1,
    ).reshape((*ribbon.u.shape, control_h.shape[1]))
    target = _pchip_eval(np.nan_to_num(ribbon.v, nan=0.0), knots, heights)
    correction = np.clip(
        target - evidence.c0_solve_m,
        -float(config["maximum_shift_m"]),
        float(config["maximum_shift_m"]),
    )
    endpoint_m = np.minimum(ribbon.u, 1.0 - ribbon.u) * 119.52849740580665
    endpoint = _smoothstep(endpoint_m / float(config["endpoint_taper_m"]))
    cross = _smoothstep(ribbon.v / 0.045) * _smoothstep((1.0 - ribbon.v) / 0.045)
    correction *= np.nan_to_num(endpoint * cross) * ribbon.mask
    correction[evidence.hard_solve] = 0.0
    return correction


def _measured_uv(
    evidence: Evidence,
    ribbon: HarmonicRibbon,
    macro_absolute: np.ndarray,
    u_count: int,
    v_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    measured_reference = np.where(
        np.isfinite(evidence.reference_m),
        evidence.reference_m,
        evidence.c0_solve_m[::4, ::4],
    )
    measured_solve = ndimage.zoom(measured_reference, 4.0, order=3, mode="nearest")[:513, :513]
    points = np.column_stack([ribbon.u[ribbon.mask], ribbon.v[ribbon.mask]])
    residual = measured_solve[ribbon.mask] - macro_absolute[ribbon.mask]
    u_axis, v_axis = np.linspace(0.0, 1.0, u_count), np.linspace(0.0, 1.0, v_count)
    uu, vv = np.meshgrid(u_axis, v_axis)
    residual_uv = griddata(points, residual, (uu, vv), method="linear", fill_value=0.0)
    return measured_solve, residual_uv, uu, vv


def _seed_indices(score: np.ndarray, count: int, distance: int) -> np.ndarray:
    peaks, _ = find_peaks(score, distance=distance)
    selected = list(peaks[np.argsort(score[peaks])[::-1][:count]]) if len(peaks) else []
    for candidate in np.argsort(score)[::-1]:
        if len(selected) >= count:
            break
        if all(abs(int(candidate) - int(existing)) >= distance for existing in selected):
            selected.append(int(candidate))
    return np.asarray(sorted(selected), dtype=np.int64)


def _least_cost_paths(
    valley: np.ndarray,
    uu: np.ndarray,
    vv: np.ndarray,
    physical_length_m: float,
    config: dict,
) -> tuple[np.ndarray, int]:
    v_count, u_count = valley.shape
    head_row = int(round(0.82 * (v_count - 1)))
    toe_row = int(round(0.14 * (v_count - 1)))
    head_band = valley[int(0.62 * v_count) : head_row + 1]
    score = np.mean(head_band, axis=0)
    spacing = max(1, int(round(float(config["minimum_seed_spacing_m"]) / physical_length_m * (u_count - 1))))
    seeds = _seed_indices(score, int(config["headcut_seed_count"]), spacing)
    lateral = int(config["path_lateral_step_cells"])
    cost = 1.0 - 0.82 * valley / max(float(np.max(valley)), 1.0e-9)
    path_field = np.zeros_like(valley, dtype=np.float64)
    accepted = 0
    for seed in seeds:
        accumulated = np.full(u_count, np.inf, dtype=np.float64)
        accumulated[seed] = cost[head_row, seed]
        back = np.zeros((head_row - toe_row + 1, u_count), dtype=np.int16)
        for step, row in enumerate(range(head_row - 1, toe_row - 1, -1), start=1):
            previous = accumulated
            candidate_cost = np.full((2 * lateral + 1, u_count), np.inf)
            offsets = np.arange(-lateral, lateral + 1)
            for oi, offset in enumerate(offsets):
                if offset < 0:
                    candidate_cost[oi, :offset] = previous[-offset:]
                elif offset > 0:
                    candidate_cost[oi, offset:] = previous[:-offset]
                else:
                    candidate_cost[oi] = previous
                candidate_cost[oi] += 0.035 * abs(offset)
            choice = np.argmin(candidate_cost, axis=0)
            accumulated = candidate_cost[choice, np.arange(u_count)] + cost[row]
            back[step] = offsets[choice]
        endpoint_window = max(spacing, 3)
        lo, hi = max(0, seed - endpoint_window), min(u_count, seed + endpoint_window + 1)
        col = lo + int(np.argmin(accumulated[lo:hi]))
        path = [(toe_row, col)]
        for step, row in enumerate(range(toe_row, head_row), start=1):
            col = int(np.clip(col + back[head_row - toe_row - step + 1, col], 0, u_count - 1))
            path.append((row + 1, col))
        if path[-1][0] != head_row:
            continue
        for row, col in path:
            path_field[row, col] = max(path_field[row, col], 0.35 + 0.65 * valley[row, col])
        accepted += 1
    if accepted < 2:
        raise ValueError("measured UV cost supports fewer than two headcut-to-toe paths")
    physical_u_pitch = physical_length_m / max(u_count - 1, 1)
    sigma_u = float(config["path_width_m"]) / physical_u_pitch
    sigma_v = 0.65 / max(12.0 / (v_count - 1), 1.0e-6)
    path_field = ndimage.gaussian_filter(path_field, sigma=(sigma_v, sigma_u))
    path_field /= max(float(np.max(path_field)), 1.0e-9)
    return path_field, accepted


def synthesize(
    evidence: Evidence,
    strip: StripModel,
    ribbon: HarmonicRibbon,
    config: dict,
) -> Result:
    control_u, control_v, control_h = _control_fields(evidence, strip, ribbon, config["macro"])
    macro_correction = _evaluate_macro(
        evidence, ribbon, control_u, control_v, control_h, config["macro"]
    )
    macro_absolute = evidence.c0_solve_m + macro_correction
    u_count, v_count = int(config["macro"]["u_samples"]), int(config["macro"]["v_samples"])
    measured_solve, residual_uv, uu, vv = _measured_uv(
        evidence, ribbon, macro_absolute, u_count, v_count
    )
    along = ndimage.gaussian_filter1d(residual_uv, 8.0, axis=1, mode="nearest")
    valley = np.maximum(along - residual_uv, 0.0)
    valley *= (vv >= 0.14) & (vv <= 0.84)
    positive = valley[valley > 0.0]
    if not len(positive):
        raise ValueError("harmonic domain contains no measured valley residual")
    threshold = float(np.percentile(positive, float(config["process"]["minimum_valley_strength_percentile"])))
    valley = np.where(valley >= threshold, valley, 0.0)
    valley = ndimage.gaussian_filter(valley, sigma=(0.8, 0.8))
    path_uv, path_count = _least_cost_paths(
        valley, uu, vv, float(strip.station_m[-1]), config["process"]
    )

    source_fine = ndimage.zoom(evidence.source_fine_m, 2.0, order=3)
    source_valid = ndimage.zoom(evidence.source_support.astype(np.uint8), 2.0, order=0) > 0
    source_band = ndimage.gaussian_filter(source_fine, 0.45) - ndimage.gaussian_filter(source_fine, 5.0)
    source_p95 = float(np.percentile(np.abs(source_band[source_valid]), 95.0))
    depth = min(
        float(config["process"]["maximum_erosion_depth_m"]),
        source_p95 * float(config["process"]["biala_p95_gain"]),
    )
    u_coordinate = np.nan_to_num(ribbon.u) * (u_count - 1)
    v_coordinate = np.nan_to_num(ribbon.v) * (v_count - 1)
    path_xy = ndimage.map_coordinates(path_uv, [v_coordinate, u_coordinate], order=1, mode="constant")
    erosion = depth * path_xy**1.2 * ribbon.mask * (~evidence.hard_solve)

    toe_row = int(round(float(config["process"]["toe_deposition_v"]) * (v_count - 1)))
    arrival = ndimage.gaussian_filter1d(path_uv[toe_row], 2.0, mode="nearest")
    deposition_uv = arrival[None] * np.exp(
        -0.5
        * ((vv - float(config["process"]["toe_deposition_v"])) / float(config["process"]["toe_deposition_width_v"])) ** 2
    )
    deposition_xy = ndimage.map_coordinates(
        deposition_uv, [v_coordinate, u_coordinate], order=1, mode="constant"
    )
    deposition_xy *= ribbon.mask * (~evidence.hard_solve)
    if float(np.sum(deposition_xy)) <= 0.0:
        raise ValueError("headcut paths do not reach a deposition apron")
    deposition = deposition_xy * (float(np.sum(erosion)) / float(np.sum(deposition_xy)))
    process = macro_correction - erosion + deposition
    process[evidence.hard_solve] = 0.0

    macro_master = ndimage.zoom(process, 4.0, order=3, mode="nearest")[:2049, :2049]
    feature = ndimage.zoom(path_xy + 0.25 * ribbon.mask, 4.0, order=3, mode="nearest")[:2049, :2049]
    continuation = ndimage.gaussian_filter(feature, 0.8) - ndimage.gaussian_filter(feature, 4.0)
    valid_master = ~evidence.hard_master
    ceiling = min(
        float(config["micro"]["absolute_p95_ceiling_m"]),
        source_p95 * float(config["micro"]["biala_p95_fraction"]),
    )
    micro = continuation * (
        ceiling / max(float(np.percentile(np.abs(continuation[valid_master]), 95.0)), 1.0e-9)
    )
    distance = ndimage.distance_transform_edt(valid_master) * MASTER_PITCH_M
    micro *= _smoothstep(distance / 8.0)
    macro_master[evidence.hard_master] = 0.0
    micro[evidence.hard_master] = 0.0
    master = evidence.c0_master_m + macro_master + micro
    master[evidence.hard_master] = evidence.c0_master_m[evidence.hard_master]
    return Result(
        master_m=master,
        macro_master_m=macro_master,
        micro_master_m=micro,
        solve_absolute_m=evidence.c0_solve_m + process,
        solve_delta_m=process,
        macro_correction_m=macro_correction,
        erosion_m=erosion,
        deposition_m=deposition,
        measured_solve_m=measured_solve,
        control_u=control_u,
        control_v=control_v,
        control_height_m=control_h,
        measured_residual_uv_m=residual_uv,
        valley_strength_uv=valley,
        path_field_uv=path_uv,
        deposition_uv=deposition_uv,
        path_count=path_count,
        source_band_p95_m=source_p95,
    )
