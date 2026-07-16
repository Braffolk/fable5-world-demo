"""Source/target preparation for the domain-scale diffusion cascade."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class StageData:
    samples: np.ndarray
    conditions: np.ndarray
    masks: np.ndarray
    origins: tuple[tuple[int, int], ...]


def normalized_gaussian(value: np.ndarray, valid: np.ndarray, sigma_cells: float) -> np.ndarray:
    weights = ndimage.gaussian_filter(valid.astype(np.float64), sigma_cells, mode="nearest")
    smoothed = ndimage.gaussian_filter(np.where(valid, value, 0.0), sigma_cells, mode="nearest")
    result = np.divide(smoothed, weights, out=np.full(value.shape, np.nan), where=weights > 1.0e-6)
    nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return np.where(np.isfinite(result), result, value[tuple(nearest)])


def block_mean(value: np.ndarray, factor: int) -> np.ndarray:
    height = value.shape[0] // factor * factor
    width = value.shape[1] // factor * factor
    return value[:height, :width].reshape(height // factor, factor, width // factor, factor).mean(axis=(1, 3))


def block_fraction(mask: np.ndarray, factor: int) -> np.ndarray:
    return block_mean(mask.astype(np.float64), factor)


def resize(value: np.ndarray, shape: tuple[int, int], order: int = 1) -> np.ndarray:
    result = ndimage.zoom(value, (shape[0] / value.shape[0], shape[1] / value.shape[1]), order=order)
    return result[: shape[0], : shape[1]]


def terrain_conditions(surface: np.ndarray, network: np.ndarray | None, pitch_m: float) -> np.ndarray:
    broad = ndimage.gaussian_filter(surface, 4.0 / pitch_m, mode="nearest")
    gy, gx = np.gradient(broad, pitch_m)
    slope = np.hypot(gx, gy)
    lap = ndimage.laplace(broad, mode="nearest") / (pitch_m * pitch_m)
    if network is None:
        ridge_valley = np.abs(lap) + 0.35 * ndimage.gaussian_gradient_magnitude(surface - broad, 1.5 / pitch_m)
        scale = max(float(np.quantile(ridge_valley, 0.98)), 1.0e-6)
        network = np.clip(ridge_valley / scale, 0.0, 1.0)
    else:
        network = np.clip(network, 0.0, 1.0)
    return np.stack((gx, gy, slope, lap, network)).astype(np.float32)


def source_surfaces(z: np.ndarray, valid: np.ndarray, pitch_m: float) -> tuple[np.ndarray, np.ndarray]:
    structural = normalized_gaussian(z, valid, 1.25 / pitch_m)
    base = normalized_gaussian(z, valid, 8.0 / pitch_m)
    return structural, base


def _stage_windows(
    target: np.ndarray,
    condition: np.ndarray,
    valid: np.ndarray,
    *,
    window_cells: int,
    stride_cells: int,
    minimum_support: float,
    holdout_start_xy: tuple[int, int],
    holdout: bool,
) -> StageData:
    samples, conditions, masks, origins = [], [], [], []
    hold_x, hold_y = holdout_start_xy
    for y in range(0, target.shape[0] - window_cells + 1, stride_cells):
        for x in range(0, target.shape[1] - window_cells + 1, stride_cells):
            is_holdout = x >= hold_x and y >= hold_y
            overlaps_holdout = x + window_cells > hold_x and y + window_cells > hold_y
            if holdout and not is_holdout:
                continue
            if not holdout and overlaps_holdout:
                continue
            window_mask = valid[y : y + window_cells, x : x + window_cells]
            if float(np.mean(window_mask)) < minimum_support:
                continue
            samples.append(target[y : y + window_cells, x : x + window_cells][None])
            conditions.append(condition[:, y : y + window_cells, x : x + window_cells])
            masks.append(window_mask)
            origins.append((x, y))
    if not samples:
        raise ValueError("no source windows satisfy the frozen support/split contract")
    return StageData(np.stack(samples), np.stack(conditions), np.stack(masks), tuple(origins))


def prepare_source_stages(
    z: np.ndarray,
    valid: np.ndarray,
    *,
    source_pitch_m: float,
    holdout_start_xy: tuple[int, int],
    minimum_support: float,
) -> tuple[StageData, StageData, StageData, StageData, dict[str, np.ndarray]]:
    structural, base = source_surfaces(z, valid, source_pitch_m)
    macro_native = structural - base

    factor1 = int(round(2.0 / source_pitch_m))
    macro_2m = block_mean(macro_native, factor1)
    base_2m = block_mean(base, factor1)
    valid_2m = block_fraction(valid, factor1) >= 0.75
    cond_2m = terrain_conditions(base_2m, None, 2.0)
    holdout_2m = (holdout_start_xy[0] // factor1, holdout_start_xy[1] // factor1)
    stage1_train = _stage_windows(macro_2m, cond_2m, valid_2m, window_cells=64, stride_cells=32, minimum_support=minimum_support, holdout_start_xy=holdout_2m, holdout=False)
    stage1_hold = _stage_windows(macro_2m, cond_2m, valid_2m, window_cells=64, stride_cells=32, minimum_support=minimum_support, holdout_start_xy=holdout_2m, holdout=True)

    factor2 = int(round(1.0 / source_pitch_m))
    structural_1m = block_mean(structural, factor2)
    base_1m = block_mean(base, factor2)
    valid_1m = block_fraction(valid, factor2) >= 0.75
    macro_truth_1m = resize(macro_2m, structural_1m.shape, order=3)
    innovation_1m = structural_1m - base_1m - macro_truth_1m
    cond_1m = np.concatenate((terrain_conditions(base_1m, None, 1.0), macro_truth_1m[None].astype(np.float32)))
    holdout_1m = (holdout_start_xy[0] // factor2, holdout_start_xy[1] // factor2)
    stage2_train = _stage_windows(innovation_1m, cond_1m, valid_1m, window_cells=64, stride_cells=32, minimum_support=minimum_support, holdout_start_xy=holdout_1m, holdout=False)
    stage2_hold = _stage_windows(innovation_1m, cond_1m, valid_1m, window_cells=64, stride_cells=32, minimum_support=minimum_support, holdout_start_xy=holdout_1m, holdout=True)
    surfaces = {"structural_1m": structural_1m, "base_1m": base_1m, "macro_2m": macro_2m, "valid_1m": valid_1m}
    return stage1_train, stage1_hold, stage2_train, stage2_hold, surfaces
