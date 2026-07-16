"""NumPy-only composition contracts for the optional learned challenger."""
from __future__ import annotations

import numpy as np


def smoothstep(value: np.ndarray) -> np.ndarray:
    x = np.clip(value, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def soft_parent_projection(
    delta_m: np.ndarray,
    hard: np.ndarray,
    factor: int,
    strength: float,
) -> np.ndarray:
    """Move active block means toward C0 without asserting exact DTM truth."""
    if factor < 1 or not 0.0 <= strength < 1.0:
        raise ValueError("soft projection requires factor >= 1 and strength in [0, 1)")
    result = np.asarray(delta_m, dtype=np.float64).copy()
    rows = result.shape[0] // factor * factor
    cols = result.shape[1] // factor * factor
    for row in range(0, rows, factor):
        for col in range(0, cols, factor):
            region = np.s_[row : row + factor, col : col + factor]
            active = ~hard[region]
            if np.any(active):
                block = result[region]
                block[active] -= strength * float(np.mean(block[active]))
                block[~active] = 0.0
    result[hard] = 0.0
    return result


def parent_mean_metrics(delta_m: np.ndarray, hard: np.ndarray, factor: int) -> dict[str, float]:
    means: list[float] = []
    for row in range(0, delta_m.shape[0] - factor + 1, factor):
        for col in range(0, delta_m.shape[1] - factor + 1, factor):
            region = np.s_[row : row + factor, col : col + factor]
            active = ~hard[region]
            if np.any(active):
                means.append(float(np.mean(delta_m[region][active])))
    values = np.abs(np.asarray(means, dtype=np.float64))
    return {
        "absolute_p95_m": float(np.percentile(values, 95.0)),
        "absolute_max_m": float(np.max(values)),
    }

