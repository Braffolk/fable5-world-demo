"""Normative F4/R4 analysis bands from the accepted terrain specification."""
from __future__ import annotations

import numpy as np

_F4 = np.asarray((1, 3, 6, 10, 12, 12, 10, 6, 3, 1), dtype=np.float64) / 64.0
_PHASES = np.asarray((-3.0 / 8.0, -1.0 / 8.0, 1.0 / 8.0, 3.0 / 8.0))


def _reflect_indices(indices: np.ndarray, size: int) -> np.ndarray:
    if size < 2:
        return np.zeros_like(indices)
    period = 2 * size
    folded = np.mod(indices, period)
    return np.where(folded < size, folded, period - folded - 1)


def _f4_axis(values: np.ndarray, axis: int) -> np.ndarray:
    size = values.shape[axis]
    if size % 4 != 0:
        raise ValueError("F4 input extent must be divisible by four")
    p = np.arange(size // 4, dtype=np.int64)
    indices = 4 * p[:, None] - 3 + np.arange(10, dtype=np.int64)[None, :]
    indices = _reflect_indices(indices, size)
    selected = np.take(values, indices, axis=axis)
    weight_shape = [1] * selected.ndim
    weight_shape[axis + 1] = 10
    return np.sum(selected * _F4.reshape(weight_shape), axis=axis + 1, dtype=np.float64)


def f4(values: np.ndarray) -> np.ndarray:
    """Apply the exact separable factor-four analysis filter east then north."""
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError("F4 requires a two-dimensional surface")
    east = _f4_axis(source, axis=1)
    return _f4_axis(east, axis=0)


def _keys_weights(t: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            -0.5 * t + t * t - 0.5 * t**3,
            1.0 - 2.5 * t * t + 1.5 * t**3,
            0.5 * t + 2.0 * t * t - 1.5 * t**3,
            -0.5 * t * t + 0.5 * t**3,
        ),
        axis=-1,
    )


def _r4_axis(values: np.ndarray, axis: int) -> np.ndarray:
    size = values.shape[axis]
    fine_size = size * 4
    fine = np.arange(fine_size, dtype=np.int64)
    coarse = fine // 4
    phase = _PHASES[fine % 4]
    indices = coarse[:, None] + np.asarray((-1, 0, 1, 2), dtype=np.int64)
    indices = _reflect_indices(indices, size)
    selected = np.take(values, indices, axis=axis)
    weights = _keys_weights(phase)
    weight_shape = [1] * selected.ndim
    weight_shape[axis] = fine_size
    weight_shape[axis + 1] = 4
    return np.sum(selected * weights.reshape(weight_shape), axis=axis + 1, dtype=np.float64)


def r4(values: np.ndarray) -> np.ndarray:
    """Reconstruct factor-four with exact Keys cubic phases from the spec."""
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError("R4 requires a two-dimensional surface")
    east = _r4_axis(source, axis=1)
    return _r4_axis(east, axis=0)


def b1_b2(height: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return B1 at 0.25 m and B2 at 0.0625 m for a fine master."""
    fine = np.asarray(height, dtype=np.float64)
    a1 = f4(fine)
    a0 = f4(a1)
    return a1 - r4(a0), fine - r4(a1)
