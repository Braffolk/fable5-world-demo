"""Structural-only reconstruction from 0.25 m samples to 0.0625 m samples.

This module does not synthesize unresolved morphology. It reconstructs a smooth
fine lattice while preserving the discrete mean of every requested parent
sample exactly up to float64 rounding.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_FACTOR = 4
_PHASES = np.array((-3.0 / 8.0, -1.0 / 8.0, 1.0 / 8.0, 3.0 / 8.0))
_OFFSETS = np.arange(-2, 3, dtype=np.int64)


def _keys_weights() -> np.ndarray:
    """Return phase weights over parent offsets -2 through +2."""
    weights = np.zeros((_FACTOR, len(_OFFSETS)), dtype=np.float64)
    for phase_index, phase in enumerate(_PHASES):
        base = int(np.floor(phase))
        t = phase - base
        cubic = (
            -0.5 * t + t * t - 0.5 * t * t * t,
            1.0 - 2.5 * t * t + 1.5 * t * t * t,
            0.5 * t + 2.0 * t * t - 1.5 * t * t * t,
            -0.5 * t * t + 0.5 * t * t * t,
        )
        for relative_offset, weight in zip(range(base - 1, base + 3), cubic):
            weights[phase_index, relative_offset + 2] = weight
    return weights


_WEIGHTS = _keys_weights()
_BUBBLE_1D = np.array(
    [u**3 * (1.0 - u) ** 3 for u in ((0.5 / 4.0), (1.5 / 4.0), (2.5 / 4.0), (3.5 / 4.0))],
    dtype=np.float64,
)
_BUBBLE_1D /= sum(_BUBBLE_1D) / _FACTOR
_BUBBLE_2D = _BUBBLE_1D[:, None] * _BUBBLE_1D[None, :]


def _validated_interval(
    interval: Sequence[int], side: int, axis_name: str
) -> tuple[int, int]:
    if len(interval) != 2:
        raise ValueError(f"{axis_name} must contain (start, stop)")
    start, stop = interval
    if isinstance(start, (bool, np.bool_)) or isinstance(stop, (bool, np.bool_)):
        raise TypeError(f"{axis_name} bounds must be integers")
    if not isinstance(start, (int, np.integer)) or not isinstance(stop, (int, np.integer)):
        raise TypeError(f"{axis_name} bounds must be integers")
    start, stop = int(start), int(stop)
    if start >= stop:
        raise ValueError(f"{axis_name} must be a non-empty half-open interval")
    if start < 2 or stop > side - 2:
        raise ValueError(
            f"{axis_name} requires two parent support samples before start and after stop"
        )
    return start, stop


def _weighted_sum5(samples: np.ndarray, count: int, weights: np.ndarray) -> np.ndarray:
    """Accumulate five aligned slices in a fixed float64 operation order."""
    result = samples[..., 0:count] * weights[0]
    for offset in range(1, 5):
        result = result + samples[..., offset : offset + count] * weights[offset]
    return result


def prolong_structural_4x(
    parent_support: np.ndarray,
    *,
    parent_rows: Sequence[int],
    parent_cols: Sequence[int],
) -> np.ndarray:
    """Reconstruct a requested parent crop onto its 4x finer sample lattice.

    ``parent_rows`` and ``parent_cols`` are half-open sample-index intervals in
    ``parent_support``. Each requested parent sample owns the four fine sample
    centers at phases ``-3/8, -1/8, +1/8, +3/8`` of the parent spacing.

    Keys cubic convolution with ``a=-0.5`` requires two parent samples on both
    sides of the requested crop. Callers must supply real neighboring authority
    samples and expand tiled reads accordingly. This function never clamps,
    wraps, reflects, or invents storage-boundary support.

    Interpolation is accumulated in float64 east then north. A separable
    ``u^3(1-u)^3`` bubble then restores every 4x4 block's discrete mean to its
    parent value. The returned shape is
    ``(4 * requested_rows, 4 * requested_cols)`` and dtype is float64.
    """
    source = np.asarray(parent_support)
    if source.ndim != 2:
        raise ValueError("parent_support must be a 2D raster")
    row_start, row_stop = _validated_interval(parent_rows, source.shape[0], "parent_rows")
    col_start, col_stop = _validated_interval(parent_cols, source.shape[1], "parent_cols")

    # Limit float64 conversion to the crop's declared five-tap support window.
    support = np.asarray(
        source[row_start - 2 : row_stop + 2, col_start - 2 : col_stop + 2],
        dtype=np.float64,
    )
    if not np.isfinite(support).all():
        raise ValueError("requested parent support contains nonfinite height")

    parent_row_count = row_stop - row_start
    parent_col_count = col_stop - col_start

    # East pass retains the north/south halo for the second separable pass.
    east = np.empty(
        (parent_row_count + 4, parent_col_count * _FACTOR), dtype=np.float64
    )
    for phase_index in range(_FACTOR):
        east[:, phase_index::_FACTOR] = _weighted_sum5(
            support, parent_col_count, _WEIGHTS[phase_index]
        )

    fine = np.empty(
        (parent_row_count * _FACTOR, parent_col_count * _FACTOR), dtype=np.float64
    )
    north_samples = np.moveaxis(east, 0, -1)
    for phase_index in range(_FACTOR):
        fine[phase_index::_FACTOR, :] = _weighted_sum5(
            north_samples, parent_row_count, _WEIGHTS[phase_index]
        ).T

    blocks = fine.reshape(parent_row_count, _FACTOR, parent_col_count, _FACTOR)
    interpolated_mean = np.zeros(
        (parent_row_count, parent_col_count), dtype=np.float64
    )
    for row_phase in range(_FACTOR):
        for col_phase in range(_FACTOR):
            interpolated_mean = interpolated_mean + blocks[:, row_phase, :, col_phase]
    interpolated_mean *= 1.0 / (_FACTOR * _FACTOR)

    authority = support[2 : 2 + parent_row_count, 2 : 2 + parent_col_count]
    delta = authority - interpolated_mean
    blocks += delta[:, None, :, None] * _BUBBLE_2D[None, :, None, :]
    return fine
