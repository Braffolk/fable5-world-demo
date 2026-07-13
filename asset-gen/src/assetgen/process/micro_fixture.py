"""Stage-1 calibrated geometry fixture, not the production morphology generator.

The functions here prove conservative reconstruction, packing, and renderer
retention. They deliberately expose fixed measured amplitudes/wavelengths and
must not be described as geology- or soil-conditioned terrain synthesis.
"""
from __future__ import annotations

import numpy as np


def c2_cell_weights(factor: int) -> np.ndarray:
    """Discrete unit-mean samples of u^3(1-u)^3 on subcell centers."""
    if factor < 2:
        raise ValueError("conservative refinement factor must be at least 2")
    u = (np.arange(factor, dtype=np.float64) + 0.5) / factor
    weights = u**3 * (1.0 - u) ** 3
    return weights / np.mean(weights)


def conservative_cell_correct(
    smooth_samples: np.ndarray,
    authority_cells: np.ndarray,
    factor: int = 16,
) -> np.ndarray:
    """Correct a smooth fine surface to exact discrete coarse-cell means.

    `smooth_samples` contains complete fine subcells, without an apron. The C2
    bubble and its first two derivatives vanish at cell edges, so corrections
    do not introduce value/slope/curvature seams into the smooth reconstruction.
    """
    fine = np.asarray(smooth_samples, dtype=np.float64)
    authority = np.asarray(authority_cells, dtype=np.float64)
    cy, cx = authority.shape
    if fine.shape != (cy * factor, cx * factor):
        raise ValueError(f"fine shape {fine.shape} != {(cy * factor, cx * factor)}")
    blocks = fine.reshape(cy, factor, cx, factor)
    error = authority - blocks.mean(axis=(1, 3))
    w = c2_cell_weights(factor)
    correction = error[:, None, :, None] * w[None, :, None, None] * w[None, None, None, :]
    return (blocks + correction).reshape(fine.shape)


def project_residual_zero_mean(
    residual: np.ndarray,
    factor: int = 16,
    allowed: np.ndarray | None = None,
) -> np.ndarray:
    """Project detail into the 1 m mean-null space without touching hard masks."""
    detail = np.asarray(residual, dtype=np.float64).copy()
    if detail.shape[0] % factor or detail.shape[1] % factor:
        raise ValueError("residual dimensions must be multiples of the refinement factor")
    if allowed is None:
        mask = np.ones(detail.shape, dtype=bool)
    else:
        mask = np.asarray(allowed, dtype=bool)
        if mask.shape != detail.shape:
            raise ValueError("hard mask shape differs from residual")
    detail[~mask] = 0.0
    cy, cx = detail.shape[0] // factor, detail.shape[1] // factor
    blocks = detail.reshape(cy, factor, cx, factor)
    masks = mask.reshape(cy, factor, cx, factor)
    means = blocks.mean(axis=(1, 3))
    w1 = c2_cell_weights(factor)
    weights = w1[None, :, None, None] * w1[None, None, None, :] * masks
    weight_means = weights.mean(axis=(1, 3))
    scale = np.divide(means, weight_means, out=np.zeros_like(means), where=weight_means > 0)
    projected = blocks - scale[:, None, :, None] * weights
    projected[~masks] = 0.0
    return projected.reshape(detail.shape)


def calibrated_fixture_residual(easting: np.ndarray, northing: np.ndarray) -> np.ndarray:
    """Absolute-coordinate retention fixture with 0.125-1 m wavelengths.

    Smooth gates make four feature families spatially legible instead of mixing
    them into decorative broadband noise. Cell-mean projection is a separate,
    mandatory step.
    """
    e = np.asarray(easting, dtype=np.float64)
    n = np.asarray(northing, dtype=np.float64)
    if e.shape != n.shape:
        raise ValueError("fixture coordinate arrays must have equal shapes")
    phase_e = e - 679424.0
    phase_n = n - 6443008.0
    zone = np.mod(np.floor(phase_e / 32.0) + np.floor(phase_n / 32.0), 4).astype(np.int8)
    out = np.zeros(e.shape, dtype=np.float64)
    families = (
        (0, 0.02, 0.125, 0.31),
        (1, 0.05, 0.25, -0.47),
        (2, 0.10, 0.50, 0.73),
        (3, 0.20, 1.00, -0.19),
    )
    for family, amplitude, wavelength, angle in families:
        along = phase_e * np.cos(angle) + phase_n * np.sin(angle)
        cross = -phase_e * np.sin(angle) + phase_n * np.cos(angle)
        carrier = np.sin(2.0 * np.pi * along / wavelength)
        modulation = 0.65 + 0.35 * np.cos(2.0 * np.pi * cross / (wavelength * 5.0))
        out += (zone == family) * amplitude * carrier * modulation
    return out
