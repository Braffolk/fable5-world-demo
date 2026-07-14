"""Frozen F4/R4 B1 evidence and conservative weak-geometry weighting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


F4_KERNEL = np.asarray(
    (0.015625, 0.046875, 0.09375, 0.15625, 0.1875, 0.1875, 0.15625, 0.09375, 0.046875, 0.015625),
    dtype=np.float64,
)


@dataclass(frozen=True)
class B1Confidence:
    b_a_fine: np.ndarray
    b_b_fine: np.ndarray
    local_energy_fine: np.ndarray
    local_confidence_fine: np.ndarray
    view_confidence_fine: np.ndarray
    weight_fine: np.ndarray
    valid_fine: np.ndarray
    c_area: float
    signal_m2: float
    noise_m2: float
    c_band: float


def _f4(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact separable 4x filter; output p consumes input 4p-3..4p+6."""
    source = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=np.bool_)
    if source.shape != mask.shape or source.ndim != 2:
        raise ValueError("F4 requires aligned two-dimensional values and support")
    out_h = max(0, (source.shape[0] - 7) // 4)
    out_w = max(0, (source.shape[1] - 7) // 4)
    horizontal = np.zeros((source.shape[0], out_w), dtype=np.float64)
    horizontal_valid = np.ones((source.shape[0], out_w), dtype=np.bool_)
    for p in range(out_w):
        selection = source[:, 4 * p + 1 : 4 * p + 11]
        selection_valid = mask[:, 4 * p + 1 : 4 * p + 11]
        horizontal[:, p] = selection @ F4_KERNEL
        horizontal_valid[:, p] = selection_valid.all(axis=1)
    output = np.zeros((out_h, out_w), dtype=np.float64)
    output_valid = np.ones((out_h, out_w), dtype=np.bool_)
    for p in range(out_h):
        selection = horizontal[4 * p + 1 : 4 * p + 11]
        output[p] = F4_KERNEL @ selection
        output_valid[p] = horizontal_valid[4 * p + 1 : 4 * p + 11].all(axis=0)
    output[~output_valid] = np.nan
    return output, output_valid


def _keys(x: np.ndarray) -> np.ndarray:
    a = -0.5
    distance = np.abs(x)
    return np.where(
        distance < 1,
        (a + 2) * distance**3 - (a + 3) * distance**2 + 1,
        np.where(
            distance < 2,
            a * distance**3 - 5 * a * distance**2 + 8 * a * distance - 4 * a,
            0.0,
        ),
    )


def _r4(values: np.ndarray, valid: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Keys a=-0.5 reconstruction at phases -3/8,-1/8,+1/8,+3/8."""
    output = np.full(shape, np.nan, dtype=np.float64)
    output_valid = np.zeros(shape, dtype=np.bool_)
    phases = (-0.375, -0.125, 0.125, 0.375)
    for y in range(shape[0]):
        cy = y // 4
        py = phases[y % 4]
        yi = np.arange(cy - 1, cy + 3)
        if yi[0] < 0 or yi[-1] >= values.shape[0]:
            continue
        wy = _keys(yi - (cy + py))
        for x in range(shape[1]):
            cx = x // 4
            px = phases[x % 4]
            xi = np.arange(cx - 1, cx + 3)
            if xi[0] < 0 or xi[-1] >= values.shape[1] or not valid[np.ix_(yi, xi)].all():
                continue
            wx = _keys(xi - (cx + px))
            output[y, x] = wy @ values[np.ix_(yi, xi)] @ wx
            output_valid[y, x] = True
    return output, output_valid


def _nearest_to_fine(values: np.ndarray, shape: tuple[int, int], fill: float) -> np.ndarray:
    output = np.full(shape, fill, dtype=np.float64)
    height = min(shape[0], values.shape[0] * 4)
    width = min(shape[1], values.shape[1] * 4)
    output[:height, :width] = np.repeat(np.repeat(values, 4, axis=0), 4, axis=1)[:height, :width]
    return output


def b1_confidence(
    height_a: np.ndarray,
    height_b: np.ndarray,
    joint_direct: np.ndarray,
    unique_geometry: np.ndarray,
    count_a: np.ndarray,
    count_b: np.ndarray,
) -> B1Confidence:
    m_c = np.asarray(joint_direct, dtype=np.bool_)
    g_c = m_c & np.asarray(unique_geometry, dtype=np.bool_)
    if not np.any(m_c):
        c_area = 0.0
    else:
        c_area = float(g_c.sum() / m_c.sum())
    a1_a, a1_valid_a = _f4(height_a, g_c)
    a1_b, a1_valid_b = _f4(height_b, g_c)
    common_a1 = a1_valid_a & a1_valid_b
    a0_a, a0_valid_a = _f4(a1_a, common_a1)
    a0_b, a0_valid_b = _f4(a1_b, common_a1)
    reconstructed_a, reconstructed_valid_a = _r4(a0_a, a0_valid_a, a1_a.shape)
    reconstructed_b, reconstructed_valid_b = _r4(a0_b, a0_valid_b, a1_b.shape)
    band_valid = common_a1 & reconstructed_valid_a & reconstructed_valid_b
    b_a = np.where(band_valid, a1_a - reconstructed_a, np.nan)
    b_b = np.where(band_valid, a1_b - reconstructed_b, np.nan)
    if np.any(band_valid):
        signal = max(0.0, float(np.mean(b_a[band_valid] * b_b[band_valid])))
        noise = float(0.5 * np.mean((b_a[band_valid] - b_b[band_valid]) ** 2))
    else:
        signal = noise = 0.0
    c_band = signal / (signal + noise) if signal + noise > 0 else 0.0
    local_energy = 0.5 * (b_a - b_b) ** 2
    denominator = noise + local_energy
    local = np.divide(
        noise,
        denominator,
        out=np.zeros_like(local_energy),
        where=denominator > 0,
    )
    local[(noise == 0) & (local_energy == 0) & band_valid] = 1.0
    # Counts are conservative minima over each 4x4 source footprint.
    coarse_count_a = np.zeros(a1_a.shape, dtype=np.float64)
    coarse_count_b = np.zeros(a1_a.shape, dtype=np.float64)
    for y in range(a1_a.shape[0]):
        for x in range(a1_a.shape[1]):
            coarse_count_a[y, x] = np.min(count_a[4 * y : 4 * y + 4, 4 * x : 4 * x + 4])
            coarse_count_b[y, x] = np.min(count_b[4 * y : 4 * y + 4, 4 * x : 4 * x + 4])
    maximum = np.maximum(coarse_count_a, coarse_count_b)
    view = np.divide(
        np.minimum(coarse_count_a, coarse_count_b),
        maximum,
        out=np.zeros_like(maximum),
        where=maximum > 0,
    )
    weight = np.minimum.reduce((np.full_like(local, c_area), np.full_like(local, c_band), local, view))
    weight[~band_valid] = 0.0
    fine_shape = height_a.shape
    fine_valid = g_c & (_nearest_to_fine(band_valid, fine_shape, 0.0) > 0)
    weight_fine = _nearest_to_fine(weight, fine_shape, 0.0)
    weight_fine[~fine_valid] = 0.0
    b_a_fine, va = _r4(b_a, band_valid, fine_shape)
    b_b_fine, vb = _r4(b_b, band_valid, fine_shape)
    fine_valid &= va & vb
    weight_fine[~fine_valid] = 0.0
    return B1Confidence(
        b_a_fine=b_a_fine,
        b_b_fine=b_b_fine,
        local_energy_fine=_nearest_to_fine(local_energy, fine_shape, np.nan),
        local_confidence_fine=_nearest_to_fine(local, fine_shape, 0.0),
        view_confidence_fine=_nearest_to_fine(view, fine_shape, 0.0),
        weight_fine=weight_fine,
        valid_fine=fine_valid,
        c_area=c_area,
        signal_m2=signal,
        noise_m2=noise,
        c_band=c_band,
    )


def capacity(weight: np.ndarray, partition: np.ndarray, cell_m: float = 0.0625) -> dict:
    """Weighted area and Kish capacity over non-overlapping central 2 m windows."""
    window_cells = round(2.0 / cell_m)
    rows = []
    for y in range(0, weight.shape[0] - window_cells + 1, window_cells):
        for x in range(0, weight.shape[1] - window_cells + 1, window_cells):
            part = partition[y : y + window_cells, x : x + window_cells]
            values = weight[y : y + window_cells, x : x + window_cells]
            roles = np.unique(part)
            if len(roles) != 1 or roles[0] == 255:
                continue
            rows.append((int(roles[0]), float(values.sum() * cell_m**2)))
    train = np.asarray([value for role, value in rows if role == 0], dtype=np.float64)
    kish = float(train.sum() ** 2 / np.sum(train**2)) if np.any(train > 0) else 0.0
    development = sum(value > 0 for role, value in rows if role == 1)
    audit = sum(value > 0 for role, value in rows if role == 2)
    return {
        "weighted_effective_train_area_m2": float(train.sum()),
        "kish_effective_nonoverlapping_train_windows": kish,
        "complete_nonzero_weight_windows_development": development,
        "complete_nonzero_weight_windows_internal_audit": audit,
        "passed": bool(train.sum() >= 32.0 and kish >= 8.0 and development >= 1 and audit >= 1),
    }
