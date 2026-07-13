import numpy as np
from scipy.ndimage import gaussian_filter

from assetgen.process.micro_fixture import project_residual_zero_mean
from assetgen.process.microtopo.projection import build_smooth_mean_null_projector


FACTOR = 16


def _project(detail: np.ndarray, allowed: np.ndarray):
    rows, cols = detail.shape[0] // FACTOR, detail.shape[1] // FACTOR
    fully_soft = allowed.reshape(rows, FACTOR, cols, FACTOR).all(axis=(1, 3))
    support = build_smooth_mean_null_projector(
        np.zeros((rows, cols)), fully_soft, factor=FACTOR
    )
    taper = support.taper_window(0, 0, rows, cols)
    means = np.where(allowed, detail * taper, 0.0).reshape(
        rows, FACTOR, cols, FACTOR
    ).mean(axis=(1, 3))
    projector = build_smooth_mean_null_projector(
        means, fully_soft, factor=FACTOR
    )
    return projector, projector.project_window(detail, allowed, row0=0, col0=0)


def _phase_rms_cv(values: np.ndarray) -> float:
    rows, cols = values.shape[0] // FACTOR, values.shape[1] // FACTOR
    phase = values.reshape(rows, FACTOR, cols, FACTOR).transpose(1, 3, 0, 2)
    rms = np.sqrt(np.mean(phase * phase, axis=(2, 3)))
    return float(np.std(rms) / np.mean(rms))


def test_global_projection_reduces_repeated_cell_phase_basis():
    rng = np.random.default_rng(2)
    detail = gaussian_filter(rng.normal(size=(384, 384)), 3) * 0.05
    allowed = np.ones(detail.shape, dtype=bool)
    old = project_residual_zero_mean(detail, FACTOR)
    _, smooth = _project(detail, allowed)
    means = smooth.reshape(24, FACTOR, 24, FACTOR).mean(axis=(1, 3))
    assert np.max(np.abs(means)) < 1e-12
    assert _phase_rms_cv(smooth) < 0.65 * _phase_rms_cv(old)


def test_partial_cell_is_failed_closed_without_tiny_support_spike():
    rng = np.random.default_rng(8)
    detail = gaussian_filter(rng.normal(size=(192, 192)), 2) * 0.08
    allowed = np.ones(detail.shape, dtype=bool)
    allowed[:, : 4 * FACTOR] = False
    allowed[5, 4 * FACTOR + 3] = False
    detail[5, 4 * FACTOR + 3] = 0.8
    _, projected = _project(detail, allowed)
    means = projected.reshape(12, FACTOR, 12, FACTOR).mean(axis=(1, 3))
    assert np.max(np.abs(means)) < 1e-12
    assert np.count_nonzero(projected[~allowed]) == 0
    assert np.max(np.abs(projected)) < 0.2
    boundary_step = np.max(np.abs(np.diff(projected, axis=1)[:, 4 * FACTOR : 8 * FACTOR]))
    assert boundary_step < 0.08


def test_global_projection_windows_are_bit_exact_crops():
    rng = np.random.default_rng(5)
    detail = gaussian_filter(rng.normal(size=(160, 192)), 2) * 0.03
    allowed = np.ones(detail.shape, dtype=bool)
    projector, whole = _project(detail, allowed)
    crop = projector.project_window(
        detail[2 * FACTOR : 7 * FACTOR, 3 * FACTOR : 9 * FACTOR],
        allowed[2 * FACTOR : 7 * FACTOR, 3 * FACTOR : 9 * FACTOR],
        row0=2,
        col0=3,
    )
    assert np.array_equal(
        crop, whole[2 * FACTOR : 7 * FACTOR, 3 * FACTOR : 9 * FACTOR]
    )
