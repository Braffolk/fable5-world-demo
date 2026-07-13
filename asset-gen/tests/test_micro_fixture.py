import numpy as np

from assetgen.process.micro_fixture import (
    calibrated_fixture_residual,
    conservative_cell_correct,
    project_residual_zero_mean,
)


def test_conservative_correction_matches_every_authority_cell_mean():
    rng = np.random.default_rng(7)
    authority = rng.normal(50.0, 3.0, (5, 7))
    smooth = rng.normal(50.0, 1.0, (5 * 16, 7 * 16))
    corrected = conservative_cell_correct(smooth, authority)
    means = corrected.reshape(5, 16, 7, 16).mean(axis=(1, 3))
    assert np.max(np.abs(means - authority)) < 1e-12


def test_masked_projection_is_zero_mean_and_exactly_zero_on_forbidden_samples():
    rng = np.random.default_rng(11)
    detail = rng.normal(0.0, 0.1, (64, 48))
    allowed = np.ones(detail.shape, dtype=bool)
    allowed[4:20, 7:29] = False
    projected = project_residual_zero_mean(detail, allowed=allowed)
    means = projected.reshape(4, 16, 3, 16).mean(axis=(1, 3))
    assert np.max(np.abs(means)) < 1e-14
    assert np.count_nonzero(projected[~allowed]) == 0


def test_fixture_is_absolute_coordinate_deterministic_and_has_calibrated_range():
    offsets = (np.arange(512) + 0.5) / 64
    e = np.stack([679424.0 + family * 32.0 + offsets for family in range(4)])
    n = np.full(e.shape, 6443008.0 + 0.5)
    a = calibrated_fixture_residual(e, n)
    b = calibrated_fixture_residual(e.copy(), n.copy())
    assert np.array_equal(a, b)
    assert 0.15 < np.max(np.abs(a)) <= 0.2
