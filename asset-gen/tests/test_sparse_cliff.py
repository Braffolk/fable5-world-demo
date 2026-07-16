from __future__ import annotations

import numpy as np

from assetgen.terrain.microtopography.erodible_slope.sparse_cliff.model import _project_one_metre_closure


def test_one_metre_projection_preserves_hard_cells_and_parent_means() -> None:
    rng = np.random.default_rng(17)
    delta = rng.normal(0.0, 0.2, (12, 16))
    hard = np.zeros_like(delta, dtype=bool)
    hard[2:6, 3] = True
    result = _project_one_metre_closure(delta, hard, 4)
    assert np.max(np.abs(result[hard])) == 0.0
    means = result.reshape(3, 4, 4, 4).mean(axis=(1, 3))
    assert np.max(np.abs(means)) < 1.0e-12


def test_hard_transition_is_not_reauthorized_by_projection() -> None:
    delta = np.ones((8, 8), dtype=np.float64)
    hard = np.zeros_like(delta, dtype=bool)
    hard[:, :4] = True
    result = _project_one_metre_closure(delta, hard, 4)
    assert np.array_equal(result[:, :4], np.zeros((8, 4)))
    assert np.max(np.abs(result[:, 4:])) == 0.0
