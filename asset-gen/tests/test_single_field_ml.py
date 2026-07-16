import numpy as np

from assetgen.terrain.microtopography.erodible_slope.single_field_ml.contracts import (
    parent_mean_metrics,
    soft_parent_projection,
)


def test_soft_parent_projection_preserves_hard_cells_without_exact_closure() -> None:
    delta = np.arange(64, dtype=np.float64).reshape(8, 8) / 100.0
    hard = np.zeros_like(delta, dtype=bool)
    hard[2, 2] = True
    projected = soft_parent_projection(delta, hard, factor=4, strength=0.75)

    assert projected[2, 2] == 0.0
    metrics = parent_mean_metrics(projected, hard, factor=4)
    assert 0.0 < metrics["absolute_max_m"] < parent_mean_metrics(delta, hard, factor=4)["absolute_max_m"]

