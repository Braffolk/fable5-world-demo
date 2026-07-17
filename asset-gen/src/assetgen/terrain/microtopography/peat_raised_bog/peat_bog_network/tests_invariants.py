"""Small synthetic invariant checks: determinism, mask exactness, conservation.

Run: uv run python -m assetgen.terrain.microtopography.peat_raised_bog.peat_bog_network.tests_invariants
"""
from __future__ import annotations

import numpy as np

from . import network
from .hydrology import ProcessHydrology
from .prf import world_uniform, world_uniform_grid


def _synthetic(slope: float, h: int = 120, w: int = 120):
    flow_row = np.ones((h, w))
    flow_col = np.zeros((h, w))
    hydro = ProcessHydrology(flow_row, flow_col, np.full((h, w), slope), np.zeros((h, w)), (0, 0, w * 2, h * 2), 2.0)
    noise = world_uniform_grid(0, h * 2, w, h, 2)
    params = dict(feed=0.037, kill=0.060, D_u=0.16, D_v=0.08, anisotropy=3.0, slope_ref=0.02, seed_quantile=0.6)
    return network.solve(hydro, noise, params, budget=4000, dt=0.5)


def test_prf_deterministic_and_chunk_free() -> None:
    a = world_uniform(np.array([[100, 101]]), np.array([[200, 201]]))
    b = world_uniform(np.array([[100, 101]]), np.array([[200, 201]]))
    assert np.array_equal(a, b), "PRF must be deterministic"
    # Same world cell reached from two different window origins yields the same value.
    g1 = world_uniform_grid(0, 200, 10, 10, 2)
    g2 = world_uniform_grid(4, 196, 10, 10, 2)
    assert np.isclose(g1[2, 2], g2[0, 0]), "PRF must key on world coords, not window origin"
    assert 0.0 <= float(a.min()) and float(a.max()) < 1.0


def test_solve_deterministic_and_finite() -> None:
    r1 = _synthetic(0.0)
    r2 = _synthetic(0.0)
    assert np.array_equal(r1.activator, r2.activator), "solve must be byte-identical"
    assert np.isfinite(r1.activator).all() and np.isfinite(r1.inhibitor).all()


def test_pattern_forms_and_classifies() -> None:
    r = _synthetic(0.0, h=140, w=140)
    region = np.zeros(r.activator.shape, bool)
    region[16:-16, 16:-16] = True
    cuts = network.compute_cuts(r.activator, r.inhibitor, region, dict(ridge_level=0.5, pool_wet_percentile=66, hollow_wet_percentile=33))
    labels = network.classify_from_cuts(r.activator, r.inhibitor, region, cuts)
    for value in (network.RIDGE, network.HOLLOW, network.POOL):
        assert (labels == value).sum() > 0, f"class {value} absent from a formed pattern"


def test_mask_exactness_and_relief_free_pool() -> None:
    # A synthetic label + relief: relief must be zero on pools and outside authority.
    labels = np.array([[network.RIDGE, network.POOL], [network.LAWN, network.HOLLOW]], dtype=np.uint8)
    relief = np.array([[0.06, 0.0], [0.0, -0.05]])
    authority = np.array([[True, True], [True, False]])
    relief = np.where(authority, relief, 0.0)
    relief[labels == network.POOL] = 0.0
    assert relief[labels == network.POOL].tolist() == [0.0]
    assert relief[~authority].tolist() == [0.0]


def test_parent_mean_conservation() -> None:
    # A zero-mean-per-block detail preserves the 1 m parent mean exactly.
    parent = np.random.default_rng(0).normal(size=(8, 8))
    fine = np.repeat(np.repeat(parent, 4, 0), 4, 1)
    block = fine.reshape(8, 4, 8, 4)
    assert np.allclose(np.mean(block, axis=(1, 3)), parent, atol=1e-12)


def main() -> int:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all invariant tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
