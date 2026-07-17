"""Synthetic invariants for v4: D-modulation monotonicity/bounds, determinism, reduction
to v3 at D_scale==1, and mire-scale gate plumbing.

Run: uv run python -m assetgen.terrain.microtopography.peat_raised_bog.peat_bog_network.tests_invariants_v4
"""
from __future__ import annotations

import numpy as np

from . import gates_v4, network, network_v4
from .hydrology import ProcessHydrology
from .prf import world_uniform_grid

PARAMS = dict(feed=0.037, kill=0.060, D_u=0.16, D_v=0.08, anisotropy=3.0, slope_ref=0.02, seed_quantile=0.6)


def _hydro(slope: np.ndarray):
    h, w = slope.shape
    return ProcessHydrology(np.ones((h, w)), np.zeros((h, w)), slope, np.zeros((h, w)), (0, 0, w * 2, h * 2), 2.0)


def test_d_scale_monotone_and_bounded() -> None:
    slope = np.linspace(0.0, 0.05, 100).reshape(10, 10)
    auth = np.ones(slope.shape, bool)
    d = network_v4.compute_d_scale(slope, auth, d_min=0.15, k=2.0)
    assert d.min() >= 0.15 - 1e-12 and d.max() <= 1.0 + 1e-12, "D_scale must stay in [d_min, 1]"
    flat = slope.ravel()
    order = np.argsort(flat)
    ds = d.ravel()[order]
    assert np.all(np.diff(ds) <= 1e-12), "D_scale must be monotone non-increasing in slope"
    # steeper -> shorter wavelength (sqrt(D) smaller).
    assert np.sqrt(d.ravel()[np.argmax(flat)]) < np.sqrt(d.ravel()[np.argmin(flat)])


def test_v4_reduces_to_v3_when_dscale_one() -> None:
    slope = np.full((80, 80), 0.01)
    hydro = _hydro(slope)
    noise = world_uniform_grid(0, 160, 80, 80, 2)
    v3 = network.solve(hydro, noise, PARAMS, budget=1500, dt=0.5)
    v4 = network_v4.solve_v4(hydro, noise, PARAMS, np.ones(slope.shape), budget=1500, dt=0.5)
    assert np.array_equal(v3.activator, v4.activator), "v4 with D_scale==1 must byte-match v3"
    assert np.array_equal(v3.inhibitor, v4.inhibitor)


def test_v4_deterministic() -> None:
    slope = np.linspace(0.0, 0.02, 80 * 80).reshape(80, 80)
    hydro = _hydro(slope)
    noise = world_uniform_grid(0, 160, 80, 80, 2)
    d = network_v4.compute_d_scale(slope, np.ones(slope.shape, bool), d_min=0.15, k=2.0)
    a = network_v4.solve_v4(hydro, noise, PARAMS, d, budget=800, dt=0.5)
    b = network_v4.solve_v4(hydro, noise, PARAMS, d, budget=800, dt=0.5)
    assert np.array_equal(a.activator, b.activator), "solve_v4 must be byte-identical"
    assert np.isfinite(a.activator).all() and np.isfinite(a.inhibitor).all()


def test_mire_scale_gate_plumbing() -> None:
    # Synthetic: horizontal ridges every 8 cells over a 128x128 region -> patterned blocks.
    labels = np.zeros((128, 128), np.uint8)
    labels[::8, :] = network.RIDGE
    region = np.ones(labels.shape, bool)
    cfg = dict(
        block_size_cells=64, min_ridge_cells=64, block_orientation_resultant_R_dom_max=0.50,
        pooled_spacing_cv_min=0.35, regime_split_R_int=0.50, min_blocks_per_regime=1, min_fraction_per_regime=0.05,
    )
    report, fields = gates_v4.compute_mire_scale_gates(labels, region, 2.0, cfg)
    assert report["block_grid"]["blocks_scanned"] == 4
    assert report["block_grid"]["patterned_blocks"] == 4
    assert fields["skeleton"].shape == labels.shape
    assert np.isfinite(fields["block_spacing"]).any()
    # Evenly-spaced synthetic strings must be LOW spacing-CV (fails the >=0.35 gate) -> the
    # gate discriminates corrugations, as designed.
    assert report["gate_b_pooled_spacing_cv"]["pooled_spacing_cv"] < 0.35


def main() -> int:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all v4 invariant tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
