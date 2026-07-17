"""Small synthetic determinism + partition-invariance checks (no heavy IO)."""
from __future__ import annotations

import numpy as np

from . import prf
from .forms import _clods, _row_boundaries
from .geometry import ParcelInfo

_PARAMS = {
    "row_spacing_mean_m": 0.22, "row_spacing_lognormal_sigma": 0.34,
    "row_spacing_clamp_m": [0.14, 0.55], "row_relief_m": 0.011, "row_amp_cv": 0.35,
    "row_dropout_fraction": 0.06, "clod_cell_m": 0.42, "clod_base_occupancy": 0.5,
    "clod_diameter_m": [0.14, 0.30], "clod_height_m": [0.008, 0.032],
    "clod_elongation_range": [0.55, 0.95], "clod_shape_exp_range": [1.7, 3.4],
}
_MIX = {
    "state_macro_cell_m": 18.0,
    "state_weights": {"seedbed": 0.25, "crusted_seedbed": 0.625, "smooth_crusted": 0.125},
    "state_form_scalings": {
        "seedbed": {"clod_density": 1.0, "clod_size": 1.0},
        "crusted_seedbed": {"clod_density": 0.55, "clod_size": 0.85},
        "smooth_crusted": {"clod_density": 0.28, "clod_size": 0.7}},
}


def test_prf_deterministic() -> None:
    assert prf.uniform(679100, 6443700, 0) == prf.uniform(679100, 6443700, 0)
    assert prf.uniform(1, 2, 0) != prf.uniform(1, 2, 1)


def test_row_boundaries_variable_and_above_nyquist() -> None:
    info = ParcelInfo(3477279, 2.38, 679253.0, 6443717.0, 240.0)
    b, w, a = _row_boundaries(info, -60.0, 60.0, _PARAMS)
    assert np.all(np.diff(b) > 0)
    assert w.min() >= 0.14 - 1e-9
    cv = float(np.std(w) / np.mean(w))
    assert cv >= 0.25, cv


def test_clods_partition_invariant() -> None:
    """A clod field cropped from a larger window equals the direct sub-window."""
    def grid(e0, e1, n0, n1, texel=0.0625):
        east = e0 + 0.03125 + np.arange(int(round((e1 - e0) / texel))) * texel
        north = n1 - 0.03125 - np.arange(int(round((n1 - n0) / texel))) * texel
        return east, north
    den = np.ones((1, 1)); siz = np.ones((1, 1))
    e_big, n_big = grid(679168.0, 679200.0, 6443744.0, 6443776.0)
    big, _ = _clods(e_big, n_big, _PARAMS, den, siz, 1.0, _MIX)
    e_sub, n_sub = grid(679180.0, 679196.0, 6443752.0, 6443768.0)
    sub, _ = _clods(e_sub, n_sub, _PARAMS, den, siz, 1.0, _MIX)
    ci = {round(float(v), 6): i for i, v in enumerate(e_big)}
    ri = {round(float(v), 6): i for i, v in enumerate(n_big)}
    cols = [ci[round(float(v), 6)] for v in e_sub]
    rows = [ri[round(float(v), 6)] for v in n_sub]
    assert np.array_equal(big[np.ix_(rows, cols)], sub), "clods not partition-invariant"


if __name__ == "__main__":
    test_prf_deterministic()
    test_row_boundaries_variable_and_above_nyquist()
    test_clods_partition_invariant()
    print("agriculture_v2 invariant tests PASS")
