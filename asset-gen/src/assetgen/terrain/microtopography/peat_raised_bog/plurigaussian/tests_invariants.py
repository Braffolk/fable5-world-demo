"""Lean invariants for the plurigaussian route (algorithms still in flux; not a QA gate)."""
from __future__ import annotations

import numpy as np

from . import anamorphosis, fields


def test_world_white_is_world_keyed_and_crop_independent() -> None:
    """A world cell gets the same innovation regardless of the grid's crop window."""
    big = fields.WorkGrid(east0=540000.0, north1=6430000.0, width=64, height=64, pitch=0.25)
    # A sub-window sharing the same world lattice origin offset.
    sub = fields.WorkGrid(east0=540000.0 + 8 * 0.25, north1=6430000.0 - 8 * 0.25, width=32, height=32, pitch=0.25)
    wb = fields.world_white(big, "fine")
    ws = fields.world_white(sub, "fine")
    assert np.allclose(wb[8:40, 8:40], ws), "world_white must be crop-independent"


def test_channels_are_independent() -> None:
    g = fields.WorkGrid(east0=540000.0, north1=6430000.0, width=48, height=48, pitch=0.25)
    a = fields.world_white(g, "fine")
    b = fields.world_white(g, "coarse")
    assert abs(float(np.corrcoef(a.ravel(), b.ravel())[0, 1])) < 0.1


def test_estonian_fractions_sum_to_one() -> None:
    f = anamorphosis.estonian_fractions()
    assert abs(sum(f.values()) - 1.0) < 1e-9
    p_lo, p_hi = anamorphosis.class_percentile_bounds()
    assert 0.0 < p_lo < p_hi < 1.0


def test_anamorphosis_is_monotone_continuous() -> None:
    p = np.linspace(0.0, 1.0, 501)
    h = anamorphosis.base_height(p)
    assert np.all(np.diff(h) >= -1e-9), "anamorphosis must be monotone (no per-class steps)"
    # continuous: bounded finite differences (no jumps that would read as blocky steps)
    assert float(np.max(np.abs(np.diff(h)))) < 0.02
