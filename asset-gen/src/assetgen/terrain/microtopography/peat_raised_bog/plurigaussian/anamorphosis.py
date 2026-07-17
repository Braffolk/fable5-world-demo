"""Gaussian anamorphosis: continuous monotone latent-percentile -> relative-height transfer.

This is the anti-checkerboard step. Height is a SMOOTH monotone function of the latent's
Gaussian percentile, NOT a per-class constant (per-class constants made the marked-form/step
route blocky). The transfer is calibrated to Moore's pooled per-class relative-height
quantiles and the reconciled 0.20-0.40 m hummock-hollow amplitude (Moore is a clipped-moss
LOWER bound; Stordalen/Ilyasov set the true amplitude ~0.30 m p95-p5).

The class thresholds (percentiles) come from an Estonian ecological area-fraction target
DERIVED from Ilyasov Table 2 (recorded below), NOT from Moore's break-dependent bands.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import ndtr

# --- Ilyasov 2026 Table 2 -> Estonian 3-class ecological area-fraction derivation ---------
# Ilyasov's finest composite classes (percent within unit): RH ridge-hummock, RD
# ridge-depression, HH hollow-hummock, HD hollow-depression. Map to the Moore 3-class
# hollow/lawn/hummock scheme by ecological wetness: hummock = RH (driest, raised),
# hollow = HD (wettest, lowest), lawn = the intermediate band RD + HH.
ILYASOV_RIDGE_HOLLOW_PATTERNED = {"RH": 21.0, "RD": 13.0, "HH": 16.0, "HD": 50.0}  # 205 ha unit
ILYASOV_RYAM_WITH_HOLLOWS = {"RH": 57.0, "RD": 20.0, "HH": 8.0, "HD": 15.0}  # 147 ha unit
# The selected pool-bearing core is a ridge-hollow-patterned bog (real ETAK Laugas pools =
# hollows) grading toward ryam; interpolate predominantly ridge-hollow-patterned with a
# light ryam grade. Recorded, not hand-tuned to a look.
RYAM_GRADE_WEIGHT = 0.25


def _three_class(comp: dict[str, float]) -> tuple[float, float, float]:
    hummock = comp["RH"]
    lawn = comp["RD"] + comp["HH"]
    hollow = comp["HD"]
    return hummock, lawn, hollow


def estonian_fractions() -> dict[str, float]:
    """Derived Estonian 3-class ecological area fractions (sum 1). Recorded in the prereg."""
    w = RYAM_GRADE_WEIGHT
    hp, lp, op = _three_class(ILYASOV_RIDGE_HOLLOW_PATTERNED)
    hr, lr, orr = _three_class(ILYASOV_RYAM_WITH_HOLLOWS)
    hummock = (1.0 - w) * hp + w * hr
    lawn = (1.0 - w) * lp + w * lr
    hollow = (1.0 - w) * op + w * orr
    total = hummock + lawn + hollow
    return {"hummock": hummock / total, "lawn": lawn / total, "hollow": hollow / total}


def class_percentile_bounds() -> tuple[float, float]:
    """Latent-percentile thresholds (p_lo, p_hi) splitting hollow | lawn | hummock."""
    f = estonian_fractions()
    p_lo = f["hollow"]
    p_hi = 1.0 - f["hummock"]
    return p_lo, p_hi


# --- Base anamorphosis shape (percentile -> relative height, metres) ----------------------
# Control points calibrated to Moore pooled per-class medians (hollow ~ -0.083, lawn ~ -0.003,
# hummock ~ +0.089) at the Estonian class boundaries, scaled so the overall p95-p5 ~ 0.30 m
# (Stordalen 0.317, Ilyasov 0.25-0.50 ridge relief). Monotone; tails reach the amplitude
# bound. The overall amplitude scalar is applied on top (calibrated to the local-relief gate).
_P = np.array([0.00, 0.05, 0.15, 0.30, 0.41, 0.50, 0.70, 0.82, 0.92, 0.95, 1.00])
_H = np.array([-0.240, -0.150, -0.100, -0.060, -0.040, -0.015,
               0.035, 0.090, 0.140, 0.150, 0.260])
_BASE = PchipInterpolator(_P, _H)


def base_height(percentile: np.ndarray) -> np.ndarray:
    """Monotone base relative height (m) for latent percentiles in [0,1]."""
    return _BASE(np.clip(percentile, 0.0, 1.0))


def anamorphosis(
    latent_std: np.ndarray,
    *,
    amplitude: float,
    ridge_field: np.ndarray | None = None,
    ridge_gain: float = 0.0,
) -> np.ndarray:
    """Continuous relative height from a standardized latent (mean 0, std 1).

    ``amplitude`` scales the whole transfer (calibrated to the measured local-relief gate).
    ``ridge_field`` (standardized) optionally extends the positive tail toward the ridge
    scale in ridge (high-coarse) zones; ``ridge_gain`` in [0, ~0.5].
    """
    percentile = ndtr(latent_std)
    height = amplitude * base_height(percentile)
    if ridge_field is not None and ridge_gain > 0.0:
        lift = ridge_gain * np.clip(ridge_field, 0.0, None) * np.clip(height, 0.0, None)
        height = height + lift
    return height
