"""Honest gate measurement on the OUTPUT relief, matching the Moore dossier estimators.

Microform gates (transiogram, area fractions, PSD slope + dominant wavelength, autocorr,
semivariogram, local relief) are measured on the DETRENDED height (subtract a sigma=0.5 m
Gaussian, exactly Moore's method) so they are apples-to-apples with the conditioning dossier.
Two-scale and coarse-patterning gates are measured on the RAW height. Every number reported.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.signal import fftconvolve

HOLLOW, LAWN, HUMMOCK = 0, 1, 2
BREAK_M = 0.06  # Moore relative-elevation dead-band
REF_SIGMA_M = 0.5  # Moore detrend reference scale


def nan_gaussian(z: np.ndarray, sigma_px: float) -> np.ndarray:
    m = np.isfinite(z).astype(float)
    zz = np.where(np.isfinite(z), z, 0.0)
    num = gaussian_filter(zz, sigma_px, mode="nearest")
    den = gaussian_filter(m, sigma_px, mode="nearest")
    den = np.where(den < 1e-6, np.nan, den)
    return num / den


def detrend(height: np.ndarray, mask: np.ndarray, pitch: float) -> np.ndarray:
    """Moore detrend: z - Gaussian(sigma=0.5 m) reference, over the authority mask only."""
    z = np.where(mask, height, np.nan)
    ref = nan_gaussian(z, REF_SIGMA_M / pitch)
    return z - ref


def classify(z_rel: np.ndarray) -> np.ndarray:
    lab = np.full(z_rel.shape, -1, np.int8)
    v = np.isfinite(z_rel)
    lab[v & (z_rel < -BREAK_M)] = HOLLOW
    lab[v & (np.abs(z_rel) <= BREAK_M)] = LAWN
    lab[v & (z_rel > BREAK_M)] = HUMMOCK
    return lab


def area_fractions(lab: np.ndarray) -> dict[str, float]:
    v = lab >= 0
    n = int(v.sum())
    return {
        "hollow": float((lab == HOLLOW).sum() / n),
        "lawn": float((lab == LAWN).sum() / n),
        "hummock": float((lab == HUMMOCK).sum() / n),
    }


def lag1_transition(lab: np.ndarray) -> np.ndarray:
    """4-neighbour lag-1 transition-probability matrix (rows from 0/1/2)."""
    T = np.zeros((3, 3))
    for ax in (0, 1):
        a = lab
        b = np.roll(lab, -1, axis=ax)
        v = (a >= 0) & (b >= 0)
        if ax == 0:
            v[-1, :] = False
        else:
            v[:, -1] = False
        for ci in range(3):
            for cj in range(3):
                T[ci, cj] += np.sum(v & (a == ci) & (b == cj))
    rowsum = T.sum(1, keepdims=True)
    return np.divide(T, np.maximum(rowsum, 1))


def _radial_bin(field2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = field2d.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    r = np.hypot(xx - cx, yy - cy).astype(int)
    tot = np.bincount(r.ravel(), field2d.ravel())
    cnt = np.bincount(r.ravel())
    return tot / np.maximum(cnt, 1), cnt


def radial_psd(z_rel: np.ndarray, pitch: float,
               band_m: tuple[float, float] = (0.5, 4.0)) -> dict[str, float]:
    """Dominant wavelength (m) + log-log radial PSD slope over the MICROFORM band, Hann-windowed.

    The slope and dominant spacing are measured over the resolvable microform wavelength band
    ``band_m`` (default 0.5-4 m). Fitting outside it is meaningless here: below the low
    ``band_m`` bound is the sigma=0.5 m detrend high-pass corner (spectrum rises, not the
    microform roll-off), and 0.25 m sampling makes the 0.5 m wavelength the Nyquist limit
    (Moore's 1 cm plots resolved the sub-0.5 m roll-off; a 0.25 m surface cannot). Restricting
    to the microform band reproduces Moore's estimator INTENT on a coarser-sampled surface.
    """
    m = np.isfinite(z_rel)
    z = np.where(m, z_rel, 0.0)
    z = z - z[m].mean()
    z = np.where(m, z, 0.0)
    hn, wn = z.shape
    wy = np.hanning(hn)[:, None]
    wx = np.hanning(wn)[None, :]
    zf = np.fft.fftshift(np.fft.fft2(z * wy * wx))
    psd = np.abs(zf) ** 2
    rad, _ = _radial_bin(psd)
    kmax = min(len(rad), min(hn // 2, wn // 2))
    rad = rad[:kmax]
    freqs = np.arange(kmax) / (max(hn, wn) * pitch)  # cycles/m
    lam_all = np.divide(1.0, freqs, out=np.full(kmax, np.inf), where=freqs > 0)
    band = (lam_all >= band_m[0]) & (lam_all <= band_m[1])
    idx = np.nonzero(band)[0]
    if idx.size < 5:
        return {"dominant_wavelength_m": None, "psd_loglog_slope": None}
    k_peak = idx[int(np.argmax(rad[idx]))]
    lam = float(1.0 / freqs[k_peak])
    with np.errstate(divide="ignore"):
        lf = np.log10(freqs[idx])
        lp = np.log10(rad[idx])
    good = np.isfinite(lf) & np.isfinite(lp)
    slope = float(np.polyfit(lf[good], lp[good], 1)[0]) if good.sum() > 4 else None
    return {"dominant_wavelength_m": lam, "psd_loglog_slope": slope}


def autocorr_scale(z_rel: np.ndarray, pitch: float) -> dict[str, float]:
    """Radial autocorrelation first-zero lag and integral length (m), Moore method."""
    m = np.isfinite(z_rel).astype(float)
    z = np.where(np.isfinite(z_rel), z_rel, 0.0)
    z = z - (z * m).sum() / m.sum()
    z = z * m
    ac = fftconvolve(z, z[::-1, ::-1], mode="full")
    norm = fftconvolve(m, m[::-1, ::-1], mode="full")
    norm = np.where(norm < 1e-6, np.nan, norm)
    ac = ac / norm
    cy, cx = np.array(ac.shape) // 2
    ac = ac / ac[cy, cx]
    h, w = ac.shape
    yy, xx = np.indices((h, w))
    r = np.hypot(xx - cx, yy - cy).astype(int)
    prof = np.bincount(r.ravel(), np.nan_to_num(ac).ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    fz = None
    for i in range(1, min(len(prof), 400)):
        if prof[i] <= 0:
            fz = i
            break
    if fz:
        return {"first_zero_lag_m": fz * pitch, "integral_length_m": float(np.sum(prof[1:fz])) * pitch}
    return {"first_zero_lag_m": None, "integral_length_m": None}


def coarse_autocorr_length(height: np.ndarray, mask: np.ndarray, pitch: float) -> float:
    """Decorrelation length (m) of the coarse component (height minus its microform detail).

    Isolates the >~5 m patterning by smoothing away microform, then measuring the radial
    autocorrelation half-height length. Used to prove the second (10-100 m) scale is present.
    """
    coarse = nan_gaussian(np.where(mask, height, np.nan), 5.0 / pitch)
    coarse = np.where(mask, coarse, np.nan)
    m = np.isfinite(coarse).astype(float)
    z = np.where(np.isfinite(coarse), coarse, 0.0)
    z = z - (z * m).sum() / m.sum()
    z = z * m
    ac = fftconvolve(z, z[::-1, ::-1], mode="full")
    norm = fftconvolve(m, m[::-1, ::-1], mode="full")
    norm = np.where(norm < 1e-6, np.nan, norm)
    ac = ac / norm
    cy, cx = np.array(ac.shape) // 2
    ac = ac / ac[cy, cx]
    h, w = ac.shape
    yy, xx = np.indices((h, w))
    r = np.hypot(xx - cx, yy - cy).astype(int)
    prof = np.bincount(r.ravel(), np.nan_to_num(ac).ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    half = None
    for i in range(1, min(len(prof), 400)):
        if prof[i] <= 0.5:
            half = i
            break
    return float(half * pitch) if half else float(min(len(prof), 400) * pitch)


def local_relief(height: np.ndarray, mask: np.ndarray, pitch: float, window_m: float) -> dict[str, float]:
    """Local hummock-hollow amplitude: (local max - local min) over a microform window."""
    from scipy.ndimage import maximum_filter, minimum_filter

    size = max(3, int(round(window_m / pitch)))
    hi = maximum_filter(height, size=size, mode="nearest")
    lo = minimum_filter(height, size=size, mode="nearest")
    amp = (hi - lo)[mask]
    return {
        "p50_m": float(np.percentile(amp, 50)),
        "p95_m": float(np.percentile(amp, 95)),
        "window_m": window_m,
    }


def pool_coupling(height: np.ndarray, open_water: np.ndarray, authority: np.ndarray,
                  pitch: float, radius_m: float) -> dict[str, float]:
    """Fraction of pool-margin cells that are low (hollow-side), and hummock-over-pool count."""
    if not open_water.any():
        return {"pool_present": False, "margin_low_fraction": None, "hummock_over_pool_cells": 0}
    dist = distance_transform_edt(~open_water) * pitch
    margin = (dist > 0) & (dist <= radius_m) & authority
    # A margin cell is "low" if its height is below the authority median (hollow-side).
    med = float(np.median(height[authority]))
    low_frac = float((height[margin] < med).mean()) if margin.any() else None
    hummock_over_pool = int((height[open_water] > med + 0.06).sum())
    return {
        "pool_present": True,
        "margin_low_fraction": low_frac,
        "margin_cells": int(margin.sum()),
        "hummock_over_pool_cells": hummock_over_pool,
    }


def periodicity_peak(z_rel: np.ndarray) -> float:
    """Ratio of the strongest non-DC radial PSD peak to its local background (periodicity).

    A clean red spectrum gives ~1; a lattice/single-wavelength stamp gives a sharp spike >> 1.
    """
    m = np.isfinite(z_rel)
    z = np.where(m, z_rel, 0.0)
    z = z - z[m].mean()
    z = np.where(m, z, 0.0)
    hn, wn = z.shape
    wy = np.hanning(hn)[:, None]
    wx = np.hanning(wn)[None, :]
    psd = np.abs(np.fft.fftshift(np.fft.fft2(z * wy * wx))) ** 2
    rad, _ = _radial_bin(psd)
    kmax = min(len(rad), min(hn // 2, wn // 2))
    rad = rad[3:kmax]
    if rad.size < 12:
        return 1.0
    # smooth background via median filter; peak-to-background ratio
    from scipy.ndimage import median_filter

    bg = median_filter(rad, size=9, mode="nearest")
    ratio = rad / np.maximum(bg, 1e-30)
    return float(np.max(ratio))
