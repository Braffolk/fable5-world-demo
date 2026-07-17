"""Honest gate evaluation for the bog v5 hummock field.

Thresholds are frozen in the preregistration BEFORE the field is measured; this module
only measures and compares. Every gate reports (value, threshold, passed).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def _pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q))


def _band(v: float, lo: float, hi: float) -> bool:
    return lo <= v <= hi


def evaluate(
    relief: np.ndarray,
    authority: np.ndarray,
    open_water: np.ndarray,
    pool_dist: np.ndarray,
    pitch: float,
    thresholds: dict,
) -> dict:
    ra = relief[authority]
    g: dict[str, dict] = {}

    def gate(name: str, value: float, passed: bool, thr) -> None:
        g[name] = {"value": round(float(value), 6), "threshold": thr, "passed": bool(passed)}

    # ---- safety (exact) ----
    pools_maxabs = float(np.max(np.abs(relief[open_water]))) if open_water.any() else 0.0
    nonauth_maxabs = float(np.max(np.abs(relief[~authority])))
    gate("safety_pools_relief_free_maxabs_m", pools_maxabs, pools_maxabs == 0.0, "== 0")
    gate("safety_nonauth_relief_free_maxabs_m", nonauth_maxabs, nonauth_maxabs == 0.0, "== 0")

    # ---- amplitude realistic ----
    p95 = _pct(ra, 95)
    p99abs = _pct(np.abs(ra), 99)
    p50 = _pct(ra, 50)
    rmax = float(ra.max())
    hollow_frac = float((ra < -0.02).mean())
    hummock_frac = float((ra > 0.15).mean())
    t = thresholds
    gate("amp_p95_m", p95, _band(p95, *t["amp_p95_band_m"]), t["amp_p95_band_m"])
    gate("amp_p99abs_m", p99abs, p99abs <= t["amp_p99abs_max_m"], t["amp_p99abs_max_m"])
    gate("amp_max_m", rmax, rmax <= t["amp_max_max_m"], t["amp_max_max_m"])
    gate("amp_p50_m", p50, _band(p50, *t["amp_p50_band_m"]), t["amp_p50_band_m"])
    gate("amp_hollow_fraction", hollow_frac, hollow_frac >= t["hollow_fraction_min"], t["hollow_fraction_min"])
    gate("amp_hummock_fraction", hummock_frac, hummock_frac >= t["hummock_fraction_min"], t["hummock_fraction_min"])

    # ---- organic / anti-artifact ----
    # Structure-tensor orientation resultant (aligned strings -> R near 1; bumps -> ~0).
    gy, gx = np.gradient(relief.astype(np.float64), pitch)
    jxx = ndimage.gaussian_filter(gx * gx, 2.0)
    jyy = ndimage.gaussian_filter(gy * gy, 2.0)
    jxy = ndimage.gaussian_filter(gx * gy, 2.0)
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    m = authority & (coh > 0)
    wsum = float(coh[m].sum())
    resultant = abs(complex(
        float((coh[m] * np.cos(2 * theta[m])).sum()) / wsum,
        float((coh[m] * np.sin(2 * theta[m])).sum()) / wsum,
    ))
    gate("orient_resultant_R", resultant, resultant <= t["orient_resultant_max"], t["orient_resultant_max"])

    # Radially-averaged power spectrum: no single sharp wavelength (Turing) peak.
    field = np.where(authority, relief - float(ra.mean()), 0.0)
    win = np.hanning(field.shape[0])[:, None] * np.hanning(field.shape[1])[None, :]
    F = np.fft.fftshift(np.fft.fft2(field * win))
    power = np.abs(F) ** 2
    H, W = field.shape
    fy = np.fft.fftshift(np.fft.fftfreq(H, d=pitch))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(W, d=pitch))[None, :]
    fr = np.sqrt(fx ** 2 + fy ** 2)
    lo_f, hi_f = 1.0 / t["spectrum_band_m"][1], 1.0 / t["spectrum_band_m"][0]
    nb = 48
    edges = np.linspace(lo_f, hi_f, nb + 1)
    bin_p = np.zeros(nb)
    bin_f = 0.5 * (edges[:-1] + edges[1:])
    for k in range(nb):
        sel = (fr >= edges[k]) & (fr < edges[k + 1])
        if sel.any():
            bin_p[k] = float(power[sel].mean())
    total = bin_p.sum()
    peak_frac = float(bin_p.max() / total) if total > 0 else 1.0
    gate("spectrum_peak_fraction", peak_frac, peak_frac <= t["spectrum_peak_fraction_max"], t["spectrum_peak_fraction_max"])
    # Excess peak prominence over the smoothed red-spectrum trend: a reaction-diffusion
    # (Turing) single-wavelength lattice shows a sharp bin rising many-fold above trend;
    # a broadband organic field does not. This replaces the invalid "dominant wavelength =
    # argmax bin" metric (argmax of a broad red-spectrum plateau is meaningless; Moore's
    # 2.4-2.9 m is a PSD-slope breakpoint, not the peak-energy location -- see preregistration
    # premise_audit_correction).
    log_p = np.log(bin_p + 1e-30)
    trend = ndimage.uniform_filter1d(log_p, 7, mode="nearest")
    prominence = float(np.exp(np.max(log_p - trend)))
    gate("spectrum_excess_peak_prominence_ratio", prominence,
         prominence <= t["spectrum_excess_peak_prominence_max"], t["spectrum_excess_peak_prominence_max"])
    # Power-law fall-off: Moore Hurst 0.14-0.26 => slope = -2(H+1) ~ -2.28..-2.52. A flat
    # (white) or peaked spectrum would fall outside this red-spectrum band.
    slope = float(np.polyfit(np.log(bin_f), log_p, 1)[0])
    gate("spectrum_loglog_slope", slope, _band(slope, *t["spectrum_loglog_slope_band"]), t["spectrum_loglog_slope_band"])

    # Hummock size + spacing morphometry (vs Moore 2019 envelope).
    lbl, ncomp = ndimage.label(relief > 0.12)
    diams = []
    if ncomp:
        areas = ndimage.sum(np.ones_like(lbl), lbl, index=np.arange(1, ncomp + 1)) * pitch * pitch
        diams = [2.0 * np.sqrt(a / np.pi) for a in areas if a >= 0.2]
    diam_med = float(np.median(diams)) if diams else 0.0
    gate("morph_hummock_diameter_median_m", diam_med, _band(diam_med, *t["hummock_diameter_band_m"]), t["hummock_diameter_band_m"])

    mx = ndimage.maximum_filter(relief, size=int(round(1.0 / pitch)) | 1)
    peaks = np.argwhere((relief == mx) & (relief > 0.12) & authority)
    spacing_med = 0.0
    if len(peaks) > 8:
        pm = peaks.astype(np.float64) * pitch
        d = np.sqrt(((pm[:, None, :] - pm[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        spacing_med = float(np.median(d.min(1)))
    gate("morph_peak_spacing_median_m", spacing_med, _band(spacing_med, *t["peak_spacing_band_m"]), t["peak_spacing_band_m"])

    # ---- pool coupling to REAL pools ----
    npool = int(open_water.sum())
    gate("pool_present_cells", npool, npool >= t["pool_present_cells_min"], t["pool_present_cells_min"])
    margin = authority & (pool_dist <= 2.0)
    margin_low_frac = float((relief[margin] < 0.05).mean()) if margin.any() else 1.0
    gate("pool_margin_low_fraction", margin_low_frac, margin_low_frac >= t["pool_margin_low_fraction_min"], t["pool_margin_low_fraction_min"])
    near = authority & (pool_dist <= 1.0)
    near_mean = float(relief[near].mean()) if near.any() else 0.0
    gate("pool_near_mean_relief_m", near_mean, near_mean <= t["pool_near_mean_relief_max_m"], t["pool_near_mean_relief_max_m"])

    diagnostics = {
        "authority_mean_relief_m": float(ra.mean()),
        "authority_std_relief_m": float(ra.std()),
        "signed_p01_m": _pct(ra, 1),
        "signed_p50_m": p50,
        "signed_p99_m": _pct(ra, 99),
        "hummock_component_count": int(ncomp),
        "detected_peak_count": int(len(peaks)),
        "core_authority_cells": int(authority.sum()),
        "core_pool_cells": npool,
    }
    passed = all(v["passed"] for v in g.values())
    return {"passed": passed, "gates": g, "diagnostics": diagnostics}
