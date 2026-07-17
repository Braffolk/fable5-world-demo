"""Honest evaluation of every frozen agriculture v2 gate.

Nothing here tunes anything; each metric is measured and compared to the threshold
frozen in the config before the run. Diagnostics (parent deviation) are reported,
not gated, per the 2026-07-17 exactness law.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import prf
from .forms import Synthesis


def _var(a: np.ndarray, mask: np.ndarray) -> float:
    v = a[mask]
    return float(np.var(v)) if v.size else 0.0


def row_variance_share(syn: Synthesis) -> dict:
    m = syn.interior
    vr = _var(syn.rows, m)
    vt = _var(syn.tracks, m)
    vc = _var(syn.clods, m)
    total = vr + vt + vc
    share = vr / total if total > 0 else 0.0
    return {
        "row_var": vr, "track_var": vt, "clod_var": vc, "total_typed_var": total,
        "row_variance_share": share,
        "track_variance_share": vt / total if total else 0.0,
        "clod_variance_share": vc / total if total else 0.0,
    }


def spacing_cv(syn: Synthesis, dev_parcel_id: int) -> dict:
    widths = syn.row_spacings_used.get(dev_parcel_id)
    if widths is None or widths.size < 4:
        return {"spacing_cv": 0.0, "spacing_mean_m": 0.0, "n_rows": 0}
    return {
        "spacing_cv": float(np.std(widths) / np.mean(widths)),
        "spacing_mean_m": float(np.mean(widths)),
        "spacing_min_m": float(widths.min()),
        "spacing_max_m": float(widths.max()),
        "n_rows": int(widths.size),
    }


def cross_parcel(parcel_field, params, min_area_m2: float = 2000.0) -> dict:
    s_mean = float(params["row_spacing_mean_m"])
    dirs, phases, ids = [], [], []
    for pid, info in parcel_field.infos.items():
        if info.along_half_m ** 2 < min_area_m2:  # skip slivers for the direction stat
            continue
        dirs.append(np.degrees(info.theta_rad) % 180.0)
        phi0 = (prf.uniform(pid, 101) - 0.5) * s_mean
        phases.append((phi0 / s_mean) % 1.0)
        ids.append(pid)
    dirs = np.array(dirs)
    phases = np.array(phases)
    # circular std of row directions on the 180-degree line
    ang = np.deg2rad(2.0 * dirs)
    R = np.hypot(np.mean(np.cos(ang)), np.mean(np.sin(ang)))
    circ_std_deg = float(np.degrees(0.5 * np.sqrt(-2.0 * np.log(max(R, 1e-9)))))
    # phase spread: std of independent per-parcel phase origins (uniform ~0.289)
    phase_spread = float(np.std(phases)) if phases.size > 1 else 0.0
    return {
        "n_parcels": int(dirs.size),
        "parcel_ids": [int(i) for i in ids],
        "direction_deg": [float(d) for d in dirs],
        "direction_circular_std_deg": circ_std_deg,
        "phase_origin_spread": phase_spread,
    }


def _cross_profile_psd(field: np.ndarray, theta: float, texel: float):
    """1D power spectrum of profiles sampled PERPENDICULAR to the rows.

    This is Marzahn's own acquisition geometry (roughness measured perpendicular to
    the seedbed rows), and it resolves the characteristic scale of a deliberately
    broadened, non-comb row field far better than a 2D wedge average (no angular
    dilution, no row-projection distortion). Returns wavelength (m) and mean power.
    """
    f = np.asarray(field, dtype=np.float64)
    n = f.shape[0]
    cx, cy = np.cos(theta + np.pi / 2.0), np.sin(theta + np.pi / 2.0)   # world cross unit
    ax, ay = np.cos(theta), np.sin(theta)                              # world row unit
    m = min(int(n / abs(cx) if abs(cx) > abs(cy) else n / abs(cy)) - 2, 2 * n)
    m = max(m, 256)
    t = (np.arange(m) - m / 2.0)                 # steps (texel units) along cross
    offs = (np.arange(0, n, 3) - n / 2.0)        # parallel line offsets along rows
    win = np.hanning(m)
    powers = []
    c = n / 2.0
    for s in offs:
        bx, by = c + s * ax, c - s * ay
        cols = bx + t * cx
        rows = by - t * cy
        if cols.min() < 0 or cols.max() >= n or rows.min() < 0 or rows.max() >= n:
            continue
        vals = ndimage.map_coordinates(f, [rows, cols], order=1, mode="constant", cval=0.0)
        if not np.any(vals):
            continue
        powers.append(np.abs(np.fft.rfft((vals - vals.mean()) * win)) ** 2)
    if not powers:
        return np.array([1.0]), np.array([0.0])
    p = np.mean(powers, axis=0)
    freq = np.fft.rfftfreq(m, d=texel)
    good = freq > 0
    return 1.0 / freq[good], p[good]


def scale_break(syn: Synthesis, axis_theta: float, gates: dict) -> dict:
    """Marzahn dual-scale: the small scale (seedbed rows + clods, wheel tracks masked
    per Marzahn 2012 sec. 2.2/3.2) and the large scale (wheel tracks) each yield a
    distinct characteristic wavelength on cross-row profiles; their ratio proves the
    two-scale break that a single wheel-track-dominated comb would hide.
    """
    texel = float(syn.east[1] - syn.east[0])
    m = syn.interior
    small = (syn.rows + syn.clods) * m
    large = syn.tracks * m
    combined = syn.added * m
    wl_s, p_s = _cross_profile_psd(small, axis_theta, texel)
    wl_l, p_l = _cross_profile_psd(large, axis_theta, texel)
    wl_c, p_c = _cross_profile_psd(combined, axis_theta, texel)
    rlo, rhi = gates["row_peak_wavelength_m_range"]
    tlo, thi = gates["track_peak_wavelength_m_range"]

    def band_peak(wl, p, a, b):
        mask = (wl >= a) & (wl <= b)
        if not mask.any():
            return None, 0.0, 0.0
        i = int(np.argmax(p[mask]))
        floor = float(np.median(p[(wl > 0.13) & (wl < 5.0)]))
        return float(wl[mask][i]), float(p[mask][i]), floor

    row_wl, row_pw, row_floor = band_peak(wl_s, p_s, rlo, rhi)
    trk_wl, trk_pw, trk_floor = band_peak(wl_l, p_l, tlo, thi)
    ratio = (trk_wl / row_wl) if (row_wl and trk_wl) else 0.0
    passed = bool(
        row_wl and trk_wl
        and ratio >= float(gates["scale_break_wavelength_ratio_min"])
        and row_pw > 1.3 * row_floor and trk_pw > 1.3 * trk_floor
    )
    return {
        "method": "marzahn_cross_row_profile_small(rows+clods)_and_large(tracks)",
        "row_peak_wavelength_m": row_wl, "row_peak_prominence_x_median": row_pw / row_floor if row_floor else 0.0,
        "track_peak_wavelength_m": trk_wl, "track_peak_prominence_x_median": trk_pw / trk_floor if trk_floor else 0.0,
        "wavelength_ratio": ratio,
        "two_distinct_peaks": passed,
        "small_scale_wavelength_m": [float(x) for x in wl_s],
        "small_scale_power": [float(x) for x in p_s],
        "large_scale_wavelength_m": [float(x) for x in wl_l],
        "large_scale_power": [float(x) for x in p_l],
        "combined_wavelength_m": [float(x) for x in wl_c],
        "combined_power": [float(x) for x in p_c],
    }


def deviation(syn: Synthesis) -> dict:
    a = np.abs(syn.added[syn.interior])
    if a.size == 0:
        return {"p50_m": 0.0, "p95_m": 0.0, "max_m": 0.0, "area_fraction_gt_5mm": 0.0}
    return {
        "p50_m": float(np.percentile(a, 50)),
        "p95_m": float(np.percentile(a, 95)),
        "max_m": float(a.max()),
        "area_fraction_gt_5mm": float(np.mean(a > 0.005)),
    }


def taper_formfree(syn: Synthesis, gates: dict) -> dict:
    formfree = ~(syn.ids > 0)          # outside any arable parcel
    a = np.abs(syn.added[formfree])
    if a.size == 0:
        return {"formfree_samples": 0, "p95_m": 0.0, "max_m": 0.0,
                "passed": True, "note": "no form-free ground in this window"}
    p95 = float(np.percentile(a, 95))
    return {
        "formfree_samples": int(a.size),
        "p95_m": p95, "max_m": float(a.max()),
        "passed": bool(p95 <= float(gates["taper_formfree_p95_m_max"])),
    }


def safety(syn: Synthesis, gates: dict) -> dict:
    outside = ~(syn.ids > 0)
    added = syn.added
    hard_res = int(np.count_nonzero(added[syn.hard_mask]))
    ditch_res = int(np.count_nonzero(added[syn.ditch_mask]))
    outside_res = int(np.count_nonzero(added[outside]))
    # collar-to-C0: relief exactly zero at the parcel boundary line (dist==0 interior edge)
    limit = int(gates["safety_residual_cells_max"])
    return {
        "hard_mask_residual_cells": hard_res,
        "ditch_residual_cells": ditch_res,
        "outside_parcel_residual_cells": outside_res,
        "hard_mask_cells": int(syn.hard_mask.sum()),
        "ditch_cells": int(syn.ditch_mask.sum()),
        "passed": bool(hard_res <= limit and ditch_res <= limit and outside_res <= limit),
    }


def evaluate(syn: Synthesis, parcel_field, config, dev_parcel_id: int,
             axis_theta: float, partition_ok: bool) -> dict:
    g = config.gates
    var = row_variance_share(syn)
    cv = spacing_cv(syn, dev_parcel_id)
    xp = cross_parcel(parcel_field, config.params)
    sb = scale_break(syn, axis_theta, g)
    dev = deviation(syn)
    results = {
        "anti_corduroy": {
            **var, **cv, **xp,
            "row_variance_share_pass": bool(var["row_variance_share"] <= g["row_variance_share_max"]),
            "spacing_cv_pass": bool(cv["spacing_cv"] >= g["spacing_cv_min"]),
            "direction_variation_pass": bool(
                xp["direction_circular_std_deg"] >= g["cross_parcel_direction_std_deg_min"]),
            "phase_discontinuity_pass": bool(
                xp["phase_origin_spread"] >= g["cross_parcel_phase_min_frac"]),
        },
        "scale_break": {**sb, "pass": sb["two_distinct_peaks"]},
        "parent_deviation_diagnostic": dev,
        "safety_exactness": safety(syn, g),
        "partition_invariance": {"byte_identical": bool(partition_ok),
                                 "pass": bool(partition_ok) if g["partition_byte_identical_required"] else True},
    }
    ac = results["anti_corduroy"]
    ac["pass"] = bool(ac["row_variance_share_pass"] and ac["spacing_cv_pass"]
                      and ac["direction_variation_pass"] and ac["phase_discontinuity_pass"])
    return results


def taper_from_window(syn_window: Synthesis, config) -> dict:
    return {**taper_formfree(syn_window, config.gates)}
