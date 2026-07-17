"""Typed cultivated forms for agriculture v2 (world-coordinate, partition-invariant).

rows    -- per-parcel smoothly-varying direction field with VARIABLE seedbed-row
           spacing (log-normal, world-PRF drawn), row-amplitude defects, dropouts,
           gentle curvature, and headland attenuation. Small (seedbed) scale.
tracks  -- wheel-track compaction lines along the same direction field, ~2 m spacing,
           sharply bounded 4-6 cm depressions with a small rim. Large scale.
clods   -- world-cell BLAKE2b marked point process; every clod is realized from a
           continuous-mark super-ellipse form family (no analytic stamp reuse).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import prf
from .geometry import ParcelField, ParcelInfo


def _compact_bump(x: np.ndarray) -> np.ndarray:
    return np.square(np.maximum(1.0 - np.square(x), 0.0))


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass
class Synthesis:
    east: np.ndarray
    north: np.ndarray
    base: np.ndarray
    rows: np.ndarray
    tracks: np.ndarray
    clods: np.ndarray
    added: np.ndarray
    height: np.ndarray
    ids: np.ndarray
    dist: np.ndarray
    track_mask: np.ndarray
    hard_mask: np.ndarray
    ditch_mask: np.ndarray
    interior: np.ndarray          # inside parcel, beyond collar, unmasked
    row_spacings_used: dict[int, np.ndarray]


def _state_scalings(east2d, north2d, mixture):
    """Per-sample operation-state form scalings from a world-PRF macro-cell field."""
    mcm = float(mixture["state_macro_cell_m"])
    weights = mixture["state_weights"]
    order = ["seedbed", "crusted_seedbed", "smooth_crusted"]
    cum = np.cumsum([weights[s] for s in order])
    scal = mixture["state_form_scalings"]
    mc_c = np.floor(east2d / mcm).astype(np.int64)
    mc_r = np.floor(north2d / mcm).astype(np.int64)
    pairs = np.stack([mc_c.ravel(), mc_r.ravel()], axis=1)
    uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
    u = np.array([prf.uniform(int(c), int(r), 9) for c, r in uniq])
    state_idx = np.searchsorted(cum, u, side="right").clip(0, 2)
    row_s = np.array([scal[order[i]]["row_amp"] for i in state_idx])[inv].reshape(east2d.shape)
    trk_s = np.array([scal[order[i]]["track_amp"] for i in state_idx])[inv].reshape(east2d.shape)
    den_s = np.array([scal[order[i]]["clod_density"] for i in state_idx])[inv].reshape(east2d.shape)
    siz_s = np.array([scal[order[i]]["clod_size"] for i in state_idx])[inv].reshape(east2d.shape)
    return row_s, trk_s, den_s, siz_s, state_idx[inv].reshape(east2d.shape)


def _parcel_coords(E, N, info: ParcelInfo, curvature: float):
    de = E - info.cx
    dn = N - info.cy
    ct, st = np.cos(info.theta_rad), np.sin(info.theta_rad)
    along = de * ct + dn * st
    cross = -de * st + dn * ct
    # gentle curvature: rows bow with distance along the parcel (bounded).
    cross = cross + curvature * np.square(along)
    return along, cross


def _row_boundaries(info: ParcelInfo, cross_min, cross_max, params):
    """World-anchored, variable-spacing seedbed-row boundary positions for a parcel."""
    s_mean = float(params["row_spacing_mean_m"])
    sigma = float(params["row_spacing_lognormal_sigma"])
    lo, hi = params["row_spacing_clamp_m"]
    phi0 = (prf.uniform(info.etak_id, 101) - 0.5) * s_mean
    reach = max(abs(cross_min - phi0), abs(cross_max - phi0)) + 4.0 * s_mean
    kmax = int(reach / float(lo)) + 3

    def spacing(k: int) -> float:
        z = prf.normal_from_two(*prf.streams((info.etak_id, k, 3), 2))
        return float(np.clip(s_mean * np.exp(sigma * z - 0.5 * sigma * sigma), lo, hi))

    def amp(k: int) -> float:
        s = prf.streams((info.etak_id, k, 7), 3)
        if s[0] < float(params["row_dropout_fraction"]):
            return 0.0
        cv = float(params["row_amp_cv"])
        return float(params["row_relief_m"]) * (1.0 + cv * (2.0 * s[1] - 1.0))

    ks = list(range(-kmax, kmax + 1))
    spac = {k: spacing(k) for k in ks}
    boundaries = np.empty(len(ks) + 1, dtype=np.float64)
    boundaries[kmax] = phi0
    acc = phi0
    for k in range(0, kmax + 1):
        boundaries[kmax + 1 + k] = acc + spac[k]
        acc += spac[k]
    acc = phi0
    for k in range(-1, -kmax - 1, -1):
        acc -= spac[k]
        boundaries[kmax + k] = acc
    widths = np.diff(boundaries)
    amps = np.array([amp(k) for k in ks], dtype=np.float64)
    return boundaries, widths, amps


def _rows_for_parcel(E, N, info, params, row_state):
    along, cross = _parcel_coords(E, N, info, float(params["row_curvature_per_m"]))
    boundaries, widths, amps = _row_boundaries(info, float(cross.min()), float(cross.max()), params)
    idx = np.clip(np.searchsorted(boundaries, cross, side="right") - 1, 0, widths.size - 1)
    u = (cross - boundaries[idx]) / widths[idx]
    ridge = np.cos(2.0 * np.pi * (u - 0.5))              # zero-mean ridge/furrow per row
    rows = amps[idx] * ridge * row_state
    # headland: attenuate seedbed rows near the boundary and overlay a turning band.
    return along, cross, rows, boundaries, widths


def _tracks_for_parcel(along, cross, params, track_state):
    s_mean = float(params["track_spacing_mean_m"])
    cv = float(params["track_spacing_cv"])
    lo = s_mean * (1.0 - 3.0 * cv)
    cmin, cmax = float(cross.min()), float(cross.max())
    reach = max(abs(cmin), abs(cmax)) + 4.0 * s_mean
    kmax = int(reach / max(lo, 0.1)) + 3
    acc = 0.0
    centers = [0.0]
    for k in range(1, kmax + 1):
        z = 2.0 * prf.streams((987654, k, 1), 1)[0] - 1.0
        acc += s_mean * (1.0 + cv * z)
        centers.append(acc)
    acc = 0.0
    for k in range(-1, -kmax - 1, -1):
        z = 2.0 * prf.streams((987654, k, 1), 1)[0] - 1.0
        acc -= s_mean * (1.0 + cv * z)
        centers.append(acc)
    centers = np.sort(np.array(centers))
    j = np.clip(np.searchsorted(centers, cross), 1, centers.size - 1)
    d = np.minimum(np.abs(cross - centers[j - 1]), np.abs(cross - centers[j]))
    half = 0.5 * float(params["track_width_m"])
    rut = _compact_bump(d / half)
    rim = _compact_bump((np.abs(d - 1.25 * half)) / (0.35 * half))
    depth = float(params["track_depth_m"])
    tracks = (-depth * rut + float(params["track_rim_fraction"]) * depth * rim) * track_state
    return tracks, rut


def _clods(east, north, params, den_scale2d, siz_scale2d, texture_mult, mixture):
    cell = float(params["clod_cell_m"])
    dmin, dmax = params["clod_diameter_m"]
    hmin, hmax = params["clod_height_m"]
    elo, ehi = params["clod_elongation_range"]
    xlo, xhi = params["clod_shape_exp_range"]
    base_occ = float(params["clod_base_occupancy"])
    mcm = float(mixture["state_macro_cell_m"])
    order = ["seedbed", "crusted_seedbed", "smooth_crusted"]
    cum = np.cumsum([mixture["state_weights"][s] for s in order])
    scal = mixture["state_form_scalings"]

    result = np.zeros((north.size, east.size), dtype=np.float64)
    e_lo, e_hi = float(east[0]), float(east[-1])
    n_lo, n_hi = float(north[-1]), float(north[0])
    r_pad = 0.5 * dmax * max(ehi, 1.0) + cell
    ic0 = int(np.floor((e_lo - r_pad) / cell))
    ic1 = int(np.floor((e_hi + r_pad) / cell))
    jc0 = int(np.floor((n_lo - r_pad) / cell))
    jc1 = int(np.floor((n_hi + r_pad) / cell))
    north_asc = north[::-1]
    n_present = 0
    diams: list[float] = []
    heights: list[float] = []

    for jc in range(jc0, jc1 + 1):
        for ic in range(ic0, ic1 + 1):
            s = prf.streams((ic, jc, 0), 8)
            cx = (ic + 0.5 + (s[1] - 0.5) * 0.7) * cell
            cy = (jc + 0.5 + (s[2] - 0.5) * 0.7) * cell
            # operation state at this clod's macro-cell drives density + size
            u_state = prf.uniform(int(np.floor(cx / mcm)), int(np.floor(cy / mcm)), 9)
            st_i = int(np.searchsorted(cum, u_state, side="right"))
            st_i = min(st_i, 2)
            occ = base_occ * scal[order[st_i]]["clod_density"]
            if s[0] >= occ:
                continue
            diameter = (dmin + (dmax - dmin) * s[3]) * scal[order[st_i]]["clod_size"] * texture_mult
            height = hmin + (hmax - hmin) * s[4]
            elong = elo + (ehi - elo) * s[5]
            orient = 2.0 * np.pi * s[6]
            shape_exp = xlo + (xhi - xlo) * s[7]
            half_major = 0.5 * diameter
            half_minor = 0.5 * diameter * elong
            col0 = max(0, int(np.searchsorted(east, cx - half_major) - 1))
            col1 = min(east.size, int(np.searchsorted(east, cx + half_major) + 1))
            a0 = max(0, int(np.searchsorted(north_asc, cy - half_major) - 1))
            a1 = min(north.size, int(np.searchsorted(north_asc, cy + half_major) + 1))
            row0, row1 = north.size - a1, north.size - a0
            if row0 >= row1 or col0 >= col1:
                continue
            de = east[col0:col1][None, :] - cx
            dn = north[row0:row1][:, None] - cy
            ct, st = np.cos(orient), np.sin(orient)
            lx = (de * ct + dn * st) / half_major
            ly = (-de * st + dn * ct) / half_minor
            radius = np.power(np.power(np.abs(lx), shape_exp) + np.power(np.abs(ly), shape_exp),
                              1.0 / shape_exp)
            cap = height * _compact_bump(radius)
            view = result[row0:row1, col0:col1]
            np.maximum(view, cap, out=view)
            n_present += 1
            diams.append(diameter)
            heights.append(height)
    stats = {
        "clods_placed": n_present,
        "diameter_m_range": [min(diams), max(diams)] if diams else [0.0, 0.0],
        "height_m_range": [min(heights), max(heights)] if heights else [0.0, 0.0],
    }
    return result, stats


def synthesize(east, north, base, parcel_field: ParcelField, config,
               hard_mask, ditch_mask) -> tuple[Synthesis, dict]:
    params = config.params
    mixture = config.mixture
    texture_mult = float(config.raw["soil_texture"]["texture_clod_size_multiplier"])
    E2d = np.broadcast_to(east[None, :], (north.size, east.size))
    N2d = np.broadcast_to(north[:, None], (north.size, east.size))
    ids, dist = parcel_field.sample(E2d, N2d)
    row_s, trk_s, den_s, siz_s, _state = _state_scalings(E2d, N2d, mixture)

    rows = np.zeros(E2d.shape, dtype=np.float64)
    tracks = np.zeros(E2d.shape, dtype=np.float64)
    rut_full = np.zeros(E2d.shape, dtype=np.float64)
    spacings_used: dict[int, np.ndarray] = {}
    headland_band = float(params["headland_band_m"])
    headland_scale = float(params["headland_row_amp_scale"])

    for pid in sorted(int(p) for p in np.unique(ids) if p > 0):
        info = parcel_field.infos[pid]
        m = ids == pid
        along, cross, rows_p, boundaries, widths = _rows_for_parcel(
            E2d, N2d, info, params, row_s)
        # headland attenuation near the parcel boundary (turn zone)
        head = np.where(dist < headland_band,
                        headland_scale + (1.0 - headland_scale) * (dist / headland_band),
                        1.0)
        rows_p = rows_p * head
        tracks_p, rut_p = _tracks_for_parcel(along, cross, params, trk_s)
        rows[m] = rows_p[m]
        tracks[m] = tracks_p[m]
        rut_full[m] = rut_p[m]
        spacings_used[pid] = widths

    clods, clod_stats = _clods(east, north, params, den_s, siz_s, texture_mult, mixture)
    clods *= (1.0 - float(params["clod_track_suppression"]) * rut_full)

    inside = ids > 0
    collar = _smoothstep(dist / float(params["boundary_collar_m"]))
    gate = inside.astype(np.float64) * collar
    gate[hard_mask | ditch_mask] = 0.0

    rows *= gate
    tracks *= gate
    clods *= gate
    added = rows + tracks + clods
    height = base + added

    interior = inside & (~hard_mask) & (~ditch_mask) & (dist > float(params["boundary_collar_m"]))
    syn = Synthesis(east, north, base, rows, tracks, clods, added, height, ids, dist,
                    (rut_full > 0.3) & inside, hard_mask, ditch_mask, interior, spacings_used)
    return syn, clod_stats
