"""Bog v5 organic hummock-hollow relief field.

A marked-point process of rounded, irregular, coalescing Sphagnum hummocks with wet
hollows between and pool margins grading into the REAL ETAK Laugas pools. This is a
deliberate mechanism BREAK from the rejected v3/v4 reaction-diffusion string generator:
there are NO thin ridge lines, NO periodic lattice, NO single characteristic wavelength.

Morphometry is grounded in Moore et al. 2019 (Biogeosciences 16, 3491-3506):
hummock-hollow at 1-10 m spatial scale, dominant scale (PSD roll-off) 2.4-2.9 m,
hummock surface ~0.20 m above the water table, hummock-hollow modal separation
mu ~0.19-0.21 m, tall-hummock modes up to 0.36-0.53 m, site-level relief sigma ~0.147 m,
hummock footprint up to a few m^2, hummock flank slopes ~16-20 deg, and a sub-metre
fractal roughness (Hurst 0.14-0.26; ~95% of variance at scales > 0.6 m).

Every stochastic mark is drawn from the world-locked BLAKE2b PRF (laas-micro-prf1),
keyed by integer world coordinates so placement is identical regardless of window,
chunking, or worker count. Deterministic float64 throughout. Amplitude is form-attributed
and mean-centred over the mire authority; relief is exactly 0 on pools/hard/non-mire.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ..peat_bog_network.prf import world_uniform


# ----------------------------------------------------------------------------- helpers
def _smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((np.asarray(x, dtype=np.float64) - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _value_noise(
    e0: int, n1: int, height: int, width: int, pitch: float, cell_m: int, salt: int
) -> np.ndarray:
    """Bilinearly-upsampled world-locked value noise at a `cell_m` metre scale.

    Node values are the BLAKE2b PRF keyed by integer world node coordinates (metres)
    offset by ``salt``; independent of the crop window (nodes are world-anchored).
    """
    # World node grid (metres) covering the raster with a one-node apron on each side.
    node_e0 = int(np.floor((e0) / cell_m) * cell_m) - cell_m
    node_n1 = int(np.ceil((n1) / cell_m) * cell_m) + cell_m
    nx = int((e0 + width * pitch - node_e0) / cell_m) + 3
    ny = int((node_n1 - (n1 - height * pitch)) / cell_m) + 3
    cols = node_e0 + np.arange(nx, dtype=np.int64) * cell_m
    rows = node_n1 - np.arange(ny, dtype=np.int64) * cell_m
    east = np.broadcast_to(cols[None, :] + salt, (ny, nx))
    north = np.broadcast_to(rows[:, None] + salt, (ny, nx))
    nodes = world_uniform(east, north)
    # Sample position of each raster cell centre in node space.
    cell_e = e0 + (np.arange(width, dtype=np.float64) + 0.5) * pitch
    cell_n = n1 - (np.arange(height, dtype=np.float64) + 0.5) * pitch
    fx = (cell_e - node_e0) / cell_m
    fy = (node_n1 - cell_n) / cell_m
    x0 = np.floor(fx).astype(np.int64)
    y0 = np.floor(fy).astype(np.int64)
    tx = (fx - x0)[None, :]
    ty = (fy - y0)[:, None]
    x0 = np.clip(x0, 0, nx - 2)
    y0 = np.clip(y0, 0, ny - 2)
    n00 = nodes[np.ix_(y0, x0)]
    n01 = nodes[np.ix_(y0, x0 + 1)]
    n10 = nodes[np.ix_(y0 + 1, x0)]
    n11 = nodes[np.ix_(y0 + 1, x0 + 1)]
    top = n00 * (1 - tx) + n01 * tx
    bot = n10 * (1 - tx) + n11 * tx
    return top * (1 - ty) + bot * ty


@dataclass(frozen=True)
class HummockPass:
    spacing_m: int          # integer world grid spacing for candidate sites
    accept: float           # base acceptance probability on fully dry ground
    h_lo: float             # hummock height range (m, before wetness gain)
    h_hi: float
    radius_ratio: float     # mean radius / height (sets flank slope ~atan(1/ratio))
    salt: int               # distinct PRF stream salt for this pass


# Three superposed populations at different scales kill any single wavelength and give
# a continuous, coalescing bumpy quilt (small hummocks infill, large complexes coalesce).
PASSES = (
    HummockPass(spacing_m=1, accept=0.85, h_lo=0.14, h_hi=0.28, radius_ratio=3.0, salt=101),
    HummockPass(spacing_m=2, accept=0.90, h_lo=0.26, h_hi=0.42, radius_ratio=2.8, salt=211),
    HummockPass(spacing_m=4, accept=0.80, h_lo=0.34, h_hi=0.52, radius_ratio=2.6, salt=331),
)

# Soft coalescence ceiling: overlapping domes merge into plateaued complexes instead of
# summing into unrealistic spikes (real hummock complexes top out ~0.4-0.5 m).
COALESCE_CAP_M = 0.55


@dataclass
class HummockField:
    relief_core: np.ndarray       # (512,512) float64 0.25 m, added to the base
    relief_halo: np.ndarray       # (1024,1024) float64, diagnostic
    authority_core: np.ndarray    # (512,512) bool
    open_water_core: np.ndarray   # (512,512) bool
    wetness_halo: np.ndarray      # (1024,1024) dryness proxy in [0,1]
    pool_dist_core: np.ndarray    # (512,512) metres to nearest open water
    hummock_count: int
    site_count: int
    meta: dict


def synthesize(
    *,
    halo_bbox: tuple[int, int, int, int],
    core_slice: tuple[slice, slice],
    pitch: float,
    authority_halo: np.ndarray,
    open_water_halo: np.ndarray,
    slope_core_p95: float,
) -> HummockField:
    e0, n0, e1, n1 = halo_bbox
    H = int(round((n1 - n0) / pitch))
    W = int(round((e1 - e0) / pitch))
    if authority_halo.shape != (H, W) or open_water_halo.shape != (H, W):
        raise ValueError("mask shape does not match halo grid")

    # ------------------------------------------------------------------ wetness proxy
    # Distance (m) from the nearest open water; pools are the wettest sinks.
    pool_dist = ndimage.distance_transform_edt(~open_water_halo) * pitch
    wp = _smoothstep(pool_dist, 0.5, 9.0)          # 0 at pool edge -> 1 far from pools
    # Large-scale dryness mosaic (~12 m) so drier/wetter patches exist independent of
    # pools; world-locked value noise, not decorative (drives the hummock/hollow mosaic).
    dryness = _value_noise(e0, n1, H, W, pitch, cell_m=12, salt=90001)
    wetness = np.clip(wp * (0.50 + 0.55 * dryness), 0.0, 1.0)

    # ------------------------------------------------------------------ hummock stamps
    relief = np.zeros((H, W), dtype=np.float64)
    max_support_m = 2.6                             # >= largest dome radius
    site_count = 0
    hummock_count = 0
    for pss in PASSES:
        s = pss.spacing_m
        margin = int(np.ceil(max_support_m))
        ce0 = int(np.floor((e0 - margin) / s) * s)
        cn1 = int(np.ceil((n1 + margin) / s) * s)
        ncx = int((e1 + margin - ce0) / s) + 1
        ncy = int((cn1 - (n0 - margin)) / s) + 1
        gx = ce0 + np.arange(ncx, dtype=np.int64) * s
        gy = cn1 - np.arange(ncy, dtype=np.int64) * s
        cell_e = np.broadcast_to(gx[None, :], (ncy, ncx))
        cell_n = np.broadcast_to(gy[:, None], (ncy, ncx))

        def stream(chan: int) -> np.ndarray:
            return world_uniform(cell_e + pss.salt + chan * 1_000_003, cell_n - chan * 7_919)

        u_jx, u_jy = stream(0), stream(1)
        u_acc = stream(2)
        u_h, u_r, u_asp, u_th, u_wrp = stream(3), stream(4), stream(5), stream(6), stream(7)

        # Continuous jittered world positions (strong jitter -> no lattice signature).
        pos_e = cell_e + (u_jx - 0.5) * s * 0.95
        pos_n = cell_n + (u_jy - 0.5) * s * 0.95
        # Cell-centre index into the halo raster to read the local wetness/authority.
        ci = np.clip(((n1 - pos_n) / pitch).astype(np.int64), 0, H - 1)
        cj = np.clip(((pos_e - e0) / pitch).astype(np.int64), 0, W - 1)
        w_site = wetness[ci, cj]
        on_auth = authority_halo[ci, cj]
        # Acceptance: drier sites host somewhat more (and taller) hummocks; the pool gate
        # (wp, folded into wetness) removes them over the wet margins entirely.
        accept = pss.accept * (0.55 + 0.45 * w_site)
        keep = on_auth & (u_acc < accept) & (pos_e > e0 - margin) & (pos_e < e1 + margin)
        keep &= (pos_n > n0 - margin) & (pos_n < n1 + margin)

        idx = np.nonzero(keep.ravel())[0]
        if idx.size == 0:
            continue
        pe = pos_e.ravel()[idx]
        pn = pos_n.ravel()[idx]
        wv = w_site.ravel()[idx]
        hh = (pss.h_lo + (pss.h_hi - pss.h_lo) * u_h.ravel()[idx]) * (0.72 + 0.28 * wv)
        rr = hh * pss.radius_ratio * (0.80 + 0.40 * u_r.ravel()[idx])
        asp = 1.0 + 0.55 * u_asp.ravel()[idx]        # anisotropy 1.0 .. 1.55
        th = 2.0 * np.pi * u_th.ravel()[idx]
        wrp = 0.18 * u_wrp.ravel()[idx]              # organic outline warp
        site_count += int(idx.size)
        hummock_count += int(idx.size)

        # Optional organic elongation ONLY where the mire is meaningfully sloped; this
        # flat raised-bog interior (p95 slope ~0.3%) stays isotropic. Rounded, never lines.
        if slope_core_p95 > 0.01:
            asp = asp * (1.0 + 2.0 * min(1.0, slope_core_p95 / 0.03))

        _stamp(relief, pe, pn, hh, rr, asp, th, wrp, e0, n1, pitch, H, W)

    # Soft-union the accumulated domes so overlaps coalesce into plateaued complexes.
    relief = COALESCE_CAP_M * np.tanh(relief / COALESCE_CAP_M)

    # ------------------------------------------------------------------ hollows + roughness
    # Hummock-free wet interstices dip below the lawn to form hollows (Belyea & Clymo).
    h_typ = 0.22
    hummock_norm = np.clip(relief / h_typ, 0.0, 1.0)
    hollow_amp = 0.09
    relief += -hollow_amp * (1.0 - hummock_norm) * wetness

    # Sub-metre Sphagnum surface roughness (Moore power-law PSD; ~95% variance > 0.6 m,
    # so this is a small, evidence-grounded correction, not decoration). Two octaves.
    rough = (
        (_value_noise(e0, n1, H, W, pitch, cell_m=1, salt=50001) - 0.5) * 0.030
        + (_value_noise(e0, n1, H, W, pitch, cell_m=2, salt=60001) - 0.5) * 0.024
    )
    rough = ndimage.gaussian_filter(rough, 0.30 / pitch, mode="nearest")
    relief += rough

    # ------------------------------------------------------------------ DC + safety
    auth = authority_halo
    relief -= float(np.mean(relief[auth]))          # mean-centre over the mire authority
    # Grade to the REAL pools: taper relief to 0 approaching open water (pool margins are
    # low hollows, not carved hummocks) and hold pools exactly relief-free.
    pool_taper = _smoothstep(pool_dist, 0.0, 1.25)
    relief *= pool_taper
    # Blend the 128 m preview core edge into the flat base (cosine ring, cosmetic).
    ci0, cj0 = core_slice[0].start, core_slice[1].start
    ci1, cj1 = core_slice[0].stop, core_slice[1].stop
    yy = np.arange(H)[:, None].astype(np.float64)
    xx = np.arange(W)[None, :].astype(np.float64)
    band = 24.0
    ey = np.minimum(np.clip((yy - ci0) / band, 0, 1), np.clip((ci1 - 1 - yy) / band, 0, 1))
    ex = np.minimum(np.clip((xx - cj0) / band, 0, 1), np.clip((cj1 - 1 - xx) / band, 0, 1))
    edge = np.minimum(ey, ex)
    relief *= 0.5 - 0.5 * np.cos(np.pi * edge)
    relief[~auth] = 0.0                              # pools/hard/non-mire exactly relief-free

    relief_core = np.ascontiguousarray(relief[core_slice], dtype=np.float64)
    pool_dist_core = np.ascontiguousarray(pool_dist[core_slice], dtype=np.float64)
    meta = {
        "authority_mean_relief_m_after_ops": float(np.mean(relief_core[authority_halo[core_slice]])),
    }
    return HummockField(
        relief_core=relief_core,
        relief_halo=relief,
        authority_core=np.ascontiguousarray(authority_halo[core_slice]),
        open_water_core=np.ascontiguousarray(open_water_halo[core_slice]),
        wetness_halo=wetness,
        pool_dist_core=pool_dist_core,
        hummock_count=hummock_count,
        site_count=site_count,
        meta=meta,
    )


def _stamp(
    relief: np.ndarray,
    pe: np.ndarray, pn: np.ndarray, hh: np.ndarray, rr: np.ndarray,
    asp: np.ndarray, th: np.ndarray, wrp: np.ndarray,
    e0: int, n1: int, pitch: float, H: int, W: int,
) -> None:
    """Add anisotropic warped Wendland-C2 domes; overlap => organic coalescence."""
    for k in range(pe.shape[0]):
        R = rr[k]
        ic = (n1 - pn[k]) / pitch
        jc = (pe[k] - e0) / pitch
        rad_px = int(np.ceil((R * max(1.0, asp[k])) / pitch)) + 1
        i0 = max(0, int(np.floor(ic)) - rad_px)
        i1 = min(H, int(np.ceil(ic)) + rad_px + 1)
        j0 = max(0, int(np.floor(jc)) - rad_px)
        j1 = min(W, int(np.ceil(jc)) + rad_px + 1)
        if i0 >= i1 or j0 >= j1:
            continue
        yy = (np.arange(i0, i1)[:, None] - ic) * pitch
        xx = (np.arange(j0, j1)[None, :] - jc) * pitch
        ct, st = np.cos(th[k]), np.sin(th[k])
        # Rotate into the dome frame; anisotropy stretches one axis.
        u = (xx * ct + yy * st)
        v = (-xx * st + yy * ct)
        alpha = np.arctan2(v, u)
        warp = 1.0 + wrp[k] * np.cos(2.0 * alpha) + 0.5 * wrp[k] * np.cos(3.0 * alpha + 1.3)
        reff = np.sqrt((u / asp[k]) ** 2 + v ** 2) / (R * warp)
        q = np.clip(reff, 0.0, 1.0)
        phi = (1.0 - q) ** 4 * (4.0 * q + 1.0)       # Wendland C2 bump, rounded dome
        relief[i0:i1, j0:j1] += hh[k] * phi
