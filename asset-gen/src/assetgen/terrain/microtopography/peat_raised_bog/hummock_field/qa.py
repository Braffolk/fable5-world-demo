"""Ground-level QA renders for the bog v5 hummock field.

The REAL acceptance gate (top-down hillshade fooled the orchestrator three times): a
low-oblique, near-eye-level look-across of the relief surface, plus a top-down hillshade
for reference. Pure-numpy heightfield raymarcher (no external renderer). Deterministic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage


def _png(path: Path, rgb: np.ndarray) -> None:
    import struct
    import zlib

    arr = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    h, w, _ = arr.shape
    raw = bytearray()
    for row in range(h):
        raw.append(0)
        raw.extend(arr[row].tobytes())

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    out = sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b"")
    path.write_bytes(out)


def hillshade(relief: np.ndarray, pitch: float, out_path: Path, az_deg: float = 315.0, el_deg: float = 30.0) -> None:
    gy, gx = np.gradient(relief.astype(np.float64), pitch)
    nz = np.ones_like(relief)
    norm = np.sqrt(gx * gx + gy * gy + 1.0)
    az, el = np.radians(az_deg), np.radians(el_deg)
    lx, ly, lz = np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)
    shade = np.clip((-gx * lx - gy * ly + nz * lz) / norm, 0, 1)
    shade = 0.15 + 0.85 * shade
    rgb = np.stack([shade * 0.82, shade * 0.78, shade * 0.66], axis=-1)
    _png(out_path, rgb)


def _surface_colour(z: np.ndarray, wet: np.ndarray, water: np.ndarray) -> np.ndarray:
    """Sphagnum-ish colour ramp: wet hollows olive, lawn green, hummock caps ochre."""
    t = np.clip((z + 0.12) / 0.42, 0.0, 1.0)[..., None]
    hollow = np.array([0.29, 0.34, 0.20])
    lawn = np.array([0.42, 0.47, 0.28])
    cap = np.array([0.66, 0.60, 0.36])
    col = np.where(t < 0.5, hollow + (lawn - hollow) * (t / 0.5), lawn + (cap - lawn) * ((t - 0.5) / 0.5))
    water_col = np.array([0.15, 0.22, 0.30])
    col = np.where(water[..., None], water_col, col)
    return col


def render_oblique(
    relief: np.ndarray,
    water: np.ndarray,
    pitch: float,
    out_path: Path,
    *,
    eye_xyz: tuple[float, float, float],
    look_xyz: tuple[float, float, float],
    sun_az_deg: float = 120.0,
    sun_el_deg: float = 14.0,
    screen: tuple[int, int] = (960, 540),
    fov_deg: float = 62.0,
) -> None:
    H, W = relief.shape
    dom = H * pitch
    z = relief.astype(np.float64)
    water_f = water.astype(np.float64)

    def sample(field: np.ndarray, px: np.ndarray, py: np.ndarray, cval: float = 0.0) -> np.ndarray:
        r = py / pitch - 0.5
        c = px / pitch - 0.5
        return ndimage.map_coordinates(field, np.stack([r, c]), order=1, mode="constant", cval=cval)

    eye = np.array(eye_xyz, dtype=np.float64)
    fwd = np.array(look_xyz, dtype=np.float64) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    sw, sh = screen
    f = 1.0 / np.tan(np.radians(fov_deg) / 2.0)
    us = (np.arange(sw) + 0.5) / sw * 2 - 1
    vs = 1 - (np.arange(sh) + 0.5) / sh * 2
    uu, vv = np.meshgrid(us * (sw / sh), vs)
    dirs = (fwd[None, None, :] * f + right[None, None, :] * uu[..., None] + up[None, None, :] * vv[..., None])
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    dirs = dirs.reshape(-1, 3)
    origin = np.broadcast_to(eye, dirs.shape).copy()

    n = dirs.shape[0]
    hit_t = np.full(n, np.inf)
    active = np.ones(n, dtype=bool)
    t = np.full(n, 0.3)
    dt, t_max = 0.14, 190.0
    prev_gap = np.zeros(n)
    while active.any() and t[active].min() < t_max:
        idx = np.nonzero(active)[0]
        ti = t[idx]
        px = origin[idx, 0] + dirs[idx, 0] * ti
        py = origin[idx, 1] + dirs[idx, 1] * ti
        rz = origin[idx, 2] + dirs[idx, 2] * ti
        inside = (px >= 0) & (px < dom) & (py >= 0) & (py < dom)
        surf = sample(z, px, py, 0.0)
        gap = rz - surf
        crossed = inside & (gap <= 0.0)
        if crossed.any():
            hit = idx[crossed]
            frac = prev_gap[hit] / np.maximum(prev_gap[hit] - gap[crossed], 1e-6)
            hit_t[hit] = ti[crossed] - dt * (1 - np.clip(frac, 0, 1))
            active[hit] = False
        prev_gap[idx] = gap
        gone = (ti > t_max) | ((py > dom + 5) & (dirs[idx, 1] > 0)) | ((py < -5) & (dirs[idx, 1] < 0))
        active[idx[gone]] = False
        t[idx] += dt

    img = np.zeros((n, 3))
    miss = ~np.isfinite(hit_t)
    # Sky gradient for misses.
    sky_v = np.clip(vv.reshape(-1), 0, 1)
    sky = np.array([0.62, 0.70, 0.82]) * (0.5 + 0.5 * sky_v[:, None]) + np.array([0.80, 0.84, 0.80]) * (0.5 - 0.5 * sky_v[:, None])
    img[miss] = sky[miss]

    hi = np.nonzero(~miss)[0]
    if hi.size:
        th = hit_t[hi]
        hx = origin[hi, 0] + dirs[hi, 0] * th
        hy = origin[hi, 1] + dirs[hi, 1] * th
        hz = sample(z, hx, hy, 0.0)
        eps = pitch
        zx = (sample(z, hx + eps, hy) - sample(z, hx - eps, hy)) / (2 * eps)
        zy = (sample(z, hx, hy + eps) - sample(z, hx, hy - eps)) / (2 * eps)
        nrm = np.stack([-zx, -zy, np.ones_like(zx)], axis=-1)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
        az, el = np.radians(sun_az_deg), np.radians(sun_el_deg)
        L = np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])
        diff = np.clip(nrm @ L, 0, 1)
        wsurf = sample(water_f, hx, hy, 0.0) > 0.5
        base = _surface_colour(hz, np.zeros_like(hz), wsurf)
        shade = (0.28 + 0.72 * diff)[:, None]
        # Distance haze.
        haze = np.clip(th / t_max, 0, 1)[:, None]
        col = base * shade
        col = col * (1 - 0.55 * haze) + np.array([0.74, 0.79, 0.85]) * (0.55 * haze)
        img[hi] = col

    _png(out_path, img.reshape(sh, sw, 3))
