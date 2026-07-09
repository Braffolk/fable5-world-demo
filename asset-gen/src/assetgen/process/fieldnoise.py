"""Break the polygon look out of categorical scatter fields.

Source data (soil / land-cover / forest-stand polygons) is piecewise-constant with hard
edges. Real ground flora and debris form a CONTINUOUS field with fuzzy meter-scale ecotones,
so shipping the raw polygon edges would make a naive client scatter a straight seam in the
forest floor. We soften at the data level:
  - domain-warp the class boundaries with globally-continuous value noise -> organic edges
  - modulate density with continuous noise -> non-flat interiors, natural thinning
Noise is evaluated in GLOBAL world coordinates so it is seamless across chunk borders.
The field remains coarse GUIDANCE — the client still jitters per instance and blends palettes
across the neighborhood; this just guarantees the guidance itself carries no hard geometry.
"""
from __future__ import annotations

import numpy as np


def _grid_coords(window_en) -> tuple[np.ndarray, np.ndarray, int, int]:
    e_min, n_min, e_max, n_max, t = window_en
    rows = round((n_max - n_min) / t)
    cols = round((e_max - e_min) / t)
    ex = e_min + (np.arange(cols)[None, :] + 0.5) * t
    nz = n_max - (np.arange(rows)[:, None] + 0.5) * t
    return np.broadcast_to(ex, (rows, cols)), np.broadcast_to(nz, (rows, cols)), rows, cols


def value_noise_global(e: np.ndarray, n: np.ndarray, cell_m: float, seed: int) -> np.ndarray:
    """Seamless coherent value noise in [0,1] over global L-EST97 meters (bilinear + smoothstep)."""
    gx, gy = e / cell_m, n / cell_m
    x0, y0 = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
    fx, fy = gx - x0, gy - y0
    ux, uy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)

    def h(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        v = ix * np.int64(374761393) + iy * np.int64(668265263) + np.int64(seed) * np.int64(69069)
        v = (v ^ (v >> np.int64(13))) * np.int64(1274126177)
        return (v & np.int64(0x7FFFFFFF)).astype(np.float64) / 0x7FFFFFFF

    a = h(x0, y0) * (1 - ux) + h(x0 + 1, y0) * ux
    b = h(x0, y0 + 1) * (1 - ux) + h(x0 + 1, y0 + 1) * ux
    return a * (1 - uy) + b * uy


def soften_field(
    class_plane: np.ndarray,
    density_plane: np.ndarray,
    window_en,
    seed: int,
    warp_m: float = 4.0,
    warp_cell_m: float = 14.0,
    density_cell_m: float = 7.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (warped class, noise-modulated density) — organic edges, non-flat interiors."""
    e, n, rows, cols = _grid_coords(window_en)
    t = window_en[4]

    # domain warp: sample the class at a noise-displaced position (few meters). Edge-pad by
    # the warp reach so the sample stays in-bounds; adjacent chunks replicate matching
    # boundary values and share the same global warp, so the seam lines up (no chunk crack).
    m = int(np.ceil(warp_m / t)) + 1
    cls_pad = np.pad(class_plane, m, mode="edge")
    den_pad = np.pad(density_plane, m, mode="edge")
    wx = (value_noise_global(e, n, warp_cell_m, seed) - 0.5) * 2 * warp_m
    wy = (value_noise_global(e, n, warp_cell_m, seed + 101) - 0.5) * 2 * warp_m
    xx, yy = np.meshgrid(np.arange(cols), np.arange(rows))
    sx = np.clip(xx + m + (wx / t), 0, cols + 2 * m - 1).astype(np.intp)
    sy = np.clip(yy + m - (wy / t), 0, rows + 2 * m - 1).astype(np.intp)  # +north = -row
    warped_class = cls_pad[sy, sx]
    warped_density = den_pad[sy, sx]

    # density variation: two octaves, keeps interiors from reading as flat fill
    noise = 0.65 * value_noise_global(e, n, density_cell_m, seed + 7) + 0.35 * value_noise_global(
        e, n, density_cell_m * 0.4, seed + 23
    )
    dmul = 0.45 + 0.75 * noise  # ~0.45..1.2, mean ~0.85
    out_density = np.clip(warped_density.astype(np.float32) * dmul, 0, 255).astype(np.uint8)
    out_density[warped_class == 0] = 0
    return warped_class, out_density
