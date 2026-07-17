"""Top-down + low-oblique diagnostic renders of the plurigaussian core relief.

Diagnostics only. The AUTHORITATIVE acceptance gate is the real-WebGPU ground-level boot of
the packed preview (top-down hillshade fooled the orchestrator three times). These renders
catch gross artifacts (dashes, lattice, blocky steps) early.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _hillshade(z: np.ndarray, pitch: float, az_deg: float, alt_deg: float) -> np.ndarray:
    gy, gx = np.gradient(z, pitch)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az = np.deg2rad(az_deg)
    alt = np.deg2rad(alt_deg)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(shade, 0.0, 1.0)


def render(result, out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    from . import synth

    out_dir.mkdir(parents=True, exist_ok=True)
    core = result.relief_core
    pitch = synth.OUTPUT_PITCH_M
    auth = result.authority[result.core_slice]
    water = result.open_water[result.core_slice]
    written: list[str] = []

    # 1. Top-down hillshade (low sun) with pools overlaid.
    hs = _hillshade(core, pitch, az_deg=315.0, alt_deg=25.0)
    rgb = cm.gray(hs)[..., :3]
    rgb[water] = np.array([0.15, 0.35, 0.7])
    fig, ax = plt.subplots(figsize=(7, 7), dpi=140)
    ax.imshow(rgb, extent=[0, 128, 0, 128])
    ax.set_title("plurigaussian v6 core relief — hillshade (sun 25deg), pools blue")
    ax.set_xlabel("m")
    ax.set_ylabel("m")
    p = out_dir / "01_hillshade_topdown.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    written.append(p.name)

    # 1b. Per-texel NEAREST hillshade of a close 16 m window, interpolation='none': proves the
    # native-0.0625 m relief has NO 4x block-replication (each 6 cm texel is distinct). Contrast
    # with the old 0.25 m upsample, where texels came in flat 4x4 blocks (the terrace defect).
    n = int(round(16.0 / pitch))  # 16 m window near the core center
    r0 = core.shape[0] // 2
    c0 = core.shape[1] // 2
    win = core[r0:r0 + n, c0:c0 + n]
    hs_nn = _hillshade(win, pitch, az_deg=315.0, alt_deg=25.0)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    ax.imshow(hs_nn, cmap="gray", interpolation="none", extent=[0, 16, 0, 16])
    ax.set_title(f"v6 6 cm relief — NEAREST hillshade, 16 m window ({n}x{n} texels, no block-replication)")
    ax.set_xlabel("m")
    ax.set_ylabel("m")
    p = out_dir / "01b_hillshade_nearest_6cm.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    written.append(p.name)

    # 2. Relief colourmap.
    fig, ax = plt.subplots(figsize=(7, 7), dpi=140)
    im = ax.imshow(np.where(auth, core, np.nan), extent=[0, 128, 0, 128],
                   cmap="BrBG", vmin=-0.2, vmax=0.2)
    fig.colorbar(im, ax=ax, label="relief (m)")
    ax.set_title("plurigaussian v6 relief (m) — hollows brown, hummocks green")
    p = out_dir / "02_relief_colour.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    written.append(p.name)

    # 3. Low-oblique perspective preview of a central 64 m transect strip (form check).
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    half = int(round(32.0 / pitch))  # central 64 m strip
    mid = core.shape[0] // 2
    sub = core[mid - half:mid + half, mid - half:mid + half]
    xx, yy = np.meshgrid(np.arange(sub.shape[1]) * pitch, np.arange(sub.shape[0]) * pitch)
    fig = plt.figure(figsize=(10, 5), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, sub, cmap="BrBG", vmin=-0.2, vmax=0.2,
                    linewidth=0, antialiased=True, rcount=180, ccount=180)
    ax.set_zlim(-0.5, 1.5)
    ax.set_box_aspect((1, 1, 0.18))
    ax.view_init(elev=8, azim=-60)
    ax.set_title("plurigaussian v6 — low-oblique (central 64 m, z exaggerated)")
    p = out_dir / "03_oblique_preview.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    written.append(p.name)

    return written
