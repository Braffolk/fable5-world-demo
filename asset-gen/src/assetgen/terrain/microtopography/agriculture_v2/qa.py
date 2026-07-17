"""Numbered decision QA renders for agriculture v2 (matplotlib Agg)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .forms import Synthesis


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hillshade(height: np.ndarray, texel: float) -> np.ndarray:
    gy, gx = np.gradient(np.asarray(height, dtype=np.float64), texel)
    nx, ny, nz = -gx, -gy, np.ones_like(gx)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    lx, ly, lz = -0.45, -0.55, 0.704
    shade = np.clip((nx * lx + ny * ly + nz * lz) / norm, 0, 1)
    return shade


def _save(fig, path: Path) -> dict:
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"path": path.name, "sha256": _sha(path), "bytes": path.stat().st_size}


def render(qa_dir: Path, dev: Synthesis, window: Synthesis, parcel_field,
           dev_theta: float, spectrum: dict, gate_results: dict) -> list[dict]:
    qa_dir.mkdir(parents=True, exist_ok=True)
    texel = float(dev.east[1] - dev.east[0])
    ext_dev = [dev.east[0], dev.east[-1], dev.north[-1], dev.north[0]]
    ext_win = [window.east[0], window.east[-1], window.north[-1], window.north[0]]
    images: list[dict] = []

    # 00 whole parcel-set organization: direction field across the cross-parcel AOI
    fig, ax = plt.subplots(figsize=(8, 8))
    idr = parcel_field.id_raster
    dirmap = np.full(idr.shape, np.nan)
    for pid, info in parcel_field.infos.items():
        dirmap[idr == pid] = np.degrees(info.theta_rad) % 180.0
    im = ax.imshow(dirmap, cmap="twilight", vmin=0, vmax=180,
                   extent=[parcel_field.e0, parcel_field.e0 + parcel_field.width * parcel_field.pitch,
                           parcel_field.n1 - parcel_field.height * parcel_field.pitch, parcel_field.n1])
    for pid, info in parcel_field.infos.items():
        if info.along_half_m < 30:
            continue
        L = info.along_half_m * 0.8
        ax.plot([info.cx - L * np.cos(info.theta_rad), info.cx + L * np.cos(info.theta_rad)],
                [info.cy - L * np.sin(info.theta_rad), info.cy + L * np.sin(info.theta_rad)],
                color="white", lw=1.5)
    ax.plot(*_rect(ext_dev), "r-", lw=1.5, label="dev tile")
    ax.plot(*_rect(ext_win), "y-", lw=1.2, label="boundary window")
    ax.set_title("00 parcel-set row-direction field (per-parcel principal axis)")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, label="row direction (deg, mod 180)", shrink=0.7)
    images.append(_save(fig, qa_dir / "00_parcel_set_direction_field.png"))

    # 01 dev-tile hillshade: variable-spacing rows + wheel tracks
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(_hillshade(dev.height, texel), cmap="gray", extent=ext_dev)
    ax.set_title("01 dev-tile hillshade: variable-spacing seedbed rows + wheel tracks")
    images.append(_save(fig, qa_dir / "01_dev_tile_hillshade.png"))

    # 02 dual-scale spectrum (Marzahn cross-row profiles: small vs large scale)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.loglog(spectrum["small_scale_wavelength_m"], spectrum["small_scale_power"],
              "-", color="#c0392b", lw=1.4, label="small scale (rows+clods, tracks masked)")
    ax.loglog(spectrum["large_scale_wavelength_m"], spectrum["large_scale_power"],
              "-", color="#27ae60", lw=1.4, label="large scale (wheel tracks)")
    ax.loglog(spectrum["combined_wavelength_m"], spectrum["combined_power"],
              "-", color="#7f8c8d", lw=0.9, alpha=0.6, label="combined added relief")
    if spectrum["row_peak_wavelength_m"]:
        ax.axvline(spectrum["row_peak_wavelength_m"], color="#c0392b", ls="--",
                   label=f"row scale {spectrum['row_peak_wavelength_m']:.2f} m")
    if spectrum["track_peak_wavelength_m"]:
        ax.axvline(spectrum["track_peak_wavelength_m"], color="#27ae60", ls="--",
                   label=f"track scale {spectrum['track_peak_wavelength_m']:.2f} m")
    ax.set_xlabel("wavelength (m)")
    ax.set_ylabel("cross-row profile power")
    ax.set_title(f"02 dual-scale break (ratio {spectrum['wavelength_ratio']:.1f}, "
                 f"two distinct scales={spectrum['two_distinct_peaks']})")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    images.append(_save(fig, qa_dir / "02_dual_scale_spectrum.png"))

    # 03 common-light closeup (16 m at tile centre)
    side = int(round(16.0 / texel))
    s0 = (dev.height.shape[0] - side) // 2
    crop = dev.height[s0:s0 + side, s0:s0 + side]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(_hillshade(crop, texel), cmap="gray")
    ax.set_title("03 16 m ground-scale closeup: rows, rut sections, resolved clods")
    ax.axis("off")
    images.append(_save(fig, qa_dir / "03_ground_closeup.png"))

    # 04 parcel-boundary transition (boundary window)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(_hillshade(window.height, texel), cmap="gray", extent=ext_win)
    edge = np.ma.masked_where(window.ids > 0, np.ones_like(window.ids, dtype=float))
    ax.imshow(edge, cmap="autumn", alpha=0.18, extent=ext_win)
    ax.set_title("04 parcel-boundary transition: relief tapers to bare ground (tinted)")
    images.append(_save(fig, qa_dir / "04_parcel_boundary_transition.png"))

    # 05 masks / deviation
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    b = float(np.percentile(np.abs(dev.added[dev.interior]), 99)) if dev.interior.any() else 0.05
    im = axs[0].imshow(dev.added, cmap="RdBu_r", vmin=-b, vmax=b, extent=ext_dev)
    axs[0].set_title("05a added relief (parent deviation)")
    fig.colorbar(im, ax=axs[0], shrink=0.7, label="m")
    owner = np.zeros(dev.ids.shape, dtype=int)
    owner[np.abs(dev.rows) >= np.maximum(np.abs(dev.tracks), np.abs(dev.clods))] = 1
    owner[np.abs(dev.tracks) > np.maximum(np.abs(dev.rows), np.abs(dev.clods))] = 2
    owner[np.abs(dev.clods) > np.maximum(np.abs(dev.rows), np.abs(dev.tracks))] = 3
    owner[~dev.interior] = 0
    axs[1].imshow(owner, cmap="tab10", vmin=0, vmax=9, extent=ext_dev)
    axs[1].set_title("05b typed ownership (0 none/1 rows/2 tracks/3 clods)")
    images.append(_save(fig, qa_dir / "05_masks_deviation_ownership.png"))
    return images


def _rect(ext):
    e0, e1, n0, n1 = ext
    return [e0, e1, e1, e0, e0], [n0, n0, n1, n1, n0]
