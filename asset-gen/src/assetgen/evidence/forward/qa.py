"""Three labeled QA views for the unqualified FORWARD terrain candidate."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

_PANEL = 420
_LABEL = 70
_COLORS = ((205, 48, 47), (38, 139, 210), (133, 153, 0), (181, 137, 0))


def _hillshade(height: np.ndarray, pixel_m: float) -> np.ndarray:
    south, east = np.gradient(height.astype(np.float64), pixel_m)
    nx = -east
    ny = south
    nz = np.ones(height.shape)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx /= norm
    ny /= norm
    nz /= norm
    light = np.asarray((-0.5, 0.5, np.sqrt(0.5)))
    shade = np.clip(nx * light[0] + ny * light[1] + nz * light[2], -0.2, 1.0)
    shade = np.clip((shade + 0.2) / 1.2, 0.0, 1.0) ** 0.75
    return np.rint(shade * 255).astype(np.uint8)


def _residual_rgb(residual: np.ndarray, scale: float) -> np.ndarray:
    value = np.clip(residual / scale, -1.0, 1.0)
    low = np.asarray((30, 91, 157), dtype=np.float32)
    mid = np.asarray((244, 240, 221), dtype=np.float32)
    high = np.asarray((184, 48, 38), dtype=np.float32)
    weight = np.abs(value)[..., None]
    endpoint = np.where((value >= 0)[..., None], high, low)
    return np.rint(mid * (1.0 - weight) + endpoint * weight).astype(np.uint8)


def _panel(rgb: np.ndarray, title: str, detail: str) -> Image.Image:
    canvas = Image.new("RGB", (_PANEL, _PANEL + _LABEL), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill="black")
    draw.text((8, 30), detail, fill=(45, 45, 45))
    image = Image.fromarray(rgb).resize((_PANEL - 16, _PANEL - 16), Image.Resampling.NEAREST)
    canvas.paste(image, (8, _LABEL))
    return canvas


def _grid(path: Path, panels: list[Image.Image], footer: str) -> None:
    columns = 2
    rows = (len(panels) + 1) // 2
    footer_h = 44
    canvas = Image.new("RGB", (columns * _PANEL, rows * panels[0].height + footer_h), "white")
    for index, panel in enumerate(panels):
        canvas.paste(panel, ((index % columns) * _PANEL, (index // columns) * panel.height))
    ImageDraw.Draw(canvas).text((10, canvas.height - 30), footer, fill=(95, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=True)


def render_candidate_qa(
    height: np.ndarray,
    patches: tuple[dict[str, Any], ...],
    qa_dir: Path,
    *,
    pixel_m: float,
) -> tuple[tuple[Path, str], ...]:
    overview = Image.fromarray(_hillshade(height, pixel_m), mode="L")
    overview.thumbnail((700, 900), Image.Resampling.BILINEAR)
    overview = overview.convert("RGB")
    draw = ImageDraw.Draw(overview)
    scale_x = overview.width / height.shape[1]
    scale_y = overview.height / height.shape[0]
    for index, patch in enumerate(patches):
        row, column = patch["row_column"]
        size = patch["height_m"].shape[0]
        box = (
            round(column * scale_x),
            round(row * scale_y),
            round((column + size) * scale_x),
            round((row + size) * scale_y),
        )
        draw.rectangle(box, outline=_COLORS[index], width=3)
        draw.text((box[0] + 2, box[1] + 2), str(index + 1), fill=_COLORS[index])
    canvas = Image.new("RGB", (overview.width, overview.height + 72), "white")
    canvas.paste(overview, (0, 52))
    title = "FORWARD MRV DTM overview: provider-processed 0.25 m grid, north up"
    ImageDraw.Draw(canvas).text((8, 10), title, fill="black")
    path1 = qa_dir / "01_dtm_overview_and_patch_locations.png"
    qa_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(path1, format="PNG", optimize=True)

    hill_panels = []
    residual_panels = []
    residual_scale = max(
        float(np.quantile(np.abs(np.concatenate([patch["residual_m"].ravel() for patch in patches])), 0.995)),
        0.01,
    )
    for index, patch in enumerate(patches):
        east, north = patch["southwest_xy_m"]
        detail = (
            f"q={patch['target_quantile']:.2f}; residual RMS={patch['residual_rms_m']:.3f} m\n"
            f"SW E {east:.3f}, N {north:.3f}"
        )
        hill = np.repeat(_hillshade(patch["height_m"], pixel_m)[..., None], 3, axis=2)
        hill_panels.append(_panel(hill, f"Patch {index + 1}: native 32 m hillshade", detail))
        residual_panels.append(
            _panel(
                _residual_rgb(patch["residual_m"], residual_scale),
                f"Patch {index + 1}: 2 m-sigma residual",
                f"height minus Gaussian baseline; shared +/-{residual_scale:.3f} m",
            )
        )
    path2 = qa_dir / "02_stratified_native_hillshade.png"
    _grid(
        path2,
        hill_panels,
        "FIXED ROUGHNESS QUANTILES / NOT HAND-PICKED / PROVIDER DTM, NOT INDEPENDENT TRUTH",
    )
    path3 = qa_dir / "03_stratified_2m_sigma_residual.png"
    _grid(
        path3,
        residual_panels,
        "ONE SHARED COLOR SCALE / KRIGING SUPPORT UNKNOWN / NO SYNTHESIS AUTHORIZATION",
    )
    return (
        (
            path1,
            "Full released Marrviken harvest-area DTM hillshade with the four deterministic 32 m patch locations. The raster is provider-processed and all pixels are finite, but the release carries no measured-versus-kriged support mask.",
        ),
        (
            path2,
            "Native-grid hillshade for patches selected nearest fixed 0.10, 0.50, 0.90, and 0.99 quantiles of 32 m band-residual RMS across a non-overlapping site grid. This exposes variance without visual cherry-picking.",
        ),
        (
            path3,
            "The same four patches after subtracting a full-raster Gaussian baseline with 2 m sigma, all rendered with one shared signed height scale. Residuals include sub-landform terrain, retained obstacles, filtering effects, and unknown kriging effects.",
        ),
    )
