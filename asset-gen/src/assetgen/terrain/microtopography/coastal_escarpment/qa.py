"""Compact visual diagnostics for coastal-escarpment capacity artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

_LABEL_H = 66
_PANEL = 512


def _hillshade(z: np.ndarray, valid: np.ndarray, cell_m: float) -> np.ndarray:
    filled = np.nan_to_num(z, nan=float(np.nanmedian(z[valid])))
    gy, gx = np.gradient(ndimage.gaussian_filter(filled, 0.55), cell_m)
    light = (-0.42 * gx - 0.58 * gy + 0.70) / np.sqrt(gx * gx + gy * gy + 1.0)
    shade = np.clip(0.5 + 0.5 * light, 0.0, 1.0)
    rgb = np.repeat(np.rint(255.0 * shade)[..., None].astype(np.uint8), 3, axis=2)
    rgb[~valid] = (116, 12, 52)
    return rgb


def _diverging(value: np.ndarray, valid: np.ndarray, scale: float) -> np.ndarray:
    normalized = np.clip(np.nan_to_num(value / scale), -1.0, 1.0)
    middle = np.asarray((238, 235, 218), dtype=np.float32)
    low = np.asarray((31, 94, 151), dtype=np.float32)
    high = np.asarray((181, 55, 43), dtype=np.float32)
    end = np.where((normalized >= 0.0)[..., None], high, low)
    rgb = np.rint(middle * (1.0 - np.abs(normalized)[..., None]) + end * np.abs(normalized)[..., None]).astype(np.uint8)
    rgb[~valid] = (116, 12, 52)
    return rgb


def _mask_rgb(direct: np.ndarray, modelable: np.ndarray, fine: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*direct.shape, 3), dtype=np.uint8)
    rgb[:] = (116, 12, 52)
    rgb[modelable] = (54, 145, 111)
    rgb[direct] = (54, 145, 111)
    rgb[fine] = (81, 186, 220)
    return rgb


def _panel(rgb: np.ndarray, title: str, detail: str, *, smooth: bool = False) -> Image.Image:
    canvas = Image.new("RGB", (_PANEL, _PANEL + _LABEL_H), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill="black")
    draw.text((10, 30), detail, fill=(55, 55, 55))
    image = Image.fromarray(np.flipud(rgb), mode="RGB")
    image = image.resize(
        (_PANEL, _PANEL),
        Image.Resampling.BILINEAR if smooth else Image.Resampling.NEAREST,
    )
    canvas.paste(image, (0, _LABEL_H))
    return canvas


def _compose(path: Path, panels: tuple[Image.Image, ...], columns: int, footer: str) -> None:
    rows = (len(panels) + columns - 1) // columns
    footer_h = 38
    canvas = Image.new("RGB", (columns * _PANEL, rows * (_PANEL + _LABEL_H) + footer_h), "white")
    for index, panel in enumerate(panels):
        canvas.paste(panel, ((index % columns) * _PANEL, (index // columns) * (_PANEL + _LABEL_H)))
    ImageDraw.Draw(canvas).text((10, canvas.height - 26), footer, fill=(95, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=True)


def render_capacity_qa(
    arrays: dict[str, np.ndarray],
    qa_dir: Path,
    *,
    cell_m: float,
    core_offset_cells: int,
) -> tuple[dict[str, Any], ...]:
    source = arrays["source_m"].astype(np.float64)
    base = arrays["base_m"].astype(np.float64)
    reconstruction = arrays["reconstruction_m"].astype(np.float64)
    direct = arrays["direct_support"] != 0
    modelable = arrays["modelable_support"] != 0
    fine = arrays["fine_authority"] != 0
    core = np.zeros(source.shape, dtype=bool)
    core[core_offset_cells:-core_offset_cells, core_offset_cells:-core_offset_cells] = True
    core_valid = modelable & core

    source_shade = _hillshade(source, direct, cell_m)
    source_selection = source_shade.copy()
    low = core_offset_cells
    high = source.shape[0] - core_offset_cells - 1
    source_selection[low : low + 2, low : high + 1] = (255, 67, 28)
    source_selection[high - 1 : high + 1, low : high + 1] = (255, 67, 28)
    source_selection[low : high + 1, low : low + 2] = (255, 67, 28)
    source_selection[low : high + 1, high - 1 : high + 1] = (255, 67, 28)
    support_rgb = _mask_rgb(direct, modelable, fine)
    path1 = qa_dir / "01-source-selection-and-support.png"
    _compose(
        path1,
        (
            _panel(source_selection, "Retained source and 128 m core", "orange = core; magenta = no direct qualified class-2 support"),
            _panel(support_rgb, "Explicit evidence masks", "cyan=fine authority, green=direct macro, magenta=unsupported; no infill"),
        ),
        2,
        "BIALA GORA SOURCE-DOMAIN CALIBRATION ONLY / VENDOR CLASS 2 IS NOT AUTHORITATIVE TRUTH",
    )

    path2 = qa_dir / "02-source-base-reconstruction-shaded.png"
    _compose(
        path2,
        (
            _panel(_hillshade(source, direct, cell_m), "Observed source", "0.5 m vendor class-2 mean; unsupported stays magenta"),
            _panel(_hillshade(base, modelable, cell_m), "Coarse structural base", "8 m normalized 2D scale-space; cliff becomes a broad ramp"),
            _panel(_hillshade(reconstruction, modelable, cell_m), "2D reconstruction", "connected face, break, gullies, ribs, toe, and ordinary ground"),
        ),
        3,
        "COMMON LIGHT / NO LINE PROFILE OR LINE-NORMAL TARGET PARAMETERIZATION",
    )

    macro = arrays["macro_component_m"].astype(np.float64)
    fine_component = arrays["fine_component_m"].astype(np.float64)
    residual = arrays["observed_residual_m"].astype(np.float64)
    macro_scale = max(float(np.nanquantile(np.abs(macro[core_valid]), 0.98)), 0.05)
    fine_valid = fine & core
    fine_scale = max(float(np.nanquantile(np.abs(fine_component[fine_valid]), 0.98)), 0.01)
    residual_valid = direct & core
    residual_scale = max(float(np.nanquantile(np.abs(residual[residual_valid]), 0.98)), 0.01)
    path3 = qa_dir / "03-reconstruction-components-and-residual.png"
    _compose(
        path3,
        (
            _panel(_diverging(macro, modelable, macro_scale), "Macro/meso component", f"source structure minus 8 m base; +/-{macro_scale:.2f} m"),
            _panel(_diverging(fine_component, fine, fine_scale), "Evidence-supported fine component", f"coherent 0.5-1.25 m band; +/-{fine_scale:.2f} m"),
            _panel(_diverging(residual, direct, residual_scale), "Observed minus reconstruction", f"direct source cells only; +/-{residual_scale:.2f} m"),
        ),
        3,
        "MACRO FORM IS NOT GENERIC ROUGHNESS / FINE BAND IS ABSENT WHERE SOURCE SUPPORT IS UNSAFE",
    )

    core_slice = np.s_[low : high + 1, low : high + 1]
    source_core = source[core_slice]
    direct_core = direct[core_slice]
    recon_core = reconstruction[core_slice]
    model_core = modelable[core_slice]
    split = source_core.shape[0] // 2
    regions = (
        (np.s_[split:, :split], "shoulder / face / transverse incision"),
        (np.s_[:split, split:], "mass-wasting ribs / toe / ordinary ground"),
    )
    panels: list[Image.Image] = []
    for region, name in regions:
        panels.extend(
            (
                _panel(_hillshade(source_core[region], direct_core[region], cell_m), f"Observed: {name}", "source-domain reference"),
                _panel(_hillshade(recon_core[region], model_core[region], cell_m), f"Reconstructed: {name}", "same light and crop"),
            )
        )
    path4 = qa_dir / "04-shaded-structural-closeups.png"
    _compose(
        path4,
        tuple(panels),
        2,
        "PASS ONLY IF CONNECTIVITY SURVIVES WITHOUT RIBS, CAPS, SLABS, OR A ROUGHENED RAMP",
    )

    return tuple(
        {
            "path": path,
            "interpretation": interpretation,
        }
        for path, interpretation in (
            (path1, "Binds the selected 128 m core and distinguishes direct macro support, stricter fine authority, and unsupported cells; no unknown cell is reconstructed."),
            (path2, "Compares the retained source, deliberately over-smoothed base, and genuinely two-dimensional connected-form reconstruction under one light."),
            (path3, "Separates macro/meso relief, support-limited fine relief, and observed-source residual rather than hiding them in a single beauty image."),
            (path4, "Shows paired source/reconstruction closeups across the scarp-incision system and the mass-wasting/toe transition."),
        )
    )
