"""Numbered visual diagnostics for the domain diffusion challenger."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

_PANEL = 520
_LABEL = 58


def _hillshade(value: np.ndarray, pitch_m: float) -> np.ndarray:
    gy, gx = np.gradient(ndimage.gaussian_filter(value, max(0.25 / pitch_m, 0.35)), pitch_m)
    light = (-0.38 * gx - 0.55 * gy + 0.74) / np.sqrt(gx * gx + gy * gy + 1.0)
    shade = np.clip(0.5 + 0.5 * light, 0.0, 1.0)
    low = np.asarray((43, 72, 60), dtype=np.float64)
    high = np.asarray((225, 214, 174), dtype=np.float64)
    return np.rint(low + shade[..., None] * (high - low)).astype(np.uint8)


def _diverging(value: np.ndarray, scale: float) -> np.ndarray:
    normalized = np.clip(value / max(scale, 1.0e-6), -1.0, 1.0)
    middle = np.asarray((239, 235, 219), dtype=np.float64)
    low = np.asarray((35, 93, 148), dtype=np.float64)
    high = np.asarray((184, 57, 43), dtype=np.float64)
    endpoint = np.where((normalized >= 0)[..., None], high, low)
    return np.rint(middle * (1.0 - np.abs(normalized)[..., None]) + endpoint * np.abs(normalized)[..., None]).astype(np.uint8)


def _scalar(value: np.ndarray, low=(31, 52, 61), high=(236, 182, 70)) -> np.ndarray:
    lo, hi = np.quantile(value, (0.02, 0.98))
    normalized = np.clip((value - lo) / max(float(hi - lo), 1.0e-8), 0.0, 1.0)
    return np.rint(np.asarray(low)[None, None] * (1.0 - normalized[..., None]) + np.asarray(high)[None, None] * normalized[..., None]).astype(np.uint8)


def _panel(rgb: np.ndarray, title: str, detail: str, smooth: bool = True) -> Image.Image:
    canvas = Image.new("RGB", (_PANEL, _PANEL + _LABEL), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((9, 7), title, fill="black")
    draw.text((9, 29), detail, fill=(55, 55, 55))
    image = Image.fromarray(np.flipud(rgb), "RGB").resize((_PANEL, _PANEL), Image.Resampling.BILINEAR if smooth else Image.Resampling.NEAREST)
    canvas.paste(image, (0, _LABEL))
    return canvas


def _compose(path: Path, panels: list[Image.Image], columns: int, footer: str) -> None:
    rows = (len(panels) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * _PANEL, rows * (_PANEL + _LABEL) + 34), "white")
    for index, panel in enumerate(panels):
        canvas.paste(panel, ((index % columns) * _PANEL, (index // columns) * (_PANEL + _LABEL)))
    ImageDraw.Draw(canvas).text((10, canvas.height - 24), footer, fill=(95, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, "PNG", optimize=True)


def write_qa(
    root: Path,
    *,
    c0: np.ndarray,
    stage1: np.ndarray,
    final: np.ndarray,
    macro: np.ndarray,
    fine: np.ndarray,
    conditions: np.ndarray,
    opportunity: np.ndarray,
    hard: np.ndarray,
    holdout_base: np.ndarray,
    holdout_truth: np.ndarray,
    holdout_stage1: np.ndarray,
    holdout_final: np.ndarray,
    pitch_m: float,
) -> dict[str, str]:
    scale = max(float(np.quantile(np.abs(macro[~hard]), 0.98)), 0.05)
    fine_scale = max(float(np.quantile(np.abs(fine[~hard]), 0.98)), 0.01)

    path1 = root / "01-target-common-light-c0-stage1-final.png"
    _compose(path1, [
        _panel(_hillshade(c0, pitch_m), "C0 corrected authority", "same light, range, and full physical master"),
        _panel(_hillshade(stage1, pitch_m), "Stage 1 domain organization", "2 m diffusion owns crest/face/toe macro only"),
        _panel(_hillshade(final, pitch_m), "Final two-stage field", "Stage 2 adds only the 1-2 m innovation band"),
    ], 3, "PASS REQUIRES CONNECTED CREST / SHOULDER / FACE ORGANIZATION, NOT MOTTLING OR STAMPS")

    path2 = root / "02-target-signed-scale-ownership.png"
    _compose(path2, [
        _panel(_diverging(macro, scale), "Stage 1 signed macro delta", f"common scale +/-{scale:.3f} m"),
        _panel(_diverging(fine, fine_scale), "Stage 2 signed innovation", f"high-pass ownership only; +/-{fine_scale:.3f} m"),
        _panel(_diverging(final - c0, scale), "Composed signed delta", "macro plus bounded within-form innovation"),
    ], 3, "STAGE 2 MAY NOT REPLACE, STAMP, OR REORGANIZE THE STAGE-1 LONG-RANGE FIELD")

    mask_rgb = np.zeros((*hard.shape, 3), dtype=np.uint8)
    mask_rgb[:] = (54, 145, 111)
    mask_rgb[hard] = (116, 12, 52)
    path3 = root / "03-target-conditioning-and-ownership.png"
    _compose(path3, [
        _panel(_scalar(conditions[2]), "C0 slope condition", "metric broad-surface slope"),
        _panel(_scalar(conditions[4]), "ALS structural network", "preserved measured network, not height authority"),
        _panel(_scalar(opportunity), "Continuous eligibility", "slope/network plus hard-distance taper"),
        _panel(mask_rgb, "Final ownership", "green=eligible/corrected authority; magenta=hard C0", smooth=False),
    ], 2, "WATER / PROTECTED / MAPPED NON-HEIGHTFIELD / COLLAR OWNERSHIP REMAINS EXACT C0")

    hold_pitch = 1.0
    path4 = root / "04-biala-geographic-holdout-common-light.png"
    _compose(path4, [
        _panel(_hillshade(holdout_base, hold_pitch), "Held-out structural observation", "same epoch; geographic block excluded from training"),
        _panel(_hillshade(holdout_truth, hold_pitch), "Held-out 1 m structural target", "direct-support R0 weak surface, not truth"),
        _panel(_hillshade(holdout_stage1, hold_pitch), "Held-out Stage 1", "domain organization from coarse condition"),
        _panel(_hillshade(holdout_final, hold_pitch), "Held-out final", "same frozen model and seed; no holdout tuning"),
    ], 2, "SAME-SITE HOLDOUT ONLY / NO ESTONIA TRANSFER, PRODUCTION, OR SURVEY-TRUTH CLAIM")

    return {
        path1.name: "Direct common-light comparison of corrected C0, the long-range Stage-1 field, and the final field over the complete target master.",
        path2.name: "Separates signed stage ownership so local refinement cannot hide or overwrite the domain-scale result.",
        path3.name: "Shows metric terrain conditions, the preserved ALS network, continuous eligibility, and exact hard ownership.",
        path4.name: "Shows the geographically separated same-epoch Biala holdout under one light; it is a bounded R0 screen, not independent transfer evidence.",
    }
