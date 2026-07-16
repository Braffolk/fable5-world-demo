"""Four diagnostic images for curvilinear-strip FLOAT review."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .evidence import Evidence
from .model import CHAIN_NAMES, Result


WIDTH, HEIGHT = 1800, 1400
PAPER = (242, 238, 225)
INK = (30, 36, 32)


def _font(size: int):
    for path in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _hillshade(height: np.ndarray, pitch: float = 0.25) -> np.ndarray:
    gy, gx = np.gradient(height.astype(np.float64), pitch)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    azimuth, altitude = np.deg2rad(315.0), np.deg2rad(38.0)
    light = np.sin(altitude) * np.cos(slope) + np.cos(altitude) * np.sin(slope) * np.cos(azimuth - aspect)
    light = np.clip((light + 0.18) / 1.18, 0.0, 1.0)
    lo, hi = np.percentile(height, [2.0, 98.0])
    elevation = np.clip((height - lo) / max(float(hi - lo), 1.0e-6), 0.0, 1.0)
    color = np.stack((0.43 + 0.35 * elevation, 0.45 + 0.28 * elevation, 0.34 + 0.21 * elevation), axis=-1)
    return np.clip(color * (0.48 + 0.72 * light[..., None]), 0.0, 1.0)


def _signed(value: np.ndarray, limit: float | None = None) -> np.ndarray:
    limit = limit or max(float(np.percentile(np.abs(value), 98.0)), 1.0e-8)
    t = np.clip(value / limit, -1.0, 1.0)
    neutral = np.asarray([0.94, 0.92, 0.85])
    red, blue = np.asarray([0.76, 0.16, 0.10]), np.asarray([0.06, 0.28, 0.57])
    return np.where((t >= 0)[..., None], neutral + t[..., None] * (red - neutral), neutral + (-t)[..., None] * (blue - neutral))


def _scalar(value: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    lo, hi = np.percentile(value, [2.0, 98.0])
    t = np.clip((value - lo) / max(float(hi - lo), 1.0e-9), 0.0, 1.0)
    base = np.asarray([0.94, 0.92, 0.85])
    return base + t[..., None] * (np.asarray(color) - base)


def _render(rgb: np.ndarray, size: tuple[int, int]) -> Image.Image:
    image = Image.fromarray(np.asarray(np.clip(rgb, 0.0, 1.0) * 255.0, dtype=np.uint8), "RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, PAPER)
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def _sheet(title: str, note: str, panels: list[tuple[str, np.ndarray]]) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
    draw = ImageDraw.Draw(image)
    draw.text((36, 22), title, font=_font(27), fill=INK)
    draw.text((36, 58), note, font=_font(16), fill=(73, 77, 68))
    panel_w, panel_h = (WIDTH - 72) // 2, (HEIGHT - 130) // 2
    for index, (label, rgb) in enumerate(panels):
        row, col = divmod(index, 2)
        x, y = 36 + col * panel_w, 94 + row * panel_h
        draw.text((x + 8, y + 5), label, font=_font(17), fill=INK)
        image.paste(_render(rgb, (panel_w - 16, panel_h - 42)), (x + 8, y + 34))
    return image


def _ownership(evidence: Evidence, result: Result) -> np.ndarray:
    base = _hillshade(evidence.c0_solve_m)
    support = np.clip(result.strip_support, 0.0, 1.0)
    rgb = base * (1.0 - 0.55 * support[..., None]) + np.asarray([0.12, 0.52, 0.44]) * 0.55 * support[..., None]
    chain = result.chain_distance_m <= 0.35
    rgb[chain] = np.asarray([0.92, 0.32, 0.10])
    rgb[evidence.hard_solve] *= 0.30
    return rgb


def _close_crop(array: np.ndarray, support: np.ndarray, padding: int = 20) -> np.ndarray:
    rows, cols = np.nonzero(support > 0.05)
    r0, r1 = max(0, int(rows.min()) - padding), min(array.shape[0], int(rows.max()) + padding + 1)
    c0, c1 = max(0, int(cols.min()) - padding), min(array.shape[1], int(cols.max()) + padding + 1)
    return array[r0:r1, c0:c1]


def write_qa(directory: Path, evidence: Evidence, result: Result, recipe_sha256: str, source_sha256s: list[str]) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    hard_view = _hillshade(evidence.c0_solve_m)
    hard_view[evidence.hard_solve] *= 0.30
    c0_close = _close_crop(evidence.c0_solve_m, result.strip_support)
    c1_close = _close_crop(result.solve_absolute_m, result.strip_support)
    delta_close = _close_crop(result.solve_delta_m, result.strip_support)
    chain_close = _close_crop(_ownership(evidence, result), result.strip_support)
    erosion_deposition = result.deposition_m - result.erosion_m
    sheets = [
        (
            "01-strip-topology-ownership.png",
            _sheet(
                "01 / Continuous curvilinear-strip topology and ownership",
                "Orange: six jointly smoothed chains. Green: compact strip. Hard ownership is dark.",
                [
                    ("accepted C0 and hard masks", hard_view),
                    ("strip and paired chain ownership", _ownership(evidence, result)),
                    ("continuous station coordinate s", _scalar(result.strip_s_m * result.strip_support, (0.10, 0.37, 0.69))),
                    ("signed normal coordinate n in owned strip", _signed(result.strip_n_m * result.strip_support)),
                ],
            ),
            "Exact ownership and single continuous crest-to-toe strip topology.",
        ),
        (
            "02-common-light-c0-c1-macro.png",
            _sheet(
                "02 / Common-light C0 and C1 macro verdict",
                "Same light, elevation palette, and scale. No browser, material, or runtime rendering.",
                [
                    ("accepted C0", _hillshade(evidence.c0_solve_m)),
                    ("curvilinear-strip C1", _hillshade(result.solve_absolute_m)),
                    ("signed C1-C0", _signed(result.solve_delta_m)),
                    ("absolute macro change", _scalar(np.abs(result.solve_delta_m), (0.67, 0.25, 0.10))),
                ],
            ),
            "Common-light whole-canvas C0/C1 macro comparison.",
        ),
        (
            "03-crest-face-bench-rill-toe-closeup.png",
            _sheet(
                "03 / Crest-face-bench-rill-toe closeup",
                "Closeup bounds come from strip ownership, not a hand-selected scene coordinate.",
                [
                    ("C0", _hillshade(c0_close)),
                    ("C1", _hillshade(c1_close)),
                    ("six chains and compact support", chain_close),
                    ("signed local change", _signed(delta_close)),
                ],
            ),
            "Close inspection of crest, paired bench breaks, face, measured rills, and toe.",
        ),
        (
            "04-strip-space-process-and-micro-bands.png",
            _sheet(
                "04 / Strip-space source, process ledger, and subordinate micro band",
                "Rills derive from connected measured strip residuals; micro is generated continuation, not truth.",
                [
                    ("measured ALS strip highpass", _signed(result.strip.measured_strip_m - result.strip.macro_strip_m)),
                    ("retained connected rill graph", _scalar(result.strip.rill_strip, (0.08, 0.32, 0.66))),
                    ("erosion blue / deposition red", _signed(erosion_deposition)),
                    ("0.0625-0.5 m generated continuation", _signed(result.micro_master_m)),
                ],
            ),
            "Measured strip residual, connected rills, conservative process, and generated fine continuation.",
        ),
    ]
    rows = []
    for filename, image, interpretation in sheets:
        path = directory / filename
        image.save(path, optimize=True)
        payload = path.read_bytes()
        rows.append({"path": filename, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), "dimensions_px": [WIDTH, HEIGHT], "interpretation": interpretation})
    index = {
        "schema_version": "laas.unconsolidated-sand-curvilinear-strip-qa/1",
        "recipe_sha256": recipe_sha256,
        "source_sha256s": sorted(source_sha256s),
        "image_count": 4,
        "images": rows,
        "chain_order": list(CHAIN_NAMES),
    }
    (directory / "index.json").write_text(json.dumps(index, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
    return index
