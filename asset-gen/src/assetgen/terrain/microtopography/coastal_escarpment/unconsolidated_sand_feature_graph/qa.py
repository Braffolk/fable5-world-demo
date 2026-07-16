"""Exactly four FLOAT-review images for the bounded graph reconstruction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .evidence import Evidence
from .model import FEATURE_NAMES, Result


WIDTH = 1800
HEIGHT = 1400
PAPER = (242, 238, 225)
INK = (32, 38, 34)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _hillshade(height: np.ndarray, *, azimuth: float = 315.0, altitude: float = 38.0) -> np.ndarray:
    gy, gx = np.gradient(height.astype(np.float64), 0.25)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az = np.deg2rad(azimuth)
    alt = np.deg2rad(altitude)
    light = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    light = np.clip((light + 0.18) / 1.18, 0.0, 1.0)
    lo, hi = np.percentile(height, [2.0, 98.0])
    elevation = np.clip((height - lo) / max(float(hi - lo), 1.0e-6), 0.0, 1.0)
    rgb = np.stack((0.43 + 0.35 * elevation, 0.45 + 0.28 * elevation, 0.34 + 0.21 * elevation), axis=-1)
    return np.clip(rgb * (0.48 + 0.72 * light[..., None]), 0.0, 1.0)


def _signed(value: np.ndarray, limit: float | None = None) -> np.ndarray:
    finite = np.isfinite(value)
    if limit is None:
        limit = max(float(np.percentile(np.abs(value[finite]), 98.0)), 1.0e-6)
    scaled = np.clip(value / limit, -1.0, 1.0)
    neutral = np.asarray([0.94, 0.92, 0.85])
    red = np.asarray([0.76, 0.16, 0.10])
    blue = np.asarray([0.06, 0.28, 0.57])
    rgb = np.where((scaled >= 0)[..., None], neutral + scaled[..., None] * (red - neutral), neutral + (-scaled)[..., None] * (blue - neutral))
    return np.clip(rgb, 0.0, 1.0)


def _scalar(value: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    valid = np.isfinite(value)
    lo, hi = np.percentile(value[valid], [2.0, 98.0])
    t = np.clip((value - lo) / max(float(hi - lo), 1.0e-9), 0.0, 1.0)
    base = np.asarray([0.94, 0.92, 0.85])
    target = np.asarray(color)
    return base + t[..., None] * (target - base)


def _as_image(rgb: np.ndarray, size: tuple[int, int]) -> Image.Image:
    image = Image.fromarray(np.asarray(np.clip(rgb, 0.0, 1.0) * 255.0, dtype=np.uint8), "RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, PAPER)
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def _sheet(title: str, panels: list[tuple[str, np.ndarray]], note: str) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
    draw = ImageDraw.Draw(image)
    draw.text((36, 22), title, font=_font(27), fill=INK)
    draw.text((36, 58), note, font=_font(16), fill=(73, 77, 68))
    top = 96
    cols = 2
    rows = (len(panels) + 1) // 2
    panel_w = (WIDTH - 72) // cols
    panel_h = (HEIGHT - top - 34) // rows
    for index, (label, rgb) in enumerate(panels):
        row, col = divmod(index, cols)
        x = 36 + col * panel_w
        y = top + row * panel_h
        draw.text((x + 8, y + 5), label, font=_font(17), fill=INK)
        rendered = _as_image(rgb, (panel_w - 16, panel_h - 42))
        image.paste(rendered, (x + 8, y + 34))
    return image


def _ownership(evidence: Evidence, result: Result) -> np.ndarray:
    base = _hillshade(evidence.c0_solve_m)
    palette = np.asarray([
        [0.0, 0.0, 0.0], [0.91, 0.34, 0.13], [0.92, 0.69, 0.13],
        [0.24, 0.67, 0.68], [0.08, 0.34, 0.68], [0.36, 0.61, 0.24],
    ])
    labels = result.feature_labels
    overlay = palette[labels]
    alpha = (labels > 0)[..., None] * 0.72
    rgb = base * (1.0 - alpha) + overlay * alpha
    rgb[evidence.hard_solve] *= 0.32
    return rgb


def _closeup(value: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.nonzero(labels > 0)
    center_row = int(np.median(rows)) if len(rows) else value.shape[0] // 2
    center_col = int(np.median(cols)) if len(cols) else value.shape[1] // 2
    half = 120
    r0, r1 = max(0, center_row - half), min(value.shape[0], center_row + half)
    c0, c1 = max(0, center_col - half), min(value.shape[1], center_col + half)
    return value[r0:r1, c0:c1], labels[r0:r1, c0:c1]


def write_qa(directory: Path, evidence: Evidence, result: Result, recipe_sha256: str, source_sha256s: list[str]) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    graph_legend = np.zeros((*result.feature_labels.shape, 3), dtype=np.float64)
    palette = np.asarray([[0.93, 0.91, 0.84], [0.91, 0.34, 0.13], [0.92, 0.69, 0.13], [0.24, 0.67, 0.68], [0.08, 0.34, 0.68], [0.36, 0.61, 0.24]])
    graph_legend[:] = palette[result.feature_labels]
    graph_legend[evidence.hard_solve] *= 0.35
    images = [
        (
            "01-overview-ownership.png",
            _sheet(
                "01 / Exact ownership and ordered measured feature graph",
                [
                    ("C0 common light; hard ownership dark", np.where(evidence.hard_solve[..., None], _hillshade(evidence.c0_solve_m) * 0.3, _hillshade(evidence.c0_solve_m))),
                    ("graph on terrain", _ownership(evidence, result)),
                    ("crest / upper break / lower bench / rill / toe", graph_legend),
                    ("measured ALS graph confidence", _scalar(result.measured_confidence, (0.08, 0.42, 0.29))),
                ],
                "EPSG:3301 [680448,6444416,680576,6444544]; 40 Liiv / 40 Jaajarvesetted only.",
            ),
            "Exact canvas, hard ownership, and ordered ALS-derived feature classes.",
        ),
        (
            "02-common-light-c0-c1-macro.png",
            _sheet(
                "02 / Common-light macro comparison",
                [
                    ("accepted C0", _hillshade(evidence.c0_solve_m)),
                    ("C1 graph solve + conservative transport", _hillshade(result.solve_absolute_m)),
                    ("signed C1-C0 macro", _signed(result.solve_delta_m)),
                    ("absolute change", _scalar(np.abs(result.solve_delta_m), (0.65, 0.24, 0.10))),
                ],
                "Same azimuth, altitude, elevation palette, and scale. No packing or runtime rendering.",
            ),
            "Common-light C0/C1 macro verdict image.",
        ),
        (
            "03-crest-face-toe-closeup.png",
            None,
            "Close view of the connected crest-face-rill-toe system.",
        ),
        (
            "04-signed-macro-micro-bands.png",
            _sheet(
                "04 / Signed macro, erosion/deposition, and generated continuation",
                [
                    ("screened-biharmonic + process macro", _signed(result.solve_delta_m)),
                    ("erosion blue / conservative deposition red", _signed(result.deposition_m - result.erosion_m)),
                    ("0.0625-0.5 m generated band; not measured truth", _signed(result.micro_m)),
                    ("feature strength controlling all continuation", _scalar(result.feature_strength, (0.10, 0.36, 0.67))),
                ],
                "Biala calibrates normalized profile/curvature and 0.25-1.25 m interpolated-lattice statistics; evidence floor remains 0.25 m.",
            ),
            "Signed macro and subordinate generated micro-continuation bands.",
        ),
    ]
    close_c0, close_labels = _closeup(evidence.c0_solve_m, result.feature_labels)
    close_c1, _ = _closeup(result.solve_absolute_m, result.feature_labels)
    close_delta, _ = _closeup(result.solve_delta_m, result.feature_labels)
    close_graph, _ = _closeup(_ownership(evidence, result), result.feature_labels)
    images[2] = (
        images[2][0],
        _sheet(
            "03 / Crest-face-rill-toe closeup",
            [
                ("C0", _hillshade(close_c0)),
                ("C1", _hillshade(close_c1)),
                ("ordered graph", close_graph),
                ("signed local change", _signed(close_delta)),
            ],
            "Crop is selected from measured graph ownership, not a hand-authored scene coordinate.",
        ),
        images[2][2],
    )

    rows = []
    for filename, image, interpretation in images:
        path = directory / filename
        assert image is not None
        image.save(path, optimize=True)
        payload = path.read_bytes()
        rows.append({
            "path": filename,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "dimensions_px": [WIDTH, HEIGHT],
            "interpretation": interpretation,
        })
    index = {
        "schema_version": "laas.unconsolidated-sand-feature-graph-qa/1",
        "recipe_sha256": recipe_sha256,
        "source_sha256s": sorted(source_sha256s),
        "image_count": 4,
        "images": rows,
        "feature_legend": {str(i + 1): name for i, name in enumerate(FEATURE_NAMES)},
    }
    (directory / "index.json").write_text(json.dumps(index, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
    return index

