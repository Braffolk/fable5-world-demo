"""Height-only diagnostic rendering for Hinsberger band qualification."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .bands import PhaseBands, RegularGrid


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _diverging(values: np.ndarray, valid: np.ndarray, extent: float) -> Image.Image:
    scaled = np.clip(values / max(extent, 1e-12), -1.0, 1.0)
    magnitude = np.abs(scaled)
    positive = scaled >= 0.0
    rgb = np.full(values.shape + (3,), 245.0, dtype=np.float64)
    positive_color = np.asarray((168.0, 72.0, 36.0))
    negative_color = np.asarray((30.0, 104.0, 132.0))
    color = np.where(positive[..., None], positive_color, negative_color)
    rgb = rgb * (1.0 - magnitude[..., None]) + color * magnitude[..., None]
    yy, xx = np.indices(values.shape)
    checker = ((xx // 8 + yy // 8) & 1).astype(np.uint8)
    rgb[~valid] = np.where(checker[~valid, None] == 0, 18, 36)
    image = Image.fromarray(np.asarray(np.clip(rgb, 0, 255), dtype=np.uint8), "RGB")
    return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def _panel(
    grid: RegularGrid,
    *,
    title: str,
    extent: float,
    size: tuple[int, int],
) -> Image.Image:
    canvas = Image.new("RGB", size, "#efece3")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill="#17201d", font=_font())
    valid_values = grid.values[grid.valid]
    rms = float(np.sqrt(np.mean(valid_values * valid_values))) if valid_values.size else 0.0
    draw.text(
        (8, 20),
        f"valid={valid_values.size:,} rms={rms:.5g} native-z",
        fill="#38433e",
        font=_font(),
    )
    plot = _diverging(grid.values, grid.valid, extent)
    plot.thumbnail((size[0] - 16, size[1] - 46), Image.Resampling.NEAREST)
    canvas.paste(plot, ((size[0] - plot.width) // 2, 40))
    return canvas


def render_height_band_qa(
    destination: Path,
    survey_id: str,
    phases: tuple[PhaseBands, PhaseBands, PhaseBands, PhaseBands],
) -> None:
    b1_values = np.concatenate([phase.b1.values[phase.b1.valid] for phase in phases])
    b2_values = np.concatenate([phase.b2.values[phase.b2.valid] for phase in phases])
    b1_extent = float(np.quantile(np.abs(b1_values), 0.99)) if b1_values.size else 1.0
    b2_extent = float(np.quantile(np.abs(b2_values), 0.99)) if b2_values.size else 1.0
    panel_size = (520, 360)
    header = 62
    canvas = Image.new("RGB", (panel_size[0] * 2, header + panel_size[1] * 4), "#ddd8cc")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), f"Hinsberger {survey_id}: F4/R4 height-band diagnostics", fill="#111916", font=_font())
    draw.text(
        (12, 28),
        "Research-only. DEM pixels only; orthomosaic pixels unread. Direct finite support; no padding.",
        fill="#34423b",
        font=_font(),
    )
    draw.text(
        (12, 44),
        f"Shared display: B1 +/-{b1_extent:.5g}, B2 +/-{b2_extent:.5g} native-z units.",
        fill="#34423b",
        font=_font(),
    )
    for row, phase in enumerate(phases):
        offset = phase.phase_xy_m
        canvas.paste(
            _panel(
                phase.b1,
                title=f"phase {row} offset=({offset[0]:.5g},{offset[1]:.5g})m  B1 0.25-1m",
                extent=b1_extent,
                size=panel_size,
            ),
            (0, header + row * panel_size[1]),
        )
        canvas.paste(
            _panel(
                phase.b2,
                title=f"phase {row} offset=({offset[0]:.5g},{offset[1]:.5g})m  B2 0.125-0.25m",
                extent=b2_extent,
                size=panel_size,
            ),
            (panel_size[0], header + row * panel_size[1]),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)


def qa_index(qa_root: Path, interpretations: dict[str, str]) -> dict[str, Any]:
    images = []
    for path in sorted(qa_root.glob("*.png")):
        encoded = path.read_bytes()
        with Image.open(path) as image:
            dimensions = [image.width, image.height]
        images.append(
            {
                "path": path.name,
                "bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "dimensions_px": dimensions,
                "interpretation": interpretations[path.name],
            }
        )
    return {"schema": "hinsberger-height-band-qa/1", "images": images}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
