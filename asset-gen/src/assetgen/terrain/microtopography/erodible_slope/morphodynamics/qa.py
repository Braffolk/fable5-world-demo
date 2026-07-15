"""Deterministic result-facing QA renders for immutable materialization."""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .solver import MorphodynamicsResult


@dataclass(frozen=True)
class QaImage:
    relative_path: str
    interpretation: str
    payload: bytes


def _signed(values: np.ndarray) -> Image.Image:
    limit = max(float(np.quantile(np.abs(values), 0.995)), 1e-12)
    scaled = np.clip(values / limit, -1.0, 1.0)
    rgb = np.stack(
        (
            238 - 100 * np.maximum(-scaled, 0.0),
            238 - 125 * np.abs(scaled),
            238 - 85 * np.maximum(scaled, 0.0),
        ),
        axis=-1,
    )
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


def _scalar(values: np.ndarray) -> Image.Image:
    finite = np.asarray(values, dtype=np.float64)
    low, high = np.quantile(finite, (0.01, 0.995))
    if not high > low:
        high = low + 1.0
    value = np.clip((finite - low) / (high - low), 0.0, 1.0)
    red = 24 + 224 * np.power(value, 0.72)
    green = 38 + 184 * np.sin(np.pi * np.clip(value, 0.0, 1.0))
    blue = 55 + 176 * np.power(1.0 - value, 0.82)
    return Image.fromarray(
        np.stack((red, green, blue), axis=-1).astype(np.uint8), mode="RGB"
    )


def _hillshade(height: np.ndarray, texel_m: float) -> Image.Image:
    south, east = np.gradient(height.astype(np.float64), texel_m)
    nx, ny, nz = -east, south, np.ones_like(height)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    shade = np.clip((-0.45 * nx - 0.55 * ny + 0.70 * nz) / norm, 0.0, 1.0)
    gray = (30 + 220 * shade).astype(np.uint8)
    return Image.fromarray(np.repeat(gray[..., None], 3, axis=2), mode="RGB")


def _panel(title: str, image: Image.Image, size: tuple[int, int]) -> Image.Image:
    panel = Image.new("RGB", (size[0], size[1] + 38), "#eee9dc")
    ImageDraw.Draw(panel).text((8, 9), title, fill="#17241c")
    panel.paste(image.resize(size, Image.Resampling.BILINEAR), (0, 38))
    return panel


def _png(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG", compress_level=9, optimize=False)
    return output.getvalue()


def _chunk_relief(
    result: MorphodynamicsResult,
    index: int,
    name: str,
) -> QaImage:
    rows, cols = result.authority.recipe.output_windows[index]
    c0 = result.authority.c0_node_m[rows, cols]
    c1 = result.c1_node_m[rows, cols]
    size = (640, 420)
    canvas = Image.new("RGB", (size[0] * 3, size[1] + 38), "#eee9dc")
    panels = (
        _panel("accepted C0 hillshade", _hillshade(c0, 0.0625), size),
        _panel("authoritative C1 hillshade", _hillshade(c1, 0.0625), size),
        _panel("signed C1 - C0", _signed(c1 - c0), size),
    )
    for panel_index, panel in enumerate(panels):
        canvas.paste(panel, (panel_index * size[0], 0))
    return QaImage(
        relative_path=f"qa/{index + 1:02d}_{name}_c0_c1_relief.png",
        interpretation=(
            f"{name} exact output-node window: accepted C0, derived C1, and signed relief"
        ),
        payload=_png(canvas),
    )


def _process_fields(result: MorphodynamicsResult) -> QaImage:
    node_rows = slice(512, 2561)
    node_cols = slice(512, 4609)
    cells = (
        slice(node_rows.start, node_rows.stop - 1),
        slice(node_cols.start, node_cols.stop - 1),
    )
    fields = (
        ("water discharge m3/s", result.hydrology.discharge_m3_s[cells]),
        ("shear Pa", result.hydrology.shear_pa[cells]),
        ("eroded substrate kg", result.state.eroded_substrate_kg[cells]),
        ("deposited colluvium kg", result.state.deposited_colluvium_kg[cells]),
        ("channel maturity", result.state.maturity[cells]),
        ("material eta", result.material.eta[cells]),
    )
    size = (520, 320)
    canvas = Image.new("RGB", (size[0] * 3, (size[1] + 38) * 2), "#eee9dc")
    for index, (title, values) in enumerate(fields):
        panel = _panel(title, _scalar(values), size)
        canvas.paste(panel, ((index % 3) * size[0], (index // 3) * (size[1] + 38)))
    return QaImage(
        relative_path="qa/03_stitched_process_fields.png",
        interpretation=(
            "stitched output-cell process fields: discharge, shear, extensive erosion and "
            "deposition, channel maturity, and world-anchored material heterogeneity"
        ),
        payload=_png(canvas),
    )


def render_diagnostics(result: MorphodynamicsResult) -> tuple[QaImage, ...]:
    """Render both exact output chunks and the stitched physical process state."""
    return (
        _chunk_relief(result, 0, "west"),
        _chunk_relief(result, 1, "east"),
        _process_fields(result),
    )
