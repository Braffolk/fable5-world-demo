"""Four bounded visual diagnostics for the aeolian-dune FLOAT candidate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .conditions import PilotConditions
from .model import FINE_PIXEL_M, SynthesisResult


PAPER = (239, 236, 223)
INK = (30, 36, 31)
SIZE = (1800, 1200)


def _font(size: int):
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _hillshade(height: np.ndarray, pitch: float, limits: tuple[float, float]) -> np.ndarray:
    gy, gx = np.gradient(height.astype(np.float64), pitch)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    azimuth = np.deg2rad(315.0)
    altitude = np.deg2rad(38.0)
    light = (
        np.sin(altitude) * np.cos(slope)
        + np.cos(altitude) * np.sin(slope) * np.cos(azimuth - aspect)
    )
    light = np.clip((light + 0.18) / 1.18, 0.0, 1.0)
    low, high = limits
    elevation = np.clip((height - low) / max(high - low, 1.0e-9), 0.0, 1.0)
    base = np.stack(
        (0.50 + 0.24 * elevation, 0.47 + 0.23 * elevation, 0.32 + 0.16 * elevation),
        axis=-1,
    )
    return np.clip(base * (0.42 + 0.76 * light[..., None]), 0.0, 1.0)


def _signed(values: np.ndarray, limit: float | None = None) -> np.ndarray:
    if limit is None:
        limit = max(float(np.quantile(np.abs(values), 0.985)), 1.0e-8)
    scaled = np.clip(values / limit, -1.0, 1.0)
    neutral = np.asarray((0.94, 0.92, 0.84))
    positive = np.asarray((0.72, 0.17, 0.09))
    negative = np.asarray((0.06, 0.30, 0.62))
    return np.where(
        (scaled >= 0)[..., None],
        neutral + scaled[..., None] * (positive - neutral),
        neutral + (-scaled)[..., None] * (negative - neutral),
    )


def _mask_rgb(conditions: PilotConditions) -> np.ndarray:
    rgb = np.zeros((*conditions.authority_fine.shape, 3), dtype=np.float64)
    rgb[:] = (0.12, 0.14, 0.12)
    rgb[conditions.forest_fine] = (0.20, 0.45, 0.19)
    rgb[conditions.authority_fine] = (0.69, 0.77, 0.34)
    rgb[conditions.hard_exclusion_fine] = (0.73, 0.17, 0.12)
    rgb[conditions.protected_fine] = (0.93, 0.55, 0.12)
    return rgb


def _panel(rgb: np.ndarray, size: tuple[int, int]) -> Image.Image:
    image = Image.fromarray(
        np.asarray(np.clip(rgb, 0.0, 1.0) * 255.0, dtype=np.uint8), "RGB"
    )
    scale = min(size[0] / image.width, size[1] / image.height)
    image = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", size, PAPER)
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def _sheet(title: str, note: str, panels: list[tuple[str, np.ndarray]]) -> Image.Image:
    image = Image.new("RGB", SIZE, PAPER)
    draw = ImageDraw.Draw(image)
    draw.text((36, 22), title, font=_font(27), fill=INK)
    draw.text((36, 61), note, font=_font(15), fill=(67, 73, 65))
    top = 100
    panel_w = (SIZE[0] - 72) // 2
    panel_h = (SIZE[1] - top - 30) // 2
    for index, (label, rgb) in enumerate(panels):
        row, col = divmod(index, 2)
        x = 36 + col * panel_w
        y = top + row * panel_h
        draw.text((x + 8, y + 4), label, font=_font(17), fill=INK)
        image.paste(_panel(rgb, (panel_w - 16, panel_h - 38)), (x + 8, y + 32))
    return image


def write_qa(
    output: Path,
    result: SynthesisResult,
    conditions: PilotConditions,
) -> list[dict]:
    output.mkdir(parents=True, exist_ok=False)
    limits = tuple(np.quantile(result.structural_fine_m, (0.01, 0.99)))
    parent_repeat = np.repeat(np.repeat(result.parent_core_m, 4, axis=0), 4, axis=1)
    residual = result.residual_fine_m.astype(np.float64)
    blur8 = ndimage.gaussian_filter(residual, 8.0 / (2.355 * FINE_PIXEL_M), mode="reflect")
    orientation = np.zeros((*residual.shape, 3), dtype=np.float64)
    gy, gx = np.gradient(ndimage.gaussian_filter(residual, 2.0 / FINE_PIXEL_M))
    theta = (np.arctan2(gy, gx) + np.pi) / (2.0 * np.pi)
    orientation[..., 0] = 0.5 + 0.5 * np.cos(theta * 2.0 * np.pi)
    orientation[..., 1] = 0.5 + 0.5 * np.cos((theta - 1.0 / 3.0) * 2.0 * np.pi)
    orientation[..., 2] = 0.5 + 0.5 * np.cos((theta - 2.0 / 3.0) * 2.0 * np.pi)
    magnitude = np.hypot(gx, gy)
    orientation *= np.clip(magnitude / max(np.quantile(magnitude, 0.98), 1.0e-8), 0.12, 1.0)[..., None]

    images = [
        (
            "01-whole-organization.png",
            _sheet(
                "01 / Aeolian organization at the complete 128 m pilot",
                "World-coordinate coast-aligned exemplar transfer; parent DTM already supplies landform position.",
                [
                    ("accepted 1 m parent / common light", _hillshade(parent_repeat, FINE_PIXEL_M, limits)),
                    ("0.25 m candidate / common light", _hillshade(result.absolute_fine_m, FINE_PIXEL_M, limits)),
                    ("added signed geometry", _signed(residual)),
                    ("structure coarser than 8 m remains parent-only", _signed(blur8)),
                ],
            ),
            "Whole-pilot organization and proof that >8 m placement remains inherited from Estonia DTM.",
        ),
        (
            "02-native-common-light-before-after.png",
            _sheet(
                "02 / Native common-light before and after",
                "Identical light and elevation limits. Bottom row is a 32 m native-scale center crop.",
                [
                    ("structural 0.25 m reconstruction", _hillshade(result.structural_fine_m, FINE_PIXEL_M, limits)),
                    ("exemplar-amplified 0.25 m", _hillshade(result.absolute_fine_m, FINE_PIXEL_M, limits)),
                    ("before / 32 m", _hillshade(result.structural_fine_m[192:320, 192:320], FINE_PIXEL_M, limits)),
                    ("after / 32 m", _hillshade(result.absolute_fine_m[192:320, 192:320], FINE_PIXEL_M, limits)),
                ],
            ),
            "Direct visual comparison under one light and one elevation transfer function.",
        ),
        (
            "03-scale-bands-and-orientation.png",
            _sheet(
                "03 / Supported scale bands and physical orientation",
                "Only >=1 m source structure is admitted; hue panel shows the coherent gradient orientation field.",
                [
                    ("1-2 m", _signed(ndimage.gaussian_filter(residual, 1.0 / (2.355 * FINE_PIXEL_M)) - ndimage.gaussian_filter(residual, 2.0 / (2.355 * FINE_PIXEL_M)))),
                    ("2-4 m", _signed(ndimage.gaussian_filter(residual, 2.0 / (2.355 * FINE_PIXEL_M)) - ndimage.gaussian_filter(residual, 4.0 / (2.355 * FINE_PIXEL_M)))),
                    ("4-8 m", _signed(ndimage.gaussian_filter(residual, 4.0 / (2.355 * FINE_PIXEL_M)) - ndimage.gaussian_filter(residual, 8.0 / (2.355 * FINE_PIXEL_M)))),
                    ("orientation / strength", orientation),
                ],
            ),
            "Band-separated geometry and orientation diagnostic for the authorized 1-8 m range.",
        ),
        (
            "04-masks-and-closure.png",
            _sheet(
                "04 / Authority, exclusions, and exact parent closure",
                "Green = authorized. Red/orange would be hard/protected. Closure uses the project smooth 4x operator.",
                [
                    ("condition authority", _mask_rgb(conditions)),
                    ("absolute parent-mean error", _signed(result.absolute_fine_m.reshape(128, 4, 128, 4).mean(axis=(1, 3)) - result.parent_core_m, 1.0e-9)),
                    ("source exemplar 1 / closed residual", _signed(result.source_residual_examples_m[0])),
                    ("source exemplar 2 / closed residual", _signed(result.source_residual_examples_m[1])),
                ],
            ),
            "Hard/protected authority and numerical closure, plus the two transferred source residual fields.",
        ),
    ]
    records: list[dict] = []
    for filename, image, interpretation in images:
        path = output / filename
        image.save(path, optimize=True)
        records.append({"path": path, "interpretation": interpretation})
    return records


def write_index(path: Path, records: list[dict], identity) -> None:
    payload = {
        "schemaVersion": "laas.sand-dune-aeolian-float-qa/1",
        "items": [
            {
                "path": record["path"].name,
                **identity(record["path"]),
                "interpretation": record["interpretation"],
            }
            for record in records
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
