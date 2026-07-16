"""Four outcome-focused QA sheets for the whole-mire form graph."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .model import Result


PAPER = (240, 237, 224)
INK = (28, 35, 30)
WIDTH, HEIGHT = 1800, 1380


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
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
    azimuth, altitude = np.deg2rad(315.0), np.deg2rad(38.0)
    light = (
        np.sin(altitude) * np.cos(slope)
        + np.cos(altitude) * np.sin(slope) * np.cos(azimuth - aspect)
    )
    light = np.clip((light + 0.12) / 1.12, 0.0, 1.0)
    low, high = limits
    elevation = np.clip((height - low) / max(high - low, 1.0e-9), 0.0, 1.0)
    base = np.stack(
        (0.32 + 0.25 * elevation, 0.40 + 0.30 * elevation, 0.23 + 0.18 * elevation),
        axis=-1,
    )
    return np.clip(base * (0.48 + 0.72 * light[..., None]), 0.0, 1.0)


def _signed(value: np.ndarray, limit: float) -> np.ndarray:
    unit = np.clip(value / max(limit, 1.0e-9), -1.0, 1.0)
    neutral = np.asarray([0.94, 0.92, 0.84])
    high = np.asarray([0.69, 0.16, 0.08])
    low = np.asarray([0.06, 0.31, 0.63])
    return np.where(
        (unit >= 0.0)[..., None],
        neutral + unit[..., None] * (high - neutral),
        neutral + (-unit)[..., None] * (low - neutral),
    )


def _scalar(value: np.ndarray, mask: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    finite = mask & np.isfinite(value)
    low, high = np.percentile(value[finite], [2.0, 98.0])
    unit = np.clip((value - low) / max(float(high - low), 1.0e-9), 0.0, 1.0)
    paper, target = np.asarray([0.94, 0.92, 0.84]), np.asarray(color)
    rgb = paper + unit[..., None] * (target - paper)
    rgb[~mask] *= 0.22
    return rgb


def _classes(classes: np.ndarray, authority: np.ndarray) -> np.ndarray:
    palette = np.asarray(
        [
            [0.72, 0.68, 0.48],  # lawn
            [0.15, 0.39, 0.62],  # flark
            [0.48, 0.31, 0.13],  # string
            [0.08, 0.29, 0.57],  # hollow
            [0.63, 0.25, 0.10],  # hummock
            [0.78, 0.55, 0.20],  # pool margin
            [0.03, 0.20, 0.42],  # pool
        ]
    )
    rgb = palette[classes]
    rgb[~authority] *= 0.20
    return rgb


def _panel(rgb: np.ndarray, size: tuple[int, int]) -> Image.Image:
    image = Image.fromarray(np.asarray(np.clip(rgb, 0.0, 1.0) * 255, dtype=np.uint8), "RGB")
    scale = min(size[0] / image.width, size[1] / image.height)
    image = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", size, PAPER)
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def _sheet(title: str, note: str, panels: list[tuple[str, np.ndarray]]) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
    draw = ImageDraw.Draw(image)
    draw.text((36, 22), title, font=_font(28), fill=INK)
    draw.text((36, 62), note, font=_font(16), fill=(70, 76, 67))
    top, panel_w = 104, (WIDTH - 72) // 2
    panel_h = (HEIGHT - top - 28) // 2
    for index, (label, rgb) in enumerate(panels):
        row, col = divmod(index, 2)
        x, y = 36 + col * panel_w, top + row * panel_h
        draw.text((x + 8, y + 4), label, font=_font(17), fill=INK)
        image.paste(_panel(rgb, (panel_w - 16, panel_h - 40)), (x + 8, y + 32))
    return image


def _identity(path: Path) -> dict[str, str | int]:
    payload = path.read_bytes()
    return {"file": path.name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def write_qa(
    directory: Path,
    result: Result,
    authority: np.ndarray,
    hard: np.ndarray,
    water: np.ndarray,
    halo_rows: slice,
    halo_cols: slice,
    core_rows: slice,
    core_cols: slice,
    metrics: dict,
    bindings: dict,
) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    limit = float(metrics["relief"]["parent_abs_max_m"])
    authority_rgb = np.zeros((*authority.shape, 3), dtype=np.float64)
    authority_rgb[:] = (0.14, 0.18, 0.13)
    authority_rgb[authority] = (0.60, 0.70, 0.32)
    authority_rgb[hard] = (0.78, 0.12, 0.08)
    authority_rgb[water] = (0.08, 0.30, 0.64)

    hr = slice(halo_rows.start * 4, halo_rows.stop * 4)
    hc = slice(halo_cols.start * 4, halo_cols.stop * 4)
    cr = slice(core_rows.start * 4, core_rows.stop * 4)
    cc = slice(core_cols.start * 4, core_cols.stop * 4)
    halo_base = result.measured_fine_m[hr, hc]
    halo_final = result.final_fine_m[hr, hc]
    halo_delta = result.fine_delta_m[hr, hc]
    halo_limits = tuple(float(v) for v in np.percentile(halo_base, [2.0, 98.0]))
    core_final = result.final_fine_m[cr, cc]
    core_delta = result.fine_delta_m[cr, cc]
    core_limits = tuple(float(v) for v in np.percentile(result.measured_fine_m[cr, cc], [2.0, 98.0]))

    closure = result.final_fine_m.reshape(
        authority.shape[0], 4, authority.shape[1], 4
    ).mean(axis=(1, 3), dtype=np.float64) - result.synthesized_parent_m
    broad = ndimage.gaussian_filter(result.parent_delta_m, 16.0)
    middle = ndimage.gaussian_filter(result.parent_delta_m, 2.0) - broad
    native = result.parent_delta_m - ndimage.gaussian_filter(result.parent_delta_m, 2.0)

    images = [
        (
            "01-whole-mire-form-organization.png",
            _sheet(
                "01 / Whole-mire form organization",
                "Domain-scale travel coordinate drives contour-parallel strings/flarks; measured depressions own pools.",
                [
                    ("crest travel distance / complete mire", _scalar(result.travel_distance_m, authority, (0.12, 0.32, 0.68))),
                    ("lawn / flark / string / hollow / hummock / margin / pool", _classes(result.form_class, authority)),
                    ("synthesized 1 m parent relief / signed", _signed(result.parent_delta_m, limit)),
                    ("authority green; hard red; water blue", authority_rgb),
                ],
            ),
        ),
        (
            "02-native-morphology-common-light.png",
            _sheet(
                "02 / Native morphology under common light",
                "Same light and elevation range; morphology is solved over the complete mire before this halo/core crop.",
                [
                    ("measured structural halo", _hillshade(halo_base, 0.25, halo_limits)),
                    ("form-graph halo", _hillshade(halo_final, 0.25, halo_limits)),
                    ("form-graph core at 0.25 m", _hillshade(core_final, 0.25, core_limits)),
                    ("core signed change / fixed parent limit", _signed(core_delta, limit)),
                ],
            ),
        ),
        (
            "03-transitions-exclusions-and-closure.png",
            _sheet(
                "03 / Support transitions, exclusions, and hierarchy closure",
                f"Outside-authority changed fine cells={metrics['integrity']['outside_changed_fine_cells']}; max 1 m closure={metrics['integrity']['closure_max_abs_m']:.3e} m.",
                [
                    ("authority / hard / water", authority_rgb),
                    ("whole-mire signed parent change", _signed(result.parent_delta_m, limit)),
                    ("absolute closure error", _scalar(np.abs(closure), np.ones_like(authority), (0.70, 0.14, 0.08))),
                    ("halo signed fine change", _signed(halo_delta, limit)),
                ],
            ),
        ),
        (
            "04-scale-bands-and-form-strength.png",
            _sheet(
                "04 / Scale bands and typed form strength",
                "Broad organization, form-scale relief, and native irregularity are separated; no independent noise layer exists.",
                [
                    (">16 m organization", _signed(broad, max(float(np.max(np.abs(broad))), 1e-9))),
                    ("2-16 m forms", _signed(middle, max(float(np.max(np.abs(middle))), 1e-9))),
                    ("<2 m measured-conditioned native relief", _signed(native, max(float(np.max(np.abs(native))), 1e-9))),
                    ("red hummock/string; blue hollow/flark/pool", _signed(result.hummock_strength + result.string_strength - result.hollow_strength - result.flark_strength - result.pool_strength, 1.5)),
                ],
            ),
        ),
    ]
    for filename, image in images:
        image.save(directory / filename, optimize=True)
    index = {
        "schema_version": "laas.microtopography-qa-index/1",
        "bindings": bindings,
        "interpretation": {
            "01": "whole-mire graph topology and exclusions",
            "02": "common-light visual morphology at native output pitch",
            "03": "support-boundary and hierarchy integrity",
            "04": "broad, form, and native-scale decomposition",
        },
        "images": [_identity(directory / filename) for filename, _ in images],
    }
    (directory / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
