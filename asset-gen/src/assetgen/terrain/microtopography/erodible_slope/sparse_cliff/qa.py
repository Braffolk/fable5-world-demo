"""Outcome-focused QA for typed sparse cliff amplification."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .model import SparseResult, TYPE_NAMES

_FONT = ImageFont.load_default()
_BG = (244, 240, 229)


def _shade(height: np.ndarray, pitch: float, lo: float, hi: float) -> Image.Image:
    gy, gx = np.gradient(height.astype(np.float64), pitch)
    normal = np.stack([-gx, gy, np.ones_like(height)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    light = np.asarray([-0.62, 0.55, 0.56])
    lighting = 0.40 + 0.60 * np.clip((normal @ light + 0.25) / 1.25, 0.0, 1.0)
    t = np.clip((height - lo) / max(hi - lo, 1.0e-8), 0.0, 1.0)
    rgb = np.asarray([58, 82, 47]) + t[..., None] * (np.asarray([190, 178, 138]) - np.asarray([58, 82, 47]))
    return Image.fromarray(np.clip(rgb * lighting[..., None], 0, 255).astype(np.uint8), "RGB")


def _signed(value: np.ndarray, limit: float) -> Image.Image:
    t = np.clip(value / max(limit, 1.0e-8), -1.0, 1.0)
    neutral = np.asarray([239.0, 236.0, 224.0])
    cold, warm = np.asarray([36.0, 90.0, 154.0]), np.asarray([190.0, 61.0, 45.0])
    rgb = np.where((t < 0)[..., None], neutral + (-t[..., None]) * (cold - neutral), neutral + t[..., None] * (warm - neutral))
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")


def _scalar(value: np.ndarray) -> Image.Image:
    t = np.clip(value, 0.0, 1.0)
    return Image.fromarray(np.clip(np.asarray([235, 232, 217]) + t[..., None] * (np.asarray([34, 110, 76]) - np.asarray([235, 232, 217])), 0, 255).astype(np.uint8), "RGB")


def _types(value: np.ndarray) -> Image.Image:
    colors = np.asarray([[232, 228, 215], [213, 132, 53], [157, 54, 45], [35, 91, 153], [127, 103, 59]], dtype=np.uint8)
    return Image.fromarray(colors[value], "RGB")


def _attribution(value: np.ndarray) -> Image.Image:
    valid = value >= 0
    rgb = np.full((*value.shape, 3), [232, 228, 215], dtype=np.uint8)
    code = value.astype(np.int64, copy=False)
    rgb[..., 0][valid] = ((code[valid] * 53 + 41) % 191 + 32).astype(np.uint8)
    rgb[..., 1][valid] = ((code[valid] * 97 + 17) % 191 + 32).astype(np.uint8)
    rgb[..., 2][valid] = ((code[valid] * 193 + 73) % 191 + 32).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def _panel(image: Image.Image, title: str, size: tuple[int, int] = (720, 430)) -> Image.Image:
    result = Image.new("RGB", (size[0], size[1] + 34), _BG)
    result.paste(image.resize(size, Image.Resampling.LANCZOS), (0, 34))
    ImageDraw.Draw(result).text((10, 10), title, fill=(25, 29, 25), font=_FONT)
    return result


def _join(images: list[Image.Image], columns: int = 2) -> Image.Image:
    width, height = max(item.width for item in images), max(item.height for item in images)
    result = Image.new("RGB", (columns * width, ((len(images) + columns - 1) // columns) * height), _BG)
    for index, item in enumerate(images):
        result.paste(item, ((index % columns) * width, (index // columns) * height))
    return result


def _sections(c0: np.ndarray, c1: np.ndarray) -> Image.Image:
    image = Image.new("RGB", (1440, 440), _BG)
    draw = ImageDraw.Draw(image)
    rows = [int(fraction * (c0.shape[0] - 1)) for fraction in (0.30, 0.50, 0.70)]
    values = np.concatenate([c0[rows].ravel(), c1[rows].ravel()])
    lo, hi = np.percentile(values, [1, 99])
    for row, color in zip(rows, ((203, 63, 82), (0, 116, 137), (225, 157, 54))):
        for surface, width in ((c0[row], 1), (c1[row], 2)):
            points = [(48 + col / (len(surface) - 1) * 1370, 398 - np.clip((height - lo) / (hi - lo), 0, 1) * 360) for col, height in enumerate(surface)]
            use = tuple(int(0.45 * channel + 0.55 * 244) for channel in color) if width == 1 else color
            draw.line(points, fill=use, width=width)
    draw.text((48, 414), "faint=C0, solid=C1; fixed rows 0.30 / 0.50 / 0.70", fill=(25, 29, 25), font=_FONT)
    return image


def write_qa(root: Path, c0: np.ndarray, hard: np.ndarray, result: SparseResult, pitch_m: float) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=False)
    valid = ~hard
    lo, hi = np.percentile(c0[valid], [1, 99])
    limit = max(float(np.percentile(np.abs(result.delta_m[valid]), 99)), 0.03)
    inset = (slice(int(0.16 * c0.shape[0]), int(0.84 * c0.shape[0])), slice(int(0.10 * c0.shape[1]), int(0.90 * c0.shape[1])))

    p1 = root / "01-common-light-c0-c1.png"
    _join([
        _panel(_shade(c0, pitch_m, lo, hi), "C0: accepted structural authority, common light"),
        _panel(_shade(result.c1_m, pitch_m, lo, hi), "C1: typed sparse amplification, same light/range"),
        _panel(_shade(c0[inset], pitch_m, lo, hi), "C0 close: shoulder, safe margins, gullies and toes"),
        _panel(_shade(result.c1_m[inset], pitch_m, lo, hi), "C1 close: same crop; source support is 0.5 m"),
    ]).save(p1)

    p2 = root / "02-typed-ownership-and-source-attribution.png"
    _join([
        _panel(_types(result.ownership), "orange=shoulder, red=safe side, blue=incision, brown=toe"),
        _panel(_attribution(result.source_attribution), "nearest selected Biala source center attribution"),
        _panel(_attribution(result.site_attribution), "irregular Wendland-C2 dominant-site ownership"),
        _panel(_scalar(result.network), "preserved ALS structural network condition"),
    ]).save(p2)

    broad = ndimage.gaussian_filter(result.delta_m, 1.0 / pitch_m)
    source_band = result.delta_m - broad
    p3 = root / "03-signed-relief-and-source-bands.png"
    _join([
        _panel(_signed(result.delta_m, limit), f"C1-C0 signed relief +/-{limit:.3f} m"),
        _panel(_signed(broad, max(float(np.percentile(np.abs(broad[valid]), 99)), 0.01)), "0.5-1.0 m supported macro/mid band"),
        _panel(_signed(source_band, max(float(np.percentile(np.abs(source_band[valid]), 99)), 0.01)), "sub-1 m interpolation view; no <0.5 m truth claim"),
        _panel(_scalar(result.condition_strength), "typed condition strength and absolute-domain handoff"),
    ]).save(p3)

    p4 = root / "04-close-sections.png"
    upper = (slice(int(0.18 * c0.shape[0]), int(0.58 * c0.shape[0])), slice(int(0.02 * c0.shape[1]), int(0.55 * c0.shape[1])))
    lower = (slice(int(0.44 * c0.shape[0]), int(0.91 * c0.shape[0])), slice(int(0.34 * c0.shape[1]), int(0.96 * c0.shape[1])))
    _join([
        _panel(_sections(c0, result.c1_m), "fixed whole-domain sections"),
        _panel(_shade(result.c1_m[upper], pitch_m, lo, hi), "C1 close A: crest/side/incision organization"),
        _panel(_shade(result.c1_m[lower], pitch_m, lo, hi), "C1 close B: gully/toe/ordinary-ground handoff"),
        _panel(_signed(result.delta_m[inset], limit), "signed close relief; inspect stops, bands and blobs"),
    ]).save(p4)
    return {
        p1.name: "common-light same-range C0/C1 comparison over the full solve and fixed close crop",
        p2.name: "typed atom ownership, nearest Biala center attribution, irregular site ownership, and ALS network condition",
        p3.name: "signed relief, honest 0.5-1.0 m source band, interpolation-only sub-band, and transition strength",
        p4.name: "fixed sections and two close shaded inspections for crest, side, incision, toe, and ordinary-ground handoff",
    }
