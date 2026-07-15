"""Four outcome-focused diagnostics for the whole-domain exemplar field."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .model import FieldResult

_FONT = ImageFont.load_default()
_BG = (244, 240, 229)


def _shade(height: np.ndarray, pitch: float, lo: float, hi: float) -> Image.Image:
    gy, gx = np.gradient(height.astype(np.float64), pitch)
    nx, ny, nz = -gx, gy, np.ones_like(height)
    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    light = np.asarray([-0.62, 0.55, 0.56])
    lighting = np.clip((nx * light[0] + ny * light[1] + nz * light[2]) / length, -0.25, 1.0)
    lighting = 0.40 + 0.60 * (lighting + 0.25) / 1.25
    t = np.clip((height - lo) / max(hi - lo, 1.0e-8), 0.0, 1.0)
    low, high = np.asarray([58, 82, 47]), np.asarray([190, 178, 138])
    rgb = low + t[..., None] * (high - low)
    return Image.fromarray(np.clip(rgb * lighting[..., None], 0, 255).astype(np.uint8), "RGB")


def _signed(value: np.ndarray, limit: float) -> Image.Image:
    t = np.clip(value / max(limit, 1.0e-8), -1.0, 1.0)
    neutral = np.asarray([239.0, 236.0, 224.0])
    cold, warm = np.asarray([36.0, 90.0, 154.0]), np.asarray([190.0, 61.0, 45.0])
    rgb = np.where((t < 0)[..., None], neutral + (-t[..., None]) * (cold - neutral), neutral + t[..., None] * (warm - neutral))
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")


def _scalar(value: np.ndarray) -> Image.Image:
    t = np.clip(value, 0.0, 1.0)
    low, high = np.asarray([235, 232, 217]), np.asarray([34, 110, 76])
    return Image.fromarray(np.clip(low + t[..., None] * (high - low), 0, 255).astype(np.uint8), "RGB")


def _types(ownership: np.ndarray) -> Image.Image:
    colors = np.asarray([
        [226, 222, 208], [205, 109, 41], [157, 54, 45], [35, 91, 153], [25, 140, 151], [127, 103, 59]
    ], dtype=np.uint8)
    return Image.fromarray(colors[ownership], "RGB")


def _panel(image: Image.Image, title: str, size: tuple[int, int] = (720, 430)) -> Image.Image:
    result = Image.new("RGB", (size[0], size[1] + 34), _BG)
    result.paste(image.resize(size, Image.Resampling.LANCZOS), (0, 34))
    ImageDraw.Draw(result).text((10, 10), title, fill=(25, 29, 25), font=_FONT)
    return result


def _join(images: list[Image.Image], columns: int = 2) -> Image.Image:
    width, height = max(x.width for x in images), max(x.height for x in images)
    rows = (len(images) + columns - 1) // columns
    result = Image.new("RGB", (columns * width, rows * height), _BG)
    for index, item in enumerate(images):
        result.paste(item, ((index % columns) * width, (index // columns) * height))
    return result


def _sections(c0: np.ndarray, c1: np.ndarray) -> Image.Image:
    width, height = 1440, 440
    image = Image.new("RGB", (width, height), _BG)
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 48, 24, width - 20, height - 42
    rows = [int(round(fraction * (c0.shape[0] - 1))) for fraction in (0.32, 0.50, 0.68)]
    values = np.concatenate([c0[rows].ravel(), c1[rows].ravel()])
    lo, hi = np.percentile(values, [1, 99])
    colors = ((203, 63, 82), (0, 116, 137), (225, 157, 54))
    draw.rectangle((left, top, right, bottom), outline=(65, 65, 60))
    for row, color in zip(rows, colors):
        for surface, faint in ((c0[row], True), (c1[row], False)):
            points = []
            for col, value in enumerate(surface):
                x = left + col / (len(surface) - 1) * (right - left)
                y = bottom - np.clip((value - lo) / max(hi - lo, 1.0e-8), 0, 1) * (bottom - top)
                points.append((x, y))
            use = tuple(int(0.35 * c + 0.65 * 244) for c in color) if faint else color
            draw.line(points, fill=use, width=1 if faint else 2)
    draw.text((left, bottom + 12), "faint=C0; solid=C1; fixed whole-domain row fractions 0.32 / 0.50 / 0.68", fill=(25, 29, 25), font=_FONT)
    return image


def write_qa(root: Path, c0: np.ndarray, hard: np.ndarray, result: FieldResult, pitch_m: float) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=False)
    valid = ~hard
    lo, hi = np.percentile(c0[valid], [1, 99])
    limit = max(float(np.percentile(np.abs(result.delta_m[valid]), 99)), 0.05)

    p1 = root / "01-common-light-c0-c1.png"
    inset = (
        slice(int(round(0.16 * c0.shape[0])), int(round(0.84 * c0.shape[0]))),
        slice(int(round(0.10 * c0.shape[1])), int(round(0.90 * c0.shape[1]))),
    )
    _join([
        _panel(_shade(c0, pitch_m, lo, hi), "C0: accepted structural authority, common light"),
        _panel(_shade(result.c1_m, pitch_m, lo, hi), "C1: whole-domain macro challenger, same light and range"),
        _panel(_shade(c0[inset], pitch_m, lo, hi), "C0: publication interior, same light and range"),
        _panel(_shade(result.c1_m[inset], pitch_m, lo, hi), "C1: publication interior, same light and range"),
    ]).save(p1)
    p2 = root / "02-typed-macro-forms.png"
    _join([
        _panel(_types(result.ownership), "orange=shoulder, red=scarp/headcut, blue=incision, cyan=seep, brown=toe"),
        _panel(_scalar(result.opportunity), "physical opportunity and exact ownership taper"),
        _panel(_scalar(result.network), "preserved ALS structural-network condition"),
        _panel(_signed(result.warped_exemplar_m, max(float(np.percentile(np.abs(result.warped_exemplar_m), 99)), 1.0)), "elastically registered 2D Biala macro field; R0 analogue only"),
    ]).save(p2)
    p3 = root / "03-signed-relief-and-scale-bands.png"
    broad = ndimage.gaussian_filter(result.delta_m, 8.0 / pitch_m)
    medium = ndimage.gaussian_filter(result.delta_m, 2.0 / pitch_m) - broad
    local = result.delta_m - ndimage.gaussian_filter(result.delta_m, 2.0 / pitch_m)
    _join([
        _panel(_signed(result.delta_m, limit), f"C1-C0 signed relief, +/-{limit:.3f} m"),
        _panel(_signed(broad, max(float(np.percentile(np.abs(broad[valid]), 99)), 0.02)), "broad form >8 m"),
        _panel(_signed(medium, max(float(np.percentile(np.abs(medium[valid]), 99)), 0.02)), "connected macro band 2-8 m"),
        _panel(_signed(local, max(float(np.percentile(np.abs(local[valid]), 99)), 0.01)), "local transition band <2 m; no fine-detail claim"),
    ]).save(p3)
    p4 = root / "04-fixed-sections-and-budget.png"
    budget = Image.new("RGB", (720, 464), _BG)
    draw = ImageDraw.Draw(budget)
    lines = [
        "SIGNED SOLID BUDGET",
        f"erosion: {float(np.sum(result.erosion_m) * pitch_m * pitch_m):.6f} m3",
        f"deposition: {float(np.sum(result.deposition_m) * pitch_m * pitch_m):.6f} m3",
        f"net: {float(np.sum(result.delta_m) * pitch_m * pitch_m):+.9f} m3",
        f"registration orientation: {result.source_orientation}",
        f"registration displacement p95: {result.registration_displacement_p95_m:.3f} m",
        "Biala supplies R0 2D capacity only; no production-transfer claim.",
    ]
    for index, line in enumerate(lines):
        draw.text((28, 30 + index * 42), line, fill=(25, 29, 25), font=_FONT)
    _join([_panel(_sections(c0, result.c1_m), "fixed sections"), budget]).save(p4)
    return {
        p1.name: "same-light, same-range whole-domain comparison of accepted C0 and generated C1 geometry",
        p2.name: "typed macro-form ownership, physical opportunity, ALS condition, and the single deformed 2D exemplar field",
        p3.name: "signed C1 relief separated into directly inspectable spatial bands",
        p4.name: "fixed whole-domain sections and the conservative signed-solid budget",
    }
