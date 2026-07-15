"""Four compact diagnostics for the connected ALS structural reference."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .evidence import StructuralInputs
from .reference import StructuralReference

_FONT = ImageFont.load_default()
_BACKGROUND = (244, 240, 229)


def _shade(height: np.ndarray, pitch: float, vmin: float, vmax: float) -> Image.Image:
    gy, gx = np.gradient(height.astype(np.float64), pitch)
    nx, ny, nz = -gx, gy, np.ones_like(height)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    light = np.asarray([-0.62, 0.55, 0.56])
    lighting = np.clip((nx * light[0] + ny * light[1] + nz * light[2]) / norm, -0.25, 1.0)
    lighting = 0.42 + 0.58 * (lighting + 0.25) / 1.25
    t = np.clip((height - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    low = np.asarray([58.0, 82.0, 47.0])
    high = np.asarray([190.0, 178.0, 138.0])
    color = low[None, None, :] * (1.0 - t[..., None]) + high[None, None, :] * t[..., None]
    return Image.fromarray(np.clip(color * lighting[..., None], 0, 255).astype(np.uint8), "RGB")


def _scalar(value: np.ndarray, low: tuple[int, int, int], high: tuple[int, int, int], limit: float) -> Image.Image:
    t = np.clip(np.nan_to_num(value, nan=0.0) / max(limit, 1e-8), 0.0, 1.0)
    a, b = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    color = a + t[..., None] * (b - a)
    return Image.fromarray(np.clip(color, 0, 255).astype(np.uint8), "RGB")


def _residual(value: np.ndarray, limit: float) -> Image.Image:
    t = np.clip(value / max(limit, 1e-8), -1.0, 1.0)
    neutral = np.asarray([239.0, 236.0, 224.0])
    cold = np.asarray([39.0, 92.0, 152.0])
    warm = np.asarray([183.0, 53.0, 47.0])
    color = np.where(
        (t < 0.0)[..., None],
        neutral + (-t[..., None]) * (cold - neutral),
        neutral + t[..., None] * (warm - neutral),
    )
    return Image.fromarray(np.clip(color, 0, 255).astype(np.uint8), "RGB")


def _panel(image: Image.Image, title: str, size: tuple[int, int] = (720, 430)) -> Image.Image:
    canvas = Image.new("RGB", (size[0], size[1] + 34), _BACKGROUND)
    canvas.paste(image.resize(size, Image.Resampling.LANCZOS), (0, 34))
    ImageDraw.Draw(canvas).text((10, 10), title, fill=(25, 29, 25), font=_FONT)
    return canvas


def _join(images: list[Image.Image], columns: int = 2) -> Image.Image:
    rows = (len(images) + columns - 1) // columns
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    result = Image.new("RGB", (columns * width, rows * height), _BACKGROUND)
    for index, image in enumerate(images):
        result.paste(image, ((index % columns) * width, (index // columns) * height))
    return result


def _network_map(reference: StructuralReference) -> Image.Image:
    image = np.full((*reference.physical_domain.shape, 3), (224, 220, 207), dtype=np.uint8)
    image[reference.physical_domain] = (202, 214, 190)
    image[reference.break_network] = (218, 141, 45)
    image[reference.ridge_network] = (184, 56, 48)
    image[reference.valley_network] = (43, 103, 159)
    return Image.fromarray(image, "RGB")


def _ownership(inputs: StructuralInputs) -> Image.Image:
    image = np.full((*inputs.hard_solve.shape, 3), (77, 137, 96), dtype=np.uint8)
    image[inputs.hard_solve] = (181, 65, 44)
    image[inputs.mapped_face_solve] = (20, 22, 22)
    return Image.fromarray(image, "RGB")


def _sections(inputs: StructuralInputs, after: np.ndarray) -> Image.Image:
    width, height = 1440, 440
    image = Image.new("RGB", (width, height), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 50, 24, width - 20, height - 42
    rows = [
        int(round((inputs.bbox_en[3] - northing) / inputs.solve_pitch_m))
        for northing in (6444436.0, 6444452.0, 6444472.0)
    ]
    start = int(round((680432.0 - inputs.bbox_en[0]) / inputs.solve_pitch_m))
    stop = int(round((680720.0 - inputs.bbox_en[0]) / inputs.solve_pitch_m)) + 1
    values = np.concatenate([
        inputs.c0_solve_m[rows, start:stop].ravel(), after[rows, start:stop].ravel()
    ])
    vmin, vmax = np.percentile(values, [1, 99])
    colors = ((209, 73, 91), (0, 121, 140), (237, 174, 73))
    draw.rectangle((left, top, right, bottom), outline=(70, 70, 65))
    for row, color in zip(rows, colors):
        for series, alpha in ((inputs.c0_solve_m[row, start:stop], 0.38), (after[row, start:stop], 1.0)):
            points = []
            for index, value in enumerate(series):
                x = left + index / max(len(series) - 1, 1) * (right - left)
                y = bottom - np.clip((value - vmin) / max(vmax - vmin, 1e-8), 0, 1) * (bottom - top)
                points.append((x, y))
            shade = tuple(int(component * alpha + 244 * (1.0 - alpha)) for component in color)
            draw.line(points, fill=shade, width=2 if alpha == 1.0 else 1)
    draw.text((left, bottom + 12), "faint=C0; solid=structural result; N 6444436 / 4452 / 4472", fill=(25, 29, 25), font=_FONT)
    return image


def write_qa(
    root: Path,
    inputs: StructuralInputs,
    reference: StructuralReference,
    residual: np.ndarray,
) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=False)
    after = inputs.c0_solve_m + residual
    vmin, vmax = np.percentile(inputs.c0_solve_m[~inputs.hard_solve], [1, 99])

    p1 = root / "01-c0-and-hard-ownership.png"
    _join([
        _panel(_shade(inputs.c0_solve_m, inputs.solve_pitch_m, vmin, vmax), "Accepted C0, common light"),
        _panel(_ownership(inputs), "green=eligible, red=hard/water/object, black=mapped face"),
    ]).save(p1)

    density_limit = max(float(np.percentile(reference.fine_fit.count, 99)), 1.0)
    spread_limit = max(float(np.percentile(reference.fine_fit.detrended_spread_m[np.isfinite(reference.fine_fit.detrended_spread_m)], 95)), 0.05)
    p2 = root / "02-als-reference-confidence.png"
    _join([
        _panel(_scalar(reference.fine_fit.count, (235, 232, 217), (29, 101, 74), density_limit), "3 m MLS qualified-point density"),
        _panel(_scalar(reference.fine_fit.detrended_spread_m, (234, 236, 221), (183, 53, 47), spread_limit), "detrended vertical spread"),
        _panel(_scalar(reference.fine_fit.last_return_fraction, (235, 232, 217), (32, 109, 155), 1.0), "last-return fraction"),
        _panel(_scalar(reference.confidence, (235, 232, 217), (42, 123, 85), 1.0), "combined reference confidence and smooth transition"),
    ]).save(p2)

    p3 = root / "03-connected-structural-network.png"
    _join([
        _panel(_network_map(reference), "green=domain, orange=break, red=ridge, blue=valley"),
        _panel(_scalar(reference.network_strength, (235, 232, 217), (122, 46, 125), 1.0), "connected multiscale structural authority"),
        _panel(_residual(reference.residual_target_m, max(float(np.percentile(np.abs(reference.residual_target_m), 99)), 0.05)), "robust ALS reference minus C0"),
        _panel(_shade(reference.reference_surface_m, inputs.reference_pitch_m, vmin, vmax), "reference surface, common light"),
    ]).save(p3)

    limit = max(float(np.percentile(np.abs(residual[~inputs.hard_solve]), 99)), 0.05)
    p4 = root / "04-before-after-final-projection.png"
    _join([
        _panel(_shade(inputs.c0_solve_m, inputs.solve_pitch_m, vmin, vmax), "Before: accepted C0"),
        _panel(_shade(after, inputs.solve_pitch_m, vmin, vmax), "After: connected ALS structural input"),
        _panel(_residual(residual, limit), f"final projected residual, +/-{limit:.3f} m"),
        _panel(_sections(inputs, after), "fixed cross-sections", (720, 430)),
    ]).save(p4)

    return {
        p1.name: "accepted C0 and the exact hard, water, object, protected, and mapped non-heightfield exclusions",
        p2.name: "local ALS support and quality fields that determine structural-reference confidence",
        p3.name: "connected multiscale break, ridge, and valley authority plus the robust ALS reference",
        p4.name: "common-light before/after, signed final correction, and fixed cross-sections",
    }
