"""Four bounded Pillow diagnostics for the ALS/TGV research result."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .evidence import AlsTgvEvidence

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


def _residual_image(value: np.ndarray, limit: float) -> Image.Image:
    t = np.clip(value / max(limit, 1e-6), -1.0, 1.0)
    neutral = np.asarray([239.0, 236.0, 224.0])
    cold = np.asarray([39.0, 92.0, 152.0])
    warm = np.asarray([183.0, 53.0, 47.0])
    color = np.where(
        (t < 0.0)[..., None],
        neutral + (-t[..., None]) * (cold - neutral),
        neutral + t[..., None] * (warm - neutral),
    )
    return Image.fromarray(np.clip(color, 0, 255).astype(np.uint8), "RGB")


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image.resize(size, Image.Resampling.LANCZOS)


def _panel(image: Image.Image, title: str, size: tuple[int, int] = (900, 540)) -> Image.Image:
    canvas = Image.new("RGB", (size[0], size[1] + 36), _BACKGROUND)
    canvas.paste(_fit(image, size), (0, 36))
    ImageDraw.Draw(canvas).text((12, 12), title, fill=(25, 29, 25), font=_FONT)
    return canvas


def _join(images: list[Image.Image], columns: int) -> Image.Image:
    rows = (len(images) + columns - 1) // columns
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    result = Image.new("RGB", (columns * width, rows * height), _BACKGROUND)
    for index, image in enumerate(images):
        result.paste(image, ((index % columns) * width, (index // columns) * height))
    return result


def _support_map(evidence: AlsTgvEvidence) -> Image.Image:
    width, height = 1200, 720
    image = Image.new("RGB", (width, height), (220, 217, 205))
    draw = ImageDraw.Draw(image)
    e0, n0, e1, n1 = evidence.bbox_en
    residual = np.clip(evidence.point_residual_m / 0.4, -1.0, 1.0)
    for x, y, value, hold in zip(evidence.point_x, evidence.point_y, residual, evidence.point_holdout):
        col = int((x - e0) / (e1 - e0) * (width - 1))
        row = int((n1 - y) / (n1 - n0) * (height - 1))
        color = (int(80 + 150 * max(value, 0)), 75, int(80 + 150 * max(-value, 0)))
        radius = 2 if hold else 0
        draw.ellipse((col - radius, row - radius, col + radius + 1, row + radius + 1), fill=(232, 183, 61) if hold else color)
    return image


def _histogram(before: np.ndarray, after: np.ndarray) -> Image.Image:
    width, height = 900, 540
    image = Image.new("RGB", (width, height), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    bounds = (-0.6, 0.6)
    hb, _ = np.histogram(before, bins=80, range=bounds)
    ha, _ = np.histogram(after, bins=80, range=bounds)
    maximum = max(int(hb.max()), int(ha.max()), 1)
    left, top, right, bottom = 55, 30, width - 20, height - 45
    draw.rectangle((left, top, right, bottom), outline=(70, 70, 65))
    for index, (a, b) in enumerate(zip(hb, ha)):
        x0 = left + index * (right - left) / len(hb)
        x1 = left + (index + 1) * (right - left) / len(hb)
        draw.rectangle((x0, bottom - a / maximum * (bottom - top), x1, bottom), fill=(85, 126, 166))
        draw.line((x0, bottom - b / maximum * (bottom - top), x1, bottom - b / maximum * (bottom - top)), fill=(189, 63, 54), width=2)
    zero = left + (0.0 - bounds[0]) / (bounds[1] - bounds[0]) * (right - left)
    draw.line((zero, top, zero, bottom), fill=(20, 20, 20), width=1)
    draw.text((left, bottom + 12), "ALS minus surface, -0.6 m to +0.6 m", fill=(25, 29, 25), font=_FONT)
    draw.text((left + 320, 10), "blue=C0 bars, red=TGV line", fill=(25, 29, 25), font=_FONT)
    return image


def _sections(evidence: AlsTgvEvidence, after: np.ndarray) -> Image.Image:
    width, height = 1800, 560
    image = Image.new("RGB", (width, height), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 55, 25, width - 25, height - 45
    rows = [int(round((evidence.bbox_en[3] - northing) / evidence.pitch_m)) for northing in (6444436.0, 6444452.0, 6444472.0)]
    start = int(round((680432.0 - evidence.bbox_en[0]) / evidence.pitch_m))
    stop = int(round((680720.0 - evidence.bbox_en[0]) / evidence.pitch_m)) + 1
    values = np.concatenate([evidence.c0_m[rows, start:stop].ravel(), after[rows, start:stop].ravel()])
    vmin, vmax = np.percentile(values, [1, 99])
    colors = ((209, 73, 91), (0, 121, 140), (237, 174, 73))
    draw.rectangle((left, top, right, bottom), outline=(70, 70, 65))
    for row, color in zip(rows, colors):
        for series, alpha in ((evidence.c0_m[row, start:stop], 0.42), (after[row, start:stop], 1.0)):
            points = []
            for index, value in enumerate(series):
                x = left + index / max(len(series) - 1, 1) * (right - left)
                y = bottom - np.clip((value - vmin) / max(vmax - vmin, 1e-6), 0, 1) * (bottom - top)
                points.append((x, y))
            shade = tuple(int(component * alpha + 244 * (1.0 - alpha)) for component in color)
            draw.line(points, fill=shade, width=2 if alpha == 1.0 else 1)
    draw.text((left, bottom + 12), "E 680432 to 680720; faint=C0, solid=ALS/TGV; N 6444436 / 4452 / 4472", fill=(25, 29, 25), font=_FONT)
    return image


def write_qa(root: Path, evidence: AlsTgvEvidence, residual: np.ndarray) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=False)
    after = evidence.c0_m + residual
    vmin, vmax = np.percentile(evidence.c0_m[evidence.active], [1, 99])

    ownership = np.zeros((*evidence.c0_m.shape, 3), dtype=np.uint8)
    ownership[:] = (225, 220, 205)
    ownership[evidence.hard_zero] = (181, 65, 44)
    ownership[evidence.mapped_face] = (20, 22, 22)
    p1 = root / "01-c0-common-light.png"
    _join([
        _panel(_shade(evidence.c0_m, evidence.pitch_m, vmin, vmax), "Accepted C0, common light"),
        _panel(Image.fromarray(ownership, "RGB"), "C0 ownership: red=hard zero, black=mapped face"),
    ], 2).save(p1)

    rr = (evidence.bbox_en[3] - evidence.point_y) / evidence.pitch_m
    cc = (evidence.point_x - evidence.bbox_en[0]) / evidence.pitch_m
    predicted = ndimage.map_coordinates(residual, [rr, cc], order=1, mode="nearest")
    hold = evidence.point_holdout
    p2 = root / "02-als-support-holdout-residual.png"
    _join([
        _panel(_support_map(evidence), "Qualified class-2 ALS: residual color, gold=withheld"),
        _panel(_histogram(evidence.point_residual_m[hold], evidence.point_residual_m[hold] - predicted[hold]), "Spatial-block holdout residual"),
    ], 2).save(p2)

    limit = max(float(np.percentile(np.abs(residual[evidence.active]), 99)), 0.05)
    mask = np.zeros((*residual.shape, 3), dtype=np.uint8)
    mask[:] = (225, 220, 205)
    mask[evidence.active] = (65, 132, 96)
    mask[evidence.imagery_masked] = (207, 155, 55)
    mask[evidence.hard_zero] = (181, 65, 44)
    mask[evidence.mapped_face] = (20, 22, 22)
    p3 = root / "03-reconstruction-residual-ownership.png"
    _join([
        _panel(_shade(after, evidence.pitch_m, vmin, vmax), "ALS/TGV reconstruction, common light", (700, 480)),
        _panel(_residual_image(residual, limit), f"Owned residual, +/-{limit:.3f} m", (700, 480)),
        _panel(Image.fromarray(mask, "RGB"), "green=active, gold=image masked, red=hard, black=face", (700, 480)),
    ], 3).save(p3)

    r0 = int(round((evidence.bbox_en[3] - 6444544.0) / evidence.pitch_m))
    r1 = int(round((evidence.bbox_en[3] - 6444416.0) / evidence.pitch_m)) + 1
    c0 = int(round((680448.0 - evidence.bbox_en[0]) / evidence.pitch_m))
    c1 = int(round((680704.0 - evidence.bbox_en[0]) / evidence.pitch_m)) + 1
    p4 = root / "04-before-after-cross-sections.png"
    _join([
        _panel(_shade(evidence.c0_m[r0:r1, c0:c1], evidence.pitch_m, vmin, vmax), "Before: accepted C0", (900, 500)),
        _panel(_shade(after[r0:r1, c0:c1], evidence.pitch_m, vmin, vmax), "After: ALS/TGV", (900, 500)),
        _panel(_sections(evidence, after), "Three east-west sections through adjacent shoulder/toe/apron", (1800, 560)),
    ], 2).save(p4)

    return {
        p1.name: "accepted C0 under the exact common light, with hard and mapped-face ownership",
        p2.name: "qualified raw class-2 ALS support and withheld spatial-block residual distributions",
        p3.name: "reconstructed surface, signed residual, and exact ownership masks",
        p4.name: "common-light output comparison and three fixed cross-sections through adjacent morphology",
    }
