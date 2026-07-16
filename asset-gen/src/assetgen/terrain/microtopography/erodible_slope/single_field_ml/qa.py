"""Four directly useful diagnostics for the learned single-field attempt."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

_FONT = ImageFont.load_default()
_BG = (244, 240, 229)


def _shade(height: np.ndarray, pitch: float, lo: float, hi: float) -> Image.Image:
    gy, gx = np.gradient(height.astype(np.float64), pitch)
    normal = np.stack((-gx, gy, np.ones_like(height)), axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    light = np.asarray((-0.62, 0.55, 0.56))
    lighting = 0.40 + 0.60 * np.clip((normal @ light + 0.25) / 1.25, 0.0, 1.0)
    t = np.clip((height - lo) / max(hi - lo, 1.0e-8), 0.0, 1.0)
    rgb = np.asarray((58, 82, 47)) + t[..., None] * (np.asarray((190, 178, 138)) - np.asarray((58, 82, 47)))
    return Image.fromarray(np.clip(rgb * lighting[..., None], 0, 255).astype(np.uint8), "RGB")


def _signed(value: np.ndarray, limit: float) -> Image.Image:
    t = np.clip(value / max(limit, 1.0e-8), -1.0, 1.0)
    neutral = np.asarray((239.0, 236.0, 224.0))
    cold, warm = np.asarray((36.0, 90.0, 154.0)), np.asarray((190.0, 61.0, 45.0))
    rgb = np.where((t < 0)[..., None], neutral + (-t[..., None]) * (cold - neutral), neutral + t[..., None] * (warm - neutral))
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")


def _scalar(value: np.ndarray, color: tuple[int, int, int] = (34, 110, 76)) -> Image.Image:
    t = np.clip(value, 0.0, 1.0)
    rgb = np.asarray((235, 232, 217)) + t[..., None] * (np.asarray(color) - np.asarray((235, 232, 217)))
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")


def _ownership(delta: np.ndarray, conditions: np.ndarray, threshold: float) -> Image.Image:
    kind = np.zeros(delta.shape, dtype=np.uint8)
    active = np.abs(delta) >= threshold
    kind[active & (delta > 0) & (conditions[1] >= conditions[2])] = 1
    kind[active & (delta < 0) & (conditions[3] < 0.50)] = 2
    kind[active & (delta < 0) & (conditions[3] >= 0.50)] = 3
    kind[active & (delta > 0) & (conditions[2] > conditions[1])] = 4
    colors = np.asarray(((230, 226, 213), (213, 132, 53), (157, 54, 45), (35, 91, 153), (127, 103, 59)), dtype=np.uint8)
    return Image.fromarray(colors[kind], "RGB")


def _panel(image: Image.Image, title: str, size: tuple[int, int] = (720, 430)) -> Image.Image:
    result = Image.new("RGB", (size[0], size[1] + 34), _BG)
    result.paste(image.resize(size, Image.Resampling.LANCZOS), (0, 34))
    ImageDraw.Draw(result).text((10, 10), title, fill=(25, 29, 25), font=_FONT)
    return result


def _join(images: list[Image.Image]) -> Image.Image:
    width, height = max(item.width for item in images), max(item.height for item in images)
    result = Image.new("RGB", (2 * width, 2 * height), _BG)
    for index, item in enumerate(images):
        result.paste(item, ((index % 2) * width, (index // 2) * height))
    return result


def _sections(c0: np.ndarray, c1: np.ndarray) -> Image.Image:
    image = Image.new("RGB", (720, 430), _BG)
    draw = ImageDraw.Draw(image)
    rows = [int(f * (c0.shape[0] - 1)) for f in (0.30, 0.50, 0.70)]
    values = np.concatenate((c0[rows].ravel(), c1[rows].ravel()))
    lo, hi = np.percentile(values, (1, 99))
    for row, color in zip(rows, ((203, 63, 82), (0, 116, 137), (225, 157, 54))):
        for surface, width in ((c0[row], 1), (c1[row], 2)):
            points = [(24 + col / (len(surface) - 1) * 672, 390 - np.clip((z - lo) / (hi - lo), 0, 1) * 350) for col, z in enumerate(surface)]
            draw.line(points, fill=color if width == 2 else tuple((v + 244) // 2 for v in color), width=width)
    draw.text((24, 407), "faint=C0, solid=C1; fixed rows .30/.50/.70", fill=(25, 29, 25), font=_FONT)
    return image


def _resize(value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return ndimage.zoom(value, (shape[0] / value.shape[0], shape[1] / value.shape[1]), order=1)[: shape[0], : shape[1]]


def write_qa(
    root: Path,
    c0: np.ndarray,
    c1: np.ndarray,
    delta: np.ndarray,
    hard: np.ndarray,
    conditions: np.ndarray,
    opportunity: np.ndarray,
    nearest_map: np.ndarray,
    anti_copy: dict[str, float],
    pitch_m: float,
) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=False)
    valid = ~hard
    lo, hi = np.percentile(c0[valid], (1, 99))
    limit = max(float(np.percentile(np.abs(delta[valid]), 99)), 0.03)
    inset = (slice(int(.16 * c0.shape[0]), int(.84 * c0.shape[0])), slice(int(.10 * c0.shape[1]), int(.90 * c0.shape[1])))

    p1 = root / "01-common-light-c0-c1.png"
    _join([
        _panel(_shade(c0, pitch_m, lo, hi), "C0 accepted structural authority; common light"),
        _panel(_shade(c1, pitch_m, lo, hi), "C1 learned single field; identical light/range"),
        _panel(_shade(c0[inset], pitch_m, lo, hi), "C0 fixed publication-interior crop"),
        _panel(_shade(c1[inset], pitch_m, lo, hi), "C1 fixed crop: inspect crest, incision, toe, phase"),
    ]).save(p1)

    broad = ndimage.gaussian_filter(delta, 2.0 / pitch_m)
    supported = delta - broad
    interpolation_only = delta - _resize(delta[::2, ::2], delta.shape)
    p2 = root / "02-scale-band-signed-relief.png"
    _join([
        _panel(_signed(delta, limit), f"C1-C0 signed relief +/-{limit:.3f} m"),
        _panel(_signed(broad, max(float(np.percentile(np.abs(broad[valid]), 99)), .01)), "coherent >2 m support"),
        _panel(_signed(supported, max(float(np.percentile(np.abs(supported[valid]), 99)), .01)), "0.5-2 m Biala-supported band"),
        _panel(_signed(interpolation_only, .02), "sub-0.5 m view is interpolation only; no truth claim"),
    ]).save(p2)

    p3 = root / "03-conditioning-ownership.png"
    _join([
        _panel(_scalar(conditions[0]), "Development-A C0 slope condition"),
        _panel(_signed(conditions[2] - conditions[1], 1.0), "signed C0 curvature: concave blue / convex red"),
        _panel(_scalar(conditions[3], (35, 91, 153)), "preserved ALS structural network cef63d8 lineage"),
        _panel(_ownership(delta, conditions, .012), "ownership: shoulder orange, side red, incision blue, toe brown"),
    ]).save(p3)

    text_panel = Image.new("RGB", (720, 430), _BG)
    draw = ImageDraw.Draw(text_panel)
    rows = ["ANTI-COPY / REPETITION"] + [f"{key}: {value:.6f}" for key, value in anti_copy.items()]
    rows += ["nearest patches use 8 m source support", "high correlation is a rejection signal, not fidelity credit"]
    for index, row in enumerate(rows):
        draw.text((24, 24 + 32 * index), row, fill=(25, 29, 25), font=_FONT)
    p4 = root / "04-nearest-source-repetition-sections.png"
    _join([
        _panel(_scalar(nearest_map, (157, 54, 45)), "nearest normalized Biala source-patch correlation"),
        _panel(_scalar(opportunity), "continuous whole-field opportunity; no placement lattice"),
        _panel(_sections(c0, c1), "fixed whole-domain sections"),
        _panel(text_panel, "anti-copy and phase-lock metrics"),
    ]).save(p4)
    return {
        p1.name: "common-light same-range C0/C1 full-domain and fixed close comparison",
        p2.name: "signed relief and honest Biala-supported scale bands",
        p3.name: "C0 slope/curvature, preserved ALS network, and learned-field ownership",
        p4.name: "nearest-source similarity, repetition/phase metrics, opportunity, and fixed sections",
    }
