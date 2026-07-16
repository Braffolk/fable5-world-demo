"""Exactly four FLOAT-review images for the coupled process challenger."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .model import Result


WIDTH = 1800
HEIGHT = 1400
PAPER = (240, 237, 224)
INK = (30, 37, 31)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _hillshade(
    height: np.ndarray,
    pitch_m: float,
    limits: tuple[float, float],
) -> np.ndarray:
    gy, gx = np.gradient(height.astype(np.float64), pitch_m)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    azimuth = np.deg2rad(315.0)
    altitude = np.deg2rad(38.0)
    light = (
        np.sin(altitude) * np.cos(slope)
        + np.cos(altitude) * np.sin(slope) * np.cos(azimuth - aspect)
    )
    light = np.clip((light + 0.15) / 1.15, 0.0, 1.0)
    low, high = limits
    elevation = np.clip((height - low) / max(high - low, 1.0e-9), 0.0, 1.0)
    base = np.stack(
        (0.36 + 0.28 * elevation, 0.42 + 0.31 * elevation, 0.26 + 0.19 * elevation),
        axis=-1,
    )
    return np.clip(base * (0.48 + 0.70 * light[..., None]), 0.0, 1.0)


def _signed(value: np.ndarray, limit: float | None = None) -> np.ndarray:
    finite = np.isfinite(value)
    if limit is None:
        limit = max(float(np.percentile(np.abs(value[finite]), 98.0)), 1.0e-9)
    scaled = np.clip(value / limit, -1.0, 1.0)
    neutral = np.asarray([0.94, 0.92, 0.84])
    red = np.asarray([0.73, 0.15, 0.08])
    blue = np.asarray([0.06, 0.30, 0.61])
    return np.where(
        (scaled >= 0)[..., None],
        neutral + scaled[..., None] * (red - neutral),
        neutral + (-scaled)[..., None] * (blue - neutral),
    )


def _scalar(value: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    finite = np.isfinite(value)
    low, high = np.percentile(value[finite], [2.0, 98.0])
    unit = np.clip((value - low) / max(float(high - low), 1.0e-9), 0.0, 1.0)
    paper = np.asarray([0.94, 0.92, 0.84])
    target = np.asarray(color)
    return paper + unit[..., None] * (target - paper)


def _classes(delta: np.ndarray, threshold: float = 0.04) -> np.ndarray:
    rgb = np.zeros((*delta.shape, 3), dtype=np.float64)
    rgb[:] = (0.73, 0.70, 0.49)
    rgb[delta < -threshold] = (0.12, 0.40, 0.66)
    rgb[delta > threshold] = (0.57, 0.30, 0.12)
    return rgb


def _probability_classes(result: Result) -> np.ndarray:
    stack = np.stack(
        (result.process_hollow_05m, result.process_lawn_05m, result.process_hummock_05m),
        axis=-1,
    )
    palette = np.asarray(
        [[0.12, 0.40, 0.66], [0.73, 0.70, 0.49], [0.57, 0.30, 0.12]]
    )
    return np.clip(stack @ palette, 0.0, 1.0)


def _trace_image(trace: np.ndarray) -> np.ndarray:
    width, height = 780, 480
    image = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(image)
    colors = ((156, 73, 30), (112, 106, 73), (31, 100, 164))
    labels = ("hummock occupancy", "lawn occupancy", "hollow occupancy")
    left, top, right, bottom = 62, 28, width - 24, height - 48
    draw.rectangle((left, top, right, bottom), outline=(120, 120, 105), width=1)
    steps = trace[:, 0]
    for offset, (color, label) in enumerate(zip(colors, labels, strict=True)):
        values = trace[:, 2 + offset]
        points = [
            (
                left + (right - left) * float((step - steps[0]) / max(steps[-1] - steps[0], 1.0)),
                bottom - (bottom - top) * float(np.clip(value, 0.0, 1.0)),
            )
            for step, value in zip(steps, values, strict=True)
        ]
        draw.line(points, fill=color, width=3)
        draw.text((left + 8, top + 8 + 21 * offset), label, font=_font(15), fill=color)
    draw.text((left, bottom + 11), "coupled coarse iterations", font=_font(15), fill=INK)
    return np.asarray(image, dtype=np.float64) / 255.0


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


def _sheet(
    title: str,
    note: str,
    panels: list[tuple[str, np.ndarray]],
) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
    draw = ImageDraw.Draw(image)
    draw.text((36, 24), title, font=_font(27), fill=INK)
    draw.text((36, 62), note, font=_font(16), fill=(70, 76, 67))
    top = 104
    panel_w = (WIDTH - 72) // 2
    panel_h = (HEIGHT - top - 34) // 2
    for index, (label, rgb) in enumerate(panels):
        row, col = divmod(index, 2)
        x = 36 + col * panel_w
        y = top + row * panel_h
        draw.text((x + 8, y + 5), label, font=_font(17), fill=INK)
        rendered = _panel(rgb, (panel_w - 16, panel_h - 42))
        image.paste(rendered, (x + 8, y + 34))
    return image


def write_qa(
    directory: Path,
    height_1m: np.ndarray,
    authority_1m: np.ndarray,
    drain_1m: np.ndarray,
    result: Result,
    halo_rows: slice,
    halo_cols: slice,
    core_rows: slice,
    core_cols: slice,
    metrics: dict,
) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    broad = ndimage.gaussian_filter(height_1m, 24.0)
    broad_residual = broad - ndimage.gaussian_filter(broad, 96.0)
    boundary = ndimage.distance_transform_edt(authority_1m)
    water = result.water_depth_1m.copy()
    water[~authority_1m] = np.nan
    directional = result.directional_eligibility_1m.copy()
    directional[~authority_1m] = np.nan
    boundary_rgb = _scalar(boundary, (0.13, 0.48, 0.31))
    boundary_rgb[drain_1m] = (0.76, 0.20, 0.10)
    boundary_rgb[~authority_1m] *= 0.25
    images = [
        (
            "01-hydrology-and-directional-eligibility.png",
            _sheet(
                "01 / Whole-mire measured conditioning and hydrology",
                "Red marks mapped drainage context. Directionality is gated by measured slope and interior support.",
                [
                    ("measured >24 m broad relief / signed", _signed(broad_residual)),
                    ("modeled water-table depth / dry red, wet pale", _scalar(water, (0.70, 0.20, 0.10))),
                    ("authority interior distance / drainage red", boundary_rgb),
                    ("directional string/flark eligibility only", _scalar(directional, (0.10, 0.32, 0.69))),
                ],
            ),
        ),
        (
            "02-coupled-state-evolution.png",
            _sheet(
                "02 / Coupled vegetation, water, and peat-state evolution",
                "Deterministic measured perturbation only; no random field, primitives, stamps, or separate fine remainder.",
                [
                    ("initial measured-conditioned process state", _signed(result.initial_process_state_05m)),
                    ("final coupled peat/accretion state", _signed(result.process_state_05m)),
                    ("continuous hollow / lawn / hummock occupancies", _probability_classes(result)),
                    ("whole-mire state trajectory", _trace_image(result.trace)),
                ],
            ),
        ),
    ]

    hr = slice(halo_rows.start * 4, halo_rows.stop * 4)
    hc = slice(halo_cols.start * 4, halo_cols.stop * 4)
    cr = slice(core_rows.start * 4, core_rows.stop * 4)
    cc = slice(core_cols.start * 4, core_cols.stop * 4)
    halo_base = result.structural_fine_m[hr, hc]
    halo_final = result.final_fine_m[hr, hc]
    halo_delta = result.delta_fine_m[hr, hc]
    limits = tuple(np.percentile(halo_base, [2.0, 98.0]))
    images.append(
        (
            "03-halo-common-light-and-classes.png",
            _sheet(
                "03 / Clear-halo common-light FLOAT comparison",
                "Same light and elevation range. Classes use only the supported 0.04 m HuHoLa lawn decision band.",
                [
                    ("measured structural 0.25 m carrier", _hillshade(halo_base, 0.25, limits)),
                    ("v3 coupled-process FLOAT", _hillshade(halo_final, 0.25, limits)),
                    ("signed process relief", _signed(halo_delta, 0.13)),
                    ("blue hollow / tan lawn / brown hummock", _classes(halo_delta)),
                ],
            ),
        )
    )
    core_base = result.structural_fine_m[cr, cc]
    core_final = result.final_fine_m[cr, cc]
    core_delta = result.delta_fine_m[cr, cc]
    eligibility_core = np.repeat(
        np.repeat(result.directional_eligibility_1m[core_rows, core_cols], 4, axis=0),
        4,
        axis=1,
    )
    images.append(
        (
            "04-core-organization-and-gates.png",
            _sheet(
                "04 / Core organization and fail-fast diagnostics",
                f"Status {metrics['status']}; exact parent means={metrics['gates']['parent_mean_exact']['passed']}; spectral peak={metrics['organization']['spectral_peak_power_fraction']:.5f}.",
                [
                    ("v3 core common light", _hillshade(core_final, 0.25, tuple(np.percentile(core_base, [2, 98])))),
                    ("core hollow / lawn / hummock organization", _classes(core_delta)),
                    ("core signed relief / fixed +/-0.13 m", _signed(core_delta, 0.13)),
                    ("directional eligibility; not a generated class", _scalar(eligibility_core, (0.10, 0.32, 0.69))),
                ],
            ),
        )
    )
    for filename, image in images:
        image.save(directory / filename, optimize=True)
