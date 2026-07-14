"""Small, claim-aware QA bundle for the Mrzezyno B1 probe."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

_INK = (27, 31, 30)
_PAPER = (240, 235, 221)
_MAGENTA = (126, 20, 73)
_UNKNOWN = (214, 154, 74)
_EXTREME = (33, 168, 181)


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _panel(image: Image.Image, title: str, subtitle: str) -> Image.Image:
    width, height = image.size
    output = Image.new("RGB", (width, height + 86), _PAPER)
    output.paste(image, (0, 86))
    draw = ImageDraw.Draw(output)
    draw.text((18, 10), title, font=_font(26, bold=True), fill=_INK)
    draw.text((18, 48), subtitle, font=_font(16), fill=_INK)
    return output


def _pair(path: Path, left: Image.Image, right: Image.Image, footer: str) -> None:
    height = max(left.height, right.height)
    output = Image.new("RGB", (left.width + right.width, height + 58), _PAPER)
    output.paste(left, (0, 0))
    output.paste(right, (left.width, 0))
    draw = ImageDraw.Draw(output)
    draw.rectangle((0, height, output.width, output.height), fill=_INK)
    draw.text((24, height + 16), footer, font=_font(18, bold=True), fill=(249, 245, 232))
    output.save(path, optimize=True)


def _resize_rgb(rgb: np.ndarray, width: int = 1200) -> Image.Image:
    image = Image.fromarray(rgb, "RGB")
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.BOX)


def _height_rgb(height: np.ndarray, valid: np.ndarray) -> np.ndarray:
    lo, hi = np.quantile(height[valid], (0.01, 0.99))
    scaled = np.clip((height - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    stops = np.asarray(
        ((31, 68, 75), (78, 117, 83), (169, 160, 100), (224, 208, 154)),
        dtype=np.float64,
    )
    position = scaled * (len(stops) - 1)
    index = np.minimum(position.astype(np.intp), len(stops) - 2)
    fraction = (position - index)[..., None]
    rgb = np.rint(stops[index] * (1 - fraction) + stops[index + 1] * fraction).astype(np.uint8)
    rgb[~valid] = _MAGENTA
    return rgb


def _hillshade_rgb(height: np.ndarray, valid: np.ndarray, pixel_m: float) -> np.ndarray:
    filled = height.copy()
    if not np.all(valid):
        _, indices = ndimage.distance_transform_edt(~valid, return_indices=True)
        filled[~valid] = filled[tuple(index[~valid] for index in indices)]
    gy, gx = np.gradient(filled, pixel_m)
    nx, ny, nz = -gx, -gy, np.ones_like(filled)
    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    light = (-0.45 * nx + 0.40 * ny + 0.80 * nz) / length
    shade = np.clip(0.48 + 0.52 * light, 0.0, 1.0)
    rgb = np.repeat(np.rint(255 * shade)[..., None].astype(np.uint8), 3, axis=2)
    rgb[~valid] = _MAGENTA
    return rgb


def _diverging(values: np.ndarray, scale: float) -> np.ndarray:
    normalized = np.clip(values / max(scale, 1e-6), -1.0, 1.0)
    low = np.asarray((34, 112, 139), dtype=np.float64)
    mid = np.asarray((239, 234, 216), dtype=np.float64)
    high = np.asarray((184, 61, 45), dtype=np.float64)
    fraction = np.abs(normalized)[..., None]
    rgb = np.where(
        (normalized < 0)[..., None],
        mid * (1 - fraction) + low * fraction,
        mid * (1 - fraction) + high * fraction,
    )
    return np.rint(rgb).astype(np.uint8)


def render_probe_qa(
    preview_height: np.ndarray,
    preview_state: np.ndarray,
    contexts: np.ndarray,
    context_records: list[Mapping[str, Any]],
    decision: Mapping[str, Any],
    qa_dir: Path,
    *,
    preview_pixel_m: float,
) -> tuple[dict[str, Any], ...]:
    qa_dir.mkdir(parents=True, exist_ok=True)
    valid = preview_state == 1
    height_panel = _panel(
        _resize_rgb(_height_rgb(preview_height, valid)),
        "2022-02 provider DEM height",
        "p01-p99 color; magenta is declared nodata or invalid extreme",
    )
    shade_panel = _panel(
        _resize_rgb(_hillshade_rgb(preview_height, valid, preview_pixel_m)),
        "Provider DEM hillshade",
        "display-only derivative; it does not establish measured support",
    )
    path1 = qa_dir / "01_source_height_and_hillshade.png"
    _pair(path1, height_panel, shade_panel, "SOURCE GEOMETRY / ALL FINITE DEM CELLS REMAIN IDW SUPPORT-UNKNOWN")

    state_rgb = np.zeros((*preview_state.shape, 3), dtype=np.uint8)
    state_rgb[preview_state == 0] = _MAGENTA
    state_rgb[preview_state == 1] = _UNKNOWN
    state_rgb[preview_state == 2] = _EXTREME
    state_image = _resize_rgb(state_rgb)
    explanation = Image.new("RGB", state_image.size, _PAPER)
    draw = ImageDraw.Draw(explanation)
    draw.text((36, 38), "Cell-state contract", font=_font(34, bold=True), fill=_INK)
    rows = (
        (_UNKNOWN, "Provider-valid IDW raster", "support distance, dry-sand semantics, and source points unknown"),
        (_MAGENTA, "Declared nodata", "never filled into scientific arrays"),
        (_EXTREME, "3 undeclared extremes", "two -34.6023 m cells and one 5.75e34 m cell; explicit invalid"),
    )
    y = 115
    for color, title, body in rows:
        draw.rectangle((42, y, 78, y + 36), fill=color)
        draw.text((98, y - 2), title, font=_font(22, bold=True), fill=_INK)
        draw.text((98, y + 32), body, font=_font(16), fill=_INK)
        y += 105
    draw.text((42, y + 10), "Authorized target cells: 0", font=_font(28, bold=True), fill=_MAGENTA)
    path2 = qa_dir / "02_support_semantics_and_invalidity.png"
    _pair(
        path2,
        _panel(state_image, "Co-registered source state", "orange is usable only for descriptive raster inspection"),
        _panel(explanation, "Why provider-valid is not observed", "unknowns are preserved rather than reconstructed"),
        "NO PER-CELL POINT SUPPORT / NO DRY-SAND MASK / NO TARGET SURFACE",
    )

    b1_scale = max(float(np.quantile(np.abs(contexts), 0.99)), 0.005)
    tiles = [_diverging(values, b1_scale) for values in contexts]
    tile_size = 300
    mosaic = Image.new("RGB", (tile_size * 3, tile_size * 2), _PAPER)
    draw = ImageDraw.Draw(mosaic)
    for index, tile in enumerate(tiles):
        image = Image.fromarray(tile, "RGB").resize((tile_size, tile_size), Image.Resampling.NEAREST)
        x = (index % 3) * tile_size
        y = (index // 3) * tile_size
        mosaic.paste(image, (x, y))
        draw.rectangle((x + 8, y + 8, x + 66, y + 40), fill=_INK)
        draw.text((x + 19, y + 11), f"C{index + 1}", font=_font(18, bold=True), fill=(255, 255, 255))
    report = Image.new("RGB", mosaic.size, _PAPER)
    report_draw = ImageDraw.Draw(report)
    report_draw.text((32, 24), "B1 separability screen", font=_font(32, bold=True), fill=_INK)
    report_draw.text((32, 72), f"Residual display range: +/-{b1_scale:.3f} m", font=_font(18), fill=_INK)
    y = 120
    for index, record in enumerate(context_records):
        report_draw.text(
            (38, y),
            f"C{index + 1}: RMS {record['b1_rms_m']:.4f} m | |B1| p95 {record['b1_abs_p95_m']:.4f} m",
            font=_font(17),
            fill=_INK,
        )
        y += 37
    evidence = decision["published_control_evidence"]
    report_draw.line((32, y + 10, report.width - 32, y + 10), fill=(120, 111, 94), width=2)
    report_draw.text((32, y + 28), "Published campaign-level elevation deviation", font=_font(19, bold=True), fill=_INK)
    report_draw.text((38, y + 64), f"median < {evidence['median_vertical_deviation_upper_bound_m']:.2f} m", font=_font(17), fill=_INK)
    report_draw.text((38, y + 96), f"p95 <= {evidence['vertical_deviation_p95_upper_bound_m']:.2f} m", font=_font(17), fill=_INK)
    report_draw.text((32, y + 145), "Band-resolved error: UNIDENTIFIABLE", font=_font(21, bold=True), fill=_MAGENTA)
    report_draw.text((32, y + 182), "B1 target-site decision: NO-GO", font=_font(25, bold=True), fill=_MAGENTA)
    path3 = qa_dir / "03_b1_signal_and_error_decision.png"
    _pair(
        path3,
        _panel(mosaic, "Descriptive canonical B1 residuals", "six mask-selected 8 m contexts; central 3 m only"),
        _panel(report, "Error evidence cannot close the gate", "published RTK values lack per-cell support and error spectrum"),
        "DESCRIPTIVE RELIEF EXISTS; QUALIFIED B1 SIGNAL/ERROR SEPARATION DOES NOT",
    )
    return (
        {"path": path1, "interpretation": "Provider height and hillshade expose the complete retained coastal strip while preserving invalid areas; both are display-only."},
        {"path": path2, "interpretation": "Every source cell is classified as declared nodata, undeclared numeric extreme, or provider-valid IDW with unknown measured support and dry-sand semantics."},
        {"path": path3, "interpretation": "Canonical B1 residuals from six mask-selected contexts are descriptive only; campaign-level RTK bounds cannot identify band-resolved total error."},
    )
