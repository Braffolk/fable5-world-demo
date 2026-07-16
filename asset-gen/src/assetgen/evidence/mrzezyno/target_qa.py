"""Render the bounded Mrzezyno morphology-target inspection bundle."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

_PAPER = (239, 235, 222)
_INK = (28, 31, 30)
_NODATA = (104, 31, 67)
_DUNE = (198, 160, 75)
_LOW = (49, 122, 133)
_UNRESOLVED = (153, 145, 126)


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _label(image: Image.Image, title: str, subtitle: str) -> Image.Image:
    output = Image.new("RGB", (image.width, image.height + 78), _PAPER)
    output.paste(image, (0, 78))
    draw = ImageDraw.Draw(output)
    draw.text((16, 8), title, font=_font(24, bold=True), fill=_INK)
    draw.text((16, 43), subtitle, font=_font(14), fill=_INK)
    return output


def _save_pair(path: Path, left: Image.Image, right: Image.Image, footer: str) -> None:
    height = max(left.height, right.height)
    output = Image.new("RGB", (left.width + right.width, height + 52), _PAPER)
    output.paste(left, (0, 0))
    output.paste(right, (left.width, 0))
    draw = ImageDraw.Draw(output)
    draw.rectangle((0, height, output.width, output.height), fill=_INK)
    draw.text((20, height + 14), footer, font=_font(17, bold=True), fill=_PAPER)
    output.save(path, optimize=True)


def _filled(height: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if np.all(valid):
        return height.astype(np.float64, copy=False)
    _, indices = ndimage.distance_transform_edt(~valid, return_indices=True)
    result = height.astype(np.float64, copy=True)
    result[~valid] = result[tuple(axis[~valid] for axis in indices)]
    return result


def _height_rgb(height: np.ndarray, valid: np.ndarray) -> np.ndarray:
    lo, hi = np.quantile(height[valid], (0.01, 0.99))
    value = np.clip((height - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    value[~valid] = 0.0
    stops = np.asarray(((30, 68, 76), (63, 118, 105), (184, 164, 94), (231, 213, 161)))
    position = value * (len(stops) - 1)
    index = np.minimum(position.astype(np.intp), len(stops) - 2)
    fraction = (position - index)[..., None]
    rgb = np.rint(stops[index] * (1.0 - fraction) + stops[index + 1] * fraction).astype(np.uint8)
    rgb[~valid] = _NODATA
    return rgb


def _hillshade_rgb(height: np.ndarray, valid: np.ndarray, pixel_m: float) -> np.ndarray:
    surface = _filled(height, valid)
    gy, gx = np.gradient(surface, pixel_m)
    length = np.sqrt(gx * gx + gy * gy + 1.0)
    light = (0.42 * gx - 0.36 * gy + 0.83) / length
    shade = np.clip(0.46 + 0.54 * light, 0.0, 1.0)
    rgb = np.repeat(np.rint(shade[..., None] * 255).astype(np.uint8), 3, axis=2)
    rgb[~valid] = _NODATA
    return rgb


def _resize(rgb: np.ndarray, width: int) -> Image.Image:
    image = Image.fromarray(rgb, "RGB")
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.BOX)


def _patch_mosaic(patches: np.ndarray, records: Sequence[Mapping[str, Any]]) -> Image.Image:
    tile = 430
    output = Image.new("RGB", (tile * 2, tile * 2), _PAPER)
    draw = ImageDraw.Draw(output)
    for index, (patch, record) in enumerate(zip(patches, records, strict=True)):
        valid = np.isfinite(patch)
        rgb = _hillshade_rgb(np.nan_to_num(patch), valid, 0.1)
        image = Image.fromarray(rgb, "RGB").resize((tile, tile), Image.Resampling.BILINEAR)
        x = index % 2 * tile
        y = index // 2 * tile
        output.paste(image, (x, y))
        draw.rectangle((x + 10, y + 10, x + 232, y + 62), fill=_INK)
        draw.text((x + 20, y + 17), record["display_label"], font=_font(18, bold=True), fill=_PAPER)
    return output


def _band_rgb(values: np.ndarray, scale: float) -> np.ndarray:
    normalized = np.clip(values / max(scale, 1e-6), -1.0, 1.0)
    low = np.asarray((35, 105, 132), dtype=np.float64)
    mid = np.asarray((238, 233, 218), dtype=np.float64)
    high = np.asarray((181, 65, 44), dtype=np.float64)
    fraction = np.abs(normalized)[..., None]
    rgb = np.where(
        (normalized < 0)[..., None],
        mid * (1.0 - fraction) + low * fraction,
        mid * (1.0 - fraction) + high * fraction,
    )
    return np.rint(rgb).astype(np.uint8)


def render_target_qa(
    height: np.ndarray,
    valid: np.ndarray,
    role_state: np.ndarray,
    patches: np.ndarray,
    patch_records: Sequence[Mapping[str, Any]],
    band_examples: np.ndarray,
    band_records: Sequence[Mapping[str, Any]],
    band_summary: Sequence[Mapping[str, Any]],
    qa_dir: Path,
) -> tuple[dict[str, Any], ...]:
    qa_dir.mkdir(parents=True, exist_ok=True)

    overview_height = _label(
        _resize(_height_rgb(height, valid), 1150),
        "01 / complete retained coastal geometry",
        "p01-p99 height; purple is declared nodata or one of three explicit invalid extremes",
    )
    overview_shade = _label(
        _resize(_hillshade_rgb(height, valid, 0.1), 1150),
        "Native-source hillshade",
        "0.10 m provider grid; display derivative only, no invented terrain",
    )
    path1 = qa_dir / "01_full_valid_geometry.png"
    _save_pair(path1, overview_height, overview_shade, "7,799,605 finite non-extreme cells retained; no nodata or extreme cell is filled")

    native = _label(
        _patch_mosaic(patches, patch_records),
        "02 / representative native morphology",
        "top row: high-relief dune-landform candidates; bottom row: low coastal surface, material unresolved",
    )
    notes = Image.new("RGB", native.size, _PAPER)
    draw = ImageDraw.Draw(notes)
    draw.text((28, 22), "What these patches can mean", font=_font(28, bold=True), fill=_INK)
    lines = (
        "HIGH RELIEF: dune-coast landform shape exemplar",
        "  not a dry-sand, vegetation-free, or per-cell support label",
        "LOW RELIEF: retained geometry only",
        "  not authorized as shore.beach_sand; wet/dry and shoreline are absent",
        "All patches remain provider IDW surfaces.",
        "No fine ripple or 0.10 m transfer-function claim is made.",
    )
    y = 90
    for line in lines:
        draw.text((36, y), line, font=_font(19, bold=line.startswith(("HIGH", "LOW"))), fill=_INK)
        y += 56
    path2 = qa_dir / "02_representative_native_morphology.png"
    _save_pair(path2, native, notes, "DUNE ROLE NARROWED TO LANDFORM SHAPE; BEACH MATERIAL ROLE REMAINS UNSUPPORTED")

    tile = 270
    band_panel = Image.new("RGB", (tile * len(band_records), tile), _PAPER)
    band_draw = ImageDraw.Draw(band_panel)
    scale = max(float(np.quantile(np.abs(band_examples), 0.99)), 0.005)
    for index, (values, record) in enumerate(zip(band_examples, band_records, strict=True)):
        image = Image.fromarray(_band_rgb(values, scale), "RGB").resize((tile, tile), Image.Resampling.BILINEAR)
        x = index * tile
        band_panel.paste(image, (x, 0))
        band_draw.rectangle((x + 8, 8, x + 142, 48), fill=_INK)
        band_draw.text((x + 16, 13), record["label"], font=_font(16, bold=True), fill=_PAPER)
    report = Image.new("RGB", (band_panel.width, 560), _PAPER)
    report_draw = ImageDraw.Draw(report)
    report_draw.text((24, 18), "Observed relief versus campaign p95 elevation bound", font=_font(26, bold=True), fill=_INK)
    report_draw.text((24, 57), "Exceeds means |band| p95 > 0.10 m; it does not identify band-resolved error.", font=_font(16), fill=_INK)
    y = 102
    for record in band_summary:
        color = (54, 116, 76) if record["exceeds_campaign_p95_bound"] else _NODATA
        report_draw.text(
            (28, y),
            f"{record['label']:>10}  dune p95 {record['dune_abs_p95_m']:.3f} m  low p95 {record['low_abs_p95_m']:.3f} m",
            font=_font(18, bold=True),
            fill=color,
        )
        y += 54
    report_draw.text((24, y + 18), "Green: observed relief clears 0.10 m in at least the dune candidates.", font=_font(16), fill=_INK)
    report_draw.text((24, y + 52), "Red: below/at the campaign bound; not a useful target band here.", font=_font(16), fill=_INK)
    path3 = qa_dir / "03_scale_band_separability.png"
    _save_pair(
        path3,
        _label(band_panel, "03 / one dune candidate by scale", f"common diverging range +/-{scale:.3f} m"),
        _label(report, "Conservative amplitude screen", "campaign median <0.05 m; p95 <=0.10 m"),
        "ONLY BANDS CLEARING THE CAMPAIGN P95 BOUND MAY SUPPLY SHAPE STATISTICS",
    )

    role_rgb = np.zeros((*role_state.shape, 3), dtype=np.uint8)
    role_rgb[role_state == 0] = _NODATA
    role_rgb[role_state == 1] = _DUNE
    role_rgb[role_state == 2] = _LOW
    role_rgb[role_state == 3] = _UNRESOLVED
    role_map = _label(
        _resize(role_rgb, 1150),
        "04 / geometry role map, not material labels",
        "gold: high-relief dune-landform candidate; blue/grey: low/transition coastal geometry unresolved",
    )
    contract = Image.new("RGB", role_map.size, _PAPER)
    contract_draw = ImageDraw.Draw(contract)
    contract_draw.text((28, 20), "Authorized next synthesis input", font=_font(27, bold=True), fill=_INK)
    contract_draw.text((34, 82), "sand.dune_aeolian", font=_font(21, bold=True), fill=_DUNE)
    contract_draw.text((34, 122), "PASS, NARROW: high-relief landform shape and clearing bands only", font=_font(17), fill=_INK)
    contract_draw.text((34, 184), "shore.beach_sand", font=_font(21, bold=True), fill=_LOW)
    contract_draw.text((34, 224), "PARK: no wet/dry sand, shoreline, or beach material separation", font=_font(17), fill=_INK)
    contract_draw.text((34, 302), "Still unresolved everywhere", font=_font(21, bold=True), fill=_INK)
    unresolved = (
        "original-point support and interpolation distance",
        "vegetation/ground confusion at fine scales",
        "material state and process direction",
        "band-resolved observation error",
    )
    y = 346
    for line in unresolved:
        contract_draw.text((48, y), f"- {line}", font=_font(17), fill=_INK)
        y += 43
    path4 = qa_dir / "04_unresolved_support_and_material.png"
    _save_pair(path4, role_map, contract, "NO BEACH LABELS / NO RIPPLE TRUTH / NO PER-CELL TARGET AUTHORITY")

    return (
        {"path": path1, "interpretation": "Complete valid provider geometry with declared nodata and all three numeric extremes excluded."},
        {"path": path2, "interpretation": "Native representative morphology split into high-relief dune-landform candidates and low coastal geometry without pretending the split is a material label."},
        {"path": path3, "interpretation": "Octave-scale observed relief compared conservatively with the published 0.10 m campaign p95 elevation-deviation bound."},
        {"path": path4, "interpretation": "Role map and explicit limits: narrowed dune-shape input only, beach-sand role parked."},
    )
