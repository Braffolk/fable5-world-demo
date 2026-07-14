"""Render the one inspectable Hovi semantic-scope diagnostic."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .model import CLASS_COLORS

_WIDTH = 1600
_MARGIN = 28
_GAP = 22
_PANEL_W = (_WIDTH - 2 * _MARGIN - _GAP) // 2
_PHOTO_H = 440
_TEXT_H = 180
_HEADER_H = 150


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def render_semantic_scope_sheet(
    quadrats: Sequence[Mapping[str, Any]], destination: Path
) -> list[dict[str, Any]]:
    """Pair publisher cover fractions with photos without classifying pixels."""
    panel_h = _PHOTO_H + _TEXT_H
    height = _HEADER_H + _MARGIN + 2 * panel_h + _MARGIN
    canvas = Image.new("RGB", (_WIDTH, height), (244, 241, 234))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (_MARGIN, 18),
        "HY_SPRUCE4 | publisher fractional-cover semantics",
        fill=(24, 38, 34),
        font=_font(30),
    )
    draw.text(
        (_MARGIN, 58),
        "Usable: quadrat and plot composition. Not usable: TLS point, view, or 6.25 cm cell labels.",
        fill=(126, 45, 34),
        font=_font(21),
    )
    draw.text(
        (_MARGIN, 92),
        "Quadrats were step-located, acquired 22 days after TLS, and published for plot averages.",
        fill=(57, 63, 59),
        font=_font(19),
    )

    panels: list[dict[str, Any]] = []
    for index, quadrat in enumerate(quadrats):
        row, column = divmod(index, 2)
        left = _MARGIN + column * (_PANEL_W + _GAP)
        top = _HEADER_H + _MARGIN + row * panel_h
        with Image.open(Path(str(quadrat["photo_path"]))) as opened:
            source_dimensions = list(opened.size)
            image = ImageOps.exif_transpose(opened).convert("RGB")
        image.thumbnail((_PANEL_W - 18, _PHOTO_H - 18), Image.Resampling.LANCZOS)
        frame = (left, top, left + _PANEL_W - 1, top + _PHOTO_H - 1)
        draw.rectangle(frame, fill=(218, 216, 208), outline=(65, 76, 70), width=1)
        canvas.paste(
            image,
            (
                left + (_PANEL_W - image.width) // 2,
                top + (_PHOTO_H - image.height) // 2,
            ),
        )

        fractions = quadrat["target_fractions"]
        label_y = top + _PHOTO_H + 10
        draw.text(
            (left + 6, label_y),
            f"quadrat {quadrat['quadrat']} | 1 m x 1 m | publisher annotations",
            fill=(18, 30, 27),
            font=_font(20),
        )
        bar_left, bar_top = left + 6, label_y + 36
        bar_width, bar_height = _PANEL_W - 12, 28
        cursor = bar_left
        ordered = ("moss_or_peat", "litter", "living_vegetation", "unknown")
        for class_name in ordered:
            value = float(fractions[class_name])
            width = round(bar_width * value)
            if class_name == ordered[-1]:
                width = bar_left + bar_width - cursor
            if width > 0:
                draw.rectangle(
                    (cursor, bar_top, cursor + width - 1, bar_top + bar_height),
                    fill=CLASS_COLORS[class_name],
                )
            cursor += width
        draw.rectangle(
            (bar_left, bar_top, bar_left + bar_width, bar_top + bar_height),
            outline=(35, 42, 38),
            width=1,
        )
        legend = (
            f"moss {fractions['moss_or_peat']:.0%}  |  litter {fractions['litter']:.0%}  |  "
            f"vascular {fractions['living_vegetation']:.0%}  |  lichen->unknown {fractions['unknown']:.0%}"
        )
        draw.text((left + 6, bar_top + 38), legend, fill=(36, 44, 40), font=_font(17))
        draw.text(
            (left + 6, bar_top + 70),
            "mineral / roots / deadwood / clasts / water: NOT MEASURED (not zero)",
            fill=(126, 45, 34),
            font=_font(17),
        )
        draw.text(
            (left + 6, bar_top + 100),
            "TLS cell authority: 0 | all scan/view/cell semantics remain unknown",
            fill=(126, 45, 34),
            font=_font(17),
        )
        panels.append(
            {
                "panel": index + 1,
                "quadrat": quadrat["quadrat"],
                "source_photo_sha256": quadrat["photo_sha256"],
                "source_dimensions_px": source_dimensions,
                "rendered_dimensions_px": list(image.size),
                "target_fractions": dict(fractions),
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)
    return panels
