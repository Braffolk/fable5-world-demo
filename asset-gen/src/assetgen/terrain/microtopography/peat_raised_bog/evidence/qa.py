"""One preregistered Moore whole-form capacity diagnostic."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .huhola import CLASS_HOLLOW, CLASS_HUMMOCK, CLASS_LAWN, ClassifiedPlot


COLORS = {
    CLASS_LAWN: np.asarray((166, 158, 112), dtype=np.uint8),
    CLASS_HOLLOW: np.asarray((38, 112, 142), dtype=np.uint8),
    CLASS_HUMMOCK: np.asarray((190, 93, 45), dtype=np.uint8),
}


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _class_image(plot: ClassifiedPlot) -> Image.Image:
    rgb = np.full(plot.classes.shape + (3,), (27, 31, 29), dtype=np.uint8)
    for class_id, color in COLORS.items():
        rgb[plot.valid & (plot.classes == class_id)] = color
    boundary = plot.valid & ~plot.interior_valid
    rgb[boundary] = (rgb[boundary].astype(np.uint16) * 45 // 100).astype(np.uint8)
    return Image.fromarray(rgb, "RGB").transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def _plot_tile(
    plot_id: str,
    plot: ClassifiedPlot,
    forms: list[dict[str, Any]],
    size: tuple[int, int],
) -> Image.Image:
    tile = Image.new("RGB", size, "#e8e2d4")
    draw = ImageDraw.Draw(tile)
    draw.text((8, 7), f"{plot_id}: {len(forms)} whole forms", fill="#17201d", font=_font())
    image = _class_image(plot)
    max_plot = (size[0] - 16, size[1] - 36)
    scale = min(max_plot[0] / max(image.width, 1), max_plot[1] / max(image.height, 1))
    image = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.NEAREST,
    )
    offset = ((size[0] - image.width) // 2, 30 + (max_plot[1] - image.height) // 2)
    tile.paste(image, offset)
    if plot.x_bounds_m.size > 1 and plot.y_bounds_m.size > 1:
        x0, x1 = float(plot.x_bounds_m[0]), float(plot.x_bounds_m[-1])
        y0, y1 = float(plot.y_bounds_m[0]), float(plot.y_bounds_m[-1])
        for form in forms:
            px = offset[0] + (float(form["centroid_east_m"]) - x0) / (x1 - x0) * image.width
            py = offset[1] + (1.0 - (float(form["centroid_north_m"]) - y0) / (y1 - y0)) * image.height
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), outline="#ffffff", width=2)
    return tile


def render_capacity_qa(
    destination: Path,
    *,
    recipe_id: str,
    result: str,
    group_rows: list[dict[str, Any]],
    representative_plots: dict[str, tuple[str, ClassifiedPlot, list[dict[str, Any]]]],
    minimum_forms: int,
    minimum_groups: int,
) -> None:
    width, height = 1600, 920
    canvas = Image.new("RGB", (width, height), "#d8d2c4")
    draw = ImageDraw.Draw(canvas)
    total_forms = sum(int(row["whole_form_count"]) for row in group_rows)
    used_groups = sum(int(row["whole_form_count"]) > 0 for row in group_rows)
    draw.text((24, 18), "Moore whole-form descriptor capacity", fill="#111916", font=_font())
    draw.text(
        (24, 38),
        f"result={result}  forms={total_forms}/{minimum_forms}  groups={used_groups}/{minimum_groups}  recipe={recipe_id[:16]}...",
        fill="#7b241c" if result != "pass" else "#1d6338",
        font=_font(),
    )
    draw.text(
        (24, 58),
        "0.50 m HuHoLa, threshold 0.04 m; dark cells are the one-cell boundary exclusion. White rings mark retained whole forms.",
        fill="#34423b",
        font=_font(),
    )

    chart = (24, 92, 1576, 330)
    draw.rectangle(chart, fill="#eee9dc", outline="#686f69")
    max_count = max([minimum_forms, *(int(row["whole_form_count"]) for row in group_rows)], default=1)
    bar_width = (chart[2] - chart[0] - 72) / max(len(group_rows), 1)
    baseline = chart[3] - 42
    chart_top = chart[1] + 54
    chart_height = baseline - chart_top
    for index, row in enumerate(group_rows):
        x0 = chart[0] + 44 + index * bar_width
        counts = row["whole_form_count_by_class"]
        bottom = baseline
        for name, color in (
            ("lawn", tuple(COLORS[CLASS_LAWN])),
            ("hollow", tuple(COLORS[CLASS_HOLLOW])),
            ("hummock", tuple(COLORS[CLASS_HUMMOCK])),
        ):
            count = int(counts[name])
            segment = chart_height * count / max(max_count, 1)
            draw.rectangle((x0, bottom - segment, x0 + bar_width * 0.56, bottom), fill=color)
            bottom -= segment
        draw.text((x0, baseline + 5), str(row["group_id"])[:12], fill="#26312c", font=_font())
        draw.text(
            (x0, max(chart_top, bottom - 15)),
            str(row["whole_form_count"]),
            fill="#26312c",
            font=_font(),
        )
    draw.text((chart[0] + 8, chart[1] + 8), "Retained whole forms by independent geographic group", fill="#17201d", font=_font())

    tile_width, tile_height = 380, 250
    for index, row in enumerate(group_rows):
        group = str(row["group_id"])
        x = 24 + (index % 4) * (tile_width + 12)
        y = 350 + (index // 4) * (tile_height + 16)
        if group in representative_plots:
            plot_id, plot, forms = representative_plots[group]
            tile = _plot_tile(f"{group}/{plot_id}", plot, forms, (tile_width, tile_height))
        else:
            tile = Image.new("RGB", (tile_width, tile_height), "#e8e2d4")
            ImageDraw.Draw(tile).text((12, 12), f"{group}: no classifiable plot", fill="#7b241c", font=_font())
        canvas.paste(tile, (x, y))

    draw.rectangle((24, 890, 38, 904), fill=tuple(COLORS[CLASS_LAWN]))
    draw.text((44, 891), "lawn", fill="#26312c", font=_font())
    draw.rectangle((100, 890, 114, 904), fill=tuple(COLORS[CLASS_HOLLOW]))
    draw.text((120, 891), "hollow", fill="#26312c", font=_font())
    draw.rectangle((188, 890, 202, 904), fill=tuple(COLORS[CLASS_HUMMOCK]))
    draw.text((208, 891), "hummock", fill="#26312c", font=_font())
    draw.text(
        (340, 891),
        "Diagnostic only: no height supervision, pixelwise training, synthesis, Estonia transfer, production, or preview authority.",
        fill="#34423b",
        font=_font(),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)


def image_record(path: Path, interpretation: str) -> dict[str, Any]:
    encoded = path.read_bytes()
    with Image.open(path) as image:
        dimensions = [image.width, image.height]
    return {
        "path": path.name,
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "dimensions_px": dimensions,
        "interpretation": interpretation,
    }
