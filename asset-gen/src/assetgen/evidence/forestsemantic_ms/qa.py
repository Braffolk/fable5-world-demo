"""Small labeled QA set for ForestSemantic-MS semantic evidence."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_CLASS_COLORS = np.asarray(
    (
        (151, 105, 66),   # ground
        (89, 170, 78),    # low vegetation
        (117, 77, 45),    # trunk
        (211, 155, 71),   # branches
        (36, 111, 62),    # foliage
        (55, 121, 151),   # woody debris
    ),
    dtype=np.uint8,
)
_CLASS_NAMES = ("ground", "low vegetation", "trunk", "branches", "foliage", "woody debris")


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _fit_grid(rgb: np.ndarray, size: tuple[int, int]) -> Image.Image:
    image = Image.fromarray(np.flipud(rgb), mode="RGB")
    image.thumbnail(size, Image.Resampling.NEAREST)
    return image


def _plot_panel(
    name: str, split: str, counts: np.ndarray, *, mode: str, width: int = 560, height: int = 390
) -> Image.Image:
    observed = counts.sum(axis=0) > 0
    if mode == "dominant":
        rgb = _CLASS_COLORS[np.argmax(counts, axis=0)]
        rgb = rgb.copy()
        detail = "dominant publisher label per 0.25 m cell"
    else:
        relevant = counts[[0, 1, 5]].astype(np.float64)
        scale = np.maximum(np.quantile(relevant, 0.99, axis=(1, 2)), 1.0)
        intensity = np.log1p(relevant) / np.log1p(scale[:, None, None])
        rgb = np.zeros((*observed.shape, 3), dtype=np.float64)
        rgb[..., 0] = 0.82 * intensity[0] + 0.65 * intensity[2]
        rgb[..., 1] = 0.72 * intensity[0] + 0.92 * intensity[1]
        rgb[..., 2] = 0.30 * intensity[0] + 0.95 * intensity[2]
        rgb = np.rint(255 * np.clip(rgb, 0.0, 1.0)).astype(np.uint8)
        detail = "ground=ochre, low vegetation=green, woody debris=blue"
    rgb[~observed] = (18, 20, 22)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), f"{name} | publisher {split.upper()} split", fill="black", font=_font())
    draw.text((12, 29), detail, fill=(65, 65, 65), font=_font())
    image = _fit_grid(rgb, (width - 24, height - 66))
    canvas.paste(image, ((width - image.width) // 2, 55))
    return canvas


def _contact(path: Path, panels: list[Image.Image], footer: str) -> None:
    cols = 2
    rows = (len(panels) + cols - 1) // cols
    footer_h = 54
    width = cols * panels[0].width
    height = rows * panels[0].height + footer_h
    canvas = Image.new("RGB", (width, height), "white")
    for index, panel in enumerate(panels):
        canvas.paste(panel, ((index % cols) * panel.width, (index // cols) * panel.height))
    ImageDraw.Draw(canvas).text(
        (12, rows * panels[0].height + 17), footer, fill=(105, 20, 25), font=_font()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=True)


def _summary_chart(path: Path, inventories: list[Mapping[str, Any]]) -> None:
    width, height = 1500, 900
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 24), "ForestSemantic-MS | actual labeled support and split audit", fill="black")
    draw.text(
        (28, 49),
        "Counts are publisher labels, not accuracy estimates. Density is points / axis-aligned file extent.",
        fill=(70, 70, 70),
    )
    left, top, chart_w, row_h = 260, 105, 1160, 105
    max_points = max(item["point_count"] for item in inventories)
    for row, item in enumerate(inventories):
        y = top + row * row_h
        draw.text((28, y + 12), f"{item['filename']} ({item['split']})", fill="black")
        total = item["point_count"]
        x = left
        for class_id, count in enumerate(item["semantic_counts"]):
            length = int(chart_w * count / max_points)
            draw.rectangle((x, y, x + length, y + 34), fill=tuple(_CLASS_COLORS[class_id]))
            x += length
        relevant = item["terrain_relevant_density_per_m2"]
        draw.text(
            (left, y + 47),
            (
                f"n={total:,}  ground={item['semantic_counts'][0]:,} ({relevant['ground']:.1f}/m2)  "
                f"low={item['semantic_counts'][1]:,} ({relevant['low_vegetation']:.1f}/m2)  "
                f"woody={item['semantic_counts'][5]:,} ({relevant['woody_debris']:.2f}/m2)"
            ),
            fill=(45, 45, 45),
        )
    legend_y = 760
    x = 28
    for index, name in enumerate(_CLASS_NAMES):
        draw.rectangle((x, legend_y, x + 20, legend_y + 16), fill=tuple(_CLASS_COLORS[index]))
        draw.text((x + 26, legend_y), name, fill="black")
        x += 200
    draw.text(
        (28, 820),
        "BOUNDARY: one Espoonlahti HeliALS corpus; no CRS VLR, view IDs, return structure, positional error, or target-band transfer.",
        fill=(105, 20, 25),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=True)


def render_qa(
    plots: list[tuple[Mapping[str, Any], np.ndarray]], qa_dir: Path
) -> tuple[dict[str, Any], ...]:
    dominant = qa_dir / "01_publisher_semantic_label_maps.png"
    _contact(
        dominant,
        [_plot_panel(item["filename"], item["split"], counts, mode="dominant") for item, counts in plots],
        "DIRECT LABEL DISPLAY / NO INTERPOLATION / DARK = NO RETURN / NORTH OR CRS ORIENTATION NOT CLAIMED",
    )
    relevant = qa_dir / "02_ground_low_vegetation_woody_support.png"
    _contact(
        relevant,
        [_plot_panel(item["filename"], item["split"], counts, mode="relevant") for item, counts in plots],
        "SEMANTIC SUPPORT, NOT SURFACE TRUTH / COLOR MIXING EXPOSES WITHIN-CELL CLASS COMPETITION",
    )
    summary = qa_dir / "03_split_class_support_summary.png"
    _summary_chart(summary, [item for item, _ in plots])
    return (
        {
            "path": dominant,
            "interpretation": "Direct 0.25 m cell maps show the spatial distribution of all six publisher semantic labels without creating an interpolated terrain surface.",
        },
        {
            "path": relevant,
            "interpretation": "Ground, low-vegetation, and woody-debris support is displayed jointly so mixed cells and sparse woody-debris supervision remain visible.",
        },
        {
            "path": summary,
            "interpretation": "Per-file class counts and axis-aligned density estimates expose train/test preservation, class imbalance, and the limited woody-debris test support.",
        },
    )
