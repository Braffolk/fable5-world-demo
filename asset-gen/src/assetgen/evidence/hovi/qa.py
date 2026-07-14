"""Small labeled QA views for Hovi raw-observation support grids."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

from .support import SupportGrid

_PANEL_W = 420
_PANEL_H = 380
_LABEL_H = 52
_TITLE_H = 42


def _sequential_rgb(values: np.ndarray) -> np.ndarray:
    value = np.clip(values.astype(np.float32, copy=False), 0.0, 1.0)
    stops = np.asarray(
        (
            (5, 10, 24),
            (18, 74, 99),
            (28, 151, 138),
            (153, 213, 111),
            (252, 232, 94),
        ),
        dtype=np.float32,
    )
    position = value * (len(stops) - 1)
    lower = np.floor(position).astype(np.intp)
    upper = np.minimum(lower + 1, len(stops) - 1)
    weight = (position - lower)[..., None]
    rgb = stops[lower] * (1.0 - weight) + stops[upper] * weight
    return np.rint(rgb).astype(np.uint8)


def _count_image(grid: SupportGrid) -> np.ndarray:
    counts = grid.point_count
    occupied = counts > 0
    value = np.zeros(counts.shape, dtype=np.float32)
    if np.any(occupied):
        logged = np.log1p(counts[occupied].astype(np.float32))
        ceiling = float(np.quantile(logged, 0.99))
        if ceiling > 0.0:
            value[occupied] = np.minimum(logged / ceiling, 1.0)
    return _sequential_rgb(np.flipud(value))


def _distance_image(grid: SupportGrid) -> np.ndarray:
    distance = grid.nearest_in_cell_center_distance_m
    occupied = np.isfinite(distance)
    value = np.zeros(distance.shape, dtype=np.float32)
    maximum = grid.resolution_m / np.sqrt(2.0)
    value[occupied] = np.minimum(distance[occupied] / maximum, 1.0)
    rgb = _sequential_rgb(np.flipud(value))
    rgb[np.flipud(~occupied)] = (112, 22, 36)
    return rgb


def _render_multiscale(
    grids: tuple[SupportGrid, ...],
    output: Path,
    *,
    title: str,
    image_fn: Callable[[SupportGrid], np.ndarray],
    detail_fn: Callable[[SupportGrid], str],
) -> None:
    columns = 3
    rows = (len(grids) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * _PANEL_W, _TITLE_H + rows * _PANEL_H), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), title, fill="black")
    for index, grid in enumerate(grids):
        column = index % columns
        row = index // columns
        left = column * _PANEL_W
        top = _TITLE_H + row * _PANEL_H
        occupied = int(np.count_nonzero(grid.point_count))
        label = (
            f"{grid.resolution_m:g} m | {grid.width} x {grid.height} | "
            f"occupied {occupied / grid.point_count.size:.2%}\n{detail_fn(grid)}"
        )
        draw.text((left + 8, top + 5), label, fill="black")
        image = Image.fromarray(image_fn(grid), mode="RGB")
        available_w = _PANEL_W - 16
        available_h = _PANEL_H - _LABEL_H - 10
        scale = min(available_w / image.width, available_h / image.height)
        target = (
            max(1, int(round(image.width * scale))),
            max(1, int(round(image.height * scale))),
        )
        image = image.resize(target, resample=Image.Resampling.NEAREST)
        image_left = left + (_PANEL_W - image.width) // 2
        image_top = top + _LABEL_H + (available_h - image.height) // 2
        canvas.paste(image, (image_left, image_top))
        draw.rectangle(
            (image_left - 1, image_top - 1, image_left + image.width, image_top + image.height),
            outline=(70, 70, 70),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def render_support_qa(
    grids: tuple[SupportGrid, ...], qa_dir: Path, *, plot_id: str
) -> tuple[tuple[Path, str], ...]:
    """Render two diagnostics; neither image represents a reconstructed ground surface."""
    count_path = qa_dir / "01_observed_point_count_multiscale.png"
    distance_path = qa_dir / "02_in_cell_center_proximity_multiscale.png"
    _render_multiscale(
        grids,
        count_path,
        title=f"{plot_id}: raw observed point count (log scale, north up)",
        image_fn=_count_image,
        detail_fn=lambda grid: f"max {int(grid.point_count.max(initial=0)):,} points/cell",
    )
    _render_multiscale(
        grids,
        distance_path,
        title=(
            f"{plot_id}: nearest observed point to cell center, within occupied cell only "
            "(north up)"
        ),
        image_fn=_distance_image,
        detail_fn=lambda grid: (
            f"red = empty; scale 0..{grid.resolution_m / np.sqrt(2.0):.4g} m"
        ),
    )
    return (
        (
            count_path,
            "Raw all-return point count per grid cell, log-scaled independently at each "
            "resolution. It is observation density, not forest-floor or ground support.",
        ),
        (
            distance_path,
            "Exact minimum planar distance from each occupied cell center to a decoded point "
            "assigned to that same cell. Red cells have no observation. This is not "
            "point-to-point spacing and does not estimate scan independence or occlusion.",
        ),
    )
