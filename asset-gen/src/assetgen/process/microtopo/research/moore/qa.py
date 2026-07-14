"""The five frozen, decision-relevant Moore materialization diagnostics."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .contracts import sha256_file


WIDTH = 1440
BACKGROUND = "#f3efe4"
INK = "#15292d"
WARNING = "#9d3325"
UNKNOWN = np.asarray((53, 58, 58), dtype=np.uint8)


def _field(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    symmetric: bool = False,
    symmetric_extent_m: float | None = None,
) -> Image.Image:
    rgb = np.empty((*values.shape, 3), dtype=np.uint8)
    rgb[:] = UNKNOWN
    finite = valid & np.isfinite(values)
    if np.any(finite):
        if symmetric:
            extent = symmetric_extent_m or max(float(np.quantile(np.abs(values[finite]), 0.98)), 1e-12)
            scaled = np.clip(0.5 + values / (2.0 * extent), 0.0, 1.0)
        else:
            lo, hi = np.quantile(values[finite], (0.02, 0.98))
            scaled = np.clip((values - lo) / max(float(hi - lo), 1e-12), 0.0, 1.0)
        rgb[finite, 0] = (31 + 205 * scaled[finite]).astype(np.uint8)
        rgb[finite, 1] = (68 + 153 * (1.0 - np.abs(2.0 * scaled[finite] - 1.0))).astype(np.uint8)
        rgb[finite, 2] = (116 + 107 * (1.0 - scaled[finite])).astype(np.uint8)
    return Image.fromarray(np.flipud(rgb), "RGB")


def _mask(mask: np.ndarray) -> Image.Image:
    rgb = np.empty((*mask.shape, 3), dtype=np.uint8)
    rgb[:] = (92, 48, 42)
    rgb[mask] = (35, 145, 91)
    return Image.fromarray(np.flipud(rgb), "RGB")


def _fit(image: Image.Image, width: int, height: int) -> Image.Image:
    result = image.copy()
    result.thumbnail((width, height), Image.Resampling.NEAREST)
    if result.width < width // 2 and result.height < height // 2:
        factor = max(1, min(width // result.width, height // result.height))
        result = result.resize((result.width * factor, result.height * factor), Image.Resampling.NEAREST)
    return result


def _document(
    title: str,
    subtitle: str,
    panels: list[tuple[str, Image.Image]],
    path: Path,
    *,
    footer: str,
) -> None:
    margin, gap, top, panel_h = 34, 22, 104, 760
    panel_w = (WIDTH - 2 * margin - gap * (len(panels) - 1)) // len(panels)
    fitted = [(label, _fit(image, panel_w, panel_h)) for label, image in panels]
    height = max(image.height for _, image in fitted)
    canvas = Image.new("RGB", (WIDTH, top + height + 112), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 18), title, fill=INK)
    draw.text((margin, 48), subtitle, fill=WARNING)
    x = margin
    for label, image in fitted:
        draw.text((x, top - 24), label, fill=INK)
        canvas.paste(image, (x, top))
        draw.rectangle((x - 1, top - 1, x + image.width, top + image.height), outline="#405052")
        x += panel_w + gap
    draw.text((margin, top + height + 28), footer, fill=WARNING)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", compress_level=9)


def _group_table(group_rows: list[dict[str, Any]]) -> Image.Image:
    image = Image.new("RGB", (900, 620), "#faf8f0")
    draw = ImageDraw.Draw(image)
    draw.text((25, 18), "Geographic-group measured support", fill=INK)
    headers = "group                 plots   B2 cells   cores   source-gate"
    draw.text((25, 60), headers, fill=INK)
    for index, row in enumerate(group_rows):
        y = 98 + 58 * index
        state = row.get("state", "not_evaluated")
        color = "#237c50" if state == "passed" else "#9d3325" if state == "failed" else "#515c5e"
        text = (
            f"{row['group_id']:<21} {row['plot_count']:>3} "
            f"{row.get('common_b2_cells', 0):>10} {row.get('spectral_core_count', 0):>7}   {state}"
        )
        draw.text((25, y), text, fill=color)
    return image


def render_qa(
    qa_root: Path,
    *,
    build_id: str,
    design_sha256: str,
    archive_sha256: str,
    preview: Any,
    group_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source = preview.source
    phases = preview.phases
    p0 = phases[0]
    four_phase_values = np.concatenate(
        [item.derived_b2.values[item.derived_b2.valid] for item in phases]
    )
    four_phase_extent = (
        max(float(np.quantile(np.abs(four_phase_values), 0.98)), 1e-12)
        if four_phase_values.size
        else 1e-12
    )
    outputs = [
        (
            "01_source_height_and_validity.png",
            "Moore source height and measured finite mask",
            f"{source.plot_id} / {source.group_id}; publisher operator is bound before resampling.",
            [("source height (m)", _field(source.z_m, np.isfinite(source.z_m))),
             ("finite source cells", _mask(np.isfinite(source.z_m)))],
            None,
            "Source is an interpolated/filtered visible moss-surface hypothesis, not 1 cm truth.",
        ),
        (
            "02_b1_height_band.png",
            "Registered A1 and strict measured-only B1",
            "No reflection or invented boundary context receives geometry-loss validity.",
            [("A1=F4(H), 0.25 m", _field(p0.a1.values, p0.a1.valid)),
             ("B1=A1-R4(A0)", _field(p0.b1.values, p0.b1.valid, symmetric=True))],
            0,
            "Empty B1 is an evidence-capacity result; it is not repaired by padding.",
        ),
        (
            "03_derived_b2_hypothesis.png",
            "Derived B2 hypothesis on complete measured support",
            "H-R4(F4(H)); diagnostic source hypothesis, never directly measured B2.",
            [("derived B2 (m)", _field(p0.derived_b2.values, p0.derived_b2.valid, symmetric=True)),
             ("complete B2 support", _mask(p0.derived_b2.valid))],
            0,
            "This diagnostic cannot establish target truth, acquisition MTF, or Estonia transfer.",
        ),
        (
            "04_four_phase_b2_stability.png",
            "All four half-source-cell B2 phases",
            "Same palette policy; no favorable phase selection or phase interpolation.",
            [(f"phase {item.phase.phase_index} {item.phase.offset_xy_m} m",
              _field(
                  item.derived_b2.values, item.derived_b2.valid,
                  symmetric=True, symmetric_extent_m=four_phase_extent,
              )) for item in phases],
            "all",
            "Metrics use exact overlap of half-open cells on four-phase common support.",
        ),
        (
            "05_valid_windows_and_groups.png",
            "Valid analysis support and leakage groups",
            "Fixed 2 m support / 1 m core / 0.5 m stride; geographic group is indivisible.",
            [("phase-0 complete common cells", _mask(preview.complete_common_cells[0])),
             ("eight frozen groups", _group_table(group_rows))],
            "all",
            "Research-only capacity screen; no production owner, transfer, or release claim.",
        ),
    ]
    records: list[dict[str, Any]] = []
    for name, title, subtitle, panels, phase, footer in outputs:
        path = qa_root / name
        _document(title, subtitle, panels, path, footer=footer)
        with Image.open(path) as image:
            dimensions = list(image.size)
        records.append(
            {
                "path": f"qa/{name}",
                "image_sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "source_and_recipe_sha256": {
                    "archive": archive_sha256,
                    "design": design_sha256,
                    "recipe": build_id,
                },
                "dimensions": dimensions,
                "units": "metres for height/relief; boolean for masks",
                "plot_and_group_ids": {"plot": source.plot_id, "group": source.group_id},
                "phase": phase,
                "interpretation": subtitle,
                "claim_boundary": footer,
            }
        )
    return records
