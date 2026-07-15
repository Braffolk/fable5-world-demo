"""Single decision-useful diagnostic for the v2 edge-corrected gate."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLORS = ((38, 112, 142), (166, 158, 112), (190, 93, 45))


def render_evidence_qa(
    destination: Path,
    *,
    recipe_id: str,
    document: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> None:
    width, height = 1600, 940
    canvas = Image.new("RGB", (width, height), "#d8d2c4")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    result = document["result"]
    draw.text((24, 18), "Moore v2 edge-corrected typed evidence", fill="#111916", font=font)
    draw.text(
        (24, 38),
        f"result={result}  recipe={recipe_id[:16]}...  independent unit=geographic group",
        fill="#7b241c" if result != "pass" else "#1d6338",
        font=font,
    )
    draw.text(
        (24, 58),
        "Exact HuHoLa 0.50 m cells; D4 translation correction; no whole-form, padding, training, synthesis, or Estonia-transfer credit.",
        fill="#34423b",
        font=font,
    )

    groups = document["groups"]
    fractions = arrays["class_fractions"]
    chart = (24, 92, 760, 396)
    draw.rectangle(chart, fill="#eee9dc", outline="#686f69")
    draw.text((36, 104), "A. Class fractions by group", fill="#17201d", font=font)
    bar_width = 76
    baseline = 352
    for group_index, group in enumerate(groups):
        x0 = 48 + group_index * 88
        bottom = baseline
        for class_index, color in enumerate(COLORS):
            segment = 220 * float(fractions[group_index, class_index])
            draw.rectangle((x0, bottom - segment, x0 + bar_width * 0.58, bottom), fill=color)
            bottom -= segment
        draw.text((x0, baseline + 8), group[:10], fill="#26312c", font=font)
        counts = arrays["class_counts"][group_index]
        draw.text((x0, 374), f"{counts[0]}/{counts[1]}/{counts[2]}", fill="#26312c", font=font)

    support = arrays["support_by_pair"]
    orbits = document["orbits"]
    support_chart = (790, 92, 1576, 396)
    draw.rectangle(support_chart, fill="#eee9dc", outline="#686f69")
    draw.text((802, 104), "B. Minimum transition-denominator support across all 9 class pairs", fill="#17201d", font=font)
    for index, orbit in enumerate(orbits):
        x = 820 + index * 80
        value = int(np.min(support[index]))
        top = 350 - value * 26
        color = "#2c7049" if value >= 6 else "#9b3c2f"
        draw.rectangle((x, top, x + 44, 350), fill=color)
        draw.text((x + 16, top - 16), str(value), fill="#26312c", font=font)
        draw.text((x, 360), f"{orbit[0]},{orbit[1]}", fill="#26312c", font=font)
    draw.line((810, 350 - 6 * 26, 1560, 350 - 6 * 26), fill="#111916", width=2)
    draw.text((804, 350 - 6 * 26 - 17), "frozen six-group threshold", fill="#111916", font=font)

    radial = (24, 426, 1576, 780)
    draw.rectangle(radial, fill="#eee9dc", outline="#686f69")
    draw.text((36, 438), "C. Group-balanced ordered same-class transition probability by radial orbit", fill="#17201d", font=font)
    mean_transitions = arrays["group_balanced_transitions"]
    distances = np.asarray(document["orbit_distance_m"], dtype=np.float64)
    x0, x1, y0, y1 = 86, 1538, 486, 742
    draw.line((x0, y1, x1, y1), fill="#5f675f", width=1)
    draw.line((x0, y0, x0, y1), fill="#5f675f", width=1)
    for tick in range(6):
        y = y1 - tick * (y1 - y0) / 5
        draw.line((x0 - 5, y, x0, y), fill="#5f675f")
        draw.text((48, y - 6), f"{tick / 5:.1f}", fill="#26312c", font=font)
    for class_index, (name, color) in enumerate(zip(document["classes"], COLORS, strict=True)):
        points = []
        for orbit_index, distance in enumerate(distances):
            value = mean_transitions[orbit_index, class_index, class_index]
            if not np.isfinite(value):
                continue
            x = x0 + (float(distance) - 0.5) / 1.5 * (x1 - x0)
            y = y1 - float(value) * (y1 - y0)
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        for point in points:
            draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=color)
        draw.rectangle((1020 + class_index * 150, 450, 1034 + class_index * 150, 464), fill=color)
        draw.text((1040 + class_index * 150, 451), name, fill="#26312c", font=font)

    draw.text((24, 806), "D. Frozen gate failures", fill="#17201d", font=font)
    for index, failure in enumerate(document["failures"]):
        draw.text((42, 830 + index * 20), f"- {failure}", fill="#7b241c", font=font)
    draw.text(
        (790, 826),
        "Cell-count labels in A are hollow/lawn/hummock. Missing curves are unsupported, not zero.",
        fill="#34423b",
        font=font,
    )
    draw.text(
        (790, 850),
        "The horizontal line in B is acceptance; every ordered class-pair denominator must reach it.",
        fill="#34423b",
        font=font,
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
