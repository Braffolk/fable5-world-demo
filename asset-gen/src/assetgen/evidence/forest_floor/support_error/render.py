"""One compact visual evidence ladder for the fail-closed decision."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1500
BACKGROUND = "#f4f0e6"
INK = "#18251f"
MUTED = "#5c685f"
PASS = "#3f7657"
BLOCK = "#a54332"
UNKNOWN = "#b9842e"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default()


def render_evidence_ladder(decision: dict[str, Any], path: Path) -> None:
    rows = [
        ("Raw metric geometry", "PRESENT", PASS, "Hovi full scans retain 16 physical scan identities"),
        ("Unique 6.25 cm sheet", "INSUFFICIENT", UNKNOWN, "234 / 4,096 sole raw hypotheses in the audited shard"),
        ("Metric surface semantics", "MISSING", BLOCK, "Photos and cover fractions are not registered geometric labels"),
        ("Independent XY/Z control", "MISSING", BLOCK, "No checkpoint error independent of registration/conversion"),
        ("Held-out same-surface reconstructions", "MISSING", BLOCK, "No semantic surface exists to reconstruct by scan split"),
        ("B1 0.25-1.0 m", "ABSTAIN", BLOCK, "Signal/error ratio and transfer gain are unidentifiable"),
        ("B2 0.125-0.25 m", "ABSTAIN", BLOCK, "Signal/error ratio and transfer gain are unidentifiable"),
    ]
    height = 260 + len(rows) * 92 + 105
    image = Image.new("RGB", (WIDTH, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    title = _font(42, bold=True)
    subtitle = _font(22)
    heading = _font(23, bold=True)
    body = _font(20)
    status_font = _font(18, bold=True)

    draw.text((70, 52), "Forest-floor B1/B2 evidence ladder", font=title, fill=INK)
    draw.text(
        (70, 112),
        "HY_SPRUCE4 + retained Evo candidates | fail-closed support/error qualification",
        font=subtitle,
        fill=MUTED,
    )
    draw.rounded_rectangle((70, 170, WIDTH - 70, 228), radius=12, fill="#e5d8c4")
    draw.text(
        (92, 184),
        "Decision: TARGET EVIDENCE INSUFFICIENT — raw variation is not yet separable from total error",
        font=heading,
        fill=INK,
    )

    y = 270
    for label, status, color, explanation in rows:
        draw.line((70, y + 76, WIDTH - 70, y + 76), fill="#d5cdbf", width=2)
        draw.text((82, y), label, font=heading, fill=INK)
        draw.rounded_rectangle((510, y - 2, 710, y + 37), radius=9, fill=color)
        status_width = draw.textbbox((0, 0), status, font=status_font)[2]
        draw.text((610 - status_width / 2, y + 6), status, font=status_font, fill="white")
        draw.text((750, y + 2), explanation, font=body, fill=MUTED)
        y += 92

    draw.text(
        (70, y + 18),
        "Unknown is preserved: no interpolation, threshold tuning, or conversion residual is promoted to truth.",
        font=body,
        fill=INK,
    )
    draw.text(
        (70, y + 52),
        f"Machine record schema: {decision['schema_version']}",
        font=_font(16),
        fill=MUTED,
    )
    image.save(path, format="PNG", optimize=False)
