"""Five decision-relevant diagnostics for the frozen pre-training gate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .bundle import FrozenBundle, sha256_file
from .contracts import ResearchSurfaceEvidence, SemanticResult


WIDTH = 1400


def _fit(image: Image.Image, width: int, height: int) -> Image.Image:
    copy = image.convert("RGB")
    copy.thumbnail((width, height), Image.Resampling.NEAREST)
    return copy


def _document(title: str, subtitle: str, panels: list[tuple[str, Image.Image]], path: Path) -> None:
    margin, gap, top, panel_h = 32, 24, 86, 720
    panel_w = (WIDTH - 2 * margin - gap * (len(panels) - 1)) // len(panels)
    fitted = [(label, _fit(image, panel_w, panel_h)) for label, image in panels]
    actual_h = max(image.height for _, image in fitted)
    canvas = Image.new("RGB", (WIDTH, top + actual_h + 105), "#f4f1e8")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 18), title, fill="#16242a", stroke_width=0)
    draw.text((margin, 45), subtitle, fill="#8b2c20")
    x = margin
    for label, image in fitted:
        draw.text((x, top - 22), label, fill="#16242a")
        canvas.paste(image, (x, top))
        draw.rectangle((x - 1, top - 1, x + image.width, top + image.height), outline="#3f4c4e")
        x += panel_w + gap
    draw.text(
        (margin, top + actual_h + 28),
        "RESEARCH ONLY | no target truth | no production transfer | missing evidence remains unknown",
        fill="#8b2c20",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", compress_level=9)


def _field(values: np.ndarray, valid: np.ndarray, *, unknown=(55, 61, 62)) -> Image.Image:
    rgb = np.empty((*values.shape, 3), dtype=np.uint8)
    rgb[:] = unknown
    if np.any(valid):
        lo, hi = np.quantile(values[valid], (0.02, 0.98))
        scaled = np.clip((values - lo) / max(float(hi - lo), 1e-9), 0, 1)
        rgb[valid, 0] = (28 + 205 * scaled[valid]).astype(np.uint8)
        rgb[valid, 1] = (70 + 145 * (1 - np.abs(scaled[valid] - 0.5) * 2)).astype(np.uint8)
        rgb[valid, 2] = (120 + 100 * (1 - scaled[valid])).astype(np.uint8)
    return Image.fromarray(np.flipud(rgb), "RGB")


def _mask(mask: np.ndarray, yes=(35, 153, 92), no=(101, 47, 42)) -> Image.Image:
    rgb = np.empty((*mask.shape, 3), dtype=np.uint8)
    rgb[:] = no
    rgb[mask] = yes
    return Image.fromarray(np.flipud(rgb), "RGB")


def _semantic_summary(semantic: SemanticResult) -> Image.Image:
    canvas = Image.new("RGB", (1000, 620), "#f8f6ef")
    draw = ImageDraw.Draw(canvas)
    rows = [("train4 calibration", semantic.calibration)] + [
        (row["file"], row) for row in semantic.audits
    ]
    draw.text((24, 18), f"Frozen threshold: {semantic.threshold!r}", fill="#17272b")
    for index, (name, row) in enumerate(rows):
        y = 75 + index * 155
        passed = row.get("status") == "passed" or row.get("passed") is True
        color = "#238b57" if passed else "#aa3528"
        ratio = row.get("forbidden_ratio_ucb95")
        accepted = int(row.get("accepted_points", 0))
        draw.text((24, y), name, fill="#17272b")
        draw.rectangle((24, y + 28, 976, y + 70), fill="#dedbd1")
        width = min(952, int(952 * min(1.0, accepted / max(accepted, 100_000))))
        draw.rectangle((24, y + 28, 24 + width, y + 70), fill=color)
        draw.text(
            (35, y + 40),
            f"accepted={accepted:,}  forbidden UCB95={ratio!r}  {'PASS' if passed else 'FAIL'}",
            fill="white" if width > 500 else "#17272b",
        )
    return canvas


def render_qa(
    bundle: FrozenBundle,
    semantic: SemanticResult,
    hovi: ResearchSurfaceEvidence,
    evo: tuple[ResearchSurfaceEvidence, ResearchSurfaceEvidence],
    qa_root: Path,
) -> list[dict]:
    source_qa = next(
        path for path, _ in bundle.inputs if "hovi-full-scan-candidate-evidence" in path.as_posix()
    )
    source_index = json.loads(source_qa.read_bytes())
    by_prefix = {Path(row["path"]).name[:2]: source_qa.parent.parent / row["path"] for row in source_index["artifacts"]}
    outputs = [
        (
            "01_hovi_sheet_hypotheses.png",
            "Hovi competing raw sheet hypotheses",
            "The compact full-scan candidate is retained; no lowest-sheet or semantic shortcut selects it.",
            [("sole raw hypotheses", Image.open(by_prefix["01"])), ("candidate count / ambiguity", Image.open(by_prefix["02"]))],
        ),
        (
            "02_semantic_included_excluded_unknown.png",
            "Frozen common-geometry semantic exclusion audit",
            "Passing means exclusion-prior usability only; audit failure makes every Hovi cell unknown.",
            [("train4 then untouched test1/test2", _semantic_summary(semantic))],
        ),
        (
            "03_support_view_disagreement.png",
            "Hovi direct view support and disagreement",
            "Scanner support is descriptive until semantic, support, and error gates all pass.",
            [("physical scan/view support", Image.open(by_prefix["03"])), ("inter-scan disagreement", Image.open(by_prefix["04"]))],
        ),
        (
            "04_b1_b2_eligibility_and_partitions.png",
            "Independent B1/B2 eligibility and frozen partitions",
            f"B1={hovi.b1_eligible.sum()} cells; B2={hovi.b2_eligible.sum()} cells; tier={hovi.tier}.",
            [("B1 0.25-1 m", _mask(hovi.b1_eligible)), ("B2 0.125-0.25 m", _mask(hovi.b2_eligible)), ("known vs unknown", _mask(hovi.p_unknown == 0))],
        ),
        (
            "05_evo_1086_1065_adapter.png",
            "Unchanged Evo R2 adapters",
            "Finite converted candidates are shown, but semantics and physical-view count remain unknown.",
            [
                (evo[0].source_id, _field(evo[0].height_m, evo[0].direct_observed)),
                (evo[1].source_id, _field(evo[1].height_m, evo[1].direct_observed)),
            ],
        ),
    ]
    records = []
    for name, title, subtitle, panels in outputs:
        path = qa_root / name
        _document(title, subtitle, panels, path)
        with Image.open(path) as image:
            dimensions = list(image.size)
        records.append(
            {
                "path": f"qa/{name}",
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "dimensions_xy": dimensions,
                "interpretation": subtitle,
            }
        )
    return records
