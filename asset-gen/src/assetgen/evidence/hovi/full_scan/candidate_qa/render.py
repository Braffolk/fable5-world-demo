"""Content-addressed PNG diagnostics for one Hovi candidate-evidence shard."""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import textwrap
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

from ..authorization import canonical_json_bytes
from ..candidate import HypothesisConfig, VerifiedSpatialManifest
from .grid import (
    AMBIGUOUS_CANDIDATES,
    HYPOTHESIS_OVERFLOW,
    SOLE_RAW_CANDIDATE,
    UNOBSERVED,
    CandidateQaGrid,
)


_INDEX_SCHEMA = "hovi-full-scan-candidate-evidence-qa-index/1.0.0"
_TITLE = "RAW CANDIDATE / UNQUALIFIED / NO SYNTHESIS AUTHORIZATION"
_CANVAS_WIDTH = 1120
_MARGIN = 24
_TITLE_HEIGHT = 72
_LEGEND_HEIGHT = 76


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _palette(
    values: np.ndarray,
    stops: tuple[tuple[int, int, int], ...],
) -> np.ndarray:
    value = np.clip(np.nan_to_num(values, nan=0.0), 0.0, 1.0)
    colors = np.asarray(stops, dtype=np.float32)
    position = value * (len(colors) - 1)
    lower = np.floor(position).astype(np.intp)
    upper = np.minimum(lower + 1, len(colors) - 1)
    weight = (position - lower)[..., None]
    return np.rint(
        colors[lower] * (1.0 - weight) + colors[upper] * weight
    ).astype(np.uint8)


def _complete_stencil(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    result = mask.copy()
    for y_offset in range(3):
        for x_offset in range(3):
            result &= padded[
                y_offset : y_offset + mask.shape[0],
                x_offset : x_offset + mask.shape[1],
            ]
    return result


def _height_hillshade(grid: CandidateQaGrid) -> np.ndarray:
    height = grid.scan_balanced_height_m.astype(np.float64)
    sole = grid.unambiguous_mask & np.isfinite(height)
    visible = _complete_stencil(sole)
    rgb = np.full((*height.shape, 3), (28, 29, 31), dtype=np.uint8)
    rgb[sole & ~visible] = (26, 72, 73)
    if not np.any(visible):
        return rgb
    low, high = np.quantile(height[visible], (0.01, 0.99))
    normalized = np.clip((height - low) / max(float(high - low), 1e-9), 0.0, 1.0)
    colored = _palette(
        normalized,
        (
            (21, 45, 55),
            (38, 96, 89),
            (123, 145, 92),
            (190, 158, 100),
            (236, 222, 181),
        ),
    ).astype(np.float32)
    dx = np.zeros(height.shape, dtype=np.float64)
    dy = np.zeros(height.shape, dtype=np.float64)
    dx[:, 1:-1] = (height[:, 2:] - height[:, :-2]) / (2.0 * grid.resolution_m)
    dy[1:-1, :] = (height[2:, :] - height[:-2, :]) / (2.0 * grid.resolution_m)
    normal = np.dstack((-dx, -dy, np.ones(height.shape, dtype=np.float64)))
    normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-12)
    light = np.asarray((-0.45, 0.55, 0.70), dtype=np.float64)
    light /= np.linalg.norm(light)
    shade = np.clip(normal @ light, 0.0, 1.0)
    colored *= (0.42 + 0.72 * shade)[..., None]
    rgb[visible] = np.clip(colored[visible], 0.0, 255.0).astype(np.uint8)
    return rgb


def _hypothesis_count(grid: CandidateQaGrid) -> np.ndarray:
    count = grid.hypothesis_count.astype(np.float32)
    maximum = max(2.0, float(np.max(count, initial=0.0)))
    rgb = _palette(
        np.minimum(count / maximum, 1.0),
        ((31, 36, 41), (60, 155, 101), (230, 154, 54), (178, 48, 66)),
    )
    rgb[grid.evidence_state == UNOBSERVED] = (20, 25, 30)
    rgb[grid.evidence_state == HYPOTHESIS_OVERFLOW] = (205, 56, 181)
    return rgb


def _unique_scan_support(grid: CandidateQaGrid) -> np.ndarray:
    normalized = grid.unique_scan_count.astype(np.float32) / 16.0
    rgb = _palette(
        normalized,
        ((25, 28, 34), (40, 74, 102), (45, 140, 139), (155, 205, 108), (244, 226, 111)),
    )
    rgb[grid.evidence_state == UNOBSERVED] = (18, 20, 24)
    return rgb


def _view_angle_support(grid: CandidateQaGrid) -> np.ndarray:
    angle = grid.maximum_pairwise_view_angle_deg
    finite = grid.unambiguous_mask & np.isfinite(angle)
    rgb = np.full((*angle.shape, 3), (24, 27, 31), dtype=np.uint8)
    rgb[grid.unambiguous_mask & ~finite] = (45, 56, 68)
    colors = _palette(
        np.clip(angle / 90.0, 0.0, 1.0),
        ((47, 58, 80), (49, 112, 142), (74, 174, 137), (220, 207, 91)),
    )
    rgb[finite] = colors[finite]
    return rgb


def _descriptive_mad(grid: CandidateQaGrid) -> np.ndarray:
    values = grid.descriptive_inter_scan_mad_m
    finite = grid.unambiguous_mask & np.isfinite(values)
    rgb = np.full((*values.shape, 3), (24, 25, 28), dtype=np.uint8)
    if not np.any(finite):
        return rgb
    ceiling = max(float(np.quantile(values[finite], 0.99)), 1e-9)
    colors = _palette(
        np.minimum(values / ceiling, 1.0),
        ((24, 35, 55), (58, 75, 128), (145, 83, 135), (218, 119, 80), (248, 220, 121)),
    )
    rgb[finite] = colors[finite]
    return rgb


def _abstention_state(grid: CandidateQaGrid) -> np.ndarray:
    rgb = np.full((*grid.evidence_state.shape, 3), (25, 25, 28), dtype=np.uint8)
    rgb[grid.evidence_state == UNOBSERVED] = (35, 55, 75)
    rgb[grid.evidence_state == SOLE_RAW_CANDIDATE] = (92, 111, 93)
    rgb[grid.evidence_state == AMBIGUOUS_CANDIDATES] = (230, 145, 48)
    rgb[grid.evidence_state == HYPOTHESIS_OVERFLOW] = (197, 54, 171)
    return rgb


def _save_panels(
    panels: tuple[tuple[str, np.ndarray], ...],
    *,
    subtitle: str,
    legend: str,
    output: Path,
) -> None:
    panel_gap = 20
    available = _CANVAS_WIDTH - 2 * _MARGIN - panel_gap * (len(panels) - 1)
    panel_width = available // len(panels)
    target_height = min(900, panel_width)
    rendered: list[tuple[str, Image.Image]] = []
    for label, rgb in panels:
        source = Image.fromarray(np.flipud(rgb), mode="RGB")
        scale = min(panel_width / source.width, target_height / source.height)
        size = (
            max(1, int(round(source.width * scale))),
            max(1, int(round(source.height * scale))),
        )
        rendered.append((label, source.resize(size, Image.Resampling.NEAREST)))
    image_height = max(image.height for _, image in rendered)
    canvas = Image.new(
        "RGB",
        (_CANVAS_WIDTH, _TITLE_HEIGHT + 26 + image_height + _LEGEND_HEIGHT),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((_MARGIN, 12), _TITLE, fill=(133, 24, 34))
    draw.text((_MARGIN, 36), subtitle, fill="black")
    left = _MARGIN
    for label, image in rendered:
        draw.text((left, _TITLE_HEIGHT), label, fill="black")
        top = _TITLE_HEIGHT + 24
        canvas.paste(image, (left, top))
        draw.rectangle(
            (left - 1, top - 1, left + image.width, top + image.height),
            outline=(45, 45, 45),
        )
        left += panel_width + panel_gap
    draw.multiline_text(
        (_MARGIN, _TITLE_HEIGHT + 32 + image_height),
        "\n".join(textwrap.wrap(legend, width=165)),
        fill="black",
        spacing=3,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=False, compress_level=9)


def _render_images(
    grid: CandidateQaGrid,
    qa_dir: Path,
) -> tuple[tuple[Path, str], ...]:
    selection = (
        f"shard ({grid.shard_x}, {grid.shard_y}), {grid.resolution_m:g} m cells, "
        "north up"
    )
    records: tuple[
        tuple[
            str,
            tuple[tuple[str, Callable[[CandidateQaGrid], np.ndarray]], ...],
            str,
            str,
        ],
        ...,
    ] = (
        (
            "01_scan_balanced_unambiguous_primary_sheet_height_hillshade.png",
            (("unambiguous primary-sheet diagnostic", _height_hillshade),),
            "colored = exactly one retained hypothesis and complete observed 3x3 stencil; teal = sole candidate hidden from derivative; dark = abstention",
            "Scan-balanced sole candidate displayed as a primary-sheet diagnostic only where the cell and its full 3x3 stencil each have exactly one retained hypothesis. It is not selected ground or primary truth; no gap fill, interpolation, or choice among competing sheets is performed.",
        ),
        (
            "02_hypothesis_count_and_ambiguity.png",
            (("retained hypothesis count", _hypothesis_count),),
            "green = one; orange/red = multiple; magenta = ceiling/assignment abstention; dark = unobserved",
            "Retained candidate-sheet count. Multiple sheets remain ambiguous; overflow or unresolved mode assignment is an explicit whole-cell abstention.",
        ),
        (
            "03_unique_scan_and_view_support.png",
            (
                ("unique scans, 0..16", _unique_scan_support),
                ("max pairwise view angle, 0..90+ deg", _view_angle_support),
            ),
            "view angle is shown only for sole candidates; blue-gray = fewer than two valid mean beam directions; neither panel is confidence",
            "Left: unique scans contributing any observed return in the cell. Right: maximum pairwise angle between per-scan mean beam directions for a sole candidate only. These are support descriptors, not qualification or confidence.",
        ),
        (
            "04_descriptive_inter_scan_mad.png",
            (("descriptive inter-scan MAD", _descriptive_mad),),
            "dark = undefined/abstained; color scale is clipped at this shard's 99th percentile and is descriptive disagreement, NOT error",
            "Median absolute deviation of one candidate-height median per contributing scan, shown only for sole candidates. It describes cross-view disagreement and is not an independent error estimate.",
        ),
        (
            "05_abstention_and_unknowns.png",
            (("evidence disposition", _abstention_state),),
            "blue = unobserved; gray-green = sole raw candidate still unqualified; orange = competing sheets; magenta = unresolved/overflow. ALL classes forbid synthesis.",
            "Evidence disposition and abstention map. Even a sole candidate retains unknown semantics, occlusion, point spacing, independent error, and full-versus-thinned transfer unless separately qualified.",
        ),
    )
    result: list[tuple[Path, str]] = []
    for filename, panel_specs, legend, interpretation in records:
        path = qa_dir / filename
        _save_panels(
            tuple((label, image_fn(grid)) for label, image_fn in panel_specs),
            subtitle=selection,
            legend=legend,
            output=path,
        )
        result.append((path, interpretation))
    return tuple(result)


def _validate_existing(final_root: Path, index_bytes: bytes, artifacts: list[dict]) -> None:
    index_path = final_root / "qa" / "index.json"
    if not index_path.is_file() or index_path.read_bytes() != index_bytes:
        raise FileExistsError("candidate QA content address contains a different index")
    for artifact in artifacts:
        path = final_root / artifact["path"]
        if (
            not path.is_file()
            or path.stat().st_size != artifact["bytes"]
            or _sha256(path) != artifact["sha256"]
        ):
            raise FileExistsError("candidate QA content address contains a changed PNG")


def publish_candidate_qa(
    grid: CandidateQaGrid,
    *,
    output_root: Path,
    manifest: VerifiedSpatialManifest,
    hypothesis_config: HypothesisConfig,
) -> Path:
    """Render and atomically publish one immutable, content-addressed QA bundle."""
    output_root = Path(os.path.abspath(os.fspath(output_root)))
    output_root.mkdir(parents=True, exist_ok=True)
    relevant_shards = manifest.shards_at(grid.shard_x, grid.shard_y)
    if not relevant_shards:
        raise ValueError("candidate QA selection has no published point-shard artifacts")
    qa_root = Path(__file__).parent
    candidate_root = qa_root.parent / "candidate"
    implementation_paths = {
        "candidate/manifest.py": candidate_root / "manifest.py",
        "candidate/records.py": candidate_root / "records.py",
        "candidate/accumulators.py": candidate_root / "accumulators.py",
        "candidate/hypotheses.py": candidate_root / "hypotheses.py",
        "candidate_qa/grid.py": qa_root / "grid.py",
        "candidate_qa/render.py": qa_root / "render.py",
        "candidate_qa/pipeline.py": qa_root / "pipeline.py",
    }
    implementation = {
        name: _sha256(path) for name, path in implementation_paths.items()
    }
    recipe = {
        "schemaVersion": _INDEX_SCHEMA,
        "selection": {
            "shardX": grid.shard_x,
            "shardY": grid.shard_y,
            "resolutionM": grid.resolution_m,
            "extentM": 4.0,
        },
        "hypothesisConfig": asdict(hypothesis_config),
        "implementationSha256": implementation,
        "displayRule": "height_only_for_observed_cells_with_exactly_one_hypothesis_and_complete_3x3_sole-candidate_stencil",
    }
    recipe_bytes = canonical_json_bytes(recipe)
    stage = Path(tempfile.mkdtemp(prefix=".candidate-qa-", dir=output_root))
    try:
        qa_dir = stage / "qa"
        rendered = _render_images(grid, qa_dir)
        artifacts: list[dict] = []
        for path, interpretation in rendered:
            digest = _sha256(path)
            renamed = path.with_name(f"{path.stem}--{digest}.png")
            path.rename(renamed)
            with Image.open(renamed) as image:
                dimensions = [image.width, image.height]
            artifacts.append(
                {
                    "path": f"qa/{renamed.name}",
                    "sha256": digest,
                    "bytes": renamed.stat().st_size,
                    "dimensionsPx": dimensions,
                    "interpretation": interpretation,
                }
            )
        index = {
            "schemaVersion": _INDEX_SCHEMA,
            "status": "raw_candidate_unqualified_visual_qa",
            "scientificRole": "diagnostic_only_not_surface_or_target_truth",
            "qualificationStatus": "raw_candidate_unqualified",
            "surfaceClaim": False,
            "targetTruth": False,
            "synthesisAuthorized": False,
            "interpolationPerformed": False,
            "manifest": {
                "sha256": manifest.manifest_sha256,
                "selectedPointShards": [
                    {
                        "scan": shard.scan,
                        "path": shard.relative_path,
                        "records": shard.records,
                        "sha256": shard.sha256,
                    }
                    for shard in relevant_shards
                ],
            },
            "recipe": recipe,
            "recipeSha256": hashlib.sha256(recipe_bytes).hexdigest(),
            "summary": {
                "cells": int(grid.evidence_state.size),
                "soleRawCandidateCells": int(np.count_nonzero(grid.unambiguous_mask)),
                "ambiguousCandidateCells": int(
                    np.count_nonzero(grid.evidence_state == AMBIGUOUS_CANDIDATES)
                ),
                "unobservedCells": int(
                    np.count_nonzero(grid.evidence_state == UNOBSERVED)
                ),
                "overflowOrUnresolvedCells": int(
                    np.count_nonzero(grid.evidence_state == HYPOTHESIS_OVERFLOW)
                ),
                "unknownReasonCellMentions": dict(grid.unknown_reason_counts),
            },
            "evidenceBoundaries": [
                "sole_candidate_is_not_selected_ground_or_primary_truth",
                "descriptive_inter_scan_mad_is_not_error",
                "view_support_is_not_confidence",
                "no_interpolation_or_gap_fill",
                "no_synthesis_authorization",
            ],
            "artifacts": artifacts,
        }
        index_bytes = canonical_json_bytes(index)
        index_sha256 = hashlib.sha256(index_bytes).hexdigest()
        index_path = qa_dir / "index.json"
        index_path.write_bytes(index_bytes)
        final_root = output_root / index_sha256
        try:
            os.rename(stage, final_root)
        except OSError:
            if not final_root.is_dir():
                raise
            _validate_existing(final_root, index_bytes, artifacts)
            shutil.rmtree(stage)
        for artifact in artifacts:
            os.chmod(final_root / artifact["path"], 0o400)
        os.chmod(final_root / "qa" / "index.json", 0o400)
        os.chmod(final_root / "qa", 0o500)
        os.chmod(final_root, 0o500)
        return final_root / "qa" / "index.json"
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
