"""Bounded whole-form exemplar transport capacity run.

The model never invents a fine-noise remainder. Every output sample comes from a
large, support-qualified measured terrain patch; overlap matching only chooses
and offsets those patches to form a larger continuous canvas.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, gaussian_filter


TEXEL_M = 0.0625
SOURCE_IDS = ("k11", "k32", "k36")
SOURCE_SHA256 = {
    "k11": "a5d78feece68a1dd30bf2721837b8bbee6ca1418f2443de15801a18191cd650b",
    "k32": "04839a7b5fd5f650958090e68fef31881c3dfa9a9c017a96d475e6e25348a23f",
    "k36": "7e120561120ea44baf346f7bbd99661487982b43e7d301c9644908abfa0b82b3",
}


@dataclass(frozen=True)
class Candidate:
    source_index: int
    source_id: str
    y: int
    x: int
    transform: int
    support_fraction: float
    values: np.ndarray


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_surface(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        z = np.asarray(data["elevation_m"], dtype=np.float64)
        measured = np.asarray(data["measured"], dtype=bool)
        confidence = np.asarray(data["confidence"], dtype=np.float64)
    support = np.isfinite(z) & (confidence > 0)
    if z.shape != (256, 256) or measured.shape != z.shape or confidence.shape != z.shape:
        raise RuntimeError(f"unexpected source shape for {path}")
    if np.any(measured & ~support):
        raise RuntimeError(f"measured cells outside support in {path}")
    return z, measured, support


def _complete_for_model(z: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Harmonic model completion; completed cells never become source evidence."""
    if not support.any():
        raise RuntimeError("empty exemplar support")
    nearest = distance_transform_edt(~support, return_distances=False, return_indices=True)
    completed = z[tuple(nearest)]
    # Relax only missing cells. This avoids nearest-cell plateaus at small TLS holes.
    for _ in range(160):
        neighbor_mean = 0.25 * (
            np.roll(completed, 1, 0)
            + np.roll(completed, -1, 0)
            + np.roll(completed, 1, 1)
            + np.roll(completed, -1, 1)
        )
        completed[~support] = neighbor_mean[~support]
        completed[0] = completed[1]
        completed[-1] = completed[-2]
        completed[:, 0] = completed[:, 1]
        completed[:, -1] = completed[:, -2]
    completed[support] = z[support]
    return completed


def _remove_plane(values: np.ndarray, support: np.ndarray) -> np.ndarray:
    yy, xx = np.indices(values.shape, dtype=np.float64)
    design = np.column_stack((np.ones(int(support.sum())), xx[support], yy[support]))
    coefficients = np.linalg.lstsq(design, values[support], rcond=None)[0]
    return values - (coefficients[0] + coefficients[1] * xx + coefficients[2] * yy)


def _transform(values: np.ndarray, transform: int) -> np.ndarray:
    if transform < 4:
        return np.rot90(values, transform)
    return np.rot90(np.fliplr(values), transform - 4)


def _candidates(
    surfaces: list[tuple[np.ndarray, np.ndarray]], patch_cells: int, step_cells: int
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for source_index, (completed, support) in enumerate(surfaces):
        source_id = SOURCE_IDS[source_index]
        for y in range(0, completed.shape[0] - patch_cells + 1, step_cells):
            for x in range(0, completed.shape[1] - patch_cells + 1, step_cells):
                patch_support = support[y : y + patch_cells, x : x + patch_cells]
                fraction = float(np.mean(patch_support))
                if fraction < 0.78:
                    continue
                patch = completed[y : y + patch_cells, x : x + patch_cells]
                residual = _remove_plane(patch, patch_support)
                for transform in range(8):
                    candidates.append(
                        Candidate(
                            source_index=source_index,
                            source_id=source_id,
                            y=y,
                            x=x,
                            transform=transform,
                            support_fraction=fraction,
                            values=np.ascontiguousarray(_transform(residual, transform)),
                        )
                    )
    if len(candidates) < 64:
        raise RuntimeError(f"only {len(candidates)} support-qualified whole-form patches")
    return candidates


def _origins(cells: int, patch_cells: int, stride_cells: int) -> list[int]:
    values = list(range(0, cells - patch_cells + 1, stride_cells))
    last = cells - patch_cells
    if values[-1] != last:
        values.append(last)
    return values


def _window(
    patch_cells: int, overlap_cells: int, *, top: bool, bottom: bool, left: bool, right: bool
) -> np.ndarray:
    axis = np.ones(patch_cells, dtype=np.float64)
    ramp = np.sin(np.linspace(0.0, np.pi * 0.5, overlap_cells, endpoint=False)) ** 2
    if top:
        axis[:overlap_cells] = ramp
    if bottom:
        axis[-overlap_cells:] = ramp[::-1]
    xaxis = np.ones(patch_cells, dtype=np.float64)
    if left:
        xaxis[:overlap_cells] = ramp
    if right:
        xaxis[-overlap_cells:] = ramp[::-1]
    return axis[:, None] * xaxis[None, :]


def _assemble(
    candidates: list[Candidate], *, cells: int, patch_cells: int, overlap_cells: int
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    stride = patch_cells - overlap_cells
    ys = _origins(cells, patch_cells, stride)
    xs = _origins(cells, patch_cells, stride)
    numerator = np.zeros((cells, cells), dtype=np.float64)
    denominator = np.zeros((cells, cells), dtype=np.float64)
    source_vote = np.zeros((len(SOURCE_IDS), cells, cells), dtype=np.float64)
    use_count = np.zeros(len(candidates), dtype=np.int32)
    source_use_count = np.zeros(len(SOURCE_IDS), dtype=np.int32)
    placements: list[dict[str, Any]] = []

    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            region = (slice(y, y + patch_cells), slice(x, x + patch_cells))
            overlap = denominator[region] > 1e-12
            current = np.divide(
                numerator[region], denominator[region], out=np.zeros((patch_cells, patch_cells)), where=overlap
            )
            ranked: list[tuple[float, int, float]] = []
            for index, candidate in enumerate(candidates):
                if overlap.any():
                    offset = float(np.mean(current[overlap] - candidate.values[overlap]))
                    difference = current[overlap] - (candidate.values[overlap] + offset)
                    seam = float(np.mean(difference * difference))
                else:
                    offset = 0.0
                    seam = 0.0
                # Reuse pressure is deliberately small: morphology compatibility owns selection.
                source_excess = int(source_use_count[candidate.source_index] - source_use_count.min())
                score = seam + use_count[index] * 0.0025 + source_excess * 0.012
                ranked.append((score, index, offset))
            ranked.sort(key=lambda item: (item[0], item[1]))
            # A spatially fixed rank avoids always selecting the same global optimum while
            # introducing no geometric signal beyond measured candidates.
            choice = ranked[(row * 7 + col * 11) % min(12, len(ranked))]
            score, index, offset = choice
            candidate = candidates[index]
            use_count[index] += 1
            source_use_count[candidate.source_index] += 1
            weight = _window(
                patch_cells,
                overlap_cells,
                top=row > 0,
                bottom=row < len(ys) - 1,
                left=col > 0,
                right=col < len(xs) - 1,
            )
            values = candidate.values + offset
            numerator[region] += weight * values
            denominator[region] += weight
            source_vote[candidate.source_index][region] += weight
            placements.append(
                {
                    "target_y": y,
                    "target_x": x,
                    "source_id": candidate.source_id,
                    "source_y": candidate.y,
                    "source_x": candidate.x,
                    "transform": candidate.transform,
                    "support_fraction": candidate.support_fraction,
                    "overlap_mse_m2": score,
                    "offset_m": offset,
                }
            )
    if np.any(denominator <= 0):
        raise RuntimeError("whole-form assembly left uncovered cells")
    output = numerator / denominator
    output -= float(np.mean(output))
    ownership = np.argmax(source_vote, axis=0).astype(np.uint8)
    return output, ownership, placements


def _percentile_rgb(values: np.ndarray, *, diverging: bool = False) -> np.ndarray:
    finite = values[np.isfinite(values)]
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if diverging:
        limit = max(abs(float(lo)), abs(float(hi)), 1e-9)
        t = np.clip(values / limit, -1.0, 1.0)
        rgb = np.empty(values.shape + (3,), dtype=np.float64)
        positive = t >= 0
        rgb[..., 0] = np.where(positive, 238, 238 + 93 * t)
        rgb[..., 1] = np.where(positive, 238 - 170 * t, 238 + 118 * t)
        rgb[..., 2] = np.where(positive, 238 - 190 * t, 238)
    else:
        t = np.clip((values - lo) / max(float(hi - lo), 1e-9), 0.0, 1.0)
        rgb = np.stack((35 + 205 * t, 70 + 160 * t, 45 + 160 * t), axis=-1)
    return np.asarray(np.clip(rgb, 0, 255), dtype=np.uint8)


def _hillshade(values: np.ndarray) -> np.ndarray:
    dy, dx = np.gradient(values, TEXEL_M)
    nx, ny, nz = -dx, -dy, np.ones_like(values)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    light = np.asarray([-0.45, -0.55, 0.70])
    shade = np.clip((nx * light[0] + ny * light[1] + nz * light[2]) / norm, 0.0, 1.0)
    return np.asarray(255 * shade, dtype=np.uint8)


def _panel(items: list[tuple[str, np.ndarray]], path: Path, subtitle: str) -> None:
    width = 512
    header = 58
    canvas = Image.new("RGB", (width * len(items), width + header), (242, 239, 226))
    draw = ImageDraw.Draw(canvas)
    for index, (label, array) in enumerate(items):
        image = Image.fromarray(array).resize((width, width), Image.Resampling.BILINEAR)
        canvas.paste(image, (index * width, header))
        draw.text((index * width + 10, 8), label, fill=(25, 28, 24))
        draw.text((index * width + 10, 29), subtitle, fill=(70, 72, 66))
    canvas.save(path, compress_level=9)


def run_forest_exemplar_transport(*, repo_root: Path, output_root: Path) -> Path:
    """Execute one deterministic R0 capacity artifact from the three calibration plots."""
    implementation_path = Path(__file__).resolve()
    source_root = repo_root / "asset-gen/data/in/microtopo-exemplars/lapinjarvi-2016/derived-v1"
    recipe = {
        "schema": "forest-whole-form-exemplar-r0/1",
        "authority": "algorithmic_capacity_only_no_estonia_transfer_no_production",
        "regime": "forest.mesic_mineral",
        "method": "support_qualified_large_patch_exemplar_transport",
        "texel_m": TEXEL_M,
        "output_cells": 512,
        "patch_cells": 160,
        "overlap_cells": 72,
        "candidate_step_cells": 16,
        "minimum_patch_support_fraction": 0.78,
        "sources": SOURCE_SHA256,
        "implementation_sha256": _sha256(implementation_path),
    }
    build_id = hashlib.sha256(_canonical_json(recipe)).hexdigest()
    destination = output_root / "sha256" / build_id
    if destination.exists():
        raise FileExistsError(f"immutable artifact already exists: {destination}")
    output_root.mkdir(parents=True, exist_ok=True)

    surfaces: list[tuple[np.ndarray, np.ndarray]] = []
    source_arrays: list[np.ndarray] = []
    for source_id in SOURCE_IDS:
        path = source_root / f"{source_id}-ground.npz"
        if _sha256(path) != SOURCE_SHA256[source_id]:
            raise RuntimeError(f"source hash differs: {path}")
        z, _measured, support = _load_surface(path)
        completed = _complete_for_model(z, support)
        source_arrays.append(_remove_plane(completed, support))
        surfaces.append((completed, support))

    candidates = _candidates(surfaces, recipe["patch_cells"], recipe["candidate_step_cells"])
    output, ownership, placements = _assemble(
        candidates,
        cells=recipe["output_cells"],
        patch_cells=recipe["patch_cells"],
        overlap_cells=recipe["overlap_cells"],
    )
    if not np.isfinite(output).all():
        raise RuntimeError("synthesis produced non-finite samples")
    dy, dx = np.gradient(output, TEXEL_M)
    metrics = {
        "candidate_count": len(candidates),
        "placement_count": len(placements),
        "source_placement_counts": {
            source_id: sum(item["source_id"] == source_id for item in placements)
            for source_id in SOURCE_IDS
        },
        "height_p01_p50_p99_m": [float(v) for v in np.percentile(output, [1, 50, 99])],
        "relief_rms_m": float(np.sqrt(np.mean(output * output))),
        "slope_p95": float(np.percentile(np.hypot(dx, dy), 95)),
        "max_adjacent_step_m": float(
            max(np.max(np.abs(np.diff(output, axis=0))), np.max(np.abs(np.diff(output, axis=1))))
        ),
    }

    with TemporaryDirectory(prefix="forest-exemplar-", dir=output_root) as temporary:
        staging = Path(temporary)
        (staging / "surface").mkdir()
        (staging / "qa").mkdir()
        (staging / "recipe.json").write_bytes(_canonical_json(recipe) + b"\n")
        np.save(staging / "surface/height_f32.npy", output.astype(np.float32), allow_pickle=False)
        np.save(staging / "surface/source_ownership_u8.npy", ownership, allow_pickle=False)
        (staging / "surface/placements.json").write_bytes(_canonical_json(placements) + b"\n")

        absolute = _percentile_rgb(output)
        shade = np.repeat(_hillshade(output)[..., None], 3, axis=2)
        _panel(
            [("SYNTHETIC ABSOLUTE RELIEF", absolute), ("GROUND-SCALE HILLSHADE", shade)],
            staging / "qa/01_synthetic_surface.png",
            "32 m x 32 m | 6.25 cm samples | R0 capacity only",
        )
        band_1m = output - gaussian_filter(output, 16.0, mode="reflect")
        band_4m = gaussian_filter(output, 16.0, mode="reflect") - gaussian_filter(
            output, 64.0, mode="reflect"
        )
        _panel(
            [("SUB-METRE FORM BAND", _percentile_rgb(band_1m, diverging=True)),
             ("1-4 METRE FORM BAND", _percentile_rgb(band_4m, diverging=True))],
            staging / "qa/02_multiscale_forms.png",
            "red high | blue low | no generated fine-noise remainder",
        )
        palette = np.asarray([[45, 123, 113], [217, 139, 71], [103, 87, 145]], dtype=np.uint8)
        ownership_rgb = palette[ownership]
        source_strip = np.concatenate([_percentile_rgb(value) for value in source_arrays], axis=1)
        source_strip = np.asarray(
            Image.fromarray(source_strip).resize((512, 512), Image.Resampling.BILINEAR)
        )
        _panel(
            [("CALIBRATION EXEMPLARS K11 | K32 | K36", source_strip),
             ("DOMINANT SOURCE OWNERSHIP", ownership_rgb)],
            staging / "qa/03_exemplar_provenance.png",
            "geometry is transported from support-qualified measured patches",
        )

        files = {}
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                files[str(path.relative_to(staging))] = {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
        manifest = {
            "schema": "forest-whole-form-exemplar-artifact/1",
            "build_id": build_id,
            "decision_scope": "inspectable_float_r0_capacity_hypothesis",
            "limitations": [
                "foreign mesic mineral forest analogue only",
                "source-hole completion is model input and not measured evidence",
                "no Estonia condition binding or national transfer authority",
                "no cook, packing, browser, production, or latest authority",
            ],
            "metrics": metrics,
            "files": files,
        }
        (staging / "manifest.json").write_bytes(_canonical_json(manifest) + b"\n")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, destination)
    return destination
