"""Research-only Estonia anchoring of the measured whole-form forest hypothesis."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from scipy.interpolate import RectBivariateSpline

from assetgen.process.micro_masks import rasterize_micro_morphology_mask
from assetgen.process.micro_fixture import conservative_cell_correct
from assetgen.process.microtopo.projection import build_smooth_mean_null_projector

from .run import (
    SOURCE_IDS,
    SOURCE_SHA256,
    TEXEL_M,
    _assemble,
    _candidates,
    _canonical_json,
    _complete_for_model,
    _hillshade,
    _load_surface,
    _panel,
    _percentile_rgb,
    _sha256,
)


TRANSACTION_ID = "009fd3d2be07d465199e6330ac9d2fc5a8fc7a16bbb5df0f55e5030ce962d3d1"
MASTER_BBOX = (679600.0, 6444720.0, 679760.0, 6444880.0)
REVIEW_BBOX = (679616.0, 6444736.0, 679744.0, 6444864.0)
FACTOR = 16


def _corrected_authority(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Load the exact accepted C0 window and a four-cell interpolation margin."""
    chunk = np.load(path, mmap_mode="r", allow_pickle=False)
    if chunk.shape != (2048, 2048) or chunk.dtype != np.float32:
        raise RuntimeError("accepted corrected C0 chunk shape or dtype differs")
    chunk_e_min = 368640 + 151 * 2048
    chunk_n_max = 6635520 - 93 * 2048
    e_min, _n_min, e_max, n_max = MASTER_BBOX
    if any(value != int(value) for value in MASTER_BBOX):
        raise RuntimeError("master bbox must follow integral C0 cell edges")
    width = int(e_max - e_min)
    height = int(n_max - MASTER_BBOX[1])
    x0 = int(e_min - chunk_e_min)
    y0 = int(chunk_n_max - n_max)
    margin = 4
    authority = np.asarray(chunk[y0 : y0 + height, x0 : x0 + width], dtype=np.float64)
    extended = np.asarray(
        chunk[
            y0 - margin : y0 + height + margin,
            x0 - margin : x0 + width + margin,
        ],
        dtype=np.float64,
    )
    if authority.shape != (160, 160) or extended.shape != (168, 168):
        raise RuntimeError("accepted corrected C0 does not cover the master bbox")
    east_centers = e_min - margin + np.arange(168, dtype=np.float64) + 0.5
    north_centers = n_max + margin - np.arange(168, dtype=np.float64) - 0.5
    return authority, extended, (east_centers, north_centers)


def _refine_c0(
    authority: np.ndarray,
    extended: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray],
    east: np.ndarray,
    north: np.ndarray,
) -> np.ndarray:
    east_centers, north_centers = axes
    # RectBivariateSpline requires increasing coordinates; rows are north-to-south.
    spline = RectBivariateSpline(
        north_centers[::-1], east_centers, extended[::-1], kx=3, ky=3, s=0.0
    )
    smooth = spline(north[::-1], east, grid=True)[::-1]
    return conservative_cell_correct(smooth, authority, FACTOR)


def _project_residual(residual: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    cell_shape = (residual.shape[0] // FACTOR, residual.shape[1] // FACTOR)
    fully_soft = allowed.reshape(
        cell_shape[0], FACTOR, cell_shape[1], FACTOR
    ).all(axis=(1, 3))
    taper_plan = build_smooth_mean_null_projector(
        np.zeros(cell_shape, dtype=np.float64), fully_soft, factor=FACTOR
    )
    taper = taper_plan.taper_window(0, 0, *cell_shape)
    tapered = np.where(allowed, residual * taper, 0.0)
    means = tapered.reshape(
        cell_shape[0], FACTOR, cell_shape[1], FACTOR
    ).mean(axis=(1, 3))
    projector = build_smooth_mean_null_projector(means, fully_soft, factor=FACTOR)
    return projector.project_window(residual, allowed, row0=0, col0=0)


def _mask_rgb(mask) -> np.ndarray:
    rgb = np.full(mask.allowed.shape + (3,), (47, 119, 79), dtype=np.uint8)
    rgb[~mask.forest] = (165, 160, 143)
    rgb[~mask.compatible_soil] = (183, 137, 82)
    rgb[mask.slope_cliff_context] = (202, 92, 72)
    rgb[mask.paved_road] = (102, 102, 102)
    rgb[mask.building] = (205, 183, 118)
    rgb[mask.water] = (48, 111, 155)
    return rgb


def _crop(array: np.ndarray) -> np.ndarray:
    halo = int((REVIEW_BBOX[0] - MASTER_BBOX[0]) / TEXEL_M)
    cells = int((REVIEW_BBOX[2] - REVIEW_BBOX[0]) / TEXEL_M)
    return np.ascontiguousarray(array[halo : halo + cells, halo : halo + cells])


def run_estonia_forest_exemplar(*, repo_root: Path, output_root: Path) -> Path:
    """Build one crop-last Estonia research candidate without cooking or packing."""
    transaction_root = (
        repo_root
        / "asset-gen/data/work/terrain-repair/taevaskoda-ahja-stage1-transaction"
        / TRANSACTION_ID
    )
    c0_path = transaction_root / "corrected-base/decoded-core/0/151_93.npy"
    transaction_path = transaction_root / "transaction.json"
    implementation_path = Path(__file__).resolve()
    exemplar_impl_path = implementation_path.with_name("run.py")
    e67_root = (
        repo_root
        / "asset-gen/data/work/microtopography/forest-whole-form-exemplar/sha256"
        / "e67feb86a7b7f3b504a5107183a7fbcef86ba786d2e357aa21cfcc7f12f0a716"
    )
    recipe = {
        "schema": "estonia-forest-whole-form-research/1",
        "authority": "research_preview_candidate_no_production_no_owner_no_latest",
        "regime": "forest.mesic_mineral",
        "method": "e67feb_whole_form_exemplar_transport",
        "transaction_id": TRANSACTION_ID,
        "transaction_sha256": _sha256(transaction_path),
        "corrected_c0_path": str(c0_path.relative_to(repo_root)),
        "corrected_c0_sha256": _sha256(c0_path),
        "source_capacity_manifest_sha256": _sha256(e67_root / "manifest.json"),
        "master_bbox_en": list(MASTER_BBOX),
        "review_bbox_en": list(REVIEW_BBOX),
        "texel_m": TEXEL_M,
        "master_cells": 2560,
        "review_cells": 2048,
        "patch_cells": 192,
        "overlap_cells": 128,
        "candidate_step_cells": 16,
        "minimum_patch_support_fraction": 0.78,
        "selector": "exact_etak_forest_plus_known_unmodified_mesic_mineral_soil",
        "projection": "aoi_wide_smooth_mean_null_crop_last",
        "sources": SOURCE_SHA256,
        "implementation_sha256": _sha256(implementation_path),
        "exemplar_implementation_sha256": _sha256(exemplar_impl_path),
    }
    build_id = hashlib.sha256(_canonical_json(recipe)).hexdigest()
    destination = output_root / "sha256" / build_id
    if destination.exists():
        raise FileExistsError(f"immutable artifact already exists: {destination}")
    output_root.mkdir(parents=True, exist_ok=True)

    source_root = (
        repo_root
        / "asset-gen/data/in/microtopo-exemplars/lapinjarvi-2016/derived-v1"
    )
    surfaces: list[tuple[np.ndarray, np.ndarray]] = []
    for source_id in SOURCE_IDS:
        source_path = source_root / f"{source_id}-ground.npz"
        if _sha256(source_path) != SOURCE_SHA256[source_id]:
            raise RuntimeError(f"source hash differs: {source_path}")
        z, _measured, support = _load_surface(source_path)
        surfaces.append((_complete_for_model(z, support), support))
    candidates = _candidates(
        surfaces, recipe["patch_cells"], recipe["candidate_step_cells"]
    )
    residual, ownership, placements = _assemble(
        candidates,
        cells=recipe["master_cells"],
        patch_cells=recipe["patch_cells"],
        overlap_cells=recipe["overlap_cells"],
    )

    e_min, _n_min, _e_max, n_max = MASTER_BBOX
    east = e_min + (np.arange(recipe["master_cells"], dtype=np.float64) + 0.5) * TEXEL_M
    north = n_max - (np.arange(recipe["master_cells"], dtype=np.float64) + 0.5) * TEXEL_M
    morphology_mask = rasterize_micro_morphology_mask(east, north)
    projected = _project_residual(residual, morphology_mask.allowed)
    authority, extended, axes = _corrected_authority(c0_path)
    c0 = _refine_c0(authority, extended, axes, east, north)
    c1 = c0 + projected

    if np.any(projected[~morphology_mask.allowed] != 0.0):
        raise AssertionError("projected morphology changed a hard-excluded sample")
    cell_means = projected.reshape(160, FACTOR, 160, FACTOR).mean(axis=(1, 3))
    max_cell_mean = float(np.max(np.abs(cell_means)))
    if max_cell_mean >= 1e-10:
        raise AssertionError("projected morphology changed accepted one-metre C0 means")

    c0_crop = _crop(c0)
    c1_crop = _crop(c1)
    residual_crop = _crop(projected)
    allowed_crop = _crop(morphology_mask.allowed)
    ownership_crop = _crop(ownership)
    mask_crop = type(morphology_mask)(
        **{
            name: _crop(getattr(morphology_mask, name))
            for name in morphology_mask.__dataclass_fields__
        }
    )
    metrics = {
        "master_mask": morphology_mask.evidence(),
        "review_mask": mask_crop.evidence(),
        "placement_count": len(placements),
        "source_placement_counts": {
            source_id: sum(item["source_id"] == source_id for item in placements)
            for source_id in SOURCE_IDS
        },
        "max_abs_residual_m": float(np.max(np.abs(residual_crop))),
        "residual_rms_allowed_m": float(
            np.sqrt(np.mean(residual_crop[allowed_crop] ** 2))
        ) if allowed_crop.any() else 0.0,
        "max_one_metre_mean_error_m": max_cell_mean,
        "max_hard_exclusion_residual_m": float(
            np.max(np.abs(residual_crop[~allowed_crop]))
        ) if (~allowed_crop).any() else 0.0,
        "c0_p01_p99_m": [float(v) for v in np.percentile(c0_crop, [1, 99])],
        "c1_p01_p99_m": [float(v) for v in np.percentile(c1_crop, [1, 99])],
    }

    with TemporaryDirectory(prefix="estonia-forest-", dir=output_root) as temporary:
        staging = Path(temporary)
        (staging / "surface").mkdir()
        (staging / "qa").mkdir()
        (staging / "recipe.json").write_bytes(_canonical_json(recipe) + b"\n")
        np.save(staging / "surface/c0_height_f32.npy", c0_crop.astype(np.float32), allow_pickle=False)
        np.save(staging / "surface/c1_height_f32.npy", c1_crop.astype(np.float32), allow_pickle=False)
        np.save(staging / "surface/residual_f32.npy", residual_crop.astype(np.float32), allow_pickle=False)
        np.save(staging / "surface/allowed_u8.npy", allowed_crop.astype(np.uint8), allow_pickle=False)
        np.save(staging / "surface/source_ownership_u8.npy", ownership_crop, allow_pickle=False)
        (staging / "surface/placements.json").write_bytes(_canonical_json(placements) + b"\n")

        c0_shade = np.repeat(_hillshade(c0_crop)[..., None], 3, axis=2)
        c1_shade = np.repeat(_hillshade(c1_crop)[..., None], 3, axis=2)
        _panel(
            [("CORRECTED C0 HILLSHADE", c0_shade), ("FOREST EXEMPLAR C1 HILLSHADE", c1_shade)],
            staging / "qa/01_estonia_c0_c1.png",
            "128 m crop centered on Taevaskoda | 6.25 cm | research candidate",
        )
        _panel(
            [("ADDED GEOMETRIC RELIEF", _percentile_rgb(residual_crop, diverging=True)),
             ("EXACT MORPHOLOGY SELECTOR", _mask_rgb(mask_crop))],
            staging / "qa/02_residual_and_selector.png",
            "red high | blue low | blue selector water | hard exclusions unchanged",
        )
        palette = np.asarray([[45, 123, 113], [217, 139, 71], [103, 87, 145]], dtype=np.uint8)
        ownership_rgb = palette[ownership_crop]
        relief_rgb = _percentile_rgb(c1_crop - c0_crop, diverging=True)
        _panel(
            [("EXEMPLAR SOURCE OWNERSHIP", ownership_rgb), ("C1 MINUS C0", relief_rgb)],
            staging / "qa/03_ownership_and_relief.png",
            "K11 teal | K32 orange | K36 violet | one master, cropped last",
        )
        closeup = max(
            (
                int(allowed_crop[y : y + 512, x : x + 512].sum()),
                -y,
                -x,
                y,
                x,
            )
            for y in range(0, 1537, 128)
            for x in range(0, 1537, 128)
        )
        close_y, close_x = closeup[-2:]
        close = np.s_[close_y : close_y + 512, close_x : close_x + 512]
        close_c0 = np.repeat(_hillshade(c0_crop[close])[..., None], 3, axis=2)
        close_c1 = np.repeat(_hillshade(c1_crop[close])[..., None], 3, axis=2)
        _panel(
            [("C0 GROUND-SCALE CLOSEUP", close_c0), ("C1 GROUND-SCALE CLOSEUP", close_c1)],
            staging / "qa/04_ground_scale_closeup.png",
            f"32 m x 32 m | allowed {allowed_crop[close].mean():.1%}",
        )

        files: dict[str, dict[str, Any]] = {}
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                files[str(path.relative_to(staging))] = {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
        manifest = {
            "schema": "estonia-forest-whole-form-research-artifact/1",
            "build_id": build_id,
            "status": "research_candidate_pending_visual_judgment",
            "limitations": [
                "best zero-budget foreign-analogue transfer hypothesis, not independent Estonia truth",
                "no production owner, national transfer, latest, cook, packing, or browser authority",
                "model-side source-hole completion is not measured evidence",
            ],
            "metrics": metrics,
            "files": files,
        }
        (staging / "manifest.json").write_bytes(_canonical_json(manifest) + b"\n")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, destination)
    return destination
