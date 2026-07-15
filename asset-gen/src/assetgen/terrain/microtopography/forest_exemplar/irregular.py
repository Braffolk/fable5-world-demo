"""World-locked irregular support transport for measured forest-floor forms."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from assetgen.process.micro_masks import rasterize_micro_morphology_mask

from .estonia import (
    FACTOR,
    MASTER_BBOX,
    REVIEW_BBOX,
    TRANSACTION_ID,
    _corrected_authority,
    _crop,
    _mask_rgb,
    _project_residual,
    _refine_c0,
)
from .run import (
    SOURCE_IDS,
    SOURCE_SHA256,
    TEXEL_M,
    Candidate,
    _candidates,
    _canonical_json,
    _complete_for_model,
    _hillshade,
    _load_surface,
    _panel,
    _percentile_rgb,
    _sha256,
)


MASK64 = (1 << 64) - 1
WORLD_ANCHOR_E = 368640.0
WORLD_ANCHOR_N = 6635520.0
SEED = 0x4C414153_46525354


@dataclass(frozen=True)
class Site:
    global_row: int
    global_col: int
    priority: int


def _mix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def _hash_xy(x: int, y: int, salt: int = 0) -> int:
    value = (
        (x & MASK64) * 0xD6E8FEB86659FD93
        ^ (y & MASK64) * 0xA5A3564E27F8862F
        ^ SEED
        ^ salt
    ) & MASK64
    return _mix64(value)


def _candidate_site(gx: int, gy: int, cell: int) -> Site:
    return Site(
        global_row=gy * cell + int(_hash_xy(gx, gy, 0xB2) % cell),
        global_col=gx * cell + int(_hash_xy(gx, gy, 0xA1) % cell),
        priority=_hash_xy(gx, gy, 0xC3),
    )


def _world_locked_sites(
    *,
    master_cells: int,
    master_bbox: tuple[float, float, float, float] = MASTER_BBOX,
    candidate_cell: int,
    minimum_distance: int,
    saturation_distance: int,
    saturation_rounds: int,
    support_radius: int,
) -> list[Site]:
    e_min, _n_min, _e_max, n_max = master_bbox
    master_col0 = int(round((e_min - WORLD_ANCHOR_E) / TEXEL_M))
    master_row0 = int(round((WORLD_ANCHOR_N - n_max) / TEXEL_M))
    margin = support_radius + minimum_distance + saturation_rounds * saturation_distance
    gx0 = (master_col0 - margin) // candidate_cell - 1
    gx1 = (master_col0 + master_cells + margin) // candidate_cell + 1
    gy0 = (master_row0 - margin) // candidate_cell - 1
    gy1 = (master_row0 + master_cells + margin) // candidate_cell + 1
    all_sites = {
        (gx, gy): _candidate_site(gx, gy, candidate_cell)
        for gy in range(gy0, gy1 + 1)
        for gx in range(gx0, gx1 + 1)
    }
    def local_minima(pool: set[tuple[int, int]], distance: int) -> set[tuple[int, int]]:
        neighbor_cells = int(np.ceil(distance / candidate_cell)) + 1
        distance2 = distance * distance
        winners: set[tuple[int, int]] = set()
        for gx, gy in pool:
            site = all_sites[gx, gy]
            rank = (site.priority, site.global_row, site.global_col)
            keep = True
            for ny in range(gy - neighbor_cells, gy + neighbor_cells + 1):
                for nx in range(gx - neighbor_cells, gx + neighbor_cells + 1):
                    key = (nx, ny)
                    if key == (gx, gy) or key not in pool:
                        continue
                    other = all_sites[key]
                    dy = other.global_row - site.global_row
                    dx = other.global_col - site.global_col
                    if dy * dy + dx * dx < distance2 and (
                        other.priority,
                        other.global_row,
                        other.global_col,
                    ) < rank:
                        keep = False
                        break
                if not keep:
                    break
            if keep:
                winners.add((gx, gy))
        return winners

    def outside_sites(
        pool: set[tuple[int, int]], sites: set[tuple[int, int]], distance: int
    ) -> set[tuple[int, int]]:
        neighbor_cells = int(np.ceil(distance / candidate_cell)) + 1
        distance2 = distance * distance
        result: set[tuple[int, int]] = set()
        for gx, gy in pool:
            site = all_sites[gx, gy]
            clear = True
            for ny in range(gy - neighbor_cells, gy + neighbor_cells + 1):
                for nx in range(gx - neighbor_cells, gx + neighbor_cells + 1):
                    key = (nx, ny)
                    if key not in sites:
                        continue
                    other = all_sites[key]
                    dy = other.global_row - site.global_row
                    dx = other.global_col - site.global_col
                    if dy * dy + dx * dx < distance2:
                        clear = False
                        break
                if not clear:
                    break
            if clear:
                result.add((gx, gy))
        return result

    available = set(all_sites)
    accepted_keys = local_minima(available, minimum_distance)
    available = outside_sites(available - accepted_keys, accepted_keys, saturation_distance)
    for _round in range(saturation_rounds):
        if not available:
            break
        winners = local_minima(available, saturation_distance)
        accepted_keys.update(winners)
        available = outside_sites(available - winners, accepted_keys, saturation_distance)
    if available:
        raise RuntimeError(
            f"site saturation did not converge after {saturation_rounds} rounds"
        )

    accepted: list[Site] = []
    for key in accepted_keys:
        site = all_sites[key]
        local_row = site.global_row - master_row0
        local_col = site.global_col - master_col0
        if (
            -support_radius < local_row < master_cells + support_radius
            and -support_radius < local_col < master_cells + support_radius
        ):
            accepted.append(site)
    accepted.sort(key=lambda site: (site.global_row, site.global_col))
    if not accepted:
        raise RuntimeError("world-locked site process produced no sites")
    return accepted


def _wendland_weight(
    rows: np.ndarray, cols: np.ndarray, center_row: int, center_col: int, radius: int
) -> np.ndarray:
    distance = np.hypot(rows - center_row, cols - center_col) / float(radius)
    one_minus = np.maximum(0.0, 1.0 - distance)
    return one_minus**4 * (4.0 * distance + 1.0)


def _select_site_patch(
    *,
    site: Site,
    values: np.ndarray,
    source_index: np.ndarray,
    source_row0: int,
    source_row1: int,
    source_col0: int,
    source_col1: int,
    weight: np.ndarray,
    active: np.ndarray,
    current_numerator: np.ndarray,
    current_denominator: np.ndarray,
    candidate_use: np.ndarray,
    source_use: np.ndarray,
) -> tuple[int, float, float, np.ndarray]:
    """Choose one measured form exactly as the accepted irregular transport does."""
    overlap = active & (current_denominator > 1e-12)
    shortlist_count = min(96, len(values))
    hashes = np.fromiter(
        (
            _mix64(site.priority ^ (index * 0x9E3779B97F4A7C15))
            for index in range(len(values))
        ),
        dtype=np.uint64,
        count=len(values),
    )
    shortlist = np.argpartition(hashes, shortlist_count - 1)[:shortlist_count]
    patch_values = values[
        shortlist,
        source_row0:source_row1,
        source_col0:source_col1,
    ].astype(np.float64)
    if overlap.any():
        current = current_numerator[overlap] / current_denominator[overlap]
        overlap_values = patch_values[:, overlap]
        overlap_weight = weight[overlap]
        weight_sum = float(overlap_weight.sum())
        offsets = (
            (current[None, :] - overlap_values) * overlap_weight[None, :]
        ).sum(axis=1) / weight_sum
        difference = current[None, :] - (overlap_values + offsets[:, None])
        scores = (difference * difference * overlap_weight[None, :]).sum(axis=1) / weight_sum
    else:
        offsets = np.zeros(shortlist_count, dtype=np.float64)
        scores = np.zeros(shortlist_count, dtype=np.float64)
    source_excess = source_use[source_index[shortlist]] - source_use.min()
    scores += 0.0025 * candidate_use[shortlist] + 0.012 * source_excess
    best_local = int(np.argmin(scores))
    selected = int(shortlist[best_local])
    offset = float(offsets[best_local])
    return selected, offset, float(scores[best_local]), patch_values[best_local] + offset


def _assemble_irregular(
    candidates: list[Candidate],
    sites: list[Site],
    *,
    master_cells: int,
    patch_cells: int,
    support_radius: int,
    master_bbox: tuple[float, float, float, float] = MASTER_BBOX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    e_min, _n_min, _e_max, n_max = master_bbox
    master_col0 = int(round((e_min - WORLD_ANCHOR_E) / TEXEL_M))
    master_row0 = int(round((WORLD_ANCHOR_N - n_max) / TEXEL_M))
    values = np.stack([candidate.values for candidate in candidates]).astype(np.float32)
    source_index = np.asarray([candidate.source_index for candidate in candidates], dtype=np.int16)
    numerator = np.zeros((master_cells, master_cells), dtype=np.float64)
    denominator = np.zeros((master_cells, master_cells), dtype=np.float64)
    dominant_weight = np.zeros((master_cells, master_cells), dtype=np.float32)
    dominant_site = np.full((master_cells, master_cells), -1, dtype=np.int32)
    source_ownership = np.zeros((master_cells, master_cells), dtype=np.uint8)
    candidate_use = np.zeros(len(candidates), dtype=np.int32)
    source_use = np.zeros(len(SOURCE_IDS), dtype=np.int32)
    placements: list[dict[str, Any]] = []
    half = patch_cells // 2

    for site_index, site in enumerate(sites):
        center_row = site.global_row - master_row0
        center_col = site.global_col - master_col0
        row0 = max(0, center_row - half)
        row1 = min(master_cells, center_row + half)
        col0 = max(0, center_col - half)
        col1 = min(master_cells, center_col + half)
        if row0 >= row1 or col0 >= col1:
            continue
        source_row0 = row0 - (center_row - half)
        source_row1 = source_row0 + (row1 - row0)
        source_col0 = col0 - (center_col - half)
        source_col1 = source_col0 + (col1 - col0)
        rows, cols = np.meshgrid(
            np.arange(row0, row1), np.arange(col0, col1), indexing="ij"
        )
        weight = _wendland_weight(rows, cols, center_row, center_col, support_radius)
        active = weight > 0
        if not active.any():
            continue
        region = np.s_[row0:row1, col0:col1]
        selected, offset, selected_score, patch = _select_site_patch(
            site=site,
            values=values,
            source_index=source_index,
            source_row0=source_row0,
            source_row1=source_row1,
            source_col0=source_col0,
            source_col1=source_col1,
            weight=weight,
            active=active,
            current_numerator=numerator[region],
            current_denominator=denominator[region],
            candidate_use=candidate_use,
            source_use=source_use,
        )
        numerator[region] += weight * patch
        denominator[region] += weight
        region_weight = dominant_weight[region]
        region_site = dominant_site[region]
        region_source = source_ownership[region]
        stronger = active & (weight > region_weight)
        region_weight[stronger] = weight[stronger]
        region_site[stronger] = site_index
        region_source[stronger] = source_index[selected]
        candidate_use[selected] += 1
        source_use[source_index[selected]] += 1
        placements.append(
            {
                "site_index": site_index,
                "global_row": site.global_row,
                "global_col": site.global_col,
                "source_id": candidates[selected].source_id,
                "source_y": candidates[selected].y,
                "source_x": candidates[selected].x,
                "transform": candidates[selected].transform,
                "support_fraction": candidates[selected].support_fraction,
                "offset_m": offset,
                "overlap_mse_m2": selected_score,
            }
        )
    if np.any(denominator <= 1e-12) or np.any(dominant_site < 0):
        uncovered = int(np.count_nonzero(denominator <= 1e-12))
        raise RuntimeError(f"irregular support left {uncovered} master samples uncovered")
    output = numerator / denominator
    output -= float(np.mean(output))
    return output, source_ownership, dominant_site, placements


def _boundary_metrics(
    residual: np.ndarray, allowed: np.ndarray, dominant_site: np.ndarray
) -> dict[str, float]:
    ratios: dict[str, float] = {}
    for axis, name in ((0, "north_south"), (1, "east_west")):
        gradient = np.abs(np.diff(residual, axis=axis))
        valid = (
            allowed[:-1] & allowed[1:]
            if axis == 0
            else allowed[:, :-1] & allowed[:, 1:]
        )
        boundary = (
            dominant_site[:-1] != dominant_site[1:]
            if axis == 0
            else dominant_site[:, :-1] != dominant_site[:, 1:]
        )
        boundary_values = gradient[valid & boundary]
        interior_values = gradient[valid & ~boundary]
        if not len(boundary_values) or not len(interior_values):
            raise RuntimeError("irregular boundary metric lacks comparison samples")
        ratios[f"{name}_mean_ratio"] = float(
            boundary_values.mean() / interior_values.mean()
        )
        ratios[f"{name}_p95_ratio"] = float(
            np.percentile(boundary_values, 95) / np.percentile(interior_values, 95)
        )
    return ratios


def run_estonia_irregular_forest_exemplar(*, repo_root: Path, output_root: Path) -> Path:
    """Build one no-grid Estonia research candidate from fixed measured sources."""
    transaction_root = (
        repo_root
        / "asset-gen/data/work/terrain-repair/taevaskoda-ahja-stage1-transaction"
        / TRANSACTION_ID
    )
    c0_path = transaction_root / "corrected-base/decoded-core/0/151_93.npy"
    transaction_path = transaction_root / "transaction.json"
    implementation_path = Path(__file__).resolve()
    e67_impl_path = implementation_path.with_name("run.py")
    anchoring_impl_path = implementation_path.with_name("estonia.py")
    recipe = {
        "schema": "estonia-forest-irregular-support-research/1",
        "authority": "research_preview_candidate_no_production_no_owner_no_latest",
        "regime": "forest.mesic_mineral",
        "method": "world_locked_multistage_saturated_sites_wendland_c2_partition",
        "transaction_id": TRANSACTION_ID,
        "transaction_sha256": _sha256(transaction_path),
        "corrected_c0_sha256": _sha256(c0_path),
        "master_bbox_en": list(MASTER_BBOX),
        "review_bbox_en": list(REVIEW_BBOX),
        "texel_m": TEXEL_M,
        "master_cells": 2560,
        "review_cells": 2048,
        "patch_cells": 192,
        "candidate_step_cells": 16,
        "minimum_patch_support_fraction": 0.78,
        "site_candidate_cell": 24,
        "site_minimum_distance_cells": 64,
        "site_saturation_distance_cells": 48,
        "site_saturation_rounds": 8,
        "support_radius_cells": 88,
        "site_seed": SEED,
        "selector": "exact_etak_forest_plus_known_unmodified_mesic_mineral_soil",
        "projection": "aoi_wide_smooth_mean_null_crop_last",
        "sources": SOURCE_SHA256,
        "implementation_sha256": _sha256(implementation_path),
        "e67_implementation_sha256": _sha256(e67_impl_path),
        "anchoring_implementation_sha256": _sha256(anchoring_impl_path),
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
    sites = _world_locked_sites(
        master_cells=recipe["master_cells"],
        candidate_cell=recipe["site_candidate_cell"],
        minimum_distance=recipe["site_minimum_distance_cells"],
        saturation_distance=recipe["site_saturation_distance_cells"],
        saturation_rounds=recipe["site_saturation_rounds"],
        support_radius=recipe["support_radius_cells"],
    )
    residual, ownership, dominant_site, placements = _assemble_irregular(
        candidates,
        sites,
        master_cells=recipe["master_cells"],
        patch_cells=recipe["patch_cells"],
        support_radius=recipe["support_radius_cells"],
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
        raise AssertionError("irregular projection changed a hard-excluded sample")
    cell_means = projected.reshape(160, FACTOR, 160, FACTOR).mean(axis=(1, 3))
    max_cell_mean = float(np.max(np.abs(cell_means)))
    if max_cell_mean >= 1e-10:
        raise AssertionError("irregular projection changed accepted one-metre C0 means")

    c0_crop = _crop(c0)
    c1_crop = _crop(c1)
    residual_crop = _crop(projected)
    ownership_crop = _crop(ownership)
    site_crop = _crop(dominant_site)
    mask_crop = type(morphology_mask)(
        **{
            name: _crop(getattr(morphology_mask, name))
            for name in morphology_mask.__dataclass_fields__
        }
    )
    allowed_crop = mask_crop.allowed
    pooled = np.concatenate([candidate.values.ravel() for candidate in candidates])
    output_values = residual_crop[allowed_crop]
    metrics = {
        "master_mask": morphology_mask.evidence(),
        "review_mask": mask_crop.evidence(),
        "site_count": len(sites),
        "placement_count": len(placements),
        "source_placement_counts": {
            source_id: sum(item["source_id"] == source_id for item in placements)
            for source_id in SOURCE_IDS
        },
        "boundary_gradient_ratios": _boundary_metrics(
            residual_crop, allowed_crop, site_crop
        ),
        "candidate_relief_p01_p99_m": [float(v) for v in np.percentile(pooled, [1, 99])],
        "output_relief_p01_p99_m": [float(v) for v in np.percentile(output_values, [1, 99])],
        "candidate_relief_rms_m": float(np.sqrt(np.mean(pooled * pooled))),
        "output_relief_rms_m": float(np.sqrt(np.mean(output_values * output_values))),
        "max_abs_candidate_relief_m": float(np.max(np.abs(pooled))),
        "max_abs_output_relief_m": float(np.max(np.abs(output_values))),
        "max_one_metre_mean_error_m": max_cell_mean,
        "max_hard_exclusion_residual_m": float(
            np.max(np.abs(residual_crop[~allowed_crop]))
        ) if (~allowed_crop).any() else 0.0,
    }

    with TemporaryDirectory(prefix="estonia-irregular-", dir=output_root) as temporary:
        staging = Path(temporary)
        (staging / "surface").mkdir()
        (staging / "qa").mkdir()
        (staging / "recipe.json").write_bytes(_canonical_json(recipe) + b"\n")
        np.save(staging / "surface/c0_height_f32.npy", c0_crop.astype(np.float32), allow_pickle=False)
        np.save(staging / "surface/c1_height_f32.npy", c1_crop.astype(np.float32), allow_pickle=False)
        np.save(staging / "surface/residual_f32.npy", residual_crop.astype(np.float32), allow_pickle=False)
        np.save(staging / "surface/allowed_u8.npy", allowed_crop.astype(np.uint8), allow_pickle=False)
        np.save(staging / "surface/source_ownership_u8.npy", ownership_crop, allow_pickle=False)
        np.save(staging / "surface/dominant_site_i32.npy", site_crop, allow_pickle=False)
        (staging / "surface/placements.json").write_bytes(_canonical_json(placements) + b"\n")

        c0_shade = np.repeat(_hillshade(c0_crop)[..., None], 3, axis=2)
        c1_shade = np.repeat(_hillshade(c1_crop)[..., None], 3, axis=2)
        _panel(
            [("CORRECTED C0 HILLSHADE", c0_shade), ("IRREGULAR FOREST C1 HILLSHADE", c1_shade)],
            staging / "qa/01_estonia_c0_c1.png",
            "128 m crop centered on Taevaskoda | no rectangular placement grid",
        )
        _panel(
            [("ADDED GEOMETRIC RELIEF", _percentile_rgb(residual_crop, diverging=True)),
             ("EXACT MORPHOLOGY SELECTOR", _mask_rgb(mask_crop))],
            staging / "qa/02_residual_and_selector.png",
            "measured forms only | hard exclusions unchanged",
        )
        palette = np.asarray([[45, 123, 113], [217, 139, 71], [103, 87, 145]], dtype=np.uint8)
        _panel(
            [("IRREGULAR SOURCE OWNERSHIP", palette[ownership_crop]),
             ("C1 MINUS C0", _percentile_rgb(residual_crop, diverging=True))],
            staging / "qa/03_irregular_ownership.png",
            "K11 teal | K32 orange | K36 violet | world-locked Matérn sites",
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
            "schema": "estonia-forest-irregular-support-artifact/1",
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
