"""Contiguous two-parent float proof for the accepted forest specialist."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import RectBivariateSpline

from assetgen.config import ASSET_GEN_ROOT, load_base
from assetgen.cook.pinned_height import PinnedBaseHeight
from assetgen.process.micro_fixture import conservative_cell_correct
from assetgen.process.micro_masks import rasterize_micro_morphology_mask
from assetgen.process.microtopo.projection import build_smooth_mean_null_projector

from .generalization import (
    BASE_MANIFEST,
    BASE_MANIFEST_SHA256,
    FACTOR,
    PARENT_M,
    _candidate_capacity,
    _hillshade,
    _load_candidates,
    _memmap,
)
from .irregular import (
    SOURCE_IDS,
    TEXEL_M,
    WORLD_ANCHOR_E,
    WORLD_ANCHOR_N,
    Site,
    _candidate_site,
    _select_site_patch,
    _wendland_weight,
)
from .run import SOURCE_SHA256, _canonical_json, _percentile_rgb, _sha256


OUTPUT_ROOT = (
    ASSET_GEN_ROOT
    / "data/work/microtopography/estonia-forest-adjacent-continuity/artifact/sha256"
)


def _read_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != "forest-mesic-mineral-adjacent-continuity/1":
        raise ValueError("unsupported adjacent-continuity config")
    if config.get("authority") != "research_float_only_no_production_no_pack_no_latest":
        raise ValueError("adjacent proof cannot authorize packing or production")
    if config.get("regime") != "forest.mesic_mineral" or config.get("texel_m") != TEXEL_M:
        raise ValueError("adjacent proof changed the accepted regime or lattice")
    bbox = tuple(map(int, config["bbox_en"]))
    if bbox != (684544, 6441472, 685056, 6442496):
        raise ValueError("adjacent proof domain differs from the frozen pair")
    parents = config["parents"]
    if [(row["cx"], row["cz"]) for row in parents] != [(617, 377), (617, 378)]:
        raise ValueError("adjacent parent identities differ")
    accepted_transport = {
        "patch_cells": 192,
        "candidate_step_cells": 16,
        "minimum_patch_support_fraction": 0.78,
        "site_candidate_cell": 24,
        "site_minimum_distance_cells": 64,
        "site_saturation_distance_cells": 48,
        "site_saturation_rounds": 8,
        "support_radius_cells": 88,
        "site_seed": 5494753648857171000,
    }
    if config["transport"] != accepted_transport:
        raise ValueError("adjacent proof retuned the accepted forest transport")
    return config


def _shape(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    e_min, n_min, e_max, n_max = bbox
    return round((n_max - n_min) / TEXEL_M), round((e_max - e_min) / TEXEL_M)


def _world_locked_sites_rect(
    *,
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    candidate_cell: int,
    minimum_distance: int,
    saturation_distance: int,
    saturation_rounds: int,
    support_radius: int,
) -> list[Site]:
    """Rectangular form of the accepted saturated world-site construction."""
    rows, cols = shape
    e_min, _n_min, _e_max, n_max = bbox
    master_col0 = int(round((e_min - WORLD_ANCHOR_E) / TEXEL_M))
    master_row0 = int(round((WORLD_ANCHOR_N - n_max) / TEXEL_M))
    margin = support_radius + minimum_distance + saturation_rounds * saturation_distance
    gx0 = (master_col0 - margin) // candidate_cell - 1
    gx1 = (master_col0 + cols + margin) // candidate_cell + 1
    gy0 = (master_row0 - margin) // candidate_cell - 1
    gy1 = (master_row0 + rows + margin) // candidate_cell + 1
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
    accepted = local_minima(available, minimum_distance)
    available = outside_sites(available - accepted, accepted, saturation_distance)
    for _round in range(saturation_rounds):
        if not available:
            break
        winners = local_minima(available, saturation_distance)
        accepted.update(winners)
        available = outside_sites(available - winners, accepted, saturation_distance)
    if available:
        raise RuntimeError("world-site saturation did not converge")
    result = []
    for key in accepted:
        site = all_sites[key]
        local_row = site.global_row - master_row0
        local_col = site.global_col - master_col0
        if (
            -support_radius < local_row < rows + support_radius
            and -support_radius < local_col < cols + support_radius
        ):
            result.append(site)
    result.sort(key=lambda site: (site.global_row, site.global_col))
    if not result:
        raise RuntimeError("world-site process produced no sites")
    return result


def _assemble(
    *,
    candidates,
    bbox: tuple[int, int, int, int],
    scratch: Path,
    config: dict[str, Any],
) -> tuple[Path, Path, Path, list[dict[str, Any]], dict[str, int], int]:
    rows, cols = _shape(bbox)
    transport = config["transport"]
    support_radius = int(transport["support_radius_cells"])
    sites = _world_locked_sites_rect(
        shape=(rows, cols),
        bbox=bbox,
        candidate_cell=int(transport["site_candidate_cell"]),
        minimum_distance=int(transport["site_minimum_distance_cells"]),
        saturation_distance=int(transport["site_saturation_distance_cells"]),
        saturation_rounds=int(transport["site_saturation_rounds"]),
        support_radius=support_radius,
    )
    values = np.stack([candidate.values for candidate in candidates]).astype(np.float32)
    source_index = np.asarray([candidate.source_index for candidate in candidates], dtype=np.int16)
    numerator = _memmap(scratch / "numerator.f64", np.float64, (rows, cols))
    denominator = _memmap(scratch / "denominator.f64", np.float64, (rows, cols))
    dominant_weight = _memmap(scratch / "dominant-weight.f32", np.float32, (rows, cols))
    dominant_site = _memmap(scratch / "dominant-site.i32", np.int32, (rows, cols), fill=-1)
    ownership = _memmap(scratch / "ownership.u8", np.uint8, (rows, cols))
    candidate_use = np.zeros(len(candidates), dtype=np.int32)
    source_use = np.zeros(len(SOURCE_IDS), dtype=np.int32)
    placements: list[dict[str, Any]] = []
    patch_cells = int(transport["patch_cells"])
    half = patch_cells // 2
    e_min, _n_min, _e_max, n_max = bbox
    master_col0 = int(round((e_min - WORLD_ANCHOR_E) / TEXEL_M))
    master_row0 = int(round((WORLD_ANCHOR_N - n_max) / TEXEL_M))
    seam_row = rows // 2
    cross_seam_sites = 0
    for site_index, site in enumerate(sites):
        center_row = site.global_row - master_row0
        center_col = site.global_col - master_col0
        row0 = max(0, center_row - half)
        row1 = min(rows, center_row + half)
        col0 = max(0, center_col - half)
        col1 = min(cols, center_col + half)
        if row0 >= row1 or col0 >= col1:
            continue
        source_row0 = row0 - (center_row - half)
        source_row1 = source_row0 + row1 - row0
        source_col0 = col0 - (center_col - half)
        source_col1 = source_col0 + col1 - col0
        grid_rows, grid_cols = np.meshgrid(
            np.arange(row0, row1), np.arange(col0, col1), indexing="ij"
        )
        weight = _wendland_weight(
            grid_rows, grid_cols, center_row, center_col, support_radius
        )
        active = weight > 0
        if not active.any():
            continue
        region = np.s_[row0:row1, col0:col1]
        selected, match_offset, score, patch = _select_site_patch(
            site=site,
            values=values,
            source_index=source_index,
            source_row0=source_row0,
            source_row1=source_row1,
            source_col0=source_col0,
            source_col1=source_col1,
            weight=weight,
            active=active,
            current_numerator=np.asarray(numerator[region]),
            current_denominator=np.asarray(denominator[region]),
            candidate_use=candidate_use,
            source_use=source_use,
        )
        numerator[region] += weight * patch
        denominator[region] += weight
        local_weight = dominant_weight[region]
        stronger = active & (weight > local_weight)
        local_weight[stronger] = weight[stronger]
        local_site = dominant_site[region]
        local_site[stronger] = site_index
        local_source = ownership[region]
        local_source[stronger] = source_index[selected]
        candidate_use[selected] += 1
        source_use[source_index[selected]] += 1
        if row0 < seam_row < row1 and np.any(active[seam_row - row0 - 1 : seam_row - row0 + 1]):
            cross_seam_sites += 1
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
                "match_offset_m": match_offset,
                "applied_offset_m": 0.0,
                "overlap_mse_m2": score,
            }
        )
    denominator.flush()
    if np.any(np.asarray(denominator) <= 1e-12) or np.any(np.asarray(dominant_site) < 0):
        raise RuntimeError("single adjacent-domain transport left samples uncovered")
    residual_path = scratch / "residual.f32"
    residual = np.memmap(residual_path, dtype=np.float32, mode="w+", shape=(rows, cols))
    row_step = 128
    total = 0.0
    for row0 in range(0, rows, row_step):
        row1 = min(rows, row0 + row_step)
        total += float(np.sum(numerator[row0:row1] / denominator[row0:row1], dtype=np.float64))
    mean = total / float(rows * cols)
    for row0 in range(0, rows, row_step):
        row1 = min(rows, row0 + row_step)
        residual[row0:row1] = numerator[row0:row1] / denominator[row0:row1] - mean
    residual.flush()
    del numerator, denominator, dominant_weight, residual
    for path in (
        scratch / "numerator.f64",
        scratch / "denominator.f64",
        scratch / "dominant-weight.f32",
    ):
        path.unlink()
    return (
        residual_path,
        scratch / "ownership.u8",
        scratch / "dominant-site.i32",
        placements,
        {source_id: int(source_use[index]) for index, source_id in enumerate(SOURCE_IDS)},
        cross_seam_sites,
    )


def _fine_mask(
    bbox: tuple[int, int, int, int], path: Path
) -> tuple[np.memmap, dict[str, int]]:
    rows, cols = _shape(bbox)
    allowed = _memmap(path, np.uint8, (rows, cols))
    evidence: dict[str, int] = {}
    e_min, _n_min, _e_max, n_max = bbox
    tile_cells = 2048
    for tile_row in range(rows // tile_cells):
        for tile_col in range(cols // tile_cells):
            tile_e = e_min + tile_col * 128
            tile_n = n_max - tile_row * 128
            east = tile_e + (np.arange(tile_cells) + 0.5) * TEXEL_M
            north = tile_n - (np.arange(tile_cells) + 0.5) * TEXEL_M
            mask = rasterize_micro_morphology_mask(east, north)
            region = np.s_[
                tile_row * tile_cells : (tile_row + 1) * tile_cells,
                tile_col * tile_cells : (tile_col + 1) * tile_cells,
            ]
            allowed[region] = mask.allowed
            for key, value in mask.evidence().items():
                if isinstance(value, int):
                    evidence[key] = evidence.get(key, 0) + value
    allowed.flush()
    return allowed, evidence


def _project(residual_path: Path, allowed: np.memmap, shape: tuple[int, int]) -> float:
    rows, cols = shape
    coarse_rows, coarse_cols = rows // FACTOR, cols // FACTOR
    residual = np.memmap(residual_path, dtype=np.float32, mode="r+", shape=shape)
    fully_soft = np.empty((coarse_rows, coarse_cols), dtype=bool)
    for row in range(coarse_rows):
        fine = np.asarray(allowed[row * FACTOR : (row + 1) * FACTOR], dtype=bool)
        fully_soft[row] = fine.reshape(FACTOR, coarse_cols, FACTOR).all(axis=(0, 2))
    taper_plan = build_smooth_mean_null_projector(
        np.zeros((coarse_rows, coarse_cols)), fully_soft, factor=FACTOR
    )
    means = np.empty((coarse_rows, coarse_cols), dtype=np.float64)
    coarse_step = 8
    for row0 in range(0, coarse_rows, coarse_step):
        block_rows = min(coarse_step, coarse_rows - row0)
        fine_rows = np.s_[row0 * FACTOR : (row0 + block_rows) * FACTOR]
        fine = np.asarray(residual[fine_rows], dtype=np.float64)
        soft = np.asarray(allowed[fine_rows], dtype=bool)
        taper = taper_plan.taper_window(row0, 0, block_rows, coarse_cols)
        means[row0 : row0 + block_rows] = np.where(soft, fine * taper, 0.0).reshape(
            block_rows, FACTOR, coarse_cols, FACTOR
        ).mean(axis=(1, 3))
    projector = build_smooth_mean_null_projector(means, fully_soft, factor=FACTOR)
    maximum_mean = 0.0
    for row0 in range(0, coarse_rows, coarse_step):
        block_rows = min(coarse_step, coarse_rows - row0)
        fine_rows = np.s_[row0 * FACTOR : (row0 + block_rows) * FACTOR]
        projected = projector.project_window(
            np.asarray(residual[fine_rows], dtype=np.float64),
            np.asarray(allowed[fine_rows], dtype=bool),
            row0=row0,
            col0=0,
        )
        block_means = projected.reshape(
            block_rows, FACTOR, coarse_cols, FACTOR
        ).mean(axis=(1, 3))
        maximum_mean = max(maximum_mean, float(np.max(np.abs(block_means))))
        residual[fine_rows] = projected.astype(np.float32)
    residual.flush()
    return maximum_mean


def _write_c1(
    *,
    base: PinnedBaseHeight,
    bbox: tuple[int, int, int, int],
    residual_path: Path,
    destination: Path,
) -> tuple[np.memmap, np.ndarray]:
    rows, cols = _shape(bbox)
    coarse_rows, coarse_cols = rows // FACTOR, cols // FACTOR
    e_min, _n_min, _e_max, n_max = bbox
    extended = base.read_cells(
        e_min - 4, n_max + 4, coarse_cols + 8, coarse_rows + 8
    ).astype(np.float64)
    authority = np.asarray(extended[4:-4, 4:-4], dtype=np.float64)
    east_centers = e_min - 4 + np.arange(coarse_cols + 8, dtype=np.float64) + 0.5
    north_centers = n_max + 4 - np.arange(coarse_rows + 8, dtype=np.float64) - 0.5
    spline = RectBivariateSpline(
        north_centers[::-1], east_centers, extended[::-1], kx=3, ky=3, s=0.0
    )
    east = e_min + (np.arange(cols, dtype=np.float64) + 0.5) * TEXEL_M
    residual = np.memmap(residual_path, dtype=np.float32, mode="r", shape=(rows, cols))
    c1 = np.lib.format.open_memmap(
        destination, mode="w+", dtype=np.float32, shape=(rows, cols)
    )
    coarse_step = 8
    for row0 in range(0, coarse_rows, coarse_step):
        block_rows = min(coarse_step, coarse_rows - row0)
        north = n_max - (
            np.arange(row0 * FACTOR, (row0 + block_rows) * FACTOR, dtype=np.float64)
            + 0.5
        ) * TEXEL_M
        smooth = spline(north[::-1], east, grid=True)[::-1]
        c0 = conservative_cell_correct(
            smooth, authority[row0 : row0 + block_rows], FACTOR
        )
        fine_rows = np.s_[row0 * FACTOR : (row0 + block_rows) * FACTOR]
        c1[fine_rows] = (c0 + residual[fine_rows]).astype(np.float32)
    c1.flush()
    return c1, authority


def _reduce8(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rows, cols = shape[0] // 8, shape[1] // 8
    result = np.empty((rows, cols), dtype=np.float32)
    for row0 in range(0, rows, 32):
        block_rows = min(32, rows - row0)
        source = np.asarray(array[row0 * 8 : (row0 + block_rows) * 8], dtype=np.float32)
        result[row0 : row0 + block_rows] = source.reshape(
            block_rows, 8, cols, 8
        ).mean(axis=(1, 3))
    return result


def _resize(image: np.ndarray, width: int) -> Image.Image:
    height = round(image.shape[0] * width / image.shape[1])
    return Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR)


def _primary_qa(
    path: Path,
    c0: np.ndarray,
    c1: np.ndarray,
    seam_row: int,
) -> None:
    c0_rgb = np.repeat(_hillshade(c0, 0.5)[..., None], 3, axis=2)
    c1_rgb = np.repeat(_hillshade(c1, 0.5)[..., None], 3, axis=2)
    full_width = 520
    top = 62
    full_c0, full_c1 = _resize(c0_rgb, full_width), _resize(c1_rgb, full_width)
    strip_half = 128
    seam_c0 = _resize(c0_rgb[seam_row - strip_half : seam_row + strip_half], full_width)
    seam_c1 = _resize(c1_rgb[seam_row - strip_half : seam_row + strip_half], full_width)
    canvas = Image.new("RGB", (1080, 1310), (244, 242, 234))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), "01  CONTIGUOUS FOREST MASTER + SHARED PARENT SEAM", fill=(20, 24, 20))
    draw.text((16, 34), "one 512 x 1024 m solve; identical common light; red = parent boundary", fill=(70, 72, 66))
    canvas.paste(full_c0, (16, top))
    canvas.paste(full_c1, (544, top))
    draw.text((16, top - 18), "PINNED C0 FULL MASTER", fill=(20, 24, 20))
    draw.text((544, top - 18), "FOREST C1 FULL MASTER", fill=(20, 24, 20))
    full_seam_y = top + round(seam_row * full_c0.height / c0.shape[0])
    draw.line((16, full_seam_y, 16 + full_width, full_seam_y), fill=(220, 40, 30), width=3)
    draw.line((544, full_seam_y, 544 + full_width, full_seam_y), fill=(220, 40, 30), width=3)
    strip_y = 1120
    canvas.paste(seam_c0, (16, strip_y))
    canvas.paste(seam_c1, (544, strip_y))
    draw.text((16, strip_y - 20), "C0 128 m SEAM STRIP", fill=(20, 24, 20))
    draw.text((544, strip_y - 20), "C1 128 m SEAM STRIP", fill=(20, 24, 20))
    line_y = strip_y + seam_c0.height // 2
    draw.line((16, line_y, 16 + full_width, line_y), fill=(220, 40, 30), width=3)
    draw.line((544, line_y, 544 + full_width, line_y), fill=(220, 40, 30), width=3)
    canvas.save(path, compress_level=9)


def _secondary_qa(
    qa: Path,
    residual: np.ndarray,
    ownership: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    allowed: np.ndarray,
    seam_row: int,
) -> list[Path]:
    paths = [
        qa / "02_residual_and_source_ownership.png",
        qa / "03_seam_ground_scale_before_after.png",
        qa / "04_parent_contexts_common_light.png",
    ]
    palette = np.asarray([[45, 123, 113], [217, 139, 71], [103, 87, 145]], dtype=np.uint8)
    residual_rgb = _percentile_rgb(residual, diverging=True)
    ownership_rgb = palette[np.clip(ownership, 0, 2)]
    residual_image = _resize(residual_rgb, 640)
    ownership_image = _resize(ownership_rgb, 640)
    canvas = Image.new("RGB", (1320, 1370), (244, 242, 234))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), "02  ADDED RELIEF + SINGLE WORLD-LOCKED OWNERSHIP FIELD", fill=(20, 24, 20))
    draw.text((16, 34), "red seam is an inspection guide, not a synthesis boundary", fill=(70, 72, 66))
    canvas.paste(residual_image, (16, 62))
    canvas.paste(ownership_image, (664, 62))
    draw.text((16, 44), "SIGNED RESIDUAL", fill=(20, 24, 20))
    draw.text((664, 44), "K11 TEAL / K32 ORANGE / K36 VIOLET", fill=(20, 24, 20))
    line_y = 62 + round(seam_row * residual_image.height / residual.shape[0])
    draw.line((16, line_y, 656, line_y), fill=(220, 40, 30), width=3)
    draw.line((664, line_y, 1304, line_y), fill=(220, 40, 30), width=3)
    canvas.save(paths[0], compress_level=9)

    # Select a 32 m seam crop with the most fully eligible one-metre cells.
    # QA arrays are reduced from 6.25 cm to 0.5 m, so one metre is 2 x 2 here.
    allowed_1m = allowed.reshape(1024, 2, PARENT_M, 2).all(axis=(1, 3))
    best = max(
        (
            int(allowed_1m[496:528, x : x + 32].sum()),
            -x,
            x,
        )
        for x in range(0, PARENT_M - 31, 16)
    )[-1]
    crop = np.s_[
        (512 - 16) * 2 : (512 + 16) * 2,
        best * 2 : (best + 32) * 2,
    ]
    c0_crop = np.repeat(_hillshade(c0[crop], 0.5)[..., None], 3, axis=2)
    c1_crop = np.repeat(_hillshade(c1[crop], 0.5)[..., None], 3, axis=2)
    left = Image.fromarray(c0_crop).resize((640, 640), Image.Resampling.BILINEAR)
    right = Image.fromarray(c1_crop).resize((640, 640), Image.Resampling.BILINEAR)
    close = Image.new("RGB", (1320, 708), (244, 242, 234))
    draw = ImageDraw.Draw(close)
    draw.text((16, 12), "03  GROUND-SCALE SHARED-SEAM CLOSEUP", fill=(20, 24, 20))
    draw.text((16, 34), f"32 x 32 m around parent boundary; parent cell x={best}", fill=(70, 72, 66))
    close.paste(left, (16, 62))
    close.paste(right, (664, 62))
    draw.text((16, 44), "C0", fill=(20, 24, 20))
    draw.text((664, 44), "C1", fill=(20, 24, 20))
    draw.line((16, 382, 656, 382), fill=(220, 40, 30), width=3)
    draw.line((664, 382, 1304, 382), fill=(220, 40, 30), width=3)
    close.save(paths[1], compress_level=9)

    shade = np.repeat(_hillshade(c1, 0.5)[..., None], 3, axis=2)
    north = Image.fromarray(shade[:1024]).resize((640, 640), Image.Resampling.BILINEAR)
    south = Image.fromarray(shade[1024:]).resize((640, 640), Image.Resampling.BILINEAR)
    parents = Image.new("RGB", (1320, 708), (244, 242, 234))
    draw = ImageDraw.Draw(parents)
    draw.text((16, 12), "04  BOTH PARENTS AT IDENTICAL COMMON LIGHT + SCALE", fill=(20, 24, 20))
    draw.text((16, 34), "new north parent (617,377) | accepted south parent (617,378)", fill=(70, 72, 66))
    parents.paste(north, (16, 62))
    parents.paste(south, (664, 62))
    draw.text((16, 44), "NEW NORTH", fill=(20, 24, 20))
    draw.text((664, 44), "ACCEPTED SOUTH", fill=(20, 24, 20))
    parents.save(paths[2], compress_level=9)
    return paths


def _identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(ASSET_GEN_ROOT.parent)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def run(config_path: Path) -> Path:
    config = _read_config(config_path)
    bbox = tuple(map(int, config["bbox_en"]))
    shape = _shape(bbox)
    recipe = {
        "schema": "forest-mesic-mineral-adjacent-continuity-recipe/1",
        "authority": config["authority"],
        "config": _identity(config_path),
        "implementation": _identity(Path(__file__).resolve()),
        "accepted_generator": {
            "generalization": _identity(Path(__file__).with_name("generalization.py")),
            "irregular": _identity(Path(__file__).with_name("irregular.py")),
        },
        "base_manifest_sha256": BASE_MANIFEST_SHA256,
        "sources": SOURCE_SHA256,
        "bbox_en": list(bbox),
        "shape": list(shape),
        "single_domain_solve_before_parent_crops": True,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    build_id = hashlib.sha256(_canonical_json(recipe)).hexdigest()
    destination = OUTPUT_ROOT / build_id
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    candidates = _load_candidates()
    capacity = _candidate_capacity(candidates)
    base = PinnedBaseHeight(
        BASE_MANIFEST,
        BASE_MANIFEST_SHA256,
        ASSET_GEN_ROOT / "data/out",
        load_base().encode,
        audit=False,
    )
    with TemporaryDirectory(
        prefix=f".{build_id}.", dir=destination.parent, delete=False
    ) as temporary:
        staging = Path(temporary)
        surface = staging / "surface"
        qa = staging / "qa"
        scratch = staging / "scratch"
        surface.mkdir()
        qa.mkdir()
        scratch.mkdir()
        (staging / "recipe.json").write_bytes(_canonical_json(recipe) + b"\n")
        (
            residual_path,
            ownership_path,
            dominant_path,
            placements,
            source_counts,
            cross_seam_sites,
        ) = _assemble(
            candidates=candidates,
            bbox=bbox,
            scratch=scratch,
            config=config,
        )
        allowed, mask_evidence = _fine_mask(bbox, scratch / "allowed.u8")
        maximum_mean = _project(residual_path, allowed, shape)
        residual = np.memmap(residual_path, dtype=np.float32, mode="r", shape=shape)
        maximum_hard = 0.0
        maximum_unprojected = 0.0
        for row0 in range(0, shape[0], 128):
            row1 = min(shape[0], row0 + 128)
            values = np.asarray(residual[row0:row1])
            maximum_unprojected = max(
                maximum_unprojected, float(np.max(np.abs(values), initial=0.0))
            )
            hard = np.where(allowed[row0:row1] == 0, np.abs(values), 0.0)
            maximum_hard = max(maximum_hard, float(np.max(hard, initial=0.0)))
        c1, authority = _write_c1(
            base=base,
            bbox=bbox,
            residual_path=residual_path,
            destination=surface / "c1_height_f32.npy",
        )
        ownership = np.memmap(ownership_path, dtype=np.uint8, mode="r", shape=shape)
        ownership_1m = np.asarray(
            ownership[FACTOR // 2 :: FACTOR, FACTOR // 2 :: FACTOR], dtype=np.uint8
        )
        np.save(surface / "source_ownership_1m_u8.npy", ownership_1m, allow_pickle=False)
        (surface / "placements.json").write_bytes(_canonical_json(placements) + b"\n")

        c1_half = _reduce8(c1, shape)
        residual_half = _reduce8(residual, shape)
        c0_half = c1_half - residual_half
        allowed_half = _reduce8(allowed, shape) >= 1.0
        seam_row = c1_half.shape[0] // 2
        seam_valid = allowed_half[seam_row - 1] & allowed_half[seam_row]
        seam_steps = np.abs(residual_half[seam_row] - residual_half[seam_row - 1])[seam_valid]
        neighborhood = np.abs(np.diff(residual_half[seam_row - 64 : seam_row + 65], axis=0))
        neighborhood_valid = (
            allowed_half[seam_row - 64 : seam_row + 64]
            & allowed_half[seam_row - 63 : seam_row + 65]
        )
        neighborhood_valid[63] = False
        local_steps = neighborhood[neighborhood_valid]
        output = residual[np.asarray(allowed, dtype=bool)]
        output_rms = float(np.sqrt(np.mean(output * output)))
        output_maximum = float(np.max(np.abs(output), initial=0.0))
        parent_allowed = [
            float(np.mean(allowed[: shape[0] // 2])),
            float(np.mean(allowed[shape[0] // 2 :])),
        ]
        seam = {
            "cross_seam_site_count": cross_seam_sites,
            "eligible_sample_count": int(seam_steps.size),
            "residual_step_mean_m": float(np.mean(seam_steps)) if seam_steps.size else None,
            "residual_step_p95_m": float(np.percentile(seam_steps, 95)) if seam_steps.size else None,
            "local_nonseam_step_mean_m": float(np.mean(local_steps)) if local_steps.size else None,
            "local_nonseam_step_p95_m": float(np.percentile(local_steps, 95)) if local_steps.size else None,
        }
        _primary_qa(qa / "01_contiguous_master_and_shared_seam.png", c0_half, c1_half, seam_row)
        pngs = [qa / "01_contiguous_master_and_shared_seam.png"]
        pngs.extend(
            _secondary_qa(
                qa,
                residual_half,
                ownership_1m,
                c0_half,
                c1_half,
                allowed_half,
                seam_row,
            )
        )
        acceptance = config["acceptance"]
        failures = []
        if min(parent_allowed) < acceptance["minimum_parent_allowed_fraction"]:
            failures.append("parent eligibility")
        if maximum_mean > acceptance["maximum_one_metre_mean_error_m"]:
            failures.append("one-metre mean closure")
        if maximum_hard > acceptance["maximum_hard_exclusion_residual_m"]:
            failures.append("hard exclusion")
        if output_maximum / capacity["maximum_abs_m"] > acceptance["maximum_output_over_candidate_abs_ratio"]:
            failures.append("measured maximum envelope")
        if output_rms / capacity["rms_m"] > acceptance["maximum_output_over_candidate_rms_ratio"]:
            failures.append("measured RMS envelope")
        if cross_seam_sites < acceptance["minimum_cross_seam_site_count"]:
            failures.append("no ownership support crosses parent seam")
        metrics = {
            "bbox_en": list(bbox),
            "shape": list(shape),
            "parent_allowed_fraction_north_south": parent_allowed,
            "mask": mask_evidence,
            "site_count": len({item["site_index"] for item in placements}),
            "placement_count": len(placements),
            "source_placement_counts": source_counts,
            "seam": seam,
            "candidate_capacity": capacity,
            "residual_rms_allowed_m": output_rms,
            "residual_p01_p99_allowed_m": [float(v) for v in np.percentile(output, [1, 99])],
            "maximum_abs_residual_m": output_maximum,
            "maximum_abs_unprojected_residual_m": maximum_unprojected,
            "output_over_candidate_abs_ratio": output_maximum / capacity["maximum_abs_m"],
            "output_over_candidate_rms_ratio": output_rms / capacity["rms_m"],
            "maximum_one_metre_mean_error_m": maximum_mean,
            "maximum_hard_exclusion_residual_m": maximum_hard,
            "authority_p01_p99_m": [float(v) for v in np.percentile(authority, [1, 99])],
        }
        (staging / "metrics.json").write_bytes(_canonical_json(metrics) + b"\n")
        del output, c1, residual, allowed, ownership
        shutil.rmtree(scratch)
        qa_index = {
            "schema": "forest-mesic-mineral-adjacent-continuity-qa/1",
            "build_id": build_id,
            "images": [
                {
                    "path": f"qa/{path.name}",
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "dimensions_px": list(Image.open(path).size),
                }
                for path in pngs
            ],
            "interpretation": [
                "01 is primary: both complete parents and a seam-centered strip use identical common light; red is only the storage-parent boundary.",
                "02 shows one signed residual and ownership field solved over the full contiguous domain before any parent crop.",
                "03 is a 32 m ground-scale before/after crop centered exactly on the shared boundary.",
                "04 compares the complete new and previously accepted parent at identical scale and light.",
            ],
        }
        (qa / "index.json").write_bytes(_canonical_json(qa_index) + b"\n")
        files = {
            str(path.relative_to(staging)): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema": "forest-mesic-mineral-adjacent-continuity-artifact/1",
            "build_id": build_id,
            "status": "inspect_float_preview" if not failures else "park_before_inspection",
            "failures": failures,
            "metrics": metrics,
            "authority": {
                "production": False,
                "packing": False,
                "browser": False,
                "latest": False,
                "national": False,
            },
            "limitations": [
                "foreign analogue research owner, not Estonia target truth or production authority",
                "one contiguous pair extends continuity evidence only; it does not authorize wide coverage",
                "the accepted forest morphology and transport parameters were not retuned",
            ],
            "files": files,
        }
        (staging / "manifest.json").write_bytes(_canonical_json(manifest) + b"\n")
        os.replace(staging, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    print(run(parser.parse_args().config))


if __name__ == "__main__":
    main()
