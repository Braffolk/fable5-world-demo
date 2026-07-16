"""Contiguous multi-parent float proofs for the accepted forest specialist."""

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


PAIR_OUTPUT_ROOT = (
    ASSET_GEN_ROOT
    / "data/work/microtopography/estonia-forest-adjacent-continuity/artifact/sha256"
)
BLOCK_OUTPUT_ROOT = (
    ASSET_GEN_ROOT
    / "data/work/microtopography/estonia-forest-adjacent-block/artifact/sha256"
)
TRANCHE_OUTPUT_ROOT = (
    ASSET_GEN_ROOT
    / "data/work/microtopography/estonia-forest-lod0-tranche/artifact/sha256"
)


def _read_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    schema = config.get("schema")
    if schema not in {
        "forest-mesic-mineral-adjacent-continuity/1",
        "forest-mesic-mineral-adjacent-block/1",
        "forest-mesic-mineral-lod0-tranche/1",
    }:
        raise ValueError("unsupported adjacent-continuity config")
    if config.get("authority") != "research_float_only_no_production_no_pack_no_latest":
        raise ValueError("adjacent proof cannot authorize packing or production")
    if config.get("regime") != "forest.mesic_mineral" or config.get("texel_m") != TEXEL_M:
        raise ValueError("adjacent proof changed the accepted regime or lattice")
    bbox = tuple(map(int, config["bbox_en"]))
    parents = [(row["cx"], row["cz"]) for row in config["parents"]]
    if schema == "forest-mesic-mineral-adjacent-continuity/1":
        if bbox != (684544, 6441472, 685056, 6442496):
            raise ValueError("adjacent proof domain differs from the frozen pair")
        if parents != [(617, 377), (617, 378)]:
            raise ValueError("adjacent parent identities differ")
    elif schema == "forest-mesic-mineral-adjacent-block/1":
        if bbox != (684544, 6441472, 685568, 6442496):
            raise ValueError("adjacent block differs from the qualified 2x2 domain")
        if parents != [(617, 377), (618, 377), (617, 378), (618, 378)]:
            raise ValueError("adjacent block parent identities differ")
    else:
        if bbox != (684032, 6440960, 686080, 6443008):
            raise ValueError("LOD0 tranche differs from the declared complete domain")
        if config.get("lod0") != [0, 154, 94]:
            raise ValueError("LOD0 tranche identity changed")
        grid = config.get("parent_grid")
        if grid != {"rows": 4, "cols": 4, "northwest": [-1, 616, 376]}:
            raise ValueError("LOD0 tranche parent grid changed")
        expected_parents = [
            (cx, cz) for cz in range(376, 380) for cx in range(616, 620)
        ]
        if parents != expected_parents:
            raise ValueError("LOD0 tranche parent identities differ")
        for index, row in enumerate(config["parents"]):
            grid_row, grid_col = divmod(index, 4)
            expected_bbox = [
                684032 + grid_col * PARENT_M,
                6443008 - (grid_row + 1) * PARENT_M,
                684032 + (grid_col + 1) * PARENT_M,
                6443008 - grid_row * PARENT_M,
            ]
            if row.get("role") != f"r{grid_row}_c{grid_col}" or row.get("bbox_en") != expected_bbox:
                raise ValueError(f"LOD0 tranche parent declaration changed at index {index}")
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


def _is_block(config: dict[str, Any]) -> bool:
    return config["schema"] == "forest-mesic-mineral-adjacent-block/1"


def _is_tranche(config: dict[str, Any]) -> bool:
    return config["schema"] == "forest-mesic-mineral-lod0-tranche/1"


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
) -> tuple[Path, Path, Path, list[dict[str, Any]], dict[str, int], dict[str, int]]:
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
    parent_cells = round(PARENT_M / TEXEL_M)
    seam_rows = tuple(range(parent_cells, rows, parent_cells))
    seam_cols = tuple(range(parent_cells, cols, parent_cells))
    crossing_sites = {"horizontal": 0, "vertical": 0, "junction": 0}
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
        crosses_horizontal = any(
            abs(center_row - seam_row) < support_radius for seam_row in seam_rows
        )
        crosses_vertical = any(
            abs(center_col - seam_col) < support_radius for seam_col in seam_cols
        )
        if crosses_horizontal:
            crossing_sites["horizontal"] += 1
        if crosses_vertical:
            crossing_sites["vertical"] += 1
        if crosses_horizontal and crosses_vertical:
            crossing_sites["junction"] += 1
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
        crossing_sites,
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


def _draw_cross(
    draw: ImageDraw.ImageDraw,
    *,
    left: int,
    top: int,
    width: int,
    height: int,
) -> None:
    draw.line((left + width // 2, top, left + width // 2, top + height), fill=(220, 40, 30), width=3)
    draw.line((left, top + height // 2, left + width, top + height // 2), fill=(220, 40, 30), width=3)


def _block_qa(
    qa: Path,
    residual: np.ndarray,
    ownership: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    allowed: np.ndarray,
) -> list[Path]:
    paths = [
        qa / "01_full_master_common_light.png",
        qa / "02_four_parent_junction_closeups.png",
        qa / "03_eligibility_hard_mask_overlay.png",
        qa / "04_ground_scale_transition_crops.png",
    ]
    shade0 = np.repeat(_hillshade(c0, 0.5)[..., None], 3, axis=2)
    shade1 = np.repeat(_hillshade(c1, 0.5)[..., None], 3, axis=2)

    full0 = _resize(shade0, 640)
    full1 = _resize(shade1, 640)
    canvas = Image.new("RGB", (1328, 718), (244, 242, 234))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), "01  FULL 2x2 MASTER AT COMMON LIGHT", fill=(20, 24, 20))
    draw.text((16, 34), "one 1024 x 1024 m solve; red cross marks storage parents only", fill=(70, 72, 66))
    canvas.paste(full0, (16, 62))
    canvas.paste(full1, (672, 62))
    draw.text((16, 44), "PINNED C0", fill=(20, 24, 20))
    draw.text((672, 44), "FOREST C1", fill=(20, 24, 20))
    _draw_cross(draw, left=16, top=62, width=640, height=640)
    _draw_cross(draw, left=672, top=62, width=640, height=640)
    canvas.save(paths[0], compress_level=9)

    center_row, center_col = c1.shape[0] // 2, c1.shape[1] // 2
    half = 128
    crop = np.s_[center_row - half : center_row + half, center_col - half : center_col + half]
    own_center = ownership.shape[0] // 2
    own_half = 64
    own_crop = ownership[
        own_center - own_half : own_center + own_half,
        own_center - own_half : own_center + own_half,
    ]
    palette = np.asarray([[45, 123, 113], [217, 139, 71], [103, 87, 145]], dtype=np.uint8)
    panels = [
        ("C0 JUNCTION", shade0[crop]),
        ("C1 JUNCTION", shade1[crop]),
        ("SIGNED RESIDUAL", _percentile_rgb(residual[crop], diverging=True)),
        ("SOURCE OWNERSHIP", palette[np.clip(own_crop, 0, 2)]),
    ]
    junction = Image.new("RGB", (1056, 1092), (244, 242, 234))
    draw = ImageDraw.Draw(junction)
    draw.text((16, 12), "02  FOUR-PARENT JUNCTION CLOSEUPS", fill=(20, 24, 20))
    draw.text((16, 34), "128 x 128 m centered on the common junction; identical light and scale", fill=(70, 72, 66))
    for index, (label, panel) in enumerate(panels):
        left = 16 + (index % 2) * 520
        top = 72 + (index // 2) * 510
        image = Image.fromarray(panel).resize((504, 504), Image.Resampling.BILINEAR)
        junction.paste(image, (left, top))
        draw.text((left, top - 18), label, fill=(20, 24, 20))
        _draw_cross(draw, left=left, top=top, width=504, height=504)
    junction.save(paths[1], compress_level=9)

    hard_rgb = np.zeros((*allowed.shape, 3), dtype=np.uint8)
    hard_rgb[allowed] = (54, 135, 72)
    hard_rgb[~allowed] = (190, 67, 48)
    overlay = shade1.copy()
    overlay[allowed] = (0.72 * overlay[allowed] + 0.28 * np.asarray((54, 135, 72))).astype(np.uint8)
    overlay[~allowed] = (0.58 * overlay[~allowed] + 0.42 * np.asarray((190, 67, 48))).astype(np.uint8)
    mask_image = _resize(hard_rgb, 640)
    overlay_image = _resize(overlay, 640)
    masks = Image.new("RGB", (1328, 718), (244, 242, 234))
    draw = ImageDraw.Draw(masks)
    draw.text((16, 12), "03  ELIGIBILITY / HARD-EXCLUSION OVERLAY", fill=(20, 24, 20))
    draw.text((16, 34), "green = forest synthesis eligible; red = hard zero residual", fill=(70, 72, 66))
    masks.paste(mask_image, (16, 62))
    masks.paste(overlay_image, (672, 62))
    draw.text((16, 44), "CATEGORICAL MASK", fill=(20, 24, 20))
    draw.text((672, 44), "MASK OVER C1", fill=(20, 24, 20))
    _draw_cross(draw, left=16, top=62, width=640, height=640)
    _draw_cross(draw, left=672, top=62, width=640, height=640)
    masks.save(paths[2], compress_level=9)

    # Choose two spatially separated 48 m windows with the strongest real
    # eligible-to-hard transition in the southeast parent.
    window = 96
    candidates: list[tuple[float, int, int]] = []
    for row in range(center_row, allowed.shape[0] - window + 1, 32):
        for col in range(center_col, allowed.shape[1] - window + 1, 32):
            fraction = float(np.mean(allowed[row : row + window, col : col + window]))
            candidates.append((min(fraction, 1.0 - fraction), row, col))
    candidates.sort(reverse=True)
    selected: list[tuple[int, int]] = []
    for score, row, col in candidates:
        if score <= 0.05:
            continue
        if all((row - other_row) ** 2 + (col - other_col) ** 2 >= 192**2 for other_row, other_col in selected):
            selected.append((row, col))
        if len(selected) == 2:
            break
    if len(selected) != 2:
        raise RuntimeError("qualified block did not yield two meaningful transition crops")
    transitions = Image.new("RGB", (1872, 1310), (244, 242, 234))
    draw = ImageDraw.Draw(transitions)
    draw.text((16, 12), "04  GROUND-SCALE REAL MASK TRANSITIONS", fill=(20, 24, 20))
    draw.text((16, 34), "two 48 x 48 m southeast crops; C1 must taper without a ridge or abrupt ownership cutoff", fill=(70, 72, 66))
    for row_index, (row, col) in enumerate(selected):
        region = np.s_[row : row + window, col : col + window]
        transition_panels = [
            ("C0", shade0[region]),
            ("C1", shade1[region]),
            ("ELIGIBLE / HARD", hard_rgb[region]),
        ]
        top = 82 + row_index * 610
        for col_index, (label, panel) in enumerate(transition_panels):
            left = 16 + col_index * 616
            image = Image.fromarray(panel).resize((600, 600), Image.Resampling.NEAREST if col_index == 2 else Image.Resampling.BILINEAR)
            transitions.paste(image, (left, top))
            draw.text((left, top - 18), f"{label}  crop {row_index + 1}", fill=(20, 24, 20))
    transitions.save(paths[3], compress_level=9)
    return paths


def _tranche_qa(
    qa: Path,
    *,
    bbox: tuple[int, int, int, int],
    residual: np.ndarray,
    c1: np.ndarray,
    residual_half: np.ndarray,
    c0_half: np.ndarray,
    c1_half: np.ndarray,
    allowed_half: np.ndarray,
    ownership_1m: np.ndarray,
) -> tuple[list[Path], list[dict[str, Any]]]:
    paths = [qa / "01_full_domain_eligibility_and_ownership.png"]
    palette = np.asarray(
        [[45, 123, 113], [217, 139, 71], [103, 87, 145]], dtype=np.uint8
    )
    allowed_rgb = np.empty((*allowed_half.shape, 3), dtype=np.uint8)
    allowed_rgb[allowed_half] = (54, 135, 72)
    allowed_rgb[~allowed_half] = (190, 67, 48)
    ownership_rgb = palette[np.clip(ownership_1m, 0, 2)]
    eligibility_image = _resize(allowed_rgb, 760)
    ownership_image = _resize(ownership_rgb, 760)
    overview = Image.new("RGB", (1568, 828), (244, 242, 234))
    draw = ImageDraw.Draw(overview)
    draw.text((16, 12), "01  COMPLETE LOD0 TRANCHE ELIGIBILITY + OWNERSHIP", fill=(20, 24, 20))
    draw.text(
        (16, 34),
        "2048 x 2048 m single-domain solve; green = eligible, red = hard zero; ownership is K11/K32/K36",
        fill=(70, 72, 66),
    )
    overview.paste(eligibility_image, (16, 62))
    overview.paste(ownership_image, (792, 62))
    draw.text((16, 44), "ELIGIBILITY / HARD EXCLUSION", fill=(20, 24, 20))
    draw.text((792, 44), "WORLD-LOCKED SOURCE OWNERSHIP", fill=(20, 24, 20))
    for boundary in (190, 380, 570):
        draw.line((16 + boundary, 62, 16 + boundary, 822), fill=(35, 35, 35), width=1)
        draw.line((792 + boundary, 62, 792 + boundary, 822), fill=(35, 35, 35), width=1)
        draw.line((16, 62 + boundary, 776, 62 + boundary), fill=(35, 35, 35), width=1)
        draw.line((792, 62 + boundary, 1552, 62 + boundary), fill=(35, 35, 35), width=1)
    overview.save(paths[0], compress_level=9)

    window = 64  # 32 m at the 0.5 m QA lattice; 512 x 512 native samples.
    candidates: list[dict[str, float | int]] = []
    for row in range(0, allowed_half.shape[0] - window + 1, window):
        for col in range(0, allowed_half.shape[1] - window + 1, window):
            region = np.s_[row : row + window, col : col + window]
            eligible = float(np.mean(allowed_half[region]))
            if eligible <= 0.05:
                continue
            relief = float(np.std(c0_half[region], dtype=np.float64))
            added = float(np.std(residual_half[region], dtype=np.float64))
            candidates.append(
                {
                    "row": row,
                    "col": col,
                    "eligible": eligible,
                    "transition": min(eligible, 1.0 - eligible),
                    "relief": relief,
                    "added": added,
                }
            )
    if not candidates:
        raise RuntimeError("LOD0 tranche yielded no eligible QA windows")

    selected: list[tuple[str, dict[str, float | int]]] = []

    def choose(label: str, key, predicate=lambda item: True) -> None:
        pool = [item for item in candidates if predicate(item)]
        pool.sort(key=key, reverse=True)
        for item in pool:
            row, col = int(item["row"]), int(item["col"])
            if all(
                (row - int(other["row"])) ** 2 + (col - int(other["col"])) ** 2
                >= 256**2
                for _other_label, other in selected
            ):
                selected.append((label, item))
                return
        raise RuntimeError(f"LOD0 tranche could not select a distinct {label} QA window")

    choose("dense eligible floor", lambda item: (item["eligible"], item["added"]))
    choose(
        "strong eligibility transition",
        lambda item: (item["transition"], item["added"]),
        lambda item: item["transition"] > 0.05,
    )
    choose(
        "higher base-relief eligible floor",
        lambda item: (item["relief"], item["eligible"]),
        lambda item: item["eligible"] >= 0.25,
    )
    choose(
        "strong added morphology",
        lambda item: (item["added"], item["eligible"]),
        lambda item: item["eligible"] >= 0.25,
    )

    selections: list[dict[str, Any]] = []
    scale = 8
    native_window = window * scale
    for index, (label, item) in enumerate(selected, start=2):
        row, col = int(item["row"]), int(item["col"])
        fine_row, fine_col = row * scale, col * scale
        fine_region = np.s_[
            fine_row : fine_row + native_window,
            fine_col : fine_col + native_window,
        ]
        native_c1 = np.asarray(c1[fine_region], dtype=np.float32)
        native_c0 = native_c1 - np.asarray(residual[fine_region], dtype=np.float32)
        shade0 = np.repeat(_hillshade(native_c0, TEXEL_M)[..., None], 3, axis=2)
        shade1 = np.repeat(_hillshade(native_c1, TEXEL_M)[..., None], 3, axis=2)
        image_path = qa / f"{index:02d}_native_{label.replace(' ', '_')}.png"
        paths.append(image_path)
        canvas = Image.new("RGB", (1072, 590), (244, 242, 234))
        draw = ImageDraw.Draw(canvas)
        east = bbox[0] + fine_col * TEXEL_M
        north = bbox[3] - fine_row * TEXEL_M
        draw.text((16, 12), f"{index:02d}  {label.upper()}", fill=(20, 24, 20))
        draw.text(
            (16, 34),
            f"32 x 32 m native 6.25 cm samples; same light; NW corner E={east:.2f} N={north:.2f}",
            fill=(70, 72, 66),
        )
        canvas.paste(Image.fromarray(shade0), (16, 62))
        canvas.paste(Image.fromarray(shade1), (544, 62))
        draw.text((16, 44), "PINNED C0", fill=(20, 24, 20))
        draw.text((544, 44), "FOREST C1", fill=(20, 24, 20))
        canvas.save(image_path, compress_level=9)
        selections.append(
            {
                "image": f"qa/{image_path.name}",
                "condition": label,
                "northwest_en": [east, north],
                "eligible_fraction": item["eligible"],
                "base_relief_std_m": item["relief"],
                "added_relief_std_m": item["added"],
            }
        )
    return paths, selections


def _axis_seam_metrics(
    residual: np.ndarray,
    allowed: np.ndarray,
    *,
    axis: int,
    seam: int | None = None,
) -> dict[str, Any]:
    if axis == 1:
        residual = residual.T
        allowed = allowed.T
    seam = residual.shape[0] // 2 if seam is None else seam
    seam_valid = allowed[seam - 1] & allowed[seam]
    seam_steps = np.abs(residual[seam] - residual[seam - 1])[seam_valid]
    neighborhood = np.abs(np.diff(residual[seam - 64 : seam + 65], axis=0))
    neighborhood_valid = allowed[seam - 64 : seam + 64] & allowed[seam - 63 : seam + 65]
    neighborhood_valid[63] = False
    local_steps = neighborhood[neighborhood_valid]
    return {
        "eligible_sample_count": int(seam_steps.size),
        "residual_step_mean_m": float(np.mean(seam_steps)) if seam_steps.size else None,
        "residual_step_p95_m": float(np.percentile(seam_steps, 95)) if seam_steps.size else None,
        "local_nonseam_step_mean_m": float(np.mean(local_steps)) if local_steps.size else None,
        "local_nonseam_step_p95_m": float(np.percentile(local_steps, 95)) if local_steps.size else None,
    }


def _identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(ASSET_GEN_ROOT.parent)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _allowed_residual_metrics(
    residual: np.ndarray, allowed: np.ndarray
) -> tuple[float, float, list[float], int]:
    count = 0
    sum_squares = 0.0
    maximum = 0.0
    sample_stride = 128
    samples: list[np.ndarray] = []
    for row0 in range(0, residual.shape[0], 128):
        row1 = min(residual.shape[0], row0 + 128)
        values = np.asarray(residual[row0:row1], dtype=np.float32)
        mask = np.asarray(allowed[row0:row1], dtype=bool)
        selected = values[mask]
        count += int(selected.size)
        sum_squares += float(np.sum(selected * selected, dtype=np.float64))
        maximum = max(maximum, float(np.max(np.abs(selected), initial=0.0)))
        if row0 % sample_stride == 0:
            sampled_values = values[0, ::sample_stride]
            sampled_mask = mask[0, ::sample_stride]
            samples.append(sampled_values[sampled_mask])
    if count == 0:
        raise RuntimeError("LOD0 tranche contains no eligible residual samples")
    sample = np.concatenate(samples)
    return (
        float(np.sqrt(sum_squares / count)),
        maximum,
        [float(value) for value in np.percentile(sample, [1, 99])],
        sample_stride,
    )


def run(config_path: Path) -> Path:
    config = _read_config(config_path)
    is_block = _is_block(config)
    is_tranche = _is_tranche(config)
    bbox = tuple(map(int, config["bbox_en"]))
    shape = _shape(bbox)
    recipe = {
        "schema": (
            "forest-mesic-mineral-lod0-tranche-recipe/1"
            if is_tranche
            else (
                "forest-mesic-mineral-adjacent-block-recipe/1"
                if is_block
                else "forest-mesic-mineral-adjacent-continuity-recipe/1"
            )
        ),
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
    output_root = (
        TRANCHE_OUTPUT_ROOT
        if is_tranche
        else (BLOCK_OUTPUT_ROOT if is_block else PAIR_OUTPUT_ROOT)
    )
    destination = output_root / build_id
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
        print("[forest-tranche] synthesis 0%: assembling world-locked residual", flush=True)
        (
            residual_path,
            ownership_path,
            dominant_path,
            placements,
            source_counts,
            crossing_sites,
        ) = _assemble(
            candidates=candidates,
            bbox=bbox,
            scratch=scratch,
            config=config,
        )
        print("[forest-tranche] synthesis 25%: residual assembled; rasterizing masks", flush=True)
        allowed, mask_evidence = _fine_mask(bbox, scratch / "allowed.u8")
        print("[forest-tranche] synthesis 40%: masks complete; projecting one-metre closure", flush=True)
        maximum_mean = _project(residual_path, allowed, shape)
        print("[forest-tranche] synthesis 60%: projection complete; writing absolute C1", flush=True)
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
        print("[forest-tranche] synthesis 75%: C1 complete; reducing diagnostics", flush=True)
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
        output_rms, output_maximum, output_percentiles, percentile_stride = (
            _allowed_residual_metrics(residual, allowed)
        )
        seam_horizontal = _axis_seam_metrics(residual_half, allowed_half, axis=0)
        seam_horizontal["cross_seam_site_count"] = crossing_sites["horizontal"]
        tranche_selections: list[dict[str, Any]] = []
        if is_tranche:
            parent_allowed = {}
            fine_parent_cells = round(PARENT_M / TEXEL_M)
            for index, parent in enumerate(config["parents"]):
                grid_row, grid_col = divmod(index, 4)
                region = np.s_[
                    grid_row * fine_parent_cells : (grid_row + 1) * fine_parent_cells,
                    grid_col * fine_parent_cells : (grid_col + 1) * fine_parent_cells,
                ]
                parent_allowed[parent["role"]] = float(np.mean(allowed[region]))
            qa_parent_cells = round(PARENT_M / 0.5)
            seam = {
                "horizontal": [
                    _axis_seam_metrics(
                        residual_half, allowed_half, axis=0, seam=boundary
                    )
                    for boundary in range(
                        qa_parent_cells, residual_half.shape[0], qa_parent_cells
                    )
                ],
                "vertical": [
                    _axis_seam_metrics(
                        residual_half, allowed_half, axis=1, seam=boundary
                    )
                    for boundary in range(
                        qa_parent_cells, residual_half.shape[1], qa_parent_cells
                    )
                ],
                "cross_seam_site_counts": {
                    "horizontal": crossing_sites["horizontal"],
                    "vertical": crossing_sites["vertical"],
                },
                "cross_junction_site_count": crossing_sites["junction"],
            }
            pngs, tranche_selections = _tranche_qa(
                qa,
                bbox=bbox,
                residual=residual,
                c1=c1,
                residual_half=residual_half,
                c0_half=c0_half,
                c1_half=c1_half,
                allowed_half=allowed_half,
                ownership_1m=ownership_1m,
            )
            print("[forest-tranche] synthesis 95%: QA complete; binding artifact", flush=True)
        elif is_block:
            half_rows, half_cols = shape[0] // 2, shape[1] // 2
            parent_allowed = {
                config["parents"][0]["role"]: float(np.mean(allowed[:half_rows, :half_cols])),
                config["parents"][1]["role"]: float(np.mean(allowed[:half_rows, half_cols:])),
                config["parents"][2]["role"]: float(np.mean(allowed[half_rows:, :half_cols])),
                config["parents"][3]["role"]: float(np.mean(allowed[half_rows:, half_cols:])),
            }
            seam_vertical = _axis_seam_metrics(residual_half, allowed_half, axis=1)
            seam_vertical["cross_seam_site_count"] = crossing_sites["vertical"]
            seam = {
                "horizontal": seam_horizontal,
                "vertical": seam_vertical,
                "cross_junction_site_count": crossing_sites["junction"],
            }
            pngs = _block_qa(
                qa,
                residual_half,
                ownership_1m,
                c0_half,
                c1_half,
                allowed_half,
            )
        else:
            parent_allowed = [
                float(np.mean(allowed[: shape[0] // 2])),
                float(np.mean(allowed[shape[0] // 2 :])),
            ]
            seam = seam_horizontal
            _primary_qa(
                qa / "01_contiguous_master_and_shared_seam.png",
                c0_half,
                c1_half,
                c1_half.shape[0] // 2,
            )
            pngs = [qa / "01_contiguous_master_and_shared_seam.png"]
            pngs.extend(
                _secondary_qa(
                    qa,
                    residual_half,
                    ownership_1m,
                    c0_half,
                    c1_half,
                    allowed_half,
                    c1_half.shape[0] // 2,
                )
            )
        acceptance = config["acceptance"]
        failures = []
        if is_tranche:
            domain_allowed = float(np.mean(allowed_half))
            if domain_allowed < acceptance["minimum_domain_allowed_fraction"]:
                failures.append("tranche eligibility")
            fractions = list(parent_allowed.values())
            if max(fractions) - min(fractions) < acceptance["minimum_parent_allowed_fraction_range"]:
                failures.append("tranche lacks a meaningful eligibility transition")
        elif is_block:
            for role, minimum in acceptance["minimum_parent_allowed_fraction_by_role"].items():
                if parent_allowed[role] < minimum:
                    failures.append(f"parent eligibility: {role}")
            fractions = list(parent_allowed.values())
            if max(fractions) - min(fractions) < acceptance["minimum_parent_allowed_fraction_range"]:
                failures.append("block lacks a meaningful eligibility transition")
        elif min(parent_allowed) < acceptance["minimum_parent_allowed_fraction"]:
            failures.append("parent eligibility")
        if maximum_mean > acceptance["maximum_one_metre_mean_error_m"]:
            failures.append("one-metre mean closure")
        if maximum_hard > acceptance["maximum_hard_exclusion_residual_m"]:
            failures.append("hard exclusion")
        if output_maximum / capacity["maximum_abs_m"] > acceptance["maximum_output_over_candidate_abs_ratio"]:
            failures.append("measured maximum envelope")
        if output_rms / capacity["rms_m"] > acceptance["maximum_output_over_candidate_rms_ratio"]:
            failures.append("measured RMS envelope")
        if is_block or is_tranche:
            minimum_axis = acceptance["minimum_cross_seam_site_count_per_axis"]
            if crossing_sites["horizontal"] < minimum_axis:
                failures.append("no ownership support crosses horizontal parent seam")
            if crossing_sites["vertical"] < minimum_axis:
                failures.append("no ownership support crosses vertical parent seam")
            if crossing_sites["junction"] < acceptance["minimum_cross_junction_site_count"]:
                failures.append("no ownership support crosses four-parent junction")
        elif crossing_sites["horizontal"] < acceptance["minimum_cross_seam_site_count"]:
            failures.append("no ownership support crosses parent seam")
        metrics = {
            "bbox_en": list(bbox),
            "shape": list(shape),
            (
                "parent_allowed_fraction_by_role"
                if is_block or is_tranche
                else "parent_allowed_fraction_north_south"
            ): parent_allowed,
            "mask": mask_evidence,
            "site_count": len({item["site_index"] for item in placements}),
            "placement_count": len(placements),
            "source_placement_counts": source_counts,
            "seam": seam,
            "candidate_capacity": capacity,
            "residual_rms_allowed_m": output_rms,
            "residual_sample_p01_p99_allowed_m": output_percentiles,
            "residual_percentile_sample_stride_cells": percentile_stride,
            "maximum_abs_residual_m": output_maximum,
            "maximum_abs_unprojected_residual_m": maximum_unprojected,
            "output_over_candidate_abs_ratio": output_maximum / capacity["maximum_abs_m"],
            "output_over_candidate_rms_ratio": output_rms / capacity["rms_m"],
            "maximum_one_metre_mean_error_m": maximum_mean,
            "maximum_hard_exclusion_residual_m": maximum_hard,
            "authority_p01_p99_m": [float(v) for v in np.percentile(authority, [1, 99])],
        }
        if is_tranche:
            metrics["qa_selections"] = tranche_selections
        (staging / "metrics.json").write_bytes(_canonical_json(metrics) + b"\n")
        del c1, residual, allowed, ownership
        shutil.rmtree(scratch)
        qa_index = {
            "schema": (
                "forest-mesic-mineral-lod0-tranche-qa/1"
                if is_tranche
                else (
                    "forest-mesic-mineral-adjacent-block-qa/1"
                    if is_block
                    else "forest-mesic-mineral-adjacent-continuity-qa/1"
                )
            ),
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
            "interpretation": (
                [
                    "01 binds complete-tranche eligibility and hard exclusions to the world-locked K11/K32/K36 ownership field; thin lines mark the 4x4 storage-parent grid only.",
                    "02-05 compare four spatially distinct 32 m C0/C1 windows using native 6.25 cm samples and identical illumination.",
                    "The four windows deliberately cover dense eligibility, a real eligibility transition, higher base relief, and strong added morphology.",
                ]
                if is_tranche
                else (
                    [
                        "01 compares the complete 2x2 C0/C1 master under identical common light; the red cross marks storage-parent boundaries only.",
                        "02 inspects C0, C1, signed residual, and source ownership at the common four-parent junction.",
                        "03 binds the eligible and hard-zero synthesis regions to their location on the C1 terrain.",
                        "04 inspects two real southeast eligibility transitions at ground scale for ridges, abrupt cutoffs, or ownership resets.",
                    ]
                    if is_block
                    else [
                        "01 is primary: both complete parents and a seam-centered strip use identical common light; red is only the storage-parent boundary.",
                        "02 shows one signed residual and ownership field solved over the full contiguous domain before any parent crop.",
                        "03 is a 32 m ground-scale before/after crop centered exactly on the shared boundary.",
                        "04 compares the complete new and previously accepted parent at identical scale and light.",
                    ]
                )
            ),
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
            "schema": (
                "forest-mesic-mineral-lod0-tranche-artifact/1"
                if is_tranche
                else (
                    "forest-mesic-mineral-adjacent-block-artifact/1"
                    if is_block
                    else "forest-mesic-mineral-adjacent-continuity-artifact/1"
                )
            ),
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
                (
                    "one complete 2048 m LOD0 tranche extends morphology and transition evidence only; it does not authorize national coverage"
                    if is_tranche
                    else (
                        "one contiguous 2x2 block extends junction and mask-transition evidence only; it does not authorize wide coverage"
                        if is_block
                        else "one contiguous pair extends continuity evidence only; it does not authorize wide coverage"
                    )
                ),
                "the accepted forest morphology and transport parameters were not retuned",
            ],
            "files": files,
        }
        (staging / "manifest.json").write_bytes(_canonical_json(manifest) + b"\n")
        os.replace(staging, destination)
    print("[forest-tranche] synthesis 100%: immutable float artifact complete", flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    print(run(parser.parse_args().config))


if __name__ == "__main__":
    main()
