"""Condition-owned, multi-site float pilot for the irregular forest specialist."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import rasterio
import shapely
from PIL import Image, ImageDraw
from scipy.interpolate import RectBivariateSpline

from assetgen.config import ASSET_GEN_ROOT, DATA_IN, load_base
from assetgen.cook.pinned_height import PinnedBaseHeight
from assetgen.process.etak_read import read_layer_window
from assetgen.process.micro_fixture import conservative_cell_correct
from assetgen.process.micro_masks import rasterize_micro_morphology_mask
from assetgen.process.microtopo.projection import build_smooth_mean_null_projector
from assetgen.process.mosaic import RasterStack, dem_sources
from assetgen.release import read_v1_index

from .irregular import (
    SOURCE_IDS,
    TEXEL_M,
    _boundary_metrics,
    _select_site_patch,
    _wendland_weight,
    _world_locked_sites,
)
from .run import (
    SOURCE_SHA256,
    Candidate,
    _candidates,
    _canonical_json,
    _complete_for_model,
    _load_surface,
    _panel,
    _percentile_rgb,
    _sha256,
)


PARENT_M = 512
FACTOR = 16
FINE_CELLS = PARENT_M * FACTOR
BASE_MANIFEST_SHA256 = "708478a57c2118eaa618867e87cc74a35e20c615999b60d7e4177b595ef5495a"
BASE_MANIFEST = ASSET_GEN_ROOT / "data/out/m/708478a57c2118ea/manifest.json"
CONFIG = ASSET_GEN_ROOT / "config/microtopography/forest-generalization-v1.json"
OUTPUT_ROOT = ASSET_GEN_ROOT / "data/work/microtopography/estonia-forest-generalization"
GEOLOGY_SOURCE = (
    ASSET_GEN_ROOT.parent
    / "docs/deep-research/microtopography-generation/library/data/egt/pinnakate-200k/q_avamus_a_200t.shp"
)


@dataclass(frozen=True)
class ParentSite:
    region_id: str
    topographic_role: str
    cx: int
    cz: int
    bbox_en: tuple[int, int, int, int]
    allowed_fraction: float
    p90_slope: float
    soil_signature: tuple[str, str, str]
    geology_signature: tuple[int, int]


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _identity_from_sidecar(path: Path) -> dict[str, Any]:
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"source identity sidecar is absent: {sidecar}")
    return {
        "path": str(path.relative_to(ASSET_GEN_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sidecar.read_text(encoding="ascii").strip().split()[0],
    }


def _parent_bbox(cx: int, cz: int) -> tuple[int, int, int, int]:
    e_min = 368640 + cx * PARENT_M
    n_max = 6635520 - cz * PARENT_M
    return e_min, n_max - PARENT_M, e_min + PARENT_M, n_max


def _aligned_parents(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    e_min, n_min, e_max, n_max = bbox
    cx0 = -((368640 - e_min) // PARENT_M)
    cx1 = (e_max - 368640) // PARENT_M - 1
    cz0 = -((n_max - 6635520) // PARENT_M)
    cz1 = (6635520 - n_min) // PARENT_M - 1
    result = []
    for cz in range(cz0, cz1 + 1):
        for cx in range(cx0, cx1 + 1):
            parent = _parent_bbox(cx, cz)
            if parent[0] >= e_min and parent[1] >= n_min and parent[2] <= e_max and parent[3] <= n_max:
                result.append((cx, cz))
    return result


def _dominant_attributes(
    source: Path,
    layer: str,
    bbox: tuple[int, int, int, int],
    fields: list[str],
) -> tuple[Any, ...]:
    geometries, columns = read_layer_window(source, layer, bbox, fields=fields)
    query = shapely.box(*bbox)
    ranked: list[tuple[float, int]] = []
    for index, geometry in enumerate(geometries):
        if geometry is None or geometry.is_empty:
            continue
        ranked.append((float(geometry.intersection(query).area), index))
    if not ranked:
        return tuple(None for _ in fields)
    index = max(ranked, key=lambda item: (item[0], -item[1]))[1]
    return tuple(columns[field][index].item() if isinstance(columns[field][index], np.generic) else columns[field][index] for field in fields)


def _condition_signatures(bbox: tuple[int, int, int, int]) -> tuple[tuple[str, str, str], tuple[int, int]]:
    soil = _dominant_attributes(
        DATA_IN / "soil/mullakaart/Mullakaart.shp",
        "Mullakaart",
        bbox,
        ["Sif1", "Lihtloimis", "Boniteet"],
    )
    geology = _dominant_attributes(
        GEOLOGY_SOURCE,
        "q_avamus_a_200t",
        bbox,
        ["lito200", "genees200"],
    )
    return tuple("" if value is None else str(value) for value in soil), tuple(-1 if value is None else int(value) for value in geology)


def _region_candidates(region: dict[str, Any], selection_texel: float) -> list[ParentSite]:
    bbox = tuple(int(value) for value in region["bbox_en"])
    e_min, n_min, e_max, n_max = bbox
    east = e_min + (np.arange(round((e_max - e_min) / selection_texel)) + 0.5) * selection_texel
    north = n_max - (np.arange(round((n_max - n_min) / selection_texel)) + 0.5) * selection_texel
    mask = rasterize_micro_morphology_mask(east, north)
    height = RasterStack(dem_sources(DATA_IN)).read_window(*bbox, selection_texel)
    if not np.isfinite(height).all():
        raise RuntimeError(f"retained DTM region is incomplete: {region['id']}")
    dy, dx = np.gradient(height.astype(np.float64), selection_texel)
    slope = np.hypot(dx, dy)
    candidates: list[ParentSite] = []
    side = round(PARENT_M / selection_texel)
    for cx, cz in _aligned_parents(bbox):
        parent = _parent_bbox(cx, cz)
        x0 = round((parent[0] - e_min) / selection_texel)
        y0 = round((n_max - parent[3]) / selection_texel)
        allowed = mask.allowed[y0 : y0 + side, x0 : x0 + side]
        allowed_fraction = float(allowed.mean())
        if not allowed.any():
            p90_slope = float("inf")
        else:
            p90_slope = float(np.percentile(slope[y0 : y0 + side, x0 : x0 + side][allowed], 90))
        candidates.append(
            ParentSite(
                region_id=region["id"],
                topographic_role=region["topographic_role"],
                cx=cx,
                cz=cz,
                bbox_en=parent,
                allowed_fraction=allowed_fraction,
                p90_slope=p90_slope,
                soil_signature=("", "", ""),
                geology_signature=(-1, -1),
            )
        )
    return candidates


def _rank(site: ParentSite) -> tuple[Any, ...]:
    if site.topographic_role == "level":
        return site.p90_slope, -site.allowed_fraction, site.cz, site.cx
    if site.topographic_role == "rolling":
        return -site.p90_slope, -site.allowed_fraction, site.cz, site.cx
    if site.topographic_role == "forest_dense":
        return -site.allowed_fraction, site.p90_slope, site.cz, site.cx
    raise ValueError(f"unknown topographic role: {site.topographic_role}")


def _selection_sources() -> dict[str, Any]:
    dem = [_identity_from_sidecar(path) for path in sorted((DATA_IN / "dem_1m").glob("*_dtm_1m.tif"))]
    return {
        "config": {"path": str(CONFIG.relative_to(ASSET_GEN_ROOT)), "sha256": _sha256(CONFIG)},
        "dem_1m": dem,
        "etak_archive_sha256": (DATA_IN / "etak/ETAK_EESTI_GPKG.zip.sha256").read_text(encoding="ascii").strip(),
        "soil_archive_sha256": (DATA_IN / "soil/Mullakaart_SHP.zip.sha256").read_text(encoding="ascii").strip(),
        "country_dtm": _identity_from_sidecar(DATA_IN / "country/DTM_10m_eesti.tif"),
        "canonical_base_manifest": {
            "path": str(BASE_MANIFEST.relative_to(ASSET_GEN_ROOT)),
            "sha256": BASE_MANIFEST_SHA256,
        },
        "geology_source_bundle": [
            {"path": str(GEOLOGY_SOURCE.with_suffix(suffix).relative_to(ASSET_GEN_ROOT.parent)), "bytes": GEOLOGY_SOURCE.with_suffix(suffix).stat().st_size}
            for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg")
        ],
    }


def _base_lod0_keys() -> set[tuple[int, int]]:
    manifest = _json(BASE_MANIFEST)
    index = BASE_MANIFEST.parent / manifest["layers"]["height"]["index"]
    return {(record.cx, record.cz) for record in read_v1_index(index) if record.lod == 0}


def _base_has_halo(site: ParentSite, keys: set[tuple[int, int]]) -> bool:
    grid = load_base().grid
    e_min, n_min, e_max, n_max = site.bbox_en
    e_min -= 4
    n_min -= 4
    e_max += 4
    n_max += 4
    cx0 = (e_min - grid.anchor_e) // grid.chunk_m
    cx1 = (e_max - 1 - grid.anchor_e) // grid.chunk_m
    cz0 = (grid.anchor_n - n_max) // grid.chunk_m
    cz1 = (grid.anchor_n - n_min - 1) // grid.chunk_m
    return all((cx, cz) in keys for cz in range(cz0, cz1 + 1) for cx in range(cx0, cx1 + 1))


def _atlas_image(config: dict[str, Any], selected: list[ParentSite], destination: Path) -> dict[str, Any]:
    with rasterio.open(DATA_IN / "country/DTM_10m_eesti.tif") as source:
        bounds = tuple(float(value) for value in source.bounds)
    texel = float(config["national_atlas_texel_m"])
    cols = round((bounds[2] - bounds[0]) / texel)
    rows = round((bounds[3] - bounds[1]) / texel)
    rgb = np.full((rows, cols, 3), (225, 220, 202), dtype=np.uint8)
    allowed_cells = 0
    forest_cells = 0
    for region in config["regions"]:
        e_min, n_min, e_max, n_max = region["bbox_en"]
        local_cols = round((e_max - e_min) / texel)
        local_rows = round((n_max - n_min) / texel)
        east = e_min + (np.arange(local_cols) + 0.5) * texel
        north = n_max - (np.arange(local_rows) + 0.5) * texel
        mask = rasterize_micro_morphology_mask(east, north)
        col0 = round((e_min - bounds[0]) / texel)
        row0 = round((bounds[3] - n_max) / texel)
        local = rgb[row0 : row0 + local_rows, col0 : col0 + local_cols]
        local[mask.forest] = (139, 157, 105)
        local[mask.allowed] = (36, 117, 74)
        allowed_cells += int(np.count_nonzero(mask.allowed))
        forest_cells += int(np.count_nonzero(mask.forest))
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    for region in config["regions"]:
        box = tuple(region["bbox_en"])
        xy = (
            round((box[0] - bounds[0]) / texel),
            round((bounds[3] - box[3]) / texel),
            round((box[2] - bounds[0]) / texel),
            round((bounds[3] - box[1]) / texel),
        )
        draw.rectangle(xy, outline=(246, 219, 89), width=3)
    for index, site in enumerate(selected, 1):
        box = site.bbox_en
        xy = (
            round((box[0] - bounds[0]) / texel),
            round((bounds[3] - box[3]) / texel),
            round((box[2] - bounds[0]) / texel),
            round((bounds[3] - box[1]) / texel),
        )
        draw.rectangle(xy, outline=(214, 62, 49), width=5)
        draw.text((xy[0] + 4, xy[1] + 4), str(index), fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    image.save(destination, compress_level=9)
    return {
        "bbox_en": list(bounds),
        "texel_m": texel,
        "shape": [rows, cols],
        "allowed_cells_in_retained_regions": allowed_cells,
        "forest_cells_in_retained_regions": forest_cells,
        "interpretation": "national context with eligibility evaluated only inside retained input regions; exact generation masks are rasterized at 0.0625 m",
    }


def select_sites() -> Path:
    config = _json(CONFIG)
    minimum = float(config["minimum_allowed_fraction"])
    base_lod0_keys = _base_lod0_keys()
    selected: list[ParentSite] = []
    used_signatures: set[tuple[tuple[str, str, str], tuple[int, int]]] = set()
    region_evidence: dict[str, Any] = {}
    for region in config["regions"]:
        candidates = _region_candidates(region, float(config["selection_texel_m"]))
        eligible = sorted(
            (
                site
                for site in candidates
                if site.allowed_fraction >= minimum and _base_has_halo(site, base_lod0_keys)
            ),
            key=_rank,
        )
        if not eligible:
            raise RuntimeError(f"no eligible full parent in {region['id']}")
        choice = None
        signature_evidence: list[dict[str, Any]] = []
        for candidate in eligible:
            soil, geology = _condition_signatures(candidate.bbox_en)
            conditioned = replace(candidate, soil_signature=soil, geology_signature=geology)
            signature_evidence.append(asdict(conditioned))
            if (soil, geology) not in used_signatures:
                choice = conditioned
                break
        if choice is None:
            raise RuntimeError(f"no distinct condition signature in {region['id']}")
        selected.append(choice)
        used_signatures.add((choice.soil_signature, choice.geology_signature))
        region_evidence[region["id"]] = {
            "candidate_count": len(candidates),
            "eligible_count": len(eligible),
            "selection_rule": region["topographic_role"],
            "requires_canonical_lod0_halo": True,
            "top_five_ranked_before_condition_query": [asdict(site) for site in eligible[:5]],
            "condition_queries_until_selection": signature_evidence,
        }
    if len({site.bbox_en for site in selected}) != 3 or len(used_signatures) != 3:
        raise RuntimeError("selection did not produce three distinct condition signatures")
    recipe = {
        "schema": "forest-mesic-mineral-site-selection/1",
        "config": config,
        "sources": _selection_sources(),
        "implementation_sha256": _sha256(Path(__file__)),
        "selected": [asdict(site) for site in selected],
        "region_evidence": region_evidence,
    }
    build_id = hashlib.sha256(_canonical_json(recipe)).hexdigest()
    destination = OUTPUT_ROOT / "selection/sha256" / build_id
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="forest-site-selection-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        (staging / "qa").mkdir()
        (staging / "selection.json").write_bytes(_canonical_json(recipe) + b"\n")
        atlas = _atlas_image(config, selected, staging / "qa/01_national_eligibility_and_sites.png")
        manifest = {
            "schema": "forest-mesic-mineral-site-selection-artifact/1",
            "build_id": build_id,
            "status": "selection_frozen_before_synthesis",
            "atlas": atlas,
            "files": {},
        }
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                manifest["files"][str(path.relative_to(staging))] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        (staging / "manifest.json").write_bytes(_canonical_json(manifest) + b"\n")
        os.replace(staging, destination)
    return destination


def _load_candidates() -> list[Candidate]:
    source_root = DATA_IN / "microtopo-exemplars/lapinjarvi-2016/derived-v1"
    surfaces: list[tuple[np.ndarray, np.ndarray]] = []
    for source_id in SOURCE_IDS:
        path = source_root / f"{source_id}-ground.npz"
        if _sha256(path) != SOURCE_SHA256[source_id]:
            raise RuntimeError(f"source hash differs: {path}")
        z, _measured, support = _load_surface(path)
        surfaces.append((_complete_for_model(z, support), support))
    return _candidates(surfaces, 192, 16)


def _memmap(path: Path, dtype, shape: tuple[int, int], *, fill=0):
    array = np.memmap(path, dtype=dtype, mode="w+", shape=shape)
    array[:] = fill
    array.flush()
    return array


def _assemble_disk(
    candidates: list[Candidate],
    bbox: tuple[int, int, int, int],
    scratch: Path,
    config: dict[str, Any],
) -> tuple[Path, Path, Path, list[dict[str, Any]], dict[str, int]]:
    transport = config["transport"]
    sites = _world_locked_sites(
        master_cells=FINE_CELLS,
        master_bbox=bbox,
        candidate_cell=int(transport["site_candidate_cell"]),
        minimum_distance=int(transport["site_minimum_distance_cells"]),
        saturation_distance=int(transport["site_saturation_distance_cells"]),
        saturation_rounds=int(transport["site_saturation_rounds"]),
        support_radius=int(transport["support_radius_cells"]),
    )
    values = np.stack([candidate.values for candidate in candidates]).astype(np.float32)
    source_index = np.asarray([candidate.source_index for candidate in candidates], dtype=np.int16)
    numerator = _memmap(scratch / "numerator.f64", np.float64, (FINE_CELLS, FINE_CELLS))
    denominator = _memmap(scratch / "denominator.f64", np.float64, (FINE_CELLS, FINE_CELLS))
    dominant_weight = _memmap(scratch / "dominant-weight.f32", np.float32, (FINE_CELLS, FINE_CELLS))
    dominant_site = _memmap(scratch / "dominant-site.i32", np.int32, (FINE_CELLS, FINE_CELLS), fill=-1)
    ownership = _memmap(scratch / "ownership.u8", np.uint8, (FINE_CELLS, FINE_CELLS))
    candidate_use = np.zeros(len(candidates), dtype=np.int32)
    source_use = np.zeros(len(SOURCE_IDS), dtype=np.int32)
    placements: list[dict[str, Any]] = []
    patch_cells = int(transport["patch_cells"])
    support_radius = int(transport["support_radius_cells"])
    half = patch_cells // 2
    e_min, _n_min, _e_max, n_max = bbox
    master_col0 = int(round((e_min - 368640.0) / TEXEL_M))
    master_row0 = int(round((6635520.0 - n_max) / TEXEL_M))
    for site_index, site in enumerate(sites):
        center_row = site.global_row - master_row0
        center_col = site.global_col - master_col0
        row0 = max(0, center_row - half)
        row1 = min(FINE_CELLS, center_row + half)
        col0 = max(0, center_col - half)
        col1 = min(FINE_CELLS, center_col + half)
        if row0 >= row1 or col0 >= col1:
            continue
        source_row0 = row0 - (center_row - half)
        source_row1 = source_row0 + row1 - row0
        source_col0 = col0 - (center_col - half)
        source_col1 = source_col0 + col1 - col0
        rows, cols = np.meshgrid(np.arange(row0, row1), np.arange(col0, col1), indexing="ij")
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
    denominator.flush()
    if np.any(np.asarray(denominator) <= 1e-12) or np.any(np.asarray(dominant_site) < 0):
        raise RuntimeError("irregular transport left full-parent samples uncovered")
    raw_path = scratch / "residual.f32"
    raw = np.memmap(raw_path, dtype=np.float32, mode="w+", shape=(FINE_CELLS, FINE_CELLS))
    row_step = 128
    total = 0.0
    for row0 in range(0, FINE_CELLS, row_step):
        row1 = min(FINE_CELLS, row0 + row_step)
        total += float(np.sum(numerator[row0:row1] / denominator[row0:row1], dtype=np.float64))
    mean = total / float(FINE_CELLS * FINE_CELLS)
    for row0 in range(0, FINE_CELLS, row_step):
        row1 = min(FINE_CELLS, row0 + row_step)
        raw[row0:row1] = numerator[row0:row1] / denominator[row0:row1] - mean
    raw.flush()
    del numerator, denominator, dominant_weight, raw
    for path in (scratch / "numerator.f64", scratch / "denominator.f64", scratch / "dominant-weight.f32"):
        path.unlink()
    return raw_path, scratch / "ownership.u8", scratch / "dominant-site.i32", placements, {
        source_id: int(source_use[index]) for index, source_id in enumerate(SOURCE_IDS)
    }


def _fine_mask(bbox: tuple[int, int, int, int], path: Path) -> tuple[np.memmap, dict[str, int]]:
    allowed = _memmap(path, np.uint8, (FINE_CELLS, FINE_CELLS))
    evidence: dict[str, int] = {}
    e_min, _n_min, _e_max, n_max = bbox
    tile_cells = 2048
    for tile_row in range(4):
        for tile_col in range(4):
            tile_e = e_min + tile_col * 128
            tile_n = n_max - tile_row * 128
            east = tile_e + (np.arange(tile_cells) + 0.5) * TEXEL_M
            north = tile_n - (np.arange(tile_cells) + 0.5) * TEXEL_M
            mask = rasterize_micro_morphology_mask(east, north)
            region = np.s_[tile_row * tile_cells : (tile_row + 1) * tile_cells, tile_col * tile_cells : (tile_col + 1) * tile_cells]
            allowed[region] = mask.allowed
            for key, value in mask.evidence().items():
                if isinstance(value, int):
                    evidence[key] = evidence.get(key, 0) + value
    allowed.flush()
    return allowed, evidence


def _project_disk(residual_path: Path, allowed: np.memmap) -> float:
    residual = np.memmap(residual_path, dtype=np.float32, mode="r+", shape=(FINE_CELLS, FINE_CELLS))
    fully_soft = np.empty((PARENT_M, PARENT_M), dtype=bool)
    for row in range(PARENT_M):
        fine = np.asarray(allowed[row * FACTOR : (row + 1) * FACTOR], dtype=bool)
        fully_soft[row] = fine.reshape(FACTOR, PARENT_M, FACTOR).all(axis=(0, 2))
    taper_plan = build_smooth_mean_null_projector(np.zeros((PARENT_M, PARENT_M)), fully_soft, factor=FACTOR)
    means = np.empty((PARENT_M, PARENT_M), dtype=np.float64)
    coarse_step = 8
    for row0 in range(0, PARENT_M, coarse_step):
        rows = min(coarse_step, PARENT_M - row0)
        fine = np.asarray(residual[row0 * FACTOR : (row0 + rows) * FACTOR], dtype=np.float64)
        soft = np.asarray(allowed[row0 * FACTOR : (row0 + rows) * FACTOR], dtype=bool)
        taper = taper_plan.taper_window(row0, 0, rows, PARENT_M)
        means[row0 : row0 + rows] = np.where(soft, fine * taper, 0.0).reshape(rows, FACTOR, PARENT_M, FACTOR).mean(axis=(1, 3))
    projector = build_smooth_mean_null_projector(means, fully_soft, factor=FACTOR)
    maximum_mean = 0.0
    for row0 in range(0, PARENT_M, coarse_step):
        rows = min(coarse_step, PARENT_M - row0)
        fine_rows = np.s_[row0 * FACTOR : (row0 + rows) * FACTOR]
        projected = projector.project_window(
            np.asarray(residual[fine_rows], dtype=np.float64),
            np.asarray(allowed[fine_rows], dtype=bool),
            row0=row0,
            col0=0,
        )
        block_means = projected.reshape(rows, FACTOR, PARENT_M, FACTOR).mean(axis=(1, 3))
        maximum_mean = max(maximum_mean, float(np.max(np.abs(block_means))))
        residual[fine_rows] = projected.astype(np.float32)
    residual.flush()
    return maximum_mean


def _refine_and_write_c1(
    *,
    base: PinnedBaseHeight,
    bbox: tuple[int, int, int, int],
    residual_path: Path,
    destination: Path,
) -> tuple[np.memmap, np.ndarray]:
    e_min, _n_min, _e_max, n_max = bbox
    extended = base.read_cells(e_min - 4, n_max + 4, PARENT_M + 8, PARENT_M + 8).astype(np.float64)
    authority = np.asarray(extended[4:-4, 4:-4], dtype=np.float64)
    east_centers = e_min - 4 + np.arange(PARENT_M + 8, dtype=np.float64) + 0.5
    north_centers = n_max + 4 - np.arange(PARENT_M + 8, dtype=np.float64) - 0.5
    spline = RectBivariateSpline(north_centers[::-1], east_centers, extended[::-1], kx=3, ky=3, s=0.0)
    east = e_min + (np.arange(FINE_CELLS, dtype=np.float64) + 0.5) * TEXEL_M
    residual = np.memmap(residual_path, dtype=np.float32, mode="r", shape=(FINE_CELLS, FINE_CELLS))
    c1 = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float32, shape=(FINE_CELLS, FINE_CELLS))
    coarse_step = 8
    for row0 in range(0, PARENT_M, coarse_step):
        rows = min(coarse_step, PARENT_M - row0)
        north = n_max - (np.arange(row0 * FACTOR, (row0 + rows) * FACTOR, dtype=np.float64) + 0.5) * TEXEL_M
        smooth = spline(north[::-1], east, grid=True)[::-1]
        c0 = conservative_cell_correct(smooth, authority[row0 : row0 + rows], FACTOR)
        fine_rows = np.s_[row0 * FACTOR : (row0 + rows) * FACTOR]
        c1[fine_rows] = (c0 + residual[fine_rows]).astype(np.float32)
    c1.flush()
    return c1, authority


def _reduce8(array: np.ndarray) -> np.ndarray:
    result = np.empty((FINE_CELLS // 8, FINE_CELLS // 8), dtype=np.float32)
    for row0 in range(0, result.shape[0], 32):
        rows = min(32, result.shape[0] - row0)
        source = np.asarray(array[row0 * 8 : (row0 + rows) * 8], dtype=np.float32)
        result[row0 : row0 + rows] = source.reshape(rows, 8, result.shape[1], 8).mean(axis=(1, 3))
    return result


def _hillshade(values: np.ndarray, texel_m: float) -> np.ndarray:
    dy, dx = np.gradient(np.asarray(values, dtype=np.float64), texel_m)
    nx, ny, nz = -dx, -dy, np.ones_like(values)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    light = np.asarray([-0.45, -0.55, 0.70])
    return np.asarray(255 * np.clip((nx * light[0] + ny * light[1] + nz * light[2]) / norm, 0.0, 1.0), dtype=np.uint8)


def _pack_mask(allowed: np.ndarray, destination: Path) -> None:
    packed = np.lib.format.open_memmap(destination, mode="w+", dtype=np.uint8, shape=(FINE_CELLS, FINE_CELLS // 8))
    for row0 in range(0, FINE_CELLS, 128):
        packed[row0 : row0 + 128] = np.packbits(np.asarray(allowed[row0 : row0 + 128], dtype=np.uint8), axis=1, bitorder="little")
    packed.flush()


def _choose_closeup(allowed_1m: np.ndarray) -> tuple[int, int]:
    return max(
        (
            int(allowed_1m[y : y + 32, x : x + 32].sum()),
            -y,
            -x,
            y,
            x,
        )
        for y in range(0, PARENT_M - 31, 16)
        for x in range(0, PARENT_M - 31, 16)
    )[-2:]


def _generate_site(
    *,
    site: ParentSite,
    candidates: list[Candidate],
    base: PinnedBaseHeight,
    config: dict[str, Any],
    root: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    site_root = root / "sites" / site.region_id
    surface_root = site_root / "surface"
    qa_root = site_root / "qa"
    scratch = site_root / "scratch"
    surface_root.mkdir(parents=True)
    qa_root.mkdir(parents=True)
    scratch.mkdir()
    raw_path, ownership_path, dominant_path, placements, source_counts = _assemble_disk(
        candidates, site.bbox_en, scratch, config
    )
    allowed, mask_evidence = _fine_mask(site.bbox_en, scratch / "allowed.u8")
    maximum_mean = _project_disk(raw_path, allowed)
    residual = np.memmap(raw_path, dtype=np.float32, mode="r", shape=(FINE_CELLS, FINE_CELLS))
    maximum_hard = 0.0
    for row0 in range(0, FINE_CELLS, 128):
        row1 = min(FINE_CELLS, row0 + 128)
        hard_values = np.where(allowed[row0:row1] == 0, np.abs(residual[row0:row1]), 0.0)
        maximum_hard = max(maximum_hard, float(np.max(hard_values, initial=0.0)))
    c1, authority = _refine_and_write_c1(
        base=base,
        bbox=site.bbox_en,
        residual_path=raw_path,
        destination=surface_root / "c1_height_f32.npy",
    )
    _pack_mask(allowed, surface_root / "allowed_packbits_u8.npy")
    ownership = np.memmap(ownership_path, dtype=np.uint8, mode="r", shape=(FINE_CELLS, FINE_CELLS))
    ownership_1m = np.asarray(ownership[FACTOR // 2 :: FACTOR, FACTOR // 2 :: FACTOR], dtype=np.uint8)
    np.save(surface_root / "source_ownership_1m_u8.npy", ownership_1m, allow_pickle=False)
    (surface_root / "placements.json").write_bytes(_canonical_json(placements) + b"\n")

    c1_half = _reduce8(c1)
    residual_half = _reduce8(residual)
    c0_half = c1_half - residual_half
    allowed_half = _reduce8(allowed) >= 1.0
    dominant = np.memmap(dominant_path, dtype=np.int32, mode="r", shape=(FINE_CELLS, FINE_CELLS))
    dominant_half = np.asarray(dominant[4::8, 4::8], dtype=np.int32)
    boundary = _boundary_metrics(residual_half, allowed_half, dominant_half)
    palette = np.asarray([[45, 123, 113], [217, 139, 71], [103, 87, 145]], dtype=np.uint8)
    _panel(
        [("PINNED C0 HILLSHADE", _hillshade(c0_half, 0.5)[..., None].repeat(3, 2)),
         ("IRREGULAR FOREST C1", _hillshade(c1_half, 0.5)[..., None].repeat(3, 2))],
        qa_root / "01_c0_c1_full_parent.png",
        f"{site.region_id} | 512 m | 0.5 m QA reduction of 6.25 cm master",
    )
    _panel(
        [("ADDED GEOMETRIC RELIEF", _percentile_rgb(residual_half, diverging=True)),
         ("SOURCE OWNERSHIP", palette[ownership_1m])],
        qa_root / "02_residual_and_ownership.png",
        f"allowed {float(np.mean(allowed)):.1%} | K11 teal K32 orange K36 violet",
    )
    allowed_1m = np.asarray(allowed.reshape(PARENT_M, FACTOR, PARENT_M, FACTOR).all(axis=(1, 3)))
    close_y, close_x = _choose_closeup(allowed_1m)
    fine = np.s_[close_y * FACTOR : (close_y + 32) * FACTOR, close_x * FACTOR : (close_x + 32) * FACTOR]
    close_c1 = np.asarray(c1[fine], dtype=np.float32)
    close_residual = np.asarray(residual[fine], dtype=np.float32)
    close_c0 = close_c1 - close_residual
    _panel(
        [("C0 GROUND-SCALE CLOSEUP", _hillshade(close_c0, TEXEL_M)[..., None].repeat(3, 2)),
         ("C1 GROUND-SCALE CLOSEUP", _hillshade(close_c1, TEXEL_M)[..., None].repeat(3, 2))],
        qa_root / "03_ground_scale_closeup.png",
        f"32 m x 32 m | parent cell ({close_x},{close_y})",
    )
    np.save(qa_root / "closeup_residual_f32.npy", close_residual, allow_pickle=False)
    output_values = residual[np.asarray(allowed, dtype=bool)]
    metrics = {
        "site": asdict(site),
        "mask": mask_evidence,
        "allowed_fraction_exact": float(np.mean(allowed)),
        "site_count": len({item["site_index"] for item in placements}),
        "placement_count": len(placements),
        "source_placement_counts": source_counts,
        "boundary_gradient_ratios_at_0_5m": boundary,
        "residual_rms_allowed_m": float(np.sqrt(np.mean(output_values * output_values))),
        "residual_p01_p99_allowed_m": [float(value) for value in np.percentile(output_values, [1, 99])],
        "maximum_abs_residual_m": float(np.max(np.abs(output_values), initial=0.0)),
        "maximum_one_metre_mean_error_m": maximum_mean,
        "maximum_hard_exclusion_residual_m": maximum_hard,
        "closeup_parent_cell_xy": [close_x, close_y],
        "authority_p01_p99_m": [float(value) for value in np.percentile(authority, [1, 99])],
    }
    (site_root / "metrics.json").write_bytes(_canonical_json(metrics) + b"\n")
    del c1, residual, allowed, ownership, dominant
    shutil.rmtree(scratch)
    return metrics, close_residual


def _nearest_copy(closeups: dict[str, np.ndarray]) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    names = sorted(closeups)
    for left_index, left_name in enumerate(names):
        left = closeups[left_name].astype(np.float64)
        left -= left.mean()
        for right_name in names[left_index + 1 :]:
            right = closeups[right_name].astype(np.float64)
            right -= right.mean()
            variants = [np.rot90(right, rotation) for rotation in range(4)] + [np.rot90(np.fliplr(right), rotation) for rotation in range(4)]
            rmse = min(float(np.sqrt(np.mean((left - variant) ** 2))) for variant in variants)
            pairs.append({"left": left_name, "right": right_name, "nearest_dihedral_rmse_m": rmse})
    return {
        "method": "zero_mean 32m closeups; minimum RMSE across eight dihedral transforms; no translation search",
        "pairs": pairs,
        "minimum_rmse_m": min(pair["nearest_dihedral_rmse_m"] for pair in pairs),
    }


def run_generalization(selection_root: Path) -> Path:
    selection = _json(selection_root / "selection.json")
    selection_manifest = selection_root / "manifest.json"
    if _json(selection_manifest).get("status") != "selection_frozen_before_synthesis":
        raise RuntimeError("site selection is not frozen before synthesis")
    config = _json(CONFIG)
    selected = [
        ParentSite(
            **{
                **row,
                "bbox_en": tuple(row["bbox_en"]),
                "soil_signature": tuple(row["soil_signature"]),
                "geology_signature": tuple(row["geology_signature"]),
            }
        )
        for row in selection["selected"]
    ]
    recipe = {
        "schema": "forest-mesic-mineral-multi-site-float-pilot/1",
        "authority": "research_float_only_no_production_no_pack_no_latest",
        "selection_manifest_sha256": _sha256(selection_manifest),
        "selection_build_id": selection_root.name,
        "config_sha256": _sha256(CONFIG),
        "base_manifest_sha256": BASE_MANIFEST_SHA256,
        "sources": SOURCE_SHA256,
        "implementation_sha256": _sha256(Path(__file__)),
        "irregular_implementation_sha256": _sha256(Path(__file__).with_name("irregular.py")),
        "selected": [asdict(site) for site in selected],
        "retained_surface": "absolute C1 float32 master only; C0 and residual remain derivable QA, not redundant retained masters",
    }
    build_id = hashlib.sha256(_canonical_json(recipe)).hexdigest()
    destination = OUTPUT_ROOT / "artifact/sha256" / build_id
    if destination.exists():
        raise FileExistsError(f"immutable artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    candidates = _load_candidates()
    base = PinnedBaseHeight(BASE_MANIFEST, BASE_MANIFEST_SHA256, ASSET_GEN_ROOT / "data/out", load_base().encode, audit=False)
    # Failed full-resolution runs are retained for diagnosis rather than silently
    # discarding multi-gigabyte completed stages.
    with TemporaryDirectory(prefix="forest-generalization-", dir=destination.parent, delete=False) as temporary:
        staging = Path(temporary)
        (staging / "qa").mkdir()
        (staging / "recipe.json").write_bytes(_canonical_json(recipe) + b"\n")
        metrics: dict[str, Any] = {}
        closeups: dict[str, np.ndarray] = {}
        for site in selected:
            metrics[site.region_id], closeups[site.region_id] = _generate_site(
                site=site,
                candidates=candidates,
                base=base,
                config=config,
                root=staging,
            )
        repetition = _nearest_copy(closeups)
        (staging / "qa/04_cross_site_repetition.json").write_bytes(_canonical_json(repetition) + b"\n")
        acceptance = config["acceptance"]
        failures: list[str] = []
        for name, site_metrics in metrics.items():
            if site_metrics["maximum_one_metre_mean_error_m"] > acceptance["maximum_one_metre_mean_error_m"]:
                failures.append(f"{name}: one-metre mean")
            if site_metrics["maximum_hard_exclusion_residual_m"] > acceptance["maximum_hard_exclusion_residual_m"]:
                failures.append(f"{name}: hard exclusion")
            for key, value in site_metrics["boundary_gradient_ratios_at_0_5m"].items():
                limit = acceptance["maximum_boundary_p95_ratio"] if "p95" in key else acceptance["maximum_boundary_mean_ratio"]
                if value > limit:
                    failures.append(f"{name}: {key}")
        if repetition["minimum_rmse_m"] < acceptance["minimum_cross_site_nearest_copy_rmse_m"]:
            failures.append("cross-site nearest-copy")
        files: dict[str, Any] = {}
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                files[str(path.relative_to(staging))] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        manifest = {
            "schema": "forest-mesic-mineral-multi-site-float-artifact/1",
            "build_id": build_id,
            "status": "research_candidate_pass" if not failures else "research_candidate_fail",
            "failures": failures,
            "site_metrics": metrics,
            "cross_site_repetition": repetition,
            "limitations": [
                "foreign analogue hypothesis, not Estonia target truth or production owner",
                "one forest regime only; no wet forest, peat, cliff, water, engineered ground, or unknown-condition synthesis",
                "float-only artifact; no cook, pack, browser, latest, or national release authority",
            ],
            "files": files,
        }
        (staging / "manifest.json").write_bytes(_canonical_json(manifest) + b"\n")
        os.replace(staging, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("select")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--selection", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "select":
        print(select_sites())
    else:
        print(run_generalization(arguments.selection))


if __name__ == "__main__":
    main()
