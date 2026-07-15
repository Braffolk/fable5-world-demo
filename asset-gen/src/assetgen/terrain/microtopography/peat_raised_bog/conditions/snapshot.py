"""Materialize the v2 raised-bog development condition snapshot."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyogrio
import pyogrio.raw
import rasterio
import rasterio.features
import rasterio.transform
import shapely
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .....config import ASSET_GEN_ROOT, DATA_IN, DATA_OUT, DATA_WORK, load_base
from .....cook.pinned_height import PinnedBaseHeight
from .....grid import ChunkId, chunks_covering_bbox_en
from .....release import audit_base_release, read_v1_index
from ....conditions.geology import extract_egt_surficial_window
from ....conditions.soil import extract_soil_window
from .select_sites import GEOLOGY_DOMAINS, _soil_support


REPOSITORY_ROOT = ASSET_GEN_ROOT.parent
PREREGISTRATION = (
    ASSET_GEN_ROOT
    / "config/microtopography/peat-raised-bog/bundle-preregistration-v2.json"
)
PREREGISTRATION_SHA256 = (
    "431b269d49f7e85457a531cc669a3422e8eee077a96958d9beb9dc7c42da0d35"
)
SELECTION = (
    DATA_WORK
    / "microtopography/peat-raised-bog/site-selection/sha256/"
    "46daa389e388a9783bff96f7caba3abedd1293bfe151f98ff13ea9d9b5c0750a/"
    "selection.json"
)
SELECTION_RECIPE_SHA256 = (
    "46daa389e388a9783bff96f7caba3abedd1293bfe151f98ff13ea9d9b5c0750a"
)
SELECTION_SHA256 = (
    "ff7cca92e6767e96c63c1d01376f836a74b0fc61ceb7824b12be240e65a837be"
)
CORRECTED_BASE = (
    DATA_WORK
    / "terrain-repair/taevaskoda-ahja-stage1-transaction/"
    "009fd3d2be07d465199e6330ac9d2fc5a8fc7a16bbb5df0f55e5030ce962d3d1/"
    "corrected-format1-release/m/3809a50163019bcb/manifest.json"
)
CORRECTED_BASE_SHA256 = (
    "3809a50163019bcb0a82e8fe9f11b3fb653d714c3d9cb660690ba0704405ebe0"
)
CORRECTED_BASE_VERIFICATION = CORRECTED_BASE.parents[2] / "verification.json"
CORRECTED_BASE_VERIFICATION_SHA256 = (
    "21d73467afbabc7213873ac13fe4b8ca0d8119f98e61670aaa0f2a19dfb60e06"
)
COUNTRY_DTM = DATA_IN / "country/DTM_10m_eesti.tif"
ETAK = DATA_IN / "etak/ETAK_EESTI_GPKG.gpkg"
BASE_CONFIG = ASSET_GEN_ROOT / "config/base.toml"
OUTPUT_ROOT = (
    DATA_WORK / "microtopography/peat-raised-bog/condition-snapshot/sha256"
)

MIRE_LAYER = "E_306_margala_a"
DITCH_TYPES = {20, 40, 50}
HALO_CELL_M = 1.0
HYDROLOGY_CELL_M = 10.0
HYDROLOGY_CONTEXT_M = 100.0

MASK_LAYERS: dict[str, tuple[str, tuple[str, ...]]] = {
    "forest": ("E_305_puittaimestik_a", ("etak_id", "vajalik_t")),
    "sea": ("E_201_meri_a", ("etak_id", "vajalik_t")),
    "lake": ("E_202_seisuveekogu_a", ("etak_id", "vajalik_t")),
    "flowing_water": ("E_203_vooluveekogu_a", ("etak_id", "vajalik_t")),
    "water_line": (
        "E_203_vooluveekogu_j",
        ("etak_id", "tyyp", "laius", "vajalik_t"),
    ),
    "cut_peat": ("E_307_turbavali_a", ("etak_id", "vajalik_t")),
    "road_area": ("E_501_tee_a", ("etak_id", "vajalik_t")),
    "road_line": ("E_501_tee_j", ("etak_id", "laius", "vajalik_t")),
    "building": ("E_401_hoone_ka", ("etak_id", "vajalik_t")),
    "high_object": ("E_402_korgrajatis_p", ("etak_id", "vajalik_t")),
    "other_object_area": ("E_403_muu_rajatis_ka", ("etak_id", "vajalik_t")),
    "other_object_point": ("E_403_muu_rajatis_p", ("etak_id", "vajalik_t")),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPOSITORY_ROOT.resolve()))


def _identity(path: Path) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _verify_bound_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    actual = _sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"bound artifact changed: {path}: {actual}")
    return json.loads(path.read_bytes())


def _read_layer(
    layer: str,
    bbox: tuple[float, float, float, float],
    columns: tuple[str, ...] = (),
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    metadata, _fids, wkbs, values = pyogrio.raw.read(
        ETAK,
        layer=layer,
        bbox=bbox,
        columns=list(columns),
        return_fids=True,
    )
    if str(metadata.get("crs")) != "EPSG:3301" or wkbs is None:
        raise ValueError(f"invalid ETAK source layer {layer}")
    geometries = np.asarray(
        [shapely.force_2d(shapely.from_wkb(bytes(value))) for value in wkbs],
        dtype=object,
    )
    return geometries, dict(zip(metadata.get("fields", ()), values, strict=True))


def _rasterize(
    geometries: Iterable[shapely.Geometry],
    shape: tuple[int, int],
    transform: rasterio.Affine,
    *,
    all_touched: bool = True,
) -> np.ndarray:
    rows = [(geometry, 1) for geometry in geometries if not geometry.is_empty]
    if not rows:
        return np.zeros(shape, dtype=bool)
    return rasterio.features.rasterize(
        rows,
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=all_touched,
        dtype="uint8",
    ).astype(bool)


def _buffer_lines(
    geometries: np.ndarray,
    widths: np.ndarray | None,
    minimum_half_width_m: float,
    selected: np.ndarray | None = None,
) -> list[shapely.Geometry]:
    rows: list[shapely.Geometry] = []
    for index, geometry in enumerate(geometries):
        if selected is not None and not bool(selected[index]):
            continue
        width = None if widths is None else widths[index]
        try:
            half_width = max(minimum_half_width_m, 0.5 * float(width))
        except (TypeError, ValueError):
            half_width = minimum_half_width_m
        rows.append(geometry.buffer(half_width, cap_style="flat"))
    return rows


def _load_selected_mire(selection: dict[str, Any]) -> shapely.Geometry:
    ids = selection["development_mire"]["member_etak_ids"]
    where = " OR ".join(f"etak_id = {int(value)}" for value in ids)
    metadata, _fids, wkbs, values = pyogrio.raw.read(
        ETAK,
        layer=MIRE_LAYER,
        where=where,
        columns=["etak_id", "vajalik_t", "muutmisaeg", "geom_muutmisaeg"],
        return_fids=True,
    )
    if str(metadata.get("crs")) != "EPSG:3301" or wkbs is None:
        raise ValueError("selected ETAK mire geometry is unavailable")
    found = sorted(int(value) for value in values[0])
    if found != sorted(int(value) for value in ids):
        raise ValueError("selected ETAK mire membership changed")
    geometry = shapely.union_all(
        [shapely.force_2d(shapely.from_wkb(bytes(value))) for value in wkbs]
    )
    expected_sha = selection["development_mire"]["geometry_sha256"]
    if hashlib.sha256(geometry.wkb).hexdigest() != expected_sha:
        raise ValueError("selected connected-mire geometry changed")
    return geometry


def _soil_semantics(feature: dict[str, Any]) -> dict[str, Any]:
    raw = feature["raw_attributes"]
    attributes = {
        key: np.asarray([raw.get(key)], dtype=object)
        for key in (
            "Siffer",
            "Sif1",
            "Osa1",
            "Sif2",
            "Osa2",
            "Sif3",
            "Osa3",
            "Sif4",
            "Osa4",
            "Loimis1",
            "Loimis2",
            "Huumus",
        )
    }
    return _soil_support(attributes, 0)


def _condition_vectors(
    halo_bbox: tuple[int, int, int, int],
    whole_bbox: tuple[int, int, int, int],
    mire: shapely.Geometry,
    soil_path: Path,
    geology_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, list[shapely.Geometry]]]:
    shape = (halo_bbox[3] - halo_bbox[1], halo_bbox[2] - halo_bbox[0])
    transform = rasterio.transform.from_origin(
        halo_bbox[0], halo_bbox[3], HALO_CELL_M, HALO_CELL_M
    )
    masks: dict[str, np.ndarray] = {
        "mire_coverage": _rasterize([mire], shape, transform),
    }
    vector_records: dict[str, Any] = {}
    whole_geometries: dict[str, list[shapely.Geometry]] = {}
    dynamic_geometries: list[shapely.Geometry] = []
    halo_layers: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    for name, (layer, fields) in MASK_LAYERS.items():
        geometries, attributes = _read_layer(layer, whole_bbox, fields)
        whole_geometries[name] = list(geometries)
        halo_layers[name] = (geometries, attributes)
        status = attributes.get("vajalik_t", np.asarray([], dtype=object))
        dynamic = np.asarray(
            [value is not None and str(value) != "Korras" for value in status],
            dtype=bool,
        )
        dynamic_geometries.extend(
            geometry for geometry, flagged in zip(geometries, dynamic, strict=True) if flagged
        )
        vector_records[name] = {
            "layer": layer,
            "feature_count_in_whole_context": len(geometries),
            "dynamic_or_noncurrent_count": int(dynamic.sum()),
        }

    forest = halo_layers["forest"][0]
    water = [
        *halo_layers["sea"][0],
        *halo_layers["lake"][0],
        *halo_layers["flowing_water"][0],
    ]
    lines, line_attrs = halo_layers["water_line"]
    line_types = line_attrs.get("tyyp", np.full(len(lines), -1))
    ditch_selected = np.asarray(
        [value is not None and int(value) in DITCH_TYPES for value in line_types],
        dtype=bool,
    )
    ditches = _buffer_lines(lines, line_attrs.get("laius"), 1.0, ditch_selected)
    road_lines, road_attrs = halo_layers["road_line"]
    roads = [
        *halo_layers["road_area"][0],
        *_buffer_lines(road_lines, road_attrs.get("laius"), 1.5),
    ]
    objects = [
        *halo_layers["high_object"][0],
        *halo_layers["other_object_area"][0],
        *halo_layers["other_object_point"][0],
    ]
    masks.update(
        {
            "forest": _rasterize(forest, shape, transform),
            "open_water": _rasterize(water, shape, transform),
            "ditch": _rasterize(ditches, shape, transform),
            "cut_peat": _rasterize(halo_layers["cut_peat"][0], shape, transform),
            "road": _rasterize(roads, shape, transform),
            "building": _rasterize(halo_layers["building"][0], shape, transform),
            "object": _rasterize(objects, shape, transform),
            "dynamic_unknown": _rasterize(dynamic_geometries, shape, transform),
        }
    )

    soil = json.loads(soil_path.read_bytes())
    soil_supported = np.zeros(shape, dtype=bool)
    soil_coverage = np.zeros(shape, dtype=bool)
    soil_records: list[dict[str, Any]] = []
    for feature in soil["features"]:
        geometry = shapely.force_2d(
            shapely.from_wkb(bytes.fromhex(feature["geometry"]["ogr_wkb_hex"]))
        )
        feature_mask = _rasterize([geometry], shape, transform)
        semantics = _soil_semantics(feature)
        soil_coverage |= feature_mask
        if semantics["supported"]:
            soil_supported |= feature_mask
        soil_records.append(
            {
                "source_fid": feature["feature_identity"]["source_fid"],
                **semantics,
            }
        )
    masks["soil_supported"] = soil_supported
    masks["soil_unknown"] = ~soil_coverage | ~soil_supported

    geology = json.loads(geology_path.read_bytes())
    geology_known = np.zeros(shape, dtype=bool)
    geology_coverage = np.zeros(shape, dtype=bool)
    lithology = np.full(shape, -1, dtype=np.int32)
    genesis = np.full(shape, -1, dtype=np.int32)
    for feature in geology["features"]:
        geometry = shapely.force_2d(
            shapely.from_wkb(bytes.fromhex(feature["geometry"]["ogr_wkb_hex"]))
        )
        feature_mask = _rasterize([geometry], shape, transform)
        geology_coverage |= feature_mask
        decoded = feature["decoded"]
        known = all(
            decoded[key]["status"] == "decoded_from_frozen_official_domain"
            for key in ("lithology", "genesis")
        )
        if known:
            geology_known |= feature_mask
            lithology[feature_mask] = int(decoded["lithology"]["code"])
            genesis[feature_mask] = int(decoded["genesis"]["code"])
    masks["geology_known"] = geology_known
    masks["geology_unknown"] = ~geology_coverage | ~geology_known
    masks["geology_lithology_code"] = lithology
    masks["geology_genesis_code"] = genesis
    hard_names = ("forest", "open_water", "ditch", "cut_peat", "road", "building", "object")
    masks["hard_exclusion"] = np.logical_or.reduce([masks[name] for name in hard_names])
    masks["ownership_unknown"] = (
        ~masks["mire_coverage"]
        | masks["soil_unknown"]
        | masks["geology_unknown"]
        | masks["dynamic_unknown"]
    )
    masks["authority"] = (
        masks["mire_coverage"]
        & ~masks["hard_exclusion"]
        & ~masks["ownership_unknown"]
    )
    metadata = {
        "etak_layers": vector_records,
        "soil": {
            "artifact": _identity(soil_path),
            "whole_context_records": soil_records,
            "coverage": soil["coverage"],
        },
        "geology": {
            "artifact": _identity(geology_path),
            "coverage": geology["coverage"],
            "decode_status_counts": geology["decode_status_counts"],
        },
    }
    return masks, metadata, whole_geometries


def _priority_fill(height: np.ndarray) -> np.ndarray:
    rows, cols = height.shape
    filled = np.asarray(height, dtype=np.float64).copy()
    visited = np.zeros(height.shape, dtype=bool)
    heap: list[tuple[float, int, int]] = []
    for row in range(rows):
        for col in (0, cols - 1):
            visited[row, col] = True
            heapq.heappush(heap, (float(filled[row, col]), row, col))
    for col in range(1, cols - 1):
        for row in (0, rows - 1):
            visited[row, col] = True
            heapq.heappush(heap, (float(filled[row, col]), row, col))
    neighbors = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    while heap:
        level, row, col = heapq.heappop(heap)
        for drow, dcol in neighbors:
            rr, cc = row + drow, col + dcol
            if not (0 <= rr < rows and 0 <= cc < cols) or visited[rr, cc]:
                continue
            visited[rr, cc] = True
            filled[rr, cc] = max(level, float(filled[rr, cc]))
            heapq.heappush(heap, (float(filled[rr, cc]), rr, cc))
    return filled.astype(np.float32)


def _flow_fields(height: np.ndarray, cell_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    filled = _priority_fill(height)
    directions = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
    flow = np.full(height.shape, -1, dtype=np.int8)
    best = np.zeros(height.shape, dtype=np.float32)
    for code, (drow, dcol) in enumerate(directions):
        shifted = np.full(height.shape, np.inf, dtype=np.float32)
        source_rows = slice(max(0, drow), min(height.shape[0], height.shape[0] + drow))
        source_cols = slice(max(0, dcol), min(height.shape[1], height.shape[1] + dcol))
        target_rows = slice(max(0, -drow), min(height.shape[0], height.shape[0] - drow))
        target_cols = slice(max(0, -dcol), min(height.shape[1], height.shape[1] - dcol))
        shifted[target_rows, target_cols] = filled[source_rows, source_cols]
        distance = cell_m * (2.0**0.5 if drow and dcol else 1.0)
        descent = (filled - shifted) / distance
        take = descent > best
        best[take] = descent[take]
        flow[take] = code
    accumulation = np.ones(height.shape, dtype=np.float64)
    order = np.argsort(filled.ravel(), kind="stable")[::-1]
    width = height.shape[1]
    for flat in order:
        row, col = divmod(int(flat), width)
        code = int(flow[row, col])
        if code < 0:
            continue
        drow, dcol = directions[code]
        rr, cc = row + drow, col + dcol
        if 0 <= rr < height.shape[0] and 0 <= cc < height.shape[1]:
            accumulation[rr, cc] += accumulation[row, col]
    wetness = np.log((accumulation * cell_m) / np.maximum(best, 1e-4))
    return flow, accumulation.astype(np.float32), wetness.astype(np.float32)


def _country_hydrology(
    mire: shapely.Geometry,
    whole_geometries: dict[str, list[shapely.Geometry]],
) -> tuple[dict[str, np.ndarray], tuple[int, int, int, int], dict[str, Any]]:
    min_e, min_n, max_e, max_n = mire.bounds
    bbox = (
        int(np.floor((min_e - HYDROLOGY_CONTEXT_M) / HYDROLOGY_CELL_M) * HYDROLOGY_CELL_M),
        int(np.floor((min_n - HYDROLOGY_CONTEXT_M) / HYDROLOGY_CELL_M) * HYDROLOGY_CELL_M),
        int(np.ceil((max_e + HYDROLOGY_CONTEXT_M) / HYDROLOGY_CELL_M) * HYDROLOGY_CELL_M),
        int(np.ceil((max_n + HYDROLOGY_CONTEXT_M) / HYDROLOGY_CELL_M) * HYDROLOGY_CELL_M),
    )
    with rasterio.open(COUNTRY_DTM) as source:
        if str(source.crs) != "EPSG:3301" or source.transform.a != 10.0 or source.transform.e != -10.0:
            raise ValueError("country DTM context grid changed")
        window = rasterio.windows.from_bounds(*bbox, source.transform)
        height = source.read(1, window=window, out_dtype="float32")
        invalid = ~np.isfinite(height) | (height == np.float32(source.nodata))
        if invalid.any():
            raise ValueError("country DTM hydrology window contains invalid cells")
    shape = height.shape
    transform = rasterio.transform.from_origin(bbox[0], bbox[3], 10.0, 10.0)
    mire_mask = _rasterize([mire], shape, transform, all_touched=False)
    water_geometries = [
        *whole_geometries["sea"],
        *whole_geometries["lake"],
        *whole_geometries["flowing_water"],
        *whole_geometries["water_line"],
    ]
    water = _rasterize(water_geometries, shape, transform)
    lines = whole_geometries["water_line"]
    ditch = _rasterize(lines, shape, transform)
    flow, accumulation, wetness = _flow_fields(height, HYDROLOGY_CELL_M)
    gy, gx = np.gradient(height.astype(np.float64), HYDROLOGY_CELL_M)
    slope = np.hypot(gx, gy).astype(np.float32)
    curvature = (
        ndimage.laplace(height.astype(np.float64), mode="nearest")
        / (HYDROLOGY_CELL_M**2)
    ).astype(np.float32)
    water_distance = ndimage.distance_transform_edt(~water).astype(np.float32) * HYDROLOGY_CELL_M
    ditch_distance = ndimage.distance_transform_edt(~ditch).astype(np.float32) * HYDROLOGY_CELL_M
    return {
        "context_height_m": height,
        "context_slope": slope,
        "context_curvature_per_m": curvature,
        "flow_direction_d8": flow,
        "flow_accumulation_cells": accumulation,
        "topographic_wetness": wetness,
        "water_distance_m": water_distance,
        "ditch_distance_m": ditch_distance,
        "whole_mire_coverage": mire_mask,
        "context_water": water,
    }, bbox, {
        "source": _identity(COUNTRY_DTM),
        "role": "official_10m_whole_mire_context_only_not_corrected_C0",
        "cell_m": HYDROLOGY_CELL_M,
        "bbox_en": list(bbox),
    }


def _corrected_height(
    halo_bbox: tuple[int, int, int, int],
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[str]]:
    audit_base_release(CORRECTED_BASE, CORRECTED_BASE_SHA256, DATA_OUT)
    manifest = json.loads(CORRECTED_BASE.read_bytes())
    index_path = CORRECTED_BASE.parent / manifest["layers"]["height"]["index"]
    records = {record.key for record in read_v1_index(index_path)}
    grid = load_base().grid
    required = [chunk for chunk in chunks_covering_bbox_en(grid, halo_bbox, 0)]
    missing = [chunk for chunk in required if (0, chunk.cx, chunk.cz) not in records]
    shape = (halo_bbox[3] - halo_bbox[1], halo_bbox[2] - halo_bbox[0])
    if missing:
        arrays = {
            "corrected_structural_height_m": np.full(shape, np.nan, dtype=np.float32),
            "corrected_structural_slope": np.full(shape, np.nan, dtype=np.float32),
            "corrected_structural_curvature_per_m": np.full(shape, np.nan, dtype=np.float32),
            "corrected_height_unknown": np.ones(shape, dtype=bool),
        }
        blocker = (
            "accepted corrected format-1 base lacks selected development LOD0: "
            + ", ".join(f"(0,{chunk.cx},{chunk.cz})" for chunk in missing)
        )
        return arrays, {
            "available": False,
            "required_lod0": [[0, chunk.cx, chunk.cz] for chunk in required],
            "missing_lod0": [[0, chunk.cx, chunk.cz] for chunk in missing],
        }, [blocker]
    reader = PinnedBaseHeight(
        CORRECTED_BASE,
        CORRECTED_BASE_SHA256,
        DATA_OUT,
        load_base().encode,
        audit=False,
    )
    padded = reader.read_cells(
        halo_bbox[0] - 1,
        halo_bbox[3] + 1,
        shape[1] + 2,
        shape[0] + 2,
    )
    height = padded[1:-1, 1:-1]
    gy, gx = np.gradient(padded.astype(np.float64), HALO_CELL_M)
    slope = np.hypot(gx, gy)[1:-1, 1:-1].astype(np.float32)
    curvature = ndimage.laplace(padded.astype(np.float64), mode="nearest")[1:-1, 1:-1].astype(np.float32)
    return {
        "corrected_structural_height_m": height.astype(np.float32),
        "corrected_structural_slope": slope,
        "corrected_structural_curvature_per_m": curvature,
        "corrected_height_unknown": np.zeros(shape, dtype=bool),
    }, {
        "available": True,
        "required_lod0": [[0, chunk.cx, chunk.cz] for chunk in required],
        "missing_lod0": [],
    }, []


def _colorize(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    finite = np.isfinite(values)
    if mask is not None:
        finite &= mask
    samples = values[finite]
    low, high = np.percentile(samples, [2, 98]) if samples.size else (0.0, 1.0)
    unit = np.clip((values - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    rgb = np.stack((40 + 180 * unit, 60 + 150 * np.sqrt(unit), 180 - 120 * unit), axis=-1)
    rgb = np.nan_to_num(rgb, nan=18.0).astype(np.uint8)
    if mask is not None:
        rgb[~mask] = (18, 18, 18)
    return rgb


def _mask_rgb(masks: dict[str, np.ndarray]) -> np.ndarray:
    rgb = np.zeros((*masks["authority"].shape, 3), dtype=np.uint8)
    rgb[:] = (34, 38, 30)
    rgb[masks["mire_coverage"]] = (76, 110, 64)
    rgb[masks["authority"]] = (129, 174, 80)
    rgb[masks["ownership_unknown"]] = (225, 155, 55)
    rgb[masks["hard_exclusion"]] = (184, 55, 42)
    rgb[masks["open_water"]] = (42, 121, 190)
    rgb[masks["ditch"]] = (65, 185, 214)
    return rgb


def _write_qa(
    path: Path,
    masks: dict[str, np.ndarray],
    hydrology: dict[str, np.ndarray],
    blockers: list[str],
) -> None:
    font = ImageFont.load_default()
    panels = [
        ("1 m halo: authority / hard / unknown", _mask_rgb(masks)),
        (
            "10 m whole-mire corrected-height status" if not blockers else "10 m whole-mire context (NOT corrected C0)",
            _colorize(hydrology["context_height_m"], hydrology["whole_mire_coverage"]),
        ),
        (
            "10 m topographic wetness",
            _colorize(hydrology["topographic_wetness"], hydrology["whole_mire_coverage"]),
        ),
        (
            "10 m flow accumulation",
            _colorize(np.log1p(hydrology["flow_accumulation_cells"]), hydrology["whole_mire_coverage"]),
        ),
    ]
    panel_w, panel_h = 640, 420
    canvas = Image.new("RGB", (panel_w * 2, panel_h * 2 + 40), (12, 14, 12))
    draw = ImageDraw.Draw(canvas)
    for index, (label, rgb) in enumerate(panels):
        image = Image.fromarray(rgb).resize((panel_w, panel_h - 26), Image.Resampling.NEAREST)
        x, y = (index % 2) * panel_w, (index // 2) * panel_h + 40
        canvas.paste(image, (x, y + 26))
        draw.text((x + 8, y + 8), label, fill=(235, 238, 225), font=font)
    title = "Raised-bog v2 development conditions"
    if blockers:
        title += " | BLOCKED: corrected C0 has no selected-site coverage"
    draw.text((10, 14), title, fill=(246, 224, 155) if blockers else (225, 240, 210), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=True)


def _verify_existing_artifact(
    manifest_path: Path,
    *,
    recipe_sha256: str,
    recipe: dict[str, Any],
    arrays: dict[str, np.ndarray],
    status: str,
    blockers: list[str],
    expected_development: dict[str, Any],
    expected_sealed: dict[str, Any],
    masks: dict[str, np.ndarray],
    hydrology: dict[str, np.ndarray],
) -> None:
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest)
    if raw_manifest != _canonical_bytes(manifest):
        raise ValueError("existing snapshot JSON is not canonical")
    if (
        manifest.get("schema_version") != "laas.peat-raised-bog-condition-snapshot/1"
        or manifest.get("recipe_sha256") != recipe_sha256
        or manifest.get("recipe") != recipe
        or manifest.get("status") != status
        or manifest.get("blockers") != blockers
        or manifest.get("development") != expected_development
        or manifest.get("sealed_abstention") != expected_sealed
    ):
        raise ValueError("existing snapshot differs from reconstructed identity or status")

    array_row = manifest.get("arrays")
    if not isinstance(array_row, dict) or array_row.get("path") != "conditions.npz":
        raise ValueError("existing snapshot lacks the required NPZ inventory")
    npz_path = manifest_path.parent / array_row["path"]
    if (
        not npz_path.is_file()
        or npz_path.stat().st_size != array_row.get("bytes")
        or _sha256(npz_path) != array_row.get("sha256")
    ):
        raise ValueError("existing condition NPZ byte identity changed")
    expected_inventory = {
        name: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values_sha256": _array_sha256(value),
        }
        for name, value in sorted(arrays.items())
    }
    if array_row.get("inventory") != expected_inventory:
        raise ValueError("existing condition array inventory differs from reconstruction")
    with np.load(npz_path, allow_pickle=False) as stored:
        if set(stored.files) != set(arrays):
            raise ValueError("existing condition NPZ has missing or unexpected arrays")
        for name, expected in arrays.items():
            actual = np.asarray(stored[name])
            if (
                actual.dtype != expected.dtype
                or actual.shape != expected.shape
                or _array_sha256(actual) != _array_sha256(expected)
            ):
                raise ValueError(f"existing condition array changed: {name}")

    qa_row = manifest.get("qa")
    if not isinstance(qa_row, dict) or qa_row.get("path") != "qa/01_conditions_and_hydrology.png":
        raise ValueError("existing snapshot lacks the required QA image")
    qa_path = manifest_path.parent / qa_row["path"]
    if (
        not qa_path.is_file()
        or qa_path.stat().st_size != qa_row.get("bytes")
        or _sha256(qa_path) != qa_row.get("sha256")
    ):
        raise ValueError("existing QA PNG byte identity changed")
    with tempfile.TemporaryDirectory(prefix="peat-condition-replay-", dir=manifest_path.parent.parent) as directory:
        reconstructed_qa = Path(directory) / "qa.png"
        _write_qa(reconstructed_qa, masks, hydrology, blockers)
        if reconstructed_qa.read_bytes() != qa_path.read_bytes():
            raise ValueError("existing QA PNG differs from reconstructed pixels/encoding")

    qa_index_path = manifest_path.parent / "qa/index.json"
    qa_index = {
        "schema_version": "laas.qa-image-index/1",
        "recipe_sha256": recipe_sha256,
        "images": [qa_row],
        "source_npz_sha256": array_row["sha256"],
    }
    if (
        not qa_index_path.is_file()
        or qa_index_path.read_bytes() != _canonical_bytes(qa_index)
    ):
        raise ValueError("existing QA index differs from reconstructed identity")


def materialize_snapshot(output_root: Path = OUTPUT_ROOT) -> Path:
    preregistration = _verify_bound_json(PREREGISTRATION, PREREGISTRATION_SHA256)
    selection = _verify_bound_json(SELECTION, SELECTION_SHA256)
    if selection.get("recipe_sha256") != SELECTION_RECIPE_SHA256:
        raise ValueError("site-selection recipe identity changed")
    verification = _verify_bound_json(
        CORRECTED_BASE_VERIFICATION, CORRECTED_BASE_VERIFICATION_SHA256
    )
    if (
        _sha256(CORRECTED_BASE) != CORRECTED_BASE_SHA256
        or verification.get("passed") is not True
        or verification.get("manifestSha256") != CORRECTED_BASE_SHA256
    ):
        raise ValueError("accepted corrected base binding is invalid")
    development = selection["development_mire"]
    sealed = selection["sealed_abstention_mire"]
    halo_bbox = tuple(int(value) for value in development["selected_halo_bounds_en"])
    if list(halo_bbox) != [int(value) for value in preregistration["site_selection"]["development"]["halo_bounds_en"]]:
        raise ValueError("v2 development halo differs from frozen selection")
    mire = _load_selected_mire(selection)
    whole_bbox = tuple(int(value) for value in (
        np.floor(mire.bounds[0]),
        np.floor(mire.bounds[1]),
        np.ceil(mire.bounds[2]),
        np.ceil(mire.bounds[3]),
    ))
    selection_source = {
        "kind": "bound_raised_bog_v2_site_selection",
        "path": _relative(SELECTION),
        "sha256": SELECTION_SHA256,
        "development_only_pixels_opened": True,
        "sealed_abstention_pixels_opened": False,
    }
    soil_path = extract_soil_window(
        whole_bbox,
        name="peat-raised-bog-v2-development-whole-mire",
        selection_source=selection_source,
    )
    geology_path = extract_egt_surficial_window(
        whole_bbox,
        name="peat-raised-bog-v2-development-whole-mire",
        domain_snapshot_path=GEOLOGY_DOMAINS,
        selection_source=selection_source,
    )
    masks, condition_metadata, whole_geometries = _condition_vectors(
        halo_bbox, whole_bbox, mire, soil_path, geology_path
    )
    corrected, corrected_metadata, blockers = _corrected_height(halo_bbox)
    masks.update(corrected)
    masks["ownership_unknown"] |= masks["corrected_height_unknown"]
    masks["authority"] &= ~masks["corrected_height_unknown"]
    hydrology, hydrology_bbox, hydrology_metadata = _country_hydrology(
        mire, whole_geometries
    )
    if blockers:
        blockers.append(
            "whole-mire hydrology is official 10 m context only; corrected-C0 hydrology remains unavailable"
        )

    recipe = {
        "schema_version": "laas.peat-raised-bog-condition-snapshot-recipe/1",
        "preregistration": _identity(PREREGISTRATION),
        "selection": {
            **_identity(SELECTION),
            "recipe_sha256": SELECTION_RECIPE_SHA256,
        },
        "corrected_base": {
            "manifest": _identity(CORRECTED_BASE),
            "verification": _identity(CORRECTED_BASE_VERIFICATION),
        },
        "canonical_sources": {
            "etak_gpkg": _identity(ETAK),
            "base_config": _identity(BASE_CONFIG),
        },
        "development": {
            "mire_id": development["normalized_mire_identifier"],
            "member_etak_ids": development["member_etak_ids"],
            "mire_geometry_sha256": development["geometry_sha256"],
            "halo_geometry_sha256": development["selected_halo_geometry_sha256"],
            "whole_mire_bounds_en": list(whole_bbox),
            "core_bounds_en": development["selected_core_bounds_en"],
            "halo_bounds_en": list(halo_bbox),
            "halo_cell_m": HALO_CELL_M,
        },
        "sealed_abstention": {
            "mire_id": sealed["normalized_mire_identifier"],
            "selection_identity_only": True,
            "pixel_sources_opened": [],
            "authority": "identically_zero",
            "unresolved_conditions": sealed["unknown_conditions"],
        },
        "condition_inputs": condition_metadata,
        "derived_condition_identity": {
            "halo_masks": {
                name: _array_sha256(masks[name])
                for name in (
                    "mire_coverage",
                    "forest",
                    "open_water",
                    "ditch",
                    "cut_peat",
                    "road",
                    "building",
                    "object",
                    "dynamic_unknown",
                    "soil_supported",
                    "soil_unknown",
                    "geology_known",
                    "geology_unknown",
                    "hard_exclusion",
                    "ownership_unknown",
                    "authority",
                    "corrected_height_unknown",
                )
            },
            "whole_mire_coverage_sha256": _array_sha256(
                hydrology["whole_mire_coverage"]
            ),
            "corrected_height_coverage": corrected_metadata,
        },
        "hydrology": hydrology_metadata,
        "implementation": _identity(Path(__file__)),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyogrio": pyogrio.__version__,
            "rasterio": rasterio.__version__,
            "shapely": shapely.__version__,
        },
        "policy": {
            "unknown_or_out_of_envelope": "C1_equals_zero",
            "hard_exclusion": "C1_equals_zero",
            "country_dtm_role": "whole_mire_context_only_not_corrected_C0",
            "no_synthesis": True,
        },
    }
    recipe_sha = hashlib.sha256(_canonical_bytes(recipe)).hexdigest()
    target = output_root / recipe_sha
    manifest_path = target / "snapshot.json"
    arrays = {**masks, **hydrology}
    status = "blocked_missing_corrected_height_coverage" if blockers else "complete"
    expected_development = {
        "halo_bbox_en": list(halo_bbox),
        "halo_shape": [halo_bbox[3] - halo_bbox[1], halo_bbox[2] - halo_bbox[0]],
        "whole_mire_bbox_en": list(whole_bbox),
        "hydrology_bbox_en": list(hydrology_bbox),
        "corrected_height": corrected_metadata,
        "authority_cells": int(masks["authority"].sum()),
        "hard_exclusion_cells": int(masks["hard_exclusion"].sum()),
        "ownership_unknown_cells": int(masks["ownership_unknown"].sum()),
    }
    expected_sealed = {
        "mire_id": sealed["normalized_mire_identifier"],
        "pixel_sources_opened": [],
        "authority_cells": 0,
        "positive_evidence_credit": False,
    }
    if manifest_path.is_file():
        _verify_existing_artifact(
            manifest_path,
            recipe_sha256=recipe_sha,
            recipe=recipe,
            arrays=arrays,
            status=status,
            blockers=blockers,
            expected_development=expected_development,
            expected_sealed=expected_sealed,
            masks=masks,
            hydrology=hydrology,
        )
        return manifest_path
    temporary = output_root / f".{recipe_sha}.tmp-{os.getpid()}"
    temporary.mkdir(parents=True, exist_ok=False)
    npz_path = temporary / "conditions.npz"
    np.savez_compressed(npz_path, **arrays)
    qa_path = temporary / "qa/01_conditions_and_hydrology.png"
    _write_qa(qa_path, masks, hydrology, blockers)
    array_inventory = {
        name: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values_sha256": _array_sha256(value),
        }
        for name, value in sorted(arrays.items())
    }
    manifest = {
        "schema_version": "laas.peat-raised-bog-condition-snapshot/1",
        "status": status,
        "recipe_sha256": recipe_sha,
        "recipe": recipe,
        "blockers": blockers,
        "development": expected_development,
        "sealed_abstention": expected_sealed,
        "arrays": {
            "path": "conditions.npz",
            "bytes": npz_path.stat().st_size,
            "sha256": _sha256(npz_path),
            "inventory": array_inventory,
        },
        "qa": {
            "path": "qa/01_conditions_and_hydrology.png",
            "bytes": qa_path.stat().st_size,
            "sha256": _sha256(qa_path),
            "interpretation": (
                "Exact development halo ownership and exclusions beside whole-mire "
                "10 m context; title explicitly marks absent corrected-C0 coverage."
            ),
        },
    }
    manifest_path_tmp = temporary / "snapshot.json"
    manifest_path_tmp.write_bytes(_canonical_bytes(manifest))
    qa_index = {
        "schema_version": "laas.qa-image-index/1",
        "recipe_sha256": recipe_sha,
        "images": [manifest["qa"]],
        "source_npz_sha256": manifest["arrays"]["sha256"],
    }
    (temporary / "qa/index.json").write_bytes(_canonical_bytes(qa_index))
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        temporary.rename(target)
    except FileExistsError:
        if not manifest_path.is_file():
            raise
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    path = materialize_snapshot(args.output_root)
    payload = json.loads(path.read_bytes())
    print(path)
    print(payload["status"])
    for blocker in payload["blockers"]:
        print(f"BLOCKER: {blocker}")
    return 0 if payload["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
