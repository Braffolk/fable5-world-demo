"""Strictly bound Development-A evidence for ALS/TGV reconstruction."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import laspy
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from scipy import ndimage
import shapely
from shapely import wkb

from .....config import ASSET_GEN_ROOT
from ...erodible_slope.morphodynamics.structural_base import (
    FINE_CANVAS_BBOX_EN,
    load_development_a_structural_base,
)


@dataclass(frozen=True)
class AlsTgvEvidence:
    bbox_en: tuple[float, float, float, float]
    pitch_m: float
    x: np.ndarray
    y: np.ndarray
    c0_m: np.ndarray
    hard_zero: np.ndarray
    active: np.ndarray
    mapped_face: np.ndarray
    imagery_masked: np.ndarray
    guide_normal_x: np.ndarray
    guide_normal_y: np.ndarray
    guide_edge_confidence: np.ndarray
    data_target_m: np.ndarray
    data_weight: np.ndarray
    train_mask: np.ndarray
    holdout_mask: np.ndarray
    point_x: np.ndarray
    point_y: np.ndarray
    point_residual_m: np.ndarray
    point_side: np.ndarray
    point_holdout: np.ndarray
    source_identities: dict[str, dict[str, Any]]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bound(row: dict[str, Any]) -> Path:
    path = ASSET_GEN_ROOT.parent / row["path"]
    if not path.is_file():
        raise ValueError(f"bound input is absent: {path}")
    if "bytes" in row and path.stat().st_size != row["bytes"]:
        raise ValueError(f"bound byte count differs: {path}")
    if _sha256_file(path) != row["sha256"]:
        raise ValueError(f"bound SHA-256 differs: {path}")
    return path


def _grid(bbox: tuple[float, float, float, float], pitch: float) -> tuple[np.ndarray, np.ndarray]:
    cols = int(round((bbox[2] - bbox[0]) / pitch)) + 1
    rows = int(round((bbox[3] - bbox[1]) / pitch)) + 1
    return (
        bbox[0] + np.arange(cols, dtype=np.float64) * pitch,
        bbox[3] - np.arange(rows, dtype=np.float64) * pitch,
    )


def _sample_domain(array: np.ndarray, bbox: tuple[float, float, float, float], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xx, yy = np.meshgrid(x, y)
    rows = bbox[3] - yy - 0.5
    cols = xx - bbox[0] - 0.5
    return ndimage.map_coordinates(array, [rows, cols], order=0, mode="nearest")


def _read_image(path: Path, bbox: tuple[float, float, float, float], shape: tuple[int, int]) -> np.ndarray:
    with rasterio.open(path) as source:
        if source.crs is None or source.crs.to_epsg() != 3301 or source.count != 3:
            raise ValueError(f"imagery contract differs: {path}")
        window = from_bounds(*bbox, transform=source.transform)
        data = source.read(
            out_shape=(3, *shape),
            window=window,
            resampling=Resampling.bilinear,
        )
    return np.moveaxis(data.astype(np.float32) / 255.0, 0, -1)


def _target_line(path: Path, target_id: int):
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in document["records"] if row["attributes"].get("etak_id") == target_id]
    if len(rows) != 1:
        raise ValueError("mapped escarpment identity is not unique")
    line = wkb.loads(bytes.fromhex(rows[0]["geometry"]["wkb_hex"]))
    if line.geom_type != "LineString" or not line.is_valid:
        raise ValueError("mapped escarpment geometry is invalid")
    return line


def _line_coordinates(line, x: np.ndarray, y: np.ndarray, c0: np.ndarray, bbox: tuple[float, float, float, float], pitch: float) -> tuple[np.ndarray, np.ndarray]:
    points = shapely.points(x, y)
    station = shapely.line_locate_point(line, points)
    before = shapely.line_interpolate_point(line, np.maximum(station - 0.5, 0.0))
    after = shapely.line_interpolate_point(line, np.minimum(station + 0.5, float(line.length)))
    tx = shapely.get_x(after) - shapely.get_x(before)
    ty = shapely.get_y(after) - shapely.get_y(before)
    scale = np.maximum(np.hypot(tx, ty), 1e-9)
    nx, ny = -ty / scale, tx / scale
    projected = shapely.line_interpolate_point(line, station)
    dx, dy = x - shapely.get_x(projected), y - shapely.get_y(projected)
    signed = dx * nx + dy * ny

    sample_station = np.linspace(0.0, float(line.length), 256)
    sample = shapely.line_interpolate_point(line, sample_station)
    sb = shapely.line_interpolate_point(line, np.maximum(sample_station - 0.5, 0.0))
    sa = shapely.line_interpolate_point(line, np.minimum(sample_station + 0.5, float(line.length)))
    stx, sty = shapely.get_x(sa) - shapely.get_x(sb), shapely.get_y(sa) - shapely.get_y(sb)
    ss = np.maximum(np.hypot(stx, sty), 1e-9)
    snx, sny = -sty / ss, stx / ss

    def height_at(px: np.ndarray, py: np.ndarray) -> np.ndarray:
        rr = (bbox[3] - py) / pitch
        cc = (px - bbox[0]) / pitch
        return ndimage.map_coordinates(c0, [rr, cc], order=1, mode="nearest")

    sx, sy = shapely.get_x(sample), shapely.get_y(sample)
    if float(np.median(height_at(sx + 8.0 * snx, sy + 8.0 * sny) - height_at(sx - 8.0 * snx, sy - 8.0 * sny))) < 0.0:
        signed = -signed
    distance = shapely.distance(points, line)
    return np.asarray(signed), np.asarray(distance)


def _aggregate_points(rows: np.ndarray, cols: np.ndarray, values: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    flat = rows * shape[1] + cols
    order = np.argsort(flat)
    flat, values = flat[order], values[order]
    unique, starts, counts = np.unique(flat, return_index=True, return_counts=True)
    target = np.zeros(shape, dtype=np.float32)
    weight = np.zeros(shape, dtype=np.float32)
    target.flat[unique] = np.asarray(
        [np.median(values[start : start + count]) for start, count in zip(starts, counts)],
        dtype=np.float32,
    )
    weight.flat[unique] = np.sqrt(np.minimum(counts, 4)).astype(np.float32)
    return target, weight


def load_evidence(config: dict[str, Any]) -> AlsTgvEvidence:
    inputs = config["inputs"]
    paths = {name: _bound(row) for name, row in inputs.items()}
    domain_manifest = json.loads(paths["domain_closure"].read_text(encoding="utf-8"))
    if not (
        domain_manifest.get("state") == "closure_pass"
        and domain_manifest.get("solver_use_authorized") is True
        and domain_manifest.get("canonical_role") == "solver_authorized_condition_authority"
    ):
        raise ValueError("domain closure is not solver-authorized")
    domain_bbox = tuple(float(v) for v in domain_manifest["recipe"]["canonical_bbox_en"])
    domain = np.load(paths["domain_npz"], allow_pickle=False)
    mapped_count = int(np.count_nonzero(domain["target_feature"]))
    mapped_nonheight_count = int(np.count_nonzero(domain["target_feature"] & domain["non_heightfield"]))
    expected = int(config["hard_laws"]["mapped_face_cells_expected"])
    if (mapped_count, mapped_nonheight_count) != (expected, expected):
        raise ValueError("mapped face/non-heightfield authority differs")

    bbox = tuple(float(v) for v in config["canvas_bbox_en"])
    if bbox != FINE_CANVAS_BBOX_EN:
        raise ValueError("solve canvas differs from accepted structural canvas")
    pitch = float(config["solve_pitch_m"])
    x, y = _grid(bbox, pitch)
    base = load_development_a_structural_base()
    stride = int(round(pitch / base.texel_m))
    c0 = np.asarray(base.c0_height_m[::stride, ::stride], dtype=np.float32)
    if c0.shape != (len(y), len(x)):
        raise ValueError("C0 solve lattice shape differs")

    masks = {
        name: _sample_domain(domain[name], domain_bbox, x, y) >= 0.5
        for name in (
            "valid", "water", "object", "non_heightfield", "protected_structure",
            "unknown", "target_feature", "vegetation_evidence",
        )
    }
    material = (
        (domain["soil_feature_index"] >= 0)
        & (domain["geology_lithology_code"] > 0)
        & (domain["geology_genesis_code"] > 0)
    )
    material = _sample_domain(material, domain_bbox, x, y) >= 0.5
    structural_hard = base.forbidden_morphology[::stride, ::stride] | base.unknown_bathymetry[::stride, ::stride]
    hard = (
        ~masks["valid"] | masks["water"] | masks["object"] | masks["non_heightfield"]
        | masks["protected_structure"] | masks["unknown"] | structural_hard
    )
    collar = int(round(float(config["outer_zero_collar_m"]) / pitch))
    hard[:collar] = True
    hard[-collar:] = True
    hard[:, :collar] = True
    hard[:, -collar:] = True

    rgb = _read_image(paths["rgb"], bbox, c0.shape)
    cir = _read_image(paths["cir"], bbox, c0.shape)
    luma = rgb @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    nir, red = cir[..., 0], cir[..., 1]
    ndvi = (nir - red) / np.maximum(nir + red, 1e-3)
    guide_config = config["imagery_guidance"]
    imagery_masked = (
        masks["vegetation_evidence"]
        | (ndvi >= float(guide_config["ndvi_vegetation_floor"]))
        | (luma <= float(guide_config["rgb_shadow_luma_ceiling"]))
    )
    luma = ndimage.gaussian_filter(luma, 1.0)
    nir = ndimage.gaussian_filter(nir, 1.0)
    gy_l, gx_l = np.gradient(luma, pitch)
    gy_n, gx_n = np.gradient(nir, pitch)
    use_nir = gx_n * gx_n + gy_n * gy_n > gx_l * gx_l + gy_l * gy_l
    gx, gy = np.where(use_nir, gx_n, gx_l), np.where(use_nir, gy_n, gy_l)
    jxx = ndimage.gaussian_filter(gx * gx, 2.0)
    jyy = ndimage.gaussian_filter(gy * gy, 2.0)
    jxy = ndimage.gaussian_filter(gx * gy, 2.0)
    angle = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    guide_nx, guide_ny = np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)
    edge = np.sqrt(np.maximum(jxx + jyy, 0.0))
    scale = float(np.percentile(edge[~imagery_masked], float(guide_config["edge_percentile"])))
    edge_confidence = np.clip(edge / max(scale, 1e-6), 0.0, 1.0).astype(np.float32)
    edge_confidence[imagery_masked] = 0.0

    line = _target_line(paths["etak"], int(config["target_etak_id"]))
    xx, yy = np.meshgrid(x, y)
    _, grid_distance = _line_coordinates(line, xx.ravel(), yy.ravel(), c0, bbox, pitch)
    adjacent = grid_distance.reshape(c0.shape) <= 48.0
    active = ~hard & material & adjacent

    points = laspy.read(paths["als"])
    point_mask = (
        (points.classification == int(config["qualification"]["als_class"]))
        & ~np.asarray(points.withheld, dtype=bool)
        & ~np.asarray(points.synthetic, dtype=bool)
        & (points.x >= bbox[0]) & (points.x <= bbox[2])
        & (points.y >= bbox[1]) & (points.y <= bbox[3])
    )
    px = np.asarray(points.x[point_mask], dtype=np.float64)
    py = np.asarray(points.y[point_mask], dtype=np.float64)
    pz = np.asarray(points.z[point_mask], dtype=np.float64)
    rr_float, cc_float = (bbox[3] - py) / pitch, (px - bbox[0]) / pitch
    point_c0 = ndimage.map_coordinates(c0, [rr_float, cc_float], order=1, mode="nearest")
    residual = pz - point_c0
    rr = np.clip(np.rint(rr_float).astype(np.int64), 0, c0.shape[0] - 1)
    cc = np.clip(np.rint(cc_float).astype(np.int64), 0, c0.shape[1] - 1)
    _, point_distance = _line_coordinates(line, px, py, c0, bbox, pitch)
    point_side, _ = _line_coordinates(line, px, py, c0, bbox, pitch)
    qualified = (
        active[rr, cc]
        & (point_distance <= 48.0)
        & (np.abs(residual) <= float(config["qualification"]["maximum_absolute_raw_residual_m"]))
    )
    px, py, residual, rr, cc, point_side = (
        value[qualified] for value in (px, py, residual, rr, cc, point_side)
    )
    blocks = (
        np.floor((px - bbox[0]) / float(config["holdout_block_m"])).astype(np.int64)
        + np.floor((py - bbox[1]) / float(config["holdout_block_m"])).astype(np.int64)
    )
    point_holdout = blocks % int(config["holdout_modulus"]) == int(config["holdout_remainder"])
    train_target, train_weight = _aggregate_points(rr[~point_holdout], cc[~point_holdout], residual[~point_holdout], c0.shape)
    hold_target, hold_weight = _aggregate_points(rr[point_holdout], cc[point_holdout], residual[point_holdout], c0.shape)
    train_weight[~active] = 0.0
    hold_weight[~active] = 0.0

    identities = {
        name: {"path": row["path"], "bytes": paths[name].stat().st_size, "sha256": row["sha256"]}
        for name, row in inputs.items()
    }
    return AlsTgvEvidence(
        bbox_en=bbox,
        pitch_m=pitch,
        x=x,
        y=y,
        c0_m=c0,
        hard_zero=hard,
        active=active,
        mapped_face=masks["target_feature"],
        imagery_masked=imagery_masked,
        guide_normal_x=guide_nx,
        guide_normal_y=guide_ny,
        guide_edge_confidence=edge_confidence,
        data_target_m=train_target,
        data_weight=train_weight,
        train_mask=train_weight > 0.0,
        holdout_mask=hold_weight > 0.0,
        point_x=px,
        point_y=py,
        point_residual_m=residual,
        point_side=np.where(point_side >= 0.0, 1, -1).astype(np.int8),
        point_holdout=point_holdout,
        source_identities=identities,
    )
