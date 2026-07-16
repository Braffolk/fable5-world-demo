"""Materialize the measured-only Valgesoo raised-bog FLOAT reconnaissance."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyogrio.raw
import rasterio
import rasterio.features
import rasterio.transform
import shapely
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .....config import ASSET_GEN_ROOT, DATA_WORK
from ....repair.prolong import prolong_structural_4x
from ..conditions import select_sites as selection


CONFIG = Path(__file__).with_name("config.json")
SCHEMA = "laas.peat-raised-bog-measured-float-config/1"
OUTPUT_ROOT = (
    DATA_WORK
    / "microtopography/peat-raised-bog/measured-float/sha256"
)
REPOSITORY_ROOT = ASSET_GEN_ROOT.parent
MIRE_LAYER = "E_306_margala_a"
SOIL = REPOSITORY_ROOT / "asset-gen/data/in/soil/mullakaart/Mullakaart.shp"
GEOLOGY = (
    REPOSITORY_ROOT
    / "docs/deep-research/microtopography-generation/library/data/egt/"
    "pinnakate-200k/q_avamus_a_200t.shp"
)

MASK_LAYERS: dict[str, tuple[str, tuple[str, ...]]] = {
    "forest": ("E_305_puittaimestik_a", ("vajalik_t",)),
    "sea": ("E_201_meri_a", ("vajalik_t",)),
    "lake": ("E_202_seisuveekogu_a", ("vajalik_t",)),
    "flowing_water": ("E_203_vooluveekogu_a", ("vajalik_t",)),
    "water_line": ("E_203_vooluveekogu_j", ("tyyp", "laius", "vajalik_t")),
    "cut_peat": ("E_307_turbavali_a", ("vajalik_t",)),
    "road_area": ("E_501_tee_a", ("vajalik_t",)),
    "road_line": ("E_501_tee_j", ("laius", "vajalik_t")),
    "building": ("E_401_hoone_ka", ("vajalik_t",)),
    "high_object": ("E_402_korgrajatis_p", ("vajalik_t",)),
    "other_object_area": ("E_403_muu_rajatis_ka", ("vajalik_t",)),
    "other_object_point": ("E_403_muu_rajatis_p", ("vajalik_t",)),
}

PAPER = (240, 237, 224)
INK = (31, 38, 32)
QA_SIZE = (1800, 1200)


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(REPOSITORY_ROOT.resolve())),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _verify_sources(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sources = config["sources"]
    declarations = [sources["dtm"], sources["etak"], sources["geology_domains"]]
    declarations.extend(sources["soil"])
    declarations.extend(sources["geology"])
    for declared in declarations:
        path = REPOSITORY_ROOT / declared["path"]
        actual = _identity(path)
        if actual["sha256"] != declared["sha256"]:
            raise ValueError(f"source hash changed: {path}: {actual['sha256']}")
        rows.append(actual)
    return rows


def _read_layer(
    path: Path,
    *,
    layer: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    columns: tuple[str, ...] = (),
    where: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    metadata, fids, wkbs, values = pyogrio.raw.read(
        path,
        layer=layer,
        bbox=bbox,
        columns=list(columns),
        where=where,
        return_fids=True,
    )
    if str(metadata.get("crs")) != "EPSG:3301" or fids is None or wkbs is None:
        raise ValueError(f"invalid EPSG:3301 vector source: {path}, layer={layer}")
    geometries = np.asarray(
        [shapely.force_2d(shapely.from_wkb(bytes(value))) for value in wkbs],
        dtype=object,
    )
    return (
        np.asarray(fids),
        geometries,
        dict(zip(metadata.get("fields", ()), values, strict=True)),
    )


def _component(config: dict[str, Any]) -> shapely.Geometry:
    target = config["target"]
    fids, geometries, attributes = _read_layer(
        REPOSITORY_ROOT / config["sources"]["etak"]["path"],
        layer=MIRE_LAYER,
        where="tyyp = 20 AND puis = 20",
        columns=(
            "etak_id",
            "tyyp_t",
            "puis_t",
            "vajalik_t",
            "muutmisaeg",
            "geom_muutmisaeg",
        ),
    )
    matches = np.flatnonzero(attributes["etak_id"] == target["minimum_etak_id"])
    if len(matches) != 1:
        raise ValueError("target ETAK member is missing or duplicated")
    target_index = int(matches[0])
    component = next(
        members
        for members in selection._connected_components(geometries)
        if target_index in members
    )
    member_ids = sorted(int(attributes["etak_id"][index]) for index in component)
    member_fids = sorted(int(fids[index]) for index in component)
    geometry = selection._union(geometries[index] for index in component)
    if member_ids != target["member_etak_ids"] or member_fids != target["source_fids"]:
        raise ValueError("connected-mire membership changed")
    expected_bounds = np.asarray(target["component_bounds_epsg3301_m"], dtype=np.float64)
    if not np.allclose(geometry.bounds, expected_bounds, rtol=0.0, atol=1.0e-7):
        raise ValueError(f"connected-mire bounds changed: {geometry.bounds}")
    if hashlib.sha256(geometry.wkb).hexdigest() != target["component_geometry_sha256"]:
        raise ValueError("connected-mire geometry changed")
    return geometry


def _rasterize(
    geometries: Iterable[shapely.Geometry],
    shape: tuple[int, int],
    transform: rasterio.Affine,
) -> np.ndarray:
    rows = [(geometry, 1) for geometry in geometries if not geometry.is_empty]
    if not rows:
        return np.zeros(shape, dtype=bool)
    return rasterio.features.rasterize(
        rows,
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)


def _buffer_lines(
    geometries: np.ndarray,
    widths: np.ndarray | None,
    minimum_half_width_m: float,
) -> list[shapely.Geometry]:
    result: list[shapely.Geometry] = []
    for index, geometry in enumerate(geometries):
        width = None if widths is None else widths[index]
        try:
            half_width = max(minimum_half_width_m, 0.5 * float(width))
        except (TypeError, ValueError):
            half_width = minimum_half_width_m
        result.append(geometry.buffer(half_width, cap_style="flat"))
    return result


def _condition_masks(
    config: dict[str, Any],
    geometry: shapely.Geometry,
    bbox: tuple[int, int, int, int],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    shape = (bbox[3] - bbox[1], bbox[2] - bbox[0])
    transform = rasterio.transform.from_origin(bbox[0], bbox[3], 1.0, 1.0)
    etak = REPOSITORY_ROOT / config["sources"]["etak"]["path"]
    layers: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    dynamic: list[shapely.Geometry] = []
    counts: dict[str, int] = {}
    for name, (layer, columns) in MASK_LAYERS.items():
        _, geometries, attributes = _read_layer(
            etak, layer=layer, bbox=bbox, columns=columns
        )
        layers[name] = (geometries, attributes)
        counts[name] = len(geometries)
        statuses = attributes.get("vajalik_t", np.asarray([], dtype=object))
        dynamic.extend(
            item
            for item, status in zip(geometries, statuses, strict=True)
            if status is not None and str(status) != "Korras"
        )

    water_lines, water_attributes = layers["water_line"]
    roads = [
        *layers["road_area"][0],
        *_buffer_lines(layers["road_line"][0], layers["road_line"][1].get("laius"), 1.5),
    ]
    water = [
        *layers["sea"][0],
        *layers["lake"][0],
        *layers["flowing_water"][0],
        *_buffer_lines(water_lines, water_attributes.get("laius"), 1.0),
    ]
    objects = [
        *layers["high_object"][0],
        *layers["other_object_area"][0],
        *layers["other_object_point"][0],
    ]
    masks = {
        "mire_coverage": _rasterize([geometry], shape, transform),
        "forest": _rasterize(layers["forest"][0], shape, transform),
        "water": _rasterize(water, shape, transform),
        "cut_peat": _rasterize(layers["cut_peat"][0], shape, transform),
        "road": _rasterize(roads, shape, transform),
        "building": _rasterize(layers["building"][0], shape, transform),
        "object": _rasterize(objects, shape, transform),
        "dynamic_unknown": _rasterize(dynamic, shape, transform),
    }

    soil_fids, soil_geometries, soil_attributes = _read_layer(
        SOIL,
        bbox=bbox,
        columns=(
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
        ),
    )
    soil_coverage = np.zeros(shape, dtype=bool)
    soil_supported = np.zeros(shape, dtype=bool)
    soil_records: list[dict[str, Any]] = []
    for index, soil_geometry in enumerate(soil_geometries):
        feature_mask = _rasterize([soil_geometry], shape, transform)
        support = selection._soil_support(soil_attributes, index)
        soil_coverage |= feature_mask
        if support["supported"]:
            soil_supported |= feature_mask
        soil_records.append(
            {
                "source_fid": int(soil_fids[index]),
                "supported": bool(support["supported"]),
                "siffer": support["siffer"],
                "loimis1": support["loimis1"],
                "humus_semantics": support["humus_semantics"],
            }
        )
    masks["soil_unknown"] = ~soil_coverage | ~soil_supported

    domains_path = REPOSITORY_ROOT / config["sources"]["geology_domains"]["path"]
    domain_payload = json.loads(domains_path.read_bytes())
    decoded = {
        row["field_name"]: {int(item["code"]) for item in row["coded_values"]}
        for row in domain_payload["domains"]
    }
    geology_fids, geology_geometries, geology_attributes = _read_layer(
        GEOLOGY, bbox=bbox, columns=("lito200", "genees200")
    )
    geology_coverage = np.zeros(shape, dtype=bool)
    geology_known = np.zeros(shape, dtype=bool)
    geology_records: list[dict[str, Any]] = []
    for index, geology_geometry in enumerate(geology_geometries):
        feature_mask = _rasterize([geology_geometry], shape, transform)
        lithology = int(geology_attributes["lito200"][index])
        genesis = int(geology_attributes["genees200"][index])
        known = (
            lithology in decoded["lito200"]
            and genesis in decoded["genees200"]
            and lithology not in {997, 998}
        )
        geology_coverage |= feature_mask
        if known:
            geology_known |= feature_mask
        geology_records.append(
            {
                "source_fid": int(geology_fids[index]),
                "lithology": lithology,
                "genesis": genesis,
                "known": known,
            }
        )
    masks["geology_unknown"] = ~geology_coverage | ~geology_known
    masks["hard_exclusion"] = np.logical_or.reduce(
        [masks[name] for name in ("forest", "water", "cut_peat", "road", "building", "object")]
    )
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
    records = {
        "etak_feature_counts": counts,
        "soil_records": soil_records,
        "geology_records": geology_records,
    }
    return masks, records


def _read_height(
    config: dict[str, Any], bbox: tuple[int, int, int, int]
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    support = int(config["method"]["real_parent_support_m"])
    support_bbox = (
        bbox[0] - support,
        bbox[1] - support,
        bbox[2] + support,
        bbox[3] + support,
    )
    path = REPOSITORY_ROOT / config["sources"]["dtm"]["path"]
    with rasterio.open(path) as source:
        if (
            str(source.crs) != "EPSG:3301"
            or source.transform.a != 1.0
            or source.transform.e != -1.0
            or source.nodata is None
        ):
            raise ValueError("local DTM grid contract changed")
        window = rasterio.windows.from_bounds(*support_bbox, source.transform)
        height = source.read(1, window=window, out_dtype="float64")
        invalid = ~np.isfinite(height) | (height == float(source.nodata))
        if invalid.any():
            raise ValueError("real two-metre parent support contains invalid DTM samples")
    expected_shape = (
        support_bbox[3] - support_bbox[1],
        support_bbox[2] - support_bbox[0],
    )
    if height.shape != expected_shape:
        raise ValueError(f"local DTM support shape changed: {height.shape}")
    return height, support_bbox


def _slices(
    outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]
) -> tuple[slice, slice]:
    if not shapely.box(*outer).covers(shapely.box(*inner)):
        raise ValueError(f"crop {inner} is outside solve bbox {outer}")
    return (
        slice(outer[3] - inner[3], outer[3] - inner[1]),
        slice(inner[0] - outer[0], inner[2] - outer[0]),
    )


def _plane_residual(height: np.ndarray) -> np.ndarray:
    rows, cols = np.indices(height.shape, dtype=np.float64)
    design = np.stack((cols.ravel(), rows.ravel(), np.ones(height.size)), axis=1)
    coefficients, *_ = np.linalg.lstsq(design, height.ravel(), rcond=None)
    return height - (coefficients[0] * cols + coefficients[1] * rows + coefficients[2])


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _hillshade(
    height: np.ndarray,
    pitch_m: float,
    elevation_limits: tuple[float, float],
) -> np.ndarray:
    gy, gx = np.gradient(height.astype(np.float64), pitch_m)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    azimuth = np.deg2rad(315.0)
    altitude = np.deg2rad(38.0)
    light = (
        np.sin(altitude) * np.cos(slope)
        + np.cos(altitude) * np.sin(slope) * np.cos(azimuth - aspect)
    )
    light = np.clip((light + 0.15) / 1.15, 0.0, 1.0)
    low, high = elevation_limits
    elevation = np.clip((height - low) / max(high - low, 1.0e-9), 0.0, 1.0)
    base = np.stack(
        (0.37 + 0.27 * elevation, 0.43 + 0.31 * elevation, 0.27 + 0.19 * elevation),
        axis=-1,
    )
    return np.clip(base * (0.48 + 0.70 * light[..., None]), 0.0, 1.0)


def _signed(values: np.ndarray, limit: float | None = None) -> np.ndarray:
    if limit is None:
        limit = max(float(np.percentile(np.abs(values), 98.0)), 1.0e-9)
    scaled = np.clip(values / limit, -1.0, 1.0)
    neutral = np.asarray([0.94, 0.92, 0.84])
    positive = np.asarray([0.73, 0.16, 0.09])
    negative = np.asarray([0.06, 0.31, 0.61])
    return np.where(
        (scaled >= 0)[..., None],
        neutral + scaled[..., None] * (positive - neutral),
        neutral + (-scaled)[..., None] * (negative - neutral),
    )


def _mask_rgb(masks: dict[str, np.ndarray]) -> np.ndarray:
    rgb = np.zeros((*masks["authority"].shape, 3), dtype=np.float64)
    rgb[:] = (0.12, 0.14, 0.12)
    rgb[masks["mire_coverage"]] = (0.36, 0.51, 0.27)
    rgb[masks["authority"]] = (0.68, 0.77, 0.35)
    rgb[masks["ownership_unknown"]] = (0.88, 0.57, 0.16)
    rgb[masks["hard_exclusion"]] = (0.70, 0.18, 0.13)
    rgb[masks["forest"]] = (0.12, 0.33, 0.16)
    rgb[masks["water"]] = (0.12, 0.42, 0.72)
    rgb[masks["road"]] = (0.66, 0.58, 0.50)
    return rgb


def _panel_image(rgb: np.ndarray, size: tuple[int, int]) -> Image.Image:
    image = Image.fromarray(
        np.asarray(np.clip(rgb, 0.0, 1.0) * 255.0, dtype=np.uint8), "RGB"
    )
    scale = min(size[0] / image.width, size[1] / image.height)
    image = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", size, PAPER)
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def _sheet(
    title: str,
    note: str,
    panels: list[tuple[str, np.ndarray]],
) -> Image.Image:
    image = Image.new("RGB", QA_SIZE, PAPER)
    draw = ImageDraw.Draw(image)
    draw.text((36, 24), title, font=_font(27), fill=INK)
    draw.text((36, 62), note, font=_font(16), fill=(70, 76, 67))
    top = 102
    panel_w = (QA_SIZE[0] - 72) // 2
    panel_h = (QA_SIZE[1] - top - 34) // 2
    for index, (label, rgb) in enumerate(panels):
        row, col = divmod(index, 2)
        x = 36 + col * panel_w
        y = top + row * panel_h
        draw.text((x + 8, y + 5), label, font=_font(17), fill=INK)
        rendered = _panel_image(rgb, (panel_w - 16, panel_h - 40))
        image.paste(rendered, (x + 8, y + 34))
    return image


def _write_qa(
    output: Path,
    masks: dict[str, np.ndarray],
    whole_height: np.ndarray,
    halo_height: np.ndarray,
    halo_fine: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    whole_limits = tuple(np.percentile(whole_height[masks["mire_coverage"]], [2, 98]))
    halo_limits = tuple(np.percentile(halo_height, [2, 98]))
    residual = _plane_residual(halo_height)
    macro = ndimage.gaussian_filter(residual, 16.0)
    meso = ndimage.gaussian_filter(halo_height, 2.0) - ndimage.gaussian_filter(
        halo_height, 16.0
    )
    whole_shade = _hillshade(whole_height, 1.0, whole_limits)
    whole_shade[~masks["mire_coverage"]] *= 0.28
    image_1 = _sheet(
        "01 / Valgesoo conditions and measured 1 m relief",
        "Complete connected mire solved; red = hard, orange = unknown, blue = water, dark green = forest.",
        [
            ("whole connected mire / conditions", _mask_rgb(masks)),
            ("measured 1 m DTM / whole mire", whole_shade),
            ("measured >16 m macro after plane removal / signed", _signed(macro)),
            ("measured 2-16 m band / signed", _signed(meso)),
        ],
    )
    image_1.save(output / "01-condition-mask-and-measured-1m-relief.png", optimize=True)

    parent_blocks = np.repeat(np.repeat(halo_height, 4, axis=0), 4, axis=1)
    signed_difference = halo_fine - parent_blocks
    center = halo_fine.shape[0] // 2
    radius = 64
    fine_close = halo_fine[center - radius : center + radius, center - radius : center + radius]
    image_2 = _sheet(
        "02 / Measured 1 m vs structural 0.25 m reconstruction",
        "Same light and elevation scale. Difference is interpolation around measured parent means, not decimetre truth.",
        [
            ("measured 1 m / 256 m halo", _hillshade(halo_height, 1.0, halo_limits)),
            ("structural 0.25 m / 256 m halo", _hillshade(halo_fine, 0.25, halo_limits)),
            (
                f"signed 0.25 m - parent / p98 {metrics['interpolation']['signed_parent_difference_abs_p98_m']:.3f} m",
                _signed(signed_difference),
            ),
            ("32 m center at 0.25 m / measured structure only", _hillshade(fine_close, 0.25, halo_limits)),
        ],
    )
    image_2.save(output / "02-common-light-1m-vs-025m-and-signed-difference.png", optimize=True)


def run(config_path: Path = CONFIG, output_root: Path = OUTPUT_ROOT) -> Path:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_bytes())
    if (
        config.get("schema_version") != SCHEMA
        or config.get("status") != "development_float_qa_only"
        or config.get("method", {}).get("synthesis") is not False
    ):
        raise ValueError("unsupported measured-FLOAT config")
    source_identities = _verify_sources(config)
    geometry = _component(config)
    target = config["target"]
    whole_bbox = tuple(
        int(value)
        for value in (
            math.floor(geometry.bounds[0]),
            math.floor(geometry.bounds[1]),
            math.ceil(geometry.bounds[2]),
            math.ceil(geometry.bounds[3]),
        )
    )
    halo_bbox = tuple(int(value) for value in target["clear_halo_bounds_epsg3301_m"])
    core_bbox = tuple(int(value) for value in target["core_bounds_epsg3301_m"])
    masks, condition_records = _condition_masks(config, geometry, whole_bbox)
    support_height, support_bbox = _read_height(config, whole_bbox)
    whole_height = support_height[2:-2, 2:-2]
    fine_whole = prolong_structural_4x(
        support_height,
        parent_rows=(2, support_height.shape[0] - 2),
        parent_cols=(2, support_height.shape[1] - 2),
    )
    blocks = fine_whole.reshape(
        whole_height.shape[0], 4, whole_height.shape[1], 4
    )
    reconstructed_means = np.mean(blocks, axis=(1, 3), dtype=np.float64)
    closure = reconstructed_means - whole_height

    halo_rows, halo_cols = _slices(whole_bbox, halo_bbox)
    core_rows, core_cols = _slices(whole_bbox, core_bbox)
    halo_height = whole_height[halo_rows, halo_cols]
    core_height = whole_height[core_rows, core_cols]
    halo_fine = fine_whole[
        slice(halo_rows.start * 4, halo_rows.stop * 4),
        slice(halo_cols.start * 4, halo_cols.stop * 4),
    ]
    core_fine = fine_whole[
        slice(core_rows.start * 4, core_rows.stop * 4),
        slice(core_cols.start * 4, core_cols.stop * 4),
    ]
    halo_masks = {name: values[halo_rows, halo_cols] for name, values in masks.items()}
    if not np.all(halo_masks["authority"]):
        raise ValueError("frozen clear halo now intersects a hard or unknown exclusion")

    halo_residual = _plane_residual(halo_height)
    core_residual = _plane_residual(core_height)
    macro = ndimage.gaussian_filter(halo_residual, 16.0)
    meso = ndimage.gaussian_filter(halo_height, 2.0) - ndimage.gaussian_filter(
        halo_height, 16.0
    )
    signed_parent_difference = halo_fine - np.repeat(
        np.repeat(halo_height, 4, axis=0), 4, axis=1
    )
    metrics = {
        "target": target,
        "surface": {
            "whole_bbox_epsg3301_m": list(whole_bbox),
            "support_bbox_epsg3301_m": list(support_bbox),
            "whole_parent_shape": list(whole_height.shape),
            "whole_fine_shape": list(fine_whole.shape),
            "halo_parent_shape": list(halo_height.shape),
            "halo_fine_shape": list(halo_fine.shape),
            "core_parent_shape": list(core_height.shape),
            "core_fine_shape": list(core_fine.shape),
        },
        "conditions": {
            "whole_mire_cells": int(np.count_nonzero(masks["mire_coverage"])),
            "authority_cells": int(np.count_nonzero(masks["authority"])),
            "hard_exclusion_cells": int(np.count_nonzero(masks["hard_exclusion"])),
            "ownership_unknown_cells": int(np.count_nonzero(masks["ownership_unknown"])),
            "halo_all_authority": bool(np.all(halo_masks["authority"])),
            "records": condition_records,
        },
        "parent_mean_closure": {
            "all_parent_cells_preserved": bool(np.array_equal(reconstructed_means, whole_height)),
            "maximum_absolute_error_m": float(np.max(np.abs(closure))),
            "rms_error_m": float(np.sqrt(np.mean(closure**2))),
        },
        "measured_relief": {
            "halo_raw_range_m": float(np.ptp(halo_height)),
            "halo_plane_residual_range_m": float(np.ptp(halo_residual)),
            "halo_plane_residual_abs_p95_m": float(np.percentile(np.abs(halo_residual), 95.0)),
            "halo_measured_over_16m_macro_range_m": float(np.ptp(macro)),
            "halo_measured_over_16m_macro_abs_p95_m": float(np.percentile(np.abs(macro), 95.0)),
            "halo_measured_2_to_16m_band_abs_p95_m": float(np.percentile(np.abs(meso), 95.0)),
            "halo_measured_2_to_16m_band_abs_p99_m": float(np.percentile(np.abs(meso), 99.0)),
            "core_raw_range_m": float(np.ptp(core_height)),
            "core_plane_residual_range_m": float(np.ptp(core_residual)),
            "core_plane_residual_abs_p95_m": float(np.percentile(np.abs(core_residual), 95.0)),
        },
        "interpolation": {
            "signed_parent_difference_min_m": float(np.min(signed_parent_difference)),
            "signed_parent_difference_max_m": float(np.max(signed_parent_difference)),
            "signed_parent_difference_abs_p98_m": float(np.percentile(np.abs(signed_parent_difference), 98.0)),
            "adds_synthesized_detail": False,
            "decimetre_truth_claim": False,
        },
        "authority": {
            "measured_dtm_is_only_height_source": True,
            "apparent_forms_are_measured_not_synthesized": True,
            "packing_runtime_shader_material_or_format_authority": False,
        },
    }

    implementation = [
        _identity(Path(__file__)),
        _identity(Path(__file__).parents[3] / "repair/prolong.py"),
        _identity(Path(selection.__file__)),
    ]
    recipe = {
        "config": _identity(config_path),
        "implementation": implementation,
        "sources": source_identities,
    }
    recipe_sha256 = hashlib.sha256(_canonical(recipe)).hexdigest()
    output = output_root / recipe_sha256
    if output.exists():
        raise FileExistsError(f"immutable measured-FLOAT artifact exists: {output}")

    metadata = {
        "schema_version": "laas.peat-raised-bog-measured-float-artifact/1",
        "build_id": recipe_sha256,
        "status": "float_qa_pending_visual_verdict",
        "recipe": recipe,
        "metrics": metrics,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rasterio": rasterio.__version__,
            "pid": os.getpid(),
        },
    }
    with tempfile.TemporaryDirectory(prefix="laas-bog-measured-float-") as temporary:
        staging = Path(temporary) / "artifact"
        staging.mkdir()
        np.savez_compressed(
            staging / "measured-float.npz",
            metadata_json_u8=np.frombuffer(_canonical(metadata), dtype=np.uint8),
            whole_measured_height_1m=whole_height.astype(np.float32),
            whole_mire_coverage_1m=masks["mire_coverage"],
            whole_authority_1m=masks["authority"],
            whole_hard_exclusion_1m=masks["hard_exclusion"],
            whole_ownership_unknown_1m=masks["ownership_unknown"],
            whole_forest_1m=masks["forest"],
            whole_water_1m=masks["water"],
            whole_road_1m=masks["road"],
            whole_building_1m=masks["building"],
            whole_object_1m=masks["object"],
            halo_measured_height_1m=halo_height.astype(np.float32),
            halo_structural_height_025m=halo_fine,
            halo_authority_1m=halo_masks["authority"],
            core_measured_height_1m=core_height.astype(np.float32),
            core_structural_height_025m=core_fine,
        )
        _write_qa(staging / "qa", masks, whole_height, halo_height, halo_fine, metrics)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staging, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    output = run(args.config, args.output_root)
    with np.load(output / "measured-float.npz", allow_pickle=False) as artifact:
        metadata = json.loads(artifact["metadata_json_u8"].tobytes())
    print(output)
    print(json.dumps(metadata["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
