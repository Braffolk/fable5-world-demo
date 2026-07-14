"""Materialize dated orthophoto and ETAK authority evidence for Taevaskoda QA.

This module is deliberately outside terrain reconstruction.  It renders immutable,
georeferenced observations and authority geometry for human review; it does not
derive height, alter shoreline geometry, or authorize morphology synthesis.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pyogrio
import rasterio
import shapely
from PIL import Image, ImageDraw
from rasterio.windows import Window
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from ..config import ASSET_GEN_ROOT, CONFIG_DIR, DATA_IN, DATA_WORK, load_base
from ..grid import ChunkId, chunk_bounds_en


_RECIPE = "taevaskoda-orthophoto-etak-review-domain/1.0.0"
_SELECTION_PATH = DATA_IN / "orthophoto" / "stage1-54472" / "retained.json"
_AUTHORITY_CONFIG_PATH = (
    CONFIG_DIR / "terrain-repair" / "taevaskoda-ahja-authority-stage1.json"
)
_STAGE_CONFIG_PATH = CONFIG_DIR / "terrain-repair" / "taevaskoda-ahja-stage1.json"
_SHORELINE_LAYER = "E_204_kaldajoon_j"
_EXPECTED_PRODUCTS = {
    "rgb": {"captureDate": "2025-07-18", "pixelM": 0.2},
    "cir": {"captureDate": "2024-05-22", "pixelM": 0.25},
}
_ORTHOPHOTO_ROOT = DATA_IN / "orthophoto"
_OUTPUT_PARENT = DATA_WORK / "orthophoto-evidence"


@dataclass(frozen=True)
class _RasterSource:
    product: str
    capture_date: str
    pixel_m: float
    archive_path: Path
    archive_sha256: str
    archive_bytes: int
    product_retention_path: Path
    product_retention_sha256: str
    extraction_manifest_path: Path
    extraction_manifest_sha256: str
    tif_path: Path
    tif_sha256: str
    tif_bytes: int
    geotiff: Mapping[str, Any]


@dataclass(frozen=True)
class _Crop:
    pixels: np.ndarray
    available: np.ndarray
    transform: tuple[float, float, float, float, float, float]
    source_intersection_bounds_en: tuple[float, float, float, float]


@dataclass(frozen=True)
class _AuthorityGeometry:
    mapped_water: BaseGeometry
    qualified_water: BaseGeometry
    qualified_centerline: BaseGeometry
    mapped_area_feature: BaseGeometry
    mapped_centerline_feature: BaseGeometry
    shorelines: tuple[BaseGeometry, ...]
    shoreline_features: tuple[Mapping[str, Any], ...]
    identity: Mapping[str, Any]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(document: object) -> bytes:
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _pretty_json(document: object) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()


def _read_object(path: Path, label: str) -> Mapping[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _asset_path(relative: str) -> Path:
    path = (ASSET_GEN_ROOT / relative).resolve()
    if not path.is_relative_to(ASSET_GEN_ROOT):
        raise ValueError(f"configured path escapes asset-gen root: {relative}")
    return path


def _relative_path(root: Path, relative: str, label: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"{label} path escapes retained root: {relative}")
    return path


def _verify_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int | None,
    label: str,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise ValueError(
            f"{label} byte length differs: expected {expected_bytes}, "
            f"got {path.stat().st_size}"
        )
    digest = _sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(f"{label} SHA-256 differs: expected {expected_sha256}, got {digest}")


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.hex()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _geometry_bytes(geometry: BaseGeometry) -> bytes:
    return shapely.to_wkb(
        shapely.force_2d(geometry),
        byte_order=1,
        output_dimension=2,
        include_srid=False,
    )


def _valid_geometry(geometry: BaseGeometry, label: str) -> BaseGeometry:
    geometry = shapely.force_2d(geometry)
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError(f"{label} geometry is empty or invalid")
    return geometry


def _load_raster_sources(selection: Mapping[str, Any]) -> dict[str, _RasterSource]:
    if (
        selection.get("schemaVersion")
        != "taevaskoda-stage1-orthophoto-selection/1.0.0"
        or selection.get("selectionId") != "taevaskoda-orthophoto-54472-stage1"
        or selection.get("sheet") != "54472"
        or selection.get("complete") is not True
    ):
        raise ValueError("orthophoto retained selection identity differs")
    if selection.get("temporalPolicy") != (
        "RGB 2025 and CIR 2024 remain separate dated observations, never a "
        "co-temporal composite."
    ):
        raise ValueError("orthophoto temporal separation policy differs")

    product_entries = selection.get("products")
    if not isinstance(product_entries, list) or {p.get("product") for p in product_entries} != {
        "rgb",
        "cir",
    }:
        raise ValueError("orthophoto selection must contain exactly RGB and CIR products")

    sources: dict[str, _RasterSource] = {}
    for selected in product_entries:
        product = str(selected["product"])
        expected = _EXPECTED_PRODUCTS[product]
        if selected.get("captureDate") != expected["captureDate"]:
            raise ValueError(f"{product} capture date differs")

        product_retention_path = _relative_path(
            _ORTHOPHOTO_ROOT, str(selected["retention"]), f"{product} retention"
        )
        product_retention_sha = str(selected["retentionSha256"])
        _verify_file(
            product_retention_path,
            expected_sha256=product_retention_sha,
            expected_bytes=None,
            label=f"{product} product retention",
        )
        product_retention = _read_object(product_retention_path, f"{product} retention")
        if (
            product_retention.get("schemaVersion") != "orthophoto-product-retention/1"
            or product_retention.get("selectionId") != selection["selectionId"]
            or product_retention.get("product") != product
            or product_retention.get("sheet") != selection["sheet"]
            or product_retention.get("captureDate") != expected["captureDate"]
        ):
            raise ValueError(f"{product} retention identity differs")

        if product_retention.get("archive") != selected.get("archive"):
            raise ValueError(f"{product} selected archive differs from product retention")
        extraction = selected["extraction"]
        if product_retention.get("extraction") != extraction:
            raise ValueError(f"{product} selected extraction differs from product retention")

        extraction_manifest_path = _relative_path(
            _ORTHOPHOTO_ROOT, str(extraction["manifest"]), f"{product} extraction manifest"
        )
        extraction_manifest_sha = str(extraction["manifestSha256"])
        _verify_file(
            extraction_manifest_path,
            expected_sha256=extraction_manifest_sha,
            expected_bytes=None,
            label=f"{product} extraction manifest",
        )
        extraction_manifest = _read_object(
            extraction_manifest_path, f"{product} extraction manifest"
        )
        if (
            extraction_manifest.get("schemaVersion")
            != "orthophoto-extraction-retention/1"
            or extraction_manifest.get("selectionId") != selection["selectionId"]
            or extraction_manifest.get("product") != product
            or extraction_manifest.get("sheet") != selection["sheet"]
            or extraction_manifest.get("captureDate") != expected["captureDate"]
            or extraction_manifest.get("zipCrcVerified") is not True
        ):
            raise ValueError(f"{product} extraction identity differs")

        archive = selected["archive"]
        archive_path = _ORTHOPHOTO_ROOT / product / str(selection["sheet"]) / str(
            archive["filename"]
        )
        _verify_file(
            archive_path,
            expected_sha256=str(archive["sha256"]),
            expected_bytes=int(archive["bytes"]),
            label=f"{product} source archive",
        )

        tif_entries = [
            entry
            for entry in extraction_manifest.get("files", [])
            if str(entry.get("relativePath", "")).lower().endswith((".tif", ".tiff"))
        ]
        if len(tif_entries) != 1:
            raise ValueError(f"{product} extraction must contain exactly one GeoTIFF")
        tif_entry = tif_entries[0]
        extraction_root = _relative_path(
            _ORTHOPHOTO_ROOT, str(extraction["root"]), f"{product} extraction root"
        )
        tif_path = _relative_path(
            extraction_root, str(tif_entry["relativePath"]), f"{product} GeoTIFF"
        )
        _verify_file(
            tif_path,
            expected_sha256=str(tif_entry["sha256"]),
            expected_bytes=int(tif_entry["bytes"]),
            label=f"{product} GeoTIFF",
        )

        geotiff = extraction_manifest.get("geoTiff")
        if not isinstance(geotiff, Mapping):
            raise ValueError(f"{product} extraction lacks GeoTIFF metadata")
        pixel_m = float(expected["pixelM"])
        with rasterio.open(tif_path) as dataset:
            observed = {
                "driver": dataset.driver,
                "crs": str(dataset.crs),
                "epsg": dataset.crs.to_epsg() if dataset.crs is not None else None,
                "width": dataset.width,
                "height": dataset.height,
                "bands": dataset.count,
                "dtypes": list(dataset.dtypes),
                "bounds": [float(value) for value in dataset.bounds],
                "res": [float(value) for value in dataset.res],
                "transform": [float(value) for value in dataset.transform[:6]],
                "nodata": dataset.nodata,
            }
        for key in (
            "driver",
            "crs",
            "epsg",
            "width",
            "height",
            "bands",
            "dtypes",
            "bounds",
            "res",
            "transform",
            "nodata",
        ):
            if observed[key] != geotiff.get(key):
                raise ValueError(f"{product} live GeoTIFF {key} differs from retention")
        if (
            observed["crs"] != "EPSG:3301"
            or observed["bands"] != 3
            or observed["dtypes"] != ["uint8", "uint8", "uint8"]
            or observed["res"] != [pixel_m, pixel_m]
        ):
            raise ValueError(f"{product} GeoTIFF is not the retained native RGB-shaped raster")

        sources[product] = _RasterSource(
            product=product,
            capture_date=str(expected["captureDate"]),
            pixel_m=pixel_m,
            archive_path=archive_path,
            archive_sha256=str(archive["sha256"]),
            archive_bytes=int(archive["bytes"]),
            product_retention_path=product_retention_path,
            product_retention_sha256=product_retention_sha,
            extraction_manifest_path=extraction_manifest_path,
            extraction_manifest_sha256=extraction_manifest_sha,
            tif_path=tif_path,
            tif_sha256=str(tif_entry["sha256"]),
            tif_bytes=int(tif_entry["bytes"]),
            geotiff=geotiff,
        )
    return sources


def _exact_etak_feature(
    source: Path,
    *,
    layer: str,
    etak_id: int,
    expected_kkr_code: str,
    expected_name: str,
) -> BaseGeometry:
    metadata, _, geometry_wkb, fields = pyogrio.raw.read(
        source,
        layer=layer,
        columns=["etak_id", "kkr_kood", "nimetus"],
        where=f"etak_id = {int(etak_id)}",
    )
    if metadata.get("crs") != "EPSG:3301" or len(geometry_wkb) != 1:
        raise ValueError(f"ETAK {layer}/{etak_id} is not one EPSG:3301 feature")
    values = dict(zip(metadata["fields"], fields, strict=True))
    if (
        int(values["etak_id"][0]) != etak_id
        or values["kkr_kood"][0] != expected_kkr_code
        or values["nimetus"][0] != expected_name
    ):
        raise ValueError(f"ETAK identity fields differ for {layer}/{etak_id}")
    return _valid_geometry(shapely.from_wkb(geometry_wkb[0]), f"ETAK {layer}/{etak_id}")


def _load_shorelines(
    source: Path,
    bounds_en: tuple[float, float, float, float],
) -> tuple[tuple[BaseGeometry, ...], tuple[Mapping[str, Any], ...]]:
    metadata, _, geometry_wkb, fields = pyogrio.raw.read(
        source,
        layer=_SHORELINE_LAYER,
        columns=["etak_id"],
        bbox=bounds_en,
    )
    if metadata.get("crs") != "EPSG:3301":
        raise ValueError(f"ETAK {_SHORELINE_LAYER} CRS differs")
    values = dict(zip(metadata["fields"], fields, strict=True))
    if "etak_id" not in values:
        raise ValueError(f"ETAK {_SHORELINE_LAYER} lacks etak_id")

    review_box = box(*bounds_en)
    rows: list[tuple[int, BaseGeometry, Mapping[str, Any]]] = []
    for index, payload in enumerate(geometry_wkb):
        geometry = _valid_geometry(
            shapely.from_wkb(payload), f"ETAK {_SHORELINE_LAYER} row {index}"
        )
        if not geometry.intersects(review_box):
            continue
        etak_id = int(values["etak_id"][index])
        rows.append(
            (
                etak_id,
                geometry,
                {
                    "layer": _SHORELINE_LAYER,
                    "etakId": etak_id,
                    "geometryWkbSha256": _sha256_bytes(_geometry_bytes(geometry)),
                    "intersectionLengthM": float(geometry.intersection(review_box).length),
                },
            )
        )
    rows.sort(key=lambda row: row[0])
    return tuple(row[1] for row in rows), tuple(row[2] for row in rows)


def _load_authority(
    bounds_en: tuple[float, float, float, float],
) -> _AuthorityGeometry:
    authority = _read_object(_AUTHORITY_CONFIG_PATH, "authority config")
    stage = _read_object(_STAGE_CONFIG_PATH, "stage config")
    if (
        authority.get("format") != 2
        or authority.get("authorityId") != "taevaskoda-ahja-structural-authority-stage1"
        or authority.get("role") != "structural_authority_0.25m_no_morphology"
        or stage.get("format") != 1
        or stage.get("pilotId") != "taevaskoda-ahja-stage1"
    ):
        raise ValueError("frozen Taevaskoda authority identity differs")

    campaign_spec = authority["campaign"]
    campaign_root = _asset_path(str(campaign_spec["root"]))
    if campaign_root.name != campaign_spec["buildSha256"]:
        raise ValueError("authority campaign root is not content-addressed by build SHA")
    campaign_files: dict[str, Mapping[str, Any]] = {}
    for filename, specification in campaign_spec["files"].items():
        path = campaign_root / filename
        _verify_file(
            path,
            expected_sha256=str(specification["sha256"]),
            expected_bytes=int(specification["bytes"]),
            label=f"authority campaign {filename}",
        )
        campaign_files[filename] = {
            "path": str(path.relative_to(ASSET_GEN_ROOT)),
            "bytes": int(specification["bytes"]),
            "sha256": str(specification["sha256"]),
        }
    campaign = _read_object(campaign_root / "campaign.json", "authority campaign")
    reach_id = str(stage["reach"]["reachId"])
    if campaign.get("reachId") != reach_id:
        raise ValueError("campaign and stage reach identities differ")

    geometry_files: dict[str, Mapping[str, Any]] = {}
    geometries: dict[str, BaseGeometry] = {}
    for role, specification in authority["geometry"].items():
        path = campaign_root / str(specification["path"])
        _verify_file(
            path,
            expected_sha256=str(specification["sha256"]),
            expected_bytes=int(specification["bytes"]),
            label=f"authority geometry {role}",
        )
        geometries[role] = _valid_geometry(
            shapely.from_wkb(path.read_bytes()), f"authority {role}"
        )
        geometry_files[role] = {
            "path": str(path.relative_to(ASSET_GEN_ROOT)),
            "bytes": int(specification["bytes"]),
            "sha256": str(specification["sha256"]),
        }
    if (
        campaign.get("waterPolygonWkbSha256")
        != authority["geometry"]["qualifiedWater"]["sha256"]
        or campaign.get("centerlineWkbSha256")
        != authority["geometry"]["qualifiedCenterline"]["sha256"]
        or not geometries["qualifiedWater"].covers(geometries["qualifiedCenterline"])
    ):
        raise ValueError("campaign and configured authority geometry differ")

    mapped_spec = authority["geometry"]["mappedWater"]
    if mapped_spec.get("closureOperation") != "normalized_union_with_qualified_water":
        raise ValueError("mapped authority closure operation differs")
    effective_mapped = _valid_geometry(
        shapely.normalize(
            shapely.union(geometries["mappedWater"], geometries["qualifiedWater"])
        ),
        "effective mapped water",
    )
    effective_sha = _sha256_bytes(_geometry_bytes(effective_mapped))
    if effective_sha != mapped_spec.get("effectiveSha256"):
        raise ValueError("effective mapped-water geometry SHA-256 differs")

    etak_spec = stage["etak"]
    etak_path = _asset_path(str(etak_spec["path"]))
    _verify_file(
        etak_path,
        expected_sha256=str(etak_spec["sourceSha256"]),
        expected_bytes=None,
        label="frozen ETAK GeoPackage",
    )
    area_spec = etak_spec["waterPolygon"]
    line_spec = etak_spec["centerline"]
    area_feature = _exact_etak_feature(
        etak_path,
        layer=str(area_spec["layer"]),
        etak_id=int(area_spec["etakId"]),
        expected_kkr_code=str(area_spec["kkrCode"]),
        expected_name=str(area_spec["name"]),
    )
    centerline_feature = _exact_etak_feature(
        etak_path,
        layer=str(line_spec["layer"]),
        etak_id=int(line_spec["etakId"]),
        expected_kkr_code=str(line_spec["kkrCode"]),
        expected_name=str(line_spec["name"]),
    )
    expected_geometry = stage["expectedGeometrySha256"]
    if (
        _sha256_bytes(_geometry_bytes(area_feature))
        != expected_geometry["forced2dWaterPolygon"]
        or _sha256_bytes(_geometry_bytes(centerline_feature))
        != expected_geometry["forced2dCenterline"]
    ):
        raise ValueError("frozen ETAK source feature geometry differs from stage config")

    shorelines, shoreline_features = _load_shorelines(etak_path, bounds_en)
    identity = {
        "authorityConfig": {
            "path": str(_AUTHORITY_CONFIG_PATH.relative_to(ASSET_GEN_ROOT)),
            "sha256": _sha256_file(_AUTHORITY_CONFIG_PATH),
            "authorityId": authority["authorityId"],
            "role": authority["role"],
        },
        "stageConfig": {
            "path": str(_STAGE_CONFIG_PATH.relative_to(ASSET_GEN_ROOT)),
            "sha256": _sha256_file(_STAGE_CONFIG_PATH),
            "pilotId": stage["pilotId"],
        },
        "campaign": {
            "root": str(campaign_root.relative_to(ASSET_GEN_ROOT)),
            "buildSha256": campaign_spec["buildSha256"],
            "contentSha256": campaign_spec["contentSha256"],
            "reachId": reach_id,
            "files": campaign_files,
        },
        "geometryFiles": geometry_files,
        "effectiveMappedWaterWkbSha256": effective_sha,
        "etak": {
            "path": str(etak_path.relative_to(ASSET_GEN_ROOT)),
            "sha256": etak_spec["sourceSha256"],
            "mappedWaterFeature": {
                "layer": area_spec["layer"],
                "etakId": area_spec["etakId"],
                "kkrCode": area_spec["kkrCode"],
                "name": area_spec["name"],
                "geometryWkbSha256": _sha256_bytes(_geometry_bytes(area_feature)),
            },
            "qualifiedCenterlineFeature": {
                "layer": line_spec["layer"],
                "etakId": line_spec["etakId"],
                "kkrCode": line_spec["kkrCode"],
                "name": line_spec["name"],
                "geometryWkbSha256": _sha256_bytes(_geometry_bytes(centerline_feature)),
            },
            "explicitShorelineFeatures": list(shoreline_features),
        },
    }
    return _AuthorityGeometry(
        mapped_water=effective_mapped,
        qualified_water=geometries["qualifiedWater"],
        qualified_centerline=geometries["qualifiedCenterline"],
        mapped_area_feature=area_feature,
        mapped_centerline_feature=centerline_feature,
        shorelines=shorelines,
        shoreline_features=shoreline_features,
        identity=identity,
    )


def _integer_cell(value: float, label: str) -> int:
    rounded = round(value)
    if abs(value - rounded) > 1e-7:
        raise ValueError(f"{label} is not on the native raster grid: {value}")
    return int(rounded)


def _crop_native(
    source: _RasterSource,
    bounds_en: tuple[float, float, float, float],
) -> _Crop:
    min_e, min_n, max_e, max_n = bounds_en
    pixel_m = source.pixel_m
    width = _integer_cell((max_e - min_e) / pixel_m, f"{source.product} crop width")
    height = _integer_cell((max_n - min_n) / pixel_m, f"{source.product} crop height")
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    available = np.zeros((height, width), dtype=bool)

    with rasterio.open(source.tif_path) as dataset:
        intersection = (
            max(min_e, float(dataset.bounds.left)),
            max(min_n, float(dataset.bounds.bottom)),
            min(max_e, float(dataset.bounds.right)),
            min(max_n, float(dataset.bounds.top)),
        )
        left, bottom, right, top = intersection
        if left >= right or bottom >= top:
            raise ValueError(f"{source.product} has no coverage of the review domain")

        source_col = _integer_cell(
            (left - float(dataset.bounds.left)) / pixel_m,
            f"{source.product} source column",
        )
        source_row = _integer_cell(
            (float(dataset.bounds.top) - top) / pixel_m,
            f"{source.product} source row",
        )
        output_col = _integer_cell(
            (left - min_e) / pixel_m, f"{source.product} output column"
        )
        output_row = _integer_cell(
            (max_n - top) / pixel_m, f"{source.product} output row"
        )
        window_width = _integer_cell(
            (right - left) / pixel_m, f"{source.product} window width"
        )
        window_height = _integer_cell(
            (top - bottom) / pixel_m, f"{source.product} window height"
        )
        window = Window(source_col, source_row, window_width, window_height)
        data = dataset.read((1, 2, 3), window=window)
        mask = dataset.read_masks(1, window=window) > 0
        expected_shape = (3, window_height, window_width)
        if data.shape != expected_shape or mask.shape != expected_shape[1:]:
            raise ValueError(f"{source.product} native crop shape differs")
        rows = slice(output_row, output_row + window_height)
        columns = slice(output_col, output_col + window_width)
        pixels[rows, columns] = np.moveaxis(data, 0, -1)
        available[rows, columns] = mask

    return _Crop(
        pixels=pixels,
        available=available,
        transform=(pixel_m, 0.0, min_e, 0.0, -pixel_m, max_n),
        source_intersection_bounds_en=intersection,
    )


def _styled_image(crop: _Crop, title: str, subtitle: str) -> Image.Image:
    pixels = crop.pixels.copy()
    yy, xx = np.indices(crop.available.shape)
    checker = ((xx // 16 + yy // 16) & 1).astype(bool)
    missing_light = np.array([49, 54, 58], dtype=np.uint8)
    missing_dark = np.array([26, 30, 33], dtype=np.uint8)
    missing_pixels = np.where(checker[..., None], missing_light, missing_dark)
    pixels[~crop.available] = missing_pixels[~crop.available]
    image = Image.fromarray(pixels, mode="RGB")
    draw = ImageDraw.Draw(image)

    missing_rows = np.flatnonzero(~np.all(crop.available, axis=1))
    if len(missing_rows):
        top = int(missing_rows.min())
        bottom = int(missing_rows.max()) + 1
        label_bottom = min(bottom, top + 84)
        draw.rectangle((8, top + 8, min(image.width - 8, 1120), label_bottom), fill=(8, 10, 12))
        draw.text((18, top + 16), title, fill=(255, 255, 255))
        draw.text((18, top + 38), subtitle, fill=(230, 230, 230))
        draw.text(
            (18, top + 60),
            "QA ONLY | CHECKER = NO RETAINED SOURCE COVERAGE",
            fill=(255, 193, 66),
        )
    return image


def _linework(geometry: BaseGeometry) -> Iterable[BaseGeometry]:
    geometry_type = geometry.geom_type
    if geometry_type in {"LineString", "LinearRing"}:
        yield geometry
    elif geometry_type == "Polygon":
        yield geometry.exterior
        yield from geometry.interiors
    elif geometry_type.startswith("Multi") or geometry_type == "GeometryCollection":
        for part in geometry.geoms:
            yield from _linework(part)


def _draw_geometry(
    draw: ImageDraw.ImageDraw,
    geometry: BaseGeometry,
    *,
    transform: tuple[float, float, float, float, float, float],
    color: tuple[int, int, int],
    width: int,
) -> None:
    pixel_m, _, min_e, _, negative_pixel_m, max_n = transform
    if negative_pixel_m != -pixel_m:
        raise ValueError("overlay transform must be north-up with square pixels")
    for line in _linework(geometry):
        coordinates = [
            ((float(e) - min_e) / pixel_m, (max_n - float(n)) / pixel_m)
            for e, n, *_ in line.coords
        ]
        if len(coordinates) >= 2:
            draw.line(coordinates, fill=color, width=width, joint="curve")


def _overlay_image(
    crop: _Crop,
    authority: _AuthorityGeometry,
) -> tuple[Image.Image, list[Mapping[str, Any]]]:
    image = _styled_image(
        crop,
        "03 ETAK STRUCTURAL AUTHORITY OVERLAY ON RGB 2025-07-18",
        "EPSG:3301 | exact geometry-to-pixel transform | source dates are not merged",
    )
    draw = ImageDraw.Draw(image)
    styles: list[Mapping[str, Any]] = [
        {
            "role": "mappedEffectiveWaterBoundary",
            "colorRgb": [34, 139, 230],
            "widthPixels": 7,
            "widthM": 1.4,
        },
        {
            "role": "explicitEtakShoreline",
            "colorRgb": [0, 255, 224],
            "widthPixels": 3,
            "widthM": 0.6,
        },
        {
            "role": "qualifiedDownstreamWaterBoundary",
            "colorRgb": [255, 194, 36],
            "widthPixels": 3,
            "widthM": 0.6,
        },
        {
            "role": "qualifiedDownstreamCenterline",
            "colorRgb": [255, 47, 199],
            "widthPixels": 3,
            "widthM": 0.6,
        },
    ]
    _draw_geometry(
        draw,
        authority.mapped_water,
        transform=crop.transform,
        color=(34, 139, 230),
        width=7,
    )
    for shoreline in authority.shorelines:
        _draw_geometry(
            draw,
            shoreline,
            transform=crop.transform,
            color=(0, 255, 224),
            width=3,
        )
    _draw_geometry(
        draw,
        authority.qualified_water,
        transform=crop.transform,
        color=(255, 194, 36),
        width=3,
    )
    _draw_geometry(
        draw,
        authority.qualified_centerline,
        transform=crop.transform,
        color=(255, 47, 199),
        width=3,
    )

    legend_y = 106
    draw.rectangle((8, legend_y - 8, 1040, legend_y + 94), fill=(8, 10, 12))
    for index, style in enumerate(styles):
        y = legend_y + index * 22
        color = tuple(int(component) for component in style["colorRgb"])
        draw.line((18, y + 6, 78, y + 6), fill=color, width=int(style["widthPixels"]))
        draw.text((90, y), str(style["role"]), fill=(255, 255, 255))
    return image, styles


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable evidence artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError(f"concurrent immutable evidence artifact differs: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _png_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG", compress_level=9, optimize=False)
    return output.getvalue()


def _source_identity(source: _RasterSource) -> Mapping[str, Any]:
    return {
        "product": source.product,
        "captureDate": source.capture_date,
        "pixelM": source.pixel_m,
        "displayPolicy": (
            "native_three_band_false_color_no_stretch_or_remap"
            if source.product == "cir"
            else "native_three_band_rgb_no_stretch_or_remap"
        ),
        "archive": {
            "path": str(source.archive_path.relative_to(ASSET_GEN_ROOT)),
            "bytes": source.archive_bytes,
            "sha256": source.archive_sha256,
        },
        "productRetention": {
            "path": str(source.product_retention_path.relative_to(ASSET_GEN_ROOT)),
            "sha256": source.product_retention_sha256,
        },
        "extractionManifest": {
            "path": str(source.extraction_manifest_path.relative_to(ASSET_GEN_ROOT)),
            "sha256": source.extraction_manifest_sha256,
        },
        "geoTiff": {
            "path": str(source.tif_path.relative_to(ASSET_GEN_ROOT)),
            "bytes": source.tif_bytes,
            "sha256": source.tif_sha256,
            "metadata": dict(source.geotiff),
        },
    }


def _review_bounds(authority: Mapping[str, Any]) -> tuple[float, float, float, float]:
    review_parent = authority["closure"]["reviewParent"]
    if review_parent != [-1, 607, 372]:
        raise ValueError("frozen review parent differs")
    lod, cx, cz = (int(value) for value in review_parent)
    bounds = tuple(float(value) for value in chunk_bounds_en(load_base().grid, ChunkId(cx, cz, lod)))
    if bounds != (679424.0, 6444544.0, 679936.0, 6445056.0):
        raise ValueError(f"derived review-parent bounds differ: {bounds}")
    return bounds


def _software_identity() -> Mapping[str, Any]:
    return {
        "implementation": "deterministic-pillow-native-raster-overlay-v1",
        "numpy": importlib.metadata.version("numpy"),
        "pillow": importlib.metadata.version("pillow"),
        "pyogrio": importlib.metadata.version("pyogrio"),
        "rasterio": importlib.metadata.version("rasterio"),
        "shapely": importlib.metadata.version("shapely"),
        "uvLockSha256": _sha256_file(ASSET_GEN_ROOT / "uv.lock"),
        "png": {"compressLevel": 9, "optimize": False, "metadata": "none"},
    }


def _image_record(
    *,
    filename: str,
    payload: bytes,
    image: Image.Image,
    source: _RasterSource,
    crop: _Crop,
    interpretation: str,
) -> Mapping[str, Any]:
    valid_pixels = int(np.count_nonzero(crop.available))
    total_pixels = int(crop.available.size)
    return {
        "file": filename,
        "sha256": _sha256_bytes(payload),
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "product": source.product,
        "captureDate": source.capture_date,
        "pixelM": source.pixel_m,
        "boundsEn": [float(value) for value in (
            crop.transform[2],
            crop.transform[5] - image.height * source.pixel_m,
            crop.transform[2] + image.width * source.pixel_m,
            crop.transform[5],
        )],
        "pixelToWorldTransform": list(crop.transform),
        "pixelCenterFormula": {
            "e": "c + (column + 0.5) * a",
            "n": "f + (row + 0.5) * e",
        },
        "sourceIntersectionBoundsEn": list(crop.source_intersection_bounds_en),
        "coverage": {
            "availablePixels": valid_pixels,
            "totalPixels": total_pixels,
            "availableFraction": valid_pixels / total_pixels,
            "missingDisplay": "labeled_checker_not_source_data",
        },
        "interpretation": interpretation,
    }


def materialize() -> Path:
    """Write the immutable evidence bundle and return its machine index path."""
    selection = _read_object(_SELECTION_PATH, "orthophoto selection")
    _verify_file(
        _SELECTION_PATH,
        expected_sha256=_sha256_file(_SELECTION_PATH),
        expected_bytes=None,
        label="orthophoto selection",
    )
    authority_config = _read_object(_AUTHORITY_CONFIG_PATH, "authority config")
    bounds_en = _review_bounds(authority_config)
    sources = _load_raster_sources(selection)
    authority = _load_authority(bounds_en)
    source_code_path = Path(__file__).resolve()

    build_identity = {
        "recipe": _RECIPE,
        "sourceCode": {
            "path": str(source_code_path.relative_to(ASSET_GEN_ROOT)),
            "sha256": _sha256_file(source_code_path),
        },
        "reviewDomain": {
            "crs": "EPSG:3301",
            "reviewParent": [-1, 607, 372],
            "boundsEn": list(bounds_en),
            "widthM": 512.0,
            "heightM": 512.0,
        },
        "orthophotoSelection": {
            "path": str(_SELECTION_PATH.relative_to(ASSET_GEN_ROOT)),
            "sha256": _sha256_file(_SELECTION_PATH),
            "selectionId": selection["selectionId"],
            "temporalPolicy": selection["temporalPolicy"],
        },
        "orthophotoSources": {
            product: _source_identity(sources[product]) for product in ("rgb", "cir")
        },
        "authority": authority.identity,
        "software": _software_identity(),
        "scientificDisposition": (
            "qa_only_no_height_derivation_no_shoreline_displacement_no_morphology_truth"
        ),
    }
    build_sha256 = _sha256_bytes(_canonical_json(build_identity))
    qa_root = _OUTPUT_PARENT / build_sha256 / "qa"

    rgb_crop = _crop_native(sources["rgb"], bounds_en)
    cir_crop = _crop_native(sources["cir"], bounds_en)
    rgb_image = _styled_image(
        rgb_crop,
        "01 MAA- JA RUUMIAMET RGB | CAPTURE 2025-07-18",
        "native 0.20 m pixels | EPSG:3301 | no resampling or color stretch",
    )
    cir_image = _styled_image(
        cir_crop,
        "02 MAA- JA RUUMIAMET CIR FALSE COLOR | CAPTURE 2024-05-22",
        "native 0.25 m three-band display | EPSG:3301 | no resampling or remap",
    )
    overlay_image, overlay_styles = _overlay_image(rgb_crop, authority)

    rendered = [
        (
            "01-rgb-2025-07-18-review-domain.png",
            rgb_image,
            sources["rgb"],
            rgb_crop,
            "Dated native RGB observation for visual surface/context evidence; the "
            "checker strip is outside retained sheet 54472 and is not invented imagery.",
        ),
        (
            "02-cir-false-color-2024-05-22-review-domain.png",
            cir_image,
            sources["cir"],
            cir_crop,
            "Dated native CIR false-color observation for vegetation and wetness-context "
            "review; it is a separate 2024 observation, not fused with 2025 RGB.",
        ),
        (
            "03-etak-shoreline-authority-overlay-review-domain.png",
            overlay_image,
            sources["rgb"],
            rgb_crop,
            "Exact frozen ETAK and qualified structural boundaries over dated RGB for "
            "human QA only; line width is visualization, not geometric uncertainty or repair.",
        ),
    ]
    images: list[Mapping[str, Any]] = []
    for filename, image, source, crop, interpretation in rendered:
        payload = _png_bytes(image)
        _write_immutable(qa_root / filename, payload)
        images.append(
            _image_record(
                filename=filename,
                payload=payload,
                image=image,
                source=source,
                crop=crop,
                interpretation=interpretation,
            )
        )

    images[2] = {
        **images[2],
        "overlay": {
            "styles": overlay_styles,
            "featureIds": {
                "mappedWater": {
                    "layer": authority.identity["etak"]["mappedWaterFeature"]["layer"],
                    "etakId": authority.identity["etak"]["mappedWaterFeature"]["etakId"],
                },
                "qualifiedCenterline": {
                    "layer": authority.identity["etak"]["qualifiedCenterlineFeature"]["layer"],
                    "etakId": authority.identity["etak"]["qualifiedCenterlineFeature"]["etakId"],
                },
                "explicitShorelines": [
                    {"layer": row["layer"], "etakId": row["etakId"]}
                    for row in authority.shoreline_features
                ],
                "qualifiedReachId": authority.identity["campaign"]["reachId"],
            },
            "geometryPolicy": (
                "coordinates_are_unmodified; only colored strokes are added for visibility"
            ),
        },
    }

    index = {
        "format": 1,
        "id": "taevaskoda-orthophoto-etak-review-domain-qa-v1",
        "buildSha256": build_sha256,
        "buildIdentity": build_identity,
        "images": images,
        "disposition": {
            "role": "qa_evidence_only",
            "heightDerivationAllowed": False,
            "shorelineDisplacementAllowed": False,
            "morphologyTruthAllowed": False,
            "runtimeOrCookInput": False,
            "temporalFusionAllowed": False,
        },
        "warnings": [
            "The review parent extends 56 m north of retained orthophoto sheet 54472.",
            "RGB and CIR dates differ and must not be interpreted as a co-temporal composite.",
            "ETAK strokes show exact coordinates but their pixel widths are visualization only.",
        ],
    }
    index_path = qa_root / "index.json"
    _write_immutable(index_path, _pretty_json(index))
    return index_path


def main() -> None:
    print(materialize())


if __name__ == "__main__":
    main()
