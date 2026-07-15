"""Materialize Development A/B condition evidence without terrain synthesis."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyogrio
import pyogrio.raw
import rasterio.features
import rasterio.transform
import shapely
from PIL import Image
from scipy import ndimage

from .....config import ASSET_GEN_ROOT, DATA_IN, DATA_WORK, load_base
from .....cook.chunkio import read_chunk, read_chunk_v2
from .....cook.encode import decode_quant16
from .....grid import ChunkId, chunk_bounds_en, chunks_covering_bbox_en
from .....release import read_v1_index
from ....conditions.geology import extract_egt_surficial_window
from ....conditions.soil import extract_soil_window
from .drainage import DrainageDomain, delineate_drainage_domain

_SCHEMA_VERSION = "laas.erodible-slope-condition-bundle/1"
_SITE_ARRAY_SCHEMA = "laas.erodible-slope-condition-grid/1"
_ETAK = DATA_IN / "etak" / "ETAK_EESTI_GPKG.gpkg"
_GRID_METERS = 1.0
_SEARCH_MARGIN_METERS = 384
_ENLARGEMENT_METERS = 128
_COLLAR_METERS = 32


@dataclass(frozen=True)
class SiteSpec:
    site_id: str
    role: str
    target_etak_id: int
    target_e: float
    target_n: float
    sheet: str
    target_outlet_ids: tuple[int, ...]
    output_chunk: tuple[int, int, int] | None = None

_ETAK_LAYERS: dict[str, tuple[str, ...]] = {
    "E_102_nolv_j": (
        "etak_id",
        "kood_t",
        "tyyp",
        "tyyp_t",
        "kaldaastang",
        "kaldaastang_t",
        "muutmisaeg",
        "geom_muutmisaeg",
        "valjavote",
    ),
    "E_203_vooluveekogu_a": (
        "etak_id",
        "kood_t",
        "kkr_kood",
        "nimetus",
        "muutmisaeg",
        "geom_muutmisaeg",
        "valjavote",
    ),
    "E_203_vooluveekogu_j": (
        "etak_id",
        "kood_t",
        "tyyp",
        "tyyp_t",
        "telje_tyyp_t",
        "laius",
        "laius_t",
        "telje_staatus_t",
        "kkr_kood",
        "nimetus",
        "muutmisaeg",
        "geom_muutmisaeg",
        "valjavote",
    ),
    "E_204_kaldajoon_j": (
        "etak_id",
        "kood_t",
        "tyyp",
        "tyyp_t",
        "muutmisaeg",
        "geom_muutmisaeg",
        "valjavote",
    ),
    "E_206_truup_j": (
        "etak_id",
        "kood_t",
        "tyyp",
        "tyyp_t",
        "muutmisaeg",
        "geom_muutmisaeg",
        "valjavote",
    ),
    "E_305_puittaimestik_a": (
        "etak_id",
        "kood_t",
        "tyyp",
        "tyyp_t",
        "muutmisaeg",
        "geom_muutmisaeg",
        "valjavote",
    ),
    "E_401_hoone_ka": (
        "etak_id",
        "kood_t",
        "tyyp",
        "tyyp_t",
        "korgus_m",
        "muutmisaeg",
        "geom_muutmisaeg",
        "valjavote",
    ),
    "E_402_korgrajatis_p": (
        "etak_id",
        "kood_t",
        "tyyp",
        "tyyp_t",
        "muutmisaeg",
        "geom_muutmisaeg",
        "valjavote",
    ),
    "E_403_muu_rajatis_ka": (
        "etak_id",
        "kood_t",
        "tyyp",
        "tyyp_t",
        "muutmisaeg",
        "geom_muutmisaeg",
        "valjavote",
    ),
    "E_403_muu_rajatis_p": (
        "etak_id",
        "kood_t",
        "tyyp",
        "tyyp_t",
        "muutmisaeg",
        "geom_muutmisaeg",
        "valjavote",
    ),
    "E_501_tee_a": (
        "etak_id",
        "kood_t",
        "tyyp",
        "tyyp_t",
        "muutmisaeg",
        "geom_muutmisaeg",
        "valjavote",
    ),
    "E_501_tee_j": (
        "etak_id",
        "kood_t",
        "tyyp",
        "tyyp_t",
        "laius",
        "teekate",
        "teekate_t",
        "muutmisaeg",
        "geom_muutmisaeg",
        "valjavote",
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ASSET_GEN_ROOT.parent))
    except ValueError:
        return str(path.resolve())


def _file_identity(
    path: Path,
    *,
    published_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "path": _repo_relative(published_path or path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _implementation_identity() -> dict[str, Any]:
    rows = []
    for name in ("drainage.py", "materialize.py"):
        path = Path(__file__).parent / name
        rows.append(_file_identity(path))
    return {
        "files": rows,
        "root_sha256": hashlib.sha256(_canonical_bytes(rows)).hexdigest(),
    }


def _runtime_identity() -> dict[str, Any]:
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "numpy": np.__version__,
        "pyogrio": pyogrio.__version__,
        "gdal": list(pyogrio.__gdal_version__),
        "shapely": shapely.__version__,
    }


def _load_preregistration(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("bundle_id") != "erodible-slope-research-bundle/1":
        raise ValueError("unexpected erodible-slope bundle identity")
    return payload


def _load_site_config(path: Path) -> tuple[list[SiteSpec], frozenset[int], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "laas.erodible-slope-condition-sites/1":
        raise ValueError("unsupported erodible-slope condition site config")
    selection_path = ASSET_GEN_ROOT.parent / payload["selection"]["path"]
    if _sha256_file(selection_path) != payload["selection"]["sha256"]:
        raise ValueError("frozen erodible-slope site selection identity changed")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection["development_replacement"]
    specs = [
        SiteSpec(
            site_id=row["id"],
            role=row["role"],
            target_etak_id=int(row["etak_id"]),
            target_e=float(row["e_m"]),
            target_n=float(row["n_m"]),
            sheet=str(row["maaamet_sheet"]),
            target_outlet_ids=tuple(int(value) for value in row["target_outlet_etak_ids"]),
            output_chunk=(
                tuple(int(value) for value in row["output_chunk"])
                if row.get("output_chunk") is not None
                else None
            ),
        )
        for row in payload["sites"]
    ]
    development_c = next(spec for spec in specs if spec.site_id == "development_c")
    if (
        development_c.target_etak_id != int(selected["etak_id"])
        or [development_c.target_e, development_c.target_n]
        != [float(value) for value in selected["representative_point_en"]]
        or list(development_c.output_chunk or ())
        != [
            int(selected["output_chunk"]["lod"]),
            int(selected["output_chunk"]["cx"]),
            int(selected["output_chunk"]["cz"]),
        ]
    ):
        raise ValueError("Development C site config differs from frozen selection")
    forbidden = frozenset(int(value) for value in payload["forbidden_etak_ids"])
    if any(spec.target_etak_id in forbidden for spec in specs):
        raise ValueError("condition site aliases a forbidden OOD ETAK identity")
    return specs, forbidden, payload


def _read_target_geometry(spec: SiteSpec) -> shapely.Geometry:
    metadata, fids, geometry_wkb, columns = pyogrio.raw.read(
        _ETAK,
        layer="E_102_nolv_j",
        where=f"etak_id = {spec.target_etak_id}",
        columns=["etak_id"],
        return_fids=True,
    )
    if len(fids) != 1 or geometry_wkb is None or int(columns[0][0]) != spec.target_etak_id:
        raise ValueError(f"ETAK target {spec.target_etak_id} is not exactly one feature")
    if str(metadata.get("crs")) != "EPSG:3301":
        raise ValueError("ETAK target is not EPSG:3301")
    return shapely.force_2d(shapely.from_wkb(bytes(geometry_wkb[0])))


def _site_bbox(target: shapely.Geometry) -> tuple[int, int, int, int]:
    min_e, min_n, max_e, max_n = target.bounds
    return (
        int(np.floor(min_e - _SEARCH_MARGIN_METERS)),
        int(np.floor(min_n - _SEARCH_MARGIN_METERS)),
        int(np.ceil(max_e + _SEARCH_MARGIN_METERS)),
        int(np.ceil(max_n + _SEARCH_MARGIN_METERS)),
    )


def _read_etak_features(
    bbox_en: tuple[int, int, int, int],
    target: shapely.Geometry,
    forbidden_etak_ids: frozenset[int],
) -> tuple[list[dict[str, Any]], dict[str, list[shapely.Geometry]]]:
    records: list[dict[str, Any]] = []
    geometries: dict[str, list[shapely.Geometry]] = {}
    for layer, fields in _ETAK_LAYERS.items():
        metadata, fids, geometry_wkb, columns = pyogrio.raw.read(
            _ETAK,
            layer=layer,
            bbox=bbox_en,
            where=" AND ".join(
                f"etak_id <> {etak_id}" for etak_id in sorted(forbidden_etak_ids)
            ),
            columns=list(fields),
            return_fids=True,
        )
        if str(metadata.get("crs")) != "EPSG:3301":
            raise ValueError(f"ETAK layer {layer} is not EPSG:3301")
        if geometry_wkb is None:
            raise ValueError(f"ETAK layer {layer} omitted geometry")
        order = np.argsort(fids, kind="stable")
        layer_geometries: list[shapely.Geometry] = []
        for source_index in order:
            raw_wkb = bytes(geometry_wkb[source_index])
            geometry = shapely.force_2d(shapely.from_wkb(raw_wkb))
            raw = {
                field: _json_scalar(columns[index][source_index])
                for index, field in enumerate(fields)
            }
            if int(raw["etak_id"]) in forbidden_etak_ids:
                raise RuntimeError("forbidden OOD geometry entered condition extraction")
            layer_geometries.append(geometry)
            records.append(
                {
                    "layer": layer,
                    "source_fid": int(fids[source_index]),
                    "attributes": raw,
                    "geometry": {
                        "encoding": "OGC_WKB_hex_from_OGR_without_clipping",
                        "wkb_hex": raw_wkb.hex(),
                        "sha256": hashlib.sha256(raw_wkb).hexdigest(),
                        "type": geometry.geom_type,
                        "bounds_en": [float(value) for value in geometry.bounds],
                        "length_m": float(geometry.length),
                        "area_m2": float(geometry.area),
                        "is_valid": bool(geometry.is_valid),
                    },
                    "target_distance_m": float(geometry.distance(target)),
                }
            )
        geometries[layer] = layer_geometries
    records.sort(key=lambda row: (row["layer"], row["attributes"]["etak_id"]))
    return records, geometries


def _authority_tiles(
    authority_root: Path,
    authority_manifest: dict[str, Any],
    bbox_en: tuple[int, int, int, int],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[list[int]]]:
    grid = load_base().grid
    chunks = chunks_covering_bbox_en(grid, bbox_en, -2)
    manifest_tiles = {tuple(row["key"]): row for row in authority_manifest["tiles"]}
    min_e, min_n, max_e, max_n = bbox_en
    shape = (max_n - min_n, max_e - min_e)
    height = np.full(shape, np.nan, dtype=np.float32)
    valid = np.zeros(shape, dtype=bool)
    unknown_bathymetry = np.zeros(shape, dtype=bool)
    forbidden_morphology = np.zeros(shape, dtype=bool)
    identities: list[dict[str, Any]] = []
    missing: list[list[int]] = []
    for chunk in chunks:
        row = manifest_tiles.get((-2, chunk.cx, chunk.cz))
        if row is None:
            missing.append([-2, chunk.cx, chunk.cz])
            continue
        path = authority_root / row["path"]
        if _sha256_file(path) != row["sha256"]:
            raise ValueError(f"accepted authority tile hash mismatch: {path}")
        with np.load(path) as source:
            source_height = np.asarray(source["height"], dtype=np.float32)
            source_valid = np.asarray(source["valid"], dtype=bool)
            source_unknown = np.asarray(source["unknown_bathymetry"], dtype=bool)
            source_forbidden = np.asarray(source["forbidden_morphology"], dtype=bool)
        if source_height.shape != (512, 512):
            raise ValueError("accepted authority tile shape changed")
        aggregate_height = source_height.reshape(128, 4, 128, 4).mean(axis=(1, 3))
        aggregate_valid = source_valid.reshape(128, 4, 128, 4).all(axis=(1, 3))
        aggregate_unknown = source_unknown.reshape(128, 4, 128, 4).any(axis=(1, 3))
        aggregate_forbidden = source_forbidden.reshape(128, 4, 128, 4).any(axis=(1, 3))
        tile_min_e, tile_min_n, tile_max_e, tile_max_n = (
            int(value) for value in chunk_bounds_en(grid, chunk)
        )
        overlap = tuple(
            int(value)
            for value in (
                max(min_e, tile_min_e),
                max(min_n, tile_min_n),
                min(max_e, tile_max_e),
                min(max_n, tile_max_n),
            )
        )
        if overlap[0] >= overlap[2] or overlap[1] >= overlap[3]:
            continue
        dst_cols = slice(overlap[0] - min_e, overlap[2] - min_e)
        dst_rows = slice(max_n - overlap[3], max_n - overlap[1])
        src_cols = slice(overlap[0] - tile_min_e, overlap[2] - tile_min_e)
        src_rows = slice(tile_max_n - overlap[3], tile_max_n - overlap[1])
        height[dst_rows, dst_cols] = aggregate_height[src_rows, src_cols]
        valid[dst_rows, dst_cols] = aggregate_valid[src_rows, src_cols]
        unknown_bathymetry[dst_rows, dst_cols] = aggregate_unknown[src_rows, src_cols]
        forbidden_morphology[dst_rows, dst_cols] = aggregate_forbidden[src_rows, src_cols]
        identities.append(
            {
                "key": row["key"],
                "path": _repo_relative(path),
                "bytes": row["bytes"],
                "sha256": row["sha256"],
                "baselinePath": _repo_relative(authority_root / row["baselinePath"]),
                "baselineBytes": row["baselineBytes"],
                "baselineSha256": row["baselineSha256"],
            }
        )
    return {
        "height": height,
        "valid": valid,
        "unknown_bathymetry": unknown_bathymetry,
        "forbidden_morphology": forbidden_morphology,
    }, identities, missing


def _corrected_lod0_window(
    materialization_path: Path,
    authority_manifest: dict[str, Any],
    bbox_en: tuple[int, int, int, int],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    transaction = json.loads(materialization_path.read_text(encoding="utf-8"))
    rows = {
        tuple(row["key"]): row
        for row in transaction.get("correctedLod0", ())
    }
    baseline_rows: dict[tuple[int, int, int], dict[str, Any]] = {}
    for tile in authority_manifest["baselineAuthority"]["tiles"]:
        for dependency in tile["dependencies"]:
            baseline_rows.setdefault(tuple(dependency["chunk"]), dependency)
    release = authority_manifest["baselineAuthority"]["release"]
    canonical_index_path = (
        DATA_WORK.parent
        / "out"
        / "m"
        / release["manifestSha256"][:16]
        / release["heightIndexPath"]
    )
    if _sha256_file(canonical_index_path) != release["heightIndexSha256"]:
        raise ValueError("pinned canonical height index identity changed")
    canonical_index = {
        record.key: record for record in read_v1_index(canonical_index_path)
    }
    grid = load_base().grid
    encode = load_base().encode
    min_e, min_n, max_e, max_n = bbox_en
    height = np.full((max_n - min_n, max_e - min_e), np.nan, dtype=np.float32)
    identities: list[dict[str, Any]] = []
    for chunk in chunks_covering_bbox_en(grid, bbox_en, 0):
        row = rows.get((0, chunk.cx, chunk.cz))
        if row is None:
            baseline = baseline_rows.get((0, chunk.cx, chunk.cz))
            if baseline is None:
                record = canonical_index.get((0, chunk.cx, chunk.cz))
                if record is None:
                    raise ValueError(
                        f"accepted corrected/canonical LOD0 lacks chunk {(0, chunk.cx, chunk.cz)}"
                    )
                hash8 = ((record.hash64 >> 32) & 0xFFFFFFFF).to_bytes(4, "big").hex()
                path = (
                    DATA_WORK.parent
                    / "out/c/height/0"
                    / f"{chunk.cx}_{chunk.cz}.{hash8}.bin"
                )
                if path.stat().st_size != record.size:
                    raise ValueError(f"pinned canonical LOD0 size mismatch: {path}")
                container_sha256 = _sha256_file(path)
                meta, payload = read_chunk(path)
                expected_decoded_sha256 = None
                identity = {
                    "key": [0, chunk.cx, chunk.cz],
                    "role": "pinned_canonical_unmodified_lod0",
                    "path": _repo_relative(path),
                    "bytes": record.size,
                    "index_hash64": record.hash64,
                    "container_sha256": container_sha256,
                }
            else:
                path = DATA_WORK.parent / "out" / baseline["content_relative_path"]
                if _sha256_file(path) != baseline["sha256"]:
                    raise ValueError(f"accepted canonical LOD0 hash mismatch: {path}")
                meta, payload = read_chunk(path)
                expected_decoded_sha256 = baseline["decoded_sha256"]
                identity = {
                    "key": baseline["chunk"],
                    "role": "pinned_canonical_unmodified_lod0",
                    "path": _repo_relative(path),
                    "bytes": baseline["bytes"],
                    "container_sha256": baseline["sha256"],
                    "decoded_values_sha256": expected_decoded_sha256,
                }
        else:
            path = materialization_path.parent / row["path"]
            if _sha256_file(path) != row["containerSha256"]:
                raise ValueError(f"accepted corrected LOD0 hash mismatch: {path}")
            meta, payload = read_chunk_v2(path)
            expected_decoded_sha256 = row["decodedValuesSha256"]
            identity = {
                "key": row["key"],
                "role": "accepted_corrected_lod0",
                "path": _repo_relative(path),
                "bytes": row["bytes"],
                "container_sha256": row["containerSha256"],
                "decoded_values_sha256": expected_decoded_sha256,
            }
        if (meta.layer, meta.lod, meta.cx, meta.cz, meta.enc) != (
            "height",
            0,
            chunk.cx,
            chunk.cz,
            1,
        ):
            raise ValueError(f"accepted corrected LOD0 header mismatch: {path}")
        values = decode_quant16(
            encode,
            payload,
            meta.res,
            meta.qoffset,
            meta.qscale,
        )
        decoded_sha256 = hashlib.sha256(
            np.ascontiguousarray(values, dtype="<f4").tobytes()
        ).hexdigest()
        if expected_decoded_sha256 is not None and decoded_sha256 != expected_decoded_sha256:
            raise ValueError(f"accepted corrected LOD0 decoded identity mismatch: {path}")
        identity["decoded_values_sha256"] = decoded_sha256
        tile_min_e, tile_min_n, tile_max_e, tile_max_n = (
            int(value) for value in chunk_bounds_en(grid, chunk)
        )
        overlap = tuple(
            int(value)
            for value in (
                max(min_e, tile_min_e),
                max(min_n, tile_min_n),
                min(max_e, tile_max_e),
                min(max_n, tile_max_n),
            )
        )
        dst_cols = slice(overlap[0] - min_e, overlap[2] - min_e)
        dst_rows = slice(max_n - overlap[3], max_n - overlap[1])
        src_cols = slice(overlap[0] - tile_min_e, overlap[2] - tile_min_e)
        src_rows = slice(tile_max_n - overlap[3], tile_max_n - overlap[1])
        height[dst_rows, dst_cols] = values[src_rows, src_cols]
        identities.append(identity)
    if not np.isfinite(height).all():
        raise ValueError("accepted corrected LOD0 does not cover condition bbox")
    return height, identities


def _transform(bbox_en: tuple[int, int, int, int]) -> rasterio.Affine:
    return rasterio.transform.from_origin(
        bbox_en[0], bbox_en[3], _GRID_METERS, _GRID_METERS
    )


def _rasterize(
    geometries: Iterable[shapely.Geometry],
    shape: tuple[int, int],
    transform: rasterio.Affine,
) -> np.ndarray:
    values = [(geometry, 1) for geometry in geometries if not geometry.is_empty]
    if not values:
        return np.zeros(shape, dtype=bool)
    return rasterio.features.rasterize(
        values,
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)


def _geometries_by_ids(
    records: list[dict[str, Any]],
    layer: str,
    ids: Iterable[int],
) -> list[shapely.Geometry]:
    wanted = set(ids)
    return [
        shapely.from_wkb(bytes.fromhex(row["geometry"]["wkb_hex"]))
        for row in records
        if row["layer"] == layer and int(row["attributes"]["etak_id"]) in wanted
    ]


def _mask_sources(
    spec: SiteSpec,
    records: list[dict[str, Any]],
    geometries: dict[str, list[shapely.Geometry]],
    shape: tuple[int, int],
    transform: rasterio.Affine,
    authority: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    water_polygons = geometries["E_203_vooluveekogu_a"]
    water_lines = geometries["E_203_vooluveekogu_j"]
    target_outlet_geometries: list[shapely.Geometry] = []
    target_ids = set(spec.target_outlet_ids)
    for layer in ("E_203_vooluveekogu_a", "E_203_vooluveekogu_j"):
        target_outlet_geometries.extend(_geometries_by_ids(records, layer, target_ids))
    if len(target_outlet_geometries) != len(target_ids):
        found = {
            int(row["attributes"]["etak_id"])
            for row in records
            if row["layer"] in ("E_203_vooluveekogu_a", "E_203_vooluveekogu_j")
            and int(row["attributes"]["etak_id"]) in target_ids
        }
        raise ValueError(f"missing target outlet ETAK IDs: {sorted(target_ids - found)}")
    target_outlet = _rasterize(target_outlet_geometries, shape, transform)
    water = _rasterize([*water_polygons, *water_lines], shape, transform)
    other_outlet = water & ~target_outlet
    vegetation_evidence = _rasterize(
        geometries["E_305_puittaimestik_a"], shape, transform
    )
    object_geometries = [
        *geometries["E_401_hoone_ka"],
        *geometries["E_402_korgrajatis_p"],
        *geometries["E_403_muu_rajatis_ka"],
        *geometries["E_403_muu_rajatis_p"],
        *geometries["E_501_tee_a"],
    ]
    object_mask = _rasterize(object_geometries, shape, transform)
    slopes = [
        row
        for row in records
        if row["layer"] == "E_102_nolv_j"
    ]
    protected_structure = _rasterize(
        [shapely.from_wkb(bytes.fromhex(row["geometry"]["wkb_hex"])) for row in slopes]
        + geometries["E_204_kaldajoon_j"],
        shape,
        transform,
    ) | authority["forbidden_morphology"]
    non_heightfield = _rasterize(
        [
            shapely.from_wkb(bytes.fromhex(row["geometry"]["wkb_hex"]))
            for row in slopes
            if row["attributes"].get("tyyp_t") == "Looduslik järsak"
        ],
        shape,
        transform,
    )
    return {
        "target_outlet": target_outlet,
        "other_outlet": other_outlet,
        "water": water | authority["unknown_bathymetry"],
        "object": object_mask,
        "vegetation_evidence": vegetation_evidence,
        "non_heightfield": non_heightfield,
        "protected_structure": protected_structure,
    }


def _condition_rasters(
    artifact_path: Path,
    *,
    kind: str,
    shape: tuple[int, int],
    transform: rasterio.Affine,
) -> dict[str, np.ndarray]:
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    coverage = np.zeros(shape, dtype=bool)
    unknown = np.zeros(shape, dtype=bool)
    primary = np.full(shape, -1, dtype=np.int32)
    secondary = np.full(shape, -1, dtype=np.int32)
    for index, row in enumerate(payload["features"]):
        geometry_key = "ogr_wkb_hex"
        geometry = shapely.force_2d(
            shapely.from_wkb(bytes.fromhex(row["geometry"][geometry_key]))
        )
        feature_mask = _rasterize([geometry], shape, transform)
        coverage |= feature_mask
        if kind == "geology":
            lithology = row["decoded"]["lithology"]
            genesis = row["decoded"]["genesis"]
            if (
                lithology["status"] != "decoded_from_frozen_official_domain"
                or genesis["status"] != "decoded_from_frozen_official_domain"
            ):
                unknown |= feature_mask
            else:
                primary[feature_mask] = int(lithology["code"])
                secondary[feature_mask] = int(genesis["code"])
        else:
            primary[feature_mask] = int(index)
    unknown |= ~coverage
    return {
        "coverage": coverage,
        "unknown": unknown,
        "primary": primary,
        "secondary": secondary,
    }


def _active_soil_semantics(
    payload: dict[str, Any],
    feature_index: np.ndarray,
    active: np.ndarray,
) -> dict[str, Any]:
    active_indices = sorted(
        int(value)
        for value in np.unique(feature_index[active])
        if int(value) >= 0
    )
    uncovered_cells = int((active & (feature_index < 0)).sum())
    allowed = {"parsed_complete_official_grammar", "missing"}
    rows: list[dict[str, Any]] = []
    complete = uncovered_cells == 0
    for index in active_indices:
        normalized = payload["features"][index]["normalized"]
        statuses = {
            field: str(normalized[field]["status"])
            for field in ("Loimis1", "Loimis2", "Huumus")
        }
        recognized = all(status in allowed for status in statuses.values())
        primary_texture_present = (
            statuses["Loimis1"] == "parsed_complete_official_grammar"
        )
        humus_present = statuses["Huumus"] == "parsed_complete_official_grammar"
        accepted = recognized and primary_texture_present and humus_present
        complete &= accepted
        rows.append(
            {
                "feature_index": index,
                "source_fid": payload["features"][index]["feature_identity"][
                    "source_fid"
                ],
                "statuses": statuses,
                "accepted": accepted,
            }
        )
    return {
        "complete": bool(complete),
        "policy": {
            "active_cells_require_source_polygon": True,
            "allowed_field_statuses": sorted(allowed),
            "primary_texture_profile_must_be_parsed": True,
            "humus_profile_must_be_parsed": True,
            "explicit_missing_secondary_texture_is_not_imputed": True,
            "missing_humus_is_unknown_not_absence": True,
            "unparseable_or_unknown_status_fails_closed": True,
        },
        "active_feature_indices": active_indices,
        "active_uncovered_cells": uncovered_cells,
        "active_features": rows,
    }


def _soil_support_mask(
    semantics: dict[str, Any],
    feature_index: np.ndarray,
) -> np.ndarray:
    accepted = {
        int(row["feature_index"])
        for row in semantics["active_features"]
        if row["accepted"]
    }
    if not accepted:
        return np.zeros(feature_index.shape, dtype=bool)
    return np.isin(feature_index, np.asarray(sorted(accepted), dtype=np.int32))


def _rgb_height(height: np.ndarray, domain: np.ndarray) -> np.ndarray:
    values = height[domain & np.isfinite(height)]
    low, high = np.percentile(values, [2.0, 98.0]) if values.size else (0.0, 1.0)
    scale = np.clip((height - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    rgb = np.stack(
        [
            35 + 190 * scale,
            55 + 160 * np.sqrt(scale),
            80 + 140 * (1.0 - scale),
        ],
        axis=-1,
    ).astype(np.uint8)
    rgb[~np.isfinite(height)] = (0, 0, 0)
    return rgb


def _topographic_seep_support(
    height: np.ndarray,
    *,
    solve_domain: np.ndarray,
    water: np.ndarray,
    object_mask: np.ndarray,
    protected_structure: np.ndarray,
    unknown: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return evidence support for a seep hypothesis, never a flux estimate."""
    fine_sigma_m = 4.0
    broad_sigma_m = 16.0
    drainage_decay_m = 48.0
    concavity_full_support_m = 0.5
    slope_rise_end = 0.05
    slope_fall_start = 0.25
    slope_fall_end = 0.75
    fine = ndimage.gaussian_filter(height, sigma=fine_sigma_m, mode="nearest")
    broad = ndimage.gaussian_filter(height, sigma=broad_sigma_m, mode="nearest")
    gradient_n, gradient_e = np.gradient(fine, _GRID_METERS, _GRID_METERS)
    slope = np.hypot(gradient_e, gradient_n)
    concavity = np.clip(
        (broad - fine) / concavity_full_support_m,
        0.0,
        1.0,
    )
    slope_rise = np.clip(slope / slope_rise_end, 0.0, 1.0)
    slope_fall = np.clip(
        (slope_fall_end - slope) / (slope_fall_end - slope_fall_start),
        0.0,
        1.0,
    )
    slope_foot = slope_rise * slope_fall
    drainage_distance = ndimage.distance_transform_edt(~water) * _GRID_METERS
    drainage_proximity = np.exp(-drainage_distance / drainage_decay_m)
    support = np.cbrt(concavity * slope_foot * drainage_proximity).astype(np.float32)
    abstain = (
        ~solve_domain
        | water
        | object_mask
        | protected_structure
        | unknown
    )
    support[abstain] = 0.0
    return support, {
        "semantics": "process_hypothesis_support_not_measured_hydrology_or_flux",
        "range": [0.0, 1.0],
        "units": "dimensionless",
        "abstention_value": 0.0,
        "derivation": "cuberoot(concavity_support * slope_foot_support * exp(-distance_to_exact_ETAK_water_or_drainage / 48m))",
        "accepted_height_gaussian_sigma_m": {
            "fine": fine_sigma_m,
            "broad": broad_sigma_m,
        },
        "concavity_support": {
            "definition": "clip((broad_height - fine_height) / 0.5m, 0, 1)",
            "full_support_m": concavity_full_support_m,
        },
        "slope_foot_support": {
            "definition": "clip(slope/0.05,0,1) * clip((0.75-slope)/(0.75-0.25),0,1)",
            "slope_units": "rise_over_run",
            "rise_end": slope_rise_end,
            "fall_start": slope_fall_start,
            "fall_end": slope_fall_end,
        },
        "drainage_distance": {
            "source": "exact_bound_ETAK_water_polygon_and_flowline_rasterization",
            "decay_m": drainage_decay_m,
        },
        "forced_abstention_masks": [
            "outside_solve_domain",
            "water",
            "object",
            "protected_structure",
            "unknown",
        ],
        "downstream_contract": "multiply only by separately derived material/profile seep-flux capacity; resulting flux remains a C1 hypothesis",
    }


def _write_png(path: Path, rgb: np.ndarray) -> None:
    image = Image.fromarray(rgb, mode="RGB")
    maximum = 1400
    if max(image.size) > maximum:
        scale = maximum / max(image.size)
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            resample=Image.Resampling.NEAREST,
        )
    image.save(path, format="PNG", optimize=True)


def _qa_images(
    root: Path,
    published_root: Path,
    ordinal: int,
    site_id: str,
    arrays: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    qa = root / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    masks = {
        name: arrays[name].astype(bool, copy=False)
        for name in (
            "solve_domain",
            "target_feature",
            "outlet",
            "water",
            "collar",
            "upstream_domain",
            "edge_leak",
            "vegetation_evidence",
            "protected_structure",
            "non_heightfield",
            "object",
            "unknown",
        )
    }
    base = _rgb_height(arrays["height"], masks["solve_domain"])
    base[masks["target_feature"]] = (255, 230, 30)
    base[masks["outlet"]] = (30, 170, 255)
    base[masks["water"]] = (10, 80, 190)
    images = [
        (
            f"{ordinal:02d}-{site_id}-height-outlet.png",
            base,
            "accepted corrected LOD0 1 m evidence height; yellow target, cyan outlet, blue water",
        )
    ]
    drainage = np.full((*arrays["height"].shape, 3), 24, dtype=np.uint8)
    drainage[masks["collar"]] = (105, 105, 105)
    drainage[masks["upstream_domain"]] = (45, 180, 80)
    drainage[masks["outlet"]] = (20, 145, 255)
    drainage[masks["edge_leak"]] = (255, 30, 20)
    images.append(
        (
            f"{ordinal + 1:02d}-{site_id}-drainage-domain.png",
            drainage,
            "green upstream basin, grey evidence collar, blue real ETAK outlet, red incomplete edge leak",
        )
    )
    typed = np.full((*arrays["height"].shape, 3), 238, dtype=np.uint8)
    typed[masks["vegetation_evidence"]] = (75, 150, 65)
    typed[masks["protected_structure"]] = (235, 150, 25)
    typed[masks["non_heightfield"]] = (170, 50, 170)
    typed[masks["object"]] = (210, 50, 40)
    typed[masks["water"]] = (30, 110, 220)
    typed[masks["unknown"]] = (20, 20, 20)
    images.append(
        (
            f"{ordinal + 2:02d}-{site_id}-typed-conditions.png",
            typed,
            "vegetation evidence green, protected structure orange, non-heightfield magenta, objects red, water blue, unknown black",
        )
    )
    seep = np.clip(arrays["topographic_seep_support_likelihood"], 0.0, 1.0)
    seep_rgb = np.stack(
        [
            12 + 35 * seep,
            18 + 205 * seep,
            24 + 225 * np.sqrt(seep),
        ],
        axis=-1,
    ).astype(np.uint8)
    seep_rgb[~masks["solve_domain"]] = (8, 8, 8)
    seep_rgb[masks["unknown"]] = (0, 0, 0)
    images.append(
        (
            f"{ordinal + 3:02d}-{site_id}-topographic-seep-support.png",
            seep_rgb,
            "dark-to-cyan process-hypothesis support from accepted DTM slope-foot concavity and exact ETAK drainage proximity; black is abstention, not zero measured flux",
        )
    )
    for name, rgb, interpretation in images:
        path = qa / name
        _write_png(path, rgb)
        with Image.open(path) as image:
            dimensions = [image.width, image.height]
        rows.append(
            {
                **_file_identity(
                    path,
                    published_path=published_root / "qa" / name,
                ),
                "dimensions": dimensions,
                "interpretation": interpretation,
            }
        )
    return rows


def _available_orthophoto(sheet: str) -> dict[str, Any]:
    result: dict[str, Any] = {"sheet": sheet, "rgb": None, "cir": None}
    for kind in ("rgb", "cir"):
        paths = sorted((DATA_IN / "orthophoto" / kind / sheet).glob("*/*.tif"))
        if len(paths) == 1:
            result[kind] = _file_identity(paths[0])
    result["complete"] = result["rgb"] is not None and result["cir"] is not None
    return result


def _site_artifacts(
    spec: SiteSpec,
    *,
    authority_root: Path,
    authority_manifest: dict[str, Any],
    accepted_materialization_path: Path,
    domain_snapshot_path: Path,
    forbidden_etak_ids: frozenset[int],
) -> dict[str, Any]:
    target = _read_target_geometry(spec)
    bbox_en = _site_bbox(target)
    records, geometries = _read_etak_features(
        bbox_en,
        target,
        forbidden_etak_ids,
    )
    authority, authority_tiles, missing_authority_tiles = _authority_tiles(
        authority_root, authority_manifest, bbox_en
    )
    corrected_height, corrected_lod0 = _corrected_lod0_window(
        accepted_materialization_path,
        authority_manifest,
        bbox_en,
    )
    authority["height"] = corrected_height
    authority["fine_mask_coverage"] = authority["valid"].copy()
    authority["valid"] = np.isfinite(corrected_height)
    transform = _transform(bbox_en)
    shape = authority["height"].shape
    masks = _mask_sources(spec, records, geometries, shape, transform, authority)
    drainage = delineate_drainage_domain(
        authority["height"],
        authority["valid"],
        masks["target_outlet"],
        masks["other_outlet"],
        collar_cells=_COLLAR_METERS,
    )
    enlarged_bbox_en = (
        bbox_en[0] - _ENLARGEMENT_METERS,
        bbox_en[1] - _ENLARGEMENT_METERS,
        bbox_en[2] + _ENLARGEMENT_METERS,
        bbox_en[3] + _ENLARGEMENT_METERS,
    )
    enlarged_records, enlarged_geometries = _read_etak_features(
        enlarged_bbox_en,
        target,
        forbidden_etak_ids,
    )
    enlarged_height, enlarged_corrected_lod0 = _corrected_lod0_window(
        accepted_materialization_path,
        authority_manifest,
        enlarged_bbox_en,
    )
    enlarged_authority = {
        "height": enlarged_height,
        "valid": np.ones(enlarged_height.shape, dtype=bool),
        "unknown_bathymetry": np.zeros(enlarged_height.shape, dtype=bool),
        "forbidden_morphology": np.zeros(enlarged_height.shape, dtype=bool),
    }
    enlarged_transform = _transform(enlarged_bbox_en)
    enlarged_shape = enlarged_authority["height"].shape
    enlarged_masks = _mask_sources(
        spec,
        enlarged_records,
        enlarged_geometries,
        enlarged_shape,
        enlarged_transform,
        enlarged_authority,
    )
    enlarged_drainage = delineate_drainage_domain(
        enlarged_authority["height"],
        enlarged_authority["valid"],
        enlarged_masks["target_outlet"],
        enlarged_masks["other_outlet"],
        collar_cells=_COLLAR_METERS,
    )
    target_feature = _rasterize([target], shape, transform)
    enlarged_target_feature = _rasterize(
        [target],
        enlarged_shape,
        enlarged_transform,
    )
    selection_source = {
        "kind": "erodible_slope_dtm_derived_physical_domain_bbox",
        "site_id": spec.site_id,
        "target_etak_id": spec.target_etak_id,
        "bbox_en": list(bbox_en),
        "grid_m": _GRID_METERS,
        "search_margin_m": _SEARCH_MARGIN_METERS,
        "minimum_evidence_collar_m": _COLLAR_METERS,
    }
    soil_path = extract_soil_window(
        bbox_en,
        name=f"erodible-slope-{spec.site_id}",
        selection_source=selection_source,
    )
    geology_path = extract_egt_surficial_window(
        bbox_en,
        name=f"erodible-slope-{spec.site_id}",
        domain_snapshot_path=domain_snapshot_path,
        selection_source=selection_source,
    )
    soil = _condition_rasters(
        soil_path, kind="soil", shape=shape, transform=transform
    )
    geology = _condition_rasters(
        geology_path, kind="geology", shape=shape, transform=transform
    )
    enlarged_selection_source = {
        **selection_source,
        "kind": "erodible_slope_dtm_derived_physical_domain_bbox_enlarged",
        "bbox_en": list(enlarged_bbox_en),
        "enlargement_m": _ENLARGEMENT_METERS,
    }
    enlarged_soil_path = extract_soil_window(
        enlarged_bbox_en,
        name=f"erodible-slope-{spec.site_id}-enlarged",
        selection_source=enlarged_selection_source,
    )
    enlarged_geology_path = extract_egt_surficial_window(
        enlarged_bbox_en,
        name=f"erodible-slope-{spec.site_id}-enlarged",
        domain_snapshot_path=domain_snapshot_path,
        selection_source=enlarged_selection_source,
    )
    enlarged_soil = _condition_rasters(
        enlarged_soil_path,
        kind="soil",
        shape=enlarged_shape,
        transform=enlarged_transform,
    )
    enlarged_geology = _condition_rasters(
        enlarged_geology_path,
        kind="geology",
        shape=enlarged_shape,
        transform=enlarged_transform,
    )
    soil_payload = json.loads(soil_path.read_text(encoding="utf-8"))
    enlarged_soil_payload = json.loads(
        enlarged_soil_path.read_text(encoding="utf-8")
    )
    soil_semantics = _active_soil_semantics(
        soil_payload,
        soil["primary"],
        drainage.solve_domain,
    )
    enlarged_soil_semantics = _active_soil_semantics(
        enlarged_soil_payload,
        enlarged_soil["primary"],
        enlarged_drainage.solve_domain,
    )
    soil_support = _soil_support_mask(soil_semantics, soil["primary"])
    enlarged_soil_support = _soil_support_mask(
        enlarged_soil_semantics,
        enlarged_soil["primary"],
    )
    unknown = (
        ~authority["valid"]
        | soil["unknown"]
        | geology["unknown"]
        | ~soil_support
        | drainage.edge_leak
    )
    seep_support, seep_support_derivation = _topographic_seep_support(
        authority["height"],
        solve_domain=drainage.solve_domain,
        water=masks["water"],
        object_mask=masks["object"],
        protected_structure=masks["protected_structure"],
        unknown=unknown,
    )
    enlarged_unknown = (
        ~enlarged_authority["valid"]
        | enlarged_soil["unknown"]
        | enlarged_geology["unknown"]
        | ~enlarged_soil_support
        | enlarged_drainage.edge_leak
    )
    enlarged_seep_support, enlarged_seep_support_derivation = (
        _topographic_seep_support(
            enlarged_authority["height"],
            solve_domain=enlarged_drainage.solve_domain,
            water=enlarged_masks["water"],
            object_mask=enlarged_masks["object"],
            protected_structure=enlarged_masks["protected_structure"],
            unknown=enlarged_unknown,
        )
    )
    arrays = {
        "height": authority["height"].astype(np.float32),
        "valid": authority["valid"].astype(np.uint8),
        "solve_domain": drainage.solve_domain.astype(np.uint8),
        "upstream_domain": drainage.upstream.astype(np.uint8),
        "outlet": drainage.outlet.astype(np.uint8),
        "collar": drainage.collar.astype(np.uint8),
        "edge_leak": drainage.edge_leak.astype(np.uint8),
        "fill_depth": drainage.fill_depth.astype(np.float32),
        "water": masks["water"].astype(np.uint8),
        "object": masks["object"].astype(np.uint8),
        "vegetation_evidence": masks["vegetation_evidence"].astype(np.uint8),
        "non_heightfield": masks["non_heightfield"].astype(np.uint8),
        "protected_structure": masks["protected_structure"].astype(np.uint8),
        "unknown": unknown.astype(np.uint8),
        "target_feature": target_feature.astype(np.uint8),
        "soil_feature_index": soil["primary"].astype(np.int32),
        "geology_lithology_code": geology["primary"].astype(np.int32),
        "geology_genesis_code": geology["secondary"].astype(np.int32),
        "topographic_seep_support_likelihood": seep_support,
    }
    soil_semantics_complete = bool(
        soil_semantics["complete"] and enlarged_soil_semantics["complete"]
    )
    target_material_support = {
        "normal_supported_cells": int((target_feature & soil_support).sum()),
        "normal_total_cells": int(target_feature.sum()),
        "enlarged_supported_cells": int(
            (enlarged_target_feature & enlarged_soil_support).sum()
        ),
        "enlarged_total_cells": int(enlarged_target_feature.sum()),
    }
    target_material_support["complete"] = bool(
        target_material_support["normal_total_cells"] > 0
        and target_material_support["normal_supported_cells"]
        == target_material_support["normal_total_cells"]
        and target_material_support["enlarged_supported_cells"]
        == target_material_support["enlarged_total_cells"]
    )
    blockers = []
    if drainage.edge_leak.any():
        blockers.append("upstream_basin_reaches_evidence_bbox_edge")
    if enlarged_drainage.edge_leak.any():
        blockers.append("upstream_basin_reaches_enlarged_evidence_bbox_edge")
    orthophoto = _available_orthophoto(spec.sheet)
    if not orthophoto["complete"]:
        blockers.append(f"dated_rgb_cir_missing_for_sheet_{spec.sheet}")
    if spec.role == "morphology_development" and not target_material_support["complete"]:
        blockers.append(f"{spec.site_id}_target_material_support_incomplete")
    facts = {
        "site_id": spec.site_id,
        "target_etak_id": spec.target_etak_id,
        "target_point_en": [spec.target_e, spec.target_n],
        "sheet": spec.sheet,
        "bbox_en": list(bbox_en),
        "shape": list(shape),
        "transform": [float(value) for value in transform[:6]],
        "crs": "EPSG:3301",
        "grid_m": _GRID_METERS,
        "collar_m": _COLLAR_METERS,
        "target_outlet_etak_ids": list(spec.target_outlet_ids),
        "development_role": spec.role,
        "target_material_support": target_material_support,
        "domain_policy": {
            "method": "conservative_fail_closed_priority_flood_outlet_partition_on_accepted_authority_1m_evidence_grid",
            "authority": "evidence_domain_delineation_only",
            "forbidden_uses": [
                "solver_water_routing",
                "generation_height_fill_or_breach",
                "form_placement",
                "amplitude_selection",
                "visible_boundary_authorization",
            ],
            "storage_chunk_edges_are_boundaries": False,
            "outer_evidence_edge_is_competing_outlet_and_edge_contact_fails_closed": True,
            "minimum_evidence_collar_m": _COLLAR_METERS,
            "later_c1_required_halo_must_not_exceed_collar": True,
            "candidate_crop_acceptance_gates": [
                "support_and_collar_proof",
                "domain_enlargement_invariance",
                "partition_invariance",
                "rotation_invariance",
            ],
            "enlarged_evidence_support": {
                "margin_added_on_every_side_m": _ENLARGEMENT_METERS,
                "role": "independent_support_for_downstream_output_crop_domain_enlargement_invariance_proof",
                "whole_window_identity_required": False,
            },
            "fill_depth_semantics": "delineation_diagnostic_only_never_generation_height_delta",
        },
        "counts": {
            "valid": int(authority["valid"].sum()),
            "upstream_domain": int(drainage.upstream.sum()),
            "outlet": int(drainage.outlet.sum()),
            "collar": int(drainage.collar.sum()),
            "solve_domain": int(drainage.solve_domain.sum()),
            "edge_leak": int(drainage.edge_leak.sum()),
            "water": int(masks["water"].sum()),
            "object": int(masks["object"].sum()),
            "vegetation_evidence": int(masks["vegetation_evidence"].sum()),
            "non_heightfield": int(masks["non_heightfield"].sum()),
            "protected_structure": int(masks["protected_structure"].sum()),
            "unknown": int(unknown.sum()),
            "soil_material_supported": int(soil_support.sum()),
        },
        "fill_depth_m": {
            "maximum": float(drainage.fill_depth.max()),
            "p99": float(np.percentile(drainage.fill_depth, 99.0)),
        },
        "authority_tiles": authority_tiles,
        "fine_structural_mask_coverage": {
            "covered_cells": int(authority["fine_mask_coverage"].sum()),
            "total_cells": int(authority["fine_mask_coverage"].size),
            "missing_lod_minus_2_tiles": missing_authority_tiles,
            "missing_semantics": "local_fine_structural_masks_unavailable; accepted_corrected_or_canonical_lod0_height_and_national_typed_masks_remain_authoritative",
        },
        "corrected_lod0": corrected_lod0,
        "enlarged_domain": {
            "bbox_en": list(enlarged_bbox_en),
            "shape": list(enlarged_shape),
            "transform": [float(value) for value in enlarged_transform[:6]],
            "grid_m": _GRID_METERS,
            "collar_m": _COLLAR_METERS,
            "corrected_lod0": enlarged_corrected_lod0,
            "counts": {
                "valid": int(enlarged_authority["valid"].sum()),
                "upstream_domain": int(enlarged_drainage.upstream.sum()),
                "outlet": int(enlarged_drainage.outlet.sum()),
                "collar": int(enlarged_drainage.collar.sum()),
                "solve_domain": int(enlarged_drainage.solve_domain.sum()),
                "edge_leak": int(enlarged_drainage.edge_leak.sum()),
                "unknown": int(enlarged_unknown.sum()),
                "soil_material_supported": int(enlarged_soil_support.sum()),
            },
            "soil_window": _file_identity(enlarged_soil_path),
            "soil_semantics": enlarged_soil_semantics,
            "geology_window": _file_identity(enlarged_geology_path),
            "topographic_seep_support_likelihood": enlarged_seep_support_derivation,
        },
        "soil_window": _file_identity(soil_path),
        "soil_semantics": soil_semantics,
        "soil_semantics_complete": soil_semantics_complete,
        "geology_window": _file_identity(geology_path),
        "orthophoto": orthophoto,
        "topographic_seep_support_likelihood": seep_support_derivation,
        "etak_records": records,
        "enlarged_etak_records": enlarged_records,
        "blockers": sorted(set(blockers)),
    }
    enlarged_arrays = {
        "height": enlarged_authority["height"].astype(np.float32),
        "valid": enlarged_authority["valid"].astype(np.uint8),
        "solve_domain": enlarged_drainage.solve_domain.astype(np.uint8),
        "upstream_domain": enlarged_drainage.upstream.astype(np.uint8),
        "outlet": enlarged_drainage.outlet.astype(np.uint8),
        "collar": enlarged_drainage.collar.astype(np.uint8),
        "edge_leak": enlarged_drainage.edge_leak.astype(np.uint8),
        "water": enlarged_masks["water"].astype(np.uint8),
        "protected_structure": enlarged_masks["protected_structure"].astype(np.uint8),
        "object": enlarged_masks["object"].astype(np.uint8),
        "vegetation_evidence": enlarged_masks["vegetation_evidence"].astype(np.uint8),
        "non_heightfield": enlarged_masks["non_heightfield"].astype(np.uint8),
        "unknown": enlarged_unknown.astype(np.uint8),
        "target_feature": enlarged_target_feature.astype(np.uint8),
        "unknown_bathymetry": enlarged_authority["unknown_bathymetry"].astype(np.uint8),
        "soil_feature_index": enlarged_soil["primary"].astype(np.int32),
        "geology_lithology_code": enlarged_geology["primary"].astype(np.int32),
        "geology_genesis_code": enlarged_geology["secondary"].astype(np.int32),
        "topographic_seep_support_likelihood": enlarged_seep_support,
    }
    return {
        "facts": facts,
        "arrays": arrays,
        "enlarged_domain_arrays": enlarged_arrays,
    }


def materialize_condition_bundle(
    *,
    preregistration_path: Path,
    site_config_path: Path,
    authority_manifest_path: Path,
    accepted_materialization_path: Path,
    domain_snapshot_path: Path,
    output_root: Path | None = None,
) -> Path:
    _load_preregistration(preregistration_path)
    site_specs, forbidden_etak_ids, _site_config = _load_site_config(site_config_path)
    authority_manifest = json.loads(authority_manifest_path.read_text(encoding="utf-8"))
    if authority_manifest.get("role") != "structural_authority_0.25m":
        raise ValueError("input is not the accepted 0.25 m structural authority")
    authority_root = authority_manifest_path.parent
    inputs = {
        "preregistration": _file_identity(preregistration_path),
        "site_config": _file_identity(site_config_path),
        "accepted_structural_authority": _file_identity(authority_manifest_path),
        "accepted_structural_authority_recipe_sha256": authority_manifest["recipeSha256"],
        "accepted_corrected_materialization": _file_identity(
            accepted_materialization_path
        ),
        "egt_domain_snapshot": _file_identity(domain_snapshot_path),
        "etak": _file_identity(_ETAK),
        "implementation": _implementation_identity(),
        "runtime": _runtime_identity(),
    }
    site_results = [
        _site_artifacts(
            spec,
            authority_root=authority_root,
            authority_manifest=authority_manifest,
            accepted_materialization_path=accepted_materialization_path,
            domain_snapshot_path=domain_snapshot_path,
            forbidden_etak_ids=forbidden_etak_ids,
        )
        for spec in site_specs
    ]
    recipe = {
        "schema_version": f"{_SCHEMA_VERSION}.recipe",
        "inputs": inputs,
        "sites": [
            {
                "site_id": result["facts"]["site_id"],
                "development_role": result["facts"]["development_role"],
                "target_etak_id": result["facts"]["target_etak_id"],
                "output_chunk": (
                    list(spec.output_chunk) if spec.output_chunk is not None else None
                ),
                "target_material_support": result["facts"]["target_material_support"],
                "bbox_en": result["facts"]["bbox_en"],
                "enlarged_domain_bbox_en": result["facts"]["enlarged_domain"]["bbox_en"],
                "authority_tiles": result["facts"]["authority_tiles"],
                "soil_window": result["facts"]["soil_window"],
                "geology_window": result["facts"]["geology_window"],
                "orthophoto": result["facts"]["orthophoto"],
            }
            for spec, result in zip(site_specs, site_results, strict=True)
        ],
        "policies": {
            "development_only": True,
            "consumed_orajogi_used": False,
            "sealed_ood_used": False,
            "forbidden_etak_ids": sorted(forbidden_etak_ids),
            "runnable_morphology_site_ids": sorted(
                result["facts"]["site_id"]
                for result in site_results
                if result["facts"]["development_role"] == "morphology_development"
            ),
            "solver_excluded_site_ids": sorted(
                result["facts"]["site_id"]
                for result in site_results
                if result["facts"]["development_role"] != "morphology_development"
            ),
            "condition_grid_m": _GRID_METERS,
            "search_margin_m": _SEARCH_MARGIN_METERS,
            "minimum_evidence_collar_m": _COLLAR_METERS,
            "all_touched_vector_rasterization": True,
            "height_authority": "exact_accepted_corrected_lod0_decoded_EH2000_centers",
            "fine_mask_authority": "conservative_any_or_all_over_exact_4x4_0.25m_structural_authority_cells_for_normal_window",
        },
    }
    recipe_sha256 = hashlib.sha256(_canonical_bytes(recipe)).hexdigest()
    parent = (
        Path(output_root)
        if output_root is not None
        else DATA_WORK
        / "microtopography"
        / "erodible-slope"
        / "conditions"
        / "sha256"
    )
    root = parent / recipe_sha256
    bundle_path = root / "bundle.json"
    if root.exists():
        if bundle_path.is_file():
            return bundle_path
        raise RuntimeError(f"incomplete condition bundle exists: {root}")
    temporary = parent / f".{recipe_sha256}.{os.getpid()}.tmp"
    if temporary.exists():
        raise RuntimeError(f"stale condition bundle temporary exists: {temporary}")
    temporary.mkdir(parents=True)

    site_rows = []
    qa_rows: list[dict[str, Any]] = []
    for index, result in enumerate(site_results):
        facts = result["facts"]
        arrays = result["arrays"]
        site_id = facts["site_id"]
        arrays_path = temporary / f"{site_id}.npz"
        np.savez_compressed(arrays_path, **arrays)
        enlarged_domain_path = temporary / f"{site_id}-domain-enlarged.npz"
        np.savez_compressed(
            enlarged_domain_path,
            **result["enlarged_domain_arrays"],
        )
        etak_path = temporary / f"{site_id}-etak.json"
        etak_path.write_text(
            json.dumps(
                {
                    "schema_version": "laas.erodible-slope-etak-window/1",
                    "site_id": site_id,
                    "source": inputs["etak"],
                    "bbox_en": facts["bbox_en"],
                    "rasterization": "exact_source_geometry_all_touched_on_bound_1m_grid",
                    "records": facts.pop("etak_records"),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        enlarged_etak_path = temporary / f"{site_id}-etak-enlarged.json"
        enlarged_etak_path.write_text(
            json.dumps(
                {
                    "schema_version": "laas.erodible-slope-etak-window/1",
                    "site_id": site_id,
                    "source": inputs["etak"],
                    "bbox_en": facts["enlarged_domain"]["bbox_en"],
                    "rasterization": "exact_source_geometry_all_touched_on_bound_1m_grid",
                    "records": facts.pop("enlarged_etak_records"),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        site_qa = _qa_images(
            temporary,
            root,
            index * 4 + 1,
            site_id,
            arrays,
        )
        qa_rows.extend(site_qa)
        site_rows.append(
            {
                **facts,
                "array_schema_version": _SITE_ARRAY_SCHEMA,
                "arrays": _file_identity(
                    arrays_path,
                    published_path=root / arrays_path.name,
                ),
                "etak_window": _file_identity(
                    etak_path,
                    published_path=root / etak_path.name,
                ),
                "enlarged_domain_arrays": _file_identity(
                    enlarged_domain_path,
                    published_path=root / enlarged_domain_path.name,
                ),
                "enlarged_etak_window": _file_identity(
                    enlarged_etak_path,
                    published_path=root / enlarged_etak_path.name,
                ),
                "qa": site_qa,
            }
        )
    qa_index = {
        "schema_version": "laas.erodible-slope-condition-qa-index/1",
        "recipe_sha256": recipe_sha256,
        "images": qa_rows,
    }
    qa_index_path = temporary / "qa" / "index.json"
    qa_index_path.write_text(
        json.dumps(qa_index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    blockers = sorted(
        {
            blocker
            for row in site_rows
            for blocker in row["blockers"]
        }
        | {
            "complete_recipe_freeze_and_run_required_before_opening_selected_sealed_ood2_9688702"
        }
    )
    bundle = {
        "schema_version": _SCHEMA_VERSION,
        "recipe_sha256": recipe_sha256,
        "authority": {
            "research_only": True,
            "r0_development_only": True,
            "target_truth": False,
            "production_owner": None,
            "estonia_transfer": "none",
            "latest_eligible": False,
            "runtime_synthesis_allowed": False,
        },
        "status": {
            "ready_for_r0_input_freeze": not any(
                blocker
                for blocker in blockers
                if "selected_sealed_ood2" not in blocker
            ),
            "ready_for_recipe_freeze_or_preview": False,
            "blockers": blockers,
        },
        "recipe": recipe,
        "site_array_fields": [
            "height",
            "valid",
            "solve_domain",
            "upstream_domain",
            "outlet",
            "collar",
            "edge_leak",
            "fill_depth",
            "water",
            "object",
            "vegetation_evidence",
            "non_heightfield",
            "protected_structure",
            "unknown",
            "target_feature",
            "soil_feature_index",
            "geology_lithology_code",
            "geology_genesis_code",
            "topographic_seep_support_likelihood",
        ],
        "enlarged_domain_array_fields": [
            "height",
            "valid",
            "solve_domain",
            "upstream_domain",
            "outlet",
            "collar",
            "edge_leak",
            "water",
            "protected_structure",
            "unknown_bathymetry",
            "object",
            "vegetation_evidence",
            "non_heightfield",
            "unknown",
            "target_feature",
            "soil_feature_index",
            "geology_lithology_code",
            "geology_genesis_code",
            "topographic_seep_support_likelihood",
        ],
        "sites": site_rows,
        "qa_index": _file_identity(
            qa_index_path,
            published_path=root / "qa" / "index.json",
        ),
        "limitations": [
            "This bundle contains condition and domain evidence only; it emits no synthetic height.",
            "The 1 m drainage grid organizes a later solve and is not morphology truth.",
            "Priority flood is restricted to fail-closed evidence-domain delineation and may not route solver water, alter generation height, place forms, set amplitudes, or authorize a visible boundary.",
            "Any generated crop must independently pass support/collar, domain-enlargement, partition, and rotation invariance gates.",
            "Vegetation polygons are evidence of vegetation presence, not individual object footprints.",
            "Mapped geology is regional conditioning and its boundaries are not synthesis seams.",
            "Orajogi ETAK 9688685 is consumed and disqualified; replacement OOD2 ETAK 9688702 is identity-frozen by the bound selection and remains sealed and unmaterialized.",
        ],
    }
    bundle_path_temporary = temporary / "bundle.json"
    bundle_path_temporary.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(root)
    return bundle_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--site-config", type=Path, required=True)
    parser.add_argument("--authority-manifest", type=Path, required=True)
    parser.add_argument("--accepted-materialization", type=Path, required=True)
    parser.add_argument("--egt-domain-snapshot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path = materialize_condition_bundle(
        preregistration_path=args.preregistration,
        site_config_path=args.site_config,
        authority_manifest_path=args.authority_manifest,
        accepted_materialization_path=args.accepted_materialization,
        domain_snapshot_path=args.egt_domain_snapshot,
        output_root=args.output_root,
    )
    print(path)


if __name__ == "__main__":
    main()
