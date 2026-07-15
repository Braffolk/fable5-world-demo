"""Full-fidelity windows from the EGT 1:200,000 surficial geology source."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyogrio
import pyogrio.raw
import shapely

from ....config import ASSET_GEN_ROOT, DATA_WORK, load_base
from ....grid import ChunkId, chunk_bounds_en
from .domains import EgtSurficialDomains, load_egt_surficial_domains

_SCHEMA_VERSION = "laas.egt-surficial-200k-window/1"
_LAYER = "q_avamus_a_200t"
_FIELDS = ("Shape_Leng", "Shape_Area", "lito200", "genees200", "GlobalID")
_REQUIRED_SOURCE_COMPONENTS = (
    ".shp",
    ".shx",
    ".dbf",
    ".prj",
    ".cpg",
    ".sbn",
    ".sbx",
    ".shp.xml",
)
_SOURCE = (
    ASSET_GEN_ROOT.parent
    / "docs"
    / "deep-research"
    / "microtopography-generation"
    / "library"
    / "data"
    / "egt"
    / "pinnakate-200k"
    / "q_avamus_a_200t.shp"
)
_EXPECTED_SCHEMA = {
    "layer_name": _LAYER,
    "crs": "EPSG:3301",
    "encoding": "UTF-8",
    "geometry_type": "Polygon Z",
    "features": 9176,
    "fields": list(_FIELDS),
    "dtypes": ["float64", "float64", "int32", "int32", "object"],
    "ogr_types": ["OFTReal", "OFTReal", "OFTInteger", "OFTInteger", "OFTString"],
    "driver": "ESRI Shapefile",
    "total_bounds": [
        369548.2049999982,
        6377140.684,
        739113.9689999968,
        6617849.331999999,
    ],
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


def _implementation_sha256() -> str:
    return _sha256_file(Path(__file__))


def _runtime_provenance() -> dict[str, Any]:
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "byteorder": sys.byteorder,
        },
        "numpy": np.__version__,
        "pyogrio": pyogrio.__version__,
        "gdal": list(pyogrio.__gdal_version__),
        "gdal_geos": list(pyogrio.__gdal_geos_version__),
        "shapely": shapely.__version__,
        "geos": shapely.geos_version_string,
    }


def _source_bundle(source: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for suffix in _REQUIRED_SOURCE_COMPONENTS:
        path = source.with_suffix(suffix)
        if not path.is_file():
            raise FileNotFoundError(f"EGT source component is absent: {path}")
        rows.append(
            {
                "path": _repo_relative(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "role": "retained_source_bundle_component",
            }
        )
    return rows


def _source_schema(source: Path) -> dict[str, Any]:
    info = pyogrio.read_info(source, layer=_LAYER)
    actual = {
        "layer_name": str(info["layer_name"]),
        "crs": str(info["crs"]),
        "encoding": str(info["encoding"]),
        "geometry_type": str(info["geometry_type"]),
        "features": int(info["features"]),
        "fields": [str(value) for value in info["fields"]],
        "dtypes": [str(value) for value in info["dtypes"]],
        "ogr_types": [str(value) for value in info["ogr_types"]],
        "driver": str(info["driver"]),
        "total_bounds": [float(value) for value in info["total_bounds"]],
    }
    if actual != _EXPECTED_SCHEMA:
        raise ValueError(
            "EGT 1:200,000 surficial source schema differs from the frozen contract: "
            f"expected={_EXPECTED_SCHEMA}, actual={actual}"
        )
    return actual


def _validate_bbox(bbox_en: tuple[float, float, float, float]) -> None:
    values = np.asarray(bbox_en, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("geology bbox must contain four finite EPSG:3301 values")
    if values[0] >= values[2] or values[1] >= values[3]:
        raise ValueError("geology bbox has non-positive extent")


def _validate_name(name: str) -> None:
    if not name or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in name
    ):
        raise ValueError("geology window name must be lowercase ASCII slug text")


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported EGT attribute value: {value!r}")


def _decode(
    domains: EgtSurficialDomains,
    domain_name: str,
    raw_code: Any,
) -> dict[str, Any]:
    if raw_code is None:
        return {"status": "missing", "code": None, "name": None}
    if isinstance(raw_code, bool) or not isinstance(raw_code, int):
        return {
            "status": "invalid_non_integer_source_code",
            "code": raw_code,
            "name": None,
        }
    try:
        name = domains.decode(domain_name, raw_code)
    except KeyError:
        return {
            "status": "unknown_code_not_in_frozen_domain",
            "code": raw_code,
            "name": None,
        }
    return {
        "status": "decoded_from_frozen_official_domain",
        "code": raw_code,
        "name": name,
    }


def extract_egt_surficial_window(
    bbox_en: tuple[float, float, float, float],
    *,
    name: str,
    domain_snapshot_path: Path,
    selection_source: dict[str, Any] | None = None,
    output_root: Path | None = None,
) -> Path:
    """Extract one immutable window while retaining each full source feature."""
    _validate_bbox(bbox_en)
    _validate_name(name)
    if not _SOURCE.is_file():
        raise FileNotFoundError(f"EGT surficial source is absent: {_SOURCE}")

    domains = load_egt_surficial_domains(domain_snapshot_path)
    source_bundle = _source_bundle(_SOURCE)
    source_schema = _source_schema(_SOURCE)
    domain_source = {
        "path": _repo_relative(domains.snapshot_path),
        "bytes": domains.snapshot_path.stat().st_size,
        "sha256": domains.snapshot_sha256,
    }
    recipe = {
        "schema_version": f"{_SCHEMA_VERSION}.recipe",
        "name": name,
        "bbox_en": [float(value) for value in bbox_en],
        "bbox_crs": "EPSG:3301",
        "selection_source": selection_source,
        "source_bundle": source_bundle,
        "source_schema": source_schema,
        "domain_snapshot": domain_source,
        "implementation_sha256": _implementation_sha256(),
        "runtime_provenance": _runtime_provenance(),
        "geometry_policy": (
            "full_unclipped_ogr_wkb_for_every_intersecting_source_feature"
        ),
        "decode_policy": (
            "decode exact raw codes only through the bound frozen official domain snapshot; "
            "preserve missing, invalid, and unknown codes explicitly"
        ),
    }
    recipe_sha256 = hashlib.sha256(_canonical_bytes(recipe)).hexdigest()
    parent = (
        Path(output_root)
        if output_root is not None
        else DATA_WORK
        / "terrain"
        / "conditions"
        / "geology"
        / "egt-surficial-200k"
        / "sha256"
    )
    root = parent / recipe_sha256
    artifact_path = root / "window.json"
    root_exists = root.exists()
    if root_exists and (
        not artifact_path.is_file()
        or {path.name for path in root.iterdir()} != {"window.json"}
    ):
        raise RuntimeError(f"incomplete EGT geology artifact exists: {root}")

    metadata, fids, geometry_wkb, columns = pyogrio.raw.read(
        _SOURCE,
        layer=_LAYER,
        bbox=tuple(float(value) for value in bbox_en),
        columns=list(_FIELDS),
        return_fids=True,
    )
    if [str(value) for value in metadata["fields"]] != list(_FIELDS):
        raise ValueError("EGT window reader field order differs from frozen schema")
    if fids is None or geometry_wkb is None:
        raise ValueError("EGT source driver did not return FID and geometry identity")
    if _source_bundle(_SOURCE) != source_bundle:
        raise RuntimeError("EGT source bundle changed during extraction")
    if _sha256_file(domains.snapshot_path) != domains.snapshot_sha256:
        raise RuntimeError("EGT domain snapshot changed during extraction")

    query = shapely.box(*bbox_en)
    query_area = float(query.area)
    order = np.argsort(fids, kind="stable")
    features: list[dict[str, Any]] = []
    intersections: list[Any] = []
    decode_status_counts: dict[str, int] = {}
    for source_index in order:
        fid = int(fids[source_index])
        raw_wkb = geometry_wkb[source_index]
        if raw_wkb is None:
            raise ValueError(f"EGT source FID {fid} has null geometry")
        wkb_bytes = bytes(raw_wkb)
        geometry = shapely.from_wkb(wkb_bytes)
        if geometry is None or geometry.is_empty:
            raise ValueError(f"EGT source FID {fid} has empty geometry")
        intersection = geometry.intersection(query)
        if intersection.is_empty:
            raise ValueError(f"spatial reader returned non-intersecting EGT FID {fid}")
        intersections.append(intersection)
        raw = {
            field: _json_scalar(columns[index][source_index])
            for index, field in enumerate(_FIELDS)
        }
        decoded = {
            "lithology": _decode(domains, "Q_Litoloogia_200", raw["lito200"]),
            "genesis": _decode(domains, "Q_Genees_200", raw["genees200"]),
        }
        for value in decoded.values():
            status = value["status"]
            decode_status_counts[status] = decode_status_counts.get(status, 0) + 1
        wkb_sha256 = hashlib.sha256(wkb_bytes).hexdigest()
        geometry_identity = hashlib.sha256(
            _canonical_bytes(
                {
                    "source_bundle": source_bundle,
                    "source_fid": fid,
                    "ogr_wkb_sha256": wkb_sha256,
                }
            )
        ).hexdigest()
        features.append(
            {
                "feature_identity": {
                    "source_fid": fid,
                    "source_fid_semantics": "driver_fid_bound_to_exact_source_bundle",
                    "global_id": raw["GlobalID"],
                    "geometry_identity_sha256": geometry_identity,
                },
                "geometry": {
                    "encoding": "OGC_WKB_hex_from_OGR_without_clipping",
                    "ogr_wkb_hex": wkb_bytes.hex(),
                    "ogr_wkb_sha256": wkb_sha256,
                    "type": geometry.geom_type,
                    "has_z": bool(geometry.has_z),
                    "bounds_en": [float(value) for value in geometry.bounds],
                    "area_m2": float(geometry.area),
                    "is_valid": bool(geometry.is_valid),
                },
                "window_intersection": {
                    "area_m2": float(intersection.area),
                    "fraction_of_query_bbox": float(intersection.area / query_area),
                },
                "raw_attributes": raw,
                "decoded": decoded,
            }
        )

    union_covered = (
        float(shapely.union_all(intersections).area) if intersections else 0.0
    )
    sum_intersections = float(sum(value.area for value in intersections))
    artifact = {
        "schema_version": _SCHEMA_VERSION,
        "status": "full_fidelity_source_window_snapshot",
        "recipe_sha256": recipe_sha256,
        "recipe": recipe,
        "query": {
            "name": name,
            "bbox_en": [float(value) for value in bbox_en],
            "crs": "EPSG:3301",
            "area_m2": query_area,
        },
        "coverage": {
            "intersecting_source_features": len(features),
            "sum_intersection_area_m2": sum_intersections,
            "union_covered_area_m2": union_covered,
            "union_coverage_fraction": union_covered / query_area,
            "overlap_area_m2": max(0.0, sum_intersections - union_covered),
            "uncovered_area_m2": max(0.0, query_area - union_covered),
        },
        "decode_status_counts": dict(sorted(decode_status_counts.items())),
        "features": features,
        "limitations": [
            "EGT 1:200,000 polygons are regional conditioning evidence, not fine geometry.",
            "Mapped polygon boundaries are uncertain and must not become hard synthesis seams.",
            "Shapefile FIDs identify features only with the exact retained source bundle.",
            "Decoded labels come only from the exact frozen official domain snapshot.",
            "Missing, invalid, and unknown codes remain explicit and are never guessed.",
        ],
    }
    if root_exists:
        existing = json.loads(artifact_path.read_text(encoding="utf-8"))
        if existing != artifact:
            raise RuntimeError(
                f"existing EGT geology artifact fails reconstruction: {artifact_path}"
            )
        return artifact_path

    temporary = parent / f".{recipe_sha256}.{os.getpid()}.tmp"
    if temporary.exists():
        raise RuntimeError(f"stale EGT geology temporary exists: {temporary}")
    temporary.mkdir(parents=True)
    (temporary / "window.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(root)
    return artifact_path


def extract_egt_surficial_chunk(
    chunk: ChunkId,
    *,
    name: str,
    domain_snapshot_path: Path,
    inspection_policy: str = "not_recorded",
    output_root: Path | None = None,
) -> Path:
    """Extract the exact footprint of a chunk from the canonical world grid."""
    grid = load_base().grid
    bounds = tuple(float(value) for value in chunk_bounds_en(grid, chunk))
    selection_source = {
        "kind": "canonical_world_grid_chunk_footprint_without_payload_apron",
        "chunk": {"lod": chunk.lod, "cx": chunk.cx, "cz": chunk.cz},
        "grid": {
            "anchor_e": grid.anchor_e,
            "anchor_n": grid.anchor_n,
            "chunk_m": grid.chunk_m,
            "lod_step": grid.lod_step,
            "chunk_res": grid.chunk_res,
        },
        "inspection_policy": inspection_policy,
    }
    return extract_egt_surficial_window(
        bounds,
        name=name,
        domain_snapshot_path=domain_snapshot_path,
        selection_source=selection_source,
        output_root=output_root,
    )
