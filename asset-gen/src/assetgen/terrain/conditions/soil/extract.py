"""Content-addressed full-fidelity Mullastikukaart window extraction."""
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
import pyproj
import shapely

from ....config import ASSET_GEN_ROOT, CONFIG_DIR, DATA_IN, DATA_WORK
from .profile import parse_humus_profile, parse_texture_profile
from .schema import (
    FIELDS,
    LAYER,
    json_scalar,
    load_tables,
    normalize_lihtloimis,
    normalize_siffer,
    normalize_soil_code,
    normalize_stoniness,
    schema_snapshot,
)

_SCHEMA_VERSION = "laas.mullastikukaart-window/1"
_REQUIRED_SOURCE_COMPONENTS = (".shp", ".shx", ".dbf", ".prj", ".cpg")
_OPTIONAL_SOURCE_COMPONENTS = (".sbn", ".sbx", ".shp.xml")
_PROFILE_AUTHORITIES = (
    {
        "path": "data/in/soil/mullalegend.pdf",
        "sha256": "69602b8fb9e0c3e99d53df32e7caa7aa5f534a91f430f2d185e0017e92502a59",
        "pages": [1, 2],
        "sections": [
            "Lõimis",
            "Mulla kores",
            "Lõimisevalem ja mulla uurimissügavus",
            "Turba ja metsakõdu lagunemisastmed",
            "Huumuslikud ja turbahorisondid ning metsakõdu",
        ],
        "role": "normative_map_formula_legend",
    },
    {
        "path": "data/in/soil/mullakaardi_seletuskiri.pdf",
        "sha256": "5db90f4db24c51cdc1860689b95cabb3590197852cd6636c6914fbd421289ae5",
        "pages": [6, 9, 10, 11],
        "sections": [
            "Digitaalses andmebaasis on olemas andmed",
            "V Mullastiku kaardile märgitavad mullaomaduste näitajad ja nende määramine välitöödel",
            "2. Mulla lõimis (mehaaniline koostis)",
            "4. Mulla huumuslike horisontide määramine",
        ],
        "role": "measurement_and_database_semantics",
    },
    {
        "path": "data/in/soil/mullakaart/Mullakaart.shp.xml",
        "sha256": "66e608361f799abf7402d4a60b29c048364f584e03004b6b06c19511abdc88da",
        "sections": ["eainfo/detailed/attr", "Esri/lineage"],
        "role": "delivered_field_and_lineage_metadata",
    },
)


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


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode("ascii") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


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
        "pyproj": pyproj.__version__,
        "proj": pyproj.proj_version_str,
    }


def _validate_bbox(bbox_en: tuple[float, float, float, float]) -> None:
    values = np.asarray(bbox_en, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("soil condition bbox must contain four finite EPSG:3301 values")
    if values[0] >= values[2] or values[1] >= values[3]:
        raise ValueError("soil condition bbox has non-positive extent")


def load_window_selection(
    path: Path,
) -> tuple[str, tuple[float, float, float, float], dict[str, Any]]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {"schema_version", "name", "bbox_en", "crs"}
    if set(payload) != expected:
        raise ValueError(
            f"soil window selection fields differ: missing={sorted(expected - set(payload))}, "
            f"extra={sorted(set(payload) - expected)}"
        )
    if payload["schema_version"] != "laas.mullastikukaart-window-selection/1":
        raise ValueError("unsupported Mullastikukaart window selection schema")
    if payload["crs"] != "EPSG:3301":
        raise ValueError("Mullastikukaart window selection must use EPSG:3301")
    bbox = tuple(float(value) for value in payload["bbox_en"])
    _validate_bbox(bbox)
    try:
        relative = str(path.resolve().relative_to(ASSET_GEN_ROOT))
    except ValueError:
        relative = str(path.resolve())
    selection_source = {
        "kind": "named_config",
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    return str(payload["name"]), bbox, selection_source


def _source_bundle(source: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for suffix in _REQUIRED_SOURCE_COMPONENTS:
        path = source.with_suffix(suffix)
        if not path.is_file():
            raise FileNotFoundError(f"Mullastikukaart component is absent: {path}")
        rows.append(
            {
                "path": str(path.relative_to(ASSET_GEN_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "role": "required_geometry_attribute_crs_or_encoding_component",
            }
        )
    for suffix in _OPTIONAL_SOURCE_COMPONENTS:
        path = source.with_suffix(suffix)
        if path.is_file():
            rows.append(
                {
                    "path": str(path.relative_to(ASSET_GEN_ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                    "role": "retained_spatial_index_or_source_metadata_component",
                }
            )
    return rows


def _table_sources() -> list[dict[str, Any]]:
    rows = []
    for name in (
        "soil-types.toml",
        "soil-texture.toml",
        "conditions/mullastikukaart-official-soil-authority.json",
    ):
        path = CONFIG_DIR / name
        rows.append(
            {
                "path": str(path.relative_to(ASSET_GEN_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return rows


def _profile_authority_sources() -> list[dict[str, Any]]:
    rows = []
    for specification in _PROFILE_AUTHORITIES:
        path = ASSET_GEN_ROOT / specification["path"]
        if not path.is_file():
            raise FileNotFoundError(f"soil profile authority is absent: {path}")
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != specification["sha256"]:
            raise ValueError(f"soil profile authority changed: {path}")
        rows.append(
            {
                **specification,
                "bytes": path.stat().st_size,
            }
        )
    return rows


def extract_soil_window(
    bbox_en: tuple[float, float, float, float],
    *,
    name: str,
    selection_source: dict[str, Any] | None = None,
    output_root: Path | None = None,
) -> Path:
    _validate_bbox(bbox_en)
    if not name or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in name
    ):
        raise ValueError("soil condition name must be lowercase ASCII slug text")
    source = DATA_IN / "soil" / "mullakaart" / "Mullakaart.shp"
    if not source.is_file():
        raise FileNotFoundError(f"Mullastikukaart source is absent: {source}")
    source_bundle = _source_bundle(source)
    table_sources = _table_sources()
    profile_authorities = _profile_authority_sources()
    source_schema = schema_snapshot(source)
    recipe = {
        "schema_version": f"{_SCHEMA_VERSION}.recipe",
        "name": name,
        "bbox_en": [float(value) for value in bbox_en],
        "bbox_crs": "EPSG:3301",
        "selection_source": selection_source,
        "source_bundle": source_bundle,
        "source_schema": source_schema,
        "normalization_tables": table_sources,
        "profile_semantics_authorities": profile_authorities,
        "implementation_sha256": _implementation_sha256(),
        "runtime_provenance": _runtime_provenance(),
        "geometry_policy": "full_unclipped_ogr_wkb_for_every_intersecting_source_feature",
        "normalization_policy": (
            "strict official grammar and exact table entries only; missing and "
            "unparseable values remain explicit with residual source spans"
        ),
    }
    recipe_sha256 = hashlib.sha256(_canonical_bytes(recipe)).hexdigest()
    parent = (
        Path(output_root)
        if output_root is not None
        else DATA_WORK
        / "terrain"
        / "conditions"
        / "soil"
        / "mullastikukaart"
        / "sha256"
    )
    root = parent / recipe_sha256
    artifact_path = root / "window.json"
    root_exists = root.exists()
    if root_exists and (
        not artifact_path.is_file()
        or {path.name for path in root.iterdir()} != {"window.json"}
    ):
        raise RuntimeError(f"incomplete soil condition artifact exists: {root}")

    metadata, fids, geometry_wkb, columns = pyogrio.raw.read(
        source,
        layer=LAYER,
        bbox=tuple(float(value) for value in bbox_en),
        columns=list(FIELDS),
        return_fids=True,
    )
    if list(metadata["fields"]) != list(FIELDS):
        raise ValueError("window reader field order differs from the frozen source schema")
    if fids is None or geometry_wkb is None:
        raise ValueError("source driver did not return FID and geometry identity")
    order = np.argsort(fids, kind="stable")
    tables = load_tables()
    if _source_bundle(source) != source_bundle:
        raise RuntimeError("Mullastikukaart source changed during window extraction")
    if _table_sources() != table_sources:
        raise RuntimeError("soil normalization tables changed during window extraction")
    if _profile_authority_sources() != profile_authorities:
        raise RuntimeError("soil profile authorities changed during window extraction")
    query = shapely.box(*bbox_en)
    query_area = float(query.area)
    features: list[dict[str, Any]] = []
    clipped_geometries: list[Any] = []
    for source_index in order:
        fid = int(fids[source_index])
        wkb = geometry_wkb[source_index]
        if wkb is None:
            raise ValueError(f"Mullastikukaart FID {fid} has null geometry")
        wkb_bytes = bytes(wkb)
        geometry = shapely.from_wkb(wkb_bytes)
        if geometry is None or geometry.is_empty:
            raise ValueError(f"Mullastikukaart FID {fid} has empty geometry")
        clipped = geometry.intersection(query)
        if clipped.is_empty:
            raise ValueError(f"spatial reader returned non-intersecting FID {fid}")
        clipped_geometries.append(clipped)
        raw = {
            field: json_scalar(columns[index][source_index])
            for index, field in enumerate(FIELDS)
        }
        normalized = {
            "Siffer": normalize_siffer(raw["Siffer"], tables),
            **{
                f"Sif{index}": normalize_soil_code(raw[f"Sif{index}"], tables)
                for index in range(1, 5)
            },
            "Lihtloimis": normalize_lihtloimis(raw["Lihtloimis"], tables),
            "Kivisus": normalize_stoniness(raw["Kivisus"]),
            "Loimis1": parse_texture_profile(raw["Loimis1"]),
            "Loimis2": parse_texture_profile(raw["Loimis2"]),
            "Huumus": parse_humus_profile(raw["Huumus"]),
        }
        wkb_sha256 = hashlib.sha256(wkb_bytes).hexdigest()
        identity = hashlib.sha256(
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
                    "geometry_identity_sha256": identity,
                },
                "geometry": {
                    "encoding": "OGC_WKB_hex_from_OGR_without_clipping",
                    "ogr_wkb_hex": wkb_bytes.hex(),
                    "ogr_wkb_sha256": wkb_sha256,
                    "type": geometry.geom_type,
                    "bounds_en": [float(value) for value in geometry.bounds],
                    "area_m2": float(geometry.area),
                    "is_valid": bool(geometry.is_valid),
                },
                "window_intersection": {
                    "area_m2": float(clipped.area),
                    "fraction_of_query_bbox": float(clipped.area / query_area),
                },
                "raw_attributes": raw,
                "normalized": normalized,
            }
        )
    covered = shapely.union_all(clipped_geometries).area if clipped_geometries else 0.0
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
            "union_covered_area_m2": float(covered),
            "union_coverage_fraction": float(covered / query_area),
            "uncovered_area_m2": float(query_area - covered),
        },
        "features": features,
        "limitations": [
            "Source polygons are 1:10,000 conditioning evidence, not centimeter geometry.",
            "Shapefile FIDs are identities only together with the exact retained source bundle.",
            "Profile strings outside the bound official grammar remain explicit with residual source spans.",
            "No missing or unparseable value is assigned a nearest class.",
        ],
    }
    if root_exists:
        existing = json.loads(artifact_path.read_text(encoding="utf-8"))
        if existing != artifact:
            raise RuntimeError(
                f"existing soil condition artifact fails full reconstruction: {artifact_path}"
            )
        return artifact_path
    temporary = parent / f".{recipe_sha256}.{os.getpid()}.tmp"
    if temporary.exists():
        raise RuntimeError(f"stale soil condition temporary exists: {temporary}")
    temporary.mkdir(parents=True)
    (temporary / "window.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(root)
    return artifact_path
