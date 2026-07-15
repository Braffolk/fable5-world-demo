"""Frozen source schema and strict table-backed normalization."""
from __future__ import annotations

import json
import re
import struct
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyogrio

from ....config import CONFIG_DIR

LAYER = "Mullakaart"
CRS = "EPSG:3301"
ENCODING = "UTF-8"
GEOMETRY_TYPE = "Polygon"
FIELDS = (
    "Siffer",
    "Sif1",
    "Osa1",
    "Sif2",
    "Osa2",
    "Sif3",
    "Osa3",
    "Sif4",
    "Osa4",
    "Boniteet",
    "Loimis1",
    "Loimis2",
    "Lihtloimis",
    "Huumus",
    "Kivisus",
    "Varv",
    "Shape_Area",
)
DTYPES = (
    "object",
    "object",
    "int32",
    "object",
    "int32",
    "object",
    "int32",
    "object",
    "int32",
    "int32",
    "object",
    "object",
    "object",
    "object",
    "object",
    "int32",
    "float64",
)
OGR_TYPES = (
    "OFTString",
    "OFTString",
    "OFTInteger",
    "OFTString",
    "OFTInteger",
    "OFTString",
    "OFTInteger",
    "OFTString",
    "OFTInteger",
    "OFTInteger",
    "OFTString",
    "OFTString",
    "OFTString",
    "OFTString",
    "OFTString",
    "OFTInteger",
    "OFTReal",
)

_QUOTE_FOLD = str.maketrans(
    {"’": "'", "′": "'", "`": "'", "”": '"', "“": '"', "″": '"'}
)
_SUBSUP = {
    ord(character): str(digit)
    for digit, characters in enumerate(["₀⁰", "₁¹", "₂²", "₃³", "₄⁴"])
    for character in characters
}
_SKELETON_RE = re.compile(r"^(rb|kr|kb|ko|v|r|k|d|p)([0-9]?)")
_CORE_RE = re.compile(r"^(pl|sl|ls|th|tt|t|l|s)([0-9]?)")
_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}
DBF_FIELDS = (
    ("Siffer", "C", 32, 0),
    ("Sif1", "C", 15, 0),
    ("Osa1", "N", 5, 0),
    ("Sif2", "C", 13, 0),
    ("Osa2", "N", 5, 0),
    ("Sif3", "C", 11, 0),
    ("Osa3", "N", 5, 0),
    ("Sif4", "C", 9, 0),
    ("Osa4", "N", 5, 0),
    ("Boniteet", "N", 5, 0),
    ("Loimis1", "C", 60, 0),
    ("Loimis2", "C", 55, 0),
    ("Lihtloimis", "C", 15, 0),
    ("Huumus", "C", 71, 0),
    ("Kivisus", "C", 16, 0),
    ("Varv", "N", 5, 0),
    ("Shape_Area", "F", 19, 11),
)


@dataclass(frozen=True)
class NormalizationTables:
    type_specs: dict[str, dict[str, Any]]
    configured_aliases: dict[str, str]
    official_codes: frozenset[str]
    official_aliases: dict[str, str]
    texture_core: dict[str, int]
    texture_skeleton: dict[str, int]


def _dbf_field_schema(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as source:
        header = source.read(32)
        if len(header) != 32:
            raise ValueError("Mullastikukaart DBF header is truncated")
        header_bytes = struct.unpack_from("<H", header, 8)[0]
        if header_bytes < 33 or (header_bytes - 33) % 32:
            raise ValueError("Mullastikukaart DBF field header length is malformed")
        descriptors = source.read(header_bytes - 32)
    if len(descriptors) != header_bytes - 32 or descriptors[-1] != 0x0D:
        raise ValueError("Mullastikukaart DBF field descriptors are truncated")
    rows = []
    for offset in range(0, len(descriptors) - 1, 32):
        descriptor = descriptors[offset : offset + 32]
        name = descriptor[:11].split(b"\0", 1)[0].decode("ascii")
        rows.append(
            {
                "name": name,
                "type": chr(descriptor[11]),
                "width": int(descriptor[16]),
                "decimal_precision": int(descriptor[17]),
            }
        )
    return rows


def schema_snapshot(source: Path) -> dict[str, Any]:
    info = pyogrio.read_info(source, layer=LAYER)
    actual = {
        "layer": info["layer_name"],
        "crs": str(info["crs"]),
        "encoding": info["encoding"],
        "geometry_type": info["geometry_type"],
        "fields": list(info["fields"]),
        "dtypes": list(info["dtypes"]),
        "ogr_types": list(info["ogr_types"]),
        "ogr_subtypes": list(info["ogr_subtypes"]),
        "fid_column": info["fid_column"],
        "feature_count": int(info["features"]),
        "total_bounds": [float(value) for value in info["total_bounds"]],
        "driver": info["driver"],
        "dbf_fields": _dbf_field_schema(source.with_suffix(".dbf")),
    }
    expected = {
        "layer": LAYER,
        "crs": CRS,
        "encoding": ENCODING,
        "geometry_type": GEOMETRY_TYPE,
        "fields": list(FIELDS),
        "dtypes": list(DTYPES),
        "ogr_types": list(OGR_TYPES),
        "ogr_subtypes": ["OFSTNone"] * len(FIELDS),
        "fid_column": "",
        "driver": "ESRI Shapefile",
        "dbf_fields": [
            {
                "name": name,
                "type": field_type,
                "width": width,
                "decimal_precision": precision,
            }
            for name, field_type, width, precision in DBF_FIELDS
        ],
    }
    drift = {
        key: {"expected": value, "actual": actual[key]}
        for key, value in expected.items()
        if actual[key] != value
    }
    if drift:
        raise ValueError(f"Mullastikukaart source schema drift: {drift}")
    return actual


def load_tables() -> NormalizationTables:
    soil_types = tomllib.loads((CONFIG_DIR / "soil-types.toml").read_text())
    textures = tomllib.loads((CONFIG_DIR / "soil-texture.toml").read_text())
    authority_path = CONFIG_DIR / "conditions/mullastikukaart-official-soil-authority.json"
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    if authority.get("schema_version") != "laas.mullastikukaart-official-soil-authority/1":
        raise ValueError("unsupported Mullastikukaart soil authority schema")
    expected_authority_fields = {
        "schema_version",
        "authority_basis",
        "official_canonical_codes",
        "official_aliases",
    }
    if set(authority) != expected_authority_fields:
        raise ValueError("Mullastikukaart soil authority fields differ")
    official_codes = frozenset(str(code) for code in authority["official_canonical_codes"])
    official_aliases = {
        str(alias): str(target) for alias, target in authority["official_aliases"].items()
    }
    if not official_codes <= set(soil_types["types"]):
        raise ValueError("official soil authority references an absent canonical type")
    if any(target not in official_codes for target in official_aliases.values()):
        raise ValueError("official soil authority alias targets a non-official type")
    return NormalizationTables(
        type_specs=dict(soil_types["types"]),
        configured_aliases=dict(soil_types.get("aliases", {})),
        official_codes=official_codes,
        official_aliases=official_aliases,
        texture_core={key: int(value) for key, value in textures["core"].items()},
        texture_skeleton={
            key: int(value) for key, value in textures["skeleton"].items()
        },
    )


def fold_typography(value: str) -> str:
    folded = value.translate(_QUOTE_FOLD).translate(_SUBSUP).strip()
    folded = re.sub(r"\s+", "", folded)

    def canonical_marks(match: re.Match[str]) -> str:
        total = sum(1 if character == "'" else 2 for character in match.group(0))
        return {1: "'", 2: '"', 3: "'\""}.get(total, "'\"")

    return re.sub(r"['\"]+", canonical_marks, folded)


def normalize_soil_code(
    raw: Any,
    tables: NormalizationTables,
) -> dict[str, Any]:
    if raw is None:
        return {"status": "missing"}
    spelling = fold_typography(str(raw))
    if spelling in tables.official_aliases:
        canonical = tables.official_aliases[spelling]
    elif spelling in tables.configured_aliases:
        return {"status": "unparseable_local_alias_not_officially_authorized"}
    else:
        canonical = spelling
    spec = tables.type_specs.get(canonical)
    if spec is None or canonical not in tables.official_codes:
        return {"status": "unparseable_no_exact_official_table_entry"}
    return {
        "status": "known_exact_table_entry",
        "canonical_code": canonical,
        "id": int(spec["id"]),
        "name_et": str(spec["name"]),
        "source_spelling_is_alias": spelling != canonical,
    }


def normalize_siffer(raw: Any, tables: NormalizationTables) -> dict[str, Any]:
    if raw is None:
        return {"status": "missing", "components": []}
    tokens = str(raw).split(";")
    components = [normalize_soil_code(token, tables) for token in tokens]
    status = (
        "known_exact_table_entries"
        if all(item["status"] == "known_exact_table_entry" for item in components)
        else "contains_unparseable_component"
    )
    return {"status": status, "components": components}


def _normalize_texture_token(
    raw: str,
    tables: NormalizationTables,
) -> dict[str, Any]:
    spelling = fold_typography(raw).lower()
    spelling = re.sub(r"[()'\"]", "", spelling)
    skeleton_key: str | None = None
    skeleton_id = 0
    match = _SKELETON_RE.match(spelling)
    rest = spelling[match.end() :] if match else spelling
    if match and spelling[: match.end()] not in ("l", "s") and (
        rest == "" or _CORE_RE.match(rest)
    ):
        family = "r" if match.group(1) == "rb" else match.group(1)
        grade = min(int(match.group(2) or 1), 5)
        skeleton_key = family if family in ("p", "ko") else f"{family}{grade}"
        if skeleton_key not in tables.texture_skeleton:
            return {"status": "unparseable_no_exact_skeleton_table_entry"}
        skeleton_id = tables.texture_skeleton[skeleton_key]
        spelling = rest
    core_match = _CORE_RE.fullmatch(spelling)
    if core_match is None:
        return {"status": "unparseable_no_exact_core_table_entry"}
    core_key = core_match.group(1) + core_match.group(2)
    if core_key not in tables.texture_core:
        core_key = core_match.group(1)
    if core_key not in tables.texture_core:
        return {"status": "unparseable_no_exact_core_table_entry"}
    return {
        "status": "known_exact_table_entries",
        "core": {"code": core_key, "id": tables.texture_core[core_key]},
        "skeleton": {"code": skeleton_key, "id": skeleton_id},
    }


def normalize_lihtloimis(raw: Any, tables: NormalizationTables) -> dict[str, Any]:
    if raw is None:
        return {"status": "missing", "layers": []}
    tokens = str(raw).split("/")
    layers = [_normalize_texture_token(token, tables) for token in tokens]
    status = (
        "known_exact_table_entries"
        if all(item["status"] == "known_exact_table_entries" for item in layers)
        else "contains_unparseable_layer"
    )
    return {"status": status, "layers": layers}


def normalize_stoniness(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {"status": "missing", "tokens": []}
    spelling = str(raw).strip()
    if spelling not in _ROMAN:
        return {"status": "unparseable_no_exact_table_entry", "tokens": []}
    return {
        "status": "known_exact_table_entry",
        "tokens": [{"source_token": spelling, "class": _ROMAN[spelling]}],
    }


def json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value
