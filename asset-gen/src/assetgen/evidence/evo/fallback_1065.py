"""Strict exact authority for the already-qualified Evo fallback plot 1065."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ...config import CONFIG_DIR
from .selection import EvoArtifact, EvoSelection

_PATH = CONFIG_DIR / "evidence" / "evo-2024-plot-1065-fallback.json"
_SCHEMA = "evo-fallback-plot-selection/1.0.0"
_PARENT_SHA256 = "50865b41135eed0a587b12546649f075c18390aa3cc89a95158d7543ed909fdb"
_DATASET_UUID = "b1dac2b9-93cb-407e-91f1-eeb79c8cdd92"
_DATASET_DOI = "10.23729/fd-5a800660-8bd8-35ef-ac9f-ac5c45f7fa77"
_EXPECTED_ARTIFACTS = (
    (
        "c95265eb-b7ec-43d0-93d4-147f5a8d74e5",
        "/Evo_TLS_2024_stand_attributes_v2.csv",
        5773,
        "b124bc20f6816c08812fdc7f061ec7e0d9e444802d3b1b67907420c97812db44",
    ),
    (
        "769b85b7-5e79-4575-89a2-e46c919922e8",
        "/Evo_TLS_2024_treeanal_pointclouds/1065_pointcloud_georef.laz",
        1_148_190_720,
        "7fc725b48a0010bd67ef9e3d6d7b8083674e418d04df7a3b015b519c1bf4c002",
    ),
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Evo 1065 {label} must be an object")
    return value


def load_evo_1065_selection(path: Path | None = None) -> EvoSelection:
    resolved = (path or _PATH).resolve()
    encoded = resolved.read_bytes()
    raw = _mapping(json.loads(encoded), "selection")
    source = _mapping(raw.get("source"), "source")
    selection = _mapping(raw.get("selection"), "plot selection")
    qualification = _mapping(raw.get("qualification"), "qualification")
    if (
        raw.get("schema_version") != _SCHEMA
        or raw.get("id") != "evo-2024-plot-1065-fallback-v1"
        or raw.get("parent_selection_config_sha256") != _PARENT_SHA256
        or source.get("dataset_uuid") != _DATASET_UUID
        or source.get("dataset_doi") != _DATASET_DOI
        or source.get("dataset_version") != 2
        or source.get("published_revision") != 2
        or source.get("file_inventory_count") != 57
        or source.get("release_bytes") != 42_192_878_905
        or source.get("license") != "CC-BY-4.0"
        or selection.get("plot_id") != "1065"
        or selection.get("site_id") != "evo.evo"
        or selection.get("campaign_id") != "evo.evo.2024"
        or selection.get("center")
        != {"epsg3067": {"easting_m": 401746.829, "northing_m": 6786780.246}}
        or selection.get("plot_size_m") != [32, 32]
        or qualification
        != {
            "role": "raw_candidate",
            "status": "unqualified",
            "target_truth": False,
            "synthesis_authorized": False,
            "transfer_ceiling": "none",
            "purpose": "compare real raw observation support with plot 1086 after plot 1086 surface inference abstained",
        }
    ):
        raise ValueError("Evo 1065 exact authority changed")
    artifacts_raw = raw.get("artifacts")
    if not isinstance(artifacts_raw, list):
        raise ValueError("Evo 1065 artifacts must be an array")
    artifacts = tuple(EvoArtifact.from_mapping(item) for item in artifacts_raw)
    observed = tuple(
        (item.file_id, item.source_path, item.bytes, item.sha256) for item in artifacts
    )
    if observed != _EXPECTED_ARTIFACTS:
        raise ValueError("Evo 1065 exact artifact tuple changed")
    record = EvoSelection(
        path=resolved,
        config_sha256=hashlib.sha256(encoded).hexdigest(),
        dataset_uuid=_DATASET_UUID,
        dataset_doi=_DATASET_DOI,
        plot_id="1065",
        site_id="evo.evo",
        campaign_id="evo.evo.2024",
        artifacts=artifacts,
    )
    record.authorized_retention_artifacts()
    return record
