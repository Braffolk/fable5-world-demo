"""Strict metadata-only freeze for the Evo 2024 plot-1086 evidence tranche."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from ...config import CONFIG_DIR

_CONFIG_PATH = CONFIG_DIR / "evidence" / "evo-2024-plot-1086.json"
_CONFIG_SHA256 = "50865b41135eed0a587b12546649f075c18390aa3cc89a95158d7543ed909fdb"
_SCHEMA = "evo-public-target-selection/1.0.0"
_SELECTION_ID = "evo-2024-plot-1086-first-conversion-v1"
_DATASET_UUID = "b1dac2b9-93cb-407e-91f1-eeb79c8cdd92"
_DATASET_DOI = "10.23729/fd-5a800660-8bd8-35ef-ac9f-ac5c45f7fa77"
_SELECTOR_PATH = "/Evo_TLS_2024_stand_attributes_v2.csv"
_PROSE_SELECTOR_FILENAME = "Evo_TLS_2024_stand_attributes_v3.csv"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_TOP_LEVEL_KEYS = {
    "schema_version",
    "id",
    "status",
    "purpose",
    "source",
    "metadata_discrepancy",
    "license",
    "access",
    "selection",
    "acquisition",
    "artifacts",
    "qualification",
    "validation",
}
_ARTIFACT_KEYS = {
    "file_id",
    "path",
    "bytes",
    "sha256",
    "kind",
    "retention_authorized",
    "downloaded_during_selection_audit",
    "retained_in_workspace",
}
_EXPECTED_ARTIFACTS = (
    (
        "c95265eb-b7ec-43d0-93d4-147f5a8d74e5",
        _SELECTOR_PATH,
        5773,
        "b124bc20f6816c08812fdc7f061ec7e0d9e444802d3b1b67907420c97812db44",
        "stand_attribute_selector_index",
        True,
        True,
        False,
    ),
    (
        "55a904fb-d582-42d0-90ac-c20bddde0a3d",
        "/Evo_TLS_2024_treeanal_pointclouds/1086_pointcloud_georef.laz",
        889454592,
        "113d5d062240361e005f4e434e1d834d445ba7f72d9d0162e25a644a06ce23f7",
        "tree_segmented_point_cloud_raw_candidate",
        False,
        False,
        False,
    ),
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Evo {label} must be an object")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Evo {label} must be a non-empty string")
    return value


@dataclass(frozen=True)
class EvoArtifact:
    file_id: str
    source_path: str
    bytes: int
    sha256: str
    kind: str
    retention_authorized: bool
    downloaded_during_selection_audit: bool
    retained_in_workspace: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "EvoArtifact":
        if set(raw) != _ARTIFACT_KEYS:
            raise ValueError("Evo artifact tuple has unknown or missing fields")
        file_id = _string(raw.get("file_id"), "artifact file_id")
        source_path = _string(raw.get("path"), "artifact path")
        sha256 = _string(raw.get("sha256"), "artifact sha256")
        byte_count = raw.get("bytes")
        pure = PurePosixPath(source_path)
        if not _UUID_RE.fullmatch(file_id):
            raise ValueError("Evo artifact file_id must be a lowercase UUID")
        if not _SHA256_RE.fullmatch(sha256):
            raise ValueError("Evo artifact sha256 must be a lowercase SHA-256")
        if (
            not source_path.startswith("/")
            or "\\" in source_path
            or any(part in {"", ".", ".."} for part in pure.parts[1:])
            or "/" + "/".join(pure.parts[1:]) != source_path
        ):
            raise ValueError("Evo artifact path must be absolute and canonical")
        if (
            not isinstance(byte_count, int)
            or isinstance(byte_count, bool)
            or byte_count <= 0
        ):
            raise ValueError("Evo artifact bytes must be a positive integer")
        flags = (
            raw.get("retention_authorized"),
            raw.get("downloaded_during_selection_audit"),
            raw.get("retained_in_workspace"),
        )
        if any(not isinstance(value, bool) for value in flags):
            raise ValueError("Evo artifact state flags must be booleans")
        return cls(
            file_id=file_id,
            source_path=source_path,
            bytes=byte_count,
            sha256=sha256,
            kind=_string(raw.get("kind"), "artifact kind"),
            retention_authorized=flags[0],
            downloaded_during_selection_audit=flags[1],
            retained_in_workspace=flags[2],
        )

    def stable_tuple(self) -> tuple[str, str, int, str, str, bool, bool, bool]:
        return (
            self.file_id,
            self.source_path,
            self.bytes,
            self.sha256,
            self.kind,
            self.retention_authorized,
            self.downloaded_during_selection_audit,
            self.retained_in_workspace,
        )


@dataclass(frozen=True)
class EvoSelection:
    path: Path
    config_sha256: str
    dataset_uuid: str
    dataset_doi: str
    plot_id: str
    site_id: str
    campaign_id: str
    artifacts: tuple[EvoArtifact, ...]

    def authorized_retention_artifacts(self) -> tuple[EvoArtifact, ...]:
        """Return the metadata-only allow-list; the LAZ remains fetch-disabled."""
        allowed = tuple(item for item in self.artifacts if item.retention_authorized)
        if len(allowed) != 1 or allowed[0].source_path != _SELECTOR_PATH:
            raise ValueError("Evo retention scope expanded beyond the selector index")
        return allowed


def load_evo_selection(path: Path | None = None) -> EvoSelection:
    """Load the exact DOI-v2 freeze and reject substitutions or scope growth."""
    resolved = (path or _CONFIG_PATH).resolve()
    encoded = resolved.read_bytes()
    config_sha256 = hashlib.sha256(encoded).hexdigest()
    if config_sha256 != _CONFIG_SHA256:
        raise ValueError("Evo frozen selection config bytes changed")
    raw = json.loads(encoded)
    if not isinstance(raw, Mapping) or set(raw) != _TOP_LEVEL_KEYS:
        raise ValueError("unsupported or non-strict Evo selection config")
    if (
        raw.get("schema_version") != _SCHEMA
        or raw.get("id") != _SELECTION_ID
        or raw.get("status") != "raw_candidate_unqualified"
    ):
        raise ValueError("Evo selection schema, identity, or status changed")

    source = _mapping(raw.get("source"), "source")
    discrepancy = _mapping(raw.get("metadata_discrepancy"), "metadata discrepancy")
    selection = _mapping(raw.get("selection"), "selection")
    qualification = _mapping(raw.get("qualification"), "qualification")
    access = _mapping(raw.get("access"), "access")
    validation = _mapping(raw.get("validation"), "validation")
    if (
        source.get("dataset_uuid") != _DATASET_UUID
        or source.get("dataset_doi") != _DATASET_DOI
        or source.get("dataset_version") != 2
        or source.get("published_revision") != 2
    ):
        raise ValueError("Evo controlling dataset version-2 identity changed")
    if discrepancy != {
        "catalog_prose_selector_filename": _PROSE_SELECTOR_FILENAME,
        "version_2_inventory_selector_path": _SELECTOR_PATH,
        "authority": "published_version_2_file_inventory",
        "policy": "fail_if_catalog_prose_filename_is_used_or_inventory_tuple_differs",
    }:
        raise ValueError("Evo v2/v3 selector discrepancy contract changed")
    if (
        selection.get("regime_id") != "forest.floor_ordinary"
        or selection.get("plot_id") != "1086"
        or selection.get("site_id") != "evo.evo"
        or selection.get("campaign_id") != "evo.evo.2024"
        or selection.get("frozen_selector", {}).get("qualifying_plot_ids")
        != ["1086", "1065"]
    ):
        raise ValueError("Evo plot-1086 selector identity changed")
    if (
        qualification.get("role") != "raw_candidate"
        or qualification.get("qualification_status") != "unqualified"
        or qualification.get("target_truth") is not False
        or qualification.get("synthesis_authorized") is not False
        or qualification.get("transfer_ceiling") != "none"
    ):
        raise ValueError("Evo qualification boundary changed")
    if (
        access.get("authorization_json_template")
        != {"cr_id": _DATASET_UUID, "file": "<full pathname>"}
        or validation.get("version_policy") != "exact_version_2_only"
        or validation.get("selector_filename_policy")
        != "exact_inventory_v2_path_only"
    ):
        raise ValueError("Evo access or version-drift policy changed")

    artifacts_raw = raw.get("artifacts")
    if not isinstance(artifacts_raw, list):
        raise ValueError("Evo artifacts must be an array")
    artifacts = tuple(EvoArtifact.from_mapping(item) for item in artifacts_raw)
    if tuple(item.stable_tuple() for item in artifacts) != _EXPECTED_ARTIFACTS:
        raise ValueError("Evo frozen artifact tuples changed")
    if discrepancy["version_2_inventory_selector_path"] != artifacts[0].source_path:
        raise ValueError("Evo selector no longer follows the version-2 inventory")
    if PurePosixPath(artifacts[0].source_path).name == _PROSE_SELECTOR_FILENAME:
        raise ValueError("Evo catalog prose v3 filename cannot control the v2 selection")

    record = EvoSelection(
        path=resolved,
        config_sha256=config_sha256,
        dataset_uuid=_string(source.get("dataset_uuid"), "dataset_uuid"),
        dataset_doi=_string(source.get("dataset_doi"), "dataset_doi"),
        plot_id=_string(selection.get("plot_id"), "plot_id"),
        site_id=_string(selection.get("site_id"), "site_id"),
        campaign_id=_string(selection.get("campaign_id"), "campaign_id"),
        artifacts=artifacts,
    )
    record.authorized_retention_artifacts()
    return record


def _main() -> None:
    parser = argparse.ArgumentParser(description="Validate the frozen Evo plot-1086 selection.")
    parser.add_argument("--selection", type=Path, default=_CONFIG_PATH)
    args = parser.parse_args()
    selection = load_evo_selection(args.selection)
    retained = selection.authorized_retention_artifacts()
    print(
        f"{selection.plot_id} {selection.config_sha256} "
        f"metadata_retention_files={len(retained)} metadata_retention_bytes={retained[0].bytes}"
    )


if __name__ == "__main__":
    _main()
