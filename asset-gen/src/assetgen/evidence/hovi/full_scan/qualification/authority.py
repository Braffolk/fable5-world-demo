"""Frozen authority for the first bounded, non-scientific E57 point probe."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
import sys
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .....config import ASSET_GEN_ROOT, CONFIG_DIR
from ..authorization import (
    canonical_json_bytes,
    load_metadata_inspection_authorization,
    open_regular_nofollow,
)
from ..extract import DEFAULT_OUTPUT_ROOT
from ..inventory import _load_extraction_binding


AUTHORITY_SCHEMA = "hovi-hy-spruce4-full-point-probe-authorization/1.0.0"
AUTHORITY_ID = "hovi-hy-spruce4-full-reader-validation-prefix-v1"
NATIVE_REPORT_SCHEMA = "laas-hovi-e57-native-probe/1.0.0"
DEFAULT_AUTHORITY = (
    CONFIG_DIR / "evidence" / "hovi-hy-spruce4-full-point-probe.json"
)
EXPECTED_AUTHORITY_BYTES = 6_847
EXPECTED_AUTHORITY_SHA256 = (
    "ab1664fd5dc37633ef19bc1e068ad58ac8fb5f6dc2ed2211536670e592137de5"
)
SELECTED_INVENTORY_BYTES = 806_369
SELECTED_INVENTORY_SHA256 = (
    "54e21049ac09c7d3f69ac7ab5990c982d12a7bd04da0618de032c8311cd3cfad"
)
SELECTED_INVENTORY_SCHEMA = "hovi-hy-spruce4-full-metadata-inventory/2.0.0"
SELECTED_IMPLEMENTATION_SHA256 = (
    "acf5654acdebab6b35ce5d2bc237cc99edcb8d768fa855eda5ef600c63241962"
)
OBJECT_SHA256 = "7889a9cfef69d57a50eaa5a9f8d224d28486d5901163e08cbdc827f92f409da2"
OBJECT_BYTES = 84_817_645_568
SCAN_ORDINAL = 0
SCAN_GUID = "0000000000000001"
PUBLISHER_RECORD_COUNT = 245_788_993
RECORD_START = 0
RECORD_END = 65_536
MANDATE_PROVENANCE = {
    "date": "2026-07-14",
    "source": "explicit_user_instruction_in_active_session",
    "instructions": ["do it", "continue autonomously"],
    "narrow_interpretation": (
        "engineer_and_run_one_bounded_non_scientific_reader_validation_probe"
    ),
    "does_not_authorize": [
        "surface_reconstruction",
        "analogue_qualification",
        "target_truth",
        "synthesis",
    ],
}
DATASET_AND_LICENSE = {
    "title": (
        "A spectral-structural characterization of European temperate, "
        "hemiboreal and boreal forests: Laboratory and field data"
    ),
    "dataset_authors": "Hovi et al.",
    "dataset_uuid": "ace2a123-00ff-4944-951e-eddbe209b70c",
    "dataset_doi": "10.23729/9a8d90cd-73e2-438d-9230-94e10e61adc9",
    "dataset_version": 1,
    "published_revision": 1,
    "publisher": "Aalto University",
    "data_paper": (
        "Rautiainen et al. 2024, Earth System Science Data 16, 5069-5097"
    ),
    "data_paper_doi": "10.5194/essd-16-5069-2024",
    "license_spdx": "CC-BY-4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/legalcode",
    "attribution_required": True,
    "license_link_required": True,
    "changes_must_be_indicated": True,
    "change_notice": (
        "LAAS decoded and repacked an authority-bounded record prefix for reader "
        "validation; it did not reconstruct or qualify a surface."
    ),
}


@dataclass(frozen=True)
class ProbeResources:
    wall_seconds: int
    cpu_seconds: int
    rss_bytes: int
    address_space_bytes: int
    stdout_bytes: int
    stderr_bytes: int
    open_files: int
    record_output_bytes: int


@dataclass(frozen=True)
class NativeReaderBinding:
    report_schema: str
    execution_authorized: bool
    executable_path: Path | None
    executable_bytes: int | None
    executable_sha256: str | None


@dataclass(frozen=True)
class PointProbeAuthority:
    config_path: Path
    config_sha256: str
    inventory_path: Path
    inventory_sha256: str
    output_root: Path
    scan_ordinal: int
    scan_guid: str
    publisher_record_count: int
    record_start: int
    record_end: int
    hard_max_decoded_records: int
    mandate_provenance: bytes
    dataset_and_license: bytes
    resources: ProbeResources
    native: NativeReaderBinding
    execution_selection_sha256: str | None = None
    execution_identity: bytes | None = None
    implementation_files: tuple[tuple[Path, str], ...] = ()


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Hovi point-probe {label} must be an object")
    return value


def _bound_asset_path(value: Any, label: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"Hovi point-probe {label} must be a relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts or "\\" in value:
        raise ValueError(f"unsafe Hovi point-probe {label}")
    return _absolute(ASSET_GEN_ROOT.joinpath(*pure.parts))


def _read_exact(path: Path, expected_bytes: int, label: str) -> bytes:
    descriptor = open_regular_nofollow(path, label)
    try:
        before = os.fstat(descriptor)
        if before.st_size != expected_bytes:
            raise ValueError(f"Hovi point-probe {label} byte count drifted")
        payload = bytearray()
        while len(payload) <= expected_bytes:
            block = os.read(descriptor, min(1 << 20, expected_bytes + 1 - len(payload)))
            if not block:
                break
            payload.extend(block)
        after = os.fstat(descriptor)
        if (
            len(payload) != expected_bytes
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        ):
            raise ValueError(f"Hovi point-probe {label} changed while being read")
        return bytes(payload)
    finally:
        os.close(descriptor)


def _validate_inventory(raw: Mapping[str, Any]) -> None:
    if (
        raw.get("schemaVersion") != SELECTED_INVENTORY_SCHEMA
        or raw.get("status") != "metadata_inventory_complete"
        or raw.get("scope") != "publisher_xml_metadata_only"
        or raw.get("pointsRead") != 0
        or raw.get("groupRecordsRead") != 0
        or raw.get("compressedVectorRecordsDecoded") != 0
        or raw.get("synthesisAuthorized") is not False
        or raw.get("targetTruth") is not False
        or raw.get("usableSurface") is not False
    ):
        raise ValueError("selected Hovi metadata inventory crossed its evidence boundary")
    implementation = _require_mapping(raw.get("implementation"), "inventory implementation")
    fingerprint = _require_mapping(
        implementation.get("loadedExecutionFingerprint"),
        "inventory implementation fingerprint",
    )
    if fingerprint.get("sha256") != SELECTED_IMPLEMENTATION_SHA256:
        raise ValueError("selected Hovi inventory implementation fingerprint drifted")
    extraction = _require_mapping(raw.get("extraction"), "inventory extraction")
    inventory_object = _require_mapping(extraction.get("object"), "inventory object")
    if (
        inventory_object.get("bytes") != OBJECT_BYTES
        or inventory_object.get("sha256") != OBJECT_SHA256
        or inventory_object.get("mode") != "0400"
        or inventory_object.get("linkCount") != 1
    ):
        raise ValueError("selected Hovi inventory no longer binds the frozen E57 object")
    inventory = _require_mapping(raw.get("inventory"), "interpreted inventory")
    root = _require_mapping(inventory.get("root"), "interpreted inventory root")
    data3d = _require_mapping(root.get("data3D"), "interpreted data3D")
    scans = data3d.get("scans")
    if not isinstance(scans, list) or len(scans) != 16:
        raise ValueError("selected Hovi scan inventory changed")
    scan = _require_mapping(scans[SCAN_ORDINAL], "selected scan")
    guid = _require_mapping(scan.get("guid"), "selected scan GUID")
    points = _require_mapping(scan.get("points"), "selected scan points")
    count = _require_mapping(
        points.get("publisherDeclaredRecordCount"),
        "selected scan publisher count",
    )
    if guid.get("value") != SCAN_GUID or count.get("value") != PUBLISHER_RECORD_COUNT:
        raise ValueError("selected Hovi scan ordinal/GUID/count tuple drifted")


def load_point_probe_authority(
    path: Path = DEFAULT_AUTHORITY,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> PointProbeAuthority:
    """Load the exact probe decision and verify all point-free evidence bindings."""
    path = _absolute(path)
    encoded = _read_exact(path, EXPECTED_AUTHORITY_BYTES, "authority config")
    config_sha256 = hashlib.sha256(encoded).hexdigest()
    if config_sha256 != EXPECTED_AUTHORITY_SHA256:
        raise ValueError("Hovi point-probe authority is not the frozen decision")
    raw = json.loads(encoded)
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "id",
        "status",
        "mandate_provenance",
        "dataset_and_license",
        "authorization",
        "retained_e57",
        "selected_metadata_inventory",
        "selection",
        "resource_limits",
        "native_reader",
        "output_contract",
        "evidence_boundary",
    }:
        raise ValueError("unsupported Hovi point-probe authority")

    authorization = _require_mapping(raw["authorization"], "authorization")
    mandate = _require_mapping(raw["mandate_provenance"], "mandate provenance")
    dataset = _require_mapping(raw["dataset_and_license"], "dataset and license")
    retained = _require_mapping(raw["retained_e57"], "retained E57 binding")
    metadata_ref = _require_mapping(
        retained.get("metadata_inspection_authorization"),
        "metadata authorization binding",
    )
    receipt_ref = _require_mapping(retained.get("extraction_receipt"), "receipt binding")
    object_ref = _require_mapping(retained.get("object"), "object binding")
    selected_ref = _require_mapping(
        raw["selected_metadata_inventory"],
        "selected metadata inventory",
    )
    selection = _require_mapping(raw["selection"], "selection")
    interval = _require_mapping(selection.get("record_interval"), "record interval")
    policy = _require_mapping(selection.get("record_policy"), "record policy")
    resources_raw = _require_mapping(raw["resource_limits"], "resource limits")
    native_raw = _require_mapping(raw["native_reader"], "native reader")
    output = _require_mapping(raw["output_contract"], "output contract")
    boundary = _require_mapping(raw["evidence_boundary"], "evidence boundary")

    if (
        raw["schema_version"] != AUTHORITY_SCHEMA
        or raw["id"] != AUTHORITY_ID
        or raw["status"] != "authorized_pending_native_build_binding"
        or dict(mandate) != MANDATE_PROVENANCE
        or dict(dataset) != DATASET_AND_LICENSE
        or authorization
        != {
            "decision": "BOUNDED_READER_VALIDATION_PROBE_GO",
            "scope": "one_frozen_raw_record_prefix_for_reader_validation_only",
            "reader_validation_only": True,
            "point_decode_authorized": True,
            "point_conversion_authorized": False,
            "surface_reconstruction_authorized": False,
            "analogue_qualification_authorized": False,
            "target_truth": False,
            "synthesis_authorized": False,
        }
        or selection.get("scan_ordinal") != SCAN_ORDINAL
        or selection.get("scan_guid") != SCAN_GUID
        or selection.get("publisher_declared_record_count") != PUBLISHER_RECORD_COUNT
        or interval
        != {
            "start_ordinal_inclusive": RECORD_START,
            "end_ordinal_exclusive": RECORD_END,
            "hard_max_decoded_records": RECORD_END - RECORD_START,
            "early_stop_required": True,
        }
        or policy
        != {
            "retain_cartesian_invalid_state_records": True,
            "source_record_ordinal_required": True,
            "coordinate_transform": "none",
            "pose_role": "provenance_only",
            "raw_fields": [
                "cartesianX_scaled_integer",
                "cartesianY_scaled_integer",
                "cartesianZ_scaled_integer",
                "intensity_f32_bits",
                "rowIndex_integer",
                "columnIndex_integer",
                "cartesianInvalidState_integer",
            ],
        }
        or output
        != {
            "native_stdout": "one_canonical_json_report",
            "native_record_output": "one_inherited_private_regular_fd",
            "record_count": RECORD_END - RECORD_START,
            "bytes_per_record": 96,
            "exact_record_bytes": 6_291_456,
            "byte_order": "little_endian",
            "record_fields": [
                "sourceOrdinal:u64@0",
                "row:i64@8",
                "column:i64@16",
                "cartesianInvalidState:i64@24",
                "rawX:i64@32",
                "rawY:i64@40",
                "rawZ:i64@48",
                "localX:f64bits@56",
                "localY:f64bits@64",
                "localZ:f64bits@72",
                "derivedRangeOrZero:f64bits@80",
                "rawIntensitySingleBits:u32@88",
                "flags:u32@92",
            ],
            "flag_bits": {
                "bit0": "range_present",
                "bit1": "intensity_finite",
                "all_other_bits": "zero",
            },
            "record_output_verification": (
                "exact_bytes_and_independent_sha256_match_native_report"
            ),
            "publication": "atomic_content_addressed_binary_then_report_commit",
            "invalid_records_retained": True,
            "pose_is_provenance_only": True,
            "eof_reached": False,
            "publisher_count_validated": False,
            "surface_claim": False,
        }
        or boundary
        != {
            "scientific_role": "reader_validation_probe_only",
            "qualification_status": "unqualified",
            "surface_claim": False,
            "analogue_qualification_authorized": False,
            "target_truth": False,
            "synthesis_authorized": False,
        }
    ):
        raise ValueError("Hovi point-probe evidence boundary changed")

    metadata_path = _bound_asset_path(metadata_ref.get("path"), "metadata authorization")
    if metadata_ref != {
        "path": "config/evidence/hovi-hy-spruce4-full-metadata-inspection.json",
        "bytes": 3188,
        "sha256": "fe44c5c05c4ccd50cda2c9a662d3c171e3a281f823691d674010ee908c4480b9",
    }:
        raise ValueError("Hovi point-probe metadata authorization binding drifted")
    metadata_authorization = load_metadata_inspection_authorization(metadata_path)
    output_root = _absolute(output_root)
    binding = _load_extraction_binding(metadata_authorization, output_root)
    if (
        receipt_ref.get("path")
        != binding.receipt_path.relative_to(ASSET_GEN_ROOT).as_posix()
        or receipt_ref.get("bytes") != len(binding.receipt_bytes)
        or receipt_ref.get("sha256")
        != hashlib.sha256(binding.receipt_bytes).hexdigest()
        or receipt_ref.get("status") != "complete"
        or object_ref.get("path")
        != binding.object_path.relative_to(ASSET_GEN_ROOT).as_posix()
        or object_ref.get("relative_path") != binding.object_relative
        or object_ref.get("bytes") != OBJECT_BYTES
        or object_ref.get("sha256") != OBJECT_SHA256
        or object_ref.get("mode") != "0400"
        or object_ref.get("link_count") != 1
        or binding.object_bytes != OBJECT_BYTES
        or binding.object_sha256 != OBJECT_SHA256
    ):
        raise ValueError("Hovi point-probe receipt/object binding drifted")

    inventory_path = _bound_asset_path(selected_ref.get("path"), "selected inventory")
    if selected_ref != {
        "path": (
            "data/in/evidence/hovi-full-metadata/inventories/sha256/54/"
            f"{SELECTED_INVENTORY_SHA256}.json"
        ),
        "bytes": SELECTED_INVENTORY_BYTES,
        "sha256": SELECTED_INVENTORY_SHA256,
        "schema_version": SELECTED_INVENTORY_SCHEMA,
        "implementation_fingerprint_sha256": SELECTED_IMPLEMENTATION_SHA256,
        "records_decoded": 0,
    }:
        raise ValueError("Hovi point-probe selected inventory binding drifted")
    inventory_bytes = _read_exact(
        inventory_path,
        SELECTED_INVENTORY_BYTES,
        "selected metadata inventory",
    )
    if (
        hashlib.sha256(inventory_bytes).hexdigest() != SELECTED_INVENTORY_SHA256
        or canonical_json_bytes(json.loads(inventory_bytes)) != inventory_bytes
    ):
        raise ValueError("selected Hovi metadata inventory bytes drifted")
    inventory = json.loads(inventory_bytes)
    if not isinstance(inventory, Mapping):
        raise ValueError("selected Hovi metadata inventory must be an object")
    _validate_inventory(inventory)

    expected_resources = {
        "wall_seconds": 60,
        "cpu_seconds": 60,
        "rss_bytes": 512 << 20,
        "address_space_bytes": 1 << 30,
        "stdout_bytes": 1 << 20,
        "stderr_bytes": 1 << 20,
        "open_files": 32,
        "record_output_bytes": 6_291_456,
    }
    if dict(resources_raw) != expected_resources:
        raise ValueError("Hovi point-probe resource limits changed")
    if native_raw != {
        "report_schema": NATIVE_REPORT_SCHEMA,
        "profile": "hy-spruce4",
        "max_file_bytes": OBJECT_BYTES,
        "max_xml_bytes": 65_536,
        "max_scans": 16,
        "max_prototype_fields": 7,
        "max_declared_records": 4_248_797_321,
        "max_json_bytes": 1 << 20,
        "e57_crate_version": "0.11.13",
        "e57_crate_sha256": (
            "fcfee41a50fbd70278c70cc477b8671ffe03951cba028cc2c057777a9ee6a3cc"
        ),
        "build_status": "awaiting_frozen_project_build_identity",
        "execution_authorized": False,
        "executable_path": None,
        "executable_bytes": None,
        "executable_sha256": None,
        "required_frozen_build_identity": [
            "repository_base_commit",
            "reader_commit",
            "source_tree_sha256",
            "cargo_lock_sha256",
            "rust_toolchain",
            "target_triple",
            "release_profile",
            "executable_bytes",
            "executable_sha256",
            "vendored_upstream_identity",
            "vendor_patch_set_sha256",
        ],
    }:
        raise ValueError("Hovi point-probe native reader boundary changed")

    return PointProbeAuthority(
        config_path=path,
        config_sha256=config_sha256,
        inventory_path=inventory_path,
        inventory_sha256=SELECTED_INVENTORY_SHA256,
        output_root=output_root,
        scan_ordinal=SCAN_ORDINAL,
        scan_guid=SCAN_GUID,
        publisher_record_count=PUBLISHER_RECORD_COUNT,
        record_start=RECORD_START,
        record_end=RECORD_END,
        hard_max_decoded_records=RECORD_END - RECORD_START,
        mandate_provenance=canonical_json_bytes(MANDATE_PROVENANCE),
        dataset_and_license=canonical_json_bytes(DATASET_AND_LICENSE),
        resources=ProbeResources(**expected_resources),
        native=NativeReaderBinding(
            report_schema=NATIVE_REPORT_SCHEMA,
            execution_authorized=False,
            executable_path=None,
            executable_bytes=None,
            executable_sha256=None,
        ),
    )


def load_probe_execution_selection(
    path: Path,
    *,
    authority: PointProbeAuthority | None = None,
) -> PointProbeAuthority:
    """Elevate the pending authority only through a content-addressed build selection."""
    authority = load_point_probe_authority() if authority is None else authority
    path = _absolute(path)
    descriptor = open_regular_nofollow(path, "point-probe execution selection")
    try:
        chunks = []
        while block := os.read(descriptor, 1 << 20):
            chunks.append(block)
        encoded = b"".join(chunks)
    finally:
        os.close(descriptor)
    digest = hashlib.sha256(encoded).hexdigest()
    if path.name != f"{digest}.json" or canonical_json_bytes(json.loads(encoded)) != encoded:
        raise ValueError("Hovi point-probe execution selection is not content-addressed JSON")
    raw = json.loads(encoded)
    if not isinstance(raw, dict) or set(raw) != {
        "schemaVersion",
        "status",
        "authorityConfigSha256",
        "repository",
        "nativeBuild",
        "pythonExecution",
        "publicationRoot",
        "evidenceBoundary",
    }:
        raise ValueError("unsupported Hovi point-probe execution selection")
    repository = _require_mapping(raw["repository"], "execution repository")
    native = _require_mapping(raw["nativeBuild"], "execution native build")
    python = _require_mapping(raw["pythonExecution"], "execution Python binding")
    boundary = _require_mapping(raw["evidenceBoundary"], "execution evidence boundary")
    files = _require_mapping(python.get("files"), "execution implementation files")
    if (
        raw["schemaVersion"] != "hovi-hy-spruce4-full-point-probe-execution/1.0.0"
        or raw["status"] != "execution_authorized"
        or raw["authorityConfigSha256"] != authority.config_sha256
        or repository.get("readerCommit")
        != "4c6e5582a077093bbdaa90fd0839ff7a516dacb8"
        or repository.get("nativeSourceTreeSha256")
        != "747c05a91f3ea9e197beba5f194b1020fbd21e28cd8a9b25a89353859ebe4775"
        or native.get("cargoLockSha256")
        != "4e172cb2f049a23395f487df481e4902794c699a86c36d10a08930418ee16feb"
        or native.get("rustToolchain") != "1.94.1"
        or native.get("rustcCommit") != "e408947bfd200af42db322daf0fadfe7e26d3bd1"
        or native.get("targetTriple") != "aarch64-apple-darwin"
        or native.get("releaseProfile")
        != {"codegenUnits": 1, "lto": "thin", "panic": "abort", "strip": "symbols"}
        or native.get("upstreamManifestSha256")
        != "1b2b48d97f564d6dbc9d07197d624f57e247f2d4d05d8d0eb78e4ec9ace90fe0"
        or native.get("vendorPatchSetSha256")
        != "64428fb355fb3c36cdc25271dd42ea0e95f853dbc4a5d7c9c60455463344ab08"
        or python.get("implementation") != "cpython"
        or python.get("version") != platform.python_version()
        or python.get("platform") != sys.platform
        or python.get("machine") != platform.machine()
        or raw["publicationRoot"]
        != "data/work/hovi-full-scan-point-probe"
        or boundary
        != {
            "readerValidationOnly": True,
            "surfaceClaim": False,
            "analogueQualificationAuthorized": False,
            "targetTruth": False,
            "synthesisAuthorized": False,
        }
    ):
        raise ValueError("Hovi point-probe execution selection crossed its authority")

    executable_path = _bound_asset_path(native.get("executablePath"), "native executable")
    executable_bytes = native.get("executableBytes")
    executable_sha256 = native.get("executableSha256")
    if (
        not isinstance(executable_bytes, int)
        or executable_bytes <= 0
        or not isinstance(executable_sha256, str)
        or len(executable_sha256) != 64
    ):
        raise ValueError("Hovi point-probe executable identity is invalid")
    descriptor = open_regular_nofollow(executable_path, "native point-probe executable")
    try:
        executable_stat = os.fstat(descriptor)
        executable_digest = hashlib.sha256()
        offset = 0
        while offset < executable_bytes:
            block = os.pread(descriptor, min(1 << 20, executable_bytes - offset), offset)
            if not block:
                break
            executable_digest.update(block)
            offset += len(block)
        if (
            offset != executable_bytes
            or executable_stat.st_size != executable_bytes
            or executable_digest.hexdigest() != executable_sha256
            or executable_stat.st_uid != os.getuid()
            or stat.S_IMODE(executable_stat.st_mode) != 0o500
            or executable_stat.st_nlink != 1
        ):
            raise ValueError("Hovi point-probe executable artifact drifted")
    finally:
        os.close(descriptor)

    implementation_files: list[tuple[Path, str]] = []
    for relative, expected_sha256 in sorted(files.items()):
        source_path = _bound_asset_path(relative, "Python implementation file")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError("Hovi point-probe implementation hash is invalid")
        descriptor = open_regular_nofollow(source_path, "Python implementation file")
        try:
            source_digest = hashlib.sha256()
            while block := os.read(descriptor, 1 << 20):
                source_digest.update(block)
        finally:
            os.close(descriptor)
        if source_digest.hexdigest() != expected_sha256:
            raise ValueError(f"Hovi point-probe implementation drifted: {relative}")
        implementation_files.append((source_path, expected_sha256))

    return replace(
        authority,
        native=NativeReaderBinding(
            report_schema=authority.native.report_schema,
            execution_authorized=True,
            executable_path=executable_path,
            executable_bytes=executable_bytes,
            executable_sha256=executable_sha256,
        ),
        execution_selection_sha256=digest,
        execution_identity=encoded,
        implementation_files=tuple(implementation_files),
    )
