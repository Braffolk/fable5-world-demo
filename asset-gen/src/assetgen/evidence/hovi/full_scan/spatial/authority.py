"""Frozen, point-free authority for full-scan spatial materialization."""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from dataclasses import dataclass
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


AUTHORITY_SCHEMA = (
    "hovi-hy-spruce4-full-spatial-materialization-authorization/1.0.0"
)
AUTHORITY_ID = "hovi-hy-spruce4-full-spatial-raw-candidate-v1"
DEFAULT_AUTHORITY = (
    CONFIG_DIR
    / "evidence"
    / "hovi-hy-spruce4-full-spatial-materialization.json"
)
EXPECTED_AUTHORITY_BYTES = 11_415
EXPECTED_AUTHORITY_SHA256 = (
    "bd3b39e6acf1de33658b7bdb63cbf7bf169478bad184f653f49093b6fcbc312a"
)
INVENTORY_BYTES = 806_369
INVENTORY_SHA256 = (
    "54e21049ac09c7d3f69ac7ab5990c982d12a7bd04da0618de032c8311cd3cfad"
)
INVENTORY_SCHEMA = "hovi-hy-spruce4-full-metadata-inventory/2.0.0"
INVENTORY_IMPLEMENTATION_SHA256 = (
    "acf5654acdebab6b35ce5d2bc237cc99edcb8d768fa855eda5ef600c63241962"
)
OBJECT_BYTES = 84_817_645_568
OBJECT_SHA256 = "7889a9cfef69d57a50eaa5a9f8d224d28486d5901163e08cbdc827f92f409da2"
PUBLISHER_RECORD_COUNT = 4_248_797_321
TREE_CEILING_BYTES = 120 << 30
LAUNCH_FREE_BYTES = 216 << 30
FREE_RESERVE_BYTES = 96 << 30
RECORD_BYTES = 28


@dataclass(frozen=True)
class ScanBinding:
    ordinal: int
    guid: str
    publisher_records: int
    pose_f64_bits: tuple[str, ...]


@dataclass(frozen=True)
class SpatialResources:
    tree_ceiling_bytes: int
    launch_free_bytes: int
    free_reserve_bytes: int
    native_max_open_files: int
    native_max_open_output_files: int
    stdout_bytes: int
    stderr_bytes: int
    wall_seconds: int
    cpu_seconds: int
    rss_bytes: int
    supervision_sample_seconds: float


@dataclass(frozen=True)
class NativeBinding:
    command: str
    report_schema: str
    execution_authorized: bool
    executable_path: Path | None
    executable_bytes: int | None
    executable_sha256: str | None


@dataclass(frozen=True)
class SpatialMaterializationAuthority:
    config_path: Path
    config_sha256: str
    inventory_path: Path
    inventory_sha256: str
    source_root: Path
    scans: tuple[ScanBinding, ...]
    resources: SpatialResources
    native: NativeBinding
    mandate_provenance: bytes
    dataset_and_license: bytes
    publication_root: Path | None = None
    execution_selection_sha256: str | None = None
    execution_identity: bytes | None = None
    implementation_files: tuple[tuple[Path, str], ...] = ()


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Hovi spatial {label} must be an object")
    return value


def _asset_path(value: Any, label: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"Hovi spatial {label} must be a relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts or "\\" in value:
        raise ValueError(f"unsafe Hovi spatial {label}")
    return _absolute(ASSET_GEN_ROOT.joinpath(*pure.parts))


def _read_exact(path: Path, expected_bytes: int, label: str) -> bytes:
    descriptor = open_regular_nofollow(path, label)
    try:
        before = os.fstat(descriptor)
        if before.st_size != expected_bytes:
            raise ValueError(f"Hovi spatial {label} byte count drifted")
        payload = bytearray()
        while len(payload) <= expected_bytes:
            block = os.read(
                descriptor,
                min(1 << 20, expected_bytes + 1 - len(payload)),
            )
            if not block:
                break
            payload.extend(block)
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if len(payload) != expected_bytes or before_identity != after_identity:
            raise ValueError(f"Hovi spatial {label} changed while being read")
        return bytes(payload)
    finally:
        os.close(descriptor)


def _f64_bits(value: Any, label: str) -> str:
    try:
        parsed = float(str(value))
    except ValueError as error:
        raise ValueError(f"Hovi spatial {label} is not a decimal float") from error
    if not math.isfinite(parsed):
        raise ValueError(f"Hovi spatial {label} is non-finite")
    bits = struct.unpack(">Q", struct.pack(">d", parsed))[0]
    return f"{bits:016x}"


def _inventory_scan_binding(scan: Mapping[str, Any], ordinal: int) -> ScanBinding:
    guid = _mapping(scan.get("guid"), f"scan {ordinal} GUID").get("value")
    points = _mapping(scan.get("points"), f"scan {ordinal} points")
    count = _mapping(
        points.get("publisherDeclaredRecordCount"),
        f"scan {ordinal} publisher count",
    ).get("value")
    pose = _mapping(scan.get("pose"), f"scan {ordinal} pose")
    semantics = _mapping(
        pose.get("transformSemantics"),
        f"scan {ordinal} transform semantics",
    )
    rotation = _mapping(pose.get("rotation"), f"scan {ordinal} rotation")
    translation = _mapping(pose.get("translation"), f"scan {ordinal} translation")
    r_components = _mapping(
        rotation.get("components"),
        f"scan {ordinal} rotation components",
    )
    t_components = _mapping(
        translation.get("components"),
        f"scan {ordinal} translation components",
    )
    if (
        semantics.get("direction") != "scan_local_to_file_frame"
        or semantics.get("formula") != "p_file = R(q) * p_scan_local + t"
        or semantics.get("quaternionComponentOrder") != ["w", "x", "y", "z"]
        or semantics.get("translationUnit") != "metre"
        or pose.get("provenance") != "publisher_declared_xml_metadata"
        or pose.get("validatedAgainstCompressedVectorRecords") is not False
    ):
        raise ValueError(f"Hovi spatial scan {ordinal} pose semantics drifted")
    values = [
        _mapping(r_components.get(component), f"scan {ordinal} rotation {component}").get(
            "semanticDecimal"
        )
        for component in ("w", "x", "y", "z")
    ] + [
        _mapping(
            t_components.get(component),
            f"scan {ordinal} translation {component}",
        ).get("semanticDecimal")
        for component in ("x", "y", "z")
    ]
    if not isinstance(guid, str) or not isinstance(count, int):
        raise ValueError(f"Hovi spatial scan {ordinal} identity is invalid")
    return ScanBinding(
        ordinal=ordinal,
        guid=guid,
        publisher_records=count,
        pose_f64_bits=tuple(
            _f64_bits(value, f"scan {ordinal} pose component {index}")
            for index, value in enumerate(values)
        ),
    )


def _configured_scans(value: Any) -> tuple[ScanBinding, ...]:
    if not isinstance(value, list) or len(value) != 16:
        raise ValueError("Hovi spatial authority must bind exactly 16 scans")
    result = []
    for expected_ordinal, item in enumerate(value):
        scan = _mapping(item, f"configured scan {expected_ordinal}")
        bits = scan.get("pose_f64_bits")
        if (
            set(scan) != {"ordinal", "guid", "publisher_records", "pose_f64_bits"}
            or scan.get("ordinal") != expected_ordinal
            or not isinstance(scan.get("guid"), str)
            or not isinstance(scan.get("publisher_records"), int)
            or not isinstance(bits, list)
            or len(bits) != 7
            or any(
                not isinstance(bit, str)
                or len(bit) != 16
                or bit != bit.lower()
                or any(character not in "0123456789abcdef" for character in bit)
                for bit in bits
            )
        ):
            raise ValueError(f"Hovi spatial configured scan {expected_ordinal} is invalid")
        result.append(
            ScanBinding(
                ordinal=expected_ordinal,
                guid=str(scan["guid"]),
                publisher_records=int(scan["publisher_records"]),
                pose_f64_bits=tuple(bits),
            )
        )
    return tuple(result)


def load_spatial_authority(
    path: Path = DEFAULT_AUTHORITY,
    *,
    source_root: Path = DEFAULT_OUTPUT_ROOT,
) -> SpatialMaterializationAuthority:
    """Load and verify the point-free spatial materialization decision."""
    path = _absolute(path)
    encoded = _read_exact(path, EXPECTED_AUTHORITY_BYTES, "authority config")
    config_sha256 = hashlib.sha256(encoded).hexdigest()
    if config_sha256 != EXPECTED_AUTHORITY_SHA256:
        raise ValueError("Hovi spatial authority is not the frozen decision")
    raw = json.loads(encoded)
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "id",
        "status",
        "mandate_provenance",
        "authorization",
        "dataset_and_license",
        "retained_e57",
        "selected_metadata_inventory",
        "scan_contract",
        "spatial_contract",
        "record_abi",
        "resource_gate",
        "native_reader",
        "artifact_policy",
        "evidence_boundary",
    }:
        raise ValueError("unsupported Hovi spatial authority")

    mandate = _mapping(raw["mandate_provenance"], "mandate provenance")
    authorization = _mapping(raw["authorization"], "authorization")
    dataset = _mapping(raw["dataset_and_license"], "dataset and license")
    retained = _mapping(raw["retained_e57"], "retained E57")
    metadata_ref = _mapping(
        retained.get("metadata_inspection_authorization"),
        "metadata authorization",
    )
    receipt_ref = _mapping(retained.get("extraction_receipt"), "extraction receipt")
    object_ref = _mapping(retained.get("object"), "E57 object")
    inventory_ref = _mapping(
        raw["selected_metadata_inventory"],
        "selected metadata inventory",
    )
    scan_contract = _mapping(raw["scan_contract"], "scan contract")
    spatial = _mapping(raw["spatial_contract"], "spatial contract")
    abi = _mapping(raw["record_abi"], "record ABI")
    resources = _mapping(raw["resource_gate"], "resource gate")
    native = _mapping(raw["native_reader"], "native reader")
    artifacts = _mapping(raw["artifact_policy"], "artifact policy")
    boundary = _mapping(raw["evidence_boundary"], "evidence boundary")

    expected_mandate = {
        "date": "2026-07-14",
        "source": "explicit_user_instruction_in_active_session",
        "instruction": "continue autonomously",
        "narrow_interpretation": (
            "prepare one exact full-scan spatial materialization boundary"
        ),
        "does_not_authorize": [
            "ground_filtering",
            "surface_reconstruction",
            "analogue_qualification",
            "target_truth",
            "synthesis",
        ],
    }
    expected_authorization = {
        "decision": "FULL_SCAN_SPATIAL_MATERIALIZATION_PREPARE",
        "full_scan_decode_authorized_after_native_binding": True,
        "spatial_materialization_authorized_after_native_binding": True,
        "execution_authorized_now": False,
        "point_conversion_authorized": False,
        "ground_filtering_authorized": False,
        "surface_reconstruction_authorized": False,
        "analogue_qualification_authorized": False,
        "target_truth": False,
        "synthesis_authorized": False,
    }
    expected_dataset = {
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
            "LAAS decoded and spatially sharded raw scan records within a bounded "
            "support AOI; it did not ground-filter, reconstruct, or qualify a surface."
        ),
    }
    if (
        raw["schema_version"] != AUTHORITY_SCHEMA
        or raw["id"] != AUTHORITY_ID
        or raw["status"] != "authorized_pending_native_command_and_build_binding"
        or dict(mandate) != expected_mandate
        or dict(authorization) != expected_authorization
        or dict(dataset) != expected_dataset
    ):
        raise ValueError("Hovi spatial mandate or license boundary changed")

    if metadata_ref != {
        "path": "config/evidence/hovi-hy-spruce4-full-metadata-inspection.json",
        "bytes": 3188,
        "sha256": "fe44c5c05c4ccd50cda2c9a662d3c171e3a281f823691d674010ee908c4480b9",
    }:
        raise ValueError("Hovi spatial metadata authorization binding drifted")
    metadata_authorization = load_metadata_inspection_authorization(
        _asset_path(metadata_ref.get("path"), "metadata authorization path")
    )
    source_root = _absolute(source_root)
    binding = _load_extraction_binding(metadata_authorization, source_root)
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
        raise ValueError("Hovi spatial receipt/object binding drifted")

    expected_inventory_ref = {
        "path": (
            "data/in/evidence/hovi-full-metadata/inventories/sha256/54/"
            f"{INVENTORY_SHA256}.json"
        ),
        "bytes": INVENTORY_BYTES,
        "sha256": INVENTORY_SHA256,
        "schema_version": INVENTORY_SCHEMA,
        "implementation_fingerprint_sha256": INVENTORY_IMPLEMENTATION_SHA256,
        "records_decoded": 0,
    }
    if dict(inventory_ref) != expected_inventory_ref:
        raise ValueError("Hovi spatial selected inventory binding drifted")
    inventory_path = _asset_path(inventory_ref.get("path"), "inventory path")
    inventory_bytes = _read_exact(inventory_path, INVENTORY_BYTES, "metadata inventory")
    if (
        hashlib.sha256(inventory_bytes).hexdigest() != INVENTORY_SHA256
        or canonical_json_bytes(json.loads(inventory_bytes)) != inventory_bytes
    ):
        raise ValueError("Hovi spatial metadata inventory bytes drifted")
    inventory = json.loads(inventory_bytes)
    if (
        not isinstance(inventory, Mapping)
        or inventory.get("schemaVersion") != INVENTORY_SCHEMA
        or inventory.get("status") != "metadata_inventory_complete"
        or inventory.get("pointsRead") != 0
        or inventory.get("compressedVectorRecordsDecoded") != 0
        or inventory.get("synthesisAuthorized") is not False
        or inventory.get("targetTruth") is not False
        or inventory.get("usableSurface") is not False
    ):
        raise ValueError("Hovi spatial inventory crossed its point-free boundary")
    implementation = _mapping(inventory.get("implementation"), "inventory implementation")
    fingerprint = _mapping(
        implementation.get("loadedExecutionFingerprint"),
        "inventory implementation fingerprint",
    )
    if fingerprint.get("sha256") != INVENTORY_IMPLEMENTATION_SHA256:
        raise ValueError("Hovi spatial inventory implementation drifted")
    interpreted = _mapping(inventory.get("inventory"), "interpreted inventory")
    root = _mapping(interpreted.get("root"), "inventory root")
    data3d = _mapping(root.get("data3D"), "inventory data3D")
    inventory_scans = data3d.get("scans")
    if not isinstance(inventory_scans, list) or len(inventory_scans) != 16:
        raise ValueError("Hovi spatial inventory must contain exactly 16 scans")
    observed_scans = tuple(
        _inventory_scan_binding(_mapping(scan, f"inventory scan {ordinal}"), ordinal)
        for ordinal, scan in enumerate(inventory_scans)
    )
    configured_scans = _configured_scans(scan_contract.get("scans"))
    if (
        set(scan_contract) != {
            "scan_count",
            "publisher_record_count_total",
            "pose_direction",
            "pose_formula",
            "pose_f64_bit_order",
            "scans",
        }
        or scan_contract.get("scan_count") != 16
        or scan_contract.get("publisher_record_count_total") != PUBLISHER_RECORD_COUNT
        or scan_contract.get("pose_direction") != "scan_local_to_file_frame"
        or scan_contract.get("pose_formula") != "p_file = R(q) * p_scan_local + t"
        or scan_contract.get("pose_f64_bit_order")
        != [
            "rotation_w",
            "rotation_x",
            "rotation_y",
            "rotation_z",
            "translation_x",
            "translation_y",
            "translation_z",
        ]
        or configured_scans != observed_scans
        or sum(scan.publisher_records for scan in configured_scans)
        != PUBLISHER_RECORD_COUNT
    ):
        raise ValueError("Hovi spatial 16-scan identity/pose contract drifted")

    expected_spatial = {
        "assignment_coordinate_frame": "publisher_file_frame",
        "stored_coordinate_frame": "raw_scan_local_scaled_integers",
        "tick_metres": 0.025,
        "tick_metres_ratio": [1, 40],
        "aoi_ticks": {
            "x_min_inclusive": -271,
            "x_max_exclusive": 1491,
            "y_min_inclusive": -244,
            "y_max_exclusive": 1351,
        },
        "shard_metres": 4,
        "shard_ticks": 160,
        "shard_indexing": "aoi_local_euclidean_floor",
        "point_shard_x_indices_inclusive": [0, 11],
        "point_shard_y_indices_inclusive": [0, 9],
        "valid_outside_aoi_policy": "count_then_drop",
        "invalid_and_direction_policy": "retain_per_scan_nonpoint_stream",
    }
    expected_abi = {
        "byte_order": "little_endian",
        "bytes_per_record": RECORD_BYTES,
        "fields": [
            "sourceOrdinal:u32@0",
            "rawX:i32@4",
            "rawY:i32@8",
            "rawZ:i32@12",
            "rawIntensitySingleBits:u32@16",
            "row:u16@20",
            "column:u16@22",
            "scanOrdinal:u8@24",
            "cartesianInvalidState:u8@25",
            "flags:u16@26",
        ],
        "flags": {"bit0": "intensity_finite", "all_other_bits": "zero"},
    }
    expected_resources = {
        "materialized_tree_ceiling_bytes": TREE_CEILING_BYTES,
        "materialized_tree_ceiling_label": "120 GiB",
        "launch_free_bytes_required": LAUNCH_FREE_BYTES,
        "launch_free_space_label": "216 GiB",
        "free_bytes_reserve": FREE_RESERVE_BYTES,
        "free_space_reserve_label": "96 GiB",
        "maximum_native_open_files_argument": 24,
        "maximum_native_open_output_files_reported": 20,
        "stdout_bytes": 1 << 20,
        "stderr_bytes": 1 << 20,
        "wall_seconds": 86_400,
        "cpu_seconds": 86_400,
        "rss_bytes": 2 << 30,
        "supervision_sample_seconds": 1,
    }
    expected_native = {
        "command": "materialize",
        "reviewed_contract_commit": (
            "7dfed50bf333880646f0c8a3222569d360e75c64"
        ),
        "report_schema": "laas-hovi-e57-spatial-materialization/1.0.0",
        "profile": "hy-spruce4",
        "max_file_bytes": OBJECT_BYTES,
        "max_xml_bytes": 65_536,
        "max_scans": 16,
        "max_prototype_fields": 7,
        "max_declared_records": PUBLISHER_RECORD_COUNT,
        "max_json_bytes": 1 << 20,
        "max_output_bytes": TREE_CEILING_BYTES,
        "max_open_files": 24,
        "shard_sink_buffer_bytes": 1 << 20,
        "build_status": "awaiting_frozen_native_command_and_build_identity",
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
    }
    expected_artifacts = {
        "native_output_is_staging_only": True,
        "native_report_atomic_publication": False,
        "independent_artifact_size_count_sha256_verification_required": True,
        "independent_scan_count_partition_verification_required": True,
        "reject_unreported_files": True,
        "manifest_is_acceptance_commit": True,
        "publish": "atomic_noreplace_directory_rename_after_manifest_fsync",
        "content_address": (
            "materializations/sha256/<manifest-sha256-prefix>/<manifest-sha256>"
        ),
    }
    expected_boundary = {
        "scientific_role": "raw_candidate_spatial_materialization",
        "qualification_status": "unqualified",
        "ground_filtered": False,
        "surface_claim": False,
        "analogue_qualification_authorized": False,
        "target_truth": False,
        "synthesis_authorized": False,
    }
    if (
        dict(spatial) != expected_spatial
        or dict(abi) != expected_abi
        or dict(resources) != expected_resources
        or dict(native) != expected_native
        or dict(artifacts) != expected_artifacts
        or dict(boundary) != expected_boundary
    ):
        raise ValueError("Hovi spatial representation/resource/evidence boundary changed")

    return SpatialMaterializationAuthority(
        config_path=path,
        config_sha256=config_sha256,
        inventory_path=inventory_path,
        inventory_sha256=INVENTORY_SHA256,
        source_root=source_root,
        scans=configured_scans,
        resources=SpatialResources(
            tree_ceiling_bytes=TREE_CEILING_BYTES,
            launch_free_bytes=LAUNCH_FREE_BYTES,
            free_reserve_bytes=FREE_RESERVE_BYTES,
            native_max_open_files=24,
            native_max_open_output_files=20,
            stdout_bytes=1 << 20,
            stderr_bytes=1 << 20,
            wall_seconds=86_400,
            cpu_seconds=86_400,
            rss_bytes=2 << 30,
            supervision_sample_seconds=1.0,
        ),
        native=NativeBinding(
            command="materialize",
            report_schema="laas-hovi-e57-spatial-materialization/1.0.0",
            execution_authorized=False,
            executable_path=None,
            executable_bytes=None,
            executable_sha256=None,
        ),
        mandate_provenance=canonical_json_bytes(expected_mandate),
        dataset_and_license=canonical_json_bytes(expected_dataset),
    )
