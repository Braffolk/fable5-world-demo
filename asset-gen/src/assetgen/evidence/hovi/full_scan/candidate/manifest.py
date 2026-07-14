"""Fail-closed binding to one published Hovi spatial materialization."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import struct
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .....config import ASSET_GEN_ROOT
from ..authorization import canonical_json_bytes, open_regular_nofollow
from ..spatial.authority import (
    SpatialMaterializationAuthority,
    load_spatial_authority,
)
from ..spatial.selection import (
    DEFAULT_EXECUTION_SELECTION_ROOT,
    load_spatial_execution_selection,
)


MANIFEST_SCHEMA = "hovi-hy-spruce4-spatial-materialization-manifest/1.0.0"
INVENTORY_SCHEMA = "hovi-hy-spruce4-full-metadata-inventory/2.0.0"
INVENTORY_SHA256 = "54e21049ac09c7d3f69ac7ab5990c982d12a7bd04da0618de032c8311cd3cfad"
OBJECT_SHA256 = "7889a9cfef69d57a50eaa5a9f8d224d28486d5901163e08cbdc827f92f409da2"
OBJECT_BYTES = 84_817_645_568
PUBLISHER_RECORDS = 4_248_797_321
RECORD_BYTES = 28
RAW_SCALE_F64_BITS = 0x3EE4F8B588E368F1
RESOLUTIONS_M = (1.0, 0.25, 0.125, 0.0625, 0.05, 0.025)
AOI_X_MIN_M = -271.0 / 40.0
AOI_Y_MIN_M = -244.0 / 40.0
SHARD_M = 4.0

_SHA256 = re.compile(r"[0-9a-f]{64}")
_POINT_PATH = re.compile(
    r"shards/scan-(?P<scan>\d{2})/x(?P<x>[+-]\d{3})_y(?P<y>[+-]\d{3})\.bin"
)
_NONPOINT_PATH = re.compile(r"nonpoints/scan-(?P<scan>\d{2})\.bin")


@dataclass(frozen=True)
class ScanTransform:
    ordinal: int
    guid: str
    publisher_records: int
    quaternion_wxyz: tuple[float, float, float, float]
    translation_xyz: tuple[float, float, float]
    raw_scale_xyz: tuple[float, float, float]
    raw_offset_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class PointShard:
    scan: int
    shard_x: int
    shard_y: int
    relative_path: str
    records: int
    bytes: int
    sha256: str


@dataclass(frozen=True)
class VerifiedSpatialManifest:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    inventory_path: Path
    authority: SpatialMaterializationAuthority
    scans: tuple[ScanTransform, ...]
    point_shards: tuple[PointShard, ...]

    def shards_for_scan(self, scan: int) -> tuple[PointShard, ...]:
        _scan(self.scans, scan)
        return tuple(shard for shard in self.point_shards if shard.scan == scan)

    def shards_at(self, shard_x: int, shard_y: int) -> tuple[PointShard, ...]:
        if not 0 <= shard_x <= 11 or not 0 <= shard_y <= 9:
            raise ValueError("Hovi shard key lies outside the frozen support AOI")
        return tuple(
            shard
            for shard in self.point_shards
            if shard.shard_x == shard_x and shard.shard_y == shard_y
        )


def _scan(scans: tuple[ScanTransform, ...], ordinal: int) -> ScanTransform:
    if not isinstance(ordinal, int) or not 0 <= ordinal < len(scans):
        raise ValueError("Hovi scan ordinal is outside the verified manifest")
    scan = scans[ordinal]
    if scan.ordinal != ordinal:
        raise ValueError("Hovi verified scan order is inconsistent")
    return scan


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Hovi manifest repeats JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Hovi manifest contains non-JSON number {value}")


def _read_regular(path: Path, label: str, *, mode: int = 0o400) -> tuple[bytes, os.stat_result]:
    descriptor = open_regular_nofollow(path, label)
    try:
        before = os.fstat(descriptor)
        if (
            before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) != mode
            or before.st_nlink != 1
        ):
            raise PermissionError(f"Hovi {label} is not a private immutable file")
        payload = bytearray()
        while block := os.read(descriptor, 1 << 20):
            payload.extend(block)
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after) or len(payload) != before.st_size:
            raise ValueError(f"Hovi {label} changed while being read")
        return bytes(payload), before
    finally:
        os.close(descriptor)


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Hovi {label} must be an object")
    return value


def _safe_relative(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Hovi {label} must be a relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts or "\\" in value:
        raise ValueError(f"unsafe Hovi {label}")
    return value


def _asset_path(value: Any, label: str) -> Path:
    relative = _safe_relative(value, label)
    pure = PurePosixPath(relative)
    return Path(os.path.abspath(os.fspath(ASSET_GEN_ROOT.joinpath(*pure.parts))))


def _f64_from_bits(value: Any, label: str) -> float:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{16}", value):
        raise ValueError(f"Hovi {label} must contain exact lowercase f64 bits")
    result = struct.unpack(">d", bytes.fromhex(value))[0]
    if not math.isfinite(result):
        raise ValueError(f"Hovi {label} is non-finite")
    return result


def _load_raw_transforms(inventory_path: Path) -> tuple[tuple[float, ...], ...]:
    payload, result = _read_regular(inventory_path, "metadata inventory")
    if result.st_size != 806_369 or hashlib.sha256(payload).hexdigest() != INVENTORY_SHA256:
        raise ValueError("Hovi metadata inventory differs from the frozen source")
    raw = json.loads(
        payload.decode("utf-8", errors="strict"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    if raw.get("schemaVersion") != INVENTORY_SCHEMA or raw.get("pointsRead") != 0:
        raise ValueError("Hovi metadata inventory crossed its point-free boundary")
    inventory = _mapping(raw.get("inventory"), "inventory")
    root = _mapping(inventory.get("root"), "inventory root")
    data3d = _mapping(root.get("data3D"), "inventory data3D")
    scans = data3d.get("scans")
    if not isinstance(scans, list) or len(scans) != 16:
        raise ValueError("Hovi metadata inventory must contain exactly 16 scans")
    transforms: list[tuple[float, ...]] = []
    for ordinal, value in enumerate(scans):
        scan = _mapping(value, f"inventory scan {ordinal}")
        points = _mapping(scan.get("points"), f"scan {ordinal} points")
        prototype = _mapping(
            points.get("prototype"),
            f"scan {ordinal} prototype",
        )
        children = prototype.get("children")
        if not isinstance(children, list):
            raise ValueError(f"Hovi scan {ordinal} prototype children are absent")
        by_name = {
            _mapping(child, "prototype child").get("localName"): child
            for child in children
        }
        values: list[float] = []
        for axis in "XYZ":
            child = _mapping(
                by_name.get(f"cartesian{axis}"),
                f"scan {ordinal} cartesian{axis}",
            )
            scale = _mapping(child.get("scale"), f"scan {ordinal} cartesian{axis} scale")
            offset = _mapping(child.get("offset"), f"scan {ordinal} cartesian{axis} offset")
            parsed_scale = float(scale.get("effectiveValue"))
            parsed_offset = float(offset.get("effectiveValue"))
            if (
                child.get("type") != "ScaledInteger"
                or struct.unpack(">Q", struct.pack(">d", parsed_scale))[0] != RAW_SCALE_F64_BITS
                or parsed_offset != 0.0
            ):
                raise ValueError(f"Hovi scan {ordinal} raw coordinate transform drifted")
            values.extend((parsed_scale, parsed_offset))
        transforms.append(tuple(values))
    return tuple(transforms)


def _verify_executable(authority: SpatialMaterializationAuthority) -> None:
    native = authority.native
    if (
        not native.execution_authorized
        or native.executable_path is None
        or native.executable_bytes is None
        or native.executable_sha256 is None
    ):
        raise PermissionError("Hovi candidate input requires execution-authorized materialization")
    descriptor = open_regular_nofollow(
        native.executable_path,
        "spatial materializer executable",
    )
    try:
        before = os.fstat(descriptor)
        digest = hashlib.sha256()
        byte_count = 0
        while block := os.read(descriptor, 1 << 20):
            digest.update(block)
            byte_count += len(block)
        after = os.fstat(descriptor)
        if (
            before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) != 0o500
            or before.st_nlink != 1
            or _identity(before) != _identity(after)
            or byte_count != native.executable_bytes
            or digest.hexdigest() != native.executable_sha256
        ):
            raise PermissionError("Hovi spatial materializer executable identity drifted")
    finally:
        os.close(descriptor)


def _verify_native_report(
    value: Any,
    authority: SpatialMaterializationAuthority,
) -> None:
    report = _mapping(value, "native report")
    document = _mapping(report.get("document"), "native report document")
    reader = _mapping(document.get("reader"), "native report reader")
    source = _mapping(document.get("source"), "native report source")
    result = _mapping(document.get("result"), "native report result")
    digest = report.get("sha256")
    byte_count = report.get("bytes")
    if (
        not isinstance(byte_count, int)
        or not 0 < byte_count <= authority.resources.stdout_bytes
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or document.get("schemaVersion") != authority.native.report_schema
        or document.get("status") != "spatial_materialization_complete"
        or document.get("profile") != "hy-spruce4"
        or reader.get("upstreamE57Version") != "0.11.13"
        or reader.get("upstreamCrateSha256")
        != "fcfee41a50fbd70278c70cc477b8671ffe03951cba028cc2c057777a9ee6a3cc"
        or reader.get("exactCountAndSectionEofValidated") is not True
        or source.get("inputMode") != "inherited_fd"
        or source.get("bytes") != str(OBJECT_BYTES)
        or result
        != {
            "allScansDecodedOnce": True,
            "scanCount": 16,
            "publisherRecordCount": str(PUBLISHER_RECORDS),
            "recordsOutsideAoiDropped": True,
            "invalidAndDirectionRetained": True,
            "groundFiltered": False,
            "surfaceClaim": False,
            "targetTruth": False,
            "synthesisAuthorized": False,
        }
    ):
        raise ValueError("Hovi native materialization report identity/boundary drifted")


def load_verified_spatial_manifest(path: Path) -> VerifiedSpatialManifest:
    """Verify a published manifest and its frozen coordinate metadata.

    Artifact bytes are verified immediately before a shard is decoded by the record
    reader. A missing materialization, mutable manifest, or non-content-addressed
    location fails closed.
    """
    path = Path(os.path.abspath(os.fspath(path)))
    if not path.is_file() or path.name != "manifest.json":
        raise FileNotFoundError(f"verified Hovi spatial manifest is absent: {path}")
    payload, _ = _read_regular(path, "spatial acceptance manifest")
    digest = hashlib.sha256(payload).hexdigest()
    if path.parent.name != digest or path.parent.parent.name != digest[:2]:
        raise ValueError("Hovi spatial manifest is not at its content address")
    raw = json.loads(
        payload.decode("utf-8", errors="strict"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    boundary = _mapping(raw.get("evidenceBoundary"), "evidence boundary")
    spatial = _mapping(raw.get("spatialContract"), "spatial contract")
    abi = _mapping(raw.get("recordAbi"), "record ABI")
    source = _mapping(raw.get("source"), "source")
    authority_ref = _mapping(raw.get("authority"), "authority reference")
    expected_spatial = {
        "coordinateFrame": "publisher_file_frame_for_assignment_only",
        "storedCoordinates": "raw_scan_local_scaled_integers",
        "tickMetres": 0.025,
        "aoiTicks": {
            "xMinInclusive": -271,
            "xMaxExclusive": 1491,
            "yMinInclusive": -244,
            "yMaxExclusive": 1351,
        },
        "shardTicks": 160,
        "shardMetres": 4.0,
        "shardIndexing": "aoi_local_euclidean_floor",
    }
    expected_abi = {
        "byteOrder": "little_endian",
        "bytesPerRecord": RECORD_BYTES,
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
        "flags": {"bit0": "intensity_finite", "allOtherBits": "zero"},
    }
    if (
        raw.get("schemaVersion") != MANIFEST_SCHEMA
        or raw.get("status") != "verified_staging_ready_for_atomic_publication"
        or spatial != expected_spatial
        or abi != expected_abi
        or boundary != {
            "scientificRole": "raw_candidate_spatial_materialization",
            "qualificationStatus": "unqualified",
            "groundFiltered": False,
            "surfaceClaim": False,
            "analogueQualificationAuthorized": False,
            "targetTruth": False,
            "synthesisAuthorized": False,
        }
        or source.get("selectedMetadataInventorySha256") != INVENTORY_SHA256
        or source.get("e57Sha256") != OBJECT_SHA256
        or source.get("e57Bytes") != OBJECT_BYTES
    ):
        raise ValueError("Hovi spatial manifest crossed the frozen evidence boundary")
    inventory_path = Path(str(source.get("selectedMetadataInventoryPath", "")))
    if not inventory_path.is_absolute():
        raise ValueError("Hovi spatial manifest metadata path is not absolute")
    raw_transforms = _load_raw_transforms(inventory_path)
    if set(authority_ref) != {
        "configPath",
        "configSha256",
        "mandateProvenance",
        "executionSelection",
    }:
        raise ValueError("Hovi spatial manifest authority reference is unsupported")
    authority_path = _asset_path(
        authority_ref.get("configPath"),
        "authority config path",
    )
    base_authority = load_spatial_authority(authority_path)
    if (
        authority_ref.get("configSha256") != base_authority.config_sha256
        or authority_ref.get("mandateProvenance")
        != json.loads(base_authority.mandate_provenance)
        or base_authority.inventory_path != inventory_path
    ):
        raise ValueError("Hovi spatial manifest does not bind the frozen authority")
    selection_ref = _mapping(
        authority_ref.get("executionSelection"),
        "execution selection",
    )
    if set(selection_ref) != {"bytes", "sha256", "document"}:
        raise ValueError("Hovi manifest execution selection has unsupported fields")
    selection_document = _mapping(
        selection_ref.get("document"),
        "execution selection document",
    )
    selection_bytes = canonical_json_bytes(selection_document)
    selection_sha256 = hashlib.sha256(selection_bytes).hexdigest()
    if (
        selection_ref.get("bytes") != len(selection_bytes)
        or selection_ref.get("sha256") != selection_sha256
    ):
        raise ValueError("Hovi embedded execution selection identity is invalid")
    selection_path = (
        DEFAULT_EXECUTION_SELECTION_ROOT
        / selection_sha256[:2]
        / f"{selection_sha256}.json"
    )
    authority = load_spatial_execution_selection(
        selection_path,
        authority=base_authority,
    )
    if (
        authority.execution_selection_sha256 != selection_sha256
        or authority.execution_identity != selection_bytes
    ):
        raise ValueError("Hovi selected execution differs from manifest embedding")
    _verify_executable(authority)
    _verify_native_report(raw.get("nativeReport"), authority)

    scan_values = raw.get("scans")
    if not isinstance(scan_values, list) or len(scan_values) != 16:
        raise ValueError("Hovi spatial manifest must contain exactly 16 scans")
    scans: list[ScanTransform] = []
    for ordinal, value in enumerate(scan_values):
        item = _mapping(value, f"manifest scan {ordinal}")
        bits = item.get("poseF64Bits")
        guid = item.get("guid")
        publisher_records = item.get("publisherRecords")
        if (
            item.get("ordinal") != ordinal
            or not isinstance(guid, str)
            or re.fullmatch(r"[0-9a-f]{16}", guid) is None
            or not isinstance(publisher_records, int)
            or publisher_records <= 0
            or not isinstance(bits, list)
            or len(bits) != 7
        ):
            raise ValueError(f"Hovi manifest scan {ordinal} identity/pose is invalid")
        pose = tuple(_f64_from_bits(bit, f"scan {ordinal} pose") for bit in bits)
        quaternion_norm = math.sqrt(sum(component * component for component in pose[:4]))
        if abs(quaternion_norm - 1.0) > 1.0e-6:
            raise ValueError(f"Hovi manifest scan {ordinal} quaternion is not unit length")
        scale_offset = raw_transforms[ordinal]
        scales = (scale_offset[0], scale_offset[2], scale_offset[4])
        offsets = (scale_offset[1], scale_offset[3], scale_offset[5])
        scans.append(
            ScanTransform(
                ordinal=ordinal,
                guid=guid,
                publisher_records=publisher_records,
                quaternion_wxyz=pose[:4],
                translation_xyz=pose[4:],
                raw_scale_xyz=scales,
                raw_offset_xyz=offsets,
            )
        )
        frozen = authority.scans[ordinal]
        if (
            guid != frozen.guid
            or publisher_records != frozen.publisher_records
            or tuple(bits) != frozen.pose_f64_bits
        ):
            raise ValueError(f"Hovi manifest scan {ordinal} differs from frozen authority")
    if sum(scan.publisher_records for scan in scans) != PUBLISHER_RECORDS:
        raise ValueError("Hovi manifest publisher scan counts do not conserve records")

    artifact_values = raw.get("artifacts")
    if not isinstance(artifact_values, list):
        raise ValueError("Hovi spatial manifest artifact inventory is absent")
    point_shards: list[PointShard] = []
    seen: set[tuple[str, int, int | None, int | None]] = set()
    artifact_records = 0
    artifact_bytes = 0
    point_records_by_scan = [0] * 16
    nonpoint_records_by_scan = [0] * 16
    for value in artifact_values:
        item = _mapping(value, "artifact")
        kind = item.get("kind")
        relative = _safe_relative(item.get("path"), "point-shard path")
        records = item.get("records")
        byte_count = item.get("bytes")
        sha256 = item.get("sha256")
        if (
            not isinstance(records, int)
            or records <= 0
            or byte_count != records * RECORD_BYTES
            or not isinstance(sha256, str)
            or _SHA256.fullmatch(sha256) is None
        ):
            raise ValueError("Hovi point-shard identity/count/hash is invalid")
        if kind == "point_shard":
            match = _POINT_PATH.fullmatch(relative)
            if match is None:
                raise ValueError("Hovi point-shard path is non-canonical")
            scan = int(match.group("scan"))
            shard_x = int(match.group("x"))
            shard_y = int(match.group("y"))
            if (
                item.get("scan") != scan
                or item.get("shardX") != shard_x
                or item.get("shardY") != shard_y
                or not 0 <= scan < 16
                or not 0 <= shard_x <= 11
                or not 0 <= shard_y <= 9
            ):
                raise ValueError("Hovi point-shard spatial identity is invalid")
            point_records_by_scan[scan] += records
            point_shards.append(
                PointShard(
                    scan,
                    shard_x,
                    shard_y,
                    relative,
                    records,
                    byte_count,
                    sha256,
                )
            )
        elif kind == "nonpoint_stream":
            match = _NONPOINT_PATH.fullmatch(relative)
            if match is None:
                raise ValueError("Hovi nonpoint-stream path is non-canonical")
            scan = int(match.group("scan"))
            shard_x = shard_y = None
            if (
                item.get("scan") != scan
                or item.get("shardX") is not None
                or item.get("shardY") is not None
                or not 0 <= scan < 16
            ):
                raise ValueError("Hovi nonpoint-stream identity is invalid")
            nonpoint_records_by_scan[scan] += records
        else:
            raise ValueError("Hovi spatial manifest contains an unknown artifact kind")
        key = (kind, scan, shard_x, shard_y)
        if key in seen:
            raise ValueError("Hovi spatial manifest repeats an artifact identity")
        seen.add(key)
        artifact_records += records
        artifact_bytes += byte_count
    if not point_shards:
        raise ValueError("Hovi spatial manifest contains no AOI point shards")
    totals = _mapping(raw.get("totals"), "materialization totals")
    if (
        totals.get("publisherRecordsDecoded") != PUBLISHER_RECORDS
        or totals.get("materializedRecords") != artifact_records
        or totals.get("materializedBytes") != artifact_bytes
        or totals.get("artifactCount") != len(artifact_values)
    ):
        raise ValueError("Hovi spatial manifest artifact totals are inconsistent")
    for ordinal, value in enumerate(scan_values):
        item = _mapping(value, f"manifest scan {ordinal}")
        if (
            item.get("validInAoi") != point_records_by_scan[ordinal]
            or item.get("directionRecords", 0) + item.get("invalidRecords", 0)
            != nonpoint_records_by_scan[ordinal]
        ):
            raise ValueError(f"Hovi manifest scan {ordinal} artifact partition is inconsistent")
    return VerifiedSpatialManifest(
        root=path.parent,
        manifest_path=path,
        manifest_sha256=digest,
        inventory_path=inventory_path,
        authority=authority,
        scans=tuple(scans),
        point_shards=tuple(
            sorted(
                point_shards,
                key=lambda value: (value.scan, value.shard_y, value.shard_x),
            )
        ),
    )
