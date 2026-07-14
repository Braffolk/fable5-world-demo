"""Independent staging-tree verification for Hovi spatial materialization."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import struct
import time
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .....config import ASSET_GEN_ROOT
from ..authorization import canonical_json_bytes
from ..extract import (
    _atomic_rename_noreplace_at,
    _fsync_directory,
    _open_or_create_private_directory_at,
    _open_or_create_private_root,
)
from .authority import (
    FREE_RESERVE_BYTES,
    OBJECT_BYTES,
    OBJECT_SHA256,
    PUBLISHER_RECORD_COUNT,
    RECORD_BYTES,
    TREE_CEILING_BYTES,
    ScanBinding,
    SpatialMaterializationAuthority,
)


_POINT_PATH = re.compile(
    r"shards/scan-(?P<scan>\d{2})/x(?P<x>[+-]\d{3})_y(?P<y>[+-]\d{3})\.bin"
)
_NONPOINT_PATH = re.compile(r"nonpoints/scan-(?P<scan>\d{2})\.bin")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_ARTIFACTS = 16 * (12 * 10 + 1)
_FILE_GUID = "{C70845D7-08E0-4185-974C-EF84E9EDEA55}"
_VENDOR_PATCH_SHA256 = (
    "7ea3abf5c71d8a942489fc2e5dd991fe857fc5e03ed48fdcdfedaf7f638d0a72"
)
_XYZ_SCALE = float("1.00000000000000008e-05")
_RECORD_DTYPE = np.dtype(
    [
        ("source", "<u4"),
        ("raw_x", "<i4"),
        ("raw_y", "<i4"),
        ("raw_z", "<i4"),
        ("intensity_bits", "<u4"),
        ("row", "<u2"),
        ("column", "<u2"),
        ("scan", "u1"),
        ("invalid", "u1"),
        ("flags", "<u2"),
    ]
)
_VERIFY_RECORDS_PER_BLOCK = 1 << 18
if _RECORD_DTYPE.itemsize != RECORD_BYTES:
    raise RuntimeError("Hovi spatial NumPy ABI is not 28 bytes")


@dataclass(frozen=True)
class InodeIdentity:
    device: int
    inode: int
    mode: int
    uid: int
    gid: int
    links: int
    size: int
    modified_ns: int
    changed_ns: int
    flags: int


@dataclass(frozen=True)
class ArtifactObservation:
    valid: int
    direction: int
    invalid: int
    finite_intensity: int
    non_finite_intensity: int


@dataclass(frozen=True)
class VerifiedArtifact:
    kind: str
    scan: int
    shard_x: int | None
    shard_y: int | None
    relative_path: str
    records: int
    bytes: int
    sha256: str
    identity: InodeIdentity | None = None


@dataclass(frozen=True)
class VerifiedSpatialStaging:
    root: Path
    root_identity: InodeIdentity
    directory_identities: tuple[tuple[str, InodeIdentity], ...]
    file_identities: tuple[tuple[str, InodeIdentity], ...]
    native_report: bytes
    native_report_sha256: str
    artifacts: tuple[VerifiedArtifact, ...]
    record_count: int
    artifact_bytes: int
    manifest: bytes
    manifest_sha256: str


def _require_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("Hovi spatial verification exceeded its absolute deadline")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Hovi spatial report repeats JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Hovi spatial report contains non-JSON number {value}")


def _compact_report(payload: bytes) -> dict[str, Any]:
    if (
        not payload.startswith(b"{")
        or not payload.endswith(b"}\n")
        or payload.count(b"\n") != 1
        or b"\r" in payload
    ):
        raise ValueError("Hovi spatial stdout is not one compact JSON report")
    in_string = False
    escaped = False
    for byte in payload[:-1]:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:
                escaped = True
            elif byte == 0x22:
                in_string = False
        elif byte == 0x22:
            in_string = True
        elif byte in b" \t\r\n":
            raise ValueError("Hovi spatial report has non-canonical whitespace")
    if in_string or escaped:
        raise ValueError("Hovi spatial report ends inside a JSON string")
    parsed = json.loads(
        payload.decode("utf-8", errors="strict"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(parsed, dict):
        raise ValueError("Hovi spatial report must be an object")
    return parsed


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Hovi spatial report {label} must be an object")
    return value


def _require_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"Hovi spatial report {label} keys changed")


def _count(value: Any, label: str) -> int:
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise ValueError(f"Hovi spatial report {label} must be an unsigned decimal")
    return int(value)


def _artifact_from_report(
    value: Any,
    scans: tuple[ScanBinding, ...],
) -> VerifiedArtifact:
    item = _mapping(value, "artifact")
    kind = item.get("kind")
    path = item.get("path")
    scan = item.get("scan")
    records = _count(item.get("records"), "artifact records")
    byte_count = _count(item.get("bytes"), "artifact bytes")
    digest = item.get("sha256")
    if (
        kind not in {"point_shard", "nonpoint_stream"}
        or not isinstance(path, str)
        or not isinstance(scan, int)
        or not 0 <= scan < len(scans)
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or records <= 0
        or byte_count != records * RECORD_BYTES
        or byte_count > TREE_CEILING_BYTES
    ):
        raise ValueError("Hovi spatial artifact identity/count/ABI is invalid")
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or "\\" in path:
        raise ValueError("Hovi spatial artifact path is unsafe")
    shard_x = shard_y = None
    if kind == "point_shard":
        match = _POINT_PATH.fullmatch(path)
        if match is None:
            raise ValueError("Hovi spatial point-shard path is non-canonical")
        shard_x = int(match.group("x"))
        shard_y = int(match.group("y"))
        if (
            int(match.group("scan")) != scan
            or item.get("shardX") != shard_x
            or item.get("shardY") != shard_y
            or not 0 <= shard_x <= 11
            or not 0 <= shard_y <= 9
            or set(item)
            != {"kind", "scan", "shardX", "shardY", "path", "records", "bytes", "sha256"}
        ):
            raise ValueError("Hovi spatial point-shard key is outside the frozen AOI")
    else:
        match = _NONPOINT_PATH.fullmatch(path)
        if (
            match is None
            or int(match.group("scan")) != scan
            or set(item) != {"kind", "scan", "path", "records", "bytes", "sha256"}
        ):
            raise ValueError("Hovi spatial nonpoint-stream key is invalid")
    return VerifiedArtifact(
        kind=kind,
        scan=scan,
        shard_x=shard_x,
        shard_y=shard_y,
        relative_path=path,
        records=records,
        bytes=byte_count,
        sha256=digest,
    )


def _inode_identity(result: os.stat_result) -> InodeIdentity:
    return InodeIdentity(
        device=result.st_dev,
        inode=result.st_ino,
        mode=result.st_mode,
        uid=result.st_uid,
        gid=result.st_gid,
        links=result.st_nlink,
        size=result.st_size,
        modified_ns=result.st_mtime_ns,
        changed_ns=result.st_ctime_ns,
        flags=int(getattr(result, "st_flags", 0)),
    )


def _open_relative_regular_at(root: int, relative_path: str) -> int:
    parts = PurePosixPath(relative_path).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("Hovi spatial artifact path is unsafe")
    directory = os.dup(root)
    try:
        for part in parts[:-1]:
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory,
            )
            os.close(directory)
            directory = child
        return os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
    finally:
        os.close(directory)


def _open_staging_root(path: Path) -> int:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        namespace = os.lstat(path)
        held = os.fstat(descriptor)
        if _inode_identity(namespace) != _inode_identity(held):
            raise ValueError("Hovi spatial staging root changed while being opened")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _read_descriptor(descriptor: int, maximum_bytes: int, deadline: float) -> bytes:
    payload = bytearray()
    while len(payload) <= maximum_bytes:
        _require_deadline(deadline)
        block = os.read(descriptor, min(1 << 20, maximum_bytes + 1 - len(payload)))
        if not block:
            break
        payload.extend(block)
    if len(payload) > maximum_bytes:
        raise ValueError("Hovi spatial file exceeds its bounded read")
    return bytes(payload)


def _tree_inventory_at(
    root: int,
    *,
    allow_writable_files: bool,
    deadline: float,
) -> tuple[dict[str, InodeIdentity], dict[str, InodeIdentity]]:
    directories = {".": _inode_identity(os.fstat(root))}
    files: dict[str, InodeIdentity] = {}
    allowed_modes = {0o400, 0o600} if allow_writable_files else {0o400}

    def walk(directory: int, prefix: str) -> None:
        _require_deadline(deadline)
        for name in sorted(os.listdir(directory)):
            _require_deadline(deadline)
            if name in {"", ".", ".."} or "/" in name or "\\" in name:
                raise ValueError("Hovi spatial staging contains an unsafe name")
            relative = f"{prefix}/{name}" if prefix else name
            namespace = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if stat.S_ISDIR(namespace.st_mode):
                child = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory,
                )
                try:
                    held = os.fstat(child)
                    if (
                        _inode_identity(namespace) != _inode_identity(held)
                        or held.st_uid != os.getuid()
                        or stat.S_IMODE(held.st_mode) != 0o700
                    ):
                        raise PermissionError(
                            "Hovi spatial staging contains a non-private directory"
                        )
                    directories[relative] = _inode_identity(held)
                    walk(child, relative)
                finally:
                    os.close(child)
            elif stat.S_ISREG(namespace.st_mode):
                if (
                    namespace.st_uid != os.getuid()
                    or namespace.st_nlink != 1
                    or stat.S_IMODE(namespace.st_mode) not in allowed_modes
                ):
                    raise PermissionError(
                        "Hovi spatial staging contains a mutable or shared file"
                    )
                files[relative] = _inode_identity(namespace)
            else:
                raise PermissionError(
                    "Hovi spatial staging contains a non-regular entry"
                )

    root_stat = os.fstat(root)
    if (
        not stat.S_ISDIR(root_stat.st_mode)
        or root_stat.st_uid != os.getuid()
        or stat.S_IMODE(root_stat.st_mode) != 0o700
    ):
        raise PermissionError("Hovi spatial staging root must be private mode 0700")
    walk(root, "")
    _require_deadline(deadline)
    directories["."] = _inode_identity(os.fstat(root))
    return directories, files


def _pose_values(binding: ScanBinding) -> tuple[float, ...]:
    return tuple(
        struct.unpack(">d", bytes.fromhex(bits))[0] for bits in binding.pose_f64_bits
    )


def _validate_record_block(
    records: np.ndarray,
    artifact: VerifiedArtifact,
    binding: ScanBinding,
    previous_source: int | None,
    seen_sources: np.ndarray,
    deadline: float,
) -> tuple[int, ArtifactObservation]:
    _require_deadline(deadline)
    source = records["source"]
    if source.size == 0:
        raise ValueError("Hovi spatial artifact contains an empty record block")
    if (
        int(source[-1]) >= binding.publisher_records
        or (previous_source is not None and int(source[0]) <= previous_source)
        or (source.size > 1 and bool(np.any(source[1:] <= source[:-1])))
    ):
        raise ValueError("Hovi spatial source ordinals are not strictly increasing")
    byte_indices = np.right_shift(source, 3).astype(np.intp)
    masks = np.left_shift(
        np.uint8(1),
        np.bitwise_and(source, 7).astype(np.uint8),
    )
    boundaries = np.empty(source.size, dtype=np.bool_)
    boundaries[0] = True
    boundaries[1:] = byte_indices[1:] != byte_indices[:-1]
    starts = np.flatnonzero(boundaries)
    unique_bytes = byte_indices[starts]
    combined_masks = np.bitwise_or.reduceat(masks, starts)
    existing = seen_sources[unique_bytes]
    if bool(np.any(np.bitwise_and(existing, combined_masks))):
        raise ValueError("Hovi spatial source ordinal occurs in multiple artifacts")
    seen_sources[unique_bytes] = np.bitwise_or(existing, combined_masks)
    _require_deadline(deadline)
    if bool(np.any(records["scan"] != artifact.scan)):
        raise ValueError("Hovi spatial record scan byte differs from its artifact")
    for field in ("raw_x", "raw_y", "raw_z"):
        if bool(np.any(records[field] == np.iinfo(np.int32).min)):
            raise ValueError("Hovi spatial raw coordinate crossed its prototype domain")
    if (
        bool(np.any(records["row"] > 11_003))
        or bool(np.any(records["column"] > 27_480))
        or bool(np.any(records["invalid"] > 2))
    ):
        raise ValueError("Hovi spatial integer field crossed its prototype domain")

    intensity = records["intensity_bits"].view("<f4")
    finite = np.isfinite(intensity)
    if (
        bool(np.any(records["flags"] != finite.astype(np.uint16)))
        or bool(np.any(intensity[finite] < 0.0))
        or bool(np.any(intensity[finite] > 1.0))
    ):
        raise ValueError("Hovi spatial intensity bits/flags crossed their domain")

    local_x = records["raw_x"].astype(np.float64) * _XYZ_SCALE
    local_y = records["raw_y"].astype(np.float64) * _XYZ_SCALE
    local_z = records["raw_z"].astype(np.float64) * _XYZ_SCALE
    qw, qx, qy, qz, tx, ty, tz = _pose_values(binding)
    uv_x = qy * local_z - qz * local_y
    uv_y = qz * local_x - qx * local_z
    uv_z = qx * local_y - qy * local_x
    uuv_x = qy * uv_z - qz * uv_y
    uuv_y = qz * uv_x - qx * uv_z
    uuv_z = qx * uv_y - qy * uv_x
    file_x = local_x + 2.0 * (qw * uv_x + uuv_x) + tx
    file_y = local_y + 2.0 * (qw * uv_y + uuv_y) + ty
    file_z = local_z + 2.0 * (qw * uv_z + uuv_z) + tz
    if not (
        bool(np.all(np.isfinite(file_x)))
        and bool(np.all(np.isfinite(file_y)))
        and bool(np.all(np.isfinite(file_z)))
    ):
        raise ValueError("Hovi spatial pose reconstruction produced non-finite XYZ")
    _require_deadline(deadline)

    invalid = records["invalid"]
    if artifact.kind == "point_shard":
        if bool(np.any(invalid != 0)):
            raise ValueError("Hovi spatial point shard contains a nonpoint record")
        tick_x = np.floor(file_x * 40.0).astype(np.int64)
        tick_y = np.floor(file_y * 40.0).astype(np.int64)
        if (
            bool(np.any(tick_x < -271))
            or bool(np.any(tick_x >= 1491))
            or bool(np.any(tick_y < -244))
            or bool(np.any(tick_y >= 1351))
        ):
            raise ValueError("Hovi spatial point shard contains an out-of-AOI record")
        shard_x = np.floor_divide(tick_x + 271, 160)
        shard_y = np.floor_divide(tick_y + 244, 160)
        if (
            artifact.shard_x is None
            or artifact.shard_y is None
            or bool(np.any(shard_x != artifact.shard_x))
            or bool(np.any(shard_y != artifact.shard_y))
        ):
            raise ValueError("Hovi spatial point record is in the wrong AOI-local shard")
        valid = int(records.size)
        direction = 0
        invalid_count = 0
    else:
        if bool(np.any((invalid != 1) & (invalid != 2))):
            raise ValueError("Hovi spatial nonpoint stream contains a point record")
        valid = 0
        direction = int(np.count_nonzero(invalid == 1))
        invalid_count = int(np.count_nonzero(invalid == 2))
    finite_count = int(np.count_nonzero(finite))
    _require_deadline(deadline)
    return (
        int(source[-1]),
        ArtifactObservation(
            valid=valid,
            direction=direction,
            invalid=invalid_count,
            finite_intensity=finite_count,
            non_finite_intensity=int(records.size) - finite_count,
        ),
    )


def _verify_artifact(
    root: int,
    artifact: VerifiedArtifact,
    binding: ScanBinding,
    seen_sources: np.ndarray,
    deadline: float,
) -> tuple[VerifiedArtifact, ArtifactObservation]:
    _require_deadline(deadline)
    descriptor = _open_relative_regular_at(root, artifact.relative_path)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) not in {0o400, 0o600}
            or before.st_nlink != 1
            or before.st_size != artifact.bytes
        ):
            raise PermissionError("Hovi spatial artifact is not a private staging inode")
        before_identity = _inode_identity(before)
        digest = hashlib.sha256()
        total_bytes = 0
        total_records = 0
        carry = b""
        previous_source: int | None = None
        observed = ArtifactObservation(0, 0, 0, 0, 0)
        block_bytes = RECORD_BYTES * _VERIFY_RECORDS_PER_BLOCK
        while True:
            _require_deadline(deadline)
            block = os.read(descriptor, block_bytes)
            if not block:
                break
            digest.update(block)
            total_bytes += len(block)
            payload = carry + block
            complete = len(payload) - (len(payload) % RECORD_BYTES)
            if complete:
                records = np.frombuffer(payload[:complete], dtype=_RECORD_DTYPE)
                previous_source, block_observed = _validate_record_block(
                    records,
                    artifact,
                    binding,
                    previous_source,
                    seen_sources,
                    deadline,
                )
                total_records += int(records.size)
                observed = ArtifactObservation(
                    valid=observed.valid + block_observed.valid,
                    direction=observed.direction + block_observed.direction,
                    invalid=observed.invalid + block_observed.invalid,
                    finite_intensity=(
                        observed.finite_intensity + block_observed.finite_intensity
                    ),
                    non_finite_intensity=(
                        observed.non_finite_intensity
                        + block_observed.non_finite_intensity
                    ),
                )
            carry = payload[complete:]
            _require_deadline(deadline)
        after = os.fstat(descriptor)
        if (
            carry
            or total_bytes != artifact.bytes
            or total_records != artifact.records
            or _inode_identity(after) != before_identity
            or digest.hexdigest() != artifact.sha256
        ):
            raise ValueError("Hovi spatial artifact hash/count/identity changed")
        if stat.S_IMODE(after.st_mode) == 0o600:
            os.fchmod(descriptor, 0o400)
            os.fsync(descriptor)
        immutable = os.fstat(descriptor)
        _require_deadline(deadline)
        if (
            immutable.st_uid != os.getuid()
            or stat.S_IMODE(immutable.st_mode) != 0o400
            or immutable.st_nlink != 1
        ):
            raise PermissionError("Hovi spatial verified artifact is not immutable")
        return replace(artifact, identity=_inode_identity(immutable)), observed
    finally:
        os.close(descriptor)


def verify_spatial_staging(
    authority: SpatialMaterializationAuthority,
    staging_root: Path,
    native_report: bytes,
    *,
    deadline: float,
) -> VerifiedSpatialStaging:
    """Hash every staged artifact and build the sole acceptance manifest."""
    _require_deadline(deadline)
    if (
        not authority.native.execution_authorized
        or authority.execution_selection_sha256 is None
        or authority.execution_identity is None
        or hashlib.sha256(authority.execution_identity).hexdigest()
        != authority.execution_selection_sha256
        or canonical_json_bytes(json.loads(authority.execution_identity))
        != authority.execution_identity
    ):
        raise RuntimeError("Hovi spatial verification requires a frozen execution selection")
    staging_root = Path(os.path.abspath(os.fspath(staging_root)))
    report = _compact_report(native_report)
    _require_deadline(deadline)
    _require_keys(
        report,
        {
            "schemaVersion",
            "status",
            "profile",
            "reader",
            "source",
            "spatialContract",
            "recordAbi",
            "scans",
            "output",
            "limits",
            "result",
        },
        "root",
    )
    reader = _mapping(report.get("reader"), "reader")
    source = _mapping(report.get("source"), "source")
    spatial = _mapping(report.get("spatialContract"), "spatial contract")
    abi = _mapping(report.get("recordAbi"), "record ABI")
    output = _mapping(report.get("output"), "output")
    limits = _mapping(report.get("limits"), "limits")
    result = _mapping(report.get("result"), "result")
    scan_reports = report.get("scans")
    if not isinstance(scan_reports, list) or len(scan_reports) != len(authority.scans):
        raise ValueError("Hovi spatial native report does not contain all 16 scans")
    _require_keys(
        reader,
        {
            "adapterVersion",
            "upstreamE57Version",
            "upstreamCrateSha256",
            "vendorPatchSetSha256",
            "exactCountAndSectionEofValidated",
        },
        "reader",
    )
    _require_keys(
        source,
        {"inputMode", "bytes", "digestComputed", "fileGuid"},
        "source",
    )
    _require_keys(
        output,
        {
            "layoutRoot",
            "bytes",
            "artifactCount",
            "atomicPublication",
            "stagingOnly",
            "artifacts",
        },
        "output",
    )

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
    expected_limits = {
        "maxFileBytes": str(OBJECT_BYTES),
        "maxXmlBytes": "65536",
        "maxScans": 16,
        "maxPrototypeFields": 7,
        "maxDeclaredRecords": str(PUBLISHER_RECORD_COUNT),
        "maxJsonBytes": 1 << 20,
        "maxOutputBytes": str(TREE_CEILING_BYTES),
        "maxOpenFiles": authority.resources.native_max_open_files,
        "reservedFiles": authority.resources.native_reserved_files,
        "maxOpenOutputFiles": authority.resources.native_max_open_output_files,
    }
    if (
        report.get("schemaVersion") != authority.native.report_schema
        or report.get("status") != "spatial_materialization_complete"
        or report.get("profile") != "hy-spruce4"
        or reader.get("adapterVersion") != "0.1.0"
        or reader.get("upstreamE57Version") != "0.11.13"
        or reader.get("upstreamCrateSha256")
        != "fcfee41a50fbd70278c70cc477b8671ffe03951cba028cc2c057777a9ee6a3cc"
        or reader.get("vendorPatchSetSha256") != _VENDOR_PATCH_SHA256
        or reader.get("exactCountAndSectionEofValidated") is not True
        or source.get("inputMode") != "inherited_fd"
        or source.get("bytes") != str(OBJECT_BYTES)
        or source.get("digestComputed") is not False
        or source.get("fileGuid") != _FILE_GUID
        or spatial != expected_spatial
        or abi != expected_abi
        or output.get("layoutRoot") != "."
        or "directory" in output
        or output.get("atomicPublication") is not False
        or output.get("stagingOnly") is not True
        or limits != expected_limits
        or result
        != {
            "allScansDecodedOnce": True,
            "scanCount": 16,
            "publisherRecordCount": str(PUBLISHER_RECORD_COUNT),
            "recordsOutsideAoiDropped": True,
            "invalidAndDirectionRetained": True,
            "groundFiltered": False,
            "surfaceClaim": False,
            "targetTruth": False,
            "synthesisAuthorized": False,
        }
    ):
        raise ValueError("Hovi spatial native report crossed the frozen authority")

    retained_by_scan: list[int] = []
    reported_routes_by_scan: list[tuple[int, int, int]] = []
    scan_manifest = []
    for binding, value in zip(authority.scans, scan_reports, strict=True):
        item = _mapping(value, f"scan {binding.ordinal}")
        _require_keys(
            item,
            {
                "ordinal",
                "guid",
                "publisherDeclaredRecords",
                "decodedRecords",
                "validInAoi",
                "validOutsideAoi",
                "directionRecords",
                "invalidRecords",
                "finiteIntensity",
                "nonFiniteIntensity",
                "poseAppliedToStoredCoordinates",
                "poseAppliedForAoiAssignment",
            },
            f"scan {binding.ordinal}",
        )
        valid_in = _count(item.get("validInAoi"), "valid-in-AOI records")
        valid_out = _count(item.get("validOutsideAoi"), "valid-outside-AOI records")
        direction = _count(item.get("directionRecords"), "direction records")
        invalid = _count(item.get("invalidRecords"), "invalid records")
        finite = _count(item.get("finiteIntensity"), "finite intensity records")
        non_finite = _count(
            item.get("nonFiniteIntensity"),
            "non-finite intensity records",
        )
        if (
            item.get("ordinal") != binding.ordinal
            or item.get("guid") != binding.guid
            or item.get("publisherDeclaredRecords") != str(binding.publisher_records)
            or item.get("decodedRecords") != str(binding.publisher_records)
            or item.get("poseAppliedToStoredCoordinates") is not False
            or item.get("poseAppliedForAoiAssignment") is not True
            or valid_in + valid_out + direction + invalid != binding.publisher_records
            or finite + non_finite != binding.publisher_records
        ):
            raise ValueError(f"Hovi spatial scan {binding.ordinal} counts are inconsistent")
        retained_by_scan.append(valid_in + direction + invalid)
        reported_routes_by_scan.append((valid_in, direction, invalid))
        scan_manifest.append(
            {
                "ordinal": binding.ordinal,
                "guid": binding.guid,
                "publisherRecords": binding.publisher_records,
                "validInAoi": valid_in,
                "validOutsideAoi": valid_out,
                "directionRecords": direction,
                "invalidRecords": invalid,
                "finiteIntensity": finite,
                "nonFiniteIntensity": non_finite,
                "poseF64Bits": list(binding.pose_f64_bits),
            }
        )

    artifact_values = output.get("artifacts")
    if (
        not isinstance(artifact_values, list)
        or len(artifact_values) > _MAX_ARTIFACTS
    ):
        raise ValueError("Hovi spatial output artifacts must be an array")
    artifacts = tuple(
        _artifact_from_report(value, authority.scans) for value in artifact_values
    )
    if (
        output.get("artifactCount") != len(artifacts)
        or len({artifact.relative_path for artifact in artifacts}) != len(artifacts)
        or len(
            {
                (artifact.kind, artifact.scan, artifact.shard_x, artifact.shard_y)
                for artifact in artifacts
            }
        )
        != len(artifacts)
    ):
        raise ValueError("Hovi spatial artifact inventory has duplicate or missing keys")

    artifact_bytes = sum(artifact.bytes for artifact in artifacts)
    artifact_records = sum(artifact.records for artifact in artifacts)
    if (
        artifact_bytes != _count(output.get("bytes"), "output bytes")
        or artifact_bytes != artifact_records * RECORD_BYTES
        or artifact_bytes > TREE_CEILING_BYTES
    ):
        raise ValueError("Hovi spatial output totals exceed the 28-byte/tree boundary")
    reported_by_scan = [0] * len(authority.scans)
    for artifact in artifacts:
        reported_by_scan[artifact.scan] += artifact.records
    if reported_by_scan != retained_by_scan:
        raise ValueError("Hovi spatial artifacts do not partition retained scan records")

    expected_directories = {".", "shards", "nonpoints"} | {
        f"shards/scan-{scan.ordinal:02d}" for scan in authority.scans
    }
    expected_files = {artifact.relative_path for artifact in artifacts}
    staging_fd = _open_staging_root(staging_root)
    try:
        initial_directories, initial_files = _tree_inventory_at(
            staging_fd,
            allow_writable_files=True,
            deadline=deadline,
        )
        existing_manifest = "manifest.json" in initial_files
        allowed_files = expected_files | (
            {"manifest.json"} if existing_manifest else set()
        )
        if (
            set(initial_directories) != expected_directories
            or set(initial_files) != allowed_files
        ):
            raise ValueError("Hovi spatial staging tree contains unreported entries")

        verified_artifacts: list[VerifiedArtifact] = []
        observed_routes = [[0, 0, 0] for _ in authority.scans]
        for binding in authority.scans:
            _require_deadline(deadline)
            seen_sources = np.zeros(
                (binding.publisher_records + 7) // 8,
                dtype=np.uint8,
            )
            for artifact in artifacts:
                if artifact.scan != binding.ordinal:
                    continue
                _require_deadline(deadline)
                verified_artifact, observation = _verify_artifact(
                    staging_fd,
                    artifact,
                    binding,
                    seen_sources,
                    deadline,
                )
                verified_artifacts.append(verified_artifact)
                observed_routes[artifact.scan][0] += observation.valid
                observed_routes[artifact.scan][1] += observation.direction
                observed_routes[artifact.scan][2] += observation.invalid
            del seen_sources
        artifacts = tuple(verified_artifacts)
        if [tuple(values) for values in observed_routes] != reported_routes_by_scan:
            raise ValueError("Hovi spatial decoded artifact routing differs from report")
        _fsync_directory(staging_fd)
        post_directories, post_files = _tree_inventory_at(
            staging_fd,
            allow_writable_files=False,
            deadline=deadline,
        )
        if (
            set(post_directories) != expected_directories
            or set(post_files) != allowed_files
            or any(
                artifact.identity != post_files.get(artifact.relative_path)
                for artifact in artifacts
            )
        ):
            raise ValueError("Hovi spatial immutable artifact inventory changed")
    finally:
        os.close(staging_fd)
    if shutil.disk_usage(staging_root).free < FREE_RESERVE_BYTES:
        raise OSError("Hovi spatial verification would violate the 96 GiB free reserve")

    artifact_manifest = [
        {
            "kind": artifact.kind,
            "scan": artifact.scan,
            "shardX": artifact.shard_x,
            "shardY": artifact.shard_y,
            "path": artifact.relative_path,
            "records": artifact.records,
            "bytes": artifact.bytes,
            "sha256": artifact.sha256,
        }
        for artifact in sorted(
            artifacts,
            key=lambda artifact: (
                artifact.scan,
                artifact.kind,
                artifact.shard_x if artifact.shard_x is not None else 0,
                artifact.shard_y if artifact.shard_y is not None else 0,
            ),
        )
    ]
    manifest = canonical_json_bytes(
        {
            "schemaVersion": "hovi-hy-spruce4-spatial-materialization-manifest/1.0.0",
            "status": "verified_staging_ready_for_atomic_publication",
            "authority": {
                "configPath": authority.config_path.relative_to(
                    ASSET_GEN_ROOT
                ).as_posix(),
                "configSha256": authority.config_sha256,
                "mandateProvenance": json.loads(authority.mandate_provenance),
                "executionSelection": {
                    "bytes": len(authority.execution_identity),
                    "sha256": authority.execution_selection_sha256,
                    "document": json.loads(authority.execution_identity),
                },
            },
            "datasetAndLicense": json.loads(authority.dataset_and_license),
            "source": {
                "e57Bytes": OBJECT_BYTES,
                "e57Sha256": OBJECT_SHA256,
                "selectedMetadataInventoryPath": authority.inventory_path.as_posix(),
                "selectedMetadataInventorySha256": authority.inventory_sha256,
            },
            "spatialContract": expected_spatial,
            "recordAbi": expected_abi,
            "scans": scan_manifest,
            "artifacts": artifact_manifest,
            "totals": {
                "publisherRecordsDecoded": PUBLISHER_RECORD_COUNT,
                "materializedRecords": artifact_records,
                "materializedBytes": artifact_bytes,
                "artifactCount": len(artifacts),
            },
            "nativeReport": {
                "bytes": len(native_report),
                "sha256": hashlib.sha256(native_report).hexdigest(),
                "document": report,
            },
            "evidenceBoundary": {
                "scientificRole": "raw_candidate_spatial_materialization",
                "qualificationStatus": "unqualified",
                "groundFiltered": False,
                "surfaceClaim": False,
                "analogueQualificationAuthorized": False,
                "targetTruth": False,
                "synthesisAuthorized": False,
            },
        }
    )
    manifest_sha256 = hashlib.sha256(manifest).hexdigest()
    _require_deadline(deadline)
    if artifact_bytes + len(manifest) > TREE_CEILING_BYTES:
        raise ValueError("Hovi spatial manifest would exceed the 120 GiB tree ceiling")
    staging_fd = _open_staging_root(staging_root)
    try:
        final_directories, final_files = _tree_inventory_at(
            staging_fd,
            allow_writable_files=False,
            deadline=deadline,
        )
        if (
            set(final_directories) != expected_directories
            or set(final_files) != allowed_files
            or any(
                artifact.identity != final_files.get(artifact.relative_path)
                for artifact in artifacts
            )
        ):
            raise ValueError("Hovi spatial verified tree changed before acceptance")
        if existing_manifest:
            manifest_descriptor = _open_relative_regular_at(staging_fd, "manifest.json")
            try:
                before = os.fstat(manifest_descriptor)
                payload = _read_descriptor(
                    manifest_descriptor,
                    len(manifest),
                    deadline,
                )
                after = os.fstat(manifest_descriptor)
                if (
                    _inode_identity(before) != _inode_identity(after)
                    or _inode_identity(after) != final_files["manifest.json"]
                    or after.st_uid != os.getuid()
                    or stat.S_IMODE(after.st_mode) != 0o400
                    or after.st_nlink != 1
                    or payload != manifest
                ):
                    raise ValueError(
                        "existing Hovi spatial manifest differs from verification"
                    )
            finally:
                os.close(manifest_descriptor)
        root_identity = _inode_identity(os.fstat(staging_fd))
        _require_deadline(deadline)
        final_directories["."] = root_identity
    finally:
        os.close(staging_fd)
    return VerifiedSpatialStaging(
        root=staging_root,
        root_identity=root_identity,
        directory_identities=tuple(sorted(final_directories.items())),
        file_identities=tuple(sorted(final_files.items())),
        native_report=native_report,
        native_report_sha256=hashlib.sha256(native_report).hexdigest(),
        artifacts=artifacts,
        record_count=artifact_records,
        artifact_bytes=artifact_bytes,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
    )


def publish_verified_spatial_staging(
    authority: SpatialMaterializationAuthority,
    verified: VerifiedSpatialStaging,
    output_root: Path,
    *,
    deadline: float,
) -> Path:
    """Commit one verified staging tree; pending native authority always refuses."""
    _require_deadline(deadline)
    if not authority.native.execution_authorized:
        raise RuntimeError("Hovi spatial publication awaits a frozen native binding")
    output_root = Path(os.path.abspath(os.fspath(output_root)))
    if verified.root.parent != output_root or verified.root.name in {"", ".", ".."}:
        raise ValueError("Hovi spatial staging is not a direct output-root child")
    if verified.artifact_bytes + len(verified.manifest) > TREE_CEILING_BYTES:
        raise ValueError("Hovi spatial publication exceeds its tree ceiling")
    if shutil.disk_usage(verified.root).free < FREE_RESERVE_BYTES:
        raise OSError("Hovi spatial publication would violate the 96 GiB reserve")

    root_fd = _open_or_create_private_root(output_root)
    staging_fd = materializations = sha_directory = prefix_directory = -1
    try:
        staging_fd = os.open(
            verified.root.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=root_fd,
        )
        namespace = os.stat(
            verified.root.name,
            dir_fd=root_fd,
            follow_symlinks=False,
        )
        if (
            _inode_identity(namespace) != verified.root_identity
            or _inode_identity(os.fstat(staging_fd)) != verified.root_identity
        ):
            raise ValueError("Hovi spatial staging root changed before publication")

        expected_directories = dict(verified.directory_identities)
        expected_files = dict(verified.file_identities)
        current_directories, current_files = _tree_inventory_at(
            staging_fd,
            allow_writable_files=False,
            deadline=deadline,
        )
        if (
            current_directories != expected_directories
            or current_files != expected_files
        ):
            raise ValueError("Hovi spatial tree changed after verification")

        existing_manifest = expected_files.get("manifest.json")
        if existing_manifest is None:
            manifest_fd = os.open(
                "manifest.json",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=staging_fd,
            )
            try:
                view = memoryview(verified.manifest)
                while view:
                    _require_deadline(deadline)
                    written = os.write(manifest_fd, view)
                    if written <= 0:
                        raise OSError("short write while committing Hovi spatial manifest")
                    view = view[written:]
                os.fchmod(manifest_fd, 0o400)
                os.fsync(manifest_fd)
                manifest_identity = _inode_identity(os.fstat(manifest_fd))
            finally:
                os.close(manifest_fd)
        else:
            manifest_fd = _open_relative_regular_at(staging_fd, "manifest.json")
            try:
                before = os.fstat(manifest_fd)
                payload = _read_descriptor(
                    manifest_fd,
                    len(verified.manifest),
                    deadline,
                )
                after = os.fstat(manifest_fd)
                if (
                    _inode_identity(before) != existing_manifest
                    or _inode_identity(after) != existing_manifest
                    or payload != verified.manifest
                ):
                    raise ValueError("existing Hovi spatial manifest is not resumable")
                manifest_identity = existing_manifest
            finally:
                os.close(manifest_fd)

        for artifact in verified.artifacts:
            _require_deadline(deadline)
            if artifact.identity is None:
                raise ValueError("Hovi spatial artifact lacks a verified identity")
            descriptor = _open_relative_regular_at(staging_fd, artifact.relative_path)
            try:
                result = os.fstat(descriptor)
                if (
                    _inode_identity(result) != artifact.identity
                    or result.st_uid != os.getuid()
                    or stat.S_IMODE(result.st_mode) != 0o400
                    or result.st_nlink != 1
                ):
                    raise PermissionError("Hovi spatial artifact changed before publication")
            finally:
                os.close(descriptor)
        _fsync_directory(staging_fd)

        materializations = _open_or_create_private_directory_at(
            root_fd,
            "materializations",
            "Hovi spatial materializations",
        )
        sha_directory = _open_or_create_private_directory_at(
            materializations,
            "sha256",
            "Hovi spatial materialization hashes",
        )
        prefix_directory = _open_or_create_private_directory_at(
            sha_directory,
            verified.manifest_sha256[:2],
            "Hovi spatial materialization hash prefix",
        )
        expected_files["manifest.json"] = manifest_identity
        expected_directories["."] = _inode_identity(os.fstat(staging_fd))
        if shutil.disk_usage(output_root).free < FREE_RESERVE_BYTES:
            raise OSError("Hovi spatial publication would violate the 96 GiB reserve")
        final_directories, final_files = _tree_inventory_at(
            staging_fd,
            allow_writable_files=False,
            deadline=deadline,
        )
        if (
            final_directories != expected_directories
            or final_files != expected_files
            or _inode_identity(os.fstat(staging_fd)) != expected_directories["."]
        ):
            raise ValueError("Hovi spatial tree changed at the publication boundary")
        _require_deadline(deadline)
        _atomic_rename_noreplace_at(
            root_fd,
            verified.root.name,
            prefix_directory,
            verified.manifest_sha256,
        )
        _fsync_directory(prefix_directory)
        _fsync_directory(root_fd)
        return (
            output_root
            / "materializations"
            / "sha256"
            / verified.manifest_sha256[:2]
            / verified.manifest_sha256
        )
    finally:
        for descriptor in (
            prefix_directory,
            sha_directory,
            materializations,
            root_fd,
            staging_fd,
        ):
            if descriptor >= 0:
                os.close(descriptor)
