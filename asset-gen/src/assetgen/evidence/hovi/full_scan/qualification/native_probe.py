"""Inherited-descriptor execution seam for the bounded Hovi reader probe."""
from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import resource
import selectors
import signal
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .....config import ASSET_GEN_ROOT
from ..authorization import (
    canonical_json_bytes,
    load_metadata_inspection_authorization,
    open_regular_nofollow,
)
from ..inventory import _open_held_extraction
from .artifacts import (
    DEFAULT_PROBE_OUTPUT_ROOT,
    create_staged_record_output,
    publish_record_output,
    publish_report,
    verify_staged_record_output,
)
from .authority import OBJECT_BYTES, OBJECT_SHA256, PointProbeAuthority


_RECORD = struct.Struct("<QqqqqqqddddII")
_XYZ_SCALE = float("1.00000000000000008e-05")
_VENDOR_PATCH_SHA256 = (
    "266cf419581580ef4198d109cbcb83f84a45ab6235d7aed5fa400631ad092dd8"
)
_FILE_GUID = "{C70845D7-08E0-4185-974C-EF84E9EDEA55}"
_PROTOTYPE = [
    {"index": 0, "name": "cartesianX", "kind": "ScaledInteger"},
    {"index": 1, "name": "cartesianY", "kind": "ScaledInteger"},
    {"index": 2, "name": "cartesianZ", "kind": "ScaledInteger"},
    {"index": 3, "name": "intensity", "kind": "Single"},
    {"index": 4, "name": "rowIndex", "kind": "Integer"},
    {"index": 5, "name": "columnIndex", "kind": "Integer"},
    {"index": 6, "name": "cartesianInvalidState", "kind": "Integer"},
]
_PROC_PIDTASKINFO = 4
_MEMORY_SAMPLE_SECONDS = 0.01


class _ProcTaskInfo(ctypes.Structure):
    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("total_user", ctypes.c_uint64),
        ("total_system", ctypes.c_uint64),
        ("threads_user", ctypes.c_uint64),
        ("threads_system", ctypes.c_uint64),
        ("policy", ctypes.c_int32),
        ("faults", ctypes.c_int32),
        ("pageins", ctypes.c_int32),
        ("cow_faults", ctypes.c_int32),
        ("messages_sent", ctypes.c_int32),
        ("messages_received", ctypes.c_int32),
        ("syscalls_mach", ctypes.c_int32),
        ("syscalls_unix", ctypes.c_int32),
        ("context_switches", ctypes.c_int32),
        ("thread_count", ctypes.c_int32),
        ("running_thread_count", ctypes.c_int32),
        ("priority", ctypes.c_int32),
    ]


class _DarwinResidentMemoryGuard:
    """Fail-closed RSS supervision for a single audited native reader process."""

    def __init__(self, limit_bytes: int) -> None:
        if sys.platform != "darwin":
            raise RuntimeError("Hovi native point-probe RSS guard requires macOS libproc")
        if ctypes.sizeof(_ProcTaskInfo) != 96:
            raise RuntimeError("macOS proc_taskinfo ABI size changed")
        self._limit_bytes = limit_bytes
        self._libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        self._libproc.proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self._libproc.proc_pidinfo.restype = ctypes.c_int

    def enforce(self, process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        task = _ProcTaskInfo()
        ctypes.set_errno(0)
        returned = self._libproc.proc_pidinfo(
            process.pid,
            _PROC_PIDTASKINFO,
            0,
            ctypes.byref(task),
            ctypes.sizeof(task),
        )
        if returned != ctypes.sizeof(task):
            observed_errno = ctypes.get_errno()
            if observed_errno == errno.ESRCH and process.poll() is not None:
                return
            raise RuntimeError(
                "macOS could not supervise native Hovi point-probe RSS "
                f"(proc_pidinfo returned {returned}, errno {observed_errno})"
            )
        if task.resident_size > self._limit_bytes:
            raise MemoryError(
                "native Hovi point probe exceeded its resident-memory limit: "
                f"{task.resident_size} > {self._limit_bytes} bytes"
            )


@dataclass(frozen=True)
class NativeProbeResult:
    record_path: Path
    record_sha256: str
    report_path: Path
    report_sha256: str
    elapsed_seconds: float


def _verify_execution_implementation(authority: PointProbeAuthority) -> None:
    if authority.execution_selection_sha256 is None or authority.execution_identity is None:
        raise RuntimeError("native Hovi point probe has no frozen execution selection")
    for path, expected_sha256 in authority.implementation_files:
        descriptor = open_regular_nofollow(path, "point-probe implementation file")
        try:
            digest = hashlib.sha256()
            while block := os.read(descriptor, 1 << 20):
                digest.update(block)
        finally:
            os.close(descriptor)
        if digest.hexdigest() != expected_sha256:
            raise ValueError(f"point-probe implementation changed: {path.name}")


def _descriptor_sha256(descriptor: int) -> str:
    before = os.fstat(descriptor)
    digest = hashlib.sha256()
    offset = 0
    size = os.fstat(descriptor).st_size
    while offset < size:
        block = os.pread(descriptor, min(1 << 20, size - offset), offset)
        if not block:
            break
        digest.update(block)
        offset += len(block)
    after = os.fstat(descriptor)
    if (
        offset != size
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    ):
        raise ValueError("native point-probe executable changed while being hashed")
    return digest.hexdigest()


def _limit_child(authority: PointProbeAuthority) -> None:
    def apply(kind: int, requested: int) -> None:
        _, inherited_hard = resource.getrlimit(kind)
        effective = (
            requested
            if inherited_hard == resource.RLIM_INFINITY
            else min(requested, inherited_hard)
        )
        resource.setrlimit(kind, (effective, effective))

    limits = authority.resources
    apply(resource.RLIMIT_CPU, limits.cpu_seconds)
    # macOS rejects a useful RLIMIT_AS because ordinary processes inherit hundreds
    # of GiB of sparse/shared mappings. The parent enforces the RSS ceiling through
    # PROC_PIDTASKINFO instead; RLIMIT_RSS itself is advisory on macOS.
    apply(resource.RLIMIT_FSIZE, limits.record_output_bytes)
    apply(resource.RLIMIT_NOFILE, limits.open_files)
    apply(resource.RLIMIT_CORE, 0)


def _terminate_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def _capture_bounded(
    process: subprocess.Popen[bytes],
    authority: PointProbeAuthority,
    started: float,
) -> tuple[bytes, bytes]:
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("native point-probe capture pipes were not created")
    streams = {
        process.stdout.fileno(): ("stdout", authority.resources.stdout_bytes),
        process.stderr.fileno(): ("stderr", authority.resources.stderr_bytes),
    }
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    selector = selectors.DefaultSelector()
    try:
        memory_guard = _DarwinResidentMemoryGuard(authority.resources.rss_bytes)
        for descriptor in streams:
            os.set_blocking(descriptor, False)
            selector.register(descriptor, selectors.EVENT_READ)
        while selector.get_map():
            memory_guard.enforce(process)
            remaining = authority.resources.wall_seconds - (time.monotonic() - started)
            if remaining <= 0:
                _terminate_group(process)
                raise TimeoutError("native Hovi point probe exceeded its hard wall limit")
            for key, _ in selector.select(
                min(remaining, _MEMORY_SAMPLE_SECONDS)
            ):
                descriptor = int(key.fd)
                label, cap = streams[descriptor]
                try:
                    block = os.read(
                        descriptor,
                        min(1 << 16, cap + 1 - len(captured[label])),
                    )
                except BlockingIOError:
                    continue
                if not block:
                    selector.unregister(descriptor)
                    continue
                captured[label].extend(block)
                if len(captured[label]) > cap:
                    _terminate_group(process)
                    raise RuntimeError(f"native Hovi point probe exceeded {label} limit")
        remaining = authority.resources.wall_seconds - (time.monotonic() - started)
        if remaining <= 0:
            _terminate_group(process)
            raise TimeoutError("native Hovi point probe exceeded its hard wall limit")
        return_code = process.wait(timeout=remaining)
    except Exception:
        if process.poll() is None:
            _terminate_group(process)
        raise
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()
    if return_code != 0:
        raise RuntimeError(
            f"native Hovi point probe failed with exit code {return_code}: "
            f"{bytes(captured['stderr']).decode('utf-8', errors='replace')}"
        )
    return bytes(captured["stdout"]), bytes(captured["stderr"])


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"native point-probe report repeats JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"native point-probe report contains non-JSON number {value}")


def _require_compact_json(payload: bytes) -> None:
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
            raise ValueError("native point-probe report has non-canonical whitespace")
    if in_string or escaped:
        raise ValueError("native point-probe report ends inside a JSON string")


def _require_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"native point-probe {label} keys changed")


def _add_bounds(bounds: list[int | float | None], value: int | float) -> None:
    bounds[0] = value if bounds[0] is None else min(bounds[0], value)
    bounds[1] = value if bounds[1] is None else max(bounds[1], value)


def _integer_bounds(bounds: list[int | float | None]) -> dict[str, str] | None:
    if bounds[0] is None or bounds[1] is None:
        return None
    return {"minimum": str(int(bounds[0])), "maximum": str(int(bounds[1]))}


def _float_bounds(bounds: list[int | float | None]) -> dict[str, float] | None:
    if bounds[0] is None or bounds[1] is None:
        return None
    return {"minimum": float(bounds[0]), "maximum": float(bounds[1])}


def _verify_record_abi(
    descriptor: int,
    authority: PointProbeAuthority,
) -> tuple[str, dict[str, Any]]:
    """Independently decode the fixed-width proof artifact before publication."""
    expected_bytes = authority.resources.record_output_bytes
    before = os.fstat(descriptor)
    if before.st_size != expected_bytes or expected_bytes != _RECORD.size * 65_536:
        raise ValueError("point-probe record output has the wrong ABI byte count")

    digest = hashlib.sha256()
    counters = {
        "validCount": 0,
        "directionCount": 0,
        "invalidCount": 0,
        "finiteIntensityCount": 0,
        "nonFiniteIntensityCount": 0,
    }
    bounds: dict[str, list[int | float | None]] = {
        name: [None, None]
        for name in (
            "rowBounds",
            "columnBounds",
            "rawXBounds",
            "rawYBounds",
            "rawZBounds",
            "localXBounds",
            "localYBounds",
            "localZBounds",
            "validLocalXBounds",
            "validLocalYBounds",
            "validLocalZBounds",
            "validRangeBoundsM",
            "finiteIntensityBounds",
        )
    }
    offset = 0
    record_ordinal = authority.record_start
    block_bytes = _RECORD.size * 8_192
    while offset < expected_bytes:
        block = os.pread(descriptor, min(block_bytes, expected_bytes - offset), offset)
        if not block or len(block) % _RECORD.size:
            raise ValueError("point-probe record output is truncated or misaligned")
        digest.update(block)
        for values in _RECORD.iter_unpack(block):
            (
                ordinal,
                row,
                column,
                invalid,
                raw_x,
                raw_y,
                raw_z,
                local_x,
                local_y,
                local_z,
                derived_range,
                intensity_bits,
                flags,
            ) = values
            if ordinal != record_ordinal:
                raise ValueError("point-probe source ordinals are not exact and contiguous")
            record_ordinal += 1
            if not 0 <= row <= 11_003 or not 0 <= column <= 27_480:
                raise ValueError("point-probe row or column is outside its prototype domain")
            if invalid not in (0, 1, 2):
                raise ValueError("point-probe invalid state is outside its prototype domain")
            if any(not -2_147_483_647 <= raw <= 2_147_483_647 for raw in (raw_x, raw_y, raw_z)):
                raise ValueError("point-probe raw coordinate is outside its prototype domain")
            expected_local = (
                float(raw_x) * _XYZ_SCALE,
                float(raw_y) * _XYZ_SCALE,
                float(raw_z) * _XYZ_SCALE,
            )
            local = (local_x, local_y, local_z)
            if any(
                not math.isfinite(actual)
                or struct.pack("<d", actual) != struct.pack("<d", expected)
                for actual, expected in zip(local, expected_local, strict=True)
            ):
                raise ValueError("point-probe local XYZ does not reconstruct from raw XYZ")
            intensity = struct.unpack("<f", struct.pack("<I", intensity_bits))[0]
            intensity_finite = math.isfinite(intensity)
            if flags & ~0b11:
                raise ValueError("point-probe record uses an undefined flag bit")
            if bool(flags & 0b01) != (invalid == 0):
                raise ValueError("point-probe range-present flag disagrees with invalid state")
            if bool(flags & 0b10) != intensity_finite:
                raise ValueError("point-probe intensity-finite flag disagrees with its raw bits")

            if invalid == 0:
                expected_range = math.sqrt(
                    local_x * local_x + local_y * local_y + local_z * local_z
                )
                tolerance = max(1.0e-12, 2.0 * math.ulp(expected_range))
                if not math.isfinite(derived_range) or not math.isclose(
                    derived_range,
                    expected_range,
                    rel_tol=0.0,
                    abs_tol=tolerance,
                ):
                    raise ValueError("point-probe derived range disagrees with local XYZ")
                counters["validCount"] += 1
                for name, value in zip(
                    (
                        "validLocalXBounds",
                        "validLocalYBounds",
                        "validLocalZBounds",
                    ),
                    local,
                    strict=True,
                ):
                    _add_bounds(bounds[name], value)
                _add_bounds(bounds["validRangeBoundsM"], derived_range)
            else:
                if derived_range != 0.0 or math.copysign(1.0, derived_range) < 0.0:
                    raise ValueError("point-probe invalid record carries a derived range")
                counters["directionCount" if invalid == 1 else "invalidCount"] += 1

            if intensity_finite:
                counters["finiteIntensityCount"] += 1
                _add_bounds(bounds["finiteIntensityBounds"], intensity)
            else:
                counters["nonFiniteIntensityCount"] += 1
            for name, value in (
                ("rowBounds", row),
                ("columnBounds", column),
                ("rawXBounds", raw_x),
                ("rawYBounds", raw_y),
                ("rawZBounds", raw_z),
                ("localXBounds", local_x),
                ("localYBounds", local_y),
                ("localZBounds", local_z),
            ):
                _add_bounds(bounds[name], value)
        offset += len(block)

    after = os.fstat(descriptor)
    if (
        record_ordinal != authority.record_end
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    ):
        raise ValueError("point-probe record output changed during ABI verification")
    result: dict[str, Any] = {
        "decodedRecords": str(authority.hard_max_decoded_records),
        "canonicalRecordBytesHashed": str(expected_bytes),
        "canonicalRecordStreamSha256": digest.hexdigest(),
        **{name: str(value) for name, value in counters.items()},
    }
    for name in ("rowBounds", "columnBounds", "rawXBounds", "rawYBounds", "rawZBounds"):
        result[name] = _integer_bounds(bounds[name])
    for name in (
        "localXBounds",
        "localYBounds",
        "localZBounds",
        "validLocalXBounds",
        "validLocalYBounds",
        "validLocalZBounds",
        "validRangeBoundsM",
        "finiteIntensityBounds",
    ):
        result[name] = _float_bounds(bounds[name])
    return digest.hexdigest(), result


def _validate_report(
    payload: bytes,
    authority: PointProbeAuthority,
    record_sha256: str,
    independently_observed: dict[str, Any],
) -> dict[str, Any]:
    if (
        not payload.startswith(b"{")
        or not payload.endswith(b"}\n")
        or payload.count(b"\n") != 1
        or b"\r" in payload
    ):
        raise ValueError("native point-probe stdout is not one canonical JSON report")
    _require_compact_json(payload)
    report = json.loads(
        payload.decode("utf-8", errors="strict"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(report, dict):
        raise ValueError("native point-probe report must be an object")
    report_authority = report.get("authority")
    reader = report.get("reader")
    source = report.get("source")
    scan = report.get("scan")
    layout = report.get("canonicalRecordLayout")
    result = report.get("result")
    output = report.get("output")
    limits = report.get("limits")
    if not all(
        isinstance(value, dict)
        for value in (report_authority, reader, source, scan, layout, output, result, limits)
    ):
        raise ValueError("native point-probe report lost a required object")
    assert isinstance(report_authority, dict)
    assert isinstance(reader, dict)
    assert isinstance(source, dict)
    assert isinstance(scan, dict)
    assert isinstance(layout, dict)
    assert isinstance(output, dict)
    assert isinstance(result, dict)
    assert isinstance(limits, dict)
    _require_keys(
        report,
        {
            "schemaVersion",
            "status",
            "profile",
            "authority",
            "reader",
            "source",
            "scan",
            "canonicalRecordLayout",
            "output",
            "result",
            "eofReached",
            "publisherCountValidated",
            "surfaceClaim",
            "recordsRetained",
            "recordOutputPublished",
            "invalidRecordsRetained",
            "poseApplied",
            "boundedProbeImplemented",
            "fullExtractionImplemented",
            "colorAvailability",
            "limits",
        },
        "root",
    )
    expected_layout = {
        "byteOrder": "little_endian",
        "bytesPerRecord": 96,
        "fields": [
            "sourceOrdinal:u64",
            "row:i64",
            "column:i64",
            "cartesianInvalidState:i64",
            "rawX:i64",
            "rawY:i64",
            "rawZ:i64",
            "localX:f64bits",
            "localY:f64bits",
            "localZ:f64bits",
            "derivedRangeOrZero:f64bits",
            "rawIntensitySingleBits:u32",
            "flags:u32",
        ],
        "flags": {"bit0": "range_present", "bit1": "intensity_finite"},
    }
    expected_pose = {
        "rotation": {
            "w": float("9.99982395361063370e-01"),
            "x": 0.0,
            "y": 0.0,
            "z": float("-5.93371451536018821e-03"),
            "wBits": "3fefffdb1494d92a",
            "xBits": "0000000000000000",
            "yBits": "0000000000000000",
            "zBits": "bf784df35c98ab0c",
        },
        "translation": {
            "x": 0.0,
            "y": 0.0,
            "z": float("1.54000000000000048e+00"),
            "xBits": "0000000000000000",
            "yBits": "0000000000000000",
            "zBits": "3ff8a3d70a3d70a6",
        },
    }
    if (
        report.get("schemaVersion") != authority.native.report_schema
        or report.get("status") != "bounded_probe_complete"
        or report.get("profile") != "hy-spruce4"
        or report_authority
        != {
            "scanOrdinal": authority.scan_ordinal,
            "scanGuid": authority.scan_guid,
            "startOrdinalInclusive": str(authority.record_start),
            "stopOrdinalExclusive": str(authority.record_end),
        }
        or reader.get("upstreamE57Version") != "0.11.13"
        or reader.get("upstreamCrateSha256")
        != "fcfee41a50fbd70278c70cc477b8671ffe03951cba028cc2c057777a9ee6a3cc"
        or reader
        != {
            "adapterVersion": "0.1.0",
            "upstreamE57Version": "0.11.13",
            "upstreamCrateSha256": (
                "fcfee41a50fbd70278c70cc477b8671ffe03951cba028cc2c057777a9ee6a3cc"
            ),
            "vendorPatchSetSha256": _VENDOR_PATCH_SHA256,
            "reusableRawBuffer": True,
            "crcImplementation": "vendored_e57_pure_rust",
            "crc32cFeature": False,
        }
        or source.get("inputMode") != "inherited_fd"
        or source.get("bytes") != str(OBJECT_BYTES)
        or source.get("digestComputed") is not False
        or source.get("fileGuid") != _FILE_GUID
        or set(source) != {"inputMode", "bytes", "digestComputed", "fileGuid"}
        or scan.get("ordinal") != authority.scan_ordinal
        or scan.get("guid") != authority.scan_guid
        or scan.get("publisherDeclaredRecords")
        != str(authority.publisher_record_count)
        or scan.get("poseApplied") is not False
        or scan.get("poseProvenanceOnly") is not True
        or scan.get("pose") != expected_pose
        or scan.get("observedPrototype") != _PROTOTYPE
        or set(scan)
        != {
            "ordinal",
            "guid",
            "publisherDeclaredRecords",
            "poseApplied",
            "poseProvenanceOnly",
            "pose",
            "observedPrototype",
        }
        or layout != expected_layout
        or output
        != {
            "mode": "inherited_fd",
            "records": str(authority.hard_max_decoded_records),
            "bytes": str(authority.resources.record_output_bytes),
            "sha256": record_sha256,
        }
        or result != independently_observed
        or report.get("eofReached") is not False
        or report.get("publisherCountValidated") is not False
        or report.get("surfaceClaim") is not False
        or report.get("recordsRetained") is not True
        or report.get("recordOutputPublished") is not False
        or report.get("invalidRecordsRetained") is not True
        or report.get("poseApplied") is not False
        or report.get("boundedProbeImplemented") is not True
        or report.get("fullExtractionImplemented") is not False
        or report.get("colorAvailability") != "absent_in_prototype"
        or limits
        != {
            "maxFileBytes": str(OBJECT_BYTES),
            "maxXmlBytes": "65536",
            "maxScans": 16,
            "maxPrototypeFields": 7,
            "maxDeclaredRecords": "4248797321",
            "maxJsonBytes": 1 << 20,
        }
    ):
        raise ValueError("native point-probe report crossed the frozen authority")
    return report


def run_native_probe(
    authority: PointProbeAuthority,
) -> NativeProbeResult:
    """Execute only the frozen build into the frozen content-addressed root."""
    result_root = DEFAULT_PROBE_OUTPUT_ROOT
    native = authority.native
    if (
        not native.execution_authorized
        or native.executable_path is None
        or native.executable_bytes is None
        or native.executable_sha256 is None
    ):
        raise RuntimeError(
            "native Hovi point-probe execution awaits a frozen project build identity"
        )
    _verify_execution_implementation(authority)

    executable_descriptor = open_regular_nofollow(
        native.executable_path,
        "native point-probe executable",
    )
    held = None
    staged = None
    process: subprocess.Popen[bytes] | None = None
    published_records = False
    try:
        metadata_authorization = load_metadata_inspection_authorization()
        binding, held = _open_held_extraction(
            metadata_authorization,
            authority.output_root,
        )
        staged = create_staged_record_output(authority.config_sha256, result_root)
        if not all(
            3 <= descriptor <= 1024
            for descriptor in (held.object_descriptor, staged.descriptor)
        ):
            raise RuntimeError("native Hovi point-probe descriptors exceed its hard range")
        executable_stat = os.fstat(executable_descriptor)
        if (
            executable_stat.st_uid != os.getuid()
            or executable_stat.st_nlink != 1
            or executable_stat.st_mode & 0o222
            or not executable_stat.st_mode & 0o100
        ):
            raise PermissionError(
                "native Hovi point-probe executable is not private and immutable"
            )
        source_identity = os.fstat(held.object_descriptor)
        if (
            executable_stat.st_size != native.executable_bytes
            or _descriptor_sha256(executable_descriptor) != native.executable_sha256
            or binding.object_sha256 != OBJECT_SHA256
            or binding.object_bytes != OBJECT_BYTES
        ):
            raise ValueError("native Hovi point-probe execution binding drifted")
        executable_fd_path = f"/dev/fd/{executable_descriptor}"
        argv = [
            executable_fd_path,
            "probe",
            "--input-fd",
            str(held.object_descriptor),
            "--output-fd",
            str(staged.descriptor),
            "--profile",
            "hy-spruce4",
            "--max-file-bytes",
            str(OBJECT_BYTES),
            "--max-xml-bytes",
            "65536",
            "--max-scans",
            "16",
            "--max-prototype-fields",
            "7",
            "--max-declared-records",
            "4248797321",
            "--max-json-bytes",
            str(1 << 20),
        ]
        started = time.monotonic()
        process = subprocess.Popen(
            argv,
            executable=executable_fd_path,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            pass_fds=(
                executable_descriptor,
                held.object_descriptor,
                staged.descriptor,
            ),
            cwd=staged.path,
            env={"LC_ALL": "C", "LANG": "C", "TZ": "UTC"},
            start_new_session=True,
            preexec_fn=lambda: _limit_child(authority),
        )
        stdout, stderr = _capture_bounded(process, authority, started)
        if stderr:
            raise RuntimeError("native Hovi point probe emitted unexpected stderr")
        source_after = os.fstat(held.object_descriptor)
        executable_after = os.fstat(executable_descriptor)
        if (
            source_identity.st_dev,
            source_identity.st_ino,
            source_identity.st_size,
            source_identity.st_mtime_ns,
            source_identity.st_ctime_ns,
        ) != (
            source_after.st_dev,
            source_after.st_ino,
            source_after.st_size,
            source_after.st_mtime_ns,
            source_after.st_ctime_ns,
        ):
            raise ValueError("held Hovi E57 object changed during the native probe")
        if (
            executable_stat.st_dev,
            executable_stat.st_ino,
            executable_stat.st_size,
            executable_stat.st_mtime_ns,
            executable_stat.st_ctime_ns,
        ) != (
            executable_after.st_dev,
            executable_after.st_ino,
            executable_after.st_size,
            executable_after.st_mtime_ns,
            executable_after.st_ctime_ns,
        ):
            raise ValueError("native point-probe executable changed during execution")
        os.fsync(staged.descriptor)
        record_sha256, independently_observed = _verify_record_abi(
            staged.descriptor,
            authority,
        )
        if record_sha256 != verify_staged_record_output(
            staged,
            expected_bytes=authority.resources.record_output_bytes,
        ):
            raise ValueError("point-probe ABI and artifact hashes disagree")
        native_report = _validate_report(
            stdout,
            authority,
            record_sha256,
            independently_observed,
        )
        record_path, published_record_sha256 = publish_record_output(
            staged,
            expected_bytes=authority.resources.record_output_bytes,
            expected_sha256=record_sha256,
        )
        published_records = True
        result_root_absolute = Path(os.path.abspath(os.fspath(result_root)))
        native_report_path, native_report_sha256 = publish_report(
            staged.root,
            result_root,
            stdout,
        )
        publication_report = canonical_json_bytes(
            {
                "schemaVersion": "hovi-hy-spruce4-full-point-probe-publication/1.0.0",
                "status": "bounded_reader_validation_probe_complete",
                "authority": {
                    "configPath": authority.config_path.relative_to(
                        ASSET_GEN_ROOT
                    ).as_posix(),
                    "configSha256": authority.config_sha256,
                    "mandateProvenance": json.loads(authority.mandate_provenance),
                    "executionSelectionSha256": authority.execution_selection_sha256,
                },
                "datasetAndLicense": json.loads(authority.dataset_and_license),
                "source": {
                    "e57Bytes": OBJECT_BYTES,
                    "e57Sha256": OBJECT_SHA256,
                    "selectedMetadataInventoryPath": authority.inventory_path.relative_to(
                        ASSET_GEN_ROOT
                    ).as_posix(),
                    "selectedMetadataInventorySha256": authority.inventory_sha256,
                },
                "selection": {
                    "scanOrdinal": authority.scan_ordinal,
                    "scanGuid": authority.scan_guid,
                    "startOrdinalInclusive": authority.record_start,
                    "endOrdinalExclusive": authority.record_end,
                    "decodedRecords": authority.hard_max_decoded_records,
                    "invalidRecordsRetained": True,
                    "poseRole": "provenance_only",
                },
                "records": {
                    "path": record_path.relative_to(result_root_absolute).as_posix(),
                    "bytes": authority.resources.record_output_bytes,
                    "sha256": published_record_sha256,
                    "byteOrder": "little_endian",
                    "bytesPerRecord": 96,
                },
                "nativeReport": {
                    "path": native_report_path.relative_to(result_root_absolute).as_posix(),
                    "bytes": len(stdout),
                    "sha256": native_report_sha256,
                    "document": native_report,
                },
                "evidenceBoundary": {
                    "scientificRole": "reader_validation_probe_only",
                    "qualificationStatus": "unqualified",
                    "eofReached": False,
                    "publisherCountValidated": False,
                    "surfaceClaim": False,
                    "analogueQualificationAuthorized": False,
                    "targetTruth": False,
                    "synthesisAuthorized": False,
                },
            }
        )
        report_path, report_sha256 = publish_report(
            staged.root,
            result_root,
            publication_report,
        )
        _verify_execution_implementation(authority)
        return NativeProbeResult(
            record_path=record_path,
            record_sha256=published_record_sha256,
            report_path=report_path,
            report_sha256=report_sha256,
            elapsed_seconds=time.monotonic() - started,
        )
    finally:
        if process is not None and process.poll() is None:
            _terminate_group(process)
        if staged is not None:
            staged.close(unlink=not published_records)
        if held is not None:
            held.close()
        os.close(executable_descriptor)
