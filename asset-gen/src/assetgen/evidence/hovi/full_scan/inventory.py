"""Receipt-gated, metadata-only inventory for the retained Hovi full scan."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import inspect
import json
import os
import re
import stat
import struct
import sys
import types
from dataclasses import MISSING, dataclass, fields, is_dataclass
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any

from ....config import ASSET_GEN_ROOT
from .authorization import (
    DEFAULT_AUTHORIZATION,
    MetadataInspectionAuthorization,
    canonical_json_bytes,
    load_metadata_inspection_authorization,
    open_regular_nofollow,
)
from .extract import (
    DEFAULT_OUTPUT_ROOT,
    PREPARED_SCHEMA,
    REPORT_SCHEMA,
    _acquire_transaction_lock,
    _atomic_rename_noreplace_at,
    _fsync_directory,
    _open_or_create_private_directory_at,
    _open_or_create_private_root,
    _open_optional_regular_at,
    _open_regular_at,
    _publication_identity,
)
from .library import E57Runtime, validate_e57_runtime
from .xml_inventory import MAX_XML_BYTES, inventory_xml


INVENTORY_SCHEMA = "hovi-hy-spruce4-full-metadata-inventory/2.0.0"
_HEADER = struct.Struct("<8sIIQQQQ")
_CODE_PATHS = (
    "src/assetgen/evidence/hovi/full_scan/authorization.py",
    "src/assetgen/evidence/hovi/full_scan/extract.py",
    "src/assetgen/evidence/hovi/full_scan/library.py",
    "src/assetgen/evidence/hovi/full_scan/xml_inventory.py",
    "src/assetgen/evidence/hovi/full_scan/inventory.py",
    "pyproject.toml",
    "uv.lock",
)
_RETAINED_MANIFEST = (
    ASSET_GEN_ROOT
    / "data/in/evidence/hovi-full-q1"
    / "cabb67ce94a2c55ee49d0f09b2a000348eb45974b26560f7e8b8d5c6cf543e63"
    / "retained.json"
)
_SUPERSEDED_INVENTORY_SHA256 = (
    "f446b813e4d5b78ff4efeadd3ea4ec7ae36fe02e54778f9847ee4d1a749a6ddd",
    "3a93b4db69c33ef0d84664f45343d2571bbb1cd081b8508e0018a88f76ff067a",
    "456be8611194693e762b6ef14f13b01b36d83a1a4b21eef9e4f3013113a518a1",
)


@dataclass(frozen=True)
class SourceSnapshot:
    path: Path
    relative_path: str
    payload: bytes
    identity: tuple[int, int, int, int, int]


@dataclass
class HeldExtraction:
    root_descriptor: int
    transaction_descriptor: int
    lock_descriptor: int
    receipt_descriptor: int
    prepared_descriptor: int
    object_descriptor: int

    def close(self) -> None:
        fcntl.flock(self.lock_descriptor, fcntl.LOCK_UN)
        for descriptor in (
            self.object_descriptor,
            self.prepared_descriptor,
            self.receipt_descriptor,
            self.lock_descriptor,
            self.transaction_descriptor,
            self.root_descriptor,
        ):
            os.close(descriptor)


@dataclass(frozen=True)
class ExtractionBinding:
    receipt_path: Path
    receipt_bytes: bytes
    receipt: dict
    prepared_path: Path
    prepared_bytes: bytes
    object_path: Path
    object_relative: str
    object_sha256: str
    object_bytes: int


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _stable_identity(result: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        result.st_dev,
        result.st_ino,
        result.st_size,
        result.st_mtime_ns,
        result.st_ctime_ns,
    )


def _read_descriptor(descriptor: int, label: str) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    before = os.fstat(descriptor)
    chunks = []
    while block := os.read(descriptor, 1 << 20):
        chunks.append(block)
    after = os.fstat(descriptor)
    payload = b"".join(chunks)
    if _stable_identity(before) != _stable_identity(after) or len(payload) != before.st_size:
        raise ValueError(f"{label} changed while being read")
    return payload


def _read_regular(path: Path, label: str) -> bytes:
    descriptor = open_regular_nofollow(path, label)
    try:
        return _read_descriptor(descriptor, label)
    finally:
        os.close(descriptor)


def _canonical_object(payload: bytes, label: str) -> dict:
    parsed = json.loads(payload)
    if not isinstance(parsed, dict) or canonical_json_bytes(parsed) != payload:
        raise ValueError(f"{label} is not canonical JSON")
    return parsed


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_extraction_binding(
    authorization: MetadataInspectionAuthorization,
    output_root: Path,
) -> ExtractionBinding:
    output_root = _absolute(output_root)
    transaction_root = output_root / "transactions" / authorization.transaction_id
    receipt_path = transaction_root / "extraction.json"
    receipt_bytes = _read_regular(receipt_path, "completed extraction receipt")
    receipt = _canonical_object(receipt_bytes, "completed extraction receipt")
    expected_keys = {
        "schemaVersion",
        "status",
        "authorizationConfigSha256",
        "retainedManifestSha256",
        "archiveSha256",
        "preparedSha256",
        "member",
        "output",
        "resumePolicy",
        "scientificRole",
        "qualificationStatus",
        "synthesisAuthorized",
    }
    member = receipt.get("member")
    if (
        set(receipt) != expected_keys
        or receipt.get("schemaVersion") != REPORT_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("authorizationConfigSha256") != authorization.config_sha256
        or receipt.get("retainedManifestSha256")
        != authorization.retained_manifest_sha256
        or receipt.get("archiveSha256") != authorization.archive_sha256
        or receipt.get("resumePolicy") != "restart_only"
        or receipt.get("scientificRole") != "raw_candidate"
        or receipt.get("qualificationStatus") != "unqualified"
        or receipt.get("synthesisAuthorized") is not False
        or not isinstance(member, dict)
        or set(member) != {"name", "bytes", "crc32", "sha256"}
        or member.get("name") != authorization.member.name
        or member.get("bytes") != authorization.member.uncompressed_bytes
        or member.get("crc32") != f"{authorization.member.crc32:08x}"
        or not _valid_sha256(member.get("sha256"))
        or not _valid_sha256(receipt.get("preparedSha256"))
    ):
        raise ValueError("completed extraction receipt is outside the authorized boundary")

    object_sha256 = member["sha256"]
    expected_relative = (
        f"objects/sha256/{object_sha256[:2]}/{object_sha256}.e57"
    )
    relative = receipt.get("output")
    if relative != expected_relative:
        raise ValueError("extraction receipt object address is not content-derived")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or "\\" in relative
        or pure.suffix != ".e57"
        or pure.name.endswith(".part")
    ):
        raise ValueError("extraction receipt object path is unsafe")
    object_path = output_root.joinpath(*pure.parts)
    object_descriptor = open_regular_nofollow(object_path, "completed E57 object")
    try:
        object_stat = os.fstat(object_descriptor)
    finally:
        os.close(object_descriptor)
    if (
        object_stat.st_size != authorization.member.uncompressed_bytes
        or object_stat.st_uid != os.getuid()
        or stat.S_IMODE(object_stat.st_mode) != 0o400
        or object_stat.st_nlink != 1
    ):
        raise ValueError("completed E57 object inode is not the immutable receipt object")

    prepared_path = transaction_root / "prepared.json"
    prepared_bytes = _read_regular(prepared_path, "prepared extraction record")
    prepared = _canonical_object(prepared_bytes, "prepared extraction record")
    prepared_member = prepared.get("member")
    if (
        hashlib.sha256(prepared_bytes).hexdigest() != receipt["preparedSha256"]
        or prepared.get("schemaVersion") != PREPARED_SCHEMA
        or prepared.get("status") != "prepared"
        or prepared.get("authorizationConfigSha256") != authorization.config_sha256
        or prepared.get("retainedManifestSha256")
        != authorization.retained_manifest_sha256
        or prepared.get("archiveSha256") != authorization.archive_sha256
        or prepared.get("object") != expected_relative
        or prepared.get("resumePolicy") != "restart_only"
        or prepared.get("synthesisAuthorized") is not False
        or not isinstance(prepared_member, dict)
        or prepared_member.get("sha256") != object_sha256
        or prepared_member.get("name") != authorization.member.name
        or prepared_member.get("uncompressedBytes")
        != authorization.member.uncompressed_bytes
        or prepared_member.get("crc32") != f"{authorization.member.crc32:08x}"
    ):
        raise ValueError("prepared extraction record is not bound to the receipt object")
    return ExtractionBinding(
        receipt_path=receipt_path,
        receipt_bytes=receipt_bytes,
        receipt=receipt,
        prepared_path=prepared_path,
        prepared_bytes=prepared_bytes,
        object_path=object_path,
        object_relative=relative,
        object_sha256=object_sha256,
        object_bytes=object_stat.st_size,
    )


def _snapshot_file(path: Path, relative_path: str) -> SourceSnapshot:
    descriptor = open_regular_nofollow(path, relative_path)
    try:
        payload = _read_descriptor(descriptor, relative_path)
        identity = _stable_identity(os.fstat(descriptor))
    finally:
        os.close(descriptor)
    return SourceSnapshot(
        path=_absolute(path),
        relative_path=relative_path,
        payload=payload,
        identity=identity,
    )


def _source_snapshots(authorization_path: Path) -> dict[str, SourceSnapshot]:
    paths = {relative: ASSET_GEN_ROOT / relative for relative in _CODE_PATHS}
    authorization_path = _absolute(authorization_path)
    paths[authorization_path.relative_to(ASSET_GEN_ROOT).as_posix()] = authorization_path
    paths[_RETAINED_MANIFEST.relative_to(ASSET_GEN_ROOT).as_posix()] = _RETAINED_MANIFEST
    return {
        relative: _snapshot_file(path, relative)
        for relative, path in sorted(paths.items())
    }


def _verify_snapshots(snapshots: dict[str, SourceSnapshot], phase: str) -> None:
    for snapshot in snapshots.values():
        descriptor = open_regular_nofollow(snapshot.path, snapshot.relative_path)
        try:
            payload = _read_descriptor(descriptor, snapshot.relative_path)
            identity = _stable_identity(os.fstat(descriptor))
        finally:
            os.close(descriptor)
        if payload != snapshot.payload or identity != snapshot.identity:
            raise ValueError(
                f"inventory source changed {phase}: {snapshot.relative_path}"
            )


def _snapshot_records(snapshots: dict[str, SourceSnapshot]) -> list[dict]:
    return [
        {
            "path": snapshot.relative_path,
            "bytes": len(snapshot.payload),
            "sha256": hashlib.sha256(snapshot.payload).hexdigest(),
            "identityHeldAtStart": {
                "device": snapshot.identity[0],
                "inode": snapshot.identity[1],
                "bytes": snapshot.identity[2],
                "mtimeNs": snapshot.identity[3],
                "ctimeNs": snapshot.identity[4],
            },
        }
        for snapshot in snapshots.values()
    ]


def _fingerprint_value(value: Any) -> Any:
    if value is MISSING:
        return {"singleton": "dataclasses.MISSING"}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return {"pythonFloatHex": value.hex()}
    if isinstance(value, bytes):
        return {"bytesHex": value.hex()}
    if isinstance(value, Path):
        return {"path": value.as_posix()}
    if isinstance(value, Decimal):
        return {"decimal": str(value)}
    if isinstance(value, types.CodeType):
        return {"code": _code_fingerprint_record(value)}
    if isinstance(value, tuple):
        return {"tuple": [_fingerprint_value(item) for item in value]}
    if isinstance(value, list):
        return {"list": [_fingerprint_value(item) for item in value]}
    if isinstance(value, (set, frozenset)):
        items = [_fingerprint_value(item) for item in value]
        return {
            "set": sorted(
                items,
                key=lambda item: canonical_json_bytes(item),
            )
        }
    if isinstance(value, dict):
        entries = [
            {
                "key": _fingerprint_value(key),
                "value": _fingerprint_value(item),
            }
            for key, item in value.items()
        ]
        return {
            "mapping": sorted(
                entries,
                key=lambda entry: canonical_json_bytes(entry["key"]),
            )
        }
    if isinstance(value, re.Pattern):
        return {
            "regex": (
                value.pattern.hex()
                if isinstance(value.pattern, bytes)
                else value.pattern
            ),
            "patternIsBytes": isinstance(value.pattern, bytes),
            "flags": value.flags,
        }
    if isinstance(value, struct.Struct):
        return {"structFormat": value.format, "size": value.size}
    if isinstance(value, (types.BuiltinFunctionType, types.BuiltinMethodType)):
        return {
            "builtinCallable": (
                f"{getattr(value, '__module__', '')}."
                f"{getattr(value, '__qualname__', value.__name__)}"
            )
        }
    if isinstance(value, type):
        return {"pythonType": f"{value.__module__}.{value.__qualname__}"}
    if value is Ellipsis:
        return {"singleton": "Ellipsis"}
    raise TypeError(
        f"loaded implementation fingerprint cannot encode {type(value).__name__}"
    )


def _code_fingerprint_record(code: types.CodeType) -> dict:
    return {
        "name": code.co_name,
        "qualname": code.co_qualname,
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "bytecodeHex": code.co_code.hex(),
        "constants": [_fingerprint_value(value) for value in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
    }


def _function_fingerprint_record(function: types.FunctionType) -> dict:
    return {
        "module": function.__module__,
        "qualname": function.__qualname__,
        "code": _code_fingerprint_record(function.__code__),
        "defaults": _fingerprint_value(function.__defaults__),
        "kwdefaults": _fingerprint_value(function.__kwdefaults__),
        "annotations": _fingerprint_value(function.__annotations__),
    }


def _class_fingerprint_record(value: type) -> dict:
    methods = []
    for name, member in sorted(vars(value).items()):
        functions = []
        if isinstance(member, (staticmethod, classmethod)):
            functions = [member.__func__]
        elif isinstance(member, types.FunctionType):
            functions = [member]
        elif isinstance(member, property):
            functions = [
                function
                for function in (member.fget, member.fset, member.fdel)
                if function is not None
            ]
        for function in functions:
            methods.append(
                {
                    "attribute": name,
                    "function": _function_fingerprint_record(function),
                }
            )
    dataclass_fields = []
    if is_dataclass(value):
        for field in fields(value):
            entry = {
                "name": field.name,
                "type": _fingerprint_value(field.type),
                "init": field.init,
                "repr": field.repr,
                "compare": field.compare,
                "kwOnly": field.kw_only,
            }
            if field.default is not MISSING:
                entry["default"] = _fingerprint_value(field.default)
            if field.default_factory is not MISSING:
                factory = field.default_factory
                entry["defaultFactory"] = (
                    f"{factory.__module__}.{factory.__qualname__}"
                )
            dataclass_fields.append(entry)
    return {
        "module": value.__module__,
        "qualname": value.__qualname__,
        "bases": [f"{base.__module__}.{base.__qualname__}" for base in value.__bases__],
        "methods": methods,
        "dataclassFields": dataclass_fields,
    }


def _module_implementation_record(module: types.ModuleType) -> dict:
    callables = []
    classes = []
    constants = []
    for name, value in sorted(vars(module).items()):
        if isinstance(value, types.FunctionType) and value.__module__ == module.__name__:
            callables.append(
                {"name": name, "function": _function_fingerprint_record(value)}
            )
        elif isinstance(value, type) and value.__module__ == module.__name__:
            classes.append({"name": name, "class": _class_fingerprint_record(value)})
        elif name.isupper() and not isinstance(value, types.ModuleType):
            constants.append({"name": name, "value": _fingerprint_value(value)})
    body = {
        "module": module.__name__,
        "callables": callables,
        "classes": classes,
        "constants": constants,
    }
    return {
        **body,
        "sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }


def _loaded_implementation_fingerprint() -> dict:
    module_names = (
        "assetgen.evidence.hovi.full_scan.authorization",
        "assetgen.evidence.hovi.full_scan.extract",
        "assetgen.evidence.hovi.full_scan.library",
        "assetgen.evidence.hovi.full_scan.xml_inventory",
        __name__,
    )
    modules = []
    for name in module_names:
        module = sys.modules.get(name)
        if module is None:
            raise RuntimeError(f"implementation module is not loaded: {name}")
        modules.append(_module_implementation_record(module))
    body = {
        "schemaVersion": "hovi-loaded-python-implementation-fingerprint/1.0.0",
        "scope": (
            "loaded callable code objects, recursive code constants, function defaults, "
            "defined classes, and uppercase module constants"
        ),
        "sourceSnapshotRelationship": (
            "separate from on-disk source snapshots; source hashes do not by themselves "
            "bind already-loaded Python execution"
        ),
        "modules": modules,
    }
    return {
        **body,
        "sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }


def _verify_loaded_implementation_fingerprint(
    expected: dict,
    phase: str,
) -> None:
    if _loaded_implementation_fingerprint() != expected:
        raise ValueError(f"loaded Python implementation changed {phase}")


def _code_hashes(snapshots: dict[str, SourceSnapshot] | None = None) -> list[dict]:
    if snapshots is None:
        snapshots = {
            relative: _snapshot_file(ASSET_GEN_ROOT / relative, relative)
            for relative in _CODE_PATHS
        }
    return [
        {
            "path": relative,
            "bytes": len(snapshots[relative].payload),
            "sha256": hashlib.sha256(snapshots[relative].payload).hexdigest(),
        }
        for relative in _CODE_PATHS
    ]


def metadata_inventory_plan(
    authorization_path: Path = DEFAULT_AUTHORIZATION,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict:
    authorization = load_metadata_inspection_authorization(authorization_path)
    runtime = validate_e57_runtime()
    output_root = _absolute(output_root)
    binding = _load_extraction_binding(authorization, output_root)
    return {
        "schemaVersion": INVENTORY_SCHEMA,
        "status": "ready_for_metadata_only_inventory",
        "authorizationConfig": {
            "path": authorization.config_path.as_posix(),
            "sha256": authorization.config_sha256,
        },
        "extractionReceipt": {
            "path": binding.receipt_path.as_posix(),
            "bytes": len(binding.receipt_bytes),
            "sha256": hashlib.sha256(binding.receipt_bytes).hexdigest(),
            "status": "complete",
        },
        "object": {
            "path": binding.object_path.as_posix(),
            "relativePath": binding.object_relative,
            "bytes": binding.object_bytes,
            "sha256": binding.object_sha256,
            "mode": "0400",
        },
        "reader": runtime.binding,
        "codeHashes": _code_hashes(),
        "xmlHardCapBytes": MAX_XML_BYTES,
        "apiBoundary": "e57.raw_xml_only",
        "rawXmlCalled": False,
        "pointsRead": 0,
        "synthesisAuthorized": False,
        "targetTruth": False,
    }


def _read_header(descriptor: int, expected_bytes: int) -> dict:
    payload = os.pread(descriptor, _HEADER.size, 0)
    if len(payload) != _HEADER.size:
        raise ValueError("completed E57 object lacks its 48-byte header")
    signature, major, minor, physical_length, xml_offset, xml_length, page_size = (
        _HEADER.unpack(payload)
    )
    if (
        signature != b"ASTM-E57"
        or major != 1
        or minor != 0
        or physical_length != expected_bytes
        or page_size != 1024
        or xml_offset >= expected_bytes
        or xml_length > MAX_XML_BYTES
    ):
        raise ValueError("completed E57 header is outside the metadata-reader boundary")
    ranges = []
    physical = xml_offset
    logical = 0
    remaining = xml_length
    while remaining:
        page_offset = physical % page_size
        if page_offset >= page_size - 4:
            raise ValueError("E57 XML begins inside a page checksum")
        span = min(remaining, page_size - 4 - page_offset)
        ranges.append(
            {
                "physicalOffset": physical,
                "logicalOffset": logical,
                "bytes": span,
            }
        )
        physical += span
        logical += span
        remaining -= span
        if remaining and physical % page_size == page_size - 4:
            physical += 4
    if ranges and ranges[-1]["physicalOffset"] + ranges[-1]["bytes"] > expected_bytes:
        raise ValueError("E57 XML physical ranges exceed the receipt object")
    return {
        "signature": signature.decode("ascii"),
        "versionMajor": major,
        "versionMinor": minor,
        "physicalLength": physical_length,
        "xmlPhysicalOffset": xml_offset,
        "xmlLogicalLength": xml_length,
        "pageSize": page_size,
        "headerReadRange": {"physicalOffset": 0, "bytes": _HEADER.size},
        "xmlLogicalPayloadPhysicalRanges": ranges,
        "xmlChecksumBytesExcludedPerPage": 4,
    }


def _open_private_directory_at(parent: int, name: str, label: str) -> int:
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"unsafe metadata inventory directory name: {name!r}")
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd=parent,
    )
    result = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(result.st_mode)
        or result.st_uid != os.getuid()
        or stat.S_IMODE(result.st_mode) != 0o700
    ):
        os.close(descriptor)
        raise PermissionError(f"metadata inventory directory is not private: {label}")
    return descriptor


def _validate_bound_regular(
    directory: int,
    name: str,
    descriptor: int,
    label: str,
    modes: set[int],
) -> os.stat_result:
    result = os.fstat(descriptor)
    namespace = os.stat(name, dir_fd=directory, follow_symlinks=False)
    if (
        not stat.S_ISREG(result.st_mode)
        or result.st_uid != os.getuid()
        or stat.S_IMODE(result.st_mode) not in modes
        or result.st_nlink != 1
        or _publication_identity(result) != _publication_identity(namespace)
    ):
        raise PermissionError(f"{label} is not the expected private regular inode")
    return result


def _read_transaction_record(
    transaction: int,
    name: str,
    label: str,
) -> tuple[int, bytes]:
    descriptor = _open_regular_at(transaction, name, label)
    try:
        _validate_bound_regular(transaction, name, descriptor, label, {0o400, 0o600})
        payload = _read_descriptor(descriptor, label)
        _canonical_object(payload, label)
        _validate_bound_regular(transaction, name, descriptor, label, {0o400, 0o600})
    except Exception:
        os.close(descriptor)
        raise
    return descriptor, payload


def _finalize_transaction_record(
    transaction: int,
    name: str,
    descriptor: int,
    label: str,
) -> None:
    _validate_bound_regular(transaction, name, descriptor, label, {0o400, 0o600})
    if stat.S_IMODE(os.fstat(descriptor).st_mode) == 0o600:
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        _fsync_directory(transaction)
    _validate_bound_regular(transaction, name, descriptor, label, {0o400})


def _open_object_at_root(
    root: int,
    relative: str,
    label: str,
) -> int:
    parts = PurePosixPath(relative).parts
    directory = os.dup(root)
    try:
        for part in parts[:-1]:
            next_directory = _open_private_directory_at(directory, part, part)
            os.close(directory)
            directory = next_directory
        descriptor = _open_regular_at(directory, parts[-1], label)
        _validate_bound_regular(
            directory,
            parts[-1],
            descriptor,
            label,
            {0o400},
        )
        return descriptor
    finally:
        os.close(directory)


def _open_held_extraction(
    authorization: MetadataInspectionAuthorization,
    output_root: Path,
) -> tuple[ExtractionBinding, HeldExtraction]:
    root = _open_or_create_private_root(output_root)
    transaction = lock = receipt = prepared = object_descriptor = -1
    transactions = -1
    try:
        transactions = _open_private_directory_at(root, "transactions", "transactions")
        transaction = _open_private_directory_at(
            transactions,
            authorization.transaction_id,
            "completed extraction transaction",
        )
        os.close(transactions)
        transactions = -1
        lock = _acquire_transaction_lock(transaction)
        receipt, receipt_bytes = _read_transaction_record(
            transaction,
            "extraction.json",
            "completed extraction receipt",
        )
        prepared, prepared_bytes = _read_transaction_record(
            transaction,
            "prepared.json",
            "prepared extraction record",
        )
        binding = _load_extraction_binding(authorization, output_root)
        if (
            binding.receipt_bytes != receipt_bytes
            or binding.prepared_bytes != prepared_bytes
        ):
            raise ValueError("held extraction records differ from their validated binding")
        object_descriptor = _open_object_at_root(
            root,
            binding.object_relative,
            "completed E57 object",
        )
        object_stat = os.fstat(object_descriptor)
        if (
            object_stat.st_size != binding.object_bytes
            or object_stat.st_uid != os.getuid()
            or stat.S_IMODE(object_stat.st_mode) != 0o400
            or object_stat.st_nlink != 1
        ):
            raise ValueError("held E57 object differs from its completed receipt")
        # Both records were validated against the object before this one-way migration.
        _finalize_transaction_record(
            transaction,
            "prepared.json",
            prepared,
            "prepared extraction record",
        )
        _finalize_transaction_record(
            transaction,
            "extraction.json",
            receipt,
            "completed extraction receipt",
        )
        _fsync_directory(transaction)
        return binding, HeldExtraction(
            root_descriptor=root,
            transaction_descriptor=transaction,
            lock_descriptor=lock,
            receipt_descriptor=receipt,
            prepared_descriptor=prepared,
            object_descriptor=object_descriptor,
        )
    except Exception:
        for descriptor in (
            object_descriptor,
            prepared,
            receipt,
            lock,
            transaction,
            transactions,
            root,
        ):
            if descriptor >= 0:
                os.close(descriptor)
        raise


def _read_canonical_inventory(
    directory: int,
    name: str,
    payload: bytes,
    label: str,
) -> int:
    descriptor = _open_regular_at(directory, name, label)
    try:
        _validate_canonical_inventory_descriptor(
            directory,
            name,
            descriptor,
            payload,
            label,
        )
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _validate_canonical_inventory_descriptor(
    directory: int,
    name: str,
    descriptor: int,
    payload: bytes,
    label: str,
) -> None:
    before = _validate_bound_regular(directory, name, descriptor, label, {0o400})
    existing = _read_descriptor(descriptor, label)
    after = _validate_bound_regular(directory, name, descriptor, label, {0o400})
    if _stable_identity(before) != _stable_identity(after):
        raise ValueError("metadata inventory inode changed during equality check")
    _canonical_object(existing, label)
    if existing != payload:
        raise ValueError("metadata inventory content-address collision")


def _reverify_held_record(
    transaction: int,
    name: str,
    descriptor: int,
    expected: bytes,
    label: str,
) -> None:
    before = _validate_bound_regular(transaction, name, descriptor, label, {0o400})
    actual = _read_descriptor(descriptor, label)
    after = _validate_bound_regular(transaction, name, descriptor, label, {0o400})
    if (
        actual != expected
        or _stable_identity(before) != _stable_identity(after)
        or canonical_json_bytes(_canonical_object(actual, label)) != actual
    ):
        raise ValueError(f"held {label} changed during metadata inventory")


def _unlink_bound_temp(
    directory: int,
    name: str,
    descriptor: int,
    expected: tuple[int, int, int, int, int],
) -> None:
    result = os.fstat(descriptor)
    namespace = os.stat(name, dir_fd=directory, follow_symlinks=False)
    if (
        _stable_identity(result) != expected
        or _stable_identity(namespace) != expected
    ):
        raise ValueError("metadata inventory temporary changed before unlink")
    os.unlink(name, dir_fd=directory)
    _fsync_directory(directory)


def _publish_inventory(
    output_root_descriptor: int,
    output_root: Path,
    payload: bytes,
) -> tuple[Path, str]:
    content_sha256 = hashlib.sha256(payload).hexdigest()
    inventories = _open_or_create_private_directory_at(
        output_root_descriptor,
        "inventories",
        "metadata inventories",
    )
    sha256_directory = prefix_directory = -1
    temporary: int | None = None
    try:
        sha256_directory = _open_or_create_private_directory_at(
            inventories,
            "sha256",
            "metadata inventory sha256 directory",
        )
        prefix_directory = _open_or_create_private_directory_at(
            sha256_directory,
            content_sha256[:2],
            "metadata inventory hash prefix",
        )
        final_name = f"{content_sha256}.json"
        temporary_name = f".{content_sha256}.json.tmp"
        final = _open_optional_regular_at(prefix_directory, final_name, final_name)
        if final is not None:
            try:
                _validate_canonical_inventory_descriptor(
                    prefix_directory,
                    final_name,
                    final,
                    payload,
                    final_name,
                )
            finally:
                os.close(final)
            _fsync_directory(prefix_directory)
            return (
                output_root / "inventories" / "sha256" / content_sha256[:2] / final_name,
                content_sha256,
            )

        temporary = _open_optional_regular_at(
            prefix_directory,
            temporary_name,
            temporary_name,
        )
        if temporary is not None:
            try:
                _validate_bound_regular(
                    prefix_directory,
                    temporary_name,
                    temporary,
                    temporary_name,
                    {0o400, 0o600},
                )
                existing = _read_descriptor(temporary, temporary_name)
                identity = _stable_identity(os.fstat(temporary))
                if existing != payload:
                    _unlink_bound_temp(
                        prefix_directory,
                        temporary_name,
                        temporary,
                        identity,
                    )
                    os.close(temporary)
                    temporary = None
                else:
                    os.fchmod(temporary, 0o400)
                    os.fsync(temporary)
            except Exception:
                if temporary is not None:
                    os.close(temporary)
                raise
        if temporary is None:
            descriptor = os.open(
                temporary_name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=prefix_directory,
            )
            temporary = descriptor
            _validate_bound_regular(
                prefix_directory,
                temporary_name,
                temporary,
                temporary_name,
                {0o600},
            )
            view = memoryview(payload)
            while view:
                written = os.write(temporary, view)
                if written <= 0:
                    raise OSError("short write while publishing metadata inventory")
                view = view[written:]
            os.fchmod(temporary, 0o400)
            os.fsync(temporary)
        _validate_bound_regular(
            prefix_directory,
            temporary_name,
            temporary,
            temporary_name,
            {0o400},
        )
        temporary_identity = _stable_identity(os.fstat(temporary))
        published_own_inode = True
        try:
            _atomic_rename_noreplace_at(
                prefix_directory,
                temporary_name,
                prefix_directory,
                final_name,
            )
        except FileExistsError:
            published_own_inode = False
            final = _read_canonical_inventory(
                prefix_directory,
                final_name,
                payload,
                final_name,
            )
            os.close(final)
            _unlink_bound_temp(
                prefix_directory,
                temporary_name,
                temporary,
                temporary_identity,
            )
        final = _read_canonical_inventory(
            prefix_directory,
            final_name,
            payload,
            final_name,
        )
        try:
            if published_own_inode and _publication_identity(os.fstat(final)) != (
                temporary_identity[0],
                temporary_identity[1],
                temporary_identity[2],
            ):
                raise ValueError("metadata inventory publication changed inode identity")
        finally:
            os.close(final)
        os.close(temporary)
        temporary = None
        _fsync_directory(prefix_directory)
        return (
            output_root / "inventories" / "sha256" / content_sha256[:2] / final_name,
            content_sha256,
        )
    finally:
        if temporary is not None:
            os.close(temporary)
        for descriptor in (prefix_directory, sha256_directory, inventories):
            if descriptor >= 0:
                os.close(descriptor)


def create_metadata_inventory(
    authorization_path: Path = DEFAULT_AUTHORIZATION,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[Path, str]:
    implementation_fingerprint = _loaded_implementation_fingerprint()
    snapshots = _source_snapshots(authorization_path)
    authorization_relative = _absolute(authorization_path).relative_to(
        ASSET_GEN_ROOT
    ).as_posix()
    retained_relative = _RETAINED_MANIFEST.relative_to(ASSET_GEN_ROOT).as_posix()
    authorization = load_metadata_inspection_authorization(
        authorization_path,
        config_bytes=snapshots[authorization_relative].payload,
        retained_manifest_bytes=snapshots[retained_relative].payload,
    )
    runtime: E57Runtime = validate_e57_runtime(
        pyproject_bytes=snapshots["pyproject.toml"].payload,
        lock_bytes=snapshots["uv.lock"].payload,
    )
    output_root = _absolute(output_root)
    binding, held = _open_held_extraction(authorization, output_root)
    try:
        _verify_loaded_implementation_fingerprint(
            implementation_fingerprint,
            "before descriptor-bound XML read",
        )
        _verify_snapshots(snapshots, "before descriptor-bound XML read")
        verified_identity = os.fstat(held.object_descriptor)
        if (
            not stat.S_ISREG(verified_identity.st_mode)
            or verified_identity.st_uid != os.getuid()
            or stat.S_IMODE(verified_identity.st_mode) != 0o400
            or verified_identity.st_nlink != 1
            or verified_identity.st_size != binding.object_bytes
        ):
            raise ValueError("held E57 object is not the immutable receipt inode")
        header = _read_header(held.object_descriptor, binding.object_bytes)
        reader_descriptor = os.dup(held.object_descriptor)
        try:
            descriptor_path = f"/dev/fd/{reader_descriptor}"
            probe_descriptor = os.open(descriptor_path, os.O_RDONLY)
            try:
                if _publication_identity(os.fstat(probe_descriptor)) != (
                    _publication_identity(os.fstat(reader_descriptor))
                ):
                    raise ValueError(
                        "/dev/fd reader path is not bound to the held E57 inode"
                    )
            finally:
                os.close(probe_descriptor)
            raw_xml = runtime.raw_xml(descriptor_path)
            if not isinstance(raw_xml, str):
                raise TypeError("e57.raw_xml returned a non-string value")
            xml_bytes = raw_xml.encode("utf-8", errors="strict")
        finally:
            os.close(reader_descriptor)
        if len(xml_bytes) > MAX_XML_BYTES or len(xml_bytes) != header["xmlLogicalLength"]:
            raise ValueError("e57.raw_xml violated the E57 header XML-length boundary")
        after = os.fstat(held.object_descriptor)
        if _stable_identity(after) != _stable_identity(verified_identity):
            raise ValueError("held receipt object changed during raw XML extraction")

        interpreted = inventory_xml(xml_bytes)
        publisher_record_total = sum(
            scan["points"]["publisherDeclaredRecordCount"]["value"]
            for scan in interpreted["root"]["data3D"]["scans"]
        )
        document = {
            "schemaVersion": INVENTORY_SCHEMA,
            "status": "metadata_inventory_complete",
            "scope": "publisher_xml_metadata_only",
            "authoritySelection": {
                "candidateStatus": "eligible_for_frozen_external_selection",
                "supersedesInventorySha256": list(
                    _SUPERSEDED_INVENTORY_SHA256
                ),
                "supersessionReasons": [
                    "incorrect_present_empty_numeric_semantics_and_payload_reread",
                    "missing_final_ABA_hardening",
                    "missing_loaded_execution_fingerprint",
                ],
            },
            "extraction": {
                "authorizationConfig": {
                    "path": authorization.config_path.as_posix(),
                    "bytes": len(snapshots[authorization_relative].payload),
                    "sha256": authorization.config_sha256,
                },
                "retainedManifest": {
                    "path": authorization.retained_manifest_path.as_posix(),
                    "bytes": authorization.retained_manifest_bytes,
                    "sha256": authorization.retained_manifest_sha256,
                    "retentionId": authorization.retention_id,
                },
                "receipt": {
                    "path": binding.receipt_path.as_posix(),
                    "bytes": len(binding.receipt_bytes),
                    "sha256": hashlib.sha256(binding.receipt_bytes).hexdigest(),
                    "status": "complete",
                    "mode": "0400",
                    "linkCount": 1,
                },
                "prepared": {
                    "path": binding.prepared_path.as_posix(),
                    "bytes": len(binding.prepared_bytes),
                    "sha256": hashlib.sha256(binding.prepared_bytes).hexdigest(),
                    "mode": "0400",
                    "linkCount": 1,
                },
                "object": {
                    "path": binding.object_path.as_posix(),
                    "relativePath": binding.object_relative,
                    "bytes": binding.object_bytes,
                    "sha256": binding.object_sha256,
                    "mode": "0400",
                    "linkCount": 1,
                },
            },
            "objectVerificationBasis": {
                "inventoryTimeFullPayloadDigestRead": False,
                "inventoryTimeCompressedPointPayloadRead": False,
                "reliedOnCompletedExtractionReceipt": True,
                "receiptBoundContentAddress": binding.object_relative,
                "receiptBoundSha256": binding.object_sha256,
                "extractionVerificationAlreadyCompleted": [
                    "uncompressed_size",
                    "zip_crc32",
                    "stream_sha256",
                    "independent_part_sha256",
                ],
                "heldInode": {
                    "device": verified_identity.st_dev,
                    "inode": verified_identity.st_ino,
                    "bytes": verified_identity.st_size,
                    "uid": verified_identity.st_uid,
                    "mode": "0400",
                    "linkCount": verified_identity.st_nlink,
                },
                "nativeReaderPathBinding": "verified_/dev/fd/<held-duplicate>",
                "headerBytesReadByInventory": _HEADER.size,
                "xmlLogicalBytesReturnedByPinnedReader": len(xml_bytes),
                "recordsDecoded": 0,
            },
            "implementation": {
                "apiBoundary": "e57.raw_xml_only",
                "reader": runtime.binding,
                "loadedExecutionFingerprint": implementation_fingerprint,
                "codeHashes": _code_hashes(snapshots),
                "sourceSnapshots": _snapshot_records(snapshots),
                "sourceSnapshotScope": (
                    "on-disk bytes and inode identity only; these snapshots do not "
                    "by themselves bind code already loaded by Python"
                ),
                "sourceSnapshotsReverifiedBeforePublication": True,
                "sourceSnapshotsReverifiedAfterPublicationBeforeAcceptance": True,
                "loadedExecutionFingerprintReverifiedBeforePublication": True,
                "loadedExecutionFingerprintReverifiedAfterPublicationBeforeAcceptance": True,
            },
            "e57Header": header,
            "xml": {
                "bytes": len(xml_bytes),
                "sha256": hashlib.sha256(xml_bytes).hexdigest(),
                "hardCapBytes": MAX_XML_BYTES,
                "dtdEntityExternalIdentifiersRejected": True,
                "networkAccess": False,
                "readViaDescriptorBoundPinnedNativeApi": True,
            },
            "publisherMetadataSummary": {
                "xmlScanElementsCount": len(
                    interpreted["root"]["data3D"]["scans"]
                ),
                "publisherDeclaredPointRecordCountTotal": publisher_record_total,
                "pointRecordCountValidatedByDecode": False,
            },
            "inventory": interpreted,
            "pointsRead": 0,
            "groupRecordsRead": 0,
            "compressedVectorRecordsDecoded": 0,
            "scientificRole": "raw_candidate_metadata",
            "qualificationStatus": "unqualified",
            "synthesisAuthorized": False,
            "targetTruth": False,
            "usableSurface": False,
        }
        payload = canonical_json_bytes(document)
        _reverify_held_record(
            held.transaction_descriptor,
            "prepared.json",
            held.prepared_descriptor,
            binding.prepared_bytes,
            "prepared extraction record",
        )
        _reverify_held_record(
            held.transaction_descriptor,
            "extraction.json",
            held.receipt_descriptor,
            binding.receipt_bytes,
            "completed extraction receipt",
        )
        _verify_loaded_implementation_fingerprint(
            implementation_fingerprint,
            "before inventory publication",
        )
        _verify_snapshots(snapshots, "before inventory publication")
        result = _publish_inventory(
            held.root_descriptor,
            output_root,
            payload,
        )
        _verify_snapshots(snapshots, "after inventory publication")
        _verify_loaded_implementation_fingerprint(
            implementation_fingerprint,
            "after inventory publication",
        )
        _reverify_held_record(
            held.transaction_descriptor,
            "prepared.json",
            held.prepared_descriptor,
            binding.prepared_bytes,
            "prepared extraction record",
        )
        _reverify_held_record(
            held.transaction_descriptor,
            "extraction.json",
            held.receipt_descriptor,
            binding.receipt_bytes,
            "completed extraction receipt",
        )
        return result
    finally:
        held.close()


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan or create the point-free Hovi E57 XML metadata inventory."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_AUTHORIZATION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                metadata_inventory_plan(args.config, args.output_root),
                indent=2,
            )
        )
        return
    path, content_sha256 = create_metadata_inventory(args.config, args.output_root)
    print(f"metadata inventory: {path}")
    print(f"metadata inventory sha256: {content_sha256}")


if __name__ == "__main__":
    _main()
