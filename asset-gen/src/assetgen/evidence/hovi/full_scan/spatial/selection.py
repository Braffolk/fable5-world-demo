"""Content-addressed execution selection for Hovi spatial materialization."""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import stat
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Mapping, Any

from .....config import ASSET_GEN_ROOT, DATA_WORK
from ..authorization import (
    canonical_json_bytes,
    load_metadata_inspection_authorization,
    open_regular_nofollow,
)
from ..extract import _open_or_create_private_root
from ..inventory import _open_held_extraction
from .authority import (
    LAUNCH_FREE_BYTES,
    OBJECT_BYTES,
    OBJECT_SHA256,
    PUBLISHER_RECORD_COUNT,
    TREE_CEILING_BYTES,
    NativeBinding,
    SpatialMaterializationAuthority,
    load_spatial_authority,
)


EXECUTION_SCHEMA = "hovi-hy-spruce4-full-spatial-materialization-execution/1.1.0"
DEFAULT_SPATIAL_OUTPUT_ROOT = DATA_WORK / "hovi-full-scan-spatial-materialization"
DEFAULT_EXECUTION_SELECTION_ROOT = (
    DEFAULT_SPATIAL_OUTPUT_ROOT / "execution-selections" / "sha256"
)
_READER_COMMIT = "e9eb37acd31084fa4a5ef6559db829de0c334301"
_ORCHESTRATOR_COMMIT = "6195439e5e382e0d62f234ec2bbba9d023c77370"
_EXECUTION_SELECTION_ENABLED = True
_NATIVE_SOURCE_TREE_SHA256 = (
    "3763dad65b8a6ea61c46bd11859ec4ac4c6ebbf10d6068b26a38008be8d1b9ca"
)
_CARGO_LOCK_SHA256 = (
    "4e172cb2f049a23395f487df481e4902794c699a86c36d10a08930418ee16feb"
)
_UPSTREAM_MANIFEST_SHA256 = (
    "0fee5bddad61489630e560947a64c637ba03a483d13a545af066de57efc2186a"
)
_VENDOR_PATCH_SET_SHA256 = (
    "e5f691c7cfac1ee336acb97a6c156d60d3b52f71d726fd7b8a44f5243db77f32"
)
_EXECUTABLE_RELATIVE = (
    "data/work/hovi-full-scan-spatial-materialization/readers/sha256/ce/"
    "ce0e26e36c7f7ffd1f8ccda7316367947e3448a045bbb07d22167f59a0279d32"
)
_EXECUTABLE_BYTES = 653_824
_EXECUTABLE_SHA256 = (
    "ce0e26e36c7f7ffd1f8ccda7316367947e3448a045bbb07d22167f59a0279d32"
)
_NUMPY_DISTRIBUTION = "numpy"
_NUMPY_VERSION = "2.4.6"
_NUMPY_ENVIRONMENT_ROOT = ASSET_GEN_ROOT / ".venv"
_IMPLEMENTATION_PATHS = frozenset(
    {
        "pyproject.toml",
        "uv.lock",
        "src/assetgen/config.py",
        "src/assetgen/evidence/hovi/full_scan/authorization.py",
        "src/assetgen/evidence/hovi/full_scan/extract.py",
        "src/assetgen/evidence/hovi/full_scan/inventory.py",
        "src/assetgen/evidence/hovi/full_scan/spatial/__init__.py",
        "src/assetgen/evidence/hovi/full_scan/spatial/artifacts.py",
        "src/assetgen/evidence/hovi/full_scan/spatial/authority.py",
        "src/assetgen/evidence/hovi/full_scan/spatial/orchestrator.py",
        "src/assetgen/evidence/hovi/full_scan/spatial/runner.py",
        "src/assetgen/evidence/hovi/full_scan/spatial/selection.py",
    }
)


@dataclass(frozen=True)
class SpatialExecutionSelection:
    executable_fd: int
    input_fd: int
    executable_path: Path
    staging_root: Path
    argv: tuple[str, ...]
    pass_fds: tuple[int, ...]
    cwd: Path
    environment: dict[str, str]
    child_umask: int
    source_identity: tuple[int, ...]
    executable_identity: tuple[int, ...]
    free_bytes_at_selection: int


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Hovi spatial execution {label} must be an object")
    return value


def _asset_path(value: Any, label: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"Hovi spatial execution {label} must be a relative path")
    parts = Path(value)
    if parts.is_absolute() or not parts.parts or ".." in parts.parts or "\\" in value:
        raise ValueError(f"unsafe Hovi spatial execution {label}")
    return _absolute(ASSET_GEN_ROOT.joinpath(*parts.parts))


def _identity(result: os.stat_result) -> tuple[int, ...]:
    return (
        result.st_dev,
        result.st_ino,
        result.st_mode,
        result.st_uid,
        result.st_gid,
        result.st_nlink,
        result.st_size,
        result.st_mtime_ns,
        result.st_ctime_ns,
        int(getattr(result, "st_flags", 0)),
    )


def _descriptor_sha256(
    descriptor: int,
    label: str,
    *,
    executable: bool = False,
) -> str:
    before = os.fstat(descriptor)
    if executable and (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.getuid()
        or before.st_nlink != 1
        or before.st_mode & 0o222
        or not before.st_mode & 0o111
    ):
        raise PermissionError("Hovi spatial executable is mutable or non-executable")
    digest = hashlib.sha256()
    offset = 0
    while offset < before.st_size:
        block = os.pread(descriptor, min(1 << 20, before.st_size - offset), offset)
        if not block:
            break
        digest.update(block)
        offset += len(block)
    after = os.fstat(descriptor)
    if (
        offset != before.st_size
        or _identity(before) != _identity(after)
        or (
            executable
            and (
                not stat.S_ISREG(after.st_mode)
                or after.st_uid != os.getuid()
                or after.st_nlink != 1
                or after.st_mode & 0o222
                or not after.st_mode & 0o111
            )
        )
    ):
        raise ValueError(f"Hovi spatial {label} changed while being hashed")
    return digest.hexdigest()


def _verify_implementation_file(path: Path, expected_sha256: str) -> None:
    descriptor = open_regular_nofollow(path, "spatial implementation file")
    try:
        if _descriptor_sha256(descriptor, "implementation file") != expected_sha256:
            raise ValueError(f"Hovi spatial implementation changed: {path.name}")
    finally:
        os.close(descriptor)


def _runtime_file_identity(path: Path, label: str) -> tuple[int, str]:
    descriptor = open_regular_nofollow(path, label)
    try:
        result = os.fstat(descriptor)
        return result.st_size, _descriptor_sha256(descriptor, label)
    finally:
        os.close(descriptor)


def _runtime_relative(path: Path, label: str) -> str:
    absolute = _absolute(path)
    try:
        return absolute.relative_to(ASSET_GEN_ROOT).as_posix()
    except ValueError as error:
        raise ValueError(f"Hovi spatial {label} is outside asset-gen") from error


def _numpy_runtime_identity() -> dict[str, Any]:
    """Fingerprint the imported distribution's declared files, not only the lock."""
    numpy = importlib.import_module(_NUMPY_DISTRIBUTION)
    distribution = importlib.metadata.distribution(_NUMPY_DISTRIBUTION)
    distribution_name = distribution.metadata.get("Name")
    imported_path_value = getattr(numpy, "__file__", None)
    imported_version = getattr(numpy, "__version__", None)
    declared_files = distribution.files
    if (
        distribution_name != _NUMPY_DISTRIBUTION
        or distribution.version != _NUMPY_VERSION
        or imported_version != _NUMPY_VERSION
        or not isinstance(imported_path_value, str)
        or not declared_files
    ):
        raise ValueError("Hovi spatial imported NumPy distribution identity drifted")

    environment_root = _absolute(_NUMPY_ENVIRONMENT_ROOT)
    distribution_root = _absolute(Path(distribution.locate_file("")))
    if distribution_root == environment_root:
        raise ValueError("Hovi spatial NumPy distribution root is not site-packages")
    try:
        distribution_root.relative_to(environment_root)
    except ValueError as error:
        raise ValueError(
            "Hovi spatial NumPy is outside the asset-gen uv environment"
        ) from error

    manifest: list[dict[str, Any]] = []
    seen: set[str] = set()
    for declared in declared_files:
        path = _absolute(Path(distribution.locate_file(declared)))
        try:
            path.relative_to(environment_root)
        except ValueError as error:
            raise ValueError(
                "Hovi spatial NumPy declares a file outside its uv environment"
            ) from error
        relative = _runtime_relative(path, "NumPy distribution file")
        if relative in seen:
            raise ValueError("Hovi spatial NumPy distribution repeats a file")
        seen.add(relative)
        size, sha256 = _runtime_file_identity(
            path,
            "NumPy distribution file",
        )
        manifest.append({"path": relative, "bytes": size, "sha256": sha256})
    manifest.sort(key=lambda item: item["path"])

    imported_path = _absolute(Path(imported_path_value))
    imported_relative = _runtime_relative(imported_path, "imported NumPy module")
    imported_size, imported_sha256 = _runtime_file_identity(
        imported_path,
        "imported NumPy module",
    )
    imported_entries = [item for item in manifest if item["path"] == imported_relative]
    if imported_entries != [
        {
            "path": imported_relative,
            "bytes": imported_size,
            "sha256": imported_sha256,
        }
    ]:
        raise ValueError("Hovi spatial imported NumPy module is not distribution-bound")

    return {
        "distribution": _NUMPY_DISTRIBUTION,
        "version": _NUMPY_VERSION,
        "environmentRoot": _runtime_relative(environment_root, "NumPy environment root"),
        "distributionRoot": _runtime_relative(
            distribution_root,
            "NumPy distribution root",
        ),
        "importedModule": _NUMPY_DISTRIBUTION,
        "importedModulePath": imported_relative,
        "importedModuleBytes": imported_size,
        "importedModuleSha256": imported_sha256,
        "declaredFiles": {
            "manifestSchema": "path-bytes-sha256-canonical-json/1.0.0",
            "count": len(manifest),
            "bytes": sum(int(item["bytes"]) for item in manifest),
            "manifestSha256": hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(),
        },
    }


def _selected_numpy_identity(
    authority: SpatialMaterializationAuthority,
) -> Mapping[str, Any]:
    if authority.execution_identity is None:
        raise RuntimeError("Hovi spatial execution has no frozen selection identity")
    parsed = json.loads(authority.execution_identity)
    if not isinstance(parsed, Mapping):
        raise ValueError("Hovi spatial execution selection identity is not an object")
    python = _mapping(parsed.get("pythonExecution"), "Python binding")
    return _mapping(python.get("numpy"), "NumPy runtime binding")


def verify_execution_implementation(
    authority: SpatialMaterializationAuthority,
) -> None:
    """Recheck every selected Python/runtime input before irreversible publication."""
    if authority.execution_selection_sha256 is None or authority.execution_identity is None:
        raise RuntimeError("Hovi spatial execution has no frozen selection")
    for path, expected_sha256 in authority.implementation_files:
        _verify_implementation_file(path, expected_sha256)
    if dict(_selected_numpy_identity(authority)) != _numpy_runtime_identity():
        raise ValueError("Hovi spatial imported NumPy runtime changed")


def load_spatial_execution_selection(
    path: Path,
    *,
    authority: SpatialMaterializationAuthority | None = None,
) -> SpatialMaterializationAuthority:
    """Elevate the pending base authority through one exact committed selection."""
    authority = load_spatial_authority() if authority is None else authority
    if not _EXECUTION_SELECTION_ENABLED:
        raise RuntimeError(
            "Hovi spatial execution selection awaits the reviewed native audit build"
        )
    if authority.native.execution_authorized:
        raise ValueError("Hovi spatial base authority is already execution-enabled")
    path = _absolute(path)
    descriptor = open_regular_nofollow(path, "spatial execution selection")
    try:
        selection_stat = os.fstat(descriptor)
        if (
            selection_stat.st_uid != os.getuid()
            or stat.S_IMODE(selection_stat.st_mode) != 0o400
            or selection_stat.st_nlink != 1
            or selection_stat.st_size > 1 << 20
        ):
            raise PermissionError("Hovi spatial execution selection is not immutable")
        encoded = bytearray()
        while block := os.read(descriptor, 1 << 20):
            encoded.extend(block)
        selection_after = os.fstat(descriptor)
        if _identity(selection_stat) != _identity(selection_after):
            raise ValueError("Hovi spatial execution selection changed while read")
    finally:
        os.close(descriptor)
    payload = bytes(encoded)
    digest = hashlib.sha256(payload).hexdigest()
    expected_path = _absolute(
        DEFAULT_EXECUTION_SELECTION_ROOT / digest[:2] / f"{digest}.json"
    )
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Hovi spatial execution selection is not JSON") from error
    if (
        path != expected_path
        or canonical_json_bytes(parsed) != payload
        or not isinstance(parsed, dict)
        or set(parsed)
        != {
            "schemaVersion",
            "status",
            "authorityConfigSha256",
            "repository",
            "nativeBuild",
            "pythonExecution",
            "publicationRoot",
            "evidenceBoundary",
        }
    ):
        raise ValueError("Hovi spatial execution selection is not content-addressed")

    repository = _mapping(parsed["repository"], "repository")
    native = _mapping(parsed["nativeBuild"], "native build")
    python = _mapping(parsed["pythonExecution"], "Python binding")
    files = _mapping(python.get("files"), "implementation files")
    numpy = _mapping(python.get("numpy"), "NumPy runtime binding")
    boundary = _mapping(parsed["evidenceBoundary"], "evidence boundary")
    realized_numpy = _numpy_runtime_identity()
    if (
        parsed["schemaVersion"] != EXECUTION_SCHEMA
        or parsed["status"] != "execution_authorized"
        or parsed["authorityConfigSha256"] != authority.config_sha256
        or dict(repository)
        != {
            "orchestratorCommit": _ORCHESTRATOR_COMMIT,
            "readerCommit": _READER_COMMIT,
            "nativeSourceTreeSha256": _NATIVE_SOURCE_TREE_SHA256,
        }
        or dict(native)
        != {
            "command": "materialize",
            "reportSchema": authority.native.report_schema,
            "cargoLockSha256": _CARGO_LOCK_SHA256,
            "rustToolchain": "1.94.1",
            "rustcCommit": "e408947bfd200af42db322daf0fadfe7e26d3bd1",
            "targetTriple": "aarch64-apple-darwin",
            "releaseProfile": {
                "codegenUnits": 1,
                "lto": "thin",
                "panic": "abort",
                "strip": "symbols",
            },
            "upstreamManifestSha256": _UPSTREAM_MANIFEST_SHA256,
            "vendorPatchSetSha256": _VENDOR_PATCH_SET_SHA256,
            "executablePath": _EXECUTABLE_RELATIVE,
            "executableBytes": _EXECUTABLE_BYTES,
            "executableSha256": _EXECUTABLE_SHA256,
        }
        or python.get("implementation") != "cpython"
        or python.get("version") != platform.python_version()
        or python.get("platform") != sys.platform
        or python.get("machine") != platform.machine()
        or set(python)
        != {"implementation", "version", "platform", "machine", "files", "numpy"}
        or set(files) != _IMPLEMENTATION_PATHS
        or dict(numpy) != realized_numpy
        or parsed["publicationRoot"]
        != "data/work/hovi-full-scan-spatial-materialization"
        or dict(boundary)
        != {
            "scientificRole": "raw_candidate_spatial_materialization",
            "groundFiltered": False,
            "surfaceClaim": False,
            "analogueQualificationAuthorized": False,
            "targetTruth": False,
            "synthesisAuthorized": False,
        }
    ):
        raise ValueError("Hovi spatial execution selection crossed its authority")

    executable_path = _asset_path(native.get("executablePath"), "native executable")
    executable = open_regular_nofollow(executable_path, "spatial native executable")
    try:
        executable_stat = os.fstat(executable)
        if (
            executable_stat.st_uid != os.getuid()
            or stat.S_IMODE(executable_stat.st_mode) != 0o500
            or executable_stat.st_nlink != 1
            or executable_stat.st_size != _EXECUTABLE_BYTES
            or _descriptor_sha256(
                executable,
                "native executable",
                executable=True,
            )
            != _EXECUTABLE_SHA256
        ):
            raise ValueError("Hovi spatial native executable artifact drifted")
    finally:
        os.close(executable)

    implementation_files: list[tuple[Path, str]] = []
    for relative, expected_sha256 in sorted(files.items()):
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise ValueError("Hovi spatial implementation hash is invalid")
        source_path = _asset_path(relative, "implementation file")
        _verify_implementation_file(source_path, expected_sha256)
        implementation_files.append((source_path, expected_sha256))

    publication_root = _asset_path(parsed["publicationRoot"], "publication root")
    return replace(
        authority,
        native=NativeBinding(
            command=authority.native.command,
            report_schema=authority.native.report_schema,
            execution_authorized=True,
            executable_path=executable_path,
            executable_bytes=_EXECUTABLE_BYTES,
            executable_sha256=_EXECUTABLE_SHA256,
        ),
        publication_root=publication_root,
        execution_selection_sha256=digest,
        execution_identity=payload,
        implementation_files=tuple(implementation_files),
    )


def spatial_materialization_plan(
    authority: SpatialMaterializationAuthority,
    *,
    output_root: Path | None = None,
) -> dict[str, Any]:
    selected_root = authority.publication_root or DEFAULT_SPATIAL_OUTPUT_ROOT
    output_root = _absolute(selected_root if output_root is None else output_root)
    if authority.publication_root is not None and output_root != authority.publication_root:
        raise ValueError("Hovi spatial execution cannot override its publication root")
    probe_path = output_root if output_root.exists() else output_root.parent
    free_bytes = shutil.disk_usage(probe_path).free
    selection_sha256 = authority.execution_selection_sha256
    return {
        "schemaVersion": "hovi-hy-spruce4-spatial-materialization-plan/1.0.0",
        "status": (
            "eligible_for_execution"
            if authority.native.execution_authorized
            else "blocked_pending_frozen_native_command_and_build"
        ),
        "authorityConfigSha256": authority.config_sha256,
        "executionSelectionSha256": selection_sha256,
        "sourceE57Sha256": OBJECT_SHA256,
        "sourceE57Bytes": OBJECT_BYTES,
        "scanCount": len(authority.scans),
        "publisherRecords": sum(scan.publisher_records for scan in authority.scans),
        "outputRoot": output_root.as_posix(),
        "stagingDirectoryName": (
            f".staging-{selection_sha256}"
            if selection_sha256 is not None
            else None
        ),
        "freeBytesObserved": free_bytes,
        "launchFreeBytesRequired": authority.resources.launch_free_bytes,
        "freeReserveBytes": authority.resources.free_reserve_bytes,
        "treeCeilingBytes": authority.resources.tree_ceiling_bytes,
        "nativeMaxOpenFilesArgument": authority.resources.native_max_open_files,
        "nativeMaxOpenOutputFilesReported": (
            authority.resources.native_max_open_output_files
        ),
        "wallSeconds": authority.resources.wall_seconds,
        "cpuSeconds": authority.resources.cpu_seconds,
        "rssBytes": authority.resources.rss_bytes,
        "executionAuthorized": authority.native.execution_authorized,
        "pointsReadByPlan": 0,
        "surfaceClaim": False,
        "synthesisAuthorized": False,
    }


@contextlib.contextmanager
def select_spatial_execution(
    authority: SpatialMaterializationAuthority,
) -> Iterator[SpatialExecutionSelection]:
    """Hold the exact binary/source inodes while exposing a direct-path argv."""
    native = authority.native
    if (
        not native.execution_authorized
        or native.executable_path is None
        or native.executable_bytes is None
        or native.executable_sha256 is None
        or authority.publication_root is None
        or authority.execution_selection_sha256 is None
    ):
        raise RuntimeError("Hovi spatial execution awaits a frozen selection")
    verify_execution_implementation(authority)
    output_root = authority.publication_root
    root_descriptor = _open_or_create_private_root(output_root)
    os.close(root_descriptor)
    free_bytes = shutil.disk_usage(output_root).free
    if free_bytes < LAUNCH_FREE_BYTES:
        raise OSError("Hovi spatial launch requires at least 216 GiB free")
    if free_bytes - TREE_CEILING_BYTES < authority.resources.free_reserve_bytes:
        raise OSError("Hovi spatial launch cannot preserve the 96 GiB reserve")
    staging_root = output_root / f".staging-{authority.execution_selection_sha256}"
    try:
        os.lstat(staging_root)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError("Hovi spatial staging directory already exists")

    executable = open_regular_nofollow(native.executable_path, "spatial native executable")
    held = None
    try:
        executable_stat = os.fstat(executable)
        executable_identity = _identity(executable_stat)
        path_stat = os.lstat(native.executable_path)
        if (
            not stat.S_ISREG(path_stat.st_mode)
            or _identity(path_stat) != executable_identity
            or executable_stat.st_uid != os.getuid()
            or stat.S_IMODE(executable_stat.st_mode) != 0o500
            or executable_stat.st_nlink != 1
            or executable_stat.st_size != native.executable_bytes
            or _descriptor_sha256(
                executable,
                "native executable",
                executable=True,
            )
            != native.executable_sha256
        ):
            raise ValueError("Hovi spatial executable binding drifted")
        metadata_authorization = load_metadata_inspection_authorization()
        binding, held = _open_held_extraction(
            metadata_authorization,
            authority.source_root,
        )
        if (
            binding.object_sha256 != OBJECT_SHA256
            or binding.object_bytes != OBJECT_BYTES
            or sum(scan.publisher_records for scan in authority.scans)
            != PUBLISHER_RECORD_COUNT
        ):
            raise ValueError("Hovi spatial held source binding drifted")
        if not 3 <= held.object_descriptor < authority.resources.native_max_open_files:
            raise RuntimeError("Hovi spatial input descriptor exceeds the NOFILE boundary")
        source_identity = _identity(os.fstat(held.object_descriptor))
        executable_path = _absolute(native.executable_path)
        argv = (
            executable_path.as_posix(),
            native.command,
            "--input-fd",
            str(held.object_descriptor),
            "--output-dir",
            staging_root.as_posix(),
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
            str(PUBLISHER_RECORD_COUNT),
            "--max-json-bytes",
            str(1 << 20),
            "--max-output-bytes",
            str(TREE_CEILING_BYTES),
            "--max-open-files",
            str(authority.resources.native_max_open_files),
        )
        yield SpatialExecutionSelection(
            executable_fd=executable,
            input_fd=held.object_descriptor,
            executable_path=executable_path,
            staging_root=staging_root,
            argv=argv,
            pass_fds=(held.object_descriptor,),
            cwd=output_root,
            environment={"LC_ALL": "C", "LANG": "C", "TZ": "UTC"},
            child_umask=0o077,
            source_identity=source_identity,
            executable_identity=executable_identity,
            free_bytes_at_selection=free_bytes,
        )
    finally:
        if held is not None:
            held.close()
        os.close(executable)
