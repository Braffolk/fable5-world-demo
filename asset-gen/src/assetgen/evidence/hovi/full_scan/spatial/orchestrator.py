"""Fail-closed execution selection for Hovi spatial materialization."""
from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .....config import DATA_WORK
from ..authorization import load_metadata_inspection_authorization, open_regular_nofollow
from ..extract import _open_or_create_private_root
from ..inventory import _open_held_extraction
from .authority import (
    LAUNCH_FREE_BYTES,
    OBJECT_BYTES,
    OBJECT_SHA256,
    PUBLISHER_RECORD_COUNT,
    TREE_CEILING_BYTES,
    SpatialMaterializationAuthority,
)


DEFAULT_SPATIAL_OUTPUT_ROOT = DATA_WORK / "hovi-full-scan-spatial-materialization"


@dataclass(frozen=True)
class SpatialExecutionSelection:
    executable_fd: int
    input_fd: int
    executable_fd_path: str
    staging_root: Path
    argv: tuple[str, ...]
    pass_fds: tuple[int, ...]
    cwd: Path
    environment: dict[str, str]
    child_umask: int
    stdout_bytes: int
    stderr_bytes: int
    free_bytes_at_selection: int


def spatial_materialization_plan(
    authority: SpatialMaterializationAuthority,
    *,
    output_root: Path = DEFAULT_SPATIAL_OUTPUT_ROOT,
) -> dict:
    output_root = Path(os.path.abspath(os.fspath(output_root)))
    probe_path = output_root if output_root.exists() else output_root.parent
    free_bytes = shutil.disk_usage(probe_path).free
    return {
        "schemaVersion": "hovi-hy-spruce4-spatial-materialization-plan/1.0.0",
        "status": (
            "eligible_for_execution_selection"
            if authority.native.execution_authorized
            else "blocked_pending_frozen_native_command_and_build"
        ),
        "authorityConfigSha256": authority.config_sha256,
        "sourceE57Sha256": OBJECT_SHA256,
        "sourceE57Bytes": OBJECT_BYTES,
        "scanCount": len(authority.scans),
        "publisherRecords": sum(scan.publisher_records for scan in authority.scans),
        "outputRoot": output_root.as_posix(),
        "stagingDirectoryName": f".staging-{authority.config_sha256}",
        "freeBytesObserved": free_bytes,
        "launchFreeBytesRequired": authority.resources.launch_free_bytes,
        "freeReserveBytes": authority.resources.free_reserve_bytes,
        "treeCeilingBytes": authority.resources.tree_ceiling_bytes,
        "nativeMaxOpenFilesArgument": authority.resources.native_max_open_files,
        "nativeMaxOpenOutputFilesReported": (
            authority.resources.native_max_open_output_files
        ),
        "executionAuthorized": authority.native.execution_authorized,
        "pointsReadByPlan": 0,
        "surfaceClaim": False,
        "synthesisAuthorized": False,
    }


def _descriptor_sha256(descriptor: int) -> str:
    expected = os.fstat(descriptor).st_size
    digest = hashlib.sha256()
    offset = 0
    while offset < expected:
        block = os.pread(descriptor, min(1 << 20, expected - offset), offset)
        if not block:
            break
        digest.update(block)
        offset += len(block)
    if offset != expected:
        raise ValueError("Hovi spatial executable changed while being hashed")
    return digest.hexdigest()


@contextlib.contextmanager
def select_spatial_execution(
    authority: SpatialMaterializationAuthority,
    *,
    output_root: Path = DEFAULT_SPATIAL_OUTPUT_ROOT,
) -> Iterator[SpatialExecutionSelection]:
    """Hold exact source/build FDs for a caller; pending authority always refuses."""
    native = authority.native
    if (
        not native.execution_authorized
        or native.executable_path is None
        or native.executable_bytes is None
        or native.executable_sha256 is None
    ):
        raise RuntimeError(
            "Hovi spatial execution awaits a frozen native command and build identity"
        )
    output_root = Path(os.path.abspath(os.fspath(output_root)))
    root_descriptor = _open_or_create_private_root(output_root)
    os.close(root_descriptor)
    free_bytes = shutil.disk_usage(output_root).free
    if free_bytes < LAUNCH_FREE_BYTES:
        raise OSError("Hovi spatial launch requires at least 216 GiB free")
    if free_bytes - TREE_CEILING_BYTES < authority.resources.free_reserve_bytes:
        raise OSError("Hovi spatial launch cannot preserve the 96 GiB reserve")
    staging_root = output_root / f".staging-{authority.config_sha256}"
    try:
        os.lstat(staging_root)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError("Hovi spatial staging directory already exists")

    executable = open_regular_nofollow(
        native.executable_path,
        "Hovi spatial native executable",
    )
    held = None
    try:
        executable_stat = os.fstat(executable)
        if (
            executable_stat.st_size != native.executable_bytes
            or _descriptor_sha256(executable) != native.executable_sha256
        ):
            raise ValueError("Hovi spatial native executable binding drifted")
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
        if not 3 <= held.object_descriptor <= 1024:
            raise RuntimeError("Hovi spatial input descriptor exceeds native hard range")
        executable_fd_path = f"/dev/fd/{executable}"
        argv = (
            executable_fd_path,
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
            executable_fd_path=executable_fd_path,
            staging_root=staging_root,
            argv=argv,
            pass_fds=(executable, held.object_descriptor),
            cwd=output_root,
            environment={"LC_ALL": "C", "LANG": "C", "TZ": "UTC"},
            child_umask=0o077,
            stdout_bytes=authority.resources.stdout_bytes,
            stderr_bytes=authority.resources.stderr_bytes,
            free_bytes_at_selection=free_bytes,
        )
    finally:
        if held is not None:
            held.close()
        os.close(executable)
