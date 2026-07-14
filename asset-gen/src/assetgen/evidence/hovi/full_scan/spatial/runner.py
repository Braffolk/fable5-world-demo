"""Supervised long-running execution for Hovi spatial materialization."""
from __future__ import annotations

import ctypes
import errno
import os
import resource
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .artifacts import publish_verified_spatial_staging, verify_spatial_staging
from .authority import SpatialMaterializationAuthority, SpatialResources
from .selection import (
    SpatialExecutionSelection,
    select_spatial_execution,
    verify_execution_implementation,
)


_PROC_PIDTASKINFO = 4
_EXIT_CONFIRM_SECONDS = 0.25


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


@dataclass(frozen=True)
class SupervisionMetrics:
    peak_rss_bytes: int
    minimum_free_bytes: int


@dataclass(frozen=True)
class SpatialMaterializationResult:
    publication_path: Path
    manifest_sha256: str
    native_report_sha256: str
    artifact_count: int
    materialized_records: int
    materialized_bytes: int
    peak_rss_bytes: int
    minimum_free_bytes: int
    elapsed_seconds: float


def _identity(result: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        result.st_dev,
        result.st_ino,
        result.st_size,
        result.st_mtime_ns,
        result.st_ctime_ns,
    )


def _confirm_child_exit(process: subprocess.Popen[bytes]) -> bool:
    try:
        process.wait(timeout=_EXIT_CONFIRM_SECONDS)
    except subprocess.TimeoutExpired:
        return False
    return True


class _DarwinResidentMemoryGuard:
    """Read exact resident bytes through the stable macOS proc_taskinfo ABI."""

    def __init__(self, limit_bytes: int) -> None:
        if sys.platform != "darwin":
            raise RuntimeError("Hovi spatial RSS supervision requires macOS libproc")
        if ctypes.sizeof(_ProcTaskInfo) != 96:
            raise RuntimeError("macOS proc_taskinfo ABI size changed")
        self._limit_bytes = limit_bytes
        self._peak_bytes = 0
        self._libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        self._libproc.proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self._libproc.proc_pidinfo.restype = ctypes.c_int

    @property
    def peak_bytes(self) -> int:
        return self._peak_bytes

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
            if observed_errno == errno.ESRCH and _confirm_child_exit(process):
                return
            raise RuntimeError(
                "macOS could not supervise Hovi spatial RSS "
                f"(proc_pidinfo returned {returned}, errno {observed_errno})"
            )
        self._peak_bytes = max(self._peak_bytes, int(task.resident_size))
        if task.resident_size > self._limit_bytes:
            raise MemoryError(
                "Hovi spatial materializer exceeded its resident-memory limit: "
                f"{task.resident_size} > {self._limit_bytes} bytes"
            )


def _limit_child(resources: SpatialResources, child_umask: int) -> None:
    def apply(kind: int, requested: int) -> None:
        _, inherited_hard = resource.getrlimit(kind)
        effective = (
            requested
            if inherited_hard == resource.RLIM_INFINITY
            else min(requested, inherited_hard)
        )
        resource.setrlimit(kind, (effective, effective))

    os.umask(child_umask)
    apply(resource.RLIMIT_CORE, 0)
    apply(resource.RLIMIT_NOFILE, resources.native_max_open_files)
    apply(resource.RLIMIT_FSIZE, resources.tree_ceiling_bytes)
    apply(resource.RLIMIT_CPU, resources.cpu_seconds)


def _terminate_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError as error:
            if error.errno in (errno.ESRCH, errno.EPERM) and _confirm_child_exit(
                process
            ):
                return
            raise
    process.wait()


def _capture_bounded(
    process: subprocess.Popen[bytes],
    resources: SpatialResources,
    output_root: Path,
    started: float,
) -> tuple[bytes, bytes, SupervisionMetrics]:
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("Hovi spatial capture pipes were not created")
    streams = {
        process.stdout.fileno(): ("stdout", resources.stdout_bytes),
        process.stderr.fileno(): ("stderr", resources.stderr_bytes),
    }
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    selector = selectors.DefaultSelector()
    memory_guard = _DarwinResidentMemoryGuard(resources.rss_bytes)
    minimum_free = shutil.disk_usage(output_root).free
    try:
        for descriptor in streams:
            os.set_blocking(descriptor, False)
            selector.register(descriptor, selectors.EVENT_READ)
        while selector.get_map():
            memory_guard.enforce(process)
            free_bytes = shutil.disk_usage(output_root).free
            minimum_free = min(minimum_free, free_bytes)
            if free_bytes < resources.free_reserve_bytes:
                raise OSError("Hovi spatial materialization crossed the 96 GiB reserve")
            remaining = resources.wall_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Hovi spatial materialization exceeded 24h wall time")
            for key, _ in selector.select(
                min(remaining, resources.supervision_sample_seconds)
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
                    raise RuntimeError(
                        f"Hovi spatial materializer exceeded {label} limit"
                    )
        remaining = resources.wall_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Hovi spatial materialization exceeded 24h wall time")
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
        detail = bytes(captured["stderr"]).decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Hovi spatial materializer failed with exit code {return_code}: {detail}"
        )
    return (
        bytes(captured["stdout"]),
        bytes(captured["stderr"]),
        SupervisionMetrics(
            peak_rss_bytes=memory_guard.peak_bytes,
            minimum_free_bytes=minimum_free,
        ),
    )


def _verify_held_inputs(selection: SpatialExecutionSelection) -> None:
    if _identity(os.fstat(selection.input_fd)) != selection.source_identity:
        raise ValueError("held Hovi E57 object changed during materialization")
    executable_stat = os.fstat(selection.executable_fd)
    path_stat = os.lstat(selection.executable_path)
    if (
        _identity(executable_stat) != selection.executable_identity
        or _identity(path_stat) != selection.executable_identity
        or not stat.S_ISREG(path_stat.st_mode)
    ):
        raise ValueError("Hovi spatial executable changed during materialization")


def run_spatial_materialization(
    authority: SpatialMaterializationAuthority,
) -> SpatialMaterializationResult:
    """Run, independently verify, and atomically publish one exact full decode."""
    if not authority.native.execution_authorized:
        raise RuntimeError("Hovi spatial execution awaits a frozen selection")
    verify_execution_implementation(authority)
    started = time.monotonic()
    process: subprocess.Popen[bytes] | None = None
    with select_spatial_execution(authority) as selection:
        try:
            process = subprocess.Popen(
                selection.argv,
                executable=selection.executable_path,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                close_fds=True,
                pass_fds=selection.pass_fds,
                cwd=selection.cwd,
                env=selection.environment,
                start_new_session=True,
                preexec_fn=lambda: _limit_child(
                    authority.resources,
                    selection.child_umask,
                ),
            )
            stdout, stderr, metrics = _capture_bounded(
                process,
                authority.resources,
                selection.cwd,
                started,
            )
            if stderr:
                raise RuntimeError("Hovi spatial materializer emitted unexpected stderr")
            _verify_held_inputs(selection)
            verify_execution_implementation(authority)
            if time.monotonic() - started >= authority.resources.wall_seconds:
                raise TimeoutError("Hovi spatial materialization exceeded 24h wall time")
            verified = verify_spatial_staging(
                authority,
                selection.staging_root,
                stdout,
            )
            _verify_held_inputs(selection)
            verify_execution_implementation(authority)
            if time.monotonic() - started >= authority.resources.wall_seconds:
                raise TimeoutError("Hovi spatial verification exceeded 24h wall time")
            publication_path = publish_verified_spatial_staging(
                authority,
                verified,
                selection.cwd,
            )
            return SpatialMaterializationResult(
                publication_path=publication_path,
                manifest_sha256=verified.manifest_sha256,
                native_report_sha256=verified.native_report_sha256,
                artifact_count=len(verified.artifacts),
                materialized_records=verified.record_count,
                materialized_bytes=verified.artifact_bytes,
                peak_rss_bytes=metrics.peak_rss_bytes,
                minimum_free_bytes=metrics.minimum_free_bytes,
                elapsed_seconds=time.monotonic() - started,
            )
        finally:
            if process is not None and process.poll() is None:
                _terminate_group(process)
