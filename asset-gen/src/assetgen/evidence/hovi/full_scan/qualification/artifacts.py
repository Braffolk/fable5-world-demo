"""Private staging and content-addressed publication for the bounded probe."""
from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from .....config import DATA_WORK
from ..extract import (
    _atomic_rename_noreplace_at,
    _fsync_directory,
    _open_optional_regular_at,
    _open_or_create_private_directory_at,
    _open_or_create_private_root,
    _publication_identity,
    _unlink_bound_regular_at,
)


DEFAULT_PROBE_OUTPUT_ROOT = DATA_WORK / "hovi-full-scan-point-probe"


@dataclass
class StagedRecordOutput:
    root: int
    descriptor: int
    name: str
    path: Path

    def close(self, *, unlink: bool) -> None:
        try:
            if unlink:
                try:
                    _unlink_bound_regular_at(
                        self.root,
                        self.name,
                        "staged point-probe record output",
                        self.descriptor,
                    )
                except FileNotFoundError:
                    pass
        finally:
            os.close(self.descriptor)
            os.close(self.root)


def _validate_private_regular(
    directory: int,
    name: str,
    descriptor: int,
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
        raise PermissionError("point-probe artifact is not the held private inode")
    return result


def _digest_descriptor(descriptor: int, expected_bytes: int) -> str:
    before = os.fstat(descriptor)
    if before.st_size != expected_bytes:
        raise ValueError("point-probe record output has the wrong byte count")
    digest = hashlib.sha256()
    total = 0
    offset = 0
    while total < expected_bytes:
        block = os.pread(descriptor, min(1 << 20, expected_bytes - total), offset)
        if not block:
            break
        digest.update(block)
        total += len(block)
        offset += len(block)
    after = os.fstat(descriptor)
    if (
        total != expected_bytes
        or _publication_identity(before) != _publication_identity(after)
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
    ):
        raise ValueError("point-probe record output changed during verification")
    return digest.hexdigest()


def create_staged_record_output(
    authority_sha256: str,
    output_root: Path = DEFAULT_PROBE_OUTPUT_ROOT,
) -> StagedRecordOutput:
    root = _open_or_create_private_root(output_root)
    name = f".{authority_sha256}.{os.getpid()}.records.tmp"
    try:
        descriptor = os.open(
            name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=root,
        )
    except Exception:
        os.close(root)
        raise
    try:
        os.fchmod(descriptor, 0o600)
        _validate_private_regular(root, name, descriptor, {0o600})
        _fsync_directory(root)
        return StagedRecordOutput(
            root=root,
            descriptor=descriptor,
            name=name,
            path=Path(os.path.abspath(os.fspath(output_root))),
        )
    except Exception:
        try:
            _unlink_bound_regular_at(
                root,
                name,
                "failed staged point-probe record output",
                descriptor,
            )
        finally:
            os.close(descriptor)
        os.close(root)
        raise


def _hash_directory(root: int, kind: str, digest: str) -> tuple[int, int, int]:
    kind_directory = _open_or_create_private_directory_at(root, kind, kind)
    sha_directory = prefix_directory = -1
    try:
        sha_directory = _open_or_create_private_directory_at(
            kind_directory,
            "sha256",
            f"{kind} sha256",
        )
        prefix_directory = _open_or_create_private_directory_at(
            sha_directory,
            digest[:2],
            f"{kind} hash prefix",
        )
        return kind_directory, sha_directory, prefix_directory
    except Exception:
        for descriptor in (prefix_directory, sha_directory, kind_directory):
            if descriptor >= 0:
                os.close(descriptor)
        raise


def _verify_existing(
    directory: int,
    name: str,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    descriptor = _open_optional_regular_at(directory, name, name)
    if descriptor is None:
        raise FileNotFoundError(name)
    try:
        _validate_private_regular(directory, name, descriptor, {0o400})
        if _digest_descriptor(descriptor, expected_bytes) != expected_sha256:
            raise ValueError("point-probe content-address collision")
    finally:
        os.close(descriptor)


def publish_record_output(
    staged: StagedRecordOutput,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> tuple[Path, str]:
    _validate_private_regular(staged.root, staged.name, staged.descriptor, {0o600})
    digest = _digest_descriptor(staged.descriptor, expected_bytes)
    if digest != expected_sha256:
        raise ValueError("point-probe record digest changed before publication")
    os.fchmod(staged.descriptor, 0o400)
    os.fsync(staged.descriptor)
    _validate_private_regular(staged.root, staged.name, staged.descriptor, {0o400})
    directories = _hash_directory(staged.root, "records", digest)
    prefix = directories[-1]
    final_name = f"{digest}.bin"
    moved = False
    try:
        try:
            _atomic_rename_noreplace_at(
                staged.root,
                staged.name,
                prefix,
                final_name,
            )
            moved = True
        except FileExistsError:
            _verify_existing(prefix, final_name, expected_bytes, digest)
            _unlink_bound_regular_at(
                staged.root,
                staged.name,
                "duplicate staged point-probe records",
                staged.descriptor,
            )
        _verify_existing(prefix, final_name, expected_bytes, digest)
        _fsync_directory(prefix)
        return (
            staged.path / "records" / "sha256" / digest[:2] / final_name,
            digest,
        )
    finally:
        for descriptor in reversed(directories):
            os.close(descriptor)
        if moved:
            staged.name = final_name


def verify_staged_record_output(
    staged: StagedRecordOutput,
    *,
    expected_bytes: int,
) -> str:
    _validate_private_regular(staged.root, staged.name, staged.descriptor, {0o600})
    return _digest_descriptor(staged.descriptor, expected_bytes)


def publish_report(
    root: int,
    output_root: Path,
    payload: bytes,
) -> tuple[Path, str]:
    digest = hashlib.sha256(payload).hexdigest()
    directories = _hash_directory(root, "reports", digest)
    prefix = directories[-1]
    final_name = f"{digest}.json"
    temporary_name = f".{digest}.{os.getpid()}.json.tmp"
    temporary = -1
    try:
        existing = _open_optional_regular_at(prefix, final_name, final_name)
        if existing is not None:
            os.close(existing)
            _verify_existing(prefix, final_name, len(payload), digest)
        else:
            temporary = os.open(
                temporary_name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=prefix,
            )
            view = memoryview(payload)
            while view:
                written = os.write(temporary, view)
                if written <= 0:
                    raise OSError("short write while publishing point-probe report")
                view = view[written:]
            _validate_private_regular(prefix, temporary_name, temporary, {0o600})
            os.fchmod(temporary, 0o400)
            os.fsync(temporary)
            try:
                _atomic_rename_noreplace_at(
                    prefix,
                    temporary_name,
                    prefix,
                    final_name,
                )
            except FileExistsError:
                _verify_existing(prefix, final_name, len(payload), digest)
                _unlink_bound_regular_at(
                    prefix,
                    temporary_name,
                    "duplicate staged point-probe report",
                    temporary,
                )
            _verify_existing(prefix, final_name, len(payload), digest)
        _fsync_directory(prefix)
        return (
            Path(os.path.abspath(os.fspath(output_root)))
            / "reports"
            / "sha256"
            / digest[:2]
            / final_name,
            digest,
        )
    finally:
        if temporary >= 0:
            try:
                _unlink_bound_regular_at(
                    prefix,
                    temporary_name,
                    "uncommitted point-probe report",
                    temporary,
                )
            except FileNotFoundError:
                pass
            os.close(temporary)
        for descriptor in reversed(directories):
            os.close(descriptor)
