"""Held-descriptor identity binding for comparison inputs."""
from __future__ import annotations

import hashlib
import os
import stat
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

from ..authorization import open_regular_nofollow


@dataclass(frozen=True)
class FileIdentity:
    device: int
    inode: int
    bytes: int
    mode: int
    uid: int
    gid: int
    links: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> "FileIdentity":
        return cls(
            device=value.st_dev,
            inode=value.st_ino,
            bytes=value.st_size,
            mode=stat.S_IMODE(value.st_mode),
            uid=value.st_uid,
            gid=value.st_gid,
            links=value.st_nlink,
            modified_ns=value.st_mtime_ns,
            changed_ns=value.st_ctime_ns,
        )


class HeldFileBinding:
    """Hash one regular inode, hold it open, then reject any identity drift."""

    def __init__(
        self,
        path: Path,
        *,
        expected_sha256: str,
        expected_bytes: int | None,
        expected_mode: int,
        label: str,
    ):
        if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
            raise ValueError(f"{label} SHA-256 must be 64 lowercase hexadecimal characters")
        self.path = Path(os.path.abspath(os.fspath(path)))
        self.label = label
        self._descriptor = open_regular_nofollow(self.path, label)
        try:
            before = os.fstat(self._descriptor)
            identity = FileIdentity.from_stat(before)
            if (
                identity.uid != os.getuid()
                or identity.links != 1
                or identity.mode != expected_mode
                or (expected_bytes is not None and identity.bytes != expected_bytes)
            ):
                raise PermissionError(f"{label} inode metadata differs from its frozen binding")
            digest = hashlib.sha256()
            while block := os.read(self._descriptor, 8 << 20):
                digest.update(block)
            after_hash = FileIdentity.from_stat(os.fstat(self._descriptor))
            if after_hash != identity or digest.hexdigest() != expected_sha256:
                raise ValueError(f"{label} changed while hashing or differs from its frozen digest")
            os.lseek(self._descriptor, 0, os.SEEK_SET)
        except BaseException:
            os.close(self._descriptor)
            self._descriptor = -1
            raise
        self.identity = identity
        self.sha256 = expected_sha256

    @contextmanager
    def binary_stream(self) -> Iterator[BinaryIO]:
        self._require_open()
        os.lseek(self._descriptor, 0, os.SEEK_SET)
        duplicate = os.dup(self._descriptor)
        with os.fdopen(duplicate, "rb", buffering=0) as stream:
            yield stream

    def verify_unchanged(self) -> None:
        self._require_open()
        if FileIdentity.from_stat(os.fstat(self._descriptor)) != self.identity:
            raise ValueError(f"{self.label} held inode changed during comparison")
        current = open_regular_nofollow(self.path, self.label)
        try:
            if FileIdentity.from_stat(os.fstat(current)) != self.identity:
                raise ValueError(f"{self.label} path no longer resolves to the held inode")
        finally:
            os.close(current)

    def document(self) -> dict[str, object]:
        return {
            "path": os.fspath(self.path),
            "sha256": self.sha256,
            "identity": asdict(self.identity),
            "binding": "nofollow-held-descriptor-sha256-and-stat-before-after",
        }

    def close(self, *, verify: bool = True) -> None:
        if self._descriptor >= 0:
            try:
                if verify:
                    self.verify_unchanged()
            finally:
                os.close(self._descriptor)
                self._descriptor = -1

    def __enter__(self) -> "HeldFileBinding":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._descriptor < 0:
            raise RuntimeError(f"{self.label} binding is closed")
