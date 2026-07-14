"""Crash-recoverable extraction of the authorized HY_SPRUCE4 E57 member."""
from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import stat
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable

from ....config import DATA_IN
from .authorization import (
    DEFAULT_AUTHORIZATION,
    MetadataInspectionAuthorization,
    canonical_json_bytes,
    load_metadata_inspection_authorization,
    open_regular_nofollow,
)


DEFAULT_OUTPUT_ROOT = DATA_IN / "evidence" / "hovi-full-metadata"
PREPARED_SCHEMA = "hovi-hy-spruce4-full-extraction-prepared/1.0.0"
REPORT_SCHEMA = "hovi-hy-spruce4-full-extraction/1.0.0"
_PART_NAME_SUFFIX = ".e57.part"
_PREPARED_NAME = "prepared.json"
_RECEIPT_NAME = "extraction.json"
_LOCK_NAME = ".transaction.lock"


@dataclass(frozen=True)
class StreamDigest:
    bytes: int
    crc32: int
    sha256: str


@dataclass(frozen=True)
class ExtractionResult:
    transaction_id: str
    output_path: Path
    output_sha256: str
    receipt_path: Path


def _normalize_output_root(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _directory_flags() -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError(
            "Hovi extraction requires O_DIRECTORY and O_NOFOLLOW"
        )
    flags = os.O_RDONLY
    return flags | os.O_DIRECTORY | os.O_NOFOLLOW


def _simple_name(name: str) -> str:
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError(f"unsafe extraction namespace component: {name!r}")
    return name


def _require_private_directory(descriptor: int, label: str) -> None:
    result = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(result.st_mode)
        or result.st_uid != os.getuid()
        or stat.S_IMODE(result.st_mode) != 0o700
    ):
        raise PermissionError(
            f"{label} must be a current-user-owned private 0700 directory"
        )


def _open_or_create_private_directory_at(parent: int, name: str, label: str) -> int:
    name = _simple_name(name)
    try:
        descriptor = os.open(name, _directory_flags(), dir_fd=parent)
    except FileNotFoundError:
        created = False
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent)
            created = True
        except FileExistsError:
            pass
        if created:
            os.fsync(parent)
        descriptor = os.open(name, _directory_flags(), dir_fd=parent)
        if created:
            os.fchmod(descriptor, 0o700)
            os.fsync(descriptor)
    try:
        _require_private_directory(descriptor, label)
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _open_existing_absolute_directory(path: Path) -> int:
    absolute = _normalize_output_root(path)
    descriptor = os.open(absolute.anchor, _directory_flags())
    try:
        for part in absolute.parts[1:]:
            next_descriptor = os.open(part, _directory_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _open_or_create_private_root(path: Path) -> int:
    absolute = _normalize_output_root(path)
    if absolute == Path(absolute.anchor):
        raise ValueError("Hovi extraction output root cannot be the filesystem root")
    parent = _open_existing_absolute_directory(absolute.parent)
    try:
        return _open_or_create_private_directory_at(
            parent,
            absolute.name,
            "Hovi extraction output root",
        )
    finally:
        os.close(parent)


def _acquire_transaction_lock(directory: int) -> int:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    try:
        descriptor = os.open(_LOCK_NAME, flags, 0o600, dir_fd=directory)
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        os.fsync(directory)
    except FileExistsError:
        descriptor = os.open(
            _LOCK_NAME,
            os.O_RDWR | os.O_NOFOLLOW,
            dir_fd=directory,
        )
    result = os.fstat(descriptor)
    namespace = os.stat(_LOCK_NAME, dir_fd=directory, follow_symlinks=False)
    if (
        not stat.S_ISREG(result.st_mode)
        or result.st_uid != os.getuid()
        or stat.S_IMODE(result.st_mode) != 0o600
        or result.st_nlink != 1
        or _publication_identity(result) != _publication_identity(namespace)
    ):
        os.close(descriptor)
        raise PermissionError(
            "Hovi extraction transaction lock must be a private current-user regular file"
        )
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    namespace = os.stat(_LOCK_NAME, dir_fd=directory, follow_symlinks=False)
    if _publication_identity(os.fstat(descriptor)) != _publication_identity(namespace):
        os.close(descriptor)
        raise ValueError("Hovi extraction transaction lock namespace changed")
    return descriptor


def _exists_at(directory: int, name: str) -> bool:
    name = _simple_name(name)
    try:
        os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _open_regular_at(directory: int, name: str, label: str) -> int:
    name = _simple_name(name)
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("Hovi extraction requires O_NOFOLLOW")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    descriptor = os.open(name, flags, dir_fd=directory)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError(f"{label} is not a regular file")
    return descriptor


def _open_optional_regular_at(directory: int, name: str, label: str) -> int | None:
    try:
        return _open_regular_at(directory, name, label)
    except FileNotFoundError:
        return None


def _open_exclusive_at(directory: int, name: str) -> BinaryIO:
    name = _simple_name(name)
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("Hovi extraction requires O_NOFOLLOW")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    descriptor = os.open(name, flags, 0o600, dir_fd=directory)
    try:
        os.fchmod(descriptor, 0o600)
        result = os.fstat(descriptor)
        if (
            not stat.S_ISREG(result.st_mode)
            or result.st_uid != os.getuid()
            or stat.S_IMODE(result.st_mode) != 0o600
            or result.st_nlink != 1
        ):
            raise PermissionError(
                "exclusive extraction file must be private, owned, and singly linked"
            )
    except Exception:
        os.close(descriptor)
        raise
    return os.fdopen(descriptor, "wb", buffering=0)


def _write_all(target: BinaryIO, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = target.write(view)
        if written is None or written <= 0:
            raise OSError("short write while materializing authorized Hovi evidence")
        view = view[written:]


def _stable_identity(result: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        result.st_dev,
        result.st_ino,
        result.st_size,
        result.st_mtime_ns,
        result.st_ctime_ns,
    )


def _publication_identity(result: os.stat_result) -> tuple[int, int, int]:
    return result.st_dev, result.st_ino, result.st_size


def _digest_descriptor(descriptor: int, block_bytes: int) -> StreamDigest:
    os.lseek(descriptor, 0, os.SEEK_SET)
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("digest source is not a regular file")
    digest = hashlib.sha256()
    crc = 0
    total = 0
    while block := os.read(descriptor, block_bytes):
        total += len(block)
        digest.update(block)
        crc = zlib.crc32(block, crc)
    after = os.fstat(descriptor)
    if _stable_identity(after) != _stable_identity(before) or total != before.st_size:
        raise ValueError("source inode changed while it was being verified")
    return StreamDigest(total, crc & 0xFFFFFFFF, digest.hexdigest())


def _read_bytes_at(directory: int, name: str, label: str) -> bytes:
    descriptor = _open_regular_at(directory, name, label)
    try:
        return _read_descriptor(descriptor, label)
    finally:
        os.close(descriptor)


def _read_descriptor(descriptor: int, label: str) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    expected = os.fstat(descriptor)
    chunks: list[bytes] = []
    while block := os.read(descriptor, 1 << 20):
        chunks.append(block)
    actual = os.fstat(descriptor)
    payload = b"".join(chunks)
    if (
        _stable_identity(actual) != _stable_identity(expected)
        or len(payload) != expected.st_size
    ):
        raise ValueError(f"{label} changed while being read")
    return payload


def _fsync_directory(descriptor: int) -> None:
    os.fsync(descriptor)


def _atomic_rename_noreplace_at(
    source_directory: int,
    source_name: str,
    destination_directory: int,
    destination_name: str,
) -> None:
    """Publish with macOS atomic no-replace rename using held directory FDs."""
    old = os.fsencode(_simple_name(source_name))
    new = os.fsencode(_simple_name(destination_name))
    library = ctypes.CDLL(None, use_errno=True)
    if not hasattr(library, "renameatx_np"):
        raise RuntimeError("macOS fd-relative atomic no-replace rename is unavailable")
    operation = library.renameatx_np
    operation.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    operation.restype = ctypes.c_int
    result = operation(
        source_directory,
        old,
        destination_directory,
        new,
        0x00000004,
    )  # RENAME_EXCL
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(destination_name)
        raise OSError(error, os.strerror(error), destination_name)
    _fsync_directory(destination_directory)
    _fsync_directory(source_directory)


def _publish_record_at(directory: int, name: str, document: dict) -> bytes:
    payload = canonical_json_bytes(document)
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    temporary = f".{_simple_name(name)}.{payload_sha256}.part"
    final_descriptor = _open_optional_regular_at(directory, name, name)
    if final_descriptor is not None:
        try:
            result = os.fstat(final_descriptor)
            namespace = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if (
                result.st_uid != os.getuid()
                or stat.S_IMODE(result.st_mode) not in {0o400, 0o600}
                or result.st_nlink != 1
                or _publication_identity(result) != _publication_identity(namespace)
                or _read_descriptor(final_descriptor, name) != payload
            ):
                raise ValueError(f"immutable transaction record differs: {name}")
            if stat.S_IMODE(result.st_mode) == 0o600:
                os.fchmod(final_descriptor, 0o400)
                os.fsync(final_descriptor)
                _fsync_directory(directory)
            result = os.fstat(final_descriptor)
            namespace = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if (
                stat.S_IMODE(result.st_mode) != 0o400
                or _publication_identity(result) != _publication_identity(namespace)
            ):
                raise ValueError(f"immutable transaction record is mutable: {name}")
        finally:
            os.close(final_descriptor)
        _fsync_directory(directory)
        return payload
    temporary_descriptor = _open_optional_regular_at(
        directory,
        temporary,
        temporary,
    )
    if temporary_descriptor is not None:
        try:
            if _read_descriptor(temporary_descriptor, temporary) != payload:
                # An uncommitted temporary may be truncated by a crash. Only the
                # no-replace final name is immutable transaction state.
                _unlink_bound_regular_at(
                    directory,
                    temporary,
                    temporary,
                    temporary_descriptor,
                )
            else:
                result = os.fstat(temporary_descriptor)
                if (
                    result.st_uid != os.getuid()
                    or stat.S_IMODE(result.st_mode) not in {0o400, 0o600}
                    or result.st_nlink != 1
                ):
                    raise PermissionError(
                        "transaction record temporary is not private and singly linked"
                    )
                os.fchmod(temporary_descriptor, 0o400)
                os.fsync(temporary_descriptor)
        finally:
            os.close(temporary_descriptor)
    if not _exists_at(directory, temporary):
        with _open_exclusive_at(directory, temporary) as target:
            _write_all(target, payload)
            target.flush()
            os.fchmod(target.fileno(), 0o400)
            os.fsync(target.fileno())
    try:
        _atomic_rename_noreplace_at(directory, temporary, directory, name)
    except FileExistsError:
        final_descriptor = _open_regular_at(directory, name, name)
        try:
            result = os.fstat(final_descriptor)
            namespace = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if (
                result.st_uid != os.getuid()
                or stat.S_IMODE(result.st_mode) != 0o400
                or result.st_nlink != 1
                or _publication_identity(result) != _publication_identity(namespace)
                or _read_descriptor(final_descriptor, name) != payload
            ):
                raise ValueError(f"immutable transaction record race differs: {name}")
        finally:
            os.close(final_descriptor)
        temporary_descriptor = _open_optional_regular_at(
            directory,
            temporary,
            temporary,
        )
        if temporary_descriptor is not None:
            try:
                _unlink_bound_regular_at(
                    directory,
                    temporary,
                    temporary,
                    temporary_descriptor,
                )
            finally:
                os.close(temporary_descriptor)
    final_descriptor = _open_regular_at(directory, name, name)
    try:
        result = os.fstat(final_descriptor)
        namespace = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if (
            result.st_uid != os.getuid()
            or stat.S_IMODE(result.st_mode) != 0o400
            or result.st_nlink != 1
            or _publication_identity(result) != _publication_identity(namespace)
            or _read_descriptor(final_descriptor, name) != payload
        ):
            raise ValueError(f"immutable transaction record publication failed: {name}")
    finally:
        os.close(final_descriptor)
    _fsync_directory(directory)
    return payload


def _unlink_bound_regular_at(
    directory: int,
    name: str,
    label: str,
    descriptor: int | None = None,
) -> None:
    held = descriptor if descriptor is not None else _open_regular_at(directory, name, label)
    try:
        before = os.fstat(held)
        namespace = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if _publication_identity(before) != _publication_identity(namespace):
            raise ValueError(f"{label} namespace changed during recovery")
        os.unlink(name, dir_fd=directory)
        _fsync_directory(directory)
    finally:
        if descriptor is None:
            os.close(held)


def _member_tuple(authorization: MetadataInspectionAuthorization) -> dict:
    member = authorization.member
    return {
        "name": member.name,
        "uncompressedBytes": member.uncompressed_bytes,
        "compressedBytes": member.compressed_bytes,
        "compressionMethod": member.compression_method,
        "flagBits": member.flag_bits,
        "crc32": f"{member.crc32:08x}",
        "localHeaderOffset": member.local_header_offset,
        "createSystem": member.create_system,
        "externalAttr": member.external_attr,
    }


def _extraction_policy_tuple(
    authorization: MetadataInspectionAuthorization,
) -> dict:
    return {
        "streamOnly": True,
        "blockBytes": authorization.block_bytes,
        "partialAddress": "authorization_config_sha256",
        "partialSuffix": _PART_NAME_SUFFIX,
        "resume": "restart_only",
        "resumeReason": (
            "A raw deflate ZIP member cannot resume at an arbitrary output byte "
            "without persisted bit-exact decompressor state."
        ),
        "interruptedPartialDisposition": "discard_and_restart_from_zero",
        "verificationBeforePublish": [
            "uncompressed_size",
            "zip_crc32",
            "stream_sha256",
            "independent_part_sha256",
        ],
        "preparedRecord": "transactions/<authorization-sha256>/prepared.json",
        "receiptRecord": "transactions/<authorization-sha256>/extraction.json",
        "recovery": "idempotent_prepared_object_receipt_state_machine",
        "transactionLock": "exclusive_advisory_lock_held_through_receipt",
        "writableDirectoryPolicy": "current_user_owned_0700",
        "preparedWithoutPayload": "clear_intent_and_restart_from_zero",
        "objectCollision": "accept_only_after_full_digest_equality",
        "publish": "atomic_noreplace_rename",
        "immutableOutput": "objects/sha256/<sha256-prefix>/<sha256>.e57",
    }


def _prepared_document(
    authorization: MetadataInspectionAuthorization,
    digest: StreamDigest,
    part_name: str,
) -> dict:
    object_relative = f"objects/sha256/{digest.sha256[:2]}/{digest.sha256}.e57"
    return {
        "schemaVersion": PREPARED_SCHEMA,
        "status": "prepared",
        "authorizationConfigSha256": authorization.config_sha256,
        "retainedManifestSha256": authorization.retained_manifest_sha256,
        "archiveSha256": authorization.archive_sha256,
        "member": {
            **_member_tuple(authorization),
            "sha256": digest.sha256,
        },
        "partial": part_name,
        "object": object_relative,
        "resumePolicy": "restart_only",
        "synthesisAuthorized": False,
    }


def _load_prepared(
    descriptor: int,
    authorization: MetadataInspectionAuthorization,
    part_name: str,
) -> tuple[bytes, StreamDigest]:
    payload = _read_descriptor(descriptor, "prepared extraction record")
    raw = json.loads(payload)
    if not isinstance(raw, dict):
        raise ValueError("prepared extraction record must be an object")
    if payload != canonical_json_bytes(raw):
        raise ValueError("prepared extraction record is not canonical JSON")
    member = raw.get("member")
    if not isinstance(member, dict):
        raise ValueError("prepared extraction record lacks its member tuple")
    digest = StreamDigest(
        bytes=member.get("uncompressedBytes"),
        crc32=int(str(member.get("crc32", "-1")), 16),
        sha256=str(member.get("sha256", "")),
    )
    if raw != _prepared_document(authorization, digest, part_name):
        raise ValueError("prepared extraction record differs from the frozen transaction")
    return payload, digest


def _receipt_document(
    authorization: MetadataInspectionAuthorization,
    prepared_payload: bytes,
    digest: StreamDigest,
) -> dict:
    return {
        "schemaVersion": REPORT_SCHEMA,
        "status": "complete",
        "authorizationConfigSha256": authorization.config_sha256,
        "retainedManifestSha256": authorization.retained_manifest_sha256,
        "archiveSha256": authorization.archive_sha256,
        "preparedSha256": hashlib.sha256(prepared_payload).hexdigest(),
        "member": {
            "name": authorization.member.name,
            "bytes": digest.bytes,
            "crc32": f"{digest.crc32:08x}",
            "sha256": digest.sha256,
        },
        "output": f"objects/sha256/{digest.sha256[:2]}/{digest.sha256}.e57",
        "resumePolicy": "restart_only",
        "scientificRole": "raw_candidate",
        "qualificationStatus": "unqualified",
        "synthesisAuthorized": False,
    }


def _require_member_digest(
    authorization: MetadataInspectionAuthorization,
    digest: StreamDigest,
) -> None:
    if (
        digest.bytes != authorization.member.uncompressed_bytes
        or digest.crc32 != authorization.member.crc32
        or len(digest.sha256) != 64
        or digest.sha256.lower() != digest.sha256
        or any(character not in "0123456789abcdef" for character in digest.sha256)
    ):
        raise ValueError("E57 size, CRC32, or SHA-256 contract is invalid")


def _open_object_directory(root: int, digest: StreamDigest) -> int:
    objects = _open_or_create_private_directory_at(root, "objects", "object root")
    try:
        sha256_root = _open_or_create_private_directory_at(
            objects,
            "sha256",
            "SHA-256 object root",
        )
    finally:
        os.close(objects)
    try:
        return _open_or_create_private_directory_at(
            sha256_root,
            digest.sha256[:2],
            "SHA-256 object prefix",
        )
    finally:
        os.close(sha256_root)


def _object_name(digest: StreamDigest) -> str:
    return f"{digest.sha256}.e57"


def _verify_object(descriptor: int, digest: StreamDigest, block_bytes: int) -> None:
    result = os.fstat(descriptor)
    if (
        result.st_uid != os.getuid()
        or stat.S_IMODE(result.st_mode) != 0o400
        or result.st_nlink != 1
    ):
        raise PermissionError("published E57 object must be private and immutable")
    actual = _digest_descriptor(descriptor, block_bytes)
    if actual != digest:
        raise ValueError("published E57 object differs from its prepared transaction")


def _verify_held_part(
    transaction_directory: int,
    part_name: str,
    part_descriptor: int,
    digest: StreamDigest,
    block_bytes: int,
) -> tuple[int, int, int, int, int]:
    os.fchmod(part_descriptor, 0o400)
    os.fsync(part_descriptor)
    if _digest_descriptor(part_descriptor, block_bytes) != digest:
        raise ValueError("prepared E57 partial differs from prepared record")
    held = os.fstat(part_descriptor)
    if held.st_uid != os.getuid() or held.st_nlink != 1:
        raise PermissionError("prepared E57 partial must be private and singly linked")
    namespace = os.stat(part_name, dir_fd=transaction_directory, follow_symlinks=False)
    if _stable_identity(held) != _stable_identity(namespace):
        raise ValueError("verified partial namespace no longer names its held inode")
    return _stable_identity(held)


def _publish_verified_part(
    transaction_directory: int,
    part_name: str,
    part_descriptor: int,
    verified_identity: tuple[int, int, int, int, int],
    object_directory: int,
    digest: StreamDigest,
    block_bytes: int,
) -> None:
    namespace = os.stat(part_name, dir_fd=transaction_directory, follow_symlinks=False)
    if (
        _stable_identity(os.fstat(part_descriptor)) != verified_identity
        or _stable_identity(namespace) != verified_identity
    ):
        raise ValueError("verified E57 partial changed before publication")
    try:
        _atomic_rename_noreplace_at(
            transaction_directory,
            part_name,
            object_directory,
            _object_name(digest),
        )
    except FileExistsError:
        destination = _open_regular_at(
            object_directory,
            _object_name(digest),
            "existing E57 object",
        )
        try:
            _verify_object(destination, digest, block_bytes)
        finally:
            os.close(destination)
        _fsync_directory(object_directory)
        if _stable_identity(os.fstat(part_descriptor)) != verified_identity:
            raise ValueError("duplicate E57 partial changed during object verification")
        _unlink_bound_regular_at(
            transaction_directory,
            part_name,
            "duplicate verified E57 partial",
            part_descriptor,
        )
        return
    destination = _open_regular_at(
        object_directory,
        _object_name(digest),
        "E57 object",
    )
    try:
        if _publication_identity(os.fstat(destination)) != _publication_identity(
            os.fstat(part_descriptor)
        ):
            raise ValueError("published E57 namespace is not the held verified inode")
    finally:
        os.close(destination)


def _stream_archive_to_part(
    authorization: MetadataInspectionAuthorization,
    transaction_directory: int,
    part_name: str,
    log: Callable[[str], None],
) -> StreamDigest:
    archive_descriptor = open_regular_nofollow(
        authorization.archive_path,
        "retained archive",
    )
    try:
        archive_identity = _stable_identity(os.fstat(archive_descriptor))
        log(f"verifying retained archive SHA-256 ({authorization.archive_bytes:,} bytes)")
        archive_digest = _digest_descriptor(archive_descriptor, authorization.block_bytes)
        if (
            archive_digest.bytes != authorization.archive_bytes
            or archive_digest.sha256 != authorization.archive_sha256
        ):
            raise ValueError("retained Hovi archive SHA-256 or byte count drifted")
        os.lseek(archive_descriptor, 0, os.SEEK_SET)
        stream_sha = hashlib.sha256()
        stream_crc = 0
        stream_bytes = 0
        archive_source = os.fdopen(
            archive_descriptor,
            "rb",
            buffering=0,
            closefd=False,
        )
        log(f"streaming {authorization.member.name} to restart-only partial")
        with archive_source, zipfile.ZipFile(archive_source, "r") as archive:
            infos = archive.infolist()
            if len(infos) != 1 or archive.comment:
                raise ValueError("retained Hovi ZIP member inventory drifted")
            info = infos[0]
            authorization.member.verify(info)
            with archive.open(info, "r") as source, _open_exclusive_at(
                transaction_directory,
                part_name,
            ) as target:
                while block := source.read(authorization.block_bytes):
                    stream_bytes += len(block)
                    if stream_bytes > authorization.member.uncompressed_bytes:
                        raise ValueError("Hovi E57 expanded beyond its authorized byte count")
                    stream_sha.update(block)
                    stream_crc = zlib.crc32(block, stream_crc)
                    _write_all(target, block)
                target.flush()
                os.fsync(target.fileno())
        if _stable_identity(os.fstat(archive_descriptor)) != archive_identity:
            raise ValueError("retained Hovi archive inode changed during extraction")
        return StreamDigest(
            stream_bytes,
            stream_crc & 0xFFFFFFFF,
            stream_sha.hexdigest(),
        )
    finally:
        os.close(archive_descriptor)


def _complete_prepared_transaction(
    authorization: MetadataInspectionAuthorization,
    output_root: Path,
    root_directory: int,
    transaction_directory: int,
    part_name: str,
    prepared_payload: bytes,
    digest: StreamDigest,
    part_descriptor: int | None,
    verified_part_identity: tuple[int, int, int, int, int] | None = None,
) -> ExtractionResult | None:
    _require_member_digest(authorization, digest)
    object_directory = _open_object_directory(root_directory, digest)
    try:
        if part_descriptor is not None:
            if verified_part_identity is None:
                verified_part_identity = _verify_held_part(
                    transaction_directory,
                    part_name,
                    part_descriptor,
                    digest,
                    authorization.block_bytes,
                )
            _publish_verified_part(
                transaction_directory,
                part_name,
                part_descriptor,
                verified_part_identity,
                object_directory,
                digest,
                authorization.block_bytes,
            )
        else:
            object_descriptor = _open_optional_regular_at(
                object_directory,
                _object_name(digest),
                "E57 object",
            )
            if object_descriptor is None:
                return None
            try:
                _verify_object(
                    object_descriptor,
                    digest,
                    authorization.block_bytes,
                )
                _fsync_directory(object_directory)
            finally:
                os.close(object_descriptor)
    finally:
        os.close(object_directory)

    receipt = _receipt_document(authorization, prepared_payload, digest)
    _publish_record_at(transaction_directory, _RECEIPT_NAME, receipt)
    object_path = output_root / receipt["output"]
    receipt_path = output_root / "transactions" / authorization.transaction_id / _RECEIPT_NAME
    return ExtractionResult(
        authorization.transaction_id,
        object_path,
        digest.sha256,
        receipt_path,
    )


def extraction_plan(
    authorization: MetadataInspectionAuthorization,
    output_root: Path,
) -> dict:
    output_root = _normalize_output_root(output_root)
    transaction_root = output_root / "transactions" / authorization.transaction_id
    return {
        "authorizationConfig": authorization.config_path.as_posix(),
        "authorizationConfigSha256": authorization.config_sha256,
        "transactionId": authorization.transaction_id,
        "retainedManifest": {
            "path": authorization.retained_manifest_path.as_posix(),
            "bytes": authorization.retained_manifest_bytes,
            "sha256": authorization.retained_manifest_sha256,
            "retentionId": authorization.retention_id,
            "status": "complete",
            "verified": True,
        },
        "archive": {
            "path": authorization.archive_path.as_posix(),
            "relativePath": authorization.archive_relative_path,
            "bytes": authorization.archive_bytes,
            "sha256": authorization.archive_sha256,
        },
        "member": _member_tuple(authorization),
        "minimumFreeBytes": authorization.minimum_free_bytes,
        "blockBytes": authorization.block_bytes,
        "extractionPolicy": _extraction_policy_tuple(authorization),
        "resume": "restart_only",
        "recovery": "idempotent_prepared_object_receipt_state_machine",
        "outputRoot": output_root.as_posix(),
        "partial": (
            transaction_root / f"{authorization.transaction_id}{_PART_NAME_SUFFIX}"
        ).as_posix(),
        "prepared": (transaction_root / _PREPARED_NAME).as_posix(),
        "objectTemplate": (
            output_root / "objects" / "sha256" / "<sha256-prefix>" / "<sha256>.e57"
        ).as_posix(),
        "receipt": (transaction_root / _RECEIPT_NAME).as_posix(),
        "synthesisAuthorized": False,
    }


def extract_authorized_e57(
    authorization_path: Path = DEFAULT_AUTHORIZATION,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    log: Callable[[str], None] = print,
) -> ExtractionResult:
    authorization = load_metadata_inspection_authorization(authorization_path)
    output_root = _normalize_output_root(output_root)
    root_directory = _open_or_create_private_root(output_root)
    try:
        transactions = _open_or_create_private_directory_at(
            root_directory,
            "transactions",
            "transaction root",
        )
        try:
            transaction_directory = _open_or_create_private_directory_at(
                transactions,
                authorization.transaction_id,
                "authorization transaction",
            )
        finally:
            os.close(transactions)
        try:
            lock_descriptor = _acquire_transaction_lock(transaction_directory)
            try:
                part_name = f"{authorization.transaction_id}{_PART_NAME_SUFFIX}"
                prepared_descriptor = _open_optional_regular_at(
                    transaction_directory,
                    _PREPARED_NAME,
                    "prepared extraction record",
                )
                receipt_descriptor = _open_optional_regular_at(
                    transaction_directory,
                    _RECEIPT_NAME,
                    "extraction receipt",
                )
                part_descriptor = _open_optional_regular_at(
                    transaction_directory,
                    part_name,
                    "restart-only E57 partial",
                )
                try:
                    if receipt_descriptor is not None and prepared_descriptor is None:
                        raise ValueError("receipt exists without its prepared transaction")
                    if prepared_descriptor is not None:
                        prepared_payload, digest = _load_prepared(
                            prepared_descriptor,
                            authorization,
                            part_name,
                        )
                        result = _complete_prepared_transaction(
                            authorization,
                            output_root,
                            root_directory,
                            transaction_directory,
                            part_name,
                            prepared_payload,
                            digest,
                            part_descriptor,
                        )
                        if result is not None:
                            return result
                        if receipt_descriptor is not None:
                            raise ValueError(
                                "completed receipt lost its content-addressed object"
                            )
                        log("clearing prepared intent that lost both partial and object")
                        _unlink_bound_regular_at(
                            transaction_directory,
                            _PREPARED_NAME,
                            "orphaned prepared intent",
                            prepared_descriptor,
                        )
                    elif part_descriptor is not None:
                        log("discarding unprepared restart-only deflate partial")
                        _unlink_bound_regular_at(
                            transaction_directory,
                            part_name,
                            "unprepared restart-only partial",
                            part_descriptor,
                        )
                finally:
                    if part_descriptor is not None:
                        os.close(part_descriptor)
                    if receipt_descriptor is not None:
                        os.close(receipt_descriptor)
                    if prepared_descriptor is not None:
                        os.close(prepared_descriptor)

                filesystem = os.fstatvfs(root_directory)
                free = filesystem.f_bavail * filesystem.f_frsize
                if free < authorization.minimum_free_bytes:
                    raise RuntimeError(
                        f"Hovi E57 extraction requires {authorization.minimum_free_bytes:,} "
                        f"free bytes before start; output filesystem has {free:,}"
                    )
                streamed = _stream_archive_to_part(
                    authorization,
                    transaction_directory,
                    part_name,
                    log,
                )
                _require_member_digest(authorization, streamed)
                part_descriptor = _open_regular_at(
                    transaction_directory,
                    part_name,
                    "extracted E57 partial",
                )
                try:
                    verified_identity = _verify_held_part(
                        transaction_directory,
                        part_name,
                        part_descriptor,
                        streamed,
                        authorization.block_bytes,
                    )
                    prepared = _prepared_document(authorization, streamed, part_name)
                    prepared_payload = _publish_record_at(
                        transaction_directory,
                        _PREPARED_NAME,
                        prepared,
                    )
                    return _complete_prepared_transaction(
                        authorization,
                        output_root,
                        root_directory,
                        transaction_directory,
                        part_name,
                        prepared_payload,
                        streamed,
                        part_descriptor,
                        verified_identity,
                    )
                finally:
                    os.close(part_descriptor)
            finally:
                os.close(lock_descriptor)
        finally:
            os.close(transaction_directory)
    finally:
        os.close(root_directory)


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan or execute the authorized restart-only HY_SPRUCE4 E57 extraction."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_AUTHORIZATION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform the 84.8 GB extraction; without this flag only print the frozen plan.",
    )
    args = parser.parse_args()
    if not args.execute:
        authorization = load_metadata_inspection_authorization(args.config)
        print(json.dumps(extraction_plan(authorization, args.output_root), indent=2))
        return
    result = extract_authorized_e57(args.config, args.output_root)
    print(f"full-scan extraction receipt: {result.receipt_path}")


if __name__ == "__main__":
    _main()
