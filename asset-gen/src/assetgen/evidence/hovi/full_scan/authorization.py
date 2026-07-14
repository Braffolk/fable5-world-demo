"""Fail-closed authorization for inspecting the retained HY_SPRUCE4 E57 metadata."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from ....config import ASSET_GEN_ROOT, CONFIG_DIR


AUTHORIZATION_SCHEMA = (
    "hovi-hy-spruce4-full-metadata-inspection-authorization/1.0.0"
)
AUTHORIZATION_ID = "hovi-hy-spruce4-full-metadata-inspection-after-retention-v1"
EXPECTED_AUTHORIZATION_BYTES = 3188
EXPECTED_AUTHORIZATION_SHA256 = (
    "fe44c5c05c4ccd50cda2c9a662d3c171e3a281f823691d674010ee908c4480b9"
)
DEFAULT_AUTHORIZATION = (
    CONFIG_DIR / "evidence" / "hovi-hy-spruce4-full-metadata-inspection.json"
)
RETAINED_SCHEMA = "hovi-hy-spruce4-full-retention/1.0.0"
EXPECTED_RETENTION_ID = (
    "cabb67ce94a2c55ee49d0f09b2a000348eb45974b26560f7e8b8d5c6cf543e63"
)
EXPECTED_ARCHIVE_SHA256 = (
    "008540c96bcbea1eb98128d92df844ee6346c11e1ec99ad626ba3c2e6a0caa1e"
)
EXPECTED_ARCHIVE_BYTES = 47_345_435_519
EXPECTED_MEMBER_NAME = "HY_SPRUCE4-full.e57"
EXPECTED_MEMBER_BYTES = 84_817_645_568
EXPECTED_MEMBER_COMPRESSED_BYTES = 47_345_435_231
EXPECTED_MEMBER_CRC32 = 0xD76A5D89
MINIMUM_FREE_BYTES = 100 * (1 << 30)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Hovi full-scan {label} must be an object")
    return value


def _directory_flags() -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError(
            "Hovi full-scan authority requires O_DIRECTORY and O_NOFOLLOW"
        )
    flags = os.O_RDONLY
    return flags | os.O_DIRECTORY | os.O_NOFOLLOW


def _open_existing_directory(path: Path, label: str) -> int:
    absolute = _absolute_path(path)
    descriptor = os.open(absolute.anchor, _directory_flags())
    try:
        for part in absolute.parts[1:]:
            next_descriptor = os.open(part, _directory_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except (FileNotFoundError, NotADirectoryError, OSError) as error:
        os.close(descriptor)
        raise ValueError(
            f"Hovi full-scan {label} has a missing, non-directory, or symlink ancestor"
        ) from error
    return descriptor


def open_regular_nofollow(path: Path, label: str) -> int:
    """Open one regular source inode without following its path or final component."""
    absolute = _absolute_path(path)
    parent_descriptor = _open_existing_directory(absolute.parent, label)
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("Hovi full-scan authority requires O_NOFOLLOW")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    try:
        descriptor = os.open(absolute.name, flags, dir_fd=parent_descriptor)
    finally:
        os.close(parent_descriptor)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError(f"Hovi full-scan {label} is not a regular file")
    return descriptor


def _safe_bound_path(relative: Any, label: str) -> Path:
    if not isinstance(relative, str):
        raise ValueError(f"Hovi full-scan {label} path must be relative")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts or "\\" in relative:
        raise ValueError(f"unsafe Hovi full-scan {label} path")
    return _absolute_path(ASSET_GEN_ROOT.joinpath(*pure.parts))


def _read_regular_nofollow(path: Path, label: str) -> bytes:
    descriptor = open_regular_nofollow(path, label)
    with os.fdopen(descriptor, "rb") as source:
        return source.read()


@dataclass(frozen=True)
class AuthorizedMember:
    name: str
    uncompressed_bytes: int
    compressed_bytes: int
    compression_method: int
    flag_bits: int
    crc32: int
    local_header_offset: int
    create_system: int
    external_attr: int

    def verify(self, info: zipfile.ZipInfo) -> None:
        actual = (
            info.filename,
            info.file_size,
            info.compress_size,
            info.compress_type,
            info.flag_bits,
            info.CRC,
            info.header_offset,
            info.create_system,
            info.external_attr,
        )
        expected = (
            self.name,
            self.uncompressed_bytes,
            self.compressed_bytes,
            self.compression_method,
            self.flag_bits,
            self.crc32,
            self.local_header_offset,
            self.create_system,
            self.external_attr,
        )
        mode = info.external_attr >> 16
        if (
            actual != expected
            or info.is_dir()
            or stat.S_ISLNK(mode)
            or info.flag_bits & 0x1
            or PurePosixPath(info.filename).is_absolute()
            or ".." in PurePosixPath(info.filename).parts
            or "\\" in info.filename
        ):
            raise ValueError("retained Hovi ZIP member inventory drifted or is unsafe")


@dataclass(frozen=True)
class MetadataInspectionAuthorization:
    config_path: Path
    config_sha256: str
    transaction_id: str
    retained_manifest_path: Path
    retained_manifest_sha256: str
    archive_path: Path
    archive_sha256: str
    archive_bytes: int
    member: AuthorizedMember
    minimum_free_bytes: int
    block_bytes: int
    retained_manifest_bytes: int
    retention_id: str
    archive_relative_path: str

    def verify_zip_inventory(self) -> None:
        descriptor = open_regular_nofollow(self.archive_path, "retained archive")
        with os.fdopen(descriptor, "rb", buffering=0) as source:
            if os.fstat(source.fileno()).st_size != self.archive_bytes:
                raise ValueError("retained Hovi archive byte count drifted")
            with zipfile.ZipFile(source, "r") as archive:
                members = archive.infolist()
                if len(members) != 1 or archive.comment:
                    raise ValueError("retained Hovi ZIP central-directory inventory drifted")
                self.member.verify(members[0])


def load_metadata_inspection_authorization(
    path: Path = DEFAULT_AUTHORIZATION,
    *,
    config_bytes: bytes | None = None,
    retained_manifest_bytes: bytes | None = None,
) -> MetadataInspectionAuthorization:
    path = _absolute_path(path)
    encoded = (
        _read_regular_nofollow(path, "metadata authorization")
        if config_bytes is None
        else config_bytes
    )
    config_sha256 = hashlib.sha256(encoded).hexdigest()
    if (
        len(encoded) != EXPECTED_AUTHORIZATION_BYTES
        or config_sha256 != EXPECTED_AUTHORIZATION_SHA256
    ):
        raise ValueError("Hovi metadata authorization bytes are not the frozen decision")
    raw = json.loads(encoded)
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "id",
        "status",
        "authorization",
        "retained_manifest",
        "archive",
        "zip_inventory",
        "resource_gate",
        "extraction_policy",
        "evidence_boundary",
    }:
        raise ValueError("unsupported Hovi full-scan metadata authorization")
    authorization = _require_mapping(raw["authorization"], "authorization")
    manifest_ref = _require_mapping(raw["retained_manifest"], "manifest binding")
    archive_ref = _require_mapping(raw["archive"], "archive binding")
    inventory = _require_mapping(raw["zip_inventory"], "ZIP inventory")
    member_ref = _require_mapping(inventory.get("member"), "ZIP member")
    resources = _require_mapping(raw["resource_gate"], "resource gate")
    policy = _require_mapping(raw["extraction_policy"], "extraction policy")
    boundary = _require_mapping(raw["evidence_boundary"], "evidence boundary")
    expected_manifest_ref = {
        "path": (
            "data/in/evidence/hovi-full-q1/"
            f"{EXPECTED_RETENTION_ID}/retained.json"
        ),
        "bytes": 3394,
        "sha256": "add8c65111e2fbb28ea617802294452a6b4fdc2b9473e40deafd3c08e630da3f",
        "retention_id": EXPECTED_RETENTION_ID,
        "status": "complete",
        "verified": True,
    }
    expected_archive_ref = {
        "relative_path": (
            "files/Laboratory_and_field_data/Terrestrial_laser_scanning/"
            "Point_clouds_full/HY_SPRUCE4-full.zip"
        ),
        "bytes": EXPECTED_ARCHIVE_BYTES,
        "sha256": EXPECTED_ARCHIVE_SHA256,
    }
    expected_member_ref = {
        "name": EXPECTED_MEMBER_NAME,
        "uncompressed_bytes": EXPECTED_MEMBER_BYTES,
        "compressed_bytes": EXPECTED_MEMBER_COMPRESSED_BYTES,
        "compression_method": zipfile.ZIP_DEFLATED,
        "flag_bits": 0,
        "crc32": f"{EXPECTED_MEMBER_CRC32:08x}",
        "local_header_offset": 0,
        "create_system": 0,
        "external_attr": 32,
    }
    if (
        raw["schema_version"] != AUTHORIZATION_SCHEMA
        or raw["id"] != AUTHORIZATION_ID
        or raw["status"] != "metadata_inspection_authorized"
        or authorization
        != {
            "decision": "METADATA_INSPECTION_GO",
            "scope": "extract_exact_retained_e57_for_metadata_inspection_only",
            "archive_extraction_authorized": True,
            "e57_metadata_inspection_authorized": True,
            "point_conversion_authorized": False,
            "surface_reconstruction_authorized": False,
            "target_truth": False,
            "usable_surface": False,
            "synthesis_authorized": False,
        }
        or manifest_ref != expected_manifest_ref
        or archive_ref != expected_archive_ref
        or inventory
        != {
            "member_count": 1,
            "archive_comment_bytes": 0,
            "member": expected_member_ref,
        }
        or resources
        != {
            "minimum_free_bytes": MINIMUM_FREE_BYTES,
            "minimum_free_space_label": "100 GiB",
            "evaluate_on_output_filesystem_before_archive_hash": True,
        }
        or policy
        != {
            "stream_only": True,
            "block_bytes": 8 << 20,
            "partial_address": "authorization_config_sha256",
            "partial_suffix": ".e57.part",
            "resume": "restart_only",
            "resume_reason": (
                "A raw deflate ZIP member cannot resume at an arbitrary output byte "
                "without persisted bit-exact decompressor state."
            ),
            "interrupted_partial_disposition": (
                "discard_and_restart_from_zero"
            ),
            "verification_before_publish": [
                "uncompressed_size",
                "zip_crc32",
                "stream_sha256",
                "independent_part_sha256",
            ],
            "prepared_record": (
                "transactions/<authorization-sha256>/prepared.json"
            ),
            "receipt_record": (
                "transactions/<authorization-sha256>/extraction.json"
            ),
            "recovery": "idempotent_prepared_object_receipt_state_machine",
            "transaction_lock": (
                "exclusive_advisory_lock_held_through_receipt"
            ),
            "writable_directory_policy": "current_user_owned_0700",
            "prepared_without_payload": "clear_intent_and_restart_from_zero",
            "object_collision": "accept_only_after_full_digest_equality",
            "publish": "atomic_noreplace_rename",
            "immutable_output": "objects/sha256/<sha256-prefix>/<sha256>.e57",
        }
        or boundary
        != {
            "role": "raw_candidate",
            "qualification_status": "unqualified",
            "transfer_ceiling": "none",
            "metadata_inventory_is_not_surface_truth": True,
            "synthesis_authorized": False,
        }
    ):
        raise ValueError("Hovi metadata-inspection authorization boundary changed")

    retained_path = _safe_bound_path(manifest_ref.get("path"), "retained manifest")
    retained_bytes = (
        _read_regular_nofollow(retained_path, "retained manifest")
        if retained_manifest_bytes is None
        else retained_manifest_bytes
    )
    if (
        len(retained_bytes) != manifest_ref.get("bytes")
        or hashlib.sha256(retained_bytes).hexdigest() != manifest_ref.get("sha256")
        or manifest_ref.get("retention_id") != EXPECTED_RETENTION_ID
        or manifest_ref.get("status") != "complete"
        or manifest_ref.get("verified") is not True
    ):
        raise ValueError("finalized Hovi retained-manifest binding drifted")
    retained = json.loads(retained_bytes)
    retained_artifact = _require_mapping(retained.get("artifact"), "retained artifact")
    if (
        retained.get("schema_version") != RETAINED_SCHEMA
        or retained.get("retention_id") != EXPECTED_RETENTION_ID
        or retained.get("status") != "complete"
        or retained.get("synthesis_authorized") is not False
        or retained_artifact.get("verified") is not True
        or retained_artifact.get("bytes") != EXPECTED_ARCHIVE_BYTES
        or retained_artifact.get("retained_bytes") != EXPECTED_ARCHIVE_BYTES
        or retained_artifact.get("sha256") != EXPECTED_ARCHIVE_SHA256
        or retained_artifact.get("relative_path") != archive_ref.get("relative_path")
    ):
        raise ValueError("finalized Hovi retention no longer authorizes metadata inspection")

    archive_relative = archive_ref.get("relative_path")
    if not isinstance(archive_relative, str):
        raise ValueError("Hovi archive relative path is invalid")
    archive_pure = PurePosixPath(archive_relative)
    if archive_pure.is_absolute() or ".." in archive_pure.parts or "\\" in archive_relative:
        raise ValueError("Hovi archive relative path is unsafe")
    archive_path = retained_path.parent.joinpath(*archive_pure.parts)
    archive_descriptor = open_regular_nofollow(archive_path, "retained archive")
    archive_stat = os.fstat(archive_descriptor)
    os.close(archive_descriptor)
    if (
        archive_ref.get("bytes") != EXPECTED_ARCHIVE_BYTES
        or archive_ref.get("sha256") != EXPECTED_ARCHIVE_SHA256
        or archive_stat.st_size != EXPECTED_ARCHIVE_BYTES
    ):
        raise ValueError("retained Hovi archive binding drifted")

    member = AuthorizedMember(
        name=member_ref["name"],
        uncompressed_bytes=member_ref["uncompressed_bytes"],
        compressed_bytes=member_ref["compressed_bytes"],
        compression_method=member_ref["compression_method"],
        flag_bits=member_ref["flag_bits"],
        crc32=int(member_ref["crc32"], 16),
        local_header_offset=member_ref["local_header_offset"],
        create_system=member_ref["create_system"],
        external_attr=member_ref["external_attr"],
    )
    if (
        inventory.get("member_count") != 1
        or inventory.get("archive_comment_bytes") != 0
        or member
        != AuthorizedMember(
            EXPECTED_MEMBER_NAME,
            EXPECTED_MEMBER_BYTES,
            EXPECTED_MEMBER_COMPRESSED_BYTES,
            zipfile.ZIP_DEFLATED,
            0,
            EXPECTED_MEMBER_CRC32,
            0,
            0,
            32,
        )
    ):
        raise ValueError("authorized Hovi ZIP inventory tuple changed")

    result = MetadataInspectionAuthorization(
        config_path=path,
        config_sha256=config_sha256,
        transaction_id=config_sha256,
        retained_manifest_path=retained_path,
        retained_manifest_sha256=str(manifest_ref["sha256"]),
        archive_path=archive_path,
        archive_sha256=EXPECTED_ARCHIVE_SHA256,
        archive_bytes=EXPECTED_ARCHIVE_BYTES,
        member=member,
        minimum_free_bytes=MINIMUM_FREE_BYTES,
        block_bytes=8 << 20,
        retained_manifest_bytes=expected_manifest_ref["bytes"],
        retention_id=EXPECTED_RETENTION_ID,
        archive_relative_path=expected_archive_ref["relative_path"],
    )
    result.verify_zip_inventory()
    return result
