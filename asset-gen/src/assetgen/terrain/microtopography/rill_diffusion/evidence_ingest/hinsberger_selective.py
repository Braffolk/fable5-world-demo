"""Retain six selected Hinsberger ZIP members without acquiring the full archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
from typing import Any
import zlib

from .....config import DATA_IN
from .hinsberger_contract import (
    API_RESPONSE_SHA256,
    ARCHIVE_BYTES,
    ARCHIVE_MD5,
    ARCHIVE_NAME,
    CENTRAL_INVENTORY_SHA256,
    CENTRAL_MEMBER_COUNT,
    DATASET_DOI,
    DOWNLOAD_URL,
    HEADER_PROBE_RANGE,
    LICENSE_SPDX,
    LICENSE_URL,
    MISSING_COMPRESSED_BYTES,
    MISSING_RANGES,
    NEW_SELECTED_BYTES_IN_HEADER_PROBE,
    PARTIAL_TRANSFER_SHA256,
    PUBLISHER_FILE_ID,
    REUSED_SELECTED_COMPRESSED_BYTES,
    SELECTED_COMPRESSED_BYTES,
    SELECTED_MEMBERS,
    SelectedMember,
)
from .range_store import RangePiece, RangeStore


_SCHEMA = "hinsberger-selective-retention/1"
_CONTENT_RANGE = re.compile(r"^Content-Range:\s*bytes\s+(\d+)-(\d+)/(\d+)\s*$", re.I | re.M)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")


def _hash_file(path: Path, *, include_crc32: bool = False) -> dict[str, Any]:
    digest = hashlib.sha256()
    crc32 = 0
    size = 0
    with path.open("rb") as source:
        while block := source.read(16 << 20):
            digest.update(block)
            if include_crc32:
                crc32 = zlib.crc32(block, crc32)
            size += len(block)
    identity = {"bytes": size, "sha256": digest.hexdigest()}
    if include_crc32:
        identity["crc32"] = f"{crc32 & 0xFFFFFFFF:08x}"
    return identity


def _safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"unsafe retained relative path: {relative}")
    return path


def _load_bound_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is absent: {path}")
    identity = _hash_file(path)
    if identity["sha256"] != expected_sha256:
        raise ValueError(f"{label} SHA-256 changed: {path}")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _verify_api(root: Path) -> dict[str, Any]:
    path = root / "figshare-api.json"
    value = _load_bound_json(path, API_RESPONSE_SHA256, "Figshare API response")
    files = value.get("files")
    if (
        value.get("doi") != DATASET_DOI
        or value.get("version") != 1
        or value.get("status") != "public"
        or value.get("size") != ARCHIVE_BYTES
        or value.get("license", {}).get("name") != "CC BY 4.0"
        or value.get("license", {}).get("url") != LICENSE_URL
        or not isinstance(files, list)
        or len(files) != 1
    ):
        raise ValueError("Figshare API dataset identity or license changed")
    artifact = files[0]
    if (
        artifact.get("id") != PUBLISHER_FILE_ID
        or artifact.get("name") != ARCHIVE_NAME
        or artifact.get("size") != ARCHIVE_BYTES
        or artifact.get("download_url") != DOWNLOAD_URL
        or artifact.get("supplied_md5") != ARCHIVE_MD5
        or artifact.get("computed_md5") != ARCHIVE_MD5
    ):
        raise ValueError("Figshare API archive tuple changed")
    return {"path": path.name, **_hash_file(path)}


def _verify_central_inventory(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = root / "archive-members.pre-download.json"
    value = _load_bound_json(path, CENTRAL_INVENTORY_SHA256, "ZIP64 central inventory")
    members = value.get("members")
    if value.get("member_count") != CENTRAL_MEMBER_COUNT or not isinstance(members, list):
        raise ValueError("ZIP64 central inventory no longer contains exactly 28 members")
    by_name = {item.get("name"): item for item in members if isinstance(item, dict)}
    if len(by_name) != CENTRAL_MEMBER_COUNT or None in by_name:
        raise ValueError("ZIP64 central inventory contains duplicate or unnamed members")
    for member in SELECTED_MEMBERS:
        actual = by_name.get(member.path)
        expected = {
            "compressed_bytes": member.compressed_bytes,
            "uncompressed_bytes": member.uncompressed_bytes,
            "crc32": f"{member.crc32:08x}",
            "compression": 8,
            "local_header_offset": member.local_header_offset,
            "flags": 0,
            "is_dir": False,
        }
        if actual is None or any(actual.get(key) != expected_value for key, expected_value in expected.items()):
            raise ValueError(f"selected ZIP64 member identity changed: {member.path}")
    return {"path": path.name, **_hash_file(path)}, by_name


def _verify_range_headers(path: Path, expected_start: int, expected_end: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"HTTP range headers are absent: {path}")
    text = path.read_text(encoding="latin-1")
    matches = _CONTENT_RANGE.findall(text)
    expected = (str(expected_start), str(expected_end), str(ARCHIVE_BYTES))
    if not matches or matches[-1] != expected or "206 Partial Content" not in text:
        raise ValueError(f"HTTP response does not bind exact range {expected_start}-{expected_end}: {path}")
    return {"path": path.name, **_hash_file(path)}


def _intersects_selected(start: int, end: int) -> bool:
    return any(
        not (end < member.local_header_offset or start > member.data_end)
        for member in SELECTED_MEMBERS
    )


def _verified_range_store(root: Path) -> tuple[RangeStore, list[dict[str, Any]]]:
    partial_path = root / "partial-transfer.json"
    partial = _load_bound_json(partial_path, PARTIAL_TRANSFER_SHA256, "partial transfer manifest")
    pieces: list[RangePiece] = []
    records: list[dict[str, Any]] = []
    for item in partial.get("pieces", []):
        start, end = item["retained_range"]
        if not _intersects_selected(start, end):
            continue
        path = _safe_path(root, item["path"])
        identity = _hash_file(path)
        if identity["bytes"] != item["bytes"] or identity["sha256"] != item["sha256"]:
            raise ValueError(f"reused retained range changed: {path}")
        header_item = item["http_headers"]
        headers_path = _safe_path(root, header_item["path"])
        headers_identity = _hash_file(headers_path)
        if headers_identity["sha256"] != header_item["sha256"]:
            raise ValueError(f"reused HTTP provenance changed: {headers_path}")
        piece = RangePiece(
            start,
            end,
            path,
            identity["sha256"],
            "reused_full_archive_stage",
            headers_path,
            headers_identity["sha256"],
        )
        pieces.append(piece)
        records.append(_piece_record(root, piece, identity, headers_identity))

    probe_start, probe_end = HEADER_PROBE_RANGE
    probe_path = root / "selective" / "header-probes" / "field8_9_dem.bin"
    probe_identity = _hash_file(probe_path) if probe_path.is_file() else None
    if probe_identity is None or probe_identity["bytes"] != probe_end - probe_start + 1:
        raise FileNotFoundError(f"complete selected local-header probe is absent: {probe_path}")
    probe_headers = probe_path.with_suffix(".headers")
    probe_header_identity = _verify_range_headers(probe_headers, probe_start, probe_end)
    probe = RangePiece(
        probe_start,
        probe_end,
        probe_path,
        probe_identity["sha256"],
        "new_selected_header_probe",
        probe_headers,
        probe_header_identity["sha256"],
    )
    pieces.append(probe)
    records.append(_piece_record(root, probe, probe_identity, probe_header_identity))

    for start, end in MISSING_RANGES:
        path = root / "selective" / "missing" / f"{start}-{end}.bin"
        if not path.is_file() or path.stat().st_size != end - start + 1:
            present = path.stat().st_size if path.exists() else 0
            raise FileNotFoundError(
                f"selected compressed range is incomplete: {path} has {present:,} bytes; "
                f"expected {end - start + 1:,}"
            )
        identity = _hash_file(path)
        headers_path = path.with_suffix(".headers")
        headers_identity = _verify_range_headers(headers_path, start, end)
        piece = RangePiece(
            start,
            end,
            path,
            identity["sha256"],
            "new_selected_compressed_range",
            headers_path,
            headers_identity["sha256"],
        )
        pieces.append(piece)
        records.append(_piece_record(root, piece, identity, headers_identity))

    store = RangeStore(pieces)
    for member in SELECTED_MEMBERS:
        store.covering_pieces(member.local_header_offset, member.data_end)
    return store, sorted(records, key=lambda item: item["range"])


def _piece_record(
    root: Path,
    piece: RangePiece,
    identity: dict[str, Any],
    headers_identity: dict[str, Any],
) -> dict[str, Any]:
    return {
        "range": [piece.start, piece.end],
        "bytes": identity["bytes"],
        "sha256": identity["sha256"],
        "path": piece.path.relative_to(root).as_posix(),
        "source": piece.source,
        "http_headers": {
            "path": piece.headers_path.relative_to(root).as_posix() if piece.headers_path else None,
            "bytes": headers_identity["bytes"],
            "sha256": headers_identity["sha256"],
        },
    }


def _read_exact(store: RangeStore, start: int, end: int) -> bytes:
    return b"".join(store.read_chunks(start, end, chunk_bytes=end - start + 1))


def _verify_local_header(store: RangeStore, member: SelectedMember) -> dict[str, Any]:
    data = _read_exact(
        store,
        member.local_header_offset,
        member.local_header_offset + member.local_header_bytes - 1,
    )
    fields = struct.unpack_from("<I5H3I2H", data)
    signature, version, flags, method, mod_time, mod_date, crc32, compressed, uncompressed, name_len, extra_len = fields
    name = data[30 : 30 + name_len].decode("utf-8")
    if (
        signature != 0x04034B50
        or flags != 0
        or method != 8
        or crc32 != member.crc32
        or compressed != member.compressed_bytes
        or uncompressed != member.uncompressed_bytes
        or name != member.path
        or 30 + name_len + extra_len != member.local_header_bytes
        or member.local_header_offset + member.local_header_bytes != member.data_start
    ):
        raise ValueError(f"selected local ZIP header changed or is unsupported: {member.path}")
    return {
        "path": member.path,
        "offset": member.local_header_offset,
        "bytes": member.local_header_bytes,
        "sha256": hashlib.sha256(data).hexdigest(),
        "version_needed": version,
        "flags": flags,
        "compression": method,
        "crc32": f"{crc32:08x}",
        "compressed_bytes": compressed,
        "uncompressed_bytes": uncompressed,
        "data_range": [member.data_start, member.data_end],
        "mod_time": mod_time,
        "mod_date": mod_date,
    }


def _extract_member(store: RangeStore, member: SelectedMember, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    if temporary.exists() or destination.exists():
        raise FileExistsError(f"selected member extraction path already exists: {destination}")
    inflater = zlib.decompressobj(-zlib.MAX_WBITS)
    compressed_sha256 = hashlib.sha256()
    output_sha256 = hashlib.sha256()
    crc32 = 0
    output_bytes = 0
    with temporary.open("wb") as target:
        for block in store.read_chunks(member.data_start, member.data_end):
            compressed_sha256.update(block)
            decoded = inflater.decompress(block)
            if inflater.unused_data or inflater.unconsumed_tail:
                raise ValueError(f"selected deflate stream ended before its contracted range: {member.path}")
            if decoded:
                output_bytes += len(decoded)
                if output_bytes > member.uncompressed_bytes:
                    raise ValueError(f"selected member inflated beyond contracted size: {member.path}")
                output_sha256.update(decoded)
                crc32 = zlib.crc32(decoded, crc32)
                target.write(decoded)
        tail = inflater.flush()
        if tail:
            output_bytes += len(tail)
            output_sha256.update(tail)
            crc32 = zlib.crc32(tail, crc32)
            target.write(tail)
        target.flush()
        os.fsync(target.fileno())
    if (
        not inflater.eof
        or inflater.unused_data
        or inflater.unconsumed_tail
        or output_bytes != member.uncompressed_bytes
        or (crc32 & 0xFFFFFFFF) != member.crc32
    ):
        raise ValueError(
            f"selected member failed deflate/size/CRC verification: {member.path}; "
            f"bytes={output_bytes:,}, crc32={crc32 & 0xFFFFFFFF:08x}"
        )
    temporary.replace(destination)
    return {
        "path": member.path,
        "retained_path": f"members/{member.path}",
        "compressed_range": [member.data_start, member.data_end],
        "compressed_bytes": member.compressed_bytes,
        "compressed_sha256": compressed_sha256.hexdigest(),
        "uncompressed_bytes": output_bytes,
        "sha256": output_sha256.hexdigest(),
        "crc32": f"{crc32 & 0xFFFFFFFF:08x}",
    }


def _implementation_identity() -> list[dict[str, Any]]:
    paths = (
        Path(__file__),
        Path(__file__).with_name("hinsberger_contract.py"),
        Path(__file__).with_name("range_store.py"),
        Path(__file__).parent.parent / "contract.py",
    )
    return [{"path": path.name, **_hash_file(path)} for path in paths]


def _validate_existing_transaction(final: Path, recipe: dict[str, Any], build_id: str) -> Path:
    recipe_path = final / "recipe.json"
    manifest_path = final / "manifest.json"
    if recipe_path.read_bytes() != _canonical_bytes(recipe):
        raise ValueError(f"existing selective-retention recipe changed: {recipe_path}")
    manifest = json.loads(manifest_path.read_bytes())
    provenance = manifest.get("range_provenance", {})
    expected_authority = {
        "research_only_selected_member_qualification": True,
        "whole_archive_retention": False,
        "absolute_height_truth": False,
        "imagery_pixel_inspection": False,
        "training": False,
        "production": False,
        "preview": False,
    }
    expected_boundary = {
        "official_file_id": PUBLISHER_FILE_ID,
        "official_total_bytes": ARCHIVE_BYTES,
        "publisher_md5": ARCHIVE_MD5,
        "whole_archive_md5_verified": False,
        "full_archive_equivalence": False,
        "central_directory_bound": True,
        "selected_local_headers_bound": True,
    }
    expected_accounting = {
        "selected_compressed_bytes": SELECTED_COMPRESSED_BYTES,
        "selected_compressed_bytes_reused": REUSED_SELECTED_COMPRESSED_BYTES,
        "selected_compressed_bytes_downloaded": MISSING_COMPRESSED_BYTES
        + NEW_SELECTED_BYTES_IN_HEADER_PROBE,
        "selected_compressed_bytes_remaining": 0,
        "new_transport_bytes": MISSING_COMPRESSED_BYTES
        + (HEADER_PROBE_RANGE[1] - HEADER_PROBE_RANGE[0] + 1),
    }
    provenance_path = _safe_path(final, str(provenance.get("path")))
    if (
        manifest.get("schema") != _SCHEMA
        or manifest.get("build_id") != build_id
        or manifest.get("recipe_sha256") != build_id
        or manifest.get("status") != "research_only_selected_members_retained"
        or manifest.get("dataset") != recipe.get("dataset")
        or manifest.get("authority") != expected_authority
        or manifest.get("archive_identity_boundary") != expected_boundary
        or manifest.get("byte_accounting") != expected_accounting
        or _hash_file(provenance_path).get("sha256") != provenance.get("sha256")
    ):
        raise ValueError(f"existing selective-retention transaction failed identity validation: {final}")
    expected_paths = {member.path for member in SELECTED_MEMBERS}
    records = manifest.get("selected_members")
    by_path = {
        record.get("path"): record
        for record in records
        if isinstance(record, dict)
    } if isinstance(records, list) else {}
    if set(by_path) != expected_paths:
        raise ValueError(f"existing selective-retention member set changed: {final}")
    for relative, record in by_path.items():
        contract = next(member for member in SELECTED_MEMBERS if member.path == relative)
        retained = _safe_path(final, str(record.get("retained_path")))
        identity = _hash_file(retained, include_crc32=True)
        if (
            record.get("retained_path") != f"members/{relative}"
            or record.get("compressed_range") != [contract.data_start, contract.data_end]
            or record.get("compressed_bytes") != contract.compressed_bytes
            or record.get("uncompressed_bytes") != contract.uncompressed_bytes
            or record.get("crc32") != f"{contract.crc32:08x}"
            or not isinstance(record.get("compressed_sha256"), str)
            or len(record["compressed_sha256"]) != 64
            or identity["bytes"] != record.get("uncompressed_bytes")
            or identity["sha256"] != record.get("sha256")
            or identity["crc32"] != record.get("crc32")
        ):
            raise ValueError(f"existing retained member changed: {relative}")
    return manifest_path


def retain_hinsberger_selected(
    staging_root: Path,
    output_root: Path | None = None,
) -> Path:
    """Verify retained ranges, stream six members, and publish a selective manifest."""
    staging_root = staging_root.resolve()
    api_identity = _verify_api(staging_root)
    central_identity, _ = _verify_central_inventory(staging_root)
    store, range_records = _verified_range_store(staging_root)
    local_headers = [_verify_local_header(store, member) for member in SELECTED_MEMBERS]
    recipe = {
        "schema": _SCHEMA,
        "dataset": {
            "doi": DATASET_DOI,
            "publisher_file_id": PUBLISHER_FILE_ID,
            "archive_name": ARCHIVE_NAME,
            "archive_bytes": ARCHIVE_BYTES,
            "publisher_md5": ARCHIVE_MD5,
            "license": {"spdx": LICENSE_SPDX, "url": LICENSE_URL},
        },
        "api": api_identity,
        "central_inventory": central_identity,
        "ranges": range_records,
        "local_headers": local_headers,
        "implementation": _implementation_identity(),
        "policy": {
            "whole_archive_md5_verified": False,
            "full_archive_equivalence": False,
            "selected_member_crc_required": True,
            "imagery_pixels_inspected": False,
            "training": False,
            "production": False,
            "preview": False,
        },
    }
    build_id = hashlib.sha256(_canonical_bytes(recipe)).hexdigest()
    parent = output_root or DATA_IN / "evidence" / "hinsberger" / "selective" / "sha256"
    final = parent / build_id
    manifest_path = final / "manifest.json"
    if manifest_path.is_file():
        return _validate_existing_transaction(final, recipe, build_id)
    staging = parent / f".{build_id}.tmp-{os.getpid()}"
    if staging.exists() or final.exists():
        raise FileExistsError(f"selective retention transaction already exists: {staging}")
    staging.mkdir(parents=True)
    (staging / "recipe.json").write_bytes(_canonical_bytes(recipe))
    (staging / "range-provenance.json").write_bytes(_canonical_bytes({"ranges": range_records}))
    extracted = [
        _extract_member(store, member, staging / "members" / member.path)
        for member in SELECTED_MEMBERS
    ]
    manifest = {
        "schema": _SCHEMA,
        "build_id": build_id,
        "status": "research_only_selected_members_retained",
        "dataset": recipe["dataset"],
        "archive_identity_boundary": {
            "official_file_id": PUBLISHER_FILE_ID,
            "official_total_bytes": ARCHIVE_BYTES,
            "publisher_md5": ARCHIVE_MD5,
            "whole_archive_md5_verified": False,
            "full_archive_equivalence": False,
            "central_directory_bound": True,
            "selected_local_headers_bound": True,
        },
        "byte_accounting": {
            "selected_compressed_bytes": SELECTED_COMPRESSED_BYTES,
            "selected_compressed_bytes_reused": REUSED_SELECTED_COMPRESSED_BYTES,
            "selected_compressed_bytes_downloaded": MISSING_COMPRESSED_BYTES
            + NEW_SELECTED_BYTES_IN_HEADER_PROBE,
            "selected_compressed_bytes_remaining": 0,
            "new_transport_bytes": MISSING_COMPRESSED_BYTES + (HEADER_PROBE_RANGE[1] - HEADER_PROBE_RANGE[0] + 1),
        },
        "recipe_sha256": hashlib.sha256((staging / "recipe.json").read_bytes()).hexdigest(),
        "range_provenance": {
            "path": "range-provenance.json",
            "sha256": hashlib.sha256((staging / "range-provenance.json").read_bytes()).hexdigest(),
        },
        "selected_members": extracted,
        "authority": {
            "research_only_selected_member_qualification": True,
            "whole_archive_retention": False,
            "absolute_height_truth": False,
            "imagery_pixel_inspection": False,
            "training": False,
            "production": False,
            "preview": False,
        },
    }
    (staging / "manifest.json").write_bytes(_canonical_bytes(manifest))
    final.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(final)
    return manifest_path


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify retained Figshare ranges and stream six selected Hinsberger ZIP members."
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=DATA_IN / "evidence" / "hinsberger" / "staging",
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    try:
        manifest = retain_hinsberger_selected(args.staging_root, args.output_root)
    except (FileNotFoundError, FileExistsError, ValueError, json.JSONDecodeError, zlib.error) as error:
        parser.exit(2, f"Hinsberger selective retention blocked: {error}\n")
    print(manifest)


if __name__ == "__main__":
    _main()
