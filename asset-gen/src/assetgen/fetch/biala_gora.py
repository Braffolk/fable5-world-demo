"""Retain the one authorized Biala Gora first-epoch archive from RepOD."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import requests

from ..config import ASSET_GEN_ROOT, DATA_IN, load_base

_DATASET_API = (
    "https://repod.icm.edu.pl/api/datasets/:persistentId/"
    "?persistentId=doi%3A10.18150%2FBHH1RC"
)
_DOWNLOAD_URL = "https://repod.icm.edu.pl/api/access/datafile/54447"
_DOI = "10.18150/BHH1RC"
_DATASET_ID = 54373
_DATASET_VERSION_ID = 3367
_FILE_ID = 54447
_FILENAME = "2022-02-27.zip"
_BYTES = 351_166_481
_MD5 = "c7a0df4c052ebfabcc40729c319f3614"
_SELECTION = (
    ASSET_GEN_ROOT.parent
    / "docs/deep-research/microtopography-generation/review/"
    "BIALA-GORA-FIRST-EPOCH-SELECTION.md"
)
_PROTOCOL = ASSET_GEN_ROOT.parent / "docs/specs/terrain/PUBLIC-TARGET-QUALIFICATION-PROTOCOL.md"
_SELECTION_SHA256 = "ec3a4c4e794c061e2bf677ff289013f313d9964e9ab604195200897a55eaa7d8"
_PROTOCOL_SHA256 = "43210c64cd12a8dc51260f80303e80e7c9b723391b2185bf6ebe9406c245e2dd"
_RESERVED_FREE_BYTES = 64 << 30
_MAX_CACHE_BYTES = 2 << 30
_SCHEMA = "biala-gora-first-epoch-retention/1.0.0"


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hashes_file(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            md5.update(block)
            sha256.update(block)
    return md5.hexdigest(), sha256.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as target:
        target.write(_canonical_json(value))
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def build_biala_retention_plan() -> tuple[dict[str, Any], str]:
    if _sha256_file(_SELECTION) != _SELECTION_SHA256:
        raise ValueError("Biala Gora frozen first-epoch selection changed")
    if _sha256_file(_PROTOCOL) != _PROTOCOL_SHA256:
        raise ValueError("public target qualification protocol changed")
    identity = {
        "schema_version": _SCHEMA,
        "authorization": {
            "scope": "exact_first_epoch_archive_only",
            "basis": "explicit_operator_authorization_2026-07-14",
            "bulk_acquisition_authorized": False,
            "network_attempts": 1,
            "max_network_bytes": _BYTES,
            "max_cache_bytes": _MAX_CACHE_BYTES,
            "minimum_reserved_free_bytes": _RESERVED_FREE_BYTES,
        },
        "authority": {
            "selection_path": _SELECTION.relative_to(ASSET_GEN_ROOT.parent).as_posix(),
            "selection_sha256": _SELECTION_SHA256,
            "protocol_path": _PROTOCOL.relative_to(ASSET_GEN_ROOT.parent).as_posix(),
            "protocol_sha256": _PROTOCOL_SHA256,
        },
        "dataset": {
            "doi": _DOI,
            "dataset_id": _DATASET_ID,
            "dataset_version": "1.0",
            "dataset_version_id": _DATASET_VERSION_ID,
            "published": "2025-01-24",
        },
        "artifact": {
            "file_id": _FILE_ID,
            "filename": _FILENAME,
            "bytes": _BYTES,
            "published_checksum": {"type": "MD5", "value": _MD5},
            "content_type": "application/zip",
            "restricted": False,
        },
        "license": {
            "spdx": "CC-BY-4.0",
            "url": "https://creativecommons.org/licenses/by/4.0/legalcode",
            "creator": "Sledziowski, Jakub",
        },
        "qualification": {
            "role": "calibration_only",
            "target_truth": False,
            "synthesis_authorized": False,
            "estonia_transfer_authorized": False,
        },
    }
    return identity, hashlib.sha256(_canonical_json(identity)).hexdigest()


def _exact_remote_file(session: requests.Session) -> dict[str, Any]:
    response = session.get(_DATASET_API, timeout=(30, 120))
    if response.status_code != 200:
        raise RuntimeError(f"RepOD metadata HTTP {response.status_code}")
    raw = response.json()
    if not isinstance(raw, Mapping) or raw.get("status") != "OK":
        raise ValueError("RepOD dataset metadata is not an OK object")
    data = raw.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("RepOD dataset metadata lacks data")
    version = data.get("latestVersion")
    if (
        data.get("id") != _DATASET_ID
        or not isinstance(version, Mapping)
        or version.get("id") != _DATASET_VERSION_ID
        or version.get("versionNumber") != 1
        or version.get("versionMinorNumber") != 0
        or version.get("versionState") != "RELEASED"
    ):
        raise ValueError("RepOD controlling dataset version drifted")
    files = version.get("files")
    if not isinstance(files, list):
        raise ValueError("RepOD file inventory is missing")
    matches = []
    for item in files:
        if isinstance(item, Mapping) and isinstance(item.get("dataFile"), Mapping):
            if item["dataFile"].get("id") == _FILE_ID:
                matches.append(item)
    if len(matches) != 1:
        raise ValueError("RepOD first-epoch file is absent or ambiguous")
    item = matches[0]
    artifact = item["dataFile"]
    expected = {
        "id": _FILE_ID,
        "filename": _FILENAME,
        "contentType": "application/zip",
        "filesize": _BYTES,
        "md5": _MD5,
    }
    if (
        any(artifact.get(key) != value for key, value in expected.items())
        or item.get("label") != _FILENAME
        or item.get("restricted") is not False
        or item.get("licenseUrl")
        != "https://creativecommons.org/licenses/by/4.0/legalcode"
    ):
        raise ValueError("RepOD first-epoch artifact tuple or license drifted")
    return {
        "dataset_id": data["id"],
        "dataset_version_id": version["id"],
        "version": "1.0",
        "version_state": version["versionState"],
        "release_time": version.get("releaseTime"),
        "file_id": artifact["id"],
        "filename": artifact["filename"],
        "bytes": artifact["filesize"],
        "md5": artifact["md5"],
        "content_type": artifact["contentType"],
        "restricted": item["restricted"],
        "license_url": item["licenseUrl"],
    }


def retain_biala_first_epoch(output_root: Path | None = None, *, log=print) -> Path:
    plan, retention_id = build_biala_retention_plan()
    root = (output_root or DATA_IN / "evidence" / "biala_gora") / retention_id
    archive = root / "files" / _FILENAME
    manifest_path = root / "retained.json"
    free_before = shutil.disk_usage(root.parent if root.parent.exists() else DATA_IN).free
    if free_before < _RESERVED_FREE_BYTES + _MAX_CACHE_BYTES:
        raise RuntimeError("Biala Gora retention would breach the 64 GiB free-space reserve")

    cfg = load_base().fetch
    session = requests.Session()
    session.headers["User-Agent"] = cfg.user_agent
    remote = _exact_remote_file(session)
    accessed = datetime.now(timezone.utc).isoformat()
    if archive.exists():
        if archive.stat().st_size != _BYTES:
            raise ValueError("retained Biala Gora archive has the wrong byte count")
        md5, sha256 = _hashes_file(archive)
        if md5 != _MD5:
            raise ValueError("retained Biala Gora archive failed published MD5")
    else:
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_name(archive.name + ".part")
        if temporary.exists():
            raise ValueError("stale Biala Gora staging archive requires inspection")
        response = session.get(_DOWNLOAD_URL, timeout=(30, 300), stream=True)
        if response.status_code != 200:
            raise RuntimeError(f"RepOD archive HTTP {response.status_code}")
        if response.url != _DOWNLOAD_URL:
            raise ValueError("RepOD archive endpoint redirected unexpectedly")
        if response.headers.get("Content-Length") != str(_BYTES):
            raise ValueError("RepOD archive Content-Length drifted")
        md5_digest = hashlib.md5(usedforsecurity=False)
        sha256_digest = hashlib.sha256()
        byte_count = 0
        try:
            with temporary.open("xb") as target:
                for block in response.iter_content(chunk_size=8 << 20):
                    if not block:
                        continue
                    byte_count += len(block)
                    if byte_count > _BYTES:
                        raise ValueError("RepOD archive exceeded its exact byte ceiling")
                    target.write(block)
                    md5_digest.update(block)
                    sha256_digest.update(block)
                target.flush()
                os.fsync(target.fileno())
            md5 = md5_digest.hexdigest()
            sha256 = sha256_digest.hexdigest()
            if byte_count != _BYTES or md5 != _MD5:
                raise ValueError("RepOD archive differs from the published tuple")
            temporary.replace(archive)
        finally:
            if temporary.exists():
                temporary.unlink()
        log(f"verified Biala Gora archive retained: {archive}")

    manifest = {
        "schema_version": _SCHEMA,
        "status": "complete",
        "retention_id": retention_id,
        "plan": plan,
        "remote_preflight": remote,
        "accessed_utc": accessed,
        "free_bytes_before": free_before,
        "artifact": {
            **plan["artifact"],
            "relative_path": archive.relative_to(root).as_posix(),
            "sha256": sha256,
            "verified": True,
        },
        "qualification": plan["qualification"],
    }
    _atomic_json(manifest_path, manifest)
    log(f"Biala Gora retention id: {retention_id}")
    log(f"Biala Gora archive SHA-256: {sha256}")
    return manifest_path


def _main() -> None:
    parser = argparse.ArgumentParser(description="Retain exact Biala Gora first epoch.")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(retain_biala_first_epoch(args.output_root))


if __name__ == "__main__":
    _main()
