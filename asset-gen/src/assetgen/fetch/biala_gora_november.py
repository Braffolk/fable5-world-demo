"""Retain the exact November Biala Gora epoch selected for a bounded probe."""
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

from ..config import DATA_IN, load_base

_DATASET_API = (
    "https://repod.icm.edu.pl/api/datasets/:persistentId/"
    "?persistentId=doi%3A10.18150%2FBHH1RC"
)
_DOWNLOAD_URL = "https://repod.icm.edu.pl/api/access/datafile/54443"
_DATASET_ID = 54373
_DATASET_VERSION_ID = 3367
_FILE_ID = 54443
_FILENAME = "2022-11-10.zip"
_BYTES = 509_963_696
_MD5 = "5f86b53f20bff55d2819e9e18546c57a"
_RESERVED_FREE_BYTES = 64 << 30
_MAX_CACHE_BYTES = 2 << 30
_SCHEMA = "biala-gora-november-probe-retention/1.0.0"


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


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


def build_retention_plan() -> tuple[dict[str, Any], str]:
    identity = {
        "schema_version": _SCHEMA,
        "authorization": {
            "scope": "exact_november_epoch_probe_only",
            "basis": "explicit_operator_selection_2026-07-16",
            "bulk_acquisition_authorized": False,
            "network_attempts": 1,
            "max_network_bytes": _BYTES,
            "max_cache_bytes": _MAX_CACHE_BYTES,
            "minimum_reserved_free_bytes": _RESERVED_FREE_BYTES,
        },
        "dataset": {
            "doi": "10.18150/BHH1RC",
            "dataset_id": _DATASET_ID,
            "dataset_version": "1.0",
            "dataset_version_id": _DATASET_VERSION_ID,
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
            "process_change_evidence": False,
        },
    }
    encoded = _canonical_json(identity)
    return identity, hashlib.sha256(encoded).hexdigest()


def _remote_file(session: requests.Session) -> dict[str, Any]:
    response = session.get(_DATASET_API, timeout=(30, 120))
    response.raise_for_status()
    raw = response.json()
    data = raw.get("data") if isinstance(raw, Mapping) else None
    version = data.get("latestVersion") if isinstance(data, Mapping) else None
    if (
        raw.get("status") != "OK"
        or data.get("id") != _DATASET_ID
        or not isinstance(version, Mapping)
        or version.get("id") != _DATASET_VERSION_ID
        or version.get("versionState") != "RELEASED"
    ):
        raise ValueError("RepOD controlling dataset version drifted")
    matches = [
        item
        for item in version.get("files", [])
        if isinstance(item, Mapping)
        and isinstance(item.get("dataFile"), Mapping)
        and item["dataFile"].get("id") == _FILE_ID
    ]
    if len(matches) != 1:
        raise ValueError("RepOD November file is absent or ambiguous")
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
        raise ValueError("RepOD November artifact tuple or license drifted")
    return {
        "dataset_id": data["id"],
        "dataset_version_id": version["id"],
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


def retain_november(output_root: Path | None = None, *, log=print) -> Path:
    plan, retention_id = build_retention_plan()
    root = (output_root or DATA_IN / "evidence" / "biala_gora") / retention_id
    archive = root / "files" / _FILENAME
    manifest_path = root / "retained.json"
    if _BYTES > _MAX_CACHE_BYTES:
        raise RuntimeError("November archive exceeds the 2 GiB cache ceiling")
    free_before = shutil.disk_usage(root.parent if root.parent.exists() else DATA_IN).free
    if free_before < _RESERVED_FREE_BYTES + _MAX_CACHE_BYTES:
        raise RuntimeError("November retention would breach the 64 GiB reserve")

    cfg = load_base().fetch
    session = requests.Session()
    session.headers["User-Agent"] = cfg.user_agent
    remote = _remote_file(session)
    if archive.exists():
        if archive.stat().st_size != _BYTES:
            raise ValueError("retained November archive has the wrong byte count")
        md5, sha256 = _hashes_file(archive)
        if md5 != _MD5:
            raise ValueError("retained November archive failed published MD5")
    else:
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_name(archive.name + ".part")
        if temporary.exists():
            raise ValueError("stale November staging archive requires inspection")
        response = session.get(_DOWNLOAD_URL, timeout=(30, 300), stream=True)
        if response.status_code != 200 or response.url != _DOWNLOAD_URL:
            raise RuntimeError("RepOD November download endpoint changed")
        if response.headers.get("Content-Length") != str(_BYTES):
            raise ValueError("RepOD November Content-Length drifted")
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
                        raise ValueError("RepOD November archive exceeded its exact byte ceiling")
                    target.write(block)
                    md5_digest.update(block)
                    sha256_digest.update(block)
                target.flush()
                os.fsync(target.fileno())
            md5 = md5_digest.hexdigest()
            sha256 = sha256_digest.hexdigest()
            if byte_count != _BYTES or md5 != _MD5:
                raise ValueError("RepOD November archive differs from the published tuple")
            temporary.replace(archive)
        finally:
            if temporary.exists():
                temporary.unlink()
    manifest = {
        "schema_version": _SCHEMA,
        "status": "complete",
        "retention_id": retention_id,
        "plan": plan,
        "remote_preflight": remote,
        "accessed_utc": datetime.now(timezone.utc).isoformat(),
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
    log(f"November archive SHA-256: {sha256}")
    log(f"November retention manifest: {manifest_path}")
    return manifest_path


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(retain_november(args.output_root))


if __name__ == "__main__":
    _main()
