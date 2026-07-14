"""Retain the exact authorized Mrzezyno 2022-02 DEM archive and evidence paper."""
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

_SELECTION = (
    ASSET_GEN_ROOT.parent
    / "docs/deep-research/microtopography-generation/review/contracts/"
    "replacement-target-selection.2026-07-14.json"
)
_SELECTION_SHA256 = "fa2437b80b712a0b6c05ffae80b96dcc571a999f8c763e1c1d8fa14d3e44f2c7"
_PROTOCOL = ASSET_GEN_ROOT.parent / "docs/specs/terrain/PUBLIC-TARGET-QUALIFICATION-PROTOCOL.md"
_PROTOCOL_SHA256 = "43210c64cd12a8dc51260f80303e80e7c9b723391b2185bf6ebe9406c245e2dd"
_API_URL = "https://zenodo.org/api/records/15476933"
_ARCHIVE_URL = "https://zenodo.org/api/records/15476933/files/2022-02.zip/content"
_ARCHIVE_NAME = "2022-02.zip"
_ARCHIVE_BYTES = 22_575_152
_ARCHIVE_MD5 = "4d3ac450101e46eb67efde2458f37ca4"
_ARCHIVE_SHA256 = "7e1ad058ca4f951603780f2798681cea7c3ad91b1be9b0bdfac1a81a23092f18"
_PAPER_URL = "https://sgp.web.amu.edu.pl/LA44/landfana.044.003.pdf"
_PAPER_NAME = "landfana.044.003.pdf"
_PAPER_BYTES = 6_073_287
_PAPER_SHA256 = "7c34d76ad3688977ff6de506a5bc67739dc4d8defde2533a7f31a9e0c0a57ee5"
_SCHEMA = "mrzezyno-2022-02-retention/1.0.0"


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hashes(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            md5.update(block)
            sha256.update(block)
    return md5.hexdigest(), sha256.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("xb") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def _copy_verified(source: Path, destination: Path, expected_bytes: int, expected_sha256: str) -> None:
    if source.stat().st_size != expected_bytes or _sha256_file(source) != expected_sha256:
        raise ValueError(f"operator-supplied evidence differs from frozen identity: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    with source.open("rb") as reader, temporary.open("xb") as writer:
        shutil.copyfileobj(reader, writer, 8 << 20)
        writer.flush()
        os.fsync(writer.fileno())
    temporary.replace(destination)


def _download_verified(
    session: requests.Session,
    url: str,
    destination: Path,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    try:
        with session.get(url, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            digest = hashlib.sha256()
            byte_count = 0
            with temporary.open("xb") as target:
                for block in response.iter_content(8 << 20):
                    if not block:
                        continue
                    byte_count += len(block)
                    if byte_count > expected_bytes:
                        raise ValueError(f"download exceeded exact byte ceiling: {url}")
                    digest.update(block)
                    target.write(block)
                target.flush()
                os.fsync(target.fileno())
        if byte_count != expected_bytes or digest.hexdigest() != expected_sha256:
            raise ValueError(f"download differs from frozen identity: {url}")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _publisher_snapshot(session: requests.Session) -> dict[str, Any]:
    response = session.get(_API_URL, timeout=(30, 120))
    response.raise_for_status()
    raw = response.json()
    metadata = raw.get("metadata") if isinstance(raw, Mapping) else None
    files = raw.get("files") if isinstance(raw, Mapping) else None
    if not isinstance(metadata, Mapping) or not isinstance(files, list):
        raise ValueError("Zenodo publisher record is incomplete")
    matches = [item for item in files if isinstance(item, Mapping) and item.get("key") == _ARCHIVE_NAME]
    if (
        raw.get("id") != 15_476_933
        or raw.get("doi") != "10.5281/zenodo.15476933"
        or raw.get("status") != "published"
        or metadata.get("license", {}).get("id") != "cc-by-4.0"
        or len(matches) != 1
        or matches[0].get("size") != _ARCHIVE_BYTES
        or matches[0].get("checksum") != f"md5:{_ARCHIVE_MD5}"
        or matches[0].get("links", {}).get("self") != _ARCHIVE_URL
    ):
        raise ValueError("Zenodo publisher identity, license, or selected artifact drifted")
    return {
        "record_id": raw["id"],
        "doi": raw["doi"],
        "status": raw["status"],
        "publication_date": metadata.get("publication_date"),
        "title": metadata.get("title"),
        "license_id": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/legalcode",
        "artifact": {
            "file_id": matches[0].get("id"),
            "key": matches[0]["key"],
            "bytes": matches[0]["size"],
            "publisher_checksum": matches[0]["checksum"],
            "url": matches[0]["links"]["self"],
        },
    }


def build_retention_plan() -> tuple[dict[str, Any], str]:
    if _sha256_file(_SELECTION) != _SELECTION_SHA256:
        raise ValueError("replacement-target selection contract changed")
    if _sha256_file(_PROTOCOL) != _PROTOCOL_SHA256:
        raise ValueError("public-target qualification protocol changed")
    selection = json.loads(_SELECTION.read_bytes())
    decision = selection.get("decision", {})
    if (
        decision.get("selected_candidate_id") != "mrzezyno_baltic_dune_2022_02"
        or decision.get("authorized_band", {}).get("id") != "B1"
        or decision.get("authorized_band", {}).get("qualification") != "candidate_only"
    ):
        raise ValueError("replacement-target selection no longer authorizes this probe")
    plan = {
        "schema_version": _SCHEMA,
        "authority": {
            "selection_path": _SELECTION.relative_to(ASSET_GEN_ROOT.parent).as_posix(),
            "selection_sha256": _SELECTION_SHA256,
            "protocol_path": _PROTOCOL.relative_to(ASSET_GEN_ROOT.parent).as_posix(),
            "protocol_sha256": _PROTOCOL_SHA256,
        },
        "scope": {
            "selected_candidate_id": decision["selected_candidate_id"],
            "role": decision["authorized_role"],
            "band": "B1",
            "candidate_only": True,
            "bulk_geometry_downloaded": False,
        },
        "dataset": {
            "doi": "10.5281/zenodo.15476933",
            "record_id": 15_476_933,
            "official_record": "https://zenodo.org/records/15476933",
            "official_api_record": _API_URL,
            "license_id": "CC-BY-4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/legalcode",
        },
        "archive": {
            "name": _ARCHIVE_NAME,
            "url": _ARCHIVE_URL,
            "bytes": _ARCHIVE_BYTES,
            "md5": _ARCHIVE_MD5,
            "sha256": _ARCHIVE_SHA256,
        },
        "paper": {
            "name": _PAPER_NAME,
            "url": _PAPER_URL,
            "doi": "10.12657/landfana-044-003",
            "bytes": _PAPER_BYTES,
            "sha256": _PAPER_SHA256,
            "role": "campaign_processing_and_control_evidence_not_geometry",
        },
    }
    return plan, hashlib.sha256(_canonical_json(plan)).hexdigest()


def retain_mrzezyno_2022_02(
    output_root: Path | None = None,
    *,
    archive_source: Path | None = None,
    paper_source: Path | None = None,
) -> Path:
    plan, retention_id = build_retention_plan()
    root = (output_root or DATA_IN / "evidence" / "mrzezyno") / retention_id
    manifest_path = root / "retained.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_bytes())
        if manifest.get("retention_id") != retention_id or manifest.get("status") != "complete":
            raise ValueError("existing Mrzezyno retention conflicts")
        return manifest_path

    cfg = load_base().fetch
    session = requests.Session()
    session.headers["User-Agent"] = cfg.user_agent
    publisher = _publisher_snapshot(session)
    archive = root / "files" / _ARCHIVE_NAME
    paper = root / "evidence" / _PAPER_NAME
    if archive_source is None:
        _download_verified(session, _ARCHIVE_URL, archive, _ARCHIVE_BYTES, _ARCHIVE_SHA256)
    else:
        _copy_verified(archive_source.resolve(), archive, _ARCHIVE_BYTES, _ARCHIVE_SHA256)
    if paper_source is None:
        _download_verified(session, _PAPER_URL, paper, _PAPER_BYTES, _PAPER_SHA256)
    else:
        _copy_verified(paper_source.resolve(), paper, _PAPER_BYTES, _PAPER_SHA256)

    md5, archive_sha256 = _hashes(archive)
    if md5 != _ARCHIVE_MD5 or archive_sha256 != _ARCHIVE_SHA256:
        raise ValueError("retained Mrzezyno archive failed its complete checksum tuple")
    manifest = {
        "schema_version": _SCHEMA,
        "status": "complete",
        "retention_id": retention_id,
        "plan": plan,
        "publisher_snapshot": publisher,
        "accessed_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "archive": {
                "path": archive.relative_to(root).as_posix(),
                "bytes": archive.stat().st_size,
                "md5": md5,
                "sha256": archive_sha256,
                "verified": True,
            },
            "paper": {
                "path": paper.relative_to(root).as_posix(),
                "bytes": paper.stat().st_size,
                "sha256": _sha256_file(paper),
                "verified": True,
            },
        },
        "qualification": {
            "retained_raw_candidate": False,
            "reason": "release_contains_provider_interpolated_dem_not_original_point_observations",
            "b1_target_site": False,
            "synthesis_authorized": False,
            "estonia_transfer_authorized": False,
        },
    }
    _atomic_bytes(manifest_path, _canonical_json(manifest))
    return manifest_path


def _main() -> None:
    parser = argparse.ArgumentParser(description="Retain exact Mrzezyno 2022-02 evidence.")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--archive-source", type=Path)
    parser.add_argument("--paper-source", type=Path)
    args = parser.parse_args()
    print(
        retain_mrzezyno_2022_02(
            args.output_root,
            archive_source=args.archive_source,
            paper_source=args.paper_source,
        )
    )


if __name__ == "__main__":
    _main()
