"""Retain the exact OPARA-1038 plot sample and its small official records."""

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
    "replacement-target-selection-2.2026-07-14.json"
)
_SELECTION_SHA256 = "69d796842ce2661b31683920be66ae0b079f7c3b275f24f2326fada07b956f99"
_PROTOCOL = ASSET_GEN_ROOT.parent / "docs/specs/terrain/PUBLIC-TARGET-QUALIFICATION-PROTOCOL.md"
_PROTOCOL_SHA256 = "43210c64cd12a8dc51260f80303e80e7c9b723391b2185bf6ebe9406c245e2dd"

_ITEM_ID = "fba06620-4280-41b3-9f16-cc183c135f08"
_BUNDLE_ID = "1ef8b5ca-f3fb-44e5-9706-747827c7c9bb"
_ARCHIVE_ID = "9bb38664-5340-4c9d-ac88-eacfdb77ccee"
_DESCRIPTION_ID = "0bb34c31-e6d8-4446-a6e1-46cd72d9bf02"
_API_ROOT = "https://opara.zih.tu-dresden.de/server/api/core"
_ITEM_URL = f"{_API_ROOT}/items/{_ITEM_ID}"
_BITSTREAMS_URL = f"{_API_ROOT}/bundles/{_BUNDLE_ID}/bitstreams?size=100"
_ARCHIVE_URL = f"{_API_ROOT}/bitstreams/{_ARCHIVE_ID}/content"
_DESCRIPTION_URL = f"{_API_ROOT}/bitstreams/{_DESCRIPTION_ID}/content"
_PAPER_URL = "https://essd.copernicus.org/articles/18/1275/2026/essd-18-1275-2026.pdf"
_METHOD_URL = "https://soil.copernicus.org/articles/11/1007/2025/soil-11-1007-2025.pdf"
_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/legalcode.txt"

_ARCHIVE_NAME = "00_B_sample_plot.7z"
_ARCHIVE_BYTES = 3_307_259_536
_ARCHIVE_MD5 = "605952140081ef2d31d6c2a4a28182c5"
_DESCRIPTION_BYTES = 786_192
_DESCRIPTION_MD5 = "27c8bcb735e05a413812f29a0f147485"
_SCHEMA = "opara-1038-plot-sample-retention/1.0.0"
_RESERVED_FREE_BYTES = 64 << 30


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _hashes(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            md5.update(block)
            sha256.update(block)
    return md5.hexdigest(), sha256.hexdigest()


def _sha256_file(path: Path) -> str:
    return _hashes(path)[1]


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("xb") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def build_opara_plan() -> tuple[dict[str, Any], str]:
    if _sha256_file(_SELECTION) != _SELECTION_SHA256:
        raise ValueError("replacement-target selection contract changed")
    if _sha256_file(_PROTOCOL) != _PROTOCOL_SHA256:
        raise ValueError("public-target qualification protocol changed")
    selection = json.loads(_SELECTION.read_bytes())
    decision = selection.get("decision", {})
    if (
        decision.get("status") != "bounded_metadata_pass"
        or decision.get("selected_candidate_id")
        != "opara_eastern_germany_bare_cultivated_soil"
        or decision.get("authorized_band") != "B1_0.25_to_1m"
    ):
        raise ValueError("replacement-target selection no longer authorizes OPARA B1")
    plan = {
        "schema_version": _SCHEMA,
        "authority": {
            "selection_path": _SELECTION.relative_to(ASSET_GEN_ROOT.parent).as_posix(),
            "selection_sha256": _SELECTION_SHA256,
            "protocol_path": _PROTOCOL.relative_to(ASSET_GEN_ROOT.parent).as_posix(),
            "protocol_sha256": _PROTOCOL_SHA256,
        },
        "dataset": {
            "doi": "10.25532/OPARA-1038",
            "item_id": _ITEM_ID,
            "official_item_url": _ITEM_URL,
            "official_bitstreams_url": _BITSTREAMS_URL,
            "license": "CC-BY-4.0",
        },
        "archive": {
            "id": _ARCHIVE_ID,
            "name": _ARCHIVE_NAME,
            "url": _ARCHIVE_URL,
            "bytes": _ARCHIVE_BYTES,
            "publisher_md5": _ARCHIVE_MD5,
        },
        "small_records": [
            {
                "id": _DESCRIPTION_ID,
                "name": "Data-description-template.pdf",
                "url": _DESCRIPTION_URL,
                "bytes": _DESCRIPTION_BYTES,
                "publisher_md5": _DESCRIPTION_MD5,
            },
            {"name": "essd-18-1275-2026.pdf", "url": _PAPER_URL},
            {"name": "soil-11-1007-2025.pdf", "url": _METHOD_URL},
            {"name": "CC-BY-4.0-legalcode.txt", "url": _LICENSE_URL},
        ],
        "authorization": {
            "candidate_role": (
                "one_foreign_bare_cultivated_loess_plot_event_B1_development_probe"
            ),
            "inventory_before_extraction": True,
            "view_geometry_before_event_freeze": False,
            "bulk_plot_archive_authorized": False,
            "bulk_slope_archive_authorized": False,
            "synthesis_authorized": False,
            "estonia_transfer_authorized": False,
        },
    }
    encoded = _canonical_json(plan)
    return plan, hashlib.sha256(encoded).hexdigest()


def _publisher_snapshot(
    item: Mapping[str, Any], bitstreams: Mapping[str, Any]
) -> dict[str, Any]:
    metadata = item.get("metadata")
    embedded = bitstreams.get("_embedded")
    rows = embedded.get("bitstreams") if isinstance(embedded, Mapping) else None
    if not isinstance(metadata, Mapping) or not isinstance(rows, list):
        raise ValueError("OPARA publisher response is incomplete")
    by_name = {row.get("name"): row for row in rows if isinstance(row, Mapping)}
    archive = by_name.get(_ARCHIVE_NAME)
    description = by_name.get("Data-description-template.pdf")
    rights = metadata.get("dc.rights", [])
    rights_uri = metadata.get("dc.rights.uri", [])
    if (
        item.get("id") != _ITEM_ID
        or item.get("withdrawn") is not False
        or item.get("discoverable") is not True
        or not any(row.get("value") == "Attribution 4.0 International" for row in rights)
        or not any(row.get("value") == "http://creativecommons.org/licenses/by/4.0/" for row in rights_uri)
        or not isinstance(archive, Mapping)
        or archive.get("id") != _ARCHIVE_ID
        or archive.get("sizeBytes") != _ARCHIVE_BYTES
        or archive.get("checkSum")
        != {"checkSumAlgorithm": "MD5", "value": _ARCHIVE_MD5}
        or archive.get("_links", {}).get("content", {}).get("href") != _ARCHIVE_URL
        or not isinstance(description, Mapping)
        or description.get("id") != _DESCRIPTION_ID
        or description.get("sizeBytes") != _DESCRIPTION_BYTES
        or description.get("checkSum")
        != {"checkSumAlgorithm": "MD5", "value": _DESCRIPTION_MD5}
    ):
        raise ValueError("OPARA item, license, or selected bitstream tuple drifted")
    return {
        "item_id": _ITEM_ID,
        "name": item.get("name"),
        "last_modified": item.get("lastModified"),
        "withdrawn": False,
        "discoverable": True,
        "license": "CC-BY-4.0",
        "selected_archive": {
            "id": archive["id"],
            "name": archive["name"],
            "bytes": archive["sizeBytes"],
            "publisher_checksum": archive["checkSum"],
            "url": _ARCHIVE_URL,
        },
    }


def _download_small(
    session: requests.Session,
    url: str,
    destination: Path,
    *,
    expected_bytes: int | None = None,
    expected_md5: str | None = None,
) -> dict[str, Any]:
    response = session.get(url, timeout=(30, 300))
    response.raise_for_status()
    payload = response.content
    if expected_bytes is not None and len(payload) != expected_bytes:
        raise ValueError(f"small OPARA record byte count changed: {url}")
    md5 = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    if expected_md5 is not None and md5 != expected_md5:
        raise ValueError(f"small OPARA record MD5 changed: {url}")
    _atomic_bytes(destination, payload)
    return {
        "path": destination.name,
        "url": url,
        "bytes": len(payload),
        "md5": md5,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "content_type": response.headers.get("Content-Type"),
    }


def _download_archive(
    session: requests.Session, destination: Path, *, log=print
) -> tuple[str, str]:
    if destination.exists():
        if destination.stat().st_size != _ARCHIVE_BYTES:
            raise ValueError("retained OPARA archive has the wrong byte count")
        md5, sha256 = _hashes(destination)
        if md5 != _ARCHIVE_MD5:
            raise ValueError("retained OPARA archive failed publisher MD5")
        return md5, sha256

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    offset = temporary.stat().st_size if temporary.exists() else 0
    if offset > _ARCHIVE_BYTES:
        raise ValueError("OPARA partial exceeds the authorized byte count")
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with session.get(_ARCHIVE_URL, headers=headers, stream=True, timeout=(30, 600)) as response:
        if offset:
            if response.status_code != 206 or not response.headers.get(
                "Content-Range", ""
            ).startswith(f"bytes {offset}-"):
                raise ValueError("OPARA server did not honor the exact resume range")
        elif response.status_code != 200:
            response.raise_for_status()
        mode = "ab" if offset else "xb"
        written = offset
        with temporary.open(mode) as target:
            for block in response.iter_content(8 << 20):
                if not block:
                    continue
                written += len(block)
                if written > _ARCHIVE_BYTES:
                    raise ValueError("OPARA download exceeded the authorized byte count")
                target.write(block)
                if written // (256 << 20) != (written - len(block)) // (256 << 20):
                    log(f"OPARA sample retained {written / (1 << 30):.2f} GiB")
            target.flush()
            os.fsync(target.fileno())
    if written != _ARCHIVE_BYTES:
        raise ValueError(f"OPARA download incomplete at {written} bytes; rerun to resume")
    md5, sha256 = _hashes(temporary)
    if md5 != _ARCHIVE_MD5:
        raise ValueError("OPARA archive failed publisher MD5")
    temporary.replace(destination)
    return md5, sha256


def retain_opara_plot_sample(output_root: Path | None = None, *, log=print) -> Path:
    plan, retention_id = build_opara_plan()
    root = (output_root or DATA_IN / "evidence" / "opara_1038") / retention_id
    manifest_path = root / "retained.json"
    if manifest_path.exists():
        retained = json.loads(manifest_path.read_bytes())
        if retained.get("retention_id") != retention_id or retained.get("status") != "complete":
            raise ValueError("existing OPARA retention conflicts")
        return manifest_path

    free = shutil.disk_usage(root.parent if root.parent.exists() else DATA_IN).free
    if free < _RESERVED_FREE_BYTES + 2 * _ARCHIVE_BYTES:
        raise RuntimeError("OPARA retention would breach the 64 GiB reserve")
    cfg = load_base().fetch
    session = requests.Session()
    session.headers["User-Agent"] = cfg.user_agent

    item_response = session.get(_ITEM_URL, timeout=(30, 120))
    item_response.raise_for_status()
    bitstreams_response = session.get(_BITSTREAMS_URL, timeout=(30, 120))
    bitstreams_response.raise_for_status()
    item_bytes = item_response.content
    bitstreams_bytes = bitstreams_response.content
    publisher = _publisher_snapshot(json.loads(item_bytes), json.loads(bitstreams_bytes))
    _atomic_bytes(root / "publisher" / "item.json", item_bytes)
    _atomic_bytes(root / "publisher" / "bitstreams.json", bitstreams_bytes)

    records = []
    records.append(
        _download_small(
            session,
            _DESCRIPTION_URL,
            root / "records" / "Data-description-template.pdf",
            expected_bytes=_DESCRIPTION_BYTES,
            expected_md5=_DESCRIPTION_MD5,
        )
    )
    for name, url in (
        ("essd-18-1275-2026.pdf", _PAPER_URL),
        ("soil-11-1007-2025.pdf", _METHOD_URL),
        ("CC-BY-4.0-legalcode.txt", _LICENSE_URL),
    ):
        records.append(_download_small(session, url, root / "records" / name))

    archive = root / "files" / _ARCHIVE_NAME
    md5, sha256 = _download_archive(session, archive, log=log)
    manifest = {
        "schema_version": _SCHEMA,
        "status": "complete",
        "retention_id": retention_id,
        "retained_utc": datetime.now(timezone.utc).isoformat(),
        "plan": plan,
        "publisher_snapshot": {
            **publisher,
            "item_sha256": hashlib.sha256(item_bytes).hexdigest(),
            "bitstreams_sha256": hashlib.sha256(bitstreams_bytes).hexdigest(),
        },
        "small_records": records,
        "archive": {
            "path": archive.relative_to(root).as_posix(),
            "bytes": archive.stat().st_size,
            "md5": md5,
            "sha256": sha256,
            "verified": True,
        },
        "qualification": {
            "state": "retained_uninspected_candidate",
            "archive_inventory_authorized": True,
            "geometry_viewed": False,
            "target_surface_authorized": False,
            "synthesis_authorized": False,
        },
    }
    _atomic_bytes(manifest_path, _canonical_json(manifest))
    return manifest_path


def _main() -> None:
    parser = argparse.ArgumentParser(description="Retain the exact OPARA-1038 plot sample.")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(retain_opara_plot_sample(args.output_root))


if __name__ == "__main__":
    _main()
