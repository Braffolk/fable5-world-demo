"""Retain the exact ForestSemantic-MS v1 corpus from Zenodo."""
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

_DOI = "10.5281/zenodo.17172162"
_RECORD_ID = 17_172_162
_RECORD_API = f"https://zenodo.org/api/records/{_RECORD_ID}"
_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/legalcode.txt"
_SCHEMA = "forestsemantic-ms-retention/1.0.0"
_RESERVED_FREE_BYTES = 64 << 30

_FILES = (
    ("train1.laz", "train", 20_990_528, "8d24c31df17ef58e58fd25f8bad23238"),
    ("train2.laz", "train", 42_592_079, "b24cf74f9feba6dce5b4014c95a6312c"),
    ("train3.laz", "train", 16_087_942, "5f360b14e2720482fe9916ca702973d0"),
    ("train4.laz", "train", 25_325_276, "b29fd542abbe35dd606793531d43ce12"),
    ("test1.laz", "test", 7_762_914, "2c36cc760602d75fee4bd466f7da9a8f"),
    ("test2.laz", "test", 18_345_926, "1476d103d889dfdde238c76a9253e8ff"),
)


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


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def build_forestsemantic_plan() -> tuple[dict[str, Any], str]:
    files = [
        {
            "filename": name,
            "split": split,
            "bytes": size,
            "published_checksum": {"type": "MD5", "value": md5},
            "url": f"{_RECORD_API}/files/{name}/content",
        }
        for name, split, size, md5 in _FILES
    ]
    plan = {
        "schema_version": _SCHEMA,
        "dataset": {
            "doi": _DOI,
            "record_id": _RECORD_ID,
            "version": "v1",
            "published": "2025-12-03",
            "publisher": "Zenodo",
        },
        "files": files,
        "split_contract": {
            "train": [name for name, split, _, _ in _FILES if split == "train"],
            "test": [name for name, split, _, _ in _FILES if split == "test"],
            "publisher_defined": True,
        },
        "license": {
            "spdx": "CC-BY-4.0",
            "legalcode_url": _LICENSE_URL,
            "attribution": (
                "Takhtkeshha et al. (2025), ForestSemantic-MS Dataset, "
                "doi:10.5281/zenodo.17172162"
            ),
        },
        "authorization": {
            "scope": "all_six_publisher_laz_files",
            "semantic_audit_only": True,
            "model_training_in_this_step": False,
            "target_height_or_error_claim": False,
            "max_network_bytes": sum(item[2] for item in _FILES) + (2 << 20),
        },
    }
    return plan, hashlib.sha256(_canonical_json(plan)).hexdigest()


def _validate_record(raw: Mapping[str, Any], plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    metadata = raw.get("metadata")
    files = raw.get("files")
    if (
        raw.get("id") != _RECORD_ID
        or raw.get("doi") != _DOI
        or not isinstance(metadata, Mapping)
        or metadata.get("version") != "v1"
        or metadata.get("publication_date") != "2025-12-03"
        or metadata.get("license") != {"id": "cc-by-4.0"}
        or not isinstance(files, list)
    ):
        raise ValueError("ForestSemantic-MS controlling record drifted")
    by_name = {item.get("key"): item for item in files if isinstance(item, Mapping)}
    if set(by_name) != {item[0] for item in _FILES}:
        raise ValueError("ForestSemantic-MS file inventory drifted")
    verified = []
    for expected in plan["files"]:
        item = by_name[expected["filename"]]
        links = item.get("links")
        if (
            item.get("size") != expected["bytes"]
            or item.get("checksum")
            != "md5:" + expected["published_checksum"]["value"]
            or not isinstance(links, Mapping)
            or links.get("self") != expected["url"]
        ):
            raise ValueError(f"ForestSemantic-MS tuple drifted: {expected['filename']}")
        verified.append(
            {
                "filename": expected["filename"],
                "bytes": item["size"],
                "checksum": item["checksum"],
                "url": links["self"],
            }
        )
    return verified


def _download_exact(
    session: requests.Session, url: str, path: Path, expected_bytes: int, expected_md5: str
) -> tuple[str, str]:
    if path.exists():
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"retained file has wrong size: {path}")
        md5, sha256 = _hashes(path)
        if md5 != expected_md5:
            raise ValueError(f"retained file failed publisher MD5: {path}")
        return md5, sha256
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    if temporary.exists():
        raise ValueError(f"stale staging file requires inspection: {temporary}")
    response = session.get(url, timeout=(30, 300), stream=True)
    response.raise_for_status()
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    byte_count = 0
    try:
        with temporary.open("xb") as target:
            for block in response.iter_content(8 << 20):
                if not block:
                    continue
                byte_count += len(block)
                if byte_count > expected_bytes:
                    raise ValueError(f"download exceeded exact size: {path.name}")
                target.write(block)
                md5.update(block)
                sha256.update(block)
            target.flush()
            os.fsync(target.fileno())
        if byte_count != expected_bytes or md5.hexdigest() != expected_md5:
            raise ValueError(f"download differs from publisher tuple: {path.name}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return md5.hexdigest(), sha256.hexdigest()


def retain_forestsemantic_ms(output_root: Path | None = None, *, log=print) -> Path:
    plan, retention_id = build_forestsemantic_plan()
    root = (output_root or DATA_IN / "evidence" / "forestsemantic_ms") / retention_id
    manifest_path = root / "retained.json"
    if manifest_path.exists():
        retained = json.loads(manifest_path.read_bytes())
        if retained.get("retention_id") != retention_id or retained.get("status") != "complete":
            raise ValueError("existing ForestSemantic-MS retention conflicts")
        return manifest_path

    free_before = shutil.disk_usage(root.parent if root.parent.exists() else DATA_IN).free
    if free_before < _RESERVED_FREE_BYTES + plan["authorization"]["max_network_bytes"]:
        raise RuntimeError("ForestSemantic-MS retention would breach the 64 GiB reserve")
    cfg = load_base().fetch
    session = requests.Session()
    session.headers["User-Agent"] = cfg.user_agent
    record_response = session.get(_RECORD_API, timeout=(30, 120))
    record_response.raise_for_status()
    record_bytes = record_response.content
    record = json.loads(record_bytes)
    remote_files = _validate_record(record, plan)
    _atomic_bytes(root / "publisher" / "zenodo-record-17172162.json", record_bytes)

    license_response = session.get(_LICENSE_URL, timeout=(30, 120))
    license_response.raise_for_status()
    license_bytes = license_response.content
    if b"Creative Commons Attribution 4.0 International Public License" not in license_bytes:
        raise ValueError("CC BY 4.0 legal code response was not recognized")
    _atomic_bytes(root / "publisher" / "CC-BY-4.0-legalcode.txt", license_bytes)

    artifacts = []
    for item in plan["files"]:
        path = root / "files" / item["split"] / item["filename"]
        md5, sha256 = _download_exact(
            session,
            item["url"],
            path,
            item["bytes"],
            item["published_checksum"]["value"],
        )
        artifacts.append(
            {
                **item,
                "relative_path": path.relative_to(root).as_posix(),
                "md5": md5,
                "sha256": sha256,
                "verified": True,
            }
        )
        log(f"verified ForestSemantic-MS {item['split']} file: {item['filename']}")

    manifest = {
        "schema_version": _SCHEMA,
        "status": "complete",
        "retention_id": retention_id,
        "retained_utc": datetime.now(timezone.utc).isoformat(),
        "free_bytes_before": free_before,
        "plan": plan,
        "publisher_snapshot": {
            "relative_path": "publisher/zenodo-record-17172162.json",
            "bytes": len(record_bytes),
            "sha256": hashlib.sha256(record_bytes).hexdigest(),
            "remote_files": remote_files,
        },
        "license_snapshot": {
            "relative_path": "publisher/CC-BY-4.0-legalcode.txt",
            "bytes": len(license_bytes),
            "sha256": hashlib.sha256(license_bytes).hexdigest(),
        },
        "artifacts": artifacts,
        "qualification": {
            "state": "retained_raw_candidate",
            "semantic_audit_authorized": True,
            "absolute_target_height_or_error_authorized": False,
        },
    }
    _atomic_bytes(manifest_path, _canonical_json(manifest))
    log(f"ForestSemantic-MS retention manifest: {manifest_path}")
    return manifest_path


def _main() -> None:
    parser = argparse.ArgumentParser(description="Retain exact ForestSemantic-MS v1 files.")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(retain_forestsemantic_ms(args.output_root))


if __name__ == "__main__":
    _main()
