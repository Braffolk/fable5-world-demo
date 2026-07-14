"""Verify and extract the exact Biala Gora first-epoch LAS member."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Mapping

from ...config import DATA_IN, DATA_WORK
from ...fetch.biala_gora import build_biala_retention_plan

_RETAINED_SCHEMA = "biala-gora-first-epoch-retention/1.0.0"
_SCHEMA = "biala-gora-first-epoch-extraction/1.0.0"
_MEMBER = "2022-02-27.las"
_MEMBER_BYTES = 605_800_031
_MEMBER_CRC32 = 0xC55BC53E
_RESERVED_FREE_BYTES = 64 << 30


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as target:
        target.write(_canonical_json(value))
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def extract_biala_las(retained_manifest: Path, output_root: Path | None = None, *, log=print) -> Path:
    plan, retention_id = build_biala_retention_plan()
    retained_manifest = retained_manifest.resolve()
    retained = json.loads(retained_manifest.read_bytes())
    if (
        not isinstance(retained, Mapping)
        or retained.get("schema_version") != _RETAINED_SCHEMA
        or retained.get("status") != "complete"
        or retained.get("retention_id") != retention_id
        or retained.get("plan") != plan
    ):
        raise ValueError("Biala Gora retained manifest differs from the exact plan")
    artifact = retained.get("artifact")
    if not isinstance(artifact, Mapping) or artifact.get("verified") is not True:
        raise ValueError("Biala Gora retained artifact is not verified")
    relative = artifact.get("relative_path")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Biala Gora retained archive path is invalid")
    archive = (retained_manifest.parent / relative).resolve()
    if (
        not archive.is_relative_to(retained_manifest.parent)
        or not archive.is_file()
        or archive.stat().st_size != plan["artifact"]["bytes"]
        or _sha256_file(archive) != artifact.get("sha256")
    ):
        raise ValueError("Biala Gora retained archive bytes changed")

    identity = {
        "schema_version": _SCHEMA,
        "archive_sha256": artifact["sha256"],
        "retained_manifest_sha256": _sha256_file(retained_manifest),
        "member": {
            "name": _MEMBER,
            "bytes": _MEMBER_BYTES,
            "crc32": f"{_MEMBER_CRC32:08x}",
            "compression": "deflate",
        },
        "implementation_sha256": _sha256_file(Path(__file__)),
        "qualification": plan["qualification"],
    }
    extraction_id = hashlib.sha256(_canonical_json(identity)).hexdigest()
    root = (
        output_root
        or DATA_WORK / "microtopography" / "biala_gora" / "source" / "sha256"
    ) / extraction_id
    manifest_path = root / "manifest.json"
    destination = root / _MEMBER
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_bytes())
        if manifest.get("status") != "complete" or manifest.get("extraction_id") != extraction_id:
            raise ValueError("existing Biala Gora extraction conflicts with the recipe")
        return manifest_path
    if root.exists() and any(root.iterdir()):
        raise ValueError("incomplete Biala Gora extraction requires inspection")
    free_before = shutil.disk_usage(root.parent if root.parent.exists() else DATA_WORK).free
    if free_before < _RESERVED_FREE_BYTES + _MEMBER_BYTES:
        raise RuntimeError("Biala Gora extraction would breach the 64 GiB reserve")

    with zipfile.ZipFile(archive) as source_zip:
        infos = source_zip.infolist()
        if len(infos) != 1:
            raise ValueError("Biala Gora archive is not the inspected single-member ZIP")
        info = infos[0]
        if (
            info.filename != _MEMBER
            or info.file_size != _MEMBER_BYTES
            or info.CRC != _MEMBER_CRC32
            or info.compress_type != zipfile.ZIP_DEFLATED
            or info.flag_bits & 0x1
            or info.extra
            or info.comment
            or source_zip.comment
        ):
            raise ValueError("Biala Gora ZIP member metadata changed")
        root.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".part")
        digest = hashlib.sha256()
        byte_count = 0
        try:
            with source_zip.open(info) as source, temporary.open("xb") as target:
                while block := source.read(8 << 20):
                    byte_count += len(block)
                    if byte_count > _MEMBER_BYTES:
                        raise ValueError("Biala Gora LAS exceeded its exact byte ceiling")
                    target.write(block)
                    digest.update(block)
                target.flush()
                os.fsync(target.fileno())
            if byte_count != _MEMBER_BYTES:
                raise ValueError("Biala Gora LAS extraction ended early")
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()
    member_sha256 = digest.hexdigest()
    manifest = {
        "schema_version": _SCHEMA,
        "status": "complete",
        "extraction_id": extraction_id,
        "identity": identity,
        "free_bytes_before": free_before,
        "source_archive_changed": False,
        "member": {
            "path": destination.relative_to(root).as_posix(),
            "bytes": destination.stat().st_size,
            "crc32": f"{_MEMBER_CRC32:08x}",
            "sha256": member_sha256,
            "verified": True,
        },
        "qualification": plan["qualification"],
    }
    _atomic_json(manifest_path, manifest)
    log(f"Biala Gora LAS SHA-256: {member_sha256}")
    log(f"Biala Gora extraction manifest: {manifest_path}")
    return manifest_path


def _main() -> None:
    _, retention_id = build_biala_retention_plan()
    parser = argparse.ArgumentParser(description="Extract exact Biala Gora first-epoch LAS.")
    parser.add_argument(
        "--retained",
        type=Path,
        default=DATA_IN / "evidence" / "biala_gora" / retention_id / "retained.json",
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(extract_biala_las(args.retained, args.output_root))


if __name__ == "__main__":
    _main()
