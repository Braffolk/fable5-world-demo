"""Deterministic artifact I/O for Hovi candidate-surface transactions."""
from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from ..records import sha256_bytes, sha256_file


def atomic_write(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as target:
        target.write(encoded)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True
    ) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            with archive.open(info, mode="w", force_zip64=True) as member:
                np.lib.format.write_array(member, arrays[name], allow_pickle=False)
    with path.open("rb") as source:
        os.fsync(source.fileno())


def store_content_file(build_root: Path, source: Path, filename: str) -> tuple[Path, str]:
    digest = sha256_file(source)
    destination = build_root / "artifacts" / "sha256" / digest / filename
    if destination.exists():
        if (
            destination.stat().st_size != source.stat().st_size
            or sha256_file(destination) != digest
        ):
            raise ValueError(f"corrupt Hovi surface artifact: {destination}")
        source.unlink()
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
    return destination, digest


def store_content_bytes(
    build_root: Path, encoded: bytes, filename: str
) -> tuple[Path, str]:
    digest = sha256_bytes(encoded)
    destination = build_root / "artifacts" / "sha256" / digest / filename
    if destination.exists():
        if destination.read_bytes() != encoded:
            raise ValueError(f"corrupt Hovi surface artifact: {destination}")
    else:
        atomic_write(destination, encoded)
    return destination, digest


def artifact_ref(build_root: Path, path: Path, digest: str) -> dict[str, Any]:
    return {
        "path": path.relative_to(build_root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest,
    }
