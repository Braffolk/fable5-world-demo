"""Deterministic machine artifacts for later transfer QA rendering."""
from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def write_machine_product(
    output_dir: Path,
    *,
    metrics: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    sources: Mapping[str, Any],
) -> Path:
    output_dir = Path(os.path.abspath(os.fspath(output_dir)))
    if output_dir.exists():
        raise FileExistsError(f"Hovi transfer output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    try:
        metrics_path = output_dir / "metrics.json"
        _write_bytes(metrics_path, canonical_json_bytes(metrics))
        arrays_path = output_dir / "observables.npz"
        _write_npz(arrays_path, arrays)
        manifest = {
            "schema_version": "hovi-full-vs-thinned-common-grid-product/1.0.0",
            "status": "complete_unqualified_numeric_comparison",
            "metrics": _reference(metrics_path),
            "observables": _reference(arrays_path),
            "sources": sources,
            "qa_pngs_generated": False,
            "synthesis_authorized": False,
            "target_truth": False,
        }
        manifest_path = output_dir / "manifest.json"
        _write_bytes(manifest_path, canonical_json_bytes(manifest))
        return manifest_path
    except BaseException:
        # Preserve a failed directory for forensic inspection; it can never masquerade as complete.
        raise


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for name in sorted(arrays):
            array = np.asarray(arrays[name])
            if array.dtype.hasobject:
                raise ValueError("Hovi transfer observables may not contain object arrays")
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o400 << 16
            with archive.open(info, "w", force_zip64=True) as member:
                np.lib.format.write_array(member, array, allow_pickle=False)
    with path.open("rb") as source:
        os.fsync(source.fileno())


def _reference(path: Path) -> dict[str, Any]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
