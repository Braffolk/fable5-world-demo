"""Safely extract and inspect the sole authorized Mrzezyno DEM."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import rasterio

from ...config import DATA_WORK
from ...fetch.mrzezyno import build_retention_plan

_SCHEMA = "mrzezyno-2022-02-extraction/1.0.0"
_RETAINED_SCHEMA = "mrzezyno-2022-02-retention/1.0.0"
_MEMBER = "2022-02/input/dem/2022-02-28.tif"
_MEMBER_BYTES = 192_715_409
_MEMBER_CRC32 = 0xF05538C1
_MEMBER_SHA256 = "717e92d7875f02177b7a8608014e92c82c0d28874b7afff2e70f5bac75207f44"
_EXPECTED_SHAPE = (4_621, 10_423)
_EXPECTED_BOUNDS = (255021.84751, 703255.8242, 256064.14751, 703717.9242)


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("xb") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def _verified_archive(retained_manifest: Path) -> tuple[Path, dict[str, Any], str]:
    encoded = retained_manifest.read_bytes()
    retained = json.loads(encoded)
    plan, retention_id = build_retention_plan()
    if (
        not isinstance(retained, Mapping)
        or retained.get("schema_version") != _RETAINED_SCHEMA
        or retained.get("status") != "complete"
        or retained.get("retention_id") != retention_id
        or retained.get("plan") != plan
    ):
        raise ValueError("Mrzezyno retention manifest differs from the frozen plan")
    artifact = retained.get("artifacts", {}).get("archive")
    if not isinstance(artifact, Mapping) or artifact.get("verified") is not True:
        raise ValueError("Mrzezyno archive is not verified")
    relative = artifact.get("path")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Mrzezyno archive path is invalid")
    archive = (retained_manifest.parent / relative).resolve()
    if (
        not archive.is_relative_to(retained_manifest.parent.resolve())
        or archive.stat().st_size != plan["archive"]["bytes"]
        or sha256_file(archive) != plan["archive"]["sha256"]
    ):
        raise ValueError("Mrzezyno retained archive bytes changed")
    return archive, retained, hashlib.sha256(encoded).hexdigest()


def extract_mrzezyno_dem(retained_manifest: Path, output_root: Path | None = None) -> Path:
    retained_manifest = retained_manifest.resolve()
    archive, retained, retained_sha256 = _verified_archive(retained_manifest)
    recipe = {
        "schema_version": _SCHEMA,
        "source_archive_sha256": retained["artifacts"]["archive"]["sha256"],
        "retained_manifest_sha256": retained_sha256,
        "member": {
            "path": _MEMBER,
            "bytes": _MEMBER_BYTES,
            "crc32": f"{_MEMBER_CRC32:08x}",
            "sha256": _MEMBER_SHA256,
        },
        "policy": {
            "extract_selected_member_only": True,
            "declared_nodata_preserved": True,
            "undeclared_extremes_preserved_as_invalid": True,
        },
    }
    build_id = hashlib.sha256(canonical_json(recipe)).hexdigest()
    root = (output_root or DATA_WORK / "microtopography" / "mrzezyno" / "source" / "sha256") / build_id
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        return manifest_path
    root.mkdir(parents=True, exist_ok=False)
    atomic_bytes(root / "recipe.json", canonical_json(recipe))
    destination = root / "2022-02-28.tif"
    temporary = destination.with_name(destination.name + ".part")
    try:
        with zipfile.ZipFile(archive) as source:
            infos = source.infolist()
            names = [item.filename for item in infos]
            if len(names) != len(set(names)):
                raise ValueError("Mrzezyno archive contains duplicate member paths")
            if any(
                Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name
                for name in names
            ):
                raise ValueError("Mrzezyno archive contains an unsafe member path")
            matches = [item for item in infos if item.filename == _MEMBER]
            if len(matches) != 1:
                raise ValueError("Mrzezyno DEM member is absent or ambiguous")
            info = matches[0]
            if (
                info.file_size != _MEMBER_BYTES
                or info.CRC != _MEMBER_CRC32
                or info.flag_bits & 0x1
            ):
                raise ValueError("Mrzezyno DEM member tuple changed or is encrypted")
            digest = hashlib.sha256()
            byte_count = 0
            with source.open(info) as reader, temporary.open("xb") as writer:
                for block in iter(lambda: reader.read(8 << 20), b""):
                    byte_count += len(block)
                    if byte_count > _MEMBER_BYTES:
                        raise ValueError("Mrzezyno DEM extraction exceeded exact size")
                    digest.update(block)
                    writer.write(block)
                writer.flush()
                os.fsync(writer.fileno())
            if byte_count != _MEMBER_BYTES or digest.hexdigest() != _MEMBER_SHA256:
                raise ValueError("Mrzezyno DEM member failed exact verification")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    extreme_cells: list[dict[str, Any]] = []
    declared_valid = 0
    with rasterio.open(destination) as dataset:
        if (
            dataset.driver != "GTiff"
            or dataset.count != 1
            or dataset.dtypes != ("float32",)
            or dataset.crs is None
            or dataset.crs.to_epsg() != 2180
            or (dataset.height, dataset.width) != _EXPECTED_SHAPE
            or not np.allclose((dataset.transform.a, dataset.transform.e), (0.1, -0.1), atol=1e-12)
            or not np.allclose(tuple(dataset.bounds), _EXPECTED_BOUNDS, atol=1e-8)
            or dataset.nodata != np.finfo(np.float32).min
        ):
            raise ValueError("Mrzezyno DEM raster identity changed")
        for _, window in dataset.block_windows(1):
            values = dataset.read(1, window=window)
            valid = (values != dataset.nodata) & np.isfinite(values)
            declared_valid += int(np.count_nonzero(valid))
            extreme_y, extreme_x = np.nonzero(valid & ((values < 0.0) | (values > 20.0)))
            for row, column in zip(extreme_y, extreme_x, strict=True):
                global_row = int(window.row_off + row)
                global_column = int(window.col_off + column)
                x, y = dataset.xy(global_row, global_column)
                extreme_cells.append(
                    {
                        "row": global_row,
                        "column": global_column,
                        "xy_m": [float(x), float(y)],
                        "value_m": float(values[row, column]),
                    }
                )
        raster = {
            "driver": dataset.driver,
            "dtype": dataset.dtypes[0],
            "crs": str(dataset.crs),
            "shape_yx": [dataset.height, dataset.width],
            "pixel_xy_m": [float(dataset.transform.a), float(dataset.transform.e)],
            "bounds_xy_m": [float(value) for value in dataset.bounds],
            "nodata": float(dataset.nodata),
            "declared_valid_cells": declared_valid,
            "declared_valid_fraction": declared_valid / (dataset.height * dataset.width),
            "undeclared_extreme_cells": extreme_cells,
            "area_or_point": dataset.tags().get("AREA_OR_POINT"),
        }
    if declared_valid != 7_799_608 or len(extreme_cells) != 3:
        raise ValueError("Mrzezyno DEM valid/extreme cell inventory changed")
    manifest = {
        "schema_version": _SCHEMA,
        "status": "complete",
        "build_id": build_id,
        "recipe_sha256": build_id,
        "retained_manifest_sha256": retained_sha256,
        "artifact": {
            "path": destination.relative_to(root).as_posix(),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        },
        "raster": raster,
        "surface_evidence": {
            "provider_ground_class_interpolated_by_idw": True,
            "original_points_present": False,
            "per_cell_measured_support_present": False,
            "interpolation_distance_present": False,
            "dry_sand_semantic_mask_present": False,
        },
    }
    atomic_bytes(manifest_path, canonical_json(manifest))
    return manifest_path


def _main() -> None:
    parser = argparse.ArgumentParser(description="Extract the exact Mrzezyno 2022-02 DEM.")
    parser.add_argument("retained_manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(extract_mrzezyno_dem(args.retained_manifest, args.output_root))


if __name__ == "__main__":
    _main()
