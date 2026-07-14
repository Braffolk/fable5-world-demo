"""Materialize one lean Biala Gora coastal-process candidate artifact."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import laspy
import numpy as np
from PIL import Image

from ...config import DATA_WORK
from .qa import render_candidate_qa

_EXTRACTION_SCHEMA = "biala-gora-first-epoch-extraction/1.0.0"
_SCHEMA = "biala-gora-process-candidate/1.0.0"
_QA_SCHEMA = "biala-gora-process-candidate-qa/1.0.0"
_CELL_M = 0.5
_CLASS_GROUND = 2


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as target:
        target.write(encoded)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def _source(extraction_manifest: Path) -> tuple[Path, dict[str, Any], str]:
    encoded = extraction_manifest.read_bytes()
    raw = json.loads(encoded)
    if (
        not isinstance(raw, Mapping)
        or raw.get("schema_version") != _EXTRACTION_SCHEMA
        or raw.get("status") != "complete"
        or raw.get("source_archive_changed") is not False
    ):
        raise ValueError("Biala Gora extraction manifest is not complete")
    member = raw.get("member")
    qualification = raw.get("qualification")
    if (
        not isinstance(member, Mapping)
        or member.get("verified") is not True
        or qualification
        != {
            "role": "calibration_only",
            "target_truth": False,
            "synthesis_authorized": False,
            "estonia_transfer_authorized": False,
        }
    ):
        raise ValueError("Biala Gora extraction qualification changed")
    relative = member.get("path")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Biala Gora extracted LAS path is invalid")
    path = (extraction_manifest.parent / relative).resolve()
    if (
        not path.is_relative_to(extraction_manifest.parent.resolve())
        or not path.is_file()
        or path.stat().st_size != member.get("bytes")
        or _sha256_file(path) != member.get("sha256")
    ):
        raise ValueError("Biala Gora extracted LAS bytes changed")
    return path, dict(raw), hashlib.sha256(encoded).hexdigest()


def _vlr(vlr: Any) -> dict[str, Any]:
    payload = vlr.record_data_bytes()
    return {
        "user_id": vlr.user_id,
        "record_id": int(vlr.record_id),
        "description": vlr.description,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _counter_update(counter: Counter[int], values: np.ndarray) -> None:
    unique, counts = np.unique(values, return_counts=True)
    counter.update({int(key): int(value) for key, value in zip(unique, counts, strict=True)})


def _finite_float32(values: np.ndarray) -> np.ndarray:
    output = values.astype(np.float32)
    output[~np.isfinite(output)] = np.nan
    return output


def build_biala_candidate(
    extraction_manifest: Path,
    output_root: Path | None = None,
    *,
    log=print,
) -> Path:
    source, extraction, extraction_sha256 = _source(extraction_manifest.resolve())
    with laspy.open(source) as reader:
        header = reader.header
        dimensions = tuple(header.point_format.dimension_names)
        required = {
            "X", "Y", "Z", "classification", "return_number", "number_of_returns",
            "red", "green", "blue",
        }
        crs = header.parse_crs()
        if (
            str(header.version) != "1.2"
            or int(header.point_format.id) != 3
            or int(header.point_format.size) != 34
            or int(header.point_count) != 17_817_615
            or not required.issubset(dimensions)
            or crs is None
            or crs.to_epsg() != 2180
        ):
            raise ValueError("Biala Gora LAS schema differs from the inspected source")
        minimum = np.asarray(header.mins, dtype=np.float64)
        maximum = np.asarray(header.maxs, dtype=np.float64)
        origin_x = math.floor(minimum[0] / _CELL_M) * _CELL_M
        origin_y = math.floor(minimum[1] / _CELL_M) * _CELL_M
        width = math.ceil((maximum[0] - origin_x) / _CELL_M)
        height = math.ceil((maximum[1] - origin_y) / _CELL_M)
        area_m2 = (maximum[0] - minimum[0]) * (maximum[1] - minimum[1])
        header_record = {
            "las_version": str(header.version),
            "point_format": int(header.point_format.id),
            "point_record_bytes": int(header.point_format.size),
            "point_count": int(header.point_count),
            "scales": [float(value) for value in header.scales],
            "offsets": [float(value) for value in header.offsets],
            "declared_min_xyz_m": minimum.tolist(),
            "declared_max_xyz_m": maximum.tolist(),
            "dimensions": list(dimensions),
            "crs_epsg": crs.to_epsg(),
            "crs_wkt": crs.to_wkt(),
            "system_identifier": header.system_identifier,
            "generating_software": header.generating_software,
            "creation_date": header.creation_date.isoformat() if header.creation_date else None,
            "vlrs": [_vlr(vlr) for vlr in header.vlrs],
        }

    recipe = {
        "schema_version": _SCHEMA,
        "implementation": {
            "candidate_py_sha256": _sha256_file(Path(__file__)),
            "qa_py_sha256": _sha256_file(Path(__file__).with_name("qa.py")),
        },
        "source": {
            "las_sha256": extraction["member"]["sha256"],
            "las_bytes": extraction["member"]["bytes"],
            "extraction_manifest_sha256": extraction_sha256,
        },
        "grid": {
            "cell_m": _CELL_M,
            "origin_xy_m": [origin_x, origin_y],
            "shape_yx": [height, width],
            "crs": "EPSG:2180",
            "north_up": True,
            "selection_basis": (
                "header footprint mean density is descriptive only; 0.5 m retains process-scale "
                "structure without claiming 0.0625 m support"
            ),
            "header_mean_points_per_m2": float(header.point_count / area_m2),
        },
        "populations": {
            "all_return": "all decoded LAS records",
            "vendor_class2": "source classification=2; vendor ground candidate, not truth",
            "multi_return": "source number_of_returns greater than one",
            "rgb_return": "at least one nonzero source RGB channel",
        },
        "qualification": extraction["qualification"],
    }
    recipe_bytes = _canonical_json(recipe)
    build_id = hashlib.sha256(recipe_bytes).hexdigest()
    root = (
        output_root
        or DATA_WORK / "microtopography" / "biala_gora" / "candidate" / "sha256"
    ) / build_id
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_bytes())
        if manifest.get("status") != "complete" or manifest.get("build_id") != build_id:
            raise ValueError("existing Biala Gora candidate conflicts with the recipe")
        return manifest_path
    if root.exists() and any(root.iterdir()):
        raise ValueError("incomplete Biala Gora candidate requires inspection")
    root.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(root / "recipe.json", recipe_bytes)

    shape = (height, width)
    cells = height * width
    all_count = np.zeros(shape, dtype=np.uint32)
    ground_count = np.zeros(shape, dtype=np.uint32)
    multi_count = np.zeros(shape, dtype=np.uint32)
    rgb_count = np.zeros(shape, dtype=np.uint32)
    all_min = np.full(shape, np.inf, dtype=np.float64)
    all_max = np.full(shape, -np.inf, dtype=np.float64)
    ground_min = np.full(shape, np.inf, dtype=np.float64)
    ground_max = np.full(shape, -np.inf, dtype=np.float64)
    ground_sum = np.zeros(shape, dtype=np.float64)
    rgb_sum = np.zeros((3, height, width), dtype=np.float64)
    classifications: Counter[int] = Counter()
    return_numbers: Counter[int] = Counter()
    return_counts: Counter[int] = Counter()
    decoded = 0
    rgb_records = 0

    with laspy.open(source) as reader:
        for points in reader.chunk_iterator(1_000_000):
            x = np.asarray(points.x)
            y = np.asarray(points.y)
            z = np.asarray(points.z)
            ix = np.floor((x - origin_x) / _CELL_M).astype(np.int64)
            iy = np.floor((y - origin_y) / _CELL_M).astype(np.int64)
            inside = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
            if not np.all(inside):
                raise ValueError("Biala Gora declared bounds exclude decoded points")
            flat = iy * width + ix
            counts = np.bincount(flat, minlength=cells)
            all_count.ravel()[:] += counts.astype(np.uint32, copy=False)
            np.minimum.at(all_min.ravel(), flat, z)
            np.maximum.at(all_max.ravel(), flat, z)

            classification = np.asarray(points.classification)
            number_of_returns = np.asarray(points.number_of_returns)
            _counter_update(classifications, classification)
            _counter_update(return_numbers, np.asarray(points.return_number))
            _counter_update(return_counts, number_of_returns)
            multi = number_of_returns > 1
            multi_count.ravel()[:] += np.bincount(
                flat[multi], minlength=cells
            ).astype(np.uint32, copy=False)

            ground = classification == _CLASS_GROUND
            if np.any(ground):
                gflat = flat[ground]
                gz = z[ground]
                ground_count.ravel()[:] += np.bincount(
                    gflat, minlength=cells
                ).astype(np.uint32, copy=False)
                ground_sum.ravel()[:] += np.bincount(gflat, weights=gz, minlength=cells)
                np.minimum.at(ground_min.ravel(), gflat, gz)
                np.maximum.at(ground_max.ravel(), gflat, gz)

            channels = np.stack(
                (
                    np.asarray(points.red, dtype=np.uint16) >> 8,
                    np.asarray(points.green, dtype=np.uint16) >> 8,
                    np.asarray(points.blue, dtype=np.uint16) >> 8,
                )
            )
            rgb_valid = np.any(channels != 0, axis=0)
            if np.any(rgb_valid):
                rflat = flat[rgb_valid]
                rgb_records += int(np.count_nonzero(rgb_valid))
                rgb_count.ravel()[:] += np.bincount(
                    rflat, minlength=cells
                ).astype(np.uint32, copy=False)
                for channel in range(3):
                    rgb_sum[channel].ravel()[:] += np.bincount(
                        rflat,
                        weights=channels[channel, rgb_valid],
                        minlength=cells,
                    )
            decoded += len(points)
            log(f"decoded Biala Gora points: {decoded:,}")
    if decoded != header.point_count or int(all_count.sum(dtype=np.uint64)) != decoded:
        raise ValueError("Biala Gora decoded point accounting differs from the header")

    all_range = all_max - all_min
    all_range[all_count == 0] = np.nan
    ground_mean = np.divide(
        ground_sum,
        ground_count,
        out=np.full(shape, np.nan, dtype=np.float64),
        where=ground_count > 0,
    )
    ground_range = ground_max - ground_min
    ground_range[ground_count == 0] = np.nan
    rgb_mean = np.divide(
        rgb_sum,
        rgb_count[None, ...],
        out=np.zeros(rgb_sum.shape, dtype=np.float64),
        where=rgb_count[None, ...] > 0,
    )
    arrays = {
        "all_return_count": all_count,
        "vendor_class2_count": ground_count,
        "multi_return_count": multi_count,
        "rgb_return_count": rgb_count,
        "rgb_mean_u8": np.moveaxis(np.rint(rgb_mean).astype(np.uint8), 0, -1),
        "all_return_z_range_m": _finite_float32(all_range),
        "vendor_class2_z_mean_m": _finite_float32(ground_mean),
        "vendor_class2_z_min_m": _finite_float32(ground_min),
        "vendor_class2_z_max_m": _finite_float32(ground_max),
        "vendor_class2_z_range_m": _finite_float32(ground_range),
    }
    artifact_path = root / "2022-02-27-process-candidate.npz"
    temporary = artifact_path.with_name(artifact_path.name + ".part")
    with temporary.open("wb") as target:
        np.savez_compressed(target, **arrays)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(artifact_path)

    qa_records = render_candidate_qa(arrays, root / "qa", cell_m=_CELL_M)
    images = []
    for index, record in enumerate(qa_records, start=1):
        path = record["path"]
        with Image.open(path) as image:
            dimensions_xy = list(image.size)
        images.append(
            {
                "index": index,
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "dimensions_xy": dimensions_xy,
                "interpretation": record["interpretation"],
            }
        )

    inventory = {
        "header": header_record,
        "decoded": {
            "point_count": decoded,
            "classification_counts": {
                str(key): value for key, value in sorted(classifications.items())
            },
            "return_number_counts": {
                str(key): value for key, value in sorted(return_numbers.items())
            },
            "number_of_returns_counts": {
                str(key): value for key, value in sorted(return_counts.items())
            },
            "rgb_nonzero_records": rgb_records,
        },
        "grid": {
            **recipe["grid"],
            "all_return_occupied_cells": int(np.count_nonzero(all_count)),
            "vendor_class2_occupied_cells": int(np.count_nonzero(ground_count)),
            "rgb_occupied_cells": int(np.count_nonzero(rgb_count)),
            "all_return_points": int(all_count.sum(dtype=np.uint64)),
            "vendor_class2_points": int(ground_count.sum(dtype=np.uint64)),
            "multi_return_points": int(multi_count.sum(dtype=np.uint64)),
        },
        "evidence_boundary": {
            "vendor_class2_is_authoritative_ground": False,
            "lowest_return_used_as_target": False,
            "interpolation_used_in_candidate_arrays": False,
            "display_only_nearest_fill_in_qa": True,
            "independent_error_distribution_available": False,
            "heightfield_validity_qualified": False,
        },
        "qualification": recipe["qualification"],
    }
    inventory_path = root / "inventory.json"
    inventory_bytes = _canonical_json(inventory)
    _atomic_bytes(inventory_path, inventory_bytes)
    qa_index = {
        "schema_version": _QA_SCHEMA,
        "build_id": build_id,
        "recipe_sha256": build_id,
        "source_sha256": extraction["member"]["sha256"],
        "artifact": {
            "path": artifact_path.relative_to(root).as_posix(),
            "bytes": artifact_path.stat().st_size,
            "sha256": _sha256_file(artifact_path),
        },
        "images": images,
        "qualification": recipe["qualification"],
    }
    qa_index_path = root / "qa" / "index.json"
    _atomic_bytes(qa_index_path, _canonical_json(qa_index))
    manifest = {
        "schema_version": _SCHEMA,
        "status": "complete",
        "build_id": build_id,
        "recipe_sha256": build_id,
        "source_sha256": extraction["member"]["sha256"],
        "environment": {
            "numpy": np.__version__,
            "laspy": laspy.__version__,
            "pillow": importlib.metadata.version("pillow"),
            "scipy": importlib.metadata.version("scipy"),
        },
        "artifacts": {
            "candidate_npz": qa_index["artifact"],
            "inventory": {
                "path": inventory_path.relative_to(root).as_posix(),
                "bytes": inventory_path.stat().st_size,
                "sha256": hashlib.sha256(inventory_bytes).hexdigest(),
            },
            "qa_index": {
                "path": qa_index_path.relative_to(root).as_posix(),
                "bytes": qa_index_path.stat().st_size,
                "sha256": _sha256_file(qa_index_path),
            },
        },
        "qualification": recipe["qualification"],
    }
    _atomic_bytes(manifest_path, _canonical_json(manifest))
    log(f"Biala Gora candidate build: {build_id}")
    log(f"Biala Gora candidate manifest: {manifest_path}")
    return manifest_path


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build Biala Gora process candidate evidence.")
    parser.add_argument("extraction_manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(build_biala_candidate(args.extraction_manifest, args.output_root))


if __name__ == "__main__":
    _main()
