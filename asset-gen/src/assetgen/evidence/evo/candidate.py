"""Convert exact Evo plot-1086 bytes into lean raw candidate evidence."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import laspy
import numpy as np
from PIL import Image

from ...config import DATA_IN, DATA_WORK
from ...fetch.evo import build_evo_plot_retention_plan
from .fallback_1065 import load_evo_1065_selection
from .qa import render_candidate_qa
from .selection import load_evo_selection
from .source import inspect_evo_source, iter_evo_points

_RETAINED_SCHEMA = "evo-plot-retention/1.0.0"
_BUILD_SCHEMA = "evo-raw-candidate-build/1.0.0"
_QA_SCHEMA = "evo-raw-candidate-qa/1.0.0"
_CELL_M = 0.0625
_PLOT_SIZE_M = 32.0


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


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Evo {label} must be an object")
    return value


def _retained_source(
    manifest_path: Path, plot_id: str
) -> tuple[Path, dict[str, Any], str]:
    plan = build_evo_plot_retention_plan(
        authorize_exact_plot=True, plot_id=plot_id
    )
    encoded = manifest_path.read_bytes()
    retained = _mapping(json.loads(encoded), "retained plot manifest")
    artifact = _mapping(retained.get("artifact"), "retained plot artifact")
    expected = plan.identity["artifact"]
    if (
        retained.get("schema_version") != _RETAINED_SCHEMA
        or retained.get("status") != "complete"
        or retained.get("retention_id") != plan.retention_id
        or retained.get("plan_sha256") != plan.retention_id
        or retained.get("authorized_scope") != f"exact_plot_{plot_id}_point_cloud_only"
        or retained.get("selector_config_retention_authorized") is not False
        or retained.get("signed_url_persisted") is not False
        or any(artifact.get(key) != value for key, value in expected.items())
        or artifact.get("verified") is not True
    ):
        raise ValueError("Evo retained plot manifest differs from the exact plan")
    relative = artifact.get("relative_path")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Evo retained plot path is invalid")
    source_path = (manifest_path.parent / relative).resolve()
    if (
        not source_path.is_relative_to(manifest_path.parent.resolve())
        or not source_path.is_file()
        or source_path.stat().st_size != expected["bytes"]
        or _sha256_file(source_path) != expected["sha256"]
    ):
        raise ValueError("Evo retained plot bytes failed frozen verification")
    return source_path, dict(retained), hashlib.sha256(encoded).hexdigest()


def _update_counter(counter: Counter[int], values: np.ndarray) -> None:
    unique, counts = np.unique(values, return_counts=True)
    counter.update({int(key): int(count) for key, count in zip(unique, counts, strict=True)})


def _finite(array: np.ndarray) -> np.ndarray:
    output = array.astype(np.float32)
    output[~np.isfinite(output)] = np.nan
    return output


def build_evo_raw_candidate(
    retained_manifest: Path,
    output_root: Path | None = None,
    *,
    plot_id: str = "1086",
    log=print,
) -> Path:
    source_path, retained, retained_sha256 = _retained_source(
        retained_manifest.resolve(), plot_id
    )
    selection = (
        load_evo_selection() if plot_id == "1086" else load_evo_1065_selection()
    )
    center = json.loads(selection.path.read_bytes())["selection"]["center"]["epsg3067"]
    center_x = float(center["easting_m"])
    center_y = float(center["northing_m"])
    minimum_x = center_x - _PLOT_SIZE_M / 2
    minimum_y = center_y - _PLOT_SIZE_M / 2
    width = int(round(_PLOT_SIZE_M / _CELL_M))
    recipe = {
        "schema_version": _BUILD_SCHEMA,
        "implementation": {
            "candidate_py_sha256": _sha256_file(Path(__file__)),
            "source_py_sha256": _sha256_file(Path(__file__).with_name("source.py")),
            "qa_py_sha256": _sha256_file(Path(__file__).with_name("qa.py")),
        },
        "source": {
            "sha256": retained["artifact"]["sha256"],
            "bytes": retained["artifact"]["bytes"],
            "retained_manifest_sha256": retained_sha256,
            "selection_config_sha256": selection.config_sha256,
        },
        "plot": {
            "plot_id": plot_id,
            "crs": "EPSG:3067",
            "vertical_datum": "N2000",
            "center_xy_m": [center_x, center_y],
            "bounds_min_inclusive_max_exclusive_xy_m": [
                minimum_x,
                minimum_y,
                minimum_x + _PLOT_SIZE_M,
                minimum_y + _PLOT_SIZE_M,
            ],
        },
        "grid": {"cell_m": _CELL_M, "shape_yx": [width, width], "north_up": True},
        "populations": {
            "all_return": "every decoded point inside the official plot square",
            "unassigned_return": "decoded points with source treeid=0; not a ground class",
            "source_ground_reference": "decoded z minus source-provided h; not independent floor truth",
        },
        "arrays": [
            "all_return_count",
            "unassigned_return_count",
            "unassigned_nearest_cell_center_m",
            "unassigned_lower_observed_z_m",
            "unassigned_upper_observed_z_m",
            "source_ground_reference_z_mean_m",
            "source_ground_reference_z_range_m",
        ],
        "qualification": {
            "role": "raw_candidate",
            "status": "unqualified",
            "target_truth": False,
            "synthesis_authorized": False,
        },
    }
    recipe_bytes = _canonical_json(recipe)
    recipe_sha256 = hashlib.sha256(recipe_bytes).hexdigest()
    build_id = recipe_sha256
    build_root = (
        output_root
        or DATA_WORK / "microtopography" / "evo" / "candidate" / "sha256"
    ) / build_id
    manifest_path = build_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_bytes())
        if manifest.get("status") != "complete" or manifest.get("build_id") != build_id:
            raise ValueError("existing Evo candidate build conflicts with the recipe")
        return manifest_path
    if build_root.exists() and any(build_root.iterdir()):
        raise ValueError("incomplete Evo candidate build requires inspection")
    build_root.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(build_root / "recipe.json", recipe_bytes)

    shape = (width, width)
    all_count = np.zeros(shape, dtype=np.uint64)
    unassigned_count = np.zeros(shape, dtype=np.uint64)
    nearest = np.full(shape, np.inf, dtype=np.float64)
    lower = np.full(shape, np.inf, dtype=np.float64)
    upper = np.full(shape, -np.inf, dtype=np.float64)
    reference_sum = np.zeros(shape, dtype=np.float64)
    reference_min = np.full(shape, np.inf, dtype=np.float64)
    reference_max = np.full(shape, -np.inf, dtype=np.float64)
    tree_ids: Counter[int] = Counter()
    returns: Counter[int] = Counter()
    return_counts: Counter[int] = Counter()
    classifications: Counter[int] = Counter()
    actual_count = 0
    inside_count = 0
    observed_min = np.full(3, np.inf, dtype=np.float64)
    observed_max = np.full(3, -np.inf, dtype=np.float64)

    for points in iter_evo_points(source_path):
        x = np.asarray(points.x)
        y = np.asarray(points.y)
        z = np.asarray(points.z)
        h = np.asarray(points.h)
        treeid = np.asarray(points.treeid)
        count = len(points)
        actual_count += count
        observed_min = np.minimum(observed_min, (x.min(), y.min(), z.min()))
        observed_max = np.maximum(observed_max, (x.max(), y.max(), z.max()))
        _update_counter(tree_ids, treeid)
        _update_counter(returns, np.asarray(points.return_number))
        _update_counter(return_counts, np.asarray(points.number_of_returns))
        _update_counter(classifications, np.asarray(points.classification))

        ix = np.floor((x - minimum_x) / _CELL_M).astype(np.int64)
        iy = np.floor((y - minimum_y) / _CELL_M).astype(np.int64)
        inside = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < width)
        if not np.any(inside):
            continue
        inside_count += int(np.count_nonzero(inside))
        ix = ix[inside]
        iy = iy[inside]
        flat = iy * width + ix
        counts = np.bincount(flat, minlength=all_count.size)
        all_count.ravel()[:] += counts.astype(np.uint64, copy=False)
        ground_reference = z[inside] - h[inside]
        reference_sum.ravel()[:] += np.bincount(
            flat, weights=ground_reference, minlength=all_count.size
        )
        np.minimum.at(reference_min.ravel(), flat, ground_reference)
        np.maximum.at(reference_max.ravel(), flat, ground_reference)

        unassigned = treeid[inside] == 0
        if not np.any(unassigned):
            continue
        uflat = flat[unassigned]
        ux = x[inside][unassigned]
        uy = y[inside][unassigned]
        uz = z[inside][unassigned]
        ucounts = np.bincount(uflat, minlength=unassigned_count.size)
        unassigned_count.ravel()[:] += ucounts.astype(np.uint64, copy=False)
        np.minimum.at(lower.ravel(), uflat, uz)
        np.maximum.at(upper.ravel(), uflat, uz)
        center_x_m = minimum_x + (ix[unassigned] + 0.5) * _CELL_M
        center_y_m = minimum_y + (iy[unassigned] + 0.5) * _CELL_M
        distance = np.hypot(ux - center_x_m, uy - center_y_m)
        np.minimum.at(nearest.ravel(), uflat, distance)
        if actual_count % 25_000_000 < count:
            log(f"decoded Evo points: {actual_count:,}")

    if actual_count <= 0 or inside_count <= 0:
        raise ValueError("Evo sequential recovery produced no usable observations")
    reference_mean = np.divide(
        reference_sum,
        all_count,
        out=np.full(shape, np.nan, dtype=np.float64),
        where=all_count > 0,
    )
    reference_range = reference_max - reference_min
    reference_range[all_count == 0] = np.nan
    lower[~np.isfinite(lower)] = np.nan
    upper[~np.isfinite(upper)] = np.nan
    nearest[~np.isfinite(nearest)] = np.nan
    arrays = {
        "all_return_count": all_count,
        "unassigned_return_count": unassigned_count,
        "unassigned_nearest_cell_center_m": _finite(nearest),
        "unassigned_lower_observed_z_m": _finite(lower),
        "unassigned_upper_observed_z_m": _finite(upper),
        "source_ground_reference_z_mean_m": _finite(reference_mean),
        "source_ground_reference_z_range_m": _finite(reference_range),
    }
    artifact_path = build_root / f"{plot_id}-raw-candidate.npz"
    temporary = artifact_path.with_name(artifact_path.name + ".part")
    with temporary.open("wb") as target:
        np.savez_compressed(target, **arrays)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(artifact_path)
    artifact_sha256 = _sha256_file(artifact_path)

    qa_records = render_candidate_qa(arrays, build_root / "qa")
    images = []
    for index, record in enumerate(qa_records, start=1):
        path = record["path"]
        with Image.open(path) as image:
            dimensions = list(image.size)
        images.append(
            {
                "index": index,
                "path": path.relative_to(build_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "dimensions_xy": dimensions,
                "interpretation": record["interpretation"],
            }
        )

    occupied = all_count > 0
    unassigned_occupied = unassigned_count > 0
    source_header = inspect_evo_source(source_path)
    inventory = {
        "source_header": source_header,
        "decoded": {
            "actual_point_count": actual_count,
            "actual_min_xyz_m": observed_min.tolist(),
            "actual_max_xyz_m": observed_max.tolist(),
            "inside_official_plot_count": inside_count,
            "outside_official_plot_count": actual_count - inside_count,
            "treeid_counts": {str(key): value for key, value in sorted(tree_ids.items())},
            "return_number_counts": {str(key): value for key, value in sorted(returns.items())},
            "number_of_returns_counts": {
                str(key): value for key, value in sorted(return_counts.items())
            },
            "classification_counts": {
                str(key): value for key, value in sorted(classifications.items())
            },
        },
        "grid": {
            "all_return_points": int(all_count.sum()),
            "all_return_occupied_cells": int(np.count_nonzero(occupied)),
            "all_return_occupied_fraction": float(np.mean(occupied)),
            "unassigned_return_points": int(unassigned_count.sum()),
            "unassigned_occupied_cells": int(np.count_nonzero(unassigned_occupied)),
            "unassigned_occupied_fraction": float(np.mean(unassigned_occupied)),
        },
        "qualification": recipe["qualification"],
    }
    inventory_bytes = _canonical_json(inventory)
    inventory_path = build_root / "inventory.json"
    _atomic_bytes(inventory_path, inventory_bytes)
    qa_index = {
        "schema_version": _QA_SCHEMA,
        "build_id": build_id,
        "recipe_sha256": recipe_sha256,
        "source_sha256": retained["artifact"]["sha256"],
        "artifact": {
            "path": artifact_path.relative_to(build_root).as_posix(),
            "bytes": artifact_path.stat().st_size,
            "sha256": artifact_sha256,
        },
        "images": images,
        "qualification": recipe["qualification"],
    }
    qa_index_path = build_root / "qa" / "index.json"
    _atomic_bytes(qa_index_path, _canonical_json(qa_index))
    manifest = {
        "schema_version": _BUILD_SCHEMA,
        "status": "complete",
        "build_id": build_id,
        "recipe_sha256": recipe_sha256,
        "source_sha256": retained["artifact"]["sha256"],
        "environment": {
            "numpy": np.__version__,
            "laspy": laspy.__version__,
            "lazrs": importlib.metadata.version("lazrs"),
            "pillow": importlib.metadata.version("pillow"),
        },
        "artifacts": {
            "candidate_npz": qa_index["artifact"],
            "inventory": {
                "path": inventory_path.relative_to(build_root).as_posix(),
                "bytes": inventory_path.stat().st_size,
                "sha256": hashlib.sha256(inventory_bytes).hexdigest(),
            },
            "qa_index": {
                "path": qa_index_path.relative_to(build_root).as_posix(),
                "bytes": qa_index_path.stat().st_size,
                "sha256": _sha256_file(qa_index_path),
            },
        },
        "qualification": recipe["qualification"],
    }
    _atomic_bytes(manifest_path, _canonical_json(manifest))
    log(f"Evo raw candidate build: {build_id}")
    log(f"Evo raw candidate manifest: {manifest_path}")
    return manifest_path


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build an exact Evo plot raw candidate.")
    parser.add_argument("--plot-id", choices=("1086", "1065"), default="1086")
    parser.add_argument(
        "--retained",
        type=Path,
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    plan = build_evo_plot_retention_plan(
        authorize_exact_plot=True, plot_id=args.plot_id
    )
    retained = args.retained or (
        DATA_IN / "evidence" / "evo" / plan.retention_id / "retained.json"
    )
    build_evo_raw_candidate(
        retained, args.output_root, plot_id=args.plot_id
    )


if __name__ == "__main__":
    _main()
