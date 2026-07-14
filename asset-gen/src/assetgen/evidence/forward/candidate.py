"""Convert the retained FORWARD DTM into an unqualified morphology candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from scipy.ndimage import gaussian_filter

from ...config import DATA_IN, DATA_WORK
from .qa import render_candidate_qa
from .source import (
    canonical_json,
    inspect_source_raster,
    load_forward_selection,
    sha256_file,
    verify_source_bytes,
)

_PATCH_M = 32.0
_BASELINE_SIGMA_M = 2.0
_EDGE_GUARD_M = 32.0
_TARGET_QUANTILES = (0.10, 0.50, 0.90, 0.99)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def _retained_source(manifest_path: Path) -> tuple[Path, dict[str, Any], str]:
    selection = load_forward_selection()
    encoded = manifest_path.read_bytes()
    retained = json.loads(encoded)
    if (
        retained.get("schema_version") != "forward-dtm-retention/1.0.0"
        or retained.get("status") != "complete"
        or retained.get("retention_id") != selection.retention_id
        or retained.get("selection_sha256") != selection.selection_sha256
    ):
        raise ValueError("FORWARD retained manifest differs from the frozen selection")
    artifact = retained.get("artifact", {})
    relative = artifact.get("relative_path")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("FORWARD retained source path is invalid")
    source = (manifest_path.parent / relative).resolve()
    if not source.is_relative_to(manifest_path.parent.resolve()):
        raise ValueError("FORWARD retained source escaped its root")
    verify_source_bytes(source, selection)
    inspect_source_raster(source, selection)
    return source, retained, hashlib.sha256(encoded).hexdigest()


def _select_patches(height: np.ndarray, pixel_m: float, transform) -> tuple[dict[str, Any], ...]:
    patch_cells = int(round(_PATCH_M / pixel_m))
    guard_cells = int(round(_EDGE_GUARD_M / pixel_m))
    baseline = gaussian_filter(
        height.astype(np.float64), sigma=_BASELINE_SIGMA_M / pixel_m, mode="reflect"
    )
    site_residual = height.astype(np.float64) - baseline
    candidates: list[dict[str, Any]] = []
    for row in range(guard_cells, height.shape[0] - guard_cells - patch_cells + 1, patch_cells):
        for column in range(
            guard_cells, height.shape[1] - guard_cells - patch_cells + 1, patch_cells
        ):
            patch = height[row : row + patch_cells, column : column + patch_cells]
            residual = site_residual[row : row + patch_cells, column : column + patch_cells]
            candidates.append(
                {
                    "row_column": [row, column],
                    "residual_rms_m": float(np.sqrt(np.mean(residual * residual))),
                    "height_m": patch,
                    "residual_m": residual,
                }
            )
    ordered = sorted(candidates, key=lambda item: (item["residual_rms_m"], item["row_column"]))
    selected = []
    for quantile in _TARGET_QUANTILES:
        rank = round(quantile * (len(ordered) - 1))
        item = dict(ordered[rank])
        row, column = item["row_column"]
        west, north = transform * (column, row)
        east, south = transform * (column + patch_cells, row + patch_cells)
        item.update(
            {
                "target_quantile": quantile,
                "rank": rank,
                "population": len(ordered),
                "southwest_xy_m": [float(west), float(south)],
                "bounds_xy_m": [float(west), float(south), float(east), float(north)],
            }
        )
        selected.append(item)
    return tuple(selected)


def build_forward_candidate(
    retained_manifest: Path,
    output_root: Path | None = None,
) -> Path:
    source, retained, retained_sha256 = _retained_source(retained_manifest.resolve())
    selection = load_forward_selection()
    recipe = {
        "schema_version": "forward-dtm-candidate-recipe/1.0.0",
        "implementation": {
            "candidate_py_sha256": sha256_file(Path(__file__)),
            "qa_py_sha256": sha256_file(Path(__file__).with_name("qa.py")),
            "source_py_sha256": sha256_file(Path(__file__).with_name("source.py")),
        },
        "source": {
            "sha256": selection.artifact["sha256"],
            "retained_manifest_sha256": retained_sha256,
            "selection_sha256": selection.selection_sha256,
        },
        "conversion": {
            "source_resampled": False,
            "patches_are_native_grid_slices": True,
            "height_dtype": "float32",
            "patch_m": _PATCH_M,
            "patch_stride_m": _PATCH_M,
            "baseline": {
                "kind": "full-raster Gaussian",
                "sigma_m": _BASELINE_SIGMA_M,
                "boundary_mode": "reflect"
            },
            "edge_guard_m": _EDGE_GUARD_M,
            "edge_guard_reason": "exclude incomplete-neighborhood and visible provider raster-edge interpolation fans",
            "selection_metric": "per-patch RMS of height minus 2 m-sigma Gaussian baseline",
            "selection_quantiles": list(_TARGET_QUANTILES),
        },
        "qualification": selection.raw["qualification"],
    }
    recipe_bytes = canonical_json(recipe)
    build_id = hashlib.sha256(recipe_bytes).hexdigest()
    build_root = (
        output_root or DATA_WORK / "microtopography" / "forward" / "candidate" / "sha256"
    ) / build_id
    manifest_path = build_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_bytes())
        if manifest.get("status") != "complete" or manifest.get("build_id") != build_id:
            raise ValueError("existing FORWARD candidate build conflicts")
        return manifest_path
    if build_root.exists() and any(build_root.iterdir()):
        raise ValueError("incomplete FORWARD candidate build requires inspection")
    build_root.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(build_root / "recipe.json", recipe_bytes)

    with rasterio.open(source) as dataset:
        source_height = dataset.read(1)
        transform = dataset.transform
    if not np.all(np.isfinite(source_height)):
        raise ValueError("FORWARD DTM unexpectedly contains non-finite samples")
    height = source_height.astype(np.float32)
    conversion_error = np.abs(source_height - height.astype(np.float64))
    patches = _select_patches(height, abs(float(transform.a)), transform)
    arrays: dict[str, np.ndarray] = {}
    patch_records = []
    for index, patch in enumerate(patches, start=1):
        arrays[f"patch_{index:02d}_height_m"] = patch["height_m"]
        arrays[f"patch_{index:02d}_2m_sigma_residual_m"] = patch["residual_m"].astype(np.float32)
        patch_records.append(
            {
                key: value
                for key, value in patch.items()
                if key not in ("height_m", "residual_m")
            }
        )
    artifact_path = build_root / "mrv-harvest-dtm-patches.npz"
    temporary_artifact = artifact_path.with_name(artifact_path.name + ".part")
    with temporary_artifact.open("wb") as target:
        np.savez_compressed(target, **arrays)
        target.flush()
        os.fsync(target.fileno())
    temporary_artifact.replace(artifact_path)

    qa = render_candidate_qa(height, patches, build_root / "qa", pixel_m=abs(float(transform.a)))
    images = []
    for index, (path, interpretation) in enumerate(qa, start=1):
        with Image.open(path) as image:
            dimensions = list(image.size)
        images.append(
            {
                "index": index,
                "path": path.relative_to(build_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "dimensions_xy": dimensions,
                "interpretation": interpretation,
            }
        )
    inventory = {
        "source_raster": retained["raster"],
        "height": {
            "minimum_m": float(height.min()),
            "maximum_m": float(height.max()),
            "mean_m": float(height.mean(dtype=np.float64)),
            "stddev_m": float(height.std(dtype=np.float64)),
            "finite_cells": int(np.count_nonzero(np.isfinite(height))),
            "float64_to_float32_max_abs_error_m": float(conversion_error.max()),
        },
        "patch_population": {
            "count": patch_records[0]["population"],
            "patch_m": _PATCH_M,
            "non_overlapping": True,
            "edge_guard_m": _EDGE_GUARD_M,
            "selected": patch_records,
        },
        "provider_processing": selection.raw["provider_dtm_processing"],
        "evidence_limits": selection.raw["qualification"]["does_not_support"],
        "qualification": selection.raw["qualification"],
    }
    inventory_bytes = canonical_json(inventory)
    inventory_path = build_root / "inventory.json"
    _atomic_bytes(inventory_path, inventory_bytes)
    qa_index = {
        "schema_version": "forward-dtm-candidate-qa/1.0.0",
        "build_id": build_id,
        "recipe_sha256": build_id,
        "source_sha256": selection.artifact["sha256"],
        "images": images,
        "qualification": selection.raw["qualification"],
    }
    qa_index_path = build_root / "qa" / "index.json"
    _atomic_bytes(qa_index_path, canonical_json(qa_index))
    manifest = {
        "schema_version": "forward-dtm-candidate-build/1.0.0",
        "status": "complete",
        "build_id": build_id,
        "recipe_sha256": build_id,
        "source_sha256": selection.artifact["sha256"],
        "artifacts": {
            "candidate_npz": {
                "path": artifact_path.relative_to(build_root).as_posix(),
                "bytes": artifact_path.stat().st_size,
                "sha256": sha256_file(artifact_path),
            },
            "inventory": {
                "path": inventory_path.relative_to(build_root).as_posix(),
                "bytes": inventory_path.stat().st_size,
                "sha256": hashlib.sha256(inventory_bytes).hexdigest(),
            },
            "qa_index": {
                "path": qa_index_path.relative_to(build_root).as_posix(),
                "bytes": qa_index_path.stat().st_size,
                "sha256": sha256_file(qa_index_path),
            },
        },
        "qualification": selection.raw["qualification"],
    }
    _atomic_bytes(manifest_path, canonical_json(manifest))
    return manifest_path


def _main() -> None:
    selection = load_forward_selection()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retained",
        type=Path,
        default=DATA_IN / "evidence" / "forward" / selection.retention_id / "retained.json",
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(build_forward_candidate(args.retained, args.output_root))


if __name__ == "__main__":
    _main()
