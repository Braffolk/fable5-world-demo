"""Materialize the retained Mrzezyno DEM as a narrowly authorized shape target."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import rasterio
from scipy import ndimage

from ...config import DATA_WORK
from .source import atomic_bytes, canonical_json, sha256_file
from .target_qa import render_target_qa

_SOURCE_SCHEMA = "mrzezyno-2022-02-extraction/1.0.0"
_SCHEMA = "mrzezyno-morphology-target/1.0.0"
_QA_SCHEMA = "mrzezyno-morphology-target-qa/1.0.0"
_SOURCE_PIXEL_M = 0.1
_EXPECTED_EXTREMES = ((4620, 1), (4620, 2), (4620, 3))
_DUNE_ELEVATION_FLOOR_M = 2.0
_LOW_ELEVATION_CEILING_M = 1.0
_PATCH_CELLS = 256
_PATCH_CORE = slice(64, 192)
_BANDS_M = ((0.2, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0))
_CAMPAIGN_MEDIAN_ERROR_M = 0.05
_CAMPAIGN_P95_ERROR_M = 0.10


def _verified_source(source_manifest: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    encoded = source_manifest.read_bytes()
    source = json.loads(encoded)
    if source.get("schema_version") != _SOURCE_SCHEMA or source.get("status") != "complete":
        raise ValueError("Mrzezyno source extraction is not complete")
    artifact = source.get("artifact", {})
    relative = artifact.get("path")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Mrzezyno DEM path is invalid")
    path = (source_manifest.parent / relative).resolve()
    if (
        not path.is_relative_to(source_manifest.parent.resolve())
        or path.stat().st_size != artifact.get("bytes")
        or sha256_file(path) != artifact.get("sha256")
    ):
        raise ValueError("Mrzezyno DEM changed")
    reference = {
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "dem_sha256": artifact["sha256"],
    }
    return path, source, reference


def _select_patches(height: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    candidates: dict[str, list[tuple[float, int, int, dict[str, float]]]] = {"dune": [], "low": []}
    half = _PATCH_CELLS // 2
    for row in range(half, height.shape[0] - half + 1, 96):
        for column in range(half, height.shape[1] - half + 1, 96):
            mask = valid[row - half : row + half, column - half : column + half]
            valid_fraction = float(np.mean(mask))
            if valid_fraction < 0.96:
                continue
            values = height[row - half : row + half, column - half : column + half][mask]
            mean = float(np.mean(values))
            p05, p95 = np.quantile(values, (0.05, 0.95))
            relief = float(p95 - p05)
            record = {"mean_elevation_m": mean, "p05_p95_relief_m": relief, "valid_fraction": valid_fraction}
            if mean >= _DUNE_ELEVATION_FLOOR_M:
                candidates["dune"].append((relief + 0.04 * mean, row, column, record))
            elif mean <= _LOW_ELEVATION_CEILING_M:
                candidates["low"].append((relief, row, column, record))

    selected: list[tuple[str, int, int, dict[str, float]]] = []
    for role in ("dune", "low"):
        for _, row, column, record in sorted(candidates[role], reverse=True):
            if all((row - other_row) ** 2 + (column - other_column) ** 2 >= 400**2 for _, other_row, other_column, _ in selected):
                selected.append((role, row, column, record))
                if sum(item[0] == role for item in selected) == 2:
                    break
        if sum(item[0] == role for item in selected) != 2:
            raise ValueError(f"Mrzezyno target has fewer than two interior {role} morphology patches")

    patches: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for role, row, column, record in selected:
        patch = height[row - half : row + half, column - half : column + half].astype(np.float32, copy=True)
        mask = valid[row - half : row + half, column - half : column + half]
        patch[~mask] = np.nan
        patches.append(patch)
        ordinal = 1 + sum(item["role"] == role for item in records)
        records.append(
            {
                "role": role,
                "display_label": f"{role.upper()} CANDIDATE {ordinal}",
                "center_row_column": [row, column],
                "size_m": _PATCH_CELLS * _SOURCE_PIXEL_M,
                **record,
            }
        )
    return np.stack(patches), records


def _fill_patch(patch: np.ndarray) -> np.ndarray:
    valid = np.isfinite(patch)
    if np.all(valid):
        return patch.astype(np.float64)
    _, indices = ndimage.distance_transform_edt(~valid, return_indices=True)
    output = patch.astype(np.float64, copy=True)
    output[~valid] = output[tuple(axis[~valid] for axis in indices)]
    return output


def _band_measurements(
    patches: np.ndarray,
    patch_records: list[Mapping[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    decomposed: list[np.ndarray] = []
    per_patch: list[dict[str, Any]] = []
    for patch, patch_record in zip(patches, patch_records, strict=True):
        filled = _fill_patch(patch)
        bands: list[np.ndarray] = []
        metrics: list[dict[str, Any]] = []
        for lower_m, upper_m in _BANDS_M:
            lower_sigma = lower_m / (2.355 * _SOURCE_PIXEL_M)
            upper_sigma = upper_m / (2.355 * _SOURCE_PIXEL_M)
            fine = ndimage.gaussian_filter(filled, lower_sigma, mode="reflect")
            coarse = ndimage.gaussian_filter(filled, upper_sigma, mode="reflect")
            values = (fine - coarse)[_PATCH_CORE, _PATCH_CORE]
            bands.append(values.astype(np.float32))
            metrics.append(
                {
                    "lower_wavelength_m": lower_m,
                    "upper_wavelength_m": upper_m,
                    "rms_m": float(np.sqrt(np.mean(values * values))),
                    "abs_p95_m": float(np.quantile(np.abs(values), 0.95)),
                    "abs_p99_m": float(np.quantile(np.abs(values), 0.99)),
                }
            )
        decomposed.append(np.stack(bands))
        per_patch.append({"role": patch_record["role"], "bands": metrics})

    summary: list[dict[str, Any]] = []
    for band_index, (lower_m, upper_m) in enumerate(_BANDS_M):
        by_role: dict[str, list[np.ndarray]] = {"dune": [], "low": []}
        for patch_index, record in enumerate(patch_records):
            by_role[record["role"]].append(decomposed[patch_index][band_index].ravel())
        dune = np.concatenate(by_role["dune"])
        low = np.concatenate(by_role["low"])
        dune_p95 = float(np.quantile(np.abs(dune), 0.95))
        low_p95 = float(np.quantile(np.abs(low), 0.95))
        summary.append(
            {
                "label": f"{lower_m:g}-{upper_m:g} m",
                "lower_wavelength_m": lower_m,
                "upper_wavelength_m": upper_m,
                "dune_abs_p95_m": dune_p95,
                "low_abs_p95_m": low_p95,
                "dune_rms_m": float(np.sqrt(np.mean(dune * dune))),
                "low_rms_m": float(np.sqrt(np.mean(low * low))),
                "exceeds_campaign_median_bound": dune_p95 > _CAMPAIGN_MEDIAN_ERROR_M,
                "exceeds_campaign_p95_bound": dune_p95 > _CAMPAIGN_P95_ERROR_M,
                "interpretation": "observed_relief_only_not_band_resolved_error_separation",
            }
        )
    examples = np.stack(decomposed)
    labels = [{"label": record["label"]} for record in summary]
    return examples, per_patch, summary


def build_mrzezyno_morphology_target(source_manifest: Path, output_root: Path | None = None) -> Path:
    source_manifest = source_manifest.resolve()
    dem_path, source, source_reference = _verified_source(source_manifest)
    recipe = {
        "schema_version": _SCHEMA,
        "source": source_reference,
        "implementation": {
            "target_py_sha256": sha256_file(Path(__file__)),
            "target_qa_py_sha256": sha256_file(Path(__file__).with_name("target_qa.py")),
        },
        "validity": {
            "mask_declared_nodata": True,
            "explicit_extreme_row_columns": [list(item) for item in _EXPECTED_EXTREMES],
            "valid_elevation_range_m": [0.0, 20.0],
        },
        "morphology_roles": {
            "dune_landform_candidate": {"minimum_elevation_m": _DUNE_ELEVATION_FLOOR_M, "semantic_label": False},
            "low_coastal_surface_unresolved": {"maximum_elevation_m": _LOW_ELEVATION_CEILING_M, "semantic_label": False},
            "transition_unresolved": {"semantic_label": False},
        },
        "band_screen": {
            "bands_m": [list(item) for item in _BANDS_M],
            "gaussian_sigma_definition": "wavelength_m/(2.355*pixel_m)",
            "campaign_median_vertical_deviation_upper_bound_m": _CAMPAIGN_MEDIAN_ERROR_M,
            "campaign_p95_vertical_deviation_upper_bound_m": _CAMPAIGN_P95_ERROR_M,
            "comparison_scope": "observed relief amplitude only; band-resolved error remains unidentified",
        },
        "patch_selection": {"cells": _PATCH_CELLS, "minimum_valid_fraction": 0.96, "count_per_role": 2},
    }
    build_id = hashlib.sha256(canonical_json(recipe)).hexdigest()
    root = (output_root or DATA_WORK / "microtopography" / "mrzezyno" / "target" / "sha256") / build_id
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        return manifest_path
    root.mkdir(parents=True, exist_ok=False)
    atomic_bytes(root / "recipe.json", canonical_json(recipe))

    with rasterio.open(dem_path) as dataset:
        height = dataset.read(1)
        declared_valid = (height != dataset.nodata) & np.isfinite(height)
        extreme = declared_valid & ((height < 0.0) | (height > 20.0))
        observed_extremes = tuple(map(tuple, np.argwhere(extreme)))
        if observed_extremes != _EXPECTED_EXTREMES:
            raise ValueError("Mrzezyno undeclared extremes changed")
        valid = declared_valid & ~extreme
        rows, columns = np.nonzero(valid)
        crop = (slice(int(rows.min()), int(rows.max()) + 1), slice(int(columns.min()), int(columns.max()) + 1))
        transform = dataset.window_transform(
            rasterio.windows.Window.from_slices((crop[0].start, crop[0].stop), (crop[1].start, crop[1].stop))
        )
        crs = str(dataset.crs)

    height = height[crop].astype(np.float32, copy=True)
    valid = valid[crop]
    height[~valid] = np.nan
    role_state = np.zeros(height.shape, dtype=np.uint8)
    role_state[valid & (height >= _DUNE_ELEVATION_FLOOR_M)] = 1
    role_state[valid & (height <= _LOW_ELEVATION_CEILING_M)] = 2
    role_state[valid & (height > _LOW_ELEVATION_CEILING_M) & (height < _DUNE_ELEVATION_FLOOR_M)] = 3

    patches, patch_records = _select_patches(height, valid)
    band_examples, per_patch_bands, band_summary = _band_measurements(patches, patch_records)
    target_path = root / "mrzezyno-morphology-target.npz"
    temporary = target_path.with_name(target_path.name + ".part")
    with temporary.open("wb") as target:
        np.savez_compressed(
            target,
            height_m=height,
            valid=np.packbits(valid, axis=None),
            valid_shape_yx=np.asarray(valid.shape, dtype=np.int32),
            valid_bitorder=np.asarray("big"),
            role_state=role_state,
            representative_patches_m=patches,
            patch_centers_row_column=np.asarray([item["center_row_column"] for item in patch_records], dtype=np.int32),
            band_examples_m=band_examples,
        )
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(target_path)

    qa_items = render_target_qa(
        height,
        valid,
        role_state,
        patches,
        patch_records,
        band_examples[0],
        [{"label": item["label"]} for item in band_summary],
        band_summary,
        root / "qa",
    )
    qa_records = []
    for item in qa_items:
        path = item["path"]
        qa_records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "interpretation": item["interpretation"],
            }
        )
    qa_index = {
        "schema_version": _QA_SCHEMA,
        "build_id": build_id,
        "source_dem_sha256": source_reference["dem_sha256"],
        "recipe_sha256": build_id,
        "images": qa_records,
    }
    qa_index_path = root / "qa" / "index.json"
    atomic_bytes(qa_index_path, canonical_json(qa_index))

    decision = {
        "verdict": "pass_narrow_dune_shape_park_beach_material",
        "sand.dune_aeolian": {
            "status": "narrow_shape_input",
            "authorized": "high-relief Baltic dune-coast landform geometry and only observed bands whose p95 relief exceeds 0.10 m",
            "not_authorized": ["fine ripples", "dry-sand labels", "vegetation-free labels", "per-cell measured truth", "Estonia transfer ownership"],
        },
        "shore.beach_sand": {
            "status": "park",
            "reason": "raster-only evidence cannot separate beach from dune transition, wet from dry sand, or shoreline/water support",
        },
        "next_synthesis_input": "masked high-relief dune-landform patches plus clearing-band amplitude/correlation statistics; use as morphology prior, never pixelwise target truth",
        "campaign_error_interpretation": "amplitude comparison only; no claim of band-resolved observation error",
    }
    decision_path = root / "decision.json"
    atomic_bytes(decision_path, canonical_json(decision))
    measurements = {
        "patches": patch_records,
        "per_patch_bands": per_patch_bands,
        "band_summary": band_summary,
    }
    measurements_path = root / "measurements.json"
    atomic_bytes(measurements_path, canonical_json(measurements))

    valid_count = int(np.count_nonzero(valid))
    if valid_count != 7_799_605:
        raise ValueError("Mrzezyno valid target inventory changed")
    manifest = {
        "schema_version": _SCHEMA,
        "status": "complete",
        "build_id": build_id,
        "recipe_sha256": build_id,
        "source": source_reference,
        "geometry": {
            "crs": crs,
            "pixel_m": _SOURCE_PIXEL_M,
            "crop_source_row_column": [crop[0].start, crop[1].start, crop[0].stop, crop[1].stop],
            "shape_yx": list(height.shape),
            "transform": list(transform)[:6],
            "valid_cells": valid_count,
            "declared_nodata_or_outside_cells": int(height.size - valid_count),
            "explicit_extreme_cells_excluded": 3,
        },
        "role_counts": {
            "dune_landform_candidate": int(np.count_nonzero(role_state == 1)),
            "low_coastal_surface_unresolved": int(np.count_nonzero(role_state == 2)),
            "transition_unresolved": int(np.count_nonzero(role_state == 3)),
        },
        "decision": decision,
        "artifacts": {
            "target": {"path": target_path.relative_to(root).as_posix(), "bytes": target_path.stat().st_size, "sha256": sha256_file(target_path)},
            "measurements": {"path": measurements_path.relative_to(root).as_posix(), "bytes": measurements_path.stat().st_size, "sha256": sha256_file(measurements_path)},
            "decision": {"path": decision_path.relative_to(root).as_posix(), "bytes": decision_path.stat().st_size, "sha256": sha256_file(decision_path)},
            "qa_index": {"path": qa_index_path.relative_to(root).as_posix(), "bytes": qa_index_path.stat().st_size, "sha256": sha256_file(qa_index_path)},
        },
        "environment": {
            "numpy": np.__version__,
            "rasterio": rasterio.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "pillow": importlib.metadata.version("pillow"),
        },
    }
    atomic_bytes(manifest_path, canonical_json(manifest))
    return manifest_path


def _main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the narrow Mrzezyno morphology target.")
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(build_mrzezyno_morphology_target(args.source_manifest, args.output_root))


if __name__ == "__main__":
    _main()
