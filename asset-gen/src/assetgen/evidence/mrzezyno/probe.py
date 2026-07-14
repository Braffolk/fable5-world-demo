"""Run the bounded Mrzezyno dry-sand B1 support/error probe."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling

from ...config import DATA_WORK
from .qa import render_probe_qa
from .source import atomic_bytes, canonical_json, sha256_file

_SOURCE_SCHEMA = "mrzezyno-2022-02-extraction/1.0.0"
_RETAINED_SCHEMA = "mrzezyno-2022-02-retention/1.0.0"
_SCHEMA = "mrzezyno-b1-support-error-probe/1.0.0"
_QA_SCHEMA = "mrzezyno-b1-support-error-qa/1.0.0"
_F4 = np.asarray((1, 3, 6, 10, 12, 12, 10, 6, 3, 1), dtype=np.float64) / 64.0
_SOURCE_CELL_M = 0.1
_CANONICAL_CELL_M = 0.0625
_CONTEXT_SOURCE_CELLS = 80
_CONTEXT_CANONICAL_CELLS = 128


def _verified_inputs(
    source_manifest: Path,
    retained_manifest: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_encoded = source_manifest.read_bytes()
    source = json.loads(source_encoded)
    retained_encoded = retained_manifest.read_bytes()
    retained = json.loads(retained_encoded)
    if source.get("schema_version") != _SOURCE_SCHEMA or source.get("status") != "complete":
        raise ValueError("Mrzezyno source extraction is not complete")
    if retained.get("schema_version") != _RETAINED_SCHEMA or retained.get("status") != "complete":
        raise ValueError("Mrzezyno retention is not complete")
    if source.get("retained_manifest_sha256") != hashlib.sha256(retained_encoded).hexdigest():
        raise ValueError("Mrzezyno extraction does not bind this retained evidence")
    artifact = source.get("artifact", {})
    path = (source_manifest.parent / artifact.get("path", "")).resolve()
    if (
        not path.is_relative_to(source_manifest.parent.resolve())
        or path.stat().st_size != artifact.get("bytes")
        or sha256_file(path) != artifact.get("sha256")
    ):
        raise ValueError("Mrzezyno extracted DEM changed")
    paper_ref = retained.get("artifacts", {}).get("paper", {})
    paper = (retained_manifest.parent / paper_ref.get("path", "")).resolve()
    if (
        not paper.is_relative_to(retained_manifest.parent.resolve())
        or paper.stat().st_size != paper_ref.get("bytes")
        or sha256_file(paper) != paper_ref.get("sha256")
    ):
        raise ValueError("Mrzezyno paper evidence changed")
    references = {
        "source_manifest": {"bytes": len(source_encoded), "sha256": hashlib.sha256(source_encoded).hexdigest()},
        "retained_manifest": {"bytes": len(retained_encoded), "sha256": hashlib.sha256(retained_encoded).hexdigest()},
        "dem": dict(artifact),
        "paper": dict(paper_ref),
    }
    return path, source, retained, references


def _window_start(valid: np.ndarray, target_x: int, context_index: int) -> tuple[int, int]:
    height, width = valid.shape
    max_x = width - _CONTEXT_SOURCE_CELLS
    offsets = [0]
    for step in range(1, 401):
        offsets.extend((-step, step))
    for offset in offsets:
        x0 = min(max(target_x + offset, 0), max_x)
        row_ok = np.all(valid[:, x0 : x0 + _CONTEXT_SOURCE_CELLS], axis=1).astype(np.int8)
        edges = np.diff(np.pad(row_ok, (1, 1)))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        runs = [(int(start), int(end)) for start, end in zip(starts, ends, strict=True) if end - start >= _CONTEXT_SOURCE_CELLS]
        if not runs:
            continue
        start, end = max(runs, key=lambda item: item[1] - item[0])
        fraction = 1.0 / 3.0 if context_index % 2 == 0 else 2.0 / 3.0
        available = end - start - _CONTEXT_SOURCE_CELLS
        y0 = start + round(fraction * available)
        if np.all(valid[y0 : y0 + _CONTEXT_SOURCE_CELLS, x0 : x0 + _CONTEXT_SOURCE_CELLS]):
            return y0, x0
    raise ValueError("could not place a mask-only 8 m context near frozen alongshore target")


def _overlap_weights() -> np.ndarray:
    source_edges = np.arange(_CONTEXT_SOURCE_CELLS + 1, dtype=np.float64) * _SOURCE_CELL_M
    target_edges = np.arange(_CONTEXT_CANONICAL_CELLS + 1, dtype=np.float64) * _CANONICAL_CELL_M
    left = np.maximum(target_edges[:-1, None], source_edges[None, :-1])
    right = np.minimum(target_edges[1:, None], source_edges[None, 1:])
    weights = np.maximum(right - left, 0.0) / _CANONICAL_CELL_M
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-14):
        raise AssertionError("exact 0.10 m to 0.0625 m area weights do not conserve constants")
    return weights


def _f4_indexed(values: np.ndarray, input_start: int) -> tuple[np.ndarray, int]:
    input_end = input_start + values.shape[0] - 1
    output_start = math.ceil((input_start + 3) / 4)
    output_end = math.floor((input_end - 6) / 4)
    outputs = np.arange(output_start, output_end + 1, dtype=np.int64)
    indices = 4 * outputs[:, None] - 3 + np.arange(10, dtype=np.int64)[None, :] - input_start
    east = np.einsum("yok,k->yo", values[:, indices], _F4, optimize=True)
    north = np.einsum("okx,k->ox", east[indices, :], _F4, optimize=True)
    return north, output_start


def _keys_weight(offset: int, t: float) -> float:
    if offset == -1:
        return -0.5 * t + t * t - 0.5 * t * t * t
    if offset == 0:
        return 1.0 - 2.5 * t * t + 1.5 * t * t * t
    if offset == 1:
        return 0.5 * t + 2.0 * t * t - 1.5 * t * t * t
    return -0.5 * t * t + 0.5 * t * t * t


def _r4_to_indices(coarse: np.ndarray, coarse_start: int, target_indices: np.ndarray) -> np.ndarray:
    output = np.empty((len(target_indices), len(target_indices)), dtype=np.float64)
    for oy, fine_y in enumerate(target_indices):
        uy = (float(fine_y) - 1.5) / 4.0
        base_y = math.floor(uy)
        ty = uy - base_y
        iy = np.asarray([base_y + offset - coarse_start for offset in (-1, 0, 1, 2)])
        wy = np.asarray([_keys_weight(offset, ty) for offset in (-1, 0, 1, 2)])
        for ox, fine_x in enumerate(target_indices):
            ux = (float(fine_x) - 1.5) / 4.0
            base_x = math.floor(ux)
            tx = ux - base_x
            ix = np.asarray([base_x + offset - coarse_start for offset in (-1, 0, 1, 2)])
            wx = np.asarray([_keys_weight(offset, tx) for offset in (-1, 0, 1, 2)])
            output[oy, ox] = wy @ coarse[np.ix_(iy, ix)] @ wx
    return output


def _canonical_b1(source_patch: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h2 = weights @ source_patch.astype(np.float64) @ weights.T
    a1, a1_start = _f4_indexed(h2, 0)
    a0, a0_start = _f4_indexed(a1, a1_start)
    a0_end = a0_start + a0.shape[0] - 1
    valid_indices = []
    for index in range(a1_start, a1_start + a1.shape[0]):
        u = (float(index) - 1.5) / 4.0
        base = math.floor(u)
        if base - 1 >= a0_start and base + 2 <= a0_end:
            valid_indices.append(index)
    target = np.asarray(valid_indices, dtype=np.int64)
    reconstructed = _r4_to_indices(a0, a0_start, target)
    local = target - a1_start
    b1 = a1[np.ix_(local, local)] - reconstructed
    if b1.shape != (12, 12):
        raise AssertionError(f"unexpected canonical B1 interior shape: {b1.shape}")
    return h2, b1


def _preview(dataset: rasterio.io.DatasetReader, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    target_width = 2_000
    target_height = max(1, round(dataset.height * target_width / dataset.width))
    masked = dataset.read(
        1,
        out_shape=(target_height, target_width),
        resampling=Resampling.average,
        masked=True,
    )
    height = masked.filled(np.nan).astype(np.float64)
    nearest = dataset.read(
        1,
        out_shape=(target_height, target_width),
        resampling=Resampling.nearest,
        masked=True,
    )
    nearest_state = np.zeros((target_height, target_width), dtype=np.uint8)
    nearest_state[~np.ma.getmaskarray(nearest)] = 1
    # The three extreme cells are sub-pixel at preview scale; their existence is textual and exact in the state artifact.
    finite = np.isfinite(height) & (height >= 0.0) & (height <= 20.0)
    nearest_state[~finite] = 0
    if np.any(finite):
        height[~finite] = float(np.median(height[finite]))
    pixel_m = _SOURCE_CELL_M * dataset.width / target_width
    return height, nearest_state, pixel_m


def build_mrzezyno_b1_probe(
    source_manifest: Path,
    retained_manifest: Path,
    output_root: Path | None = None,
) -> Path:
    source_manifest = source_manifest.resolve()
    retained_manifest = retained_manifest.resolve()
    dem, source, retained, references = _verified_inputs(source_manifest, retained_manifest)
    recipe = {
        "schema_version": _SCHEMA,
        "implementation": {
            "probe_py_sha256": sha256_file(Path(__file__)),
            "qa_py_sha256": sha256_file(Path(__file__).with_name("qa.py")),
        },
        "inputs": references,
        "claim": {
            "regime": "southern_baltic_dry_sand_beach_dune_scarp",
            "site": "Mrzezyno",
            "campaign": "2022-02-28",
            "surface": "single_valued_exposed_dry_sand_surface",
            "band": "B1",
            "wavelength_m": [0.25, 1.0],
            "included": ["dry_exposed_beach_sand", "dry_exposed_dune_sand", "single_valued_dune_scarp"],
            "excluded": ["water", "bathymetry", "wet_or_dynamic_shoreline", "vegetation", "objects", "overhangs", "unknown_semantics"],
            "forbidden": ["B2", "fine_sand_ripples", "generic_terrain", "Estonia_transfer", "production_ownership"],
        },
        "analysis": {
            "context_count": 6,
            "context_size_m": 8.0,
            "context_selection": "six_even_alongshore_targets_then_nearest_all-provider-valid_window; height-blind",
            "source_cell_m": _SOURCE_CELL_M,
            "canonical_cell_m": _CANONICAL_CELL_M,
            "canonical_lattice_phase": (
                "context-local northwest source-cell edge; fixed before height analysis; "
                "not optimized and not evidence of phase robustness"
            ),
            "resampling": "exact_separable_source_cell_overlap; source values treated as constant IDW raster cells",
            "band_operator": "canonical_float64_F4_R4_B1",
            "scientific_use": "descriptive_signal_screen_only",
        },
    }
    build_id = hashlib.sha256(canonical_json(recipe)).hexdigest()
    root = (output_root or DATA_WORK / "microtopography" / "mrzezyno" / "probe" / "sha256") / build_id
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        return manifest_path
    root.mkdir(parents=True, exist_ok=False)
    atomic_bytes(root / "recipe.json", canonical_json(recipe))

    with rasterio.open(dem) as dataset:
        height = dataset.read(1)
        declared_valid = (height != dataset.nodata) & np.isfinite(height)
        extreme = declared_valid & ((height < 0.0) | (height > 20.0))
        provider_valid = declared_valid & ~extreme
        state = np.zeros(height.shape, dtype=np.uint8)
        state[provider_valid] = 1
        state[extreme] = 2
        valid_y, valid_x = np.nonzero(provider_valid)
        target_x = np.rint(np.linspace(valid_x.min() + 80, valid_x.max() - 160, 6)).astype(int)
        starts = [_window_start(provider_valid, int(x), index) for index, x in enumerate(target_x)]
        weights = _overlap_weights()
        h2_contexts = []
        b1_contexts = []
        context_records = []
        for index, (row, column) in enumerate(starts, start=1):
            patch = height[row : row + 80, column : column + 80]
            h2, b1 = _canonical_b1(patch, weights)
            h2_contexts.append(h2)
            b1_contexts.append(b1)
            x, y = dataset.xy(row, column)
            context_records.append(
                {
                    "id": f"C{index}",
                    "source_window_row_col_height_width": [row, column, 80, 80],
                    "northwest_source_cell_center_xy_m": [float(x), float(y)],
                    "selection_uses_height": False,
                    "provider_valid_cells": 6_400,
                    "measured_support_cells": 0,
                    "dry_sand_semantically_authorized_cells": 0,
                    "canonical_b1_cells": int(b1.size),
                    "b1_rms_m": float(np.sqrt(np.mean(b1 * b1))),
                    "b1_abs_p50_m": float(np.quantile(np.abs(b1), 0.5)),
                    "b1_abs_p95_m": float(np.quantile(np.abs(b1), 0.95)),
                    "b1_abs_max_m": float(np.max(np.abs(b1))),
                }
            )
        preview_height, preview_state, preview_pixel_m = _preview(dataset, state)

    arrays_path = root / "mrzezyno-b1-probe.npz"
    temporary = arrays_path.with_name(arrays_path.name + ".part")
    with temporary.open("xb") as target:
        np.savez_compressed(
            target,
            cell_state_u8=state,
            context_source_window_yx=np.asarray(starts, dtype=np.int32),
            context_canonical_height_m_f64=np.asarray(h2_contexts, dtype=np.float64),
            context_b1_m_f64=np.asarray(b1_contexts, dtype=np.float64),
        )
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(arrays_path)
    decision = {
        "schema_version": _SCHEMA,
        "decision": "no_go_as_b1_target_site",
        "claim": recipe["claim"],
        "source_state_counts": {
            "declared_nodata": int(np.count_nonzero(state == 0)),
            "provider_valid_idw_support_unknown": int(np.count_nonzero(state == 1)),
            "undeclared_extreme_invalid": int(np.count_nonzero(state == 2)),
            "authorized_target_cells": 0,
        },
        "published_control_evidence": {
            "source": "Landform Analysis 44 (2025), doi:10.12657/landfana-044-003, journal pages 59-62",
            "campaign_relationship": "paper explicitly covers the 2022-02-28 L1 campaign",
            "rtk_receiver_assumed_horizontal_accuracy_upper_bound_m": 0.03,
            "rtk_receiver_assumed_vertical_accuracy_upper_bound_m": 0.05,
            "median_vertical_deviation_upper_bound_m": 0.05,
            "vertical_deviation_p95_upper_bound_m": 0.10,
            "scope": "per-campaign LiDAR DEM elevations compared with RTK profiles and GCPs",
            "missing": ["released_profile_points", "sample_count_by_campaign", "error_distribution", "band_resolved_error_spectrum", "per_cell_support_distance"],
            "sfm_checkpoint_rmse_not_used_as_lidar_truth": {"planimetry_upper_bound_m": 0.03, "elevation_upper_bound_m": 0.04},
        },
        "descriptive_b1_contexts": context_records,
        "gates": {
            "original_observations_present": False,
            "per_cell_measured_support_present": False,
            "support_threshold_0_05m_evaluable": False,
            "dry_sand_semantics_present": False,
            "water_vegetation_object_exclusion_evaluable": False,
            "band_resolved_total_error_present": False,
            "maximum_error_signal_ratio_evaluable": False,
            "minimum_transfer_gain_evaluable": False,
            "independent_site_split_possible": False,
        },
        "reasons": [
            "release_contains_only_an_idw_dem_not_original_lidar_points",
            "all_provider_valid_cells_lack_measured_support_and_interpolation_distance",
            "no_co_registered_dry_sand_water_vegetation_or_object_semantics",
            "campaign_vertical_deviation_bounds_are_not_a_b1_error_distribution",
            "one_campaign_at_one_site_cannot_supply_independent_site_or_campaign_validation",
        ],
        "allowed_use": ["coastal_process_visual_inspection", "converter_stress_input", "descriptive_b1_amplitude_screen"],
        "not_authorized": ["qualified_target_B1", "qualified_target_B2", "synthesis_training_target", "Estonia_transfer", "release_ownership"],
        "resume_condition": "publisher releases original classified points plus reconstructable support/view identity and co-registered dry-sand semantics, or equivalent independent cell-level evidence",
    }
    decision_path = root / "decision.json"
    atomic_bytes(decision_path, canonical_json(decision))
    qa_records = render_probe_qa(
        preview_height,
        preview_state,
        np.asarray(b1_contexts),
        context_records,
        decision,
        root / "qa",
        preview_pixel_m=preview_pixel_m,
    )
    images = []
    for index, record in enumerate(qa_records, start=1):
        path = record["path"]
        with Image.open(path) as image:
            dimensions = list(image.size)
        images.append(
            {
                "index": index,
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "dimensions_xy": dimensions,
                "interpretation": record["interpretation"],
            }
        )
    qa_index = {
        "schema_version": _QA_SCHEMA,
        "build_id": build_id,
        "source_dem_sha256": source["artifact"]["sha256"],
        "recipe_sha256": build_id,
        "images": images,
    }
    qa_index_path = root / "qa" / "index.json"
    atomic_bytes(qa_index_path, canonical_json(qa_index))
    manifest = {
        "schema_version": _SCHEMA,
        "status": "complete_fail_closed",
        "build_id": build_id,
        "decision": decision["decision"],
        "environment": {
            "numpy": np.__version__,
            "rasterio": rasterio.__version__,
            "pillow": importlib.metadata.version("pillow"),
            "scipy": importlib.metadata.version("scipy"),
        },
        "artifacts": {
            "arrays": {"path": arrays_path.relative_to(root).as_posix(), "bytes": arrays_path.stat().st_size, "sha256": sha256_file(arrays_path)},
            "decision": {"path": decision_path.relative_to(root).as_posix(), "bytes": decision_path.stat().st_size, "sha256": sha256_file(decision_path)},
            "qa_index": {"path": qa_index_path.relative_to(root).as_posix(), "bytes": qa_index_path.stat().st_size, "sha256": sha256_file(qa_index_path)},
        },
        "qualification": {
            "b1_target_site": False,
            "b2_target_site": False,
            "synthesis_authorized": False,
            "estonia_transfer_authorized": False,
        },
    }
    atomic_bytes(manifest_path, canonical_json(manifest))
    return manifest_path


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the bounded Mrzezyno B1 probe.")
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("retained_manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(build_mrzezyno_b1_probe(args.source_manifest, args.retained_manifest, args.output_root))


if __name__ == "__main__":
    _main()
