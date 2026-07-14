"""Immutable HY_SPRUCE4 merged-cloud candidate-surface transaction."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping

import laspy
import numpy as np
from PIL import Image
from scipy import ndimage

from ....config import CONFIG_DIR, DATA_WORK
from ..records import (
    HoviSelection,
    RetainedSelection,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from .artifacts import (
    artifact_ref,
    atomic_write,
    store_content_bytes,
    store_content_file,
    write_deterministic_npz,
)
from .histogram import (
    accumulate_selected_mode_footprint,
    accumulate_structural_histogram,
    accumulate_tangent_sheared_histogram,
    accumulate_windowed_histogram,
    aggregate_histogram,
    grid_from_bounds,
)
from .model import Reason, SurfaceCandidate, SurfaceConfig
from .modes import (
    discover_context_component,
    extract_vertical_modes,
    interpolate_small_supported_holes,
    predict_same_sheet_tangent,
    select_conditioned_sheet,
)
from .qa import render_surface_qa
from .topology import audit_direct_topology

_RECIPE_SCHEMA = "hovi-candidate-surface-recipe/1.0.0"
_EVIDENCE_SCHEMA = "hovi-candidate-surface-evidence/1.0.0"
_MANIFEST_SCHEMA = "hovi-candidate-surface-build/1.0.0"
_QA_SCHEMA = "hovi-candidate-surface-qa/1.0.0"
_PLOT_ID = "HY_SPRUCE4"
_REGISTRATION_MAE_RE = re.compile(
    r"for Enabled Constraints\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*m"
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Hovi surface {label} must be an object")
    return value


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    encoded = path.read_bytes()
    value = json.loads(encoded)
    if not isinstance(value, dict):
        raise ValueError(f"Hovi surface {label} must be a JSON object")
    return value, sha256_bytes(encoded)


def _verify_relative_artifact(root: Path, reference: Mapping[str, Any]) -> Path:
    relative = reference.get("path")
    digest = reference.get("sha256")
    byte_count = reference.get("bytes")
    if (
        not isinstance(relative, str)
        or Path(relative).is_absolute()
        or not isinstance(digest, str)
        or len(digest) != 64
        or not isinstance(byte_count, int)
    ):
        raise ValueError("Hovi surface input artifact reference is invalid")
    path = (root / relative).resolve()
    if (
        not path.is_relative_to(root.resolve())
        or not path.is_file()
        or path.stat().st_size != byte_count
        or sha256_file(path) != digest
    ):
        raise ValueError(f"Hovi surface input artifact failed verification: {relative}")
    return path


def _validate_support(
    manifest_path: Path, *, source_sha256: str, retention_id: str
) -> tuple[dict[str, Any], dict[str, Any], str, Path]:
    manifest, manifest_sha = _load_json(manifest_path, "raw-support manifest")
    if (
        manifest.get("schema_version") != "hovi-observation-support-build/1.0.0"
        or manifest.get("status") != "complete"
        or manifest.get("plot_id") != _PLOT_ID
        or manifest.get("scientific_role") != "raw_candidate"
        or manifest.get("qualification_status") != "unqualified"
        or manifest.get("transfer_ceiling") != "none"
        or manifest.get("synthesis_authorized") is not False
    ):
        raise ValueError("Hovi surface requires the complete unqualified raw-support build")
    inventory_path = _verify_relative_artifact(
        manifest_path.parent, _mapping(manifest.get("inventory"), "support inventory ref")
    )
    support_npz_path = _verify_relative_artifact(
        manifest_path.parent, _mapping(manifest.get("support_npz"), "support NPZ ref")
    )
    inventory, inventory_sha = _load_json(inventory_path, "raw-support inventory")
    disposition = _mapping(inventory.get("scientific_disposition"), "support disposition")
    source = _mapping(inventory.get("source_artifact"), "support source")
    retention = _mapping(inventory.get("retention"), "support retention")
    if (
        inventory.get("schema_version") != "hovi-observation-support-inventory/1.0.0"
        or inventory_sha != manifest["inventory"]["sha256"]
        or source.get("sha256") != source_sha256
        or retention.get("retention_id") != retention_id
        or disposition.get("role") != "raw_candidate"
        or disposition.get("qualification_status") != "unqualified"
        or disposition.get("transfer_ceiling") != "none"
        or disposition.get("synthesis_authorized") is not False
    ):
        raise ValueError("Hovi surface raw-support evidence identity changed")
    return manifest, inventory, manifest_sha, support_npz_path


def _validate_condition(path: Path, retention_id: str) -> tuple[dict[str, Any], str]:
    record, digest = _load_json(path, "condition evidence")
    qualification = _mapping(record.get("qualification"), "condition qualification")
    plot = _mapping(record.get("plot_identity"), "condition plot")
    retention = _mapping(record.get("retention"), "condition retention")
    if (
        record.get("schema_version") != "hovi-condition-semantics-evidence/1.0.0"
        or path.parent.name != digest
        or plot.get("plot_id") != _PLOT_ID
        or retention.get("retention_id") != retention_id
        or qualification.get("role") != "raw_candidate"
        or qualification.get("qualification_status") != "unqualified"
        or qualification.get("transfer_ceiling") != "none"
        or qualification.get("synthesis_authorized") is not False
        or qualification.get("target_truth") is not False
    ):
        raise ValueError("Hovi surface condition evidence identity changed")
    return record, digest


def _validate_photo_qa(
    path: Path, *, condition_sha256: str, retention_id: str
) -> tuple[dict[str, Any], str]:
    record, digest = _load_json(path, "semantic-photo QA index")
    qualification = _mapping(record.get("qualification"), "photo QA qualification")
    claims = _mapping(record.get("claims"), "photo QA claims")
    if (
        record.get("schema_version") != "hovi-semantic-photo-qa-index/1.0.0"
        or record.get("status") != "complete"
        or record.get("plot_id") != _PLOT_ID
        or record.get("retention_id") != retention_id
        or record.get("condition_evidence_sha256") != condition_sha256
        or qualification.get("qualification_status") != "unqualified"
        or qualification.get("transfer_ceiling") != "none"
        or qualification.get("synthesis_authorized") is not False
        or claims.get("metric_registration") is not False
        or claims.get("pixel_classification") is not False
        or claims.get("ground_height_claim") is not False
        or claims.get("target_truth_claim") is not False
    ):
        raise ValueError("Hovi surface semantic-photo QA boundary changed")
    for output in record.get("outputs", []):
        _verify_relative_artifact(path.parent.parent, _mapping(output, "photo QA output"))
    return record, digest


def _registration_record(path: Path) -> dict[str, Any]:
    encoded = path.read_bytes()
    text = encoded.decode("utf-8-sig")
    match = _REGISTRATION_MAE_RE.search(text)
    if match is None:
        raise ValueError("Hovi registration diagnostics lack enabled-constraint MAE")
    return {
        "sha256": sha256_bytes(encoded),
        "bytes": len(encoded),
        "enabled_constraint_mean_absolute_error_m": float(match.group(1)),
        "interpretation": (
            "project-level registration diagnostic only; merged points lack scan identity, "
            "so no per-cell independent error can be derived"
        ),
    }


def _environment() -> dict[str, str]:
    return {
        "numpy": np.__version__,
        "scipy": importlib.metadata.version("scipy"),
        "laspy": laspy.__version__,
        "lazrs": importlib.metadata.version("lazrs"),
        "pillow": importlib.metadata.version("pillow"),
    }


def _implementation_identity() -> dict[str, Any]:
    module_root = Path(__file__).parent
    module_names = (
        "__init__.py",
        "artifacts.py",
        "convert.py",
        "histogram.py",
        "model.py",
        "modes.py",
        "qa.py",
        "topology.py",
    )
    modules = {}
    for name in module_names:
        path = module_root / name
        modules[name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "package": "assetgen.evidence.hovi.surface",
        "entrypoint": "assetgen.evidence.hovi.surface.convert:extract_hovi_candidate_surface",
        "modules": modules,
    }


def _nominal_core(
    condition: Mapping[str, Any], grid
) -> tuple[np.ndarray, tuple[float, float]]:
    protocol = _mapping(condition.get("scan_protocol"), "scan protocol")
    transform = np.asarray(protocol.get("homogeneous_xy_transform"), dtype=np.float64)
    plot = _mapping(condition.get("plot_identity"), "plot identity")
    coordinates = _mapping(plot.get("coordinates"), "plot coordinates")
    center_utm = _mapping(coordinates.get("selection_epsg25835"), "EPSG:25835 center")
    target = np.asarray(
        [center_utm.get("easting_m"), center_utm.get("northing_m")], dtype=np.float64
    )
    if transform.shape != (3, 3) or not np.all(np.isfinite(transform)) or not np.all(
        np.isfinite(target)
    ):
        raise ValueError("Hovi scan transform or nominal plot center is invalid")
    local = np.linalg.solve(transform[:2, :2], target - transform[:2, 2])
    layout = np.asarray(protocol.get("nominal_layout_m"), dtype=np.float64)
    if layout.shape != (2,) or np.any(layout <= 0.0):
        raise ValueError("Hovi nominal plot layout is invalid")
    x = grid.origin_x_m + (np.arange(grid.width) + 0.5) * grid.resolution_m
    y = grid.origin_y_m + (np.arange(grid.height) + 0.5) * grid.resolution_m
    mask = (
        (np.abs(x[None, :] - local[0]) <= 0.5 * layout[0])
        & (np.abs(y[:, None] - local[1]) <= 0.5 * layout[1])
    )
    return mask, (float(local[0]), float(local[1]))


def _eroded_analysis_core(observed_mask: np.ndarray, grid, erosion_m: float) -> np.ndarray:
    evidence_domain = np.asarray(observed_mask, dtype=bool)
    padded = np.pad(evidence_domain, 1, mode="constant", constant_values=False)
    distance = ndimage.distance_transform_edt(padded, sampling=grid.resolution_m)[1:-1, 1:-1]
    return evidence_domain & (distance >= erosion_m)


def _distribution(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0, "min": None, "p05": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "median": float(np.median(finite)),
        "p95": float(np.quantile(finite, 0.95)),
        "max": float(np.max(finite)),
    }


def _mask_metrics(mask: np.ndarray, resolution_m: float) -> dict[str, Any]:
    labels, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(2, 1))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    cells = int(np.count_nonzero(mask))
    return {
        "cells": cells,
        "area_m2": cells * resolution_m * resolution_m,
        "component_count": int(count),
        "largest_component_cells": int(sizes.max(initial=0)),
        "largest_component_area_m2": float(
            sizes.max(initial=0) * resolution_m * resolution_m
        ),
    }


def _quality_metrics(candidate: SurfaceCandidate) -> dict[str, Any]:
    core = candidate.analysis_core_mask
    core_count = int(np.count_nonzero(core))
    direct = candidate.direct_mask & core
    ambiguity = candidate.ambiguity_mask & core
    inferred = candidate.inferred_mask & core
    competitor_mask = candidate.above_sheet_competitor_mask & core
    labels, count = ndimage.label(direct, structure=ndimage.generate_binary_structure(2, 1))
    component_sizes = np.bincount(labels.ravel(), minlength=count + 1)
    largest = int(component_sizes[1:].max(initial=0))
    height = candidate.absolute_local_z_m
    jumps = []
    horizontal = direct[:, :-1] & direct[:, 1:]
    vertical = direct[:-1, :] & direct[1:, :]
    jumps.extend(np.abs(height[:, 1:] - height[:, :-1])[horizontal].tolist())
    jumps.extend(np.abs(height[1:, :] - height[:-1, :])[vertical].tolist())
    jump_array = np.asarray(jumps, dtype=np.float32)
    return {
        "analysis_core_cells": core_count,
        "direct_cells": int(np.count_nonzero(direct)),
        "direct_fraction": float(np.count_nonzero(direct) / max(core_count, 1)),
        "inferred_cells": int(np.count_nonzero(inferred)),
        "inferred_fraction": float(np.count_nonzero(inferred) / max(core_count, 1)),
        "ambiguity_cells": int(np.count_nonzero(ambiguity)),
        "ambiguity_fraction": float(np.count_nonzero(ambiguity) / max(core_count, 1)),
        "above_sheet_competitor_cells": int(np.count_nonzero(competitor_mask)),
        "above_sheet_competitor_fraction": float(np.count_nonzero(competitor_mask) / max(core_count, 1)),
        "largest_direct_component_cells": largest,
        "largest_direct_component_fraction": float(largest / max(core_count, 1)),
        "neighbor_jump_m": {
            "count": int(jump_array.size),
            "median": float(np.median(jump_array)) if jump_array.size else None,
            "p95": float(np.quantile(jump_array, 0.95)) if jump_array.size else None,
            "p99": float(np.quantile(jump_array, 0.99)) if jump_array.size else None,
            "max": float(jump_array.max()) if jump_array.size else None,
        },
        "direct_height_m": _distribution(height[direct]),
        "direct_selected_support_count": _distribution(candidate.selected_support_count[direct]),
        "direct_selection_score": _distribution(candidate.selection_score[direct]),
        "direct_competing_mode_ratio": _distribution(candidate.competing_mode_ratio[direct]),
        "direct_footprint_quadrants": _distribution(candidate.footprint_quadrant_count[direct]),
        "direct_footprint_subcells": _distribution(candidate.footprint_subcell_count[direct]),
    }


def _array_record(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {"dtype": str(value.dtype), "shape": list(value.shape)}
        for name, value in arrays.items()
    }


def extract_hovi_candidate_surface(
    retained_path: Path,
    support_manifest_path: Path,
    condition_evidence_path: Path,
    photo_qa_index_path: Path,
    *,
    selection_path: Path = CONFIG_DIR / "hovi-public-targets.json",
    work_root: Path = DATA_WORK,
    config: SurfaceConfig = SurfaceConfig(),
    log: Callable[[str], None] = print,
) -> Path:
    selection = HoviSelection.load(selection_path)
    plot = selection.development_plot(_PLOT_ID)
    source_selected = plot.geometry_artifact()
    retained = RetainedSelection.load(retained_path, selection=selection)
    source = retained.artifact_for(source_selected)
    if source.plot_id != _PLOT_ID:
        raise ValueError("Hovi surface source belongs to another plot")
    log("verifying retained HY_SPRUCE4 merged LAZ")
    if sha256_file(source.local_path) != source.sha256:
        raise ValueError("Hovi surface source bytes changed")

    support_manifest, support_inventory, support_manifest_sha, support_npz_path = _validate_support(
        support_manifest_path,
        source_sha256=source.sha256,
        retention_id=retained.retention_id,
    )
    condition, condition_sha = _validate_condition(
        condition_evidence_path, retained.retention_id
    )
    photo_qa, photo_qa_sha = _validate_photo_qa(
        photo_qa_index_path,
        condition_sha256=condition_sha,
        retention_id=retained.retention_id,
    )
    registration_selected = next(
        item for item in plot.artifacts if item.kind == "registration_diagnostic"
    )
    registration_source = retained.artifact_for(registration_selected)
    registration = _registration_record(registration_source.local_path)
    if registration["sha256"] != registration_selected.sha256:
        raise ValueError("Hovi registration diagnostic bytes changed")

    recipe = {
        "schema_version": _RECIPE_SCHEMA,
        "plot_id": _PLOT_ID,
        "selection_sha256": selection.sha256,
        "retention_id": retained.retention_id,
        "source": {
            "file_id": source.file_id,
            "path": source.source_path,
            "bytes": source.bytes,
            "sha256": source.sha256,
            "kind": source.kind,
        },
        "support_manifest": {
            "build_id": support_manifest["build_id"],
            "sha256": support_manifest_sha,
            "inventory_sha256": support_manifest["inventory"]["sha256"],
            "npz_sha256": support_manifest["support_npz"]["sha256"],
        },
        "condition_evidence_sha256": condition_sha,
        "semantic_photo_qa_sha256": photo_qa_sha,
        "registration_diagnostic": registration,
        "algorithm": config.as_record(),
        "implementation": _implementation_identity(),
        "environment": _environment(),
    }
    recipe_bytes = canonical_json_bytes(recipe)
    build_id = sha256_bytes(recipe_bytes)
    build_root = work_root / "microtopography" / "hovi" / "surface" / build_id
    manifest_path = build_root / "manifest.json"
    if manifest_path.exists():
        manifest, _ = _load_json(manifest_path, "existing surface manifest")
        if (
            manifest.get("schema_version") != _MANIFEST_SCHEMA
            or manifest.get("build_id") != build_id
            or manifest.get("status") != "complete"
        ):
            raise ValueError("existing Hovi surface build has conflicting identity")
        return manifest_path
    build_root.mkdir(parents=True, exist_ok=True)
    atomic_write(build_root / "recipe.json", recipe_bytes)

    log("building raw 0.25 m full-height bootstrap evidence")
    structural_hist = accumulate_structural_histogram(
        source.local_path,
        xy_resolution_m=config.structural_resolution_m,
        z_bin_m=config.structural_z_bin_m,
        chunk_points=config.histogram_chunk_points,
        memory_limit_bytes=config.histogram_memory_limit_bytes,
        log=log,
    )
    context_hist = aggregate_histogram(structural_hist, config.context_resolution_m)
    structural_grid = structural_hist.grid
    context_grid = context_hist.grid
    context_modes = extract_vertical_modes(
        context_hist.counts,
        z_bin_m=context_hist.z_bin_m,
        z_base_m=context_hist.z_origin_m,
        mode_count=config.mode_count,
        mode_radius_bins=config.structural_mode_radius_bins,
        min_mode_support=config.context_min_mode_support,
        underburden_clearance_m=config.context_underburden_clearance_m,
    )
    bootstrap_step_m = (
        config.context_neighbor_step_m
        + config.tangent_max_slope * config.context_resolution_m
    )
    bootstrap_component_mask, bootstrap_diagnostics = discover_context_component(
        context_modes,
        context_hist.grid,
        context_hist.counts.sum(axis=1) > 0,
        config=config,
        max_neighbor_step_m=bootstrap_step_m,
    )
    bootstrap_sheet = select_conditioned_sheet(
        context_modes,
        context_hist.grid,
        config,
        min_mode_support=config.context_min_mode_support,
        neighbor_step_m=bootstrap_step_m,
        candidate_mask=bootstrap_component_mask,
    )
    context_bootstrap_tangent = predict_same_sheet_tangent(
        context_hist.grid,
        context_hist.grid,
        bootstrap_sheet.selected_height_m,
        bootstrap_sheet.direct_mask,
        neighbor_step_m=config.context_neighbor_step_m,
        radius_cells=config.tangent_radius_cells,
        minimum_cells=config.tangent_minimum_cells,
        maximum_residual_m=config.context_tangent_max_residual_m,
        maximum_slope=config.tangent_max_slope,
    )
    log("rebuilding 1 m full-height modes in bootstrap tangent coordinates")
    context_sheared_hist = accumulate_tangent_sheared_histogram(
        source.local_path,
        grid=context_grid,
        slope_x=context_bootstrap_tangent.parent_slope_x,
        slope_y=context_bootstrap_tangent.parent_slope_y,
        z_bin_m=config.structural_z_bin_m,
        maximum_slope=config.tangent_max_slope,
        chunk_points=config.histogram_chunk_points,
        memory_limit_bytes=config.histogram_memory_limit_bytes,
        stage_name="context",
        log=log,
    )
    context_sheared_modes = extract_vertical_modes(
        context_sheared_hist.counts,
        z_bin_m=context_sheared_hist.z_bin_m,
        z_base_m=context_sheared_hist.z_origin_m,
        mode_count=config.mode_count,
        mode_radius_bins=config.structural_mode_radius_bins,
        min_mode_support=config.context_min_mode_support,
        underburden_clearance_m=config.context_underburden_clearance_m,
    )
    component_mask, component_diagnostics = discover_context_component(
        context_sheared_modes,
        context_grid,
        context_sheared_hist.counts.sum(axis=1) > 0,
        config=config,
        max_neighbor_step_m=config.context_neighbor_step_m,
        continuation_slope_x=context_bootstrap_tangent.parent_slope_x,
        continuation_slope_y=context_bootstrap_tangent.parent_slope_y,
    )
    component_diagnostics["bootstrap"] = bootstrap_diagnostics
    context_sheet = select_conditioned_sheet(
        context_sheared_modes,
        context_grid,
        config,
        min_mode_support=config.context_min_mode_support,
        neighbor_step_m=config.context_neighbor_step_m,
        continuation_slope_x=context_bootstrap_tangent.parent_slope_x,
        continuation_slope_y=context_bootstrap_tangent.parent_slope_y,
        candidate_mask=component_mask,
    )
    component_coverage = float(component_diagnostics.get("selected_coverage_fraction", 0.0))
    log(
        f"context component coverage {component_coverage:.1%}; "
        f"direct {np.mean(context_sheet.direct_mask):.1%}"
    )

    context_tangent = predict_same_sheet_tangent(
        context_grid,
        structural_grid,
        context_sheet.selected_height_m,
        context_sheet.direct_mask,
        prior_slope_x=context_bootstrap_tangent.parent_slope_x,
        prior_slope_y=context_bootstrap_tangent.parent_slope_y,
        neighbor_step_m=config.context_neighbor_step_m,
        radius_cells=config.tangent_radius_cells,
        minimum_cells=config.tangent_minimum_cells,
        maximum_residual_m=config.context_tangent_max_residual_m,
        maximum_slope=config.tangent_max_slope,
    )
    context_reference = context_tangent.child_center_z_m
    del structural_hist, context_hist, context_modes, context_sheared_hist, context_sheared_modes
    log("building 0.25 m full-height modes in context tangent coordinates")
    structural_hist = accumulate_tangent_sheared_histogram(
        source.local_path,
        grid=structural_grid,
        slope_x=context_tangent.child_slope_x,
        slope_y=context_tangent.child_slope_y,
        z_bin_m=config.structural_z_bin_m,
        maximum_slope=config.tangent_max_slope,
        chunk_points=config.histogram_chunk_points,
        memory_limit_bytes=config.histogram_memory_limit_bytes,
        stage_name="structural",
        log=log,
    )
    structural_modes = extract_vertical_modes(
        structural_hist.counts,
        z_bin_m=structural_hist.z_bin_m,
        z_base_m=structural_hist.z_origin_m,
        mode_count=config.mode_count,
        mode_radius_bins=config.structural_mode_radius_bins,
        min_mode_support=config.structural_min_mode_support,
        underburden_clearance_m=config.structural_underburden_clearance_m,
    )
    structural_sheet = select_conditioned_sheet(
        structural_modes,
        structural_hist.grid,
        config,
        min_mode_support=config.structural_min_mode_support,
        neighbor_step_m=config.structural_neighbor_step_m,
        parent_height_m=context_reference,
        parent_tolerance_m=config.structural_parent_tolerance_m,
        continuation_slope_x=context_tangent.child_slope_x,
        continuation_slope_y=context_tangent.child_slope_y,
    )
    del structural_modes

    with laspy.open(source.local_path) as reader:
        mins = np.asarray(reader.header.mins, dtype=np.float64)
        maxs = np.asarray(reader.header.maxs, dtype=np.float64)
    final_grid = grid_from_bounds(
        config.final_resolution_m, mins[0], mins[1], maxs[0], maxs[1]
    )
    expected_grid = next(
        item
        for item in support_inventory["support_grids"]
        if item["resolution_m"] == config.final_resolution_m
    )
    if (
        expected_grid["shape_yx"] != [final_grid.height, final_grid.width]
        or not np.allclose(
            expected_grid["origin_xy_m"], [final_grid.origin_x_m, final_grid.origin_y_m], atol=1e-12
        )
    ):
        raise ValueError("Hovi final grid differs from the raw-support inventory")
    with np.load(support_npz_path, allow_pickle=False) as support_arrays:
        support_key = "point_count__0p0625m"
        if support_key not in support_arrays:
            raise ValueError("Hovi raw-support NPZ lacks the final-grid point counts")
        raw_support_count = support_arrays[support_key]
    if raw_support_count.shape != (final_grid.height, final_grid.width):
        raise ValueError("Hovi raw-support final-grid counts have the wrong shape")
    raw_observed_fine = raw_support_count > 0
    fine_tangent = predict_same_sheet_tangent(
        structural_hist.grid,
        final_grid,
        structural_sheet.selected_height_m,
        structural_sheet.direct_mask,
        prior_slope_x=context_tangent.child_slope_x,
        prior_slope_y=context_tangent.child_slope_y,
        neighbor_step_m=config.structural_neighbor_step_m,
        radius_cells=config.tangent_radius_cells,
        minimum_cells=config.tangent_minimum_cells,
        maximum_residual_m=config.structural_tangent_max_residual_m,
        maximum_slope=config.tangent_max_slope,
    )
    structural_reference = fine_tangent.child_center_z_m
    del structural_hist

    log("building 0.0625 m windowed vertical-mode evidence")
    fine_hist = accumulate_windowed_histogram(
        source.local_path,
        grid=final_grid,
        reference_z_m=structural_reference,
        slope_x=fine_tangent.child_slope_x,
        slope_y=fine_tangent.child_slope_y,
        half_window_m=config.final_half_window_m,
        z_bin_m=config.final_z_bin_m,
        chunk_points=config.histogram_chunk_points,
        memory_limit_bytes=config.histogram_memory_limit_bytes,
        log=log,
    )
    fine_modes = extract_vertical_modes(
        fine_hist.counts,
        z_bin_m=fine_hist.z_bin_m,
        z_base_m=structural_reference.reshape(-1) - config.final_half_window_m,
        mode_count=config.mode_count,
        mode_radius_bins=config.final_mode_radius_bins,
        min_mode_support=config.final_min_mode_support,
        underburden_clearance_m=config.final_underburden_clearance_m,
    )
    fine_sheet = select_conditioned_sheet(
        fine_modes,
        final_grid,
        config,
        min_mode_support=config.final_min_mode_support,
        neighbor_step_m=config.final_neighbor_step_m,
        parent_height_m=structural_reference,
        parent_tolerance_m=config.final_parent_tolerance_m,
        continuation_slope_x=fine_tangent.child_slope_x,
        continuation_slope_y=fine_tangent.child_slope_y,
    )
    tangent_window_observed = (
        fine_hist.counts.sum(axis=1, dtype=np.uint64).reshape(final_grid.height, final_grid.width)
        + fine_hist.below_window_count
        + fine_hist.above_window_count
    ) > 0
    nominal_layout_mask, local_center = _nominal_core(condition, final_grid)
    pre_erosion_domain = raw_observed_fine & nominal_layout_mask
    analysis_core = _eroded_analysis_core(
        raw_observed_fine & nominal_layout_mask,
        final_grid,
        config.analysis_core_erosion_m,
    )
    del fine_modes, fine_hist

    footprint_half_band = (
        (config.final_mode_radius_bins + 0.5) * config.final_z_bin_m
        + config.footprint_registration_allowance_m
    )
    log("measuring selected fine-mode XY footprint support")
    footprint = accumulate_selected_mode_footprint(
        source.local_path,
        grid=final_grid,
        selected_center_z_m=fine_sheet.selected_height_m,
        slope_x=fine_tangent.child_slope_x,
        slope_y=fine_tangent.child_slope_y,
        half_band_m=footprint_half_band,
        subcell_axis=config.footprint_subcell_axis,
        chunk_points=config.histogram_chunk_points,
        log=log,
    )
    footprint_valid = (
        (footprint.selected_point_count >= config.final_min_mode_support)
        & (footprint.quadrant_count >= config.minimum_footprint_quadrants)
        & (footprint.subcell_count >= config.minimum_footprint_subcells)
    )
    direct_candidate_mask = (
        fine_sheet.direct_mask
        & footprint_valid
        & fine_tangent.child_valid_mask
        & analysis_core
    )
    direct_mask, topology_rejected_mask, topology_diagnostics = audit_direct_topology(
        direct_candidate_mask,
        fine_sheet.selected_height_m,
        structural_reference,
        fine_tangent.child_slope_x,
        fine_tangent.child_slope_y,
        final_grid,
        edge_residual_limit_m=config.final_neighbor_step_m,
        minimum_component_cells=config.topology_minimum_component_cells,
        structural_anchor_residual_m=config.topology_structural_anchor_residual_m,
        minimum_anchor_fraction=config.topology_minimum_anchor_fraction,
    )

    eligible_interpolation = (
        analysis_core
        & np.isfinite(structural_reference)
        & ~fine_sheet.ambiguity_mask
        & ~fine_sheet.above_sheet_competitor_mask
    )
    interpolated_height, interpolation_mask = interpolate_small_supported_holes(
        fine_sheet.selected_height_m,
        direct_mask,
        fine_sheet.selection_score,
        eligible_interpolation,
        fine_sheet.ambiguity_mask,
        fine_sheet.above_sheet_competitor_mask,
        analysis_core,
        final_grid,
        config,
    )
    absolute = np.full(fine_sheet.selected_height_m.shape, np.nan, dtype=np.float32)
    absolute[direct_mask] = fine_sheet.selected_height_m[direct_mask]
    absolute[interpolation_mask] = interpolated_height[interpolation_mask]
    inferred = interpolation_mask
    unsupported = analysis_core & ~(direct_mask | inferred)
    selection_score = fine_sheet.selection_score.copy()
    selection_score[inferred] = np.minimum(
        0.35, ndimage.maximum_filter(selection_score, size=3)[inferred]
    )
    selection_score[~analysis_core] = 0.0

    reasons = np.zeros(absolute.shape, dtype=np.uint16)
    has_mode = np.isfinite(fine_sheet.selected_height_m)
    reasons[has_mode] |= int(Reason.DIRECT_VERTICAL_MODE)
    reasons[fine_sheet.broad_window_consistent_mask] |= int(Reason.BROAD_TANGENT_WINDOW)
    reasons[fine_sheet.neighbor_consistent_mask] |= int(Reason.NEIGHBOR_CONSISTENT)
    reasons[fine_sheet.ambiguity_mask] |= int(Reason.AMBIGUOUS_COMPETITOR)
    reasons[fine_sheet.above_sheet_competitor_mask] |= int(Reason.ABOVE_SHEET_COMPETITOR)
    reasons[unsupported] |= int(Reason.UNSUPPORTED)
    reasons[inferred] |= int(Reason.INTERPOLATED_SMALL_HOLE)
    reasons[has_mode & (fine_sheet.selection_score < config.min_direct_selection_score)] |= int(
        Reason.LOW_SELECTION_SCORE
    )
    reasons[~analysis_core] |= int(Reason.OUTSIDE_ANALYSIS_CORE)
    reasons[has_mode & ~footprint_valid] |= int(Reason.INSUFFICIENT_XY_FOOTPRINT)
    reasons[~fine_tangent.child_valid_mask] |= int(Reason.INVALID_TANGENT_REFERENCE)
    reasons[topology_rejected_mask] |= int(Reason.DISCONNECTED_DIRECT_CANDIDATE)
    reasons[
        fine_sheet.selected_underburden_fraction > config.max_selected_underburden_fraction
    ] |= int(Reason.SUBSTANTIAL_UNDERBURDEN)
    candidate = SurfaceCandidate(
        grid=final_grid,
        absolute_local_z_m=absolute,
        mode_candidate_z_m=fine_sheet.selected_height_m,
        selection_score=selection_score,
        reason_bits=reasons,
        direct_mask=direct_mask,
        inferred_mask=inferred,
        ambiguity_mask=fine_sheet.ambiguity_mask,
        above_sheet_competitor_mask=fine_sheet.above_sheet_competitor_mask,
        unsupported_mask=unsupported,
        interpolation_mask=interpolation_mask,
        analysis_core_mask=analysis_core,
        nominal_layout_mask=nominal_layout_mask,
        observed_fine_mask=raw_observed_fine,
        tangent_window_observed_mask=tangent_window_observed,
        tangent_reference_valid_mask=fine_tangent.child_valid_mask,
        pre_erosion_domain_mask=pre_erosion_domain,
        topology_rejected_mask=topology_rejected_mask,
        selected_support_count=footprint.selected_point_count,
        selected_underburden_fraction=fine_sheet.selected_underburden_fraction,
        competing_mode_ratio=fine_sheet.competing_mode_ratio,
        footprint_quadrant_count=footprint.quadrant_count,
        footprint_subcell_count=footprint.subcell_count,
        structural_reference_z_m=structural_reference,
    )
    metrics = _quality_metrics(candidate)
    diagnostic_candidate = candidate
    geometry_pass = (
        component_coverage >= config.minimum_context_component_coverage
        and metrics["direct_fraction"] >= config.minimum_analysis_direct_coverage
        and metrics["ambiguity_fraction"] <= config.maximum_analysis_ambiguity_fraction
        and metrics["largest_direct_component_fraction"] >= 0.45
    )
    disposition = "preview_geometry_plausible_for_q1" if geometry_pass else "abstention"
    authorized_absolute = np.full_like(candidate.absolute_local_z_m, np.nan)
    authorized_mask = np.zeros_like(candidate.direct_mask, dtype=np.uint8)

    arrays = {
        "preview_absolute_local_z_m": candidate.absolute_local_z_m,
        "mode_candidate_z_m": candidate.mode_candidate_z_m,
        "selection_score": candidate.selection_score,
        "reason_bits": candidate.reason_bits,
        "preview_direct_mask": candidate.direct_mask.astype(np.uint8),
        "preview_inferred_mask": candidate.inferred_mask.astype(np.uint8),
        "authorized_absolute_local_z_m": authorized_absolute,
        "authorized_surface_mask": authorized_mask,
        "ambiguity_mask": candidate.ambiguity_mask.astype(np.uint8),
        "above_sheet_competitor_mask": candidate.above_sheet_competitor_mask.astype(np.uint8),
        "unsupported_mask": candidate.unsupported_mask.astype(np.uint8),
        "interpolation_mask": candidate.interpolation_mask.astype(np.uint8),
        "analysis_core_mask": candidate.analysis_core_mask.astype(np.uint8),
        "nominal_layout_mask": candidate.nominal_layout_mask.astype(np.uint8),
        "observed_fine_mask": candidate.observed_fine_mask.astype(np.uint8),
        "tangent_window_observed_mask": candidate.tangent_window_observed_mask.astype(np.uint8),
        "tangent_reference_valid_mask": candidate.tangent_reference_valid_mask.astype(np.uint8),
        "pre_erosion_domain_mask": candidate.pre_erosion_domain_mask.astype(np.uint8),
        "topology_rejected_mask": candidate.topology_rejected_mask.astype(np.uint8),
        "selected_support_count": candidate.selected_support_count,
        "selected_underburden_fraction": candidate.selected_underburden_fraction,
        "competing_mode_ratio": candidate.competing_mode_ratio,
        "footprint_quadrant_count": candidate.footprint_quadrant_count,
        "footprint_subcell_count": candidate.footprint_subcell_count,
        "structural_reference_z_m": candidate.structural_reference_z_m,
    }
    temporary_npz = build_root / ".candidate-surface.npz.part"
    write_deterministic_npz(temporary_npz, arrays)
    npz_path, npz_sha = store_content_file(build_root, temporary_npz, "candidate-surface.npz")
    evidence_record = {
        "schema_version": _EVIDENCE_SCHEMA,
        "build_id": build_id,
        "disposition": disposition,
        "scientific_disposition": {
            "role": "raw_candidate",
            "qualification_status": "unqualified",
            "target_truth": False,
            "usable_surface": False,
            "transfer_ceiling": "none",
            "synthesis_authorized": False,
        },
        "semantic_boundary": {
            "bare_earth_claim": False,
            "stable_surface_truth_claim": False,
            "usable_surface_claim": False,
            "photos_used_as_dense_labels": False,
            "photos_metric_registered": False,
            "geometry_mask_meaning": (
                "vertical and spatial evidence only; it cannot distinguish moss, litter, "
                "root, clast, deadwood, stem, or living vegetation semantics"
            ),
            "preserved_gap": (
                "semantic photo QA visibly confirms mixed moss/litter, roots, deadwood/stems, "
                "and understory, but supplies no per-cell ownership labels"
            ),
        },
        "method": {
            "lowest_point_filtering": False,
            "decorative_smoothing_or_noise": False,
            "direct_height_alteration": False,
            "selection": (
                "raw full-height bootstrap -> tangent-sheared full-height 1 m modes -> "
                "one-label-or-null robust context sheet -> tangent-sheared full-height "
                "0.25 m modes -> tangent-residual 0.0625 m modes -> selected-mode XY "
                "footprint gate -> emitted-direct topology audit"
            ),
            "local_relief_policy": (
                "direct fine-cell mode centroids are retained unchanged when support, "
                "one-sidedness, parent, and neighbor gates pass"
            ),
        },
        "grid": final_grid.as_record(),
        "nominal_plot_local_center_xy_m": list(local_center),
        "registration": registration,
        "context_component": component_diagnostics,
        "quality_metrics_before_abstention": metrics,
        "direct_topology_audit": topology_diagnostics,
        "analysis_domain": {
            "nominal_layout": _mask_metrics(nominal_layout_mask, final_grid.resolution_m),
            "observed_within_nominal": _mask_metrics(
                raw_observed_fine & nominal_layout_mask, final_grid.resolution_m
            ),
            "tangent_window_observed_within_nominal": _mask_metrics(
                tangent_window_observed & nominal_layout_mask,
                final_grid.resolution_m,
            ),
            "tangent_valid_within_nominal": _mask_metrics(
                fine_tangent.child_valid_mask & nominal_layout_mask,
                final_grid.resolution_m,
            ),
            "pre_erosion_outer_domain": _mask_metrics(
                pre_erosion_domain, final_grid.resolution_m
            ),
            "eroded_analysis_core": _mask_metrics(
                analysis_core, final_grid.resolution_m
            ),
            "erosion_m": config.analysis_core_erosion_m,
        },
        "scale_support": {
            "context_direct_cells": int(np.count_nonzero(context_sheet.direct_mask)),
            "context_tangent_cells": int(
                np.count_nonzero(context_tangent.parent_valid_mask)
            ),
            "structural_direct_cells": int(
                np.count_nonzero(structural_sheet.direct_mask)
            ),
            "structural_tangent_cells": int(
                np.count_nonzero(fine_tangent.parent_valid_mask)
            ),
        },
        "footprint_observation": {
            "half_band_m": footprint_half_band,
            "decoded_points": footprint.decoded_points,
            "candidate_points": footprint.candidate_points,
            "included_points": footprint.included_points,
            "selected_point_count_all_modes": _distribution(
                footprint.selected_point_count[np.isfinite(fine_sheet.selected_height_m)]
            ),
            "quadrant_count_all_modes": _distribution(
                footprint.quadrant_count[np.isfinite(fine_sheet.selected_height_m)]
            ),
            "subcell_count_all_modes": _distribution(
                footprint.subcell_count[np.isfinite(fine_sheet.selected_height_m)]
            ),
        },
        "gates": {
            "geometry_pass": geometry_pass,
            "sole_consequence": "warrant_Q1_full_scan_audit",
            "usable_surface": False,
            "minimum_context_component_coverage": config.minimum_context_component_coverage,
            "minimum_analysis_direct_coverage": config.minimum_analysis_direct_coverage,
            "maximum_analysis_ambiguity_fraction": config.maximum_analysis_ambiguity_fraction,
            "minimum_largest_direct_component_fraction": 0.45,
        },
        "full_scan_requirement": (
            "Q1 full scans are still required to audit independent views, incidence, "
            "occlusion, thinning bias, and scan disagreement before any qualification"
        ),
        "authorized_surface": {
            "present": False,
            "reason": "Q1 preview extraction cannot authorize target truth or synthesis",
            "npz_arrays": [
                "authorized_absolute_local_z_m",
                "authorized_surface_mask",
            ],
        },
        "inputs": {
            "source_sha256": source.sha256,
            "raw_support_manifest_sha256": support_manifest_sha,
            "condition_evidence_sha256": condition_sha,
            "semantic_photo_qa_sha256": photo_qa_sha,
        },
        "candidate_npz": {
            **artifact_ref(build_root, npz_path, npz_sha),
            "arrays": _array_record(arrays),
        },
        "reason_bits": {reason.name: int(reason) for reason in Reason},
    }
    evidence_bytes = canonical_json_bytes(evidence_record)
    evidence_path, evidence_sha = store_content_bytes(
        build_root, evidence_bytes, "candidate-surface-evidence.json"
    )

    qa_records = []
    for path, interpretation in render_surface_qa(
        diagnostic_candidate,
        build_root / "qa",
        disposition=disposition,
    ):
        with Image.open(path) as image:
            dimensions = list(image.size)
        qa_records.append(
            {
                "path": path.relative_to(build_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "dimensions_xy": dimensions,
                "interpretation": interpretation,
                "preview_only_unusable": True,
                "abstention_diagnostic": not geometry_pass,
            }
        )
    qa_index = {
        "schema_version": _QA_SCHEMA,
        "build_id": build_id,
        "disposition": disposition,
        "preview_only": True,
        "usable_surface": False,
        "source_sha256": source.sha256,
        "recipe_sha256": build_id,
        "candidate_evidence_sha256": evidence_sha,
        "candidate_npz_sha256": npz_sha,
        "semantic_photo_qa_sha256": photo_qa_sha,
        "images": qa_records,
    }
    qa_bytes = canonical_json_bytes(qa_index)
    qa_path = build_root / "qa" / "index.json"
    atomic_write(qa_path, qa_bytes)
    atomic_write(
        qa_path.with_suffix(".json.sha256"),
        (sha256_bytes(qa_bytes) + "\n").encode("ascii"),
    )
    manifest = {
        "schema_version": _MANIFEST_SCHEMA,
        "status": "complete",
        "build_id": build_id,
        "plot_id": _PLOT_ID,
        "disposition": disposition,
        "scientific_role": "raw_candidate",
        "qualification_status": "unqualified",
        "transfer_ceiling": "none",
        "synthesis_authorized": False,
        "usable_surface": False,
        "candidate_evidence": artifact_ref(build_root, evidence_path, evidence_sha),
        "candidate_npz": artifact_ref(build_root, npz_path, npz_sha),
        "qa_index": artifact_ref(build_root, qa_path, sha256_bytes(qa_bytes)),
    }
    manifest_bytes = canonical_json_bytes(manifest)
    atomic_write(manifest_path, manifest_bytes)
    atomic_write(
        manifest_path.with_suffix(".json.sha256"),
        (sha256_bytes(manifest_bytes) + "\n").encode("ascii"),
    )
    log(f"Hovi surface {disposition}: {manifest_path}")
    return manifest_path


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a conservative unqualified HY_SPRUCE4 candidate surface."
    )
    parser.add_argument("--retained", type=Path, required=True)
    parser.add_argument("--support-manifest", type=Path, required=True)
    parser.add_argument("--condition-evidence", type=Path, required=True)
    parser.add_argument("--photo-qa-index", type=Path, required=True)
    parser.add_argument(
        "--selection", type=Path, default=CONFIG_DIR / "hovi-public-targets.json"
    )
    parser.add_argument("--work-root", type=Path, default=DATA_WORK)
    args = parser.parse_args()
    path = extract_hovi_candidate_surface(
        args.retained,
        args.support_manifest,
        args.condition_evidence,
        args.photo_qa_index,
        selection_path=args.selection,
        work_root=args.work_root,
    )
    print(f"candidate surface manifest: {path}")


if __name__ == "__main__":
    _main()
