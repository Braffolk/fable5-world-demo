"""Atomic content-addressed Moore M0 pre-training materialization."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .....config import ASSET_GEN_ROOT, DATA_WORK
from ..provenance import numerical_environment_identity
from .archive import SourceGrid, load_all_grids
from .bands import AreaGrid, materialize_bands
from .contracts import canonical_json, load_frozen_design, sha256_file
from .gate import PlotMaterialization, evaluate_source_gate, materialize_plot
from .qa import render_qa
from .resample import materialize_four_phases


def _atomic_json(path: Path, value: Any) -> None:
    payload = canonical_json(value)
    temporary = path.with_name(path.name + ".part")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def _grid_arrays(prefix: str, grid: AreaGrid, output: dict[str, np.ndarray]) -> None:
    output[f"{prefix}_values_f64"] = grid.values.astype(np.float64, copy=False)
    output[f"{prefix}_valid"] = grid.valid.astype(np.bool_, copy=False)
    output[f"{prefix}_x_bounds_m_f64"] = grid.x_bounds_m.astype(np.float64, copy=False)
    output[f"{prefix}_y_bounds_m_f64"] = grid.y_bounds_m.astype(np.float64, copy=False)


def _save_plot(path: Path, plot: PlotMaterialization) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {
        "source_height_after_demsmooth_m_f64": plot.source.z_m,
        "source_finite": np.isfinite(plot.source.z_m),
        "source_x_centers_m_f64": plot.source.x_centers_m,
        "source_y_centers_m_f64": plot.source.y_centers_m,
        "overlay_x_bounds_m_f64": plot.overlay_x_bounds_m,
        "overlay_y_bounds_m_f64": plot.overlay_y_bounds_m,
        "four_phase_common_support": plot.common_support,
    }
    phase_counts = []
    for index, item in enumerate(plot.phases):
        prefix = f"phase_{index}"
        _grid_arrays(f"{prefix}_height", item.phase.height, arrays)
        arrays[f"{prefix}_finite_overlap_fraction_f64"] = item.phase.finite_overlap_fraction
        arrays[f"{prefix}_overlap_valid"] = item.phase.overlap_valid
        arrays[f"{prefix}_target_valid"] = item.phase.target_valid
        arrays[f"{prefix}_boundary_or_incomplete"] = item.phase.boundary_or_incomplete
        for name in ("a1", "a0", "o1_q4", "o0_q4q4", "b1", "derived_b2"):
            _grid_arrays(f"{prefix}_{name}", getattr(item, name), arrays)
        arrays[f"{prefix}_complete_common_cells"] = plot.complete_common_cells[index]
        phase_counts.append(
            {
                "phase": index,
                "offset_xy_m": list(item.phase.offset_xy_m),
                "target_valid_cells": int(np.count_nonzero(item.phase.target_valid)),
                "a1_valid_cells": int(np.count_nonzero(item.a1.valid)),
                "a0_valid_cells": int(np.count_nonzero(item.a0.valid)),
                "strict_b1_valid_cells": int(np.count_nonzero(item.b1.valid)),
                "derived_b2_valid_cells": int(np.count_nonzero(item.derived_b2.valid)),
                "complete_common_b2_cells": int(np.count_nonzero(plot.complete_common_cells[index])),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as target:
        np.savez_compressed(target, **arrays)
        target.flush()
        os.fsync(target.fileno())
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "plot_id": plot.source.plot_id,
        "group_id": plot.source.group_id,
        "archive_member": plot.source.archive_member,
        "archive_member_sha256": plot.source.archive_member_sha256,
        "raw_z_sha256": plot.source.raw_z_sha256,
        "filtered_z_sha256": plot.source.filtered_z_sha256,
        "source_shape_yx": list(plot.source.z_m.shape),
        "source_finite_cells": plot.source.finite_count,
        "source_finite_area_m2": plot.source.finite_count * 0.0001,
        "source_boundary_origin_xy_m": list(plot.source.source_boundary_origin_xy_m),
        "orientation": plot.source.orientation,
        "phase_counts": phase_counts,
        "window_candidates": len(plot.windows),
        "accepted_windows": sum(int(row["accepted"]) for row in plot.windows),
        "spectral_core_count": sum(len(cores) for cores in plot.spectral_cores),
        "eligible_form_count": sum(len(forms) for forms in plot.forms),
    }


def _splits(group_order: list[str], plots: list[PlotMaterialization]) -> dict[str, Any]:
    by_group = {
        group: sorted(plot.source.plot_id for plot in plots if plot.source.group_id == group)
        for group in group_order
    }
    all_plot_ids = [plot.source.plot_id for plot in plots]
    if len(set(all_plot_ids)) != len(all_plot_ids):
        raise AssertionError("duplicate Moore plot identity")
    if set().union(*(set(values) for values in by_group.values())) != set(all_plot_ids):
        raise AssertionError("plot leakage-group coverage failed")
    outer = []
    for held_out in group_order:
        development = [group for group in group_order if group != held_out]
        inner = [
            {
                "selection_group": selection,
                "training_groups": [group for group in development if group != selection],
            }
            for selection in development
        ]
        outer.append(
            {
                "held_out_group": held_out,
                "development_groups": development,
                "inner_leave_one_group_out": inner,
            }
        )
    return {
        "independent_unit": "geographic_group",
        "group_plot_ids": by_group,
        "outer_leave_one_group_out": outer,
        "final_frozen_audit_group": "nobel",
        "final_preview_plot": "Lambda",
        "random_plot_or_window_split_allowed": False,
        "seney_members_cross_split": False,
    }


def _recipe(design: Any, member_ledger: dict[str, dict[str, object]]) -> dict[str, Any]:
    module_root = Path(__file__).parent
    return {
        "schema_version": "moore-m0-materialization-recipe/1.0.0",
        "checkpoint": design.value["first_executable_checkpoint"]["id"],
        "design": {
            "path": design.path.relative_to(design.repository_root).as_posix(),
            "sha256": design.sha256,
            "design_id": design.value["design_id"],
        },
        "normative_sha256": design.normative_sha256,
        "archive": {
            "path": design.archive_path.relative_to(design.repository_root).as_posix(),
            "sha256": design.value["source"]["archive"]["sha256"],
            "bytes": design.value["source"]["archive"]["bytes"],
            "members": member_ledger,
        },
        "operators": {
            "publisher_observation": design.value["source"]["publisher_observation_operator"],
            "phases": design.value["phases"],
            "masks": design.value["masks"],
            "windows": design.value["windows"],
            "source_b2_gate": design.value["source_b2_gate"],
        },
        "implementation": {
            "modules": {path.name: sha256_file(path) for path in sorted(module_root.glob("*.py"))},
            "environment": numerical_environment_identity(design.repository_root),
        },
        "authority": {
            "model_training": False,
            "candidate_selection": False,
            "packing_or_preview": False,
            "latest_pointer_update": False,
        },
    }


def _verify_existing(path: Path, recipe_id: str) -> Path:
    manifest = json.loads(path.read_bytes())
    if manifest.get("recipe_id") != recipe_id or manifest.get("status") != "complete":
        raise ValueError("existing Moore materialization conflicts with the current recipe")
    root = path.parent
    if sha256_file(root / "recipe.json") != manifest["recipe_sha256"]:
        raise ValueError("existing Moore recipe changed")
    for row in manifest["plots"]:
        artifact = root / row["path"]
        if sha256_file(artifact) != row["sha256"] or artifact.stat().st_size != row["bytes"]:
            raise ValueError(f"existing Moore plot artifact changed: {row['path']}")
    for field in ("windows_and_splits", "source_b2_gate"):
        row = manifest[field]
        artifact = root / row["path"]
        if sha256_file(artifact) != row["sha256"]:
            raise ValueError(f"existing Moore {field} artifact changed")
    for row in manifest["qa"]:
        artifact = root / row["path"]
        if sha256_file(artifact) != row["image_sha256"] or artifact.stat().st_size != row["bytes"]:
            raise ValueError(f"existing Moore QA artifact changed: {row['path']}")
    if sha256_file(root / "qa" / "index.json") != manifest["qa_index_sha256"]:
        raise ValueError("existing Moore QA index changed")
    return path


def materialize_moore_m0(output_root: Path | None = None, *, log=print) -> Path:
    design = load_frozen_design(ASSET_GEN_ROOT.parent)
    base = output_root or DATA_WORK / "microtopography" / "moore" / "m0-materialization" / "sha256"
    base.mkdir(parents=True, exist_ok=True)
    extraction = base / f".extracting-{os.getpid()}"
    if extraction.exists():
        raise ValueError("stale Moore extraction directory requires inspection")
    extraction.mkdir()
    try:
        grids, member_ledger = load_all_grids(design, extraction)
    except BaseException:
        if extraction.exists():
            shutil.rmtree(extraction)
        raise
    recipe = _recipe(design, member_ledger)
    recipe_id = hashlib.sha256(canonical_json(recipe)).hexdigest()
    final_root = base / recipe_id
    manifest_path = final_root / "manifest.json"
    if manifest_path.exists():
        return _verify_existing(manifest_path, recipe_id)
    temporary = base / f".{recipe_id}.preparing-{os.getpid()}"
    if temporary.exists() or final_root.exists():
        raise ValueError("incomplete Moore artifact requires inspection")
    temporary.mkdir()
    try:
        _atomic_json(temporary / "recipe.json", recipe)
        group_order = [row["id"] for row in design.value["geographic_groups"]]
        plots: list[PlotMaterialization] = []
        for plot_order, source in enumerate(grids):
            phase_bands = tuple(materialize_bands(phase) for phase in materialize_four_phases(source))
            plot = materialize_plot(
                source, phase_bands, group_order=group_order.index(source.group_id), plot_order=plot_order
            )
            plots.append(plot)
            log(f"Moore M0 materialized {plot_order + 1}/68: {source.group_id}/{source.plot_id}")
        gate, group_rows = evaluate_source_gate(
            plots, group_order, design.value["source_b2_gate"]["thresholds"]
        )
        plot_rows = []
        for plot in plots:
            relative = Path("plots") / plot.source.group_id / f"{plot.source.plot_id}.npz"
            row = _save_plot(temporary / relative, plot)
            row["path"] = relative.as_posix()
            plot_rows.append(row)
        windows = {
            "schema_version": "moore-m0-windows/1.0.0",
            "recipe_id": recipe_id,
            "window_contract": design.value["windows"],
            "plots": [
                {
                    "plot_id": plot.source.plot_id,
                    "group_id": plot.source.group_id,
                    "windows": plot.windows,
                }
                for plot in plots
            ],
            "splits": _splits(group_order, plots),
        }
        _atomic_json(temporary / "windows-and-splits.json", windows)
        _atomic_json(temporary / "source-b2-gate.json", {"result": gate, "groups": group_rows})
        preview = next(plot for plot in plots if plot.source.plot_id == "Lambda")
        qa = render_qa(
            temporary / "qa", build_id=recipe_id, design_sha256=design.sha256,
            archive_sha256=design.value["source"]["archive"]["sha256"],
            preview=preview, group_rows=group_rows,
        )
        qa_index = {
            "schema_version": "moore-m0-materialization-qa/1.0.0",
            "recipe_id": recipe_id,
            "artifacts": qa,
        }
        _atomic_json(temporary / "qa" / "index.json", qa_index)
        finite_count = sum(plot.source.finite_count for plot in plots)
        manifest = {
            "schema_version": "moore-m0-materialization/1.0.0",
            "status": "complete",
            "recipe_id": recipe_id,
            "recipe_sha256": sha256_file(temporary / "recipe.json"),
            "design_sha256": design.sha256,
            "archive_sha256": design.value["source"]["archive"]["sha256"],
            "inventory": {
                "grid_count": len(plots),
                "rec_grid_count": sum(plot.source.plot_id.startswith("REC_") for plot in plots),
                "named_dems_grid_count": sum(not plot.source.plot_id.startswith("REC_") for plot in plots),
                "finite_cell_count": finite_count,
                "finite_area_m2": finite_count * 0.0001,
                "geographic_group_count": len(group_order),
            },
            "plots": plot_rows,
            "windows_and_splits": {
                "path": "windows-and-splits.json",
                "sha256": sha256_file(temporary / "windows-and-splits.json"),
            },
            "source_b2_gate": {
                "path": "source-b2-gate.json",
                "sha256": sha256_file(temporary / "source-b2-gate.json"),
                **gate,
            },
            "window_and_split_checks": {
                "fixed_window_shapes_enumerated": True,
                "accepted_window_count": sum(row["accepted_windows"] for row in plot_rows),
                "complete_spectral_core_count": sum(row["spectral_core_count"] for row in plot_rows),
                "geographic_group_partition_complete": True,
                "seney_members_cross_split": False,
                "random_window_or_plot_split_used": False,
                "leakage_check_passed": True,
                "capacity_sufficient_for_training": False,
            },
            "qa": qa,
            "qa_index_sha256": sha256_file(temporary / "qa" / "index.json"),
            "training_authorized": False,
            "D_training_authorized": False,
            "R_training_authorized": False,
            "E_execution_authorized": False,
            "publication_authorized": False,
            "forbidden_claims": design.value["forbidden_claims"],
        }
        _atomic_json(temporary / "manifest.json", manifest)
        temporary.replace(final_root)
    except BaseException:
        # Preserve a failed preparation for forensic inspection; it can never be
        # confused with a complete content-addressed artifact.
        raise
    log(f"Moore M0 materialization complete: {recipe_id}; gate={gate['state']}")
    return final_root / "manifest.json"


if __name__ == "__main__":
    print(materialize_moore_m0())
