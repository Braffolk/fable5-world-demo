"""Immutable correlated within-site crop evaluation artifact."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .....config import DATA_WORK
from .artifact import (
    _canonical_bytes,
    _implementation_identity,
    _sha256_file,
    _site_metrics,
    _write_deterministic_npz,
)
from .diagnostics import rotation_error
from .forms import build_form_plan, render_fine_surface
from .model import CropEvaluation, ProcessConfig, SlopeDomain
from .process import solve_process
from .qa import render_qa


def _shared_solve_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    process_grid = {
        key: value
        for key, value in metrics["process_grid"].items()
        if key != "crop_material_supported_samples"
    }
    forms = {
        key: metrics["forms"][key]
        for key in (
            "channel_heads",
            "retained_head_candidates",
            "streamlines",
            "rill_centerline_samples",
            "headcut_samples",
            "seep_samples",
            "maximum_support_radius_m",
            "collar_m",
        )
    }
    gates = {
        key: metrics["gates"][key]
        for key in ("sediment_mass_conservative", "support_strictly_inside_collar")
    }
    return {
        "source_site_id": metrics["site_id"],
        "process_grid": process_grid,
        "water_budget_m3": metrics["water_budget_m3"],
        "sediment_budget_kg": metrics["sediment_budget_kg"],
        "relief_budget_m3": metrics["relief_budget_m3"],
        "forms": forms,
        "gates": gates,
        "hard_gate_pass": all(gates.values()),
    }


def _correlated_crop_metrics(
    evaluation: CropEvaluation,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    forms = {
        key: metrics["forms"][key]
        for key in (
            "typed_crop_fraction",
            "residual_abs_p99_m",
            "residual_min_m",
            "residual_max_m",
        )
    }
    crop_gate_names = (
        "partition_exact",
        "domain_enlargement_residual_max_abs_le_0_001_m",
        "domain_enlargement_typed_ownership_exact",
        "hard_exclusion_exact",
        "recognizable_connected_form_proxy",
    )
    gates = {key: metrics["gates"][key] for key in crop_gate_names}
    hard_gate_names = tuple(
        key for key in crop_gate_names if key != "recognizable_connected_form_proxy"
    )
    return {
        "crop_id": evaluation.crop_id,
        "source_site_id": evaluation.source_site_id,
        "solve_group_id": evaluation.solve_group_id,
        "evidence_role": evaluation.evidence_role,
        "output_chunk": list(evaluation.output_chunk),
        "bbox_en": list(evaluation.bbox_en),
        "fine_grid": {
            "shape": [2049, 2049],
            "texel_m": 0.0625,
            "material_supported_samples": metrics["process_grid"][
                "crop_material_supported_samples"
            ],
        },
        "forms": forms,
        "causal_ablation": metrics["causal_ablation"],
        "invariance": metrics["invariance"],
        "gates": gates,
        "shared_solve_gate_references": [
            "sediment_mass_conservative",
            "support_strictly_inside_collar",
        ],
        "hard_gate_pass": all(gates[key] for key in hard_gate_names),
        "visible_morphology_gate_pass": gates["recognizable_connected_form_proxy"],
        "crop_result": metrics["site_result"],
    }


def _surface_edge(surface: Any, column: int) -> dict[str, np.ndarray]:
    return {
        "c0_height_m": surface.c0_height_m[:, column].copy(),
        "c1_height_m": surface.c1_height_m[:, column].copy(),
        "residual_m": surface.residual_m[:, column].copy(),
        "incision_m": surface.incision_m[:, column].copy(),
        "deposition_m": surface.deposition_m[:, column].copy(),
        "ownership": surface.ownership[:, column].copy(),
        "hard_exclusion": surface.hard_exclusion[:, column].copy(),
        "material_supported": surface.material_supported[:, column].copy(),
    }


def _shared_edge_metrics(
    west: dict[str, np.ndarray],
    east: dict[str, np.ndarray],
    *,
    boundary_e_m: float,
) -> dict[str, Any]:
    floating = {
        name: (west[name], east[name])
        for name in (
            "c0_height_m",
            "c1_height_m",
            "residual_m",
            "incision_m",
            "deposition_m",
        )
    }
    categorical = {
        name: (west[name], east[name])
        for name in ("ownership", "hard_exclusion", "material_supported")
    }
    maximum = {
        name: float(np.max(np.abs(left - right), initial=0.0))
        for name, (left, right) in floating.items()
    }
    exact = {
        name: bool(np.array_equal(left, right))
        for name, (left, right) in {**floating, **categorical}.items()
    }
    return {
        "shared_boundary_e_m": boundary_e_m,
        "samples": int(west["c1_height_m"].shape[0]),
        "floating_max_abs": maximum,
        "array_exact": exact,
        "all_arrays_exact": all(exact.values()),
    }


def materialize_correlated_crop_evaluation(
    *,
    domains: dict[str, SlopeDomain],
    enlarged_domains: dict[str, SlopeDomain],
    evaluations: tuple[CropEvaluation, ...],
    evidence_accounting: dict[str, Any],
    collars_m: dict[str, float],
    config: ProcessConfig,
    config_path: Path,
    condition_bundle_path: Path,
    condition_bundle_sha256: str,
    evaluation_plan_path: Path,
    evaluation_plan_sha256: str,
    output_parent: Path | None = None,
) -> Path:
    if {value.crop_id for value in evaluations} != {
        "development_a_west",
        "development_a_east",
    }:
        raise ValueError("correlated evaluation requires exactly A-west and A-east")
    if any(value.source_site_id != "development_a" for value in evaluations):
        raise ValueError("correlated crops must retain Development A source identity")
    solve_groups = {value.solve_group_id for value in evaluations}
    if solve_groups != {"development_a_whole_domain_10a5"}:
        raise ValueError("correlated crops must share one frozen solve group")

    domain = domains["development_a"]
    enlarged_domain = enlarged_domains["development_a"]
    implementation = _implementation_identity()
    recipe = {
        "schema_version": "laas.erodible-slope-correlated-crop-recipe/1",
        "condition_bundle": {
            "path": str(condition_bundle_path),
            "sha256": condition_bundle_sha256,
        },
        "evaluation_plan": {
            "path": str(evaluation_plan_path),
            "sha256": evaluation_plan_sha256,
        },
        "config": {
            "path": str(config_path),
            "sha256": _sha256_file(config_path),
            "value": config.__dict__,
        },
        "solve_group": {
            "solve_group_id": evaluations[0].solve_group_id,
            "source_site_id": "development_a",
            "source_identity": domain.source_identity,
            "enlarged_source_identity": enlarged_domain.source_identity,
            "baseline_solve_count": 1,
            "seep_off_control_solve_count": 1,
            "enlarged_control_solve_count": 1,
        },
        "crops": [
            {
                "crop_id": value.crop_id,
                "source_site_id": value.source_site_id,
                "solve_group_id": value.solve_group_id,
                "output_chunk": list(value.output_chunk),
                "bbox_en": list(value.bbox_en),
                "evidence_role": value.evidence_role,
            }
            for value in evaluations
        ],
        "evidence_accounting": evidence_accounting,
        "implementation": implementation,
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "byteorder": sys.byteorder,
            "numpy": np.__version__,
        },
        "controls": [
            "C0_cubic_1m_diagnostic_carrier_not_structural_authority",
            "C1_one_shared_whole_development_a_domain_morphology_process",
            "one_shared_seep_support_off_control",
            "one_shared_enlarged_domain_control",
        ],
        "prohibitions": [
            "no_runtime_synthesis",
            "no_noise_fbm_sines_or_stamps",
            "no_per_chunk_process_state",
            "no_duplicate_baseline_or_control_solve_per_crop",
            "no_development_a_east_alias_as_development_c",
            "no_independent_site_validation_claim",
            "no_production_or_preview_credit",
        ],
    }
    recipe_sha256 = hashlib.sha256(_canonical_bytes(recipe)).hexdigest()
    parent = (
        Path(output_parent)
        if output_parent is not None
        else DATA_WORK
        / "microtopography"
        / "erodible-slope"
        / "correlated-crop-evaluation"
        / "sha256"
    )
    root = parent / recipe_sha256
    temporary = parent / f".{recipe_sha256}.{os.getpid()}.tmp"
    if temporary.exists():
        raise RuntimeError(f"stale correlated evaluation transaction exists: {temporary}")
    temporary.mkdir(parents=True)

    rotation = rotation_error(config)
    process = solve_process(domain, config)
    plan = build_form_plan(domain, process, config)
    seep_off_domain = replace(
        domain,
        seep_likelihood=np.zeros_like(domain.seep_likelihood),
    )
    seep_off_process = solve_process(seep_off_domain, config)
    seep_off_plan = build_form_plan(seep_off_domain, seep_off_process, config)
    enlarged_process = solve_process(enlarged_domain, config)
    enlarged_plan = build_form_plan(enlarged_domain, enlarged_process, config)

    crop_metrics: dict[str, Any] = {}
    crop_edges: dict[str, dict[str, np.ndarray]] = {}
    full_metrics: list[dict[str, Any]] = []
    for evaluation in evaluations:
        surface = render_fine_surface(
            domain, process, plan, config, bbox_en=evaluation.bbox_en
        )
        partitioned = render_fine_surface(
            domain,
            process,
            plan,
            config,
            bbox_en=evaluation.bbox_en,
            row_block=137,
        )
        seep_off = render_fine_surface(
            seep_off_domain,
            seep_off_process,
            seep_off_plan,
            config,
            bbox_en=evaluation.bbox_en,
        )
        enlarged_surface = render_fine_surface(
            enlarged_domain,
            enlarged_process,
            enlarged_plan,
            config,
            bbox_en=evaluation.bbox_en,
        )
        measured = _site_metrics(
            domain,
            process,
            plan,
            surface,
            seep_off,
            enlarged_surface,
            partitioned,
            config,
            collars_m["development_a"],
        )
        full_metrics.append(measured)
        metrics = _correlated_crop_metrics(evaluation, measured)
        crop_metrics[evaluation.crop_id] = metrics
        crop_edges[evaluation.crop_id] = _surface_edge(
            surface, -1 if evaluation.crop_id == "development_a_west" else 0
        )
        crop_root = temporary / "crops" / evaluation.crop_id
        crop_root.mkdir(parents=True)
        _write_deterministic_npz(
            crop_root / "surface.npz",
            {
                "c0_height_m": surface.c0_height_m.astype("<f4"),
                "c1_height_m": surface.c1_height_m.astype("<f4"),
                "c1_minus_c0_m": surface.residual_m.astype("<f4"),
                "incision_m": surface.incision_m.astype("<f4"),
                "deposition_m": surface.deposition_m.astype("<f4"),
                "ownership": surface.ownership,
                "hard_exclusion": surface.hard_exclusion.astype(np.uint8),
                "material_supported": surface.material_supported.astype(np.uint8),
                "seep_off_residual_m": seep_off.residual_m.astype("<f4"),
            },
        )
        (crop_root / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        render_qa(
            crop_root / "qa",
            domain=domain,
            process=process,
            plan=plan,
            surface=surface,
            config=config,
            source_hashes={
                "condition_bundle": condition_bundle_sha256,
                "config": _sha256_file(config_path),
                "evaluation_plan": evaluation_plan_sha256,
            },
            recipe_sha256=recipe_sha256,
            diagnostics={
                "partition max abs (m)": metrics["invariance"]["partition_max_abs_m"],
                "domain enlargement max abs (m)": metrics["invariance"][
                    "domain_enlargement_max_abs_m"
                ],
                **rotation,
                "support radius (m)": measured["forms"]["maximum_support_radius_m"],
                "condition collar (m)": collars_m["development_a"],
            },
            crop_id=evaluation.crop_id,
            solve_group_id=evaluation.solve_group_id,
        )

    shared_metrics = _shared_solve_metrics(full_metrics[0])
    shared_metrics["solve_group_id"] = evaluations[0].solve_group_id
    shared_metrics["baseline_solve_count"] = 1
    shared_metrics["seep_off_control_solve_count"] = 1
    shared_metrics["enlarged_control_solve_count"] = 1
    edge = _shared_edge_metrics(
        crop_edges["development_a_west"],
        crop_edges["development_a_east"],
        boundary_e_m=evaluations[0].bbox_en[2],
    )
    crop_hard_pass = all(value["hard_gate_pass"] for value in crop_metrics.values())
    visible_pass = all(
        value["visible_morphology_gate_pass"] for value in crop_metrics.values()
    )
    rotation_pass = max(
        rotation["rotation_water_max_relative"],
        rotation["rotation_area_max_relative"],
    ) <= 1e-10
    state = (
        "r0_correlated_within_site_two_crop_visual_survivor_non_authorizing"
        if shared_metrics["hard_gate_pass"]
        and crop_hard_pass
        and visible_pass
        and rotation_pass
        and edge["all_arrays_exact"]
        else "r0_correlated_within_site_two_crop_visual_rejected"
    )
    manifest = {
        "schema_version": "laas.erodible-slope-correlated-crop-evaluation/1",
        "recipe_sha256": recipe_sha256,
        "recipe": recipe,
        "state": state,
        "evidence_accounting": evidence_accounting,
        "research_only": True,
        "production_owner": False,
        "preview_authorized": False,
        "recipe_freeze_authorized": False,
        "solve_groups": {evaluations[0].solve_group_id: shared_metrics},
        "crops": crop_metrics,
        "shared_edge": edge,
        "rotation_qualification": rotation,
        "limitations": [
            "A-west and A-east are adjacent views of one Development A condition, solve, and realization.",
            "The two crops receive one development-site credit and zero independent-validation, OOD, production, or preview credit.",
            "Development A-east is not Development C; rejected Development C history is unchanged.",
            "C1 remains a condition-bound R0 process hypothesis rather than measured morphology fit.",
            "The cubic 1 m C0 carrier remains diagnostic only and is not fine structural authority or a cookable master.",
        ],
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (temporary / "recipe.json").write_text(
        json.dumps(recipe, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (temporary / "shared-solve-metrics.json").write_text(
        json.dumps(shared_metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if root.exists():
        existing = sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file())
        rebuilt = sorted(
            path.relative_to(temporary) for path in temporary.rglob("*") if path.is_file()
        )
        if existing != rebuilt or any(
            _sha256_file(root / relative) != _sha256_file(temporary / relative)
            for relative in rebuilt
        ):
            raise RuntimeError(
                "existing correlated-crop evaluation differs from deterministic reconstruction"
            )
        shutil.rmtree(temporary)
        return root / "manifest.json"
    parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(root)
    return root / "manifest.json"
