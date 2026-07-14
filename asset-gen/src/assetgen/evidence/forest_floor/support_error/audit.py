"""Bind retained forest-floor evidence to the canonical B1/B2 qualification gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "forest-floor-support-error-qualification/1.0.0"
BANDS = {
    "B1": {"wavelength_m": [0.25, 1.0], "support_threshold_m": 0.05},
    "B2": {"wavelength_m": [0.125, 0.25], "support_threshold_m": 0.03},
}

INPUT_PATHS = {
    "canonical_spec": "docs/specs/terrain/MICROTOPOGRAPHY.md",
    "qualification_protocol": "docs/specs/terrain/PUBLIC-TARGET-QUALIFICATION-PROTOCOL.md",
    "hovi_condition": (
        "asset-gen/data/work/microtopography/hovi/conditions/sha256/"
        "22841590061fb60238ee370220c58a635d669a6e3e620549283e788d986fdc8e/"
        "HY_SPRUCE4-condition-evidence.json"
    ),
    "hovi_photo_semantics": (
        "asset-gen/data/work/microtopography/hovi/photo-semantics/"
        "043cecae6d96784da4062d61c5378727d60e7a4bf2284b075482e4d9e1d2432e/"
        "qa/index.json"
    ),
    "hovi_thinned_surface": (
        "asset-gen/data/work/microtopography/hovi/surface/"
        "1e5f674af3fc4d09fdf619ef154f425df22d148a0c619cdc2f93253a32e6440e/"
        "manifest.json"
    ),
    "hovi_full_scan_qa": (
        "asset-gen/data/work/hovi-full-scan-candidate-evidence/"
        "261e2e9999aecf270c262b64118775bcf21bcae5f7d462589685e6063fcf48f4/"
        "qa/index.json"
    ),
    "evo_1086_surface": (
        "asset-gen/data/work/microtopography/evo/surface/sha256/"
        "bba7a00ad160cbbdb1193fb50d89f629283eb48a31f005f0fa0e7168263fdb3d/"
        "manifest.json"
    ),
    "evo_1065_surface": (
        "asset-gen/data/work/microtopography/evo/surface/sha256/"
        "674116851f1f22acc5143ed0cec0ed0eac7c8d764458cfd562f4a6b5d5835f3c/"
        "manifest.json"
    ),
}

IMPLEMENTATION_PATHS = (
    "asset-gen/src/assetgen/evidence/forest_floor/support_error/audit.py",
    "asset-gen/src/assetgen/evidence/forest_floor/support_error/artifacts.py",
    "asset-gen/src/assetgen/evidence/forest_floor/support_error/render.py",
    "asset-gen/src/assetgen/evidence/forest_floor/support_error/__main__.py",
)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("rb") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _input_reference(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Required retained evidence is missing: {relative_path}")
    return {
        "path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _verified_relative_json(
    root: Path,
    parent_relative_path: str,
    reference: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(set(reference) >= {"path", "bytes", "sha256"}, "Artifact reference is incomplete")
    relative = str(Path(parent_relative_path).parent / str(reference["path"]))
    bound = _input_reference(root, relative)
    _require(bound["bytes"] == reference["bytes"], f"Referenced byte count changed: {relative}")
    _require(bound["sha256"] == reference["sha256"], f"Referenced digest changed: {relative}")
    return _load_json(root / relative), bound


def build_support_error_decision(repo_root: Path) -> dict[str, Any]:
    """Return the smallest honest decision supported by retained forest-floor evidence."""
    root = repo_root.resolve()
    inputs = {name: _input_reference(root, path) for name, path in INPUT_PATHS.items()}

    condition = _load_json(root / INPUT_PATHS["hovi_condition"])
    photos = _load_json(root / INPUT_PATHS["hovi_photo_semantics"])
    thinned_manifest = _load_json(root / INPUT_PATHS["hovi_thinned_surface"])
    full_qa = _load_json(root / INPUT_PATHS["hovi_full_scan_qa"])
    evo_1086 = _load_json(root / INPUT_PATHS["evo_1086_surface"])
    evo_1065 = _load_json(root / INPUT_PATHS["evo_1065_surface"])

    _require(
        photos.get("condition_evidence_sha256") == inputs["hovi_condition"]["sha256"],
        "Hovi photo evidence no longer binds the retained condition record",
    )
    _require(
        condition.get("forest_floor_semantics", {}).get("geometric_semantics") is False,
        "Hovi condition evidence unexpectedly claims geometric semantics",
    )
    _require(
        photos.get("claims", {}).get("metric_registration") is False
        and photos.get("claims", {}).get("pixel_classification") is False,
        "Hovi photo evidence boundary changed",
    )
    _require(
        thinned_manifest.get("qualification_status") == "unqualified"
        and thinned_manifest.get("usable_surface") is False,
        "Hovi thinned candidate is no longer the expected unqualified artifact",
    )
    thinned_evidence, thinned_evidence_ref = _verified_relative_json(
        root,
        INPUT_PATHS["hovi_thinned_surface"],
        thinned_manifest["candidate_evidence"],
    )
    _require(
        thinned_evidence.get("scientific_disposition", {}).get("usable_surface") is False,
        "Hovi thinned candidate evidence unexpectedly authorizes a surface",
    )
    _require(
        full_qa.get("qualificationStatus") == "raw_candidate_unqualified"
        and full_qa.get("surfaceClaim") is False
        and full_qa.get("targetTruth") is False,
        "Hovi full-scan QA boundary changed",
    )
    hovi_summary = full_qa["summary"]
    _require(
        hovi_summary["cells"]
        == hovi_summary["soleRawCandidateCells"]
        + hovi_summary["ambiguousCandidateCells"]
        + hovi_summary["overflowOrUnresolvedCells"],
        "Hovi full-scan disposition does not conserve cells",
    )
    selected_shards = full_qa["manifest"]["selectedPointShards"]
    scan_ids = sorted({int(value["scan"]) for value in selected_shards})
    _require(scan_ids == list(range(16)), "Hovi full-scan QA no longer binds all 16 scans")

    evo_records: dict[str, Any] = {}
    for name, manifest in (("1086", evo_1086), ("1065", evo_1065)):
        _require(
            manifest.get("qualification", {}).get("status")
            == "surface_semantics_and_error_unresolved",
            f"Evo {name} qualification boundary changed",
        )
        inventory, inventory_ref = _verified_relative_json(
            root,
            INPUT_PATHS[f"evo_{name}_surface"],
            manifest["artifacts"]["inventory"],
        )
        evo_records[name] = {
            "manifest_sha256": inputs[f"evo_{name}_surface"]["sha256"],
            "inventory": inventory_ref,
            "direct_observed_fraction": inventory["grid"]["direct_observed_fraction"],
            "geometrically_valid_fraction": inventory["grid"]["geometrically_valid_fraction"],
            "nearest_support_p95_m": inventory["grid"]["nearest_support_p95_m"],
            "plane_residual_rms_p95_m": inventory["grid"]["plane_residual_rms_p95_m"],
            "source_group_role": "redundancy_group_not_mapped_to_physical_view",
            "qualification_status": manifest["qualification"]["status"],
        }

    semantics = condition["forest_floor_semantics"]
    band_results = {
        band: {
            **definition,
            "status": "target_evidence_insufficient",
            "effective_support_ucb95_m": None,
            "support_gate": "not_evaluable_without_accepted_semantic_surface",
            "recoverable_signal_rms_m": None,
            "recoverable_signal_status": "unidentifiable",
            "measurement_error_rms_m": None,
            "error_signal_ratio_ucb95": None,
            "transfer_gain_lcb95": None,
            "decision_reasons": [
                "surface_contract_and_metric_semantic_labels_missing",
                "held_out_scan_reconstructions_of_same_surface_missing",
                "independent_horizontal_and_vertical_control_missing",
                "acquisition_response_and_converter_transfer_unbounded",
            ],
        }
        for band, definition in BANDS.items()
    }

    decision: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_fail_closed",
        "decision": "target_evidence_insufficient",
        "scope": {
            "regime": "foreign_hemiboreal_mature_spruce_forest_floor_feasibility",
            "site": "Hovi_HY_SPRUCE4",
            "claim_role": "support_error_feasibility_only",
            "surface_contract": "not_frozen",
            "target_surface_claim": False,
            "qualified_exemplar": False,
            "qualified_target_B1": False,
            "qualified_target_B2": False,
            "estonia_transfer": False,
            "synthesis_authorized": False,
        },
        "canonical_gate": {
            "bands": BANDS,
            "minimum_independent_view_groups_per_context": 2,
            "context_size_m": 8.0,
            "initial_contexts_per_site": 6,
            "maximum_error_signal_ratio": 0.5,
            "minimum_transfer_gain": 0.707,
            "unknown_policy": "abstain",
        },
        "input_artifacts": inputs,
        "implementation_artifacts": {
            path: _input_reference(root, path) for path in IMPLEMENTATION_PATHS
        },
        "derived_input_artifacts": {
            "hovi_thinned_candidate_evidence": thinned_evidence_ref,
        },
        "observed_evidence": {
            "hovi_condition": {
                "geometric_semantics": semantics["geometric_semantics"],
                "semantic_method_scope": semantics["method_scope"],
                "semantic_interpretation_limit": semantics["interpretation_limit"],
                "mean_cover_fraction": {
                    name: value["mean_fraction"]
                    for name, value in semantics["components"].items()
                },
            },
            "hovi_full_scan_central_shard": {
                "physical_scan_ids": scan_ids,
                "cells": hovi_summary["cells"],
                "sole_raw_candidate_cells": hovi_summary["soleRawCandidateCells"],
                "ambiguous_candidate_cells": hovi_summary["ambiguousCandidateCells"],
                "overflow_or_unresolved_cells": hovi_summary["overflowOrUnresolvedCells"],
                "sole_raw_candidate_fraction": (
                    hovi_summary["soleRawCandidateCells"] / hovi_summary["cells"]
                ),
                "surface_semantics_unknown_cells": hovi_summary[
                    "unknownReasonCellMentions"
                ]["surface_semantics_unknown"],
                "interpolation_performed": full_qa["interpolationPerformed"],
                "scientific_role": full_qa["scientificRole"],
            },
            "hovi_thinned_preview": {
                "merged_scan_identity": "absent",
                "direct_fraction": thinned_evidence["quality_metrics_before_abstention"][
                    "direct_fraction"
                ],
                "registration_mean_absolute_error_m": thinned_evidence["registration"][
                    "enabled_constraint_mean_absolute_error_m"
                ],
                "registration_role": thinned_evidence["registration"]["interpretation"],
                "usable_surface": False,
            },
            "evo_sparse_candidates": evo_records,
        },
        "signal_error_separation": {
            "observed_raw_geometry_present": True,
            "physical_view_identity_present": "Hovi_full_scans_only",
            "metric_semantic_surface_present": False,
            "held_out_same_surface_repeatability_present": False,
            "independent_checkpoint_xy_error_present": False,
            "independent_checkpoint_z_error_present": False,
            "independent_registration_error_present": False,
            "semantic_choice_error_present": False,
            "interpolation_error_present": False,
            "interpretation": (
                "Raw height variation cannot be assigned to terrain signal versus registration, "
                "measurement, occlusion, or semantic-sheet choice. Inter-scan disagreement and "
                "local plane residuals remain descriptive observables, not independent error."
            ),
        },
        "band_results": band_results,
        "forbidden_shortcuts": [
            "do_not_treat_sole_raw_hypothesis_as_ground",
            "do_not_treat_inter_scan_mad_as_independent_error",
            "do_not_treat_project_registration_residual_as_per_cell_total_error",
            "do_not_treat_evo_source_ids_as_physical_views",
            "do_not_interpolate_unknown_cells_into_support",
            "do_not_infer_physical_smoothness_from_unidentifiable_signal",
        ],
        "objective_resume_conditions": [
            "Freeze one physical surface contract for mineral soil, stable moss/litter, roots, clasts, deadwood, and living vegetation.",
            "Register cell-level semantic labels to raw scans and independently double-audit at least 20 percent; ambiguous cells invalidate.",
            "Freeze disjoint physical scan groups and reconstruct the same accepted surface independently in at least six non-overlapping 8 m contexts.",
            "Provide independent horizontal and vertical checkpoints or a calibrated independent measurement not used by registration/conversion.",
            "Compute per-cell r_eff from accepted-surface observations and bound context p95 against 0.05 m for B1 and 0.03 m for B2.",
            "Forward-sample held-out acquisition geometry, reconvert, and apply canonical F4/R4 bands to estimate transfer and signal/error intervals.",
        ],
    }
    return decision
