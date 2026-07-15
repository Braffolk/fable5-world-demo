"""Separated construction, sealed evaluation, and human decision workflow."""

from __future__ import annotations

import hashlib
import json
import platform
import tempfile
from datetime import datetime
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import numpy as np
import scipy

from assetgen.process.microtopo.model import GroundSurface

from .contracts import (
    ScreenConfig,
    canonical_json,
    sha256_file,
    verify_static_closure,
)
from .frozen_anchor import EXPECTED_SEMANTIC_SHA256
from .operators import SurfaceArrays, build_atoms, ideal_area_degrade, reconstruct_sparse_one
from .qa import assignment_trace, gate_failures, numeric_diagnostics, write_visual_panel


def _load_surface(contract: Any) -> SurfaceArrays:
    surface = GroundSurface.load(contract.path)
    if surface.source_id != contract.source_id or surface.qa_status != "approved":
        raise ValueError(f"surface identity/approval mismatch for {contract.source_id}")
    if surface.elevation_m.shape != contract.shape:
        raise ValueError(f"surface shape mismatch for {contract.source_id}")
    if surface.texel_m != contract.texel_m:
        raise ValueError(f"surface texel mismatch for {contract.source_id}")
    if (surface.origin_x_m, surface.origin_y_m) != contract.origin_m:
        raise ValueError(f"surface origin mismatch for {contract.source_id}")
    if surface.role not in {"calibration", "holdout"}:
        raise ValueError(f"unexpected extraction role for {contract.source_id}")
    return SurfaceArrays(
        surface.source_id, surface.elevation_m, surface.measured, surface.confidence
    )


def _low(surface: SurfaceArrays, algorithm: dict[str, Any]) -> Any:
    return ideal_area_degrade(
        surface,
        factor=int(algorithm["factor"]),
        min_direct_fraction=float(algorithm["degradation"]["min_direct_fraction"]),
        min_support_fraction=float(algorithm["degradation"]["min_support_fraction"]),
        min_mean_confidence=float(algorithm["degradation"]["min_mean_confidence"]),
    )


def _atoms(surface: SurfaceArrays, low: Any, algorithm: dict[str, Any]) -> Any:
    return build_atoms(
        surface,
        low,
        factor=int(algorithm["factor"]),
        patch_cells=int(algorithm["low_patch_cells"]),
        stride_cells=int(algorithm["low_stride_cells"]),
        min_low_support_fraction=float(
            algorithm["dictionary"]["min_low_support_fraction"]
        ),
        min_high_direct_fraction=float(
            algorithm["dictionary"]["min_high_direct_fraction"]
        ),
        min_atom_norm_m=float(algorithm["dictionary"]["min_atom_norm_m"]),
    )


def _evaluate(
    target: SurfaceArrays,
    atoms: list[Any],
    algorithm: dict[str, Any],
    gates: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    low = _low(target, algorithm)
    target_support = (
        np.isfinite(target.elevation_m)
        & np.isfinite(target.confidence)
        & (target.confidence > 0)
    )
    result = reconstruct_sparse_one(
        low,
        atoms,
        factor=int(algorithm["factor"]),
        patch_cells=int(algorithm["low_patch_cells"]),
        stride_cells=int(algorithm["low_stride_cells"]),
        min_common_support_fraction=float(
            algorithm["matching"]["min_common_support_fraction"]
        ),
        target_high_support=target_support,
        target_high_confidence=target.confidence,
    )
    metrics = numeric_diagnostics(
        target,
        result,
        stride_hr=int(algorithm["low_stride_cells"]) * int(algorithm["factor"]),
    )
    metrics["numeric_failures"] = gate_failures(metrics, gates)
    return result, metrics


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }


def _json_object_from_bytes(data: bytes, source: str) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {source}")
            result[key] = value
        return result

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid UTF-8 JSON in {source}") from exc
    value = json.loads(text, object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {source}")
    return value


def _json_without_duplicate_keys(path: Path) -> dict[str, Any]:
    return _json_object_from_bytes(path.read_bytes(), str(path))


def _seal_content_artifact(
    config: ScreenConfig,
    *,
    stage: str,
    staging: Path,
) -> Path:
    records: list[dict[str, str]] = []
    for path in sorted(staging.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"artifact may not contain symlink: {path}")
        if path.is_file():
            records.append(
                {
                    "path": path.relative_to(staging).as_posix(),
                    "sha256": sha256_file(path),
                }
            )
    content = {
        "schema": "forest-sparse-content-manifest/1",
        "stage": stage,
        "config_sha256": config.config_sha256,
        "semantic_sha256": EXPECTED_SEMANTIC_SHA256,
        "files": records,
    }
    content_path = staging / "content-manifest.json"
    content_path.write_bytes(canonical_json(content))
    identity = sha256_file(content_path)
    destination_parent = (
        config.root / str(config.raw["output"]["root"]) / stage / "sha256"
    )
    destination_parent.mkdir(parents=True, exist_ok=True)
    destination = destination_parent / identity
    if destination.exists():
        raise FileExistsError(f"immutable result already exists: {destination}")
    staging.replace(destination)
    manifest_path = destination / "manifest.json"
    _validate_content_artifact(config, manifest_path, stage)
    return manifest_path


def _write_review_artifact(
    config: ScreenConfig,
    *,
    stage: str,
    payload: dict[str, Any],
    rendered: list[tuple[str, SurfaceArrays, Any]],
    extra_files: dict[str, bytes] | None = None,
) -> Path:
    stage_root = (
        config.root / str(config.raw["output"]["root"]) / stage
    )
    stage_root.mkdir(parents=True, exist_ok=True)
    output_root = Path(tempfile.mkdtemp(prefix=".staging-", dir=stage_root))
    qa_root = output_root / "qa"
    qa_root.mkdir(parents=True)
    files: list[dict[str, str]] = []
    for index, (source_id, truth, reconstruction) in enumerate(rendered, start=1):
        image_path = qa_root / f"{index:02d}_{source_id}_height_relief_and_assignments.png"
        write_visual_panel(
            image_path,
            truth,
            reconstruction,
            f"{source_id}: frozen Guerin sparse {stage}",
        )
        trace_path = qa_root / f"{index:02d}_{source_id}_assignment_trace.json"
        trace_path.write_bytes(
            canonical_json(
                {
                    "schema": "forest-sparse-assignment-trace/1",
                    "target_source_id": source_id,
                    "low_cell_m": config.raw["algorithm"]["coarse_texel_m"],
                    "assignments": assignment_trace(reconstruction),
                }
            )
        )
        files.extend(
            {
                "path": path.relative_to(output_root).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in (image_path, trace_path)
        )
    manifest_path = output_root / "manifest.json"
    manifest_path.write_bytes(canonical_json(payload))
    qa_index = {
        "schema": "forest-sparse-review-index/1",
        "stage": stage,
        "manifest": {
            "path": "manifest.json",
            "sha256": sha256_file(manifest_path),
        },
        "review_files": sorted(files, key=lambda item: item["path"]),
        "visual_gates": config.raw["rejection_gates"]["visual"],
    }
    qa_index_path = output_root / "qa-index.json"
    qa_index_path.write_bytes(canonical_json(qa_index))
    template = {
        "schema": "forest-sparse-visual-adjudication/1",
        "stage": stage,
        "config_sha256": config.config_sha256,
        "semantic_sha256": EXPECTED_SEMANTIC_SHA256,
        "artifact_content_manifest_sha256": None,
        "reviewer_identity": None,
        "authoring_method": None,
        "adjudicated_at": None,
        "decision": None,
        "reviewed_files": qa_index["review_files"],
        "gate_decisions": [
            {"id": gate["id"], "passed": None, "notes": ""}
            for gate in config.raw["rejection_gates"]["visual"]
        ],
    }
    (output_root / "visual-adjudication.template.json").write_bytes(
        canonical_json(template)
    )
    for relative, content in sorted((extra_files or {}).items()):
        posix = PurePosixPath(relative)
        if posix.is_absolute() or ".." in posix.parts or not posix.parts:
            raise ValueError(f"invalid extra artifact path: {relative}")
        destination = output_root.joinpath(*posix.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ValueError(f"extra artifact path collides: {relative}")
        destination.write_bytes(content)
    return _seal_content_artifact(
        config, stage=stage, staging=output_root
    )


def run_construction(config: ScreenConfig) -> Path:
    """Run construction folds only. This function has no k19 load path."""
    closure = verify_static_closure(config)
    algorithm = config.raw["algorithm"]
    gates = config.raw["rejection_gates"]
    construction = [_load_surface(item) for item in config.construction]
    lows = {surface.source_id: _low(surface, algorithm) for surface in construction}
    atom_groups = {
        surface.source_id: _atoms(surface, lows[surface.source_id], algorithm)
        for surface in construction
    }
    fold_results: dict[str, Any] = {}
    rendered: list[tuple[str, SurfaceArrays, Any]] = []
    for target in construction:
        dictionary = [
            atom
            for source_id, atoms in atom_groups.items()
            if source_id != target.source_id
            for atom in atoms
        ]
        reconstruction, metrics = _evaluate(target, dictionary, algorithm, gates)
        fold_results[target.source_id] = metrics
        rendered.append((target.source_id, target, reconstruction))
    numeric_passed = all(
        not result["numeric_failures"] for result in fold_results.values()
    )
    payload = {
        "schema": "forest-sparse-construction-result/1",
        "config_sha256": config.config_sha256,
        "semantic_sha256": EXPECTED_SEMANTIC_SHA256,
        "static_closure": closure,
        "environment": _environment(),
        "dictionary_atom_counts": {
            key: len(value) for key, value in atom_groups.items()
        },
        "construction_folds": fold_results,
        "numeric_passed": numeric_passed,
        "visual_gate_status": "pending_human_authored_outside_generator_receipt"
        if numeric_passed else "not_reached_numeric_failure",
        "evaluation_authorized": False,
        "authorized_outcome": "none",
        "failure_action": None if numeric_passed else "park_without_tuning",
    }
    return _write_review_artifact(
        config, stage="construction", payload=payload, rendered=rendered
    )


def _validate_content_artifact(
    config: ScreenConfig,
    manifest_path: Path,
    stage: str,
    *,
    captured_files: dict[str, bytes] | None = None,
) -> dict[str, Any]:
    captured = captured_files or {}
    captured_seen: set[str] = set()
    resolved = manifest_path.resolve()
    expected_parent = (
        config.root / str(config.raw["output"]["root"]) / stage / "sha256"
    ).resolve()
    if resolved.name != "manifest.json" or resolved.parent.parent != expected_parent:
        raise ValueError(f"{stage} manifest is outside the frozen immutable artifact root")
    content_path = resolved.with_name("content-manifest.json")
    if sha256_file(content_path) != resolved.parent.name:
        raise ValueError(f"{stage} directory identity does not match content manifest")
    content = _json_without_duplicate_keys(content_path)
    if set(content) != {"schema", "stage", "config_sha256", "semantic_sha256", "files"}:
        raise ValueError(f"{stage} content manifest schema is not exact")
    expected_header = {
        "schema": "forest-sparse-content-manifest/1",
        "stage": stage,
        "config_sha256": config.config_sha256,
        "semantic_sha256": EXPECTED_SEMANTIC_SHA256,
    }
    for key, value in expected_header.items():
        if content.get(key) != value:
            raise ValueError(f"{stage} content manifest {key} mismatch")
    expected_files = {"content-manifest.json"}
    seen: set[str] = set()
    for record in content.get("files", []):
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(f"{stage} content file record is not exact")
        relative = str(record["path"])
        posix = PurePosixPath(relative)
        if posix.is_absolute() or ".." in posix.parts or not posix.parts:
            raise ValueError(f"unsafe content path: {relative}")
        if relative in seen or relative == "content-manifest.json":
            raise ValueError(f"duplicate/reserved content path: {relative}")
        seen.add(relative)
        candidate = resolved.parent.joinpath(*posix.parts)
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"missing/non-regular content file: {relative}")
        if relative in captured:
            actual_sha256 = hashlib.sha256(captured[relative]).hexdigest()
            captured_seen.add(relative)
        else:
            actual_sha256 = sha256_file(candidate)
        if actual_sha256 != record["sha256"]:
            raise ValueError(f"content file hash mismatch: {relative}")
        expected_files.add(relative)
    if captured_seen != set(captured):
        raise ValueError(f"captured content paths are not exact: {sorted(set(captured)-captured_seen)}")
    actual_files: set[str] = set()
    for candidate in resolved.parent.rglob("*"):
        if candidate.is_symlink():
            raise ValueError(f"artifact contains a symlink: {candidate}")
        if candidate.is_file():
            actual_files.add(candidate.relative_to(resolved.parent).as_posix())
    if actual_files != expected_files:
        raise ValueError(
            f"{stage} artifact file-set mismatch: missing={sorted(expected_files-actual_files)}, "
            f"added={sorted(actual_files-expected_files)}"
        )
    core = {"manifest.json", "content-manifest.json"}
    if stage == "construction":
        expected_stage_files = core | {
            "qa-index.json",
            "visual-adjudication.template.json",
            *(
                f"qa/{index:02d}_{source_id}_height_relief_and_assignments.png"
                for index, source_id in enumerate(("k11", "k32", "k36"), start=1)
            ),
            *(
                f"qa/{index:02d}_{source_id}_assignment_trace.json"
                for index, source_id in enumerate(("k11", "k32", "k36"), start=1)
            ),
        }
    elif stage == "evaluation":
        expected_stage_files = core | {
            "qa-index.json",
            "visual-adjudication.template.json",
            "qa/01_k19_height_relief_and_assignments.png",
            "qa/01_k19_assignment_trace.json",
            "inputs/construction-visual-receipt.json",
        }
    elif stage == "decision":
        expected_stage_files = core | {
            "inputs/construction-visual-receipt.json",
            "inputs/evaluation-visual-receipt.json",
        }
    else:
        raise ValueError(f"unsupported artifact stage: {stage}")
    if actual_files != expected_stage_files:
        raise ValueError(
            f"{stage} stage schema mismatch: missing={sorted(expected_stage_files-actual_files)}, "
            f"added={sorted(actual_files-expected_stage_files)}"
        )
    return content


def _load_review_index(
    manifest_path: Path, content: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    manifest = _json_without_duplicate_keys(manifest_path)
    index_path = manifest_path.with_name("qa-index.json")
    index = _json_without_duplicate_keys(index_path)
    if set(index) != {"schema", "stage", "manifest", "review_files", "visual_gates"}:
        raise ValueError("review index schema is not exact")
    if index.get("schema") != "forest-sparse-review-index/1":
        raise ValueError("unsupported review index schema")
    if not isinstance(index.get("manifest"), dict) or set(index["manifest"]) != {
        "path", "sha256"
    }:
        raise ValueError("review index manifest record is not exact")
    if index["manifest"]["path"] != "manifest.json":
        raise ValueError("review index manifest path changed")
    if index["manifest"]["sha256"] != sha256_file(manifest_path):
        raise ValueError("review index does not bind supplied manifest")
    content_qa = {
        record["path"]: record["sha256"]
        for record in content["files"]
        if str(record["path"]).startswith("qa/")
    }
    indexed_qa: dict[str, str] = {}
    for record in index["review_files"]:
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError("review file record is not exact")
        posix = PurePosixPath(str(record["path"]))
        if posix.is_absolute() or ".." in posix.parts or not posix.parts:
            raise ValueError(f"unsafe review path: {record['path']}")
        candidate = manifest_path.parent.joinpath(*posix.parts)
        if sha256_file(candidate) != record["sha256"]:
            raise ValueError(f"review file hash mismatch: {candidate}")
        if record["path"] in indexed_qa:
            raise ValueError(f"duplicate indexed review file: {record['path']}")
        indexed_qa[record["path"]] = record["sha256"]
    if indexed_qa != content_qa:
        raise ValueError("QA index has a review-file addition or omission")
    return manifest, index, index_path


def _validate_visual_receipt_object(
    config: ScreenConfig,
    *,
    stage: str,
    approval: dict[str, Any],
    artifact_content_manifest_sha256: str,
    expected_reviewed_files: list[dict[str, str]] | None,
    require_pass: bool,
) -> None:
    receipt_contract = config.raw["human_gate_workflow"]["receipt_contract"]
    if set(approval) != set(receipt_contract["required_receipt_keys"]):
        raise ValueError("visual receipt top-level schema is not exact")
    if approval.get("schema") != "forest-sparse-visual-adjudication/1":
        raise ValueError("unsupported visual approval schema")
    expected = {
        "stage": stage,
        "config_sha256": config.config_sha256,
        "semantic_sha256": EXPECTED_SEMANTIC_SHA256,
        "artifact_content_manifest_sha256": artifact_content_manifest_sha256,
    }
    for key, value in expected.items():
        if approval.get(key) != value:
            raise ValueError(f"visual approval {key} mismatch")
    if approval.get("reviewer_identity") != receipt_contract["required_reviewer_identity"]:
        raise ValueError("visual receipt reviewer identity is not the configured identity")
    method = approval.get("authoring_method")
    if method not in receipt_contract["authoring_method_enum"]:
        raise ValueError("visual receipt authoring method is outside the configured enum")
    if method != receipt_contract["required_authoring_method"]:
        raise ValueError("visual receipt authoring method is not the required method")
    timestamp = approval.get("adjudicated_at")
    if not isinstance(timestamp, str):
        raise ValueError("visual receipt requires an ISO-8601 timestamp")
    try:
        parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("visual receipt timestamp is not ISO-8601") from exc
    if parsed_timestamp.tzinfo is None:
        raise ValueError("visual receipt timestamp must include a timezone")
    reviewed_files = approval.get("reviewed_files")
    if not isinstance(reviewed_files, list):
        raise ValueError("visual receipt reviewed_files must be a list")
    if expected_reviewed_files is not None and reviewed_files != expected_reviewed_files:
        raise ValueError("visual approval must bind every indexed review file")
    for record in reviewed_files:
        if not isinstance(record, dict) or set(record) != set(
            receipt_contract["required_review_file_keys"]
        ):
            raise ValueError("visual receipt review-file schema is not exact")
    expected_ids = [gate["id"] for gate in config.raw["rejection_gates"]["visual"]]
    decisions = approval.get("gate_decisions", [])
    if [item.get("id") for item in decisions] != expected_ids:
        raise ValueError("visual approval gate order/identity mismatch")
    for decision in decisions:
        if set(decision) != set(receipt_contract["required_gate_decision_keys"]):
            raise ValueError(f"gate {decision.get('id')} schema is not exact")
        if not isinstance(decision.get("passed"), bool):
            raise ValueError(f"gate {decision.get('id')} has no boolean decision")
        if not str(decision.get("notes", "")).strip():
            raise ValueError(f"gate {decision.get('id')} requires review notes")
    passed = all(bool(item["passed"]) for item in decisions)
    if approval.get("decision") != ("pass" if passed else "fail"):
        raise ValueError("visual approval aggregate decision is inconsistent")
    if require_pass and not passed:
        raise ValueError("construction visual adjudication failed; k19 remains sealed")


def _validate_visual_approval(
    config: ScreenConfig,
    *,
    stage: str,
    manifest_path: Path,
    approval_path: Path,
    require_pass: bool,
) -> tuple[dict[str, Any], str, bytes]:
    content = _validate_content_artifact(config, manifest_path, stage)
    _, index, _ = _load_review_index(manifest_path, content)
    if index.get("stage") != stage:
        raise ValueError("review index stage mismatch")
    if index.get("visual_gates") != config.raw["rejection_gates"]["visual"]:
        raise ValueError("review index visual gates differ from frozen config")
    approval_bytes = approval_path.read_bytes()
    approval = _json_object_from_bytes(approval_bytes, str(approval_path))
    _validate_visual_receipt_object(
        config,
        stage=stage,
        approval=approval,
        artifact_content_manifest_sha256=sha256_file(
            manifest_path.with_name("content-manifest.json")
        ),
        expected_reviewed_files=index["review_files"],
        require_pass=require_pass,
    )
    return approval, hashlib.sha256(approval_bytes).hexdigest(), approval_bytes


def run_evaluation(
    config: ScreenConfig,
    *,
    construction_manifest: Path,
    visual_approval: Path,
) -> Path:
    """Open k19 only after a human-authored outside-generator receipt validates."""
    verify_static_closure(config)
    construction_content = _validate_content_artifact(
        config, construction_manifest, "construction"
    )
    construction_payload, _, _ = _load_review_index(
        construction_manifest, construction_content
    )
    if construction_payload.get("schema") != "forest-sparse-construction-result/1":
        raise ValueError("not a construction result")
    if construction_payload.get("config_sha256") != config.config_sha256:
        raise ValueError("construction result config mismatch")
    if construction_payload.get("numeric_passed") is not True:
        raise ValueError("construction numerics failed; k19 remains sealed")
    approval, approval_sha256, approval_bytes = _validate_visual_approval(
        config,
        stage="construction",
        manifest_path=construction_manifest,
        approval_path=visual_approval,
        require_pass=True,
    )

    algorithm = config.raw["algorithm"]
    gates = config.raw["rejection_gates"]
    construction = [_load_surface(item) for item in config.construction]
    final_dictionary = [
        atom
        for surface in construction
        for atom in _atoms(surface, _low(surface, algorithm), algorithm)
    ]
    # No function above this line opens the evaluation-only surface.
    evaluation = _load_surface(config.evaluation)
    reconstruction, metrics = _evaluate(
        evaluation, final_dictionary, algorithm, gates
    )
    numeric_passed = not metrics["numeric_failures"]
    payload = {
        "schema": "forest-sparse-evaluation-result/1",
        "config_sha256": config.config_sha256,
        "semantic_sha256": EXPECTED_SEMANTIC_SHA256,
        "construction_content_manifest_sha256": sha256_file(
            construction_manifest.with_name("content-manifest.json")
        ),
        "construction_manifest_sha256": sha256_file(construction_manifest),
        "construction_visual_approval_sha256": approval_sha256,
        "construction_visual_reviewer_identity": approval["reviewer_identity"],
        "construction_visual_authoring_method": approval["authoring_method"],
        "environment": _environment(),
        "evaluation_only_k19": metrics,
        "numeric_passed": numeric_passed,
        "visual_gate_status": "pending_human_authored_outside_generator_receipt"
        if numeric_passed else "not_reached_numeric_failure",
        "authorized_outcome": "none",
        "failure_action": None if numeric_passed else "park_without_tuning",
    }
    return _write_review_artifact(
        config,
        stage="evaluation",
        payload=payload,
        rendered=[("k19", evaluation, reconstruction)],
        extra_files={
            "inputs/construction-visual-receipt.json": approval_bytes,
        },
    )


def finalize_evaluation(
    config: ScreenConfig,
    *,
    evaluation_manifest: Path,
    visual_approval: Path,
) -> Path:
    """Persist the human-gated terminal decision without opening any surface."""
    verify_static_closure(config)
    construction_receipt = (
        evaluation_manifest.parent / "inputs" / "construction-visual-receipt.json"
    )
    construction_receipt_bytes = construction_receipt.read_bytes()
    evaluation_content = _validate_content_artifact(
        config,
        evaluation_manifest,
        "evaluation",
        captured_files={
            "inputs/construction-visual-receipt.json": construction_receipt_bytes
        },
    )
    evaluation_payload, _, _ = _load_review_index(
        evaluation_manifest, evaluation_content
    )
    if evaluation_payload.get("schema") != "forest-sparse-evaluation-result/1":
        raise ValueError("not an evaluation result")
    if evaluation_payload.get("config_sha256") != config.config_sha256:
        raise ValueError("evaluation result config mismatch")
    if evaluation_payload.get("numeric_passed") is not True:
        raise ValueError("evaluation numerics failed; terminal outcome is already park")
    approval, approval_sha256, approval_bytes = _validate_visual_approval(
        config,
        stage="evaluation",
        manifest_path=evaluation_manifest,
        approval_path=visual_approval,
        require_pass=False,
    )
    passed = approval["decision"] == "pass"
    payload = {
        "schema": "forest-sparse-terminal-decision/1",
        "config_sha256": config.config_sha256,
        "semantic_sha256": EXPECTED_SEMANTIC_SHA256,
        "evaluation_content_manifest_sha256": sha256_file(
            evaluation_manifest.with_name("content-manifest.json")
        ),
        "evaluation_manifest_sha256": sha256_file(evaluation_manifest),
        "evaluation_visual_approval_sha256": approval_sha256,
        "evaluation_visual_reviewer_identity": approval["reviewer_identity"],
        "evaluation_visual_authoring_method": approval["authoring_method"],
        "visual_gate_status": "passed" if passed else "failed",
        "authorized_outcome": "algorithm_capacity_only" if passed else "none",
        "failure_action": None if passed else "park_without_tuning",
    }
    stage_root = config.root / str(config.raw["output"]["root"]) / "decision"
    stage_root.mkdir(parents=True, exist_ok=True)
    output_root = Path(tempfile.mkdtemp(prefix=".staging-", dir=stage_root))
    output = output_root / "manifest.json"
    output.write_bytes(canonical_json(payload))
    inputs = output_root / "inputs"
    inputs.mkdir()
    construction_receipt_sha256 = hashlib.sha256(construction_receipt_bytes).hexdigest()
    if construction_receipt_sha256 != evaluation_payload.get(
        "construction_visual_approval_sha256"
    ):
        raise ValueError("evaluation artifact construction receipt binding mismatch")
    construction_approval = _json_object_from_bytes(
        construction_receipt_bytes, str(construction_receipt)
    )
    _validate_visual_receipt_object(
        config,
        stage="construction",
        approval=construction_approval,
        artifact_content_manifest_sha256=str(
            evaluation_payload.get("construction_content_manifest_sha256")
        ),
        expected_reviewed_files=None,
        require_pass=True,
    )
    if construction_approval["reviewer_identity"] != evaluation_payload.get(
        "construction_visual_reviewer_identity"
    ):
        raise ValueError("preserved construction receipt reviewer binding mismatch")
    if construction_approval["authoring_method"] != evaluation_payload.get(
        "construction_visual_authoring_method"
    ):
        raise ValueError("preserved construction receipt method binding mismatch")
    inputs.joinpath("construction-visual-receipt.json").write_bytes(
        construction_receipt_bytes
    )
    inputs.joinpath("evaluation-visual-receipt.json").write_bytes(approval_bytes)
    return _seal_content_artifact(
        config, stage="decision", staging=output_root
    )
