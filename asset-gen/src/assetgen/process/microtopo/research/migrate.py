"""Republish the consumed a1ce3c audit with corrected provenance and missingness.

This module never imports or calls semantic fitting/audit code. It accepts only the
exact immutable legacy file ledger below and preserves the frozen NO-GO decision.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from ....config import ASSET_GEN_ROOT, DATA_WORK
from .bundle import canonical_json, sha256_file
from .provenance import numerical_environment_identity


LEGACY_RECIPE_ID = "a1ce3cabee55f73ad8a8fe005e272e12c1807766ed42db10b21c86211a39e135"
LEGACY_FILE_SHA256 = {
    "manifest.json": "6790e89c09d8ab0481ca2dcf214033ffee5130f00cf360b33c2750caaa2b11f7",
    "qa/01_hovi_sheet_hypotheses.png": "afd15f951a1e895259c1ccf1c9860579cf614dbd54300baab07b6f872c23ab0f",
    "qa/02_semantic_included_excluded_unknown.png": "071da9d7b0dc260bc5cefbb66a46b905c71be8912cce50f849332919a997e93a",
    "qa/03_support_view_disagreement.png": "636b1fd3135c61478a8aab03586bfd2625f2553a644beaab7e69804d565f2d6a",
    "qa/04_b1_b2_eligibility_and_partitions.png": "2337853d8034d16fd0a3ec2d6748da58bbf318c8c658d70cdfab3c2e9ce548d5",
    "qa/05_evo_1086_1065_adapter.png": "13413df90eb979f07c15c3d39e5b122dc2b837b9a6388ebf6fd7d06e3a3d13d3",
    "qa/index.json": "198e37643e6f1d5452dae2ecfa5478c37fb48762fdc562e97d1515647777495a",
    "recipe.json": LEGACY_RECIPE_ID,
    "semantic/forestsemantic-geometry-exclusion.joblib": "f032c50302de6fb51db2d22915fbdaf6061e147ee0a0e6d0f619209be9bdab82",
    "semantic/semantic-freeze.8258570b11d6a72a330ac54ed7f07adeeed054b821526ac7c7d0515c1c5a7732.json": "8258570b11d6a72a330ac54ed7f07adeeed054b821526ac7c7d0515c1c5a7732",
    "semantic/semantic-result.json": "d58c43a771c5bfe8fd68570b806eec8e3fd70ef11c1cae2e37fef3e5823430c4",
    "surfaces/evo-1065.npz": "6a0025565ec3ec55ed41ee7fc872279139020af8b9c2633802982132172da37a",
    "surfaces/evo-1086.npz": "96f7c535d86411299acea80c2608edfc7f30b351dfcad8cd23489016b3f8b505",
    "surfaces/hovi-hy-spruce4.npz": "cf34f3405a2d9aebdae7956f18d9bb135b998774dca37b393a5fb7e6fe4daecf",
}
DECISION_SHA256 = {
    "semantic_gate": "943aca0b44fe1a4bfda4d20cdb4ddd80fca58af5bbe910b2acd1c62b1bf6d7f1",
    "band_decision": "75b6f803c7f8d2d6c3bac8f8228af204a13f07d08598307502f74d3c661c59c2",
    "hovi_gate": "79a6c07f1a3b2b0ba51e1afb9cc162866fa75421b025a251dfe5ff486b31c3c8",
}


def _json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    payload = canonical_json(value)
    temporary = path.with_name(path.name + ".part")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def _legacy_root(base: Path) -> Path:
    return base / LEGACY_RECIPE_ID


def _verify_legacy(base: Path) -> tuple[Path, dict]:
    root = _legacy_root(base)
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_files != set(LEGACY_FILE_SHA256):
        raise ValueError("legacy forest gate file set changed")
    for relative, expected in LEGACY_FILE_SHA256.items():
        if sha256_file(root / relative) != expected:
            raise ValueError(f"legacy forest gate byte changed: {relative}")
    manifest = json.loads((root / "manifest.json").read_bytes())
    if (
        manifest.get("recipe_id") != LEGACY_RECIPE_ID
        or manifest.get("status") != "complete"
        or manifest.get("model_training_authorized") is not False
        or manifest.get("synthesis_authorized") is not False
    ):
        raise ValueError("legacy forest gate identity/authority changed")
    for field, expected in DECISION_SHA256.items():
        if _json_sha256(manifest[field]) != expected:
            raise ValueError(f"legacy frozen decision changed: {field}")
    for row in manifest["surfaces"]:
        legacy_path = root / "surfaces" / row["path"]
        if sha256_file(legacy_path) != row["sha256"] or legacy_path.stat().st_size != row["bytes"]:
            raise ValueError(f"legacy surface manifest binding changed: {row['path']}")
    for row in manifest["qa"]:
        path = root / row["path"]
        if sha256_file(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError(f"legacy QA manifest binding changed: {row['path']}")
    return root, manifest


def _rewrite_surface(source_path: Path, destination: Path, source_id: str) -> dict:
    with np.load(source_path) as legacy:
        values = {name: legacy[name] for name in legacy.files}
    shape = values["height_m_f64"].shape
    if values["semantic_probabilities_f32"].shape != (*shape, 6):
        raise ValueError(f"legacy semantic probability shape changed: {source_id}")
    legacy_view_groups = values.pop("view_count_u8")
    values.pop("group_a_support_u8")
    values.pop("group_b_support_u8")
    values["semantic_probabilities_f32"] = np.full((*shape, 6), np.nan, dtype=np.float32)
    values["semantic_probability_available"] = np.zeros((*shape, 6), dtype=np.bool_)
    values["physical_view_count_f32"] = np.full(shape, np.nan, dtype=np.float32)
    values["physical_view_count_available"] = np.zeros(shape, dtype=np.bool_)
    values["group_a_support_f32"] = np.full(shape, np.nan, dtype=np.float32)
    values["group_b_support_f32"] = np.full(shape, np.nan, dtype=np.float32)
    if source_id.startswith("evo."):
        values["redundancy_group_count_u8"] = legacy_view_groups.astype(np.uint8)
        values["redundancy_group_count_available"] = values["direct_observed"].astype(np.bool_)
    else:
        values["redundancy_group_count_u8"] = np.zeros(shape, dtype=np.uint8)
        values["redundancy_group_count_available"] = np.zeros(shape, dtype=np.bool_)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as target:
        np.savez_compressed(target, **values)
        target.flush()
        os.fsync(target.fileno())
    return {
        "path": f"surfaces/{destination.name}",
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
        "serialization": {
            "semantic_probability": "six float32 NaN channels with availability=false",
            "physical_view_count": "float32 NaN with availability=false",
            "redundancy_group_count": (
                "known non-physical source groups retained separately"
                if source_id.startswith("evo.")
                else "unavailable"
            ),
        },
    }


def _migration_recipe(repository_root: Path, legacy_manifest: dict) -> dict:
    module_root = Path(__file__).parent
    return {
        "schema_version": "forest-floor-research-artifact-migration/1.0.0",
        "operation": "serialization_and_provenance_republication_only",
        "legacy_recipe_id": LEGACY_RECIPE_ID,
        "legacy_manifest_sha256": LEGACY_FILE_SHA256["manifest.json"],
        "legacy_file_sha256": LEGACY_FILE_SHA256,
        "frozen_decision_sha256": DECISION_SHA256,
        "frozen_semantic_result_sha256": legacy_manifest["semantic_result_sha256"],
        "implementation": {
            "modules": {
                path.name: sha256_file(path)
                for path in sorted(module_root.glob("*.py"))
            },
            "environment": numerical_environment_identity(repository_root),
        },
        "corrections": [
            "surface paths are root-relative and include surfaces/",
            "unavailable semantic probabilities are NaN with explicit availability=false",
            "unknown physical view counts/support are NaN with explicit availability=false",
            "Evo redundancy-group counts remain separate from physical view count",
            "Python SciPy joblib uv and uv.lock identities are recipe-bound",
        ],
        "prohibitions": {
            "semantic_fit_or_audit": "not imported or called",
            "ForestSemantic_test_reopen": False,
            "Hovi_scan": False,
            "decision_change": False,
        },
    }


def migrate_consumed_forest_gate(output_root: Path | None = None, *, log=print) -> Path:
    repository_root = ASSET_GEN_ROOT.parent
    base = output_root or DATA_WORK / "microtopography" / "forest-floor-research" / "sha256"
    legacy_root, legacy_manifest = _verify_legacy(base)
    recipe = _migration_recipe(repository_root, legacy_manifest)
    recipe_id = hashlib.sha256(canonical_json(recipe)).hexdigest()
    final_root = base / recipe_id
    manifest_path = final_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_bytes())
        if manifest.get("recipe_id") != recipe_id or manifest.get("status") != "complete":
            raise ValueError("existing migrated forest artifact conflicts")
        return manifest_path
    temporary = base / f".{recipe_id}.preparing-{os.getpid()}"
    if temporary.exists() or final_root.exists():
        raise ValueError("incomplete migrated forest artifact requires inspection")
    temporary.mkdir(parents=True)
    _atomic_json(temporary / "recipe.json", recipe)

    for relative in LEGACY_FILE_SHA256:
        if relative.startswith("qa/") and relative != "qa/index.json":
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(legacy_root / relative, destination)
        elif relative.startswith("semantic/"):
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(legacy_root / relative, destination)

    migrated_surfaces = []
    legacy_by_name = {row["source_id"]: row for row in legacy_manifest["surfaces"]}
    for source_id, legacy_row in sorted(legacy_by_name.items()):
        source = legacy_root / "surfaces" / legacy_row["path"]
        destination = temporary / "surfaces" / legacy_row["path"]
        serialization = _rewrite_surface(source, destination, source_id)
        migrated_surfaces.append({**legacy_row, **serialization})

    qa = legacy_manifest["qa"]
    qa_index = {
        "schema_version": "forest-floor-pretraining-qa/1.0.1",
        "recipe_id": recipe_id,
        "migrated_from_recipe_id": LEGACY_RECIPE_ID,
        "artifacts": qa,
    }
    _atomic_json(temporary / "qa" / "index.json", qa_index)
    manifest = {
        **legacy_manifest,
        "schema_version": "microtopography-research-surface/1.0.1",
        "recipe_id": recipe_id,
        "recipe_sha256": sha256_file(temporary / "recipe.json"),
        "surfaces": migrated_surfaces,
        "qa_index_sha256": sha256_file(temporary / "qa" / "index.json"),
        "migration": {
            "operation": recipe["operation"],
            "legacy_recipe_id": LEGACY_RECIPE_ID,
            "legacy_manifest_sha256": LEGACY_FILE_SHA256["manifest.json"],
            "legacy_file_sha256": LEGACY_FILE_SHA256,
            "frozen_decision_sha256": DECISION_SHA256,
            "ForestSemantic_test_reopened": False,
            "model_refit": False,
            "audit_recomputed": False,
            "Hovi_scanned": False,
        },
        "provenance": {
            **legacy_manifest["provenance"],
            "legacy_provenance": legacy_manifest["provenance"],
            "implementation": recipe["implementation"],
            "full_environment_bound": True,
        },
    }
    for field, expected in DECISION_SHA256.items():
        if _json_sha256(manifest[field]) != expected:
            raise AssertionError(f"migration changed frozen decision: {field}")
    _atomic_json(temporary / "manifest.json", manifest)
    _verify_migrated(temporary, manifest)
    temporary.replace(final_root)
    log(f"migrated forest gate: {recipe_id}")
    return final_root / "manifest.json"


def _verify_migrated(root: Path, manifest: dict) -> None:
    for row in manifest["surfaces"]:
        path = root / row["path"]
        if sha256_file(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise AssertionError(f"migrated surface binding failed: {row['path']}")
        with np.load(path) as surface:
            semantic = surface["semantic_probabilities_f32"]
            semantic_available = surface["semantic_probability_available"]
            view = surface["physical_view_count_f32"]
            view_available = surface["physical_view_count_available"]
            if (
                np.any(semantic_available)
                or np.any(np.isfinite(semantic))
                or np.any(view_available)
                or np.any(np.isfinite(view))
            ):
                raise AssertionError(f"migrated missingness is not explicit: {row['path']}")
    for row in manifest["qa"]:
        path = root / row["path"]
        if sha256_file(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise AssertionError(f"migrated QA binding failed: {row['path']}")
    if sha256_file(root / "semantic" / "semantic-result.json") != manifest["semantic_result_sha256"]:
        raise AssertionError("migrated semantic result changed")


if __name__ == "__main__":
    print(migrate_consumed_forest_gate())
