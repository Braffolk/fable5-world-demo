"""One-shot orchestration for the frozen forest pre-training gate."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np

from ....config import ASSET_GEN_ROOT, DATA_WORK
from .bundle import canonical_json, load_frozen_bundle, sha256_file
from .contracts import SemanticMethod
from .hovi import reconstruct_hovi_surface
from .qa import render_qa
from .semantics import fit_and_audit_semantics
from .surfaces import adapt_evo, failed_hovi_surface, save_surface


def _atomic_json(path: Path, value: dict) -> None:
    payload = canonical_json(value)
    temporary = path.with_name(path.name + ".part")
    temporary.write_bytes(payload)
    with temporary.open("rb") as source:
        os.fsync(source.fileno())
    temporary.replace(path)


def _implementation_identity() -> dict:
    root = Path(__file__).parent
    return {
        "modules": {path.name: sha256_file(path) for path in sorted(root.glob("*.py"))},
        "environment": {
            "numpy": np.__version__,
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "laspy": importlib.metadata.version("laspy"),
            "pillow": importlib.metadata.version("pillow"),
        },
    }


def build_forest_floor_research_gate(
    output_root: Path | None = None,
    *,
    log=print,
) -> Path:
    repository_root = ASSET_GEN_ROOT.parent
    bundle = load_frozen_bundle(repository_root)
    method = SemanticMethod()
    recipe = {
        "schema_version": "forest-floor-pretraining-gate-recipe/1.0.0",
        "bundle_id": bundle.bundle_identity,
        "contract_sha256": bundle.contract_sha256,
        "current_normative_input_sha256": {
            path.relative_to(repository_root).as_posix(): digest
            for path, digest in bundle.current_input_sha256
        },
        "nonnormative_tracking_input_sha256": {
            path.relative_to(repository_root).as_posix(): digest
            for path, digest in bundle.tracking_input_sha256
        },
        "implementation": _implementation_identity(),
        "method": method.__dict__,
        "roles": {
            "fit": ["ForestSemantic-MS train1", "train2", "train3"],
            "calibration": ["ForestSemantic-MS train4"],
            "frozen_semantic_audit": ["ForestSemantic-MS test1", "test2"],
            "weak_geometry": ["Hovi HY_SPRUCE4"],
            "r2_geometry_audit": ["Evo 1086", "Evo 1065"],
            "excluded": ["HY_PINE2", "JS_SPRUCE1", "LUKE", "Taevaskoda"],
        },
    }
    recipe_id = hashlib.sha256(canonical_json(recipe)).hexdigest()
    base = output_root or DATA_WORK / "microtopography" / "forest-floor-research" / "sha256"
    final_root = base / recipe_id
    manifest_path = final_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_bytes())
        if manifest.get("recipe_id") != recipe_id or manifest.get("status") != "complete":
            raise ValueError("existing forest research gate conflicts")
        return manifest_path
    temporary = base / f".{recipe_id}.preparing-{os.getpid()}"
    if temporary.exists() or final_root.exists():
        raise ValueError("incomplete forest research gate requires inspection")
    temporary.mkdir(parents=True)
    _atomic_json(temporary / "recipe.json", recipe)

    semantic = fit_and_audit_semantics(bundle, temporary / "semantic", method, log=log)
    if semantic.audit_passed:
        hovi, hovi_gate = reconstruct_hovi_surface(bundle, semantic, log=log)
    else:
        hovi = failed_hovi_surface(bundle, semantic)
        hovi_gate = {
            "status": "not_run_semantic_audit_failed",
            "capacity": {
                "weighted_effective_train_area_m2": 0.0,
                "kish_effective_nonoverlapping_train_windows": 0.0,
                "complete_nonzero_weight_windows_development": 0,
                "complete_nonzero_weight_windows_internal_audit": 0,
                "passed": False,
            },
        }
    evo = (
        adapt_evo(bundle, "bba7a00ad160cbbdb1193fb50d89f629283eb48a31f005f0fa0e7168263fdb3d", "1086"),
        adapt_evo(bundle, "674116851f1f22acc5143ed0cec0ed0eac7c8d764458cfd562f4a6b5d5835f3c", "1065"),
    )
    surface_root = temporary / "surfaces"
    surfaces = [
        save_surface(surface_root / "hovi-hy-spruce4.npz", hovi),
        save_surface(surface_root / "evo-1086.npz", evo[0]),
        save_surface(surface_root / "evo-1065.npz", evo[1]),
    ]
    qa = render_qa(bundle, semantic, hovi, evo, temporary / "qa")
    b1_go = bool(semantic.audit_passed and hovi_gate["capacity"]["passed"])
    manifest = {
        "schema_version": "microtopography-research-surface/1.0.0",
        "recipe_id": recipe_id,
        "status": "complete",
        "authority": "research_only",
        "target_truth": False,
        "synthesis_authorized": False,
        "model_training_authorized": False,
        "production_owner": None,
        "estonia_transfer": "none",
        "latest_eligible": False,
        "semantic_gate": {
            "threshold": semantic.threshold,
            "audit_passed": semantic.audit_passed,
            "calibration": semantic.calibration,
            "audits": semantic.audits,
        },
        "band_decision": {
            "B1": "research_trainable" if b1_go else "R1_research_ineligible",
            "B2": "research_ineligible_by_this_contract",
            "production": "target_evidence_insufficient",
            "go_no": "GO_research_B1" if b1_go else "NO_GO_before_model_training",
            "reason": (
                "frozen semantic audit and continuous weak-geometry capacity passed"
                if b1_go
                else "frozen semantic or continuous weak-geometry capacity gate failed"
            ),
        },
        "hovi_gate": hovi_gate,
        "surfaces": surfaces,
        "qa": qa,
        "recipe_sha256": sha256_file(temporary / "recipe.json"),
        "semantic_result_sha256": sha256_file(temporary / "semantic" / "semantic-result.json"),
        "provenance": {
            "evidence_tiers": {
                "Hovi HY_SPRUCE4": hovi.tier,
                "Evo 1086": "R2_research_audit",
                "Evo 1065": "R2_research_audit",
            },
            "normative_inputs": {
                path.relative_to(repository_root).as_posix(): digest
                for path, digest in bundle.inputs
            },
            "implementation": recipe["implementation"],
            "licenses": {
                "Hovi": "CC BY 4.0; bound through the immutable spatial manifest",
                "Evo": "bound through each immutable converted-surface manifest",
                "ForestSemantic-MS": "bound through retained source and semantic-evidence manifests",
            },
            "total_surface_error": None,
            "interpretation": "weak research geometry only; no target truth or Estonia transfer",
        },
    }
    _atomic_json(temporary / "qa" / "index.json", {"schema_version": "forest-floor-pretraining-qa/1.0.0", "recipe_id": recipe_id, "artifacts": qa})
    manifest["qa_index_sha256"] = sha256_file(temporary / "qa" / "index.json")
    _atomic_json(temporary / "manifest.json", manifest)
    temporary.replace(final_root)
    log(f"forest floor pre-training gate: {manifest['band_decision']['go_no']} {recipe_id}")
    return final_root / "manifest.json"
