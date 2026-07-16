"""Materialize one immutable, unpacked FLOAT QA attempt."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .....config import ASSET_GEN_ROOT, DATA_WORK
from .evidence import MASTER_PITCH_M, SOLVE_PITCH_M, TARGET_BBOX_EN, load
from .model import FEATURE_NAMES, synthesize
from .qa import write_qa


SCHEMA = "laas.unconsolidated-sand-feature-graph-config/1"
ROOT = DATA_WORK / "microtopography" / "coastal-escarpment" / "unconsolidated-sand-feature-graph" / "sha256"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, relative: str | None = None) -> dict[str, Any]:
    return {"path": relative or str(path.resolve().relative_to(ASSET_GEN_ROOT.parent)), "bytes": path.stat().st_size, "sha256": _sha(path)}


def run(config_path: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA or config.get("status") != "development_float_qa_only":
        raise ValueError("unsupported feature-graph config")
    if config["regime"] != {
        "owner": "unconsolidated_glaciolacustrine_sand",
        "lithology_code": 40,
        "lithology_name_et": "Liiv",
        "genesis_code": 40,
        "genesis_name_et": "Jaajarvesetted",
    }:
        raise ValueError("regime boundary differs")
    evidence = load(config)
    result = synthesize(evidence, config)
    implementation_paths = [Path(__file__).with_name(name) for name in ("evidence.py", "model.py", "qa.py", "run.py")]
    recipe = {
        "config": _identity(config_path),
        "implementation": [_identity(path) for path in implementation_paths],
        "source_sha256s": sorted(row["sha256"] for row in config["inputs"].values()),
    }
    recipe_sha = hashlib.sha256(_canonical(recipe)).hexdigest()
    output = ROOT / recipe_sha
    if output.exists():
        raise FileExistsError(f"immutable attempt already exists: {output}")

    hard_delta = result.master_m.astype(np.float64) - evidence.c0_master_m
    collar = int(round(float(config["solver"]["outer_collar_m"]) / MASTER_PITCH_M))
    collar_values = np.concatenate((hard_delta[:collar].ravel(), hard_delta[-collar:].ravel(), hard_delta[:, :collar].ravel(), hard_delta[:, -collar:].ravel()))
    graph = result.feature_labels > 0
    before = np.abs(evidence.c0_solve_m - result.measured_target_m)
    after = np.abs(result.solve_absolute_m.astype(np.float64) - result.measured_target_m)
    graph_before = float(np.sqrt(np.mean(before[graph] ** 2)))
    graph_after = float(np.sqrt(np.mean(after[graph] ** 2)))
    cell_area = SOLVE_PITCH_M**2
    eroded = float(np.sum(result.erosion_m, dtype=np.float64) * cell_area)
    deposited = float(np.sum(result.deposition_m, dtype=np.float64) * cell_area)
    conservation = abs(deposited - eroded) / max(eroded, 1.0e-12)
    edge_graph = np.concatenate((graph[0], graph[-1], graph[:, 0], graph[:, -1]))
    labels, counts = np.unique(result.feature_labels[result.feature_labels > 0], return_counts=True)
    feature_counts = {FEATURE_NAMES[int(label) - 1]: int(count) for label, count in zip(labels, counts)}
    mechanical_pass = (
        np.isfinite(result.master_m).all()
        and float(np.max(np.abs(hard_delta[evidence.hard_master]))) == 0.0
        and float(np.max(np.abs(collar_values))) == 0.0
        and graph_after < graph_before
        and conservation <= 1.0e-6
        and not np.any(edge_graph)
    )
    metrics = {
        "state": "float_qa_pending_visual_verdict" if mechanical_pass else "mechanical_failure",
        "target": {"etak_id": 1826743, "bbox_epsg3301_m": list(TARGET_BBOX_EN), "game_center_xz_m": [311872.0, 191040.0]},
        "regime": config["regime"],
        "surface": {"master_pitch_m": MASTER_PITCH_M, "master_shape": list(result.master_m.shape), "solve_pitch_m": SOLVE_PITCH_M, "solve_shape": list(result.solve_delta_m.shape), "absolute_single_valued": True, "finite": bool(np.isfinite(result.master_m).all())},
        "hard_gates": {"all_hard_max_abs_m": float(np.max(np.abs(hard_delta[evidence.hard_master]))), "outer_collar_max_abs_m": float(np.max(np.abs(collar_values))), "pass": mechanical_pass},
        "feature_graph": {"ordered_classes": list(FEATURE_NAMES), "solve_cell_counts": feature_counts, "measured_target_rmse_before_m": graph_before, "measured_target_rmse_after_m": graph_after, "adherence_improved": graph_after < graph_before, "outer_edge_cells": int(np.count_nonzero(edge_graph))},
        "transport_ledger": {"eroded_m3": eroded, "deposited_m3": deposited, "net_m3": deposited - eroded, "relative_conservation_error": conservation},
        "relief": {"minimum_delta_m": float(np.min(hard_delta)), "maximum_delta_m": float(np.max(hard_delta)), "absolute_delta_p95_m": float(np.percentile(np.abs(hard_delta[~evidence.hard_master]), 95.0)), "micro_absolute_p95_m": float(np.percentile(np.abs(result.micro_m[~evidence.hard_master]), 95.0))},
        "biala_statistics": {"named_band_m": [0.25, 1.25], "native_source_pitch_m": 0.5, "lower_band_samples_are_interpolated": True, "absolute_p50_m": result.source_statistics.erosion_band_abs_p50_m, "absolute_p95_m": result.source_statistics.erosion_band_abs_p95_m, "absolute_p99_m": result.source_statistics.erosion_band_abs_p99_m, "normalized_profile_samples": int(len(result.source_statistics.normalized_profile)), "normalized_curvature_abs_p95_per_m": result.source_statistics.curvature_abs_p95_per_m},
        "evidence_boundary": {"effective_morphology_floor_m": 0.25, "master_samples_m": [0.0625, 0.125, 0.25, 0.5], "sub_0_25_m_role": "band_limited_generated_continuation_subordinate_to_measured_graph", "sub_0_25_m_measured_truth": False, "estonia_transfer_claim": False, "packing_or_runtime_authorized": False},
        "solver": {"kind": "one_full_canvas_2d_screened_biharmonic_absolute_surface_with_height_and_tangent_normal_graph_constraints", "cg_info": result.solver_info, "one_metre_mean_closure_enforced": False},
    }

    with tempfile.TemporaryDirectory(prefix="laas-sand-graph-") as temporary:
        staging = Path(temporary) / "artifact"
        staging.mkdir()
        np.save(staging / "master-absolute-f32.npy", result.master_m, allow_pickle=False)
        np.savez_compressed(
            staging / "research-fields.npz",
            solve_absolute_m=result.solve_absolute_m,
            solve_delta_m=result.solve_delta_m,
            macro_m=result.macro_m,
            micro_m=result.micro_m,
            feature_labels=result.feature_labels,
            feature_strength=result.feature_strength,
            erosion_m=result.erosion_m,
            deposition_m=result.deposition_m,
            measured_target_m=result.measured_target_m,
            measured_confidence=result.measured_confidence,
            biala_normalized_profile=result.source_statistics.normalized_profile,
            biala_normalized_curvature=result.source_statistics.normalized_curvature,
        )
        (staging / "metrics.json").write_bytes(_canonical(metrics))
        qa_index = write_qa(staging / "qa", evidence, result, recipe_sha, [row["sha256"] for row in config["inputs"].values()])
        manifest = {
            "schema_version": "laas.unconsolidated-sand-feature-graph-artifact/1",
            "build_id": recipe_sha,
            "status": metrics["state"],
            "recipe": recipe,
            "sources": evidence.source_identities,
            "outputs": {
                "master": _identity(staging / "master-absolute-f32.npy", "master-absolute-f32.npy"),
                "research_fields": _identity(staging / "research-fields.npz", "research-fields.npz"),
                "metrics": _identity(staging / "metrics.json", "metrics.json"),
                "qa_index": {**_identity(staging / "qa" / "index.json", "qa/index.json"), "image_count": qa_index["image_count"]},
            },
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "pid": os.getpid()},
        }
        (staging / "manifest.json").write_bytes(_canonical(manifest))
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staging, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    output = run(args.config.resolve())
    print(output)


if __name__ == "__main__":
    main()
