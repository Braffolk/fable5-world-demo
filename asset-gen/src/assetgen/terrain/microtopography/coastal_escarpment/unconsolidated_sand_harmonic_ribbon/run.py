"""Materialize one injective harmonic-ribbon FLOAT attempt."""
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
from ..unconsolidated_sand_curvilinear_strip.evidence import (
    MASTER_PITCH_M,
    SOLVE_PITCH_M,
    TARGET_BBOX_EN,
    bound_path,
    load,
)
from ..unconsolidated_sand_curvilinear_strip.model import fit_strip
from .domain import build_harmonic_ribbon
from .model import synthesize
from .qa import write_qa


SCHEMA = "laas.unconsolidated-sand-harmonic-ribbon-config/1"
ROOT = DATA_WORK / "microtopography" / "coastal-escarpment" / "unconsolidated-sand-harmonic-ribbon" / "sha256"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, logical: str | None = None) -> dict[str, Any]:
    return {"path": logical or str(path.resolve().relative_to(ASSET_GEN_ROOT.parent)), "bytes": path.stat().st_size, "sha256": _sha(path)}


def run(config_path: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA or config.get("status") != "development_float_qa_only":
        raise ValueError("unsupported harmonic-ribbon config")
    expected_regime = {"owner": "unconsolidated_glaciolacustrine_sand", "lithology_code": 40, "lithology_name_et": "Liiv", "genesis_code": 40, "genesis_name_et": "Jaajarvesetted"}
    if config.get("regime") != expected_regime:
        raise ValueError("regime boundary differs")
    chain_config_path = bound_path(config["inputs"]["chain_config"])
    chain_config = json.loads(chain_config_path.read_text(encoding="utf-8"))
    evidence = load(chain_config)
    strip = fit_strip(evidence, chain_config["strip"])
    ribbon = build_harmonic_ribbon(evidence, strip, config["domain"])
    result = synthesize(evidence, strip, ribbon, config)

    local_files = [Path(__file__).with_name(name) for name in ("domain.py", "model.py", "qa.py", "run.py")]
    dependency_files = [
        Path(__file__).parents[1] / "unconsolidated_sand_curvilinear_strip" / name
        for name in ("evidence.py", "model.py", "qa.py")
    ]
    recipe = {
        "config": _identity(config_path),
        "chain_config": _identity(chain_config_path),
        "implementation": [_identity(path) for path in local_files],
        "bound_chain_dependencies": [_identity(path) for path in dependency_files],
        "source_sha256s": sorted(row["sha256"] for row in chain_config["inputs"].values()),
    }
    recipe_sha = hashlib.sha256(_canonical(recipe)).hexdigest()
    output = ROOT / recipe_sha
    if output.exists():
        raise FileExistsError(f"immutable attempt already exists: {output}")

    delta_master = result.master_m - evidence.c0_master_m
    collar = int(round(float(chain_config["strip"]["outer_collar_m"]) / MASTER_PITCH_M))
    collar_values = np.concatenate((delta_master[:collar].ravel(), delta_master[-collar:].ravel(), delta_master[:, :collar].ravel(), delta_master[:, -collar:].ravel()))
    edge = np.concatenate((result.solve_delta_m[0], result.solve_delta_m[-1], result.solve_delta_m[:, 0], result.solve_delta_m[:, -1]))
    area = SOLVE_PITCH_M**2
    eroded = float(np.sum(result.erosion_m, dtype=np.float64) * area)
    deposited = float(np.sum(result.deposition_m, dtype=np.float64) * area)
    conservation = abs(deposited - eroded) / max(eroded, 1.0e-12)
    before = float(np.sqrt(np.mean((evidence.c0_solve_m[ribbon.mask] - result.measured_solve_m[ribbon.mask]) ** 2)))
    after = float(np.sqrt(np.mean((result.solve_absolute_m[ribbon.mask] - result.measured_solve_m[ribbon.mask]) ** 2)))
    mechanical = (
        np.isfinite(result.master_m).all()
        and float(np.max(np.abs(delta_master[evidence.hard_master]))) == 0.0
        and float(np.max(np.abs(collar_values))) == 0.0
        and float(np.max(np.abs(edge))) == 0.0
        and conservation <= 1.0e-6
        and ribbon.jacobian_wrong_sign_count == 0
        and ribbon.jacobian_interior_zero_count == 0
        and ribbon.control_order_fraction >= float(config["domain"]["minimum_control_order_fraction"])
        and result.path_count >= 2
        and after < before
    )
    metrics = {
        "state": "float_qa_pending_visual_verdict" if mechanical else "mechanical_failure",
        "target": {"etak_id": 1826743, "bbox_epsg3301_m": list(TARGET_BBOX_EN), "game_center_xz_m": [311872.0, 191040.0]},
        "regime": expected_regime,
        "surface": {"master_pitch_m": MASTER_PITCH_M, "master_shape": list(result.master_m.shape), "master_dtype": str(result.master_m.dtype), "solve_pitch_m": SOLVE_PITCH_M, "finite": bool(np.isfinite(result.master_m).all()), "single_valued": True},
        "domain": {"construction": "topology_regularized_measured_cross_strip_cell_union_with_convex_boundary_discrete_harmonic_coordinates", "area_m2": ribbon.polygon_area_m2, "solver_residual": ribbon.solver_residual, "jacobian_min_abs": ribbon.jacobian_min_abs, "jacobian_wrong_sign_count": ribbon.jacobian_wrong_sign_count, "jacobian_interior_zero_count": ribbon.jacobian_interior_zero_count, "allowed_all_boundary_degenerate_count": ribbon.boundary_degenerate_count, "control_order_fraction": ribbon.control_order_fraction, "corner_error_max_m": ribbon.corner_error_max_m, "closest_line_ownership": False, "global_xy_smoothing": False},
        "mechanical_gates": {"all_hard_max_abs_m": float(np.max(np.abs(delta_master[evidence.hard_master]))), "outer_collar_max_abs_m": float(np.max(np.abs(collar_values))), "outer_edge_process_max_abs_m": float(np.max(np.abs(edge))), "pass": mechanical},
        "macro": {"ribbon_rmse_before_m": before, "ribbon_rmse_after_m": after, "minimum_delta_m": float(np.min(delta_master)), "maximum_delta_m": float(np.max(delta_master)), "absolute_delta_p95_m": float(np.percentile(np.abs(delta_master[~evidence.hard_master]), 95.0))},
        "process": {"headcut_to_toe_path_count": result.path_count, "eroded_m3": eroded, "deposited_m3": deposited, "net_m3": deposited - eroded, "relative_conservation_error": conservation, "micro_absolute_p95_m": float(np.percentile(np.abs(result.micro_master_m[~evidence.hard_master]), 95.0))},
        "evidence_boundary": {"effective_morphology_floor_m": 0.25, "biala_native_pitch_m": 0.5, "biala_band_abs_p95_m": result.source_band_p95_m, "sub_0_25_m_role": "band_limited_generated_continuation", "sub_0_25_m_measured_truth": False, "owner": "unconsolidated_glaciolacustrine_sand_only", "sandstone_or_carbonate_claim": False, "packing_runtime_material_or_production_authorized": False},
    }

    with tempfile.TemporaryDirectory(prefix="laas-sand-harmonic-") as temporary:
        staging = Path(temporary) / "artifact"
        staging.mkdir()
        np.save(staging / "master-absolute-f64.npy", result.master_m.astype(np.float64), allow_pickle=False)
        np.savez_compressed(
            staging / "research-fields.npz",
            solve_absolute_m=result.solve_absolute_m.astype(np.float32),
            solve_delta_m=result.solve_delta_m.astype(np.float32),
            macro_master_m=result.macro_master_m.astype(np.float32),
            micro_master_m=result.micro_master_m.astype(np.float32),
            macro_correction_m=result.macro_correction_m.astype(np.float32),
            erosion_m=result.erosion_m.astype(np.float32),
            deposition_m=result.deposition_m.astype(np.float32),
            harmonic_mask=ribbon.mask.astype(np.uint8),
            harmonic_u=np.nan_to_num(ribbon.u).astype(np.float32),
            harmonic_v=np.nan_to_num(ribbon.v).astype(np.float32),
            control_u=result.control_u.astype(np.float32),
            control_v=result.control_v.astype(np.float32),
            control_height_m=result.control_height_m.astype(np.float32),
            measured_residual_uv_m=result.measured_residual_uv_m.astype(np.float32),
            valley_strength_uv=result.valley_strength_uv.astype(np.float32),
            path_field_uv=result.path_field_uv.astype(np.float32),
            deposition_uv=result.deposition_uv.astype(np.float32),
        )
        (staging / "metrics.json").write_bytes(_canonical(metrics))
        qa = write_qa(staging / "qa", evidence, ribbon, result, recipe_sha, recipe["source_sha256s"])
        manifest = {
            "schema_version": "laas.unconsolidated-sand-harmonic-ribbon-artifact/1",
            "build_id": recipe_sha,
            "status": metrics["state"],
            "recipe": recipe,
            "sources": evidence.source_identities,
            "outputs": {"master": _identity(staging / "master-absolute-f64.npy", "master-absolute-f64.npy"), "research_fields": _identity(staging / "research-fields.npz", "research-fields.npz"), "metrics": _identity(staging / "metrics.json", "metrics.json"), "qa_index": {**_identity(staging / "qa" / "index.json", "qa/index.json"), "image_count": qa["image_count"]}},
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
    print(run(args.config.resolve()))


if __name__ == "__main__":
    main()
