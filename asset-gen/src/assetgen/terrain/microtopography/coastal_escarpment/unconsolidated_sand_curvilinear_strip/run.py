"""Run one immutable curvilinear-strip FLOAT attempt."""
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
from .model import CHAIN_NAMES, synthesize
from .qa import write_qa


SCHEMA = "laas.unconsolidated-sand-curvilinear-strip-config/1"
ROOT = DATA_WORK / "microtopography" / "coastal-escarpment" / "unconsolidated-sand-curvilinear-strip" / "sha256"


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
        raise ValueError("unsupported curvilinear-strip config")
    expected_regime = {"owner": "unconsolidated_glaciolacustrine_sand", "lithology_code": 40, "lithology_name_et": "Liiv", "genesis_code": 40, "genesis_name_et": "Jaajarvesetted"}
    if config.get("regime") != expected_regime:
        raise ValueError("regime boundary differs")
    evidence = load(config)
    result = synthesize(evidence, config)
    implementation = [Path(__file__).with_name(name) for name in ("evidence.py", "model.py", "qa.py", "run.py")]
    recipe = {"config": _identity(config_path), "implementation": [_identity(path) for path in implementation], "source_sha256s": sorted(row["sha256"] for row in config["inputs"].values())}
    recipe_sha = hashlib.sha256(_canonical(recipe)).hexdigest()
    output = ROOT / recipe_sha
    if output.exists():
        raise FileExistsError(f"immutable attempt already exists: {output}")

    delta_master = result.master_m - evidence.c0_master_m
    collar = int(round(float(config["strip"]["outer_collar_m"]) / MASTER_PITCH_M))
    collar_values = np.concatenate((delta_master[:collar].ravel(), delta_master[-collar:].ravel(), delta_master[:, :collar].ravel(), delta_master[:, -collar:].ravel()))
    area = SOLVE_PITCH_M**2
    eroded = float(np.sum(result.erosion_m, dtype=np.float64) * area)
    deposited = float(np.sum(result.deposition_m, dtype=np.float64) * area)
    conservation = abs(deposited - eroded) / max(eroded, 1.0e-12)
    edge_process = np.concatenate((result.solve_delta_m[0], result.solve_delta_m[-1], result.solve_delta_m[:, 0], result.solve_delta_m[:, -1]))
    paired_fraction = float(np.mean(result.strip.valid_station))
    graph_mask = (result.chain_distance_m <= 0.5) & (result.strip_support > 0.5) & ~evidence.hard_solve
    measured = np.where(np.isfinite(evidence.reference_m), evidence.reference_m, evidence.c0_solve_m[::4, ::4])
    measured_solve = np.asarray(__import__("scipy").ndimage.zoom(measured, 4.0, order=3)[:513, :513])
    before = float(np.sqrt(np.mean((evidence.c0_solve_m[graph_mask] - measured_solve[graph_mask]) ** 2)))
    after = float(np.sqrt(np.mean((result.solve_absolute_m[graph_mask] - measured_solve[graph_mask]) ** 2)))
    mechanical = (
        np.isfinite(result.master_m).all()
        and float(np.max(np.abs(delta_master[evidence.hard_master]))) == 0.0
        and float(np.max(np.abs(collar_values))) == 0.0
        and conservation <= 1.0e-6
        and float(np.max(np.abs(edge_process))) == 0.0
        and paired_fraction >= float(config["strip"]["minimum_paired_station_fraction"])
        and after < before
    )
    metrics = {
        "state": "float_qa_pending_visual_verdict" if mechanical else "mechanical_failure",
        "target": {"etak_id": 1826743, "bbox_epsg3301_m": list(TARGET_BBOX_EN), "game_center_xz_m": [311872.0, 191040.0]},
        "regime": expected_regime,
        "surface": {"master_pitch_m": MASTER_PITCH_M, "master_shape": list(result.master_m.shape), "master_dtype": str(result.master_m.dtype), "solve_pitch_m": SOLVE_PITCH_M, "finite": bool(np.isfinite(result.master_m).all()), "single_valued": True},
        "method": {"kind": "one_continuous_curvilinear_strip_with_joint_along_s_chain_fields_and_piecewise_monotone_cross_sections", "global_xy_pde": False, "independent_station_ownership": False, "one_metre_mean_closure": False, "chain_order": list(CHAIN_NAMES)},
        "mechanical_gates": {"all_hard_max_abs_m": float(np.max(np.abs(delta_master[evidence.hard_master]))), "outer_collar_max_abs_m": float(np.max(np.abs(collar_values))), "outer_edge_process_max_abs_m": float(np.max(np.abs(edge_process))), "pass": mechanical},
        "strip": {"line_length_m": float(evidence.etak_line.length), "station_count": int(len(result.strip.station_m)), "paired_station_fraction": paired_fraction, "graph_rmse_before_m": before, "graph_rmse_after_m": after, "support_fraction": float(np.mean(result.strip_support > 0.0)), "rill_strip_fraction": float(np.mean(result.strip.rill_strip > 0.01))},
        "transport": {"eroded_m3": eroded, "deposited_m3": deposited, "net_m3": deposited - eroded, "relative_conservation_error": conservation},
        "relief": {"minimum_delta_m": float(np.min(delta_master)), "maximum_delta_m": float(np.max(delta_master)), "absolute_delta_p95_m": float(np.percentile(np.abs(delta_master[~evidence.hard_master]), 95.0)), "micro_absolute_p95_m": float(np.percentile(np.abs(result.micro_master_m[~evidence.hard_master]), 95.0))},
        "evidence_boundary": {"effective_morphology_floor_m": 0.25, "biala_native_pitch_m": 0.5, "biala_band_abs_p95_m": result.source_band_p95_m, "sub_0_25_m_role": "band_limited_generated_continuation", "sub_0_25_m_measured_truth": False, "sandstone_carbonate_till_or_generic_cliff_claim": False, "packing_runtime_or_production_authorized": False},
    }

    with tempfile.TemporaryDirectory(prefix="laas-sand-strip-") as temporary:
        staging = Path(temporary) / "artifact"
        staging.mkdir()
        np.save(staging / "master-absolute-f64.npy", result.master_m.astype(np.float64), allow_pickle=False)
        np.savez_compressed(
            staging / "research-fields.npz",
            solve_absolute_m=result.solve_absolute_m.astype(np.float32),
            solve_delta_m=result.solve_delta_m.astype(np.float32),
            macro_master_m=result.macro_master_m.astype(np.float32),
            micro_master_m=result.micro_master_m.astype(np.float32),
            erosion_m=result.erosion_m.astype(np.float32),
            deposition_m=result.deposition_m.astype(np.float32),
            strip_s_m=result.strip_s_m.astype(np.float32),
            strip_n_m=result.strip_n_m.astype(np.float32),
            strip_support=result.strip_support.astype(np.float32),
            chain_n_m=result.strip.chain_n_m.astype(np.float32),
            chain_height_m=result.strip.chain_height_m.astype(np.float32),
            measured_strip_m=result.strip.measured_strip_m.astype(np.float32),
            macro_strip_m=result.strip.macro_strip_m.astype(np.float32),
            rill_strip=result.strip.rill_strip.astype(np.float32),
        )
        (staging / "metrics.json").write_bytes(_canonical(metrics))
        qa = write_qa(staging / "qa", evidence, result, recipe_sha, [row["sha256"] for row in config["inputs"].values()])
        manifest = {
            "schema_version": "laas.unconsolidated-sand-curvilinear-strip-artifact/1",
            "build_id": recipe_sha,
            "status": metrics["state"],
            "recipe": recipe,
            "sources": evidence.source_identities,
            "outputs": {
                "master": _identity(staging / "master-absolute-f64.npy", "master-absolute-f64.npy"),
                "research_fields": _identity(staging / "research-fields.npz", "research-fields.npz"),
                "metrics": _identity(staging / "metrics.json", "metrics.json"),
                "qa_index": {**_identity(staging / "qa" / "index.json", "qa/index.json"), "image_count": qa["image_count"]},
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
    print(run(args.config.resolve()))


if __name__ == "__main__":
    main()
