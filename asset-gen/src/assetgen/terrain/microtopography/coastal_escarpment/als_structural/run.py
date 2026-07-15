"""Materialize one immutable Development-A connected ALS structural artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from .....config import ASSET_GEN_ROOT, DATA_WORK
from .evidence import StructuralInputs, load_inputs
from .model import ProjectionResult, solve_projection
from .qa import write_qa
from .reference import StructuralReference, build_reference

_SCHEMA = "laas.coastal-escarpment-als-structural-config/1"
_ARTIFACT_ROOT = DATA_WORK / "microtopography" / "coastal-escarpment" / "als-structural" / "sha256"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, logical_path: str | None = None) -> dict[str, Any]:
    if logical_path is None:
        logical_path = str(path.resolve().relative_to(ASSET_GEN_ROOT.parent))
    return {"path": logical_path, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _read_config(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != _SCHEMA:
        raise ValueError("unsupported connected ALS structural config")
    if document.get("status") != "development_only" or document.get("target_etak_id") != 1826743:
        raise ValueError("Development-A boundary differs")
    if document.get("output_chunks") != [[-2, 2436, 1492], [-2, 2437, 1492]]:
        raise ValueError("adjacent output pair differs")
    laws = document.get("hard_laws", {})
    required = {
        "mapped_non_heightfield_face_stays_c0": True,
        "no_arbitrary_distance_corridor": True,
        "line_profile_target_forbidden": True,
        "biala_geometry_or_parameters_forbidden": True,
        "packed_output_requires_offline_acceptance": True,
        "taevaskoda_sentinel_opened": False,
    }
    if laws != required:
        raise ValueError("hard-law boundary differs")
    return document


def _holdout_metrics(inputs: StructuralInputs, residual: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    rr = (inputs.bbox_en[3] - inputs.point_y) / inputs.solve_pitch_m
    cc = (inputs.point_x - inputs.bbox_en[0]) / inputs.solve_pitch_m
    predicted = ndimage.map_coordinates(residual, [rr, cc], order=1, mode="nearest")
    point_residual = inputs.point_z - inputs.point_c0_m
    block_m = float(config["holdout_block_m"])
    bx = np.floor((inputs.point_x - inputs.bbox_en[0]) / block_m).astype(np.int64)
    by = np.floor((inputs.point_y - inputs.bbox_en[1]) / block_m).astype(np.int64)
    block_id = by * 10000 + bx
    result: dict[str, Any] = {}
    gates = config["gates"]
    for side, name in ((-1, "low"), (1, "high")):
        selected = inputs.point_holdout & (inputs.point_side == side)
        rows = []
        for identity in np.unique(block_id[selected]):
            block = selected & (block_id == identity)
            if int(np.count_nonzero(block)) < 5:
                continue
            rows.append((
                abs(float(np.median(point_residual[block]))),
                abs(float(np.median(point_residual[block] - predicted[block]))),
            ))
        c0_error = float(np.mean([row[0] for row in rows])) if rows else float("inf")
        candidate_error = float(np.mean([row[1] for row in rows])) if rows else float("inf")
        support = (
            int(np.count_nonzero(selected)) >= int(gates["minimum_holdout_points_per_side"])
            and len(rows) >= int(gates["minimum_holdout_blocks_per_side"])
        )
        result[name] = {
            "holdout_points": int(np.count_nonzero(selected)),
            "qualified_blocks": len(rows),
            "c0_block_median_mae_m": c0_error,
            "candidate_block_median_mae_m": candidate_error,
            "improvement_fraction": 1.0 - candidate_error / c0_error if c0_error > 0.0 else 0.0,
            "support_pass": support,
            "beats_c0": candidate_error < c0_error,
        }
    result["pass"] = all(row["support_pass"] and row["beats_c0"] for row in result.values())
    return result


def _transition_jump(inputs: StructuralInputs, reference: StructuralReference, residual: np.ndarray) -> dict[str, Any]:
    supported = reference.solve_confidence > 1e-6
    jumps: list[np.ndarray] = []
    for axis in (0, 1):
        edge = np.diff(supported.astype(np.int8), axis=axis) != 0
        hard = np.take(inputs.hard_solve, range(inputs.hard_solve.shape[axis] - 1), axis=axis) | np.take(inputs.hard_solve, range(1, inputs.hard_solve.shape[axis]), axis=axis)
        first = np.take(residual, range(residual.shape[axis] - 1), axis=axis)
        second = np.take(residual, range(1, residual.shape[axis]), axis=axis)
        jumps.append(np.abs(first - second)[edge & ~hard])
    values = np.concatenate([value for value in jumps if value.size]) if any(value.size for value in jumps) else np.zeros(1)
    return {
        "definition": "adjacent non-hard samples crossing the actual zero-support transition",
        "interfaces": int(values.size),
        "p99_m": float(np.percentile(values, 99)),
        "maximum_m": float(np.max(values)),
    }


def _write_npy(path: Path, value: np.ndarray) -> None:
    with path.open("wb") as destination:
        np.lib.format.write_array(destination, np.asarray(value, dtype="<f4", order="C"), allow_pickle=False)


def _build(inputs: StructuralInputs, config: dict[str, Any], include_holdout: bool) -> tuple[StructuralReference, ProjectionResult]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        reference = build_reference(inputs, config, include_holdout=include_holdout)
    return reference, solve_projection(inputs, reference, config)


def run(config_path: Path) -> Path:
    config_payload = config_path.read_bytes()
    config = _read_config(config_path)
    inputs = load_inputs(config)
    training_reference, training_projection = _build(inputs, config, include_holdout=False)
    holdout = _holdout_metrics(inputs, training_projection.residual_m, config)
    reference, projection = _build(inputs, config, include_holdout=True)
    residual = projection.residual_m
    reconstructed = inputs.c0_solve_m + residual
    collar = int(round(float(config["outer_zero_collar_m"]) / inputs.solve_pitch_m))
    collar_values = np.concatenate([
        residual[:collar].ravel(), residual[-collar:].ravel(),
        residual[:, :collar].ravel(), residual[:, -collar:].ravel(),
    ])
    hard = {
        "mapped_face_source_cells": int(config["gates"]["mapped_face_source_cells"]),
        "mapped_face_solve_nodes": int(np.count_nonzero(inputs.mapped_face_solve)),
        "mapped_face_residual_max_abs_m": float(np.max(np.abs(residual[inputs.mapped_face_solve]))),
        "all_hard_residual_max_abs_m": float(np.max(np.abs(residual[inputs.hard_solve]))),
        "outer_collar_residual_max_abs_m": float(np.max(np.abs(collar_values))),
    }
    hard["pass"] = all(value == 0.0 for key, value in hard.items() if key.endswith("_max_abs_m"))
    transition = _transition_jump(inputs, reference, residual)
    transition["pass"] = transition["maximum_m"] <= float(config["gates"]["maximum_boundary_jump_m"])
    valid = residual[~inputs.hard_solve]
    accepted = holdout["pass"] and hard["pass"] and transition["pass"]
    metrics = {
        "status": "development_structural_input_accepted" if accepted else "development_structural_input_rejected",
        "spatial": {"bbox_en": list(inputs.bbox_en), "pitch_m": inputs.solve_pitch_m, "shape": list(residual.shape)},
        "als": {
            "qualified_class2_points": len(inputs.point_x),
            "training_points": int(np.count_nonzero(~inputs.point_holdout)),
            "holdout_points": int(np.count_nonzero(inputs.point_holdout)),
            "reference_supported_cells": int(np.count_nonzero(reference.confidence > 0.0)),
            "physical_domain_cells": int(np.count_nonzero(reference.physical_domain)),
        },
        "holdout": holdout,
        "hard_laws": hard,
        "transition": transition,
        "network": {
            "break_cells": int(np.count_nonzero(reference.break_network)),
            "ridge_cells": int(np.count_nonzero(reference.ridge_network)),
            "valley_cells": int(np.count_nonzero(reference.valley_network)),
            "network_cells": int(np.count_nonzero(reference.network_strength)),
        },
        "residual": {
            "minimum_m": float(np.min(valid)), "maximum_m": float(np.max(valid)),
            "p01_m": float(np.percentile(valid, 1)), "p50_m": float(np.percentile(valid, 50)),
            "p99_m": float(np.percentile(valid, 99)),
            "changed_over_2cm_fraction": float(np.mean(np.abs(valid) > 0.02)),
        },
        "solver": {
            "family": "robust_multiscale_quadratic_als_reference_plus_second_order_tgv_projection",
            "iterations": projection.iterations,
            "last_ten_iteration_max_delta_m": projection.convergence_delta_m,
            "line_profile_target": False,
            "arbitrary_distance_corridor": False,
        },
        "credit": {
            "role": "corrected_structural_input_for_future_stage_2e_challenger",
            "added_process_morphology": False,
            "morphology_specialist": False,
            "production": False,
            "packing": False,
            "browser": False,
        },
    }

    with tempfile.TemporaryDirectory(prefix="laas-als-structural-") as temporary:
        staging = Path(temporary) / "artifact"
        staging.mkdir()
        arrays = {
            "reconstruction-f32.npy": reconstructed,
            "residual-f32.npy": residual,
            "reference-surface-f32.npy": reference.reference_surface_m,
            "reference-confidence-f32.npy": reference.confidence,
            "network-strength-f32.npy": reference.network_strength,
        }
        for name, value in arrays.items():
            _write_npy(staging / name, value)
        (staging / "metrics.json").write_bytes(_canonical(metrics) + b"\n")
        content = hashlib.sha256(hashlib.sha256(config_payload).digest())
        for name in sorted(arrays):
            content.update((staging / name).read_bytes())
        content.update((staging / "metrics.json").read_bytes())
        build_id = content.hexdigest()
        final = _ARTIFACT_ROOT / build_id
        if final.exists():
            return final / "manifest.json"

        interpretations = write_qa(staging / "qa", inputs, reference, residual)
        qa_rows = []
        for name, interpretation in interpretations.items():
            path = staging / "qa" / name
            with Image.open(path) as image:
                dimensions = list(image.size)
            qa_rows.append({**_identity(path, f"qa/{name}"), "dimensions": dimensions, "interpretation": interpretation})
        (staging / "qa" / "index.json").write_bytes(_canonical({"images": qa_rows}) + b"\n")
        source_files = [Path(__file__)] + [Path(__file__).with_name(name) for name in ("evidence.py", "reference.py", "model.py", "qa.py")]
        manifest = {
            "schema_version": "laas.coastal-escarpment-als-structural-artifact/1",
            "build_id": build_id, "state": metrics["status"],
            "config": _identity(config_path), "sources": inputs.source_identities,
            "implementation": [_identity(path) for path in source_files],
            "outputs": {
                **{name.removesuffix("-f32.npy").replace("-", "_"): _identity(staging / name, name) for name in arrays},
                "metrics": _identity(staging / "metrics.json", "metrics.json"),
                "qa_index": _identity(staging / "qa" / "index.json", "qa/index.json"),
            },
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "pid": os.getpid()},
        }
        (staging / "manifest.json").write_bytes(_canonical(manifest) + b"\n")
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final)
    return final / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
