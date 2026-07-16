"""Materialize one immutable whole-mire form-scale raised-bog FLOAT attempt."""
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
from scipy import ndimage

from .....config import ASSET_GEN_ROOT, DATA_WORK
from .model import Result, simulate
from .qa import write_qa


CONFIG = Path(__file__).with_name("config.json")
SCHEMA = "laas.peat-raised-bog-formscale-graph-config/1"
REPOSITORY_ROOT = ASSET_GEN_ROOT.parent
OUTPUT_ROOT = DATA_WORK / "microtopography/peat-raised-bog/formscale-graph/sha256"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(REPOSITORY_ROOT.resolve())),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _crop_slices(outer: list[int], inner: list[int]) -> tuple[slice, slice]:
    return (
        slice(outer[3] - inner[3], outer[3] - inner[1]),
        slice(inner[0] - outer[0], inner[2] - outer[0]),
    )


def _component_metrics(mask: np.ndarray, pitch_m: float = 1.0) -> dict[str, Any]:
    labels, count = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())[1:] * pitch_m**2
    return {
        "count": int(count),
        "median_area_m2": float(np.median(sizes)) if sizes.size else 0.0,
        "maximum_area_m2": float(np.max(sizes)) if sizes.size else 0.0,
    }


def _metrics(
    result: Result,
    authority: np.ndarray,
    hard: np.ndarray,
    water: np.ndarray,
    core_rows: slice,
    core_cols: slice,
) -> dict[str, Any]:
    blocks = result.final_fine_m.reshape(authority.shape[0], 4, authority.shape[1], 4)
    means = blocks.mean(axis=(1, 3), dtype=np.float64)
    closure = means - result.synthesized_parent_m
    fine_authority = np.repeat(np.repeat(authority, 4, axis=0), 4, axis=1)
    outside = result.fine_delta_m[~fine_authority]
    core_delta = result.parent_delta_m[core_rows, core_cols]
    classes = result.form_class[authority]
    names = ("lawn", "flark", "string", "hollow", "hummock", "pool_margin", "pool")
    fractions = {name: float(np.count_nonzero(classes == index) / classes.size) for index, name in enumerate(names)}
    pool_components = _component_metrics(result.form_class == 6)
    string_components = _component_metrics(result.string_strength > 0.50)
    return {
        "status": "float_qa_pending_holistic_visual_verdict",
        "integrity": {
            "closure_max_abs_m": float(np.max(np.abs(closure))),
            "outside_changed_fine_cells": int(np.count_nonzero(outside)),
            "hard_changed_parent_cells": int(np.count_nonzero(result.parent_delta_m[hard])),
            "water_changed_parent_cells": int(np.count_nonzero(result.parent_delta_m[water])),
            "finite": bool(np.isfinite(result.final_fine_m).all()),
        },
        "relief": {
            "parent_min_m": float(np.min(result.parent_delta_m[authority])),
            "parent_max_m": float(np.max(result.parent_delta_m[authority])),
            "parent_abs_max_m": float(np.max(np.abs(result.parent_delta_m[authority]))),
            "parent_abs_p95_m": float(np.percentile(np.abs(result.parent_delta_m[authority]), 95.0)),
            "core_abs_p95_m": float(np.percentile(np.abs(core_delta), 95.0)),
            "fine_abs_p99_m": float(np.percentile(np.abs(result.fine_delta_m[fine_authority]), 99.0)),
        },
        "organization": {
            "class_fractions": fractions,
            "pool_components": pool_components,
            "string_components": string_components,
            "travel_distance_p95_m": float(np.percentile(result.travel_distance_m[authority], 95.0)),
            "string_strong_cells": int(np.count_nonzero(result.string_strength > 0.50)),
            "flark_strong_cells": int(np.count_nonzero(result.flark_strength > 0.50)),
        },
        "authority": {
            "float_only": True,
            "whole_mire_solve_crop_last": True,
            "raw_dtm_is_observation_not_immutable_parent": True,
            "pack_runtime_format_or_production_authority": False,
        },
    }


def run(config_path: Path = CONFIG, output_root: Path = OUTPUT_ROOT) -> Path:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_bytes())
    if (
        config.get("schema_version") != SCHEMA
        or config.get("status") != "development_float_qa_only"
        or config.get("evidence_boundary", {}).get("not_process_v3_retune") is not True
    ):
        raise ValueError("unsupported form-scale graph config")
    declaration = config["inputs"]["measured_float"]
    measured_path = REPOSITORY_ROOT / declaration["path"]
    measured_identity = _identity(measured_path)
    if measured_identity["sha256"] != declaration["sha256"]:
        raise ValueError("bound measured FLOAT changed")
    with np.load(measured_path, allow_pickle=False) as measured:
        measured_metadata = json.loads(measured["metadata_json_u8"].tobytes())
        height = np.asarray(measured["whole_measured_height_1m"], dtype=np.float64)
        authority = np.asarray(measured["whole_authority_1m"], dtype=bool)
        hard = np.asarray(measured["whole_hard_exclusion_1m"], dtype=bool)
        water = np.asarray(measured["whole_water_1m"], dtype=bool)
    if measured_metadata["metrics"]["target"]["normalized_mire_identifier"] != config["target"]["normalized_mire_identifier"]:
        raise ValueError("Valgesoo target binding changed")
    whole_bbox = measured_metadata["metrics"]["surface"]["whole_bbox_epsg3301_m"]
    halo_rows, halo_cols = _crop_slices(whole_bbox, config["target"]["clear_halo_bounds_epsg3301_m"])
    core_rows, core_cols = _crop_slices(whole_bbox, config["target"]["core_bounds_epsg3301_m"])
    if not np.all(authority[halo_rows, halo_cols]):
        raise ValueError("frozen clear halo lost authority")

    result = simulate(height, authority, config)
    metrics = _metrics(result, authority, hard, water, core_rows, core_cols)
    implementation = [_identity(Path(__file__)), _identity(Path(__file__).with_name("model.py")), _identity(Path(__file__).with_name("qa.py"))]
    recipe = {"config": _identity(config_path), "implementation": implementation, "inputs": [measured_identity]}
    build_id = hashlib.sha256(_canonical(recipe)).hexdigest()
    output = output_root / build_id
    if output.exists():
        raise FileExistsError(f"immutable form-scale graph attempt exists: {output}")
    metadata = {
        "schema_version": "laas.peat-raised-bog-formscale-graph-float-artifact/1",
        "build_id": build_id,
        "status": metrics["status"],
        "recipe": recipe,
        "metrics": metrics,
        "evidence_boundary": config["evidence_boundary"],
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pid": os.getpid()},
    }
    hr, hc = slice(halo_rows.start * 4, halo_rows.stop * 4), slice(halo_cols.start * 4, halo_cols.stop * 4)
    cr, cc = slice(core_rows.start * 4, core_rows.stop * 4), slice(core_cols.start * 4, core_cols.stop * 4)
    with tempfile.TemporaryDirectory(prefix="laas-bog-formscale-graph-") as temporary:
        staging = Path(temporary) / "artifact"
        staging.mkdir()
        np.savez_compressed(
            staging / "formscale-graph-float.npz",
            metadata_json_u8=np.frombuffer(_canonical(metadata), dtype=np.uint8),
            whole_synthesized_parent_1m=result.synthesized_parent_m.astype(np.float32),
            whole_parent_delta_1m=result.parent_delta_m.astype(np.float32),
            whole_form_class_1m=result.form_class,
            whole_travel_distance_1m=result.travel_distance_m.astype(np.float32),
            whole_string_strength_1m=result.string_strength.astype(np.float32),
            whole_flark_strength_1m=result.flark_strength.astype(np.float32),
            whole_pool_strength_1m=result.pool_strength.astype(np.float32),
            halo_measured_height_025m=result.measured_fine_m[hr, hc].astype(np.float32),
            halo_final_height_025m=result.final_fine_m[hr, hc].astype(np.float32),
            halo_delta_025m=result.fine_delta_m[hr, hc].astype(np.float32),
            core_final_height_025m=result.final_fine_m[cr, cc].astype(np.float32),
            core_delta_025m=result.fine_delta_m[cr, cc].astype(np.float32),
        )
        write_qa(
            staging / "qa", result, authority, hard, water,
            halo_rows, halo_cols, core_rows, core_cols, metrics,
            {"build_id": build_id, "recipe_sha256": hashlib.sha256(_canonical(recipe)).hexdigest(), "input_sha256": measured_identity["sha256"]},
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staging, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    output = run(args.config, args.output_root)
    with np.load(output / "formscale-graph-float.npz", allow_pickle=False) as artifact:
        metadata = json.loads(artifact["metadata_json_u8"].tobytes())
    print(output)
    print(json.dumps(metadata["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
