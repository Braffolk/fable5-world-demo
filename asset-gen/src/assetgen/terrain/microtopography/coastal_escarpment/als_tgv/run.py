"""Materialize the bounded Development-A ALS/TGV float transaction."""
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
from PIL import Image
from scipy import ndimage

from .....config import ASSET_GEN_ROOT, DATA_WORK
from .evidence import load_evidence
from .model import solve_tgv
from .qa import write_qa

_SCHEMA = "laas.coastal-escarpment-als-tgv-config/1"
_ARTIFACT_ROOT = DATA_WORK / "microtopography" / "coastal-escarpment" / "als-tgv" / "sha256"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, logical_path: str | None = None) -> dict[str, Any]:
    if logical_path is None:
        logical_path = str(path.resolve().relative_to(ASSET_GEN_ROOT.parent))
    return {
        "path": logical_path,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _read_config(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != _SCHEMA:
        raise ValueError("unsupported ALS/TGV config")
    if document.get("status") != "development_only" or document.get("target_etak_id") != 1826743:
        raise ValueError("Development-A boundary differs")
    if document.get("output_chunks") != [[-2, 2436, 1492], [-2, 2437, 1492]]:
        raise ValueError("adjacent output pair differs")
    if document.get("shared_parent") != [-1, 609, 373]:
        raise ValueError("shared parent differs")
    laws = document.get("hard_laws", {})
    if laws.get("biala_geometry_or_parameters_forbidden") is not True or laws.get("taevaskoda_sentinel_opened") is not False:
        raise ValueError("research/sentinel boundary differs")
    return document


def _holdout_metrics(evidence, residual: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    rr = (evidence.bbox_en[3] - evidence.point_y) / evidence.pitch_m
    cc = (evidence.point_x - evidence.bbox_en[0]) / evidence.pitch_m
    predicted = ndimage.map_coordinates(residual, [rr, cc], order=1, mode="nearest")
    block_m = float(config["holdout_block_m"])
    bx = np.floor((evidence.point_x - evidence.bbox_en[0]) / block_m).astype(np.int64)
    by = np.floor((evidence.point_y - evidence.bbox_en[1]) / block_m).astype(np.int64)
    block_id = by * 10000 + bx
    result: dict[str, Any] = {}
    minimum_points = int(config["qualification"]["minimum_holdout_points_per_side"])
    for side, name in ((-1, "low"), (1, "high")):
        selected = evidence.point_holdout & (evidence.point_side == side)
        rows = []
        for identity in np.unique(block_id[selected]):
            block = selected & (block_id == identity)
            if int(np.count_nonzero(block)) < 5:
                continue
            rows.append(
                (
                    abs(float(np.median(evidence.point_residual_m[block]))),
                    abs(float(np.median(evidence.point_residual_m[block] - predicted[block]))),
                    int(np.count_nonzero(block)),
                )
            )
        c0_error = float(np.mean([row[0] for row in rows])) if rows else float("inf")
        candidate_error = float(np.mean([row[1] for row in rows])) if rows else float("inf")
        result[name] = {
            "holdout_points": int(np.count_nonzero(selected)),
            "qualified_blocks": len(rows),
            "c0_block_median_mae_m": c0_error,
            "candidate_block_median_mae_m": candidate_error,
            "improvement_fraction": 1.0 - candidate_error / c0_error if c0_error > 0.0 else 0.0,
            "support_pass": int(np.count_nonzero(selected)) >= minimum_points and len(rows) >= 8,
            "beats_c0": candidate_error < c0_error,
        }
    result["pass"] = all(row["support_pass"] and row["beats_c0"] for row in result.values())
    return result


def _write_npy(path: Path, value: np.ndarray) -> None:
    with path.open("wb") as destination:
        np.lib.format.write_array(destination, np.asarray(value, dtype="<f4", order="C"), allow_pickle=False)


def run(config_path: Path) -> Path:
    config_payload = config_path.read_bytes()
    config = _read_config(config_path)
    evidence = load_evidence(config)
    result = solve_tgv(evidence, config)
    residual = result.residual_m
    reconstructed = evidence.c0_m + residual

    holdout = _holdout_metrics(evidence, residual, config)
    collar = int(round(float(config["outer_zero_collar_m"]) / evidence.pitch_m))
    collar_values = np.concatenate(
        [residual[:collar].ravel(), residual[-collar:].ravel(), residual[:, :collar].ravel(), residual[:, -collar:].ravel()]
    )
    hard_metrics = {
        "mapped_face_source_cells": int(config["hard_laws"]["mapped_face_cells_expected"]),
        "mapped_face_solve_nodes": int(np.count_nonzero(evidence.mapped_face)),
        "mapped_face_residual_max_abs_m": float(np.max(np.abs(residual[evidence.mapped_face]))),
        "all_hard_residual_max_abs_m": float(np.max(np.abs(residual[evidence.hard_zero]))),
        "outer_collar_residual_max_abs_m": float(np.max(np.abs(collar_values))),
    }
    hard_metrics["pass"] = all(value == 0.0 for key, value in hard_metrics.items() if key.endswith("_max_abs_m"))
    active_values = residual[evidence.active]
    metrics = {
        "status": "development_float_candidate_accepted" if holdout["pass"] and hard_metrics["pass"] else "development_float_candidate_rejected",
        "spatial": {
            "bbox_en": list(evidence.bbox_en),
            "pitch_m": evidence.pitch_m,
            "shape": list(residual.shape),
            "active_nodes": int(np.count_nonzero(evidence.active)),
            "hard_zero_nodes": int(np.count_nonzero(evidence.hard_zero)),
        },
        "als": {
            "qualified_raw_class2_points": len(evidence.point_x),
            "training_points": int(np.count_nonzero(~evidence.point_holdout)),
            "holdout_points": int(np.count_nonzero(evidence.point_holdout)),
            "training_nodes": int(np.count_nonzero(evidence.train_mask)),
            "holdout_nodes": int(np.count_nonzero(evidence.holdout_mask)),
        },
        "holdout": holdout,
        "hard_laws": hard_metrics,
        "residual": {
            "minimum_m": float(np.min(active_values)),
            "maximum_m": float(np.max(active_values)),
            "p01_m": float(np.percentile(active_values, 1)),
            "p50_m": float(np.percentile(active_values, 50)),
            "p99_m": float(np.percentile(active_values, 99)),
            "changed_over_2cm_fraction": float(np.mean(np.abs(active_values) > 0.02)),
        },
        "solver": {
            "family": "second_order_tgv_primal_dual",
            "iterations": result.iterations,
            "last_ten_iteration_max_delta_m": result.convergence_delta_m,
            "raw_point_data_metric": "Huber",
            "image_role": "masked_anisotropic_regularization_only",
            "line_profile_target": False,
        },
        "credit": {
            "development_a_only": True,
            "independent_transfer": False,
            "production": False,
            "packing": False,
            "browser": False,
            "sandstone_wall": False,
        },
    }

    with tempfile.TemporaryDirectory(prefix="laas-als-tgv-") as temporary:
        staging = Path(temporary) / "artifact"
        staging.mkdir()
        _write_npy(staging / "reconstruction-f32.npy", reconstructed)
        _write_npy(staging / "residual-f32.npy", residual)
        (staging / "metrics.json").write_bytes(_canonical(metrics) + b"\n")
        content = hashlib.sha256()
        content.update(hashlib.sha256(config_payload).digest())
        content.update((staging / "reconstruction-f32.npy").read_bytes())
        content.update((staging / "residual-f32.npy").read_bytes())
        content.update((staging / "metrics.json").read_bytes())
        build_id = content.hexdigest()
        final = _ARTIFACT_ROOT / build_id
        if final.exists():
            shutil.rmtree(staging)
            return final / "manifest.json"

        interpretations = write_qa(staging / "qa", evidence, residual)
        qa_rows = []
        for name, interpretation in interpretations.items():
            path = staging / "qa" / name
            with Image.open(path) as image:
                dimensions = list(image.size)
            qa_rows.append({**_identity(path, f"qa/{name}"), "dimensions": dimensions, "interpretation": interpretation})
        (staging / "qa" / "index.json").write_bytes(_canonical({"images": qa_rows}) + b"\n")

        source_files = [Path(__file__), Path(__file__).with_name("evidence.py"), Path(__file__).with_name("model.py"), Path(__file__).with_name("qa.py")]
        manifest = {
            "schema_version": "laas.coastal-escarpment-als-tgv-artifact/1",
            "build_id": build_id,
            "state": metrics["status"],
            "config": _identity(config_path),
            "sources": evidence.source_identities,
            "implementation": [_identity(path) for path in source_files],
            "outputs": {
                "reconstruction": _identity(staging / "reconstruction-f32.npy", "reconstruction-f32.npy"),
                "residual": _identity(staging / "residual-f32.npy", "residual-f32.npy"),
                "metrics": _identity(staging / "metrics.json", "metrics.json"),
                "qa_index": _identity(staging / "qa" / "index.json", "qa/index.json"),
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pid": os.getpid(),
            },
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
