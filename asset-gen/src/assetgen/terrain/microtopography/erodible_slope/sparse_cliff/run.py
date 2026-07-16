"""Materialize one immutable Development-A typed sparse cliff artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from .....config import ASSET_GEN_ROOT, DATA_WORK
from ....microtopography.coastal_escarpment.als_structural.evidence import load_inputs
from .model import TYPE_NAMES, synthesize
from .qa import write_qa

_SCHEMA = "laas.erodible-slope-sparse-cliff-config/1"
_ROOT = DATA_WORK / "microtopography" / "erodible-slope" / "sparse-cliff-development-a" / "sha256"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bound(row: dict[str, Any]) -> Path:
    path = ASSET_GEN_ROOT.parent / row["path"]
    if path.stat().st_size != row["bytes"] or _sha(path) != row["sha256"]:
        raise ValueError(f"bound input differs: {path}")
    return path


def _identity(path: Path, logical: str | None = None) -> dict[str, Any]:
    return {"path": logical or str(path.resolve().relative_to(ASSET_GEN_ROOT.parent)), "bytes": path.stat().st_size, "sha256": _sha(path)}


def _write_npy(path: Path, value: np.ndarray, dtype: str) -> None:
    with path.open("wb") as target:
        np.lib.format.write_array(target, np.asarray(value, dtype=dtype, order="C"), allow_pickle=False)


def _one_metre_error(delta: np.ndarray, factor: int) -> float:
    rows = delta.shape[0] // factor * factor
    cols = delta.shape[1] // factor * factor
    means = delta[:rows, :cols].reshape(rows // factor, factor, cols // factor, factor).mean(axis=(1, 3))
    return float(np.max(np.abs(means)))


def run(config_path: Path) -> Path:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config.get("schema_version") != _SCHEMA or config.get("status") != "development_r0_only":
        raise ValueError("unsupported sparse-cliff config")
    expected_laws = {
        "absolute_c0_authority": True,
        "mapped_non_heightfield_face_stays_c0": True,
        "outer_collar_stays_c0": True,
        "protected_hard_and_water_stay_c0": True,
        "one_metre_parent_mean_stays_c0": True,
        "taevaskoda_or_ood_opened": False,
    }
    if config["hard_laws"] != expected_laws:
        raise ValueError("hard-law boundary differs")
    paths = {name: _bound(row) for name, row in config["inputs"].items()}
    structural_config = json.loads(paths["structural_config"].read_text(encoding="utf-8"))
    inputs = load_inputs(structural_config)
    network = np.load(paths["network"], allow_pickle=False)
    with np.load(paths["biala_capacity"], allow_pickle=False) as source:
        offset = int(round(float(config["source_core_offset_m"]) / float(config["model"]["source_pitch_m"])))
        cells = int(round(float(config["source_core_m"]) / float(config["model"]["source_pitch_m"])))
        core = np.s_[offset : offset + cells, offset : offset + cells]
        source_reconstruction = source["reconstruction_m"][core].astype(np.float64)
        source_macro = source["macro_component_m"][core].astype(np.float64)
        source_support = source["direct_support"][core].astype(bool)
    median = float(np.nanmedian(source_reconstruction[source_support]))
    source_reconstruction = np.nan_to_num(source_reconstruction, nan=median)
    source_macro = np.nan_to_num(source_macro, nan=0.0)

    result = synthesize(
        inputs.c0_solve_m,
        inputs.hard_solve,
        network,
        source_reconstruction,
        source_macro,
        source_support,
        inputs.bbox_en,
        config["model"],
        inputs.solve_pitch_m,
    )
    delta = result.delta_m.astype(np.float64)
    hard_max = float(np.max(np.abs(delta[inputs.hard_solve])))
    face_max = float(np.max(np.abs(delta[inputs.mapped_face_solve])))
    collar = int(round(float(structural_config["outer_zero_collar_m"]) / inputs.solve_pitch_m))
    collar_values = np.concatenate((delta[:collar].ravel(), delta[-collar:].ravel(), delta[:, :collar].ravel(), delta[:, -collar:].ravel()))
    closure = _one_metre_error(delta, int(round(1.0 / inputs.solve_pitch_m)))
    valid = ~inputs.hard_solve
    distance = np.asarray(ndimage.distance_transform_edt(valid) * inputs.solve_pitch_m)
    transition = valid & (distance > 0.0) & (distance <= float(config["model"]["transition_m"]))
    interior = valid & (distance >= 2.0 * float(config["model"]["transition_m"]))
    gradient = np.maximum(np.abs(np.diff(delta, axis=0, prepend=delta[:1])), np.abs(np.diff(delta, axis=1, prepend=delta[:, :1])))
    transition_ratio = float(np.percentile(gradient[transition], 95) / max(np.percentile(gradient[interior], 95), 1.0e-9))
    source_ids = [int(value) for value in np.unique(result.source_attribution[result.source_attribution >= 0])]
    placement_counts = {name: sum(row["kind"] == name for row in result.placements) for name in TYPE_NAMES}
    metrics = {
        "state": "development_r0_visual_pending",
        "spatial": {"bbox_en": list(inputs.bbox_en), "pitch_m": inputs.solve_pitch_m, "shape": list(delta.shape), "solve": "one_connected_absolute_master"},
        "support_scale_claim": {"source_pitch_m": 0.5, "paired_low_pitch_m": 1.0, "artifact_pitch_m": inputs.solve_pitch_m, "artifact_pitch_role": "interpolation_and_existing_C0_lattice_only", "truth_below_0_5_m": False},
        "hard_laws": {"all_hard_max_abs_m": hard_max, "mapped_face_max_abs_m": face_max, "outer_collar_max_abs_m": float(np.max(np.abs(collar_values))), "one_metre_mean_error_m": closure, "pass": hard_max == 0.0 and face_max == 0.0 and closure <= 1.0e-7},
        "transition": {"p95_gradient_over_interior_ratio": transition_ratio},
        "relief": {"minimum_m": float(np.min(delta[valid])), "maximum_m": float(np.max(delta[valid])), "abs_p50_m": float(np.percentile(np.abs(delta[valid]), 50)), "abs_p95_m": float(np.percentile(np.abs(delta[valid]), 95)), "changed_over_2cm_fraction": float(np.mean(np.abs(delta[valid]) > 0.02))},
        "dictionary": {"mechanism": "typed coupled 1m-low/0.5m-high sparsity-one OMP with irregular Wendland-C2 partition", "atom_counts": result.atom_counts, "placement_counts": placement_counts, "site_count": len(result.placements), "distinct_source_centers": len(source_ids)},
        "evidence_boundary": {"biala_role": "R0 Baltic coastal sand/till analogue source-domain capacity only", "estonia_predictive_transfer": False, "packing": False, "production": False, "six_centimeter_support": False},
    }

    with tempfile.TemporaryDirectory(prefix="laas-sparse-cliff-") as temporary:
        staging = Path(temporary) / "artifact"
        staging.mkdir()
        arrays: dict[str, tuple[np.ndarray, str]] = {
            "c0-f32.npy": (inputs.c0_solve_m, "<f4"),
            "c1-f32.npy": (result.c1_m, "<f4"),
            "delta-f32.npy": (result.delta_m, "<f4"),
            "ownership-u8.npy": (result.ownership, "u1"),
            "source-attribution-i32.npy": (result.source_attribution, "<i4"),
            "site-attribution-i32.npy": (result.site_attribution, "<i4"),
        }
        for name, (value, dtype) in arrays.items():
            _write_npy(staging / name, value, dtype)
        (staging / "metrics.json").write_bytes(_canonical(metrics))
        (staging / "placements.json").write_bytes(_canonical({"placements": result.placements}))
        content = hashlib.sha256(hashlib.sha256(config_bytes).digest())
        for name in sorted(arrays):
            content.update((staging / name).read_bytes())
        content.update((staging / "metrics.json").read_bytes())
        content.update((staging / "placements.json").read_bytes())
        build_id = content.hexdigest()
        final = _ROOT / build_id
        if final.exists():
            return final / "manifest.json"
        interpretations = write_qa(staging / "qa", inputs.c0_solve_m, inputs.hard_solve, result, inputs.solve_pitch_m)
        images = []
        for name, interpretation in interpretations.items():
            path = staging / "qa" / name
            with Image.open(path) as image:
                dimensions = list(image.size)
            images.append({**_identity(path, f"qa/{name}"), "dimensions": dimensions, "interpretation": interpretation})
        (staging / "qa" / "index.json").write_bytes(_canonical({"images": images}))
        sources = [Path(__file__), Path(__file__).with_name("model.py"), Path(__file__).with_name("qa.py")]
        outputs = {name.replace(".npy", "").replace("-", "_"): _identity(staging / name, name) for name in arrays}
        outputs.update({"metrics": _identity(staging / "metrics.json", "metrics.json"), "placements": _identity(staging / "placements.json", "placements.json"), "qa_index": _identity(staging / "qa" / "index.json", "qa/index.json")})
        manifest = {
            "schema_version": "laas.erodible-slope-sparse-cliff-artifact/1",
            "build_id": build_id,
            "state": metrics["state"],
            "config": _identity(config_path),
            "inputs": {name: _identity(path) for name, path in paths.items()},
            "implementation": [_identity(path) for path in sources],
            "outputs": outputs,
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "pid": os.getpid()},
        }
        (staging / "manifest.json").write_bytes(_canonical(manifest))
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
