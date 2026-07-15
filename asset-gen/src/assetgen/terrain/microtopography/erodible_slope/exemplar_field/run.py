"""Materialize one immutable Development-A whole-domain macro challenger."""
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

from .....config import ASSET_GEN_ROOT, DATA_WORK
from ....microtopography.coastal_escarpment.als_structural.evidence import load_inputs
from .model import synthesize
from .qa import write_qa

_SCHEMA = "laas.erodible-slope-exemplar-field-config/1"
_ROOT = DATA_WORK / "microtopography" / "erodible-slope" / "exemplar-field-development-a" / "sha256"


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
    return {
        "path": logical or str(path.resolve().relative_to(ASSET_GEN_ROOT.parent)),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _write_npy(path: Path, value: np.ndarray) -> None:
    with path.open("wb") as target:
        np.lib.format.write_array(target, np.asarray(value, dtype="<f4", order="C"), allow_pickle=False)


def run(config_path: Path) -> Path:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config.get("schema_version") != _SCHEMA or config.get("status") != "development_r0_only":
        raise ValueError("unsupported exemplar-field config")
    laws = config["hard_laws"]
    if laws != {
        "absolute_c0_authority": True,
        "crop_after_whole_domain_solve": True,
        "mapped_non_heightfield_face_stays_c0": True,
        "outer_collar_stays_c0": True,
        "protected_hard_and_water_stay_c0": True,
        "taevaskoda_or_ood_opened": False,
    }:
        raise ValueError("hard-law boundary differs")

    paths = {name: _bound(row) for name, row in config["inputs"].items()}
    structural_config = json.loads(paths["structural_config"].read_text(encoding="utf-8"))
    inputs = load_inputs(structural_config)
    network = np.load(paths["network"], allow_pickle=False)
    with np.load(paths["biala_capacity"], allow_pickle=False) as source:
        offset = int(round(float(config["source_core_offset_m"]) / float(config["source_pitch_m"])))
        cells = int(round(float(config["source_core_m"]) / float(config["source_pitch_m"])))
        core = np.s_[offset : offset + cells, offset : offset + cells]
        source_base = source["base_m"][core].astype(np.float64)
        source_macro = source["macro_component_m"][core].astype(np.float64)
        source_support = source["direct_support"][core].astype(bool)
    median = float(np.nanmedian(source_base[source_support]))
    source_base = np.nan_to_num(source_base, nan=median)
    source_macro = np.nan_to_num(source_macro, nan=0.0)

    result = synthesize(
        inputs.c0_solve_m,
        inputs.hard_solve,
        network,
        source_base,
        source_macro,
        source_support,
        config["model"],
        inputs.solve_pitch_m,
    )
    delta = result.delta_m
    hard_max = float(np.max(np.abs(delta[inputs.hard_solve])))
    collar = int(round(float(structural_config["outer_zero_collar_m"]) / inputs.solve_pitch_m))
    collar_values = np.concatenate([delta[:collar].ravel(), delta[-collar:].ravel(), delta[:, :collar].ravel(), delta[:, -collar:].ravel()])
    valid = ~inputs.hard_solve
    erosion_volume = float(np.sum(result.erosion_m) * inputs.solve_pitch_m**2)
    deposition_volume = float(np.sum(result.deposition_m) * inputs.solve_pitch_m**2)
    net_volume = float(np.sum(delta) * inputs.solve_pitch_m**2)
    maximum_neighbor = max(float(np.max(np.abs(np.diff(delta, axis=0)))), float(np.max(np.abs(np.diff(delta, axis=1)))))
    metrics = {
        "state": "development_r0_visual_rejected",
        "spatial": {"bbox_en": list(inputs.bbox_en), "pitch_m": inputs.solve_pitch_m, "shape": list(delta.shape), "solve": "one_connected_absolute_master_crop_last"},
        "hard_laws": {
            "all_hard_max_abs_m": hard_max,
            "mapped_face_max_abs_m": float(np.max(np.abs(delta[inputs.mapped_face_solve]))),
            "outer_collar_max_abs_m": float(np.max(np.abs(collar_values))),
            "pass": hard_max == 0.0 and float(np.max(np.abs(collar_values))) == 0.0,
        },
        "relief": {
            "minimum_m": float(np.min(delta[valid])), "maximum_m": float(np.max(delta[valid])),
            "abs_p50_m": float(np.percentile(np.abs(delta[valid]), 50)),
            "abs_p95_m": float(np.percentile(np.abs(delta[valid]), 95)),
            "changed_over_2cm_fraction": float(np.mean(np.abs(delta[valid]) > 0.02)),
            "maximum_adjacent_delta_change_m": maximum_neighbor,
        },
        "conservation": {
            "erosion_m3": erosion_volume, "deposition_m3": deposition_volume,
            "net_m3": net_volume, "relative_error": abs(net_volume) / max(erosion_volume, 1.0e-12),
        },
        "forms": {str(identity): int(np.count_nonzero(result.ownership == identity)) for identity in range(1, 6)},
        "registration": {"orientation": result.source_orientation, "displacement_p95_m": result.registration_displacement_p95_m},
        "visual_decision": {
            "accepted": False,
            "worth_packing": False,
            "judgment": "common-light macro remains too soft; long paired right-bank bands and broad lower-slope lobes remain visibly generator-shaped, while seep relief is negligible",
            "attempt_budget_exhausted": True,
        },
        "evidence_boundary": {
            "biala_role": "R0 Baltic coastal sand/till analogue 2D representation capacity only",
            "substrate_process_match": "erodible unconsolidated mineral scarp with runoff incision, shallow mass wasting, seep influence, and colluvial toe",
            "estonia_predictive_transfer": False, "production": False, "packing": False,
        },
    }

    with tempfile.TemporaryDirectory(prefix="laas-exemplar-field-") as temporary:
        staging = Path(temporary) / "artifact"
        staging.mkdir()
        arrays = {
            "c0-f32.npy": inputs.c0_solve_m,
            "c1-f32.npy": result.c1_m,
            "delta-f32.npy": result.delta_m,
            "ownership-f32.npy": result.ownership.astype(np.float32),
        }
        for name, value in arrays.items():
            _write_npy(staging / name, value)
        (staging / "metrics.json").write_bytes(_canonical(metrics))
        content = hashlib.sha256(hashlib.sha256(config_bytes).digest())
        for name in sorted(arrays):
            content.update((staging / name).read_bytes())
        content.update((staging / "metrics.json").read_bytes())
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
        manifest = {
            "schema_version": "laas.erodible-slope-exemplar-field-artifact/1", "build_id": build_id,
            "state": metrics["state"], "config": _identity(config_path),
            "inputs": {name: _identity(path) for name, path in paths.items()},
            "implementation": [_identity(path) for path in sources],
            "outputs": {**{name.removesuffix("-f32.npy").replace("-", "_"): _identity(staging / name, name) for name in arrays}, "metrics": _identity(staging / "metrics.json", "metrics.json"), "qa_index": _identity(staging / "qa" / "index.json", "qa/index.json")},
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
