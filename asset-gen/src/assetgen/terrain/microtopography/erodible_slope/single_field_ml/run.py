"""Train and materialize one Development-A learned single-field attempt."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from .....config import ASSET_GEN_ROOT, DATA_WORK
from ....microtopography.coastal_escarpment.als_structural.evidence import load_inputs
from .contracts import parent_mean_metrics, smoothstep, soft_parent_projection
from .model import generate, terrain_conditions, train_pyramid
from .qa import write_qa

_SCHEMA = "laas.erodible-slope-single-field-ml-config/1"
_ROOT = DATA_WORK / "microtopography" / "erodible-slope" / "single-field-ml-development-a" / "sha256"


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


def _write_npy(path: Path, value: np.ndarray, dtype: str = "<f4") -> None:
    with path.open("wb") as target:
        np.lib.format.write_array(target, np.asarray(value, dtype=dtype, order="C"), allow_pickle=False)


def _fill_nearest(value: np.ndarray, valid: np.ndarray) -> np.ndarray:
    indices = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return np.where(valid, value, value[tuple(indices)])


def _resize(value: np.ndarray, shape: tuple[int, int], order: int = 1) -> np.ndarray:
    return ndimage.zoom(value, (shape[0] / value.shape[0], shape[1] / value.shape[1]), order=order)[:shape[0], :shape[1]]


def _patches(value: np.ndarray, size: int, stride: int) -> tuple[np.ndarray, tuple[int, int]]:
    windows = np.lib.stride_tricks.sliding_window_view(value, (size, size))[::stride, ::stride]
    shape = windows.shape[:2]
    matrix = windows.reshape(-1, size * size).astype(np.float64)
    matrix -= matrix.mean(axis=1, keepdims=True)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1.0e-9)
    return matrix, shape


def _anti_copy(source: np.ndarray, generated: np.ndarray, pitch_m: float) -> tuple[dict[str, float], np.ndarray]:
    size = int(round(8.0 / pitch_m))
    stride = max(2, size // 2)
    source_patches, _ = _patches(source, size, stride)
    generated_patches, map_shape = _patches(generated, size, stride)
    nearest = np.max(generated_patches @ source_patches.T, axis=1)
    nearest_map = _resize(nearest.reshape(map_shape), generated.shape, order=1)
    centered = generated - ndimage.gaussian_filter(generated, 2.0 / pitch_m)
    centered -= float(np.mean(centered))
    variance = float(np.sum(centered * centered))
    axis_correlations = []
    for lag in range(2, min(33, generated.shape[0] // 3, generated.shape[1] // 3)):
        axis_correlations.append(abs(float(np.sum(centered[:, :-lag] * centered[:, lag:]) / max(variance, 1.0e-9))))
        axis_correlations.append(abs(float(np.sum(centered[:-lag] * centered[lag:]) / max(variance, 1.0e-9))))
    checker = np.indices(generated.shape).sum(axis=0) & 1
    checker_score = abs(float(np.mean(centered * (2.0 * checker - 1.0)))) / max(float(np.std(centered)), 1.0e-9)
    gy, gx = np.gradient(generated)
    energy_ratio = max(float(np.mean(gx * gx)), float(np.mean(gy * gy))) / max(min(float(np.mean(gx * gx)), float(np.mean(gy * gy))), 1.0e-12)
    return {
        "nearest_source_correlation_p50": float(np.percentile(nearest, 50)),
        "nearest_source_correlation_p95": float(np.percentile(nearest, 95)),
        "near_copy_fraction_over_0_98": float(np.mean(nearest > 0.98)),
        "maximum_axis_periodic_correlation_lag_2_32": max(axis_correlations),
        "checker_phase_score": checker_score,
        "gradient_axis_energy_ratio": energy_ratio,
    }, nearest_map


def _device(torch: Any) -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run(config_path: Path) -> Path:
    import torch

    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config.get("schema_version") != _SCHEMA or config.get("status") != "development_r0_only":
        raise ValueError("unsupported learned single-field config")
    expected_laws = {
        "absolute_c0_authority": True,
        "crop_after_whole_domain_generation": True,
        "mapped_non_heightfield_face_stays_c0": True,
        "outer_collar_stays_c0": True,
        "protected_hard_and_water_stay_c0": True,
        "soft_one_metre_consistency_not_exact_raw_dtm": True,
        "taevaskoda_or_ood_opened": False,
    }
    if config["hard_laws"] != expected_laws:
        raise ValueError("hard-law boundary differs")
    paths = {name: _bound(row) for name, row in config["inputs"].items()}
    structural_config = json.loads(paths["structural_config"].read_text(encoding="utf-8"))
    inputs = load_inputs(structural_config)
    network = np.load(paths["network"], allow_pickle=False).astype(np.float64)
    source_pitch = float(config["source_pitch_m"])
    with np.load(paths["biala_capacity"], allow_pickle=False) as source_npz:
        offset = int(round(float(config["source_core_offset_m"]) / source_pitch))
        cells = int(round(float(config["source_core_m"]) / source_pitch))
        core = np.s_[offset : offset + cells, offset : offset + cells]
        reconstruction = source_npz["reconstruction_m"][core].astype(np.float64)
        support = source_npz["direct_support"][core].astype(bool)
    reconstruction = _fill_nearest(reconstruction, support)
    source_residual = reconstruction - ndimage.gaussian_filter(reconstruction, float(config["model"]["source_detrend_sigma_m"]) / source_pitch)
    source_condition = terrain_conditions(reconstruction, None, source_pitch)

    device = _device(torch)
    started = time.perf_counter()
    trained = train_pyramid(source_residual, source_condition, config["model"], device)
    training_seconds = time.perf_counter() - started

    model_pitch = float(config["model_pitch_m"])
    stride = int(round(model_pitch / inputs.solve_pitch_m))
    target_shape = ((inputs.c0_solve_m.shape[0] - 1) // stride + 1, (inputs.c0_solve_m.shape[1] - 1) // stride + 1)
    target_c0 = inputs.c0_solve_m[::stride, ::stride].astype(np.float64)
    target_hard = inputs.hard_solve[::stride, ::stride]
    target_network = _resize(network, target_shape)
    target_condition = terrain_conditions(target_c0, target_network, model_pitch)
    generated = generate(trained, target_condition, target_shape, config["model"], device)
    generated -= ndimage.gaussian_filter(
        generated,
        float(config["model"]["maximum_form_support_m"]) / model_pitch,
    )
    coupling = float(config["model"]["signed_condition_coupling"])
    incision_condition = np.maximum(target_condition[2], target_condition[3])
    toe_condition = np.maximum(
        target_condition[1],
        ndimage.gaussian_filter(target_condition[2], 2.0 / model_pitch) * (1.0 - target_condition[0]),
    )
    generated = (
        np.maximum(generated, 0.0) * ((1.0 - coupling) + coupling * toe_condition)
        - np.maximum(-generated, 0.0) * ((1.0 - coupling) + coupling * incision_condition)
    )
    valid_model = ~target_hard
    distance_model = ndimage.distance_transform_edt(valid_model) * model_pitch
    opportunity = smoothstep((target_condition[0] - float(config["model"]["minimum_slope"])) / float(config["model"]["slope_ramp"]))
    opportunity *= 0.35 + 0.65 * target_condition[3]
    opportunity *= smoothstep(distance_model / float(config["model"]["hard_transition_m"]))
    opportunity[target_hard] = 0.0
    generated *= opportunity
    p95 = float(np.percentile(np.abs(generated[valid_model]), 95.0))
    generated *= float(config["model"]["target_abs_p95_m"]) / max(p95, 1.0e-8)
    generated = np.clip(generated, -float(config["model"]["maximum_relief_m"]), float(config["model"]["maximum_relief_m"]))

    delta = _resize(generated, inputs.c0_solve_m.shape, order=3)
    fine_distance = ndimage.distance_transform_edt(~inputs.hard_solve) * inputs.solve_pitch_m
    delta *= smoothstep(fine_distance / float(config["model"]["hard_transition_m"]))
    delta = soft_parent_projection(delta, inputs.hard_solve, int(round(1.0 / inputs.solve_pitch_m)), float(config["model"]["soft_parent_strength"]))
    delta = np.clip(delta, -float(config["model"]["maximum_relief_m"]), float(config["model"]["maximum_relief_m"]))
    delta[inputs.hard_solve] = 0.0
    c1 = inputs.c0_solve_m.astype(np.float64) + delta
    c1[inputs.hard_solve] = inputs.c0_solve_m[inputs.hard_solve]
    conditions_fine = np.stack([_resize(channel, delta.shape) for channel in target_condition])
    opportunity_fine = _resize(opportunity, delta.shape)
    source_for_copy = source_residual
    generated_for_copy = generated
    anti_copy, nearest_map_model = _anti_copy(source_for_copy, generated_for_copy, model_pitch)
    nearest_map = _resize(nearest_map_model, delta.shape)
    parent_metrics = parent_mean_metrics(delta, inputs.hard_solve, int(round(1.0 / inputs.solve_pitch_m)))
    collar = int(round(float(structural_config["outer_zero_collar_m"]) / inputs.solve_pitch_m))
    collar_values = np.concatenate((delta[:collar].ravel(), delta[-collar:].ravel(), delta[:, :collar].ravel(), delta[:, -collar:].ravel()))
    valid = ~inputs.hard_solve
    metrics = {
        "state": "development_r0_visual_pending",
        "training": {"device": device, "seconds": training_seconds, "torch": torch.__version__, "parameter_count": trained.parameter_count, "losses": trained.losses},
        "spatial": {"bbox_en": list(inputs.bbox_en), "pitch_m": inputs.solve_pitch_m, "model_pitch_m": model_pitch, "shape": list(delta.shape), "solve": "one_full_physical_master_crop_last"},
        "support_scale_claim": {"source_pitch_m": source_pitch, "generated_support_floor_m": source_pitch, "artifact_pitch_m": inputs.solve_pitch_m, "artifact_pitch_role": "existing_C0_lattice_and_interpolation_only", "truth_below_0_5_m": False, "six_centimeter_truth": False},
        "hard_laws": {"all_hard_max_abs_m": float(np.max(np.abs(delta[inputs.hard_solve]))), "mapped_face_max_abs_m": float(np.max(np.abs(delta[inputs.mapped_face_solve]))), "outer_collar_max_abs_m": float(np.max(np.abs(collar_values))), "pass": bool(np.max(np.abs(delta[inputs.hard_solve])) == 0.0 and np.max(np.abs(collar_values)) == 0.0)},
        "soft_parent_consistency": {"projection_strength": float(config["model"]["soft_parent_strength"]), **parent_metrics, "exact_raw_dtm_projection": False},
        "relief": {"minimum_m": float(np.min(delta[valid])), "maximum_m": float(np.max(delta[valid])), "abs_p50_m": float(np.percentile(np.abs(delta[valid]), 50)), "abs_p95_m": float(np.percentile(np.abs(delta[valid]), 95)), "changed_over_2cm_fraction": float(np.mean(np.abs(delta[valid]) > .02)), "positive_volume_m3": float(np.sum(np.maximum(delta, 0.0)) * inputs.solve_pitch_m**2), "negative_volume_m3": float(np.sum(np.maximum(-delta, 0.0)) * inputs.solve_pitch_m**2)},
        "anti_copy": anti_copy,
        "evidence_boundary": {"biala_role": "R0 Baltic erodible mineral scarp single-field representation challenger only", "estonia_predictive_transfer": False, "production": False, "packing": False},
    }

    with tempfile.TemporaryDirectory(prefix="laas-single-field-ml-") as temporary:
        staging = Path(temporary) / "artifact"
        staging.mkdir()
        arrays = {"c0-f32.npy": inputs.c0_solve_m, "c1-f32.npy": c1, "delta-f32.npy": delta, "opportunity-f32.npy": opportunity_fine, "nearest-source-correlation-f32.npy": nearest_map}
        for name, value in arrays.items():
            _write_npy(staging / name, value)
        checkpoint = {"schema_version": "laas.erodible-slope-single-field-checkpoint/1", "config_sha256": hashlib.sha256(config_bytes).hexdigest(), "normalization_m": trained.normalization_m, "scales": trained.scales, "noise_amplitudes": trained.noise_amplitudes, "states": trained.states, "training_seed": config["model"]["training_seed"]}
        torch.save(checkpoint, staging / "checkpoint.pt", _use_new_zipfile_serialization=False)
        (staging / "metrics.json").write_bytes(_canonical(metrics))
        content = hashlib.sha256(hashlib.sha256(config_bytes).digest())
        for name in sorted(arrays):
            content.update((staging / name).read_bytes())
        content.update((staging / "checkpoint.pt").read_bytes())
        content.update((staging / "metrics.json").read_bytes())
        build_id = content.hexdigest()
        final = _ROOT / build_id
        if final.exists():
            return final / "manifest.json"
        interpretations = write_qa(staging / "qa", inputs.c0_solve_m, c1, delta, inputs.hard_solve, conditions_fine, opportunity_fine, nearest_map, anti_copy, inputs.solve_pitch_m)
        images = []
        for name, interpretation in interpretations.items():
            path = staging / "qa" / name
            with Image.open(path) as image:
                dimensions = list(image.size)
            images.append({**_identity(path, f"qa/{name}"), "dimensions": dimensions, "interpretation": interpretation})
        (staging / "qa" / "index.json").write_bytes(_canonical({"images": images}))
        implementation = [Path(__file__), Path(__file__).with_name("model.py"), Path(__file__).with_name("contracts.py"), Path(__file__).with_name("qa.py")]
        outputs = {name.removesuffix(".npy").replace("-", "_"): _identity(staging / name, name) for name in arrays}
        outputs.update({"checkpoint": _identity(staging / "checkpoint.pt", "checkpoint.pt"), "metrics": _identity(staging / "metrics.json", "metrics.json"), "qa_index": _identity(staging / "qa" / "index.json", "qa/index.json")})
        manifest = {"schema_version": "laas.erodible-slope-single-field-ml-artifact/1", "build_id": build_id, "state": metrics["state"], "config": _identity(config_path), "inputs": {name: _identity(path) for name, path in paths.items()}, "implementation": [_identity(path) for path in implementation], "outputs": outputs, "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "device": device, "pid": os.getpid()}}
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
