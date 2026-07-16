"""Train and materialize one bounded coarse-structure-first pixel diffusion run."""
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
from ...coastal_escarpment.als_structural.evidence import load_inputs
from ..single_field_ml.contracts import parent_mean_metrics, smoothstep, soft_parent_projection
from .data import prepare_source_stages, resize, terrain_conditions
from .model import DiffusionModel, sample, train
from .qa import write_qa

_SCHEMA = "laas.erodible-slope-domain-diffusion-config/1"
_ROOT = DATA_WORK / "microtopography" / "erodible-slope" / "domain-diffusion-development-a" / "sha256"
_TRAINING_CACHE = DATA_WORK / "microtopography" / "erodible-slope" / "domain-diffusion-development-a" / "training-cache"


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


def _device(torch: Any) -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _model_record(model: DiffusionModel) -> dict[str, Any]:
    return {
        "condition_mean": model.condition_mean.tolist(),
        "condition_std": model.condition_std.tolist(),
        "output_scale_m": model.output_scale_m,
        "losses": model.losses,
        "parameter_count": model.parameter_count,
        "state": model.state,
    }


def _model_from_record(record: dict[str, Any]) -> DiffusionModel:
    return DiffusionModel(
        state=record["state"],
        condition_mean=np.asarray(record["condition_mean"], dtype=np.float32),
        condition_std=np.asarray(record["condition_std"], dtype=np.float32),
        output_scale_m=float(record["output_scale_m"]),
        losses=[float(value) for value in record["losses"]],
        parameter_count=int(record["parameter_count"]),
    )


def _holdout_metrics(truth: np.ndarray, base: np.ndarray, generated: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    truth_delta = truth - base
    generated_delta = generated - base
    tgy, tgx = np.gradient(truth_delta)
    ggy, ggx = np.gradient(generated_delta)
    return {
        "truth_delta_rms_m": float(np.sqrt(np.mean(truth_delta[valid] ** 2))),
        "generated_delta_rms_m": float(np.sqrt(np.mean(generated_delta[valid] ** 2))),
        "truth_gradient_rms": float(np.sqrt(np.mean((tgx[valid] ** 2 + tgy[valid] ** 2)))),
        "generated_gradient_rms": float(np.sqrt(np.mean((ggx[valid] ** 2 + ggy[valid] ** 2)))),
        "signed_delta_correlation": float(np.corrcoef(truth_delta[valid], generated_delta[valid])[0, 1]),
    }


def run(config_path: Path) -> Path:
    import torch

    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config.get("schema_version") != _SCHEMA or config.get("status") != "development_r0_only":
        raise ValueError("unsupported domain-diffusion config")
    paths = {name: _bound(row) for name, row in config["inputs"].items()}
    structural_config = json.loads(paths["structural_config"].read_text(encoding="utf-8"))
    inputs = load_inputs(structural_config)
    network = np.load(paths["network"], allow_pickle=False).astype(np.float64)
    with np.load(paths["biala_candidate"], allow_pickle=False) as source:
        source_z = source["vendor_class2_z_mean_m"].astype(np.float64)
        count = source["vendor_class2_count"]
        vertical_range = source["vendor_class2_z_range_m"].astype(np.float64)
    source_valid = np.isfinite(source_z) & (count >= config["source_support"]["minimum_returns"]) & np.isfinite(vertical_range) & (vertical_range <= config["source_support"]["maximum_range_m"])
    stage1_train, stage1_hold, stage2_train, stage2_hold, source_surfaces = prepare_source_stages(
        source_z,
        source_valid,
        source_pitch_m=float(config["source_pitch_m"]),
        holdout_start_xy=tuple(config["geographic_holdout_start_xy_cells"]),
        minimum_support=float(config["source_support"]["minimum_window_support"]),
    )
    device = _device(torch)
    cache_identity = hashlib.sha256(config_bytes + Path(__file__).with_name("model.py").read_bytes() + Path(__file__).with_name("data.py").read_bytes()).hexdigest()
    cache_path = _TRAINING_CACHE / f"{cache_identity}.pt"
    started = time.perf_counter()
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        stage1_model = _model_from_record(cached["stage1"])
        stage2_model = _model_from_record(cached["stage2"])
        training_seconds = float(cached["training_seconds"])
    else:
        stage1_model = train(stage1_train.samples, stage1_train.conditions, stage1_train.masks, iterations=config["stage1"]["iterations"], batch_size=config["training"]["batch_size"], width=config["training"]["width"], learning_rate=config["training"]["learning_rate"], seed=config["stage1"]["training_seed"], device=device)
        stage2_model = train(stage2_train.samples, stage2_train.conditions, stage2_train.masks, iterations=config["stage2"]["iterations"], batch_size=config["training"]["batch_size"], width=config["training"]["width"], learning_rate=config["training"]["learning_rate"], seed=config["stage2"]["training_seed"], device=device)
        training_seconds = time.perf_counter() - started
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_cache = cache_path.with_suffix(".part")
        torch.save({"stage1": _model_record(stage1_model), "stage2": _model_record(stage2_model), "training_seconds": training_seconds}, temporary_cache, _use_new_zipfile_serialization=False)
        os.replace(temporary_cache, cache_path)

    target_shape_2m = ((inputs.c0_solve_m.shape[0] - 1) // 8 + 1, (inputs.c0_solve_m.shape[1] - 1) // 8 + 1)
    target_c0_2m = resize(inputs.c0_solve_m, target_shape_2m, 1)
    target_network_2m = resize(network, target_shape_2m, 1)
    condition_2m = terrain_conditions(target_c0_2m, target_network_2m, 2.0)
    macro_2m = sample(stage1_model, condition_2m, width=config["training"]["width"], seed=config["stage1"]["generation_seed"], ddim_steps=config["training"]["ddim_steps"], device=device)
    macro_2m = np.clip(macro_2m, -config["stage1"]["maximum_relief_m"], config["stage1"]["maximum_relief_m"])

    target_shape_1m = ((inputs.c0_solve_m.shape[0] - 1) // 4 + 1, (inputs.c0_solve_m.shape[1] - 1) // 4 + 1)
    target_c0_1m = resize(inputs.c0_solve_m, target_shape_1m, 1)
    target_network_1m = resize(network, target_shape_1m, 1)
    macro_1m = resize(macro_2m, target_shape_1m, 3)
    condition_1m = np.concatenate((terrain_conditions(target_c0_1m, target_network_1m, 1.0), macro_1m[None].astype(np.float32)))
    fine_1m = sample(stage2_model, condition_1m, width=config["training"]["width"], seed=config["stage2"]["generation_seed"], ddim_steps=config["training"]["ddim_steps"], device=device)
    fine_1m -= ndimage.gaussian_filter(fine_1m, config["stage2"]["maximum_owned_wavelength_m"] / 2.0, mode="nearest")
    fine_1m = np.clip(fine_1m, -config["stage2"]["maximum_relief_m"], config["stage2"]["maximum_relief_m"])

    network_fine = resize(network, inputs.c0_solve_m.shape, 1)
    conditions_fine = terrain_conditions(inputs.c0_solve_m.astype(np.float64), network_fine, inputs.solve_pitch_m)
    distance = ndimage.distance_transform_edt(~inputs.hard_solve) * inputs.solve_pitch_m
    slope = conditions_fine[2]
    opportunity = smoothstep((slope - config["ownership"]["minimum_slope"]) / config["ownership"]["slope_ramp"])
    opportunity = np.maximum(opportunity, config["ownership"]["network_weight"] * np.clip(network_fine, 0.0, 1.0))
    opportunity *= smoothstep(distance / config["ownership"]["hard_transition_m"])
    opportunity[inputs.hard_solve] = 0.0

    macro = resize(macro_2m, inputs.c0_solve_m.shape, 3) * opportunity
    fine = resize(fine_1m, inputs.c0_solve_m.shape, 3) * opportunity
    macro[inputs.hard_solve] = 0.0
    fine[inputs.hard_solve] = 0.0
    delta = macro + fine
    delta = soft_parent_projection(delta, inputs.hard_solve, int(round(1.0 / inputs.solve_pitch_m)), config["ownership"]["soft_parent_strength"])
    delta = np.clip(delta, -config["ownership"]["maximum_total_relief_m"], config["ownership"]["maximum_total_relief_m"])
    delta[inputs.hard_solve] = 0.0
    c0 = inputs.c0_solve_m.astype(np.float64)
    stage1_surface = c0 + macro
    final = c0 + delta

    hold1_condition = stage1_hold.conditions[0]
    hold1_macro = sample(stage1_model, hold1_condition, width=config["training"]["width"], seed=config["stage1"]["holdout_seed"], ddim_steps=config["training"]["ddim_steps"], device=device)
    hold1_macro = np.clip(hold1_macro, -config["stage1"]["maximum_relief_m"], config["stage1"]["maximum_relief_m"])
    hold2_condition = stage2_hold.conditions[0].copy()
    hold2_origin_x, hold2_origin_y = stage2_hold.origins[0]
    hold1_origin_x, hold1_origin_y = stage1_hold.origins[0]
    macro_offset_x = (hold2_origin_x - 2 * hold1_origin_x) // 2
    macro_offset_y = (hold2_origin_y - 2 * hold1_origin_y) // 2
    macro_cells = hold2_condition.shape[-1] // 2
    hold1_crop = hold1_macro[macro_offset_y : macro_offset_y + macro_cells, macro_offset_x : macro_offset_x + macro_cells]
    hold2_condition[-1] = resize(hold1_crop, hold2_condition.shape[-2:], 3)
    hold2_fine = sample(stage2_model, hold2_condition, width=config["training"]["width"], seed=config["stage2"]["holdout_seed"], ddim_steps=config["training"]["ddim_steps"], device=device)
    hold2_fine -= ndimage.gaussian_filter(hold2_fine, config["stage2"]["maximum_owned_wavelength_m"] / 2.0, mode="nearest")
    hold2_fine = np.clip(hold2_fine, -config["stage2"]["maximum_relief_m"], config["stage2"]["maximum_relief_m"])
    hold_base = source_surfaces["base_1m"]
    x0, y0 = stage2_hold.origins[0]
    size = stage2_hold.samples.shape[-1]
    hold_base = hold_base[y0 : y0 + size, x0 : x0 + size]
    hold_truth = hold_base + stage2_hold.conditions[0, -1] + stage2_hold.samples[0, 0]
    hold_stage1 = hold_base + resize(hold1_crop, hold_base.shape, 3)
    hold_final = hold_stage1 + hold2_fine
    hold_valid = stage2_hold.masks[0]

    valid = ~inputs.hard_solve
    collar = int(round(float(structural_config["outer_zero_collar_m"]) / inputs.solve_pitch_m))
    collar_values = np.concatenate((delta[:collar].ravel(), delta[-collar:].ravel(), delta[:, :collar].ravel(), delta[:, -collar:].ravel()))
    metrics = {
        "state": "development_r0_visual_pending",
        "architecture": {"family": "metric_pixel_diffusion", "stage1": "2 m domain organization", "stage2": "1-2 m within-form innovation only", "natural_image_prior": False, "latent_rgb_vae": False},
        "training": {"device": device, "torch": torch.__version__, "seconds": training_seconds, "stage1_windows": len(stage1_train.origins), "stage2_windows": len(stage2_train.origins), "stage1_origins": stage1_train.origins, "stage2_origin_count": len(stage2_train.origins), "stage1_losses": stage1_model.losses, "stage2_losses": stage2_model.losses, "parameter_count_each": stage1_model.parameter_count},
        "geographic_holdout": {"same_epoch_same_site": True, "start_xy_source_cells": config["geographic_holdout_start_xy_cells"], "stage1_origins": stage1_hold.origins, "stage2_origin_count": len(stage2_hold.origins), **_holdout_metrics(hold_truth, hold_base, hold_final, hold_valid)},
        "hard_laws": {"all_hard_max_abs_m": float(np.max(np.abs(delta[inputs.hard_solve]))), "mapped_face_max_abs_m": float(np.max(np.abs(delta[inputs.mapped_face_solve]))), "outer_collar_max_abs_m": float(np.max(np.abs(collar_values))), "pass": bool(np.max(np.abs(delta[inputs.hard_solve])) == 0.0 and np.max(np.abs(collar_values)) == 0.0)},
        "stage_ownership": {"macro_rms_m": float(np.sqrt(np.mean(macro[valid] ** 2))), "fine_rms_m": float(np.sqrt(np.mean(fine[valid] ** 2))), "fine_to_macro_rms_ratio": float(np.sqrt(np.mean(fine[valid] ** 2)) / max(np.sqrt(np.mean(macro[valid] ** 2)), 1.0e-9)), "stage2_lowpass_rms_m": float(np.sqrt(np.mean(ndimage.gaussian_filter(fine, 4.0 / inputs.solve_pitch_m)[valid] ** 2)))},
        "relief": {"minimum_m": float(np.min(delta[valid])), "maximum_m": float(np.max(delta[valid])), "abs_p95_m": float(np.percentile(np.abs(delta[valid]), 95)), "changed_over_2cm_fraction": float(np.mean(np.abs(delta[valid]) > 0.02))},
        "soft_parent_consistency": {"exact_raw_dtm_projection": False, **parent_mean_metrics(delta, inputs.hard_solve, int(round(1.0 / inputs.solve_pitch_m)))},
        "evidence_boundary": {"biala_role": "same-site R0 Baltic erodible mineral scarp weak surface", "source_support_floor_m": 0.5, "generated_stage1_pitch_m": 2.0, "generated_stage2_pitch_m": 1.0, "artifact_pitch_role": "interpolated onto accepted target lattice", "estonia_predictive_transfer": False, "production": False, "packing": False},
    }

    with tempfile.TemporaryDirectory(prefix="laas-domain-diffusion-") as temporary:
        staging = Path(temporary) / "artifact"
        staging.mkdir()
        arrays = {"c0-f32.npy": c0, "stage1-c1-f32.npy": stage1_surface, "final-c1-f32.npy": final, "stage1-macro-delta-f32.npy": macro, "stage2-fine-delta-f32.npy": fine, "opportunity-f32.npy": opportunity}
        for name, value in arrays.items():
            _write_npy(staging / name, value)
        checkpoint = {"schema_version": "laas.erodible-slope-domain-diffusion-checkpoint/1", "config_sha256": hashlib.sha256(config_bytes).hexdigest(), "stage1": _model_record(stage1_model), "stage2": _model_record(stage2_model)}
        torch.save(checkpoint, staging / "checkpoint.pt", _use_new_zipfile_serialization=False)
        (staging / "metrics.json").write_bytes(_canonical(metrics))
        content = hashlib.sha256(hashlib.sha256(config_bytes).digest())
        for name in sorted(arrays):
            content.update((staging / name).read_bytes())
        content.update((staging / "checkpoint.pt").read_bytes())
        content.update((staging / "metrics.json").read_bytes())
        build_id = content.hexdigest()
        final_root = _ROOT / build_id
        if final_root.exists():
            return final_root / "manifest.json"
        interpretations = write_qa(staging / "qa", c0=c0, stage1=stage1_surface, final=final, macro=macro, fine=fine, conditions=conditions_fine, opportunity=opportunity, hard=inputs.hard_solve, holdout_base=hold_base, holdout_truth=hold_truth, holdout_stage1=hold_stage1, holdout_final=hold_final, pitch_m=inputs.solve_pitch_m)
        images = []
        for name, interpretation in interpretations.items():
            path = staging / "qa" / name
            with Image.open(path) as image:
                dimensions = list(image.size)
            images.append({**_identity(path, f"qa/{name}"), "dimensions": dimensions, "interpretation": interpretation})
        (staging / "qa" / "index.json").write_bytes(_canonical({"images": images}))
        implementation = [Path(__file__), Path(__file__).with_name("model.py"), Path(__file__).with_name("data.py"), Path(__file__).with_name("qa.py")]
        outputs = {name.removesuffix(".npy").replace("-", "_"): _identity(staging / name, name) for name in arrays}
        outputs.update({"checkpoint": _identity(staging / "checkpoint.pt", "checkpoint.pt"), "metrics": _identity(staging / "metrics.json", "metrics.json"), "qa_index": _identity(staging / "qa" / "index.json", "qa/index.json")})
        manifest = {"schema_version": "laas.erodible-slope-domain-diffusion-artifact/1", "build_id": build_id, "state": metrics["state"], "config": _identity(config_path), "inputs": {name: _identity(path) for name, path in paths.items()}, "implementation": [_identity(path) for path in implementation], "outputs": outputs, "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "device": device, "pid": os.getpid()}}
        (staging / "manifest.json").write_bytes(_canonical(manifest))
        final_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final_root)
    return final_root / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
