"""Materialize one immutable coupled-process raised-bog v3 FLOAT attempt."""
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
SCHEMA = "laas.peat-raised-bog-process-v3-config/1"
REPOSITORY_ROOT = ASSET_GEN_ROOT.parent
OUTPUT_ROOT = (
    DATA_WORK / "microtopography/peat-raised-bog/process-v3/sha256"
)


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")


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


def _verify_input(declaration: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = REPOSITORY_ROOT / declaration["path"]
    identity = _identity(path)
    if identity["sha256"] != declaration["sha256"]:
        raise ValueError(f"bound v3 input changed: {path}: {identity['sha256']}")
    return path, identity


def _verify_evidence(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    paths: dict[str, Path] = {}
    for name, declaration in config["inputs"].items():
        path, identity = _verify_input(declaration)
        paths[name] = path
        identities.append(identity)
    evidence = json.loads(paths["edge_corrected_v2_evidence"].read_bytes())
    if (
        evidence.get("result") != "reject"
        or evidence.get("retained_orbits_full_eight_groups") != []
        or evidence.get("classes") != ["hollow", "lawn", "hummock"]
    ):
        raise ValueError("v2 rejection/evidence boundary changed")
    conditional = {
        row["class"]: row
        for row in evidence["group_balanced_class_conditional_quantiles"]
    }
    if (
        conditional["lawn"]["support_group_count"] != 8
        or conditional["hummock"]["support_group_count"] != 0
        or conditional["hollow"]["support_group_count"] != 1
        or conditional["lawn"]["hhdh_decision_margin_m_quantiles"][5] != 0.04
    ):
        raise ValueError("supported HuHoLa lawn margin or unsupported extreme classes changed")
    forms = json.loads(paths["moore_whole_form_capacity"].read_bytes())
    whole_forms = sum(int(group["whole_form_count"]) for group in forms["groups"])
    if whole_forms != 1 or len(forms["forms"]) != 1:
        raise ValueError("rejected Moore whole-form capacity changed")
    boundary = {
        "v2_result": "reject",
        "retained_typed_transition_orbits": [],
        "supported_class_conditional_margins": ["lawn"],
        "lawn_support_group_count": 8,
        "lawn_hhdh_decision_half_band_m": 0.04,
        "unsupported_hollow_support_group_count": 1,
        "unsupported_hummock_support_group_count": 0,
        "complete_moore_form_count": 1,
        "uses_extreme_class_prevalence_or_transitions": False,
        "estonia_transfer_claim": False,
    }
    return identities, boundary


def _crop_slices(
    outer: list[int], inner: list[int]
) -> tuple[slice, slice]:
    return (
        slice(outer[3] - inner[3], outer[3] - inner[1]),
        slice(inner[0] - outer[0], inner[2] - outer[0]),
    )


def _class_metrics(delta: np.ndarray, threshold: float) -> dict[str, Any]:
    hollow = delta < -threshold
    hummock = delta > threshold
    lawn = ~(hollow | hummock)
    total = delta.size
    direct = int(
        np.count_nonzero(hollow[:, :-1] & hummock[:, 1:])
        + np.count_nonzero(hummock[:, :-1] & hollow[:, 1:])
        + np.count_nonzero(hollow[:-1, :] & hummock[1:, :])
        + np.count_nonzero(hummock[:-1, :] & hollow[1:, :])
    )
    component_rows: dict[str, Any] = {}
    for name, selected in (("hollow", hollow), ("lawn", lawn), ("hummock", hummock)):
        labels, count = ndimage.label(selected)
        sizes = np.bincount(labels.ravel())[1:]
        component_rows[name] = {
            "component_count": int(count),
            "components_at_least_0p25m2": int(np.count_nonzero(sizes >= 4)),
            "median_area_m2": float(np.median(sizes) * 0.25**2) if sizes.size else 0.0,
            "maximum_area_m2": float(np.max(sizes) * 0.25**2) if sizes.size else 0.0,
        }
    return {
        "fractions": {
            "hollow": float(np.count_nonzero(hollow) / total),
            "lawn": float(np.count_nonzero(lawn) / total),
            "hummock": float(np.count_nonzero(hummock) / total),
        },
        "direct_hummock_hollow_adjacencies": direct,
        "components": component_rows,
    }


def _spectral_peak_fraction(delta: np.ndarray) -> float:
    window_y = np.hanning(delta.shape[0])
    window_x = np.hanning(delta.shape[1])
    transformed = np.fft.rfft2((delta - np.mean(delta)) * window_y[:, None] * window_x[None, :])
    power = np.abs(transformed) ** 2
    power[0, 0] = 0.0
    return float(np.max(power) / max(float(np.sum(power)), 1.0e-30))


def _seam_ratio(delta: np.ndarray) -> float:
    east = np.abs(np.diff(delta, axis=1))
    south = np.abs(np.diff(delta, axis=0))
    east_boundary = np.arange(1, delta.shape[1]) % 4 == 0
    south_boundary = np.arange(1, delta.shape[0]) % 4 == 0
    boundary = np.concatenate(
        (east[:, east_boundary].ravel(), south[south_boundary, :].ravel())
    )
    interior = np.concatenate(
        (east[:, ~east_boundary].ravel(), south[~south_boundary, :].ravel())
    )
    return float(np.mean(boundary) / max(float(np.mean(interior)), 1.0e-12))


def _metrics(
    config: dict[str, Any],
    height: np.ndarray,
    authority: np.ndarray,
    result: Result,
    halo_rows: slice,
    halo_cols: slice,
    core_rows: slice,
    core_cols: slice,
    evidence_boundary: dict[str, Any],
) -> dict[str, Any]:
    final_blocks = result.final_fine_m.reshape(height.shape[0], 4, height.shape[1], 4)
    final_means = np.mean(final_blocks, axis=(1, 3), dtype=np.float64)
    closure = final_means - height
    authority_fine = np.repeat(np.repeat(authority, 4, axis=0), 4, axis=1)
    outside_delta = result.delta_fine_m[~authority_fine]
    hr = slice(halo_rows.start * 4, halo_rows.stop * 4)
    hc = slice(halo_cols.start * 4, halo_cols.stop * 4)
    cr = slice(core_rows.start * 4, core_rows.stop * 4)
    cc = slice(core_cols.start * 4, core_cols.stop * 4)
    halo_delta = result.delta_fine_m[hr, hc]
    core_delta = result.delta_fine_m[cr, cc]
    classes = _class_metrics(
        core_delta, float(config["evidence_policy"]["huhola_lawn_decision_half_band_m"])
    )
    spectral = _spectral_peak_fraction(core_delta)
    seam = _seam_ratio(core_delta)
    process_mask = np.repeat(np.repeat(authority, 2, axis=0), 2, axis=1)
    process_change = float(
        np.sqrt(
            np.mean(
                (
                    result.process_state_05m[process_mask]
                    - result.initial_process_state_05m[process_mask]
                )
                ** 2
            )
        )
    )
    simplex = result.hummock_1m + result.lawn_1m + result.hollow_1m
    gates_config = config["gates"]
    fractions = classes["fractions"]
    low_lawn, high_lawn = gates_config["core_lawn_fraction_range"]
    low_relief, high_relief = gates_config["core_abs_delta_p99_m_range"]
    p99 = float(np.percentile(np.abs(core_delta), 99.0))
    gates = {
        "parent_mean_exact": {
            "passed": bool(np.array_equal(final_means, height)),
            "maximum_absolute_error_m": float(np.max(np.abs(closure))),
        },
        "hard_unknown_outside_unchanged": {
            "passed": bool(np.count_nonzero(outside_delta) == 0),
            "nonzero_fine_cells": int(np.count_nonzero(outside_delta)),
        },
        "finite_and_bounded_states": {
            "passed": bool(
                np.isfinite(result.final_fine_m).all()
                and np.min(result.water_depth_1m[authority]) >= 0.0
                and np.max(result.water_depth_1m[authority]) <= 1.0
                and np.max(np.abs(simplex[authority] - 1.0)) <= 1.0e-12
            ),
            "simplex_max_abs_error": float(np.max(np.abs(simplex[authority] - 1.0))),
        },
        "three_state_presence": {
            "passed": bool(
                fractions["hollow"] >= gates_config["minimum_core_fraction_per_extreme_class"]
                and fractions["hummock"] >= gates_config["minimum_core_fraction_per_extreme_class"]
                and low_lawn <= fractions["lawn"] <= high_lawn
            ),
            "fractions": fractions,
        },
        "lawn_separates_extremes": {
            "passed": classes["direct_hummock_hollow_adjacencies"]
            <= gates_config["maximum_direct_hummock_hollow_adjacencies"],
            "direct_adjacencies": classes["direct_hummock_hollow_adjacencies"],
        },
        "bounded_relief": {
            "passed": low_relief <= p99 <= high_relief,
            "core_abs_delta_p99_m": p99,
        },
        "nonperiodic_spectrum": {
            "passed": spectral <= gates_config["maximum_spectral_single_bin_power_fraction"],
            "spectral_peak_power_fraction": spectral,
        },
        "no_parent_grid_lock": {
            "passed": seam <= gates_config["maximum_parent_seam_to_interior_jump_ratio"],
            "seam_to_interior_mean_jump_ratio": seam,
        },
        "material_process_change": {
            "passed": process_change >= gates_config["minimum_process_change_rms"],
            "process_state_change_rms": process_change,
        },
    }
    all_passed = all(row["passed"] for row in gates.values())
    status = "float_qa_pending_visual_verdict" if all_passed else "rejected_fail_fast_gates"
    return {
        "status": status,
        "all_fail_fast_gates_passed": all_passed,
        "gates": gates,
        "evidence_boundary": evidence_boundary,
        "organization": {
            **classes,
            "spectral_peak_power_fraction": spectral,
            "parent_seam_to_interior_jump_ratio": seam,
            "directional_eligible_1m_fraction": float(
                np.count_nonzero(result.directional_eligibility_1m[authority] > 0.0)
                / np.count_nonzero(authority)
            ),
            "directional_strong_1m_fraction": float(
                np.count_nonzero(result.directional_eligibility_1m[authority] >= 0.5)
                / np.count_nonzero(authority)
            ),
        },
        "relief": {
            "halo_delta_min_m": float(np.min(halo_delta)),
            "halo_delta_max_m": float(np.max(halo_delta)),
            "halo_abs_delta_p95_m": float(np.percentile(np.abs(halo_delta), 95.0)),
            "halo_abs_delta_p99_m": float(np.percentile(np.abs(halo_delta), 99.0)),
            "core_abs_delta_p95_m": float(np.percentile(np.abs(core_delta), 95.0)),
            "core_abs_delta_p99_m": p99,
        },
        "dynamics": {
            "coarse_trace_rows": int(len(result.trace)),
            "final_mean_water_depth": float(np.mean(result.water_depth_1m[authority])),
            "final_mean_hummock_occupancy": float(np.mean(result.hummock_1m[authority])),
            "final_mean_lawn_occupancy": float(np.mean(result.lawn_1m[authority])),
            "final_mean_hollow_occupancy": float(np.mean(result.hollow_1m[authority])),
            "process_state_change_rms": process_change,
        },
        "authority": {
            "float_only": True,
            "measured_carrier_preserved": True,
            "v3_process_hypothesis_not_v2_replay": True,
            "estonia_calibration_or_production_owner": False,
            "packing_runtime_shader_material_or_format_authority": False,
        },
    }


def run(config_path: Path = CONFIG, output_root: Path = OUTPUT_ROOT) -> Path:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_bytes())
    if (
        config.get("schema_version") != SCHEMA
        or config.get("status") != "development_float_qa_only"
        or config.get("v3_deviation", {}).get("not_a_v2_replay") is not True
    ):
        raise ValueError("unsupported raised-bog process-v3 config")
    input_identities, evidence_boundary = _verify_evidence(config)
    measured_path = REPOSITORY_ROOT / config["inputs"]["measured_float"]["path"]
    with np.load(measured_path, allow_pickle=False) as measured:
        measured_metadata = json.loads(measured["metadata_json_u8"].tobytes())
        height = np.asarray(measured["whole_measured_height_1m"], dtype=np.float64)
        authority = np.asarray(measured["whole_authority_1m"], dtype=bool)
        drain = np.asarray(measured["whole_water_1m"] | measured["whole_road_1m"], dtype=bool)
    target = config["target"]
    if (
        measured_metadata["metrics"]["target"]["normalized_mire_identifier"]
        != target["normalized_mire_identifier"]
        or measured_metadata["metrics"]["target"]["clear_halo_bounds_epsg3301_m"]
        != target["clear_halo_bounds_epsg3301_m"]
    ):
        raise ValueError("measured FLOAT target binding changed")
    whole_bbox = measured_metadata["metrics"]["surface"]["whole_bbox_epsg3301_m"]
    halo_rows, halo_cols = _crop_slices(whole_bbox, target["clear_halo_bounds_epsg3301_m"])
    core_rows, core_cols = _crop_slices(whole_bbox, target["core_bounds_epsg3301_m"])
    if not np.all(authority[halo_rows, halo_cols]):
        raise ValueError("v3 frozen halo is no longer fully supported")

    result = simulate(height, authority, drain, config)
    metrics = _metrics(
        config,
        height,
        authority,
        result,
        halo_rows,
        halo_cols,
        core_rows,
        core_cols,
        evidence_boundary,
    )
    implementation = [
        _identity(Path(__file__)),
        _identity(Path(__file__).with_name("model.py")),
        _identity(Path(__file__).with_name("qa.py")),
    ]
    recipe = {
        "config": _identity(config_path),
        "implementation": implementation,
        "inputs": input_identities,
    }
    recipe_sha256 = hashlib.sha256(_canonical(recipe)).hexdigest()
    output = output_root / recipe_sha256
    if output.exists():
        raise FileExistsError(f"immutable process-v3 attempt exists: {output}")
    metadata = {
        "schema_version": "laas.peat-raised-bog-process-v3-float-artifact/1",
        "build_id": recipe_sha256,
        "status": metrics["status"],
        "recipe": recipe,
        "metrics": metrics,
        "v3_deviation": config["v3_deviation"],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pid": os.getpid(),
        },
    }
    hr = slice(halo_rows.start * 4, halo_rows.stop * 4)
    hc = slice(halo_cols.start * 4, halo_cols.stop * 4)
    cr = slice(core_rows.start * 4, core_rows.stop * 4)
    cc = slice(core_cols.start * 4, core_cols.stop * 4)
    with tempfile.TemporaryDirectory(prefix="laas-bog-process-v3-") as temporary:
        staging = Path(temporary) / "artifact"
        staging.mkdir()
        np.savez_compressed(
            staging / "process-v3-float.npz",
            metadata_json_u8=np.frombuffer(_canonical(metadata), dtype=np.uint8),
            water_depth_1m=result.water_depth_1m.astype(np.float32),
            hummock_occupancy_1m=result.hummock_1m.astype(np.float32),
            lawn_occupancy_1m=result.lawn_1m.astype(np.float32),
            hollow_occupancy_1m=result.hollow_1m.astype(np.float32),
            peat_state_1m=result.peat_state_1m.astype(np.float32),
            directional_eligibility_1m=result.directional_eligibility_1m.astype(np.float32),
            process_state_05m=result.process_state_05m.astype(np.float32),
            process_hummock_05m=result.process_hummock_05m.astype(np.float32),
            process_lawn_05m=result.process_lawn_05m.astype(np.float32),
            process_hollow_05m=result.process_hollow_05m.astype(np.float32),
            dynamics_trace=result.trace,
            halo_structural_height_025m=result.structural_fine_m[hr, hc],
            halo_process_height_025m=result.final_fine_m[hr, hc],
            halo_process_delta_025m=result.delta_fine_m[hr, hc],
            core_structural_height_025m=result.structural_fine_m[cr, cc],
            core_process_height_025m=result.final_fine_m[cr, cc],
            core_process_delta_025m=result.delta_fine_m[cr, cc],
        )
        write_qa(
            staging / "qa",
            height,
            authority,
            drain,
            result,
            halo_rows,
            halo_cols,
            core_rows,
            core_cols,
            metrics,
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
    with np.load(output / "process-v3-float.npz", allow_pickle=False) as artifact:
        metadata = json.loads(artifact["metadata_json_u8"].tobytes())
    print(output)
    print(json.dumps(metadata["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
