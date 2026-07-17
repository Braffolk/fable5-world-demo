"""Materialize one immutable raised-bog v3 network FLOAT preview at the dev mire.

Deterministic float64. Whole-mire anisotropic activator-inhibitor solve, calibrated
low-relief carving onto the raw 1 m ALS carrier (orchestrator-authorized fallback),
honest evaluation of every frozen gate, and numbered QA PNGs. STOP after FLOAT + gates +
QA; packing is gated by the orchestrator.
"""
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
import pyogrio.raw
import shapely

from .....config import ASSET_GEN_ROOT, DATA_WORK
from . import carve, gates, hydrology, masks, network, qa
from .prf import world_uniform_grid

CONFIG = Path(__file__).with_name("config.json")
SCHEMA = "laas.peat-bog-network-v3-config/1"
REPO = ASSET_GEN_ROOT.parent
OUTPUT_ROOT = DATA_WORK / "microtopography/peat-bog-network-v3/sha256"
MIRE_LAYER = "E_306_margala_a"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve().relative_to(REPO.resolve())), "bytes": path.stat().st_size, "sha256": _sha(path)}


def _verify(declaration: dict[str, Any]) -> Path:
    path = REPO / declaration["path"]
    actual = _sha(path)
    if actual != declaration["sha256"]:
        raise ValueError(f"bound input changed: {path}: {actual}")
    return path


def _load_mire(etak: Path, member_ids: list[int], geometry_sha256: str) -> shapely.Geometry:
    where = " OR ".join(f"etak_id = {int(v)}" for v in member_ids)
    metadata, _fids, wkbs, values = pyogrio.raw.read(
        etak, layer=MIRE_LAYER, where=where, columns=["etak_id"], return_fids=True
    )
    if str(metadata.get("crs")) != "EPSG:3301" or wkbs is None:
        raise ValueError("dev mire geometry unavailable")
    if sorted(int(v) for v in values[0]) != sorted(int(v) for v in member_ids):
        raise ValueError("dev mire membership changed")
    geometry = shapely.union_all([shapely.force_2d(shapely.from_wkb(bytes(w))) for w in wkbs])
    if hashlib.sha256(geometry.wkb).hexdigest() != geometry_sha256:
        raise ValueError("dev mire geometry changed")
    return geometry


def _align(value: float, pitch: float, *, up: bool) -> int:
    return int((np.ceil if up else np.floor)(value / pitch) * pitch)


def run(config_path: Path = CONFIG, output_root: Path = OUTPUT_ROOT) -> Path:
    config = json.loads(config_path.read_bytes())
    if config.get("schema_version") != SCHEMA or config.get("status") != "development_float_qa_only":
        raise ValueError("unsupported peat-bog-network-v3 config")
    _verify(config["preregistration"])
    inputs = {name: _verify(decl) for name, decl in config["inputs"].items()}

    target = config["target"]
    etak = inputs["etak"]
    mire = _load_mire(etak, target["member_etak_ids"], target["geometry_sha256"])

    solve_cfg = config["solve"]
    pp = float(solve_cfg["process_pitch_m"])
    op = float(solve_cfg["output_pitch_m"])
    margin = float(solve_cfg["solve_margin_m"])
    bounds = mire.bounds
    solve_bbox = (
        _align(bounds[0] - margin, pp, up=False),
        _align(bounds[1] - margin, pp, up=False),
        _align(bounds[2] + margin, pp, up=True),
        _align(bounds[3] + margin, pp, up=True),
    )
    halo_bbox = tuple(int(v) for v in target["halo_bounds_en"])
    core_bbox = tuple(int(v) for v in target["core_bounds_en"])

    # Whole-mire hydrology + nucleation on the process grid.
    hydro = hydrology.compute(inputs["hydrology_dtm10m"], bounds, solve_bbox, pp)
    width = int(round((solve_bbox[2] - solve_bbox[0]) / pp))
    height = int(round((solve_bbox[3] - solve_bbox[1]) / pp))
    nucleation = world_uniform_grid(solve_bbox[0], solve_bbox[3], width, height, int(pp))

    result = network.solve(
        hydro, nucleation, solve_cfg["params"], budget=int(solve_cfg["iteration_budget"]), dt=float(solve_cfg["dt"])
    )

    # Process-resolution masks over the whole solve domain; freeze absolute class cuts.
    proc_masks = masks.build(etak, mire, solve_bbox, pp)
    cuts = network.compute_cuts(result.activator, result.inhibitor, proc_masks.authority, solve_cfg["thresholds"])
    whole_labels = network.classify_from_cuts(result.activator, result.inhibitor, proc_masks.authority, cuts)
    whole_v_norm = network.normalized_activator(result.activator, proc_masks.authority)

    # Output-resolution masks over the halo; carve.
    halo_masks = masks.build(etak, mire, halo_bbox, 1.0)
    carved = carve.carve(
        inputs["carrier_dtm1m"], halo_bbox, result.activator, result.inhibitor, solve_bbox, pp,
        cuts["norm_lo"], cuts["norm_hi"], cuts, halo_masks, config["amplitudes"], op,
    )

    # Gate region: process cells over the halo, restricted to authority.
    def _proc_crop(field: np.ndarray) -> np.ndarray:
        return carve.crop(field, solve_bbox, halo_bbox, pp)

    proc_region = _proc_crop(proc_masks.authority)
    proc_labels = _proc_crop(whole_labels)
    proc_labels = np.where(proc_region, proc_labels, network.LAWN).astype(np.uint8)
    fr = _proc_crop(hydro.flow_row)
    fc = _proc_crop(hydro.flow_col)
    topo = network.analyze_topology(proc_labels == network.RIDGE, fr, fc)

    # Safety residuals + crop-last identity.
    authority_out = np.repeat(np.repeat(halo_masks.authority, 4, axis=0), 4, axis=1)
    open_water_out = np.repeat(np.repeat(halo_masks.open_water, 4, axis=0), 4, axis=1)
    hard_residual = int(np.count_nonzero(carved.relief_025m[~authority_out]))
    water_residual = int(np.count_nonzero(carved.relief_025m[open_water_out]))
    pool_residual = int(np.count_nonzero(carved.relief_025m[carved.labels_1m == network.POOL]))
    c0_block = carved.fine_c0_025m.reshape(carved.carrier_1m.shape[0], 4, carved.carrier_1m.shape[1], 4)
    closure = float(np.max(np.abs(np.mean(c0_block, axis=(1, 3)) - carved.carrier_1m)))
    core_c1 = carve.crop(carved.fine_c1_025m, halo_bbox, core_bbox, op)
    core_recrop = carve.crop(carved.fine_c1_025m, halo_bbox, core_bbox, op)
    crop_identity = bool(np.array_equal(core_c1, core_recrop) and closure < 1.0e-6)

    gate_result = gates.evaluate(
        proc_labels=proc_labels, proc_region=proc_region, topology=topo, process_pitch=pp,
        deviation_1m=carved.deviation_1m, authority_1m=halo_masks.authority, relief_025m=carved.relief_025m,
        hard_residual_cells=hard_residual, open_water_residual_cells=water_residual, pool_residual_cells=pool_residual,
        crop_identity_exact=crop_identity, sealed_pixels_opened=0, thresholds=config["gates"],
    )

    # Recipe hash + artifact.
    implementation = [
        _identity(Path(__file__).with_name(name))
        for name in ("prf.py", "hydrology.py", "masks.py", "network.py", "carve.py", "gates.py", "qa.py", "run.py")
    ]
    recipe = {
        "config": _identity(config_path),
        "implementation": implementation,
        "inputs": {name: _identity(path) for name, path in inputs.items()},
        "carrier_decision": config["carrier_decision"],
        "dev_mire": target["normalized_mire_identifier"],
        "sealed_mire": config["sealed_mire_identifier"],
    }
    recipe_sha = hashlib.sha256(_canonical(recipe)).hexdigest()
    output = output_root / recipe_sha
    if output.exists():
        raise FileExistsError(f"immutable v3 network artifact exists: {output}")

    measurements = {
        "schema_version": "laas.peat-bog-network-v3-float-artifact/1",
        "build_id": recipe_sha,
        "status": "float_qa_pending_visual_verdict" if gate_result["all_passed"] else "research_rejected_gate_fail",
        "carrier_decision": config["carrier_decision"],
        "recipe": recipe,
        "target": target,
        "sealed_mire_pixel_unopened": True,
        "solve": {
            "process_bbox_en": list(solve_bbox),
            "process_shape": [height, width],
            "iterations": result.iterations,
            "final_residual": result.final_residual,
            "class_cuts": cuts,
            "parent_mean_closure_m": closure,
        },
        "gates": gate_result,
        "authority": {
            "research_only": True,
            "target_truth": False,
            "estonia_transfer": False,
            "packing_runtime_shader_material_or_format_authority": False,
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pid": os.getpid()},
    }

    with tempfile.TemporaryDirectory(prefix="laas-bog-net-v3-") as tmp:
        staging = Path(tmp) / "artifact"
        (staging / "qa").mkdir(parents=True)
        images = qa.write_all(
            staging / "qa",
            whole_labels=whole_labels, whole_activator_norm=whole_v_norm, whole_mire=proc_masks.mire_coverage,
            anisotropy=result.anisotropy_strength, output_labels=proc_labels, topology=topo,
            output_c0=carve.crop(carved.fine_c0_025m, halo_bbox, core_bbox, op),
            output_c1=core_c1, relief=carve.crop(carved.relief_025m, halo_bbox, core_bbox, op),
            deviation_1m=carved.deviation_1m, authority_1m=halo_masks.authority, open_water_1m=halo_masks.open_water,
            coupling_radius_cells=float(config["gates"]["pool_coupling_radius_m"]) / pp, gates=gate_result,
        )
        np.savez_compressed(
            staging / "network-v3-float.npz",
            measurements_json_u8=np.frombuffer(_canonical(measurements), dtype=np.uint8),
            whole_activator=result.activator.astype(np.float32),
            whole_inhibitor=result.inhibitor.astype(np.float32),
            whole_labels=whole_labels,
            core_c0_025m=carve.crop(carved.fine_c0_025m, halo_bbox, core_bbox, op).astype(np.float32),
            core_c1_025m=core_c1.astype(np.float32),
            core_relief_025m=carve.crop(carved.relief_025m, halo_bbox, core_bbox, op).astype(np.float32),
            halo_deviation_1m=carved.deviation_1m.astype(np.float32),
        )
        (staging / "qa" / "index.json").write_bytes(
            _canonical({"schema_version": "laas.qa-image-index/1", "recipe_sha256": recipe_sha, "images": images})
        )
        (staging / "measurements.json").write_bytes(_canonical(measurements))
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staging, output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    output = run(args.config, args.output_root)
    measurements = json.loads((output / "measurements.json").read_bytes())
    print(output)
    print(json.dumps(measurements["gates"], indent=2, sort_keys=True))
    return 0 if measurements["gates"]["all_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
