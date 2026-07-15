"""Bounded source-domain Biala Gora coastal-escarpment capacity screen.

This is deliberately not a transferable generator. It tests whether a masked,
two-dimensional multiscale representation can retain connected cliff/process
morphology without the ribs and caps produced by earlier line-based solvers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from assetgen.config import ASSET_GEN_ROOT, DATA_WORK
from assetgen.terrain.microtopography.coastal_escarpment.qa import render_capacity_qa

_SCHEMA = "laas.biala-coastal-escarpment-capacity/1"
_CONFIG_SCHEMA = "laas.biala-coastal-escarpment-capacity-config/1"
_QA_SCHEMA = "laas.biala-coastal-escarpment-capacity-qa/1"
_ARTIFACT_ROOT = DATA_WORK / "terrain" / "microtopography" / "biala-coastal-escarpment-capacity" / "sha256"


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def _normalized_gaussian(z: np.ndarray, valid: np.ndarray, sigma_cells: float) -> np.ndarray:
    weights = ndimage.gaussian_filter(valid.astype(np.float64), sigma_cells, mode="nearest")
    values = ndimage.gaussian_filter(np.where(valid, z, 0.0), sigma_cells, mode="nearest")
    return np.divide(values, weights, out=np.full(z.shape, np.nan), where=weights > 1.0e-6)


def _load_config(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    config = json.loads(raw)
    if config.get("schema_version") != _CONFIG_SCHEMA:
        raise ValueError("unsupported Biala coastal-escarpment capacity config")
    return config, raw


def build_capacity_screen(config_path: Path, output_root: Path | None = None) -> Path:
    config, config_bytes = _load_config(config_path)
    source_manifest = ASSET_GEN_ROOT.parent / config["source"]["manifest"]
    source_manifest_bytes = source_manifest.read_bytes()
    source_record = json.loads(source_manifest_bytes)
    if source_record.get("build_id") != config["source"]["build_id"] or source_record.get("status") != "complete":
        raise ValueError("retained Biala source binding changed")
    artifact_record = source_record["artifacts"]["candidate_npz"]
    source_npz = source_manifest.parent / artifact_record["path"]
    if _sha256(source_npz) != artifact_record["sha256"]:
        raise ValueError("retained Biala candidate bytes changed")

    implementation_paths = (Path(__file__), Path(__file__).with_name("qa.py"))
    recipe = {
        "schema_version": _SCHEMA,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "source": {
            "build_id": source_record["build_id"],
            "manifest_sha256": hashlib.sha256(source_manifest_bytes).hexdigest(),
            "artifact_sha256": artifact_record["sha256"],
            "qualification": source_record["qualification"],
        },
        "implementation": {path.name: _sha256(path) for path in implementation_paths},
        "selection": config["selection"],
        "support": config["support"],
        "reconstruction": config["reconstruction"],
        "evidence_boundary": config["evidence_boundary"],
    }
    recipe_bytes = _canonical_json(recipe)
    build_id = hashlib.sha256(recipe_bytes).hexdigest()
    root = (output_root or _ARTIFACT_ROOT) / build_id
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_bytes())
        if manifest.get("build_id") != build_id or manifest.get("status") != "complete":
            raise ValueError("existing capacity artifact conflicts with recipe")
        return manifest_path
    if root.exists() and any(root.iterdir()):
        raise ValueError("incomplete capacity artifact requires inspection")
    root.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(root / "recipe.json", recipe_bytes)

    with np.load(source_npz) as source:
        z_full = source["vendor_class2_z_mean_m"].astype(np.float64)
        count_full = source["vendor_class2_count"].astype(np.uint32)
        range_full = source["vendor_class2_z_range_m"].astype(np.float64)
    cell_m = float(config["selection"]["cell_m"])
    center_x, center_y = map(int, config["selection"]["center_xy_cells"])
    core_cells = int(round(float(config["selection"]["core_m"]) / cell_m))
    halo_cells = int(round(float(config["selection"]["halo_m"]) / cell_m))
    total_cells = core_cells + 2 * halo_cells
    if core_cells % 2 or total_cells % 2:
        raise ValueError("core and halo must produce even cell counts")
    radius = total_cells // 2
    selection = np.s_[center_y - radius : center_y + radius, center_x - radius : center_x + radius]
    z = z_full[selection]
    count = count_full[selection]
    vertical_range = range_full[selection]
    if z.shape != (total_cells, total_cells):
        raise ValueError("selected source window is incomplete")

    direct = (
        np.isfinite(z)
        & (count >= int(config["support"]["minimum_vendor_class2_returns"]))
        & np.isfinite(vertical_range)
        & (vertical_range <= float(config["support"]["maximum_vendor_class2_range_m"]))
    )
    maximum_interpolation_m = float(config["support"]["maximum_interpolation_distance_m"])
    if maximum_interpolation_m != 0.0:
        raise ValueError("this capacity screen may not publish interpolated unknown cells")
    modelable = direct.copy()
    base = _normalized_gaussian(
        z,
        direct,
        float(config["reconstruction"]["base_sigma_m"]) / cell_m,
    )
    structural = _normalized_gaussian(
        z,
        direct,
        float(config["reconstruction"]["structural_sigma_m"]) / cell_m,
    )
    structural[~modelable] = np.nan
    base[~modelable] = np.nan
    fine_surface = _normalized_gaussian(
        z,
        direct,
        float(config["reconstruction"]["fine_sigma_m"]) / cell_m,
    )

    gy, gx = np.gradient(np.nan_to_num(structural, nan=float(np.nanmedian(structural[modelable]))), cell_m)
    slope = np.hypot(gx, gy)
    local_density = ndimage.uniform_filter(direct.astype(np.float64), size=5, mode="constant")
    fine = (
        direct
        & (vertical_range <= float(config["support"]["fine_maximum_vendor_class2_range_m"]))
        & (count >= int(config["support"]["fine_minimum_vendor_class2_returns"]))
        & (local_density >= float(config["support"]["fine_minimum_local_support_fraction"]))
        & (slope <= float(config["support"]["fine_maximum_slope_rise_run"]))
    )
    fine_component = np.where(fine, fine_surface - structural, 0.0)
    reconstruction = structural + fine_component
    reconstruction[~modelable] = np.nan
    macro_component = structural - base
    observed_residual = np.where(direct, z - reconstruction, np.nan)

    core_offset = halo_cells
    core_slice = np.s_[core_offset : core_offset + core_cells, core_offset : core_offset + core_cells]
    direct_core = direct[core_slice]
    modelable_core = modelable[core_slice]
    fine_core = fine[core_slice]
    residual_core = observed_residual[core_slice]
    if float(direct_core.mean()) < float(config["acceptance"]["minimum_direct_core_fraction"]):
        raise ValueError("selected core no longer meets direct-support floor")
    if float(modelable_core.mean()) < float(config["acceptance"]["minimum_modelable_core_fraction"]):
        raise ValueError("selected core no longer meets bounded-interpolation floor")

    arrays = {
        "source_m": z.astype(np.float32),
        "base_m": base.astype(np.float32),
        "reconstruction_m": reconstruction.astype(np.float32),
        "macro_component_m": macro_component.astype(np.float32),
        "fine_component_m": fine_component.astype(np.float32),
        "observed_residual_m": observed_residual.astype(np.float32),
        "direct_support": direct.astype(np.uint8),
        "modelable_support": modelable.astype(np.uint8),
        "fine_authority": fine.astype(np.uint8),
        "vendor_class2_count": count,
        "vendor_class2_range_m": vertical_range.astype(np.float32),
    }
    artifact_path = root / "biala-source-domain-capacity.npz"
    temporary = artifact_path.with_name(artifact_path.name + ".part")
    with temporary.open("wb") as target:
        np.savez_compressed(target, **arrays)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(artifact_path)

    qa_records = render_capacity_qa(arrays, root / "qa", cell_m=cell_m, core_offset_cells=core_offset)
    images = []
    for index, record in enumerate(qa_records, start=1):
        path = record["path"]
        with Image.open(path) as image:
            dimensions = list(image.size)
        images.append(
            {
                "index": index,
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "dimensions_xy": dimensions,
                "interpretation": record["interpretation"],
            }
        )

    source_origin = config["source"]["origin_xy_m"]
    x0 = float(source_origin[0]) + (center_x - core_cells // 2) * cell_m
    y0 = float(source_origin[1]) + (center_y - core_cells // 2) * cell_m
    finite_residual = residual_core[np.isfinite(residual_core)]
    metrics = {
        "core_bounds_epsg2180_m": [x0, y0, x0 + core_cells * cell_m, y0 + core_cells * cell_m],
        "core_direct_support_fraction": float(direct_core.mean()),
        "core_modelable_support_fraction": float(modelable_core.mean()),
        "core_fine_authority_fraction": float(fine_core.mean()),
        "core_observed_residual_abs_p50_m": float(np.quantile(np.abs(finite_residual), 0.50)),
        "core_observed_residual_abs_p95_m": float(np.quantile(np.abs(finite_residual), 0.95)),
        "core_observed_residual_abs_max_m": float(np.max(np.abs(finite_residual))),
        "core_source_relief_p98_p02_m": float(np.nanquantile(z[core_slice], 0.98) - np.nanquantile(z[core_slice], 0.02)),
        "macro_component_abs_p95_m": float(np.nanquantile(np.abs(macro_component[core_slice]), 0.95)),
        "fine_component_abs_p95_m": float(np.quantile(np.abs(fine_component[core_slice][fine_core]), 0.95)),
    }
    qa_index = {
        "schema_version": _QA_SCHEMA,
        "build_id": build_id,
        "artifact": {
            "path": artifact_path.relative_to(root).as_posix(),
            "bytes": artifact_path.stat().st_size,
            "sha256": _sha256(artifact_path),
        },
        "images": images,
        "metrics": metrics,
        "interpretation": config["evidence_boundary"],
    }
    qa_index_path = root / "qa" / "index.json"
    qa_index_bytes = _canonical_json(qa_index)
    _atomic_bytes(qa_index_path, qa_index_bytes)
    manifest = {
        "schema_version": _SCHEMA,
        "status": "complete",
        "build_id": build_id,
        "recipe_sha256": build_id,
        "artifacts": {
            "float_npz": qa_index["artifact"],
            "qa_index": {
                "path": qa_index_path.relative_to(root).as_posix(),
                "bytes": len(qa_index_bytes),
                "sha256": hashlib.sha256(qa_index_bytes).hexdigest(),
            },
        },
        "metrics": metrics,
        "qualification": {
            "source_domain_capacity_only": True,
            "predictive_generator": False,
            "estonia_transfer_authorized": False,
            "production_or_owner_authority": False,
            "fine_support_m": cell_m,
            "six_centimeter_support": False,
        },
    }
    _atomic_bytes(manifest_path, _canonical_json(manifest))
    return manifest_path


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build bounded Biala coastal-escarpment capacity screen.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(build_capacity_screen(args.config, args.output_root))


if __name__ == "__main__":
    _main()
