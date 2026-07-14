"""Infer conservative direct floor-sheet candidates from selected Evo plots."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from PIL import Image

from ....config import DATA_IN, DATA_WORK
from ....fetch.evo import build_evo_plot_retention_plan
from ..candidate import _retained_source
from ..source import iter_evo_points
from .qa import render_surface_qa

_SCHEMA = "evo-floor-sheet-candidate/1.0.0"
_QA_SCHEMA = "evo-floor-sheet-candidate-qa/1.0.0"
_RAW_BUILD_IDS = {
    "1086": "9a5598c601d2af9c8b7b7e2cd8d101d4d589a331c0b8f23b2e7ce4dd2d630929",
    "1065": "4d41d72a7c19a3f5aca8032d6a0db269481c2683c5c1041ec6d5ab4a1e8e8e91",
}
_EXPECTED_SOURCE_COUNTS = {
    "1086": (302_050_000, 146_688_000),
    "1065": (394_450_000, 296_803_717),
}
_CELL_M = 0.0625
_PLOT_M = 32.0
_INLIER_HALF_BAND_M = 0.02
_MIN_SOURCE_SCAN_IDS = 3
_MIN_CONSENSUS_FRACTION = 2.0 / 3.0


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def _load_raw_candidate(
    path: Path, plot_id: str
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any], str]:
    manifest_bytes = path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (
        manifest.get("status") != "complete"
        or manifest.get("build_id") != _RAW_BUILD_IDS[plot_id]
        or manifest.get("qualification", {}).get("role") != "raw_candidate"
        or manifest.get("qualification", {}).get("synthesis_authorized") is not False
    ):
        raise ValueError("Evo raw candidate authority changed")
    recipe_path = path.parent / "recipe.json"
    recipe_bytes = recipe_path.read_bytes()
    if hashlib.sha256(recipe_bytes).hexdigest() != manifest["recipe_sha256"]:
        raise ValueError("Evo raw candidate recipe failed verification")
    recipe = json.loads(recipe_bytes)
    if recipe.get("plot", {}).get("plot_id") != plot_id:
        raise ValueError("Evo raw candidate plot identity changed")
    record = manifest["artifacts"]["candidate_npz"]
    artifact = path.parent / record["path"]
    if (
        not artifact.is_file()
        or artifact.stat().st_size != record["bytes"]
        or _sha256_file(artifact) != record["sha256"]
    ):
        raise ValueError("Evo raw candidate artifact failed verification")
    with np.load(artifact, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if arrays["source_ground_reference_z_mean_m"].shape != (512, 512):
        raise ValueError("Evo raw candidate grid changed")
    return arrays, manifest, recipe, hashlib.sha256(manifest_bytes).hexdigest()


def _new_moments(size: int) -> dict[str, np.ndarray]:
    return {name: np.zeros(size, dtype=np.float64) for name in ("n", "x", "y", "z", "xx", "xy", "yy", "xz", "yz", "zz")}


def _accumulate(
    moments: dict[str, np.ndarray], flat: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> None:
    size = moments["n"].size
    values = {
        "n": None,
        "x": x,
        "y": y,
        "z": z,
        "xx": x * x,
        "xy": x * y,
        "yy": y * y,
        "xz": x * z,
        "yz": y * z,
        "zz": z * z,
    }
    for name, weights in values.items():
        moments[name] += np.bincount(flat, weights=weights, minlength=size)


def _solve(moments: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = moments["n"]
    positive = n > 0
    mean = {name: np.divide(moments[name], n, out=np.zeros_like(n), where=positive) for name in ("x", "y", "z", "xx", "xy", "yy", "xz", "yz", "zz")}
    cxx = mean["xx"] - mean["x"] ** 2
    cxy = mean["xy"] - mean["x"] * mean["y"]
    cyy = mean["yy"] - mean["y"] ** 2
    cxz = mean["xz"] - mean["x"] * mean["z"]
    cyz = mean["yz"] - mean["y"] * mean["z"]
    determinant = cxx * cyy - cxy * cxy
    rank_valid = positive & (determinant > 1e-12)
    slope_x = np.divide(cxz * cyy - cyz * cxy, determinant, out=np.zeros_like(n), where=rank_valid)
    slope_y = np.divide(cyz * cxx - cxz * cxy, determinant, out=np.zeros_like(n), where=rank_valid)
    intercept = mean["z"] - slope_x * mean["x"] - slope_y * mean["y"]
    residual_variance = mean["zz"] - mean["z"] ** 2 - slope_x * cxz - slope_y * cyz
    residual_rms = np.sqrt(np.maximum(residual_variance, 0.0))
    return intercept, slope_x, slope_y, residual_rms, rank_valid


def _popcount32(mask: np.ndarray) -> np.ndarray:
    bytes_view = mask.view(np.uint8).reshape(mask.shape + (4,))
    table = np.asarray([int(i).bit_count() for i in range(256)], dtype=np.uint8)
    return np.sum(table[bytes_view], axis=-1, dtype=np.uint8)


def _inside_points(points, minimum_x: float, minimum_y: float, width: int):
    x = np.asarray(points.x)
    y = np.asarray(points.y)
    ix = np.floor((x - minimum_x) / _CELL_M).astype(np.int64)
    iy = np.floor((y - minimum_y) / _CELL_M).astype(np.int64)
    inside = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < width)
    return x, y, ix, iy, inside


def build_evo_surface_candidate(
    plot_id: str,
    retained_manifest: Path,
    raw_candidate_manifest: Path,
    output_root: Path | None = None,
    *,
    log=print,
) -> Path:
    if plot_id not in _RAW_BUILD_IDS:
        raise ValueError(f"unsupported Evo surface plot: {plot_id}")
    source_path, retained, retained_sha256 = _retained_source(
        retained_manifest.resolve(), plot_id
    )
    raw, raw_manifest, raw_recipe, raw_manifest_sha256 = _load_raw_candidate(
        raw_candidate_manifest.resolve(), plot_id
    )
    if raw_manifest["source_sha256"] != retained["artifact"]["sha256"]:
        raise ValueError("Evo raw candidate and retained source differ")
    center_x, center_y = map(float, raw_recipe["plot"]["center_xy_m"])
    minimum_x = center_x - _PLOT_M / 2.0
    minimum_y = center_y - _PLOT_M / 2.0
    width = int(round(_PLOT_M / _CELL_M))
    reference = raw["source_ground_reference_z_mean_m"].astype(np.float64)
    reference_flat = reference.ravel()

    recipe = {
        "schema_version": _SCHEMA,
        "implementation": {
            "infer_py_sha256": _sha256_file(Path(__file__)),
            "qa_py_sha256": _sha256_file(Path(__file__).with_name("qa.py")),
            "source_py_sha256": _sha256_file(Path(__file__).parents[1] / "source.py"),
        },
        "source": {
            "plot_id": plot_id,
            "sha256": retained["artifact"]["sha256"],
            "bytes": retained["artifact"]["bytes"],
            "retained_manifest_sha256": retained_sha256,
            "raw_candidate_build_id": raw_manifest["build_id"],
            "raw_candidate_manifest_sha256": raw_manifest_sha256,
        },
        "surface_contract": {
            "name": "source-normalized directly visible forest-floor sheet candidate",
            "intended_physical_surface": "upper stable forest-floor sheet including litter, moss, embedded roots and stones; excluding living above-floor vegetation, stems, detached deadwood and other objects",
            "evidence_boundary": "the source has no audited labels separating stable organics, embedded roots, low living vegetation and detached deadwood; the candidate therefore remains semantically unresolved",
        },
        "point_acceptance_rule": {
            "predicate": "inside official 32 m square AND treeid=0 AND h exactly 0.00 m AND return_number=number_of_returns",
            "basis": "h is source-provided at 0.01 m resolution; h=0 is a source-normalized sheet signal, treeid=0 excludes segmented trees, and last return reduces foreground interception; none is independently authoritative ground",
            "first_fit": "ordinary least-squares plane in cell-centered x/y and z-(mean source z-h)",
            "robust_refit": f"refit observations within +/-{_INLIER_HALF_BAND_M:.2f} m of the first plane",
        },
        "view_rule": {
            "field": "point_source_id",
            "accepted_storage_domain": "unsigned IDs 0..31; actual non-empty IDs are recorded in inventory",
            "physical_station_mapping": "unknown and deliberately not inferred; plot 1065 has 19 non-empty IDs despite a documented nine-position upright/tilted acquisition",
            "minimum_source_scan_ids": _MIN_SOURCE_SCAN_IDS,
            "use": "three IDs provide source-group redundancy only, not a guaranteed count of independent physical views",
            "effective_view_count": "unknown",
        },
        "heightfield_rule": {
            "cell_m": _CELL_M,
            "direct_only": True,
            "interpolation": "forbidden",
            "center_bracketing": "all four cell-center quadrants require accepted observations",
            "minimum_plane_consensus_fraction": _MIN_CONSENSUS_FRACTION,
            "consensus_basis": "accepted direct observations must outnumber excluded alternatives at least two to one; not tuned to imagery",
        },
        "known_unknowns": {
            "semantic_confidence": "not calibrated",
            "sigma_xy_m": "not calibrated",
            "sigma_z_m": "not calibrated",
            "effective_footprint_m": "not reconstructable because scan origins/ranges are absent",
            "effective_view_count": "not reconstructable because point_source_id-to-station mapping is absent",
            "water_or_dynamic": "not independently labelled; plot selection is ordinary forest",
        },
        "source_point_reconstruction": {
            "record_identity": "zero-based sequential record index from exact source bytes via iter_evo_points",
            "cell_index": "floor((y-min_y)/0.0625)*512 + floor((x-min_x)/0.0625)",
            "candidate_predicate": "the exact point_acceptance_rule above",
        },
        "qualification": {
            "role": "converted_surface_candidate",
            "status": "surface_semantics_and_error_unresolved",
            "target_truth": False,
            "synthesis_authorized": False,
        },
    }
    recipe_bytes = _canonical_json(recipe)
    build_id = hashlib.sha256(recipe_bytes).hexdigest()
    build_root = (output_root or DATA_WORK / "microtopography" / "evo" / "surface" / "sha256") / build_id
    manifest_path = build_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_bytes())
        if manifest.get("status") != "complete" or manifest.get("build_id") != build_id:
            raise ValueError("existing Evo surface build conflicts with recipe")
        return manifest_path
    if build_root.exists() and any(build_root.iterdir()):
        raise ValueError("incomplete Evo surface build requires inspection")
    build_root.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(build_root / "recipe.json", recipe_bytes)

    size = width * width
    first = _new_moments(size)
    all_source_counts = np.zeros(32, dtype=np.uint64)
    eligible_source_counts = np.zeros(32, dtype=np.uint64)
    decoded_count = inside_count = eligible_count = 0

    for points in iter_evo_points(source_path):
        x, y, ix, iy, inside = _inside_points(points, minimum_x, minimum_y, width)
        decoded_count += len(points)
        inside_count += int(np.count_nonzero(inside))
        if not np.any(inside):
            continue
        point_source = np.asarray(points.point_source_id)
        inside_sources = point_source[inside].astype(np.int64)
        if np.any(inside_sources > 31):
            raise ValueError("Evo point_source_id escaped the supported 0..31 range")
        all_source_counts += np.bincount(inside_sources, minlength=32).astype(np.uint64)
        eligible = inside.copy()
        eligible[inside] &= (
            (np.asarray(points.treeid)[inside] == 0)
            & (np.asarray(points.h)[inside] == 0.0)
            & (np.asarray(points.return_number)[inside] == np.asarray(points.number_of_returns)[inside])
        )
        if not np.any(eligible):
            continue
        flat = iy[eligible] * width + ix[eligible]
        source_ids = point_source[eligible].astype(np.int64)
        eligible_source_counts += np.bincount(source_ids, minlength=32).astype(np.uint64)
        eligible_count += len(flat)
        center_cell_x = minimum_x + (ix[eligible] + 0.5) * _CELL_M
        center_cell_y = minimum_y + (iy[eligible] + 0.5) * _CELL_M
        dx = x[eligible] - center_cell_x
        dy = y[eligible] - center_cell_y
        dz = np.asarray(points.z)[eligible] - reference_flat[flat]
        _accumulate(first, flat, dx, dy, dz)
        if decoded_count % 25_000_000 < len(points):
            log(f"Evo floor first fit decoded: {decoded_count:,}")

    expected_decoded, expected_inside = _EXPECTED_SOURCE_COUNTS[plot_id]
    if decoded_count != expected_decoded or inside_count != expected_inside:
        raise ValueError("Evo recovered source population changed")
    first_intercept, first_slope_x, first_slope_y, _, first_rank = _solve(first)

    refined = _new_moments(size)
    refined_scan_mask = np.zeros(size, dtype=np.uint32)
    quadrant_mask = np.zeros(size, dtype=np.uint8)
    nearest = np.full(size, np.inf, dtype=np.float64)
    second_decoded = second_inside = initial_rejected_count = 0

    for points in iter_evo_points(source_path):
        x, y, ix, iy, inside = _inside_points(points, minimum_x, minimum_y, width)
        second_decoded += len(points)
        second_inside += int(np.count_nonzero(inside))
        if not np.any(inside):
            continue
        eligible = inside.copy()
        eligible[inside] &= (
            (np.asarray(points.treeid)[inside] == 0)
            & (np.asarray(points.h)[inside] == 0.0)
            & (np.asarray(points.return_number)[inside] == np.asarray(points.number_of_returns)[inside])
        )
        if not np.any(eligible):
            continue
        flat = iy[eligible] * width + ix[eligible]
        center_cell_x = minimum_x + (ix[eligible] + 0.5) * _CELL_M
        center_cell_y = minimum_y + (iy[eligible] + 0.5) * _CELL_M
        dx = x[eligible] - center_cell_x
        dy = y[eligible] - center_cell_y
        dz = np.asarray(points.z)[eligible] - reference_flat[flat]
        residual = dz - (first_intercept[flat] + first_slope_x[flat] * dx + first_slope_y[flat] * dy)
        accepted = first_rank[flat] & (np.abs(residual) <= _INLIER_HALF_BAND_M)
        initial_rejected_count += int(np.count_nonzero(~accepted))
        if not np.any(accepted):
            continue
        aflat = flat[accepted]
        adx = dx[accepted]
        ady = dy[accepted]
        adz = dz[accepted]
        _accumulate(refined, aflat, adx, ady, adz)
        source_ids = np.asarray(points.point_source_id)[eligible][accepted].astype(np.uint32)
        np.bitwise_or.at(refined_scan_mask, aflat, np.uint32(1) << source_ids)
        quadrants = (adx >= 0.0).astype(np.uint8) + 2 * (ady >= 0.0).astype(np.uint8)
        np.bitwise_or.at(quadrant_mask, aflat, np.uint8(1) << quadrants)
        np.minimum.at(nearest, aflat, np.hypot(adx, ady))
        if second_decoded % 25_000_000 < len(points):
            log(f"Evo floor robust fit decoded: {second_decoded:,}")

    if second_decoded != decoded_count or second_inside != inside_count:
        raise ValueError("Evo second sequential pass differs from the first")
    intercept, slope_x, slope_y, residual_rms, rank_valid = _solve(refined)
    initial_n = first["n"]
    refined_n = refined["n"]
    inlier_fraction = np.divide(refined_n, initial_n, out=np.zeros(size), where=initial_n > 0)
    scan_count = _popcount32(refined_scan_mask)
    observed = rank_valid & (refined_n >= 3)
    enough_source_groups = scan_count >= _MIN_SOURCE_SCAN_IDS
    bracketed = quadrant_mask == 0b1111
    consensus = inlier_fraction >= _MIN_CONSENSUS_FRACTION
    heightfield_valid = observed & enough_source_groups & bracketed & consensus

    ambiguity = np.zeros(size, dtype=np.uint8)
    ambiguity[initial_n == 0] |= 1
    ambiguity[(initial_n > 0) & ~enough_source_groups] |= 2
    ambiguity[(initial_n > 0) & enough_source_groups & ~bracketed] |= 4
    ambiguity[(initial_n > 0) & enough_source_groups & bracketed & ~rank_valid] |= 8
    ambiguity[(initial_n > 0) & enough_source_groups & bracketed & rank_valid & ~consensus] |= 16

    height = reference_flat + intercept
    height[~observed] = np.nan
    nearest[~np.isfinite(nearest)] = np.nan
    unknown = np.full(size, np.nan, dtype=np.float32)
    arrays = {
        "height_m_f64": height.reshape(width, width),
        "valid_observed": observed.reshape(width, width),
        "heightfield_valid": heightfield_valid.reshape(width, width),
        "semantic_class": np.where(observed, 1, 0).astype(np.uint8).reshape(width, width),
        "semantic_confidence": unknown.reshape(width, width),
        "view_group_count": scan_count.reshape(width, width),
        "effective_view_count": unknown.reshape(width, width),
        "nearest_support_m": nearest.astype(np.float32).reshape(width, width),
        "effective_footprint_m": unknown.reshape(width, width),
        "interpolation_distance_m": np.where(observed, 0.0, np.nan).astype(np.float32).reshape(width, width),
        "occluded": (~observed).reshape(width, width),
        "interpolated": np.zeros((width, width), dtype=bool),
        "water_or_dynamic": np.zeros((width, width), dtype=bool),
        "human_invalid": np.zeros((width, width), dtype=bool),
        "sigma_xy_m": unknown.reshape(width, width),
        "sigma_z_m": unknown.reshape(width, width),
        "source_point_ids_or_reconstructable_index": np.arange(size, dtype=np.uint32).reshape(width, width),
        "source_scan_mask": refined_scan_mask.reshape(width, width),
        "support_point_count": refined_n.astype(np.uint32).reshape(width, width),
        "quadrant_mask": quadrant_mask.reshape(width, width),
        "local_plane_slope": np.hypot(slope_x, slope_y).astype(np.float32).reshape(width, width),
        "local_plane_residual_rms_m": residual_rms.astype(np.float32).reshape(width, width),
        "inlier_fraction": inlier_fraction.astype(np.float32).reshape(width, width),
        "ambiguity_flags": ambiguity.reshape(width, width),
        "source_ground_reference_z_range_m": raw["source_ground_reference_z_range_m"].astype(np.float32),
    }

    artifact_path = build_root / f"{plot_id}-floor-sheet-candidate.npz"
    temporary = artifact_path.with_name(artifact_path.name + ".part")
    with temporary.open("wb") as target:
        np.savez_compressed(target, **arrays)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(artifact_path)
    qa_records = render_surface_qa(arrays, build_root / "qa", _CELL_M)
    images = []
    for index, record in enumerate(qa_records, start=1):
        path = record["path"]
        with Image.open(path) as image:
            dimensions = list(image.size)
        images.append({
            "index": index,
            "path": path.relative_to(build_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "dimensions_xy": dimensions,
            "interpretation": record["interpretation"],
        })

    observed_grid = arrays["valid_observed"]
    valid_grid = arrays["heightfield_valid"]
    nearest_grid = arrays["nearest_support_m"]
    inventory = {
        "source": {
            "decoded_point_count_each_pass": decoded_count,
            "inside_official_plot_count_each_pass": inside_count,
            "all_inside_source_id_counts": {
                str(index): int(value)
                for index, value in enumerate(all_source_counts)
                if value
            },
            "eligible_source_id_counts": {
                str(index): int(value)
                for index, value in enumerate(eligible_source_counts)
                if value
            },
        },
        "selection": {
            "eligible_direct_points": eligible_count,
            "first_fit_rejected_points": initial_rejected_count,
            "robust_fit_points": int(refined_n.sum()),
        },
        "grid": {
            "cells": size,
            "direct_observed_cells": int(np.count_nonzero(observed_grid)),
            "direct_observed_fraction": float(np.mean(observed_grid)),
            "geometrically_valid_cells": int(np.count_nonzero(valid_grid)),
            "geometrically_valid_fraction": float(np.mean(valid_grid)),
            "nearest_support_p95_m": float(np.nanquantile(nearest_grid[observed_grid], 0.95)),
            "view_group_count_quantiles": np.quantile(arrays["view_group_count"][observed_grid], (0.05, 0.5, 0.95)).tolist(),
            "plane_residual_rms_p95_m": float(np.nanquantile(arrays["local_plane_residual_rms_m"][observed_grid], 0.95)),
        },
        "ambiguity_flag_bits": {
            "1": "no exact source-normalized eligible return",
            "2": "fewer than three source-ID groups; insufficient redundancy",
            "4": "accepted observations do not bracket the cell center in all four quadrants",
            "8": "local x/y plane is rank deficient",
            "16": "less than two-thirds of eligible observations support the robust plane",
        },
        "semantic_class": {"0": "no direct candidate", "1": "source-normalized floor-sheet candidate; stable-organic semantics unresolved"},
        "qualification": recipe["qualification"],
    }
    inventory_path = build_root / "inventory.json"
    inventory_bytes = _canonical_json(inventory)
    _atomic_bytes(inventory_path, inventory_bytes)
    qa_index = {
        "schema_version": _QA_SCHEMA,
        "build_id": build_id,
        "source_sha256": retained["artifact"]["sha256"],
        "candidate_sha256": _sha256_file(artifact_path),
        "images": images,
        "qualification": recipe["qualification"],
    }
    qa_index_path = build_root / "qa" / "index.json"
    _atomic_bytes(qa_index_path, _canonical_json(qa_index))
    manifest = {
        "schema_version": _SCHEMA,
        "status": "complete",
        "build_id": build_id,
        "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
        "source_sha256": retained["artifact"]["sha256"],
        "environment": {
            "numpy": np.__version__,
            "laspy": laspy.__version__,
            "pillow": importlib.metadata.version("pillow"),
        },
        "artifacts": {
            "candidate_npz": {"path": artifact_path.name, "bytes": artifact_path.stat().st_size, "sha256": _sha256_file(artifact_path)},
            "inventory": {"path": inventory_path.name, "bytes": inventory_path.stat().st_size, "sha256": hashlib.sha256(inventory_bytes).hexdigest()},
            "qa_index": {"path": qa_index_path.relative_to(build_root).as_posix(), "bytes": qa_index_path.stat().st_size, "sha256": _sha256_file(qa_index_path)},
        },
        "qualification": recipe["qualification"],
    }
    _atomic_bytes(manifest_path, _canonical_json(manifest))
    log(f"Evo floor-sheet candidate: {build_id}")
    log(f"Evo floor-sheet manifest: {manifest_path}")
    return manifest_path


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build an Evo direct floor-sheet candidate.")
    parser.add_argument("--plot-id", choices=tuple(_RAW_BUILD_IDS), default="1086")
    parser.add_argument("--retained", type=Path)
    parser.add_argument("--raw-candidate", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    plan = build_evo_plot_retention_plan(
        authorize_exact_plot=True, plot_id=args.plot_id
    )
    retained = args.retained or (
        DATA_IN / "evidence" / "evo" / plan.retention_id / "retained.json"
    )
    raw_candidate = args.raw_candidate or (
        DATA_WORK
        / "microtopography"
        / "evo"
        / "candidate"
        / "sha256"
        / _RAW_BUILD_IDS[args.plot_id]
        / "manifest.json"
    )
    build_evo_surface_candidate(
        args.plot_id, retained, raw_candidate, args.output_root
    )


if __name__ == "__main__":
    _main()
