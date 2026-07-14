"""End-to-end cultivated agricultural R0 surface and QA materialization."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from ....config import ASSET_GEN_ROOT, DATA_WORK
from .base import corrected_source_slope, fine_payload_base, rgb_conditioning_crop
from .forms import evaluate_clods, evaluate_operation_forms
from .hydrology import synthesize_hydrology
from .model import load_conditions
from .qa import render_qa, sha256_file, write_index

_ALGORITHM_VERSION = "agriculture.cultivated.r0/2"
_SOURCE_HASHES = {
    "marzahn_2012_dual_scale_soil_roughness_pdf": "37e55852fbf610eb51ef132f7c3ba5f346df9e619736c6d785a85b50ff7b98ee",
    "schott_2024_multiscale_erosion_pdf": "91f76e1a920ec3629ad065f927e0ecd1bb6498f67bf2c44217aa817d379290aa",
    "grenier_2024_controlled_patterns_pdf": "d317cee806fe836af59479b552c9679d3e9537c8bc88bee66e90d3a36b2b05e1",
    "schott_official_repository_commit": "64fe87d57d0ea904f54eb0ec24d19da08bebd737",
}
_OWNER_NAMES = ("base", "rows", "tracks", "clods", "incision", "deposition")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode("ascii") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _array_metrics(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min_m": float(array.min()),
        "max_m": float(array.max()),
        "mean_m": float(array.mean()),
        "rms_m": float(np.sqrt(np.mean(np.square(array)))),
        "p01_m": float(np.quantile(array, 0.01)),
        "p99_m": float(np.quantile(array, 0.99)),
    }


def _save_array(path: Path, values: np.ndarray, dtype: str) -> dict[str, Any]:
    np.save(path, np.asarray(values, dtype=dtype), allow_pickle=False)
    return {
        "path": f"surface/{path.name}",
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "dtype": np.dtype(dtype).name,
        "shape": list(values.shape),
    }


def build_cultivated_r0(
    config_path: Path,
    *,
    output_root: Path | None = None,
) -> Path:
    config_path = Path(config_path)
    conditions = load_conditions(config_path)
    config_sha256 = sha256_file(config_path)
    bound_sources = {
        "corrected_base": {
            "path": conditions.corrected_base_path,
            "sha256": conditions.corrected_base_sha256,
        },
        "rgb_orthophoto": {
            "path": conditions.rgb_orthophoto_path,
            "sha256": conditions.rgb_orthophoto_sha256,
        },
        "cir_orthophoto": {
            "path": conditions.cir_orthophoto_path,
            "sha256": conditions.cir_orthophoto_sha256,
        },
    }
    for source in bound_sources.values():
        actual = sha256_file(ASSET_GEN_ROOT / source["path"])
        if actual != source["sha256"]:
            raise ValueError(f"bound agriculture source differs: {source['path']}")
    qa_source_hashes = {
        **_SOURCE_HASHES,
        **{
            f"bound_{name}_sha256": source["sha256"]
            for name, source in bound_sources.items()
        },
    }
    implementation_sha256 = _implementation_sha256()
    recipe = {
        "schema_version": "laas.agriculture-cultivated-r0.recipe/1",
        "algorithm_version": _ALGORITHM_VERSION,
        "implementation_sha256": implementation_sha256,
        "condition_file_sha256": config_sha256,
        "conditions": conditions.canonical(),
        "source_hashes": _SOURCE_HASHES,
        "bound_real_sources": bound_sources,
        "representation": {
            "master": "one_absolute_heightfield",
            "texel_m": conditions.texel_m,
            "intervals": 2048,
            "samples": 2049,
            "nyquist_wavelength_m": 0.125,
            "below_nyquist_policy": "absent_not_synthesized",
            "parent_policy": "derive_later_from_decoded_children",
            "per_1m_zero_mean_projection": False,
        },
    }
    recipe_sha256 = hashlib.sha256(_canonical_bytes(recipe)).hexdigest()
    parent = (
        Path(output_root)
        if output_root is not None
        else DATA_WORK / "microtopography" / "agriculture" / "sha256"
    )
    build_root = parent / recipe_sha256
    manifest_path = build_root / "manifest.json"
    if manifest_path.exists():
        return manifest_path
    if build_root.exists():
        raise RuntimeError(f"incomplete content-addressed build already exists: {build_root}")
    temporary = parent / f".{recipe_sha256}.{os.getpid()}.tmp"
    if temporary.exists():
        raise RuntimeError(f"stale agriculture build temporary exists: {temporary}")
    surface_root = temporary / "surface"
    qa_root = temporary / "qa"
    surface_root.mkdir(parents=True)
    qa_root.mkdir(parents=True)

    try:
        base_surface = fine_payload_base(conditions)
        source_slope = corrected_source_slope(conditions)
        conditioning_rgb = rgb_conditioning_crop(conditions)
        base = base_surface.height
        operations = evaluate_operation_forms(
            base_surface.eastings, base_surface.northings, conditions
        )
        clods = evaluate_clods(
            base_surface.eastings,
            base_surface.northings,
            operations.track_mask,
            conditions,
        )
        hydrology = synthesize_hydrology(conditions)
        contributions = (
            operations.rows,
            operations.tracks,
            clods.height,
            hydrology.incision,
            hydrology.deposition,
        )
        height = np.asarray(base, dtype=np.float32).copy()
        for contribution in contributions:
            height += np.asarray(contribution, dtype=np.float32)
        if height.shape != (2049, 2049) or not np.isfinite(height).all():
            raise AssertionError("agriculture R0 did not produce one finite 2049-square master")

        contribution_stack = np.stack(
            [np.zeros_like(base), *contributions], axis=0
        )
        owner = np.argmax(np.abs(contribution_stack), axis=0).astype(np.uint8)
        residual = height.astype(np.float64) - base.astype(np.float64)
        metrics: dict[str, Any] = {
            "condition_evidence": {
                "fine_chunk_lod_cx_cz": [
                    conditions.fine_lod,
                    conditions.fine_cx,
                    conditions.fine_cz,
                ],
                "parcel_etak_id": conditions.parcel_etak_id,
                "parcel_boundary_clearance_m": conditions.parcel_boundary_clearance_m,
                "native_corrected_dtm_median_slope_deg": source_slope.median_slope_deg,
                "native_corrected_dtm_downhill_aspect_deg_east_of_grid_north": (
                    source_slope.downhill_aspect_deg_east_of_grid_north
                ),
                "soil_distribution": conditions.soil_distribution,
                "substrate_evidence_status": conditions.substrate_evidence_status,
                "hard_mask_intersections": conditions.hard_mask_intersections,
            },
            "surface": _array_metrics(height),
            "residual": _array_metrics(residual),
            "forms": {
                name: _array_metrics(contribution_stack[index])
                for index, name in enumerate(_OWNER_NAMES)
            },
            "ownership_fraction": {
                name: float(np.mean(owner == index))
                for index, name in enumerate(_OWNER_NAMES)
            },
            "clods": {
                "candidate_cells": clods.candidate_count,
                "retained_forms": clods.retained_count,
                "subtype_counts_rounded_blocky_platy": list(clods.subtype_counts),
                "diameter_range_m": list(clods.diameter_range_m),
                "height_range_m": list(clods.height_range_m),
            },
            "hydrology": {
                "process_texel_m": hydrology.process_texel_m,
                "flow_context_halo_m": conditions.flow_context_halo_m,
                "channel_cells_in_tile": hydrology.channel_cells,
                "process_boundary_outlets": hydrology.outlet_count,
                "generated_sediment_m3": hydrology.generated_sediment_m3,
                "deposited_sediment_m3": hydrology.deposited_sediment_m3,
                "exported_sediment_m3": hydrology.exported_sediment_m3,
                "mass_balance_error_m3": float(
                    hydrology.generated_sediment_m3
                    - hydrology.deposited_sediment_m3
                    - hydrology.exported_sediment_m3
                ),
            },
            "development_references_not_acceptance_knobs": {
                "marzahn_small_scale_rms_m": [0.0024, 0.0124],
                "marzahn_large_scale_rms_m": [0.0112, 0.0243],
                "marzahn_wheel_track_relief_m": [0.04, 0.06],
            },
        }
        arrays = [
            _save_array(surface_root / "master_height_f32.npy", height, "<f4"),
            _save_array(surface_root / "base_height_f32.npy", base, "<f4"),
            _save_array(
                surface_root / "typed_contributions_f32.npy",
                contribution_stack,
                "<f4",
            ),
            _save_array(surface_root / "form_ownership_u8.npy", owner, "u1"),
            _save_array(
                surface_root / "flow_area_025m_f32.npy",
                hydrology.flow_area_coarse,
                "<f4",
            ),
            _save_array(
                surface_root / "channel_mask_025m_u8.npy",
                hydrology.channel_coarse,
                "u1",
            ),
        ]
        artifacts = render_qa(
            qa_root,
            recipe_sha256=recipe_sha256,
            source_hashes=qa_source_hashes,
            height=height,
            base=base,
            owner=owner,
            flow_area=hydrology.flow_area_coarse,
            incision=hydrology.incision,
            deposition=hydrology.deposition,
            conditioning_rgb=conditioning_rgb,
            texel_m=conditions.texel_m,
        )
        qa_index = write_index(
            qa_root,
            recipe_sha256=recipe_sha256,
            source_hashes=qa_source_hashes,
            metrics=metrics,
            artifacts=artifacts,
        )
        (temporary / "recipe.json").write_text(
            json.dumps(recipe, indent=2, sort_keys=True) + "\n"
        )
        manifest = {
            "schema_version": "laas.agriculture-cultivated-r0.manifest/1",
            "status": "research_development_only",
            "recipe_sha256": recipe_sha256,
            "recipe_path": "recipe.json",
            "condition_file_sha256": config_sha256,
            "implementation_sha256": implementation_sha256,
            "arrays": arrays,
            "qa_index": {
                "path": "qa/index.json",
                "sha256": sha256_file(qa_index),
                "bytes": qa_index.stat().st_size,
            },
            "metrics": metrics,
            "release_eligible": False,
            "claim_boundary": (
                "Row orientation is observed in the bound orthophoto; management state, "
                "clod inventory, rut section, and rainfall event are development hypotheses."
            ),
            "soil_conditioning_policy": (
                "Observed LP/E2I fractions are preserved as conditions but are not "
                "translated into roughness parameters without calibrated code semantics."
            ),
            "mask_reapplication": (
                "identity_after_composition because the frozen water/building/"
                "paved-road/forest masks are all empty on this tile"
            ),
            "abstains_for_release": True,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(build_root)
    except BaseException:
        # Preserve a failed content-addressed temporary for diagnosis; a rerun fails closed.
        raise
    return manifest_path
