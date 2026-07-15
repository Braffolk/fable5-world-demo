"""Immutable C0/C1 candidate transaction and focused qualifications."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .....config import DATA_WORK
from .diagnostics import rotation_error, support_radius_m
from .forms import build_form_plan, render_fine_surface
from .model import ProcessConfig, SlopeDomain
from .process import solve_process
from .qa import render_qa


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _implementation_identity() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(Path(__file__).parent.glob("*.py")):
        rows.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return rows


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            with archive.open(info, mode="w", force_zip64=True) as member:
                np.lib.format.write_array(
                    member,
                    np.asarray(arrays[name]),
                    allow_pickle=False,
                )


def _site_metrics(
    domain: SlopeDomain,
    process: Any,
    plan: Any,
    surface: Any,
    seep_off: Any,
    enlarged_surface: Any,
    partitioned_surface: Any,
    config: ProcessConfig,
    collar_m: float,
) -> dict[str, Any]:
    ponded_mask = process.routing.ponded_water_m3 > 0.0
    ponded_depth = np.divide(
        process.routing.ponded_water_m3[ponded_mask],
        process.routing.contributing_area_m2[ponded_mask],
        out=np.zeros(np.count_nonzero(ponded_mask), dtype=np.float64),
        where=process.routing.contributing_area_m2[ponded_mask] > 0.0,
    )
    partition_max = float(
        np.max(np.abs(surface.c1_height_m - partitioned_surface.c1_height_m))
    )
    enlargement_max = float(
        np.max(np.abs(surface.residual_m - enlarged_surface.residual_m))
    )
    enlargement_ownership_exact = bool(
        np.array_equal(surface.ownership, enlarged_surface.ownership)
    )
    hard_max = float(
        np.max(np.abs(surface.residual_m[surface.hard_exclusion]), initial=0.0)
    )
    radius = support_radius_m(plan)
    residual_active = surface.residual_m[~surface.hard_exclusion]
    residual_p99 = (
        float(np.quantile(np.abs(residual_active), 0.99))
        if residual_active.size
        else 0.0
    )
    typed_fraction = float(np.mean(surface.ownership != 0))
    domain_material_supported_cells = int(
        np.count_nonzero(domain.material_rule >= 0)
    )
    crop_material_supported_samples = int(
        np.count_nonzero(surface.material_supported)
    )
    gates = {
        "partition_exact": partition_max == 0.0,
        "domain_enlargement_residual_max_abs_le_0_001_m": enlargement_max <= 0.001,
        "domain_enlargement_typed_ownership_exact": enlargement_ownership_exact,
        "hard_exclusion_exact": hard_max == 0.0,
        "support_strictly_inside_collar": radius < collar_m,
        "sediment_mass_conservative": abs(process.mass_balance_error_kg)
        <= max(
            1e-12,
            process.generated_sediment_kg
            * config.mass_balance_relative_tolerance,
        ),
        "recognizable_connected_form_proxy": plan.streamlines > 0
        and residual_p99 >= 0.003
        and typed_fraction > 0.0001,
    }
    gates = {name: bool(value) for name, value in gates.items()}
    return {
        "site_id": domain.site_id,
        "process_grid": {
            "bbox_en": list(domain.bbox_en),
            "shape": list(domain.height_m.shape),
            "texel_m": domain.texel_m,
            "active_cells": int(np.count_nonzero(domain.active)),
            "domain_material_supported_cells": domain_material_supported_cells,
            "crop_material_supported_samples": crop_material_supported_samples,
            "abstained_unknown_cells": int(np.count_nonzero(domain.unknown)),
        },
        "water_budget_m3": {
            "source": process.routing.source_water_m3,
            "exported_real_outlets": process.routing.exported_water_m3,
            "retained_finite_depression_storage": process.routing.trapped_water_m3,
            "exported_fraction": (
                process.routing.exported_water_m3 / process.routing.source_water_m3
                if process.routing.source_water_m3 > 0.0
                else 0.0
            ),
            "strict_lower_sink_cells": int(
                np.count_nonzero(
                    domain.active
                    & ~domain.outlet
                    & (process.routing.target_a < 0)
                    & (process.routing.target_b < 0)
                )
            ),
            "ponded_basins": int(np.count_nonzero(ponded_mask)),
            "truly_closed_basins": process.routing.closed_depression_cells,
            "cumulative_spill_m3": float(
                np.sum(process.routing.spilled_water_m3)
            ),
            "finite_capacity_m3": float(
                np.sum(process.routing.depression_capacity_m3)
            ),
            "mean_ponded_depth_over_contributing_area_m": {
                "p50": float(np.quantile(ponded_depth, 0.5)) if ponded_depth.size else 0.0,
                "p90": float(np.quantile(ponded_depth, 0.9)) if ponded_depth.size else 0.0,
                "p99": float(np.quantile(ponded_depth, 0.99)) if ponded_depth.size else 0.0,
                "max": float(np.max(ponded_depth, initial=0.0)),
            },
        },
        "sediment_budget_kg": {
            "detached": process.generated_sediment_kg,
            "deposited": float(np.sum(process.deposited_kg)),
            "exported_real_outlets": process.exported_sediment_kg,
            "mass_balance_error": process.mass_balance_error_kg,
        },
        "relief_budget_m3": {
            "represented_erosion": plan.represented_erosion_volume_m3,
            "unrepresented_erosion": plan.unrepresented_erosion_volume_m3,
            "represented_deposition": plan.represented_deposition_volume_m3,
            "unrepresented_deposition": plan.unrepresented_deposition_volume_m3,
        },
        "forms": {
            "channel_heads": plan.channel_heads,
            "retained_head_candidates": plan.retained_head_candidates,
            "streamlines": plan.streamlines,
            "rill_centerline_samples": len(plan.rill_points_en),
            "headcut_samples": len(plan.headcut_points_en),
            "seep_samples": len(plan.seep_points_en),
            "maximum_support_radius_m": radius,
            "collar_m": collar_m,
            "typed_crop_fraction": typed_fraction,
            "residual_abs_p99_m": residual_p99,
            "residual_min_m": float(np.min(surface.residual_m)),
            "residual_max_m": float(np.max(surface.residual_m)),
        },
        "causal_ablation": {
            "id": "seep_support_off",
            "c1_minus_ablation_rms_m": float(
                np.sqrt(np.mean((surface.c1_height_m - seep_off.c1_height_m) ** 2))
            ),
            "purpose": "isolates localized seep-support water and seep-form consequences",
        },
        "invariance": {
            "partition_max_abs_m": partition_max,
            "domain_enlargement_max_abs_m": enlargement_max,
            "domain_enlargement_typed_ownership_exact": enlargement_ownership_exact,
            "hard_exclusion_max_abs_m": hard_max,
        },
        "gates": gates,
        "hard_gate_pass": all(value for key, value in gates.items() if key != "recognizable_connected_form_proxy"),
        "visible_morphology_gate_pass": gates["recognizable_connected_form_proxy"],
        "site_result": (
            "strict_condition_abstention_no_complete_material_support"
            if crop_material_supported_samples == 0
            else (
                "development_visible_morphology_preflight_pass"
                if gates["recognizable_connected_form_proxy"]
                else "development_visible_morphology_preflight_fail"
            )
        ),
    }


def materialize_candidate(
    *,
    domains: dict[str, SlopeDomain],
    enlarged_domains: dict[str, SlopeDomain],
    crops_en: dict[str, tuple[float, float, float, float]],
    collars_m: dict[str, float],
    config: ProcessConfig,
    config_path: Path,
    condition_bundle_path: Path,
    condition_bundle_sha256: str,
    output_parent: Path | None = None,
) -> Path:
    if set(domains) != {"development_a", "development_c"}:
        raise ValueError("candidate requires exactly Development A and C")
    if set(enlarged_domains) != set(domains) or set(crops_en) != set(domains):
        raise ValueError("candidate domains, enlargements, and crops differ")
    implementation = _implementation_identity()
    recipe = {
        "schema_version": "laas.erodible-slope-c1-recipe/1",
        "condition_bundle": {
            "path": str(condition_bundle_path),
            "sha256": condition_bundle_sha256,
        },
        "config": {
            "path": str(config_path),
            "sha256": _sha256_file(config_path),
            "value": config.__dict__,
        },
        "crops_en": {name: list(crops_en[name]) for name in sorted(crops_en)},
        "source_identities": {
            name: domains[name].source_identity for name in sorted(domains)
        },
        "enlarged_source_identities": {
            name: enlarged_domains[name].source_identity
            for name in sorted(enlarged_domains)
        },
        "implementation": implementation,
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "byteorder": sys.byteorder,
            "numpy": np.__version__,
        },
        "controls": [
            "C0_cubic_1m_diagnostic_carrier_not_structural_authority",
            "C1_whole_domain_morphology_process",
        ],
        "causal_ablations": ["seep_support_off"],
        "prohibitions": [
            "no_runtime_synthesis",
            "no_noise_fbm_sines_or_stamps",
            "no_per_chunk_process_state",
            "no_priority_flood_solver_routing",
            "no_Orajogi_or_Taevaskoda_tuning",
            "no_consumed_9688685_or_sealed_9688702_inspection",
        ],
    }
    recipe_sha256 = hashlib.sha256(_canonical_bytes(recipe)).hexdigest()
    parent = (
        Path(output_parent)
        if output_parent is not None
        else DATA_WORK
        / "microtopography"
        / "erodible-slope"
        / "candidate"
        / "sha256"
    )
    root = parent / recipe_sha256
    temporary = parent / f".{recipe_sha256}.{os.getpid()}.tmp"
    if temporary.exists():
        raise RuntimeError(f"stale candidate transaction exists: {temporary}")
    temporary.mkdir(parents=True)
    rotation = rotation_error(config)
    site_metrics: dict[str, Any] = {}
    for site_id in sorted(domains):
        domain = domains[site_id]
        process = solve_process(domain, config)
        plan = build_form_plan(domain, process, config)
        surface = render_fine_surface(
            domain, process, plan, config, bbox_en=crops_en[site_id]
        )
        partitioned = render_fine_surface(
            domain,
            process,
            plan,
            config,
            bbox_en=crops_en[site_id],
            row_block=137,
        )
        seep_off_domain = replace(
            domain,
            seep_likelihood=np.zeros_like(domain.seep_likelihood),
        )
        seep_off_process = solve_process(seep_off_domain, config)
        seep_off_plan = build_form_plan(seep_off_domain, seep_off_process, config)
        seep_off = render_fine_surface(
            seep_off_domain,
            seep_off_process,
            seep_off_plan,
            config,
            bbox_en=crops_en[site_id],
        )
        enlarged = enlarged_domains[site_id]
        enlarged_process = solve_process(enlarged, config)
        enlarged_plan = build_form_plan(enlarged, enlarged_process, config)
        enlarged_surface = render_fine_surface(
            enlarged,
            enlarged_process,
            enlarged_plan,
            config,
            bbox_en=crops_en[site_id],
        )
        metrics = _site_metrics(
            domain,
            process,
            plan,
            surface,
            seep_off,
            enlarged_surface,
            partitioned,
            config,
            collars_m[site_id],
        )
        site_metrics[site_id] = metrics
        site_root = temporary / site_id
        site_root.mkdir()
        _write_deterministic_npz(
            site_root / "surface.npz",
            {
                "c0_height_m": surface.c0_height_m.astype("<f4"),
                "c1_height_m": surface.c1_height_m.astype("<f4"),
                "c1_minus_c0_m": surface.residual_m.astype("<f4"),
                "incision_m": surface.incision_m.astype("<f4"),
                "deposition_m": surface.deposition_m.astype("<f4"),
                "ownership": surface.ownership,
                "hard_exclusion": surface.hard_exclusion.astype(np.uint8),
                "material_supported": surface.material_supported.astype(np.uint8),
                "seep_off_residual_m": seep_off.residual_m.astype("<f4"),
            },
        )
        (site_root / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        render_qa(
            site_root / "qa",
            domain=domain,
            process=process,
            plan=plan,
            surface=surface,
            config=config,
            source_hashes={
                "condition_bundle": condition_bundle_sha256,
                "config": _sha256_file(config_path),
            },
            recipe_sha256=recipe_sha256,
            diagnostics={
                "partition max abs (m)": metrics["invariance"]["partition_max_abs_m"],
                "domain enlargement max abs (m)": metrics["invariance"]["domain_enlargement_max_abs_m"],
                **rotation,
                "support radius (m)": metrics["forms"]["maximum_support_radius_m"],
                "condition collar (m)": collars_m[site_id],
            },
        )

    hard_pass = all(value["hard_gate_pass"] for value in site_metrics.values())
    visible_pass = all(
        value["visible_morphology_gate_pass"] for value in site_metrics.values()
    )
    rotation_pass = max(
        rotation["rotation_water_max_relative"],
        rotation["rotation_area_max_relative"],
    ) <= 1e-10
    state = (
        "r0_development_preflight_visual_survivor_non_authorizing"
        if hard_pass and visible_pass and rotation_pass
        else "r0_development_preflight_rejected"
    )
    manifest = {
        "schema_version": "laas.erodible-slope-candidate/1",
        "recipe_sha256": recipe_sha256,
        "recipe": recipe,
        "state": state,
        "research_only": True,
        "production_owner": False,
        "preview_authorized": False,
        "recipe_freeze_authorized": False,
        "authority_blocker": (
            "Selected sealed OOD2 ETAK 9688702 remains unopened until a complete recipe "
            "freeze and A/C development run. This artifact is development-only and cannot "
            "advance the Stage-2E state machine."
        ),
        "rotation_qualification": rotation,
        "routing_preflight_baseline_development_a": {
            "method": "strict_lower_raw_height_without_finite_spill",
            "strict_lower_sink_cells": 6500,
            "source_water_m3": 931.050839104369,
            "exported_water_m3": 15.991639013858933,
            "trapped_water_m3": 915.0592000905092,
            "single_sink_implied_depth_m": {
                "p50": 0.027362376051338407,
                "p99": 1.9773660815125125,
                "max": 13.744061392093352,
            },
            "interpretation": "rejected as raw-DTM pit trapping before finite hypsometric spill",
        },
        "sites": site_metrics,
        "limitations": [
            "Development A/C are one conservative leakage group and supply no independent truth.",
            "C1 is a condition-bound process hypothesis, not measured morphology fit.",
            "The cubic 1 m C0 carrier is a morphology diagnostic only, not the accepted fine structural authority or a cookable absolute master.",
            "Any accepted future owner must compose the world-coordinate residual onto exact accepted fine structural authority, then derive parent closure.",
            "The pre-correction A enlargement delta was 0.00036822748 m; the preregistered gate is residual max <=0.001 m plus exact typed ownership, while partition identity remains exact.",
            "Priority flood selected conservative evidence domains only; C1 routing never consumes filled height.",
            "Unrepresented cap-limited sediment relief is reported rather than hidden by amplitude rescaling.",
        ],
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (temporary / "recipe.json").write_text(
        json.dumps(recipe, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if root.exists():
        existing = sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file())
        rebuilt = sorted(
            path.relative_to(temporary) for path in temporary.rglob("*") if path.is_file()
        )
        if existing != rebuilt or any(
            _sha256_file(root / relative) != _sha256_file(temporary / relative)
            for relative in rebuilt
        ):
            raise RuntimeError("existing candidate differs from deterministic reconstruction")
        shutil.rmtree(temporary)
        return root / "manifest.json"
    parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(root)
    return root / "manifest.json"
