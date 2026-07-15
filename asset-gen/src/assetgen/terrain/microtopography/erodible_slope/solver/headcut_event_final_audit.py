"""Fail-closed final audit of the frozen headcut-event emitted surfaces."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .....config import DATA_WORK
from .artifact import _canonical_bytes, _sha256_file

_SOURCE_RECIPE = "72f953d9588e811bf2b027efa3f500b6de245581632cbee84037b769e8e61b3d"
_SOURCE_MANIFEST_SHA256 = "b6445b9c73c3ffce3ffe1f9d0e2533a881b0b06cf0494f3352e5d59db8b3e18e"
_CONFIG_SHA256 = "3efc46f38828379f606d7d1da3cdf8fb0d2ae3ad320292848645f1e42ac6be0a"
_EVENT_PLAN_SHA256 = "527fe245bd5ed0a0a9dbdeeda482af90b201cb533fceafa2bde4f6fc18854cc5"
_METRICS_SHA256 = "68e7ffc962b967b0f5e1835c0b5a7f93afef0b299e7e5dcbe24610f65bae52e7"
_SURFACE_SHA256 = {
    "development_a_west": "bf7b2c04ce85b69612beaf31f245d40d858417c197a1328472277c7d002e6c27",
    "development_a_east": "e39f7000bd95247319f8122045db9cb3353213f270ec0b236a095e035461fe6b",
}
_TEXEL_M = 0.0625


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def _source_root() -> Path:
    return (
        DATA_WORK
        / "microtopography"
        / "erodible-slope"
        / "headcut-event-evaluation"
        / "sha256"
        / _SOURCE_RECIPE
    )


def _validate_sources(repo_root: Path, source_root: Path) -> None:
    identities = {
        repo_root / "asset-gen/config/microtopography/erodible-slope/r0-headcut-event-v1.json": _CONFIG_SHA256,
        source_root / "manifest.json": _SOURCE_MANIFEST_SHA256,
        source_root / "event-plan.npz": _EVENT_PLAN_SHA256,
        source_root / "metrics.json": _METRICS_SHA256,
        source_root / "crops/development_a_west/surface.npz": _SURFACE_SHA256["development_a_west"],
        source_root / "crops/development_a_east/surface.npz": _SURFACE_SHA256["development_a_east"],
    }
    for path, expected in identities.items():
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"frozen headcut-event audit input differs: {path}")


def _load_emitted(source_root: Path) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for crop in ("development_a_west", "development_a_east"):
        with np.load(source_root / "crops" / crop / "surface.npz", allow_pickle=False) as source:
            result[crop] = {name: source[name] for name in source.files}
    west = result["development_a_west"]
    east = result["development_a_east"]
    for name in ("incision_m", "deposition_m", "c1_minus_c0_m"):
        if west[name].shape != (2049, 2049) or east[name].shape != (2049, 2049):
            raise ValueError(f"unexpected emitted crop shape for {name}")
    for name in west:
        if not np.array_equal(west[name][:, -1], east[name][:, 0]):
            raise ValueError(f"frozen shared edge is not exact for {name}")
    return result


def _integrate_stitched(
    emitted: dict[str, dict[str, np.ndarray]],
    field: str,
) -> float:
    west = emitted["development_a_west"][field]
    east = emitted["development_a_east"][field]
    # The east edge of west and west edge of east are the same emitted samples.
    samples = np.sum(west[:, :-1], dtype=np.float64) + np.sum(east, dtype=np.float64)
    return float(samples * (_TEXEL_M**2))


def _corrected_metrics(
    source_metrics: dict[str, Any],
    emitted: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    incision = _integrate_stitched(emitted, "incision_m")
    deposition = _integrate_stitched(emitted, "deposition_m")
    residual = _integrate_stitched(emitted, "c1_minus_c0_m")
    surplus = deposition - incision
    tolerance = max(1e-12, incision * 1e-10)
    crop_metrics = json.loads(json.dumps(source_metrics["crops"]))
    for crop, metrics in crop_metrics.items():
        metrics["checks"]["repetition_no_comb_qualified"] = False
        metrics["repetition_no_comb"] = {
            "status": "unevaluated_fail_closed",
            "reason": "frozen config prohibits combs/repeated stamps but preregistered implementation has no metric",
            "post_output_threshold_invented": False,
        }
        metrics["recognizable_form_gate_pass"] = False
    global_checks = {
        name: value
        for name, value in source_metrics["global_checks"].items()
        if name not in {"hydrology_off_empty", "volume_conservative"}
    }
    global_checks["emitted_surface_volume_conservative"] = abs(surplus) <= tolerance
    result = "r0_headcut_event_two_crop_final_audit_rejected"
    return {
        "result": result,
        "source_recipe_sha256": _SOURCE_RECIPE,
        "sample_integration": {
            "texel_m": _TEXEL_M,
            "sample_area_m2": _TEXEL_M**2,
            "stitch_policy": "west_without_shared_east_column_plus_complete_east",
            "shared_edge_samples_counted_once": 2049,
        },
        "emitted_surface_volume_budget_m3": {
            "incision": incision,
            "deposition": deposition,
            "deposition_minus_incision_surplus": surplus,
            "integrated_emitted_c1_minus_c0": residual,
            "conservation_error_abs": abs(surplus),
            "tolerance": tolerance,
            "pass": abs(surplus) <= tolerance,
        },
        "plan_volume_diagnostic_m3": {
            **source_metrics["volume_budget_m3"],
            "scientific_gate": False,
            "interpretation": "event-plan bookkeeping only; does not qualify emitted surface conservation",
        },
        "negative_controls": {
            "corrected_only_c0": {
                "status": "carrier_reference",
                "positive_ablation_credit": 0,
            },
            "hydrology_off": {
                "event_count": source_metrics["negative_controls"]["hydrology_off_event_count"],
                "status": "switch_plumbing_diagnostic_only",
                "scientific_gate": False,
                "positive_ablation_credit": 0,
            },
        },
        "crops": crop_metrics,
        "shared_edge": source_metrics["shared_edge"],
        "global_checks": global_checks,
        "both_visual_proxies_pass": False,
        "overall_pass": False,
    }


def _result_png(path: Path, metrics: dict[str, Any]) -> None:
    canvas = Image.new("RGB", (1600, 960), "#f3efe4")
    draw = ImageDraw.Draw(canvas)
    draw.text((55, 38), "Headcut-event final audit: emitted surfaces are authoritative", font=_font(31, True), fill="#172622")
    emitted = metrics["emitted_surface_volume_budget_m3"]
    lines = (
        ("RESULT", metrics["result"]),
        ("Emitted incision", f"{emitted['incision']:.15f} m3"),
        ("Emitted deposition", f"{emitted['deposition']:.15f} m3"),
        ("Deposition surplus", f"+{emitted['deposition_minus_incision_surplus']:.15f} m3"),
        ("Conservation tolerance", f"{emitted['tolerance']:.15e} m3"),
        ("Emitted conservation", "FAIL"),
        ("Plan conservation", "diagnostic only, zero gate credit"),
        ("Hydrology-off empty plan", "switch-plumbing only, zero positive ablation credit"),
        ("West recognizable morphology", "FAIL CLOSED: repetition/no-comb metric unevaluated"),
        ("East recognizable morphology", "FAIL: no emitted event"),
        ("Both visual proxies", "FAIL"),
    )
    y = 115
    for label, value in lines:
        draw.text((70, y), f"{label}:", font=_font(22, True), fill="#33463f")
        draw.text((465, y), value, font=_font(22), fill="#8d2f23" if "FAIL" in value else "#33463f")
        y += 60
    bound = max(emitted["incision"], emitted["deposition"], 1e-12)
    for index, (label, value, color) in enumerate(
        (
            ("incision", emitted["incision"], "#9e4f37"),
            ("deposition", emitted["deposition"], "#b99544"),
        )
    ):
        y0 = 815 + index * 55
        draw.text((70, y0), label, font=_font(18, True), fill="#263c35")
        draw.rectangle((230, y0, 230 + int(1050 * value / bound), y0 + 30), fill=color)
        draw.text((1300, y0), f"{value:.9f} m3", font=_font(18), fill="#263c35")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", compress_level=9)


def materialize_final_audit(output_parent: Path | None = None) -> Path:
    source_root = _source_root()
    asset_gen_root = Path(__file__).resolve().parents[6]
    repo_root = asset_gen_root.parent
    _validate_sources(repo_root, source_root)
    implementation_path = Path(__file__).resolve()
    recipe = {
        "schema_version": "laas.erodible-slope-headcut-event-final-audit-recipe/1",
        "source": {
            "recipe_sha256": _SOURCE_RECIPE,
            "manifest_sha256": _SOURCE_MANIFEST_SHA256,
            "config_sha256": _CONFIG_SHA256,
            "event_plan_sha256": _EVENT_PLAN_SHA256,
            "metrics_sha256": _METRICS_SHA256,
            "surface_sha256": _SURFACE_SHA256,
        },
        "implementation": {
            "path": str(implementation_path.relative_to(repo_root)),
            "bytes": implementation_path.stat().st_size,
            "sha256": _sha256_file(implementation_path),
        },
        "corrections": [
            "gate_on_stitched_emitted_surface_volume_not_event_plan_totals",
            "fail_closed_when_repetition_no_comb_metric_is_unimplemented",
            "hydrology_off_is_switch_plumbing_only_with_zero_positive_credit",
        ],
        "synthesis_rerun": False,
        "frozen_config_changed": False,
        "post_output_threshold_added": False,
    }
    recipe_sha256 = hashlib.sha256(_canonical_bytes(recipe)).hexdigest()
    parent = output_parent or DATA_WORK / "microtopography" / "erodible-slope" / "headcut-event-final-audit" / "sha256"
    root = Path(parent) / recipe_sha256
    temporary = Path(parent) / f".{recipe_sha256}.{os.getpid()}.tmp"
    temporary.mkdir(parents=True)
    emitted = _load_emitted(source_root)
    source_metrics = json.loads((source_root / "metrics.json").read_text(encoding="utf-8"))
    metrics = _corrected_metrics(source_metrics, emitted)
    (temporary / "recipe.json").write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (temporary / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    qa_root = temporary / "qa"
    qa_root.mkdir()
    for name in ("01_headcut_event_morphology.png", "02_ownership_and_hydrology.png"):
        shutil.copyfile(source_root / "qa" / name, qa_root / name)
    _result_png(qa_root / "03_final_audit_result.png", metrics)
    images = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        for path in sorted(qa_root.glob("*.png"))
    ]
    (qa_root / "index.json").write_text(
        json.dumps(
            {
                "schema_version": "laas.microtopography-qa-index/1",
                "recipe_sha256": recipe_sha256,
                "source_recipe_sha256": _SOURCE_RECIPE,
                "images": images,
                "interpretation": {
                    "01_headcut_event_morphology.png": "Frozen emitted west/east morphology, unchanged.",
                    "02_ownership_and_hydrology.png": "Frozen event ownership and conditioning, unchanged.",
                    "03_final_audit_result.png": "Corrected emitted-volume, repetition/no-comb, and hydrology-off audit status.",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    files = [
        {
            "path": str(path.relative_to(temporary)),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(temporary.rglob("*"))
        if path.is_file()
    ]
    manifest = {
        "schema_version": "laas.erodible-slope-headcut-event-final-audit/1",
        "recipe_sha256": recipe_sha256,
        "state": metrics["result"],
        "source_recipe_sha256": _SOURCE_RECIPE,
        "metrics": metrics,
        "files": files,
        "research_only": True,
        "preview_authorized": False,
        "production_owner": False,
    }
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if root.exists():
        rebuilt = sorted(path.relative_to(temporary) for path in temporary.rglob("*") if path.is_file())
        existing = sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file())
        if rebuilt != existing or any(
            _sha256_file(temporary / path) != _sha256_file(root / path) for path in rebuilt
        ):
            raise RuntimeError("existing final audit differs from deterministic replay")
        shutil.rmtree(temporary)
        return root / "manifest.json"
    Path(parent).mkdir(parents=True, exist_ok=True)
    temporary.replace(root)
    return root / "manifest.json"


def main() -> None:
    print(materialize_final_audit())


if __name__ == "__main__":
    main()
