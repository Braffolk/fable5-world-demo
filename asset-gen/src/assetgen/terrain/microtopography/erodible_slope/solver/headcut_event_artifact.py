"""Immutable one-shot evaluation of the frozen headcut-event R0 hypothesis."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .....config import DATA_WORK
from .artifact import _canonical_bytes, _sha256_file, _write_deterministic_npz
from .headcut_event import (
    FORM_GULLY,
    FORM_HEADCUT,
    FORM_TOE,
    HeadcutEvent,
    HeadcutEventConfig,
    HeadcutEventPlan,
    build_headcut_event_plan,
    render_headcut_event_surface,
)
from .model import CropEvaluation, ProcessConfig, SlopeDomain
from .process import ProcessResult, solve_process
from .qa import _diverging, _earth, _fit, _font, _owner, _scale


def _validate_frozen_bindings(config: HeadcutEventConfig, repo_root: Path) -> None:
    for section in ("bindings", "implementation"):
        for name, row in config.values[section].items():
            path = repo_root / row["path"]
            if not path.is_file():
                raise FileNotFoundError(f"frozen {section} path absent: {name}: {path}")
            if path.stat().st_size != row["bytes"] or _sha256_file(path) != row["sha256"]:
                raise ValueError(f"frozen {section} identity differs: {name}")


def _event_document(event: HeadcutEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "seed_en": list(event.seed_en),
        "age_years": event.age_years,
        "head_en": list(event.head_en),
        "head_downhill_en": list(event.head_downhill_en),
        "head_half_width_m": event.head_half_width_m,
        "head_depth_m": event.head_depth_m,
        "fan_origin_en": list(event.fan_origin_en),
        "fan_downhill_en": list(event.fan_downhill_en),
        "fan_length_m": event.fan_length_m,
        "fan_half_width_m": event.fan_half_width_m,
        "fan_amplitude_m": event.fan_amplitude_m,
        "erosion_volume_m3": event.erosion_volume_m3,
        "deposition_volume_m3": event.deposition_volume_m3,
        "retreat_length_m": event.retreat_length_m,
        "incision_length_m": event.incision_length_m,
        "opportunity_peak": event.opportunity_peak,
        "seep_peak": event.seep_peak,
        "path_samples": len(event.path_points_en),
    }


def _event_catalog_sha256(plan: HeadcutEventPlan) -> str:
    document = []
    for event in plan.events:
        row = _event_document(event)
        row["path_points_sha256"] = hashlib.sha256(
            np.asarray(event.path_points_en, dtype="<f8").tobytes()
        ).hexdigest()
        row["path_radius_sha256"] = hashlib.sha256(
            np.asarray(event.path_radius_m, dtype="<f8").tobytes()
        ).hexdigest()
        row["path_depth_sha256"] = hashlib.sha256(
            np.asarray(event.path_depth_m, dtype="<f8").tobytes()
        ).hexdigest()
        document.append(row)
    return hashlib.sha256(_canonical_bytes(document)).hexdigest()


def _plan_arrays(plan: HeadcutEventPlan) -> dict[str, np.ndarray]:
    points: list[np.ndarray] = []
    radius: list[np.ndarray] = []
    depth: list[np.ndarray] = []
    event_id: list[np.ndarray] = []
    offsets = [0]
    for event in plan.events:
        points.append(event.path_points_en)
        radius.append(event.path_radius_m)
        depth.append(event.path_depth_m)
        event_id.append(np.full(len(event.path_points_en), event.event_id, dtype=np.int32))
        offsets.append(offsets[-1] + len(event.path_points_en))
    return {
        "opportunity": plan.opportunity.astype("<f4"),
        "opportunity_mask": plan.opportunity_mask.astype(np.uint8),
        "path_points_en": np.concatenate(points).astype("<f8") if points else np.empty((0, 2), dtype="<f8"),
        "path_radius_m": np.concatenate(radius).astype("<f4") if radius else np.empty((0,), dtype="<f4"),
        "path_depth_m": np.concatenate(depth).astype("<f4") if depth else np.empty((0,), dtype="<f4"),
        "path_event_id": np.concatenate(event_id) if event_id else np.empty((0,), dtype=np.int32),
        "path_offsets": np.asarray(offsets, dtype="<i8"),
        "seed_en": np.asarray([event.seed_en for event in plan.events], dtype="<f8").reshape((-1, 2)),
        "head_en": np.asarray([event.head_en for event in plan.events], dtype="<f8").reshape((-1, 2)),
        "fan_origin_en": np.asarray([event.fan_origin_en for event in plan.events], dtype="<f8").reshape((-1, 2)),
        "erosion_volume_m3": np.asarray([event.erosion_volume_m3 for event in plan.events], dtype="<f8"),
        "deposition_volume_m3": np.asarray([event.deposition_volume_m3 for event in plan.events], dtype="<f8"),
    }


def _largest_span_m(mask: np.ndarray, texel_m: float) -> float:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    maximum = 0.0
    for label in range(1, count + 1):
        rows, cols = np.nonzero(labels == label)
        if not rows.size:
            continue
        span = np.hypot(rows.max() - rows.min(), cols.max() - cols.min()) * texel_m
        maximum = max(maximum, float(span))
    return maximum


def _crop_metrics(
    evaluation: CropEvaluation,
    surface: Any,
    partitioned: Any,
    enlarged: Any,
    config: HeadcutEventConfig,
) -> dict[str, Any]:
    gates = config.gates
    typed = surface.ownership != 0
    fraction = float(np.mean(typed))
    incision_mask = surface.incision_m >= 0.001
    span = _largest_span_m(incision_mask, surface.texel_m)
    partition_max = float(np.max(np.abs(surface.c1_height_m - partitioned.c1_height_m), initial=0.0))
    enlargement_max = float(np.max(np.abs(surface.residual_m - enlarged.residual_m), initial=0.0))
    ownership_exact = bool(np.array_equal(surface.ownership, enlarged.ownership))
    hard_max = float(np.max(np.abs(surface.residual_m[surface.hard_exclusion]), initial=0.0))
    measured = {
        "headcut_pixels": int(np.count_nonzero(surface.ownership == FORM_HEADCUT)),
        "gully_pixels": int(np.count_nonzero(surface.ownership == FORM_GULLY)),
        "toe_pixels": int(np.count_nonzero(surface.ownership == FORM_TOE)),
        "typed_fraction": fraction,
        "incision_max_m": float(np.max(surface.incision_m, initial=0.0)),
        "deposition_max_m": float(np.max(surface.deposition_m, initial=0.0)),
        "residual_min_m": float(np.min(surface.residual_m, initial=0.0)),
        "residual_max_m": float(np.max(surface.residual_m, initial=0.0)),
        "largest_connected_incision_span_m": span,
    }
    checks = {
        "headcut_present": measured["headcut_pixels"] > 0,
        "connected_gully_present": measured["gully_pixels"] > 0,
        "depositional_toe_present": measured["toe_pixels"] > 0,
        "incision_depth_visible": measured["incision_max_m"] >= gates["minimum_incision_max_m"],
        "deposition_relief_visible": measured["deposition_max_m"] >= gates["minimum_deposition_max_m"],
        "connected_incision_span": span >= gates["minimum_connected_incision_span_m"],
        "typed_fraction_lower": fraction >= gates["minimum_typed_fraction"],
        "typed_fraction_upper": fraction <= gates["maximum_typed_fraction"],
        "partition_exact": partition_max == 0.0,
        "enlargement_residual": enlargement_max <= gates["maximum_enlargement_residual_m"],
        "enlargement_ownership_exact": ownership_exact,
        "hard_exclusion_exact": hard_max == 0.0,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    return {
        "crop_id": evaluation.crop_id,
        "bbox_en": list(evaluation.bbox_en),
        "output_chunk": list(evaluation.output_chunk),
        "measurements": measured,
        "invariance": {
            "partition_max_abs_m": partition_max,
            "enlargement_max_abs_m": enlargement_max,
            "enlargement_ownership_exact": ownership_exact,
            "hard_exclusion_max_abs_m": hard_max,
        },
        "checks": checks,
        "recognizable_form_gate_pass": all(checks.values()),
    }


def _surface_edge(surface: Any, column: int) -> dict[str, np.ndarray]:
    return {
        name: getattr(surface, name)[:, column].copy()
        for name in (
            "c0_height_m",
            "c1_height_m",
            "residual_m",
            "incision_m",
            "deposition_m",
            "ownership",
            "hard_exclusion",
            "material_supported",
        )
    }


def _shared_edge(west: dict[str, np.ndarray], east: dict[str, np.ndarray]) -> dict[str, Any]:
    exact = {name: bool(np.array_equal(west[name], east[name])) for name in west}
    return {
        "samples": int(next(iter(west.values())).size),
        "array_exact": exact,
        "all_arrays_exact": all(exact.values()),
    }


def _labeled_grid(
    title: str,
    panels: tuple[tuple[str, Image.Image], ...],
    *,
    columns: int,
) -> Image.Image:
    panel_w, panel_h, label_h = 520, 430, 34
    rows = int(np.ceil(len(panels) / columns))
    canvas = Image.new("RGB", (columns * panel_w, 64 + rows * (panel_h + label_h)), "#f3efe4")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 16), title, font=_font(26, True), fill="#172622")
    for index, (label, image) in enumerate(panels):
        row, col = divmod(index, columns)
        x = col * panel_w
        y = 64 + row * (panel_h + label_h)
        draw.text((x + 12, y + 4), label, font=_font(17, True), fill="#263c35")
        panel = image.resize((panel_w - 16, panel_h), Image.Resampling.BILINEAR)
        canvas.paste(panel, (x + 8, y + label_h))
    return canvas


def _overlay_events(
    image: Image.Image,
    bbox: tuple[float, float, float, float],
    events: tuple[HeadcutEvent, ...],
) -> Image.Image:
    result = image.convert("RGB").resize((640, 640), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(result)
    e0, n0, e1, n1 = bbox
    def point(value: tuple[float, float] | np.ndarray) -> tuple[int, int]:
        return (
            int(round((float(value[0]) - e0) / (e1 - e0) * 639)),
            int(round((n1 - float(value[1])) / (n1 - n0) * 639)),
        )
    for event in events:
        points = [point(value) for value in event.path_points_en]
        if len(points) >= 2:
            draw.line(points, fill="#ffffff", width=2)
        hx, hy = point(event.head_en)
        fx, fy = point(event.fan_origin_en)
        draw.ellipse((hx - 4, hy - 4, hx + 4, hy + 4), fill="#ffcf33")
        draw.ellipse((fx - 4, fy - 4, fx + 4, fy + 4), fill="#00e6a0")
    return result


def _render_qa(
    qa_root: Path,
    *,
    crops: dict[str, tuple[CropEvaluation, Any]],
    domain: SlopeDomain,
    process: ProcessResult,
    plan: HeadcutEventPlan,
    metrics: dict[str, Any],
    recipe_sha256: str,
    source_hashes: dict[str, str],
) -> None:
    del process
    qa_root.mkdir(parents=True)
    names = ("development_a_west", "development_a_east")
    morphology: list[tuple[str, Image.Image]] = []
    conditioning: list[tuple[str, Image.Image]] = []
    from .forms import _sample
    for name in names:
        evaluation, surface = crops[name]
        morphology.extend(
            (
                (f"{name}: signed C1-C0", _diverging(surface.residual_m)),
                (f"{name}: incision", _earth(surface.incision_m)),
                (f"{name}: conserved toe/fan", _earth(surface.deposition_m)),
            )
        )
        ee = np.linspace(evaluation.bbox_en[0], evaluation.bbox_en[2], 513)
        nn = np.linspace(evaluation.bbox_en[3], evaluation.bbox_en[1], 513)
        ee, nn = np.meshgrid(ee, nn)
        opportunity = _sample(domain, plan.opportunity, ee, nn, order=1)
        seep = _sample(domain, domain.seep_likelihood, ee, nn, order=1)
        conditioning.extend(
            (
                (
                    f"{name}: ownership + routed events",
                    _overlay_events(_owner(surface.ownership), evaluation.bbox_en, plan.events),
                ),
                (f"{name}: failure opportunity", _earth(opportunity)),
                (f"{name}: seep support", _earth(seep)),
            )
        )
    _labeled_grid(
        "01 Headcut-event morphology: adjacent crops, one realization",
        tuple(morphology),
        columns=3,
    ).save(qa_root / "01_headcut_event_morphology.png", format="PNG", compress_level=9)
    _labeled_grid(
        "02 Physical ownership and conditioning (yellow=headcut, green=toe)",
        tuple(conditioning),
        columns=3,
    ).save(qa_root / "02_ownership_and_hydrology.png", format="PNG", compress_level=9)

    e0, n0, e1, n1 = metrics["synthesis_bbox_en"]
    rows, cols = np.indices(domain.height_m.shape, dtype=np.float64)
    from .forms import _indices_to_en
    east, north = _indices_to_en(domain, rows, cols)
    window = (east >= e0) & (east < e1) & (north > n0) & (north <= n1)
    rr, cc = np.nonzero(window)
    crop_height = domain.height_m[rr.min() : rr.max() + 1, cc.min() : cc.max() + 1]
    crop_opp = plan.opportunity[rr.min() : rr.max() + 1, cc.min() : cc.max() + 1]
    base = np.asarray(_earth(crop_height), dtype=np.float64)
    heat = np.asarray(_earth(crop_opp), dtype=np.float64)
    alpha = (0.7 * _scale(crop_opp))[..., None]
    blended = Image.fromarray(np.clip(base * (1.0 - alpha) + heat * alpha, 0, 255).astype(np.uint8), "RGB")
    overview = _overlay_events(blended, (e0, n0, e1, n1), plan.events)
    text_image = Image.new("RGB", (640, 640), "#f3efe4")
    draw = ImageDraw.Draw(text_image)
    lines = [
        f"result: {metrics['result']}",
        f"events: {metrics['event_count']} / components: {metrics['candidate_components']}",
        f"erosion: {metrics['volume_budget_m3']['erosion']:.9f} m3",
        f"deposition: {metrics['volume_budget_m3']['deposition']:.9f} m3",
        f"mass error: {metrics['volume_budget_m3']['error']:.3e} m3",
        f"+384 catalog exact: {metrics['enlargement_event_catalog_exact']}",
        f"hydrology-off events: {metrics['negative_controls']['hydrology_off_event_count']}",
        "",
        "rejected components:",
        *[f"  {name}: {value}" for name, value in sorted(plan.rejected_components.items())],
        "",
        "crop recognizable-form gates:",
        *[f"  {name}: {value['recognizable_form_gate_pass']}" for name, value in metrics["crops"].items()],
        "",
        f"recipe: {recipe_sha256[:24]}...",
    ]
    y = 28
    for line in lines:
        draw.text((24, y), line, font=_font(16, line.endswith(":") or line.startswith("result")), fill="#243a34")
        y += 32
    _labeled_grid(
        "03 Frozen single-run result and conservation",
        (
            ("Whole 256 x 128 m window: failure opportunity and routes", overview),
            ("Machine result, controls, and exact volume budget", text_image),
        ),
        columns=2,
    ).save(qa_root / "03_result_and_conservation.png", format="PNG", compress_level=9)

    images = []
    for path in sorted(qa_root.glob("*.png")):
        images.append({"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    index = {
        "schema_version": "laas.microtopography-qa-index/1",
        "recipe_sha256": recipe_sha256,
        "source_hashes": source_hashes,
        "images": images,
        "interpretation": {
            "01_headcut_event_morphology.png": "Both adjacent fine crops; signed residual, incision, and conservative deposition.",
            "02_ownership_and_hydrology.png": "Typed ownership over physical failure opportunity and seep support; white paths are event routes.",
            "03_result_and_conservation.png": "Whole publication window, exact volume budget, frozen gates, and rejection counts.",
        },
    }
    (qa_root / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def materialize_headcut_event_evaluation(
    *,
    domain: SlopeDomain,
    enlarged_domain: SlopeDomain,
    evaluations: tuple[CropEvaluation, ...],
    process_config: ProcessConfig,
    event_config: HeadcutEventConfig,
    event_config_path: Path,
    repo_root: Path,
    output_parent: Path | None = None,
) -> Path:
    _validate_frozen_bindings(event_config, repo_root)
    process = solve_process(domain, process_config)
    enlarged_process = solve_process(enlarged_domain, process_config)
    plan = build_headcut_event_plan(domain, process, process_config, event_config)
    enlarged_plan = build_headcut_event_plan(enlarged_domain, enlarged_process, process_config, event_config)
    hydrology_off = build_headcut_event_plan(
        domain, process, process_config, event_config, hydrology_enabled=False
    )
    recipe = {
        "schema_version": "laas.erodible-slope-headcut-event-recipe/1",
        "config": {
            "path": str(event_config_path.relative_to(repo_root)),
            "bytes": event_config_path.stat().st_size,
            "sha256": _sha256_file(event_config_path),
            "value": event_config.values,
        },
        "canonical_source_identity": domain.source_identity,
        "enlarged_source_identity": enlarged_domain.source_identity,
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "byteorder": sys.byteorder,
            "numpy": np.__version__,
        },
        "execution": "one_frozen_canonical_run_no_parameter_sweep",
    }
    recipe_sha256 = hashlib.sha256(_canonical_bytes(recipe)).hexdigest()
    parent = output_parent or DATA_WORK / "microtopography" / "erodible-slope" / "headcut-event-evaluation" / "sha256"
    root = Path(parent) / recipe_sha256
    temporary = Path(parent) / f".{recipe_sha256}.{os.getpid()}.tmp"
    temporary.mkdir(parents=True)

    catalog_sha = _event_catalog_sha256(plan)
    enlarged_catalog_sha = _event_catalog_sha256(enlarged_plan)
    catalog_exact = catalog_sha == enlarged_catalog_sha
    volume_error = plan.erosion_volume_m3 - plan.deposition_volume_m3 - plan.exported_volume_m3
    volume_tolerance = max(
        1e-12,
        plan.erosion_volume_m3 * float(event_config.gates["maximum_volume_relative_error"]),
    )
    crop_metrics: dict[str, Any] = {}
    crop_surfaces: dict[str, tuple[CropEvaluation, Any]] = {}
    edges: dict[str, dict[str, np.ndarray]] = {}
    for evaluation in evaluations:
        surface = render_headcut_event_surface(domain, plan, process_config, bbox_en=evaluation.bbox_en)
        partitioned = render_headcut_event_surface(
            domain, plan, process_config, bbox_en=evaluation.bbox_en, row_block=137
        )
        enlarged_surface = render_headcut_event_surface(
            enlarged_domain, enlarged_plan, process_config, bbox_en=evaluation.bbox_en
        )
        measured = _crop_metrics(evaluation, surface, partitioned, enlarged_surface, event_config)
        crop_metrics[evaluation.crop_id] = measured
        crop_surfaces[evaluation.crop_id] = (evaluation, surface)
        edges[evaluation.crop_id] = _surface_edge(
            surface, -1 if evaluation.crop_id.endswith("west") else 0
        )
        crop_root = temporary / "crops" / evaluation.crop_id
        crop_root.mkdir(parents=True)
        _write_deterministic_npz(
            crop_root / "surface.npz",
            {
                "c0_height_m": surface.c0_height_m.astype("<f4"),
                "c1_height_m": surface.c1_height_m.astype("<f4"),
                "c1_minus_c0_m": surface.residual_m.astype("<f4"),
                "incision_m": surface.incision_m.astype("<f4"),
                "deposition_m": surface.deposition_m.astype("<f4"),
                "ownership": surface.ownership,
                "hard_exclusion": surface.hard_exclusion.astype(np.uint8),
                "material_supported": surface.material_supported.astype(np.uint8),
            },
        )
        (crop_root / "metrics.json").write_text(
            json.dumps(measured, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    shared = _shared_edge(edges["development_a_west"], edges["development_a_east"])
    global_checks = {
        "event_count_within_budget": len(plan.events) <= int(event_config.budgets["maximum_events"]),
        "erosion_volume_within_budget": plan.erosion_volume_m3 <= float(event_config.budgets["maximum_total_erosion_volume_m3"]),
        "support_within_budget": plan.maximum_support_radius_m <= float(event_config.budgets["maximum_support_radius_m"]),
        "volume_conservative": abs(volume_error) <= volume_tolerance,
        "hydrology_off_empty": len(hydrology_off.events) == 0,
        "enlargement_event_catalog_exact": catalog_exact,
        "shared_edge_exact": shared["all_arrays_exact"],
    }
    global_checks = {name: bool(value) for name, value in global_checks.items()}
    crop_pass = all(value["recognizable_form_gate_pass"] for value in crop_metrics.values())
    result = "r0_headcut_event_two_crop_survivor_non_authorizing" if crop_pass and all(global_checks.values()) else "r0_headcut_event_two_crop_rejected"
    metrics = {
        "result": result,
        "synthesis_bbox_en": list(event_config.synthesis_bbox_en),
        "event_count": len(plan.events),
        "candidate_components": plan.candidate_components,
        "rejected_components": plan.rejected_components,
        "event_catalog_sha256": catalog_sha,
        "enlarged_event_catalog_sha256": enlarged_catalog_sha,
        "enlargement_event_catalog_exact": catalog_exact,
        "maximum_support_radius_m": plan.maximum_support_radius_m,
        "volume_budget_m3": {
            "erosion": plan.erosion_volume_m3,
            "deposition": plan.deposition_volume_m3,
            "exported": plan.exported_volume_m3,
            "error": volume_error,
            "tolerance": volume_tolerance,
        },
        "negative_controls": {
            "corrected_only_c0": True,
            "hydrology_off_event_count": len(hydrology_off.events),
        },
        "crops": crop_metrics,
        "shared_edge": shared,
        "global_checks": global_checks,
        "both_visual_proxies_pass": crop_pass,
    }
    _write_deterministic_npz(temporary / "event-plan.npz", _plan_arrays(plan))
    (temporary / "event-catalog.json").write_text(
        json.dumps([_event_document(event) for event in plan.events], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (temporary / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (temporary / "recipe.json").write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    source_hashes = {
        name: row["sha256"]
        for section in ("bindings", "implementation")
        for name, row in event_config.values[section].items()
    }
    _render_qa(
        temporary / "qa",
        crops=crop_surfaces,
        domain=domain,
        process=process,
        plan=plan,
        metrics=metrics,
        recipe_sha256=recipe_sha256,
        source_hashes=source_hashes,
    )
    files = []
    for path in sorted(temporary.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": str(path.relative_to(temporary)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    manifest = {
        "schema_version": "laas.erodible-slope-headcut-event-evaluation/1",
        "recipe_sha256": recipe_sha256,
        "state": result,
        "research_only": True,
        "production_owner": False,
        "preview_authorized": False,
        "recipe_freeze_authorized": False,
        "metrics": metrics,
        "events": [_event_document(event) for event in plan.events],
        "files": files,
        "limitations": [
            "Development A west/east are adjacent crops from one condition, solve, and event realization.",
            "This R0 hypothesis has no measured morphology truth, independent-site validation, OOD credit, production authority, or preview authority.",
            "Vertical cliffs, caves, undercuts, detached blocks, water, and non-heightfield talus remain excluded.",
        ],
    }
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if root.exists():
        rebuilt = sorted(path.relative_to(temporary) for path in temporary.rglob("*") if path.is_file())
        existing = sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file())
        if rebuilt != existing or any(_sha256_file(temporary / path) != _sha256_file(root / path) for path in rebuilt):
            raise RuntimeError("existing headcut-event artifact differs from deterministic reconstruction")
        shutil.rmtree(temporary)
        return root / "manifest.json"
    Path(parent).mkdir(parents=True, exist_ok=True)
    temporary.replace(root)
    return root / "manifest.json"
