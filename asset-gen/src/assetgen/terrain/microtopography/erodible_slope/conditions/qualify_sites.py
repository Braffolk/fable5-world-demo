"""Qualify target-adjacent process opportunity without imagery or synthesis."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyogrio.raw
import shapely
from scipy import ndimage

from .....config import ASSET_GEN_ROOT, DATA_IN, DATA_WORK, load_base
from .....grid import ChunkId, chunk_bounds_en, chunks_covering_bbox_en
from . import select_sites
from .materialize import SiteSpec, _site_artifacts
from .opportunity import typed_process_opportunity

_REPO = ASSET_GEN_ROOT.parent
_ETAK = DATA_IN / "etak" / "ETAK_EESTI_GPKG.gpkg"
_SELECTION = (
    DATA_WORK
    / "microtopography/erodible-slope/site-selection/sha256/"
    "aa59aca7a5209b9504be5c3778b6764d6ae6f5c3bf3de26eab967a9274d7365d/"
    "selection.json"
)
_CONDITIONS = (
    DATA_WORK
    / "microtopography/erodible-slope/conditions/sha256/"
    "10a5ea53b007f3e6e2e878ccebd70a9712716042139ca35178310655fcb85b8b/"
    "bundle.json"
)
_FAILED_CANDIDATE = (
    DATA_WORK
    / "microtopography/erodible-slope/candidate/sha256/"
    "90df9d5cbb36cf21d6871d346d448716489d1f1c3dc5d8a1a7ce55a0d69c5d2c/"
    "manifest.json"
)
_OLDER_FAILED_CANDIDATE = (
    DATA_WORK
    / "microtopography/erodible-slope/candidate/sha256/"
    "98c131d4826761e09746c4bd3f02ab34d423f8d7bb94cca2a79422437a898b52/"
    "manifest.json"
)
_PROCESS_CONFIG = ASSET_GEN_ROOT / "config/microtopography/erodible-slope/solver-c1-v1.json"
_CURRENT_ETAK_ID = 9688719
_SEALED_OOD2_ETAK_ID = 9688702
_FORBIDDEN_ETAK_IDS = frozenset({9688685, _SEALED_OOD2_ETAK_ID})
_HISTORICAL_ETAK_IDS = frozenset(
    {1826743, 1826691, 9688685, _CURRENT_ETAK_ID, _SEALED_OOD2_ETAK_ID}
)
_HISTORICAL_POINTS = (
    (680551.76, 6444450.80),
    (679692.03, 6442784.16),
    (681429.34, 6443825.32),
    (680752.8418668837, 6441183.486937005),
)
_SEARCH_BBOX = (675000.0, 6440000.0, 685000.0, 6445000.0)
_MIN_SITE_DISTANCE_M = 1000.0
_MAX_OUTLET_DISTANCE_M = 64.0
_TARGET_ADJACENCY_M = 32.0
_TARGET_CONTACT_M = 4.0
_MIN_COMPONENT_CELLS = 512
_MIN_COMPONENT_SPAN_M = 32.0
_MIN_CONTACT_CELLS = 16
_MIN_LINE_M = 32.0
_MIN_EDGE_CLEARANCE_M = 16.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(_REPO)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _bound_path(identity: dict[str, Any]) -> Path:
    path = _REPO / identity["path"]
    if path.stat().st_size != identity["bytes"] or _sha256(path) != identity["sha256"]:
        raise ValueError(f"bound input differs: {path}")
    return path


def _read_arrays(identity: dict[str, Any]) -> dict[str, np.ndarray]:
    with np.load(_bound_path(identity), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _array_identity(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    rows = []
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest = hashlib.sha256()
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.tobytes())
        rows.append(
            {
                "name": name,
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "sha256": digest.hexdigest(),
            }
        )
    return {
        "arrays": rows,
        "root_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
    }


def _target_geometry(etak_id: int) -> tuple[shapely.Geometry, dict[str, Any]]:
    metadata, fids, wkbs, values = pyogrio.raw.read(
        _ETAK,
        layer="E_102_nolv_j",
        where=f"etak_id = {etak_id}",
        columns=["etak_id", "tyyp_t", "kaldaastang_t", "muutmisaeg", "geom_muutmisaeg"],
        return_fids=True,
    )
    if (
        str(metadata.get("crs")) != "EPSG:3301"
        or fids is None
        or wkbs is None
        or len(fids) != 1
        or int(values[0][0]) != etak_id
    ):
        raise ValueError(f"ETAK target identity differs: {etak_id}")
    raw = bytes(wkbs[0])
    geometry = shapely.force_2d(shapely.from_wkb(raw))
    return geometry, {
        "etak_id": etak_id,
        "source_fid": int(fids[0]),
        "geometry_sha256": hashlib.sha256(raw).hexdigest(),
        "length_m": float(geometry.length),
        "attributes": {
            "tyyp_t": values[1][0],
            "kaldaastang_t": values[2][0],
            "muutmisaeg": str(values[3][0]),
            "geom_muutmisaeg": str(values[4][0]),
        },
    }


def _crop_slices(
    bbox_en: list[int] | tuple[int, int, int, int],
    bounds: tuple[int, int, int, int],
) -> tuple[slice, slice]:
    min_e, min_n, max_e, max_n = (int(value) for value in bbox_en)
    if not (
        min_e <= bounds[0] < bounds[2] <= max_e
        and min_n <= bounds[1] < bounds[3] <= max_n
    ):
        raise ValueError("candidate crop lies outside its condition window")
    rows = slice(max_n - bounds[3], max_n - bounds[1])
    columns = slice(bounds[0] - min_e, bounds[2] - min_e)
    if (rows.stop - rows.start, columns.stop - columns.start) != (128, 128):
        raise ValueError("candidate crop is not the exact 128x128 LOD -2 footprint")
    return rows, columns


def _component_result(mask: np.ndarray, distance: np.ndarray) -> dict[str, Any]:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    components = []
    row_grid, column_grid = np.indices(mask.shape)
    interior = (
        (row_grid >= _MIN_EDGE_CLEARANCE_M)
        & (row_grid < mask.shape[0] - _MIN_EDGE_CLEARANCE_M)
        & (column_grid >= _MIN_EDGE_CLEARANCE_M)
        & (column_grid < mask.shape[1] - _MIN_EDGE_CLEARANCE_M)
    )
    for label in range(1, count + 1):
        cells = labels == label
        rows, columns = np.nonzero(cells)
        area = int(rows.size)
        near = int(np.count_nonzero(cells & (distance <= _TARGET_CONTACT_M)))
        span = float(
            np.hypot(np.ptp(rows), np.ptp(columns)) if rows.size else 0.0
        )
        interior_cells = int(np.count_nonzero(cells & interior))
        passed = (
            area >= _MIN_COMPONENT_CELLS
            and near >= _MIN_CONTACT_CELLS
            and span >= _MIN_COMPONENT_SPAN_M
            and interior_cells > 0
        )
        components.append(
            {
                "label": label,
                "cells": area,
                "near_target_cells": near,
                "span_m": span,
                "interior_cells": interior_cells,
                "passed": bool(passed),
            }
        )
    components.sort(
        key=lambda row: (
            not row["passed"],
            -row["cells"],
            -row["near_target_cells"],
            -row["span_m"],
            row["label"],
        )
    )
    best = components[0] if components else None
    result: dict[str, Any] = {
        "opportunity_cells": int(np.count_nonzero(mask)),
        "component_count": count,
        "best_component": best,
        "passed": bool(best is not None and best["passed"]),
    }
    if best is not None and best["interior_cells"] > 0:
        cells = (labels == best["label"]) & interior
        rows, columns = np.nonzero(cells)
        order = np.lexsort((columns, rows, distance[rows, columns]))
        result["representative_cell_rc"] = [
            int(rows[order[0]]),
            int(columns[order[0]]),
        ]
    return result


def _crop_opportunity(
    arrays: dict[str, np.ndarray],
    typed: dict[str, Any],
    bbox_en: list[int],
    bounds: tuple[int, int, int, int],
) -> dict[str, Any]:
    rows, columns = _crop_slices(bbox_en, bounds)
    target = arrays["target_feature"].astype(bool, copy=False)
    distance = ndimage.distance_transform_edt(~target)
    crop_distance = distance[rows, columns]
    corridor = (crop_distance > 0.0) & (crop_distance <= _TARGET_ADJACENCY_M)
    result: dict[str, Any] = {
        "hard_semantics": {
            "target_feature_cells": int(np.count_nonzero(target[rows, columns])),
            "target_protected_cells": int(
                np.count_nonzero(
                    target[rows, columns]
                    & arrays["protected_structure"][rows, columns].astype(bool)
                )
            ),
            "target_form_active_cells": int(
                np.count_nonzero(
                    target[rows, columns] & typed["form_active"][rows, columns]
                )
            ),
        },
        "domain_diagnostics": typed["diagnostics"],
        "types": {},
    }
    for name in ("runoff", "seep"):
        mask = typed[name][rows, columns] & corridor
        row = _component_result(mask, crop_distance)
        row["mask_sha256"] = hashlib.sha256(
            np.ascontiguousarray(mask.astype(np.uint8)).tobytes()
        ).hexdigest()
        result["types"][name] = row
    result["passed_types"] = [
        name for name, row in result["types"].items() if row["passed"]
    ]
    return result


def _line_point(
    geometry: shapely.Geometry,
    bounds: tuple[int, int, int, int],
) -> shapely.Point | None:
    inner = shapely.box(
        bounds[0] + _MIN_EDGE_CLEARANCE_M,
        bounds[1] + _MIN_EDGE_CLEARANCE_M,
        bounds[2] - _MIN_EDGE_CLEARANCE_M,
        bounds[3] - _MIN_EDGE_CLEARANCE_M,
    )
    parts = [
        part
        for part in shapely.get_parts(geometry.intersection(inner))
        if part.geom_type in {"LineString", "LinearRing"} and part.length > 0.0
    ]
    if not parts:
        return None
    parts.sort(key=lambda part: (-float(part.length), hashlib.sha256(part.wkb).hexdigest()))
    return parts[0].interpolate(float(parts[0].length) * 0.5)


def _evaluate_chunks(
    geometry: shapely.Geometry,
    normal_arrays: dict[str, np.ndarray],
    enlarged_arrays: dict[str, np.ndarray],
    normal_soil: dict[str, Any],
    enlarged_soil: dict[str, Any],
    normal_bbox: list[int],
    enlarged_bbox: list[int],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    grid = load_base().grid
    results = []
    normal_typed = typed_process_opportunity(normal_arrays, normal_soil, config)
    enlarged_typed = typed_process_opportunity(
        enlarged_arrays, enlarged_soil, config
    )
    chunks = sorted(
        chunks_covering_bbox_en(grid, geometry.bounds, -2),
        key=lambda chunk: (chunk.cz, chunk.cx),
    )
    for chunk in chunks:
        bounds = tuple(int(value) for value in chunk_bounds_en(grid, chunk))
        line_m = float(geometry.intersection(shapely.box(*bounds)).length)
        point = _line_point(geometry, bounds)
        if line_m < _MIN_LINE_M or point is None:
            continue
        normal = _crop_opportunity(
            normal_arrays, normal_typed, normal_bbox, bounds
        )
        enlarged = _crop_opportunity(
            enlarged_arrays, enlarged_typed, enlarged_bbox, bounds
        )
        stable_types = [
            name
            for name in ("runoff", "seep")
            if normal["types"][name]["passed"]
            and enlarged["types"][name]["passed"]
            and normal["types"][name]["mask_sha256"]
            == enlarged["types"][name]["mask_sha256"]
        ]
        results.append(
            {
                "chunk": {
                    "lod": -2,
                    "cx": chunk.cx,
                    "cz": chunk.cz,
                    "bounds_en": list(bounds),
                },
                "target_line_intersection_m": line_m,
                "target_representative_point_en": [float(point.x), float(point.y)],
                "normal": normal,
                "enlarged": enlarged,
                "stable_qualifying_types": stable_types,
                "passed": bool(stable_types),
            }
        )
    return results


def _nearest_outlet(geometry: shapely.Geometry) -> dict[str, Any] | None:
    min_e, min_n, max_e, max_n = geometry.bounds
    bbox = (
        min_e - _MAX_OUTLET_DISTANCE_M,
        min_n - _MAX_OUTLET_DISTANCE_M,
        max_e + _MAX_OUTLET_DISTANCE_M,
        max_n + _MAX_OUTLET_DISTANCE_M,
    )
    choices = []
    for layer in ("E_203_vooluveekogu_a", "E_203_vooluveekogu_j"):
        _metadata, fids, wkbs, values = pyogrio.raw.read(
            _ETAK,
            layer=layer,
            bbox=bbox,
            columns=["etak_id"],
            return_fids=True,
        )
        if fids is None or wkbs is None:
            continue
        for index, raw in enumerate(wkbs):
            distance = float(geometry.distance(shapely.from_wkb(bytes(raw))))
            if distance <= _MAX_OUTLET_DISTANCE_M:
                choices.append(
                    (distance, layer, int(values[0][index]), int(fids[index]))
                )
    if not choices:
        return None
    distance, layer, etak_id, source_fid = min(choices)
    return {
        "etak_id": etak_id,
        "source_fid": source_fid,
        "layer": layer,
        "distance_m": distance,
        "maximum_distance_m": _MAX_OUTLET_DISTANCE_M,
    }


def _metadata_candidates() -> list[dict[str, Any]]:
    excluded = ",".join(str(value) for value in sorted(_HISTORICAL_ETAK_IDS))
    _metadata, fids, wkbs, values = pyogrio.raw.read(
        _ETAK,
        layer="E_102_nolv_j",
        bbox=_SEARCH_BBOX,
        where=f"etak_id NOT IN ({excluded})",
        columns=["etak_id", "tyyp_t", "kaldaastang_t", "muutmisaeg", "geom_muutmisaeg"],
        return_fids=True,
    )
    if fids is None or wkbs is None:
        raise ValueError("ETAK candidate query returned no identity arrays")
    sheet_boxes = {
        sheet: shapely.box(*bounds) for sheet, bounds in select_sites._SHEETS.items()
    }
    orthophoto = select_sites._load_orthophoto_evidence()
    candidates = []
    for index, raw in enumerate(wkbs):
        etak_id = int(values[0][index])
        geometry = shapely.force_2d(shapely.from_wkb(bytes(raw)))
        if not select_sites._eligible_slope_role(values[1][index], values[2][index]):
            continue
        if (
            not geometry.is_valid
            or geometry.geom_type not in {"LineString", "MultiLineString"}
            or not (select_sites._MIN_LENGTH_M <= geometry.length <= select_sites._MAX_LENGTH_M)
        ):
            continue
        if min(
            geometry.distance(shapely.Point(east, north))
            for east, north in _HISTORICAL_POINTS
        ) < _MIN_SITE_DISTANCE_M:
            continue
        prospective = shapely.box(
            geometry.bounds[0] - select_sites._PROSPECTIVE_DOMAIN_MARGIN_M,
            geometry.bounds[1] - select_sites._PROSPECTIVE_DOMAIN_MARGIN_M,
            geometry.bounds[2] + select_sites._PROSPECTIVE_DOMAIN_MARGIN_M,
            geometry.bounds[3] + select_sites._PROSPECTIVE_DOMAIN_MARGIN_M,
        )
        sheets = [sheet for sheet, box_ in sheet_boxes.items() if box_.covers(prospective)]
        if len(sheets) != 1 or any(
            kind not in orthophoto[sheets[0]] for kind in ("rgb", "cir")
        ):
            continue
        outlet = _nearest_outlet(geometry)
        if outlet is None:
            continue
        midpoint = geometry.interpolate(float(geometry.length) * 0.5)
        candidates.append(
            {
                "identity": {
                    "etak_id": etak_id,
                    "source_fid": int(fids[index]),
                    "geometry_sha256": hashlib.sha256(bytes(raw)).hexdigest(),
                    "length_m": float(geometry.length),
                    "attributes": {
                        "tyyp_t": values[1][index],
                        "kaldaastang_t": values[2][index],
                        "muutmisaeg": str(values[3][index]),
                        "geom_muutmisaeg": str(values[4][index]),
                    },
                },
                "geometry": geometry,
                "midpoint": midpoint,
                "sheet": sheets[0],
                "outlet": outlet,
            }
        )
    candidates.sort(key=lambda row: row["identity"]["etak_id"])
    return candidates


def _condition_candidate(
    candidate: dict[str, Any],
    condition_bundle: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    inputs = condition_bundle["recipe"]["inputs"]
    authority_path = _bound_path(inputs["accepted_structural_authority"])
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    etak_id = candidate["identity"]["etak_id"]
    site_id = f"candidate_{etak_id}"
    result = _site_artifacts(
        SiteSpec(
            site_id=site_id,
            role="morphology_development",
            target_etak_id=etak_id,
            target_e=float(candidate["midpoint"].x),
            target_n=float(candidate["midpoint"].y),
            sheet=candidate["sheet"],
            target_outlet_ids=(candidate["outlet"]["etak_id"],),
        ),
        authority_root=authority_path.parent,
        authority_manifest=authority,
        accepted_materialization_path=_bound_path(
            inputs["accepted_corrected_materialization"]
        ),
        domain_snapshot_path=_bound_path(inputs["egt_domain_snapshot"]),
        forbidden_etak_ids=_FORBIDDEN_ETAK_IDS,
    )
    facts = result["facts"]
    obsolete_blocker = f"{site_id}_target_material_support_incomplete"
    active_blockers = [
        blocker for blocker in facts["blockers"] if blocker != obsolete_blocker
    ]
    normal_soil = json.loads(
        _bound_path(facts["soil_window"]).read_text(encoding="utf-8")
    )
    enlarged_soil = json.loads(
        _bound_path(facts["enlarged_domain"]["soil_window"]).read_text(
            encoding="utf-8"
        )
    )
    evaluations = (
        []
        if active_blockers
        else _evaluate_chunks(
            candidate["geometry"],
            result["arrays"],
            result["enlarged_domain_arrays"],
            normal_soil,
            enlarged_soil,
            facts["bbox_en"],
            facts["enlarged_domain"]["bbox_en"],
            config,
        )
    )
    passed = next((row for row in evaluations if row["passed"]), None)
    return {
        "identity": candidate["identity"],
        "sheet": candidate["sheet"],
        "outlet": candidate["outlet"],
        "condition": {
            "bbox_en": facts["bbox_en"],
            "enlarged_bbox_en": facts["enlarged_domain"]["bbox_en"],
            "soil_window": facts["soil_window"],
            "geology_window": facts["geology_window"],
            "normal_arrays": _array_identity(result["arrays"]),
            "enlarged_arrays": _array_identity(result["enlarged_domain_arrays"]),
            "reported_blockers": facts["blockers"],
            "ignored_obsolete_target_line_material_blocker": (
                obsolete_blocker in facts["blockers"]
            ),
            "active_blockers": active_blockers,
        },
        "chunk_evaluations": evaluations,
        "passed": passed is not None,
        "selected_chunk": passed,
    }


def qualify_sites(output_root: Path | None = None) -> Path:
    condition_bundle = json.loads(_CONDITIONS.read_text(encoding="utf-8"))
    config = json.loads(_PROCESS_CONFIG.read_text(encoding="utf-8"))
    control = next(
        row for row in condition_bundle["sites"] if row["site_id"] == "development_a"
    )
    control_geometry, control_identity = _target_geometry(1826743)
    control_evaluations = _evaluate_chunks(
        control_geometry,
        _read_arrays(control["arrays"]),
        _read_arrays(control["enlarged_domain_arrays"]),
        json.loads(_bound_path(control["soil_window"]).read_text(encoding="utf-8")),
        json.loads(
            _bound_path(control["enlarged_domain"]["soil_window"]).read_text(
                encoding="utf-8"
            )
        ),
        control["bbox_en"],
        control["enlarged_domain"]["bbox_en"],
        config,
    )
    if not any(row["passed"] for row in control_evaluations):
        raise RuntimeError("typed opportunity gate rejects the Development A positive control")
    current = next(
        row for row in condition_bundle["sites"] if row["site_id"] == "development_c"
    )
    current_geometry, current_identity = _target_geometry(_CURRENT_ETAK_ID)
    same_feature = _evaluate_chunks(
        current_geometry,
        _read_arrays(current["arrays"]),
        _read_arrays(current["enlarged_domain_arrays"]),
        json.loads(_bound_path(current["soil_window"]).read_text(encoding="utf-8")),
        json.loads(
            _bound_path(current["enlarged_domain"]["soil_window"]).read_text(
                encoding="utf-8"
            )
        ),
        current["bbox_en"],
        current["enlarged_domain"]["bbox_en"],
        config,
    )
    selected: dict[str, Any] | None = None
    metadata_candidates: list[dict[str, Any]] = []
    evaluated_candidates: list[dict[str, Any]] = []
    same_pass = next((row for row in same_feature if row["passed"]), None)
    if same_pass is not None:
        selected = {
            "source": "same_etak_feature",
            "identity": current_identity,
            "selected_chunk": same_pass,
        }
    else:
        metadata_candidates = _metadata_candidates()
        for candidate in metadata_candidates:
            evaluated = _condition_candidate(candidate, condition_bundle, config)
            evaluated_candidates.append(evaluated)
            if evaluated["passed"]:
                selected = {
                    "source": "next_eligible_etak_feature",
                    "identity": evaluated["identity"],
                    "sheet": evaluated["sheet"],
                    "outlet": evaluated["outlet"],
                    "selected_chunk": evaluated["selected_chunk"],
                }
                break

    sources = {
        "superseded_selection": _identity(_SELECTION),
        "superseded_condition_bundle": _identity(_CONDITIONS),
        "rejected_candidate_90df": _identity(_FAILED_CANDIDATE),
        "rejected_candidate_98c131": _identity(_OLDER_FAILED_CANDIDATE),
        "process_config_physical_inputs_only": _identity(_PROCESS_CONFIG),
        "etak": _identity(_ETAK),
        "base_config": _identity(ASSET_GEN_ROOT / "config/base.toml"),
        "grid_implementation": _identity(ASSET_GEN_ROOT / "src/assetgen/grid.py"),
        "opportunity_implementation": _identity(Path(__file__).with_name("opportunity.py")),
        "qualification_implementation": _identity(Path(__file__)),
    }
    recipe = {
        "schema_version": "laas.erodible-slope-opportunity-selection/1.recipe",
        "sources": sources,
        "search_order": "same ETAK feature canonical LOD -2 chunks by (cz,cx), then eligible ETAK features by etak_id and their chunks by (cz,cx); stop at first pass",
        "ood2_policy": "ETAK 9688702 excluded in the source SQL query; no OOD2 geometry, conditions, pixels, or candidate output accessed",
        "opportunity": {
            "common_form_active": "valid AND solve_domain AND upstream_domain AND NOT collar/outlet/unknown/water/object/non_heightfield/protected_structure AND supported material",
            "runoff": "bound 40 mm event and material runoff fraction/vegetation attenuation; conservative strictly-downhill symmetric multiple-flow accumulation; bound form eligibility requires contributing area >=36m2, DTM slope >=0.012, and shear above material critical shear",
            "seep": "positive bound topographic-seep support on common form-active cells",
            "typed_alternatives": True,
        },
        "thresholds": {
            "target_adjacency_m": _TARGET_ADJACENCY_M,
            "target_contact_m": _TARGET_CONTACT_M,
            "minimum_connected_component_cells_at_1m": _MIN_COMPONENT_CELLS,
            "minimum_component_span_m": _MIN_COMPONENT_SPAN_M,
            "minimum_near_target_cells": _MIN_CONTACT_CELLS,
            "minimum_target_line_intersection_m": _MIN_LINE_M,
            "minimum_representative_edge_clearance_m": _MIN_EDGE_CLEARANCE_M,
            "normal_enlarged_opportunity_mask_identity_required": True,
            "runoff_form_eligibility": {
                "minimum_contributing_area_m2": float(config["rill_area_threshold_m2"]),
                "minimum_slope": float(config["min_slope"]),
                "minimum_shear": "strictly above the bound material critical_shear_pa",
                "basis": "solver/forms.py _channel_heads form eligibility; positive eroded-depth precondition represented input-only by physical shear exceedance",
            },
            "basis": {
                "32m": "existing condition collar, one quarter of the 128m crop, twice the 16m ground-close review footprint, and existing minimum mapped-line extent",
                "512_cells": "two complete 16m by 16m close-review footprints, preventing a tiny local patch from standing in for connected visible morphology",
                "16_contact_cells": "one 16m-equivalent target-side contact set inside the 4m adjacency band, preventing remote-basin qualification",
            },
        },
        "forbidden": [
            "imagery_pixel_read",
            "manual_candidate_geometry_inspection",
            "visual_candidate_ranking",
            "synthesis_or_solver_output_selection",
            "solver_or_runtime_change",
            "sealed_ood2_access",
        ],
        "same_feature_identity": current_identity,
        "positive_control": {
            "identity": control_identity,
            "evaluations": control_evaluations,
            "purpose": "prove the general gate accepts independently established connected process opportunity; never candidate ranking",
        },
        "same_feature_evaluations": same_feature,
        "eligible_metadata_candidate_count": len(metadata_candidates),
        "evaluated_candidates_until_first_pass": evaluated_candidates,
        "selected": selected,
    }
    recipe_sha256 = hashlib.sha256(
        json.dumps(recipe, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
    ).hexdigest()
    artifact = {
        "schema_version": "laas.erodible-slope-opportunity-selection-proposal/1",
        "recipe_sha256": recipe_sha256,
        "status": (
            "qualified_replacement_proposed"
            if selected is not None
            else "no_qualified_second_development_site_in_retained_search_domain"
        ),
        "development_replacement": selected,
        "positive_control": recipe["positive_control"],
        "same_feature_evaluations": same_feature,
        "evaluated_candidates_until_first_pass": evaluated_candidates,
        "rejected_history": {
            "aa59_selection": "rejected: counted the protected target line instead of adjacent form-active opportunity",
            "10a5_condition_bundle": "retained as immutable historical condition evidence; its Development C crop qualification is rejected",
            "90df_candidate": "rejected historical synthesis result; sparse neighboring forms do not qualify the site",
            "98c131_candidate": "rejected historical condition-abstention result",
        },
        "sealed_ood2": "unchanged and sealed; excluded at source and never accessed by this qualification",
        "conclusion": (
            "No replacement is proposed. Keep Development A as the only runnable morphology-development site and leave the second role vacant until a new metadata-eligible site passes the typed opportunity gate."
            if selected is None
            else "The first deterministic typed-opportunity pass is proposed as the replacement; no later candidate was opened."
        ),
        "recipe": recipe,
    }
    parent = output_root or DATA_WORK / "microtopography/erodible-slope/site-selection/sha256"
    root = parent / recipe_sha256
    path = root / "selection-proposal.json"
    text = json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    reasoning = (
        "# Erodible-slope opportunity selection proposal\n\n"
        f"Recipe `{recipe_sha256}` replaces target-line counts with typed adjacent "
        "process opportunity. Protected mapped lines remain immutable. Runoff and "
        "seep qualify independently, but either must form the same sufficiently "
        "large target-touching component in normal and enlarged evidence windows.\n\n"
        f"Result: `{artifact['status']}`. {artifact['conclusion']}\n"
    )
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError("existing opportunity selection proposal differs")
        if (root / "reasoning.md").read_text(encoding="utf-8") != reasoning:
            raise RuntimeError("existing opportunity reasoning differs")
        return path
    temporary = parent / f".{recipe_sha256}.{os.getpid()}.tmp"
    temporary.mkdir(parents=True)
    (temporary / "selection-proposal.json").write_text(text, encoding="utf-8")
    (temporary / "reasoning.md").write_text(reasoning, encoding="utf-8")
    parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(root)
    return path


if __name__ == "__main__":
    print(qualify_sites())
