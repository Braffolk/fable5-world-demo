"""Refine a frozen Development crop using condition relevance, without imagery."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyogrio.raw
import shapely

from .....config import ASSET_GEN_ROOT, DATA_IN, DATA_WORK, load_base
from .....grid import ChunkId, chunk_bounds_en, chunks_covering_bbox_en
from .select_sites import _COARSE_TEXTURES, _COHESIVE_TEXTURES, _surface_texture

_REPO = ASSET_GEN_ROOT.parent
_ETAK = DATA_IN / "etak" / "ETAK_EESTI_GPKG.gpkg"
_PRIOR_SELECTION = (
    DATA_WORK
    / "microtopography/erodible-slope/site-selection/sha256/"
    "15d76d27aade90051e74fa39b96db35ba396df25d14733593f761bc17f0ed2e1/"
    "selection.json"
)
_CONDITION_BUNDLE = (
    DATA_WORK
    / "microtopography/erodible-slope/conditions/sha256/"
    "fa43f7075214eb0f0eef5b3977427823dcf10a09157e53ac9a9f61b774c5077d/"
    "bundle.json"
)
_DEVELOPMENT_ID = 9688719
_OLD_CHUNK = ChunkId(2438, 1519, -2)
_MIN_SOLVE_FRACTION = 0.25
_MIN_TARGET_SOLVE_CELLS = 32
_MIN_MATERIAL_FRACTION = 0.25
_MIN_EDGE_CLEARANCE_M = 16.0
_MIN_TARGET_LINE_M = 32.0


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


def _read_target() -> tuple[shapely.Geometry, dict[str, Any]]:
    metadata, fids, wkbs, columns = pyogrio.raw.read(
        _ETAK,
        layer="E_102_nolv_j",
        where=f"etak_id = {_DEVELOPMENT_ID}",
        columns=["etak_id", "tyyp_t", "kaldaastang_t", "muutmisaeg", "geom_muutmisaeg"],
        return_fids=True,
    )
    if (
        str(metadata.get("crs")) != "EPSG:3301"
        or fids is None
        or wkbs is None
        or len(fids) != 1
        or int(columns[0][0]) != _DEVELOPMENT_ID
    ):
        raise ValueError("Development C target identity differs")
    raw_wkb = bytes(wkbs[0])
    return shapely.force_2d(shapely.from_wkb(raw_wkb)), {
        "source_fid": int(fids[0]),
        "geometry_sha256": hashlib.sha256(raw_wkb).hexdigest(),
        "etak_attributes": {
            "tyyp_t": columns[1][0],
            "kaldaastang_t": columns[2][0],
            "muutmisaeg": str(columns[3][0]),
            "geom_muutmisaeg": str(columns[4][0]),
        },
    }


def _bound_path(identity: dict[str, Any]) -> Path:
    path = _REPO / identity["path"]
    if path.stat().st_size != identity["bytes"] or _sha256(path) != identity["sha256"]:
        raise ValueError(f"bound condition input differs: {path}")
    return path


def _material_support(
    arrays: dict[str, np.ndarray],
    soil_document: dict[str, Any],
) -> np.ndarray:
    features = soil_document["features"]
    families: list[str | None] = []
    for feature in features:
        normalized = feature["normalized"]
        if normalized["Loimis2"]["status"] not in {
            "parsed_complete_official_grammar",
            "missing",
        }:
            families.append(None)
        else:
            families.append(
                _surface_texture(normalized["Loimis1"], normalized["Huumus"])
            )
    soil_index = arrays["soil_feature_index"]
    lithology = arrays["geology_lithology_code"]
    genesis = arrays["geology_genesis_code"]
    result = np.zeros(soil_index.shape, dtype=bool)
    for index, family in enumerate(families):
        if family is None:
            continue
        cells = soil_index == index
        if family in _COARSE_TEXTURES:
            result |= cells & (lithology == 40) & np.isin(
                genesis, (10, 30, 40, 60, 100)
            )
        if family in _COHESIVE_TEXTURES:
            result |= cells & (lithology == 50) & (genesis == 50)
    return result


def _chunk_slices(
    bounds: tuple[int, int, int, int],
    site: dict[str, Any],
) -> tuple[slice, slice]:
    bounds = tuple(int(value) for value in bounds)
    min_e, min_n, max_e, max_n = (int(value) for value in site["bbox_en"])
    if not (
        min_e <= bounds[0] < bounds[2] <= max_e
        and min_n <= bounds[1] < bounds[3] <= max_n
    ):
        raise ValueError(f"candidate chunk lies outside materialized condition grid: {bounds}")
    rows = slice(max_n - bounds[3], max_n - bounds[1])
    columns = slice(bounds[0] - min_e, bounds[2] - min_e)
    if rows.stop - rows.start != 128 or columns.stop - columns.start != 128:
        raise ValueError("candidate condition crop is not exactly 128x128 process cells")
    return rows, columns


def _point_cell(
    point: shapely.Point,
    site: dict[str, Any],
) -> tuple[int, int]:
    min_e, _, _, max_n = (int(value) for value in site["bbox_en"])
    column = int(np.floor(float(point.x) - min_e))
    row = int(np.floor(max_n - float(point.y)))
    return row, column


def _active_representative(
    geometry: shapely.Geometry,
    bounds: tuple[int, int, int, int],
    site: dict[str, Any],
    active_target: np.ndarray,
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
    parts.sort(key=lambda part: hashlib.sha256(part.wkb).hexdigest())
    for part in parts:
        sample_count = max(1, int(np.ceil(float(part.length))))
        for ordinal in range(sample_count):
            distance = min(float(part.length), ordinal + 0.5)
            point = part.interpolate(distance)
            row, column = _point_cell(point, site)
            if active_target[row, column]:
                return point
    return None


def _evaluate_chunk(
    chunk: ChunkId,
    geometry: shapely.Geometry,
    site: dict[str, Any],
    arrays: dict[str, np.ndarray],
    material_supported: np.ndarray,
) -> dict[str, Any]:
    grid = load_base().grid
    bounds = tuple(int(value) for value in chunk_bounds_en(grid, chunk))
    rows, columns = _chunk_slices(bounds, site)
    valid = arrays["valid"].astype(bool, copy=False)
    solve = arrays["solve_domain"].astype(bool, copy=False)
    target = arrays["target_feature"].astype(bool, copy=False)
    unknown = arrays["unknown"].astype(bool, copy=False)
    soil_covered = arrays["soil_feature_index"] >= 0
    geology_decoded = (
        (arrays["geology_lithology_code"] >= 0)
        & (arrays["geology_genesis_code"] >= 0)
    )
    line_m = float(geometry.intersection(shapely.box(*bounds)).length)
    crop_valid = valid[rows, columns]
    crop_solve = solve[rows, columns]
    crop_target = target[rows, columns]
    crop_unknown = unknown[rows, columns]
    crop_soil_covered = soil_covered[rows, columns]
    crop_geology_decoded = geology_decoded[rows, columns]
    crop_material_supported = material_supported[rows, columns]
    crop_active_target = (
        crop_target
        & crop_solve
        & crop_material_supported
        & crop_valid
        & ~crop_unknown
    )
    solve_count = int(np.count_nonzero(crop_solve))
    target_count = int(np.count_nonzero(crop_target))
    target_solve_count = int(np.count_nonzero(crop_target & crop_solve))
    material_count = int(np.count_nonzero(crop_material_supported))
    target_material_count = int(np.count_nonzero(crop_target & crop_material_supported))
    target_solve_material_count = int(
        np.count_nonzero(crop_target & crop_solve & crop_material_supported)
    )
    active_target_count = int(np.count_nonzero(crop_active_target))
    active_target = target & solve & material_supported & valid & ~unknown
    point = _active_representative(geometry, bounds, site, active_target)
    result = {
        "chunk": {"lod": -2, "cx": chunk.cx, "cz": chunk.cz, "bounds_en": list(bounds)},
        "metrics": {
            "process_cells": 128 * 128,
            "valid_cells": int(np.count_nonzero(crop_valid)),
            "soil_source_covered_cells": int(np.count_nonzero(crop_soil_covered)),
            "geology_decoded_cells": int(np.count_nonzero(crop_geology_decoded)),
            "unknown_cells": int(np.count_nonzero(crop_unknown)),
            "solve_domain_cells": solve_count,
            "solve_domain_fraction": solve_count / (128 * 128),
            "target_feature_cells": target_count,
            "target_feature_solve_domain_cells": target_solve_count,
            "target_feature_material_supported_cells": target_material_count,
            "target_feature_solve_material_supported_cells": target_solve_material_count,
            "active_target_cells": active_target_count,
            "material_supported_cells": material_count,
            "material_supported_fraction": material_count / (128 * 128),
            "exact_target_line_intersection_m": line_m,
        },
        "gates": {
            "solve_domain_fraction_at_least_0_25": solve_count / (128 * 128) >= _MIN_SOLVE_FRACTION,
            "target_feature_solve_cells_at_least_32": target_solve_count >= _MIN_TARGET_SOLVE_CELLS,
            "material_supported_fraction_at_least_0_25": material_count / (128 * 128) >= _MIN_MATERIAL_FRACTION,
            "exact_target_line_intersection_at_least_32m": line_m >= _MIN_TARGET_LINE_M,
            "active_exact_target_representative_at_least_16m_inside": point is not None,
        },
    }
    result["passed"] = all(result["gates"].values())
    if point is not None:
        result["representative_point_en"] = [float(point.x), float(point.y)]
    return result


def _amendment(artifact: dict[str, Any]) -> str:
    selected = artifact["development_replacement"]
    chunk = selected["output_chunk"]
    point = selected["representative_point_en"]
    failed = artifact["rejected_development_c_crops"][0]
    return f"""# Required Stage 2E Development-C Crop Amendment Draft

- Supersede selection recipe `15d76d27aade90051e74fa39b96db35ba396df25d14733593f761bc17f0ed2e1` with condition-relevance refinement `{artifact['recipe_sha256']}`; ETAK Development C identity `9688719` and sealed OOD2 identity `9688702` do not change.
- Replace rejected Development C LOD -2 crop `({failed['chunk']['cx']},{failed['chunk']['cz']})` with exact canonical chunk `({chunk['cx']},{chunk['cz']})`, bounds `{chunk['bounds_en']}`, representative EPSG:3301 point `{point[0]:.6f}, {point[1]:.6f}`.
- The replacement crop is the first same-feature chunk in frozen `(cz,cx)` order that passes all predeclared condition-relevance gates: solve-domain coverage >=25%, target/solve intersection >=32 cells, material support >=25%, exact target-line intersection >=32 m, and an active exact-line representative >=16 m inside the chunk.
- Every crop-scoped support count and criterion proof is rebuilt from the replacement chunk's bound condition arrays and exact ETAK geometry; no support or crop proof from the rejected chunk is inherited.
- Retain the former crop as rejected condition-irrelevant development evidence; its correct zero C1 residual is not a synthesis failure.
- Carry OOD2 forward unchanged and sealed. Do not materialize, inspect, render, or open its pixels or source geometry before complete recipe freeze.
- Development C remains R0 process-development evidence only. This amendment changes no solver coefficient, production authority, runtime, or wire format.
"""


def refine_development_crop(output_root: Path | None = None) -> Path:
    prior = json.loads(_PRIOR_SELECTION.read_text(encoding="utf-8"))
    bundle = json.loads(_CONDITION_BUNDLE.read_text(encoding="utf-8"))
    site = next(row for row in bundle["sites"] if row["site_id"] == "development_c")
    if int(site["target_etak_id"]) != _DEVELOPMENT_ID or site["blockers"]:
        raise ValueError("Development C condition bundle is not the expected unblocked source")
    arrays_path = _bound_path(site["arrays"])
    soil_path = _bound_path(site["soil_window"])
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    soil_document = json.loads(soil_path.read_text(encoding="utf-8"))
    material_supported = _material_support(arrays, soil_document)
    geometry, geometry_identity = _read_target()
    prior_development = prior["development_replacement"]
    if geometry_identity["geometry_sha256"] != prior_development["geometry_sha256"]:
        raise ValueError("Development C geometry differs from frozen identity")
    if geometry_identity["source_fid"] != prior_development["source_fid"]:
        raise ValueError("Development C source FID differs from frozen identity")
    if geometry_identity["etak_attributes"] != prior_development["etak_attributes"]:
        raise ValueError("Development C attributes differ from frozen identity")
    if not geometry.is_valid or geometry.geom_type not in {"LineString", "MultiLineString"}:
        raise ValueError("Development C no longer has valid line geometry")
    if geometry_identity["etak_attributes"]["tyyp_t"] != "Looduslik järsak" and not (
        geometry_identity["etak_attributes"]["tyyp_t"] == "Nõlv"
        and geometry_identity["etak_attributes"]["kaldaastang_t"] == "Jah"
    ):
        raise ValueError("Development C no longer has an eligible slope role")

    grid = load_base().grid
    ordered_chunks = sorted(
        chunks_covering_bbox_en(grid, geometry.bounds, -2),
        key=lambda chunk: (chunk.cz, chunk.cx),
    )
    old_result = _evaluate_chunk(_OLD_CHUNK, geometry, site, arrays, material_supported)
    evaluations: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for chunk in ordered_chunks:
        if chunk == _OLD_CHUNK:
            continue
        result = _evaluate_chunk(chunk, geometry, site, arrays, material_supported)
        evaluations.append(result)
        if result["passed"]:
            selected = result
            break
    if selected is None:
        raise RuntimeError(
            "no alternative canonical Development C chunk passed the frozen condition-relevance gate; "
            f"evaluated={evaluations}"
        )

    sources = {
        "prior_selection": _identity(_PRIOR_SELECTION),
        "condition_bundle": _identity(_CONDITION_BUNDLE),
        "condition_arrays": _identity(arrays_path),
        "condition_soil_window": _identity(soil_path),
        "etak": _identity(_ETAK),
        "base_config": _identity(ASSET_GEN_ROOT / "config/base.toml"),
        "grid_implementation": _identity(ASSET_GEN_ROOT / "src/assetgen/grid.py"),
        "implementation": _identity(Path(__file__)),
    }
    recipe = {
        "schema_version": "laas.erodible-slope-condition-relevance-selection/1.recipe",
        "sources": sources,
        "development_etak_id": _DEVELOPMENT_ID,
        "old_chunk": [-2, _OLD_CHUNK.cx, _OLD_CHUNK.cz],
        "candidate_order": "same ETAK feature canonical LOD -2 chunks in cz,cx ascending order; stop at first pass",
        "gates": {
            "minimum_solve_domain_fraction": _MIN_SOLVE_FRACTION,
            "minimum_target_feature_solve_cells": _MIN_TARGET_SOLVE_CELLS,
            "minimum_material_supported_fraction": _MIN_MATERIAL_FRACTION,
            "minimum_representative_edge_clearance_m": _MIN_EDGE_CLEARANCE_M,
            "minimum_exact_target_line_intersection_m": _MIN_TARGET_LINE_M,
            "representative_cell": "exact target-feature AND solve-domain AND C1 material-supported",
        },
        "forbidden": [
            "imagery_pixel_read",
            "human_candidate_geometry_inspection",
            "sealed_ood2_materialization_or_inspection",
            "visual_ranking",
            "solver_change",
        ],
    }
    recipe_sha256 = hashlib.sha256(
        json.dumps(recipe, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    selected_chunk = selected["chunk"]
    metrics = selected["metrics"]
    process_cells = metrics["process_cells"]
    crop_soil_complete = metrics["soil_source_covered_cells"] == process_cells
    crop_geology_complete = metrics["geology_decoded_cells"] == process_cells
    target_point = shapely.Point(*selected["representative_point_en"])
    target_row, target_column = _point_cell(target_point, site)
    target_point_active = bool(
        arrays["valid"][target_row, target_column]
        and not arrays["unknown"][target_row, target_column]
        and arrays["target_feature"][target_row, target_column]
        and arrays["solve_domain"][target_row, target_column]
        and material_supported[target_row, target_column]
    )
    if not target_point_active:
        raise RuntimeError("selected representative is not active in the bound condition arrays")
    development = {
        "etak_id": prior_development["etak_id"],
        "source_fid": prior_development["source_fid"],
        "geometry_sha256": prior_development["geometry_sha256"],
        "primary_sheet": prior_development["primary_sheet"],
        "required_orthophoto_sheets": prior_development["required_orthophoto_sheets"],
        "length_m": float(geometry.length),
        "etak_attributes": geometry_identity["etak_attributes"],
        "representative_point_en": selected["representative_point_en"],
        "output_chunk": {
            **selected_chunk,
            "minimum_target_line_intersection_m": _MIN_TARGET_LINE_M,
            "representative_point_minimum_edge_clearance_m": _MIN_EDGE_CLEARANCE_M,
            "target_line_intersection_m": selected["metrics"]["exact_target_line_intersection_m"],
        },
        "support": {
            "crop_bbox_en": selected_chunk["bounds_en"],
            "process_cells": process_cells,
            "valid_cells": metrics["valid_cells"],
            "soil_source_covered_cells": metrics["soil_source_covered_cells"],
            "geology_decoded_cells": metrics["geology_decoded_cells"],
            "unknown_cells": metrics["unknown_cells"],
            "material_supported_cells": metrics["material_supported_cells"],
            "supported_crop_fraction": metrics["material_supported_fraction"],
            "minimum_supported_crop_fraction": _MIN_MATERIAL_FRACTION,
            "target_feature_cells": metrics["target_feature_cells"],
            "target_feature_material_supported_cells": metrics[
                "target_feature_material_supported_cells"
            ],
            "target_feature_solve_material_supported_cells": metrics[
                "target_feature_solve_material_supported_cells"
            ],
            "active_target_cells": metrics["active_target_cells"],
            "unsupported_material_or_unknown_cells_remain_fail_closed_abstention": True,
        },
        "criterion_pass_proof": {
            "eligible_natural_or_bank_escarpment_rechecked_from_bound_etak": True,
            "valid_line_geometry_rechecked_from_bound_etak": True,
            "output_crop_m": 128,
            "output_crop_is_exact_canonical_lod_minus_2_chunk": True,
            "representative_point_at_least_16m_inside_chunk": selected["gates"][
                "active_exact_target_representative_at_least_16m_inside"
            ],
            "target_line_intersection_at_least_32m": selected["gates"][
                "exact_target_line_intersection_at_least_32m"
            ],
            "crop_soil_source_coverage_complete": crop_soil_complete,
            "crop_geology_decoded_coverage_complete": crop_geology_complete,
            "solve_domain_fraction_at_least_0_25": selected["gates"][
                "solve_domain_fraction_at_least_0_25"
            ],
            "target_feature_solve_cells_at_least_32": selected["gates"][
                "target_feature_solve_cells_at_least_32"
            ],
            "supported_crop_fraction_at_least_0_25": selected["gates"][
                "material_supported_fraction_at_least_0_25"
            ],
            "target_point_is_valid_non_unknown_target_solve_and_material_supported": target_point_active,
            "unsupported_material_or_unknown_cells_remain_fail_closed_abstention": True,
            "retained_dated_rgb_cir_available_in_bound_condition_bundle": bool(
                site["orthophoto"]["complete"]
            ),
        },
        "site_identity_provenance": {
            "source_recipe_sha256": prior["recipe_sha256"],
            "scope": "same exact ETAK feature identity; prior crop support and crop criteria excluded",
            "rechecked_fields": [
                "source_fid",
                "geometry_sha256",
                "etak_attributes",
            ],
        },
        "condition_relevance": {
            "source_condition_bundle_recipe_sha256": bundle["recipe_sha256"],
            "metrics": selected["metrics"],
            "gates": selected["gates"],
        },
    }
    artifact: dict[str, Any] = {
        "schema_version": "laas.erodible-slope-replacement-selection/3",
        "recipe_sha256": recipe_sha256,
        "supersedes_recipe_sha256": prior["recipe_sha256"],
        "status": "development_crop_refined_from_bound_condition_arrays_without_imagery_or_human_geometry_inspection",
        "development_replacement": development,
        "sealed_ood_replacement": prior["sealed_ood_replacement"],
        "sealed_ood_disclosure_policy": prior["sealed_ood_disclosure_policy"],
        "rejected_development_c_crops": [old_result],
        "same_feature_evaluations_until_first_pass": evaluations,
        "prior_sites": prior["prior_sites"],
        "recipe": recipe,
    }
    parent = output_root or DATA_WORK / "microtopography/erodible-slope/site-selection/sha256"
    root = parent / recipe_sha256
    path = root / "selection.json"
    amendment = _amendment(artifact)
    selection_text = json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != selection_text:
            raise RuntimeError("existing condition-relevance selection differs")
        if (root / "amendment-draft.md").read_text(encoding="utf-8") != amendment:
            raise RuntimeError("existing condition-relevance amendment differs")
        return path
    temporary = parent / f".{recipe_sha256}.{os.getpid()}.tmp"
    temporary.mkdir(parents=True)
    (temporary / "selection.json").write_text(selection_text, encoding="utf-8")
    (temporary / "amendment-draft.md").write_text(amendment, encoding="utf-8")
    parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(root)
    return path


if __name__ == "__main__":
    print(refine_development_crop())
