"""Freeze replacement erodible-slope sites before human geometry/pixel inspection."""
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
from ....conditions.soil.profile import parse_humus_profile, parse_texture_profile

_REPO = ASSET_GEN_ROOT.parent
_BASE_CONFIG = ASSET_GEN_ROOT / "config" / "base.toml"
_GRID_IMPLEMENTATION = ASSET_GEN_ROOT / "src" / "assetgen" / "grid.py"
_ETAK = DATA_IN / "etak" / "ETAK_EESTI_GPKG.gpkg"
_SOIL = DATA_IN / "soil" / "mullakaart" / "Mullakaart.shp"
_GEOLOGY_ROOT = (
    _REPO
    / "docs/deep-research/microtopography-generation/library/data/egt/"
    "pinnakate-200k"
)
_GEOLOGY = _GEOLOGY_ROOT / "q_avamus_a_200t.shp"
_DOMAIN = (
    DATA_WORK
    / "terrain/conditions/geology/egt-surficial-200k-domains/sha256/"
    "8d5c7ba4ee72ded6f5b1bd3ce3de4604b0c53c524c7a7a8500b7999aaa689a71/"
    "snapshot.json"
)
_LEAKAGE_BUNDLE = (
    DATA_WORK
    / "microtopography/erodible-slope/conditions/sha256/"
    "efc5771a92ce40311c38431336f73e2cffc37d3e03e5d86164dff04c5c44619a/"
    "bundle.json"
)
_ORTHOPHOTO = {
    sheet: {
        kind: DATA_IN / "orthophoto" / kind / sheet / "retained.json"
        for kind in ("rgb", "cir")
    }
    for sheet in ("54472", "54481")
}
_SHEETS = {
    "54472": (675000.0, 6440000.0, 680000.0, 6445000.0),
    "54481": (680000.0, 6440000.0, 685000.0, 6445000.0),
}

# These are evidence-leakage exclusions, not scene-specific morphology rules.
_EXCLUDED_POINTS = {
    1826743: (680551.76, 6444450.80, "development_a_and_taevaskoda"),
    1826691: (679692.03, 6442784.16, "development_b_rejected_unknown_humus"),
    9688685: (681429.34, 6443825.32, "orajogi_consumed"),
}
_EXCLUDED_ENLARGED_DOMAINS = {
    "development_a": (679969.0, 6443925.0, 681149.0, 6444998.0),
    "development_b_rejected": (678916.0, 6442122.0, 680459.0, 6443429.0),
}
_MIN_EXISTING_DISTANCE_M = 1000.0
_MIN_OOD_SEPARATION_M = 1024.0
_MIN_LENGTH_M = 20.0
_MAX_LENGTH_M = 600.0
_OUTPUT_CROP_M = 128.0
_MIN_CHUNK_EDGE_CLEARANCE_M = 16.0
_MIN_TARGET_LINE_IN_CHUNK_M = 32.0
_MIN_SUPPORTED_CROP_FRACTION = 0.25
_PROSPECTIVE_DOMAIN_MARGIN_M = 512.0
_COVERAGE_TOLERANCE_M2 = 0.01
_COARSE_TEXTURES = {
    "sand",
    "fine_sand",
    "silty_fine_sand",
    "silty_sand",
    "sandy_loam",
    "fine_sandy_loam",
    "silty_sandy_loam",
}
_COHESIVE_TEXTURES = {
    "sandy_loam",
    "fine_sandy_loam",
    "silty_sandy_loam",
    "loam",
    "fine_sandy_loam_to_loam",
    "silty_loam",
    "clay",
}


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


def _bundle_identity(path: Path) -> list[dict[str, Any]]:
    suffixes = (".shp", ".shx", ".dbf", ".prj", ".cpg", ".sbn", ".sbx", ".shp.xml")
    stem = path.with_suffix("")
    return [_identity(Path(str(stem) + suffix)) for suffix in suffixes if Path(str(stem) + suffix).is_file()]


def _read(
    path: Path,
    *,
    layer: str | None,
    bbox: tuple[float, ...],
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...], np.ndarray]:
    metadata, fids, wkbs, values = pyogrio.raw.read(
        path,
        layer=layer,
        bbox=bbox,
        columns=columns,
        return_fids=True,
    )
    if str(metadata["crs"]) != "EPSG:3301" or fids is None or wkbs is None:
        raise ValueError(f"invalid spatial source: {path}")
    geometries = np.asarray(
        [shapely.force_2d(shapely.from_wkb(bytes(wkb))) for wkb in wkbs],
        dtype=object,
    )
    return fids, geometries, values, wkbs


def _load_orthophoto_evidence() -> dict[str, dict[str, dict[str, Any]]]:
    evidence: dict[str, dict[str, dict[str, Any]]] = {}
    for sheet, kinds in _ORTHOPHOTO.items():
        evidence[sheet] = {}
        for kind, path in kinds.items():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("sheet") != sheet or payload.get("product") != kind:
                raise ValueError(f"retained orthophoto manifest identity differs: {path}")
            extraction = _REPO / "asset-gen/data/in/orthophoto" / payload["extraction"]["root"]
            tiffs = sorted(extraction.glob("*.tif"))
            if len(tiffs) != 1:
                raise ValueError(f"retained orthophoto TIFF is not exactly one file: {path}")
            evidence[sheet][kind] = {
                "manifest": _identity(path),
                "capture_date": payload["captureDate"],
                "archive_sha256": payload["archive"]["sha256"],
                # Availability only: selection never opens image pixels.
                "tiff": _identity(tiffs[0]),
            }
    return evidence


def _surface_texture(loimis: dict[str, Any], huumus: dict[str, Any]) -> str | None:
    if loimis.get("status") != "parsed_complete_official_grammar":
        return None
    if huumus.get("status") != "parsed_complete_official_grammar":
        return None
    expressions = loimis.get("expressions")
    if not isinstance(expressions, list) or len(expressions) != 1:
        return None
    layers = expressions[0].get("layers")
    if not isinstance(layers, list) or not layers:
        return None
    material = layers[0].get("material")
    if not isinstance(material, dict) or material.get("kind") not in {
        "mineral_fine_earth",
        "mineral_skeletal_material",
    }:
        return None
    texture = material.get("texture")
    family = texture.get("family") if isinstance(texture, dict) else None
    if not isinstance(family, str):
        return None
    skeleton = material.get("skeleton")
    if isinstance(skeleton, dict):
        content = skeleton.get("content_class")
        if content == "ungraded_or_over_70_percent" or (
            isinstance(content, int) and content >= 5
        ):
            return None
    for expression in huumus.get("expressions", []):
        for sequence in expression.get("sequences", []):
            if any(term.get("kind") == "peat_horizon" for term in sequence.get("terms", [])):
                return None
    return family


def _material_compatible(texture: str, lithology: int, genesis: int) -> bool:
    sand = texture in _COARSE_TEXTURES and lithology == 40 and genesis in {10, 30, 40, 60, 100}
    till = texture in _COHESIVE_TEXTURES and lithology == 50 and genesis == 50
    return sand or till


def _eligible_slope_role(tyyp_t: Any, kaldaastang_t: Any) -> bool:
    return tyyp_t == "Looduslik järsak" or (
        tyyp_t == "Nõlv" and kaldaastang_t == "Jah"
    )


def _coverage_complete(geometries: list[shapely.Geometry], window: shapely.Geometry) -> bool:
    if not geometries:
        return False
    covered = shapely.union_all([geometry.intersection(window) for geometry in geometries])
    return float(window.area - covered.area) <= _COVERAGE_TOLERANCE_M2


def _select_output_chunk(
    grid: Any,
    geometry: shapely.Geometry,
) -> tuple[shapely.Point, ChunkId, tuple[int, int, int, int], float] | None:
    choices: list[
        tuple[float, int, int, shapely.Point, ChunkId, tuple[int, int, int, int]]
    ] = []
    for chunk in chunks_covering_bbox_en(grid, geometry.bounds, -2):
        bounds = chunk_bounds_en(grid, chunk)
        chunk_box = shapely.box(*bounds)
        line_in_chunk = geometry.intersection(chunk_box)
        line_length = float(line_in_chunk.length)
        if line_length < _MIN_TARGET_LINE_IN_CHUNK_M:
            continue
        inner = shapely.box(
            bounds[0] + _MIN_CHUNK_EDGE_CLEARANCE_M,
            bounds[1] + _MIN_CHUNK_EDGE_CLEARANCE_M,
            bounds[2] - _MIN_CHUNK_EDGE_CLEARANCE_M,
            bounds[3] - _MIN_CHUNK_EDGE_CLEARANCE_M,
        )
        inner_parts = [
            part
            for part in shapely.get_parts(geometry.intersection(inner))
            if part.geom_type in {"LineString", "LinearRing"} and part.length > 0.0
        ]
        if not inner_parts:
            continue
        inner_parts.sort(
            key=lambda part: (-float(part.length), hashlib.sha256(part.wkb).hexdigest())
        )
        point = inner_parts[0].interpolate(0.5, normalized=True)
        clearance = min(
            float(point.x) - bounds[0],
            bounds[2] - float(point.x),
            float(point.y) - bounds[1],
            bounds[3] - float(point.y),
        )
        if clearance + 1e-9 < _MIN_CHUNK_EDGE_CLEARANCE_M:
            continue
        choices.append((-line_length, chunk.cz, chunk.cx, point, chunk, bounds))
    if not choices:
        return None
    choices.sort(key=lambda row: row[:3])
    negative_length, _, _, point, chunk, bounds = choices[0]
    return point, chunk, bounds, -negative_length


def _candidate_support(
    point: shapely.Point,
    crop: shapely.Geometry,
    *,
    soils: np.ndarray,
    soil_values: tuple[np.ndarray, ...],
    soil_tree: shapely.STRtree,
    geologies: np.ndarray,
    geology_values: tuple[np.ndarray, ...],
    geology_tree: shapely.STRtree,
    decoded: dict[str, dict[int, str]],
    rejections: dict[str, int],
) -> dict[str, Any] | None:
    soil_indices = [
        int(index)
        for index in soil_tree.query(crop, predicate="intersects")
        if soils[int(index)].intersection(crop).area > 0.0
    ]
    geology_indices = [
        int(index)
        for index in geology_tree.query(crop, predicate="intersects")
        if geologies[int(index)].intersection(crop).area > 0.0
    ]
    if not _coverage_complete([soils[index] for index in soil_indices], crop):
        rejections["incomplete_soil_coverage"] += 1
        return None
    if not _coverage_complete([geologies[index] for index in geology_indices], crop):
        rejections["incomplete_geology_coverage"] += 1
        return None

    textures: dict[int, str] = {}
    unsupported_soil_features = 0
    for index in soil_indices:
        loimis1 = parse_texture_profile(soil_values[0][index])
        loimis2 = parse_texture_profile(soil_values[1][index])
        huumus = parse_humus_profile(soil_values[2][index])
        if loimis2["status"] not in {"parsed_complete_official_grammar", "missing"}:
            unsupported_soil_features += 1
            continue
        texture = _surface_texture(loimis1, huumus)
        if texture is None:
            unsupported_soil_features += 1
            continue
        textures[index] = texture

    decoded_geology: dict[int, tuple[int, int]] = {}
    for index in geology_indices:
        lithology = int(geology_values[0][index])
        genesis = int(geology_values[1][index])
        if lithology not in decoded["lito200"] or genesis not in decoded["genees200"]:
            rejections["undecoded_geology"] += 1
            return None
        if lithology in {997, 998}:
            rejections["unknown_geology"] += 1
            return None
        decoded_geology[index] = (lithology, genesis)

    supported_parts: list[shapely.Geometry] = []
    for soil_index, texture in textures.items():
        for geology_index in geology_indices:
            overlap = soils[soil_index].intersection(geologies[geology_index]).intersection(crop)
            if overlap.area > _COVERAGE_TOLERANCE_M2 and _material_compatible(
                texture, *decoded_geology[geology_index]
            ):
                supported_parts.append(overlap)
    supported = shapely.union_all(supported_parts) if supported_parts else shapely.GeometryCollection()
    supported_fraction = float(supported.area / crop.area)
    if not supported.covers(point):
        rejections["target_point_not_material_supported"] += 1
        return None
    if supported_fraction < _MIN_SUPPORTED_CROP_FRACTION:
        rejections["supported_crop_fraction_below_minimum"] += 1
        return None
    return {
        "crop_bbox_en": [float(value) for value in crop.bounds],
        "soil_feature_count": len(soil_indices),
        "geology_feature_count": len(geology_indices),
        "surface_texture_families": sorted(set(textures.values())),
        "unsupported_soil_feature_count": unsupported_soil_features,
        "supported_crop_fraction": supported_fraction,
        "minimum_supported_crop_fraction": _MIN_SUPPORTED_CROP_FRACTION,
        "geology_pairs": sorted(
            {
                (
                    lithology,
                    decoded["lito200"][lithology],
                    genesis,
                    decoded["genees200"][genesis],
                )
                for lithology, genesis in decoded_geology.values()
            }
        ),
    }


def _ood_abstention_proof(
    point: shapely.Point,
    *,
    soils: np.ndarray,
    soil_values: tuple[np.ndarray, ...],
    soil_tree: shapely.STRtree,
    geologies: np.ndarray,
    geology_values: tuple[np.ndarray, ...],
    geology_tree: shapely.STRtree,
    decoded: dict[str, dict[int, str]],
) -> dict[str, Any] | None:
    soil_indices = [int(index) for index in soil_tree.query(point, predicate="intersects")]
    geology_indices = [int(index) for index in geology_tree.query(point, predicate="intersects")]
    if len(soil_indices) != 1 or len(geology_indices) != 1:
        return None
    soil_index = soil_indices[0]
    geology_index = geology_indices[0]
    lithology = int(geology_values[0][geology_index])
    genesis = int(geology_values[1][geology_index])
    if lithology not in decoded["lito200"] or genesis not in decoded["genees200"]:
        return None
    if lithology in {997, 998}:
        return None
    loimis1 = parse_texture_profile(soil_values[0][soil_index])
    loimis2 = parse_texture_profile(soil_values[1][soil_index])
    huumus = parse_humus_profile(soil_values[2][soil_index])
    texture = _surface_texture(loimis1, huumus)
    if loimis1["status"] != "parsed_complete_official_grammar":
        reason = "loimis1_missing_or_unparseable"
    elif huumus["status"] != "parsed_complete_official_grammar":
        reason = "huumus_missing_or_unparseable"
    elif loimis2["status"] not in {"parsed_complete_official_grammar", "missing"}:
        reason = "loimis2_unparseable"
    elif texture is None:
        reason = "surface_nonmineral_peat_or_unresolved"
    elif not _material_compatible(texture, lithology, genesis):
        reason = "soil_geology_pair_outside_reviewed_c1_rules"
    else:
        return None
    return {
        "target_is_outside_frozen_c1_condition_envelope": True,
        "abstention_reason": reason,
        "geology_decoded_from_frozen_official_domain": True,
        "positive_morphology_credit": False,
        "allowed_result": "bounded_abstention_only",
    }


def _disclosed_candidate(
    candidate: dict[str, Any],
    *,
    sealed: bool,
) -> dict[str, Any]:
    common = {
        "etak_id": candidate["etak_id"],
        "source_fid": candidate["source_fid"],
        "geometry_sha256": candidate["geometry_sha256"],
        "representative_point_en": candidate["representative_point_en"],
        "primary_sheet": candidate["primary_sheet"],
        "required_orthophoto_sheets": candidate["required_orthophoto_sheets"],
        "output_chunk": candidate["output_chunk"],
        "criterion_pass_proof": candidate["criterion_pass_proof"],
    }
    if sealed:
        return common
    return {
        **common,
        "length_m": candidate["length_m"],
        "etak_attributes": candidate["etak_attributes"],
        "support": candidate["support"],
    }


def _amendment_markdown(artifact: dict[str, Any]) -> str:
    development = artifact["development_replacement"]
    ood = artifact["sealed_ood_replacement"]
    development_point = development["representative_point_en"]
    ood_point = ood["representative_point_en"]
    return f"""# Required Stage 2E Role Amendment Draft

- Bind replacement selection `{artifact['recipe_sha256']}` as the authority for site-role identity; its `selection.json` must be hash-bound by preregistration.
- Retain `development_a` (`ETAK 1826743`) as Development. Replace rejected `development_b` with `development_c` (`ETAK {development['etak_id']}`, EPSG:3301 `{development_point[0]:.6f}, {development_point[1]:.6f}`, sheet `{development['primary_sheet']}`, LOD -2 chunk `{development['output_chunk']['cx']},{development['output_chunk']['cz']}` with bounds `{development['output_chunk']['bounds_en']}`, geometry SHA-256 `{development['geometry_sha256']}`). Development C is R0 process-development evidence only, never target truth or transfer evidence.
- Development C is non-abstaining at its frozen target and has `{development['support']['supported_crop_fraction']:.6f}` compatible support in the 128 m crop. Unsupported cells remain explicit abstention; materialization must still prove solve-domain coverage, outlet/collar integrity, and invariance before C1 runs.
- Preserve `development_b` (`ETAK 1826691`) as rejected because unknown Huumus intersects support; never default or impute its material rule.
- Replace consumed `orajogi_ood` with sealed `erodible_slope_ood_2` (`ETAK {ood['etak_id']}`, EPSG:3301 `{ood_point[0]:.6f}, {ood_point[1]:.6f}`, sheet `{ood['primary_sheet']}`, LOD -2 chunk `{ood['output_chunk']['cx']},{ood['output_chunk']['cz']}` with bounds `{ood['output_chunk']['bounds_en']}`, geometry SHA-256 `{ood['geometry_sha256']}`). Its only allowed result is bounded abstention and it receives no positive morphology credit.
- Preserve `orajogi_ood` (`ETAK 9688685`) as consumed and disqualified. Update `sealed_ood_gate.site_id` to `erodible_slope_ood_2`; do not open its pixels, source-feature bounds/shape, detailed geometry, or render before every existing one-shot gate is frozen.
- No solver coefficient, runtime behavior, wire format, production-owner claim, or release eligibility changes are authorized by this amendment.
"""


def select_replacements(output_root: Path | None = None) -> Path:
    bbox = (675000.0, 6440000.0, 685000.0, 6445000.0)
    grid = load_base().grid
    orthophoto = _load_orthophoto_evidence()
    domains = json.loads(_DOMAIN.read_text(encoding="utf-8"))["domains"]
    decoded = {
        row["field_name"]: {int(item["code"]): item["name"] for item in row["coded_values"]}
        for row in domains
    }
    slope_fids, slopes, slope_values, slope_wkbs = _read(
        _ETAK,
        layer="E_102_nolv_j",
        bbox=bbox,
        columns=["etak_id", "tyyp_t", "kaldaastang_t", "muutmisaeg", "geom_muutmisaeg"],
    )
    soil_fids, soils, soil_values, _ = _read(
        _SOIL,
        layer=None,
        bbox=bbox,
        columns=["Loimis1", "Loimis2", "Huumus"],
    )
    geology_fids, geologies, geology_values, _ = _read(
        _GEOLOGY,
        layer=None,
        bbox=bbox,
        columns=["lito200", "genees200"],
    )
    del soil_fids, geology_fids
    soil_tree = shapely.STRtree(soils)
    geology_tree = shapely.STRtree(geologies)
    sheet_boxes = {sheet: shapely.box(*bounds) for sheet, bounds in _SHEETS.items()}
    available_domain = shapely.union_all(list(sheet_boxes.values()))
    leakage_domains = shapely.union_all(
        [shapely.box(*bounds) for bounds in _EXCLUDED_ENLARGED_DOMAINS.values()]
    )
    leakage_point_buffers = shapely.union_all(
        [shapely.Point(e, n).buffer(_MIN_EXISTING_DISTANCE_M) for e, n, _ in _EXCLUDED_POINTS.values()]
    )

    # Physical/evidence filters run before the stable etak_id ordering.
    candidates: list[dict[str, Any]] = []
    ood_candidates: list[dict[str, Any]] = []
    filter_counts = {
        "source_features": len(slopes),
        "eligible_natural_or_bank_escarpment": 0,
        "valid_length": 0,
        "outside_leakage_with_retained_crop": 0,
        "complete_supported_crop": 0,
    }
    support_rejections = {
        "incomplete_soil_coverage": 0,
        "incomplete_geology_coverage": 0,
        "undecoded_geology": 0,
        "unknown_geology": 0,
        "target_point_not_material_supported": 0,
        "supported_crop_fraction_below_minimum": 0,
    }
    for index in range(len(slopes)):
        etak_id = int(slope_values[0][index])
        geometry = slopes[index]
        if etak_id in _EXCLUDED_POINTS or not _eligible_slope_role(
            slope_values[1][index], slope_values[2][index]
        ):
            continue
        filter_counts["eligible_natural_or_bank_escarpment"] += 1
        if not geometry.is_valid or geometry.geom_type not in {"LineString", "MultiLineString"}:
            continue
        length_m = float(geometry.length)
        if not (_MIN_LENGTH_M <= length_m <= _MAX_LENGTH_M):
            continue
        filter_counts["valid_length"] += 1
        output_selection = _select_output_chunk(grid, geometry)
        if output_selection is None:
            continue
        point, output_chunk, output_bounds, target_line_in_chunk_m = output_selection
        if output_bounds[2] - output_bounds[0] != _OUTPUT_CROP_M:
            raise ValueError("canonical LOD -2 footprint differs from frozen 128 m crop")
        crop = shapely.box(*output_bounds)
        prospective_domain = shapely.box(
            geometry.bounds[0] - _PROSPECTIVE_DOMAIN_MARGIN_M,
            geometry.bounds[1] - _PROSPECTIVE_DOMAIN_MARGIN_M,
            geometry.bounds[2] + _PROSPECTIVE_DOMAIN_MARGIN_M,
            geometry.bounds[3] + _PROSPECTIVE_DOMAIN_MARGIN_M,
        )
        if (
            not available_domain.covers(prospective_domain)
            or geometry.intersects(leakage_point_buffers)
            or crop.intersects(leakage_point_buffers)
            or prospective_domain.intersects(leakage_domains)
        ):
            continue
        primary_sheets = [sheet for sheet, box in sheet_boxes.items() if box.covers(point)]
        required_sheets = [sheet for sheet, box in sheet_boxes.items() if box.covers(prospective_domain)]
        if len(primary_sheets) != 1 or len(required_sheets) != 1:
            continue
        if any(kind not in orthophoto[sheet] for sheet in required_sheets for kind in ("rgb", "cir")):
            continue
        filter_counts["outside_leakage_with_retained_crop"] += 1
        base_candidate = {
            "etak_id": etak_id,
            "source_fid": int(slope_fids[index]),
            "geometry_sha256": hashlib.sha256(bytes(slope_wkbs[index])).hexdigest(),
            "representative_point_en": [float(point.x), float(point.y)],
            "primary_sheet": primary_sheets[0],
            "required_orthophoto_sheets": sorted(required_sheets),
            "output_chunk": {
                "lod": output_chunk.lod,
                "cx": output_chunk.cx,
                "cz": output_chunk.cz,
                "bounds_en": list(output_bounds),
                "target_line_intersection_m": target_line_in_chunk_m,
                "minimum_target_line_intersection_m": _MIN_TARGET_LINE_IN_CHUNK_M,
                "representative_point_minimum_edge_clearance_m": _MIN_CHUNK_EDGE_CLEARANCE_M,
            },
            "length_m": length_m,
            "etak_attributes": {
                "tyyp_t": slope_values[1][index],
                "kaldaastang_t": slope_values[2][index],
                "muutmisaeg": str(slope_values[3][index]),
                "geom_muutmisaeg": str(slope_values[4][index]),
            },
        }
        support = _candidate_support(
            point,
            crop,
            soils=soils,
            soil_values=soil_values,
            soil_tree=soil_tree,
            geologies=geologies,
            geology_values=geology_values,
            geology_tree=geology_tree,
            decoded=decoded,
            rejections=support_rejections,
        )
        if support is None:
            ood_proof = _ood_abstention_proof(
                point,
                soils=soils,
                soil_values=soil_values,
                soil_tree=soil_tree,
                geologies=geologies,
                geology_values=geology_values,
                geology_tree=geology_tree,
                decoded=decoded,
            )
            if ood_proof is not None:
                ood_candidates.append(
                    {
                        **base_candidate,
                        "criterion_pass_proof": {
                            "eligible_natural_or_bank_escarpment": True,
                            "valid_single_valued_line_metadata": True,
                            "output_crop_is_exact_canonical_lod_minus_2_chunk": True,
                            "representative_point_at_least_16m_inside_chunk": True,
                            "target_line_intersection_at_least_32m": True,
                            "outside_consumed_and_development_leakage": True,
                            "retained_dated_rgb_cir_available": True,
                            **ood_proof,
                        },
                    }
                )
            continue
        filter_counts["complete_supported_crop"] += 1
        candidates.append(
            {
                **base_candidate,
                "support": support,
                "criterion_pass_proof": {
                    "eligible_natural_or_bank_escarpment": True,
                    "valid_single_valued_line_metadata": True,
                    "output_crop_m": _OUTPUT_CROP_M,
                    "output_crop_is_exact_canonical_lod_minus_2_chunk": True,
                    "representative_point_at_least_16m_inside_chunk": True,
                    "target_line_intersection_at_least_32m": True,
                    "prospective_domain_margin_m": _PROSPECTIVE_DOMAIN_MARGIN_M,
                    "prospective_domain_inside_one_retained_sheet": True,
                    "outside_consumed_and_development_leakage": True,
                    "retained_dated_rgb_cir_available": True,
                    "crop_soil_coverage_complete": True,
                    "crop_geology_coverage_complete": True,
                    "target_point_loimis1_huumus_complete_nonmissing": True,
                    "target_point_material_pair_supported_by_c1": True,
                    "supported_crop_fraction_at_least_0_25": True,
                    "unsupported_crop_cells_remain_fail_closed_abstention": True,
                },
            }
        )
    candidates.sort(key=lambda row: row["etak_id"])
    ood_candidates.sort(key=lambda row: row["etak_id"])
    if not candidates:
        raise RuntimeError(
            "retained 54481/54472 sheets have no eligible replacement Development site; "
            f"aggregate filter counts={filter_counts}; support rejections={support_rejections}"
        )
    development = candidates[0]
    ood = next(
        (
            row
            for row in ood_candidates
            if np.hypot(
                row["representative_point_en"][0] - development["representative_point_en"][0],
                row["representative_point_en"][1] - development["representative_point_en"][1],
            )
            >= _MIN_OOD_SEPARATION_M
        ),
        None,
    )
    if ood is None:
        maximum_separation = max(
            (
                float(
                    np.hypot(
                        row["representative_point_en"][0] - development["representative_point_en"][0],
                        row["representative_point_en"][1] - development["representative_point_en"][1],
                    )
                )
                for row in ood_candidates
            ),
            default=0.0,
        )
        raise RuntimeError(
            "retained sheets have no distinct eligible sealed OOD replacement; "
            f"supported_count={len(candidates)}, ood_count={len(ood_candidates)}, "
            f"maximum_separation_m={maximum_separation:.3f}"
        )

    source_identities = {
        "etak": [_identity(_ETAK)],
        "soil_bundle": _bundle_identity(_SOIL),
        "geology_bundle": _bundle_identity(_GEOLOGY),
        "egt_domain": _identity(_DOMAIN),
        "leakage_authority": _identity(_LEAKAGE_BUNDLE),
        "orthophoto_availability": orthophoto,
        "grid_config": _identity(_BASE_CONFIG),
        "grid_implementation": _identity(_GRID_IMPLEMENTATION),
    }
    recipe = {
        "schema_version": "laas.erodible-slope-replacement-selection/2.recipe",
        "sources": source_identities,
        "implementation": _identity(Path(__file__)),
        "ordering": "etak_id_ascending_after_all_physical_and_evidence_filters",
        "criteria": {
            "retained_sheet_union": _SHEETS,
            "type": "Looduslik jarask, or Nolv with kaldaastang=Jah",
            "length_m": [_MIN_LENGTH_M, _MAX_LENGTH_M],
            "output_crop_m": _OUTPUT_CROP_M,
            "output_crop": "exact canonical LOD -2 ChunkId from load_base().grid at the automated representative point",
            "representative_point_selection": "choose the chunk with greatest target-line intersection after requiring at least 32 m in the chunk; ties use cz then cx; choose midpoint of the longest line part inside the chunk inset by 16 m",
            "minimum_chunk_edge_clearance_m": _MIN_CHUNK_EDGE_CLEARANCE_M,
            "minimum_target_line_in_chunk_m": _MIN_TARGET_LINE_IN_CHUNK_M,
            "prospective_domain_margin_m": _PROSPECTIVE_DOMAIN_MARGIN_M,
            "minimum_existing_site_distance_m": _MIN_EXISTING_DISTANCE_M,
            "excluded_enlarged_domains_en": _EXCLUDED_ENLARGED_DOMAINS,
            "minimum_development_ood_separation_m": _MIN_OOD_SEPARATION_M,
            "minimum_development_ood_separation_basis": "eight 128 m output-crop widths",
            "soil": "crop union complete; target point has parsed nonmissing Loimis1 and Huumus, parsed-or-missing Loimis2, mineral surface and non-peat humus; unsupported polygons remain abstention",
            "geology": "crop union complete; target point maps to reviewed C1 sand_mineral or till_mineral rule; at least 25 percent of crop has a compatible soil/geology overlap",
            "orthophoto": "retained RGB and CIR manifests and hash-bound extracted TIFFs available for the single sheet covering the prospective domain; image pixels never opened",
            "forbidden_before_freeze": [
                "human_pixel_inspection",
                "human_detailed_candidate_geometry_inspection",
                "manual_visual_ranking",
                "parameter_tuning",
            ],
        },
    }
    recipe_sha256 = hashlib.sha256(
        json.dumps(recipe, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    artifact: dict[str, Any] = {
        "schema_version": "laas.erodible-slope-replacement-selection/2",
        "recipe_sha256": recipe_sha256,
        "status": "identities_frozen_after_automated_metadata_geometry_filtering_before_any_human_pixel_or_detailed_geometry_inspection",
        "development_replacement": _disclosed_candidate(development, sealed=False),
        "sealed_ood_replacement": _disclosed_candidate(ood, sealed=True),
        "sealed_ood_disclosure_policy": "identity/hash/representative coordinate/canonical chunk and criterion proof only; no source-feature bounds or shape, render, pixels, or manual inspection before the one-shot gate",
        "prior_sites": [
            {
                "id": "development_b",
                "etak_id": 1826691,
                "state": "rejected",
                "reason": "unknown Huumus intersected its process support; no material-rule default allowed",
            },
            {
                "id": "orajogi_ood",
                "etak_id": 9688685,
                "state": "consumed_and_disqualified",
                "reason": "geometry/evidence was inspected before recipe freeze",
            },
        ],
        "eligible_count_not_exported_or_ranked": len(candidates),
        "ood_eligible_count_not_exported_or_ranked": len(ood_candidates),
        "aggregate_filter_counts": filter_counts,
        "aggregate_support_rejections": support_rejections,
        "recipe": recipe,
    }
    parent = output_root or DATA_WORK / "microtopography/erodible-slope/site-selection/sha256"
    root = parent / recipe_sha256
    path = root / "selection.json"
    amendment = _amendment_markdown(artifact)
    amendment_path = root / "amendment-draft.md"
    selection_text = json.dumps(
        artifact, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != selection_text:
            raise RuntimeError("existing replacement selection differs")
        if not amendment_path.is_file() or amendment_path.read_text(encoding="utf-8") != amendment:
            raise RuntimeError("existing replacement amendment draft differs")
        return path
    temporary = parent / f".{recipe_sha256}.{os.getpid()}.tmp"
    temporary.mkdir(parents=True)
    (temporary / "selection.json").write_text(selection_text, encoding="utf-8")
    (temporary / "amendment-draft.md").write_text(amendment, encoding="utf-8")
    parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(root)
    return path


if __name__ == "__main__":
    print(select_replacements())
