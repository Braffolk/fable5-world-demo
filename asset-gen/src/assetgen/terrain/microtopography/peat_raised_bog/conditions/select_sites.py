"""Freeze raised-bog development and sealed-audit sites without pixel inspection."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyogrio.raw
import shapely

from .....config import ASSET_GEN_ROOT, DATA_IN, DATA_WORK, load_base
from ....conditions.soil.profile import parse_humus_profile, parse_texture_profile


REPOSITORY_ROOT = ASSET_GEN_ROOT.parent
PREREGISTRATION = (
    ASSET_GEN_ROOT
    / "config/microtopography/peat-raised-bog/bundle-preregistration-v1.json"
)
PREREGISTRATION_SHA256 = (
    "21d972723f7c2641ce6a9fb7c15e171e85f3378dc83863cc93cb833be94bf115"
)
ETAK = DATA_IN / "etak/ETAK_EESTI_GPKG.gpkg"
SOIL = DATA_IN / "soil/mullakaart/Mullakaart.shp"
SOIL_DOCUMENTS = (
    DATA_IN / "soil/mullakaardi_seletuskiri.pdf",
    DATA_IN / "soil/mullalegend.pdf",
)
GEOLOGY = (
    REPOSITORY_ROOT
    / "docs/deep-research/microtopography-generation/library/data/egt/"
    "pinnakate-200k/q_avamus_a_200t.shp"
)
GEOLOGY_DOMAINS = (
    DATA_WORK
    / "terrain/conditions/geology/egt-surficial-200k-domains/sha256/"
    "8d5c7ba4ee72ded6f5b1bd3ce3de4604b0c53c524c7a7a8500b7999aaa689a71/"
    "snapshot.json"
)
OUTPUT_ROOT = DATA_WORK / "microtopography/peat-raised-bog/site-selection/sha256"

MIRE_LAYER = "E_306_margala_a"
FOREST_LAYER = "E_305_puittaimestik_a"
CUT_PEAT_LAYER = "E_307_turbavali_a"
WATER_AREA_LAYER = "E_203_vooluveekogu_a"
WATER_LINE_LAYER = "E_203_vooluveekogu_j"
BUILDING_LAYER = "E_401_hoone_ka"
ROAD_AREA_LAYER = "E_501_tee_a"
ROAD_LINE_LAYER = "E_501_tee_j"

CORE_M = 128.0
HALO_M = 64.0
HALO_FOOTPRINT_M = CORE_M + 2.0 * HALO_M
CORE_AREA_M2 = CORE_M * CORE_M
COVERAGE_TOLERANCE_M2 = 0.01
TAEVASKODA_GAME_XZ = (311123.082, 190723.435)
LEAKAGE_EXCLUSION_RADIUS_M = 1024.0
DITCH_TYPES = {20, 40, 50}  # canal, main ditch, ditch
PURE_RAISED_BOG_PREFIX = "R"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPOSITORY_ROOT.resolve()))


def _identity(path: Path) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _bundle_identity(path: Path) -> list[dict[str, Any]]:
    stem = path.with_suffix("")
    suffixes = (".shp", ".shx", ".dbf", ".prj", ".cpg", ".sbn", ".sbx", ".shp.xml")
    return [
        _identity(Path(f"{stem}{suffix}"))
        for suffix in suffixes
        if Path(f"{stem}{suffix}").is_file()
    ]


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("ascii")


def _read(
    path: Path,
    *,
    layer: str | None = None,
    columns: Iterable[str] = (),
    bbox: tuple[float, float, float, float] | None = None,
    where: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    metadata, fids, wkbs, values = pyogrio.raw.read(
        path,
        layer=layer,
        columns=list(columns),
        bbox=bbox,
        where=where,
        return_fids=True,
    )
    if str(metadata.get("crs")) != "EPSG:3301" or fids is None or wkbs is None:
        raise ValueError(f"invalid EPSG:3301 vector source: {path} layer={layer}")
    geometries = np.asarray(
        [
            shapely.force_2d(shapely.from_wkb(bytes(wkb)))
            for wkb in wkbs
        ],
        dtype=object,
    )
    fields = [str(field) for field in metadata.get("fields", ())]
    return np.asarray(fids), geometries, dict(zip(fields, values, strict=True))


def _verify_preregistration() -> dict[str, Any]:
    actual = _sha256(PREREGISTRATION)
    if actual != PREREGISTRATION_SHA256:
        raise ValueError(f"raised-bog preregistration hash changed: {actual}")
    value = json.loads(PREREGISTRATION.read_bytes())
    if value.get("schema_version") != "laas.peat-raised-bog-r0-research-bundle-preregistration/1":
        raise ValueError("raised-bog preregistration schema changed")
    if value.get("bundle_id") != "peat-raised-bog-r0-research-bundle/1":
        raise ValueError("raised-bog bundle identity changed")
    selection = value.get("site_selection", {})
    if selection.get("state") != "blocked_until_content_addressed_selection_exists":
        raise ValueError("site-selection preregistration state changed")
    if selection.get("development", {}).get("count") != 1:
        raise ValueError("development-site count changed")
    if selection.get("sealed_abstention_audit", {}).get("count") != 1:
        raise ValueError("sealed-audit count changed")
    seam = value.get("whole_domain_and_seam_law", {})
    if seam.get("development_core_m") != [128, 128] or seam.get("fine_halo_min_m") != 64:
        raise ValueError("development core or halo changed")
    soil = value.get("surface_scope", {}).get("soil_semantics", {})
    expected = (
        "distinct_humus_horizon_not_applicable only when every bound soil component "
        "is raised-bog and its complete Loimis profile parses as peat; official retained "
        "soil documentation defines deep bog profiles as T-only and peat thickness/"
        "decomposition in the texture formula"
    )
    if soil.get("pure_raised_bog_blank_humus") != expected:
        raise ValueError("pure raised-bog Huumus semantics changed")
    if soil.get("mixed_or_nonbog_blank_humus") != "ownership_unknown":
        raise ValueError("mixed/non-bog blank-Huumus law changed")
    if soil.get("zero_imputation_forbidden") is not True:
        raise ValueError("zero-imputation prohibition changed")
    return value


def _all_peat(profile: dict[str, Any]) -> bool:
    expressions = profile.get("expressions", [])
    if not expressions:
        return False
    for expression in expressions:
        layers = expression.get("layers", [])
        if not layers:
            return False
        for layer in layers:
            if layer.get("material", {}).get("kind") != "peat":
                return False
            bottom = layer.get("bottom_depth_cm")
            if bottom is None or float(bottom.get("max_cm", 0)) <= 0:
                return False
    return True


def _raised_bog_components(attributes: dict[str, Any], index: int) -> bool:
    found = False
    for component in range(1, 5):
        code = attributes[f"Sif{component}"][index]
        share = attributes[f"Osa{component}"][index]
        if code is None or str(code).strip() == "":
            if share not in (None, 0):
                return False
            continue
        found = True
        if not str(code).strip().startswith(PURE_RAISED_BOG_PREFIX):
            return False
    return found


def _soil_support(attributes: dict[str, Any], index: int) -> dict[str, Any]:
    loimis1 = parse_texture_profile(attributes["Loimis1"][index])
    loimis2 = parse_texture_profile(attributes["Loimis2"][index])
    huumus = parse_humus_profile(attributes["Huumus"][index])
    complete_peat = (
        loimis1.get("status") == "parsed_complete_official_grammar"
        and _all_peat(loimis1)
        and loimis2.get("status") in {"parsed_complete_official_grammar", "missing"}
    )
    pure_bog = _raised_bog_components(attributes, index)
    if huumus.get("status") == "parsed_complete_official_grammar":
        humus_semantics = "explicit_complete_humus_or_peat_horizon_formula"
        humus_resolved = True
    elif huumus.get("status") == "missing" and pure_bog and complete_peat:
        humus_semantics = "distinct_humus_horizon_not_applicable_pure_bog_profile"
        humus_resolved = True
    else:
        humus_semantics = "ownership_unknown"
        humus_resolved = False
    return {
        "supported": bool(complete_peat and pure_bog and humus_resolved),
        "complete_peat": bool(complete_peat),
        "pure_raised_bog_components": bool(pure_bog),
        "humus_semantics": humus_semantics,
        "siffer": None if attributes["Siffer"][index] is None else str(attributes["Siffer"][index]),
        "loimis1": None if attributes["Loimis1"][index] is None else str(attributes["Loimis1"][index]),
        "loimis2": None if attributes["Loimis2"][index] is None else str(attributes["Loimis2"][index]),
        "huumus": None if attributes["Huumus"][index] is None else str(attributes["Huumus"][index]),
    }


def _union(geometries: Iterable[shapely.Geometry]) -> shapely.Geometry:
    values = [geometry for geometry in geometries if geometry is not None and not geometry.is_empty]
    return shapely.union_all(values) if values else shapely.GeometryCollection()


def _covered_by(geometry: shapely.Geometry, parts: Iterable[shapely.Geometry]) -> bool:
    covered = _union(part.intersection(geometry) for part in parts)
    return float(geometry.area - covered.area) <= COVERAGE_TOLERANCE_M2


def _layer_geometries(
    layer: str,
    bbox: tuple[float, float, float, float],
    columns: Iterable[str] = (),
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    _, geometries, attributes = _read(ETAK, layer=layer, bbox=bbox, columns=columns)
    return geometries, attributes


def _buffered_lines(
    geometries: np.ndarray,
    widths: np.ndarray | None,
    minimum_half_width: float,
) -> list[shapely.Geometry]:
    result: list[shapely.Geometry] = []
    for index, geometry in enumerate(geometries):
        width = None if widths is None else widths[index]
        try:
            half_width = max(minimum_half_width, 0.5 * float(width))
        except (TypeError, ValueError):
            half_width = minimum_half_width
        result.append(geometry.buffer(half_width, cap_style="flat"))
    return result


def _aligned_halos(geometry: shapely.Geometry) -> list[tuple[shapely.Geometry, shapely.Geometry]]:
    if geometry.is_empty:
        return []
    min_e, min_n, max_e, max_n = geometry.bounds
    half = 0.5 * HALO_FOOTPRINT_M
    eastings = np.arange(
        math.ceil((min_e + half) / CORE_M) * CORE_M,
        math.floor((max_e - half) / CORE_M) * CORE_M + 1.0,
        CORE_M,
        dtype=np.float64,
    )
    northings = np.arange(
        math.ceil((min_n + half) / CORE_M) * CORE_M,
        math.floor((max_n - half) / CORE_M) * CORE_M + 1.0,
        CORE_M,
        dtype=np.float64,
    )
    if eastings.size == 0 or northings.size == 0:
        return []
    rows: list[tuple[shapely.Geometry, shapely.Geometry]] = []
    for north in northings[::-1]:
        for east in eastings:
            halo = shapely.box(east - half, north - half, east + half, north + half)
            if geometry.covers(halo):
                core_half = 0.5 * CORE_M
                core = shapely.box(
                    east - core_half,
                    north - core_half,
                    east + core_half,
                    north + core_half,
                )
                rows.append((core, halo))
    return rows


def _geology_support(
    bbox: tuple[float, float, float, float],
    decoded: dict[str, dict[int, str]],
) -> tuple[shapely.Geometry, list[dict[str, Any]], int]:
    fids, geometries, attributes = _read(
        GEOLOGY,
        bbox=bbox,
        columns=("lito200", "genees200"),
    )
    accepted: list[shapely.Geometry] = []
    records: list[dict[str, Any]] = []
    unknown = 0
    for index, geometry in enumerate(geometries):
        lithology = int(attributes["lito200"][index])
        genesis = int(attributes["genees200"][index])
        known = (
            lithology in decoded["lito200"]
            and genesis in decoded["genees200"]
            and lithology not in {997, 998}
        )
        if known:
            accepted.append(geometry)
        else:
            unknown += 1
        records.append(
            {
                "source_fid": int(fids[index]),
                "lithology": lithology,
                "lithology_name": decoded["lito200"].get(lithology),
                "genesis": genesis,
                "genesis_name": decoded["genees200"].get(genesis),
                "known": known,
            }
        )
    return _union(accepted), records, unknown


def _candidate(
    *,
    geometry: shapely.Geometry,
    attributes: dict[str, np.ndarray],
    member_indices: list[int],
    source_fids: np.ndarray,
    decoded: dict[str, dict[int, str]],
    taevaskoda_exclusion: shapely.Geometry,
) -> dict[str, Any]:
    etak_ids = sorted(int(attributes["etak_id"][index]) for index in member_indices)
    component_fids = sorted(int(source_fids[index]) for index in member_indices)
    etak_id = etak_ids[0]
    identifier = f"etak-component-{etak_id:010d}"
    statuses = sorted({str(attributes["vajalik_t"][index]) for index in member_indices})
    base = {
        "normalized_mire_identifier": identifier,
        "minimum_etak_id": etak_id,
        "member_etak_ids": etak_ids,
        "source_fids": component_fids,
        "bounds_en": [float(value) for value in geometry.bounds],
        "candidate_area_m2": float(geometry.area),
        "geometry_sha256": hashlib.sha256(geometry.wkb).hexdigest(),
        "etak": {
            "types": sorted({str(attributes["tyyp_t"][index]) for index in member_indices}),
            "wooded_states": sorted({str(attributes["puis_t"][index]) for index in member_indices}),
            "record_statuses": statuses,
            "modified": sorted({str(attributes["muutmisaeg"][index]) for index in member_indices}),
            "geometry_modified": sorted({str(attributes["geom_muutmisaeg"][index]) for index in member_indices}),
        },
    }
    if geometry.intersects(taevaskoda_exclusion):
        return {**base, "global_exclusion": "taevaskoda_1024m_leakage_domain"}

    bbox = geometry.bounds
    forest, _ = _layer_geometries(FOREST_LAYER, bbox)
    cut_peat, _ = _layer_geometries(CUT_PEAT_LAYER, bbox)
    water_areas, _ = _layer_geometries(WATER_AREA_LAYER, bbox)
    water_lines, water_attributes = _layer_geometries(
        WATER_LINE_LAYER, bbox, ("tyyp", "tyyp_t", "laius")
    )
    buildings, _ = _layer_geometries(BUILDING_LAYER, bbox)
    road_areas, _ = _layer_geometries(ROAD_AREA_LAYER, bbox)
    road_lines, road_attributes = _layer_geometries(ROAD_LINE_LAYER, bbox, ("laius",))

    ditch_indices = [
        offset
        for offset, value in enumerate(water_attributes.get("tyyp", ()))
        if value is not None and int(value) in DITCH_TYPES
    ]
    ditch_geometries = [water_lines[offset] for offset in ditch_indices]
    known_drainage = any(geometry.intersects(item) for item in ditch_geometries)
    known_cut_peat = any(geometry.intersects(item) for item in cut_peat)

    hard = _union(
        [
            *forest,
            *cut_peat,
            *water_areas,
            *_buffered_lines(water_lines, water_attributes.get("laius"), 1.0),
            *buildings,
            *road_areas,
            *_buffered_lines(road_lines, road_attributes.get("laius"), 1.5),
        ]
    )
    clear_bog = geometry.difference(hard)
    clear_halos = _aligned_halos(clear_bog)
    if not clear_halos:
        return {
            **base,
            "known_hard_exclusion": True,
            "known_drainage": known_drainage,
            "known_cut_peat": known_cut_peat,
            "eligible_core_count": 0,
            "unknown_conditions": [],
        }

    soil_fids, soils, soil_attributes = _read(
        SOIL,
        bbox=bbox,
        columns=(
            "Siffer", "Sif1", "Osa1", "Sif2", "Osa2", "Sif3", "Osa3",
            "Sif4", "Osa4", "Loimis1", "Loimis2", "Huumus",
        ),
    )
    soil_records = [_soil_support(soil_attributes, offset) for offset in range(len(soils))]
    soil_supported = _union(
        soils[offset] for offset, record in enumerate(soil_records) if record["supported"]
    )
    geology_supported, geology_records, geology_unknown_features = _geology_support(
        bbox, decoded
    )
    fully_supported = clear_bog.intersection(soil_supported).intersection(geology_supported)
    supported_halos = _aligned_halos(fully_supported)

    unknown_conditions: list[str] = []
    if not any(_covered_by(halo, [soil_supported]) for _, halo in clear_halos):
        unknown_conditions.append("soil_profile_or_humus_semantics")
    if not any(_covered_by(halo, [geology_supported]) for _, halo in clear_halos):
        unknown_conditions.append("geology_coverage_or_domain")
    if statuses != ["Korras"]:
        unknown_conditions.append("condition_ownership")
    if known_drainage or known_cut_peat:
        supported_halos = []

    chosen_pool = supported_halos if supported_halos else clear_halos
    chosen_core, chosen_halo = chosen_pool[0]
    intersecting_soils = []
    for offset, soil_geometry in enumerate(soils):
        if soil_geometry.intersection(chosen_halo).area > COVERAGE_TOLERANCE_M2:
            intersecting_soils.append(
                {
                    "source_fid": int(soil_fids[offset]),
                    "overlap_m2": float(soil_geometry.intersection(chosen_halo).area),
                    **soil_records[offset],
                }
            )
    return {
        **base,
        "known_hard_exclusion": False,
        "known_drainage": known_drainage,
        "known_cut_peat": known_cut_peat,
        "eligible_core_count": len(supported_halos),
        "eligible_core_area_m2": len(supported_halos) * CORE_AREA_M2,
        "unknown_conditions": unknown_conditions,
        "selected_core_bounds_en": [float(value) for value in chosen_core.bounds],
        "selected_halo_bounds_en": [float(value) for value in chosen_halo.bounds],
        "selected_halo_geometry_sha256": hashlib.sha256(chosen_halo.wkb).hexdigest(),
        "soil_records_at_selected_halo": intersecting_soils,
        "geology_records_in_candidate_bbox": geology_records,
        "geology_unknown_feature_count": geology_unknown_features,
        "hard_object_counts_in_candidate_bbox": {
            "forest": len(forest),
            "cut_peat": len(cut_peat),
            "water_area": len(water_areas),
            "water_line": len(water_lines),
            "ditch": len(ditch_indices),
            "building": len(buildings),
            "road_area": len(road_areas),
            "road_line": len(road_lines),
        },
    }


def _connected_components(geometries: np.ndarray) -> list[list[int]]:
    """Connected ETAK polygons form one physical selection unit."""
    parent = np.arange(len(geometries), dtype=np.int64)

    def find(value: int) -> int:
        while int(parent[value]) != value:
            parent[value] = parent[int(parent[value])]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    tree = shapely.STRtree(geometries)
    pairs = tree.query(geometries, predicate="intersects")
    for left, right in zip(pairs[0], pairs[1], strict=True):
        left_index = int(left)
        right_index = int(right)
        if left_index < right_index:
            union(left_index, right_index)
    grouped: dict[int, list[int]] = {}
    for index in range(len(geometries)):
        grouped.setdefault(find(index), []).append(index)
    return list(grouped.values())


def _source_bindings() -> dict[str, Any]:
    return {
        "preregistration": _identity(PREREGISTRATION),
        "etak": _identity(ETAK),
        "soil": _bundle_identity(SOIL),
        "soil_semantics_documents": [_identity(path) for path in SOIL_DOCUMENTS],
        "geology": _bundle_identity(GEOLOGY),
        "geology_domains": _identity(GEOLOGY_DOMAINS),
        "implementation": _identity(Path(__file__)),
    }


def _write_artifact(output_root: Path, recipe: dict[str, Any], result: dict[str, Any]) -> Path:
    recipe_sha256 = hashlib.sha256(_canonical_bytes(recipe)).hexdigest()
    target = output_root / recipe_sha256
    artifact_path = target / "selection.json"
    artifact = {
        "schema_version": "laas.peat-raised-bog-site-selection/1",
        "recipe_sha256": recipe_sha256,
        "recipe": recipe,
        **result,
    }
    payload = _canonical_bytes(artifact)
    if artifact_path.is_file():
        if artifact_path.read_bytes() != payload:
            raise ValueError(f"existing selection artifact differs: {artifact_path}")
        return artifact_path
    temporary = output_root / f".{recipe_sha256}.tmp-{os.getpid()}"
    temporary.mkdir(parents=True, exist_ok=False)
    (temporary / "selection.json").write_bytes(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.rename(target)
    except FileExistsError:
        if artifact_path.read_bytes() != payload:
            raise
        temporary.rmdir()
    return artifact_path


def select_sites(output_root: Path = OUTPUT_ROOT) -> Path:
    preregistration = _verify_preregistration()
    domain_payload = json.loads(GEOLOGY_DOMAINS.read_bytes())
    decoded = {
        row["field_name"]: {
            int(value["code"]): str(value["name"]) for value in row["coded_values"]
        }
        for row in domain_payload["domains"]
    }
    grid = load_base().grid
    taevaskoda_e = grid.anchor_e + TAEVASKODA_GAME_XZ[0]
    taevaskoda_n = grid.anchor_n - TAEVASKODA_GAME_XZ[1]
    taevaskoda_exclusion = shapely.Point(taevaskoda_e, taevaskoda_n).buffer(
        LEAKAGE_EXCLUSION_RADIUS_M
    )
    mire_fids, mires, mire_attributes = _read(
        ETAK,
        layer=MIRE_LAYER,
        where="tyyp = 20 AND puis = 20",
        columns=(
            "etak_id", "tyyp_t", "puis_t", "vajalik_t", "muutmisaeg",
            "geom_muutmisaeg",
        ),
    )
    components = _connected_components(mires)
    candidates: list[dict[str, Any]] = []
    rejected_geometry = 0
    below_minimum_area = 0
    for member_indices in components:
        member_geometries = [mires[index] for index in member_indices]
        if any(
            not geometry.is_valid
            or geometry.geom_type not in {"Polygon", "MultiPolygon"}
            for geometry in member_geometries
        ):
            rejected_geometry += 1
            continue
        geometry = _union(member_geometries)
        if not geometry.is_valid or geometry.geom_type not in {"Polygon", "MultiPolygon"}:
            rejected_geometry += 1
            continue
        if float(geometry.area) < HALO_FOOTPRINT_M**2:
            below_minimum_area += 1
            continue
        candidates.append(
            _candidate(
                geometry=geometry,
                attributes=mire_attributes,
                member_indices=member_indices,
                source_fids=mire_fids,
                decoded=decoded,
                taevaskoda_exclusion=taevaskoda_exclusion,
            )
        )

    development_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("eligible_core_count", 0) > 0
        and not candidate.get("unknown_conditions")
        and not candidate.get("known_drainage")
        and not candidate.get("known_cut_peat")
        and not candidate.get("global_exclusion")
    ]
    development_candidates.sort(
        key=lambda row: (
            -float(row["eligible_core_area_m2"]),
            str(row["normalized_mire_identifier"]),
        )
    )
    development = development_candidates[0] if development_candidates else None

    audit_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("unknown_conditions")
        and not candidate.get("known_hard_exclusion")
        and not candidate.get("known_drainage")
        and not candidate.get("known_cut_peat")
        and not candidate.get("global_exclusion")
        and (
            development is None
            or (
                candidate["normalized_mire_identifier"]
                != development["normalized_mire_identifier"]
                and not shapely.box(*candidate["bounds_en"]).intersects(
                    shapely.box(*development["bounds_en"])
                )
                and not shapely.box(*candidate["selected_halo_bounds_en"]).intersects(
                    shapely.box(*development["selected_halo_bounds_en"])
                )
            )
        )
    ]
    audit_candidates.sort(
        key=lambda row: (
            -len(row["unknown_conditions"]),
            -float(row["candidate_area_m2"]),
            str(row["normalized_mire_identifier"]),
        )
    )
    audit = audit_candidates[0] if audit_candidates else None

    blockers: list[str] = []
    if development is None:
        blockers.append("no development mire resolves every frozen metadata predicate")
    if audit is None:
        blockers.append("no physically distinct sealed abstention mire satisfies the frozen ordering")
    sources = _source_bindings()
    recipe = {
        "schema_version": "laas.peat-raised-bog-site-selection-recipe/1",
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "selection_rules": preregistration["site_selection"],
        "soil_semantics": preregistration["surface_scope"]["soil_semantics"],
        "core_m": CORE_M,
        "halo_m": HALO_M,
        "alignment_m": CORE_M,
        "candidate_unit": (
            "one connected component of intersecting ETAK open nonwooded raised-bog "
            "features, normalized by minimum etak_id"
        ),
        "taevaskoda_exclusion": {
            "center_en": [taevaskoda_e, taevaskoda_n],
            "radius_m": LEAKAGE_EXCLUSION_RADIUS_M,
        },
        "sources": sources,
        "pixel_sources_opened": [],
    }
    status = "selected" if not blockers else "blocked_predicates_unresolved"
    result = {
        "status": status,
        "blockers": blockers,
        "development_mire": development,
        "sealed_abstention_mire": audit,
        "inspection_state": {
            "dtm_opened": False,
            "rgb_opened": False,
            "cir_opened": False,
            "detailed_geometry_pixels_opened": False,
            "structured_vector_metadata_only": True,
        },
        "national_scan": {
            "open_nonwooded_raised_bog_features": len(mires),
            "connected_components": len(components),
            "candidate_components_at_least_256m_square_area": len(candidates),
            "invalid_geometry_components": rejected_geometry,
            "below_minimum_area_components": below_minimum_area,
            "development_eligible_components": len(development_candidates),
            "sealed_audit_eligible_components": len(audit_candidates),
            "rejection_counts": {
                "taevaskoda_leakage": sum(bool(row.get("global_exclusion")) for row in candidates),
                "no_clear_halo": sum(row.get("eligible_core_count") == 0 and row.get("known_hard_exclusion") is True for row in candidates),
                "known_drainage": sum(bool(row.get("known_drainage")) for row in candidates),
                "known_cut_peat": sum(bool(row.get("known_cut_peat")) for row in candidates),
                "unresolved_conditions": sum(bool(row.get("unknown_conditions")) for row in candidates),
            },
        },
        "sealed_audit_law": {
            "allowed_pass": "C1_is_identically_absent",
            "positive_evidence_credit": False,
            "post_open_tuning_allowed": False,
        },
    }
    return _write_artifact(output_root, recipe, result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    artifact = select_sites(args.output_root)
    payload = json.loads(artifact.read_bytes())
    print(artifact)
    print(payload["status"])
    for blocker in payload["blockers"]:
        print(f"BLOCKER: {blocker}")
    return 0 if payload["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
