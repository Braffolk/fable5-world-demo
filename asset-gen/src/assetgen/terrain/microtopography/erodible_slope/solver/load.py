"""Fail-closed binding from the evidence condition bundle to solver inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .....config import ASSET_GEN_ROOT, load_base
from .....grid import ChunkId, chunk_bounds_en, chunk_id_for_en
from .model import ProcessConfig, SlopeDomain

_BUNDLE_SCHEMA = "laas.erodible-slope-condition-bundle/1"
_GRID_SCHEMA = "laas.erodible-slope-condition-grid/1"
_SITE_CONFIG_SCHEMA = "laas.erodible-slope-condition-sites/1"
_RUNNABLE_SITES = frozenset({"development_a", "development_c"})
_EXCLUDED_SITES = frozenset({"development_b"})
_FORBIDDEN_ETAK_IDS = frozenset({9688685, 9688702})
_REQUIRED_ARRAYS = {
    "height",
    "valid",
    "solve_domain",
    "upstream_domain",
    "outlet",
    "collar",
    "edge_leak",
    "water",
    "object",
    "vegetation_evidence",
    "non_heightfield",
    "protected_structure",
    "unknown",
    "target_feature",
    "soil_feature_index",
    "geology_lithology_code",
    "geology_genesis_code",
    "topographic_seep_support_likelihood",
}
_ALLOWED_EXTRA_ARRAYS = {"fill_depth", "unknown_bathymetry"}
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bound_path(identity: dict[str, Any]) -> Path:
    path = ASSET_GEN_ROOT.parent / identity["path"]
    if not path.is_file():
        raise FileNotFoundError(f"bound condition source is absent: {path}")
    if path.stat().st_size != identity["bytes"] or _sha256_file(path) != identity["sha256"]:
        raise ValueError(f"bound condition source identity differs: {path}")
    return path


def _surface_material(feature: dict[str, Any]) -> str | None:
    normalized = feature.get("normalized", {})
    texture_profile = normalized.get("Loimis1", {})
    deeper_profile = normalized.get("Loimis2", {})
    humus = normalized.get("Huumus", {})
    if texture_profile.get("status") != "parsed_complete_official_grammar":
        return None
    if deeper_profile.get("status") not in {
        "parsed_complete_official_grammar",
        "missing",
    }:
        return None
    if humus.get("status") != "parsed_complete_official_grammar":
        return None
    expressions = texture_profile.get("expressions")
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
    if not isinstance(texture, dict) or not isinstance(texture.get("family"), str):
        return None
    skeleton = material.get("skeleton")
    if isinstance(skeleton, dict):
        content = skeleton.get("content_class")
        if content == "ungraded_or_over_70_percent" or (
            isinstance(content, int) and content >= 5
        ):
            return None
    for expression in humus.get("expressions", []):
        for sequence in expression.get("sequences", []):
            if any(
                term.get("kind") == "peat_horizon"
                for term in sequence.get("terms", [])
            ):
                return None
    return str(texture["family"])


def _material_rules(
    arrays: dict[str, np.ndarray],
    soil_document: dict[str, Any],
    config: ProcessConfig,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
    names = tuple(config.material_rules)
    name_to_index = {name: index for index, name in enumerate(names)}
    if set(names) != {"sand_mineral", "till_mineral"}:
        raise ValueError("C1 material-rule inventory differs from reviewed classifier")
    features = soil_document.get("features")
    if not isinstance(features, list):
        raise ValueError("soil window has no feature inventory")
    feature_material = [_surface_material(feature) for feature in features]
    soil_index = arrays["soil_feature_index"].astype(np.int64, copy=False)
    if np.any(soil_index >= len(features)):
        raise ValueError("soil feature index exceeds its exact bound window")
    lithology = arrays["geology_lithology_code"]
    genesis = arrays["geology_genesis_code"]
    result = np.full(soil_index.shape, -1, dtype=np.int16)
    for index, texture in enumerate(feature_material):
        if texture is None:
            continue
        feature_cells = soil_index == index
        if texture in _COARSE_TEXTURES:
            compatible = feature_cells & (lithology == 40) & np.isin(
                genesis, (10, 30, 40, 60, 100)
            )
            result[compatible] = name_to_index["sand_mineral"]
        if texture in _COHESIVE_TEXTURES:
            compatible = feature_cells & (lithology == 50) & (genesis == 50)
            result[compatible] = name_to_index["till_mineral"]
    return result, names, {
        "soil_features": len(features),
        "soil_features_with_complete_surface_material_and_humus": int(
            sum(value is not None for value in feature_material)
        ),
        "sand_mineral_cells": int(np.count_nonzero(result == name_to_index["sand_mineral"])),
        "till_mineral_cells": int(np.count_nonzero(result == name_to_index["till_mineral"])),
        "abstained_material_cells": int(np.count_nonzero(result < 0)),
    }


def _load_domain_record(
    record: dict[str, Any],
    *,
    site_id: str,
    config: ProcessConfig,
    role: str,
) -> SlopeDomain:
    if record.get("array_schema_version") != _GRID_SCHEMA:
        raise ValueError(f"{site_id} {role} array schema differs")
    if record.get("grid_m") != config.process_texel_m:
        raise ValueError(f"{site_id} {role} grid differs from process config")
    if record.get("blockers"):
        raise ValueError(f"{site_id} {role} condition blockers remain: {record['blockers']}")
    array_path = _bound_path(record["arrays"])
    with np.load(array_path, allow_pickle=False) as source:
        fields = set(source.files)
        if not _REQUIRED_ARRAYS <= fields or not (
            fields - _REQUIRED_ARRAYS
        ) <= _ALLOWED_EXTRA_ARRAYS:
            raise ValueError(
                f"{site_id} {role} fields differ: missing={sorted(_REQUIRED_ARRAYS - fields)}, "
                f"extra={sorted(fields - _REQUIRED_ARRAYS - _ALLOWED_EXTRA_ARRAYS)}"
            )
        arrays = {name: np.asarray(source[name]) for name in source.files}
    shape = tuple(record["shape"])
    if any(value.shape != shape for value in arrays.values()):
        raise ValueError(f"{site_id} {role} array shapes differ from manifest")
    if np.any(arrays["edge_leak"]):
        raise ValueError(f"{site_id} {role} physical domain reaches evidence edge")
    soil_path = _bound_path(record["soil_window"])
    soil_document = json.loads(soil_path.read_text(encoding="utf-8"))
    material, material_names, material_summary = _material_rules(
        arrays, soil_document, config
    )
    valid = arrays["valid"].astype(bool)
    solve = arrays["solve_domain"].astype(bool)
    outlet = arrays["outlet"].astype(bool)
    water = arrays["water"].astype(bool)
    objects = arrays["object"].astype(bool)
    non_heightfield = arrays["non_heightfield"].astype(bool)
    protected = arrays["protected_structure"].astype(bool)
    unknown = arrays["unknown"].astype(bool)
    hard_exclusion = water | objects | non_heightfield | protected
    routing_barrier = objects | non_heightfield | (water & ~outlet)
    source_identity = {
        "role": role,
        "arrays": record["arrays"],
        "soil_window": record["soil_window"],
        "geology_window": record["geology_window"],
        "material_assignment": material_summary,
        "topographic_seep_support_likelihood": record[
            "topographic_seep_support_likelihood"
        ],
        "domain_policy": record["domain_policy"],
    }
    return SlopeDomain(
        site_id=site_id,
        bbox_en=tuple(float(value) for value in record["bbox_en"]),
        texel_m=float(record["grid_m"]),
        height_m=arrays["height"].astype(np.float64),
        valid=valid,
        solve_domain=solve,
        upstream_domain=arrays["upstream_domain"].astype(bool),
        outlet=outlet,
        collar=arrays["collar"].astype(bool),
        routing_barrier=routing_barrier,
        hard_exclusion=hard_exclusion,
        unknown=unknown,
        vegetation_cover=arrays["vegetation_evidence"].astype(np.float64),
        seep_likelihood=arrays[
            "topographic_seep_support_likelihood"
        ].astype(np.float64),
        upstream_water_m3=np.zeros(shape, dtype=np.float64),
        material_rule=material,
        material_rule_names=material_names,
        source_identity=source_identity,
    )


def _runnable_crops(
    document: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, tuple[float, float, float, float]]]:
    recipe = document.get("recipe")
    if not isinstance(recipe, dict):
        raise ValueError("condition bundle lacks recipe authority")
    policies = recipe.get("policies")
    if not isinstance(policies, dict):
        raise ValueError("condition bundle lacks recipe policies")
    runnable = policies.get("runnable_morphology_site_ids")
    excluded = policies.get("solver_excluded_site_ids")
    forbidden = policies.get("forbidden_etak_ids")
    if not isinstance(runnable, list) or set(runnable) != _RUNNABLE_SITES:
        raise ValueError("condition bundle must authorize exactly Development A/C")
    if not isinstance(excluded, list) or set(excluded) != _EXCLUDED_SITES:
        raise ValueError("condition bundle must solver-exclude exactly Development B")
    if not isinstance(forbidden, list) or set(forbidden) != _FORBIDDEN_ETAK_IDS:
        raise ValueError("condition bundle forbidden ETAK identities differ")
    if policies.get("sealed_ood_used") is not False:
        raise ValueError("sealed OOD2 must remain unopened")

    site_records = {row["site_id"]: row for row in document.get("sites", [])}
    if not _RUNNABLE_SITES <= set(site_records):
        raise ValueError("condition bundle lacks a runnable morphology site")
    for site_id in _RUNNABLE_SITES:
        if site_records[site_id].get("development_role") != "morphology_development":
            raise ValueError(f"{site_id} is not bound as morphology development")
    if site_records.get("development_b", {}).get("development_role") != (
        "strict_material_abstention_negative"
    ):
        raise ValueError("Development B exclusion role differs")

    inputs = recipe.get("inputs")
    if not isinstance(inputs, dict) or not isinstance(inputs.get("site_config"), dict):
        raise ValueError("condition bundle lacks bound site configuration")
    site_config = json.loads(
        _bound_path(inputs["site_config"]).read_text(encoding="utf-8")
    )
    if site_config.get("schema_version") != _SITE_CONFIG_SCHEMA:
        raise ValueError("unsupported erodible-slope site configuration")
    configured = {row["id"]: row for row in site_config.get("sites", [])}
    if set(configured) != _RUNNABLE_SITES | _EXCLUDED_SITES:
        raise ValueError("bound site configuration inventory differs")
    if set(site_config.get("forbidden_etak_ids", [])) != _FORBIDDEN_ETAK_IDS:
        raise ValueError("bound site configuration forbidden identities differ")

    recipe_sites = {row["site_id"]: row for row in recipe.get("sites", [])}
    if set(recipe_sites) != set(configured):
        raise ValueError("condition recipe and bound site configuration differ")
    grid = load_base().grid
    chunks: dict[str, ChunkId] = {}
    for site_id in sorted(_RUNNABLE_SITES):
        row = configured[site_id]
        if row.get("role") != "morphology_development":
            raise ValueError(f"{site_id} configured role differs")
        point_chunk = chunk_id_for_en(grid, float(row["e_m"]), float(row["n_m"]), -2)
        explicit = row.get("output_chunk")
        recipe_explicit = recipe_sites[site_id].get("output_chunk")
        if explicit != recipe_explicit:
            raise ValueError(f"{site_id} output chunk differs from condition recipe")
        if explicit is None:
            chunk = point_chunk
        else:
            if not (
                isinstance(explicit, list)
                and len(explicit) == 3
                and all(isinstance(value, int) for value in explicit)
            ):
                raise ValueError(f"{site_id} output chunk is malformed")
            chunk = ChunkId(cx=explicit[1], cz=explicit[2], lod=explicit[0])
            if chunk != point_chunk:
                raise ValueError(f"{site_id} output chunk does not contain target point")
        chunks[site_id] = chunk
    if configured["development_a"].get("output_chunk") is not None:
        raise ValueError("Development A crop must remain point-derived")
    if chunks["development_c"] != ChunkId(2438, 1519, -2):
        raise ValueError("Development C explicit output chunk differs")

    crops = {
        site_id: tuple(float(value) for value in chunk_bounds_en(grid, chunk))
        for site_id, chunk in chunks.items()
    }
    return {site_id: site_records[site_id] for site_id in sorted(_RUNNABLE_SITES)}, crops


def load_condition_bundle(
    path: Path,
    config: ProcessConfig,
) -> tuple[
    dict[str, SlopeDomain],
    dict[str, SlopeDomain],
    dict[str, tuple[float, float, float, float]],
    dict[str, float],
    str,
]:
    path = Path(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != _BUNDLE_SCHEMA:
        raise ValueError("unsupported erodible-slope condition bundle")
    status = document.get("status", {})
    if status.get("ready_for_r0_input_freeze") is not True:
        raise ValueError(f"condition bundle is not R0-input-ready: {status}")
    if status.get("ready_for_recipe_freeze_or_preview") is not False:
        raise ValueError("replacement-OOD authority boundary is not explicit")
    sites, crops = _runnable_crops(document)
    domains: dict[str, SlopeDomain] = {}
    enlarged: dict[str, SlopeDomain] = {}
    collars: dict[str, float] = {}
    for site_id in sorted(sites):
        record = sites[site_id]
        domains[site_id] = _load_domain_record(
            record,
            site_id=site_id,
            config=config,
            role="normal",
        )
        enlarged_metadata = record.get("enlarged_domain")
        enlarged_arrays = record.get("enlarged_domain_arrays")
        if not isinstance(enlarged_metadata, dict) or not isinstance(
            enlarged_arrays, dict
        ):
            raise ValueError(f"{site_id} lacks bound enlarged-domain record")
        enlarged_record = {
            **enlarged_metadata,
            "array_schema_version": record["array_schema_version"],
            "arrays": enlarged_arrays,
            "domain_policy": record["domain_policy"],
        }
        enlarged[site_id] = _load_domain_record(
            enlarged_record,
            site_id=site_id,
            config=config,
            role="enlarged_128m",
        )
        collars[site_id] = float(record["domain_policy"]["minimum_evidence_collar_m"])
    for site_id, crop in crops.items():
        for role, domain in (
            ("normal", domains[site_id]),
            ("enlarged", enlarged[site_id]),
        ):
            e0, n0, e1, n1 = domain.bbox_en
            if not (e0 <= crop[0] < crop[2] <= e1 and n0 <= crop[1] < crop[3] <= n1):
                raise ValueError(f"{site_id} crop lies outside {role} physical domain")
    return domains, enlarged, crops, collars, _sha256_file(path)
