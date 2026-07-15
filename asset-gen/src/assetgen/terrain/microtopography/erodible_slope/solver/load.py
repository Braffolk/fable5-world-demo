"""Fail-closed binding from the evidence condition bundle to solver inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .....config import ASSET_GEN_ROOT, load_base
from .....grid import ChunkId, chunk_bounds_en, chunk_id_for_en
from .model import CropEvaluation, ProcessConfig, SlopeDomain

_BUNDLE_SCHEMA = "laas.erodible-slope-condition-bundle/1"
_GRID_SCHEMA = "laas.erodible-slope-condition-grid/1"
_SITE_CONFIG_SCHEMA = "laas.erodible-slope-condition-sites/1"
_EVALUATION_SCHEMA = "laas.erodible-slope-correlated-crop-evaluation/1"
_DOMAIN_CLOSURE_SCHEMA = "laas.erodible-slope-domain-closure/1"
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
_CLOSURE_ARRAYS = _REQUIRED_ARRAYS | {"material_rule"}
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
            explicit_bounds = row.get("output_bounds_en")
            if explicit_bounds is not None and tuple(explicit_bounds) != chunk_bounds_en(
                grid, chunk
            ):
                raise ValueError(f"{site_id} output bounds differ from chunk grid")
        chunks[site_id] = chunk
    if configured["development_a"].get("output_chunk") is not None:
        raise ValueError("Development A crop must remain point-derived")
    if chunks["development_c"] != ChunkId(2438, 1518, -2):
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


def _validate_crop_support(domain: SlopeDomain, crop: CropEvaluation, role: str) -> None:
    e0, n0, e1, n1 = domain.bbox_en
    ce0, cn0, ce1, cn1 = crop.bbox_en
    if not (e0 <= ce0 < ce1 <= e1 and n0 <= cn0 < cn1 <= n1):
        raise ValueError(f"{crop.crop_id} lies outside Development A {role} domain")
    col0 = int(round((ce0 - e0) / domain.texel_m))
    col1 = int(round((ce1 - e0) / domain.texel_m))
    row0 = int(round((n1 - cn1) / domain.texel_m))
    row1 = int(round((n1 - cn0) / domain.texel_m))
    window = np.s_[row0:row1, col0:col1]
    expected = (
        int(round((cn1 - cn0) / domain.texel_m)),
        int(round((ce1 - ce0) / domain.texel_m)),
    )
    if domain.height_m[window].shape != expected:
        raise ValueError(f"{crop.crop_id} {role} process window shape differs")
    if not np.all(domain.valid[window] & domain.solve_domain[window]):
        raise ValueError(f"{crop.crop_id} has invalid or unsolved {role} cells")
    if np.any(domain.collar[window]):
        raise ValueError(f"{crop.crop_id} intersects the {role} evidence collar")


def _load_closed_domain(
    path: Path,
    *,
    bbox_en: tuple[float, float, float, float],
    site_id: str,
    role: str,
    config: ProcessConfig,
    closure_identity: dict[str, Any],
) -> SlopeDomain:
    with np.load(path, allow_pickle=False) as source:
        if set(source.files) != _CLOSURE_ARRAYS:
            raise ValueError(f"{role} closure-domain fields differ")
        arrays = {name: np.asarray(source[name]) for name in source.files}
    expected_shape = (
        int(round((bbox_en[3] - bbox_en[1]) / config.process_texel_m)),
        int(round((bbox_en[2] - bbox_en[0]) / config.process_texel_m)),
    )
    if any(value.shape != expected_shape for value in arrays.values()):
        raise ValueError(f"{role} closure-domain shape differs")
    if np.any(arrays["edge_leak"]):
        raise ValueError(f"{role} closure domain reaches its evidence edge")
    water = arrays["water"].astype(bool)
    objects = arrays["object"].astype(bool)
    non_heightfield = arrays["non_heightfield"].astype(bool)
    protected = arrays["protected_structure"].astype(bool)
    outlet = arrays["outlet"].astype(bool)
    return SlopeDomain(
        site_id=site_id,
        bbox_en=bbox_en,
        texel_m=config.process_texel_m,
        height_m=arrays["height"].astype(np.float64),
        valid=arrays["valid"].astype(bool),
        solve_domain=arrays["solve_domain"].astype(bool),
        upstream_domain=arrays["upstream_domain"].astype(bool),
        outlet=outlet,
        collar=arrays["collar"].astype(bool),
        routing_barrier=objects | non_heightfield | (water & ~outlet),
        hard_exclusion=water | objects | non_heightfield | protected,
        unknown=arrays["unknown"].astype(bool),
        vegetation_cover=arrays["vegetation_evidence"].astype(np.float64),
        seep_likelihood=arrays[
            "topographic_seep_support_likelihood"
        ].astype(np.float64),
        upstream_water_m3=np.zeros(expected_shape, dtype=np.float64),
        material_rule=arrays["material_rule"].astype(np.int16),
        material_rule_names=tuple(config.material_rules),
        source_identity={
            "role": role,
            "domain_closure": closure_identity,
            "arrays": {
                "path": str(path.relative_to(ASSET_GEN_ROOT.parent)),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            },
        },
    )


def _load_authorized_domain_closure(
    identity: dict[str, Any],
    config: ProcessConfig,
) -> tuple[SlopeDomain, SlopeDomain]:
    manifest_path = _bound_path(identity)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    closure = manifest.get("closure", {})
    dependency = closure.get("dependency_set_identity", {})
    condition_identity = closure.get("condition_identity", {})
    if (
        manifest.get("schema_version") != _DOMAIN_CLOSURE_SCHEMA
        or manifest.get("recipe_sha256") != manifest_path.parent.name
        or manifest.get("state") != "closure_pass"
        or manifest.get("solver_use_authorized") is not True
        or manifest.get("canonical_role") != "solver_authorized_condition_authority"
        or manifest.get("sealed_ood_used") is not False
        or closure.get("closure_pass") is not True
        or closure.get("canonical_dependency_touches_storage_edge") is not False
        or closure.get("control_dependency_touches_storage_edge") is not False
        or closure.get("next_required_expansion") is not None
        or dependency.get("exact") is not True
        or dependency.get("canonical_only_cells") != 0
        or dependency.get("control_dependency_cells_outside_canonical") != 0
        or dependency.get("control_only_cells_inside_canonical") != 0
        or closure.get("raw_dinf_branch_identity", {}).get("a", {}).get("exact") is not True
        or closure.get("raw_dinf_branch_identity", {}).get("b", {}).get("exact") is not True
        or closure.get("finite_depression_spill_identity", {}).get("exact") is not True
        or not isinstance(condition_identity, dict)
        or not condition_identity
        or any(
            row.get("exact") is not True
            for row in condition_identity.values()
        )
    ):
        raise ValueError("correlated evaluation domain closure is not solver-authorized")
    counts = manifest.get("routing_dependency_counts", {})
    canonical_count = counts.get("canonical", {}).get("dependency_cells")
    control_count = counts.get("control", {}).get("dependency_cells")
    if canonical_count != control_count or not isinstance(canonical_count, int):
        raise ValueError("domain-closure dependency counts differ")
    outputs = manifest.get("outputs", {})
    canonical_path = _bound_path(outputs.get("canonical_domain", {}))
    control_path = _bound_path(outputs.get("control_domain", {}))
    recipe = manifest.get("recipe", {})
    if (
        recipe.get("canonical_expansion_from_original_m") != 256
        or recipe.get("control_expansion_from_original_m") != 384
    ):
        raise ValueError("domain-closure expansion authority differs")
    canonical_bbox = tuple(float(value) for value in recipe.get("canonical_bbox_en", ()))
    control_bbox = tuple(float(value) for value in recipe.get("control_bbox_en", ()))
    if len(canonical_bbox) != 4 or len(control_bbox) != 4:
        raise ValueError("domain-closure bounding boxes are malformed")
    closure_identity = {
        "path": str(manifest_path.relative_to(ASSET_GEN_ROOT.parent)),
        "bytes": identity["bytes"],
        "sha256": identity["sha256"],
        "recipe_sha256": manifest["recipe_sha256"],
        "dependency_cells": canonical_count,
        "closure_pass": True,
    }
    return (
        _load_closed_domain(
            canonical_path,
            bbox_en=canonical_bbox,
            site_id="development_a",
            role="canonical_plus256",
            config=config,
            closure_identity=closure_identity,
        ),
        _load_closed_domain(
            control_path,
            bbox_en=control_bbox,
            site_id="development_a",
            role="control_plus384",
            config=config,
            closure_identity=closure_identity,
        ),
    )


def load_crop_evaluation_plan(
    path: Path,
    *,
    condition_bundle_path: Path,
    config: ProcessConfig,
) -> tuple[
    tuple[CropEvaluation, ...],
    dict[str, Any],
    str,
    SlopeDomain,
    SlopeDomain,
]:
    path = Path(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version",
        "condition_bundle",
        "domain_closure",
        "source_site_id",
        "solve_group_id",
        "crops",
        "evidence_accounting",
    }
    if set(document) != expected_fields or document.get("schema_version") != _EVALUATION_SCHEMA:
        raise ValueError("unsupported correlated-crop evaluation plan")
    bound_bundle = _bound_path(document["condition_bundle"])
    if bound_bundle.resolve() != Path(condition_bundle_path).resolve():
        raise ValueError("evaluation plan condition bundle differs from runner input")
    canonical_domain, control_domain = _load_authorized_domain_closure(
        document["domain_closure"], config
    )
    if document.get("source_site_id") != "development_a":
        raise ValueError("correlated evaluation must retain Development A identity")
    if document.get("solve_group_id") != "development_a_whole_domain_e316add6":
        raise ValueError("correlated evaluation solve-group identity differs")
    accounting = document.get("evidence_accounting")
    required_accounting = {
        "evidence_label": "r0_correlated_within_site_two_crop_visual_evaluation",
        "development_site_credit": 1,
        "independent_validation_site_credit": 0,
        "ood_holdout_credit": 0,
        "production_credit": 0,
        "preview_authorized": False,
        "recipe_freeze_authorized": False,
    }
    if accounting != required_accounting:
        raise ValueError("correlated evaluation evidence accounting differs")
    expected_chunks = {
        "development_a_west": ChunkId(cx=2436, cz=1492, lod=-2),
        "development_a_east": ChunkId(cx=2437, cz=1492, lod=-2),
    }
    rows = document.get("crops")
    if not isinstance(rows, list) or len(rows) != len(expected_chunks):
        raise ValueError("correlated evaluation requires exactly A-west and A-east")
    grid = load_base().grid
    evaluations: list[CropEvaluation] = []
    for row in rows:
        if set(row) != {"crop_id", "output_chunk", "evidence_role"}:
            raise ValueError("correlated crop fields differ")
        crop_id = row["crop_id"]
        if crop_id not in expected_chunks:
            raise ValueError(f"unexpected correlated crop {crop_id}")
        values = row["output_chunk"]
        if not (
            isinstance(values, list)
            and len(values) == 3
            and all(isinstance(value, int) for value in values)
        ):
            raise ValueError(f"{crop_id} output chunk is malformed")
        chunk = ChunkId(cx=values[1], cz=values[2], lod=values[0])
        if chunk != expected_chunks[crop_id]:
            raise ValueError(f"{crop_id} output chunk differs")
        if row["evidence_role"] != "correlated_within_site_r0_visual_evaluation":
            raise ValueError(f"{crop_id} evidence role differs")
        evaluation = CropEvaluation(
            crop_id=crop_id,
            source_site_id="development_a",
            solve_group_id=document["solve_group_id"],
            output_chunk=(chunk.lod, chunk.cx, chunk.cz),
            bbox_en=tuple(float(value) for value in chunk_bounds_en(grid, chunk)),
            evidence_role=row["evidence_role"],
        )
        _validate_crop_support(canonical_domain, evaluation, "canonical-plus256")
        _validate_crop_support(control_domain, evaluation, "control-plus384")
        evaluations.append(evaluation)
    if {row.crop_id for row in evaluations} != set(expected_chunks):
        raise ValueError("correlated crop identities are not unique and complete")
    evaluations.sort(key=lambda value: value.output_chunk)
    return (
        tuple(evaluations),
        accounting,
        _sha256_file(path),
        canonical_domain,
        control_domain,
    )
