"""Development A condition expansion and routing-dependency closure evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import zipfile
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from .....config import ASSET_GEN_ROOT, DATA_WORK, load_base
from .....grid import ChunkId, chunk_bounds_en
from ....conditions.geology import extract_egt_surficial_window
from ....conditions.soil import extract_soil_window
from ..solver.load import _load_domain_record, _material_rules
from ..solver.model import ProcessConfig, SlopeDomain, load_process_config
from ..solver.routing import _depression_spills, route_continuous
from .drainage import delineate_drainage_domain
from .materialize import (
    SiteSpec,
    _active_soil_semantics,
    _condition_rasters,
    _corrected_lod0_window,
    _file_identity,
    _mask_sources,
    _rasterize,
    _read_etak_features,
    _read_target_geometry,
    _soil_support_mask,
    _topographic_seep_support,
    _transform,
)

_PLAN_SCHEMA = "laas.erodible-slope-domain-closure-plan/1"
_BUNDLE_SCHEMA = "laas.erodible-slope-condition-bundle/1"
_ARRAY_SCHEMA = "laas.erodible-slope-condition-grid/1"
_COLLAR_M = 32
_DOMAIN_FIELDS = (
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
    "material_rule",
)
_CLOSURE_CONDITION_FIELDS = tuple(
    name for name in _DOMAIN_FIELDS if name != "soil_feature_index"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")


def _bound(identity: dict[str, Any]) -> Path:
    if set(identity) != {"path", "bytes", "sha256"}:
        raise ValueError("bound file identity fields differ")
    path = ASSET_GEN_ROOT.parent / identity["path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != identity["bytes"] or _sha256_file(path) != identity["sha256"]:
        raise ValueError(f"bound file identity differs: {path}")
    return path


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
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
                np.lib.format.write_array(member, np.asarray(arrays[name]), allow_pickle=False)


def _implementation_identity() -> list[dict[str, Any]]:
    paths = (
        Path(__file__),
        Path(__file__).with_name("materialize.py"),
        Path(__file__).with_name("drainage.py"),
        Path(__file__).parent.parent / "solver" / "load.py",
        Path(__file__).parent.parent / "solver" / "model.py",
        Path(__file__).parent.parent / "solver" / "routing.py",
    )
    return [_file_identity(path) for path in paths]


def _load_plan(path: Path) -> tuple[dict[str, Any], Path, Path, Path]:
    document = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "condition_bundle",
        "site_config",
        "solver_config",
        "source_site_id",
        "canonical_expansion_from_original_m",
        "control_expansion_from_original_m",
        "next_expansion_step_m",
        "crops",
        "forbidden_etak_ids",
        "sealed_ood_used",
    }
    optional = {"canonical_authority"}
    if (
        not expected <= set(document) <= expected | optional
        or document.get("schema_version") != _PLAN_SCHEMA
    ):
        raise ValueError("unsupported domain-closure plan")
    if document["source_site_id"] != "development_a":
        raise ValueError("domain closure is restricted to Development A")
    canonical_expansion = int(document["canonical_expansion_from_original_m"])
    control_expansion = int(document["control_expansion_from_original_m"])
    step = int(document["next_expansion_step_m"])
    if (
        canonical_expansion not in {128, 256}
        or control_expansion != canonical_expansion + step
        or step != 128
    ):
        raise ValueError("domain-closure expansion schedule differs")
    if canonical_expansion == 128 and "canonical_authority" in document:
        raise ValueError("initial canonical must come from the condition bundle")
    if canonical_expansion == 256 and "canonical_authority" not in document:
        raise ValueError("promoted +256 canonical authority is not bound")
    if set(document["forbidden_etak_ids"]) != {9688685, 9688702}:
        raise ValueError("forbidden ETAK identities differ")
    if document["sealed_ood_used"] is not False:
        raise ValueError("sealed OOD2 must remain unused")
    expected_crops = {
        "development_a_west": [-2, 2436, 1492],
        "development_a_east": [-2, 2437, 1492],
    }
    if {
        row.get("crop_id"): row.get("output_chunk") for row in document["crops"]
    } != expected_crops:
        raise ValueError("domain-closure crops differ")
    return (
        document,
        _bound(document["condition_bundle"]),
        _bound(document["site_config"]),
        _bound(document["solver_config"]),
    )


def _site_spec(path: Path) -> SiteSpec:
    document = json.loads(path.read_text(encoding="utf-8"))
    if set(document.get("forbidden_etak_ids", [])) != {9688685, 9688702}:
        raise ValueError("site configuration forbidden identities differ")
    rows = [row for row in document.get("sites", []) if row.get("id") == "development_a"]
    if len(rows) != 1:
        raise ValueError("site configuration lacks exactly one Development A")
    row = rows[0]
    return SiteSpec(
        site_id="development_a",
        role=str(row["role"]),
        target_etak_id=int(row["etak_id"]),
        target_e=float(row["e_m"]),
        target_n=float(row["n_m"]),
        sheet=str(row["maaamet_sheet"]),
        target_outlet_ids=tuple(int(value) for value in row["target_outlet_etak_ids"]),
    )


def _expanded_bbox(
    bbox: tuple[int, int, int, int], expansion_m: int
) -> tuple[int, int, int, int]:
    return (
        bbox[0] - expansion_m,
        bbox[1] - expansion_m,
        bbox[2] + expansion_m,
        bbox[3] + expansion_m,
    )


def _load_canonical(
    bundle: dict[str, Any],
    config: ProcessConfig,
) -> tuple[SlopeDomain, dict[str, np.ndarray], dict[str, Any]]:
    rows = [row for row in bundle["sites"] if row.get("site_id") == "development_a"]
    if len(rows) != 1:
        raise ValueError("condition bundle lacks exactly one Development A")
    site = rows[0]
    metadata = site["enlarged_domain"]
    record = {
        **metadata,
        "array_schema_version": site["array_schema_version"],
        "arrays": site["enlarged_domain_arrays"],
        "domain_policy": site["domain_policy"],
    }
    domain = _load_domain_record(
        record,
        site_id="development_a",
        config=config,
        role="canonical_plus128",
    )
    array_path = _bound(site["enlarged_domain_arrays"])
    with np.load(array_path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    arrays["material_rule"] = domain.material_rule
    arrays["material_rule_names"] = np.asarray(domain.material_rule_names, dtype="U32")
    return domain, arrays, site


def _load_promoted_canonical(
    authority_path: Path,
    expected_bbox: tuple[int, int, int, int],
    config: ProcessConfig,
) -> tuple[SlopeDomain, dict[str, np.ndarray], dict[str, Any]]:
    manifest = json.loads(authority_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "laas.erodible-slope-domain-closure/1":
        raise ValueError("unsupported promoted canonical authority")
    recipe = manifest.get("recipe", {})
    if tuple(recipe.get("control_bbox_en", ())) != expected_bbox:
        raise ValueError("promoted canonical bbox differs")
    domain_identity = manifest.get("outputs", {}).get("control_domain")
    if not isinstance(domain_identity, dict):
        raise ValueError("promoted canonical domain is not bound")
    domain_path = _bound(domain_identity)
    with np.load(domain_path, allow_pickle=False) as source:
        if not set(_DOMAIN_FIELDS) <= set(source.files):
            raise ValueError("promoted canonical arrays are incomplete")
        arrays = {name: np.asarray(source[name]) for name in source.files}
    valid = arrays["valid"].astype(bool)
    solve = arrays["solve_domain"].astype(bool)
    outlet = arrays["outlet"].astype(bool)
    water = arrays["water"].astype(bool)
    objects = arrays["object"].astype(bool)
    non_heightfield = arrays["non_heightfield"].astype(bool)
    protected = arrays["protected_structure"].astype(bool)
    hard = water | objects | non_heightfield | protected
    domain = SlopeDomain(
        site_id="development_a",
        bbox_en=tuple(float(value) for value in expected_bbox),
        texel_m=1.0,
        height_m=arrays["height"].astype(np.float64),
        valid=valid,
        solve_domain=solve,
        upstream_domain=arrays["upstream_domain"].astype(bool),
        outlet=outlet,
        collar=arrays["collar"].astype(bool),
        routing_barrier=objects | non_heightfield | (water & ~outlet),
        hard_exclusion=hard,
        unknown=arrays["unknown"].astype(bool),
        vegetation_cover=arrays["vegetation_evidence"].astype(np.float64),
        seep_likelihood=arrays["topographic_seep_support_likelihood"].astype(
            np.float64
        ),
        upstream_water_m3=np.zeros(arrays["height"].shape, dtype=np.float64),
        material_rule=arrays["material_rule"].astype(np.int16),
        material_rule_names=tuple(config.material_rules),
        source_identity={
            "role": "promoted_canonical",
            "authority_manifest": _file_identity(authority_path),
            "domain": _file_identity(domain_path),
        },
    )
    return domain, arrays, domain.source_identity


def _build_control(
    *,
    bundle: dict[str, Any],
    bundle_path: Path,
    plan_path: Path,
    spec: SiteSpec,
    control_bbox: tuple[int, int, int, int],
    control_expansion_m: int,
    config: ProcessConfig,
) -> tuple[SlopeDomain, dict[str, np.ndarray], dict[str, Any]]:
    inputs = bundle["recipe"]["inputs"]
    accepted_materialization = _bound(inputs["accepted_corrected_materialization"])
    authority_manifest_path = _bound(inputs["accepted_structural_authority"])
    domain_snapshot = _bound(inputs["egt_domain_snapshot"])
    etak_path = _bound(inputs["etak"])
    expected_etak = ASSET_GEN_ROOT / "data/in/etak/ETAK_EESTI_GPKG.gpkg"
    if etak_path.resolve() != expected_etak.resolve():
        raise ValueError("condition bundle ETAK source path differs")
    authority_manifest = json.loads(authority_manifest_path.read_text(encoding="utf-8"))
    target = _read_target_geometry(spec)
    records, geometries = _read_etak_features(
        control_bbox, target, frozenset({9688685, 9688702})
    )
    height, corrected_lod0 = _corrected_lod0_window(
        accepted_materialization,
        authority_manifest,
        control_bbox,
    )
    authority = {
        "height": height,
        "valid": np.isfinite(height),
        "unknown_bathymetry": np.zeros(height.shape, dtype=bool),
        "forbidden_morphology": np.zeros(height.shape, dtype=bool),
    }
    transform = _transform(control_bbox)
    masks = _mask_sources(
        spec, records, geometries, height.shape, transform, authority
    )
    drainage = delineate_drainage_domain(
        height,
        authority["valid"],
        masks["target_outlet"],
        masks["other_outlet"],
        collar_cells=_COLLAR_M,
    )
    selection_source = {
        "kind": "erodible_slope_development_a_domain_closure_control",
        "condition_bundle": _file_identity(bundle_path),
        "plan": _file_identity(plan_path),
        "site_id": "development_a",
        "target_etak_id": spec.target_etak_id,
        "bbox_en": list(control_bbox),
        "grid_m": 1,
        "expansion_from_original_m": control_expansion_m,
        "forbidden_etak_ids": [9688685, 9688702],
        "sealed_ood_used": False,
    }
    soil_path = extract_soil_window(
        control_bbox,
        name="erodible-slope-development-a-domain-closure-control",
        selection_source=selection_source,
    )
    geology_path = extract_egt_surficial_window(
        control_bbox,
        name="erodible-slope-development-a-domain-closure-control",
        domain_snapshot_path=domain_snapshot,
        selection_source=selection_source,
    )
    soil = _condition_rasters(
        soil_path, kind="soil", shape=height.shape, transform=transform
    )
    geology = _condition_rasters(
        geology_path, kind="geology", shape=height.shape, transform=transform
    )
    soil_document = json.loads(soil_path.read_text(encoding="utf-8"))
    soil_semantics = _active_soil_semantics(
        soil_document, soil["primary"], drainage.solve_domain
    )
    soil_support = _soil_support_mask(soil_semantics, soil["primary"])
    unknown = (
        ~authority["valid"]
        | soil["unknown"]
        | geology["unknown"]
        | ~soil_support
        | drainage.edge_leak
    )
    seep_support, seep_derivation = _topographic_seep_support(
        height,
        solve_domain=drainage.solve_domain,
        water=masks["water"],
        object_mask=masks["object"],
        protected_structure=masks["protected_structure"],
        unknown=unknown,
    )
    arrays = {
        "height": height.astype(np.float32),
        "valid": authority["valid"].astype(np.uint8),
        "solve_domain": drainage.solve_domain.astype(np.uint8),
        "upstream_domain": drainage.upstream.astype(np.uint8),
        "outlet": drainage.outlet.astype(np.uint8),
        "collar": drainage.collar.astype(np.uint8),
        "edge_leak": drainage.edge_leak.astype(np.uint8),
        "water": masks["water"].astype(np.uint8),
        "object": masks["object"].astype(np.uint8),
        "vegetation_evidence": masks["vegetation_evidence"].astype(np.uint8),
        "non_heightfield": masks["non_heightfield"].astype(np.uint8),
        "protected_structure": masks["protected_structure"].astype(np.uint8),
        "unknown": unknown.astype(np.uint8),
        "target_feature": _rasterize([target], height.shape, transform).astype(np.uint8),
        "unknown_bathymetry": authority["unknown_bathymetry"].astype(np.uint8),
        "soil_feature_index": soil["primary"].astype(np.int32),
        "geology_lithology_code": geology["primary"].astype(np.int32),
        "geology_genesis_code": geology["secondary"].astype(np.int32),
        "topographic_seep_support_likelihood": seep_support.astype(np.float32),
        "fill_depth": drainage.fill_depth.astype(np.float32),
    }
    material, material_names, material_summary = _material_rules(
        arrays, soil_document, config
    )
    arrays["material_rule"] = material
    arrays["material_rule_names"] = np.asarray(material_names, dtype="U32")
    valid = arrays["valid"].astype(bool)
    solve = arrays["solve_domain"].astype(bool)
    outlet = arrays["outlet"].astype(bool)
    water = arrays["water"].astype(bool)
    objects = arrays["object"].astype(bool)
    non_heightfield = arrays["non_heightfield"].astype(bool)
    protected = arrays["protected_structure"].astype(bool)
    hard = water | objects | non_heightfield | protected
    domain = SlopeDomain(
        site_id="development_a",
        bbox_en=tuple(float(value) for value in control_bbox),
        texel_m=1.0,
        height_m=height.astype(np.float64),
        valid=valid,
        solve_domain=solve,
        upstream_domain=arrays["upstream_domain"].astype(bool),
        outlet=outlet,
        collar=arrays["collar"].astype(bool),
        routing_barrier=objects | non_heightfield | (water & ~outlet),
        hard_exclusion=hard,
        unknown=arrays["unknown"].astype(bool),
        vegetation_cover=arrays["vegetation_evidence"].astype(np.float64),
        seep_likelihood=seep_support.astype(np.float64),
        upstream_water_m3=np.zeros(height.shape, dtype=np.float64),
        material_rule=material,
        material_rule_names=material_names,
        source_identity={
            "role": f"control_plus{control_expansion_m}",
            "condition_bundle": _file_identity(bundle_path),
            "corrected_lod0": corrected_lod0,
            "soil_window": _file_identity(soil_path),
            "geology_window": _file_identity(geology_path),
            "material_assignment": material_summary,
        },
    )
    facts = {
        "bbox_en": list(control_bbox),
        "shape": list(height.shape),
        "corrected_lod0": corrected_lod0,
        "soil_window": _file_identity(soil_path),
        "geology_window": _file_identity(geology_path),
        "soil_semantics": soil_semantics,
        "material_assignment": material_summary,
        "seep_support": seep_derivation,
        "etak_records": records,
        "counts": {
            "valid": int(valid.sum()),
            "solve_domain": int(solve.sum()),
            "upstream_domain": int(domain.upstream_domain.sum()),
            "outlet": int(outlet.sum()),
            "collar": int(domain.collar.sum()),
            "edge_leak": int(arrays["edge_leak"].sum()),
            "unknown": int(domain.unknown.sum()),
            "material_supported": int((material >= 0).sum()),
        },
    }
    return domain, arrays, facts


def _crop_masks(
    domain: SlopeDomain, crops: list[dict[str, Any]]
) -> tuple[np.ndarray, dict[str, int]]:
    mask = np.zeros(domain.height_m.shape, dtype=bool)
    counts: dict[str, int] = {}
    grid = load_base().grid
    e0, n0, _, n1 = domain.bbox_en
    for row in crops:
        lod, cx, cz = row["output_chunk"]
        bounds = chunk_bounds_en(grid, ChunkId(cx=cx, cz=cz, lod=lod))
        ce0, cn0, ce1, cn1 = bounds
        rows = slice(int(n1 - cn1), int(n1 - cn0))
        columns = slice(int(ce0 - e0), int(ce1 - e0))
        if domain.height_m[rows, columns].shape != (128, 128):
            raise ValueError(f"{row['crop_id']} is outside or misaligned with domain")
        selected = domain.active[rows, columns]
        mask[rows, columns] = selected
        counts[row["crop_id"]] = int(selected.sum())
    return mask, counts


def _reverse_closure(
    target_a: np.ndarray,
    target_b: np.ndarray,
    weight_a: np.ndarray,
    weight_b: np.ndarray,
    spill_target: np.ndarray,
    seeds: np.ndarray,
) -> np.ndarray:
    flat_a = target_a.ravel()
    flat_b = target_b.ravel()
    flat_wa = weight_a.ravel()
    flat_wb = weight_b.ravel()
    flat_spill = spill_target.ravel()
    source_index = np.arange(flat_a.size, dtype=np.int64)
    use_a = (flat_a >= 0) & (flat_wa > 0.0)
    use_b = (flat_b >= 0) & (flat_wb > 0.0)
    use_spill = flat_spill >= 0
    sources = np.concatenate(
        (source_index[use_a], source_index[use_b], source_index[use_spill])
    )
    destinations = np.concatenate(
        (flat_a[use_a], flat_b[use_b], flat_spill[use_spill])
    )
    order = np.argsort(destinations, kind="stable")
    sources = sources[order]
    destinations = destinations[order]
    counts = np.bincount(destinations, minlength=flat_a.size)
    offsets = np.empty(flat_a.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    dependency = seeds.ravel().copy()
    queue = deque(int(value) for value in np.flatnonzero(dependency))
    while queue:
        destination = queue.popleft()
        for source in sources[offsets[destination] : offsets[destination + 1]]:
            source_value = int(source)
            if not dependency[source_value]:
                dependency[source_value] = True
                queue.append(source_value)
    return dependency.reshape(seeds.shape)


def _routing_evidence(
    domain: SlopeDomain,
    crops: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    result = route_continuous(
        domain, local_water_depth_m=np.zeros(domain.height_m.shape, dtype=np.float64)
    )
    spill_target, capacity, spill_order, spill_depth, closed = _depression_spills(
        domain,
        result.target_a,
        result.target_b,
        result.weight_a,
        result.weight_b,
    )
    if not np.array_equal(
        capacity.reshape(domain.height_m.shape), result.depression_capacity_m3
    ):
        raise RuntimeError("diagnostic spill capacity differs from routed capacity")
    seeds, crop_counts = _crop_masks(domain, crops)
    dependency = _reverse_closure(
        result.target_a,
        result.target_b,
        result.weight_a,
        result.weight_b,
        spill_target.reshape(domain.height_m.shape),
        seeds,
    )
    flat_dependency = dependency.ravel()
    indices = np.flatnonzero(flat_dependency)
    flat_a = result.target_a.ravel()
    flat_b = result.target_b.ravel()
    flat_wa = result.weight_a.ravel()
    flat_wb = result.weight_b.ravel()
    flat_spill = spill_target.ravel()
    rows, cols = domain.height_m.shape
    boundary = np.zeros((rows, cols), dtype=bool)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    arrays = {
        "crop_seed_flat_index": np.flatnonzero(seeds).astype(np.int64),
        "dependency_flat_index": indices.astype(np.int64),
        "target_a_local_flat": flat_a[indices].astype(np.int64),
        "target_b_local_flat": flat_b[indices].astype(np.int64),
        "weight_a": flat_wa[indices].astype(np.float64),
        "weight_b": flat_wb[indices].astype(np.float64),
        "spill_target_local_flat": flat_spill[indices].astype(np.int64),
        "depression_capacity_m3": capacity.ravel()[indices].astype(np.float64),
        "spill_order_m": spill_order.ravel()[indices].astype(np.float64),
        "spill_depth": spill_depth.ravel()[indices].astype(np.int32),
    }
    facts = {
        "crop_active_seed_cells": crop_counts,
        "dependency_cells": int(indices.size),
        "raw_branch_a_edges": int(np.count_nonzero((flat_a >= 0) & flat_dependency)),
        "raw_branch_b_edges": int(np.count_nonzero((flat_b >= 0) & flat_dependency)),
        "split_branch_sources": int(
            np.count_nonzero((flat_a >= 0) & (flat_b >= 0) & flat_dependency)
        ),
        "spill_dependency_sinks": int(
            np.count_nonzero((flat_spill != -1) & flat_dependency)
        ),
        "closed_depression_cells_domain": int(closed),
        "dependency_touches_storage_edge": bool(np.any(dependency & boundary)),
    }
    return {
        **arrays,
        "dependency_mask": dependency,
        "target_a": result.target_a,
        "target_b": result.target_b,
        "weight_a_full": result.weight_a,
        "weight_b_full": result.weight_b,
        "spill_target": spill_target.reshape(domain.height_m.shape),
        "capacity": capacity.reshape(domain.height_m.shape),
        "spill_order": spill_order.reshape(domain.height_m.shape),
        "spill_depth_full": spill_depth.reshape(domain.height_m.shape),
    }, facts


def _embed_slices(
    inner: SlopeDomain, outer: SlopeDomain
) -> tuple[slice, slice]:
    e0, n0, e1, n1 = inner.bbox_en
    oe0, on0, oe1, on1 = outer.bbox_en
    if not (oe0 <= e0 < e1 <= oe1 and on0 <= n0 < n1 <= on1):
        raise ValueError("canonical domain is not inside control domain")
    return (
        slice(int(on1 - n1), int(on1 - n0)),
        slice(int(e0 - oe0), int(e1 - oe0)),
    )


def _target_world_keys(
    targets: np.ndarray,
    domain: SlopeDomain,
    reference: SlopeDomain,
) -> np.ndarray:
    flat = targets.ravel()
    result = flat.copy()
    used = flat >= 0
    if not np.any(used):
        return result
    rows, cols = np.unravel_index(flat[used], domain.height_m.shape)
    east = int(domain.bbox_en[0]) + cols
    north = int(domain.bbox_en[3]) - 1 - rows
    width = reference.height_m.shape[1]
    result[used] = (north - int(reference.bbox_en[1])) * width + (
        east - int(reference.bbox_en[0])
    )
    return result


def _mismatch(a: np.ndarray, b: np.ndarray) -> tuple[int, float | None]:
    equal = (a == b) | (np.isnan(a) & np.isnan(b)) if a.dtype.kind == "f" else a == b
    count = int(np.count_nonzero(~equal))
    if count == 0 or a.dtype.kind not in "fiu":
        return count, 0.0 if count == 0 and a.dtype.kind in "fiu" else None
    difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
    finite = difference[np.isfinite(difference)]
    return count, float(np.max(finite, initial=0.0))


def _closure_metrics(
    canonical: SlopeDomain,
    control: SlopeDomain,
    canonical_arrays: dict[str, np.ndarray],
    control_arrays: dict[str, np.ndarray],
    canonical_routing: dict[str, np.ndarray],
    control_routing: dict[str, np.ndarray],
    plan: dict[str, Any],
) -> dict[str, Any]:
    rows, columns = _embed_slices(canonical, control)
    canonical_dependency = canonical_routing["dependency_mask"]
    control_dependency = control_routing["dependency_mask"]
    control_overlap = control_dependency[rows, columns]
    canonical_only = canonical_dependency & ~control_overlap
    control_only_overlap = control_overlap & ~canonical_dependency
    outside = control_dependency.copy()
    outside[rows, columns] = False

    crop_mask, _ = _crop_masks(canonical, plan["crops"])
    compare_mask = canonical_dependency | crop_mask
    canonical_indices = np.flatnonzero(compare_mask)
    can_rows, can_cols = np.unravel_index(canonical_indices, canonical.height_m.shape)
    control_rows = can_rows + rows.start
    control_cols = can_cols + columns.start
    control_indices = np.ravel_multi_index(
        (control_rows, control_cols), control.height_m.shape
    )

    condition_identity: dict[str, Any] = {}
    for name in _CLOSURE_CONDITION_FIELDS:
        left = np.asarray(canonical_arrays[name]).ravel()[canonical_indices]
        right = np.asarray(control_arrays[name]).ravel()[control_indices]
        count, maximum = _mismatch(left, right)
        condition_identity[name] = {
            "mismatch_cells": count,
            "max_abs": maximum,
            "exact": count == 0,
        }

    branch_identity: dict[str, Any] = {}
    for branch in ("a", "b"):
        left_target = _target_world_keys(
            canonical_routing[f"target_{branch}"], canonical, control
        )[canonical_indices]
        right_target = _target_world_keys(
            control_routing[f"target_{branch}"], control, control
        )[control_indices]
        target_count, _ = _mismatch(left_target, right_target)
        left_weight = canonical_routing[f"weight_{branch}_full"].ravel()[
            canonical_indices
        ]
        right_weight = control_routing[f"weight_{branch}_full"].ravel()[control_indices]
        weight_count, weight_max = _mismatch(left_weight, right_weight)
        branch_identity[branch] = {
            "target_mismatch_cells": target_count,
            "weight_mismatch_cells": weight_count,
            "weight_max_abs": weight_max,
            "exact": target_count == 0 and weight_count == 0,
        }

    canonical_spill = _target_world_keys(
        canonical_routing["spill_target"], canonical, control
    )[canonical_indices]
    control_spill = _target_world_keys(
        control_routing["spill_target"], control, control
    )[control_indices]
    spill_target_count, _ = _mismatch(canonical_spill, control_spill)
    spill_identity: dict[str, Any] = {
        "target_mismatch_cells": spill_target_count,
    }
    for name in ("capacity", "spill_order", "spill_depth_full"):
        left = canonical_routing[name].ravel()[canonical_indices]
        right = control_routing[name].ravel()[control_indices]
        count, maximum = _mismatch(left, right)
        spill_identity[name] = {
            "mismatch_cells": count,
            "max_abs": maximum,
            "exact": count == 0,
        }
    spill_identity["exact"] = spill_target_count == 0 and all(
        spill_identity[name]["exact"]
        for name in ("capacity", "spill_order", "spill_depth_full")
    )

    dependency_identity = {
        "canonical_only_cells": int(canonical_only.sum()),
        "control_only_cells_inside_canonical": int(control_only_overlap.sum()),
        "control_dependency_cells_outside_canonical": int(outside.sum()),
        "exact": not (
            np.any(canonical_only) or np.any(control_only_overlap) or np.any(outside)
        ),
    }
    conditions_exact = all(row["exact"] for row in condition_identity.values())
    branches_exact = all(row["exact"] for row in branch_identity.values())
    closure_pass = (
        dependency_identity["exact"]
        and conditions_exact
        and branches_exact
        and spill_identity["exact"]
        and not canonical_routing["facts"]["dependency_touches_storage_edge"]
        and not control_routing["facts"]["dependency_touches_storage_edge"]
    )
    return {
        "closure_pass": bool(closure_pass),
        "dependency_set_identity": dependency_identity,
        "condition_identity": condition_identity,
        "raw_dinf_branch_identity": branch_identity,
        "finite_depression_spill_identity": spill_identity,
        "canonical_dependency_touches_storage_edge": canonical_routing["facts"][
            "dependency_touches_storage_edge"
        ],
        "control_dependency_touches_storage_edge": control_routing["facts"][
            "dependency_touches_storage_edge"
        ],
        "next_required_expansion": None
        if closure_pass
        else {
            "promote_canonical_expansion_from_original_m": plan[
                "control_expansion_from_original_m"
            ],
            "materialize_control_expansion_from_original_m": plan[
                "control_expansion_from_original_m"
            ]
            + plan["next_expansion_step_m"],
            "additional_margin_each_side_m": plan["next_expansion_step_m"],
            "reason": "dependency closure or identity is not stable; expansion must continue without trimming or masking",
        },
    }


def materialize_domain_closure(plan_path: Path, output_parent: Path | None = None) -> Path:
    plan_path = Path(plan_path)
    plan, bundle_path, site_config_path, solver_config_path = _load_plan(plan_path)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("schema_version") != _BUNDLE_SCHEMA:
        raise ValueError("unsupported condition bundle")
    policies = bundle["recipe"]["policies"]
    if set(policies["forbidden_etak_ids"]) != {9688685, 9688702}:
        raise ValueError("condition bundle forbidden identities differ")
    if policies.get("sealed_ood_used") is not False:
        raise ValueError("condition bundle consumed sealed OOD2")
    config = load_process_config(solver_config_path)
    spec = _site_spec(site_config_path)
    bundle_canonical, bundle_canonical_arrays, site = _load_canonical(bundle, config)
    original_bbox = tuple(int(value) for value in site["bbox_en"])
    canonical_expansion = int(plan["canonical_expansion_from_original_m"])
    control_expansion = int(plan["control_expansion_from_original_m"])
    expected_canonical = _expanded_bbox(original_bbox, canonical_expansion)
    if canonical_expansion == 128:
        canonical = bundle_canonical
        canonical_arrays = bundle_canonical_arrays
        canonical_source = {
            "source_arrays": site["enlarged_domain_arrays"],
            "source_metadata": site["enlarged_domain"],
        }
    else:
        canonical, canonical_arrays, canonical_identity = _load_promoted_canonical(
            _bound(plan["canonical_authority"]), expected_canonical, config
        )
        canonical_source = {"promoted_authority": canonical_identity}
    if tuple(int(value) for value in canonical.bbox_en) != expected_canonical:
        raise ValueError("canonical Development A domain differs")
    control_bbox = _expanded_bbox(original_bbox, control_expansion)
    control, control_arrays, control_facts = _build_control(
        bundle=bundle,
        bundle_path=bundle_path,
        plan_path=plan_path,
        spec=spec,
        control_bbox=control_bbox,
        control_expansion_m=control_expansion,
        config=config,
    )
    canonical_routing, canonical_routing_facts = _routing_evidence(
        canonical, plan["crops"]
    )
    control_routing, control_routing_facts = _routing_evidence(control, plan["crops"])
    canonical_routing["facts"] = canonical_routing_facts
    control_routing["facts"] = control_routing_facts
    closure = _closure_metrics(
        canonical,
        control,
        canonical_arrays,
        control_arrays,
        canonical_routing,
        control_routing,
        plan,
    )

    recipe = {
        "schema_version": "laas.erodible-slope-domain-closure-recipe/1",
        "plan": _file_identity(plan_path),
        "condition_bundle": _file_identity(bundle_path),
        "site_config": _file_identity(site_config_path),
        "solver_config": _file_identity(solver_config_path),
        "source_site_id": "development_a",
        "original_bbox_en": list(original_bbox),
        "canonical_bbox_en": list(expected_canonical),
        "control_bbox_en": list(control_bbox),
        "canonical_expansion_from_original_m": canonical_expansion,
        "control_expansion_from_original_m": control_expansion,
        "canonical_authority": _file_identity(_bound(plan["canonical_authority"]))
        if "canonical_authority" in plan
        else None,
        "crops": plan["crops"],
        "control_sources": {
            key: control.source_identity[key]
            for key in ("corrected_lod0", "soil_window", "geology_window")
        },
        "implementation": _implementation_identity(),
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "byteorder": sys.byteorder,
            "numpy": np.__version__,
        },
        "routing_dependency_semantics": {
            "raw_multi_flow": "exact solver route_continuous target_a/target_b and weights",
            "finite_depression_spill": "exact solver _depression_spills target/capacity/order/depth",
            "closure": "reverse reachability from both crop active cells over every positive raw branch plus every finite spill-to-sink edge",
            "identity_space": "EPSG:3301 one-metre cell world coordinates",
            "forbidden_actions": [
                "no_dependency_trimming",
                "no_boundary_masking",
                "no_threshold_weakening",
                "no_solver_math_change",
            ],
        },
        "forbidden_etak_ids": [9688685, 9688702],
        "sealed_ood_used": False,
        "imagery_inspected": False,
    }
    recipe_sha256 = hashlib.sha256(_canonical_bytes(recipe)).hexdigest()
    parent = (
        Path(output_parent)
        if output_parent is not None
        else DATA_WORK
        / "microtopography"
        / "erodible-slope"
        / "conditions"
        / "domain-closure"
        / "sha256"
    )
    root = parent / recipe_sha256
    if root.exists():
        manifest = root / "manifest.json"
        if not manifest.is_file():
            raise RuntimeError("existing closure transaction is incomplete")
        return manifest
    temporary = parent / f".{recipe_sha256}.{os.getpid()}.tmp"
    if temporary.exists():
        raise RuntimeError(f"stale closure transaction exists: {temporary}")
    temporary.mkdir(parents=True)
    canonical_domain_path = temporary / f"canonical-plus{canonical_expansion}-domain.npz"
    control_domain_path = temporary / f"control-plus{control_expansion}-domain.npz"
    canonical_routing_path = (
        temporary / f"canonical-plus{canonical_expansion}-routing-dependencies.npz"
    )
    control_routing_path = (
        temporary / f"control-plus{control_expansion}-routing-dependencies.npz"
    )
    _write_npz(
        canonical_domain_path,
        {name: canonical_arrays[name] for name in _DOMAIN_FIELDS},
    )
    _write_npz(
        control_domain_path,
        {name: control_arrays[name] for name in _DOMAIN_FIELDS},
    )
    routing_fields = (
        "crop_seed_flat_index",
        "dependency_flat_index",
        "target_a_local_flat",
        "target_b_local_flat",
        "weight_a",
        "weight_b",
        "spill_target_local_flat",
        "depression_capacity_m3",
        "spill_order_m",
        "spill_depth",
    )
    _write_npz(
        canonical_routing_path,
        {name: canonical_routing[name] for name in routing_fields},
    )
    _write_npz(
        control_routing_path,
        {name: control_routing[name] for name in routing_fields},
    )
    etak_path = temporary / f"control-plus{control_expansion}-etak.json"
    etak_path.write_text(
        json.dumps(
            {
                "schema_version": "laas.erodible-slope-domain-closure-etak/1",
                "bbox_en": list(control_bbox),
                "forbidden_etak_ids": [9688685, 9688702],
                "sealed_ood_used": False,
                "records": control_facts.pop("etak_records"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    metrics = {
        "schema_version": "laas.erodible-slope-domain-closure-metrics/1",
        "recipe_sha256": recipe_sha256,
        "canonical": {
            "bbox_en": list(canonical.bbox_en),
            "shape": list(canonical.height_m.shape),
            **canonical_source,
            "routing_dependencies": canonical_routing_facts,
        },
        "control": {
            **control_facts,
            "routing_dependencies": control_routing_facts,
        },
        "closure": closure,
    }
    metrics_path = temporary / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_identities = {
        "canonical_domain": _file_identity(
            canonical_domain_path, published_path=root / canonical_domain_path.name
        ),
        "control_domain": _file_identity(
            control_domain_path, published_path=root / control_domain_path.name
        ),
        "canonical_routing_dependencies": _file_identity(
            canonical_routing_path, published_path=root / canonical_routing_path.name
        ),
        "control_routing_dependencies": _file_identity(
            control_routing_path, published_path=root / control_routing_path.name
        ),
        "control_etak": _file_identity(etak_path, published_path=root / etak_path.name),
        "metrics": _file_identity(metrics_path, published_path=root / metrics_path.name),
    }
    manifest = {
        "schema_version": "laas.erodible-slope-domain-closure/1",
        "recipe_sha256": recipe_sha256,
        "recipe": recipe,
        "state": "closure_pass" if closure["closure_pass"] else "closure_failed_expand_again",
        "solver_use_authorized": bool(closure["closure_pass"]),
        "canonical_role": "solver_authorized_condition_authority"
        if closure["closure_pass"]
        else "proposed_only_until_dependency_closure_passes",
        "closure": closure,
        "routing_dependency_counts": {
            "canonical": canonical_routing_facts,
            "control": control_routing_facts,
        },
        "outputs": output_identities,
        "sealed_ood_used": False,
        "imagery_inspected": False,
    }
    (temporary / "recipe.json").write_text(
        json.dumps(recipe, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(root)
    return root / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path)
    args = parser.parse_args()
    print(materialize_domain_closure(args.plan, args.output_parent))


if __name__ == "__main__":
    main()
