"""Independent verifier for the Development-A ALS/TGV packed research preview."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from .....config import EncodeConfig, load_base
from .....cook.chunkio import read_chunk_v2
from .....cook.encode import decode_quant16, encode_quant16_checked
from .....cook.micro_hierarchy import assemble_parent_source_memmap, box_mean4_striped
from .....height_geom import HeightChunkId, plan_hero, plan_parent_set
from .....micro_verify import evidence_sha256, transient_merkle_root
from .....release import audit_base_release, micro_verification_binding, read_v1_index
from ....repair.base_transaction import AuditedFormat1HeightSource
from ....repair.pinned_baseline import PinnedDecodedBaseline
from ....repair.prolong import prolong_structural_4x
from .evidence import load_evidence


VERIFIER_ID = "assetgen.als-tgv-structural-preview-verify.v1"
RECIPE_KIND = "research-als-tgv-structural-preview-v1"
ARTIFACT_SCHEMA = "laas.coastal-escarpment-als-tgv-artifact/1"
ARTIFACT_STATE = "development_float_candidate_accepted"
ARTIFACT_BBOX = (680416.0, 6444384.0, 680736.0, 6444576.0)
SOURCE_PITCH_M = 0.25
PARENTS = tuple(
    sorted(
        (
            HeightChunkId(-1, 608, 372),
            HeightChunkId(-1, 609, 372),
            HeightChunkId(-1, 608, 373),
            HeightChunkId(-1, 609, 373),
        )
    )
)
AUTHORITY = HeightChunkId(0, 152, 93)
FINE_CORE = 2048


def verifier_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _gate(evidence: Any) -> dict[str, Any]:
    return {"passed": True, "evidence": evidence, "evidenceSha256": evidence_sha256(evidence)}


def _load_json(path: Path, sha_path: Path | None = None) -> dict[str, Any]:
    blob = path.read_bytes()
    if sha_path is not None and hashlib.sha256(blob).hexdigest() != sha_path.read_text().strip():
        raise ValueError(f"JSON identity mismatch: {path}")
    return json.loads(blob)


def _decoded(path: Path, encode: EncodeConfig):
    meta, payload = read_chunk_v2(path)
    return meta, payload, decode_quant16(encode, payload, meta.res, meta.qoffset, meta.qscale)


def _index_blob(manifest_path: Path, layer: str) -> bytes:
    manifest = json.loads(manifest_path.read_bytes())
    return (manifest_path.parent / manifest["layers"][layer]["index"]).read_bytes()


def _height_inventory(manifest_path: Path) -> dict[tuple[int, int, int], tuple[int, int]]:
    manifest = json.loads(manifest_path.read_bytes())
    index = manifest_path.parent / manifest["layers"]["height"]["index"]
    return {record.key: (record.size, record.hash64) for record in read_v1_index(index)}


def _load_artifact(inputs: dict[str, Any]):
    artifact_root = Path(inputs["artifact"]["root"])
    manifest_path = artifact_root / "manifest.json"
    if _sha256(manifest_path) != inputs["artifact"]["manifestSha256"]:
        raise ValueError("ALS/TGV artifact manifest changed")
    manifest = json.loads(manifest_path.read_bytes())
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA
        or manifest.get("build_id") != inputs["artifact"]["buildId"]
        or manifest.get("state") != ARTIFACT_STATE
        or manifest.get("outputs") != inputs["artifact"]["outputs"]
    ):
        raise ValueError("ALS/TGV artifact boundary changed")
    config_path = Path(__file__).parents[7] / manifest["config"]["path"]
    if _sha256(config_path) != inputs["artifact"]["configSha256"]:
        raise ValueError("ALS/TGV config changed")
    config = json.loads(config_path.read_bytes())
    arrays = {}
    for name in ("reconstruction", "residual"):
        row = manifest["outputs"][name]
        path = artifact_root / row["path"]
        if path.stat().st_size != row["bytes"] or _sha256(path) != row["sha256"]:
            raise ValueError(f"ALS/TGV {name} changed")
        arrays[name] = np.load(path, mmap_mode="r")
    for name in ("metrics", "qa_index"):
        row = manifest["outputs"][name]
        path = artifact_root / row["path"]
        if path.stat().st_size != row["bytes"] or _sha256(path) != row["sha256"]:
            raise ValueError(f"ALS/TGV {name} changed")
    metrics = json.loads((artifact_root / manifest["outputs"]["metrics"]["path"]).read_bytes())
    evidence = load_evidence(config)
    reconstruction = arrays["reconstruction"]
    residual = arrays["residual"]
    hard = evidence.hard_zero | evidence.mapped_face
    if (
        reconstruction.shape != (769, 1281)
        or residual.shape != reconstruction.shape
        or not np.array_equal(reconstruction, evidence.c0_m + residual)
        or np.any(residual[hard] != 0.0)
        or metrics.get("status") != ARTIFACT_STATE
        or metrics.get("holdout", {}).get("pass") is not True
        or metrics.get("hard_laws", {}).get("pass") is not True
    ):
        raise ValueError("ALS/TGV accepted numerical evidence changed")
    return evidence.c0_m, reconstruction, hard, metrics


def _sample_support(base, chunk, c0, c1, hard):
    origin_e = base.grid.anchor_e + chunk.cx * 128.0
    origin_n = base.grid.anchor_n - chunk.cz * 128.0
    indices = np.arange(-4, 516, dtype=np.float64)
    east = origin_e + (indices + 0.5) * SOURCE_PITCH_M
    north = origin_n - (indices + 0.5) * SOURCE_PITCH_M
    cols = (east - ARTIFACT_BBOX[0]) / SOURCE_PITCH_M
    rows = (ARTIFACT_BBOX[3] - north) / SOURCE_PITCH_M
    valid_r = (rows >= 0.0) & (rows <= c0.shape[0] - 1)
    valid_c = (cols >= 0.0) & (cols <= c0.shape[1] - 1)
    overlay = valid_r[:, None] & valid_c[None, :]
    sampled_c0 = np.zeros((520, 520), dtype=np.float64)
    sampled_c1 = np.zeros((520, 520), dtype=np.float64)
    target_hard = np.zeros((520, 520), dtype=bool)
    if overlay.any():
        rr = rows[valid_r]
        cc = cols[valid_c]
        grid_r, grid_c = np.meshgrid(rr, cc, indexing="ij")
        target = np.ix_(valid_r, valid_c)
        sampled_c0[target] = ndimage.map_coordinates(c0, [grid_r, grid_c], order=1, mode="nearest")
        sampled_c1[target] = ndimage.map_coordinates(c1, [grid_r, grid_c], order=1, mode="nearest")
        r0, r1 = np.floor(grid_r).astype(np.int64), np.ceil(grid_r).astype(np.int64)
        c0i, c1i = np.floor(grid_c).astype(np.int64), np.ceil(grid_c).astype(np.int64)
        target_hard[target] = hard[r0, c0i] | hard[r0, c1i] | hard[r1, c0i] | hard[r1, c1i]
    return sampled_c0, sampled_c1, overlay, overlay & target_hard


def _expected_core(baseline, base, chunk, c0, c1, hard):
    support = baseline.reconstruct(chunk, halo_samples=4).tile.height
    baseline_fine = prolong_structural_4x(support, parent_rows=(4, 516), parent_cols=(4, 516))
    sampled_c0, sampled_c1, overlay, target_hard = _sample_support(base, chunk, c0, c1, hard)
    c0_support = np.array(support, dtype=np.float64, copy=True)
    c1_support = np.array(support, dtype=np.float64, copy=True)
    c0_support[overlay] = sampled_c0[overlay]
    c1_support[overlay] = np.where(target_hard[overlay], sampled_c0[overlay], sampled_c1[overlay])
    c0_fine = prolong_structural_4x(c0_support, parent_rows=(4, 516), parent_cols=(4, 516))
    c1_fine = prolong_structural_4x(c1_support, parent_rows=(4, 516), parent_cols=(4, 516))
    fine_overlay = np.repeat(np.repeat(overlay[4:516, 4:516], 4, axis=0), 4, axis=1)
    fine_hard = np.repeat(np.repeat(target_hard[4:516, 4:516], 4, axis=0), 4, axis=1)
    expected = np.array(baseline_fine, dtype=np.float64, copy=True)
    expected[fine_overlay] = c1_fine[fine_overlay]
    expected[fine_overlay & fine_hard] = c0_fine[fine_overlay & fine_hard]
    return expected, c0_fine, fine_overlay & fine_hard


def _max_seams(paths: dict[HeightChunkId, Path], encode: EncodeConfig) -> tuple[float, int]:
    maximum = 0.0
    comparisons = 0
    available = set(paths)
    for chunk in sorted(available):
        east = HeightChunkId(-2, chunk.cx + 1, chunk.cz)
        south = HeightChunkId(-2, chunk.cx, chunk.cz + 1)
        values = _decoded(paths[chunk], encode)[2]
        if east in available:
            maximum = max(maximum, float(np.max(np.abs(values[:, -1] - _decoded(paths[east], encode)[2][:, 0]))))
            comparisons += 1
        if south in available:
            maximum = max(maximum, float(np.max(np.abs(values[-1, :] - _decoded(paths[south], encode)[2][0, :]))))
            comparisons += 1
    return maximum, comparisons


def _parent_overlay_mask(base, parent):
    east = base.grid.anchor_e + parent.cx * 512.0 + (np.arange(2048) + 0.5) * SOURCE_PITCH_M
    north = base.grid.anchor_n - parent.cz * 512.0 - (np.arange(2048) + 0.5) * SOURCE_PITCH_M
    return (
        (north[:, None] >= ARTIFACT_BBOX[1])
        & (north[:, None] <= ARTIFACT_BBOX[3])
        & (east[None, :] >= ARTIFACT_BBOX[0])
        & (east[None, :] <= ARTIFACT_BBOX[2])
    )


def verify_als_tgv_preview(build_digest: str, base_manifest_path: Path, base_out_root: Path, work_root: Path) -> Path:
    base = load_base()
    build_root = work_root / "builds" / build_digest
    expectation = _load_json(build_root / "expectation.json", build_root / "expectation.sha256")
    plan = _load_json(build_root / "plan.json", build_root / "plan.sha256")
    if (
        expectation.get("recipeKind") != RECIPE_KIND
        or plan.get("microRecipeKind") != RECIPE_KIND
        or expectation.get("recipeSha256") != build_digest
        or plan.get("recipeSha256") != build_digest
        or expectation.get("verifier") != {"id": VERIFIER_ID, "sourceSha256": verifier_source_sha256()}
    ):
        raise ValueError("ALS/TGV recipe or verifier identity differs")
    inputs = expectation["recipeInputs"]
    c0, c1, hard, metrics = _load_artifact(inputs)
    coverage = plan_parent_set(PARENTS)
    expected_coverage = {
        "sites": [{"siteId": "development-a-adjacent-escarpment", "reviewBboxEn": list(ARTIFACT_BBOX)}],
        "parents": [[c.lod, c.cx, c.cz] for c in coverage.parents],
        "publishedFine": [[c.lod, c.cx, c.cz] for c in coverage.published_fine],
        "transientSupport": [[c.lod, c.cx, c.cz] for c in coverage.transient_support],
        "authorities": [[c.lod, c.cx, c.cz] for c in coverage.authorities_lod0],
    }
    if plan.get("microCoverage") != expected_coverage:
        raise ValueError("ALS/TGV coverage differs from the four-parent closure")
    evidence = _load_json(build_root / "evidence/als-tgv-structural-preview.json")
    identities = {HeightChunkId(*row["key"]): row for row in evidence["children"]}
    all_fine = (*coverage.published_fine, *coverage.transient_support)
    if set(identities) != set(all_fine):
        raise ValueError("ALS/TGV evidence lacks the exact fine closure")
    published = set(coverage.published_fine)
    paths: dict[HeightChunkId, Path] = {}
    headers = []
    for chunk in all_fine:
        root = build_root / ("chunks" if chunk in published else "transient")
        path = root / "height/-2" / f"{chunk.cx}_{chunk.cz}.lac"
        identity = identities[chunk]
        if path.stat().st_size != identity["bytes"] or _sha256(path) != identity["sha256"]:
            raise ValueError(f"ALS/TGV fine chunk changed: {chunk}")
        meta, _payload, _values = _decoded(path, base.encode)
        if (
            (meta.lod, meta.cx, meta.cz) != (chunk.lod, chunk.cx, chunk.cz)
            or meta.res != 2049
            or abs(meta.qscale - 0.002) > 1e-8
            or abs(meta.qoffset - 30.0) > 1e-8
        ):
            raise ValueError(f"ALS/TGV fine header differs: {chunk}")
        paths[chunk] = path
        headers.append({"key": identity["key"], "qoffset": meta.qoffset, "qscale": meta.qscale})

    source_manifest = Path(inputs["sourceBase"]["manifest"])
    source_sha = inputs["sourceBase"]["manifestSha256"]
    baseline = PinnedDecodedBaseline(
        manifest_path=source_manifest,
        manifest_sha256=source_sha,
        content_root=base_out_root,
        encode=base.encode,
        cache_chunks=4,
    )
    maximum_structural_error = 0.0
    maximum_hard_error = 0.0
    affected_chunks = 0
    for chunk in coverage.published_fine:
        expected, expected_c0, fine_hard = _expected_core(baseline, base, chunk, c0, c1, hard)
        decoded = _decoded(paths[chunk], base.encode)[2][:-1, :-1]
        maximum_structural_error = max(
            maximum_structural_error,
            float(np.max(np.abs(decoded.astype(np.float64) - expected))),
        )
        if fine_hard.any():
            affected_chunks += 1
            maximum_hard_error = max(
                maximum_hard_error,
                float(np.max(np.abs(decoded[fine_hard].astype(np.float64) - expected_c0[fine_hard]))),
            )
    if maximum_structural_error > 0.00101 or maximum_hard_error > 0.00101:
        raise ValueError(
            f"ALS/TGV packed reconstruction errors structural={maximum_structural_error}, hard={maximum_hard_error}"
        )
    maximum_seam, seam_comparisons = _max_seams(paths, base.encode)
    if maximum_seam != 0.0:
        raise ValueError(f"ALS/TGV decoded fine seam is {maximum_seam} m")

    parent_rows = []
    parent_decoded: dict[HeightChunkId, np.ndarray] = {}
    parent_evidence = {HeightChunkId(*row["key"]): row for row in evidence["parents"]}
    for parent in PARENTS:
        hero = plan_hero(parent.cx, parent.cz)
        mosaic = assemble_parent_source_memmap(
            build_root / f"scratch/verify-als-tgv-parent-{parent.cx}-{parent.cz}.f32",
            hero,
            lambda chunk: _decoded(paths[chunk], base.encode)[2],
        )
        expected_parent = box_mean4_striped(mosaic)
        path = build_root / f"chunks/height/-1/{parent.cx}_{parent.cz}.lac"
        meta, payload, decoded = _decoded(path, base.encode)
        identity = parent_evidence[parent]
        if path.stat().st_size != identity["bytes"] or _sha256(path) != identity["sha256"]:
            raise ValueError(f"ALS/TGV parent changed: {parent}")
        expected_payload, expected_offset, expected_scale = encode_quant16_checked(
            base.encode, expected_parent, meta.qscale, meta.qoffset
        )
        if payload != expected_payload or meta.qoffset != expected_offset or meta.qscale != expected_scale:
            raise ValueError(f"ALS/TGV parent is not byte-exactly child-derived: {parent}")
        parent_decoded[parent] = decoded
        parent_rows.append({"key": identity["key"], "sha256": identity["sha256"], "payloadByteExact": True})
    parent_seam = 0.0
    for parent in PARENTS:
        east = HeightChunkId(-1, parent.cx + 1, parent.cz)
        south = HeightChunkId(-1, parent.cx, parent.cz + 1)
        if east in parent_decoded:
            parent_seam = max(parent_seam, float(np.max(np.abs(parent_decoded[parent][:, -1] - parent_decoded[east][:, 0]))))
        if south in parent_decoded:
            parent_seam = max(parent_seam, float(np.max(np.abs(parent_decoded[parent][-1, :] - parent_decoded[south][0, :]))))
    if parent_seam != 0.0:
        raise ValueError(f"ALS/TGV decoded parent seam is {parent_seam} m")

    corrected = inputs["correctedBase"]
    if corrected["manifestSha256"] != expectation["baseManifestSha256"]:
        raise ValueError("ALS/TGV corrected base differs from expectation")
    audit_base_release(base_manifest_path, corrected["manifestSha256"], base_out_root)
    corrected_source = AuditedFormat1HeightSource(
        manifest_path=base_manifest_path,
        manifest_sha256=corrected["manifestSha256"],
        content_root=base_out_root,
        encode=base.encode,
        cache_chunks=2,
    )
    authority_values = corrected_source.load(AUTHORITY).decoded[:-1, :-1]
    maximum_lod0_error = 0.0
    for parent in PARENTS:
        reduced = box_mean4_striped(parent_decoded[parent][:2048, :2048])
        mask = _parent_overlay_mask(base, parent).reshape(512, 4, 512, 4).any(axis=(1, 3))
        x0 = (parent.cx - AUTHORITY.cx * 4) * 512
        z0 = (parent.cz - AUTHORITY.cz * 4) * 512
        actual = authority_values[z0 : z0 + 512, x0 : x0 + 512]
        maximum_lod0_error = max(
            maximum_lod0_error,
            float(np.max(np.abs(actual[mask] - reduced[mask]), initial=0.0)),
        )
    if maximum_lod0_error > 0.00501:
        raise ValueError(f"ALS/TGV corrected LOD0 closure is {maximum_lod0_error} m")

    audit_base_release(source_manifest, source_sha, base_out_root)
    source_doc = json.loads(source_manifest.read_bytes())
    inherited_layers = sorted(set(source_doc["layers"]) - {"height"})
    for layer in inherited_layers:
        if _index_blob(source_manifest, layer) != _index_blob(base_manifest_path, layer):
            raise ValueError(f"ALS/TGV corrected base changed inherited {layer}")
    before_height = _height_inventory(source_manifest)
    after_height = _height_inventory(base_manifest_path)
    if set(before_height) != set(after_height):
        raise ValueError("ALS/TGV corrected base changed inherited height keys")
    changed_height = {key for key in before_height if before_height[key] != after_height[key]}
    transaction_plan = _load_json(build_root / "corrected-base/resolved-hierarchy-plan.json")
    planned_height = {tuple(row["chunk"]) for row in transaction_plan["artifacts"]}
    if not changed_height.issubset(planned_height):
        raise ValueError("ALS/TGV corrected base contains unrelated hierarchy changes")

    transient_artifacts = [
        {"key": identities[c]["key"], "sha256": identities[c]["sha256"], "size": identities[c]["bytes"]}
        for c in coverage.transient_support
    ]
    hard_laws = metrics["hard_laws"]
    gates = {
        "coverage": _gate({**expected_coverage, "publishedCount": 68, "transientCount": len(coverage.transient_support)}),
        "parentClosure": _gate({
            "parents": parent_rows,
            "correctedHierarchyPlanSha256": corrected["hierarchyPlanSha256"],
        }),
        "headers": _gate({"fine": headers, "parentQscaleM": 0.005, "fixedQoffsetM": 30.0}),
        "decodedHierarchy": _gate({
            "maxLod0DecodedChildErrorM": maximum_lod0_error,
            "maskedSpliceThroughLod4": True,
            "changedHeightKeys": [list(key) for key in sorted(changed_height)],
            "noStaleUnrelatedAncestorChanges": True,
        }),
        "aprons": _gate({
            "maxDecodedFineSeamErrorM": maximum_seam,
            "maxDecodedParentSeamErrorM": parent_seam,
            "comparisons": seam_comparisons,
        }),
        "determinism": _gate({
            "artifactManifestSha256": inputs["artifact"]["manifestSha256"],
            "sourceBaseManifestSha256": source_sha,
            "verifierSourceSha256": verifier_source_sha256(),
        }),
        "hardMasks": _gate({
            "maxStructuralRoundTripErrorM": maximum_structural_error,
            "maxHardC0RoundTripErrorM": maximum_hard_error,
            "artifactAllHardResidualM": hard_laws["all_hard_residual_max_abs_m"],
            "artifactMappedFaceResidualM": hard_laws["mapped_face_residual_max_abs_m"],
            "artifactOuterCollarResidualM": hard_laws["outer_collar_residual_max_abs_m"],
            "conservativeStencilOwnership": True,
            "affectedPublishedChunkCount": affected_chunks,
            "fineMorphologyClaimBelow025M": "none",
            "inheritedLayerIndexesByteExact": inherited_layers,
        }),
    }
    report = {
        **micro_verification_binding(build_digest, work_root),
        "verifier": VERIFIER_ID,
        "verifierSourceSha256": verifier_source_sha256(),
        "pilotOnly": True,
        "passed": True,
        "gates": gates,
        "transientArtifacts": transient_artifacts,
        "transientSupportMerkleRoot": transient_merkle_root(transient_artifacts),
    }
    path = build_root / "micro-verify.json"
    blob = _json_bytes(report)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(blob)
    temporary.replace(path)
    return path
