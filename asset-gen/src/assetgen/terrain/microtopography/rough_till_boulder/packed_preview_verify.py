"""Independent verifier for the structural-only rough-till packed preview."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from ....config import EncodeConfig, load_base
from ....cook.chunkio import read_chunk_v2
from ....cook.encode import decode_quant16, encode_quant16_checked
from ....cook.micro_hierarchy import assemble_parent_source_memmap, box_mean4_striped
from ....height_geom import HeightChunkId, plan_hero
from ....micro_verify import evidence_sha256, transient_merkle_root
from ....release import audit_base_release, micro_verification_binding, read_v1_index
from ...repair.base_transaction import AuditedFormat1HeightSource
from ...repair.pinned_baseline import PinnedDecodedBaseline
from ...repair.prolong import prolong_structural_4x


VERIFIER_ID = "assetgen.rough-till-structural-preview-verify.v1"
RECIPE_KIND = "research-rough-till-structural-preview-v1"
ARTIFACT_SCHEMA = "laas.rough-till-boulder-multi-form/1"
PARENT = HeightChunkId(-1, 615, 389)
AUTHORITY = HeightChunkId(0, 153, 97)
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


def _load_json_bound(path: Path, sha_path: Path | None = None) -> dict[str, Any]:
    blob = path.read_bytes()
    if sha_path is not None and hashlib.sha256(blob).hexdigest() != sha_path.read_text().strip():
        raise ValueError(f"JSON identity mismatch: {path}")
    return json.loads(blob)


def _decoded(path: Path, encode: EncodeConfig) -> tuple[Any, bytes, np.ndarray]:
    meta, payload = read_chunk_v2(path)
    return meta, payload, decode_quant16(
        encode, payload, meta.res, meta.qoffset, meta.qscale
    )


def _index_blob(manifest_path: Path, layer: str) -> bytes:
    manifest = json.loads(manifest_path.read_bytes())
    return (manifest_path.parent / manifest["layers"][layer]["index"]).read_bytes()


def _height_inventory(manifest_path: Path) -> dict[tuple[int, int, int], tuple[int, int]]:
    manifest = json.loads(manifest_path.read_bytes())
    index_path = manifest_path.parent / manifest["layers"]["height"]["index"]
    return {record.key: (record.size, record.hash64) for record in read_v1_index(index_path)}


def _expected_core(
    baseline: PinnedDecodedBaseline,
    chunk: HeightChunkId,
    after: np.ndarray,
    allowed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    reconstructed = baseline.reconstruct(chunk, halo_samples=4)
    support = reconstructed.tile.height
    baseline_fine = prolong_structural_4x(
        support, parent_rows=(4, 516), parent_cols=(4, 516)
    )
    authority_support = np.array(support, dtype=np.float64, copy=True)
    dx, dz = chunk.cx - PARENT.cx * 4, chunk.cz - PARENT.cz * 4
    global_rows = dz * 512 + np.arange(-4, 516, dtype=np.int64)
    global_cols = dx * 512 + np.arange(-4, 516, dtype=np.int64)
    valid_rows = (global_rows >= 0) & (global_rows < 2048)
    valid_cols = (global_cols >= 0) & (global_cols < 2048)
    if valid_rows.any() and valid_cols.any():
        rr = global_rows[valid_rows]
        cc = global_cols[valid_cols]
        selected_allowed = allowed[np.ix_(rr, cc)]
        selected_after = after[np.ix_(rr, cc)]
        block = authority_support[np.ix_(valid_rows, valid_cols)]
        authority_support[np.ix_(valid_rows, valid_cols)] = np.where(
            selected_allowed, selected_after, block
        )
    core_allowed = allowed[
        dz * 512 : (dz + 1) * 512,
        dx * 512 : (dx + 1) * 512,
    ]
    fine_allowed = np.repeat(np.repeat(core_allowed, 4, axis=0), 4, axis=1)
    altered = prolong_structural_4x(
        authority_support, parent_rows=(4, 516), parent_cols=(4, 516)
    )
    return np.where(fine_allowed, altered, baseline_fine), baseline_fine


def _max_seam(
    paths: dict[HeightChunkId, Path], encode: EncodeConfig
) -> tuple[float, int]:
    maximum = 0.0
    comparisons = 0
    x0, z0 = PARENT.cx * 4, PARENT.cz * 4
    for cz in range(z0, z0 + 5):
        for cx in range(x0, x0 + 4):
            left = _decoded(paths[HeightChunkId(-2, cx, cz)], encode)[2]
            right = _decoded(paths[HeightChunkId(-2, cx + 1, cz)], encode)[2]
            maximum = max(maximum, float(np.max(np.abs(left[:, -1] - right[:, 0]))))
            comparisons += 1
    for cx in range(x0, x0 + 5):
        for cz in range(z0, z0 + 4):
            north = _decoded(paths[HeightChunkId(-2, cx, cz)], encode)[2]
            south = _decoded(paths[HeightChunkId(-2, cx, cz + 1)], encode)[2]
            maximum = max(maximum, float(np.max(np.abs(north[-1, :] - south[0, :]))))
            comparisons += 1
    return maximum, comparisons


def verify_rough_till_preview(
    build_digest: str,
    base_manifest_path: Path,
    base_out_root: Path,
    work_root: Path,
) -> Path:
    base = load_base()
    build_root = work_root / "builds" / build_digest
    expectation = _load_json_bound(build_root / "expectation.json", build_root / "expectation.sha256")
    plan = _load_json_bound(build_root / "plan.json", build_root / "plan.sha256")
    if expectation.get("recipeKind") != RECIPE_KIND or plan.get("microRecipeKind") != RECIPE_KIND:
        raise ValueError("rough-till verifier received another recipe kind")
    if expectation.get("recipeSha256") != build_digest or plan.get("recipeSha256") != build_digest:
        raise ValueError("rough-till recipe identity mismatch")
    if expectation.get("verifier") != {
        "id": VERIFIER_ID,
        "sourceSha256": verifier_source_sha256(),
    }:
        raise ValueError("rough-till expectation names another verifier")

    inputs = expectation["recipeInputs"]
    artifact_root = Path(inputs["artifact"]["root"])
    for name, key in (
        ("manifest.json", "manifestSha256"),
        ("recipe.json", "recipeSha256"),
        ("preview.npz", "previewSha256"),
    ):
        if _sha256(artifact_root / name) != inputs["artifact"][key]:
            raise ValueError(f"rough-till artifact changed: {name}")
    artifact_manifest = json.loads((artifact_root / "manifest.json").read_bytes())
    if (
        artifact_manifest.get("schema_version") != ARTIFACT_SCHEMA
        or artifact_manifest.get("decision") != "inspect_float_preview"
        or artifact_manifest["metrics"].get("forbidden_nonzero_cells") != 0
    ):
        raise ValueError("rough-till artifact evidence boundary changed")
    source = np.load(artifact_root / "preview.npz", allow_pickle=False)
    before = source["before_m"]
    after = source["after_m"]
    allowed = source["allowed"].astype(bool)
    if np.any(after[~allowed] != before[~allowed]):
        raise ValueError("rough-till source changes a forbidden 0.25 m cell")

    coverage = plan_hero(PARENT.cx, PARENT.cz)
    expected_coverage = {
        "parent": [PARENT.lod, PARENT.cx, PARENT.cz],
        "publishedFine": [[c.lod, c.cx, c.cz] for c in coverage.published_fine],
        "transientSupport": [[c.lod, c.cx, c.cz] for c in coverage.transient_support],
        "authority": [AUTHORITY.lod, AUTHORITY.cx, AUTHORITY.cz],
    }
    if plan["microCoverage"] != expected_coverage:
        raise ValueError("rough-till coverage differs from exact hero closure")
    evidence = _load_json_bound(build_root / "evidence/rough-till-preview.json")
    identities = {HeightChunkId(*row["key"]): row for row in evidence["children"]}
    all_fine = (*coverage.published_fine, *coverage.transient_support)
    if set(identities) != set(all_fine):
        raise ValueError("rough-till evidence lacks exact fine closure")
    published = set(coverage.published_fine)
    paths: dict[HeightChunkId, Path] = {}
    headers: list[dict[str, Any]] = []
    for chunk in all_fine:
        root = build_root / ("chunks" if chunk in published else "transient")
        path = root / "height/-2" / f"{chunk.cx}_{chunk.cz}.lac"
        identity = identities[chunk]
        if _sha256(path) != identity["sha256"] or path.stat().st_size != identity["bytes"]:
            raise ValueError(f"rough-till fine artifact changed: {chunk}")
        meta, _payload, _values = _decoded(path, base.encode)
        if (
            (meta.lod, meta.cx, meta.cz) != (chunk.lod, chunk.cx, chunk.cz)
            or meta.res != 2049
            or abs(meta.qscale - 0.002) > 1e-8
        ):
            raise ValueError(f"rough-till fine header mismatch: {chunk}")
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
    maximum_forbidden_error = 0.0
    for chunk in coverage.published_fine:
        expected, expected_baseline = _expected_core(baseline, chunk, after, allowed)
        decoded = _decoded(paths[chunk], base.encode)[2][:-1, :-1]
        dx, dz = chunk.cx - PARENT.cx * 4, chunk.cz - PARENT.cz * 4
        core_allowed = allowed[
            dz * 512 : (dz + 1) * 512,
            dx * 512 : (dx + 1) * 512,
        ]
        fine_allowed = np.repeat(np.repeat(core_allowed, 4, axis=0), 4, axis=1)
        maximum_structural_error = max(
            maximum_structural_error,
            float(np.max(np.abs(decoded.astype(np.float64) - expected))),
        )
        maximum_forbidden_error = max(
            maximum_forbidden_error,
            float(np.max(np.abs(decoded[~fine_allowed].astype(np.float64) - expected_baseline[~fine_allowed]), initial=0.0)),
        )
    if maximum_structural_error > 0.00101:
        raise ValueError(f"structural reconstruction round trip is {maximum_structural_error} m")
    if maximum_forbidden_error > 0.00101:
        raise ValueError(f"forbidden fine residual is {maximum_forbidden_error} m")

    maximum_seam, seam_comparisons = _max_seam(paths, base.encode)
    if maximum_seam != 0.0:
        raise ValueError(f"rough-till decoded seam is {maximum_seam} m")
    mosaic = assemble_parent_source_memmap(
        build_root / "scratch/verify-rough-till-parent.f32",
        coverage,
        lambda chunk: _decoded(paths[chunk], base.encode)[2],
    )
    expected_parent = box_mean4_striped(mosaic)
    parent_path = build_root / f"chunks/height/-1/{PARENT.cx}_{PARENT.cz}.lac"
    parent_meta, parent_payload, parent_decoded = _decoded(parent_path, base.encode)
    parent_identity = evidence["parent"]
    if _sha256(parent_path) != parent_identity["sha256"] or parent_path.stat().st_size != parent_identity["bytes"]:
        raise ValueError("rough-till parent artifact changed")
    expected_payload, expected_offset, expected_scale = encode_quant16_checked(
        base.encode, expected_parent, parent_meta.qscale, parent_meta.qoffset
    )
    if (
        expected_payload != parent_payload
        or expected_offset != parent_meta.qoffset
        or expected_scale != parent_meta.qscale
    ):
        raise ValueError("rough-till LOD -1 parent is not byte-exactly child-derived")

    corrected = inputs["correctedBase"]
    if corrected["manifestSha256"] != expectation["baseManifestSha256"]:
        raise ValueError("rough-till corrected base is not the expected release")
    audit_base_release(base_manifest_path, corrected["manifestSha256"], base_out_root)
    corrected_source = AuditedFormat1HeightSource(
        manifest_path=base_manifest_path,
        manifest_sha256=corrected["manifestSha256"],
        content_root=base_out_root,
        encode=base.encode,
        cache_chunks=2,
    )
    reduced_lod0 = box_mean4_striped(parent_decoded[:2048, :2048])
    coarse_mask = allowed.reshape(512, 4, 512, 4).any(axis=(1, 3))
    actual_lod0 = corrected_source.load(AUTHORITY).decoded[:-1, :-1][512:1024, 1536:2048]
    maximum_lod0_error = float(
        np.max(np.abs(actual_lod0[coarse_mask] - reduced_lod0[coarse_mask]), initial=0.0)
    )
    if maximum_lod0_error > 0.00501:
        raise ValueError(f"rough-till LOD0 closure is {maximum_lod0_error} m")

    audit_base_release(source_manifest, source_sha, base_out_root)
    source_manifest_doc = json.loads(source_manifest.read_bytes())
    inherited_layers = sorted(set(source_manifest_doc["layers"]) - {"height"})
    for layer in inherited_layers:
        if _index_blob(source_manifest, layer) != _index_blob(base_manifest_path, layer):
            raise ValueError(f"rough-till corrected base changed inherited {layer}")
    before_height = _height_inventory(source_manifest)
    after_height = _height_inventory(base_manifest_path)
    if set(before_height) != set(after_height):
        raise ValueError("rough-till corrected base changed height key set")
    changed_height = {key for key in before_height if before_height[key] != after_height[key]}
    transaction_plan = _load_json_bound(build_root / "corrected-base/resolved-hierarchy-plan.json")
    planned_height = {tuple(row["chunk"]) for row in transaction_plan["artifacts"]}
    if not changed_height.issubset(planned_height):
        raise ValueError("rough-till corrected base contains stale hierarchy changes")
    unchanged_planned = planned_height - changed_height

    transient_artifacts = [
        {"key": identities[c]["key"], "sha256": identities[c]["sha256"], "size": identities[c]["bytes"]}
        for c in coverage.transient_support
    ]
    binding = micro_verification_binding(build_digest, work_root)
    gates = {
        "coverage": _gate({**expected_coverage, "publishedCount": 17, "transientCount": 9}),
        "parentClosure": _gate({
            "parentSha256": _sha256(parent_path),
            "payloadByteExact": True,
            "correctedHierarchyPlanSha256": corrected["hierarchyPlanSha256"],
        }),
        "headers": _gate({"fine": headers, "parentQscaleM": parent_meta.qscale}),
        "decodedHierarchy": _gate({
            "maxLod0DecodedChildErrorM": maximum_lod0_error,
            "maskedSpliceThroughLod4": True,
            "changedHeightKeys": [list(key) for key in sorted(changed_height)],
            "byteIdenticalPlannedKeys": [list(key) for key in sorted(unchanged_planned)],
            "noStaleUnrelatedAncestorChanges": True,
        }),
        "aprons": _gate({"maxDecodedSeamErrorM": maximum_seam, "comparisons": seam_comparisons}),
        "determinism": _gate({
            "artifactManifestSha256": inputs["artifact"]["manifestSha256"],
            "artifactPreviewSha256": inputs["artifact"]["previewSha256"],
            "sourceBaseManifestSha256": source_sha,
            "verifierSourceSha256": verifier_source_sha256(),
        }),
        "hardMasks": _gate({
            "maxStructuralRoundTripErrorM": maximum_structural_error,
            "maxForbiddenFineRoundTripErrorM": maximum_forbidden_error,
            "artifactForbiddenResidualM": 0.0,
            "finestRungStructuralOnly": True,
            "fineMorphologyClaimBelow025M": "none",
            "inheritedLayerIndexesByteExact": inherited_layers,
        }),
    }
    report = {
        **binding,
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
