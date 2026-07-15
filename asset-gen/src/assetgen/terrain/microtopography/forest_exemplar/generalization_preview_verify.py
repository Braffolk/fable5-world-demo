"""Independent verifier for the multi-site forest master preview."""
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
from ....release import audit_base_release, micro_verification_binding
from ...repair.base_transaction import AuditedFormat1HeightSource


VERIFIER_ID = "assetgen.forest-generalization-preview-verify.v1"
RECIPE_KIND = "research-microtopography-generalization-preview-v1"
SITE_ID = "southeast-northcentral-retained"
PARENT = HeightChunkId(-1, 606, 361)
AUTHORITY = HeightChunkId(0, 151, 90)
REVIEW_BBOX = (678912, 6450176, 679424, 6450688)


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
    values = decode_quant16(encode, payload, meta.res, meta.qoffset, meta.qscale)
    return meta, payload, values


def _max_seam(paths: dict[HeightChunkId, Path], encode: EncodeConfig) -> float:
    maximum = 0.0
    x0, z0 = PARENT.cx * 4, PARENT.cz * 4
    for cz in range(z0, z0 + 5):
        for cx in range(x0, x0 + 4):
            left = _decoded(paths[HeightChunkId(-2, cx, cz)], encode)[2]
            right = _decoded(paths[HeightChunkId(-2, cx + 1, cz)], encode)[2]
            maximum = max(maximum, float(np.max(np.abs(left[:, -1] - right[:, 0]))))
    for cx in range(x0, x0 + 5):
        for cz in range(z0, z0 + 4):
            north = _decoded(paths[HeightChunkId(-2, cx, cz)], encode)[2]
            south = _decoded(paths[HeightChunkId(-2, cx, cz + 1)], encode)[2]
            maximum = max(maximum, float(np.max(np.abs(north[-1, :] - south[0, :]))))
    return maximum


def verify_generalization_preview(
    build_digest: str,
    base_manifest_path: Path,
    base_out_root: Path,
    work_root: Path,
) -> Path:
    base = load_base()
    build_root = work_root / "builds" / build_digest
    expectation = _load_json_bound(
        build_root / "expectation.json", build_root / "expectation.sha256"
    )
    plan = _load_json_bound(build_root / "plan.json", build_root / "plan.sha256")
    if expectation.get("recipeKind") != RECIPE_KIND or plan.get("microRecipeKind") != RECIPE_KIND:
        raise ValueError("forest generalization verifier received another recipe kind")
    if expectation.get("recipeSha256") != build_digest or plan.get("recipeSha256") != build_digest:
        raise ValueError("forest generalization recipe identity mismatch")
    if expectation.get("verifier") != {
        "id": VERIFIER_ID,
        "sourceSha256": verifier_source_sha256(),
    }:
        raise ValueError("forest generalization expectation names another verifier")

    inputs = expectation["recipeInputs"]
    artifact_root = Path(inputs["artifact"]["root"])
    if _sha256(artifact_root / "manifest.json") != inputs["artifact"]["manifestSha256"]:
        raise ValueError("accepted forest generalization manifest changed")
    artifact_manifest = json.loads((artifact_root / "manifest.json").read_bytes())
    if artifact_manifest.get("status") != "research_candidate_pass":
        raise ValueError("forest generalization artifact no longer records a pass")
    for relative, identity in artifact_manifest["files"].items():
        path = artifact_root / relative
        if _sha256(path) != identity["sha256"] or path.stat().st_size != identity["bytes"]:
            raise ValueError(f"accepted forest generalization artifact changed: {relative}")
    metrics = artifact_manifest["site_metrics"][SITE_ID]
    if (
        metrics["maximum_hard_exclusion_residual_m"] != 0.0
        or metrics["maximum_abs_applied_offset_m"] != 0.0
    ):
        raise ValueError("accepted forest generalization hard-mask or datum gate changed")

    coverage = plan_hero(PARENT.cx, PARENT.cz)
    expected_coverage = {
        "parent": [PARENT.lod, PARENT.cx, PARENT.cz],
        "publishedFine": [[c.lod, c.cx, c.cz] for c in coverage.published_fine],
        "transientSupport": [[c.lod, c.cx, c.cz] for c in coverage.transient_support],
        "authority": [AUTHORITY.lod, AUTHORITY.cx, AUTHORITY.cz],
    }
    if plan["microCoverage"] != expected_coverage:
        raise ValueError("forest generalization coverage differs from exact parent closure")
    evidence = _load_json_bound(build_root / "evidence/forest-generalization-preview.json")
    identities = {HeightChunkId(*row["key"]): row for row in evidence["children"]}
    if set(identities) != set((*coverage.published_fine, *coverage.transient_support)):
        raise ValueError("forest generalization evidence lacks exact fine closure")
    paths: dict[HeightChunkId, Path] = {}
    header_rows: list[dict[str, Any]] = []
    for chunk in (*coverage.published_fine, *coverage.transient_support):
        root = build_root / ("chunks" if chunk in coverage.published_fine else "transient")
        path = root / "height/-2" / f"{chunk.cx}_{chunk.cz}.lac"
        identity = identities[chunk]
        if _sha256(path) != identity["sha256"] or path.stat().st_size != identity["bytes"]:
            raise ValueError(f"fine artifact changed: {chunk}")
        meta, _payload, _values = _decoded(path, base.encode)
        if (
            (meta.lod, meta.cx, meta.cz) != (chunk.lod, chunk.cx, chunk.cz)
            or meta.res != 2049
            or abs(meta.qscale - 0.002) > 1e-8
        ):
            raise ValueError(f"fine header mismatch: {chunk}")
        paths[chunk] = path
        header_rows.append({"key": identity["key"], "qoffset": meta.qoffset, "qscale": meta.qscale})

    c1 = np.load(
        artifact_root / f"sites/{SITE_ID}/surface/c1_height_f32.npy", mmap_mode="r"
    )
    maximum_c1_error = 0.0
    x0, z0 = PARENT.cx * 4, PARENT.cz * 4
    for chunk in coverage.published_fine:
        dx, dz = chunk.cx - x0, chunk.cz - z0
        decoded = _decoded(paths[chunk], base.encode)[2][:-1, :-1]
        expected = c1[
            dz * 2048 : (dz + 1) * 2048,
            dx * 2048 : (dx + 1) * 2048,
        ]
        maximum_c1_error = max(
            maximum_c1_error,
            float(np.max(np.abs(decoded.astype(np.float64) - expected.astype(np.float64)))),
        )
    if maximum_c1_error > 0.00101:
        raise ValueError(f"decoded C1 master error is {maximum_c1_error} m")

    max_seam = _max_seam(paths, base.encode)
    if max_seam != 0.0:
        raise ValueError(f"decoded fine seam is {max_seam} m")
    mosaic = assemble_parent_source_memmap(
        build_root / "scratch/verify-forest-generalization-parent-source.f32",
        coverage,
        lambda chunk: _decoded(paths[chunk], base.encode)[2],
    )
    expected_parent = box_mean4_striped(mosaic)
    parent_path = build_root / f"chunks/height/-1/{PARENT.cx}_{PARENT.cz}.lac"
    parent_meta, parent_payload, parent_decoded = _decoded(parent_path, base.encode)
    expected_payload, expected_offset, expected_scale = encode_quant16_checked(
        base.encode, expected_parent, parent_meta.qscale, parent_meta.qoffset
    )
    if (
        expected_payload != parent_payload
        or expected_offset != parent_meta.qoffset
        or expected_scale != parent_meta.qscale
    ):
        raise ValueError("LOD -1 parent is not byte-exactly derived from decoded children")

    corrected = inputs["correctedBase"]
    if corrected["manifestSha256"] != expectation["baseManifestSha256"]:
        raise ValueError("corrected base is not the expected release")
    audit_base_release(base_manifest_path, corrected["manifestSha256"], base_out_root)
    source = AuditedFormat1HeightSource(
        manifest_path=base_manifest_path,
        manifest_sha256=corrected["manifestSha256"],
        content_root=base_out_root,
        encode=base.encode,
        cache_chunks=2,
    )
    decoded_lod0 = source.load(AUTHORITY).decoded[:-1, :-1]
    reduced_lod0 = box_mean4_striped(parent_decoded[:2048, :2048])
    packed = np.load(
        artifact_root / f"sites/{SITE_ID}/surface/allowed_packbits_u8.npy", mmap_mode="r"
    )
    allowed = np.unpackbits(packed, axis=1, bitorder="little")[:, :8192].astype(bool)
    coarse_mask = allowed.reshape(512, 16, 512, 16).any(axis=(1, 3))
    origin_e = base.grid.anchor_e + AUTHORITY.cx * base.grid.chunk_m
    origin_n = base.grid.anchor_n - AUTHORITY.cz * base.grid.chunk_m
    x = REVIEW_BBOX[0] - origin_e
    y = origin_n - REVIEW_BBOX[3]
    actual = decoded_lod0[y : y + 512, x : x + 512][coarse_mask]
    expected = reduced_lod0[coarse_mask]
    max_lod0_error = float(np.max(np.abs(actual - expected), initial=0.0))
    if max_lod0_error > 0.00501:
        raise ValueError(f"corrected LOD0 differs from decoded-child closure by {max_lod0_error} m")

    transient_artifacts = [
        {"key": identities[c]["key"], "sha256": identities[c]["sha256"], "size": identities[c]["bytes"]}
        for c in coverage.transient_support
    ]
    binding = micro_verification_binding(build_digest, work_root)
    gates = {
        "coverage": _gate({**expected_coverage, "publishedCount": 17, "transientCount": 9}),
        "parentClosure": _gate({
            "parentSha256": _sha256(parent_path),
            "decodedChildCount": 25,
            "payloadByteExact": True,
            "correctedHierarchyPlanSha256": corrected["hierarchyPlanSha256"],
        }),
        "headers": _gate({"fine": header_rows, "parentQscale": parent_meta.qscale}),
        "decodedHierarchy": _gate({
            "maxLod0DecodedChildErrorM": max_lod0_error,
            "limitM": 0.00501,
            "maskedSpliceThroughLod4": True,
        }),
        "aprons": _gate({"maxDecodedSeamErrorM": max_seam, "comparisons": 40}),
        "determinism": _gate({
            "artifactManifestSha256": inputs["artifact"]["manifestSha256"],
            "sourceBaseManifestSha256": inputs["sourceBase"]["manifestSha256"],
            "verifierSourceSha256": verifier_source_sha256(),
        }),
        "hardMasks": _gate({
            "maxC1RoundTripErrorM": maximum_c1_error,
            "maxArtifactHardExclusionResidualM": metrics["maximum_hard_exclusion_residual_m"],
            "maxArtifactAppliedOffsetM": metrics["maximum_abs_applied_offset_m"],
            "waterAndStructuralExclusionsPreserved": True,
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
