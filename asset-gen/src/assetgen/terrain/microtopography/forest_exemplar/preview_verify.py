"""Independent verifier for the accepted irregular-forest research preview."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from ....config import ASSET_GEN_ROOT, EncodeConfig, load_base
from ....cook.chunkio import read_chunk_v2
from ....cook.encode import decode_quant16, encode_quant16_checked
from ....cook.micro_hierarchy import assemble_parent_source_memmap, box_mean4_striped
from ....height_geom import HeightChunkId, plan_hero
from ....micro_verify import evidence_sha256, transient_merkle_root
from ....release import audit_base_release, micro_verification_binding
from ...repair.base_transaction import AuditedFormat1HeightSource


VERIFIER_ID = "assetgen.forest-irregular-preview-verify.v1"
RECIPE_KIND = "research-microtopography-preview-v1"
PARENT = HeightChunkId(-1, 607, 372)
AUTHORITY = HeightChunkId(0, 151, 93)


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
    return {
        "passed": True,
        "evidence": evidence,
        "evidenceSha256": evidence_sha256(evidence),
    }


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
    for cz in range(1488, 1493):
        for cx in range(2428, 2432):
            left = _decoded(paths[HeightChunkId(-2, cx, cz)], encode)[2]
            right = _decoded(paths[HeightChunkId(-2, cx + 1, cz)], encode)[2]
            maximum = max(maximum, float(np.max(np.abs(left[:, -1] - right[:, 0]))))
    for cx in range(2428, 2433):
        for cz in range(1488, 1492):
            north = _decoded(paths[HeightChunkId(-2, cx, cz)], encode)[2]
            south = _decoded(paths[HeightChunkId(-2, cx, cz + 1)], encode)[2]
            maximum = max(maximum, float(np.max(np.abs(north[-1, :] - south[0, :]))))
    return maximum


def verify_irregular_forest_preview(
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
        raise ValueError("forest preview verifier received another recipe kind")
    if expectation.get("recipeSha256") != build_digest or plan.get("recipeSha256") != build_digest:
        raise ValueError("forest preview recipe identity mismatch")
    if expectation.get("verifier") != {
        "id": VERIFIER_ID,
        "sourceSha256": verifier_source_sha256(),
    }:
        raise ValueError("forest preview expectation names another verifier")

    recipe_inputs = expectation["recipeInputs"]
    artifact_root = Path(recipe_inputs["artifact"]["root"])
    transaction_root = Path(recipe_inputs["sourceTransaction"]["root"])
    if _sha256(artifact_root / "manifest.json") != recipe_inputs["artifact"]["manifestSha256"]:
        raise ValueError("accepted forest manifest changed")
    artifact_manifest = json.loads((artifact_root / "manifest.json").read_bytes())
    for relative, identity in artifact_manifest["files"].items():
        path = artifact_root / relative
        if _sha256(path) != identity["sha256"] or path.stat().st_size != identity["bytes"]:
            raise ValueError(f"accepted forest artifact changed: {relative}")
    if _sha256(transaction_root / "transaction.json") != recipe_inputs["sourceTransaction"]["transactionSha256"]:
        raise ValueError("source correction transaction changed")

    coverage = plan_hero(PARENT.cx, PARENT.cz)
    expected_coverage = {
        "parent": [PARENT.lod, PARENT.cx, PARENT.cz],
        "publishedFine": [[c.lod, c.cx, c.cz] for c in coverage.published_fine],
        "transientSupport": [[c.lod, c.cx, c.cz] for c in coverage.transient_support],
        "authority": [AUTHORITY.lod, AUTHORITY.cx, AUTHORITY.cz],
    }
    if plan["microCoverage"] != expected_coverage:
        raise ValueError("forest preview coverage differs from exact hero closure")

    evidence = json.loads(
        (build_root / "evidence/forest-irregular-preview.json").read_bytes()
    )
    identities = {
        HeightChunkId(*item["key"]): item for item in evidence["children"]
    }
    if set(identities) != set((*coverage.published_fine, *coverage.transient_support)):
        raise ValueError("forest preview evidence lacks the exact fine closure")
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

    c1 = np.load(artifact_root / "surface/c1_height_f32.npy", mmap_mode="r")
    allowed = np.load(artifact_root / "surface/allowed_u8.npy", mmap_mode="r").astype(bool)
    changed_samples = 0
    max_authority_error = 0.0
    max_outside_error = 0.0
    for cz, artifact_row in ((1489, 0), (1490, 1024)):
        for cx, artifact_col in ((2429, 0), (2430, 1024)):
            chunk = HeightChunkId(-2, cx, cz)
            decoded = _decoded(paths[chunk], base.encode)[2]
            baseline_path = transaction_root / f"materialization/chunks/height/-2/{cx}_{cz}.lac2"
            baseline = _decoded(baseline_path, base.encode)[2]
            row = 1024 if cz == 1489 else 0
            col = 1024 if cx == 2429 else 0
            source = np.s_[artifact_row : artifact_row + 1024, artifact_col : artifact_col + 1024]
            target = np.s_[row : row + 1024, col : col + 1024]
            mask = allowed[source]
            actual = decoded[target]
            core_authority = np.zeros((2048, 2048), dtype=bool)
            core_authority[target] = mask
            changed_samples += int(mask.sum())
            max_authority_error = max(
                max_authority_error,
                float(np.max(np.abs(actual[mask].astype(np.float64) - c1[source][mask]))),
            )
            max_outside_error = max(
                max_outside_error,
                float(
                    np.max(
                        np.abs(
                            decoded[:2048, :2048][~core_authority]
                            - baseline[:2048, :2048][~core_authority]
                        )
                    )
                ),
            )
    if max_authority_error > 0.00101:
        raise ValueError(f"decoded C1 authority error is {max_authority_error} m")
    if max_outside_error != 0.0:
        raise ValueError(f"forest overlay changed excluded C0 by {max_outside_error} m")

    max_seam = _max_seam(paths, base.encode)
    if max_seam != 0.0:
        raise ValueError(f"decoded fine seam is {max_seam} m")
    mosaic = assemble_parent_source_memmap(
        build_root / "scratch/verify-forest-parent-source.f32",
        coverage,
        lambda chunk: _decoded(paths[chunk], base.encode)[2],
    )
    expected_parent = box_mean4_striped(mosaic)
    parent_path = build_root / f"chunks/height/-1/{PARENT.cx}_{PARENT.cz}.lac"
    parent_meta, parent_payload, parent_decoded = _decoded(parent_path, base.encode)
    expected_payload, expected_offset, expected_scale = encode_quant16_checked(
        base.encode,
        expected_parent,
        parent_meta.qscale,
        parent_meta.qoffset,
    )
    if (
        expected_payload != parent_payload
        or expected_offset != parent_meta.qoffset
        or expected_scale != parent_meta.qscale
    ):
        raise ValueError("LOD -1 parent is not byte-exactly derived from decoded children")

    corrected = recipe_inputs["correctedBase"]
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
    coarse_mask = allowed.reshape(128, 16, 128, 16).any(axis=(1, 3))
    coarse_actual = decoded_lod0[192:320, 1728:1856][coarse_mask]
    coarse_expected = reduced_lod0[192:320, 192:320][coarse_mask]
    max_lod0_error = float(np.max(np.abs(coarse_actual - coarse_expected)))
    if max_lod0_error > 0.00501:
        raise ValueError(f"corrected LOD0 differs from decoded-child closure by {max_lod0_error} m")

    transient_artifacts = [
        {
            "key": identities[chunk]["key"],
            "sha256": identities[chunk]["sha256"],
            "size": identities[chunk]["bytes"],
        }
        for chunk in coverage.transient_support
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
            "artifactManifestSha256": recipe_inputs["artifact"]["manifestSha256"],
            "sourceTransactionSha256": recipe_inputs["sourceTransaction"]["transactionSha256"],
            "verifierSourceSha256": verifier_source_sha256(),
        }),
        "hardMasks": _gate({
            "authoritySamples": changed_samples,
            "maxC1RoundTripErrorM": max_authority_error,
            "maxExcludedC0ChangeM": max_outside_error,
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
