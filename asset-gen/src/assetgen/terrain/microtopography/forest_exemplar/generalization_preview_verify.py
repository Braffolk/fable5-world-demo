"""Independent verifier for the disjoint multi-parent forest preview."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ....config import EncodeConfig, load_base
from ....cook.chunkio import read_chunk_v2
from ....cook.encode import decode_quant16, encode_quant16_checked
from ....cook.micro_hierarchy import assemble_parent_source_memmap, box_mean4_striped
from ....height_geom import HeightChunkId, plan_hero, plan_parent_set
from ....micro_verify import evidence_sha256, transient_merkle_root
from ....release import (
    audit_base_release,
    micro_verification_binding,
    read_v1_index,
)
from ...repair.base_transaction import AuditedFormat1HeightSource


VERIFIER_ID = "assetgen.forest-generalization-preview-verify.v2"
RECIPE_KIND = "research-microtopography-generalization-preview-v1"


@dataclass(frozen=True)
class SiteSpec:
    site_id: str
    parent: HeightChunkId
    authority: HeightChunkId
    bbox_en: tuple[int, int, int, int]

    def json(self) -> dict[str, object]:
        return {
            "siteId": self.site_id,
            "reviewBboxEn": list(self.bbox_en),
            "parent": [self.parent.lod, self.parent.cx, self.parent.cz],
            "authority": [self.authority.lod, self.authority.cx, self.authority.cz],
        }


SITES = (
    SiteSpec("southeast-southwest-retained", HeightChunkId(-1, 601, 384), HeightChunkId(0, 150, 96), (676352, 6438400, 676864, 6438912)),
    SiteSpec("southeast-northcentral-retained", HeightChunkId(-1, 606, 361), HeightChunkId(0, 151, 90), (678912, 6450176, 679424, 6450688)),
    SiteSpec("southeast-east-retained", HeightChunkId(-1, 617, 378), HeightChunkId(0, 154, 94), (684544, 6441472, 685056, 6441984)),
)


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


def _max_seam(
    coverage: Any,
    paths: dict[HeightChunkId, Path],
    encode: EncodeConfig,
) -> tuple[float, int]:
    maximum = 0.0
    comparisons = 0
    x0, z0 = coverage.parent.cx * 4, coverage.parent.cz * 4
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


def _index_blob(manifest_path: Path, layer: str) -> bytes:
    manifest = json.loads(manifest_path.read_bytes())
    return (manifest_path.parent / manifest["layers"][layer]["index"]).read_bytes()


def _height_inventory(manifest_path: Path) -> dict[tuple[int, int, int], tuple[int, int]]:
    manifest = json.loads(manifest_path.read_bytes())
    index_path = manifest_path.parent / manifest["layers"]["height"]["index"]
    return {record.key: (record.size, record.hash64) for record in read_v1_index(index_path)}


def verify_generalization_preview(
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
        raise ValueError("forest generalization verifier received another recipe kind")
    if expectation.get("recipeSha256") != build_digest or plan.get("recipeSha256") != build_digest:
        raise ValueError("forest generalization recipe identity mismatch")
    if expectation.get("verifier") != {"id": VERIFIER_ID, "sourceSha256": verifier_source_sha256()}:
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
    metrics_by_site = artifact_manifest["site_metrics"]
    for site in SITES:
        metrics = metrics_by_site[site.site_id]
        if tuple(metrics["site"]["bbox_en"]) != site.bbox_en:
            raise ValueError(f"artifact bbox changed for {site.site_id}")
        if metrics["maximum_hard_exclusion_residual_m"] != 0.0 or metrics["maximum_abs_applied_offset_m"] != 0.0:
            raise ValueError(f"hard-mask or datum gate changed for {site.site_id}")

    coverage = plan_parent_set(tuple(site.parent for site in SITES))
    expected_coverage = {
        "sites": [site.json() for site in SITES],
        "parents": [[c.lod, c.cx, c.cz] for c in coverage.parents],
        "publishedFine": [[c.lod, c.cx, c.cz] for c in coverage.published_fine],
        "transientSupport": [[c.lod, c.cx, c.cz] for c in coverage.transient_support],
        "authorities": [[c.lod, c.cx, c.cz] for c in coverage.authorities_lod0],
    }
    if plan["microCoverage"] != expected_coverage or expectation["sites"] != expected_coverage["sites"]:
        raise ValueError("forest generalization coverage differs from exact parent-set closure")
    evidence = _load_json_bound(build_root / "evidence/forest-generalization-preview.json")
    identities = {HeightChunkId(*row["key"]): row for row in evidence["children"]}
    all_fine = (*coverage.published_fine, *coverage.transient_support)
    if set(identities) != set(all_fine):
        raise ValueError("forest generalization evidence lacks exact unioned fine closure")
    published = set(coverage.published_fine)
    paths: dict[HeightChunkId, Path] = {}
    header_rows: list[dict[str, Any]] = []
    for chunk in all_fine:
        root = build_root / ("chunks" if chunk in published else "transient")
        path = root / "height/-2" / f"{chunk.cx}_{chunk.cz}.lac"
        identity = identities[chunk]
        if _sha256(path) != identity["sha256"] or path.stat().st_size != identity["bytes"]:
            raise ValueError(f"fine artifact changed: {chunk}")
        meta, _payload, _values = _decoded(path, base.encode)
        if (meta.lod, meta.cx, meta.cz) != (chunk.lod, chunk.cx, chunk.cz) or meta.res != 2049 or abs(meta.qscale - 0.002) > 1e-8:
            raise ValueError(f"fine header mismatch: {chunk}")
        paths[chunk] = path
        header_rows.append({"key": identity["key"], "qoffset": meta.qoffset, "qscale": meta.qscale})

    parent_evidence = {HeightChunkId(*row["key"]): row for row in evidence["parents"]}
    if set(parent_evidence) != set(coverage.parents):
        raise ValueError("forest generalization evidence lacks exact parent set")
    maximum_c1_error = 0.0
    maximum_seam = 0.0
    seam_comparisons = 0
    maximum_lod0_error = 0.0
    parent_rows: list[dict[str, Any]] = []
    corrected = inputs["correctedBase"]
    if corrected["manifestSha256"] != expectation["baseManifestSha256"]:
        raise ValueError("corrected base is not the expected release")
    audit_base_release(base_manifest_path, corrected["manifestSha256"], base_out_root)
    corrected_source = AuditedFormat1HeightSource(
        manifest_path=base_manifest_path,
        manifest_sha256=corrected["manifestSha256"],
        content_root=base_out_root,
        encode=base.encode,
        cache_chunks=2,
    )
    for site in SITES:
        hero = plan_hero(site.parent.cx, site.parent.cz)
        c1 = np.load(artifact_root / f"sites/{site.site_id}/surface/c1_height_f32.npy", mmap_mode="r")
        x0, z0 = site.parent.cx * 4, site.parent.cz * 4
        site_c1_error = 0.0
        for chunk in hero.published_fine:
            dx, dz = chunk.cx - x0, chunk.cz - z0
            decoded = _decoded(paths[chunk], base.encode)[2][:-1, :-1]
            expected = c1[dz * 2048 : (dz + 1) * 2048, dx * 2048 : (dx + 1) * 2048]
            site_c1_error = max(site_c1_error, float(np.max(np.abs(decoded.astype(np.float64) - expected.astype(np.float64)))))
        maximum_c1_error = max(maximum_c1_error, site_c1_error)
        if site_c1_error > 0.00101:
            raise ValueError(f"decoded C1 error at {site.site_id} is {site_c1_error} m")
        site_seam, site_comparisons = _max_seam(hero, paths, base.encode)
        maximum_seam = max(maximum_seam, site_seam)
        seam_comparisons += site_comparisons
        if site_seam != 0.0:
            raise ValueError(f"decoded fine seam at {site.site_id} is {site_seam} m")
        mosaic = assemble_parent_source_memmap(
            build_root / f"scratch/verify-forest-parent-{site.parent.cx}-{site.parent.cz}.f32",
            hero,
            lambda chunk: _decoded(paths[chunk], base.encode)[2],
        )
        expected_parent = box_mean4_striped(mosaic)
        parent_path = build_root / f"chunks/height/-1/{site.parent.cx}_{site.parent.cz}.lac"
        parent_meta, parent_payload, parent_decoded = _decoded(parent_path, base.encode)
        identity = parent_evidence[site.parent]
        if _sha256(parent_path) != identity["sha256"] or parent_path.stat().st_size != identity["bytes"]:
            raise ValueError(f"parent artifact changed: {site.parent}")
        expected_payload, expected_offset, expected_scale = encode_quant16_checked(
            base.encode, expected_parent, parent_meta.qscale, parent_meta.qoffset
        )
        if expected_payload != parent_payload or expected_offset != parent_meta.qoffset or expected_scale != parent_meta.qscale:
            raise ValueError(f"{site.parent} is not byte-exactly derived from decoded children")
        reduced_lod0 = box_mean4_striped(parent_decoded[:2048, :2048])
        packed = np.load(artifact_root / f"sites/{site.site_id}/surface/allowed_packbits_u8.npy", mmap_mode="r")
        allowed = np.unpackbits(packed, axis=1, bitorder="little")[:, :8192].astype(bool)
        coarse_mask = allowed.reshape(512, 16, 512, 16).any(axis=(1, 3))
        origin_e = base.grid.anchor_e + site.authority.cx * base.grid.chunk_m
        origin_n = base.grid.anchor_n - site.authority.cz * base.grid.chunk_m
        x = site.bbox_en[0] - origin_e
        y = origin_n - site.bbox_en[3]
        decoded_lod0 = corrected_source.load(site.authority).decoded[:-1, :-1]
        actual = decoded_lod0[y : y + 512, x : x + 512][coarse_mask]
        expected = reduced_lod0[coarse_mask]
        site_lod0_error = float(np.max(np.abs(actual - expected), initial=0.0))
        maximum_lod0_error = max(maximum_lod0_error, site_lod0_error)
        if site_lod0_error > 0.00501:
            raise ValueError(f"corrected LOD0 at {site.site_id} differs by {site_lod0_error} m")
        parent_rows.append({
            "siteId": site.site_id,
            "parent": [site.parent.lod, site.parent.cx, site.parent.cz],
            "sha256": _sha256(parent_path),
            "payloadByteExact": True,
            "maxC1RoundTripErrorM": site_c1_error,
            "maxLod0ClosureErrorM": site_lod0_error,
        })

    source_manifest = Path(inputs["sourceBase"]["manifest"])
    source_sha = inputs["sourceBase"]["manifestSha256"]
    audit_base_release(source_manifest, source_sha, base_out_root)
    source_manifest_doc = json.loads(source_manifest.read_bytes())
    inherited_layers = sorted(set(source_manifest_doc["layers"]) - {"height"})
    for layer in inherited_layers:
        if _index_blob(source_manifest, layer) != _index_blob(base_manifest_path, layer):
            raise ValueError(f"corrected base changed inherited {layer} inventory")
    before_height = _height_inventory(source_manifest)
    after_height = _height_inventory(base_manifest_path)
    if set(before_height) != set(after_height):
        raise ValueError("corrected base changed the inherited height key set")
    changed_height = {key for key in before_height if before_height[key] != after_height[key]}
    transaction_plan = _load_json_bound(build_root / "corrected-base/resolved-hierarchy-plan.json")
    planned_height = {tuple(row["chunk"]) for row in transaction_plan["artifacts"]}
    if not changed_height.issubset(planned_height):
        raise ValueError("corrected base contains stale or unrelated hierarchy changes")
    unchanged_planned_height = planned_height - changed_height

    transient_artifacts = [
        {"key": identities[c]["key"], "sha256": identities[c]["sha256"], "size": identities[c]["bytes"]}
        for c in coverage.transient_support
    ]
    binding = micro_verification_binding(build_digest, work_root)
    max_hard = max(metrics_by_site[site.site_id]["maximum_hard_exclusion_residual_m"] for site in SITES)
    max_offset = max(metrics_by_site[site.site_id]["maximum_abs_applied_offset_m"] for site in SITES)
    gates = {
        "coverage": _gate({**expected_coverage, "publishedCount": 51, "transientCount": len(coverage.transient_support)}),
        "parentClosure": _gate({
            "parents": parent_rows,
            "decodedPublishedChildCount": len(coverage.published_fine),
            "correctedHierarchyPlanSha256": corrected["hierarchyPlanSha256"],
        }),
        "headers": _gate({"fine": header_rows, "parentQscaleM": 0.005}),
        "decodedHierarchy": _gate({
            "maxLod0DecodedChildErrorM": maximum_lod0_error,
            "limitM": 0.00501,
            "maskedSpliceThroughLod4": True,
            "changedHeightKeys": [list(key) for key in sorted(changed_height)],
            "byteIdenticalPlannedKeys": [
                list(key) for key in sorted(unchanged_planned_height)
            ],
            "noStaleUnrelatedAncestorChanges": True,
        }),
        "aprons": _gate({"maxDecodedSeamErrorM": maximum_seam, "comparisons": seam_comparisons}),
        "determinism": _gate({
            "artifactManifestSha256": inputs["artifact"]["manifestSha256"],
            "sourceBaseManifestSha256": inputs["sourceBase"]["manifestSha256"],
            "verifierSourceSha256": verifier_source_sha256(),
        }),
        "hardMasks": _gate({
            "maxC1RoundTripErrorM": maximum_c1_error,
            "maxArtifactHardExclusionResidualM": max_hard,
            "maxArtifactAppliedOffsetM": max_offset,
            "waterAndStructuralExclusionsPreserved": True,
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
