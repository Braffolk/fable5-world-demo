"""Independent verifier for the complete packed forest LOD0 tranche."""

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
from ....process.micro_masks import rasterize_micro_morphology_mask
from ....release import (
    audit_base_release,
    micro_verification_binding,
    read_v1_index,
)
from ...repair.base_transaction import AuditedFormat1HeightSource


VERIFIER_ID = "assetgen.forest-lod0-tranche-preview-verify.v1"
RECIPE_KIND = "research-forest-lod0-tranche-preview-v1"
RECIPE_ID = "laas.micro.forest-lod0-tranche-preview.recipe.v1"
ARTIFACT_SCHEMA = "forest-mesic-mineral-lod0-tranche-artifact/1"
MASTER_SHAPE = (32768, 32768)
MASTER_BBOX = (684032, 6440960, 686080, 6443008)
MASTER_FINE_ORIGIN = (2464, 1504)
AUTHORITY = HeightChunkId(0, 154, 94)
TEXEL_M = 0.0625
TRANCHE_QOFFSET_M = 30.0


@dataclass(frozen=True)
class SiteSpec:
    site_id: str
    parent: HeightChunkId
    bbox_en: tuple[int, int, int, int]

    def json(self) -> dict[str, object]:
        return {
            "siteId": self.site_id,
            "reviewBboxEn": list(self.bbox_en),
            "parent": [self.parent.lod, self.parent.cx, self.parent.cz],
            "authority": [AUTHORITY.lod, AUTHORITY.cx, AUTHORITY.cz],
        }


SITES = tuple(
    SiteSpec(
        f"forest-lod0-154-94-r{row}-c{col}",
        HeightChunkId(-1, 616 + col, 376 + row),
        (
            684032 + col * 512,
            6443008 - (row + 1) * 512,
            684032 + (col + 1) * 512,
            6443008 - row * 512,
        ),
    )
    for row in range(4)
    for col in range(4)
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
    return (
        meta,
        payload,
        decode_quant16(encode, payload, meta.res, meta.qoffset, meta.qscale),
    )


def _index_blob(manifest_path: Path, layer: str) -> bytes:
    manifest = json.loads(manifest_path.read_bytes())
    return (manifest_path.parent / manifest["layers"][layer]["index"]).read_bytes()


def _height_inventory(manifest_path: Path) -> dict[tuple[int, int, int], tuple[int, int]]:
    manifest = json.loads(manifest_path.read_bytes())
    index_path = manifest_path.parent / manifest["layers"]["height"]["index"]
    return {record.key: (record.size, record.hash64) for record in read_v1_index(index_path)}


def _verify_masks(
    build_root: Path, inputs: dict[str, Any]
) -> tuple[dict[HeightChunkId, np.ndarray], list[dict[str, Any]]]:
    allowed_by_parent: dict[HeightChunkId, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for site in SITES:
        identity = inputs["masks"][site.site_id]
        path = build_root / identity["path"]
        if _sha256(path) != identity["sha256"]:
            raise ValueError(f"staged mask changed at {site.site_id}")
        packed = np.load(path, mmap_mode="r")
        if packed.shape != (8192, 1024) or packed.dtype != np.uint8:
            raise ValueError(f"invalid staged mask at {site.site_id}")
        e_min, _n_min, _e_max, n_max = site.bbox_en
        for tile_row in range(4):
            for tile_col in range(4):
                tile_e = e_min + tile_col * 128
                tile_n = n_max - tile_row * 128
                east = tile_e + (np.arange(2048) + 0.5) * TEXEL_M
                north = tile_n - (np.arange(2048) + 0.5) * TEXEL_M
                expected = np.packbits(
                    rasterize_micro_morphology_mask(east, north).allowed,
                    axis=1,
                    bitorder="little",
                )
                actual = packed[
                    tile_row * 2048 : (tile_row + 1) * 2048,
                    tile_col * 256 : (tile_col + 1) * 256,
                ]
                if not np.array_equal(actual, expected):
                    raise ValueError(f"staged mask differs from source at {site.site_id}")
        allowed = np.unpackbits(packed, axis=1, bitorder="little")[:, :8192].astype(bool)
        allowed_by_parent[site.parent] = allowed
        rows.append(
            {
                "siteId": site.site_id,
                "sha256": identity["sha256"],
                "allowedCells": int(np.count_nonzero(allowed)),
            }
        )
    return allowed_by_parent, rows


def verify_lod0_tranche_preview(
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
        raise ValueError("forest tranche verifier received another recipe kind")
    if expectation.get("recipeSha256") != build_digest or plan.get("recipeSha256") != build_digest:
        raise ValueError("forest tranche recipe identity mismatch")
    if expectation.get("verifier") != {
        "id": VERIFIER_ID,
        "sourceSha256": verifier_source_sha256(),
    }:
        raise ValueError("forest tranche expectation names another verifier")

    inputs = expectation["recipeInputs"]
    if inputs.get("id") != RECIPE_ID:
        raise ValueError("forest tranche recipe ID changed")
    artifact_root = Path(inputs["artifact"]["root"])
    if _sha256(artifact_root / "manifest.json") != inputs["artifact"]["manifestSha256"]:
        raise ValueError("forest tranche manifest changed")
    artifact = json.loads((artifact_root / "manifest.json").read_bytes())
    metrics = artifact.get("metrics", {})
    if (
        artifact.get("schema") != ARTIFACT_SCHEMA
        or artifact.get("build_id") != artifact_root.name
        or artifact.get("status") != "inspect_float_preview"
        or artifact.get("failures") != []
        or tuple(metrics.get("bbox_en", ())) != MASTER_BBOX
        or tuple(metrics.get("shape", ())) != MASTER_SHAPE
        or metrics.get("maximum_hard_exclusion_residual_m") != 0.0
        or float(metrics.get("maximum_one_metre_mean_error_m", 1.0)) > 1e-12
    ):
        raise ValueError("forest tranche float gates changed")
    for relative, identity in artifact["files"].items():
        path = artifact_root / relative
        if _sha256(path) != identity["sha256"] or path.stat().st_size != identity["bytes"]:
            raise ValueError(f"forest tranche artifact changed: {relative}")
    master = np.load(artifact_root / "surface/c1_height_f32.npy", mmap_mode="r")

    coverage = plan_parent_set(tuple(site.parent for site in SITES))
    expected_coverage = {
        "sites": [site.json() for site in SITES],
        "parents": [[c.lod, c.cx, c.cz] for c in coverage.parents],
        "publishedFine": [[c.lod, c.cx, c.cz] for c in coverage.published_fine],
        "transientSupport": [[c.lod, c.cx, c.cz] for c in coverage.transient_support],
        "authorities": [[c.lod, c.cx, c.cz] for c in coverage.authorities_lod0],
    }
    if (
        plan["microCoverage"] != expected_coverage
        or expectation["sites"] != expected_coverage["sites"]
        or expectation["expectedPublished"]
        != [["height", c.lod, c.cx, c.cz] for c in (*coverage.published_fine, *coverage.parents)]
    ):
        raise ValueError("forest tranche coverage differs from exact parent-set closure")
    evidence = _load_json_bound(build_root / "evidence/forest-lod0-tranche-preview.json")
    identities = {HeightChunkId(*row["key"]): row for row in evidence["children"]}
    all_fine = (*coverage.published_fine, *coverage.transient_support)
    if set(identities) != set(all_fine):
        raise ValueError("forest tranche evidence lacks exact fine closure")
    published = set(coverage.published_fine)
    paths: dict[HeightChunkId, Path] = {}
    header_rows: list[dict[str, Any]] = []
    maximum_c1_error = 0.0
    for chunk in all_fine:
        root = build_root / ("chunks" if chunk in published else "transient")
        path = root / "height/-2" / f"{chunk.cx}_{chunk.cz}.lac"
        identity = identities[chunk]
        if _sha256(path) != identity["sha256"] or path.stat().st_size != identity["bytes"]:
            raise ValueError(f"fine artifact changed: {chunk}")
        meta, _payload, decoded = _decoded(path, base.encode)
        if (
            (meta.lod, meta.cx, meta.cz) != (chunk.lod, chunk.cx, chunk.cz)
            or meta.res != 2049
            or abs(meta.qscale - 0.002) > 1e-8
        ):
            raise ValueError(f"fine header mismatch: {chunk}")
        if chunk in published:
            dx = chunk.cx - MASTER_FINE_ORIGIN[0]
            dz = chunk.cz - MASTER_FINE_ORIGIN[1]
            expected = master[
                dz * 2048 : (dz + 1) * 2048,
                dx * 2048 : (dx + 1) * 2048,
            ]
            maximum_c1_error = max(
                maximum_c1_error,
                float(np.max(np.abs(decoded[:-1, :-1].astype(np.float64) - expected.astype(np.float64)))),
            )
        paths[chunk] = path
        header_rows.append({"key": identity["key"], "qoffset": meta.qoffset, "qscale": meta.qscale})
    if maximum_c1_error > 0.00101:
        raise ValueError(f"decoded tranche C1 error is {maximum_c1_error} m")

    maximum_seam = 0.0
    seam_comparisons = 0
    fine_set = set(all_fine)
    for chunk in all_fine:
        current = _decoded(paths[chunk], base.encode)[2]
        east = HeightChunkId(-2, chunk.cx + 1, chunk.cz)
        south = HeightChunkId(-2, chunk.cx, chunk.cz + 1)
        if east in fine_set:
            neighbor = _decoded(paths[east], base.encode)[2]
            maximum_seam = max(maximum_seam, float(np.max(np.abs(current[:, -1] - neighbor[:, 0]))))
            seam_comparisons += 1
        if south in fine_set:
            neighbor = _decoded(paths[south], base.encode)[2]
            maximum_seam = max(maximum_seam, float(np.max(np.abs(current[-1, :] - neighbor[0, :]))))
            seam_comparisons += 1
    if maximum_seam != 0.0:
        raise ValueError(f"decoded fine seam is {maximum_seam} m")

    allowed_by_parent, mask_rows = _verify_masks(build_root, inputs)
    parent_evidence = {HeightChunkId(*row["key"]): row for row in evidence["parents"]}
    if set(parent_evidence) != set(coverage.parents):
        raise ValueError("forest tranche evidence lacks exact parent set")
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
    decoded_lod0 = corrected_source.load(AUTHORITY).decoded[:-1, :-1]
    parent_decoded: dict[HeightChunkId, np.ndarray] = {}
    parent_rows: list[dict[str, Any]] = []
    maximum_lod0_error = 0.0
    for site in SITES:
        hero = plan_hero(site.parent.cx, site.parent.cz)
        mosaic = assemble_parent_source_memmap(
            build_root / f"scratch/verify-forest-tranche-parent-{site.parent.cx}-{site.parent.cz}.f32",
            hero,
            lambda chunk: _decoded(paths[chunk], base.encode)[2],
        )
        expected_parent = box_mean4_striped(mosaic)
        parent_path = build_root / f"chunks/height/-1/{site.parent.cx}_{site.parent.cz}.lac"
        meta, payload, decoded_parent = _decoded(parent_path, base.encode)
        identity = parent_evidence[site.parent]
        if _sha256(parent_path) != identity["sha256"] or parent_path.stat().st_size != identity["bytes"]:
            raise ValueError(f"parent artifact changed: {site.parent}")
        expected_payload, expected_offset, expected_scale = encode_quant16_checked(
            base.encode, expected_parent, meta.qscale, meta.qoffset
        )
        if (
            payload != expected_payload
            or meta.qoffset != expected_offset
            or meta.qscale != expected_scale
            or meta.qoffset != TRANCHE_QOFFSET_M
            or abs(meta.qscale - 0.005) > 1e-8
        ):
            raise ValueError(f"parent is not byte-exactly derived: {site.parent}")
        parent_decoded[site.parent] = decoded_parent
        reduced_lod0 = box_mean4_striped(decoded_parent[:2048, :2048])
        coarse_mask = allowed_by_parent[site.parent].reshape(512, 16, 512, 16).any(axis=(1, 3))
        row = site.parent.cz - 376
        col = site.parent.cx - 616
        actual = decoded_lod0[
            row * 512 : (row + 1) * 512,
            col * 512 : (col + 1) * 512,
        ][coarse_mask]
        expected = reduced_lod0[coarse_mask]
        error = float(np.max(np.abs(actual - expected), initial=0.0))
        maximum_lod0_error = max(maximum_lod0_error, error)
        if error > 0.00501:
            raise ValueError(f"corrected LOD0 differs at {site.site_id} by {error} m")
        parent_rows.append(
            {
                "siteId": site.site_id,
                "parent": [site.parent.lod, site.parent.cx, site.parent.cz],
                "sha256": _sha256(parent_path),
                "payloadByteExact": True,
                "maxLod0ClosureErrorM": error,
            }
        )

    parent_shared_seam = 0.0
    shared_parent_comparisons = 0
    for parent, values in parent_decoded.items():
        for neighbor, axis in (
            (HeightChunkId(-1, parent.cx + 1, parent.cz), "east"),
            (HeightChunkId(-1, parent.cx, parent.cz + 1), "south"),
        ):
            if neighbor not in parent_decoded:
                continue
            other = parent_decoded[neighbor]
            error = (
                float(np.max(np.abs(values[:, -1] - other[:, 0])))
                if axis == "east"
                else float(np.max(np.abs(values[-1, :] - other[0, :])))
            )
            parent_shared_seam = max(parent_shared_seam, error)
            shared_parent_comparisons += 1
    if shared_parent_comparisons != 24 or parent_shared_seam > 1e-5:
        raise ValueError(
            f"parent seam closure failed: {shared_parent_comparisons}, {parent_shared_seam} m"
        )

    source_manifest = Path(inputs["sourceBase"]["manifest"])
    source_sha = inputs["sourceBase"]["manifestSha256"]
    audit_base_release(source_manifest, source_sha, base_out_root)
    source_doc = json.loads(source_manifest.read_bytes())
    inherited_layers = sorted(set(source_doc["layers"]) - {"height"})
    for layer in inherited_layers:
        if _index_blob(source_manifest, layer) != _index_blob(base_manifest_path, layer):
            raise ValueError(f"corrected base changed inherited {layer} inventory")
    before_height = _height_inventory(source_manifest)
    after_height = _height_inventory(base_manifest_path)
    if set(before_height) != set(after_height):
        raise ValueError("corrected base changed inherited height keys")
    changed_height = {key for key in before_height if before_height[key] != after_height[key]}
    transaction_plan = _load_json_bound(build_root / "corrected-base/resolved-hierarchy-plan.json")
    planned_height = {tuple(row["chunk"]) for row in transaction_plan["artifacts"]}
    if not changed_height.issubset(planned_height):
        raise ValueError("corrected base contains unrelated hierarchy changes")
    unchanged_planned_height = planned_height - changed_height

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
        "coverage": _gate(
            {
                **expected_coverage,
                "publishedCount": len(coverage.published_fine) + len(coverage.parents),
                "transientCount": len(coverage.transient_support),
            }
        ),
        "parentClosure": _gate(
            {
                "parents": parent_rows,
                "decodedPublishedChildCount": len(coverage.published_fine),
                "correctedHierarchyPlanSha256": corrected["hierarchyPlanSha256"],
                "sharedParentSeamErrorM": parent_shared_seam,
                "sharedParentBoundaryComparisons": shared_parent_comparisons,
            }
        ),
        "headers": _gate(
            {
                "fine": header_rows,
                "parentQoffsetM": TRANCHE_QOFFSET_M,
                "parentQscaleM": 0.005,
            }
        ),
        "decodedHierarchy": _gate(
            {
                "maxLod0DecodedChildErrorM": maximum_lod0_error,
                "limitM": 0.00501,
                "maskedSpliceThroughLod4": True,
                "changedHeightKeys": [list(key) for key in sorted(changed_height)],
                "byteIdenticalPlannedKeys": [
                    list(key) for key in sorted(unchanged_planned_height)
                ],
                "noStaleUnrelatedAncestorChanges": True,
            }
        ),
        "aprons": _gate(
            {
                "maxDecodedSeamErrorM": maximum_seam,
                "comparisons": seam_comparisons,
                "sharedParentSeamErrorM": parent_shared_seam,
            }
        ),
        "determinism": _gate(
            {
                "artifactManifestSha256": inputs["artifact"]["manifestSha256"],
                "sourceBaseManifestSha256": source_sha,
                "verifierSourceSha256": verifier_source_sha256(),
            }
        ),
        "hardMasks": _gate(
            {
                "maxC1RoundTripErrorM": maximum_c1_error,
                "maxArtifactHardExclusionResidualM": metrics[
                    "maximum_hard_exclusion_residual_m"
                ],
                "maxArtifactOneMetreMeanErrorM": metrics[
                    "maximum_one_metre_mean_error_m"
                ],
                "independentlyRecomputedMasks": mask_rows,
                "waterAndStructuralExclusionsPreserved": True,
                "inheritedLayerIndexesByteExact": inherited_layers,
            }
        ),
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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_json_bytes(report))
    temporary.replace(path)
    return path
