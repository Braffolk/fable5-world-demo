"""Independent verifier for a contiguous multi-parent forest preview."""

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
from ....height_geom import HeightChunkId, plan_hero, plan_parent_set
from ....micro_verify import evidence_sha256, transient_merkle_root
from ....process.micro_masks import rasterize_micro_morphology_mask
from ....release import audit_base_release, micro_verification_binding, read_v1_index
from ...repair.base_transaction import AuditedFormat1HeightSource
from .adjacent_preview import (
    TEXEL_M,
    _layout_for_schema,
)


VERIFIER_ID = "assetgen.forest-generalization-preview-verify.v2"
RECIPE_KIND = "research-microtopography-generalization-preview-v1"


def _verifier_source_sha256() -> str:
    return hashlib.sha256(
        Path(__file__).with_name("generalization_preview_verify.py").read_bytes()
    ).hexdigest()


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


def _verify_mask(
    *,
    build_root: Path,
    site: Any,
    identity: dict[str, Any],
) -> tuple[np.ndarray, dict[str, int]]:
    relative = Path(identity["path"])
    path = (build_root / relative).resolve()
    if build_root.resolve() not in path.parents:
        raise ValueError("morphology mask escaped the build root")
    if _sha256(path) != identity["sha256"]:
        raise ValueError(f"staged morphology mask changed: {site.site_id}")
    packed = np.load(path, mmap_mode="r")
    if packed.shape != (8192, 1024) or packed.dtype != np.uint8:
        raise ValueError(f"invalid morphology mask shape: {site.site_id}")
    evidence: dict[str, int] = {}
    e_min, _n_min, _e_max, n_max = site.bbox_en
    for tile_row in range(4):
        for tile_col in range(4):
            tile_e = e_min + tile_col * 128
            tile_n = n_max - tile_row * 128
            east = tile_e + (np.arange(2048) + 0.5) * TEXEL_M
            north = tile_n - (np.arange(2048) + 0.5) * TEXEL_M
            mask = rasterize_micro_morphology_mask(east, north)
            actual = np.packbits(mask.allowed, axis=1, bitorder="little")
            rows = np.s_[tile_row * 2048 : (tile_row + 1) * 2048]
            cols = np.s_[tile_col * 256 : (tile_col + 1) * 256]
            if not np.array_equal(actual, packed[rows, cols]):
                raise ValueError(f"morphology mask recomputation changed: {site.site_id}")
            for key, value in mask.evidence().items():
                if isinstance(value, int):
                    evidence[key] = evidence.get(key, 0) + value
    if evidence != identity["evidence"]:
        raise ValueError(f"morphology mask evidence changed: {site.site_id}")
    allowed = np.unpackbits(packed, axis=1, bitorder="little")[:, :8192].astype(bool)
    return allowed, evidence


def verify_adjacent_preview(
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
    inputs = expectation["recipeInputs"]
    layout = _layout_for_schema(inputs.get("artifact", {}).get("schema", ""))
    sites = layout.sites
    if inputs.get("id") != layout.recipe_id:
        raise ValueError("adjacent verifier received another recipe")
    if expectation.get("recipeKind") != RECIPE_KIND or plan.get("microRecipeKind") != RECIPE_KIND:
        raise ValueError("adjacent verifier received another recipe kind")
    if expectation.get("recipeSha256") != build_digest or plan.get("recipeSha256") != build_digest:
        raise ValueError("adjacent recipe identity mismatch")
    expected_verifier = {"id": VERIFIER_ID, "sourceSha256": _verifier_source_sha256()}
    if expectation.get("verifier") != expected_verifier:
        raise ValueError("adjacent expectation names another verifier")

    artifact_root = Path(inputs["artifact"]["root"])
    if _sha256(artifact_root / "manifest.json") != inputs["artifact"]["manifestSha256"]:
        raise ValueError("adjacent artifact manifest changed")
    artifact_manifest = json.loads((artifact_root / "manifest.json").read_bytes())
    metrics = artifact_manifest.get("metrics", {})
    if (
        artifact_manifest.get("schema") != layout.schema
        or artifact_manifest.get("status") != "inspect_float_preview"
        or artifact_manifest.get("failures") != []
        or tuple(metrics.get("bbox_en", ())) != layout.master_bbox
        or tuple(metrics.get("shape", ())) != layout.master_shape
        or metrics.get("maximum_hard_exclusion_residual_m") != 0.0
        or float(metrics.get("maximum_one_metre_mean_error_m", 1.0)) > 1e-12
    ):
        raise ValueError("adjacent artifact gates changed")
    for relative, identity in artifact_manifest["files"].items():
        path = artifact_root / relative
        if _sha256(path) != identity["sha256"] or path.stat().st_size != identity["bytes"]:
            raise ValueError(f"adjacent artifact changed: {relative}")
    master = np.load(artifact_root / "surface/c1_height_f32.npy", mmap_mode="r")
    if master.shape != layout.master_shape or master.dtype != np.float32:
        raise ValueError("adjacent master shape changed")

    coverage = plan_parent_set(tuple(site.parent for site in sites))
    expected_coverage = {
        "sites": [site.json() for site in sites],
        "parents": [[c.lod, c.cx, c.cz] for c in coverage.parents],
        "publishedFine": [[c.lod, c.cx, c.cz] for c in coverage.published_fine],
        "transientSupport": [[c.lod, c.cx, c.cz] for c in coverage.transient_support],
        "authorities": [[c.lod, c.cx, c.cz] for c in coverage.authorities_lod0],
    }
    if plan["microCoverage"] != expected_coverage or expectation["sites"] != expected_coverage["sites"]:
        raise ValueError("adjacent coverage differs from exact parent closure")
    evidence = _load_json_bound(build_root / "evidence/forest-adjacent-preview.json")
    identities = {HeightChunkId(*row["key"]): row for row in evidence["children"]}
    all_fine = (*coverage.published_fine, *coverage.transient_support)
    if set(identities) != set(all_fine):
        raise ValueError("adjacent evidence lacks exact fine closure")
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
        if (
            (meta.lod, meta.cx, meta.cz) != (chunk.lod, chunk.cx, chunk.cz)
            or meta.res != 2049
            or abs(meta.qscale - 0.002) > 1e-8
        ):
            raise ValueError(f"fine header mismatch: {chunk}")
        paths[chunk] = path
        header_rows.append(
            {"key": identity["key"], "qoffset": meta.qoffset, "qscale": meta.qscale}
        )

    allowed_by_parent: dict[HeightChunkId, np.ndarray] = {}
    mask_rows: dict[str, Any] = {}
    for site in sites:
        allowed, mask_evidence = _verify_mask(
            build_root=build_root, site=site, identity=inputs["masks"][site.site_id]
        )
        allowed_by_parent[site.parent] = allowed
        mask_rows[site.site_id] = mask_evidence

    parent_evidence = {HeightChunkId(*row["key"]): row for row in evidence["parents"]}
    if set(parent_evidence) != set(coverage.parents):
        raise ValueError("adjacent evidence lacks exact parent set")
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

    maximum_c1_error = 0.0
    maximum_seam = 0.0
    seam_comparisons = 0
    maximum_lod0_error = 0.0
    parent_rows: list[dict[str, Any]] = []
    parent_decoded: dict[HeightChunkId, np.ndarray] = {}
    for site in sites:
        hero = plan_hero(site.parent.cx, site.parent.cz)
        x0, z0 = site.parent.cx * 4, site.parent.cz * 4
        site_c1_error = 0.0
        for chunk in hero.published_fine:
            dx = chunk.cx - layout.master_fine_origin[0]
            dz = chunk.cz - layout.master_fine_origin[1]
            decoded = _decoded(paths[chunk], base.encode)[2][:-1, :-1]
            expected = master[
                dz * 2048 : (dz + 1) * 2048,
                dx * 2048 : (dx + 1) * 2048,
            ]
            site_c1_error = max(
                site_c1_error,
                float(np.max(np.abs(decoded.astype(np.float64) - expected.astype(np.float64)))),
            )
        maximum_c1_error = max(maximum_c1_error, site_c1_error)
        if site_c1_error > 0.00101:
            raise ValueError(f"decoded C1 error at {site.site_id} is {site_c1_error} m")
        site_seam, site_comparisons = _max_seam(hero, paths, base.encode)
        maximum_seam = max(maximum_seam, site_seam)
        seam_comparisons += site_comparisons
        if site_seam != 0.0:
            raise ValueError(f"decoded fine seam at {site.site_id} is {site_seam} m")

        mosaic = assemble_parent_source_memmap(
            build_root / f"scratch/verify-adjacent-parent-{site.parent.cx}-{site.parent.cz}.f32",
            hero,
            lambda chunk: _decoded(paths[chunk], base.encode)[2],
        )
        expected_parent = box_mean4_striped(mosaic)
        parent_path = build_root / f"chunks/height/-1/{site.parent.cx}_{site.parent.cz}.lac"
        parent_meta, parent_payload, decoded_parent = _decoded(parent_path, base.encode)
        identity = parent_evidence[site.parent]
        if _sha256(parent_path) != identity["sha256"] or parent_path.stat().st_size != identity["bytes"]:
            raise ValueError(f"parent artifact changed: {site.parent}")
        if abs(parent_meta.qscale - 0.005) > 1e-8:
            raise ValueError(f"parent qscale changed: {site.parent}")
        expected_payload, expected_offset, expected_scale = encode_quant16_checked(
            base.encode, expected_parent, parent_meta.qscale, parent_meta.qoffset
        )
        if (
            expected_payload != parent_payload
            or expected_offset != parent_meta.qoffset
            or expected_scale != parent_meta.qscale
        ):
            raise ValueError(f"{site.parent} is not byte-exactly derived from decoded children")
        parent_decoded[site.parent] = decoded_parent

        reduced_lod0 = box_mean4_striped(decoded_parent[:2048, :2048])
        allowed = allowed_by_parent[site.parent]
        coarse_mask = allowed.reshape(512, 16, 512, 16).any(axis=(1, 3))
        origin_e = base.grid.anchor_e + site.authority.cx * base.grid.chunk_m
        origin_n = base.grid.anchor_n - site.authority.cz * base.grid.chunk_m
        x = site.bbox_en[0] - origin_e
        y = origin_n - site.bbox_en[3]
        decoded_lod0 = corrected_source.load(site.authority).decoded[:-1, :-1]
        actual = decoded_lod0[y : y + 512, x : x + 512][coarse_mask]
        expected_lod0 = reduced_lod0[coarse_mask]
        site_lod0_error = float(np.max(np.abs(actual - expected_lod0), initial=0.0))
        maximum_lod0_error = max(maximum_lod0_error, site_lod0_error)
        if site_lod0_error > 0.00501:
            raise ValueError(f"corrected LOD0 at {site.site_id} differs by {site_lod0_error} m")
        parent_rows.append(
            {
                "siteId": site.site_id,
                "parent": [site.parent.lod, site.parent.cx, site.parent.cz],
                "sha256": _sha256(parent_path),
                "payloadByteExact": True,
                "maxC1RoundTripErrorM": site_c1_error,
                "maxLod0ClosureErrorM": site_lod0_error,
            }
        )

    parents = sorted(parent_decoded)
    parent_shared_seam = 0.0
    shared_child_seam = 0.0
    shared_parent_comparisons = 0
    for first in parents:
        for second in parents:
            if first.cz == second.cz and second.cx == first.cx + 1:
                parent_shared_seam = max(
                    parent_shared_seam,
                    float(
                        np.max(
                            np.abs(
                                parent_decoded[first][:, -1]
                                - parent_decoded[second][:, 0]
                            )
                        )
                    ),
                )
                boundary_cx = second.cx * 4
                for cz in range(first.cz * 4, first.cz * 4 + 4):
                    left = _decoded(
                        paths[HeightChunkId(-2, boundary_cx - 1, cz)], base.encode
                    )[2]
                    right = _decoded(
                        paths[HeightChunkId(-2, boundary_cx, cz)], base.encode
                    )[2]
                    shared_child_seam = max(
                        shared_child_seam,
                        float(np.max(np.abs(left[:, -1] - right[:, 0]))),
                    )
                shared_parent_comparisons += 1
            if first.cx == second.cx and second.cz == first.cz + 1:
                parent_shared_seam = max(
                    parent_shared_seam,
                    float(
                        np.max(
                            np.abs(
                                parent_decoded[first][-1, :]
                                - parent_decoded[second][0, :]
                            )
                        )
                    ),
                )
                boundary_cz = second.cz * 4
                for cx in range(first.cx * 4, first.cx * 4 + 4):
                    north = _decoded(
                        paths[HeightChunkId(-2, cx, boundary_cz - 1)], base.encode
                    )[2]
                    south = _decoded(
                        paths[HeightChunkId(-2, cx, boundary_cz)], base.encode
                    )[2]
                    shared_child_seam = max(
                        shared_child_seam,
                        float(np.max(np.abs(north[-1, :] - south[0, :]))),
                    )
                shared_parent_comparisons += 1
    if shared_parent_comparisons == 0:
        raise ValueError("adjacent preview contains no shared parent boundaries")
    # Parent chunks select independent integer-metre qoffsets. Their wire lattices
    # coincide, while float32 decode order can differ by one ULP around 60 metres.
    if parent_shared_seam > 1e-5:
        raise ValueError(f"decoded shared parent seam is {parent_shared_seam} m")
    if shared_child_seam != 0.0:
        raise ValueError(f"decoded shared child seam is {shared_child_seam} m")

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
        raise ValueError("corrected base changed inherited height keys")
    changed_height = {key for key in before_height if before_height[key] != after_height[key]}
    transaction_plan = _load_json_bound(
        build_root / "corrected-base/resolved-hierarchy-plan.json"
    )
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
                "sharedParentSeamLimitM": 1e-5,
                "sharedParentBoundaryComparisons": shared_parent_comparisons,
            }
        ),
        "headers": _gate({"fine": header_rows, "parentQscaleM": 0.005}),
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
                "sharedChildSeamErrorM": shared_child_seam,
                "sharedParentSeamErrorM": parent_shared_seam,
                "sharedParentSeamLimitM": 1e-5,
                "sharedParentBoundaryComparisons": shared_parent_comparisons,
            }
        ),
        "determinism": _gate(
            {
                "artifactManifestSha256": inputs["artifact"]["manifestSha256"],
                "sourceBaseManifestSha256": inputs["sourceBase"]["manifestSha256"],
                "verifierSourceSha256": _verifier_source_sha256(),
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
        "verifierSourceSha256": _verifier_source_sha256(),
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
