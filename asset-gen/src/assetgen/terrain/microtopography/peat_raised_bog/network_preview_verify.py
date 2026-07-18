"""Independent verifier for the accepted peat-raised-bog network v4 research preview.

Recomputes the baseline LOD -2 cores from the pinned base and the bog relief from the
accepted v4 float, then checks the packed overlay against the same gate family as the
forest single-exemplar preview: coverage, parent closure, headers, decoded-hierarchy
LOD0 closure, apron seams, determinism, and hard-mask / excluded-C0 preservation.
"""
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
from ....cook.pinned_height import PinnedBaseHeight
from ....height_geom import HeightChunkId, plan_hero
from ....micro_verify import evidence_sha256, transient_merkle_root
from ....release import audit_base_release, micro_verification_binding
from ...repair.base_transaction import AuditedFormat1HeightSource


VERIFIER_ID = "assetgen.peat-bog-network-preview-verify.v1"
RECIPE_KIND = "research-peat-bog-network-preview-v1"
PARENT = HeightChunkId(-1, 335, 402)
AUTHORITY = HeightChunkId(0, 83, 100)
CORE_BBOX = (540224.0, 6429504.0, 540352.0, 6429632.0)
RELIEF_PITCH_M = 0.0625  # native finest-rung pitch; relief placed 1:1 into the fine core
FINE_CORE = 2048
AUTH_WINDOW = (slice(1088, 1216), slice(1600, 1728))
PARENT_WINDOW = (slice(64, 192), slice(64, 192))


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


def _load_json_bound(path: Path, sha_path: Path) -> dict[str, Any]:
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != sha_path.read_text().strip():
        raise ValueError(f"JSON identity mismatch: {path}")
    return json.loads(blob)


def _decoded(path: Path, encode: EncodeConfig) -> tuple[Any, bytes, np.ndarray]:
    meta, payload = read_chunk_v2(path)
    values = decode_quant16(encode, payload, meta.res, meta.qoffset, meta.qscale)
    return meta, payload, values


def verify_bog_network_preview(
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
        raise ValueError("bog network verifier received another recipe kind")
    if expectation.get("recipeSha256") != build_digest or plan.get("recipeSha256") != build_digest:
        raise ValueError("bog network recipe identity mismatch")
    if expectation.get("verifier") != {"id": VERIFIER_ID, "sourceSha256": verifier_source_sha256()}:
        raise ValueError("bog network expectation names another verifier")

    recipe_inputs = expectation["recipeInputs"]
    artifact_root = Path(recipe_inputs["artifact"]["root"])
    float_name = recipe_inputs["artifact"].get("floatName", "network-v4-float.npz")
    if _sha256(artifact_root / float_name) != recipe_inputs["artifact"]["floatSha256"]:
        raise ValueError("accepted bog float changed")
    if _sha256(artifact_root / "measurements.json") != recipe_inputs["artifact"]["measurementsSha256"]:
        raise ValueError("accepted bog measurements changed")
    float_npz = np.load(artifact_root / float_name)
    relief = np.asarray(float_npz["core_relief_00625m"], dtype=np.float64)
    if relief.shape != (FINE_CORE, FINE_CORE):
        raise ValueError(f"bog core relief must be {FINE_CORE} square")
    # Reinterpreted "open-water relief-free" safety gate (v7 #104 pool-depth carve): open water
    # must carry NO positive microform bump; it may carry only a monotone, non-positive Laugas
    # bed wedge (0 >= relief >= -dmax) that opens the water column under the INHERITED waterY
    # plane. Assert it directly on the frozen float here (not a static label).
    open_water_core = np.asarray(float_npz["open_water_core"], dtype=bool)
    water_relief = relief[open_water_core]
    water_positive_cells = int((water_relief > 1e-6).sum())
    if water_positive_cells != 0:
        raise ValueError(f"open water carries {water_positive_cells} positive-relief cells (fake microform)")
    water_relief_min_m = float(water_relief.min()) if water_relief.size else 0.0

    coverage = plan_hero(PARENT.cx, PARENT.cz)
    expected_coverage = {
        "parent": [PARENT.lod, PARENT.cx, PARENT.cz],
        "publishedFine": [[c.lod, c.cx, c.cz] for c in coverage.published_fine],
        "transientSupport": [[c.lod, c.cx, c.cz] for c in coverage.transient_support],
        "authority": [AUTHORITY.lod, AUTHORITY.cx, AUTHORITY.cz],
    }
    if plan["microCoverage"] != expected_coverage:
        raise ValueError("bog preview coverage differs from exact hero closure")

    evidence = json.loads((build_root / "evidence/peat-bog-network-preview.json").read_bytes())
    identities = {HeightChunkId(*item["key"]): item for item in evidence["children"]}
    if set(identities) != set((*coverage.published_fine, *coverage.transient_support)):
        raise ValueError("bog preview evidence lacks the exact fine closure")

    corrected = recipe_inputs["correctedBase"]
    if corrected["manifestSha256"] != expectation["baseManifestSha256"]:
        raise ValueError("corrected base is not the expected release")
    audit_base_release(base_manifest_path, corrected["manifestSha256"], base_out_root)
    pinned = PinnedBaseHeight(
        base_manifest_path, corrected["manifestSha256"], base_out_root, base.encode
    )
    # The corrected base only re-splices height around the core; the baseline cores the
    # packer used come from the *pinned source* base, whose height LOD0 outside the core
    # window is byte-identical to the corrected base. Recompute from the source base.
    source_base = Path(recipe_inputs["sourceBase"]["manifest"])
    source_sha = recipe_inputs["sourceBase"]["manifestSha256"]
    if _sha256(source_base) != source_sha:
        raise ValueError("pinned source base manifest changed")
    audit_base_release(source_base, source_sha, base_out_root)
    source_pinned = PinnedBaseHeight(source_base, source_sha, base_out_root, base.encode)
    from ..forest_exemplar.generalization_preview import _baseline_core

    def core_grid(chunk: HeightChunkId) -> tuple[np.ndarray, np.ndarray]:
        step = base.grid.chunk_m * (base.grid.lod_step ** chunk.lod)
        texel = step / FINE_CORE
        origin_e = base.grid.anchor_e + chunk.cx * step
        origin_n = base.grid.anchor_n - chunk.cz * step
        east = origin_e + (np.arange(FINE_CORE) + 0.5) * texel
        north = origin_n - (np.arange(FINE_CORE) + 0.5) * texel
        return east, north

    expected_cores: dict[HeightChunkId, np.ndarray] = {}
    applied_masks: dict[HeightChunkId, np.ndarray] = {}

    def expected_core(chunk: HeightChunkId) -> np.ndarray:
        if chunk not in expected_cores:
            values = np.asarray(_baseline_core(source_pinned, chunk), dtype=np.float64)
            east, north = core_grid(chunk)
            e0, n0, e1, n1 = CORE_BBOX
            col_in = (east >= e0) & (east < e1)
            row_in = (north >= n0) & (north < n1)
            mask = np.zeros((FINE_CORE, FINE_CORE), dtype=bool)
            if col_in.any() and row_in.any():
                rr = np.nonzero(row_in)[0]
                cc = np.nonzero(col_in)[0]
                # Native 0.0625 m relief, world-aligned to the fine-core lattice: place 1:1 (each
                # in-core fine texel <- its own relief texel), NOT a 4x4 nearest block.
                rel_r0 = int(round((n1 - north[rr[0]]) / RELIEF_PITCH_M - 0.5))
                rel_c0 = int(round((east[cc[0]] - e0) / RELIEF_PITCH_M - 0.5))
                block = relief[rel_r0:rel_r0 + rr.size, rel_c0:rel_c0 + cc.size]
                if block.shape != (rr.size, cc.size):
                    raise ValueError("relief core does not cover the in-core fine texels 1:1")
                grid = np.ix_(rr, cc)
                values[grid] = values[grid] + block
                mask[grid] = True
            expected_cores[chunk] = values.astype(np.float32)
            applied_masks[chunk] = mask
        return expected_cores[chunk]

    published = set(coverage.published_fine)
    paths: dict[HeightChunkId, Path] = {}
    header_rows: list[dict[str, Any]] = []
    max_core_error = 0.0
    changed_samples = 0
    for chunk in (*coverage.published_fine, *coverage.transient_support):
        root = build_root / ("chunks" if chunk in published else "transient")
        path = root / "height/-2" / f"{chunk.cx}_{chunk.cz}.lac"
        identity = identities[chunk]
        if _sha256(path) != identity["sha256"] or path.stat().st_size != identity["bytes"]:
            raise ValueError(f"fine artifact changed: {chunk}")
        meta, _payload, decoded = _decoded(path, base.encode)
        if (
            (meta.lod, meta.cx, meta.cz) != (chunk.lod, chunk.cx, chunk.cz)
            or meta.res != FINE_CORE + 1
            or abs(meta.qscale - 0.002) > 1e-8
        ):
            raise ValueError(f"fine header mismatch: {chunk}")
        exp = expected_core(chunk)
        core_err = float(np.max(np.abs(decoded[:-1, :-1].astype(np.float64) - exp.astype(np.float64))))
        max_core_error = max(max_core_error, core_err)
        changed_samples += int(applied_masks[chunk].sum())
        paths[chunk] = path
        header_rows.append({"key": identity["key"], "qoffset": meta.qoffset, "qscale": meta.qscale})
    if max_core_error > 0.00101:
        raise ValueError(f"decoded fine core deviates from baseline+relief by {max_core_error} m")

    # Apron seams across every adjacent published/transient pair.
    max_seam = 0.0
    comparisons = 0
    decoded_cache = {c: _decoded(p, base.encode)[2] for c, p in paths.items()}
    for chunk, left in decoded_cache.items():
        east = HeightChunkId(-2, chunk.cx + 1, chunk.cz)
        south = HeightChunkId(-2, chunk.cx, chunk.cz + 1)
        if east in decoded_cache:
            max_seam = max(max_seam, float(np.max(np.abs(left[:, -1] - decoded_cache[east][:, 0]))))
            comparisons += 1
        if south in decoded_cache:
            max_seam = max(max_seam, float(np.max(np.abs(left[-1, :] - decoded_cache[south][0, :]))))
            comparisons += 1
    if max_seam != 0.0:
        raise ValueError(f"decoded fine seam is {max_seam} m")

    # Parent byte-exactly derived from decoded children.
    mosaic = assemble_parent_source_memmap(
        build_root / "scratch/verify-bog-parent-source.f32",
        coverage,
        lambda chunk: decoded_cache[chunk] if chunk in decoded_cache else _decoded(paths[chunk], base.encode)[2],
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

    # LOD0 closure: corrected authority masked cells == box-mean of decoded children.
    source = AuditedFormat1HeightSource(
        manifest_path=base_manifest_path,
        manifest_sha256=corrected["manifestSha256"],
        content_root=base_out_root,
        encode=base.encode,
        cache_chunks=2,
    )
    decoded_lod0 = source.load(AUTHORITY).decoded[:-1, :-1]
    reduced_lod0 = box_mean4_striped(parent_decoded[:2048, :2048])
    max_lod0_error = float(
        np.max(np.abs(decoded_lod0[AUTH_WINDOW] - reduced_lod0[PARENT_WINDOW]))
    )
    if max_lod0_error > 0.00501:
        raise ValueError(f"corrected LOD0 differs from decoded-child closure by {max_lod0_error} m")

    # Excluded C0 outside the core is unchanged from the pure pinned baseline.
    max_outside_error = 0.0
    for chunk in coverage.published_fine:
        if not applied_masks[chunk].any():
            continue
        baseline = np.asarray(_baseline_core(source_pinned, chunk), dtype=np.float64)
        decoded = decoded_cache[chunk][:-1, :-1].astype(np.float64)
        outside = ~applied_masks[chunk]
        max_outside_error = max(
            max_outside_error, float(np.max(np.abs(decoded[outside] - baseline[outside])))
        )
    if max_outside_error > 0.00101:
        raise ValueError(f"bog overlay changed excluded C0 by {max_outside_error} m")

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
            "decodedChildCount": len(coverage.published_fine) + len(coverage.transient_support),
            "payloadByteExact": True,
            "correctedHierarchyPlanSha256": corrected["hierarchyPlanSha256"],
        }),
        "headers": _gate({"fine": header_rows, "parentQscale": parent_meta.qscale}),
        "decodedHierarchy": _gate({
            "maxLod0DecodedChildErrorM": max_lod0_error,
            "limitM": 0.00501,
            "maskedSpliceThroughLod4": True,
        }),
        "aprons": _gate({"maxDecodedSeamErrorM": max_seam, "comparisons": comparisons}),
        "determinism": _gate({
            "artifactFloatSha256": recipe_inputs["artifact"]["floatSha256"],
            "sourceBaseManifestSha256": source_sha,
            "verifierSourceSha256": verifier_source_sha256(),
        }),
        "hardMasks": _gate({
            "carvedSamples": changed_samples,
            "maxCoreRoundTripErrorM": max_core_error,
            "maxExcludedC0ChangeM": max_outside_error,
            "openWaterNoPositiveMicroform": True,
            "openWaterPositiveReliefCells": water_positive_cells,
            "openWaterBedWedgeMinReliefM": water_relief_min_m,
            "openWaterGate": "reinterpreted_#104_monotone_nonpositive_bed_wedge_waterY_inherited",
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
