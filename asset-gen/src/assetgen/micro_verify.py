"""Independent verifier and evidence envelope for the Stage-1 fixture release."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

VERIFIER_ID = "assetgen.micro-verify.v1"
VERIFIER_CAN_AUTHORIZE_RELEASE = True


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def verifier_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def evidence_sha256(evidence: Any) -> str:
    return hashlib.sha256(_json_bytes(evidence)).hexdigest()


def transient_merkle_root(artifacts: list[dict[str, Any]]) -> str:
    ordered = sorted(artifacts, key=lambda item: tuple(item["key"]))
    return hashlib.sha256(_json_bytes(ordered)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _gate(evidence: Any) -> dict[str, Any]:
    return {
        "passed": True,
        "evidence": evidence,
        "evidenceSha256": evidence_sha256(evidence),
    }


def _gate_not_applicable(evidence: Any) -> dict[str, Any]:
    return {
        "status": "notApplicable",
        "applicable": False,
        "evidence": evidence,
        "evidenceSha256": evidence_sha256(evidence),
    }


def _decoded_authority_mean_error(
    decoded: Any, authority: Any, factor: int = 16
) -> float:
    import numpy as np

    values = np.asarray(decoded)
    coarse = np.asarray(authority, dtype=np.float64)
    rows, cols = coarse.shape
    core = values[: rows * factor, : cols * factor]
    means = core.reshape(rows, factor, cols, factor).mean(
        axis=(1, 3), dtype=np.float64
    )
    return float(np.max(np.abs(means - coarse)))


def _decoded_rejected_residual(
    decoded: Any, expected_conservative: Any, allowed: Any
) -> tuple[int, float]:
    import numpy as np

    actual = np.asarray(decoded)
    expected = np.asarray(expected_conservative)
    soft = np.asarray(allowed, dtype=bool)
    if actual.shape != expected.shape or actual.shape != soft.shape:
        raise ValueError("decoded rejection check arrays differ in shape")
    rejected = ~soft
    count = int(np.count_nonzero(rejected))
    maximum = (
        float(np.max(np.abs(actual[rejected] - expected[rejected])))
        if count
        else 0.0
    )
    return count, maximum


def verify_micro_fixture(
    build_digest: str,
    base_manifest_path: Path,
    base_out_root: Path,
    work_root: Path,
) -> Path:
    """Reopen and independently regenerate every fixture artifact.

    Cooker sidecars are cross-checked only after all measurements have been
    derived from LAC2 bytes, the frozen plan, and the pinned LOD0 release.
    """
    import numpy as np

    from .config import load_base
    from .cook.chunkio import read_chunk_v2
    from .cook.encode import decode_quant16, encode_quant16_checked
    from .cook.micro_fixture_cook import _fit_global_base, _fixture_chunk_surface
    from .cook.micro_hierarchy import (
        assemble_parent_source_memmap,
        box_mean4_striped,
        dependency_merkle_root,
    )
    from .cook.pinned_height import PinnedBaseHeight
    from .height_geom import HeightChunkId, chunk_origin_en_units, plan_hero
    from .micro_config import load_micro_config
    from .release import micro_verification_binding

    build_root = work_root / "builds" / build_digest
    plan_blob = (build_root / "plan.json").read_bytes()
    if hashlib.sha256(plan_blob).hexdigest() != (build_root / "plan.sha256").read_text().strip():
        raise ValueError("micro verifier: plan digest mismatch")
    plan = json.loads(plan_blob)
    if plan.get("manifestFormat") != 2 or plan.get("recipeSha256") != build_digest:
        raise ValueError("micro verifier: plan identity mismatch")
    coverage_doc = plan["microCoverage"]
    parent = HeightChunkId(*coverage_doc["parent"])
    coverage = plan_hero(parent.cx, parent.cz)
    expected_doc = {
        "parent": [coverage.parent.lod, coverage.parent.cx, coverage.parent.cz],
        "publishedFine": [[c.lod, c.cx, c.cz] for c in coverage.published_fine],
        "transientSupport": [[c.lod, c.cx, c.cz] for c in coverage.transient_support],
        "authority": [
            coverage.authority_lod0.lod,
            coverage.authority_lod0.cx,
            coverage.authority_lod0.cz,
        ],
    }
    if coverage_doc != expected_doc:
        raise ValueError("micro verifier: plan coverage is not the exact geometry closure")

    base = load_base()
    micro = load_micro_config(base)
    published = set(coverage.published_fine)
    all_fine = (*coverage.published_fine, *coverage.transient_support)
    decoded: dict[HeightChunkId, np.ndarray] = {}
    fine_payloads: dict[HeightChunkId, bytes] = {}
    fine_artifacts: dict[HeightChunkId, dict[str, Any]] = {}
    header_rows = []
    for chunk in all_fine:
        root = build_root / ("chunks" if chunk in published else "transient")
        path = root / "height" / str(chunk.lod) / f"{chunk.cx}_{chunk.cz}.lac"
        meta, payload = read_chunk_v2(path)
        origin_e_u, origin_n_u = chunk_origin_en_units(base.grid, chunk)
        if (
            meta.layer, meta.lod, meta.cx, meta.cz, meta.enc, meta.res, meta.count,
            meta.origin_e, meta.origin_n,
        ) != (
            "height", chunk.lod, chunk.cx, chunk.cz, 1, 2049, 0,
            origin_e_u / 32.0, origin_n_u / 32.0,
        ):
            raise ValueError(f"micro verifier: invalid fine header {chunk}")
        if abs(meta.qscale - micro.selected_fine_qscale_m) > 1e-8:
            raise ValueError(f"micro verifier: invalid fine qscale {chunk}")
        values = decode_quant16(base.encode, payload, meta.res, meta.qoffset, meta.qscale)
        if not np.isfinite(values).all():
            raise ValueError(f"micro verifier: nonfinite fine decode {chunk}")
        digest = _sha256_file(path)
        decoded[chunk] = values
        fine_payloads[chunk] = payload
        fine_artifacts[chunk] = {
            "key": [chunk.lod, chunk.cx, chunk.cz],
            "sha256": digest,
            "size": path.stat().st_size,
            "qoffset": meta.qoffset,
            "qscale": meta.qscale,
        }
        header_rows.append({"key": [chunk.lod, chunk.cx, chunk.cz], "sha256": digest})
    qoffsets = {item["qoffset"] for item in fine_artifacts.values()}
    if len(qoffsets) != 1:
        raise ValueError("micro verifier: fine closure does not share one qoffset")

    parent_path = (
        build_root / "chunks" / "height" / str(parent.lod) / f"{parent.cx}_{parent.cz}.lac"
    )
    parent_meta, parent_payload = read_chunk_v2(parent_path)
    parent_origin = chunk_origin_en_units(base.grid, parent)
    if (
        parent_meta.layer, parent_meta.lod, parent_meta.cx, parent_meta.cz,
        parent_meta.enc, parent_meta.res, parent_meta.count,
        parent_meta.origin_e, parent_meta.origin_n,
    ) != (
        "height", -1, parent.cx, parent.cz, 1, 2049, 0,
        parent_origin[0] / 32.0, parent_origin[1] / 32.0,
    ):
        raise ValueError("micro verifier: invalid parent header")
    parent_decoded = decode_quant16(
        base.encode, parent_payload, parent_meta.res, parent_meta.qoffset, parent_meta.qscale
    )

    mosaic = assemble_parent_source_memmap(
        build_root / "scratch" / "verify-parent-source.f32",
        coverage,
        decoded.__getitem__,
    )
    derived_parent = box_mean4_striped(mosaic)
    expected_payload, expected_offset, expected_scale = encode_quant16_checked(
        base.encode, derived_parent, micro.parent_qscale_m
    )
    if (
        parent_payload != expected_payload
        or parent_meta.qoffset != expected_offset
        or parent_meta.qscale != expected_scale
    ):
        raise ValueError("micro verifier: parent is not byte-exactly derived from decoded children")

    max_seam = 0.0
    fine_cx0, fine_cz0 = parent.cx * 4, parent.cz * 4
    for dz in range(5):
        for dx in range(4):
            left = decoded[HeightChunkId(-2, fine_cx0 + dx, fine_cz0 + dz)]
            right = decoded[HeightChunkId(-2, fine_cx0 + dx + 1, fine_cz0 + dz)]
            max_seam = max(max_seam, float(np.max(np.abs(left[:, -1] - right[:, 0]))))
    for dz in range(4):
        for dx in range(5):
            north = decoded[HeightChunkId(-2, fine_cx0 + dx, fine_cz0 + dz)]
            south = decoded[HeightChunkId(-2, fine_cx0 + dx, fine_cz0 + dz + 1)]
            max_seam = max(max_seam, float(np.max(np.abs(north[-1, :] - south[0, :]))))
    if max_seam != 0.0:
        raise ValueError(f"micro verifier: decoded apron seam is {max_seam} m")

    reader = PinnedBaseHeight(
        base_manifest_path, micro.base_manifest_sha256, base_out_root, base.encode
    )
    spline = _fit_global_base(reader, base, coverage)
    first_origin = chunk_origin_en_units(base.grid, coverage.published_fine[0])
    domain_authority = reader.read_cells(
        first_origin[0] // 32, first_origin[1] // 32, 640, 640
    )
    canonical_fine_qoffset = float(np.floor(float(domain_authority.min()) - 2.0))
    max_regeneration_error = 0.0
    max_authority_mean_error = 0.0
    for chunk in all_fine:
        expected = _fixture_chunk_surface(reader, spline, base, chunk)
        canonical_payload, canonical_offset, canonical_scale = encode_quant16_checked(
            base.encode,
            expected,
            micro.selected_fine_qscale_m,
            canonical_fine_qoffset,
        )
        if (
            fine_payloads[chunk] != canonical_payload
            or fine_artifacts[chunk]["qoffset"] != canonical_offset
            or fine_artifacts[chunk]["qscale"] != canonical_scale
        ):
            raise ValueError(f"micro verifier: noncanonical fine payload {chunk}")
        actual = decoded[chunk].astype(np.float64)
        error = float(np.max(np.abs(expected - actual)))
        peak = np.float32(np.max(np.abs(decoded[chunk])))
        limit = float(fine_artifacts[chunk]["qscale"]) * 0.5 + 2.0 * abs(float(np.spacing(peak)))
        if error > limit:
            raise ValueError(f"micro verifier: fixture regeneration mismatch {chunk}: {error}")
        max_regeneration_error = max(max_regeneration_error, error)
        origin_e_u, origin_n_u = chunk_origin_en_units(base.grid, chunk)
        authority = reader.read_cells(origin_e_u // 32, origin_n_u // 32, 128, 128)
        means = actual[:2048, :2048].reshape(128, 16, 128, 16).mean(axis=(1, 3))
        max_authority_mean_error = max(
            max_authority_mean_error,
            float(np.max(np.abs(means - authority.astype(np.float64)))),
        )
    if max_authority_mean_error > micro.selected_fine_qscale_m * 0.125:
        raise ValueError(
            f"micro verifier: decoded authority mean error {max_authority_mean_error} m"
        )

    actual_dependencies = [
        (chunk, fine_artifacts[chunk]["sha256"]) for chunk in all_fine
    ]
    dependency_root = dependency_merkle_root(actual_dependencies)
    cook_sidecar = json.loads((build_root / "evidence" / "fixture-cook.json").read_bytes())
    if (
        cook_sidecar.get("recipeSha256") != build_digest
        or cook_sidecar.get("dependencyMerkleRoot") != dependency_root
        or cook_sidecar.get("parent", {}).get("sha256") != _sha256_file(parent_path)
    ):
        raise ValueError("micro verifier: parent sidecar is not bound to actual dependencies")

    transient_artifacts = [
        {
            "key": fine_artifacts[chunk]["key"],
            "sha256": fine_artifacts[chunk]["sha256"],
            "size": fine_artifacts[chunk]["size"],
        }
        for chunk in coverage.transient_support
    ]
    binding = micro_verification_binding(build_digest, work_root)
    gates = {
        "coverage": _gate({**expected_doc, "stagedCount": 17, "transientCount": 9}),
        "parentClosure": _gate({
            "dependencyMerkleRoot": dependency_root,
            "derivedParentSha256": _sha256_file(parent_path),
            "reducer": "decoded-f32-fixed-order-box-mean-16",
        }),
        "headers": _gate({"fine": header_rows, "parentSha256": _sha256_file(parent_path)}),
        "decodedHierarchy": _gate({
            "maxAuthorityMeanErrorM": max_authority_mean_error,
            "maxFixtureRegenerationErrorM": max_regeneration_error,
            "parentPayloadByteExact": True,
        }),
        "aprons": _gate({"maxDecodedSeamErrorM": max_seam, "comparisons": 40}),
        "determinism": _gate({
            "independentRegeneration": "absolute-coordinate fixture -> canonical quant16 payload",
            "allFinePayloadsByteExact": 25,
            "parentPayloadMatched": True,
        }),
        "hardMasks": _gate_not_applicable({
            "fixturePolicy": "no hard-mask inputs; this gate is fixture-only",
            "productionMorphologyAuthorized": False,
        }),
    }
    report = {
        **binding,
        "verifier": VERIFIER_ID,
        "verifierSourceSha256": verifier_source_sha256(),
        "fixtureOnly": True,
        "gates": gates,
        "transientArtifacts": transient_artifacts,
        "transientSupportMerkleRoot": transient_merkle_root(transient_artifacts),
    }
    path = build_root / "micro-verify.json"
    blob = _json_bytes(report)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as output:
        output.write(blob)
        output.flush()
        import os

        os.fsync(output.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return path


def verify_micro_synthesis(
    build_digest: str,
    base_manifest_path: Path,
    base_out_root: Path,
    work_root: Path,
) -> Path:
    """Verify production-pilot packing without authorizing morphology or masks."""
    import numpy as np
    from scipy.interpolate import RectBivariateSpline

    from .config import ASSET_GEN_ROOT, DATA_IN, load_base
    from .cook.chunkio import read_chunk_v2
    from .cook.encode import decode_quant16, encode_quant16_checked
    from .cook.micro_hierarchy import (
        assemble_parent_source_memmap,
        box_mean4_striped,
        dependency_merkle_root,
    )
    from .cook.pinned_height import PinnedBaseHeight
    from .height_geom import HeightChunkId, chunk_origin_en_units, plan_hero
    from .micro_config import load_micro_config
    from .micro_recipe import _synthesis_environment_identity
    from .process.micro_fixture import conservative_cell_correct
    from .process.microtopo import load_exemplar_bank
    from .process.micro_masks import rasterize_micro_morphology_mask
    from .release import micro_verification_binding

    build_root = work_root / "builds" / build_digest
    plan_blob = (build_root / "plan.json").read_bytes()
    if hashlib.sha256(plan_blob).hexdigest() != (build_root / "plan.sha256").read_text().strip():
        raise ValueError("micro synthesis verifier: plan digest mismatch")
    plan = json.loads(plan_blob)
    if (
        plan.get("manifestFormat") != 2
        or plan.get("recipeSha256") != build_digest
        or plan.get("microRecipeKind") != "measured-synthesis-pilot"
    ):
        raise ValueError("micro synthesis verifier: plan identity mismatch")

    expectation_blob = (build_root / "expectation.json").read_bytes()
    if hashlib.sha256(expectation_blob).hexdigest() != (
        build_root / "expectation.sha256"
    ).read_text().strip():
        raise ValueError("micro synthesis verifier: expectation digest mismatch")
    expectation = json.loads(expectation_blob)
    if (
        expectation.get("recipeSha256") != build_digest
        or expectation.get("recipeKind") != "measured-synthesis-pilot"
    ):
        raise ValueError("micro synthesis verifier: expectation identity mismatch")
    if expectation.get("recipeInputs", {}).get(
        "environment"
    ) != _synthesis_environment_identity(ASSET_GEN_ROOT):
        raise ValueError("micro synthesis verifier: numerical environment mismatch")

    coverage_doc = plan["microCoverage"]
    parent = HeightChunkId(*coverage_doc["parent"])
    coverage = plan_hero(parent.cx, parent.cz)
    expected_doc = {
        "parent": [coverage.parent.lod, coverage.parent.cx, coverage.parent.cz],
        "publishedFine": [[c.lod, c.cx, c.cz] for c in coverage.published_fine],
        "transientSupport": [[c.lod, c.cx, c.cz] for c in coverage.transient_support],
        "authority": [
            coverage.authority_lod0.lod,
            coverage.authority_lod0.cx,
            coverage.authority_lod0.cz,
        ],
    }
    if coverage_doc != expected_doc:
        raise ValueError("micro synthesis verifier: invalid coverage closure")

    base = load_base()
    micro = load_micro_config(base)
    reader = PinnedBaseHeight(
        base_manifest_path,
        micro.base_manifest_sha256,
        base_out_root,
        base.encode,
    )
    published = set(coverage.published_fine)
    all_fine = (*coverage.published_fine, *coverage.transient_support)
    first_e_u, first_n_u = chunk_origin_en_units(
        base.grid, coverage.published_fine[0]
    )
    first_e, first_n = first_e_u // 32, first_n_u // 32
    margin_cells = 4
    authority_grid = reader.read_cells(
        first_e - margin_cells,
        first_n + margin_cells,
        640 + 2 * margin_cells,
        640 + 2 * margin_cells,
    ).astype(np.float64)
    authority_east = (
        first_e - margin_cells
        + np.arange(authority_grid.shape[1], dtype=np.float64)
        + 0.5
    )
    authority_south = (
        base.grid.anchor_n
        - (first_n + margin_cells)
        + np.arange(authority_grid.shape[0], dtype=np.float64)
        + 0.5
    )
    base_spline = RectBivariateSpline(
        authority_south, authority_east, authority_grid, kx=3, ky=3, s=0.0
    )
    paths: dict[HeightChunkId, Path] = {}
    artifacts: dict[HeightChunkId, dict[str, Any]] = {}
    header_rows: list[dict[str, Any]] = []
    qoffsets: set[float] = set()
    for chunk in all_fine:
        root = build_root / ("chunks" if chunk in published else "transient")
        path = root / "height" / str(chunk.lod) / f"{chunk.cx}_{chunk.cz}.lac"
        meta, payload = read_chunk_v2(path)
        origin = chunk_origin_en_units(base.grid, chunk)
        if (
            meta.layer,
            meta.lod,
            meta.cx,
            meta.cz,
            meta.enc,
            meta.res,
            meta.count,
            meta.origin_e,
            meta.origin_n,
        ) != (
            "height",
            -2,
            chunk.cx,
            chunk.cz,
            1,
            2049,
            0,
            origin[0] / 32.0,
            origin[1] / 32.0,
        ):
            raise ValueError(f"micro synthesis verifier: invalid fine header {chunk}")
        if abs(meta.qscale - micro.selected_fine_qscale_m) > 1e-8:
            raise ValueError(f"micro synthesis verifier: invalid fine qscale {chunk}")
        decoded = decode_quant16(base.encode, payload, meta.res, meta.qoffset, meta.qscale)
        if not np.isfinite(decoded).all():
            raise ValueError(f"micro synthesis verifier: nonfinite fine chunk {chunk}")
        digest = _sha256_file(path)
        paths[chunk] = path
        qoffsets.add(meta.qoffset)
        artifacts[chunk] = {
            "key": [chunk.lod, chunk.cx, chunk.cz],
            "sha256": digest,
            "size": path.stat().st_size,
            "qoffset": meta.qoffset,
            "qscale": meta.qscale,
        }
        header_rows.append({"key": artifacts[chunk]["key"], "sha256": digest})
    if len(qoffsets) != 1:
        raise ValueError("micro synthesis verifier: fine closure lacks a shared qoffset")

    parent_path = (
        build_root / "chunks" / "height" / str(parent.lod) / f"{parent.cx}_{parent.cz}.lac"
    )
    parent_meta, parent_payload = read_chunk_v2(parent_path)
    parent_origin = chunk_origin_en_units(base.grid, parent)
    if (
        parent_meta.layer,
        parent_meta.lod,
        parent_meta.cx,
        parent_meta.cz,
        parent_meta.enc,
        parent_meta.res,
        parent_meta.count,
        parent_meta.origin_e,
        parent_meta.origin_n,
    ) != (
        "height",
        -1,
        parent.cx,
        parent.cz,
        1,
        2049,
        0,
        parent_origin[0] / 32.0,
        parent_origin[1] / 32.0,
    ):
        raise ValueError("micro synthesis verifier: invalid parent header")
    if abs(parent_meta.qscale - micro.parent_qscale_m) > 1e-8:
        raise ValueError("micro synthesis verifier: invalid parent qscale")

    def decoded(chunk: HeightChunkId) -> np.ndarray:
        meta, payload = read_chunk_v2(paths[chunk])
        return decode_quant16(base.encode, payload, meta.res, meta.qoffset, meta.qscale)

    mosaic = assemble_parent_source_memmap(
        build_root / "scratch" / "verify-synthesis-parent-source.f32",
        coverage,
        decoded,
    )
    derived_parent = box_mean4_striped(mosaic)
    expected_payload, expected_offset, expected_scale = encode_quant16_checked(
        base.encode, derived_parent, micro.parent_qscale_m
    )
    if (
        parent_payload != expected_payload
        or parent_meta.qoffset != expected_offset
        or parent_meta.qscale != expected_scale
    ):
        raise ValueError("micro synthesis verifier: parent is not derived from decoded children")

    max_seam = 0.0
    fine_cx0, fine_cz0 = parent.cx * 4, parent.cz * 4
    for dz in range(5):
        left = decoded(HeightChunkId(-2, fine_cx0, fine_cz0 + dz))
        for dx in range(1, 5):
            right = decoded(HeightChunkId(-2, fine_cx0 + dx, fine_cz0 + dz))
            max_seam = max(max_seam, float(np.max(np.abs(left[:, -1] - right[:, 0]))))
            left = right
    for dx in range(5):
        north = decoded(HeightChunkId(-2, fine_cx0 + dx, fine_cz0))
        for dz in range(1, 5):
            south = decoded(HeightChunkId(-2, fine_cx0 + dx, fine_cz0 + dz))
            max_seam = max(max_seam, float(np.max(np.abs(north[-1, :] - south[0, :]))))
            north = south
    if max_seam != 0.0:
        raise ValueError(f"micro synthesis verifier: decoded apron seam is {max_seam} m")

    sidecar_path = build_root / "evidence" / "micro-synthesis-cook.json"
    sidecar = json.loads(sidecar_path.read_bytes())
    if (
        sidecar.get("recipeSha256") != build_digest
        or sidecar.get("cook") != "exemplar-driven-production-pilot-v1"
        or sidecar.get("cookRevision") != plan.get("cookRev")
        or sidecar.get("seed") != micro.seed
        or sidecar.get("exemplarTopK") != micro.exemplar_top_k
        or sidecar.get("maskIntegration", {}).get("status")
        != "etak-soil-fail-closed-fine-v2"
        or sidecar.get("maskIntegration", {}).get("applied") is not True
        or sidecar.get("maskIntegration", {}).get(
            "maxAbsRejectedMorphologyResidualM"
        )
        != 0.0
    ):
        raise ValueError("micro synthesis verifier: invalid cook evidence identity")
    sidecar_children = {
        tuple(item.get("key", ())): item for item in sidecar.get("children", [])
    }
    expected_child_keys = {(item.lod, item.cx, item.cz) for item in all_fine}
    if set(sidecar_children) != expected_child_keys:
        raise ValueError("micro synthesis verifier: cook evidence child set mismatch")
    for chunk, actual in artifacts.items():
        claimed = sidecar_children[(chunk.lod, chunk.cx, chunk.cz)]
        if (
            claimed.get("sha256") != actual["sha256"]
            or claimed.get("size") != actual["size"]
            or claimed.get("qoffset") != actual["qoffset"]
            or claimed.get("qscale") != actual["qscale"]
        ):
            raise ValueError(f"micro synthesis verifier: cook evidence differs for {chunk}")
    parent_sha = _sha256_file(parent_path)
    if sidecar.get("parent", {}).get("sha256") != parent_sha:
        raise ValueError("micro synthesis verifier: cook evidence parent mismatch")
    dependencies = [(chunk, artifacts[chunk]["sha256"]) for chunk in all_fine]
    dependency_root = dependency_merkle_root(dependencies)
    if sidecar.get("dependencyMerkleRoot") != dependency_root:
        raise ValueError("micro synthesis verifier: dependency binding mismatch")
    cook_verification = sidecar.get("verification", {})
    claimed_round_trips = [
        (item.get("maxRoundTripErrorM"), item.get("roundTripLimitM"))
        for item in (*sidecar.get("children", []), sidecar.get("parent", {}))
    ]
    if any(
        not isinstance(error, (int, float))
        or not isinstance(limit, (int, float))
        or not np.isfinite(error)
        or not np.isfinite(limit)
        or error < 0
        or error > limit
        for error, limit in claimed_round_trips
    ):
        raise ValueError("micro synthesis verifier: invalid round-trip evidence")
    max_round_trip = max(float(error) for error, _ in claimed_round_trips)
    if (
        cook_verification.get("scope")
        != ["quantization-round-trip", "decoded-apron-seams"]
        or cook_verification.get("maxDecodedSeamErrorM") != max_seam
        or cook_verification.get("maxRoundTripErrorM") != max_round_trip
    ):
        raise ValueError("micro synthesis verifier: cook verification evidence mismatch")

    exemplar = sidecar.get("exemplarManifest", {})
    exemplar_path = Path(str(exemplar.get("path", "")))
    exemplar_sha = _sha256_file(exemplar_path)
    recipe_inputs = expectation.get("recipeInputs", {})
    if (
        exemplar.get("sha256") != exemplar_sha
        or recipe_inputs.get("exemplarManifestSha256") != exemplar_sha
    ):
        raise ValueError("micro synthesis verifier: exemplar manifest identity mismatch")
    exemplar_doc = json.loads(exemplar_path.read_bytes())
    if recipe_inputs.get("exemplarBankSha256") != exemplar_doc.get("bankSha256"):
        raise ValueError("micro synthesis verifier: exemplar bank identity mismatch")
    bank = load_exemplar_bank(exemplar_path)

    etak_sources = sorted((DATA_IN / "etak").glob("*.gpkg"))
    if len(etak_sources) != 1:
        raise ValueError("micro synthesis verifier: expected exactly one ETAK GeoPackage")
    if recipe_inputs.get("etakGpkgSha256") != _sha256_file(etak_sources[0]):
        raise ValueError("micro synthesis verifier: ETAK mask source identity mismatch")
    soil_sources = sorted((DATA_IN / "soil" / "mullakaart").glob("Mullakaart.*"))
    actual_soil_sources = {
        str(path.relative_to(ASSET_GEN_ROOT)): _sha256_file(path)
        for path in soil_sources
    }
    if recipe_inputs.get("soilSourceSha256") != actual_soil_sources:
        raise ValueError("micro synthesis verifier: soil source identity mismatch")
    source_bindings = sidecar.get("maskIntegration", {}).get("sourceBindings", {})
    if (
        source_bindings.get("etakGpkgSha256") != recipe_inputs.get("etakGpkgSha256")
        or source_bindings.get("soilSourceSha256") != actual_soil_sources
    ):
        raise ValueError("micro synthesis verifier: mask provenance differs from recipe")
    claimed_masks = {
        tuple(item.get("key", ())): item
        for item in sidecar.get("maskIntegration", {}).get("chunks", [])
    }
    if set(claimed_masks) != expected_child_keys:
        raise ValueError("micro synthesis verifier: morphology evidence child set mismatch")
    mask_totals = {name: 0 for name in (
        "cells", "allowedCells", "forestCells", "compatibleSoilCells",
        "unknownSoilCells", "unsupportedSoilContextCells", "nonForestCells",
        "drainedSoilCells", "slopeCliffContextCells", "waterCells",
        "buildingCells", "pavedRoadCells",
    )}
    max_authority_mean_error = 0.0
    authority_mean_limit = 0.0
    max_decoded_rejected_residual = 0.0
    decoded_rejected_samples = 0
    for chunk in all_fine:
        origin_e_u, origin_n_u = chunk_origin_en_units(base.grid, chunk)
        origin_e, origin_n = origin_e_u / 32.0, origin_n_u / 32.0
        texel = 0.0625
        east = origin_e + (np.arange(2064, dtype=np.float64) + 0.5) * texel
        north = origin_n - (np.arange(2064, dtype=np.float64) + 0.5) * texel
        mask = rasterize_micro_morphology_mask(east, north)
        actual_mask = mask.evidence()
        if claimed_masks[(chunk.lod, chunk.cx, chunk.cz)] != {
            **actual_mask,
            "key": [chunk.lod, chunk.cx, chunk.cz],
        }:
            raise ValueError(
                f"micro synthesis verifier: morphology evidence differs for {chunk}"
            )
        for name in mask_totals:
            mask_totals[name] += actual_mask[name]

        meta, payload = read_chunk_v2(paths[chunk])
        decoded_fine = decode_quant16(
            base.encode, payload, meta.res, meta.qoffset, meta.qscale
        )
        authority = reader.read_cells(
            int(origin_e), int(origin_n), 128, 128
        ).astype(np.float64)
        max_authority_mean_error = max(
            max_authority_mean_error,
            _decoded_authority_mean_error(decoded_fine, authority),
        )
        decoded_peak = np.float32(np.max(np.abs(decoded_fine)))
        authority_mean_limit = max(
            authority_mean_limit,
            meta.qscale * 0.5 + 2.0 * abs(float(np.spacing(decoded_peak))),
        )

        south = base.grid.anchor_n - north
        smooth = base_spline(south, east, grid=True)
        authority_apron = reader.read_cells(
            int(origin_e), int(origin_n), 129, 129
        )
        conservative = conservative_cell_correct(
            smooth, authority_apron, factor=16
        )[:2049, :2049]
        expected_payload, expected_offset, expected_scale = encode_quant16_checked(
            base.encode,
            conservative,
            meta.qscale,
            meta.qoffset,
        )
        expected_conservative = decode_quant16(
            base.encode,
            expected_payload,
            conservative.shape[0],
            expected_offset,
            expected_scale,
        )
        rejected_count, rejected_maximum = _decoded_rejected_residual(
            decoded_fine,
            expected_conservative,
            mask.allowed[:2049, :2049],
        )
        decoded_rejected_samples += rejected_count
        max_decoded_rejected_residual = max(
            max_decoded_rejected_residual, rejected_maximum
        )

    if max_authority_mean_error > authority_mean_limit:
        raise ValueError(
            "micro synthesis verifier: decoded fine hierarchy differs from LOD0 "
            f"authority by {max_authority_mean_error} m"
        )
    if max_decoded_rejected_residual != 0.0:
        raise ValueError(
            "micro synthesis verifier: decoded rejected residual is "
            f"{max_decoded_rejected_residual} m"
        )

    transient_artifacts = [
        {
            "key": artifacts[chunk]["key"],
            "sha256": artifacts[chunk]["sha256"],
            "size": artifacts[chunk]["size"],
        }
        for chunk in coverage.transient_support
    ]
    binding = micro_verification_binding(build_digest, work_root)
    gates = {
        "coverage": _gate({**expected_doc, "stagedCount": 17, "transientCount": 9}),
        "parentClosure": _gate({
            "dependencyMerkleRoot": dependency_root,
            "derivedParentSha256": parent_sha,
            "reducer": "decoded-f32-fixed-order-box-mean-16",
        }),
        "headers": _gate({"fine": header_rows, "parentSha256": parent_sha}),
        "decodedHierarchy": _gate({
            "maxLod0AuthorityMeanErrorM": max_authority_mean_error,
            "lod0AuthorityMeanLimitM": authority_mean_limit,
            "parentPayloadByteExactFromDecodedChildren": True,
        }),
        "aprons": _gate({"maxDecodedSeamErrorM": max_seam, "comparisons": 40}),
        "determinism": _gate({
            "recipeBoundGenerator": True,
            "seed": micro.seed,
            "exemplarManifestSha256": exemplar_sha,
            "exemplarBankSha256": exemplar_doc["bankSha256"],
            "bankPatches": bank.patches_m.shape[0],
        }),
        "hardMasks": _gate({
            "source": "recipe-bound ETAK GeoPackage and Mullakaart components",
            "analogueStatus": "foreign-analogue-low-confidence",
            "semantics": [
                "exact-etak-forest",
                "known-mineral-non-peat-productive-mesic-soil",
                "slope-cliff-context-excluded",
            ],
            "unpavedRoadsSuppressed": False,
            "decodedRejectedSamples": decoded_rejected_samples,
            "maxAbsDecodedRejectedResidualM": max_decoded_rejected_residual,
            "totals": mask_totals,
            "productionMorphologyAuthorized": False,
        }),
    }
    report = {
        **binding,
        "verifier": VERIFIER_ID,
        "verifierSourceSha256": verifier_source_sha256(),
        "pilotOnly": True,
        "gates": gates,
        "transientArtifacts": transient_artifacts,
        "transientSupportMerkleRoot": transient_merkle_root(transient_artifacts),
    }
    path = build_root / "micro-verify.json"
    blob = _json_bytes(report)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as output:
        output.write(blob)
        output.flush()
        import os

        os.fsync(output.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return path
