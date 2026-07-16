"""Pack the accepted Development-A ALS/TGV surface as a bounded research preview."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from .....config import DATA_OUT, DATA_WORK, BaseConfig, load_base
from .....cook.micro_hierarchy import assemble_parent_source_memmap, box_mean4_striped
from .....height_geom import HeightChunkId, plan_hero, plan_parent_set
from .....release import create_build_plan, materialize_corrected_format1_base, materialize_preview
from ....repair.base_transaction import (
    AuditedFormat1HeightSource,
    CorrectedLod0Core,
    build_corrected_base_transaction,
)
from ....repair.pinned_baseline import PinnedDecodedBaseline
from ....repair.prolong import prolong_structural_4x
from ...erodible_slope.morphodynamics.structural_base import (
    StructuralC0TransitionSupport,
    load_development_a_structural_transition_support,
)
from ...forest_exemplar.preview import _decoded, _immutable, _json_bytes, _sha256, _write_height
from .evidence import load_evidence
from .packed_preview_verify import VERIFIER_ID, verifier_source_sha256


RECIPE_KIND = "research-als-tgv-structural-preview-v1"
COOK_REVISION = 2
ARTIFACT_SCHEMA = "laas.coastal-escarpment-als-tgv-artifact/1"
CONFIG_SCHEMA = "laas.coastal-escarpment-als-tgv-config/1"
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
FINE_QSCALE = 0.002
PARENT_QSCALE = 0.005
QOFFSET = 30.0


def _bound_file(root: Path, row: dict[str, Any]) -> Path:
    path = root / row["path"]
    if path.stat().st_size != row["bytes"] or _sha256(path) != row["sha256"]:
        raise ValueError(f"ALS/TGV artifact output changed: {path}")
    return path


def _load_artifact(artifact_root: Path) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    manifest_path = artifact_root / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA
        or manifest.get("build_id") != artifact_root.name
        or manifest.get("state") != ARTIFACT_STATE
    ):
        raise ValueError("ALS/TGV artifact is not the accepted Development-A float candidate")
    config_row = manifest["config"]
    config_path = Path(__file__).parents[7] / config_row["path"]
    if config_path.stat().st_size != config_row["bytes"] or _sha256(config_path) != config_row["sha256"]:
        raise ValueError("ALS/TGV config identity changed")
    config = json.loads(config_path.read_bytes())
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or tuple(config.get("canvas_bbox_en", ())) != ARTIFACT_BBOX
        or config.get("solve_pitch_m") != SOURCE_PITCH_M
        or config.get("status") != "development_only"
    ):
        raise ValueError("ALS/TGV spatial/config contract changed")
    reconstruction = np.load(_bound_file(artifact_root, manifest["outputs"]["reconstruction"]), mmap_mode="r")
    residual = np.load(_bound_file(artifact_root, manifest["outputs"]["residual"]), mmap_mode="r")
    metrics_path = _bound_file(artifact_root, manifest["outputs"]["metrics"])
    _bound_file(artifact_root, manifest["outputs"]["qa_index"])
    metrics = json.loads(metrics_path.read_bytes())
    expected_shape = (769, 1281)
    if (
        reconstruction.shape != expected_shape
        or residual.shape != expected_shape
        or reconstruction.dtype != np.float32
        or residual.dtype != np.float32
        or not np.isfinite(reconstruction).all()
        or not np.isfinite(residual).all()
    ):
        raise ValueError("ALS/TGV float lattice contract changed")
    hard_laws = metrics.get("hard_laws", {})
    if (
        metrics.get("status") != ARTIFACT_STATE
        or metrics.get("holdout", {}).get("pass") is not True
        or hard_laws.get("pass") is not True
        or hard_laws.get("all_hard_residual_max_abs_m") != 0.0
        or hard_laws.get("mapped_face_residual_max_abs_m") != 0.0
        or hard_laws.get("outer_collar_residual_max_abs_m") != 0.0
    ):
        raise ValueError("ALS/TGV acceptance or hard-law evidence changed")
    evidence = load_evidence(config)
    hard = evidence.hard_zero | evidence.mapped_face
    if not np.array_equal(reconstruction, evidence.c0_m + residual):
        raise ValueError("ALS/TGV reconstruction no longer equals C0 plus residual")
    if np.any(residual[hard] != 0.0):
        raise ValueError("ALS/TGV residual changes hard or mapped-face authority")
    return manifest, config, evidence.c0_m, reconstruction, hard


def _recipe_identity(artifact_root: Path, source_base_manifest: Path) -> tuple[str, dict[str, Any]]:
    manifest, config, _c0, _c1, _hard = _load_artifact(artifact_root)
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("packed_preview_verify.py"),
        Path(__file__).with_name("evidence.py"),
        Path(__file__).parents[4] / "height_geom.py",
        Path(__file__).parents[4] / "cook/micro_hierarchy.py",
        Path(__file__).parents[3] / "repair/base_transaction.py",
        Path(__file__).parents[3] / "repair/pinned_baseline.py",
        Path(__file__).parents[3] / "repair/prolong.py",
        Path(__file__).parents[2] / "erodible_slope/morphodynamics/structural_base.py",
        Path(__file__).parents[4] / "release.py",
    )
    inputs: dict[str, Any] = {
        "id": "laas.micro.development-a-als-tgv-structural-preview.recipe.v1",
        "artifact": {
            "root": artifact_root.as_posix(),
            "manifestSha256": _sha256(artifact_root / "manifest.json"),
            "buildId": manifest["build_id"],
            "configSha256": manifest["config"]["sha256"],
            "outputs": manifest["outputs"],
        },
        "sourceBase": {
            "manifest": source_base_manifest.as_posix(),
            "manifestSha256": _sha256(source_base_manifest),
        },
        "coverage": {
            "artifactBboxEn": list(ARTIFACT_BBOX),
            "parents": [[p.lod, p.cx, p.cz] for p in PARENTS],
            "authority": [AUTHORITY.lod, AUTHORITY.cx, AUTHORITY.cz],
            "reason": "complete four-parent closure carries the nonzero artifact halo to its zero collar",
        },
        "binding": {
            "sourceGridConvention": "inclusive 0.25 m nodes",
            "packedAuthorityConvention": "0.25 m storage centers sampled bilinearly from source nodes",
            "hardOwnership": "any bilinear stencil touching hard or mapped-face authority selects C0",
            "c0Ownership": "accepted structural support with quintic delta closure to pinned baseline",
        },
        "finestRungSemantics": {
            "sourcePitchM": SOURCE_PITCH_M,
            "packedPitchM": 0.0625,
            "operator": "tensor-product Keys cubic a=-0.5 plus exact parent-cell mean bubble",
            "morphologyClaim": "none below 0.25 m",
            "previewAuthorization": "explicit bounded packing task; original float artifact remains non-production",
        },
        "configHardLaws": config["hard_laws"],
        "sourceSha256": {
            path.relative_to(Path(__file__).parents[5]).as_posix(): _sha256(path)
            for path in source_paths
        },
    }
    blob = _json_bytes(inputs)
    digest = hashlib.sha256(b"laas.micro.development-a-als-tgv-structural-preview.recipe.v1\0" + blob).hexdigest()
    return digest, inputs


def _sample_artifact_support(
    base: BaseConfig,
    chunk: HeightChunkId,
    c0: np.ndarray,
    c1: np.ndarray,
    hard: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    footprint = 128.0
    origin_e = base.grid.anchor_e + chunk.cx * footprint
    origin_n = base.grid.anchor_n - chunk.cz * footprint
    indices = np.arange(-4, 516, dtype=np.float64)
    east = origin_e + (indices + 0.5) * SOURCE_PITCH_M
    north = origin_n - (indices + 0.5) * SOURCE_PITCH_M
    cols = (east - ARTIFACT_BBOX[0]) / SOURCE_PITCH_M
    rows = (ARTIFACT_BBOX[3] - north) / SOURCE_PITCH_M
    valid_rows = (rows >= 0.0) & (rows <= c0.shape[0] - 1)
    valid_cols = (cols >= 0.0) & (cols <= c0.shape[1] - 1)
    sampled_c0 = np.zeros((520, 520), dtype=np.float64)
    sampled_c1 = np.zeros((520, 520), dtype=np.float64)
    conservative_hard = np.zeros((520, 520), dtype=bool)
    if valid_rows.any() and valid_cols.any():
        rr = rows[valid_rows]
        cc = cols[valid_cols]
        grid_r, grid_c = np.meshgrid(rr, cc, indexing="ij")
        target = np.ix_(valid_rows, valid_cols)
        sampled_c0[target] = ndimage.map_coordinates(c0, [grid_r, grid_c], order=1, mode="nearest")
        sampled_c1[target] = ndimage.map_coordinates(c1, [grid_r, grid_c], order=1, mode="nearest")
        r0 = np.floor(grid_r).astype(np.int64)
        r1 = np.ceil(grid_r).astype(np.int64)
        c0i = np.floor(grid_c).astype(np.int64)
        c1i = np.ceil(grid_c).astype(np.int64)
        conservative_hard[target] = hard[r0, c0i] | hard[r0, c1i] | hard[r1, c0i] | hard[r1, c1i]
    overlay = valid_rows[:, None] & valid_cols[None, :]
    return sampled_c0, sampled_c1, overlay & conservative_hard


def _smooth_axis_weight(values: np.ndarray, inner_min: float, inner_max: float, outer_min: float, outer_max: float) -> np.ndarray:
    distance = np.maximum(np.maximum(inner_min - values, values - inner_max), 0.0)
    clearance = np.where(values < inner_min, inner_min - outer_min, outer_max - inner_max)
    t = np.clip(distance / clearance, 0.0, 1.0)
    return 1.0 - (t * t * t * (t * (t * 6.0 - 15.0) + 10.0))


def _sample_c0_transition(
    base: BaseConfig,
    chunk: HeightChunkId,
    transition: StructuralC0TransitionSupport,
) -> tuple[np.ndarray, np.ndarray]:
    origin_e = base.grid.anchor_e + chunk.cx * 128.0
    origin_n = base.grid.anchor_n - chunk.cz * 128.0
    indices = np.arange(-4, 516, dtype=np.float64)
    east = origin_e + (indices + 0.5) * SOURCE_PITCH_M
    north = origin_n - (indices + 0.5) * SOURCE_PITCH_M
    e0, n0, e1, n1 = transition.bbox_en
    cols = (east - e0) / transition.pitch_m
    rows = (n1 - north) / transition.pitch_m
    valid_rows = (rows >= 0.0) & (rows <= transition.c0_height_m.shape[0] - 1)
    valid_cols = (cols >= 0.0) & (cols <= transition.c0_height_m.shape[1] - 1)
    sampled = np.zeros((520, 520), dtype=np.float64)
    if valid_rows.any() and valid_cols.any():
        rr, cc = np.meshgrid(rows[valid_rows], cols[valid_cols], indexing="ij")
        sampled[np.ix_(valid_rows, valid_cols)] = ndimage.map_coordinates(
            transition.c0_height_m,
            [rr, cc],
            order=1,
            mode="nearest",
        )
    weight = (
        _smooth_axis_weight(east, ARTIFACT_BBOX[0], ARTIFACT_BBOX[2], e0, e1)[None, :]
        * _smooth_axis_weight(north, ARTIFACT_BBOX[1], ARTIFACT_BBOX[3], n0, n1)[:, None]
    )
    valid = valid_rows[:, None] & valid_cols[None, :]
    weight[~valid] = 0.0
    return sampled, weight


def _stage_fine(
    base: BaseConfig,
    build_root: Path,
    source_base_manifest: Path,
    content_root: Path,
    c0: np.ndarray,
    c1: np.ndarray,
    hard: np.ndarray,
) -> tuple[dict[HeightChunkId, Path], list[dict[str, Any]], dict[str, str]]:
    coverage = plan_parent_set(PARENTS)
    baseline = PinnedDecodedBaseline(
        manifest_path=source_base_manifest,
        manifest_sha256=_sha256(source_base_manifest),
        content_root=content_root,
        encode=base.encode,
        cache_chunks=4,
    )
    scratch = build_root / "scratch/als-tgv-fine-cores"
    scratch.mkdir(parents=True, exist_ok=True)
    dependency_roots: dict[str, str] = {}
    transition = load_development_a_structural_transition_support()

    def core(chunk: HeightChunkId) -> np.ndarray:
        path = scratch / f"{chunk.cx}_{chunk.cz}.npy"
        reconstructed = baseline.reconstruct(chunk, halo_samples=4)
        dependency_roots[f"{chunk.cx},{chunk.cz}"] = reconstructed.dependency_root_sha256
        if path.exists():
            return np.load(path, mmap_mode="r")
        support = reconstructed.tile.height
        baseline_fine = prolong_structural_4x(support, parent_rows=(4, 516), parent_cols=(4, 516))
        sampled_c0, sampled_c1, support_hard = _sample_artifact_support(base, chunk, c0, c1, hard)
        transition_c0, transition_weight = _sample_c0_transition(base, chunk, transition)
        origin_e = base.grid.anchor_e + chunk.cx * 128.0
        origin_n = base.grid.anchor_n - chunk.cz * 128.0
        indices = np.arange(-4, 516, dtype=np.float64)
        east = origin_e + (indices + 0.5) * SOURCE_PITCH_M
        north = origin_n - (indices + 0.5) * SOURCE_PITCH_M
        overlay = (
            (north[:, None] >= ARTIFACT_BBOX[1])
            & (north[:, None] <= ARTIFACT_BBOX[3])
            & (east[None, :] >= ARTIFACT_BBOX[0])
            & (east[None, :] <= ARTIFACT_BBOX[2])
        )
        c0_support = np.asarray(support, dtype=np.float64) + transition_weight * (
            transition_c0 - np.asarray(support, dtype=np.float64)
        )
        c1_support = np.array(c0_support, copy=True)
        c0_support[overlay] = sampled_c0[overlay]
        c1_support[overlay] = np.where(support_hard[overlay], sampled_c0[overlay], sampled_c1[overlay])
        c0_fine = prolong_structural_4x(c0_support, parent_rows=(4, 516), parent_cols=(4, 516))
        c1_fine = prolong_structural_4x(c1_support, parent_rows=(4, 516), parent_cols=(4, 516))
        c0_overlay = transition_weight[4:516, 4:516] > 0.0
        core_overlay = np.repeat(np.repeat(c0_overlay, 4, axis=0), 4, axis=1)
        residual_overlay = np.repeat(np.repeat(overlay[4:516, 4:516], 4, axis=0), 4, axis=1)
        core_hard = np.repeat(np.repeat(support_hard[4:516, 4:516], 4, axis=0), 4, axis=1)
        values = np.array(baseline_fine, dtype=np.float32, copy=True)
        values[core_overlay] = c0_fine[core_overlay]
        values[residual_overlay] = c1_fine[residual_overlay]
        values[residual_overlay & core_hard] = c0_fine[residual_overlay & core_hard]
        temporary = path.with_suffix(".tmp.npy")
        np.save(temporary, values, allow_pickle=False)
        temporary.replace(path)
        return np.load(path, mmap_mode="r")

    published = set(coverage.published_fine)
    paths: dict[HeightChunkId, Path] = {}
    identities: list[dict[str, Any]] = []
    for chunk in (*coverage.published_fine, *coverage.transient_support):
        values = np.empty((FINE_CORE + 1, FINE_CORE + 1), dtype=np.float32)
        values[:-1, :-1] = core(chunk)
        values[:-1, -1] = core(HeightChunkId(-2, chunk.cx + 1, chunk.cz))[:, 0]
        values[-1, :-1] = core(HeightChunkId(-2, chunk.cx, chunk.cz + 1))[0, :]
        values[-1, -1] = core(HeightChunkId(-2, chunk.cx + 1, chunk.cz + 1))[0, 0]
        root = build_root / ("chunks" if chunk in published else "transient")
        destination = root / "height/-2" / f"{chunk.cx}_{chunk.cz}.lac"
        destination.parent.mkdir(parents=True, exist_ok=True)
        identity = _write_height(destination, base, chunk, values, FINE_QSCALE, qoffset=QOFFSET)
        identity["role"] = "als-tgv-structural-published" if chunk in published else "apron-support"
        paths[chunk] = destination
        identities.append(identity)
    return paths, identities, dependency_roots


def _stage_parents(base: BaseConfig, build_root: Path, paths: dict[HeightChunkId, Path]):
    identities: list[dict[str, Any]] = []
    decoded: dict[HeightChunkId, np.ndarray] = {}
    for parent in PARENTS:
        coverage = plan_hero(parent.cx, parent.cz)
        mosaic = assemble_parent_source_memmap(
            build_root / f"scratch/als-tgv-parent-{parent.cx}-{parent.cz}.f32",
            coverage,
            lambda chunk: _decoded(paths[chunk], base),
        )
        values = box_mean4_striped(mosaic)
        destination = build_root / "chunks/height/-1" / f"{parent.cx}_{parent.cz}.lac"
        destination.parent.mkdir(parents=True, exist_ok=True)
        identity = _write_height(destination, base, parent, values, PARENT_QSCALE, qoffset=QOFFSET)
        identities.append(identity)
        decoded[parent] = _decoded(destination, base)
    return identities, decoded


def _parent_overlay_mask(base: BaseConfig, parent: HeightChunkId) -> np.ndarray:
    east = base.grid.anchor_e + parent.cx * 512.0 + (np.arange(2048) + 0.5) * SOURCE_PITCH_M
    north = base.grid.anchor_n - parent.cz * 512.0 - (np.arange(2048) + 0.5) * SOURCE_PITCH_M
    return (
        (north[:, None] >= ARTIFACT_BBOX[1])
        & (north[:, None] <= ARTIFACT_BBOX[3])
        & (east[None, :] >= ARTIFACT_BBOX[0])
        & (east[None, :] <= ARTIFACT_BBOX[2])
    )


def _corrected_base(
    base: BaseConfig,
    build_root: Path,
    source_manifest: Path,
    parent_decoded: dict[HeightChunkId, np.ndarray],
    content_root: Path,
):
    source_sha = _sha256(source_manifest)
    source = AuditedFormat1HeightSource(
        manifest_path=source_manifest,
        manifest_sha256=source_sha,
        content_root=content_root,
        encode=base.encode,
        cache_chunks=4,
    )
    target = np.array(source.load(AUTHORITY).decoded[:-1, :-1], dtype=np.float64, copy=True)
    affected = np.zeros((2048, 2048), dtype=bool)
    for parent in PARENTS:
        reduced = box_mean4_striped(parent_decoded[parent][:2048, :2048])
        parent_mask = _parent_overlay_mask(base, parent).reshape(512, 4, 512, 4).any(axis=(1, 3))
        x0 = (parent.cx - AUTHORITY.cx * 4) * 512
        z0 = (parent.cz - AUTHORITY.cz * 4) * 512
        window = np.s_[z0 : z0 + 512, x0 : x0 + 512]
        target[window] = reduced
        affected[window] = parent_mask
    transaction = build_corrected_base_transaction(
        source=source,
        staging_root=build_root / "corrected-base",
        corrected_lod0={AUTHORITY: CorrectedLod0Core(target, affected)},
        grid=base.grid,
        encode=base.encode,
    )
    replacements = {
        ("height", row.chunk.lod, row.chunk.cx, row.chunk.cz): build_root / "corrected-base" / row.relative_path
        for row in transaction.plan.artifacts
    }
    release = materialize_corrected_format1_base(
        base=base,
        recipe_sha256=build_root.name,
        pinned_manifest_path=source_manifest,
        pinned_manifest_sha256=source_sha,
        content_root=content_root,
        release_root=build_root / "corrected-format1-release",
        replacements=replacements,
    )
    return transaction, release


def materialize_als_tgv_preview(
    *,
    artifact_root: Path,
    source_base_manifest: Path,
    content_root: Path = DATA_OUT,
    work_root: Path = DATA_WORK,
) -> Path:
    base = load_base()
    artifact_root = artifact_root.resolve()
    source_base_manifest = source_base_manifest.resolve()
    build_digest, inputs = _recipe_identity(artifact_root, source_base_manifest)
    build_root = work_root / "builds" / build_digest
    build_root.mkdir(parents=True, exist_ok=True)
    _manifest, _config, c0, c1, hard = _load_artifact(artifact_root)
    paths, fine_identities, dependency_roots = _stage_fine(
        base, build_root, source_base_manifest, content_root, c0, c1, hard
    )
    parent_identities, parent_decoded = _stage_parents(base, build_root, paths)
    transaction, corrected_release = _corrected_base(
        base, build_root, source_base_manifest, parent_decoded, content_root
    )
    coverage = plan_parent_set(PARENTS)
    inputs["correctedBase"] = {
        "manifest": corrected_release.manifest_path.as_posix(),
        "manifestSha256": corrected_release.manifest_sha256,
        "verificationSha256": corrected_release.verification_sha256,
        "hierarchyPlanSha256": transaction.plan_sha256,
    }
    expectation = {
        "format": 1,
        "recipeKind": RECIPE_KIND,
        "recipeSha256": build_digest,
        "baseManifestSha256": corrected_release.manifest_sha256,
        "recipeInputs": inputs,
        "grid": {
            "anchorE": base.grid.anchor_e,
            "anchorN": base.grid.anchor_n,
            "chunkMeters": base.grid.chunk_m,
            "chunkRes": base.grid.chunk_res,
            "lodStep": base.grid.lod_step,
        },
        "sites": [{"siteId": "development-a-adjacent-escarpment", "reviewBboxEn": list(ARTIFACT_BBOX)}],
        "parents": [[c.lod, c.cx, c.cz] for c in coverage.parents],
        "publishedFine": [[c.lod, c.cx, c.cz] for c in coverage.published_fine],
        "transientSupport": [[c.lod, c.cx, c.cz] for c in coverage.transient_support],
        "authorities": [[c.lod, c.cx, c.cz] for c in coverage.authorities_lod0],
        "expectedPublished": [["height", c.lod, c.cx, c.cz] for c in (*coverage.published_fine, *coverage.parents)],
        "verifier": {"id": VERIFIER_ID, "sourceSha256": verifier_source_sha256()},
    }
    expectation_blob = _json_bytes(expectation)
    _immutable(build_root / "expectation.json", expectation_blob)
    _immutable(build_root / "expectation.sha256", (hashlib.sha256(expectation_blob).hexdigest() + "\n").encode())
    evidence = {
        "format": 1,
        "cook": "development-a-als-tgv-structural-preview-v1",
        "cookRevision": COOK_REVISION,
        "recipeSha256": build_digest,
        "artifactManifestSha256": inputs["artifact"]["manifestSha256"],
        "children": fine_identities,
        "parents": parent_identities,
        "dependencyRoots": dependency_roots,
        "fineMorphologyClaim": "none below 0.25 m",
        "correctedBase": inputs["correctedBase"],
    }
    _immutable(build_root / "evidence/als-tgv-structural-preview.json", _json_bytes(evidence))
    create_build_plan(
        base,
        build_digest,
        COOK_REVISION,
        work_root=work_root,
        base_manifest_path=corrected_release.manifest_path,
        base_manifest_sha256=corrected_release.manifest_sha256,
        base_out_root=content_root,
        manifest_format=2,
        micro_parents=tuple((parent.cx, parent.cz) for parent in coverage.parents),
    )
    return materialize_preview(build_digest, work_root=work_root, out_root=content_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--source-base-manifest", type=Path, required=True)
    parser.add_argument("--content-root", type=Path, default=DATA_OUT)
    parser.add_argument("--work-root", type=Path, default=DATA_WORK)
    print(materialize_als_tgv_preview(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
