"""Pack the accepted rough-till float surface as a structural-only preview."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ....config import DATA_OUT, DATA_WORK, BaseConfig, load_base
from ....cook.micro_hierarchy import assemble_parent_source_memmap, box_mean4_striped
from ....height_geom import HeightChunkId, plan_hero
from ....release import create_build_plan, materialize_corrected_format1_base, materialize_preview
from ...repair.base_transaction import (
    AuditedFormat1HeightSource,
    CorrectedLod0Core,
    build_corrected_base_transaction,
)
from ...repair.pinned_baseline import PinnedDecodedBaseline
from ...repair.prolong import prolong_structural_4x
from ..forest_exemplar.preview import _decoded, _immutable, _json_bytes, _sha256, _write_height
from .packed_preview_verify import VERIFIER_ID, verifier_source_sha256


RECIPE_KIND = "research-rough-till-structural-preview-v1"
COOK_REVISION = 1
ARTIFACT_SCHEMA = "laas.rough-till-boulder-multi-form/1"
PARENT = HeightChunkId(-1, 615, 389)
AUTHORITY = HeightChunkId(0, 153, 97)
REVIEW_BBOX = (683520, 6435840, 684032, 6436352)
FINE_QSCALE = 0.002
PARENT_QSCALE = 0.005
PARENT_C1_RES = 2048
FINE_CORE = 2048


def _artifact_identity(artifact_root: Path) -> dict[str, object]:
    manifest_path = artifact_root / "manifest.json"
    recipe_path = artifact_root / "recipe.json"
    preview_path = artifact_root / "preview.npz"
    manifest = json.loads(manifest_path.read_bytes())
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA
        or manifest.get("build_sha256") != artifact_root.name
        or manifest.get("decision") != "inspect_float_preview"
        or manifest.get("authority") != {
            "browser": False,
            "estonia_transfer_truth": False,
            "packing": False,
            "production": False,
        }
    ):
        raise ValueError("rough-till artifact is not the accepted bounded float candidate")
    if manifest["metrics"].get("forbidden_nonzero_cells") != 0:
        raise ValueError("rough-till artifact no longer preserves hard exclusions")
    return {
        "root": artifact_root.as_posix(),
        "manifestSha256": _sha256(manifest_path),
        "recipeSha256": _sha256(recipe_path),
        "previewSha256": _sha256(preview_path),
        "previewBytes": preview_path.stat().st_size,
        "sourceBuildSha256": artifact_root.name,
    }


def _recipe_identity(
    artifact_root: Path, source_base_manifest: Path
) -> tuple[str, dict[str, object]]:
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("packed_preview_verify.py"),
        Path(__file__).parents[3] / "height_geom.py",
        Path(__file__).parents[3] / "cook/chunkio.py",
        Path(__file__).parents[3] / "cook/encode.py",
        Path(__file__).parents[3] / "cook/micro_hierarchy.py",
        Path(__file__).parents[2] / "repair/base_transaction.py",
        Path(__file__).parents[2] / "repair/pinned_baseline.py",
        Path(__file__).parents[2] / "repair/prolong.py",
        Path(__file__).parents[3] / "release.py",
    )
    inputs: dict[str, object] = {
        "id": "laas.micro.rough-till-structural-preview.recipe.v1",
        "artifact": _artifact_identity(artifact_root),
        "sourceBase": {
            "manifest": source_base_manifest.as_posix(),
            "manifestSha256": _sha256(source_base_manifest),
        },
        "coverage": {
            "reviewBboxEn": list(REVIEW_BBOX),
            "parent": [PARENT.lod, PARENT.cx, PARENT.cz],
            "authority": [AUTHORITY.lod, AUTHORITY.cx, AUTHORITY.cz],
        },
        "finestRungSemantics": {
            "kind": "structural-only-finest-reconstruction",
            "sourcePitchM": 0.25,
            "packedPitchM": 0.0625,
            "operator": "tensor-product Keys cubic a=-0.5 plus exact parent-cell mean bubble",
            "morphologyClaim": "none below 0.25 m",
        },
        "sourceSha256": {
            path.relative_to(Path(__file__).parents[4]).as_posix(): _sha256(path)
            for path in source_paths
        },
    }
    blob = _json_bytes(inputs)
    return hashlib.sha256(
        b"laas.micro.rough-till-structural-preview.recipe.v1\0" + blob
    ).hexdigest(), inputs


def _artifact_arrays(artifact_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.load(artifact_root / "preview.npz", allow_pickle=False)
    before = source["before_m"]
    after = source["after_m"]
    allowed = source["allowed"].astype(bool)
    expected = (PARENT_C1_RES, PARENT_C1_RES)
    if before.shape != expected or after.shape != expected or allowed.shape != expected:
        raise ValueError("rough-till artifact arrays must be 2048 square at 0.25 m")
    if before.dtype != np.float32 or after.dtype != np.float32:
        raise ValueError("rough-till height arrays must be float32")
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("rough-till height arrays contain nonfinite values")
    if np.any(after[~allowed] != before[~allowed]):
        raise ValueError("rough-till artifact changes a forbidden 0.25 m cell")
    return before, after, allowed


def _stage_fine(
    base: BaseConfig,
    build_root: Path,
    artifact_root: Path,
    source_base_manifest: Path,
    content_root: Path,
) -> tuple[dict[HeightChunkId, Path], list[dict[str, object]], dict[str, object]]:
    coverage = plan_hero(PARENT.cx, PARENT.cz)
    _before, after, allowed = _artifact_arrays(artifact_root)
    source_sha = _sha256(source_base_manifest)
    baseline = PinnedDecodedBaseline(
        manifest_path=source_base_manifest,
        manifest_sha256=source_sha,
        content_root=content_root,
        encode=base.encode,
        cache_chunks=4,
    )
    scratch = build_root / "scratch/rough-till-fine-cores"
    scratch.mkdir(parents=True, exist_ok=True)
    dependency_roots: dict[str, str] = {}

    def core_path(chunk: HeightChunkId) -> Path:
        return scratch / f"{chunk.cx}_{chunk.cz}.npy"

    def core(chunk: HeightChunkId) -> np.ndarray:
        path = core_path(chunk)
        if path.exists():
            return np.load(path, mmap_mode="r")
        reconstructed = baseline.reconstruct(chunk, halo_samples=4)
        support = reconstructed.tile.height
        baseline_fine = prolong_structural_4x(
            support, parent_rows=(4, 516), parent_cols=(4, 516)
        )
        authority_support = np.array(support, dtype=np.float64, copy=True)
        fine_allowed = np.zeros((FINE_CORE, FINE_CORE), dtype=bool)
        dx, dz = chunk.cx - PARENT.cx * 4, chunk.cz - PARENT.cz * 4
        global_rows = dz * 512 + np.arange(-4, 516, dtype=np.int64)
        global_cols = dx * 512 + np.arange(-4, 516, dtype=np.int64)
        valid_rows = (global_rows >= 0) & (global_rows < PARENT_C1_RES)
        valid_cols = (global_cols >= 0) & (global_cols < PARENT_C1_RES)
        if valid_rows.any() and valid_cols.any():
            rr = global_rows[valid_rows]
            cc = global_cols[valid_cols]
            selected_allowed = allowed[np.ix_(rr, cc)]
            selected_after = after[np.ix_(rr, cc)]
            block = authority_support[np.ix_(valid_rows, valid_cols)]
            authority_support[np.ix_(valid_rows, valid_cols)] = np.where(
                selected_allowed, selected_after, block
            )
        if 0 <= dx < 4 and 0 <= dz < 4:
            core_allowed = allowed[
                dz * 512 : (dz + 1) * 512,
                dx * 512 : (dx + 1) * 512,
            ]
            fine_allowed = np.repeat(np.repeat(core_allowed, 4, axis=0), 4, axis=1)
        altered_fine = prolong_structural_4x(
            authority_support, parent_rows=(4, 516), parent_cols=(4, 516)
        )
        values = np.where(fine_allowed, altered_fine, baseline_fine).astype(np.float32)
        temporary = path.with_suffix(".tmp.npy")
        np.save(temporary, values, allow_pickle=False)
        temporary.replace(path)
        dependency_roots[f"{chunk.cx},{chunk.cz}"] = reconstructed.dependency_root_sha256
        return np.load(path, mmap_mode="r")

    published = set(coverage.published_fine)
    paths: dict[HeightChunkId, Path] = {}
    identities: list[dict[str, object]] = []
    for chunk in (*coverage.published_fine, *coverage.transient_support):
        values = np.empty((FINE_CORE + 1, FINE_CORE + 1), dtype=np.float32)
        values[:-1, :-1] = core(chunk)
        values[:-1, -1] = core(HeightChunkId(-2, chunk.cx + 1, chunk.cz))[:, 0]
        values[-1, :-1] = core(HeightChunkId(-2, chunk.cx, chunk.cz + 1))[0, :]
        values[-1, -1] = core(HeightChunkId(-2, chunk.cx + 1, chunk.cz + 1))[0, 0]
        root = build_root / ("chunks" if chunk in published else "transient")
        destination = root / "height/-2" / f"{chunk.cx}_{chunk.cz}.lac"
        destination.parent.mkdir(parents=True, exist_ok=True)
        identity = _write_height(
            destination, base, chunk, values, FINE_QSCALE, qoffset=30.0
        )
        identity["role"] = (
            "structural-only-published" if chunk in published else "structural-base-apron-support"
        )
        paths[chunk] = destination
        identities.append(identity)
    return paths, identities, {
        "reconstruction": "pinned-decoded-keys-bubble-4x then Keys-bubble 4x",
        "dependencyRoots": dependency_roots,
        "fineMorphologyClaim": "none below 0.25 m",
    }


def _stage_parent(
    base: BaseConfig, build_root: Path, paths: dict[HeightChunkId, Path]
) -> tuple[dict[str, object], np.ndarray]:
    coverage = plan_hero(PARENT.cx, PARENT.cz)
    mosaic = assemble_parent_source_memmap(
        build_root / "scratch/rough-till-parent-source.f32",
        coverage,
        lambda chunk: _decoded(paths[chunk], base),
    )
    values = box_mean4_striped(mosaic)
    destination = build_root / "chunks/height/-1" / f"{PARENT.cx}_{PARENT.cz}.lac"
    destination.parent.mkdir(parents=True, exist_ok=True)
    identity = _write_height(destination, base, PARENT, values, PARENT_QSCALE)
    return identity, _decoded(destination, base)


def _corrected_base(
    base: BaseConfig,
    build_root: Path,
    source_manifest: Path,
    parent_decoded: np.ndarray,
    artifact_root: Path,
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
    reduced = box_mean4_striped(parent_decoded[:2048, :2048])
    _before, _after, allowed = _artifact_arrays(artifact_root)
    coarse_mask = allowed.reshape(512, 4, 512, 4).any(axis=(1, 3))
    window = np.s_[512:1024, 1536:2048]
    target[window] = reduced
    affected = np.zeros((2048, 2048), dtype=bool)
    affected[window] = coarse_mask
    transaction = build_corrected_base_transaction(
        source=source,
        staging_root=build_root / "corrected-base",
        corrected_lod0={AUTHORITY: CorrectedLod0Core(target, affected)},
        grid=base.grid,
        encode=base.encode,
    )
    replacements = {
        ("height", row.chunk.lod, row.chunk.cx, row.chunk.cz):
        build_root / "corrected-base" / row.relative_path
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


def materialize_rough_till_preview(
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
    paths, fine_identities, reconstruction = _stage_fine(
        base, build_root, artifact_root, source_base_manifest, content_root
    )
    parent_identity, parent_decoded = _stage_parent(base, build_root, paths)
    transaction, corrected_release = _corrected_base(
        base, build_root, source_base_manifest, parent_decoded, artifact_root, content_root
    )
    coverage = plan_hero(PARENT.cx, PARENT.cz)
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
        "parent": [PARENT.lod, PARENT.cx, PARENT.cz],
        "publishedFine": [[c.lod, c.cx, c.cz] for c in coverage.published_fine],
        "transientSupport": [[c.lod, c.cx, c.cz] for c in coverage.transient_support],
        "authority": [AUTHORITY.lod, AUTHORITY.cx, AUTHORITY.cz],
        "expectedPublished": [
            ["height", c.lod, c.cx, c.cz]
            for c in (*coverage.published_fine, coverage.parent)
        ],
        "verifier": {"id": VERIFIER_ID, "sourceSha256": verifier_source_sha256()},
    }
    expectation_blob = _json_bytes(expectation)
    _immutable(build_root / "expectation.json", expectation_blob)
    _immutable(
        build_root / "expectation.sha256",
        (hashlib.sha256(expectation_blob).hexdigest() + "\n").encode(),
    )
    evidence = {
        "format": 1,
        "cook": "rough-till-structural-only-finest-preview-v1",
        "cookRevision": COOK_REVISION,
        "recipeSha256": build_digest,
        "artifactManifestSha256": inputs["artifact"]["manifestSha256"],
        "children": fine_identities,
        "parent": parent_identity,
        "reconstruction": reconstruction,
        "correctedBase": inputs["correctedBase"],
    }
    _immutable(build_root / "evidence/rough-till-preview.json", _json_bytes(evidence))
    create_build_plan(
        base,
        build_digest,
        COOK_REVISION,
        work_root=work_root,
        base_manifest_path=corrected_release.manifest_path,
        base_manifest_sha256=corrected_release.manifest_sha256,
        base_out_root=content_root,
        manifest_format=2,
        micro_parent=(PARENT.cx, PARENT.cz),
    )
    return materialize_preview(build_digest, work_root=work_root, out_root=content_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--source-base-manifest", type=Path, required=True)
    parser.add_argument("--content-root", type=Path, default=DATA_OUT)
    parser.add_argument("--work-root", type=Path, default=DATA_WORK)
    print(materialize_rough_till_preview(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
