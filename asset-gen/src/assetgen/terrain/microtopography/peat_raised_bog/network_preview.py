"""Pack the accepted peat-raised-bog self-organized-network v4 float as a
research-only, immutable, non-latest format-2 preview.

The bog v4 float (gates all pass, orchestrator-accepted) carves a 128 m core over
the reselected pool-bearing mire etak-component-0004071135. The core straddles the
2x2 LOD -2 block whose LOD -1 parent is (-1, 335, 402) and whose LOD0 authority is
(0, 83, 100). The ETAK Laugas (bog pools) carry a monotone non-positive #104 bed-depth
wedge (v7 pool-depth carve) so the inherited waterY plane floats above a real carved bed;
the synthesized string/hollow relief lands only on the land between them.
Packing mirrors the forest single-exemplar preview: baseline fine cores are derived
from the pinned bog base, the bog relief is added inside the core (world-coordinate
placement keeps decoded seams exact), the LOD -1 parent is box-mean derived from the
decoded children, and the LOD0 authority + coarser ancestors are re-spliced through a
corrected format-1 base so LOD0 closure holds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ....config import ASSET_GEN_ROOT, DATA_OUT, DATA_WORK, BaseConfig, load_base
from ....cook.micro_hierarchy import assemble_parent_source_memmap, box_mean4_striped
from ....cook.pinned_height import PinnedBaseHeight
from ....height_geom import HeightChunkId, plan_hero
from ....release import (
    create_build_plan,
    materialize_corrected_format1_base,
    materialize_preview,
)
from ...repair.base_transaction import (
    AuditedFormat1HeightSource,
    CorrectedLod0Core,
    build_corrected_base_transaction,
)
from ..forest_exemplar.generalization_preview import FINE_CORE, _baseline_core
from ..forest_exemplar.preview import (
    _decoded,
    _immutable,
    _json_bytes,
    _sha256,
    _write_height,
)
from .network_preview_verify import VERIFIER_ID, verifier_source_sha256


RECIPE_KIND = "research-peat-bog-network-preview-v1"
RECIPE_ID = "laas.micro.peat-bog-network-preview.recipe.v1"
COOK_REVISION = 3
ARTIFACT_SCHEMA = "laas.peat-raised-bog-r0-research-bundle-preregistration/4"
PARENT = HeightChunkId(-1, 335, 402)
AUTHORITY = HeightChunkId(0, 83, 100)
CORE_BBOX = (540224.0, 6429504.0, 540352.0, 6429632.0)
CORE_M = 128.0
# The relief is synthesized natively at the LOD -2 finest pitch (0.0625 m == FINE_CORE over the
# 128 m core), so it is placed 1:1 into the fine core. (It was 0.25 m and 4x nearest-upsampled,
# which stamped flat 0.25 m terraces into the 6 cm fine rung.)
RELIEF_PITCH_M = 0.0625
FINE_QSCALE = 0.002
PARENT_QSCALE = 0.005
# A single shared quant offset across all fine chunks so adjacent chunks land on the
# same quantization grid and apron seams decode bit-identically (exact-zero seam gate).
# The pool-bearing mire sits ~57 m; offset 0 with qscale 0.002 spans 0..131 m, no overflow.
FINE_QOFFSET = 0.0
# LOD0 authority (0,83,100) 1 m window covering the 128 m core.
AUTH_WINDOW = (slice(1088, 1216), slice(1600, 1728))
# LOD -1 parent (-1,335,402) 1 m (box-mean-of-fine) window covering the core.
PARENT_WINDOW = (slice(64, 192), slice(64, 192))


def _artifact_root() -> Path:
    return (
        DATA_WORK
        / "microtopography/peat-bog-network-v4/sha256"
        / "b29a72d8faf0ef72163e0c9a6440991f6a560e1eb410b21cfa32a7f4c51c7979"
    )


DEFAULT_FLOAT_NAME = "network-v4-float.npz"


def _load_relief(artifact_root: Path, float_name: str = DEFAULT_FLOAT_NAME) -> np.ndarray:
    npz = np.load(artifact_root / float_name)
    relief = np.asarray(npz["core_relief_00625m"], dtype=np.float64)
    if relief.shape != (FINE_CORE, FINE_CORE):
        raise ValueError(f"bog core relief must be {FINE_CORE} square 0.0625 m, got {relief.shape}")
    return relief


def _recipe_identity(
    artifact_root: Path, source_base_manifest: Path, float_name: str = DEFAULT_FLOAT_NAME
) -> tuple[str, dict[str, Any]]:
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("network_preview_verify.py"),
        ASSET_GEN_ROOT / "src/assetgen/terrain/microtopography/forest_exemplar/generalization_preview.py",
        ASSET_GEN_ROOT / "src/assetgen/terrain/microtopography/forest_exemplar/preview.py",
        ASSET_GEN_ROOT / "src/assetgen/cook/chunkio.py",
        ASSET_GEN_ROOT / "src/assetgen/cook/encode.py",
        ASSET_GEN_ROOT / "src/assetgen/cook/micro_hierarchy.py",
        ASSET_GEN_ROOT / "src/assetgen/cook/pinned_height.py",
        ASSET_GEN_ROOT / "src/assetgen/terrain/repair/base_transaction.py",
        ASSET_GEN_ROOT / "src/assetgen/release.py",
    )
    inputs: dict[str, Any] = {
        "id": RECIPE_ID,
        "artifact": {
            "root": artifact_root.as_posix(),
            "schema": ARTIFACT_SCHEMA,
            "buildId": artifact_root.name,
            "floatName": float_name,
            "measurementsSha256": _sha256(artifact_root / "measurements.json"),
            "floatSha256": _sha256(artifact_root / float_name),
        },
        "sourceBase": {
            "manifest": source_base_manifest.as_posix(),
            "manifestSha256": _sha256(source_base_manifest),
        },
        "coverage": {
            "coreBboxEn": list(CORE_BBOX),
            "parent": [PARENT.lod, PARENT.cx, PARENT.cz],
            "authority": [AUTHORITY.lod, AUTHORITY.cx, AUTHORITY.cz],
        },
        "sourceSha256": {
            path.relative_to(ASSET_GEN_ROOT).as_posix(): _sha256(path)
            for path in source_paths
        },
    }
    blob = _json_bytes(inputs)
    return hashlib.sha256((RECIPE_ID + "\0").encode() + blob).hexdigest(), inputs


def _core_grid(base: BaseConfig, chunk: HeightChunkId) -> tuple[np.ndarray, np.ndarray]:
    """World E (per column) and N (per row) centers of a LOD -2 chunk at 0.0625 m."""
    texel = base.grid.chunk_m * (base.grid.lod_step ** chunk.lod) / FINE_CORE
    origin_e = base.grid.anchor_e + chunk.cx * base.grid.chunk_m * (base.grid.lod_step ** chunk.lod)
    origin_n = base.grid.anchor_n - chunk.cz * base.grid.chunk_m * (base.grid.lod_step ** chunk.lod)
    east = origin_e + (np.arange(FINE_CORE) + 0.5) * texel
    north = origin_n - (np.arange(FINE_CORE) + 0.5) * texel
    return east, north


def _carved_core(
    base: BaseConfig,
    pinned: PinnedBaseHeight,
    relief: np.ndarray,
    chunk: HeightChunkId,
) -> tuple[np.ndarray, np.ndarray]:
    """Baseline LOD -2 core (2048^2) plus the bog relief inside the 128 m core.

    Returns (values, applied_mask); relief is placed by WORLD coordinate so the
    shared value at any chunk boundary is identical from either side.
    """
    values = np.asarray(_baseline_core(pinned, chunk), dtype=np.float64)
    east, north = _core_grid(base, chunk)
    e0, n0, e1, n1 = CORE_BBOX
    col_in = (east >= e0) & (east < e1)
    row_in = (north >= n0) & (north < n1)
    mask = np.zeros((FINE_CORE, FINE_CORE), dtype=bool)
    if col_in.any() and row_in.any():
        rr = np.nonzero(row_in)[0]
        cc = np.nonzero(col_in)[0]
        # The relief is native 0.0625 m over CORE_BBOX and world-aligned to the fine-core lattice,
        # so each in-core fine texel maps to its OWN relief texel (1:1) -- NOT a 4x4 block. The
        # in-core rows/cols are contiguous runs; recover the world-aligned relief index of the
        # first in-core texel and take the matching contiguous relief slice.
        rel_r0 = int(round((n1 - north[rr[0]]) / RELIEF_PITCH_M - 0.5))
        rel_c0 = int(round((east[cc[0]] - e0) / RELIEF_PITCH_M - 0.5))
        block = relief[rel_r0:rel_r0 + rr.size, rel_c0:rel_c0 + cc.size]
        if block.shape != (rr.size, cc.size):
            raise ValueError("relief core does not cover the in-core fine texels 1:1")
        grid = np.ix_(rr, cc)
        values[grid] = values[grid] + block
        mask[grid] = True
    return values.astype(np.float32), mask


def _stage_fine(
    base: BaseConfig,
    build_root: Path,
    pinned: PinnedBaseHeight,
    relief: np.ndarray,
) -> tuple[dict[HeightChunkId, Path], list[dict[str, Any]], dict[HeightChunkId, np.ndarray]]:
    coverage = plan_hero(PARENT.cx, PARENT.cz)
    cores: dict[HeightChunkId, np.ndarray] = {}
    masks: dict[HeightChunkId, np.ndarray] = {}

    def core(chunk: HeightChunkId) -> np.ndarray:
        if chunk not in cores:
            values, mask = _carved_core(base, pinned, relief, chunk)
            cores[chunk] = values
            masks[chunk] = mask
        return cores[chunk]

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
        identity = _write_height(destination, base, chunk, values, FINE_QSCALE, qoffset=FINE_QOFFSET)
        identity["role"] = "carved-core" if masks.get(chunk, np.zeros(1)).any() else "baseline"
        paths[chunk] = destination
        identities.append(identity)
    return paths, identities, masks


def _stage_parent(
    base: BaseConfig,
    build_root: Path,
    paths: dict[HeightChunkId, Path],
) -> tuple[Path, dict[str, Any], np.ndarray]:
    coverage = plan_hero(PARENT.cx, PARENT.cz)
    mosaic = assemble_parent_source_memmap(
        build_root / "scratch/bog-parent-source.f32",
        coverage,
        lambda chunk: _decoded(paths[chunk], base),
    )
    parent_values = box_mean4_striped(mosaic)
    destination = build_root / "chunks/height/-1" / f"{PARENT.cx}_{PARENT.cz}.lac"
    destination.parent.mkdir(parents=True, exist_ok=True)
    identity = _write_height(destination, base, PARENT, parent_values, PARENT_QSCALE)
    return destination, identity, _decoded(destination, base)


def _corrected_base(
    base: BaseConfig,
    build_root: Path,
    source_manifest: Path,
    source_manifest_sha256: str,
    parent_decoded: np.ndarray,
    content_root: Path,
):
    source = AuditedFormat1HeightSource(
        manifest_path=source_manifest,
        manifest_sha256=source_manifest_sha256,
        content_root=content_root,
        encode=base.encode,
        cache_chunks=4,
    )
    inherited = source.load(AUTHORITY).decoded[:-1, :-1]
    target = np.array(inherited, dtype=np.float64, copy=True)
    parent_lod0 = box_mean4_striped(parent_decoded[:2048, :2048])
    target[AUTH_WINDOW] = parent_lod0[PARENT_WINDOW]
    affected = np.zeros((2048, 2048), dtype=bool)
    affected[AUTH_WINDOW] = True
    transaction = build_corrected_base_transaction(
        source=source,
        staging_root=build_root / "corrected-base",
        corrected_lod0={AUTHORITY: CorrectedLod0Core(target, affected)},
        grid=base.grid,
        encode=base.encode,
    )
    replacements = {
        ("height", artifact.chunk.lod, artifact.chunk.cx, artifact.chunk.cz):
        build_root / "corrected-base" / artifact.relative_path
        for artifact in transaction.plan.artifacts
    }
    release = materialize_corrected_format1_base(
        base=base,
        recipe_sha256=build_root.name,
        pinned_manifest_path=source_manifest,
        pinned_manifest_sha256=source_manifest_sha256,
        content_root=content_root,
        release_root=build_root / "corrected-format1-release",
        replacements=replacements,
    )
    return transaction, release


def materialize_bog_network_preview(
    *,
    source_base_manifest: Path,
    artifact_root: Path | None = None,
    float_name: str = DEFAULT_FLOAT_NAME,
    content_root: Path = DATA_OUT,
    work_root: Path = DATA_WORK,
) -> Path:
    base = load_base()
    artifact_root = (artifact_root or _artifact_root()).resolve()
    source_base_manifest = source_base_manifest.resolve()
    source_base_sha = _sha256(source_base_manifest)
    relief = _load_relief(artifact_root, float_name)
    build_digest, recipe_inputs = _recipe_identity(artifact_root, source_base_manifest, float_name)
    build_root = work_root / "builds" / build_digest
    build_root.mkdir(parents=True, exist_ok=True)
    print(f"[bog-network-pack] recipe {build_digest}", flush=True)

    pinned = PinnedBaseHeight(source_base_manifest, source_base_sha, content_root, base.encode)
    paths, fine_identities, masks = _stage_fine(base, build_root, pinned, relief)
    parent_path, parent_identity, parent_decoded = _stage_parent(base, build_root, paths)
    corrected_transaction, corrected_release = _corrected_base(
        base, build_root, source_base_manifest, source_base_sha, parent_decoded, content_root
    )
    coverage = plan_hero(PARENT.cx, PARENT.cz)
    recipe_inputs["correctedBase"] = {
        "manifest": corrected_release.manifest_path.as_posix(),
        "manifestSha256": corrected_release.manifest_sha256,
        "verificationSha256": corrected_release.verification_sha256,
        "hierarchyPlanSha256": corrected_transaction.plan_sha256,
    }
    changed_fine = sorted(
        [c.lod, c.cx, c.cz] for c, m in masks.items() if c in set(coverage.published_fine) and m.any()
    )
    expectation = {
        "format": 1,
        "recipeKind": RECIPE_KIND,
        "recipeSha256": build_digest,
        "baseManifestSha256": corrected_release.manifest_sha256,
        "recipeInputs": recipe_inputs,
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
        "cook": "accepted-peat-bog-network-v4-preview-v1",
        "cookRevision": COOK_REVISION,
        "recipeSha256": build_digest,
        "artifactBuildId": artifact_root.name,
        "authority": {
            "research_only": True,
            "target_truth": False,
            "production_owner": None,
            "estonia_transfer": "none",
            "latest_eligible": False,
        },
        "children": fine_identities,
        "parent": parent_identity,
        "changedFine": changed_fine,
        "correctedBase": recipe_inputs["correctedBase"],
    }
    _immutable(build_root / "evidence/peat-bog-network-preview.json", _json_bytes(evidence))
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


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-base-manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--float-name", type=str, default=DEFAULT_FLOAT_NAME)
    parser.add_argument("--content-root", type=Path, default=DATA_OUT)
    parser.add_argument("--work-root", type=Path, default=DATA_WORK)
    args = parser.parse_args()
    print(materialize_bog_network_preview(
        source_base_manifest=args.source_base_manifest,
        artifact_root=args.artifact_root,
        float_name=args.float_name,
        content_root=args.content_root,
        work_root=args.work_root,
    ))


if __name__ == "__main__":
    _main()
