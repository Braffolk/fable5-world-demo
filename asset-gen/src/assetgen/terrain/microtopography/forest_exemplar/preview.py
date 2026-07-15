"""Pack an accepted irregular-forest surface as an immutable format-2 preview."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from ....config import ASSET_GEN_ROOT, DATA_OUT, DATA_WORK, BaseConfig, load_base
from ....cook.chunkio import ChunkMeta, read_chunk_v2, write_chunk_v2
from ....cook.encode import decode_quant16, encode_quant16_checked
from ....cook.micro_hierarchy import assemble_parent_source_memmap, box_mean4_striped
from ....height_geom import HeightChunkId, chunk_origin_en_units, plan_hero
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
from .preview_verify import VERIFIER_ID, verifier_source_sha256


RECIPE_KIND = "research-microtopography-preview-v1"
COOK_REVISION = 1
PARENT = HeightChunkId(-1, 607, 372)
AUTHORITY = HeightChunkId(0, 151, 93)
REVIEW_BBOX = (679616.0, 6444736.0, 679744.0, 6444864.0)
FINE_QSCALE = 0.002
PARENT_QSCALE = 0.005


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable artifact differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _decoded(path: Path, base: BaseConfig) -> np.ndarray:
    meta, payload = read_chunk_v2(path)
    return decode_quant16(base.encode, payload, meta.res, meta.qoffset, meta.qscale)


def _write_height(
    path: Path,
    base: BaseConfig,
    chunk: HeightChunkId,
    values: np.ndarray,
    qscale: float,
    qoffset: float | None = None,
) -> dict[str, object]:
    payload, qoffset, wire_qscale = encode_quant16_checked(
        base.encode, values, qscale, qoffset
    )
    decoded = decode_quant16(base.encode, payload, values.shape[0], qoffset, wire_qscale)
    origin_e_u, origin_n_u = chunk_origin_en_units(base.grid, chunk)
    meta = ChunkMeta(
        layer="height",
        lod=chunk.lod,
        enc=1,
        cx=chunk.cx,
        cz=chunk.cz,
        res=values.shape[0],
        count=0,
        origin_e=origin_e_u / 32.0,
        origin_n=origin_n_u / 32.0,
        qoffset=qoffset,
        qscale=wire_qscale,
    )
    write_chunk_v2(path, meta, payload)
    return {
        "key": [chunk.lod, chunk.cx, chunk.cz],
        "path": path.as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "qoffset": qoffset,
        "qscale": wire_qscale,
        "maxRoundTripErrorM": float(
            np.max(np.abs(decoded.astype(np.float64) - np.asarray(values, dtype=np.float64)))
        ),
    }


def _artifact_files(artifact_root: Path) -> dict[str, str]:
    manifest = json.loads((artifact_root / "manifest.json").read_bytes())
    if (
        manifest.get("schema") != "estonia-forest-irregular-support-artifact/1"
        or manifest.get("build_id") != artifact_root.name
    ):
        raise ValueError("irregular forest artifact identity is invalid")
    for relative, identity in manifest["files"].items():
        path = artifact_root / relative
        if path.stat().st_size != identity["bytes"] or _sha256(path) != identity["sha256"]:
            raise ValueError(f"irregular forest artifact file changed: {relative}")
    return {relative: value["sha256"] for relative, value in manifest["files"].items()}


def _recipe_identity(
    artifact_root: Path,
    transaction_root: Path,
    source_base_manifest: Path,
) -> tuple[str, dict[str, object]]:
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("preview_verify.py"),
        ASSET_GEN_ROOT / "src/assetgen/cook/chunkio.py",
        ASSET_GEN_ROOT / "src/assetgen/cook/encode.py",
        ASSET_GEN_ROOT / "src/assetgen/cook/micro_hierarchy.py",
        ASSET_GEN_ROOT / "src/assetgen/terrain/repair/base_transaction.py",
        ASSET_GEN_ROOT / "src/assetgen/release.py",
    )
    inputs: dict[str, object] = {
        "id": "laas.micro.forest-irregular-preview.recipe.v1",
        "artifact": {
            "root": artifact_root.as_posix(),
            "manifestSha256": _sha256(artifact_root / "manifest.json"),
            "recipeSha256": _sha256(artifact_root / "recipe.json"),
            "files": _artifact_files(artifact_root),
        },
        "sourceTransaction": {
            "root": transaction_root.as_posix(),
            "transactionSha256": _sha256(transaction_root / "transaction.json"),
            "materializationSha256": _sha256(
                transaction_root / "materialization/transaction.json"
            ),
        },
        "sourceBase": {
            "manifest": source_base_manifest.as_posix(),
            "manifestSha256": _sha256(source_base_manifest),
        },
        "coverage": {
            "reviewBboxEn": list(REVIEW_BBOX),
            "parent": [PARENT.lod, PARENT.cx, PARENT.cz],
            "authority": [AUTHORITY.lod, AUTHORITY.cx, AUTHORITY.cz],
        },
        "sourceSha256": {
            path.relative_to(ASSET_GEN_ROOT).as_posix(): _sha256(path)
            for path in source_paths
        },
    }
    blob = _json_bytes(inputs)
    return hashlib.sha256(b"laas.micro.forest-irregular-preview.recipe.v1\0" + blob).hexdigest(), inputs


def _baseline_path(transaction_root: Path, chunk: HeightChunkId) -> Path:
    return (
        transaction_root
        / "materialization/chunks/height"
        / str(chunk.lod)
        / f"{chunk.cx}_{chunk.cz}.lac2"
    )


def _apply_artifact(
    base: BaseConfig,
    transaction_root: Path,
    artifact_root: Path,
) -> dict[HeightChunkId, np.ndarray]:
    c1 = np.load(artifact_root / "surface/c1_height_f32.npy", mmap_mode="r")
    allowed = np.load(artifact_root / "surface/allowed_u8.npy", mmap_mode="r")
    if c1.shape != (2048, 2048) or allowed.shape != c1.shape:
        raise ValueError("irregular forest review arrays must be 2048 square")
    result: dict[HeightChunkId, np.ndarray] = {}
    for cz, artifact_row in ((1489, 0), (1490, 1024)):
        for cx, artifact_col in ((2429, 0), (2430, 1024)):
            chunk = HeightChunkId(-2, cx, cz)
            values = np.array(_decoded(_baseline_path(transaction_root, chunk), base), copy=True)
            row = 1024 if cz == 1489 else 0
            col = 1024 if cx == 2429 else 0
            source = np.s_[artifact_row : artifact_row + 1024, artifact_col : artifact_col + 1024]
            target = np.s_[row : row + 1024, col : col + 1024]
            authority = np.asarray(allowed[source], dtype=bool)
            values[target] = np.where(authority, c1[source], values[target])
            result[chunk] = values

    # LAC2 stores one east/south neighbor sample. Rebind those aprons after all
    # four authority cores have been overlaid so decoded seams remain exact.
    for chunk, values in result.items():
        east = HeightChunkId(-2, chunk.cx + 1, chunk.cz)
        south = HeightChunkId(-2, chunk.cx, chunk.cz + 1)
        southeast = HeightChunkId(-2, chunk.cx + 1, chunk.cz + 1)
        east_values = result.get(east)
        south_values = result.get(south)
        southeast_values = result.get(southeast)
        if east_values is None:
            east_values = _decoded(_baseline_path(transaction_root, east), base)
        if south_values is None:
            south_values = _decoded(_baseline_path(transaction_root, south), base)
        if southeast_values is None:
            southeast_values = _decoded(_baseline_path(transaction_root, southeast), base)
        values[:-1, -1] = east_values[:-1, 0]
        values[-1, :-1] = south_values[0, :-1]
        values[-1, -1] = southeast_values[0, 0]
    return result


def _stage_fine(
    base: BaseConfig,
    build_root: Path,
    transaction_root: Path,
    artifact_root: Path,
) -> tuple[dict[HeightChunkId, Path], list[dict[str, object]]]:
    coverage = plan_hero(PARENT.cx, PARENT.cz)
    changed = _apply_artifact(base, transaction_root, artifact_root)
    paths: dict[HeightChunkId, Path] = {}
    identities: list[dict[str, object]] = []
    for chunk in (*coverage.published_fine, *coverage.transient_support):
        root = build_root / ("chunks" if chunk in coverage.published_fine else "transient")
        destination = root / "height" / str(chunk.lod) / f"{chunk.cx}_{chunk.cz}.lac"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if chunk in changed:
            identity = _write_height(
                destination, base, chunk, changed[chunk], FINE_QSCALE, qoffset=30.0
            )
        else:
            source = _baseline_path(transaction_root, chunk)
            if destination.exists():
                if _sha256(destination) != _sha256(source):
                    raise ValueError(f"immutable staged fine chunk differs: {destination}")
            else:
                shutil.copyfile(source, destination)
            meta, _ = read_chunk_v2(destination)
            identity = {
                "key": [chunk.lod, chunk.cx, chunk.cz],
                "path": destination.as_posix(),
                "sha256": _sha256(destination),
                "bytes": destination.stat().st_size,
                "qoffset": meta.qoffset,
                "qscale": meta.qscale,
                "maxRoundTripErrorM": 0.0,
            }
        paths[chunk] = destination
        identities.append(identity)
    return paths, identities


def _stage_parent(
    base: BaseConfig,
    build_root: Path,
    paths: dict[HeightChunkId, Path],
) -> tuple[Path, dict[str, object], np.ndarray]:
    coverage = plan_hero(PARENT.cx, PARENT.cz)
    mosaic = assemble_parent_source_memmap(
        build_root / "scratch/forest-parent-source.f32",
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
    artifact_root: Path,
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
    allowed = np.load(artifact_root / "surface/allowed_u8.npy", mmap_mode="r").astype(bool)
    coarse_mask = allowed.reshape(128, 16, 128, 16).any(axis=(1, 3))
    target_window = np.s_[192:320, 1728:1856]
    parent_window = np.s_[192:320, 192:320]
    target[target_window] = parent_lod0[parent_window]
    affected = np.zeros((2048, 2048), dtype=bool)
    affected[target_window] = coarse_mask
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


def materialize_irregular_forest_preview(
    *,
    artifact_root: Path,
    transaction_root: Path,
    source_base_manifest: Path,
    content_root: Path = DATA_OUT,
    work_root: Path = DATA_WORK,
) -> Path:
    base = load_base()
    artifact_root = artifact_root.resolve()
    transaction_root = transaction_root.resolve()
    source_base_manifest = source_base_manifest.resolve()
    build_digest, recipe_inputs = _recipe_identity(
        artifact_root, transaction_root, source_base_manifest
    )
    build_root = work_root / "builds" / build_digest
    build_root.mkdir(parents=True, exist_ok=True)
    paths, fine_identities = _stage_fine(
        base, build_root, transaction_root, artifact_root
    )
    _parent_path, parent_identity, parent_decoded = _stage_parent(base, build_root, paths)
    source_base_sha = _sha256(source_base_manifest)
    corrected_transaction, corrected_release = _corrected_base(
        base,
        build_root,
        source_base_manifest,
        source_base_sha,
        parent_decoded,
        artifact_root,
        content_root,
    )
    coverage = plan_hero(PARENT.cx, PARENT.cz)
    recipe_inputs["correctedBase"] = {
        "manifest": corrected_release.manifest_path.as_posix(),
        "manifestSha256": corrected_release.manifest_sha256,
        "verificationSha256": corrected_release.verification_sha256,
        "hierarchyPlanSha256": corrected_transaction.plan_sha256,
    }
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
        "cook": "accepted-irregular-forest-absolute-overlay-v1",
        "cookRevision": COOK_REVISION,
        "recipeSha256": build_digest,
        "artifactManifestSha256": _sha256(artifact_root / "manifest.json"),
        "sourceMaterializationSha256": _sha256(
            transaction_root / "materialization/transaction.json"
        ),
        "children": fine_identities,
        "parent": parent_identity,
        "changedFine": [[-2, cx, cz] for cz in (1489, 1490) for cx in (2429, 2430)],
        "correctedBase": recipe_inputs["correctedBase"],
    }
    _immutable(
        build_root / "evidence/forest-irregular-preview.json", _json_bytes(evidence)
    )
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
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--transaction-root", type=Path, required=True)
    parser.add_argument("--source-base-manifest", type=Path, required=True)
    parser.add_argument("--content-root", type=Path, default=DATA_OUT)
    parser.add_argument("--work-root", type=Path, default=DATA_WORK)
    args = parser.parse_args()
    print(materialize_irregular_forest_preview(**vars(args)))


if __name__ == "__main__":
    _main()
