"""Pack the accepted disjoint forest masters as one immutable format-2 preview."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline

from ....config import DATA_OUT, DATA_WORK, BaseConfig, load_base
from ....cook.micro_hierarchy import assemble_parent_source_memmap, box_mean4_striped
from ....cook.pinned_height import PinnedBaseHeight
from ....height_geom import (
    HeightChunkId,
    chunk_origin_en_units,
    plan_hero,
    plan_parent_set,
)
from ....release import create_build_plan, materialize_corrected_format1_base, materialize_preview
from ....process.micro_fixture import conservative_cell_correct
from ...repair.base_transaction import (
    AuditedFormat1HeightSource,
    CorrectedLod0Core,
    build_corrected_base_transaction,
)
from .generalization_preview_verify import VERIFIER_ID, verifier_source_sha256
from .preview import _decoded, _immutable, _json_bytes, _sha256, _write_height


RECIPE_KIND = "research-microtopography-generalization-preview-v1"
COOK_REVISION = 2
ARTIFACT_SCHEMA = "forest-mesic-mineral-multi-site-float-artifact/1"
FINE_QSCALE = 0.002
PARENT_QSCALE = 0.005
FINE_CORE = 2048
FACTOR = 16


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
    SiteSpec(
        "southeast-southwest-retained",
        HeightChunkId(-1, 601, 384),
        HeightChunkId(0, 150, 96),
        (676352, 6438400, 676864, 6438912),
    ),
    SiteSpec(
        "southeast-northcentral-retained",
        HeightChunkId(-1, 606, 361),
        HeightChunkId(0, 151, 90),
        (678912, 6450176, 679424, 6450688),
    ),
    SiteSpec(
        "southeast-east-retained",
        HeightChunkId(-1, 617, 378),
        HeightChunkId(0, 154, 94),
        (684544, 6441472, 685056, 6441984),
    ),
)


def _artifact_files(artifact_root: Path) -> dict[str, str]:
    manifest = json.loads((artifact_root / "manifest.json").read_bytes())
    if (
        manifest.get("schema") != ARTIFACT_SCHEMA
        or manifest.get("build_id") != artifact_root.name
        or manifest.get("status") != "research_candidate_pass"
    ):
        raise ValueError("forest generalization artifact is not the frozen passing candidate")
    metrics_by_site = manifest.get("site_metrics", {})
    for site in SITES:
        metrics = metrics_by_site.get(site.site_id)
        if metrics is None or tuple(metrics["site"]["bbox_en"]) != site.bbox_en:
            raise ValueError(f"forest generalization artifact lacks {site.site_id}")
    for relative, identity in manifest["files"].items():
        path = artifact_root / relative
        if path.stat().st_size != identity["bytes"] or _sha256(path) != identity["sha256"]:
            raise ValueError(f"forest generalization artifact changed: {relative}")
    return {relative: row["sha256"] for relative, row in manifest["files"].items()}


def _recipe_identity(
    artifact_root: Path,
    source_base_manifest: Path,
) -> tuple[str, dict[str, object]]:
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("generalization_preview_verify.py"),
        Path(__file__).with_name("preview.py"),
        Path(__file__).parents[3] / "height_geom.py",
        Path(__file__).parents[3] / "cook/chunkio.py",
        Path(__file__).parents[3] / "cook/encode.py",
        Path(__file__).parents[3] / "cook/micro_hierarchy.py",
        Path(__file__).parents[3] / "cook/pinned_height.py",
        Path(__file__).parents[2] / "repair/base_transaction.py",
        Path(__file__).parents[3] / "release.py",
    )
    inputs: dict[str, object] = {
        "id": "laas.micro.forest-generalization-multi-parent-preview.recipe.v2",
        "artifact": {
            "root": artifact_root.as_posix(),
            "manifestSha256": _sha256(artifact_root / "manifest.json"),
            "recipeSha256": _sha256(artifact_root / "recipe.json"),
            "files": _artifact_files(artifact_root),
            "siteIds": [site.site_id for site in SITES],
        },
        "sourceBase": {
            "manifest": source_base_manifest.as_posix(),
            "manifestSha256": _sha256(source_base_manifest),
        },
        "coverage": {"sites": [site.json() for site in SITES]},
        "sourceSha256": {
            path.relative_to(Path(__file__).parents[4]).as_posix(): _sha256(path)
            for path in source_paths
        },
    }
    blob = _json_bytes(inputs)
    return hashlib.sha256(
        b"laas.micro.forest-generalization-multi-parent-preview.recipe.v2\0" + blob
    ).hexdigest(), inputs


def _baseline_core(source: PinnedBaseHeight, chunk: HeightChunkId) -> np.ndarray:
    origin_e_u, origin_n_u = chunk_origin_en_units(source.grid, chunk)
    e_min = origin_e_u // 32
    n_max = origin_n_u // 32
    extended = source.read_cells(e_min - 4, n_max + 4, 136, 136).astype(np.float64)
    authority = extended[4:-4, 4:-4]
    east_centers = e_min - 4 + np.arange(136, dtype=np.float64) + 0.5
    north_centers = n_max + 4 - np.arange(136, dtype=np.float64) - 0.5
    spline = RectBivariateSpline(
        north_centers[::-1], east_centers, extended[::-1], kx=3, ky=3, s=0.0
    )
    east = e_min + (np.arange(FINE_CORE, dtype=np.float64) + 0.5) / FACTOR
    north = n_max - (np.arange(FINE_CORE, dtype=np.float64) + 0.5) / FACTOR
    smooth = spline(north[::-1], east, grid=True)[::-1]
    return conservative_cell_correct(smooth, authority, FACTOR).astype(np.float32)


def _stage_fine(
    base: BaseConfig,
    build_root: Path,
    artifact_root: Path,
    source_base_manifest: Path,
) -> tuple[dict[HeightChunkId, Path], list[dict[str, object]]]:
    coverage = plan_parent_set(tuple(site.parent for site in SITES))
    masters: dict[HeightChunkId, np.ndarray] = {}
    for site in SITES:
        c1 = np.load(
            artifact_root / f"sites/{site.site_id}/surface/c1_height_f32.npy",
            mmap_mode="r",
        )
        if c1.shape != (8192, 8192) or c1.dtype != np.float32:
            raise ValueError(f"{site.site_id} master must be float32 8192 square")
        masters[site.parent] = c1
    source_sha = _sha256(source_base_manifest)
    pinned = PinnedBaseHeight(source_base_manifest, source_sha, DATA_OUT, base.encode)
    scratch = build_root / "scratch/fine-cores"
    scratch.mkdir(parents=True, exist_ok=True)

    def core_path(chunk: HeightChunkId) -> Path:
        return scratch / f"{chunk.cx}_{chunk.cz}.npy"

    def core(chunk: HeightChunkId) -> np.ndarray:
        for parent, master in masters.items():
            dx, dz = chunk.cx - parent.cx * 4, chunk.cz - parent.cz * 4
            if 0 <= dx < 4 and 0 <= dz < 4:
                return master[
                    dz * FINE_CORE : (dz + 1) * FINE_CORE,
                    dx * FINE_CORE : (dx + 1) * FINE_CORE,
                ]
        path = core_path(chunk)
        if not path.exists():
            values = _baseline_core(pinned, chunk)
            temporary = path.with_suffix(".tmp.npy")
            np.save(temporary, values, allow_pickle=False)
            temporary.replace(path)
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
        identity["role"] = "accepted-master" if chunk in published else "base-apron-support"
        paths[chunk] = destination
        identities.append(identity)
    return paths, identities


def _stage_parents(
    base: BaseConfig,
    build_root: Path,
    paths: dict[HeightChunkId, Path],
) -> tuple[list[dict[str, object]], dict[HeightChunkId, np.ndarray]]:
    identities: list[dict[str, object]] = []
    decoded: dict[HeightChunkId, np.ndarray] = {}
    for site in SITES:
        coverage = plan_hero(site.parent.cx, site.parent.cz)
        mosaic = assemble_parent_source_memmap(
            build_root / f"scratch/forest-generalization-parent-{site.parent.cx}-{site.parent.cz}.f32",
            coverage,
            lambda chunk: _decoded(paths[chunk], base),
        )
        values = box_mean4_striped(mosaic)
        destination = build_root / "chunks/height/-1" / f"{site.parent.cx}_{site.parent.cz}.lac"
        destination.parent.mkdir(parents=True, exist_ok=True)
        identity = _write_height(destination, base, site.parent, values, PARENT_QSCALE)
        identity["siteId"] = site.site_id
        identities.append(identity)
        decoded[site.parent] = _decoded(destination, base)
    return identities, decoded


def _corrected_base(
    base: BaseConfig,
    build_root: Path,
    source_manifest: Path,
    parent_decoded: dict[HeightChunkId, np.ndarray],
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
    targets: dict[HeightChunkId, np.ndarray] = {}
    masks: dict[HeightChunkId, np.ndarray] = {}
    for site in SITES:
        target = targets.setdefault(
            site.authority,
            np.array(source.load(site.authority).decoded[:-1, :-1], dtype=np.float64, copy=True),
        )
        affected = masks.setdefault(site.authority, np.zeros((2048, 2048), dtype=bool))
        parent_lod0 = box_mean4_striped(parent_decoded[site.parent][:2048, :2048])
        packed = np.load(
            artifact_root / f"sites/{site.site_id}/surface/allowed_packbits_u8.npy",
            mmap_mode="r",
        )
        allowed = np.unpackbits(packed, axis=1, bitorder="little")[:, :8192].astype(bool)
        coarse_mask = allowed.reshape(512, 16, 512, 16).any(axis=(1, 3))
        origin_e = base.grid.anchor_e + site.authority.cx * base.grid.chunk_m
        origin_n = base.grid.anchor_n - site.authority.cz * base.grid.chunk_m
        x0 = site.bbox_en[0] - origin_e
        y0 = origin_n - site.bbox_en[3]
        window = np.s_[y0 : y0 + 512, x0 : x0 + 512]
        if affected[window].any():
            raise ValueError(f"corrected LOD0 ownership overlaps at {site.site_id}")
        target[window] = parent_lod0
        affected[window] = coarse_mask
    corrected = {
        authority: CorrectedLod0Core(targets[authority], masks[authority])
        for authority in sorted(targets)
    }
    transaction = build_corrected_base_transaction(
        source=source,
        staging_root=build_root / "corrected-base",
        corrected_lod0=corrected,
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


def materialize_generalization_preview(
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
    paths, fine_identities = _stage_fine(base, build_root, artifact_root, source_base_manifest)
    parent_identities, parent_decoded = _stage_parents(base, build_root, paths)
    corrected_transaction, corrected_release = _corrected_base(
        base, build_root, source_base_manifest, parent_decoded, artifact_root, content_root
    )
    coverage = plan_parent_set(tuple(site.parent for site in SITES))
    inputs["correctedBase"] = {
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
        "recipeInputs": inputs,
        "grid": {
            "anchorE": base.grid.anchor_e,
            "anchorN": base.grid.anchor_n,
            "chunkMeters": base.grid.chunk_m,
            "chunkRes": base.grid.chunk_res,
            "lodStep": base.grid.lod_step,
        },
        "sites": [site.json() for site in SITES],
        "parents": [[c.lod, c.cx, c.cz] for c in coverage.parents],
        "publishedFine": [[c.lod, c.cx, c.cz] for c in coverage.published_fine],
        "transientSupport": [[c.lod, c.cx, c.cz] for c in coverage.transient_support],
        "authorities": [[c.lod, c.cx, c.cz] for c in coverage.authorities_lod0],
        "expectedPublished": [
            ["height", c.lod, c.cx, c.cz]
            for c in (*coverage.published_fine, *coverage.parents)
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
        "cook": "accepted-forest-generalization-multi-parent-absolute-master-v2",
        "cookRevision": COOK_REVISION,
        "recipeSha256": build_digest,
        "artifactManifestSha256": _sha256(artifact_root / "manifest.json"),
        "sites": [site.json() for site in SITES],
        "children": fine_identities,
        "parents": parent_identities,
        "changedFine": [[c.lod, c.cx, c.cz] for c in coverage.published_fine],
        "correctedBase": inputs["correctedBase"],
    }
    _immutable(
        build_root / "evidence/forest-generalization-preview.json", _json_bytes(evidence)
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
        micro_parents=tuple((parent.cx, parent.cz) for parent in coverage.parents),
    )
    return materialize_preview(build_digest, work_root=work_root, out_root=content_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--source-base-manifest", type=Path, required=True)
    parser.add_argument("--content-root", type=Path, default=DATA_OUT)
    parser.add_argument("--work-root", type=Path, default=DATA_WORK)
    print(materialize_generalization_preview(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
