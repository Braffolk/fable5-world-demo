"""Pack an accepted contiguous forest master as a format-2 preview."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ....config import ASSET_GEN_ROOT, DATA_OUT, DATA_WORK, BaseConfig, load_base
from ....cook.micro_hierarchy import assemble_parent_source_memmap, box_mean4_striped
from ....cook.pinned_height import PinnedBaseHeight
from ....height_geom import HeightChunkId, plan_hero, plan_parent_set
from ....process.micro_masks import rasterize_micro_morphology_mask
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
from .generalization_preview import (
    FINE_CORE,
    FINE_QSCALE,
    PARENT_QSCALE,
    RECIPE_KIND,
    SiteSpec,
    _baseline_core,
)
from .generalization_preview_verify import VERIFIER_ID, verifier_source_sha256
from .preview import _decoded, _immutable, _json_bytes, _sha256, _write_height


COOK_REVISION = 4
TEXEL_M = 0.0625


@dataclass(frozen=True)
class ArtifactLayout:
    schema: str
    master_shape: tuple[int, int]
    master_bbox: tuple[int, int, int, int]
    master_fine_origin: tuple[int, int]
    sites: tuple[SiteSpec, ...]
    recipe_id: str
    cook_id: str


PAIR_SITES = (
    SiteSpec(
        "east-adjacent-north",
        HeightChunkId(-1, 617, 377),
        HeightChunkId(0, 154, 94),
        (684544, 6441984, 685056, 6442496),
    ),
    SiteSpec(
        "east-adjacent-south",
        HeightChunkId(-1, 617, 378),
        HeightChunkId(0, 154, 94),
        (684544, 6441472, 685056, 6441984),
    ),
)

BLOCK_SITES = (
    SiteSpec(
        "east-adjacent-northwest",
        HeightChunkId(-1, 617, 377),
        HeightChunkId(0, 154, 94),
        (684544, 6441984, 685056, 6442496),
    ),
    SiteSpec(
        "east-adjacent-northeast",
        HeightChunkId(-1, 618, 377),
        HeightChunkId(0, 154, 94),
        (685056, 6441984, 685568, 6442496),
    ),
    SiteSpec(
        "east-adjacent-southwest",
        HeightChunkId(-1, 617, 378),
        HeightChunkId(0, 154, 94),
        (684544, 6441472, 685056, 6441984),
    ),
    SiteSpec(
        "east-adjacent-southeast-transition",
        HeightChunkId(-1, 618, 378),
        HeightChunkId(0, 154, 94),
        (685056, 6441472, 685568, 6441984),
    ),
)

COMPOSITION_SITES = (
    SiteSpec(
        "forest-boulder-composition-positive",
        HeightChunkId(-1, 609, 388),
        HeightChunkId(0, 152, 97),
        (680448, 6436352, 680960, 6436864),
    ),
)
ROCK_ONLY_SITES = (
    SiteSpec(
        "mapped-boulder-body-socket-rock-only-1145648",
        HeightChunkId(-1, 615, 390),
        HeightChunkId(0, 153, 97),
        (683520, 6435328, 684032, 6435840),
    ),
)

PAIR_LAYOUT = ArtifactLayout(
    "forest-mesic-mineral-adjacent-continuity-artifact/1",
    (16384, 8192),
    (684544, 6441472, 685056, 6442496),
    (2468, 1508),
    PAIR_SITES,
    "laas.micro.forest-adjacent-two-parent-preview.recipe.v1",
    "accepted-forest-adjacent-two-parent-absolute-master-v1",
)
BLOCK_LAYOUT = ArtifactLayout(
    "forest-mesic-mineral-adjacent-block-artifact/1",
    (16384, 16384),
    (684544, 6441472, 685568, 6442496),
    (2468, 1508),
    BLOCK_SITES,
    "laas.micro.forest-adjacent-four-parent-preview.recipe.v1",
    "accepted-forest-adjacent-four-parent-absolute-master-v1",
)
COMPOSITION_LAYOUT = ArtifactLayout(
    "forest-mesic-mineral-mapped-boulder-composition-positive/3.artifact/1",
    (8192, 8192),
    (680448, 6436352, 680960, 6436864),
    (2436, 1552),
    COMPOSITION_SITES,
    # Reuse the already released contiguous-master verification dispatch.  The
    # artifact schema and content-addressed inputs keep this recipe distinct.
    "laas.micro.forest-adjacent-two-parent-preview.recipe.v1",
    "accepted-forest-mapped-boulder-composition-absolute-master-v2",
)
ROCK_ONLY_LAYOUT = ArtifactLayout(
    "mapped-boulder-body-socket-rock-only/1.artifact/1",
    (8192, 8192),
    (683520, 6435328, 684032, 6435840),
    (2460, 1560),
    ROCK_ONLY_SITES,
    "laas.micro.mapped-boulder-rock-only-preview.recipe.v1",
    "accepted-mapped-boulder-rock-only-absolute-master-v1",
)


def _layout_for_schema(schema: str) -> ArtifactLayout:
    for layout in (PAIR_LAYOUT, BLOCK_LAYOUT, COMPOSITION_LAYOUT, ROCK_ONLY_LAYOUT):
        if layout.schema == schema:
            return layout
    raise ValueError(f"unsupported adjacent forest artifact schema: {schema}")


def _artifact_layout(artifact_root: Path) -> ArtifactLayout:
    manifest = json.loads((artifact_root / "manifest.json").read_bytes())
    return _layout_for_schema(manifest.get("schema", ""))


def _composition_families(
    artifact_root: Path, manifest: dict[str, Any]
) -> tuple[int, ...]:
    recipe = json.loads((artifact_root / "recipe.json").read_bytes())
    identity = recipe.get("config", {})
    config_path = (ASSET_GEN_ROOT.parent / identity.get("path", "")).resolve()
    if (
        ASSET_GEN_ROOT.parent.resolve() not in config_path.parents
        or not config_path.is_file()
        or config_path.stat().st_size != identity.get("bytes")
        or _sha256(config_path) != identity.get("sha256")
    ):
        raise ValueError("composition config binding changed")
    config = json.loads(config_path.read_bytes())
    families = tuple(
        sorted({int(row["source_family"]) for row in config.get("mapped_boulders", ())})
    )
    metric_families = manifest.get("metrics", {}).get("rock", {}).get("families", {})
    ownership_families = {
        int(code) for code in manifest.get("ownership", {}) if int(code) not in {0, 1}
    }
    if (
        not families
        or not set(families).issubset({2, 3, 4})
        or manifest.get("schema") != config.get("schema", "") + ".artifact/1"
        or {int(code) for code in metric_families} != set(families)
        or ownership_families != set(families)
        or any(
            int(metric_families[str(code)].get("owned_fine_cells", 0)) <= 0
            for code in families
        )
    ):
        raise ValueError("composition mapped-family binding changed")
    return families


def _artifact_files(
    artifact_root: Path, layout: ArtifactLayout
) -> dict[str, str]:
    manifest = json.loads((artifact_root / "manifest.json").read_bytes())
    metrics = manifest.get("metrics", {})
    common_invalid = (
        manifest.get("schema") != layout.schema
        or manifest.get("build_id") != artifact_root.name
        or manifest.get("failures") != []
        or tuple(metrics.get("bbox_en", ())) != layout.master_bbox
        or tuple(metrics.get("shape", ())) != layout.master_shape
    )
    if layout == COMPOSITION_LAYOUT:
        rock = metrics.get("rock", {})
        families = rock.get("families", {})
        expected_families = _composition_families(artifact_root, manifest)
        invalid = (
            common_invalid
            or manifest.get("status") != "ready_to_pack"
            or metrics.get("maximum_abstained_residual_m") != 0.0
            or metrics.get("maximum_boulder_pile_residual_m") != 0.0
            or rock.get("generic_till_owned_cells") != 0
            or {int(code) for code in families} != set(expected_families)
            or any(
                not row.get("anchor_owned")
                or float(row.get("parent_mean_max_error_m", 1.0)) > 1e-10
                or float(row.get("whole_window_forest_carrier_maximum_error_m", 1.0))
                > 1e-12
                or float(row.get("rock_support_forest_overlap_fraction", 0.0)) < 0.9
                or float(row.get("maximum_rock_support_to_forest_distance_m", 99.0)) > 2.1
                or float(row.get("anchor_forest_interior_clearance_m", 0.0)) < 6.0
                for row in families.values()
            )
        )
    elif layout == ROCK_ONLY_LAYOUT:
        rock = metrics.get("rock", {})
        families = rock.get("families", {})
        expected_families = _composition_families(artifact_root, manifest)
        invalid = (
            common_invalid
            or manifest.get("status") != "inspect_float_preview"
            or manifest.get("authority", {}).get("forest_transfer") is not False
            or metrics.get("maximum_abstained_residual_m") != 0.0
            or metrics.get("maximum_boulder_pile_residual_m") != 0.0
            or rock.get("generic_till_owned_cells") != 0
            or set(expected_families) != {4}
            or {int(code) for code in families} != {4}
            or any(
                not row.get("anchor_owned")
                or int(row.get("etak_id", 0)) != 1145648
                or float(row.get("parent_mean_max_error_m", 1.0)) > 1e-10
                or float(row.get("whole_window_base_carrier_maximum_error_m", 1.0))
                > 1e-12
                for row in families.values()
            )
        )
    else:
        invalid = (
            common_invalid
            or manifest.get("status") != "inspect_float_preview"
            or metrics.get("maximum_hard_exclusion_residual_m") != 0.0
            or float(metrics.get("maximum_one_metre_mean_error_m", 1.0)) > 1e-12
        )
    if invalid:
        raise ValueError("adjacent forest artifact is not the accepted float candidate")
    for relative, identity in manifest["files"].items():
        path = artifact_root / relative
        if path.stat().st_size != identity["bytes"] or _sha256(path) != identity["sha256"]:
            raise ValueError(f"adjacent forest artifact changed: {relative}")
    master = np.load(artifact_root / "surface/c1_height_f32.npy", mmap_mode="r")
    if master.shape != layout.master_shape or master.dtype != np.float32:
        raise ValueError(
            f"adjacent forest master must be float32 {layout.master_shape}"
        )
    return {relative: row["sha256"] for relative, row in manifest["files"].items()}


def _recipe_identity(
    artifact_root: Path,
    source_base_manifest: Path,
    layout: ArtifactLayout,
) -> tuple[str, dict[str, Any]]:
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("adjacent_preview_verify.py"),
        Path(__file__).with_name("generalization_preview_verify.py"),
        Path(__file__).with_name("generalization_preview.py"),
        Path(__file__).with_name("preview.py"),
        Path(__file__).parents[3] / "height_geom.py",
        Path(__file__).parents[3] / "process/micro_masks.py",
        Path(__file__).parents[3] / "cook/chunkio.py",
        Path(__file__).parents[3] / "cook/encode.py",
        Path(__file__).parents[3] / "cook/micro_hierarchy.py",
        Path(__file__).parents[2] / "repair/base_transaction.py",
        Path(__file__).parents[3] / "release.py",
    )
    inputs: dict[str, Any] = {
        "id": layout.recipe_id,
        "artifact": {
            "root": artifact_root.as_posix(),
            "schema": layout.schema,
            "manifestSha256": _sha256(artifact_root / "manifest.json"),
            "recipeSha256": _sha256(artifact_root / "recipe.json"),
            "files": _artifact_files(artifact_root, layout),
        },
        "sourceBase": {
            "manifest": source_base_manifest.as_posix(),
            "manifestSha256": _sha256(source_base_manifest),
        },
        "coverage": {"sites": [site.json() for site in layout.sites]},
        "singleContiguousMaster": {
            "bboxEn": list(layout.master_bbox),
            "shape": list(layout.master_shape),
            "fineOrigin": list(layout.master_fine_origin),
        },
        "sourceSha256": {
            path.relative_to(Path(__file__).parents[4]).as_posix(): _sha256(path)
            for path in source_paths
        },
    }
    blob = _json_bytes(inputs)
    digest = hashlib.sha256(
        (layout.recipe_id + "\0").encode() + blob
    ).hexdigest()
    return digest, inputs


def _save_npy_immutable(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npy")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    np.save(temporary, values, allow_pickle=False)
    try:
        _immutable(path, temporary.read_bytes())
    finally:
        temporary.unlink(missing_ok=True)


def _stage_masks(
    build_root: Path,
    artifact_root: Path,
    layout: ArtifactLayout,
) -> tuple[dict[HeightChunkId, Path], dict[str, dict[str, int]]]:
    paths: dict[HeightChunkId, Path] = {}
    evidence_by_site: dict[str, dict[str, int]] = {}
    tile_cells = 2048
    for site in layout.sites:
        packed = np.empty((8192, 1024), dtype=np.uint8)
        evidence: dict[str, int] = {}
        if layout in (COMPOSITION_LAYOUT, ROCK_ONLY_LAYOUT):
            ownership = np.load(
                artifact_root / "surface/composition_ownership_u8.npy",
                mmap_mode="r",
            )
            if ownership.shape != (8192, 8192) or ownership.dtype != np.uint8:
                raise ValueError("composition ownership must be uint8 8192 square")
            packed[:] = np.packbits(ownership > 0, axis=1, bitorder="little")
            manifest = json.loads((artifact_root / "manifest.json").read_bytes())
            families = _composition_families(artifact_root, manifest)
            actual_families = set(np.unique(ownership).tolist()) - {0, 1}
            if actual_families != set(families):
                raise ValueError("composition ownership families changed")
            evidence = {
                "ownedCells": int(np.count_nonzero(ownership)),
                **{
                    f"mappedFamily{code}OwnedCells": int(np.count_nonzero(ownership == code))
                    for code in families
                },
            }
            if layout == COMPOSITION_LAYOUT:
                evidence["forestOwnedCells"] = int(np.count_nonzero(ownership == 1))
            path = build_root / "evidence/masks" / f"{site.parent.cx}_{site.parent.cz}.npy"
            _save_npy_immutable(path, packed)
            paths[site.parent] = path
            evidence_by_site[site.site_id] = evidence
            continue
        e_min, _n_min, _e_max, n_max = site.bbox_en
        for tile_row in range(4):
            for tile_col in range(4):
                tile_e = e_min + tile_col * 128
                tile_n = n_max - tile_row * 128
                east = tile_e + (np.arange(tile_cells) + 0.5) * TEXEL_M
                north = tile_n - (np.arange(tile_cells) + 0.5) * TEXEL_M
                mask = rasterize_micro_morphology_mask(east, north)
                rows = np.s_[tile_row * tile_cells : (tile_row + 1) * tile_cells]
                cols = np.s_[tile_col * 256 : (tile_col + 1) * 256]
                packed[rows, cols] = np.packbits(
                    mask.allowed, axis=1, bitorder="little"
                )
                for key, value in mask.evidence().items():
                    if isinstance(value, int):
                        evidence[key] = evidence.get(key, 0) + value
        path = build_root / "evidence/masks" / f"{site.parent.cx}_{site.parent.cz}.npy"
        _save_npy_immutable(path, packed)
        paths[site.parent] = path
        evidence_by_site[site.site_id] = evidence
    return paths, evidence_by_site


def _stage_fine(
    base: BaseConfig,
    build_root: Path,
    artifact_root: Path,
    source_base_manifest: Path,
    layout: ArtifactLayout,
) -> tuple[dict[HeightChunkId, Path], list[dict[str, Any]]]:
    coverage = plan_parent_set(tuple(site.parent for site in layout.sites))
    master = np.load(artifact_root / "surface/c1_height_f32.npy", mmap_mode="r")
    source_sha = _sha256(source_base_manifest)
    pinned = PinnedBaseHeight(source_base_manifest, source_sha, DATA_OUT, base.encode)
    scratch = build_root / "scratch/fine-cores"
    scratch.mkdir(parents=True, exist_ok=True)

    def core_path(chunk: HeightChunkId) -> Path:
        return scratch / f"{chunk.cx}_{chunk.cz}.npy"

    def core(chunk: HeightChunkId) -> np.ndarray:
        dx = chunk.cx - layout.master_fine_origin[0]
        dz = chunk.cz - layout.master_fine_origin[1]
        if (
            0 <= dx < layout.master_shape[1] // FINE_CORE
            and 0 <= dz < layout.master_shape[0] // FINE_CORE
        ):
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
    identities: list[dict[str, Any]] = []
    for chunk in (*coverage.published_fine, *coverage.transient_support):
        values = np.empty((FINE_CORE + 1, FINE_CORE + 1), dtype=np.float32)
        values[:-1, :-1] = core(chunk)
        values[:-1, -1] = core(HeightChunkId(-2, chunk.cx + 1, chunk.cz))[:, 0]
        values[-1, :-1] = core(HeightChunkId(-2, chunk.cx, chunk.cz + 1))[0, :]
        values[-1, -1] = core(
            HeightChunkId(-2, chunk.cx + 1, chunk.cz + 1)
        )[0, 0]
        root = build_root / ("chunks" if chunk in published else "transient")
        destination = root / "height/-2" / f"{chunk.cx}_{chunk.cz}.lac"
        destination.parent.mkdir(parents=True, exist_ok=True)
        identity = _write_height(
            destination, base, chunk, values, FINE_QSCALE, qoffset=30.0
        )
        identity["role"] = (
            "contiguous-accepted-master" if chunk in published else "base-apron-support"
        )
        paths[chunk] = destination
        identities.append(identity)
    return paths, identities


def _stage_parents(
    base: BaseConfig,
    build_root: Path,
    paths: dict[HeightChunkId, Path],
    layout: ArtifactLayout,
) -> tuple[list[dict[str, Any]], dict[HeightChunkId, np.ndarray]]:
    identities: list[dict[str, Any]] = []
    decoded: dict[HeightChunkId, np.ndarray] = {}
    for site in layout.sites:
        coverage = plan_hero(site.parent.cx, site.parent.cz)
        mosaic = assemble_parent_source_memmap(
            build_root / f"scratch/forest-adjacent-parent-{site.parent.cx}-{site.parent.cz}.f32",
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


def _unpack_allowed(path: Path) -> np.ndarray:
    packed = np.load(path, mmap_mode="r")
    if packed.shape != (8192, 1024) or packed.dtype != np.uint8:
        raise ValueError(f"invalid staged morphology mask: {path}")
    return np.unpackbits(packed, axis=1, bitorder="little")[:, :8192].astype(bool)


def _corrected_base(
    base: BaseConfig,
    build_root: Path,
    source_manifest: Path,
    parent_decoded: dict[HeightChunkId, np.ndarray],
    mask_paths: dict[HeightChunkId, Path],
    content_root: Path,
    layout: ArtifactLayout,
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
    for site in layout.sites:
        target = targets.setdefault(
            site.authority,
            np.array(source.load(site.authority).decoded[:-1, :-1], dtype=np.float64, copy=True),
        )
        affected = masks.setdefault(site.authority, np.zeros((2048, 2048), dtype=bool))
        parent_lod0 = box_mean4_striped(parent_decoded[site.parent][:2048, :2048])
        allowed = _unpack_allowed(mask_paths[site.parent])
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


def materialize_adjacent_preview(
    *,
    artifact_root: Path,
    source_base_manifest: Path,
    content_root: Path = DATA_OUT,
    work_root: Path = DATA_WORK,
) -> Path:
    base = load_base()
    artifact_root = artifact_root.resolve()
    source_base_manifest = source_base_manifest.resolve()
    layout = _artifact_layout(artifact_root)
    build_digest, inputs = _recipe_identity(
        artifact_root, source_base_manifest, layout
    )
    build_root = work_root / "builds" / build_digest
    build_root.mkdir(parents=True, exist_ok=True)
    mask_paths, mask_metrics = _stage_masks(build_root, artifact_root, layout)
    paths, fine_identities = _stage_fine(
        base, build_root, artifact_root, source_base_manifest, layout
    )
    parent_identities, parent_decoded = _stage_parents(
        base, build_root, paths, layout
    )
    corrected_transaction, corrected_release = _corrected_base(
        base,
        build_root,
        source_base_manifest,
        parent_decoded,
        mask_paths,
        content_root,
        layout,
    )
    coverage = plan_parent_set(tuple(site.parent for site in layout.sites))
    inputs["masks"] = {
        site.site_id: {
            "path": mask_paths[site.parent].relative_to(build_root).as_posix(),
            "sha256": _sha256(mask_paths[site.parent]),
            "kind": (
                "composition-ownership"
                if layout in (COMPOSITION_LAYOUT, ROCK_ONLY_LAYOUT)
                else "recomputed-morphology-allowed"
            ),
            "evidence": mask_metrics[site.site_id],
        }
        for site in layout.sites
    }
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
        "sites": [site.json() for site in layout.sites],
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
        "cook": layout.cook_id,
        "cookRevision": COOK_REVISION,
        "recipeSha256": build_digest,
        "artifactManifestSha256": _sha256(artifact_root / "manifest.json"),
        "sites": [site.json() for site in layout.sites],
        "children": fine_identities,
        "parents": parent_identities,
        "masks": inputs["masks"],
        "changedFine": [[c.lod, c.cx, c.cz] for c in coverage.published_fine],
        "correctedBase": inputs["correctedBase"],
    }
    _immutable(build_root / "evidence/forest-adjacent-preview.json", _json_bytes(evidence))
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
    print(materialize_adjacent_preview(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
