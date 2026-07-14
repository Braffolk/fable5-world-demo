"""Cook a restartable structural authority from pinned height and contextual DTM evidence."""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rasterio
import shapely

from ...config import GridConfig
from ...height_geom import (
    UNITS_PER_METER,
    HeightChunkId,
    chunk_origin_en_units,
    footprint_m,
)
from ...process.mosaic import RasterStack
from .baseline import BaselineTile, SampleGrid
from .closure import (
    MIXED_AUTHORITY_ADJACENCY,
    MIXED_AUTHORITY_CLOSURE_VERSION,
    MIXED_AUTHORITY_COLLAR_SAMPLES,
    close_mixed_water_authority,
)
from .model import FlowProfile, StructuralTile
from .pinned_baseline import PinnedDecodedBaseline, ReconstructedBaselineTile
from .storage import (
    BaselineTileArtifact,
    StructuralTileArtifact,
    decode_baseline_tile,
    decode_structural_tile,
    encode_authority_manifest,
    write_baseline_tile,
    write_structural_tile,
)
from .water import evaluate_flowing_water

_AUTHORITY_SIDE = 512
_AUTHORITY_TEXEL_M = 0.25
_SOURCE_TEXEL_M = 1.0
_RECIPE_VERSION = "structural-authority-cook/4"
_GEOMETRY_TOLERANCE_M = 1e-6
AUTHORITY_EVIDENCE_BINDING_VERSION = "qualification-and-campaign-content/1"
_PAIR_COMPLETION_VERSION = "structural-authority-pair-completion/1"


@dataclass(frozen=True)
class StructuralAuthorityCook:
    recipe_sha256: str
    profile_qualification_sha256: str
    campaign_content_sha256: str
    artifacts: tuple[StructuralTileArtifact, ...]
    manifest_path: Path
    manifest_sha256: str


@dataclass(frozen=True)
class AuthorityEvidenceBinding:
    """Typed identities with deliberately non-interchangeable scientific roles."""

    profile_qualification_sha256: str
    campaign_content_sha256: str

    def validated(self) -> AuthorityEvidenceBinding:
        return AuthorityEvidenceBinding(
            _validated_digest(
                self.profile_qualification_sha256, "profile_qualification_sha256"
            ),
            _validated_digest(self.campaign_content_sha256, "campaign_content_sha256"),
        )


def _validated_digest(value: str, name: str) -> str:
    try:
        if len(value) != 64 or len(bytes.fromhex(value)) != 32 or value != value.lower():
            raise ValueError
    except ValueError as error:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest") from error
    return value


def _source_stack(
    dtm_sources: Sequence[Path] | RasterStack,
) -> tuple[RasterStack, tuple[Path, ...]]:
    if isinstance(dtm_sources, RasterStack):
        stack = dtm_sources
        paths = tuple(Path(source.path) for source in stack.sources)
    else:
        paths = tuple(Path(path) for path in dtm_sources)
        if not paths:
            raise ValueError("at least one real 1 m DTM source is required")
        stack = RasterStack(list(paths))
    if not paths or len(paths) != len(stack.sources):
        raise ValueError("RasterStack has no inspectable DTM sources")
    for path, source in zip(paths, stack.sources, strict=True):
        if not path.is_file():
            raise FileNotFoundError(path)
        if not np.isclose(source.res, _SOURCE_TEXEL_M, rtol=0.0, atol=1e-9):
            raise ValueError(f"structural authority requires real 1 m DTM support: {path}")
        with rasterio.open(path) as dataset:
            epsg = None if dataset.crs is None else dataset.crs.to_epsg()
            if epsg != 3301:
                raise ValueError(f"DTM source is not EPSG:3301: {path}")
            if not np.allclose(dataset.res, (1.0, 1.0), rtol=0.0, atol=1e-9):
                raise ValueError(f"DTM source is not a 1 m raster: {path}")
            transform = dataset.transform
            if not np.allclose(
                (transform.a, transform.b, transform.d, transform.e),
                (1.0, 0.0, 0.0, -1.0),
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError(f"DTM source is not an unrotated north-up grid: {path}")
            if not np.allclose(
                (transform.c, transform.f),
                np.rint((transform.c, transform.f)),
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError(f"DTM source does not use integer-metre grid edges: {path}")
    return stack, paths


def _validated_tiles(tiles: Sequence[HeightChunkId]) -> tuple[HeightChunkId, ...]:
    ordered = tuple(sorted(tiles))
    if not ordered:
        raise ValueError("at least one LOD-2 alignment tile is required")
    if len(set(ordered)) != len(ordered):
        raise ValueError("requested alignment tiles must be unique")
    if any(chunk.lod != -2 for chunk in ordered):
        raise ValueError("structural authority requests must use LOD-2 alignment tiles")
    return ordered


def _validated_geometry(water_polygon, centerline) -> None:
    if (
        water_polygon is None
        or water_polygon.is_empty
        or water_polygon.geom_type not in {"Polygon", "MultiPolygon"}
        or not water_polygon.is_valid
    ):
        raise ValueError("water_polygon must be a nonempty valid Polygon/MultiPolygon")
    if (
        centerline is None
        or centerline.is_empty
        or centerline.geom_type != "LineString"
        or not centerline.is_simple
        or not centerline.is_valid
    ):
        raise ValueError("centerline must be one nonempty valid simple LineString")
    if bool(shapely.has_z(water_polygon)) or bool(shapely.has_z(centerline)):
        raise ValueError("water geometry must be two-dimensional EPSG:3301 geometry")


def _validated_profile(
    profile: FlowProfile, centerline, *, profile_qualification_sha256: str
) -> None:
    if not isinstance(profile, FlowProfile):
        raise TypeError("profile must be a qualified FlowProfile")
    if (
        not np.isfinite(profile.longest_missing_span_m)
        or profile.longest_missing_span_m < 0.0
        or np.isinf(profile.observation_y).any()
        or not np.any(profile.accepted)
        or profile.source_artifact_sha256 != profile_qualification_sha256
    ):
        raise ValueError("profile is not finite or bound to the supplied qualification evidence")
    tolerance = 1e-8
    if (
        profile.station_m[0] < -tolerance
        or profile.station_m[-1] > float(centerline.length) + tolerance
    ):
        raise ValueError("profile station domain leaves the supplied centerline")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _update_field(digest, name: str, value: bytes) -> None:
    encoded_name = name.encode("utf-8")
    digest.update(len(encoded_name).to_bytes(4, "big"))
    digest.update(encoded_name)
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _array_bytes(values: np.ndarray, dtype: str) -> bytes:
    array = np.asarray(values, dtype=np.dtype(dtype)).copy()
    if np.issubdtype(array.dtype, np.floating):
        array[np.isnan(array)] = np.nan
    return np.ascontiguousarray(array).tobytes(order="C")


def _recipe_digest(
    *,
    grid: GridConfig,
    source_paths: tuple[Path, ...],
    mapped_water_polygon,
    qualified_water_polygon,
    centerline,
    profile: FlowProfile,
    tiles: tuple[HeightChunkId, ...],
    evidence_binding: AuthorityEvidenceBinding,
    baseline_source: PinnedDecodedBaseline,
) -> str:
    digest = hashlib.sha256()
    fixed = {
        "version": _RECIPE_VERSION,
        "grid": {
            "anchorE": grid.anchor_e,
            "anchorN": grid.anchor_n,
            "chunkMeters": grid.chunk_m,
            "chunkResolution": grid.chunk_res,
            "lodStep": grid.lod_step,
            "lods": list(grid.lods),
        },
        "tiles": [[tile.lod, tile.cx, tile.cz] for tile in tiles],
        "evidenceBinding": {
            "version": AUTHORITY_EVIDENCE_BINDING_VERSION,
            "profileQualificationSha256": evidence_binding.profile_qualification_sha256,
            "campaignContentSha256": evidence_binding.campaign_content_sha256,
        },
        "profile": {
            "reachId": profile.reach_id,
            "epochId": profile.epoch_id,
            "orientation": profile.orientation.value,
            "longestMissingSpanMeters": profile.longest_missing_span_m,
            "sourceArtifactSha256": profile.source_artifact_sha256,
        },
        "mixedAuthorityClosure": {
            "version": MIXED_AUTHORITY_CLOSURE_VERSION,
            "collarSamples": MIXED_AUTHORITY_COLLAR_SAMPLES,
            "collarMeters": MIXED_AUTHORITY_COLLAR_SAMPLES * _AUTHORITY_TEXEL_M,
            "adjacency": MIXED_AUTHORITY_ADJACENCY,
            "boundaryCorrection": "exact-zero",
            "clearance": "water-minus-existing-bed-depth",
            "contextEdgePolicy": "fail-if-removed-support-reaches-edge",
        },
        "geometryToleranceMeters": _GEOMETRY_TOLERANCE_M,
        "baselineAuthority": _baseline_release_identity(baseline_source),
    }
    _update_field(
        digest,
        "fixed",
        json.dumps(fixed, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )
    _update_field(
        digest,
        "mappedWaterPolygonWkb",
        shapely.to_wkb(
            mapped_water_polygon, byte_order=1, output_dimension=2, include_srid=True
        ),
    )
    _update_field(
        digest,
        "qualifiedWaterPolygonWkb",
        shapely.to_wkb(
            qualified_water_polygon, byte_order=1, output_dimension=2, include_srid=True
        ),
    )
    _update_field(
        digest,
        "centerlineWkb",
        shapely.to_wkb(centerline, byte_order=1, output_dimension=2, include_srid=True),
    )
    for name, values, dtype in (
        ("stationM", profile.station_m, "<f8"),
        ("observationY", profile.observation_y, "<f8"),
        ("accepted", profile.accepted, "u1"),
        ("waterY", profile.water_y, "<f8"),
    ):
        _update_field(digest, name, _array_bytes(values, dtype))
    for index, path in enumerate(source_paths):
        _update_field(
            digest,
            f"contextualDtmEvidence[{index}]",
            bytes.fromhex(_sha256_file(path)),
        )
    return digest.hexdigest()


def _tile_grid(grid: GridConfig, chunk: HeightChunkId) -> SampleGrid:
    if not np.isclose(
        footprint_m(grid, chunk.lod),
        _AUTHORITY_SIDE * _AUTHORITY_TEXEL_M,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("LOD-2 alignment footprint is not 128 m")
    origin_e_units, origin_n_units = chunk_origin_en_units(grid, chunk)
    west = origin_e_units / UNITS_PER_METER
    north = origin_n_units / UNITS_PER_METER
    return SampleGrid(
        center_e=west + _AUTHORITY_TEXEL_M / 2.0,
        center_n=north - _AUTHORITY_TEXEL_M / 2.0,
        texel_m=_AUTHORITY_TEXEL_M,
        rows=_AUTHORITY_SIDE,
        cols=_AUTHORITY_SIDE,
    )


def _closure_grid(target: SampleGrid) -> SampleGrid:
    collar = MIXED_AUTHORITY_COLLAR_SAMPLES
    return SampleGrid(
        center_e=target.center_e - collar * target.texel_m,
        center_n=target.center_n + collar * target.texel_m,
        texel_m=target.texel_m,
        rows=target.rows + 2 * collar,
        cols=target.cols + 2 * collar,
    )


def _grid_footprint(grid: SampleGrid):
    half = grid.texel_m / 2.0
    west = grid.center_e - half
    east = grid.center_e + (grid.cols - 0.5) * grid.texel_m
    north = grid.center_n + half
    south = grid.center_n - (grid.rows - 0.5) * grid.texel_m
    return shapely.box(west, south, east, north)


def _crop_closure(tile: StructuralTile) -> StructuralTile:
    collar = MIXED_AUTHORITY_COLLAR_SAMPLES
    region = np.s_[collar:-collar, collar:-collar]
    return StructuralTile(
        height=tile.height[region],
        valid=tile.valid[region],
        unknown_bathymetry=tile.unknown_bathymetry[region],
        forbidden_morphology=tile.forbidden_morphology[region],
    )


def _crop_baseline(tile: BaselineTile) -> BaselineTile:
    collar = MIXED_AUTHORITY_COLLAR_SAMPLES
    region = np.s_[collar:-collar, collar:-collar]
    return BaselineTile(
        height=tile.height[region],
        valid=tile.valid[region],
    )


def _build_tile(
    *,
    grid: GridConfig,
    baseline_source: PinnedDecodedBaseline,
    mapped_water_polygon,
    qualified_water_polygon,
    centerline,
    profile: FlowProfile,
    chunk: HeightChunkId,
) -> tuple[BaselineTile, StructuralTile, ReconstructedBaselineTile]:
    target = _closure_grid(_tile_grid(grid, chunk))
    reconstructed = baseline_source.reconstruct(
        chunk, halo_samples=MIXED_AUTHORITY_COLLAR_SAMPLES
    )
    baseline = reconstructed.tile
    if baseline.height.shape != (target.rows, target.cols):
        raise AssertionError("pinned baseline collar does not match the closure grid")
    if mapped_water_polygon.disjoint(_grid_footprint(target)):
        baseline_tile = _crop_baseline(baseline)
        empty = np.zeros(baseline_tile.height.shape, dtype=bool)
        return (
            baseline_tile,
            StructuralTile(
                height=baseline_tile.height,
                valid=baseline_tile.valid,
                unknown_bathymetry=empty,
                forbidden_morphology=empty,
            ),
            reconstructed,
        )
    east = np.tile(target.eastings(), target.rows)
    north = np.repeat(target.northings(), target.cols)
    qualified = evaluate_flowing_water(
        easting=east,
        northing=north,
        water_polygon=qualified_water_polygon,
        centerline=centerline,
        profile=profile,
    )
    mapped_wet = np.asarray(
        shapely.covers(mapped_water_polygon, shapely.points(east, north)), dtype=bool
    )
    if np.any(qualified.wet & ~mapped_wet):
        raise ValueError("qualified water geometry leaves mapped water ownership")
    water = type(qualified)(
        wet=mapped_wet,
        supported=qualified.supported,
        water_y=qualified.water_y,
        bed_y=qualified.bed_y,
    )
    closure = close_mixed_water_authority(baseline, water)
    if np.any(closure.removed_support[0, :]) or np.any(
        closure.removed_support[-1, :]
    ) or np.any(closure.removed_support[:, 0]) or np.any(
        closure.removed_support[:, -1]
    ):
        raise ValueError(
            "mixed-authority abstention reached the declared cook collar edge"
        )
    tile = closure.tile
    dry = ~water.wet.reshape(baseline.height.shape)
    if not np.array_equal(tile.valid[dry], baseline.valid[dry]) or not np.array_equal(
        tile.height[dry], baseline.height[dry], equal_nan=True
    ):
        raise AssertionError("structural composition changed dry pinned-baseline samples")
    if not np.array_equal(tile.forbidden_morphology.ravel(), water.wet):
        raise AssertionError("structural composition did not forbid every wet sample")
    baseline_tile = _crop_baseline(baseline)
    structural_tile = _crop_closure(tile)
    unowned = ~structural_tile.unknown_bathymetry
    if not np.array_equal(
        structural_tile.height[unowned], baseline_tile.height[unowned], equal_nan=True
    ) or not np.array_equal(
        structural_tile.valid[unowned], baseline_tile.valid[unowned]
    ):
        raise AssertionError("composed authority changed unowned baseline samples")
    return baseline_tile, structural_tile, reconstructed


def _baseline_tile_identity(
    reconstructed: ReconstructedBaselineTile,
) -> dict:
    dependencies = []
    for dependency in reconstructed.dependencies:
        row = asdict(dependency)
        row["chunk"] = [
            dependency.chunk.lod,
            dependency.chunk.cx,
            dependency.chunk.cz,
        ]
        dependencies.append(row)
    return {
        "key": [
            reconstructed.chunk.lod,
            reconstructed.chunk.cx,
            reconstructed.chunk.cz,
        ],
        "closureHaloSamples": MIXED_AUTHORITY_COLLAR_SAMPLES,
        "manifestSha256": reconstructed.manifest_sha256,
        "reconstructionVersion": reconstructed.reconstruction_version,
        "dependencyRootSha256": reconstructed.dependency_root_sha256,
        "sourceAuthoritySha256": reconstructed.authority_sha256,
        "maximumMeanErrorMeters": reconstructed.maximum_mean_error_m,
        "meanErrorLimitMeters": reconstructed.mean_error_limit_m,
        "dependencies": dependencies,
    }


def _baseline_release_identity(source: PinnedDecodedBaseline) -> dict:
    identity = source.release_identity
    return {
        "manifestSha256": identity.manifest_sha256,
        "heightIndexPath": identity.height_index_relative_path,
        "heightIndexBytes": identity.height_index_bytes,
        "heightIndexSha256": identity.height_index_sha256,
        "reconstructionVersion": identity.reconstruction_version,
    }


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable structural manifest differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _completion_path(root: Path, chunk: HeightChunkId) -> Path:
    return root / "completion" / f"{chunk.cx}_{chunk.cz}.json"


def _evidence_document(binding: AuthorityEvidenceBinding) -> dict:
    return {
        "version": AUTHORITY_EVIDENCE_BINDING_VERSION,
        "profileQualificationSha256": binding.profile_qualification_sha256,
        "campaignContentSha256": binding.campaign_content_sha256,
    }


def _artifact_document(artifact: StructuralTileArtifact) -> dict:
    return {
        "structural": {
            "path": artifact.relative_path,
            "bytes": artifact.bytes,
            "sha256": artifact.sha256,
        },
        "baseline": {
            "path": artifact.baseline_relative_path,
            "bytes": artifact.baseline_bytes,
            "sha256": artifact.baseline_sha256,
        },
    }


def _checked_relative(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("authority completion artifact path is unsafe")
    return root / candidate


def _verify_completion(
    *,
    root: Path,
    path: Path,
    chunk: HeightChunkId,
    recipe_sha256: str,
    evidence_binding: AuthorityEvidenceBinding,
    baseline_release: dict,
) -> tuple[StructuralTileArtifact, dict]:
    try:
        document = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid authority pair completion: {path}") from error
    required = {
        "format",
        "version",
        "key",
        "recipeSha256",
        "evidenceBinding",
        "pinnedBaseline",
        "artifacts",
    }
    if (
        set(document) != required
        or document["format"] != 1
        or document["version"] != _PAIR_COMPLETION_VERSION
        or document["key"] != [chunk.lod, chunk.cx, chunk.cz]
        or document["recipeSha256"] != recipe_sha256
        or document["evidenceBinding"] != _evidence_document(evidence_binding)
        or document["pinnedBaseline"].get("release") != baseline_release
    ):
        raise ValueError(f"authority pair completion identity differs: {chunk}")
    tile_identity = document["pinnedBaseline"].get("tile")
    if not isinstance(tile_identity, dict) or tile_identity.get("key") != [
        chunk.lod,
        chunk.cx,
        chunk.cz,
    ]:
        raise ValueError(f"authority pair pinned dependency identity differs: {chunk}")
    artifacts = document["artifacts"]
    if set(artifacts) != {"structural", "baseline"}:
        raise ValueError(f"authority pair artifact roles differ: {chunk}")
    checked: dict[str, tuple[str, int, str, bytes]] = {}
    for role in ("structural", "baseline"):
        row = artifacts[role]
        if set(row) != {"path", "bytes", "sha256"}:
            raise ValueError(f"authority pair {role} identity differs: {chunk}")
        artifact_path = _checked_relative(root, row["path"])
        payload = artifact_path.read_bytes()
        if len(payload) != row["bytes"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise ValueError(f"authority pair {role} artifact differs: {chunk}")
        checked[role] = (row["path"], row["bytes"], row["sha256"], payload)
    structural = decode_structural_tile(checked["structural"][3])
    baseline = decode_baseline_tile(checked["baseline"][3])
    unowned = ~structural.unknown_bathymetry
    if not np.array_equal(
        structural.height[unowned], baseline.height[unowned], equal_nan=True
    ) or not np.array_equal(structural.valid[unowned], baseline.valid[unowned]):
        raise ValueError(f"authority pair changed unowned baseline samples: {chunk}")
    artifact = StructuralTileArtifact(
        chunk=chunk,
        relative_path=checked["structural"][0],
        bytes=checked["structural"][1],
        sha256=checked["structural"][2],
        evidence_sha256=evidence_binding.campaign_content_sha256,
        baseline_relative_path=checked["baseline"][0],
        baseline_bytes=checked["baseline"][1],
        baseline_sha256=checked["baseline"][2],
    )
    return artifact, tile_identity


def cook_structural_authority(
    *,
    grid: GridConfig,
    dtm_sources: Sequence[Path] | RasterStack,
    mapped_water_polygon,
    qualified_water_polygon,
    centerline,
    profile: FlowProfile,
    requested_tiles: Sequence[HeightChunkId],
    evidence_binding: AuthorityEvidenceBinding,
    output_root: Path,
    baseline_source: PinnedDecodedBaseline,
) -> StructuralAuthorityCook:
    """Cook complete requested structural tiles with explicit wet abstention.

    DTM source order is priority order. Every source must be a real 1 m
    EPSG:3301 raster; callers must not include a DSM or coarse fallback. These
    rasters remain hash-bound repair evidence and never define the canonical
    dry surface. The latter is reconstructed exclusively from ``baseline_source``.
    Mapped wet outside the qualified subreach preserves the baseline and is
    morphology-forbidden. Only the qualified subreach receives a synthetic bed.
    Each pair is built once and receives an immutable completion sidecar only
    after both artifacts exist. A restart verifies and skips completed pairs.
    """
    evidence = evidence_binding.validated()
    tiles = _validated_tiles(requested_tiles)
    _validated_geometry(mapped_water_polygon, centerline)
    _validated_geometry(qualified_water_polygon, centerline)
    if not mapped_water_polygon.covers(qualified_water_polygon):
        raise ValueError("qualified water polygon must stay inside mapped water")
    if centerline.difference(qualified_water_polygon).length > _GEOMETRY_TOLERANCE_M:
        raise ValueError("qualified water polygon does not own the profile centerline")
    _validated_profile(
        profile,
        centerline,
        profile_qualification_sha256=evidence.profile_qualification_sha256,
    )
    _, source_paths = _source_stack(dtm_sources)
    recipe = _recipe_digest(
        grid=grid,
        source_paths=source_paths,
        mapped_water_polygon=mapped_water_polygon,
        qualified_water_polygon=qualified_water_polygon,
        centerline=centerline,
        profile=profile,
        tiles=tiles,
        evidence_binding=evidence,
        baseline_source=baseline_source,
    )

    materialized: list[StructuralTileArtifact] = []
    baseline_identities: list[dict] = []
    baseline_release = _baseline_release_identity(baseline_source)
    for chunk in tiles:
        completion = _completion_path(output_root, chunk)
        if completion.is_file():
            artifact, baseline_identity = _verify_completion(
                root=output_root,
                path=completion,
                chunk=chunk,
                recipe_sha256=recipe,
                evidence_binding=evidence,
                baseline_release=baseline_release,
            )
            materialized.append(artifact)
            baseline_identities.append(baseline_identity)
            continue
        baseline_tile, structural_tile, reconstructed = _build_tile(
            grid=grid,
            baseline_source=baseline_source,
            mapped_water_polygon=mapped_water_polygon,
            qualified_water_polygon=qualified_water_polygon,
            centerline=centerline,
            profile=profile,
            chunk=chunk,
        )
        baseline_artifact: BaselineTileArtifact = write_baseline_tile(
            output_root, chunk, baseline_tile
        )
        artifact = write_structural_tile(
            output_root,
            chunk,
            structural_tile,
            evidence_sha256=evidence.campaign_content_sha256,
            baseline_artifact=baseline_artifact,
        )
        baseline_identity = _baseline_tile_identity(reconstructed)
        completion_document = {
            "format": 1,
            "version": _PAIR_COMPLETION_VERSION,
            "key": [chunk.lod, chunk.cx, chunk.cz],
            "recipeSha256": recipe,
            "evidenceBinding": _evidence_document(evidence),
            "pinnedBaseline": {
                "release": baseline_release,
                "tile": baseline_identity,
            },
            "artifacts": _artifact_document(artifact),
        }
        _write_immutable(
            completion,
            (json.dumps(completion_document, sort_keys=True, separators=(",", ":")) + "\n").encode(),
        )
        verified_artifact, verified_identity = _verify_completion(
            root=output_root,
            path=completion,
            chunk=chunk,
            recipe_sha256=recipe,
            evidence_binding=evidence,
            baseline_release=baseline_release,
        )
        materialized.append(verified_artifact)
        baseline_identities.append(verified_identity)
    expected_completions = {_completion_path(output_root, chunk) for chunk in tiles}
    actual_completions = set((output_root / "completion").glob("*.json"))
    if actual_completions != expected_completions:
        raise ValueError("authority pair completion inventory differs from requested support")
    artifacts = tuple(materialized)
    manifest = encode_authority_manifest(
        recipe_sha256=recipe,
        profile_qualification_sha256=evidence.profile_qualification_sha256,
        campaign_content_sha256=evidence.campaign_content_sha256,
        evidence_binding_version=AUTHORITY_EVIDENCE_BINDING_VERSION,
        artifacts=artifacts,
        baseline_authority={
            "release": baseline_release,
            "tiles": baseline_identities,
        },
    )
    manifest_path = output_root / "manifest.json"
    _write_immutable(manifest_path, manifest)
    return StructuralAuthorityCook(
        recipe_sha256=recipe,
        profile_qualification_sha256=evidence.profile_qualification_sha256,
        campaign_content_sha256=evidence.campaign_content_sha256,
        artifacts=artifacts,
        manifest_path=manifest_path,
        manifest_sha256=hashlib.sha256(manifest).hexdigest(),
    )
