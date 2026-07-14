"""Immutable inputs for independently rerasterized coarse/fine water closure."""
from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely

from ...config import ASSET_GEN_ROOT, load_base
from ...height_geom import HeightChunkId
from .model import FlowOrientation, FlowProfile


SHARED_WATER_CLOSURE_VERSION = "halo-aware-shared-water-closure/2"
SHARED_WATER_RASTERIZER_ID = "laas.terrain.shared-water-closure-raster.v1"
SHARED_CLOSURE_MANIFEST_VERSION = "expanded-fine-shared-closure-manifest/1"
FINE_CORE_SIDE = 2049
FINE_HALO_SAMPLES = 16
FINE_EXPANDED_SIDE = FINE_CORE_SIDE + 2 * FINE_HALO_SAMPLES
_CONTENT_DOMAIN = b"laas.terrain.repair.shared-closure-manifest.v1\0"
_SOURCE_PATHS = (
    "src/assetgen/terrain/repair/closure.py",
    "src/assetgen/terrain/repair/fine_water.py",
    "src/assetgen/terrain/repair/hierarchy.py",
    "src/assetgen/terrain/repair/shared_closure.py",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: str, name: str) -> str:
    try:
        if value != value.lower() or len(bytes.fromhex(value)) != 32:
            raise ValueError
    except (AttributeError, ValueError) as error:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest") from error
    return value


def _canonical_json(document: object) -> bytes:
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable shared-closure artifact differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError(f"concurrent shared-closure artifact differs: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _geometry_payload(geometry, name: str) -> bytes:
    if geometry is None or geometry.is_empty or not geometry.is_valid:
        raise ValueError(f"{name} geometry must be nonempty and valid")
    return shapely.to_wkb(
        shapely.force_2d(geometry),
        byte_order=1,
        output_dimension=2,
        include_srid=False,
    )


def _profile_payloads(profile: FlowProfile) -> tuple[bytes, bytes]:
    if not isinstance(profile, FlowProfile):
        raise TypeError("shared closure requires a qualified FlowProfile")
    metadata = {
        "format": 1,
        "reachId": profile.reach_id,
        "epochId": profile.epoch_id,
        "orientation": profile.orientation.value,
        "longestMissingSpanM": profile.longest_missing_span_m,
        "sourceArtifactSha256": profile.source_artifact_sha256,
    }
    output = io.BytesIO()
    np.savez(
        output,
        station_m=np.asarray(profile.station_m, dtype="<f8"),
        observation_y=np.asarray(profile.observation_y, dtype="<f8"),
        accepted=np.asarray(profile.accepted, dtype="u1"),
        water_y=np.asarray(profile.water_y, dtype="<f8"),
    )
    return _canonical_json(metadata), output.getvalue()


def _entry(path: str, payload: bytes) -> dict:
    return {"path": path, "bytes": len(payload), "sha256": _sha256_bytes(payload)}


@dataclass(frozen=True)
class SharedClosureManifest:
    root: Path
    path: Path
    manifest_sha256: str
    content_sha256: str


@dataclass(frozen=True)
class SharedClosureInputs:
    manifest_path: Path
    manifest_sha256: str
    authority_manifest_sha256: str
    campaign_content_sha256: str
    campaign_qualification_sha256: str
    authority_support: tuple[HeightChunkId, ...]
    mapped_water: object
    qualified_water: object
    centerline: object
    profile: FlowProfile


def write_shared_closure_manifest(
    *,
    output_root: Path,
    authority_manifest_sha256: str,
    campaign_content_sha256: str,
    campaign_qualification_sha256: str,
    authority_support: tuple[HeightChunkId, ...],
    mapped_water_polygon,
    qualified_water_polygon,
    centerline,
    profile: FlowProfile,
) -> SharedClosureManifest:
    """Freeze exact scientific inputs before any fine ownership is materialized."""
    authority_sha = _digest(authority_manifest_sha256, "authority manifest")
    campaign_sha = _digest(campaign_content_sha256, "campaign content")
    qualification_sha = _digest(
        campaign_qualification_sha256, "campaign qualification"
    )
    if (
        not authority_support
        or len(set(authority_support)) != len(authority_support)
        or tuple(sorted(authority_support)) != authority_support
        or any(chunk.lod != -2 for chunk in authority_support)
    ):
        raise ValueError("shared closure requires sorted unique LOD-2 authority support")
    if profile.source_artifact_sha256 != qualification_sha:
        raise ValueError("shared closure profile names another qualification artifact")
    payloads = {
        "mapped-water.wkb": _geometry_payload(mapped_water_polygon, "mapped water"),
        "qualified-water.wkb": _geometry_payload(
            qualified_water_polygon, "qualified water"
        ),
        "centerline.wkb": _geometry_payload(centerline, "centerline"),
    }
    profile_metadata, profile_arrays = _profile_payloads(profile)
    payloads["profile.json"] = profile_metadata
    payloads["profile.npz"] = profile_arrays
    base = load_base()
    identity = {
        "format": 1,
        "manifestVersion": SHARED_CLOSURE_MANIFEST_VERSION,
        "closureVersion": SHARED_WATER_CLOSURE_VERSION,
        "rasterizerId": SHARED_WATER_RASTERIZER_ID,
        "authorityManifestSha256": authority_sha,
        "campaignContentSha256": campaign_sha,
        "campaignQualificationSha256": qualification_sha,
        "grid": {
            "anchorE": base.grid.anchor_e,
            "anchorN": base.grid.anchor_n,
            "chunkMeters": base.grid.chunk_m,
            "chunkResolution": base.grid.chunk_res,
            "lodStep": base.grid.lod_step,
        },
        "fineCoreSide": FINE_CORE_SIDE,
        "haloSamples": FINE_HALO_SAMPLES,
        "expandedSide": FINE_EXPANDED_SIDE,
        "authoritySupport": [
            [chunk.lod, chunk.cx, chunk.cz] for chunk in authority_support
        ],
        "sourceSha256": {
            relative: _sha256_file(ASSET_GEN_ROOT / relative)
            for relative in _SOURCE_PATHS
        },
        "inputs": {
            "mappedWaterWkb": _entry("inputs/mapped-water.wkb", payloads["mapped-water.wkb"]),
            "qualifiedWaterWkb": _entry(
                "inputs/qualified-water.wkb", payloads["qualified-water.wkb"]
            ),
            "centerlineWkb": _entry("inputs/centerline.wkb", payloads["centerline.wkb"]),
            "profileMetadata": _entry("inputs/profile.json", profile_metadata),
            "profileArrays": _entry("inputs/profile.npz", profile_arrays),
        },
    }
    content_sha = hashlib.sha256(_CONTENT_DOMAIN + _canonical_json(identity)).hexdigest()
    document = {**identity, "contentSha256": content_sha}
    root = Path(output_root) / content_sha
    for name, payload in payloads.items():
        _write_immutable(root / "inputs" / name, payload)
    path = root / "manifest.json"
    manifest_payload = _canonical_json(document)
    _write_immutable(path, manifest_payload)
    return SharedClosureManifest(
        root=root,
        path=path,
        manifest_sha256=_sha256_bytes(manifest_payload),
        content_sha256=content_sha,
    )


def _checked_input(root: Path, entry: dict, name: str) -> Path:
    relative = Path(entry.get("path", ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe shared-closure {name} path")
    path = root / relative
    if (
        not path.is_file()
        or path.stat().st_size != entry.get("bytes")
        or _sha256_file(path) != entry.get("sha256")
    ):
        raise ValueError(f"shared-closure {name} integrity mismatch")
    return path


def load_shared_closure_manifest(path: Path) -> SharedClosureInputs:
    """Integrity-check and reconstruct the exact scientific closure inputs."""
    path = Path(path)
    document = json.loads(path.read_bytes())
    if (
        document.get("format") != 1
        or document.get("manifestVersion") != SHARED_CLOSURE_MANIFEST_VERSION
        or document.get("closureVersion") != SHARED_WATER_CLOSURE_VERSION
        or document.get("rasterizerId") != SHARED_WATER_RASTERIZER_ID
        or document.get("fineCoreSide") != FINE_CORE_SIDE
        or document.get("haloSamples") != FINE_HALO_SAMPLES
        or document.get("expandedSide") != FINE_EXPANDED_SIDE
        or set(document.get("inputs", {}))
        != {
            "mappedWaterWkb",
            "qualifiedWaterWkb",
            "centerlineWkb",
            "profileMetadata",
            "profileArrays",
        }
    ):
        raise ValueError("shared-closure manifest contract differs")
    base = load_base()
    if document.get("grid") != {
        "anchorE": base.grid.anchor_e,
        "anchorN": base.grid.anchor_n,
        "chunkMeters": base.grid.chunk_m,
        "chunkResolution": base.grid.chunk_res,
        "lodStep": base.grid.lod_step,
    }:
        raise ValueError("shared-closure grid differs from the active cook grid")
    expected_sources = {
        relative: _sha256_file(ASSET_GEN_ROOT / relative) for relative in _SOURCE_PATHS
    }
    if document.get("sourceSha256") != expected_sources:
        raise ValueError("shared-closure scientific source identity differs")
    identity = {key: value for key, value in document.items() if key != "contentSha256"}
    expected_content = hashlib.sha256(
        _CONTENT_DOMAIN + _canonical_json(identity)
    ).hexdigest()
    if document.get("contentSha256") != expected_content or path.parent.name != expected_content:
        raise ValueError("shared-closure content address differs")
    inputs = document["inputs"]
    mapped_path = _checked_input(path.parent, inputs["mappedWaterWkb"], "mapped water")
    qualified_path = _checked_input(
        path.parent, inputs["qualifiedWaterWkb"], "qualified water"
    )
    centerline_path = _checked_input(path.parent, inputs["centerlineWkb"], "centerline")
    metadata_path = _checked_input(path.parent, inputs["profileMetadata"], "profile metadata")
    arrays_path = _checked_input(path.parent, inputs["profileArrays"], "profile arrays")
    metadata = json.loads(metadata_path.read_bytes())
    if set(metadata) != {
        "format",
        "reachId",
        "epochId",
        "orientation",
        "longestMissingSpanM",
        "sourceArtifactSha256",
    } or metadata.get("format") != 1:
        raise ValueError("shared-closure profile metadata contract differs")
    with np.load(arrays_path, allow_pickle=False) as archive:
        if set(archive.files) != {"station_m", "observation_y", "accepted", "water_y"}:
            raise ValueError("shared-closure profile array set differs")
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    profile = FlowProfile(
        reach_id=metadata["reachId"],
        epoch_id=metadata["epochId"],
        station_m=arrays["station_m"],
        observation_y=arrays["observation_y"],
        accepted=arrays["accepted"].astype(np.bool_),
        water_y=arrays["water_y"],
        orientation=FlowOrientation(metadata["orientation"]),
        longest_missing_span_m=float(metadata["longestMissingSpanM"]),
        source_artifact_sha256=metadata["sourceArtifactSha256"],
    )
    support = tuple(HeightChunkId(*key) for key in document["authoritySupport"])
    if (
        not support
        or support != tuple(sorted(support))
        or len(support) != len(set(support))
        or any(chunk.lod != -2 for chunk in support)
    ):
        raise ValueError("shared-closure authority support is not canonical")
    authority_sha = _digest(document["authorityManifestSha256"], "authority manifest")
    campaign_sha = _digest(document["campaignContentSha256"], "campaign content")
    qualification_sha = _digest(
        document["campaignQualificationSha256"], "campaign qualification"
    )
    if profile.source_artifact_sha256 != qualification_sha:
        raise ValueError("shared-closure profile differs from campaign qualification")
    mapped = shapely.from_wkb(mapped_path.read_bytes())
    qualified = shapely.from_wkb(qualified_path.read_bytes())
    centerline = shapely.from_wkb(centerline_path.read_bytes())
    if (
        mapped.is_empty
        or qualified.is_empty
        or centerline.is_empty
        or not mapped.is_valid
        or not qualified.is_valid
        or not centerline.is_valid
        or not mapped.covers(qualified)
        or centerline.geom_type != "LineString"
        or not centerline.is_simple
    ):
        raise ValueError("shared-closure scientific geometry contract differs")
    return SharedClosureInputs(
        manifest_path=path,
        manifest_sha256=_sha256_file(path),
        authority_manifest_sha256=authority_sha,
        campaign_content_sha256=campaign_sha,
        campaign_qualification_sha256=qualification_sha,
        authority_support=support,
        mapped_water=mapped,
        qualified_water=qualified,
        centerline=centerline,
        profile=profile,
    )


def rerasterize_fine_shared_closure_for_verification(
    *,
    chunk: HeightChunkId,
    scientific_manifest_path: Path,
    authority_manifest_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independently reconstruct and rerasterize one exact fine ownership core."""
    from .authority_inventory import StructuralAuthorityInventory
    from .fine_water import reevaluate_fine_flowing_water
    from .hierarchy import assemble_fine_surface_window

    inputs = load_shared_closure_manifest(scientific_manifest_path)
    authority_path = Path(authority_manifest_path)
    if _sha256_file(authority_path) != inputs.authority_manifest_sha256:
        raise ValueError("shared closure names another authority manifest")
    if chunk not in set(inputs.authority_support):
        raise ValueError("fine rerasterization chunk leaves declared authority support")
    inventory = StructuralAuthorityInventory(
        manifest_path=authority_path,
        manifest_sha256=inputs.authority_manifest_sha256,
        expected_chunks=inputs.authority_support,
        cache_tiles=12,
    )
    window = assemble_fine_surface_window(
        chunk,
        load_structural_height=lambda dependency: inventory.load(
            dependency
        ).structural.height,
        load_baseline_height=lambda dependency: inventory.load_canonical(
            dependency
        ).tile.height,
    )
    result = reevaluate_fine_flowing_water(
        window,
        grid=load_base().grid,
        mapped_water_polygon=inputs.mapped_water,
        qualified_water_polygon=inputs.qualified_water,
        centerline=inputs.centerline,
        profile=inputs.profile,
    )
    baseline = np.ascontiguousarray(window.baseline_height[window.core_slice])
    return result.authority, result.abstained, baseline
