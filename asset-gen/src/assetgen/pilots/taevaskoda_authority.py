"""Cook the frozen Taevaskoda/Ahja Stage-1 structural authority closure."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import shapely

from ..config import ASSET_GEN_ROOT, DATA_WORK, load_base
from ..height_geom import HeightChunkId, UNITS_PER_METER
from ..terrain.repair.cook import AuthorityEvidenceBinding, cook_structural_authority
from ..terrain.repair.model import FlowOrientation, FlowProfile
from ..terrain.repair.pinned_baseline import PinnedDecodedBaseline
from ..terrain.repair.plan import (
    StructuralRepairPlan,
    chunk_set_bounds_en_units,
    plan_structural_repair,
)
from ..terrain.repair.storage import (
    AUTHORITY_MANIFEST_FORMAT,
    baseline_tile_contract,
)


_DOMAIN = b"laas-taevaskoda-ahja-authority-orchestration-v2\0"
AUTHORITY_EVIDENCE_BINDING_VERSION = "qualification-and-campaign-content/1"
_DEFAULT_CONFIG = (
    ASSET_GEN_ROOT / "config/terrain-repair/taevaskoda-ahja-authority-stage1.json"
)
_SOURCE_PATHS = (
    "config/base.toml",
    "pyproject.toml",
    "uv.lock",
    "src/assetgen/config.py",
    "src/assetgen/cook/chunkio.py",
    "src/assetgen/cook/encode.py",
    "src/assetgen/cook/micro_hierarchy.py",
    "src/assetgen/grid.py",
    "src/assetgen/height_geom.py",
    "src/assetgen/process/mosaic.py",
    "src/assetgen/release.py",
    "src/assetgen/pilots/taevaskoda_authority.py",
    "src/assetgen/terrain/repair/baseline.py",
    "src/assetgen/terrain/repair/closure.py",
    "src/assetgen/terrain/repair/components.py",
    "src/assetgen/terrain/repair/cook.py",
    "src/assetgen/terrain/repair/model.py",
    "src/assetgen/terrain/repair/pinned_baseline.py",
    "src/assetgen/terrain/repair/plan.py",
    "src/assetgen/terrain/repair/prolong.py",
    "src/assetgen/terrain/repair/storage.py",
    "src/assetgen/terrain/repair/water.py",
)


@dataclass(frozen=True)
class AuthorityPilotConfig:
    path: Path
    raw: dict[str, Any]
    canonical_sha256: str
    authority_id: str
    campaign_root: Path
    dtm_sources: tuple[Path, ...]
    pinned_manifest: Path
    pinned_content_root: Path
    plan: StructuralRepairPlan


@dataclass(frozen=True)
class VerifiedAuthorityInputs:
    pilot: dict[str, Any]
    campaign: dict[str, Any]
    profile: FlowProfile
    mapped_water: Any
    qualified_water: Any
    centerline: Any
    file_sha256: dict[str, str]
    source_sha256: dict[str, str]
    input_identity: dict[str, Any]
    input_sha256: str
    baseline_source: PinnedDecodedBaseline
    evidence_binding: AuthorityEvidenceBinding


def baseline_authority_identity(spec: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the immutable independent-baseline identity bound into each cook."""
    if spec is None:
        spec = json.loads(_DEFAULT_CONFIG.read_bytes())["pinnedBase"]
    return {
        "source": "pinned browser-decoded format-1 LOD0 height release",
        "persistedArtifact": True,
        "contract": baseline_tile_contract(),
        "release": {
            "manifest": spec["manifest"],
            "manifestBytes": spec["manifestBytes"],
            "manifestSha256": spec["manifestSha256"],
            "heightIndex": spec["heightIndex"],
            "heightIndexBytes": spec["heightIndexBytes"],
            "heightIndexSha256": spec["heightIndexSha256"],
            "contentRoot": spec["contentRoot"],
        },
        "reconstruction": {
            "kind": "Keys cubic plus exact 4x mean-preserving bubble",
            "sourceLod": 0,
            "sourceTexelMeters": 1.0,
            "outputTexelMeters": 0.25,
            "boundedDecodedChunkLru": int(spec["cacheChunks"]),
        },
    }


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _npy_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(output, np.asarray(array), allow_pickle=False)
    return output.getvalue()


def _asset_path(relative: str) -> Path:
    path = (ASSET_GEN_ROOT / relative).resolve()
    if not path.is_relative_to(ASSET_GEN_ROOT):
        raise ValueError(f"authority input escapes asset-gen root: {relative}")
    return path


def _require_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{context} keys differ: missing={sorted(expected - set(value))}, "
            f"unexpected={sorted(set(value) - expected)}"
        )


def _chunk_range(spec: dict[str, Any]) -> tuple[HeightChunkId, ...]:
    lod = int(spec["lod"])
    cx0, cx1 = (int(value) for value in spec["cx"])
    cz0, cz1 = (int(value) for value in spec["cz"])
    if cx1 < cx0 or cz1 < cz0:
        raise ValueError("authority closure range is reversed")
    return tuple(
        HeightChunkId(lod, cx, cz)
        for cz in range(cz0, cz1 + 1)
        for cx in range(cx0, cx1 + 1)
    )


def load_authority_config(path: Path = _DEFAULT_CONFIG) -> AuthorityPilotConfig:
    """Load and structurally validate the frozen orchestration config without I/O hashing."""
    path = Path(path).resolve()
    raw = json.loads(path.read_bytes())
    _require_keys(
        raw,
        {
            "format",
            "authorityId",
            "role",
            "pinnedBase",
            "campaign",
            "geometry",
            "dtm",
            "closure",
        },
        "authority config",
    )
    if (
        raw["format"] != 2
        or raw["authorityId"] != "taevaskoda-ahja-structural-authority-stage1"
        or raw["role"] != "structural_authority_0.25m_no_morphology"
    ):
        raise ValueError("unsupported structural authority orchestration config")
    closure = raw["closure"]
    authority = HeightChunkId(*closure["authorityLod0"])
    corrected = tuple(HeightChunkId(*key) for key in closure["correctedLod0"])
    review = HeightChunkId(*closure["reviewParent"])
    plan = plan_structural_repair(corrected, review)
    if plan.authority_lod0 != authority:
        raise ValueError("declared review authority differs from canonical parent")
    declared_lod1 = _chunk_range(closure["lod1Reducer"])
    declared_lod2 = _chunk_range(closure["lod2Reducer"])
    declared_authority = _chunk_range(closure["authoritySupport"])
    declared_review = _chunk_range(closure["publishedReview"])
    if (
        int(closure["lod1Reducer"]["count"]) != len(declared_lod1)
        or tuple(sorted(declared_lod1)) != plan.lod1_reducer
        or int(closure["lod2Reducer"]["count"]) != len(declared_lod2)
        or tuple(sorted(declared_lod2)) != plan.lod2_reducer
        or int(closure["authoritySupport"]["count"]) != len(declared_authority)
        or tuple(sorted(declared_authority)) != plan.authority_support
        or int(closure["publishedReview"]["count"]) != len(declared_review)
        or tuple(sorted(declared_review)) != plan.published_lod2
    ):
        raise ValueError("declared structural closure differs from canonical plan")
    bounds_units = chunk_set_bounds_en_units(load_base().grid, plan.lod2_reducer)
    bounds_m = [value // UNITS_PER_METER for value in bounds_units]
    if bounds_m != closure["lod2Reducer"]["boundsEn"]:
        raise ValueError("declared structural closure bounds differ from canonical grid")
    authority_bounds = chunk_set_bounds_en_units(load_base().grid, plan.authority_support)
    authority_bounds_m = [value // UNITS_PER_METER for value in authority_bounds]
    if authority_bounds_m != closure["authoritySupport"]["boundsEn"]:
        raise ValueError("declared authority support bounds differ from canonical grid")
    sheets = raw["dtm"]["orderedSheets"]
    if [sheet["sheet"] for sheet in sheets] != ["54472", "54474", "54481", "54483"]:
        raise ValueError("Stage-1 authority requires the exact four ordered DTM sheets")
    pinned = raw["pinnedBase"]
    _require_keys(
        pinned,
        {
            "manifest",
            "manifestBytes",
            "manifestSha256",
            "heightIndex",
            "heightIndexBytes",
            "heightIndexSha256",
            "contentRoot",
            "cacheChunks",
        },
        "pinned base",
    )
    if (
        pinned["manifestSha256"]
        != "708478a57c2118eaa618867e87cc74a35e20c615999b60d7e4177b595ef5495a"
        or int(pinned["cacheChunks"]) < 1
    ):
        raise ValueError("Stage-1 pinned base identity differs")
    return AuthorityPilotConfig(
        path=path,
        raw=raw,
        canonical_sha256=_sha256_bytes(_canonical_json(raw)),
        authority_id=raw["authorityId"],
        campaign_root=_asset_path(raw["campaign"]["root"]),
        dtm_sources=tuple(_asset_path(sheet["path"]) for sheet in sheets),
        pinned_manifest=_asset_path(pinned["manifest"]),
        pinned_content_root=_asset_path(pinned["contentRoot"]),
        plan=plan,
    )


def _verified_file(path: Path, spec: dict[str, Any], context: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(spec["bytes"]):
        raise ValueError(f"{context} byte size differs")
    digest = _sha256_file(path)
    if digest != spec["sha256"]:
        raise ValueError(f"{context} SHA-256 differs")
    return digest


def _load_profile(
    metadata: dict[str, Any], arrays_path: Path, arrays_blob: bytes
) -> FlowProfile:
    if (
        metadata.get("format") != 1
        or metadata.get("scientificRole")
        != "multi_tile_campaign_visible_water_evidence"
        or metadata.get("discontinuityFreeReach") is not True
        or metadata.get("topology", {}).get("cappedReachGeometry") is not True
        or metadata.get("campaign", {}).get("decision", {}).get("kind")
        != "qualified_profile"
        or metadata.get("arraysNpzSha256") != _sha256_bytes(arrays_blob)
    ):
        raise ValueError("campaign does not authorize a structural water profile")
    expected_names = {
        "campaign_station_m",
        "campaign_observation_y",
        "campaign_sample_count",
        "campaign_accepted",
        "campaign_water_y",
    }
    with np.load(arrays_path, allow_pickle=False) as archive:
        if set(archive.files) != expected_names:
            raise ValueError("campaign array set differs")
        values = {name: np.array(archive[name], copy=True) for name in sorted(archive.files)}
    for name, array in values.items():
        descriptor = metadata["arrays"].get(name)
        actual = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": _sha256_bytes(_npy_bytes(array)),
        }
        if descriptor != actual:
            raise ValueError(f"campaign array descriptor differs: {name}")
    decision = metadata["campaign"]["decision"]
    return FlowProfile(
        reach_id=metadata["reachId"],
        epoch_id=metadata["campaign"]["campaignId"],
        station_m=values["campaign_station_m"],
        observation_y=values["campaign_observation_y"],
        accepted=values["campaign_accepted"],
        water_y=values["campaign_water_y"],
        orientation=FlowOrientation(decision["orientation"]),
        longest_missing_span_m=float(decision["longestMissingSpanM"]),
        source_artifact_sha256=metadata["campaign"]["qualificationSha256"],
    )


def _verify_inputs(config: AuthorityPilotConfig) -> VerifiedAuthorityInputs:
    raw = config.raw
    file_hashes: dict[str, str] = {}
    for name, spec in raw["campaign"]["files"].items():
        file_hashes[f"campaign/{name}"] = _verified_file(
            config.campaign_root / name, spec, f"campaign {name}"
        )
    pilot = json.loads((config.campaign_root / "pilot.json").read_bytes())
    campaign_path = config.campaign_root / "campaign.json"
    arrays_path = config.campaign_root / "campaign.npz"
    campaign_blob = campaign_path.read_bytes()
    arrays_blob = arrays_path.read_bytes()
    campaign = json.loads(campaign_blob)
    campaign_spec = raw["campaign"]
    if (
        pilot.get("format") != 1
        or pilot.get("pilotId") != "taevaskoda-ahja-stage1"
        or pilot.get("buildSha256") != campaign_spec["buildSha256"]
        or pilot.get("campaignContentSha256") != campaign_spec["contentSha256"]
        or pilot.get("campaignMetadataSha256") != _sha256_bytes(campaign_blob)
        or pilot.get("campaignArraysSha256") != _sha256_bytes(arrays_blob)
    ):
        raise ValueError("campaign pilot identity differs from authority config")
    encoded_content = _sha256_bytes(
        b"laas-als-water-campaign-v1\0" + campaign_blob + arrays_blob
    )
    if encoded_content != campaign_spec["contentSha256"]:
        raise ValueError("campaign content digest differs")
    profile = _load_profile(campaign, arrays_path, arrays_blob)

    geometries: dict[str, Any] = {}
    for role, spec in raw["geometry"].items():
        path = config.campaign_root / spec["path"]
        digest = _verified_file(path, spec, f"geometry {role}")
        file_hashes[f"geometry/{role}"] = digest
        geometry = shapely.force_2d(shapely.from_wkb(path.read_bytes()))
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError(f"geometry {role} is empty or invalid")
        geometries[role] = geometry
    if (
        campaign["waterPolygonWkbSha256"] != raw["geometry"]["qualifiedWater"]["sha256"]
        or campaign["centerlineWkbSha256"]
        != raw["geometry"]["qualifiedCenterline"]["sha256"]
        or not geometries["qualifiedWater"].covers(geometries["qualifiedCenterline"])
    ):
        raise ValueError("campaign and configured authority geometries disagree")
    mapped_spec = raw["geometry"]["mappedWater"]
    if mapped_spec.get("closureOperation") != "normalized_union_with_qualified_water":
        raise ValueError("mapped-water ULP closure operation differs")
    mapped_water = shapely.normalize(
        shapely.union(geometries["mappedWater"], geometries["qualifiedWater"])
    )
    mapped_wkb = shapely.to_wkb(mapped_water, byte_order=1, include_srid=False)
    if (
        not mapped_water.is_valid
        or not mapped_water.covers(geometries["mappedWater"])
        or not mapped_water.covers(geometries["qualifiedWater"])
        or _sha256_bytes(mapped_wkb) != mapped_spec.get("effectiveSha256")
    ):
        raise ValueError("effective mapped-water ULP closure differs")
    geometries["mappedWater"] = mapped_water

    for path, spec in zip(
        config.dtm_sources, raw["dtm"]["orderedSheets"], strict=True
    ):
        digest = _verified_file(path, spec, f"DTM sheet {spec['sheet']}")
        file_hashes[f"dtm/{spec['sheet']}"] = digest
        with rasterio.open(path) as dataset:
            bounds = [float(value) for value in dataset.bounds]
            if (
                dataset.crs is None
                or dataset.crs.to_epsg() != 3301
                or dataset.count != 1
                or dataset.shape != (5000, 5000)
                or dataset.dtypes != ("float32",)
                or tuple(dataset.res) != (1.0, 1.0)
                or bounds != [float(value) for value in spec["boundsEn"]]
            ):
                raise ValueError(f"DTM sheet metadata differs: {spec['sheet']}")

    pinned = raw["pinnedBase"]
    pinned_manifest_sha = _verified_file(
        config.pinned_manifest,
        {"bytes": pinned["manifestBytes"], "sha256": pinned["manifestSha256"]},
        "pinned base manifest",
    )
    pinned_index_path = _asset_path(pinned["heightIndex"])
    _verified_file(
        pinned_index_path,
        {"bytes": pinned["heightIndexBytes"], "sha256": pinned["heightIndexSha256"]},
        "pinned base height index",
    )
    baseline_source = PinnedDecodedBaseline(
        manifest_path=config.pinned_manifest,
        manifest_sha256=pinned_manifest_sha,
        content_root=config.pinned_content_root,
        encode=load_base().encode,
        cache_chunks=int(pinned["cacheChunks"]),
    )

    source_hashes = {
        relative: _sha256_file(ASSET_GEN_ROOT / relative) for relative in _SOURCE_PATHS
    }
    closure = {
        "authorityLod0": list(config.raw["closure"]["authorityLod0"]),
        "correctedLod0": [[c.lod, c.cx, c.cz] for c in config.plan.corrected_lod0],
        "reviewParent": list(config.raw["closure"]["reviewParent"]),
        "lod1Reducer": [[c.lod, c.cx, c.cz] for c in config.plan.lod1_reducer],
        "lod2Reducer": [[c.lod, c.cx, c.cz] for c in config.plan.lod2_reducer],
        "authoritySupport": [
            [c.lod, c.cx, c.cz] for c in config.plan.authority_support
        ],
        "publishedReview": [[c.lod, c.cx, c.cz] for c in config.plan.published_lod2],
        "reviewSupport": [
            [c.lod, c.cx, c.cz] for c in config.plan.review_lod2_support
        ],
    }
    identity = {
        "format": 2,
        "authorityId": config.authority_id,
        "role": raw["role"],
        "configCanonicalSha256": config.canonical_sha256,
        "campaign": {
            "buildSha256": campaign_spec["buildSha256"],
            "contentSha256": campaign_spec["contentSha256"],
            "qualificationSha256": profile.source_artifact_sha256,
            "filesSha256": {
                key: value for key, value in file_hashes.items() if key.startswith("campaign/")
            },
        },
        "profile": {
            "reachId": profile.reach_id,
            "epochId": profile.epoch_id,
            "orientation": profile.orientation.value,
            "longestMissingSpanM": profile.longest_missing_span_m,
            "stationCount": int(profile.station_m.size),
            "arrays": campaign["arrays"],
        },
        "geometrySha256": {
            "mappedWater": {
                "source": mapped_spec["sha256"],
                "operation": mapped_spec["closureOperation"],
                "effective": mapped_spec["effectiveSha256"],
            },
            "qualifiedWater": raw["geometry"]["qualifiedWater"]["sha256"],
            "qualifiedCenterline": raw["geometry"]["qualifiedCenterline"]["sha256"],
        },
        "contextualDtmEvidence": [
            {
                "sheet": spec["sheet"],
                "path": spec["path"],
                "bytes": spec["bytes"],
                "sha256": file_hashes[f"dtm/{spec['sheet']}"],
                "boundsEn": spec["boundsEn"],
                "sourceUrl": spec["sourceUrl"],
            }
            for spec in raw["dtm"]["orderedSheets"]
        ],
        "baselineAuthority": baseline_authority_identity(pinned),
        "closure": closure,
        "sourceSha256": source_hashes,
    }
    input_sha = _sha256_bytes(_DOMAIN + _canonical_json(identity))
    return VerifiedAuthorityInputs(
        pilot=pilot,
        campaign=campaign,
        profile=profile,
        mapped_water=geometries["mappedWater"],
        qualified_water=geometries["qualifiedWater"],
        centerline=geometries["qualifiedCenterline"],
        file_sha256=file_hashes,
        source_sha256=source_hashes,
        input_identity=identity,
        input_sha256=input_sha,
        baseline_source=baseline_source,
        evidence_binding=AuthorityEvidenceBinding(
            profile_qualification_sha256=profile.source_artifact_sha256,
            campaign_content_sha256=campaign_spec["contentSha256"],
        ),
    )


def _write_immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"immutable authority orchestration differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError(f"concurrent immutable authority conflict: {path}")
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _authority_artifacts(
    manifest_path: Path,
    expected_keys: set[tuple[int, int, int]],
    expected_baseline_release: dict[str, Any],
) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_bytes())
    if (
        manifest.get("format") != AUTHORITY_MANIFEST_FORMAT
        or manifest.get("baselineContract") != baseline_tile_contract()
    ):
        raise ValueError("existing authority manifest contract differs")
    baseline_authority = manifest.get("baselineAuthority")
    if (
        not isinstance(baseline_authority, dict)
        or baseline_authority.get("release") != expected_baseline_release
    ):
        raise ValueError("existing authority pinned baseline release differs")
    baseline_rows = baseline_authority.get("tiles")
    if not isinstance(baseline_rows, list):
        raise ValueError("existing authority has no pinned baseline tile identities")
    rows = manifest.get("tiles")
    if not isinstance(rows, list):
        raise ValueError("existing authority manifest has no paired tile inventory")
    actual: set[tuple[int, int, int]] = set()
    structural_paths: set[Path] = set()
    baseline_paths: set[Path] = set()
    normalized: list[dict[str, Any]] = []
    root = manifest_path.parent.resolve()

    def artifact_path(relative: Any, role: str) -> Path:
        if not isinstance(relative, str):
            raise ValueError(f"existing authority {role} has no path")
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"existing authority {role} path leaves manifest root")
        return path

    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("key"), list):
            raise ValueError("existing authority has an invalid paired tile row")
        key = tuple(row["key"])
        if (
            len(key) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) for value in key)
            or key in actual
        ):
            raise ValueError(f"existing authority has an invalid/duplicate key: {key}")
        actual.add(key)
        structural = artifact_path(row.get("path"), "structural tile")
        baseline = artifact_path(row.get("baselinePath"), "baseline tile")
        if structural in structural_paths or baseline in baseline_paths:
            raise ValueError("existing authority paired tiles alias an artifact")
        structural_paths.add(structural)
        baseline_paths.add(baseline)
        if not structural.is_file() or not baseline.is_file():
            raise ValueError(f"existing authority pair is missing: {key}")
        structural_size = structural.stat().st_size
        baseline_size = baseline.stat().st_size
        structural_sha = _sha256_file(structural)
        baseline_sha = _sha256_file(baseline)
        if (
            row.get("bytes") != structural_size
            or row.get("sha256") != structural_sha
            or row.get("baselineBytes") != baseline_size
            or row.get("baselineSha256") != baseline_sha
        ):
            raise ValueError(f"existing authority paired tile differs: {key}")
        normalized.append(
            {
                "key": list(key),
                "path": row["path"],
                "bytes": structural_size,
                "sha256": structural_sha,
                "baselinePath": row["baselinePath"],
                "baselineBytes": baseline_size,
                "baselineSha256": baseline_sha,
            }
        )
    if structural_paths & baseline_paths:
        raise ValueError("existing structural and baseline inventories alias artifacts")
    if actual != expected_keys:
        raise ValueError("existing authority support tile set differs")
    if {
        tuple(row.get("key", ())) for row in baseline_rows if isinstance(row, dict)
    } != expected_keys or len(baseline_rows) != len(expected_keys):
        raise ValueError("existing pinned baseline tile identity set differs")
    normalized.sort(key=lambda row: tuple(row["key"]))
    return normalized


def _verify_existing(path: Path, inputs: VerifiedAuthorityInputs) -> bool:
    if not path.is_file():
        return False
    document = json.loads(path.read_bytes())
    if (
        document.get("format") != 2
        or document.get("buildSha256") != inputs.input_sha256
        or document.get("inputIdentity") != inputs.input_identity
    ):
        raise ValueError("existing authority orchestration identity differs")
    root = path.parent
    authority = document["authority"]
    manifest_path = root / authority["manifest"]
    if _sha256_file(manifest_path) != authority["manifestSha256"]:
        raise ValueError("existing structural authority manifest differs")
    expected_keys = {
        tuple(key) for key in inputs.input_identity["closure"]["authoritySupport"]
    }
    artifacts = _authority_artifacts(
        manifest_path,
        expected_keys,
        _manifest_baseline_release(inputs),
    )
    structural_bytes = sum(item["bytes"] for item in artifacts)
    baseline_bytes = sum(item["baselineBytes"] for item in artifacts)
    if (
        len(artifacts) != authority["pairCount"]
        or authority["structuralTileCount"] != len(artifacts)
        or authority["baselineTileCount"] != len(artifacts)
        or structural_bytes != authority["structuralBytes"]
        or baseline_bytes != authority["baselineBytes"]
        or structural_bytes + baseline_bytes != authority["totalBytes"]
        or _sha256_bytes(_canonical_json(artifacts))
        != authority["artifactSetSha256"]
    ):
        raise ValueError("existing paired authority artifact set differs")
    return True


def _assert_inputs_unchanged(
    config: AuthorityPilotConfig, inputs: VerifiedAuthorityInputs
) -> None:
    for name, spec in config.raw["campaign"]["files"].items():
        if _sha256_file(config.campaign_root / name) != inputs.file_sha256[f"campaign/{name}"]:
            raise ValueError(f"campaign input changed during authority cook: {name}")
    for role, spec in config.raw["geometry"].items():
        if _sha256_file(config.campaign_root / spec["path"]) != inputs.file_sha256[
            f"geometry/{role}"
        ]:
            raise ValueError(f"geometry input changed during authority cook: {role}")
    for path, spec in zip(
        config.dtm_sources, config.raw["dtm"]["orderedSheets"], strict=True
    ):
        if _sha256_file(path) != inputs.file_sha256[f"dtm/{spec['sheet']}"]:
            raise ValueError(f"DTM input changed during authority cook: {spec['sheet']}")
    for relative, expected in inputs.source_sha256.items():
        if _sha256_file(ASSET_GEN_ROOT / relative) != expected:
            raise ValueError(f"recipe source changed during authority cook: {relative}")
    pinned = config.raw["pinnedBase"]
    if _sha256_file(config.pinned_manifest) != pinned["manifestSha256"]:
        raise ValueError("pinned base manifest changed during authority cook")
    if _sha256_file(_asset_path(pinned["heightIndex"])) != pinned["heightIndexSha256"]:
        raise ValueError("pinned base height index changed during authority cook")


def _manifest_baseline_release(inputs: VerifiedAuthorityInputs) -> dict[str, Any]:
    release = inputs.baseline_source.release_identity
    return {
        "manifestSha256": release.manifest_sha256,
        "heightIndexPath": release.height_index_relative_path,
        "heightIndexBytes": release.height_index_bytes,
        "heightIndexSha256": release.height_index_sha256,
        "reconstructionVersion": release.reconstruction_version,
    }


def run_authority_cook(
    config_path: Path = _DEFAULT_CONFIG,
    *,
    work_root: Path = DATA_WORK,
) -> Path:
    """Verify frozen inputs and cook the complete paired source-authority support."""
    config = load_authority_config(config_path)
    inputs = _verify_inputs(config)
    output_root = (
        Path(work_root)
        / "terrain-repair"
        / config.authority_id
        / inputs.input_sha256
    )
    orchestration_path = output_root / "orchestration.json"
    if _verify_existing(orchestration_path, inputs):
        return orchestration_path

    authority_root = output_root / "authority"
    result = cook_structural_authority(
        grid=load_base().grid,
        dtm_sources=config.dtm_sources,
        mapped_water_polygon=inputs.mapped_water,
        qualified_water_polygon=inputs.qualified_water,
        centerline=inputs.centerline,
        profile=inputs.profile,
        requested_tiles=config.plan.authority_support,
        evidence_binding=inputs.evidence_binding,
        output_root=authority_root,
        baseline_source=inputs.baseline_source,
    )
    _assert_inputs_unchanged(config, inputs)
    expected_keys = {
        (chunk.lod, chunk.cx, chunk.cz) for chunk in config.plan.authority_support
    }
    artifacts = _authority_artifacts(
        result.manifest_path,
        expected_keys,
        _manifest_baseline_release(inputs),
    )
    if len(artifacts) != len(config.plan.authority_support):
        raise AssertionError("authority cook did not return the complete paired support")
    artifact_set_sha = _sha256_bytes(_canonical_json(artifacts))
    structural_bytes = sum(item["bytes"] for item in artifacts)
    baseline_bytes = sum(item["baselineBytes"] for item in artifacts)
    manifest_relative = result.manifest_path.relative_to(output_root).as_posix()
    document = {
        "format": 2,
        "authorityId": config.authority_id,
        "role": config.raw["role"],
        "morphology": "absent",
        "buildSha256": inputs.input_sha256,
        "inputIdentity": inputs.input_identity,
        "campaign": {
            "buildSha256": config.raw["campaign"]["buildSha256"],
            "contentSha256": config.raw["campaign"]["contentSha256"],
            "qualificationSha256": inputs.profile.source_artifact_sha256,
        },
        "recipe": {
            "cookRecipeSha256": result.recipe_sha256,
            "sourceSha256": inputs.source_sha256,
        },
        "authority": {
            "manifest": manifest_relative,
            "manifestSha256": result.manifest_sha256,
            "pairCount": len(artifacts),
            "structuralTileCount": len(artifacts),
            "baselineTileCount": len(artifacts),
            "artifactSetSha256": artifact_set_sha,
            "structuralBytes": structural_bytes,
            "baselineBytes": baseline_bytes,
            "totalBytes": structural_bytes + baseline_bytes,
        },
        "determinism": {
            "inputIdentitySha256": inputs.input_sha256,
            "orderedArtifactSetSha256": artifact_set_sha,
            "immutableCreateOrVerify": True,
        },
    }
    _write_immutable(orchestration_path, _canonical_json(document))
    return orchestration_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--work-root", type=Path, default=DATA_WORK)
    arguments = parser.parse_args()
    print(run_authority_cook(arguments.config, work_root=arguments.work_root))


if __name__ == "__main__":
    main()
