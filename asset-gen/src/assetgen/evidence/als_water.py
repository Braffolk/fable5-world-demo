"""Provenance-bound ALS evidence qualification for flowing-water profiles.

This layer reports what each epoch supports. It deliberately does not select an
epoch, interpret class 2 as water, repair terrain, or publish cooked assets.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import laspy
import numpy as np
import pyproj
import shapely
from shapely.geometry.base import BaseGeometry

from ..terrain.repair.model import AbstainedReach, FlowProfile
from ..terrain.repair.water import fit_flow_profile, station_locations


_QUALIFICATION_METHOD = "als-water-huber-hampel-pava-gaussian-v3"
_ENDPOINT_CAP_TOLERANCE_M = 1.5


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _valid_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


def _readonly_1d(values: np.ndarray, dtype: np.dtype, name: str) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    result.flags.writeable = False
    return result


def _validated_class_contract(
    *,
    classification_authority: str,
    visible_water_class: int,
    ground_class: int,
    noise_classes: tuple[int, ...],
) -> tuple[int, ...]:
    if not classification_authority:
        raise ValueError("classification_authority is required")
    if visible_water_class != 9:
        raise ValueError("only LAS class 9 may be qualified as visible-water evidence")
    if ground_class != 2:
        raise ValueError("LAS ground semantics must identify class 2 explicitly")
    noise = tuple(sorted(set(int(value) for value in noise_classes)))
    if visible_water_class in noise or ground_class in noise:
        raise ValueError("water and ground classes cannot also be noise classes")
    return noise


@dataclass(frozen=True)
class EpochClassSemantics:
    """Explicit classification contract for one independently acquired epoch."""

    epoch_id: str
    source_artifact_sha256: str
    classification_authority: str
    visible_water_class: int
    ground_class: int
    noise_classes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.epoch_id:
            raise ValueError("epoch_id is required")
        if not _valid_sha256(self.source_artifact_sha256):
            raise ValueError("source_artifact_sha256 must be a SHA-256 hex digest")
        noise = _validated_class_contract(
            classification_authority=self.classification_authority,
            visible_water_class=self.visible_water_class,
            ground_class=self.ground_class,
            noise_classes=self.noise_classes,
        )
        object.__setattr__(self, "noise_classes", noise)


@dataclass(frozen=True)
class RetainedInventoryBundle:
    """One independently retained selection and its bound inventory."""

    retained_path: Path
    inventory_path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "retained_path", Path(self.retained_path).resolve())
        object.__setattr__(self, "inventory_path", Path(self.inventory_path).resolve())


@dataclass(frozen=True)
class WaterCampaignContract:
    """One hydrological snapshot assembled from an ordered set of tile artifacts."""

    campaign_id: str
    source_artifact_sha256s: tuple[str, ...]
    expected_acquisition_year: int
    expected_artifact_type: str
    horizontal_crs: str
    vertical_crs: str
    allowed_point_source_ids: tuple[int, ...]
    classification_authority: str
    visible_water_class: int
    ground_class: int
    noise_classes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.campaign_id or not self.source_artifact_sha256s:
            raise ValueError("campaign_id and source artifacts are required")
        if (
            self.expected_acquisition_year < 1900
            or not self.expected_artifact_type
            or not self.horizontal_crs
            or not self.vertical_crs
        ):
            raise ValueError("campaign acquisition and CRS semantics are required")
        for crs_name in (self.horizontal_crs, self.vertical_crs):
            try:
                authority = pyproj.CRS.from_user_input(crs_name).to_authority()
            except pyproj.exceptions.CRSError as error:
                raise ValueError(f"invalid campaign CRS authority {crs_name}") from error
            if authority is None or f"{authority[0]}:{authority[1]}" != crs_name:
                raise ValueError("campaign CRS fields must use canonical authority identifiers")
        point_sources = tuple(int(value) for value in self.allowed_point_source_ids)
        if (
            not point_sources
            or len(set(point_sources)) != len(point_sources)
            or any(value < 0 or value > 65535 for value in point_sources)
        ):
            raise ValueError("allowed_point_source_ids must be unique LAS source IDs")
        artifacts = tuple(self.source_artifact_sha256s)
        if len(set(artifacts)) != len(artifacts) or not all(_valid_sha256(item) for item in artifacts):
            raise ValueError("campaign artifacts must be unique SHA-256 hex digests")
        noise = _validated_class_contract(
            classification_authority=self.classification_authority,
            visible_water_class=self.visible_water_class,
            ground_class=self.ground_class,
            noise_classes=self.noise_classes,
        )
        object.__setattr__(self, "source_artifact_sha256s", artifacts)
        object.__setattr__(self, "allowed_point_source_ids", point_sources)
        object.__setattr__(self, "noise_classes", noise)


@dataclass(frozen=True)
class EpochQualificationMetrics:
    decoded_points: int
    visible_water_labels: int
    ground_labels_not_used: int
    noise_labels_excluded: int
    withheld_water_excluded: int
    overlap_water_excluded: int
    eligible_water_points: int
    inside_water_polygon: int
    outside_water_polygon: int
    observation_geometry_excluded: int
    point_source_counts: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class CampaignArtifactEvidence:
    selection_id: str
    tile_id: str
    source_artifact_sha256: str
    source_filename: str
    acquisition_year: int
    artifact_type: str
    retained_manifest_sha256: str
    inventory_sha256: str
    metrics: EpochQualificationMetrics

    def __post_init__(self) -> None:
        if not self.selection_id or not self.tile_id or not self.source_filename:
            raise ValueError("campaign artifact selection, tile, and filename are required")
        digests = (
            self.source_artifact_sha256,
            self.retained_manifest_sha256,
            self.inventory_sha256,
        )
        if not all(_valid_sha256(value) for value in digests):
            raise ValueError("campaign artifact provenance fields must be SHA-256 digests")


@dataclass(frozen=True)
class AlsWaterCampaignEvidence:
    reach_id: str
    discontinuity_free_reach: bool
    capped_reach_geometry: bool
    campaign: WaterCampaignContract
    source_campaign_sha256: str
    qualification_sha256: str
    water_polygon_wkb_sha256: str
    centerline_wkb_sha256: str
    observation_exclusion_wkb_sha256: str | None
    station_m: np.ndarray
    observation_y: np.ndarray
    sample_count: np.ndarray
    artifacts: tuple[CampaignArtifactEvidence, ...]
    profile: FlowProfile | AbstainedReach

    def __post_init__(self) -> None:
        station = _readonly_1d(self.station_m, np.dtype("<f8"), "station_m")
        observation = _readonly_1d(self.observation_y, np.dtype("<f8"), "observation_y")
        count = _readonly_1d(self.sample_count, np.dtype("<i4"), "sample_count")
        if station.size < 2 or observation.shape != station.shape or count.shape != station.shape:
            raise ValueError("campaign station arrays must have equal length >= 2")
        if not np.isfinite(station).all() or np.any(count < 0):
            raise ValueError("campaign station coordinates and sample counts must be valid")
        if self.discontinuity_free_reach is not True:
            raise ValueError("campaign evidence requires a discontinuity-free reach")
        if self.capped_reach_geometry is not True:
            raise ValueError("campaign evidence requires explicitly capped reach geometry")
        digests = (
            self.source_campaign_sha256,
            self.qualification_sha256,
            self.water_polygon_wkb_sha256,
            self.centerline_wkb_sha256,
        )
        if not all(_valid_sha256(value) for value in digests):
            raise ValueError("campaign and geometry identities must be SHA-256 digests")
        if (
            self.observation_exclusion_wkb_sha256 is not None
            and not _valid_sha256(self.observation_exclusion_wkb_sha256)
        ):
            raise ValueError("observation exclusion identity must be a SHA-256 digest")
        artifact_ids = tuple(item.source_artifact_sha256 for item in self.artifacts)
        if artifact_ids != self.campaign.source_artifact_sha256s:
            raise ValueError("campaign artifact evidence must preserve contracted source order")
        if self.profile.reach_id != self.reach_id:
            raise ValueError("campaign profile belongs to a different reach")
        if isinstance(self.profile, FlowProfile):
            if (
                self.profile.epoch_id != self.campaign.campaign_id
                or self.profile.source_artifact_sha256 != self.qualification_sha256
                or not np.array_equal(self.profile.station_m, station)
                or not np.array_equal(self.profile.observation_y, observation, equal_nan=True)
            ):
                raise ValueError("campaign profile is not bound to its evidence record")
        object.__setattr__(self, "station_m", station)
        object.__setattr__(self, "observation_y", observation)
        object.__setattr__(self, "sample_count", count)


@dataclass(frozen=True)
class EpochStationEvidence:
    epoch_id: str
    source_artifact_sha256: str
    source_filename: str
    class_semantics: EpochClassSemantics
    station_m: np.ndarray
    observation_y: np.ndarray
    sample_count: np.ndarray
    metrics: EpochQualificationMetrics
    profile: FlowProfile | AbstainedReach

    def __post_init__(self) -> None:
        station = _readonly_1d(self.station_m, np.dtype("<f8"), "station_m")
        observation = _readonly_1d(self.observation_y, np.dtype("<f8"), "observation_y")
        count = _readonly_1d(self.sample_count, np.dtype("<i4"), "sample_count")
        if station.size < 2 or observation.shape != station.shape or count.shape != station.shape:
            raise ValueError("station evidence arrays must have equal length >= 2")
        if not np.isfinite(station).all() or np.any(count < 0):
            raise ValueError("station coordinates and sample counts must be valid")
        object.__setattr__(self, "station_m", station)
        object.__setattr__(self, "observation_y", observation)
        object.__setattr__(self, "sample_count", count)


@dataclass(frozen=True)
class AlsWaterEvidence:
    reach_id: str
    discontinuity_free_reach: bool
    retained_manifest_sha256: str
    inventory_sha256: str
    water_polygon_wkb_sha256: str
    centerline_wkb_sha256: str
    epochs: tuple[EpochStationEvidence, ...]


@dataclass(frozen=True)
class EncodedAlsWaterEvidence:
    metadata_json: bytes
    arrays_npz: bytes
    content_sha256: str


def _load_bound_manifests(
    retained_path: Path, inventory_path: Path
) -> tuple[bytes, bytes, dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    retained_bytes = retained_path.read_bytes()
    inventory_bytes = inventory_path.read_bytes()
    retained = json.loads(retained_bytes)
    inventory = json.loads(inventory_bytes)
    retained_sha = _sha256_bytes(retained_bytes)
    if retained.get("complete") is not True or retained.get("missingFiles"):
        raise ValueError("retained ALS manifest is not complete")
    if inventory.get("retainedManifestSha256") != retained_sha:
        raise ValueError("ALS inventory is not bound to the supplied retained manifest")
    if retained.get("selectionId") != inventory.get("selectionId"):
        raise ValueError("retained manifest and inventory selection IDs differ")

    retained_by_sha: dict[str, dict[str, Any]] = {}
    for artifact in retained.get("artifacts", []):
        sha = artifact.get("sha256")
        if not isinstance(sha, str) or not _valid_sha256(sha) or sha in retained_by_sha:
            raise ValueError("retained manifest contains invalid or duplicate artifact IDs")
        retained_by_sha[sha] = artifact
    inventory_artifacts = inventory.get("artifacts", [])
    inventory_by_sha = {artifact.get("sha256"): artifact for artifact in inventory_artifacts}
    if (
        len(inventory_by_sha) != len(inventory_artifacts)
        or set(inventory_by_sha) != set(retained_by_sha)
        or None in inventory_by_sha
    ):
        raise ValueError("ALS inventory artifact set differs from retained manifest")
    for sha, artifact in retained_by_sha.items():
        inventoried = inventory_by_sha[sha]
        for field in ("bytes", "filename", "relativePath"):
            if inventoried.get(field) != artifact.get(field):
                raise ValueError(f"inventory disagrees on {field} for artifact {sha}")
    return retained_bytes, inventory_bytes, retained, inventory, retained_by_sha


def _plan_geometry(geometry: BaseGeometry, expected: tuple[str, ...], name: str) -> BaseGeometry:
    if geometry is None or geometry.geom_type not in expected or geometry.is_empty:
        raise ValueError(f"{name} must be a nonempty {'/'.join(expected)}")
    result = shapely.force_2d(geometry)
    if not result.is_valid:
        raise ValueError(f"{name} is invalid")
    bounds = np.asarray(result.bounds, dtype=np.float64)
    if bounds.shape != (4,) or not np.isfinite(bounds).all():
        raise ValueError(f"{name} bounds must be finite")
    return result


def _crs_authorities(crs: pyproj.CRS) -> set[str]:
    result: set[str] = set()
    authority = crs.to_authority()
    if authority is not None:
        result.add(f"{authority[0]}:{authority[1]}")
    identifier = crs.to_json_dict().get("id")
    if isinstance(identifier, dict) and identifier.get("authority") and identifier.get("code"):
        result.add(f"{identifier['authority']}:{identifier['code']}")
    for child in crs.sub_crs_list:
        result.update(_crs_authorities(child))
    return result


def _require_campaign_crs(
    crs: pyproj.CRS | None, campaign: WaterCampaignContract, context: str
) -> None:
    if crs is None:
        raise ValueError(f"{context} has no parseable CRS")
    authorities = _crs_authorities(crs)
    expected = {campaign.horizontal_crs, campaign.vertical_crs}
    if not expected.issubset(authorities):
        raise ValueError(f"{context} CRS authorities {sorted(authorities)} do not satisfy {sorted(expected)}")


def _line_endpoint_tangents(line: BaseGeometry) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.asarray(line.coords, dtype=np.float64)[:, :2]
    deltas = np.diff(coordinates, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    valid = np.flatnonzero(lengths > 0.0)
    if valid.size == 0:
        raise ValueError("centerline has no nonzero segment")
    start_tangent = deltas[valid[0]] / lengths[valid[0]]
    end_tangent = deltas[valid[-1]] / lengths[valid[-1]]
    return coordinates[0], start_tangent, coordinates[-1], end_tangent


@dataclass(frozen=True)
class _StreamedArtifact:
    point_station_m: np.ndarray
    point_y: np.ndarray
    metrics: EpochQualificationMetrics


def _stream_qualified_artifact(
    *,
    source_path: Path,
    source_artifact_sha256: str,
    visible_water_class: int,
    ground_class: int,
    noise_classes: tuple[int, ...],
    polygon: BaseGeometry,
    line: BaseGeometry,
    chunk_points: int,
    context: str,
    tile_bounds: tuple[float, float, float, float] | None = None,
    campaign: WaterCampaignContract | None = None,
    observation_exclusion: BaseGeometry | None = None,
) -> _StreamedArtifact:
    if source_path.suffix.lower() != ".laz":
        raise ValueError("ALS water qualification requires retained LAZ artifacts")
    if _sha256_file(source_path) != source_artifact_sha256:
        raise ValueError(f"retained artifact bytes differ for {context}")
    decoded = water_labels = ground_labels = noise_label_count = 0
    withheld_water = overlap_water = eligible_water = inside_count = outside_count = 0
    observation_excluded = 0
    station_parts: list[np.ndarray] = []
    height_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    with laspy.open(source_path) as reader:
        if campaign is not None:
            try:
                actual_crs = reader.header.parse_crs()
            except Exception as error:
                raise ValueError(f"{context} LAS header CRS cannot be parsed") from error
            _require_campaign_crs(actual_crs, campaign, f"{context} LAS header")
        if tile_bounds is not None:
            min_x, min_y, max_x, max_y = tile_bounds
            header_min_x, header_min_y = map(float, reader.header.mins[:2])
            header_max_x, header_max_y = map(float, reader.header.maxs[:2])
            if not (
                min_x <= header_min_x < max_x
                and min_y <= header_min_y < max_y
                and min_x <= header_max_x < max_x
                and min_y <= header_max_y < max_y
            ):
                raise ValueError(f"{context} LAS header extent escapes half-open tile ownership")
        dimensions = set(reader.header.point_format.dimension_names)
        required = {"classification", "withheld", "point_source_id", "X", "Y", "Z"}
        if not required.issubset(dimensions):
            raise ValueError(f"{context} lacks required LAS dimensions")
        has_overlap = "overlap" in dimensions
        start_point, start_tangent, end_point, end_tangent = _line_endpoint_tangents(line)
        for points in reader.chunk_iterator(chunk_points):
            decoded += len(points)
            all_x = np.asarray(points.x, dtype=np.float64)
            all_y = np.asarray(points.y, dtype=np.float64)
            if tile_bounds is not None:
                min_x, min_y, max_x, max_y = tile_bounds
                owned = (
                    (all_x >= min_x)
                    & (all_x < max_x)
                    & (all_y >= min_y)
                    & (all_y < max_y)
                )
                if not np.all(owned):
                    raise ValueError(f"{context} contains points outside half-open tile ownership")
            classes = np.asarray(points.classification, dtype=np.uint8)
            withheld = np.asarray(points.withheld, dtype=bool)
            overlap = (
                np.asarray(points.overlap, dtype=bool)
                if has_overlap
                else np.zeros(len(points), dtype=bool)
            )
            water = classes == visible_water_class
            ground_labels += int(np.count_nonzero(classes == ground_class))
            noise_label_count += int(np.count_nonzero(np.isin(classes, noise_classes)))
            water_labels += int(np.count_nonzero(water))
            excluded_withheld = water & withheld
            withheld_water += int(np.count_nonzero(excluded_withheld))
            excluded_overlap = water & ~withheld & overlap
            overlap_water += int(np.count_nonzero(excluded_overlap))
            eligible = water & ~withheld & ~overlap
            eligible_water += int(np.count_nonzero(eligible))
            if not np.any(eligible):
                continue
            x = all_x[eligible]
            y = all_y[eligible]
            z = np.asarray(points.z, dtype=np.float64)[eligible]
            point_sources = np.asarray(points.point_source_id, dtype=np.uint16)[eligible]
            points_2d = shapely.points(x, y)
            inside = np.asarray(shapely.contains(polygon, points_2d), dtype=bool)
            outside_count += int(np.count_nonzero(~inside))
            if observation_exclusion is not None and np.any(inside):
                excluded = inside & np.asarray(
                    shapely.intersects(observation_exclusion, points_2d), dtype=bool
                )
                observation_excluded += int(np.count_nonzero(excluded))
                inside &= ~excluded
            inside_count += int(np.count_nonzero(inside))
            if np.any(inside):
                inside_sources = point_sources[inside]
                if campaign is not None and not np.isin(
                    inside_sources, campaign.allowed_point_source_ids
                ).all():
                    unexpected = sorted(
                        set(map(int, inside_sources)) - set(campaign.allowed_point_source_ids)
                    )
                    raise ValueError(f"{context} has unexpected in-reach point_source_ids {unexpected}")
                inside_xy = np.column_stack((x[inside], y[inside]))
                located = np.asarray(
                    shapely.line_locate_point(line, points_2d[inside]), dtype=np.float64
                )
                before_start = (inside_xy - start_point) @ start_tangent
                after_end = (inside_xy - end_point) @ end_tangent
                if np.any(
                    ((located <= 1e-9) & (before_start < -_ENDPOINT_CAP_TOLERANCE_M))
                    | (
                        (located >= line.length - 1e-9)
                        & (after_end > _ENDPOINT_CAP_TOLERANCE_M)
                    )
                ):
                    raise ValueError(f"{context} has in-polygon evidence beyond a reach endpoint cap")
                station_parts.append(
                    located
                )
                height_parts.append(z[inside])
                source_parts.append(inside_sources)
        if decoded != reader.header.point_count:
            raise ValueError(f"decoded count differs from LAS header for {context}")
    point_station = (
        np.concatenate(station_parts) if station_parts else np.empty(0, dtype=np.float64)
    )
    point_height = (
        np.concatenate(height_parts) if height_parts else np.empty(0, dtype=np.float64)
    )
    point_sources = (
        np.concatenate(source_parts) if source_parts else np.empty(0, dtype=np.uint16)
    )
    source_ids, source_counts = np.unique(point_sources, return_counts=True)
    return _StreamedArtifact(
        point_station_m=point_station,
        point_y=point_height,
        metrics=EpochQualificationMetrics(
            decoded_points=decoded,
            visible_water_labels=water_labels,
            ground_labels_not_used=ground_labels,
            noise_labels_excluded=noise_label_count,
            withheld_water_excluded=withheld_water,
            overlap_water_excluded=overlap_water,
            eligible_water_points=eligible_water,
            inside_water_polygon=inside_count,
            outside_water_polygon=outside_count,
            observation_geometry_excluded=observation_excluded,
            point_source_counts=tuple(
                (int(source_id), int(count))
                for source_id, count in zip(source_ids, source_counts, strict=True)
            ),
        ),
    )


def qualify_als_flowing_water(
    *,
    retained_path: Path,
    inventory_path: Path,
    water_polygon: BaseGeometry,
    centerline: BaseGeometry,
    reach_id: str,
    discontinuity_free_reach: bool,
    epochs: Sequence[EpochClassSemantics],
    chunk_points: int = 1_000_000,
) -> AlsWaterEvidence:
    """Qualify each explicit epoch independently against exact reach geometry."""
    retained_path = Path(retained_path).resolve()
    inventory_path = Path(inventory_path).resolve()
    if not reach_id or chunk_points < 1:
        raise ValueError("reach_id and a positive chunk_points are required")
    if discontinuity_free_reach is not True:
        raise ValueError(
            "flow-profile fitting requires an explicitly segmented discontinuity-free reach"
        )
    epoch_contracts = tuple(epochs)
    if not epoch_contracts:
        raise ValueError("at least one explicit epoch contract is required")
    if len({item.epoch_id for item in epoch_contracts}) != len(epoch_contracts):
        raise ValueError("epoch IDs must be unique")
    if len({item.source_artifact_sha256 for item in epoch_contracts}) != len(epoch_contracts):
        raise ValueError("each epoch must bind a distinct retained artifact")

    retained_bytes, inventory_bytes, _, _, artifacts = _load_bound_manifests(
        retained_path, inventory_path
    )
    polygon = _plan_geometry(water_polygon, ("Polygon", "MultiPolygon"), "water_polygon")
    line = _plan_geometry(centerline, ("LineString",), "centerline")
    if not line.is_simple or line.length < 1.0 or not polygon.covers(line):
        raise ValueError("centerline must be simple, at least 1 m long, and covered by water_polygon")
    stations = np.arange(0.0, np.floor(line.length) + 1.0, 1.0, dtype=np.float64)
    if stations.size < 2:
        raise ValueError("centerline does not support at least two 1 m stations")

    root = retained_path.parent.resolve()
    evidence: list[EpochStationEvidence] = []
    for semantics in epoch_contracts:
        artifact = artifacts.get(semantics.source_artifact_sha256)
        if artifact is None:
            raise ValueError(f"epoch {semantics.epoch_id} is not in the retained manifest")
        relative = Path(artifact["relativePath"])
        source_path = (root / relative).resolve()
        if not source_path.is_relative_to(root) or not source_path.is_file():
            raise ValueError(f"retained artifact path is missing or escapes its root: {relative}")
        if source_path.stat().st_size != artifact["bytes"]:
            raise ValueError(f"retained artifact bytes differ for epoch {semantics.epoch_id}")
        streamed = _stream_qualified_artifact(
            source_path=source_path,
            source_artifact_sha256=semantics.source_artifact_sha256,
            visible_water_class=semantics.visible_water_class,
            ground_class=semantics.ground_class,
            noise_classes=semantics.noise_classes,
            polygon=polygon,
            line=line,
            chunk_points=chunk_points,
            context=f"epoch {semantics.epoch_id}",
        )
        observation, sample_count = station_locations(
            streamed.point_station_m, streamed.point_y, stations
        )
        profile = fit_flow_profile(
            reach_id=reach_id,
            epoch_id=semantics.epoch_id,
            source_artifact_sha256=semantics.source_artifact_sha256,
            station_m=stations,
            observation_y=observation,
        )
        evidence.append(
            EpochStationEvidence(
                epoch_id=semantics.epoch_id,
                source_artifact_sha256=semantics.source_artifact_sha256,
                source_filename=artifact["filename"],
                class_semantics=semantics,
                station_m=stations,
                observation_y=observation,
                sample_count=sample_count,
                metrics=streamed.metrics,
                profile=profile,
            )
        )

    polygon_wkb = shapely.to_wkb(polygon, byte_order=1, include_srid=False)
    line_wkb = shapely.to_wkb(line, byte_order=1, include_srid=False)
    return AlsWaterEvidence(
        reach_id=reach_id,
        discontinuity_free_reach=True,
        retained_manifest_sha256=_sha256_bytes(retained_bytes),
        inventory_sha256=_sha256_bytes(inventory_bytes),
        water_polygon_wkb_sha256=_sha256_bytes(polygon_wkb),
        centerline_wkb_sha256=_sha256_bytes(line_wkb),
        epochs=tuple(evidence),
    )


@dataclass(frozen=True)
class _BoundCampaignArtifact:
    selection_id: str
    tile_id: str
    tile_bounds: tuple[float, float, float, float]
    retained_manifest_sha256: str
    inventory_sha256: str
    root: Path
    artifact: dict[str, Any]
    inventory_artifact: dict[str, Any]


def _tile_bounds_record(tile: dict[str, Any]) -> tuple[float, float, float, float]:
    bounds = tile.get("bounds")
    try:
        result = (
            float(bounds["min_x"]),
            float(bounds["min_y"]),
            float(bounds["max_x_exclusive"]),
            float(bounds["max_y_exclusive"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("campaign bundle requires finite exclusive tile bounds") from error
    if (
        not np.isfinite(result).all()
        or result[0] >= result[2]
        or result[1] >= result[3]
    ):
        raise ValueError("campaign bundle has an invalid tile declaration")
    return result


def _retained_tile_bounds(
    retained: dict[str, Any], expected_crs: str
) -> dict[str, tuple[float, float, float, float]]:
    declared = retained.get("tiles")
    if declared is None:
        single = retained.get("tile")
        declared = [single] if isinstance(single, dict) else None
    if not isinstance(declared, list) or not declared:
        raise ValueError("campaign bundle requires one or more tile declarations")

    result: dict[str, tuple[float, float, float, float]] = {}
    for tile in declared:
        tile_id = tile.get("id") if isinstance(tile, dict) else None
        tile_bounds = _tile_bounds_record(tile) if isinstance(tile, dict) else None
        if (
            not isinstance(tile_id, str)
            or not tile_id
            or tile_id in result
            or tile.get("crs") != expected_crs
            or tile_bounds is None
        ):
            raise ValueError("campaign bundle has an invalid tile declaration")
        result[tile_id] = tile_bounds
    return result


def _campaign_sources(
    bundles: Sequence[RetainedInventoryBundle], campaign: WaterCampaignContract
) -> tuple[_BoundCampaignArtifact, ...]:
    declared = tuple(bundles)
    if not declared:
        raise ValueError("at least one retained/inventory bundle is required")
    all_artifacts: dict[str, _BoundCampaignArtifact] = {}
    retained_digests: set[str] = set()
    inventory_digests: set[str] = set()
    selection_ids: set[str] = set()
    contributing_bundles = 0
    for bundle in declared:
        retained_bytes, inventory_bytes, retained, inventory, artifacts = _load_bound_manifests(
            bundle.retained_path, bundle.inventory_path
        )
        retained_sha = _sha256_bytes(retained_bytes)
        inventory_sha = _sha256_bytes(inventory_bytes)
        selection_id = retained.get("selectionId")
        if (
            not isinstance(selection_id, str)
            or not selection_id
            or retained_sha in retained_digests
            or inventory_sha in inventory_digests
            or selection_id in selection_ids
        ):
            raise ValueError("campaign bundles contain duplicate source declarations")
        retained_digests.add(retained_sha)
        inventory_digests.add(inventory_sha)
        selection_ids.add(selection_id)
        tiles = _retained_tile_bounds(retained, campaign.horizontal_crs)
        inventory_by_sha = {
            artifact["sha256"]: artifact for artifact in inventory["artifacts"]
        }
        selected_here = 0
        for sha, artifact in artifacts.items():
            if sha in all_artifacts:
                raise ValueError("campaign bundles declare an overlapping retained artifact")
            artifact_tile = artifact.get("tile")
            if isinstance(artifact_tile, dict):
                embedded_tile = artifact_tile
                artifact_tile = embedded_tile.get("id")
                if (
                    not isinstance(artifact_tile, str)
                    or artifact_tile not in tiles
                    or embedded_tile.get("crs") != campaign.horizontal_crs
                    or _tile_bounds_record(embedded_tile) != tiles[artifact_tile]
                ):
                    raise ValueError("campaign artifact embedded tile declaration is inconsistent")
            if artifact_tile is None and len(tiles) == 1:
                artifact_tile = next(iter(tiles))
            if not isinstance(artifact_tile, str) or artifact_tile not in tiles:
                raise ValueError("campaign artifact is not bound to a declared tile")
            bound = _BoundCampaignArtifact(
                selection_id=selection_id,
                tile_id=artifact_tile,
                tile_bounds=tiles[artifact_tile],
                retained_manifest_sha256=retained_sha,
                inventory_sha256=inventory_sha,
                root=bundle.retained_path.parent.resolve(),
                artifact=artifact,
                inventory_artifact=inventory_by_sha[sha],
            )
            all_artifacts[sha] = bound
            if sha in campaign.source_artifact_sha256s:
                selected_here += 1
        if selected_here:
            contributing_bundles += 1
    if contributing_bundles != len(declared):
        raise ValueError("every campaign bundle must contribute a contracted artifact")
    try:
        ordered = tuple(all_artifacts[sha] for sha in campaign.source_artifact_sha256s)
    except KeyError as error:
        raise ValueError(f"campaign artifact is absent from retained bundles: {error.args[0]}") from error

    for source in ordered:
        retained_artifact = source.artifact
        inventoried = source.inventory_artifact
        if (
            retained_artifact.get("year") != campaign.expected_acquisition_year
            or inventoried.get("year") != campaign.expected_acquisition_year
        ):
            raise ValueError("campaign artifact acquisition year differs from contract")
        if (
            retained_artifact.get("type") != campaign.expected_artifact_type
            or inventoried.get("type") != campaign.expected_artifact_type
        ):
            raise ValueError("campaign artifact type differs from contract")
        header = inventoried.get("header")
        if not isinstance(header, dict) or header.get("crsParseError") is not None:
            raise ValueError("campaign inventory lacks a valid parsed CRS")
        try:
            inventory_crs = pyproj.CRS.from_wkt(header["crsWkt"])
        except (KeyError, TypeError, pyproj.exceptions.CRSError) as error:
            raise ValueError("campaign inventory CRS cannot be parsed") from error
        _require_campaign_crs(inventory_crs, campaign, "campaign inventory")
        try:
            min_x, min_y = map(float, header["mins"][:2])
            max_x, max_y = map(float, header["maxs"][:2])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("campaign inventory lacks LAS header extents") from error
        owner_min_x, owner_min_y, owner_max_x, owner_max_y = source.tile_bounds
        if not (
            owner_min_x <= min_x < owner_max_x
            and owner_min_y <= min_y < owner_max_y
            and owner_min_x <= max_x < owner_max_x
            and owner_min_y <= max_y < owner_max_y
        ):
            raise ValueError("campaign inventory extent escapes half-open tile ownership")

    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            lx0, ly0, lx1, ly1 = left.tile_bounds
            rx0, ry0, rx1, ry1 = right.tile_bounds
            overlaps_interior = min(lx1, rx1) > max(lx0, rx0) and min(ly1, ry1) > max(ly0, ry0)
            if left.tile_id == right.tile_id or overlaps_interior:
                raise ValueError("campaign artifacts declare duplicate or spatially overlapping tiles")
    return ordered


def _campaign_source_digest(
    campaign: WaterCampaignContract, sources: Sequence[_BoundCampaignArtifact]
) -> str:
    identity = {
        "format": 1,
        "campaignId": campaign.campaign_id,
        "expectedAcquisitionYear": campaign.expected_acquisition_year,
        "expectedArtifactType": campaign.expected_artifact_type,
        "horizontalCrs": campaign.horizontal_crs,
        "verticalCrs": campaign.vertical_crs,
        "allowedPointSourceIds": list(campaign.allowed_point_source_ids),
        "classificationAuthority": campaign.classification_authority,
        "visibleWaterClass": campaign.visible_water_class,
        "groundClassNotWater": campaign.ground_class,
        "noiseClasses": list(campaign.noise_classes),
        "sources": [
            {
                "artifactSha256": sha,
                "selectionId": source.selection_id,
                "tileId": source.tile_id,
                "retainedManifestSha256": source.retained_manifest_sha256,
                "inventorySha256": source.inventory_sha256,
            }
            for sha, source in zip(campaign.source_artifact_sha256s, sources, strict=True)
        ],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(b"laas-als-water-campaign-sources-v1\0" + encoded)


def _qualification_digest(
    *,
    source_campaign_sha256: str,
    reach_id: str,
    polygon_sha256: str,
    centerline_sha256: str,
    observation_exclusion_sha256: str | None,
    discontinuity_free_reach: bool,
    capped_reach_geometry: bool,
) -> str:
    identity = {
        "format": 1,
        "method": _QUALIFICATION_METHOD,
        "sourceCampaignSha256": source_campaign_sha256,
        "reachId": reach_id,
        "waterPolygonWkbSha256": polygon_sha256,
        "centerlineWkbSha256": centerline_sha256,
        "observationExclusionWkbSha256": observation_exclusion_sha256,
        "topology": {
            "unbranchedSimpleCenterline": True,
            "discontinuityFreeReach": discontinuity_free_reach,
            "cappedReachGeometry": capped_reach_geometry,
        },
        "endpointCapToleranceM": _ENDPOINT_CAP_TOLERANCE_M,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(b"laas-als-water-qualification-v1\0" + encoded)


def qualify_als_flowing_water_campaign(
    *,
    bundles: Sequence[RetainedInventoryBundle],
    campaign: WaterCampaignContract,
    water_polygon: BaseGeometry,
    centerline: BaseGeometry,
    reach_id: str,
    discontinuity_free_reach: bool,
    capped_reach_geometry: bool,
    observation_exclusion_geometry: BaseGeometry | None = None,
    chunk_points: int = 1_000_000,
) -> AlsWaterCampaignEvidence:
    """Fit one campaign after concatenating qualified observations across its tiles."""
    if not reach_id or chunk_points < 1:
        raise ValueError("reach_id and a positive chunk_points are required")
    if discontinuity_free_reach is not True:
        raise ValueError(
            "flow-profile fitting requires an explicitly segmented discontinuity-free reach"
        )
    if capped_reach_geometry is not True:
        raise ValueError("flow-profile fitting requires explicitly capped reach geometry")
    polygon = _plan_geometry(water_polygon, ("Polygon", "MultiPolygon"), "water_polygon")
    line = _plan_geometry(centerline, ("LineString",), "centerline")
    exclusion = (
        None
        if observation_exclusion_geometry is None
        else _plan_geometry(
            observation_exclusion_geometry,
            ("Polygon", "MultiPolygon"),
            "observation_exclusion_geometry",
        )
    )
    if not line.is_simple or line.length < 1.0 or not polygon.covers(line):
        raise ValueError("centerline must be simple, at least 1 m long, and covered by water_polygon")
    stations = np.arange(0.0, np.floor(line.length) + 1.0, 1.0, dtype=np.float64)
    sources = _campaign_sources(bundles, campaign)
    source_campaign_sha = _campaign_source_digest(campaign, sources)
    polygon_wkb = shapely.to_wkb(polygon, byte_order=1, include_srid=False)
    line_wkb = shapely.to_wkb(line, byte_order=1, include_srid=False)
    exclusion_wkb = (
        None
        if exclusion is None
        else shapely.to_wkb(exclusion, byte_order=1, include_srid=False)
    )
    polygon_sha = _sha256_bytes(polygon_wkb)
    line_sha = _sha256_bytes(line_wkb)
    exclusion_sha = None if exclusion_wkb is None else _sha256_bytes(exclusion_wkb)
    qualification_sha = _qualification_digest(
        source_campaign_sha256=source_campaign_sha,
        reach_id=reach_id,
        polygon_sha256=polygon_sha,
        centerline_sha256=line_sha,
        observation_exclusion_sha256=exclusion_sha,
        discontinuity_free_reach=True,
        capped_reach_geometry=True,
    )
    station_parts: list[np.ndarray] = []
    height_parts: list[np.ndarray] = []
    artifact_evidence: list[CampaignArtifactEvidence] = []
    for sha, source in zip(campaign.source_artifact_sha256s, sources, strict=True):
        artifact = source.artifact
        relative = Path(artifact["relativePath"])
        source_path = (source.root / relative).resolve()
        if not source_path.is_relative_to(source.root) or not source_path.is_file():
            raise ValueError(f"retained artifact path is missing or escapes its root: {relative}")
        if source_path.stat().st_size != artifact["bytes"]:
            raise ValueError(f"retained artifact size differs for campaign source {sha}")
        streamed = _stream_qualified_artifact(
            source_path=source_path,
            source_artifact_sha256=sha,
            visible_water_class=campaign.visible_water_class,
            ground_class=campaign.ground_class,
            noise_classes=campaign.noise_classes,
            polygon=polygon,
            line=line,
            chunk_points=chunk_points,
            context=f"campaign {campaign.campaign_id} tile {source.tile_id}",
            tile_bounds=source.tile_bounds,
            campaign=campaign,
            observation_exclusion=exclusion,
        )
        station_parts.append(streamed.point_station_m)
        height_parts.append(streamed.point_y)
        artifact_evidence.append(
            CampaignArtifactEvidence(
                selection_id=source.selection_id,
                tile_id=source.tile_id,
                source_artifact_sha256=sha,
                source_filename=artifact["filename"],
                acquisition_year=campaign.expected_acquisition_year,
                artifact_type=campaign.expected_artifact_type,
                retained_manifest_sha256=source.retained_manifest_sha256,
                inventory_sha256=source.inventory_sha256,
                metrics=streamed.metrics,
            )
        )
    point_station = np.concatenate(station_parts)
    point_y = np.concatenate(height_parts)
    observation, sample_count = station_locations(point_station, point_y, stations)
    profile = fit_flow_profile(
        reach_id=reach_id,
        epoch_id=campaign.campaign_id,
        source_artifact_sha256=qualification_sha,
        station_m=stations,
        observation_y=observation,
    )
    return AlsWaterCampaignEvidence(
        reach_id=reach_id,
        discontinuity_free_reach=True,
        capped_reach_geometry=True,
        campaign=campaign,
        source_campaign_sha256=source_campaign_sha,
        qualification_sha256=qualification_sha,
        water_polygon_wkb_sha256=polygon_sha,
        centerline_wkb_sha256=line_sha,
        observation_exclusion_wkb_sha256=exclusion_sha,
        station_m=stations,
        observation_y=observation,
        sample_count=sample_count,
        artifacts=tuple(artifact_evidence),
        profile=profile,
    )


def _npy_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(output, np.asarray(array), allow_pickle=False)
    return output.getvalue()


def _deterministic_npz(arrays: Sequence[tuple[str, np.ndarray]]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, array in arrays:
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(array))
    return output.getvalue()


def encode_als_water_evidence(evidence: AlsWaterEvidence) -> EncodedAlsWaterEvidence:
    """Encode deterministic canonical metadata and array payloads for recipe binding."""
    arrays: list[tuple[str, np.ndarray]] = []
    epoch_metadata: list[dict[str, Any]] = []
    for index, epoch in enumerate(evidence.epochs):
        prefix = f"epoch_{index:03d}"
        accepted = (
            epoch.profile.accepted
            if isinstance(epoch.profile, FlowProfile)
            else np.zeros(epoch.station_m.shape, dtype=np.bool_)
        )
        water_y = (
            epoch.profile.water_y
            if isinstance(epoch.profile, FlowProfile)
            else np.full(epoch.station_m.shape, np.nan, dtype="<f8")
        )
        epoch_arrays = (
            (f"{prefix}_station_m", np.asarray(epoch.station_m, dtype="<f8")),
            (f"{prefix}_observation_y", np.asarray(epoch.observation_y, dtype="<f8")),
            (f"{prefix}_sample_count", np.asarray(epoch.sample_count, dtype="<i4")),
            (f"{prefix}_accepted", np.asarray(accepted, dtype=np.bool_)),
            (f"{prefix}_water_y", np.asarray(water_y, dtype="<f8")),
        )
        arrays.extend(epoch_arrays)
        if isinstance(epoch.profile, FlowProfile):
            decision: dict[str, Any] = {
                "kind": "qualified_profile",
                "orientation": epoch.profile.orientation.value,
                "longestMissingSpanM": epoch.profile.longest_missing_span_m,
            }
        else:
            decision = {
                "kind": "abstained",
                "reason": epoch.profile.reason.value,
                "detail": epoch.profile.detail,
            }
        epoch_metadata.append(
            {
                "epochId": epoch.epoch_id,
                "sourceArtifactSha256": epoch.source_artifact_sha256,
                "sourceFilename": epoch.source_filename,
                "classSemantics": {
                    "classificationAuthority": epoch.class_semantics.classification_authority,
                    "visibleWaterClass": epoch.class_semantics.visible_water_class,
                    "groundClassNotWater": epoch.class_semantics.ground_class,
                    "noiseClasses": list(epoch.class_semantics.noise_classes),
                },
                "metrics": {
                    "decodedPoints": epoch.metrics.decoded_points,
                    "visibleWaterLabels": epoch.metrics.visible_water_labels,
                    "groundLabelsNotUsed": epoch.metrics.ground_labels_not_used,
                    "noiseLabelsExcluded": epoch.metrics.noise_labels_excluded,
                    "withheldWaterExcluded": epoch.metrics.withheld_water_excluded,
                    "overlapWaterExcluded": epoch.metrics.overlap_water_excluded,
                    "eligibleWaterPoints": epoch.metrics.eligible_water_points,
                    "insideWaterPolygon": epoch.metrics.inside_water_polygon,
                    "outsideWaterPolygon": epoch.metrics.outside_water_polygon,
                    "observationGeometryExcluded": epoch.metrics.observation_geometry_excluded,
                    "pointSourceCounts": [list(item) for item in epoch.metrics.point_source_counts],
                },
                "decision": decision,
                "arrays": {
                    name: {
                        "dtype": np.asarray(array).dtype.str,
                        "shape": list(np.asarray(array).shape),
                        "sha256": _sha256_bytes(_npy_bytes(array)),
                    }
                    for name, array in epoch_arrays
                },
            }
        )
    npz = _deterministic_npz(arrays)
    metadata = {
        "format": 1,
        "scientificRole": "per_epoch_visible_water_evidence",
        "sourceSelectionPolicy": None,
        "reachId": evidence.reach_id,
        "discontinuityFreeReach": evidence.discontinuity_free_reach,
        "retainedManifestSha256": evidence.retained_manifest_sha256,
        "inventorySha256": evidence.inventory_sha256,
        "waterPolygonWkbSha256": evidence.water_polygon_wkb_sha256,
        "centerlineWkbSha256": evidence.centerline_wkb_sha256,
        "arraysNpzSha256": _sha256_bytes(npz),
        "epochs": epoch_metadata,
    }
    metadata_json = (json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n").encode()
    content_sha = _sha256_bytes(b"laas-als-water-evidence-v1\0" + metadata_json + npz)
    return EncodedAlsWaterEvidence(metadata_json, npz, content_sha)


def encode_als_water_campaign_evidence(
    evidence: AlsWaterCampaignEvidence,
) -> EncodedAlsWaterEvidence:
    """Encode a deterministic multi-artifact campaign evidence record."""
    accepted = (
        evidence.profile.accepted
        if isinstance(evidence.profile, FlowProfile)
        else np.zeros(evidence.station_m.shape, dtype=np.bool_)
    )
    water_y = (
        evidence.profile.water_y
        if isinstance(evidence.profile, FlowProfile)
        else np.full(evidence.station_m.shape, np.nan, dtype="<f8")
    )
    arrays = (
        ("campaign_station_m", np.asarray(evidence.station_m, dtype="<f8")),
        ("campaign_observation_y", np.asarray(evidence.observation_y, dtype="<f8")),
        ("campaign_sample_count", np.asarray(evidence.sample_count, dtype="<i4")),
        ("campaign_accepted", np.asarray(accepted, dtype=np.bool_)),
        ("campaign_water_y", np.asarray(water_y, dtype="<f8")),
    )
    if isinstance(evidence.profile, FlowProfile):
        decision: dict[str, Any] = {
            "kind": "qualified_profile",
            "orientation": evidence.profile.orientation.value,
            "longestMissingSpanM": evidence.profile.longest_missing_span_m,
        }
    else:
        decision = {
            "kind": "abstained",
            "reason": evidence.profile.reason.value,
            "detail": evidence.profile.detail,
        }
    npz = _deterministic_npz(arrays)
    metadata = {
        "format": 1,
        "scientificRole": "multi_tile_campaign_visible_water_evidence",
        "sourceSelectionPolicy": None,
        "reachId": evidence.reach_id,
        "discontinuityFreeReach": evidence.discontinuity_free_reach,
        "campaign": {
            "campaignId": evidence.campaign.campaign_id,
            "sourceCampaignSha256": evidence.source_campaign_sha256,
            "qualificationSha256": evidence.qualification_sha256,
            "qualificationMethod": _QUALIFICATION_METHOD,
            "orderedSourceArtifactSha256s": list(
                evidence.campaign.source_artifact_sha256s
            ),
            "expectedAcquisitionYear": evidence.campaign.expected_acquisition_year,
            "expectedArtifactType": evidence.campaign.expected_artifact_type,
            "horizontalCrs": evidence.campaign.horizontal_crs,
            "verticalCrs": evidence.campaign.vertical_crs,
            "allowedPointSourceIds": list(evidence.campaign.allowed_point_source_ids),
            "classSemantics": {
                "classificationAuthority": evidence.campaign.classification_authority,
                "visibleWaterClass": evidence.campaign.visible_water_class,
                "groundClassNotWater": evidence.campaign.ground_class,
                "noiseClasses": list(evidence.campaign.noise_classes),
            },
            "decision": decision,
        },
        "waterPolygonWkbSha256": evidence.water_polygon_wkb_sha256,
        "centerlineWkbSha256": evidence.centerline_wkb_sha256,
        "observationExclusionWkbSha256": evidence.observation_exclusion_wkb_sha256,
        "topology": {
            "unbranchedSimpleCenterline": True,
            "discontinuityFreeReach": evidence.discontinuity_free_reach,
            "cappedReachGeometry": evidence.capped_reach_geometry,
            "endpointCapToleranceM": _ENDPOINT_CAP_TOLERANCE_M,
        },
        "sources": [
            {
                "selectionId": source.selection_id,
                "tileId": source.tile_id,
                "sourceArtifactSha256": source.source_artifact_sha256,
                "sourceFilename": source.source_filename,
                "acquisitionYear": source.acquisition_year,
                "artifactType": source.artifact_type,
                "retainedManifestSha256": source.retained_manifest_sha256,
                "inventorySha256": source.inventory_sha256,
                "metrics": {
                    "decodedPoints": source.metrics.decoded_points,
                    "visibleWaterLabels": source.metrics.visible_water_labels,
                    "groundLabelsNotUsed": source.metrics.ground_labels_not_used,
                    "noiseLabelsExcluded": source.metrics.noise_labels_excluded,
                    "withheldWaterExcluded": source.metrics.withheld_water_excluded,
                    "overlapWaterExcluded": source.metrics.overlap_water_excluded,
                    "eligibleWaterPoints": source.metrics.eligible_water_points,
                    "insideWaterPolygon": source.metrics.inside_water_polygon,
                    "outsideWaterPolygon": source.metrics.outside_water_polygon,
                    "observationGeometryExcluded": source.metrics.observation_geometry_excluded,
                    "pointSourceCounts": [
                        list(item) for item in source.metrics.point_source_counts
                    ],
                },
            }
            for source in evidence.artifacts
        ],
        "arraysNpzSha256": _sha256_bytes(npz),
        "arrays": {
            name: {
                "dtype": np.asarray(array).dtype.str,
                "shape": list(np.asarray(array).shape),
                "sha256": _sha256_bytes(_npy_bytes(array)),
            }
            for name, array in arrays
        },
    }
    metadata_json = (json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n").encode()
    content_sha = _sha256_bytes(b"laas-als-water-campaign-v1\0" + metadata_json + npz)
    return EncodedAlsWaterEvidence(metadata_json, npz, content_sha)
