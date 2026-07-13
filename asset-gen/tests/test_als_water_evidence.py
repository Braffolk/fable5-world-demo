import hashlib
import json
from pathlib import Path

import laspy
import numpy as np
from pyproj import CRS
from pyproj.crs import CompoundCRS
import pytest
from shapely.geometry import LineString, box

from assetgen.evidence.als_water import (
    EpochClassSemantics,
    RetainedInventoryBundle,
    WaterCampaignContract,
    encode_als_water_campaign_evidence,
    encode_als_water_evidence,
    qualify_als_flowing_water,
    qualify_als_flowing_water_campaign,
)
from assetgen.terrain.repair.model import FlowProfile


_TEST_CRS = CompoundCRS("test EST97 + EVRF2007", [CRS.from_epsg(3301), CRS.from_epsg(5621)])
_TEST_YEAR = 2025
_TEST_TYPE = "lidar_laz_test"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_campaign_bundle(
    root: Path,
    *,
    tile_id: str,
    bounds: tuple[float, float, float, float],
    station_start: int,
    station_stop: int,
    point_source_id: int = 321,
    extra_x: tuple[float, ...] = (),
) -> tuple[RetainedInventoryBundle, str]:
    source = root / "raw" / f"{tile_id}.laz"
    source.parent.mkdir(parents=True)
    stations = np.arange(station_start, station_stop + 1, dtype=np.float64)
    x = np.repeat(stations, 4) + np.tile((-0.3, -0.1, 0.1, 0.3), stations.size)
    y = np.tile((-0.1, 0.1, -0.1, 0.1), stations.size)
    if extra_x:
        x = np.concatenate((x, np.asarray(extra_x, dtype=np.float64)))
        y = np.concatenate((y, np.zeros(len(extra_x), dtype=np.float64)))
    header = laspy.LasHeader(point_format=8, version="1.4")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.add_crs(_TEST_CRS)
    points = laspy.LasData(header)
    points.x, points.y, points.z = x, y, 30.0 - 0.01 * x
    points.classification = np.full(x.size, 9, dtype=np.uint8)
    points.point_source_id = np.full(x.size, point_source_id, dtype=np.uint16)
    points.write(source)
    source_sha = _sha256(source)
    artifact = {
        "filename": source.name,
        "relativePath": source.relative_to(root).as_posix(),
        "bytes": source.stat().st_size,
        "sha256": source_sha,
        "year": _TEST_YEAR,
        "type": _TEST_TYPE,
    }
    retained = {
        "complete": True,
        "missingFiles": [],
        "selectionId": f"selection-{tile_id}",
        "tile": {
            "id": tile_id,
            "crs": "EPSG:3301",
            "bounds": {
                "min_x": bounds[0],
                "min_y": bounds[1],
                "max_x_exclusive": bounds[2],
                "max_y_exclusive": bounds[3],
            },
        },
        "artifacts": [artifact],
    }
    retained_path = root / "retained.json"
    retained_path.write_text(json.dumps(retained, sort_keys=True), encoding="utf-8")
    with laspy.open(source) as reader:
        inventory_artifact = dict(artifact)
        inventory_artifact["header"] = {
            "mins": list(map(float, reader.header.mins)),
            "maxs": list(map(float, reader.header.maxs)),
            "crsWkt": reader.header.parse_crs().to_wkt(),
            "crsParseError": None,
        }
    inventory = {
        "selectionId": retained["selectionId"],
        "retainedManifestSha256": _sha256(retained_path),
        "artifacts": [inventory_artifact],
    }
    inventory_path = root / "inventory.json"
    inventory_path.write_text(json.dumps(inventory, sort_keys=True), encoding="utf-8")
    return RetainedInventoryBundle(retained_path, inventory_path), source_sha


def _campaign_contract(
    artifacts: tuple[str, ...],
    *,
    allowed_sources: tuple[int, ...] = (321,),
    year: int = _TEST_YEAR,
    artifact_type: str = _TEST_TYPE,
    vertical_crs: str = "EPSG:5621",
) -> WaterCampaignContract:
    return WaterCampaignContract(
        campaign_id="synthetic-campaign",
        source_artifact_sha256s=artifacts,
        expected_acquisition_year=year,
        expected_artifact_type=artifact_type,
        horizontal_crs="EPSG:3301",
        vertical_crs=vertical_crs,
        allowed_point_source_ids=allowed_sources,
        classification_authority="synthetic campaign fixture",
        visible_water_class=9,
        ground_class=2,
        noise_classes=(7, 18),
    )


def _merge_campaign_bundle(
    root: Path,
    entries: list[tuple[RetainedInventoryBundle, str]],
) -> RetainedInventoryBundle:
    retained_documents = [json.loads(bundle.retained_path.read_text()) for bundle, _ in entries]
    artifacts = []
    tiles = []
    for document in retained_documents:
        artifact = dict(document["artifacts"][0])
        artifact["relativePath"] = str(
            (Path(document["artifacts"][0]["relativePath"])).as_posix()
        )
        artifact["tile"] = document["tile"]
        source = entries[len(artifacts)][0].retained_path.parent / artifact["relativePath"]
        destination = root / artifact["relativePath"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        artifacts.append(artifact)
        tiles.append(document["tile"])
    retained = {
        "complete": True,
        "missingFiles": [],
        "selectionId": "multi-tile-selection",
        "tiles": tiles,
        "artifacts": artifacts,
    }
    retained_path = root / "retained.json"
    retained_path.write_text(json.dumps(retained, sort_keys=True), encoding="utf-8")
    inventory = {
        "selectionId": retained["selectionId"],
        "retainedManifestSha256": _sha256(retained_path),
        "artifacts": [
            next(
                item
                for item in json.loads(bundle.inventory_path.read_text())["artifacts"]
                if item["sha256"] == sha
            )
            for bundle, sha in entries
        ],
    }
    inventory_path = root / "inventory.json"
    inventory_path.write_text(json.dumps(inventory, sort_keys=True), encoding="utf-8")
    return RetainedInventoryBundle(retained_path, inventory_path)


def test_qualifies_only_visible_water_and_encodes_deterministically(tmp_path: Path) -> None:
    root = tmp_path / "retained"
    source = root / "raw" / "epoch.laz"
    source.parent.mkdir(parents=True)
    stations = np.arange(41, dtype=np.float64)
    x = np.repeat(stations, 4) + np.tile((-0.3, -0.1, 0.1, 0.3), stations.size)
    y = np.tile((-0.1, 0.1, -0.1, 0.1), stations.size)
    z = 20.0 - 0.01 * x
    classifications = np.full(x.size, 9, dtype=np.uint8)
    withheld = np.zeros(x.size, dtype=bool)
    overlap = np.zeros(x.size, dtype=bool)
    withheld[4] = True
    overlap[8] = True
    # Ground, noise, and an out-of-polygon water label must remain factual metrics only.
    x = np.concatenate((x, [20.0, 20.0, 20.0, 20.0]))
    y = np.concatenate((y, [0.0, 0.0, 0.0, 4.0]))
    z = np.concatenate((z, [99.0, 88.0, 77.0, 66.0]))
    classifications = np.concatenate((classifications, [2, 7, 18, 9])).astype(np.uint8)
    withheld = np.concatenate((withheld, [False] * 4))
    overlap = np.concatenate((overlap, [False] * 4))

    header = laspy.LasHeader(point_format=8, version="1.4")
    header.scales = np.array([0.01, 0.01, 0.01])
    points = laspy.LasData(header)
    points.x, points.y, points.z = x, y, z
    points.classification = classifications
    points.withheld = withheld
    points.overlap = overlap
    points.point_source_id = np.full(x.size, 321, dtype=np.uint16)
    points.write(source)
    source_sha = _sha256(source)

    retained = {
        "complete": True,
        "missingFiles": [],
        "selectionId": "synthetic-contract",
        "artifacts": [
            {
                "filename": source.name,
                "relativePath": source.relative_to(root).as_posix(),
                "bytes": source.stat().st_size,
                "sha256": source_sha,
            }
        ],
    }
    retained_path = root / "retained.json"
    retained_path.write_text(json.dumps(retained, sort_keys=True), encoding="utf-8")
    inventory = {
        "selectionId": "synthetic-contract",
        "retainedManifestSha256": _sha256(retained_path),
        "artifacts": retained["artifacts"],
    }
    inventory_path = root / "inventory.json"
    inventory_path.write_text(json.dumps(inventory, sort_keys=True), encoding="utf-8")
    semantics = EpochClassSemantics(
        epoch_id="flight-a",
        source_artifact_sha256=source_sha,
        classification_authority="synthetic LAS 1.4 fixture",
        visible_water_class=9,
        ground_class=2,
        noise_classes=(7, 18),
    )

    evidence = qualify_als_flowing_water(
        retained_path=retained_path,
        inventory_path=inventory_path,
        water_polygon=box(-1.0, -1.0, 41.0, 1.0),
        centerline=LineString([(0.0, 0.0), (40.0, 0.0)]),
        reach_id="synthetic-reach",
        discontinuity_free_reach=True,
        epochs=[semantics],
        chunk_points=17,
    )
    epoch = evidence.epochs[0]
    assert isinstance(epoch.profile, FlowProfile)
    assert epoch.metrics.ground_labels_not_used == 1
    assert epoch.metrics.noise_labels_excluded == 2
    assert epoch.metrics.withheld_water_excluded == 1
    assert epoch.metrics.overlap_water_excluded == 1
    assert epoch.metrics.outside_water_polygon == 1
    assert epoch.metrics.inside_water_polygon == 162
    assert epoch.metrics.point_source_counts == ((321, 162),)
    assert np.nanmax(epoch.observation_y) < 21.0
    encoded_a = encode_als_water_evidence(evidence)
    encoded_b = encode_als_water_evidence(evidence)
    assert encoded_a == encoded_b
    assert json.loads(encoded_a.metadata_json)["sourceSelectionPolicy"] is None
    assert hashlib.sha256(encoded_a.arrays_npz).hexdigest() == json.loads(
        encoded_a.metadata_json
    )["arraysNpzSha256"]

    with pytest.raises(ValueError, match="only LAS class 9"):
        EpochClassSemantics(
            epoch_id="invalid",
            source_artifact_sha256=source_sha,
            classification_authority="invalid fixture",
            visible_water_class=2,
            ground_class=2,
            noise_classes=(7, 18),
        )


def test_campaign_concatenates_tiles_and_preserves_nested_source_identity(
    tmp_path: Path,
) -> None:
    bundle_a, sha_a = _write_campaign_bundle(
        tmp_path / "tile-a",
        tile_id="tile-a",
        bounds=(-0.3, -1.0, 40.5, 1.0),
        station_start=0,
        station_stop=40,
    )
    bundle_b, sha_b = _write_campaign_bundle(
        tmp_path / "tile-b",
        tile_id="tile-b",
        bounds=(40.5, -1.0, 81.0, 1.0),
        station_start=41,
        station_stop=80,
    )
    campaign = _campaign_contract((sha_b, sha_a))
    evidence = qualify_als_flowing_water_campaign(
        bundles=[bundle_a, bundle_b],
        campaign=campaign,
        water_polygon=box(-1.0, -1.0, 81.0, 1.0),
        centerline=LineString([(0.0, 0.0), (80.0, 0.0)]),
        reach_id="two-tile-reach",
        discontinuity_free_reach=True,
        capped_reach_geometry=True,
        observation_exclusion_geometry=box(19.5, -0.5, 20.5, 0.5),
        chunk_points=19,
    )
    assert isinstance(evidence.profile, FlowProfile)
    assert evidence.profile.source_artifact_sha256 == evidence.qualification_sha256
    assert evidence.source_campaign_sha256 != evidence.qualification_sha256
    assert tuple(source.source_artifact_sha256 for source in evidence.artifacts) == (sha_b, sha_a)
    assert tuple(source.tile_id for source in evidence.artifacts) == ("tile-b", "tile-a")
    assert all(source.metrics.point_source_counts == ((321, 160),) for source in evidence.artifacts)
    assert np.all(evidence.sample_count >= 4)
    assert sum(source.metrics.observation_geometry_excluded for source in evidence.artifacts) == 4
    encoded_a = encode_als_water_campaign_evidence(evidence)
    encoded_b = encode_als_water_campaign_evidence(evidence)
    assert encoded_a == encoded_b
    metadata = json.loads(encoded_a.metadata_json)
    assert metadata["sourceSelectionPolicy"] is None
    assert metadata["campaign"]["orderedSourceArtifactSha256s"] == [sha_b, sha_a]
    assert metadata["observationExclusionWkbSha256"] is not None
    assert [source["tileId"] for source in metadata["sources"]] == ["tile-b", "tile-a"]
    assert len({source["retainedManifestSha256"] for source in metadata["sources"]}) == 2
    assert len({source["inventorySha256"] for source in metadata["sources"]}) == 2
    alternate_reach = qualify_als_flowing_water_campaign(
        bundles=[bundle_a, bundle_b],
        campaign=campaign,
        water_polygon=box(-1.0, -1.0, 81.0, 1.0),
        centerline=LineString([(0.0, 0.0), (80.0, 0.0)]),
        reach_id="same-sources-different-reach",
        discontinuity_free_reach=True,
        capped_reach_geometry=True,
        observation_exclusion_geometry=box(19.5, -0.5, 20.5, 0.5),
        chunk_points=19,
    )
    assert alternate_reach.source_campaign_sha256 == evidence.source_campaign_sha256
    assert alternate_reach.qualification_sha256 != evidence.qualification_sha256

    with pytest.raises(ValueError, match="duplicate source declarations"):
        qualify_als_flowing_water_campaign(
            bundles=[bundle_a, bundle_a],
            campaign=_campaign_contract((sha_a,)),
            water_polygon=box(-1.0, -1.0, 41.0, 1.0),
            centerline=LineString([(0.0, 0.0), (40.0, 0.0)]),
            reach_id="invalid-reach",
            discontinuity_free_reach=True,
            capped_reach_geometry=True,
        )


def test_campaign_accepts_multi_tile_retained_selection(tmp_path: Path) -> None:
    bundle_a, sha_a = _write_campaign_bundle(
        tmp_path / "source-a",
        tile_id="tile-a",
        bounds=(-0.3, -1.0, 40.5, 1.0),
        station_start=0,
        station_stop=40,
    )
    bundle_b, sha_b = _write_campaign_bundle(
        tmp_path / "source-b",
        tile_id="tile-b",
        bounds=(40.5, -1.0, 81.0, 1.0),
        station_start=41,
        station_stop=80,
    )
    merged = _merge_campaign_bundle(
        tmp_path / "merged-selection", [(bundle_a, sha_a), (bundle_b, sha_b)]
    )
    evidence = qualify_als_flowing_water_campaign(
        bundles=[merged],
        campaign=_campaign_contract((sha_a, sha_b)),
        water_polygon=box(-1.0, -1.0, 81.0, 1.0),
        centerline=LineString([(0.0, 0.0), (80.0, 0.0)]),
        reach_id="multi-tile-selection-reach",
        discontinuity_free_reach=True,
        capped_reach_geometry=True,
        chunk_points=19,
    )
    assert isinstance(evidence.profile, FlowProfile)
    assert tuple(source.tile_id for source in evidence.artifacts) == ("tile-a", "tile-b")
    assert len({source.retained_manifest_sha256 for source in evidence.artifacts}) == 1


@pytest.mark.parametrize(
    ("contract_kwargs", "match"),
    [
        ({"year": _TEST_YEAR - 1}, "acquisition year"),
        ({"artifact_type": "different_campaign_type"}, "artifact type"),
        ({"vertical_crs": "EPSG:5703"}, "CRS authorities"),
        ({"allowed_sources": (999,)}, "unexpected in-reach point_source_ids"),
    ],
)
def test_campaign_rejects_cross_campaign_semantics(
    tmp_path: Path, contract_kwargs: dict, match: str
) -> None:
    bundle, sha = _write_campaign_bundle(
        tmp_path / "source",
        tile_id="tile-a",
        bounds=(-1.0, -1.0, 41.0, 1.0),
        station_start=0,
        station_stop=40,
    )
    with pytest.raises(ValueError, match=match):
        qualify_als_flowing_water_campaign(
            bundles=[bundle],
            campaign=_campaign_contract((sha,), **contract_kwargs),
            water_polygon=box(-1.0, -1.0, 41.0, 1.0),
            centerline=LineString([(0.0, 0.0), (40.0, 0.0)]),
            reach_id="semantic-rejection",
            discontinuity_free_reach=True,
            capped_reach_geometry=True,
        )


def test_campaign_rejects_actual_las_max_edge_outside_half_open_owner(
    tmp_path: Path,
) -> None:
    bundle, sha = _write_campaign_bundle(
        tmp_path / "source",
        tile_id="tile-a",
        bounds=(-1.0, -1.0, 40.5, 1.0),
        station_start=0,
        station_stop=40,
        extra_x=(40.5,),
    )
    # Make inventory metadata appear owned; the independent real LAS header check must still fail.
    inventory = json.loads(bundle.inventory_path.read_text())
    inventory["artifacts"][0]["header"]["maxs"][0] = 40.49
    bundle.inventory_path.write_text(json.dumps(inventory, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="LAS header extent escapes half-open"):
        qualify_als_flowing_water_campaign(
            bundles=[bundle],
            campaign=_campaign_contract((sha,)),
            water_polygon=box(-1.0, -1.0, 41.0, 1.0),
            centerline=LineString([(0.0, 0.0), (40.0, 0.0)]),
            reach_id="owner-rejection",
            discontinuity_free_reach=True,
            capped_reach_geometry=True,
        )


def test_campaign_rejects_uncapped_endpoint_evidence(tmp_path: Path) -> None:
    bundle, sha = _write_campaign_bundle(
        tmp_path / "source",
        tile_id="tile-a",
        bounds=(-6.0, -1.0, 41.0, 1.0),
        station_start=0,
        station_stop=40,
        extra_x=(-5.0,),
    )
    with pytest.raises(ValueError, match="beyond a reach endpoint cap"):
        qualify_als_flowing_water_campaign(
            bundles=[bundle],
            campaign=_campaign_contract((sha,)),
            water_polygon=box(-6.0, -1.0, 41.0, 1.0),
            centerline=LineString([(0.0, 0.0), (40.0, 0.0)]),
            reach_id="cap-rejection",
            discontinuity_free_reach=True,
            capped_reach_geometry=True,
        )
