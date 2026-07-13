"""Run the provenance-bound Stage-1 Taevaskoda/Ahja water campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyogrio
import shapely
from shapely.geometry.base import BaseGeometry
from shapely.ops import substring

from ..config import ASSET_GEN_ROOT, DATA_WORK
from ..evidence.als_water import (
    RetainedInventoryBundle,
    WaterCampaignContract,
    encode_als_water_campaign_evidence,
    qualify_als_flowing_water_campaign,
)
from ..evidence.reach import cap_water_polygon_to_centerline
from ..terrain.repair.model import FlowProfile

_DOMAIN = b"laas-taevaskoda-ahja-structural-pilot-v1\0"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _geometry_bytes(geometry: BaseGeometry) -> bytes:
    return shapely.to_wkb(geometry, byte_order=1, include_srid=False)


def _geometry_sha256(geometry: BaseGeometry) -> str:
    return _sha256_bytes(_geometry_bytes(geometry))


def _asset_path(relative: str) -> Path:
    path = (ASSET_GEN_ROOT / relative).resolve()
    if not path.is_relative_to(ASSET_GEN_ROOT):
        raise ValueError(f"configured path escapes asset-gen root: {relative}")
    return path


def _require_sha256(actual: str, expected: str, context: str) -> None:
    if actual != expected:
        raise ValueError(f"{context} SHA-256 differs: expected {expected}, got {actual}")


def _read_exact_etak_feature(
    source: Path,
    *,
    layer: str,
    etak_id: int,
    expected_kkr_code: str,
    expected_name: str,
) -> BaseGeometry:
    metadata, _, geometry_wkb, fields = pyogrio.raw.read(
        source,
        layer=layer,
        columns=["etak_id", "kkr_kood", "nimetus"],
        where=f"etak_id = {int(etak_id)}",
    )
    if metadata.get("crs") != "EPSG:3301" or len(geometry_wkb) != 1:
        raise ValueError(f"ETAK {layer}/{etak_id} is not one EPSG:3301 feature")
    values = dict(zip(metadata["fields"], fields, strict=True))
    if (
        int(values["etak_id"][0]) != etak_id
        or values["kkr_kood"][0] != expected_kkr_code
        or values["nimetus"][0] != expected_name
    ):
        raise ValueError(f"ETAK identity fields differ for {layer}/{etak_id}")
    geometry = shapely.force_2d(shapely.from_wkb(geometry_wkb[0]))
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError(f"ETAK geometry is empty or invalid for {layer}/{etak_id}")
    return geometry


@dataclass(frozen=True)
class _SnapResult:
    geometry: BaseGeometry
    tolerance_m: float
    missing_before_m: float
    hausdorff_displacement_m: float
    symmetric_difference_m2: float


def _close_coverage_sliver(polygon: BaseGeometry, line: BaseGeometry, ulps: int) -> _SnapResult:
    """Snap roundoff-only boundary vertices to source line vertices, or fail closed."""
    if ulps < 1:
        raise ValueError("coverage snap ULP multiplier must be positive")
    scale = max(abs(value) for value in (*polygon.bounds, *line.bounds))
    tolerance = float(ulps * np.spacing(scale))
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("coverage snap tolerance is not finite and positive")
    missing_before = float(line.difference(polygon).length)
    if missing_before > tolerance:
        raise ValueError(
            f"line escapes capped polygon by {missing_before:.17g} m, beyond ULP tolerance"
        )
    snapped = shapely.normalize(shapely.snap(polygon, line, tolerance))
    displacement = float(polygon.hausdorff_distance(snapped))
    symmetric_difference = float(polygon.symmetric_difference(snapped).area)
    area_limit = tolerance * max(1.0, float(polygon.length))
    if (
        snapped.geom_type != "Polygon"
        or not snapped.is_valid
        or displacement > tolerance
        or symmetric_difference > area_limit
        or not snapped.covers(line)
        or line.difference(snapped).length != 0.0
    ):
        raise ValueError("ULP-bounded snap did not produce exact, valid line coverage")
    return _SnapResult(
        geometry=snapped,
        tolerance_m=tolerance,
        missing_before_m=missing_before,
        hausdorff_displacement_m=displacement,
        symmetric_difference_m2=symmetric_difference,
    )


def _write_immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"immutable pilot artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_campaign(config_path: Path) -> Path:
    config_path = Path(config_path).resolve()
    raw = json.loads(config_path.read_bytes())
    if raw.get("format") != 1 or raw.get("pilotId") != "taevaskoda-ahja-stage1":
        raise ValueError("unsupported Taevaskoda structural pilot config")

    etak = raw["etak"]
    etak_path = _asset_path(etak["path"])
    _require_sha256(_sha256_file(etak_path), etak["sourceSha256"], "ETAK dataset")
    centerline_spec = etak["centerline"]
    polygon_spec = etak["waterPolygon"]
    full_line = _read_exact_etak_feature(
        etak_path,
        layer=centerline_spec["layer"],
        etak_id=int(centerline_spec["etakId"]),
        expected_kkr_code=centerline_spec["kkrCode"],
        expected_name=centerline_spec["name"],
    )
    full_polygon = _read_exact_etak_feature(
        etak_path,
        layer=polygon_spec["layer"],
        etak_id=int(polygon_spec["etakId"]),
        expected_kkr_code=polygon_spec["kkrCode"],
        expected_name=polygon_spec["name"],
    )
    expected_geometry = raw["expectedGeometrySha256"]
    _require_sha256(
        _geometry_sha256(full_line), expected_geometry["forced2dCenterline"], "centerline"
    )
    _require_sha256(
        _geometry_sha256(full_polygon),
        expected_geometry["forced2dWaterPolygon"],
        "water polygon",
    )

    full_cap = cap_water_polygon_to_centerline(full_polygon, full_line)
    _require_sha256(_geometry_sha256(full_cap), expected_geometry["fullAuditedCap"], "full cap")
    downstream_start = float(raw["reach"]["downstreamStartFullStationM"])
    downstream_line = shapely.force_2d(substring(full_line, downstream_start, full_line.length))
    if downstream_line.geom_type != "LineString" or downstream_line.length < 1.0:
        raise ValueError("audited downstream substring is not a usable LineString")
    _require_sha256(
        _geometry_sha256(downstream_line),
        expected_geometry["downstreamCenterline"],
        "downstream line",
    )
    raw_downstream_cap = cap_water_polygon_to_centerline(full_cap, downstream_line)
    _require_sha256(
        _geometry_sha256(raw_downstream_cap),
        expected_geometry["downstreamCapBeforeUlpSnap"],
        "raw downstream cap",
    )
    snap = _close_coverage_sliver(
        raw_downstream_cap,
        downstream_line,
        int(raw["reach"]["coverageSnapUlps"]),
    )
    downstream_cap = snap.geometry
    _require_sha256(
        _geometry_sha256(downstream_cap), expected_geometry["downstreamCap"], "downstream cap"
    )

    source = raw["sourceCampaign"]
    semantics = source["heightSemantics"]
    contract_identity = (
        int(source["year"]),
        source["artifactType"],
        source["horizontalCrs"],
        semantics["authoritativeHeightSystem"],
        semantics["authoritativeVerticalCrs"],
        semantics["observedSourceHeaderVerticalCrs"],
        tuple(source["allowedPointSourceIds"]),
    )
    if contract_identity != (
        2019,
        "lidar_laz_tava",
        "EPSG:3301",
        "EH2000",
        "EPSG:9663",
        "EPSG:5621",
        (19317,),
    ):
        raise ValueError("Taevaskoda Stage-1 source campaign contract differs")
    if (
        semantics["relationshipPolicy"]
        != "provider_declared_closest_epsg_encoding_no_numeric_transform"
    ):
        raise ValueError("unsupported source-header/authoritative-height relationship")
    bundles = tuple(
        RetainedInventoryBundle(
            _asset_path(item["retainedPath"]),
            _asset_path(item["inventoryPath"]),
        )
        for item in source["bundles"]
    )
    campaign = WaterCampaignContract(
        campaign_id=source["campaignId"],
        source_artifact_sha256s=tuple(source["orderedSourceArtifactSha256s"]),
        expected_acquisition_year=int(source["year"]),
        expected_artifact_type=source["artifactType"],
        horizontal_crs=source["horizontalCrs"],
        vertical_crs=semantics["observedSourceHeaderVerticalCrs"],
        allowed_point_source_ids=tuple(source["allowedPointSourceIds"]),
        classification_authority=source["classificationAuthorityUrl"],
        visible_water_class=int(source["visibleWaterClass"]),
        ground_class=int(source["groundClassNotWater"]),
        noise_classes=tuple(source["noiseClasses"]),
    )
    evidence = qualify_als_flowing_water_campaign(
        bundles=bundles,
        campaign=campaign,
        water_polygon=downstream_cap,
        centerline=downstream_line,
        reach_id=raw["reach"]["reachId"],
        discontinuity_free_reach=True,
        capped_reach_geometry=True,
        observation_exclusion_geometry=None,
    )
    if not isinstance(evidence.profile, FlowProfile):
        raise ValueError(f"audited downstream campaign abstained: {evidence.profile.detail}")
    encoded = encode_als_water_campaign_evidence(evidence)
    encoded_metadata = json.loads(encoded.metadata_json)
    if encoded_metadata["observationExclusionWkbSha256"] is not None:
        raise ValueError("Stage-1 pilot must not invent bridge exclusion geometry")

    config_identity = _canonical_json(raw)
    geometry_payloads = {
        "full-centerline.wkb": _geometry_bytes(full_line),
        "full-water-cap.wkb": _geometry_bytes(full_cap),
        "downstream-centerline.wkb": _geometry_bytes(downstream_line),
        "downstream-water-cap.wkb": _geometry_bytes(downstream_cap),
    }
    digest = hashlib.sha256(_DOMAIN)
    digest.update(config_identity)
    for name, payload in geometry_payloads.items():
        digest.update(name.encode() + b"\0" + payload)
    digest.update(encoded.metadata_json)
    digest.update(encoded.arrays_npz)
    build_sha = digest.hexdigest()
    output_root = DATA_WORK / "terrain-repair" / raw["pilotId"] / build_sha
    limitation = raw["limitations"]["bridgeObservationMask"]
    manifest = {
        "format": 1,
        "pilotId": raw["pilotId"],
        "buildSha256": build_sha,
        "configPath": config_path.relative_to(ASSET_GEN_ROOT).as_posix(),
        "configCanonicalSha256": _sha256_bytes(config_identity),
        "campaignContentSha256": encoded.content_sha256,
        "campaignMetadataSha256": _sha256_bytes(encoded.metadata_json),
        "campaignArraysSha256": _sha256_bytes(encoded.arrays_npz),
        "heightSemantics": semantics,
        "officialSemanticSources": source["officialSemanticSources"],
        "geometrySha256": {
            name: _sha256_bytes(payload) for name, payload in geometry_payloads.items()
        },
        "coverageClosure": {
            "method": "snap_existing_cap_vertices_to_exact_centerline_within_coordinate_ulp_bound",
            "toleranceM": snap.tolerance_m,
            "missingBeforeM": snap.missing_before_m,
            "hausdorffDisplacementM": snap.hausdorff_displacement_m,
            "symmetricDifferenceM2": snap.symmetric_difference_m2,
            "exactCoverageAfter": downstream_cap.covers(downstream_line),
        },
        "observationExclusionGeometry": None,
        "limitations": {"bridgeObservationMask": limitation},
        "decision": encoded_metadata["campaign"]["decision"],
    }
    outputs = {
        "campaign.json": encoded.metadata_json,
        "campaign.npz": encoded.arrays_npz,
        "pilot.json": _canonical_json(manifest),
        **geometry_payloads,
    }
    for name, content in outputs.items():
        _write_immutable(output_root / name, content)
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ASSET_GEN_ROOT / "config/terrain-repair/taevaskoda-ahja-stage1.json",
    )
    output = run_campaign(parser.parse_args().config)
    print(output)


if __name__ == "__main__":
    main()
