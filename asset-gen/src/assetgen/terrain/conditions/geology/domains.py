"""Strict parser and immutable snapshot contract for EGT 1:200k domains."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ....config import DATA_WORK

SCHEMA_VERSION = "laas.egt-surficial-200k-domains/1"
SERVICE_ITEM_ID = "a940f8e1315b48ccb0f829804e23b228"
SERVICE_ITEM_URL = (
    "https://gis.egt.ee/portal/sharing/rest/content/items/"
    f"{SERVICE_ITEM_ID}?f=pjson"
)
MAP_SERVICE_URL = (
    "https://gis.egt.ee/arcgis/rest/services/Q_200_WM_MIL1/MapServer"
)


@dataclass(frozen=True)
class DomainSpec:
    layer_id: int
    layer_name: str
    field_name: str
    domain_name: str

    @property
    def url(self) -> str:
        return f"{MAP_SERVICE_URL}/{self.layer_id}?f=pjson"


DOMAIN_SPECS = (
    DomainSpec(34, "Setete litoloogia", "lito200", "Q_Litoloogia_200"),
    DomainSpec(35, "Setete genees", "genees200", "Q_Genees_200"),
)


@dataclass(frozen=True)
class EgtSurficialDomains:
    snapshot_path: Path
    snapshot_sha256: str
    domains: Mapping[str, Mapping[int, str]]

    def decode(self, domain_name: str, code: int) -> str:
        domain = self.domains.get(domain_name)
        if domain is None:
            raise KeyError(f"unknown EGT domain: {domain_name}")
        try:
            return domain[int(code)]
        except KeyError as error:
            raise KeyError(f"unknown {domain_name} code: {code}") from error


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _parse_service_item(payload: bytes) -> dict[str, Any]:
    item = _object(payload, "EGT service item")
    expected = {
        "id": SERVICE_ITEM_ID,
        "title": "Q_200_WM_MIL1",
        "type": "Map Service",
        "access": "public",
        "url": MAP_SERVICE_URL,
    }
    drift = {
        key: {"expected": value, "actual": item.get(key)}
        for key, value in expected.items()
        if item.get(key) != value
    }
    if drift:
        raise ValueError(f"EGT service-item identity drift: {drift}")
    return {
        "id": item["id"],
        "title": item["title"],
        "type": item["type"],
        "access": item["access"],
        "owner": item.get("owner"),
        "created": item.get("created"),
        "modified": item.get("modified"),
        "service_url": item["url"],
        "license_info": item.get("licenseInfo"),
    }


def _parse_domain_layer(payload: bytes, spec: DomainSpec) -> dict[str, Any]:
    layer = _object(payload, f"EGT MapServer layer {spec.layer_id}")
    expected = {
        "id": spec.layer_id,
        "name": spec.layer_name,
        "type": "Feature Layer",
        "geometryType": "esriGeometryPolygon",
        "serviceItemId": SERVICE_ITEM_ID,
    }
    drift = {
        key: {"expected": value, "actual": layer.get(key)}
        for key, value in expected.items()
        if layer.get(key) != value
    }
    if drift:
        raise ValueError(f"EGT layer identity drift: {drift}")
    spatial_reference = layer.get("spatialReference")
    if not isinstance(spatial_reference, dict) or spatial_reference.get("wkid") != 3301:
        raise ValueError(f"EGT layer {spec.layer_id} is not EPSG:3301")
    fields = layer.get("fields")
    if not isinstance(fields, list):
        raise ValueError(f"EGT layer {spec.layer_id} fields are absent")
    matches = [field for field in fields if field.get("name") == spec.field_name]
    if len(matches) != 1:
        raise ValueError(
            f"EGT layer {spec.layer_id} must expose exactly one {spec.field_name} field"
        )
    field = matches[0]
    if field.get("type") != "esriFieldTypeSmallInteger":
        raise ValueError(f"EGT {spec.field_name} field type changed")
    domain = field.get("domain")
    if (
        not isinstance(domain, dict)
        or domain.get("type") != "codedValue"
        or domain.get("name") != spec.domain_name
    ):
        raise ValueError(f"EGT {spec.field_name} coded domain changed")
    source_values = domain.get("codedValues")
    if not isinstance(source_values, list) or not source_values:
        raise ValueError(f"EGT {spec.domain_name} has no coded values")
    coded_values: list[dict[str, Any]] = []
    seen_codes: set[int] = set()
    for index, value in enumerate(source_values):
        if not isinstance(value, dict):
            raise ValueError(f"EGT {spec.domain_name} value {index} is not an object")
        code = value.get("code")
        name = value.get("name")
        if isinstance(code, bool) or not isinstance(code, int):
            raise ValueError(f"EGT {spec.domain_name} code {code!r} is not an integer")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"EGT {spec.domain_name} code {code} has no name")
        if code in seen_codes:
            raise ValueError(f"EGT {spec.domain_name} repeats code {code}")
        seen_codes.add(code)
        coded_values.append({"code": code, "name": name})
    return {
        "layer_id": spec.layer_id,
        "layer_name": spec.layer_name,
        "layer_url": spec.url,
        "field_name": spec.field_name,
        "field_type": field["type"],
        "domain_name": domain["name"],
        "domain_description": domain.get("description"),
        "merge_policy": domain.get("mergePolicy"),
        "split_policy": domain.get("splitPolicy"),
        "coded_values": coded_values,
    }


def build_egt_surficial_domain_snapshot(
    *,
    service_item_json: bytes,
    layer_json: Mapping[int, bytes],
    source_documents: list[dict[str, Any]],
    output_parent: Path | None = None,
) -> Path:
    if set(layer_json) != {spec.layer_id for spec in DOMAIN_SPECS}:
        raise ValueError("EGT domain snapshot requires exactly layers 34 and 35")
    item = _parse_service_item(service_item_json)
    domains = [
        _parse_domain_layer(layer_json[spec.layer_id], spec) for spec in DOMAIN_SPECS
    ]
    sources_by_url = {row.get("requested_url"): row for row in source_documents}
    expected_urls = {SERVICE_ITEM_URL, *(spec.url for spec in DOMAIN_SPECS)}
    if set(sources_by_url) != expected_urls:
        raise ValueError("EGT source-document inventory differs from the frozen URLs")
    expected_payloads = {
        SERVICE_ITEM_URL: service_item_json,
        **{spec.url: layer_json[spec.layer_id] for spec in DOMAIN_SPECS},
    }
    normalized_sources: list[dict[str, Any]] = []
    for url in sorted(expected_urls):
        row = sources_by_url[url]
        payload = expected_payloads[url]
        if (
            row.get("exact_json_sha256") != _sha256(payload)
            or row.get("bytes") != len(payload)
            or row.get("status") != 200
            or row.get("final_url") != url
        ):
            raise ValueError(f"EGT retrieval metadata does not bind {url}")
        normalized_sources.append(dict(row))
    identity = {
        "schema_version": SCHEMA_VERSION,
        "service_item": item,
        "source_documents": normalized_sources,
        "domains": domains,
    }
    snapshot_sha256 = _sha256(canonical_json_bytes(identity))
    parent = (
        Path(output_parent)
        if output_parent is not None
        else DATA_WORK
        / "terrain"
        / "conditions"
        / "geology"
        / "egt-surficial-200k-domains"
        / "sha256"
    )
    root = parent / snapshot_sha256
    snapshot_path = root / "snapshot.json"
    payload = canonical_json_bytes(identity)
    if snapshot_path.exists():
        if snapshot_path.read_bytes() != payload:
            raise ValueError(f"immutable EGT domain snapshot differs: {snapshot_path}")
        return snapshot_path
    if root.exists():
        raise RuntimeError(f"incomplete EGT domain snapshot exists: {root}")
    temporary = parent / f".{snapshot_sha256}.{os.getpid()}.tmp"
    if temporary.exists():
        raise RuntimeError(f"stale EGT domain temporary exists: {temporary}")
    temporary.mkdir(parents=True)
    (temporary / "snapshot.json").write_bytes(payload)
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(root)
    return snapshot_path


def load_egt_surficial_domains(snapshot_path: Path) -> EgtSurficialDomains:
    snapshot_path = Path(snapshot_path)
    payload = snapshot_path.read_bytes()
    document = _object(payload, "EGT surficial-domain snapshot")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported EGT surficial-domain snapshot schema")
    expected_sha = snapshot_path.parent.name
    actual_sha = _sha256(payload)
    if len(expected_sha) != 64 or actual_sha != expected_sha:
        raise ValueError("EGT domain snapshot path does not match its canonical JSON SHA-256")
    parsed: dict[str, dict[int, str]] = {}
    rows = document.get("domains")
    if not isinstance(rows, list):
        raise ValueError("EGT domain snapshot lacks domains")
    for spec in DOMAIN_SPECS:
        matches = [row for row in rows if row.get("domain_name") == spec.domain_name]
        if len(matches) != 1:
            raise ValueError(f"snapshot must contain exactly one {spec.domain_name}")
        values = matches[0].get("coded_values")
        if not isinstance(values, list):
            raise ValueError(f"snapshot {spec.domain_name} lacks coded values")
        mapping = {int(value["code"]): str(value["name"]) for value in values}
        if len(mapping) != len(values):
            raise ValueError(f"snapshot {spec.domain_name} repeats a code")
        parsed[spec.domain_name] = mapping
    return EgtSurficialDomains(snapshot_path, actual_sha, parsed)
