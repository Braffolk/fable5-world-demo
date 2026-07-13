"""Immutable acquisition of preregistered public ALS evidence."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from ..config import CONFIG_DIR, DATA_IN, BaseConfig
from .http import PoliteSession

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PRIMARY_SCHEMA = "taevaskoda-als-selection/1.0.0"
_PRIMARY_SELECTION_ID = "taevaskoda-als-444679-stage1"
_ADJACENT_SCHEMA = "taevaskoda-ahja-als-adjacent-selection/1.0.0"
_ADJACENT_SELECTION_ID = "taevaskoda-ahja-als-adjacent-2019-stage1"
_SOURCE_INDEX_URL = (
    "https://geoportaal.maaamet.ee/index.php?lang_id=1&plugin_act=otsing&page_id=614"
    "&kaardiruut=444679&andmetyyp=lidar_laz_tava"
)
_LICENSE = {
    "url": "https://geoportaal.maaruum.ee/avaandmete-litsents",
    "effective_date": "2025-01-01",
}
_ATTRIBUTION = {
    "provider": "Maa- ja Ruumiamet",
    "dataset": "Airborne laser scanning height points, tile 444679",
    "include_data_age_or_extraction_date": True,
    "include_license_text_or_url": True,
}
_OFFICIAL_HOSTS = ("geoportaal.maaamet.ee", "geoportaal.maaruum.ee")
_FROZEN_FILES = (
    (2011, "lidar_laz_tava", "444679_2011_tava.laz", "corroboration", None, None),
    (2015, "lidar_laz_tava", "444679_2015_tava.laz", "corroboration", None, None),
    (
        2016,
        "lidar_laz_madal",
        "444679_2016_madal.laz",
        "conditional_corroboration",
        None,
        None,
    ),
    (2017, "lidar_laz_mets", "444679_2017_mets.laz", "complementary", None, None),
    (2019, "lidar_laz_tava", "444679_2019_tava.laz", "corroboration", None, None),
    (2021, "lidar_laz_mets", "444679_2021_mets.laz", "complementary", None, None),
    (
        2023,
        "lidar_laz_tava",
        "444679_2023_tava.laz",
        "primary_candidate",
        68_066_160,
        "9c50c123f14841c717d0d123d2d08061b87a51baf6d806ef269fabf1036a9fb7",
    ),
    (
        2024,
        "lidar_laz_mets",
        "444679_2024_mets.laz",
        "complementary_change",
        None,
        None,
    ),
)
_ADJACENT_PURPOSE = (
    "close the selected Ahja profile across primary-tile boundaries without treating ALS "
    "storage tiles as hydrologic endpoints"
)
_ADJACENT_QUALIFICATION = (
    "apply the existing class-9 support, continuity, orientation, bank-consistency, and "
    "abstention gates to the extended 2019 profile"
)
_ADJACENT_ATTRIBUTION = {
    **_ATTRIBUTION,
    "dataset": "Airborne laser scanning height points, tiles 445679 and 444680",
}
_ADJACENT_FILES = (
    (
        "445679",
        (679000.0, 6445000.0, 680000.0, 6446000.0),
        "39.6 MB",
        "445679_2019_tava.laz",
    ),
    (
        "444680",
        (680000.0, 6444000.0, 681000.0, 6445000.0),
        "44.3 MB",
        "444680_2019_tava.laz",
    ),
)
_ADJACENT_EXCLUDED_TILES = [
    {
        "id": "445680",
        "reason": (
            "the selected principal centerline has zero length in this tile and its ETAK water "
            "area is outside the publication and 8 m repair-support domain"
        ),
    }
]


def _source_index_url(tile_id: str, source_type: str) -> str:
    return (
        "https://geoportaal.maaamet.ee/index.php?lang_id=1&plugin_act=otsing&page_id=614"
        f"&kaardiruut={tile_id}&andmetyyp={source_type}"
    )


def _artifact_url(source_type: str, filename: str, tile_id: str = "444679") -> str:
    return (
        "https://geoportaal.maaamet.ee/index.php?lang_id=1&plugin_act=otsing"
        f"&kaardiruut={tile_id}&andmetyyp={source_type}&dl=1&f={filename}&page_id=614"
    )


@dataclass(frozen=True)
class AlsArtifact:
    year: int
    source_type: str
    filename: str
    role: str
    url: str
    expected_bytes: int | None
    expected_sha256: str | None
    manifest_metadata: dict[str, Any]


@dataclass(frozen=True)
class AlsSourceIndex:
    snapshot_name: str
    url: str


@dataclass(frozen=True)
class AlsSelection:
    selection_id: str
    tile_id: str
    epsg: int
    source_indexes: tuple[AlsSourceIndex, ...]
    license_url: str
    raw: dict[str, Any]
    files: tuple[AlsArtifact, ...]
    manifest_metadata: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_bounds(
    tile: dict[str, Any], tile_id: str, expected: tuple[float, ...]
) -> None:
    if (
        set(tile) != {"id", "crs", "bounds"}
        or tile["id"] != tile_id
        or tile["crs"] != "EPSG:3301"
    ):
        raise ValueError(f"ALS selection requires the exact EPSG:3301 tile {tile_id} contract")
    bounds = tile["bounds"]
    keys = ("min_x", "min_y", "max_x_exclusive", "max_y_exclusive")
    if set(bounds) != set(keys) or tuple(float(bounds[key]) for key in keys) != expected:
        raise ValueError(f"ALS selection moved away from official tile {tile_id}")


def _validate_artifact_url(item: dict[str, Any], tile_id: str) -> None:
    parsed = urlsplit(item["canonical_url"])
    if (
        parsed.scheme != "https"
        or parsed.netloc != "geoportaal.maaamet.ee"
        or item["canonical_url"] != _artifact_url(item["type"], item["filename"], tile_id)
    ):
        raise ValueError(f"non-official ALS endpoint {item['canonical_url']}")
    query = parse_qs(parsed.query, keep_blank_values=True)
    if query != {
        "lang_id": ["1"],
        "plugin_act": ["otsing"],
        "kaardiruut": [tile_id],
        "andmetyyp": [item["type"]],
        "dl": ["1"],
        "f": [item["filename"]],
        "page_id": ["614"],
    }:
        raise ValueError(f"ALS artifact URL is not the frozen official query: {item['filename']}")


def _load_primary_selection(raw: dict[str, Any]) -> AlsSelection:
    required = {
        "schema_version",
        "id",
        "stage",
        "morphology_target",
        "tile",
        "source_index_url",
        "license",
        "required_attribution",
        "files",
    }
    if set(raw) != required or raw["schema_version"] != _PRIMARY_SCHEMA:
        raise ValueError("unsupported or non-strict ALS selection config")
    if (
        raw["id"] != _PRIMARY_SELECTION_ID
        or raw["stage"] != "stage1-structural-repair"
        or raw["morphology_target"] is not False
    ):
        raise ValueError("Taevaskoda ALS must remain structural-repair evidence only")
    tile = raw["tile"]
    _validate_bounds(tile, "444679", (679000.0, 6444000.0, 680000.0, 6445000.0))
    if raw["source_index_url"] != _SOURCE_INDEX_URL:
        raise ValueError("ALS selection source index is not the frozen official query")
    if raw["license"] != _LICENSE or raw["required_attribution"] != _ATTRIBUTION:
        raise ValueError("ALS selection changed its controlling license or attribution")
    if len(raw["files"]) != len(_FROZEN_FILES):
        raise ValueError("ALS selection must contain exactly eight frozen epochs")
    records: list[AlsArtifact] = []
    for item, expected in zip(raw["files"], _FROZEN_FILES, strict=True):
        if set(item) != {
            "year", "type", "filename", "role", "canonical_url", "bytes", "sha256"
        }:
            raise ValueError("ALS file selection contains unknown or missing fields")
        actual = (
            item["year"],
            item["type"],
            item["filename"],
            item["role"],
            item["bytes"],
            item["sha256"],
        )
        if actual != expected:
            raise ValueError(f"ALS file decision changed for {item['filename']}")
        _validate_artifact_url(item, "444679")
        expected_sha = item["sha256"]
        if expected_sha is not None and not _SHA256_RE.fullmatch(expected_sha):
            raise ValueError(f"invalid expected SHA-256 for {item['filename']}")
        records.append(
            AlsArtifact(
                year=int(item["year"]),
                source_type=str(item["type"]),
                filename=str(item["filename"]),
                role=str(item["role"]),
                url=str(item["canonical_url"]),
                expected_bytes=item["bytes"],
                expected_sha256=expected_sha,
                manifest_metadata={},
            )
        )
    return AlsSelection(
        selection_id=str(raw["id"]),
        tile_id=str(tile["id"]),
        epsg=3301,
        source_indexes=(AlsSourceIndex("source-index.html", str(raw["source_index_url"])),),
        license_url=str(raw["license"]["url"]),
        raw=raw,
        files=tuple(records),
        manifest_metadata={"tile": raw["tile"]},
    )


def _load_adjacent_selection(raw: dict[str, Any]) -> AlsSelection:
    required = {
        "schema_version",
        "id",
        "stage",
        "morphology_target",
        "purpose",
        "primary_selection_id",
        "principal_centerline_etak_id",
        "repair_collar_m",
        "corroborating_epoch_required",
        "qualification_after_retention",
        "license",
        "required_attribution",
        "files",
        "excluded_tiles",
    }
    if set(raw) != required or raw["schema_version"] != _ADJACENT_SCHEMA:
        raise ValueError("unsupported or non-strict ALS selection config")
    if (
        raw["id"] != _ADJACENT_SELECTION_ID
        or raw["stage"] != "stage1-structural-repair"
        or raw["morphology_target"] is not False
        or raw["purpose"] != _ADJACENT_PURPOSE
        or raw["primary_selection_id"] != _PRIMARY_SELECTION_ID
        or raw["principal_centerline_etak_id"] != 2356024
        or raw["repair_collar_m"] != 8.0
        or raw["corroborating_epoch_required"] is not False
        or raw["qualification_after_retention"] != _ADJACENT_QUALIFICATION
        or raw["license"] != _LICENSE
        or raw["required_attribution"] != _ADJACENT_ATTRIBUTION
        or raw["excluded_tiles"] != _ADJACENT_EXCLUDED_TILES
        or len(raw["files"]) != 2
    ):
        raise ValueError("adjacent ALS closure decision changed")

    records: list[AlsArtifact] = []
    indexes: list[AlsSourceIndex] = []
    for item, (tile_id, bounds, reported_size, filename) in zip(
        raw["files"], _ADJACENT_FILES, strict=True
    ):
        if set(item) != {
            "tile",
            "source_index_url",
            "year",
            "type",
            "filename",
            "role",
            "canonical_url",
            "index_reported_size",
            "bytes",
            "sha256",
        }:
            raise ValueError("adjacent ALS file selection contains unknown or missing fields")
        _validate_bounds(item["tile"], tile_id, bounds)
        expected_index = _source_index_url(tile_id, "lidar_laz_tava")
        if (
            item["source_index_url"] != expected_index
            or item["year"] != 2019
            or item["type"] != "lidar_laz_tava"
            or item["filename"] != filename
            or item["role"] != "primary_profile_spatial_closure"
            or item["index_reported_size"] != reported_size
            or item["bytes"] is not None
            or item["sha256"] is not None
        ):
            raise ValueError(f"adjacent ALS file decision changed for tile {tile_id}")
        _validate_artifact_url(item, tile_id)
        indexes.append(AlsSourceIndex(f"source-index-{tile_id}.html", expected_index))
        records.append(
            AlsArtifact(
                year=2019,
                source_type="lidar_laz_tava",
                filename=filename,
                role="primary_profile_spatial_closure",
                url=str(item["canonical_url"]),
                expected_bytes=None,
                expected_sha256=None,
                manifest_metadata={"tile": item["tile"]},
            )
        )
    return AlsSelection(
        selection_id=_ADJACENT_SELECTION_ID,
        tile_id="multi",
        epsg=3301,
        source_indexes=tuple(indexes),
        license_url=str(raw["license"]["url"]),
        raw=raw,
        files=tuple(records),
        manifest_metadata={
            "tiles": [item["tile"] for item in raw["files"]],
            "primarySelectionId": raw["primary_selection_id"],
        },
    )


def load_als_selection(path: Path | None = None, *, content: bytes | None = None) -> AlsSelection:
    path = path or CONFIG_DIR / "taevaskoda-als.json"
    raw = json.loads(content if content is not None else path.read_bytes())
    schema = raw.get("schema_version") if isinstance(raw, dict) else None
    if schema == _PRIMARY_SCHEMA:
        return _load_primary_selection(raw)
    if schema == _ADJACENT_SCHEMA:
        return _load_adjacent_selection(raw)
    raise ValueError("unsupported or non-strict ALS selection config")


def _safe_relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _snapshot_bytes(root: Path, source: Path, content: bytes, digest: str) -> Path:
    snapshot = root / "snapshots" / "selection" / digest / source.name
    if snapshot.exists():
        if snapshot.read_bytes() != content:
            raise ValueError(f"selection snapshot digest collision: {snapshot}")
        return snapshot
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    temporary = snapshot.with_suffix(snapshot.suffix + ".part")
    temporary.write_bytes(content)
    temporary.replace(snapshot)
    return snapshot


def _provenance_entry(root: Path, artifact: Path, provenance: Path) -> dict[str, Any]:
    return {
        "relativePath": artifact.relative_to(root).as_posix(),
        "bytes": artifact.stat().st_size,
        "sha256": _sha256_file(artifact),
        "provenance": provenance.relative_to(root).as_posix(),
        "provenanceSha256": _sha256_file(provenance),
    }


def _has_las_magic(path: Path) -> bool:
    with path.open("rb") as source:
        return source.read(4) == b"LASF"


def _verify_retained_entry(
    root: Path,
    item: AlsArtifact,
    entry: dict[str, Any],
) -> dict[str, Any]:
    raw_path = root / str(entry["relativePath"])
    provenance = root / str(entry["provenance"])
    sidecar = raw_path.with_suffix(raw_path.suffix + ".sha256")
    expected_path = Path("raw") / "sha256" / str(entry["sha256"]) / item.filename
    expected_provenance = Path("provenance") / "http" / f"{item.filename}.json"
    if Path(str(entry["relativePath"])) != expected_path:
        raise ValueError(f"retained ALS artifact is not content-addressed: {item.filename}")
    if Path(str(entry["provenance"])) != expected_provenance:
        raise ValueError(f"retained ALS provenance moved unexpectedly: {item.filename}")
    if not raw_path.is_file() or not sidecar.is_file() or not provenance.is_file():
        raise ValueError(f"retained ALS artifact tuple is incomplete: {item.filename}")
    digest = _sha256_file(raw_path)
    record = json.loads(provenance.read_bytes())
    if (
        digest != entry["sha256"]
        or sidecar.read_text(encoding="ascii").strip() != digest
        or raw_path.stat().st_size != entry["bytes"]
        or _sha256_file(provenance) != entry["provenanceSha256"]
        or record.get("requestedUrl") != item.url
        or record.get("bytes") != entry["bytes"]
        or record.get("sha256") != digest
        or record.get("status") != 200
        or not _has_las_magic(raw_path)
        or entry.get("year") != item.year
        or entry.get("type") != item.source_type
        or entry.get("filename") != item.filename
        or entry.get("role") != item.role
        or any(entry.get(key) != value for key, value in item.manifest_metadata.items())
    ):
        raise ValueError(f"retained ALS artifact failed provenance verification: {item.filename}")
    if item.expected_bytes is not None and entry["bytes"] != item.expected_bytes:
        raise ValueError(f"retained ALS byte count changed: {item.filename}")
    if item.expected_sha256 is not None and digest != item.expected_sha256:
        raise ValueError(f"retained ALS digest changed: {item.filename}")
    return dict(entry)


def _recover_content_addressed(
    root: Path,
    item: AlsArtifact,
    provenance: Path,
) -> dict[str, Any] | None:
    candidates = tuple((root / "raw" / "sha256").glob(f"*/{item.filename}"))
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError(f"multiple retained byte identities found for {item.filename}")
    artifact = candidates[0]
    digest = artifact.parent.name
    if not _SHA256_RE.fullmatch(digest) or _sha256_file(artifact) != digest:
        raise ValueError(f"invalid content-address path for {item.filename}")
    sidecar = artifact.with_suffix(artifact.suffix + ".sha256")
    staging_sidecar = (root / "staging" / item.filename).with_suffix(
        Path(item.filename).suffix + ".sha256"
    )
    if not sidecar.exists() and staging_sidecar.exists():
        if staging_sidecar.read_text(encoding="ascii").strip() != digest:
            raise ValueError(f"staged sidecar does not match retained bytes: {item.filename}")
        staging_sidecar.replace(sidecar)
    if not sidecar.is_file() or not provenance.is_file():
        raise ValueError(f"orphaned retained ALS artifact requires manual audit: {item.filename}")
    entry = {
        **_provenance_entry(root, artifact, provenance),
        "year": item.year,
        "type": item.source_type,
        "filename": item.filename,
        "role": item.role,
        **item.manifest_metadata,
    }
    return _verify_retained_entry(root, item, entry)


def _promote_content_addressed(root: Path, staging: Path, provenance: Path) -> dict[str, Any]:
    digest = _sha256_file(staging)
    destination = root / "raw" / "sha256" / digest / staging.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_sidecar = staging.with_suffix(staging.suffix + ".sha256")
    destination_sidecar = destination.with_suffix(destination.suffix + ".sha256")
    if destination.exists():
        if _sha256_file(destination) != digest:
            raise ValueError(f"content-address collision at {destination}")
        staging.unlink()
    else:
        staging.replace(destination)
    if destination_sidecar.exists():
        if destination_sidecar.read_text(encoding="ascii").strip() != digest:
            raise ValueError(f"content-address sidecar collision at {destination_sidecar}")
        staging_sidecar.unlink(missing_ok=True)
    else:
        staging_sidecar.replace(destination_sidecar)
    return _provenance_entry(root, destination, provenance)


def fetch_als_selection(
    base: BaseConfig,
    selection_path: Path | None = None,
    *,
    years: tuple[int, ...] = (),
    log=print,
) -> Path:
    selection_path = selection_path or CONFIG_DIR / "taevaskoda-als.json"
    selection_path = selection_path.resolve()
    selection_bytes = selection_path.read_bytes()
    selection_sha256 = hashlib.sha256(selection_bytes).hexdigest()
    selection = load_als_selection(selection_path, content=selection_bytes)
    selected = tuple(item for item in selection.files if not years or item.year in years)
    if not selected:
        raise ValueError("requested ALS years do not select an artifact")
    unknown_years = set(years) - {item.year for item in selection.files}
    if unknown_years:
        raise ValueError(f"ALS selection has no campaigns for years {sorted(unknown_years)}")

    als_root = (DATA_IN / "public" / "als").resolve()
    root = (als_root / selection.selection_id).resolve()
    if root.parent != als_root:
        raise ValueError("ALS selection ID escapes the public evidence root")
    selection_snapshot = _snapshot_bytes(root, selection_path, selection_bytes, selection_sha256)
    provenance_root = root / "provenance" / "http"
    session = PoliteSession(base.fetch)
    snapshots = tuple(
        (source.snapshot_name, source.url) for source in selection.source_indexes
    ) + (("license.html", selection.license_url),)
    snapshot_entries: list[dict[str, Any]] = []
    for filename, url in snapshots:
        path = root / "snapshots" / filename
        provenance = provenance_root / f"{filename}.json"
        session.download_recorded(
            url,
            path,
            provenance,
            min_bytes=256,
            allowed_final_hosts=_OFFICIAL_HOSTS,
            allow_html=True,
        )
        snapshot_entries.append(
            {"name": filename, "url": url, **_provenance_entry(root, path, provenance)}
        )

    output = root / "retained.json"
    existing_by_name: dict[str, dict[str, Any]] = {}
    if output.exists():
        existing = json.loads(output.read_bytes())
        if (
            existing.get("selectionId") != selection.selection_id
            or existing.get("selectionConfigSha256") != selection_sha256
        ):
            raise ValueError("existing ALS retention manifest belongs to another selection")
        existing_by_name = {
            str(entry["filename"]): entry for entry in existing.get("artifacts", [])
        }

    fetched_by_name: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(selected, start=1):
        if item.filename in existing_by_name:
            retained = _verify_retained_entry(root, item, existing_by_name[item.filename])
            downloaded = False
        else:
            staging = root / "staging" / item.filename
            provenance = provenance_root / f"{item.filename}.json"
            recovered = _recover_content_addressed(root, item, provenance)
            if recovered is not None:
                retained = recovered
                downloaded = False
            else:
                session.download_recorded(
                    item.url,
                    staging,
                    provenance,
                    min_bytes=227,
                    expected_bytes=item.expected_bytes,
                    expected_sha256=item.expected_sha256,
                    expected_magic=b"LASF",
                    allowed_final_hosts=_OFFICIAL_HOSTS,
                )
                retained = _promote_content_addressed(root, staging, provenance)
                downloaded = True
        fetched_by_name[item.filename] = {
            **retained,
            "year": item.year,
            "type": item.source_type,
            "filename": item.filename,
            "role": item.role,
            **item.manifest_metadata,
        }
        log(
            f"[{index}/{len(selected)}] {'fetched' if downloaded else 'verified'} "
            f"{item.filename} ({retained['bytes'] / 1e6:.1f} MB)"
        )

    artifacts: list[dict[str, Any]] = []
    missing_files: list[str] = []
    for item in selection.files:
        entry = fetched_by_name.get(item.filename) or existing_by_name.get(item.filename)
        if entry is None:
            provenance = provenance_root / f"{item.filename}.json"
            entry = _recover_content_addressed(root, item, provenance)
        if entry is not None:
            artifacts.append(_verify_retained_entry(root, item, entry))
        else:
            missing_files.append(item.filename)

    retained = {
        "format": 1,
        "selectionId": selection.selection_id,
        "selectionConfig": _safe_relative(selection_path, CONFIG_DIR.parent),
        "selectionConfigSha256": selection_sha256,
        "selectionSnapshot": selection_snapshot.relative_to(root).as_posix(),
        "selectionSnapshotSha256": _sha256_file(selection_snapshot),
        "generatedUtc": datetime.now(timezone.utc).isoformat(),
        "requestedYears": [item.year for item in selected],
        "complete": not missing_files,
        "missingFiles": missing_files,
        "morphologyTarget": False,
        "license": selection.raw["license"],
        "requiredAttribution": selection.raw["required_attribution"],
        "sourceSnapshots": snapshot_entries,
        "artifacts": artifacts,
        **selection.manifest_metadata,
    }
    temporary = output.with_suffix(".json.part")
    temporary.write_text(json.dumps(retained, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return output
