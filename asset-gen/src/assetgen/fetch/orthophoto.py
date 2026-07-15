"""Acquire workbook-bound RGB/CIR orthophoto evidence for frozen sheets."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import time
import zipfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit
from xml.etree import ElementTree

import rasterio
import requests
from affine import Affine
from rasterio.windows import Window

from ..config import ASSET_GEN_ROOT, DATA_IN, BaseConfig
from .http import PoliteSession


_WORKBOOK = (
    ASSET_GEN_ROOT.parent
    / "docs/deep-research/microtopography-generation/library/data/maaamet"
    / "tomba_etak_avaandmed.xlsx"
)
_WORKBOOK_SHA256 = "fbef1433eff6116174cd1bbeca6e6550e093eeed794c14d583af77e03970b4b5"
_DEVELOPMENT_SELECTIONS = (
    ASSET_GEN_ROOT
    / "config/evidence/orthophoto-development-sheets-v1.json"
)
_OFFICIAL_HOSTS = ("geoportaal.maaamet.ee", "geoportaal.maaruum.ee")
_RETRYABLE_HTTP = {429, 500, 502, 503, 504}
_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
_XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_DOC_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
_ATTRIBUTION = {
    "provider": "Maa- ja Ruumiamet",
    "terms": "CC BY 4.0 equivalent open-data terms declared by the project",
    "includeSourceAndAcquisitionDate": True,
}


@dataclass(frozen=True)
class OrthophotoArtifact:
    product: str
    workbook_sheet: str
    source_type: str
    filename: str
    capture_date: str
    expected_bytes: int
    expected_sha256: str | None
    pixel_size_m: float
    raster_size: int
    url: str


@dataclass(frozen=True)
class OrthophotoSelection:
    selection_id: str
    schema: str
    sheet: str
    sheet_bounds: tuple[float, float, float, float]
    retention_directory: str
    log_label: str
    temporal_policy: str
    artifacts: tuple[OrthophotoArtifact, ...]


def _artifact(
    *,
    sheet: str,
    product: str,
    workbook_sheet: str,
    source_type: str,
    capture_date: str,
    expected_bytes: int,
    expected_sha256: str | None,
    pixel_size_m: float,
    raster_size: int,
) -> OrthophotoArtifact:
    filename = f"{sheet}_OF_{product.upper()}_GeoTIFF_{capture_date.replace('-', '_')}.zip"
    return OrthophotoArtifact(
        product=product,
        workbook_sheet=workbook_sheet,
        source_type=source_type,
        filename=filename,
        capture_date=capture_date,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
        pixel_size_m=pixel_size_m,
        raster_size=raster_size,
        url=(
            "https://geoportaal.maaamet.ee/index.php?lang_id=1&plugin_act=otsing"
            f"&kaardiruut={sheet}&andmetyyp={source_type}&dl=1"
            f"&f={filename}&page_id=610"
        ),
    )


_TEMPORAL_POLICY = (
    "RGB 2025 and CIR 2024 remain separate dated observations, never a "
    "co-temporal composite."
)
_SELECTIONS = {
    "54472": OrthophotoSelection(
        selection_id="taevaskoda-orthophoto-54472-stage1",
        schema="taevaskoda-stage1-orthophoto-selection/1.0.0",
        sheet="54472",
        sheet_bounds=(675000.0, 6440000.0, 680000.0, 6445000.0),
        retention_directory="stage1-54472",
        log_label="Stage-1 orthophoto",
        temporal_policy=_TEMPORAL_POLICY,
        artifacts=(
            _artifact(
                sheet="54472",
                product="rgb",
                workbook_sheet="Ortofotod 2016-2025",
                source_type="ortofoto_eesti_rgb",
                capture_date="2025-07-18",
                expected_bytes=207_351_332,
                expected_sha256="ea6b108232540c0966a8112ee5236039e9f5774d5a4fbf242ee6e5e488cae83b",
                pixel_size_m=0.2,
                raster_size=25_000,
            ),
            _artifact(
                sheet="54472",
                product="cir",
                workbook_sheet="Ortofoto_CIR",
                source_type="ortofoto_eesti_cir",
                capture_date="2024-05-22",
                expected_bytes=133_805_748,
                expected_sha256="92dd4dc6bc67d58983baf919c980af43f78cfe4ec3b196d9e57b06ce778a4a20",
                pixel_size_m=0.25,
                raster_size=20_000,
            ),
        ),
    ),
    "54481": OrthophotoSelection(
        selection_id="erodible-slope-orthophoto-54481-current",
        schema="orthophoto-frozen-sheet-selection/1.0.0",
        sheet="54481",
        sheet_bounds=(680000.0, 6440000.0, 685000.0, 6445000.0),
        retention_directory="sheet-54481",
        log_label="Frozen sheet 54481 orthophoto",
        temporal_policy=_TEMPORAL_POLICY,
        artifacts=(
            _artifact(
                sheet="54481",
                product="rgb",
                workbook_sheet="Ortofotod 2016-2025",
                source_type="ortofoto_eesti_rgb",
                capture_date="2025-07-18",
                expected_bytes=194_611_964,
                expected_sha256="beb2cce12095409a4124890c1e95ebac90f4bfb53aa9263e8622c0688a9f14d6",
                pixel_size_m=0.2,
                raster_size=25_000,
            ),
            _artifact(
                sheet="54481",
                product="cir",
                workbook_sheet="Ortofoto_CIR",
                source_type="ortofoto_eesti_cir",
                capture_date="2024-05-22",
                expected_bytes=125_683_915,
                expected_sha256="fb1ed7a1e02d282d56c03d07e8d255f33eb80f2e89d51d49e0fe21d7c5382c8f",
                pixel_size_m=0.25,
                raster_size=20_000,
            ),
        ),
    ),
}


def _file_identity(path: Path) -> dict[str, Any]:
    return {
        "path": _safe_relative(path, ASSET_GEN_ROOT.parent),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _load_development_selections(
) -> tuple[dict[str, OrthophotoSelection], dict[str, Any], int]:
    document = json.loads(_DEVELOPMENT_SELECTIONS.read_text(encoding="utf-8"))
    if document.get("schemaVersion") != "orthophoto-frozen-sheet-selection-set/1":
        raise ValueError("unsupported development orthophoto selection schema")
    workbook = document.get("workbook", {})
    if (
        workbook.get("path") != _safe_relative(_WORKBOOK, ASSET_GEN_ROOT.parent)
        or workbook.get("sha256") != _WORKBOOK_SHA256
    ):
        raise ValueError("development orthophoto selection binds another workbook")
    excluded = {int(value) for value in document.get("excludedEtakIds", [])}
    candidate_ids = {
        int(value)
        for row in document.get("sheets", [])
        for value in row.get("candidateEtakIds", [])
    }
    if 9688702 not in excluded or candidate_ids & excluded:
        raise ValueError("development orthophoto selection does not exclude sealed OOD2")
    selections: dict[str, OrthophotoSelection] = {}
    for row in document.get("sheets", []):
        sheet = str(row["sheet"])
        if sheet in selections or not re.fullmatch(r"\d{5}", sheet):
            raise ValueError(f"duplicate or invalid orthophoto sheet id: {sheet}")
        bounds = tuple(float(value) for value in row["sheetBoundsEn"])
        if len(bounds) != 4 or bounds[2] - bounds[0] != 5000 or bounds[3] - bounds[1] != 5000:
            raise ValueError(f"orthophoto sheet {sheet} is not one 5 km square")
        artifacts = tuple(
            _artifact(
                sheet=sheet,
                product=str(artifact["product"]),
                workbook_sheet=str(artifact["workbookSheet"]),
                source_type=str(artifact["sourceType"]),
                capture_date=str(artifact["captureDate"]),
                expected_bytes=int(artifact["expectedBytes"]),
                expected_sha256=(
                    str(artifact["expectedSha256"])
                    if artifact.get("expectedSha256") is not None
                    else None
                ),
                pixel_size_m=float(artifact["pixelSizeM"]),
                raster_size=int(artifact["rasterSize"]),
            )
            for artifact in row["artifacts"]
        )
        if [artifact.product for artifact in artifacts] != ["rgb", "cir"]:
            raise ValueError(f"orthophoto sheet {sheet} must freeze RGB then CIR")
        selections[sheet] = OrthophotoSelection(
            selection_id=str(row["selectionId"]),
            schema="orthophoto-frozen-sheet-retention/1.0.0",
            sheet=sheet,
            sheet_bounds=bounds,
            retention_directory=str(row["retentionDirectory"]),
            log_label="National Development orthophoto",
            temporal_policy=str(row["temporalPolicy"]),
            artifacts=artifacts,
        )
    maximum_attempts = int(document.get("maximumAttemptsPerProduct", 0))
    if maximum_attempts not in (1, 2):
        raise ValueError("development orthophoto attempts must be capped at one or two")
    return selections, _file_identity(_DEVELOPMENT_SELECTIONS), maximum_attempts


class _DownloadIntegrityError(ValueError):
    pass


class _RangeUnsupported(_DownloadIntegrityError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(document: object) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable orthophoto artifact changed: {path}")
        return
    _write_atomic(path, payload)


def _safe_relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    return ["".join(node.itertext()) for node in root.findall(f"{_XLSX_NS}si")]


def _worksheet_path(archive: zipfile.ZipFile, sheet_name: str) -> str:
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    ids = {
        relation.attrib["Id"]: relation.attrib["Target"]
        for relation in relationships.findall(f"{_REL_NS}Relationship")
    }
    matches = [
        sheet
        for sheet in workbook.findall(f"{_XLSX_NS}sheets/{_XLSX_NS}sheet")
        if sheet.attrib.get("name") == sheet_name
    ]
    if len(matches) != 1:
        raise ValueError(f"workbook must contain exactly one {sheet_name!r} sheet")
    target = ids.get(matches[0].attrib.get(_DOC_REL, ""))
    if target is None:
        raise ValueError(f"workbook relationship is missing for {sheet_name!r}")
    path = PurePosixPath("xl") / target
    if path.is_absolute() or ".." in path.parts or path.as_posix() not in archive.namelist():
        raise ValueError(f"unsafe or missing worksheet target for {sheet_name!r}")
    return path.as_posix()


def _cell_value(cell: ElementTree.Element, shared: list[str]) -> str:
    value = cell.find(f"{_XLSX_NS}v")
    if value is None or value.text is None:
        return ""
    if cell.attrib.get("t") == "s":
        index = int(value.text)
        if index < 0 or index >= len(shared):
            raise ValueError("workbook shared-string index is out of range")
        return shared[index]
    return value.text


def _validate_workbook(
    selection: OrthophotoSelection,
) -> tuple[bytes, list[dict[str, Any]]]:
    workbook = _WORKBOOK.read_bytes()
    digest = hashlib.sha256(workbook).hexdigest()
    if digest != _WORKBOOK_SHA256:
        raise ValueError(f"official orthophoto workbook changed: {_WORKBOOK}")
    decisions: list[dict[str, Any]] = []
    with zipfile.ZipFile(_WORKBOOK) as archive:
        if archive.testzip() is not None:
            raise ValueError("official orthophoto workbook failed ZIP CRC validation")
        shared = _shared_strings(archive)
        for artifact in selection.artifacts:
            expected_command = f'wget --content-disposition "{artifact.url}"'
            sheet_path = _worksheet_path(archive, artifact.workbook_sheet)
            root = ElementTree.fromstring(archive.read(sheet_path))
            matches: list[tuple[int, str]] = []
            for row in root.iter(f"{_XLSX_NS}row"):
                values: dict[str, str] = {}
                for cell in row.findall(f"{_XLSX_NS}c"):
                    reference = cell.attrib.get("r", "")
                    column = "".join(character for character in reference if character.isalpha())
                    values[column] = _cell_value(cell, shared)
                if values.get("A") == artifact.filename:
                    matches.append((int(row.attrib["r"]), values.get("B", "")))
            if len(matches) != 1 or matches[0][1] != expected_command:
                raise ValueError(
                    f"official workbook row changed or is duplicated for {artifact.filename}"
                )
            filename_date = artifact.filename.removesuffix(".zip").rsplit("_", 3)[-3:]
            if "-".join(filename_date) != artifact.capture_date:
                raise ValueError(f"archive date does not match frozen capture date: {artifact.filename}")
            decisions.append(
                {
                    "product": artifact.product,
                    "sheet": artifact.workbook_sheet,
                    "worksheetPath": sheet_path,
                    "row": matches[0][0],
                    "filename": artifact.filename,
                    "captureDate": artifact.capture_date,
                    "url": artifact.url,
                    "expectedBytes": artifact.expected_bytes,
                }
            )
    return workbook, decisions


def _response_metadata(response: requests.Response) -> dict[str, Any]:
    return {
        "requestedUrl": response.request.url,
        "redirects": [
            {"url": prior.url, "status": prior.status_code} for prior in response.history
        ],
        "finalUrl": response.url,
        "status": response.status_code,
        "headers": {
            key: response.headers.get(key)
            for key in (
                "Content-Type",
                "Content-Length",
                "Content-Range",
                "Content-Disposition",
                "ETag",
                "Last-Modified",
                "Accept-Ranges",
                "Content-Encoding",
            )
        },
    }


def _validate_official_response(response: requests.Response, artifact: OrthophotoArtifact) -> None:
    final = urlsplit(response.url)
    if final.scheme != "https" or final.netloc not in _OFFICIAL_HOSTS:
        raise _DownloadIntegrityError(f"orthophoto redirected outside official hosts: {response.url}")
    content_type = response.headers.get("Content-Type", "").lower()
    if "html" in content_type:
        raise _DownloadIntegrityError(
            f"orthophoto endpoint returned HTML for {artifact.filename}"
        )
    if response.headers.get("Content-Encoding") not in (None, "", "identity"):
        raise _DownloadIntegrityError("encoded HTTP transfer cannot prove archive byte identity")


def _parse_content_range(
    response: requests.Response, start: int, end: int, total: int
) -> None:
    match = _CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
    if match is None or tuple(map(int, match.groups())) != (start, end, total):
        raise _DownloadIntegrityError(
            f"unexpected Content-Range for retained prefix: "
            f"{response.headers.get('Content-Range')!r}"
        )
    length = response.headers.get("Content-Length")
    if length is None or int(length) != end - start + 1:
        raise _DownloadIntegrityError("range response has an unexpected Content-Length")


def _compare_retained_prefix(
    session: PoliteSession,
    artifact: OrthophotoArtifact,
    part: Path,
    retained_bytes: int,
) -> dict[str, Any]:
    session._pace(artifact.url)
    headers = {"Range": f"bytes=0-{retained_bytes - 1}"}
    with session.session.get(artifact.url, headers=headers, stream=True, timeout=300) as response:
        if response.status_code in _RETRYABLE_HTTP:
            raise requests.HTTPError(str(response.status_code), response=response)
        response.raise_for_status()
        _validate_official_response(response, artifact)
        if response.status_code != 206:
            raise _RangeUnsupported("server refused byte-range prefix validation")
        _parse_content_range(response, 0, retained_bytes - 1, artifact.expected_bytes)
        compared = 0
        with part.open("rb") as local:
            for remote in response.iter_content(1 << 20):
                if not remote:
                    continue
                if local.read(len(remote)) != remote:
                    raise _DownloadIntegrityError(
                        f"retained prefix differs from official bytes: {artifact.filename}"
                    )
                compared += len(remote)
            if compared != retained_bytes or local.read(1):
                raise _DownloadIntegrityError("retained prefix length comparison was incomplete")
        return _response_metadata(response)


def _write_resume_state(
    state_path: Path,
    selection: OrthophotoSelection,
    artifact: OrthophotoArtifact,
    retained_bytes: int,
    attempts: int,
    initial_bytes: int,
) -> None:
    _write_atomic(
        state_path,
        _json_bytes(
            {
                "schemaVersion": "orthophoto-download-resume/1",
                "selectionId": selection.selection_id,
                "filename": artifact.filename,
                "url": artifact.url,
                "expectedBytes": artifact.expected_bytes,
                "initialRetainedBytes": initial_bytes,
                "retainedBytes": retained_bytes,
                "attempts": attempts,
                "updatedUtc": datetime.now(timezone.utc).isoformat(),
            }
        ),
    )


def _download_archive(
    session: PoliteSession,
    selection: OrthophotoSelection,
    artifact: OrthophotoArtifact,
    part: Path,
    log,
) -> dict[str, Any]:
    part.parent.mkdir(parents=True, exist_ok=True)
    state_path = part.with_suffix(part.suffix + ".state.json")
    staged_provenance = part.with_suffix(part.suffix + ".http.json")
    initial_bytes = part.stat().st_size if part.exists() else 0
    if initial_bytes > artifact.expected_bytes:
        raise ValueError(f"partial archive exceeds frozen byte count: {part}")
    prior_checkpoint_bytes: int | None = None
    if state_path.exists():
        state = json.loads(state_path.read_bytes())
        if (
            state.get("selectionId") != selection.selection_id
            or state.get("filename") != artifact.filename
            or state.get("url") != artifact.url
            or state.get("expectedBytes") != artifact.expected_bytes
        ):
            raise ValueError(f"partial archive resume state belongs to another artifact: {part}")
        prior_checkpoint_bytes = int(state.get("retainedBytes", -1))

    if initial_bytes == artifact.expected_bytes and staged_provenance.exists():
        retained_record = json.loads(staged_provenance.read_bytes())
        retained_sha256 = _sha256_file(part)
        if (
            retained_record.get("selectionId") != selection.selection_id
            or retained_record.get("product") != artifact.product
            or retained_record.get("filename") != artifact.filename
            or retained_record.get("requestedUrl") != artifact.url
            or retained_record.get("bytes") != artifact.expected_bytes
            or retained_record.get("sha256") != retained_sha256
            or (
                artifact.expected_sha256 is not None
                and retained_sha256 != artifact.expected_sha256
            )
        ):
            raise ValueError(f"staged HTTP provenance does not bind {part}")
        return retained_record

    last_request: dict[str, Any] | None = None
    range_validated_bytes = 0
    restart_events: list[dict[str, Any]] = []
    started = datetime.now(timezone.utc).isoformat()
    for attempt in range(1, session.cfg.max_retries + 1):
        retained = part.stat().st_size if part.exists() else 0
        _write_resume_state(
            state_path, selection, artifact, retained, attempt, initial_bytes
        )
        try:
            if retained:
                log(
                    f"validating {retained / 1e6:.1f} MB retained prefix for "
                    f"{artifact.filename}"
                )
                last_request = _compare_retained_prefix(session, artifact, part, retained)
                range_validated_bytes = retained
            if retained == artifact.expected_bytes:
                break

            headers = {"Range": f"bytes={retained}-{artifact.expected_bytes - 1}"} if retained else {}
            session._pace(artifact.url)
            with session.session.get(
                artifact.url, headers=headers, stream=True, timeout=300
            ) as response:
                if response.status_code in _RETRYABLE_HTTP:
                    raise requests.HTTPError(str(response.status_code), response=response)
                response.raise_for_status()
                _validate_official_response(response, artifact)
                if retained:
                    if response.status_code != 206:
                        raise _DownloadIntegrityError(
                            "server refused byte-range resume; partial bytes were not overwritten"
                        )
                    _parse_content_range(
                        response,
                        retained,
                        artifact.expected_bytes - 1,
                        artifact.expected_bytes,
                    )
                else:
                    length = response.headers.get("Content-Length")
                    if response.status_code != 200 or length is None:
                        raise _DownloadIntegrityError("fresh download lacks exact HTTP length")
                    if int(length) != artifact.expected_bytes:
                        raise _DownloadIntegrityError(
                            f"HTTP byte count changed for {artifact.filename}: {length}"
                        )
                last_request = _response_metadata(response)
                mode = "ab" if retained else "wb"
                with part.open(mode) as target:
                    for block in response.iter_content(1 << 20):
                        if not block:
                            continue
                        target.write(block)
                        if target.tell() > artifact.expected_bytes:
                            raise _DownloadIntegrityError("orthophoto response exceeded frozen size")
                    target.flush()
                    os.fsync(target.fileno())
            if part.stat().st_size != artifact.expected_bytes:
                raise requests.ConnectionError(
                    f"short orthophoto response retained {part.stat().st_size} bytes"
                )
            break
        except _RangeUnsupported:
            digest = _sha256_file(part)
            quarantine = part.parent / "unresumable" / digest / part.name
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            if quarantine.exists():
                if _sha256_file(quarantine) != digest:
                    raise ValueError(f"unresumable-prefix quarantine collision: {quarantine}")
                part.unlink()
            else:
                part.replace(quarantine)
            if state_path.exists():
                quarantine_state = quarantine.with_suffix(quarantine.suffix + ".state.json")
                if quarantine_state.exists():
                    state_path.unlink()
                else:
                    state_path.replace(quarantine_state)
            restart_events.append(
                {
                    "reason": "server_range_unsupported",
                    "retainedBytes": retained,
                    "sha256": digest,
                    "quarantine": quarantine.relative_to(part.parent).as_posix(),
                }
            )
            initial_bytes = 0
            range_validated_bytes = 0
            log(
                f"server does not support safe resume for {artifact.filename}; "
                "preserved prefix and restarting cleanly"
            )
            continue
        except _DownloadIntegrityError:
            _write_resume_state(
                state_path,
                selection,
                artifact,
                part.stat().st_size if part.exists() else 0,
                attempt,
                initial_bytes,
            )
            raise
        except requests.RequestException as error:
            _write_resume_state(
                state_path,
                selection,
                artifact,
                part.stat().st_size if part.exists() else 0,
                attempt,
                initial_bytes,
            )
            if attempt == session.cfg.max_retries or (
                isinstance(error, requests.HTTPError)
                and error.response is not None
                and error.response.status_code not in _RETRYABLE_HTTP
            ):
                raise
            log(f"retrying {artifact.filename} after {type(error).__name__}: {error}")
            time.sleep(2.0 ** (attempt - 1))
    else:
        raise AssertionError("unreachable")

    if not part.is_file() or part.stat().st_size != artifact.expected_bytes:
        raise _DownloadIntegrityError(f"orthophoto download did not reach exact size: {part}")
    with part.open("rb") as source:
        if source.read(4) != b"PK\x03\x04":
            raise _DownloadIntegrityError(f"orthophoto response is not a ZIP archive: {part}")
    if last_request is None:
        raise AssertionError("completed orthophoto download lacks an HTTP identity request")
    record = {
        "schemaVersion": "orthophoto-http-provenance/1",
        "selectionId": selection.selection_id,
        "product": artifact.product,
        "filename": artifact.filename,
        "requestedUrl": artifact.url,
        "fetchStartedUtc": started,
        "fetchCompletedUtc": datetime.now(timezone.utc).isoformat(),
        "initialRetainedBytes": initial_bytes,
        "rangeValidatedBytes": range_validated_bytes,
        "checkpointRetainedBytes": prior_checkpoint_bytes,
        "restartEvents": restart_events,
        "finalRequest": last_request,
        "bytes": artifact.expected_bytes,
        "sha256": _sha256_file(part),
    }
    if (
        artifact.expected_sha256 is not None
        and record["sha256"] != artifact.expected_sha256
    ):
        raise _DownloadIntegrityError(
            f"orthophoto SHA-256 changed for {artifact.filename}: {record['sha256']}"
        )
    _write_atomic(staged_provenance, _json_bytes(record))
    return record


def _safe_zip_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    result: list[zipfile.ZipInfo] = []
    seen: set[str] = set()
    for info in archive.infolist():
        relative = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        if (
            relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
            or "\\" in info.filename
            or info.filename in seen
            or stat.S_ISLNK(mode)
            or info.flag_bits & 0x1
        ):
            raise ValueError(f"unsafe orthophoto ZIP member: {info.filename!r}")
        seen.add(info.filename)
        if not info.is_dir():
            result.append(info)
    if not result:
        raise ValueError("orthophoto ZIP contains no regular files")
    if sum(info.file_size for info in result) > 8 * (1 << 30):
        raise ValueError("orthophoto ZIP exceeds the bounded extraction budget")
    return result


def _extract_archive(archive_path: Path, destination: Path) -> list[dict[str, Any]]:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"orthophoto ZIP CRC failed for member {bad!r}")
        members = _safe_zip_members(archive)
        extracted: list[dict[str, Any]] = []
        for info in members:
            relative = PurePosixPath(info.filename)
            output = destination.joinpath(*relative.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with archive.open(info) as source, output.open("xb") as target:
                while block := source.read(1 << 20):
                    digest.update(block)
                    target.write(block)
            extracted.append(
                {
                    "relativePath": relative.as_posix(),
                    "bytes": output.stat().st_size,
                    "sha256": digest.hexdigest(),
                    "zipCrc32": f"{info.CRC:08x}",
                    "compressedBytes": info.compress_size,
                }
            )
    return extracted


def _validate_geotiff(
    root: Path,
    selection: OrthophotoSelection,
    artifact: OrthophotoArtifact,
) -> dict[str, Any]:
    candidates = [
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    ]
    if len(candidates) != 1:
        raise ValueError(f"{artifact.product} archive must contain exactly one GeoTIFF")
    path = candidates[0]
    expected_transform = Affine(
        artifact.pixel_size_m,
        0.0,
        selection.sheet_bounds[0],
        0.0,
        -artifact.pixel_size_m,
        selection.sheet_bounds[3],
    )
    with rasterio.open(path) as source:
        bounds = tuple(float(value) for value in source.bounds)
        if (
            source.driver != "GTiff"
            or source.crs is None
            or source.crs.to_epsg() != 3301
            or source.width != artifact.raster_size
            or source.height != artifact.raster_size
            or source.count != 3
            or tuple(source.dtypes) != ("uint8", "uint8", "uint8")
            or not source.transform.almost_equals(expected_transform, precision=1e-9)
            or any(
                not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6)
                for actual, expected in zip(
                    bounds, selection.sheet_bounds, strict=True
                )
            )
        ):
            raise ValueError(
                f"unexpected {artifact.product} GeoTIFF schema: driver={source.driver}, "
                f"crs={source.crs}, size={source.width}x{source.height}, "
                f"bands={source.count}, dtypes={source.dtypes}, transform={source.transform}, "
                f"bounds={bounds}"
            )
        corners = (
            Window(0, 0, 1, 1),
            Window(source.width - 1, 0, 1, 1),
            Window(0, source.height - 1, 1, 1),
            Window(source.width - 1, source.height - 1, 1, 1),
        )
        for window in corners:
            sample = source.read((1, 2, 3), window=window)
            if sample.shape != (3, 1, 1):
                raise ValueError(f"{artifact.product} GeoTIFF corner read failed")
        return {
            "relativePath": path.relative_to(root).as_posix(),
            "driver": source.driver,
            "crs": source.crs.to_string(),
            "epsg": source.crs.to_epsg(),
            "width": source.width,
            "height": source.height,
            "bands": source.count,
            "dtypes": list(source.dtypes),
            "transform": list(source.transform)[:6],
            "bounds": list(bounds),
            "res": list(source.res),
            "colorInterp": [item.name for item in source.colorinterp],
            "nodata": source.nodata,
        }


def _verify_extraction(
    root: Path,
    selection: OrthophotoSelection,
    artifact: OrthophotoArtifact,
    archive_sha256: str,
    document: dict[str, Any],
) -> None:
    if (
        document.get("schemaVersion") != "orthophoto-extraction-retention/1"
        or document.get("selectionId") != selection.selection_id
        or document.get("sheet") != selection.sheet
        or document.get("product") != artifact.product
        or document.get("archiveFilename") != artifact.filename
        or document.get("archiveSha256") != archive_sha256
        or document.get("captureDate") != artifact.capture_date
        or document.get("zipCrcVerified") is not True
    ):
        raise ValueError(f"orthophoto extraction identity changed: {root}")
    for entry in document.get("files", []):
        relative = PurePosixPath(str(entry.get("relativePath", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("orthophoto extraction manifest contains an unsafe path")
        path = root.joinpath(*relative.parts)
        if (
            not path.is_file()
            or path.stat().st_size != entry.get("bytes")
            or _sha256_file(path) != entry.get("sha256")
        ):
            raise ValueError(f"retained orthophoto extraction changed: {path}")
    if _validate_geotiff(root, selection, artifact) != document.get("geoTiff"):
        raise ValueError(f"retained {artifact.product} GeoTIFF metadata changed")


def _materialize_extraction(
    part: Path,
    product_root: Path,
    selection: OrthophotoSelection,
    artifact: OrthophotoArtifact,
    archive_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    final_root = product_root / archive_sha256
    manifest_name = "retention.json"
    if final_root.exists():
        manifest = final_root / manifest_name
        if not manifest.is_file():
            raise ValueError(f"orphaned content-addressed extraction: {final_root}")
        document = json.loads(manifest.read_bytes())
        _verify_extraction(
            final_root, selection, artifact, archive_sha256, document
        )
        return manifest, document

    temporary = product_root / f".{archive_sha256}.extracting"
    files = _extract_archive(part, temporary)
    geotiff = _validate_geotiff(temporary, selection, artifact)
    document = {
        "schemaVersion": "orthophoto-extraction-retention/1",
        "selectionId": selection.selection_id,
        "sheet": selection.sheet,
        "product": artifact.product,
        "captureDate": artifact.capture_date,
        "archiveFilename": artifact.filename,
        "archiveBytes": artifact.expected_bytes,
        "archiveSha256": archive_sha256,
        "zipCrcVerified": True,
        "files": files,
        "geoTiff": geotiff,
    }
    _write_immutable(temporary / manifest_name, _json_bytes(document))
    temporary.replace(final_root)
    return final_root / manifest_name, document


def _promote_json_content_addressed(
    root: Path, filename: str, document: dict[str, Any]
) -> tuple[Path, str]:
    payload = _json_bytes(document)
    digest = hashlib.sha256(payload).hexdigest()
    path = root / "sha256" / digest / filename
    _write_immutable(path, payload)
    return path, digest


def _verify_product_retention(
    orthophoto_root: Path,
    selection: OrthophotoSelection,
    artifact: OrthophotoArtifact,
    pointer: Path,
) -> dict[str, Any]:
    document = json.loads(pointer.read_bytes())
    product_root = pointer.parent
    archive = product_root / artifact.filename
    archive_sha256 = str(document.get("archive", {}).get("sha256", ""))
    sidecar = archive.with_suffix(archive.suffix + ".sha256")
    provenance = orthophoto_root / str(document.get("httpProvenance", {}).get("path", ""))
    provenance_document = json.loads(provenance.read_bytes()) if provenance.is_file() else {}
    final_request = provenance_document.get("finalRequest", {})
    final_url = urlsplit(str(final_request.get("finalUrl", "")))
    extraction = product_root / archive_sha256
    extraction_manifest = extraction / "retention.json"
    pointer_sha256 = _sha256_file(pointer)
    content_addressed = (
        product_root / "retention" / "sha256" / pointer_sha256 / "retained.json"
    )
    if (
        document.get("schemaVersion") != "orthophoto-product-retention/1"
        or document.get("selectionId") != selection.selection_id
        or document.get("product") != artifact.product
        or document.get("sheet") != selection.sheet
        or document.get("captureDate") != artifact.capture_date
        or document.get("sourceUrl") != artifact.url
        or document.get("archive", {}).get("filename") != artifact.filename
        or document.get("archive", {}).get("bytes") != artifact.expected_bytes
        or not archive.is_file()
        or archive.stat().st_size != artifact.expected_bytes
        or _sha256_file(archive) != archive_sha256
        or (
            artifact.expected_sha256 is not None
            and archive_sha256 != artifact.expected_sha256
        )
        or not sidecar.is_file()
        or sidecar.read_text(encoding="ascii").strip() != archive_sha256
        or not provenance.is_file()
        or _sha256_file(provenance) != document.get("httpProvenance", {}).get("sha256")
        or provenance_document.get("selectionId") != selection.selection_id
        or provenance_document.get("product") != artifact.product
        or provenance_document.get("filename") != artifact.filename
        or provenance_document.get("requestedUrl") != artifact.url
        or provenance_document.get("bytes") != artifact.expected_bytes
        or provenance_document.get("sha256") != archive_sha256
        or final_request.get("status") not in (200, 206)
        or final_url.scheme != "https"
        or final_url.netloc not in _OFFICIAL_HOSTS
        or not extraction_manifest.is_file()
        or _sha256_file(extraction_manifest) != document.get("extraction", {}).get(
            "manifestSha256"
        )
        or not content_addressed.is_file()
        or content_addressed.read_bytes() != pointer.read_bytes()
    ):
        raise ValueError(f"retained orthophoto product tuple changed: {pointer}")
    extraction_document = json.loads(extraction_manifest.read_bytes())
    _verify_extraction(
        extraction, selection, artifact, archive_sha256, extraction_document
    )
    return document


def _fetch_product(
    orthophoto_root: Path,
    workbook_sha256: str,
    workbook_row: dict[str, Any],
    selection: OrthophotoSelection,
    artifact: OrthophotoArtifact,
    session: PoliteSession,
    log,
) -> tuple[Path, dict[str, Any]]:
    product_root = orthophoto_root / artifact.product / selection.sheet
    pointer = product_root / "retained.json"
    if pointer.exists():
        return pointer, _verify_product_retention(
            orthophoto_root, selection, artifact, pointer
        )

    archive = product_root / artifact.filename
    if archive.exists():
        raise ValueError(f"orphaned orthophoto archive requires manual audit: {archive}")
    part = archive.with_suffix(archive.suffix + ".part")
    http_record = _download_archive(session, selection, artifact, part, log)
    archive_sha256 = _sha256_file(part)
    if http_record.get("sha256") != archive_sha256:
        raise ValueError(f"staged orthophoto HTTP provenance changed: {part}")
    extraction_manifest, extraction_document = _materialize_extraction(
        part, product_root, selection, artifact, archive_sha256
    )

    provenance_path, provenance_sha256 = _promote_json_content_addressed(
        product_root / "provenance" / "http", f"{artifact.filename}.json", http_record
    )
    sidecar_temporary = archive.with_suffix(archive.suffix + ".sha256.part")
    sidecar_temporary.write_text(archive_sha256 + "\n", encoding="ascii")
    part.replace(archive)
    sidecar_temporary.replace(archive.with_suffix(archive.suffix + ".sha256"))
    part.with_suffix(part.suffix + ".state.json").unlink(missing_ok=True)
    part.with_suffix(part.suffix + ".http.json").unlink(missing_ok=True)

    product_document = {
        "schemaVersion": "orthophoto-product-retention/1",
        "selectionId": selection.selection_id,
        "sheet": selection.sheet,
        "product": artifact.product,
        "captureDate": artifact.capture_date,
        "sourceType": artifact.source_type,
        "sourceUrl": artifact.url,
        "workbookSha256": workbook_sha256,
        "workbookDecision": workbook_row,
        "archive": {
            "filename": artifact.filename,
            "bytes": artifact.expected_bytes,
            "sha256": archive_sha256,
        },
        "httpProvenance": {
            "path": _safe_relative(provenance_path, orthophoto_root),
            "sha256": provenance_sha256,
        },
        "extraction": {
            "root": _safe_relative(extraction_manifest.parent, orthophoto_root),
            "manifest": _safe_relative(extraction_manifest, orthophoto_root),
            "manifestSha256": _sha256_file(extraction_manifest),
            "fileCount": len(extraction_document["files"]),
        },
        "attribution": _ATTRIBUTION,
    }
    content_path, _ = _promote_json_content_addressed(
        product_root / "retention", "retained.json", product_document
    )
    _write_immutable(pointer, content_path.read_bytes())
    return pointer, _verify_product_retention(
        orthophoto_root, selection, artifact, pointer
    )


def fetch_frozen_orthophoto_sheet(
    base: BaseConfig,
    sheet: str,
    *,
    log=print,
) -> Path:
    """Fetch and verify RGB/CIR artifacts for one explicitly frozen sheet."""
    development_selections, selection_manifest, maximum_attempts = (
        _load_development_selections()
    )
    selection = _SELECTIONS.get(sheet) or development_selections.get(sheet)
    if selection is None:
        raise ValueError(f"orthophoto sheet is not frozen for acquisition: {sheet}")
    workbook, decisions = _validate_workbook(selection)
    workbook_sha256 = hashlib.sha256(workbook).hexdigest()
    orthophoto_root = (DATA_IN / "orthophoto").resolve()
    stage_root = orthophoto_root / selection.retention_directory
    snapshot = stage_root / "snapshots" / "workbook" / workbook_sha256 / _WORKBOOK.name
    _write_immutable(snapshot, workbook)

    fetch_config = (
        replace(base.fetch, max_retries=maximum_attempts)
        if sheet in development_selections
        else base.fetch
    )
    session = PoliteSession(fetch_config)
    products: list[dict[str, Any]] = []
    for index, (artifact, decision) in enumerate(
        zip(selection.artifacts, decisions, strict=True), 1
    ):
        log(
            f"[{index}/{len(selection.artifacts)}] {selection.log_label} "
            f"{artifact.product.upper()}"
        )
        pointer, document = _fetch_product(
            orthophoto_root,
            workbook_sha256,
            decision,
            selection,
            artifact,
            session,
            log,
        )
        products.append(
            {
                "product": artifact.product,
                "captureDate": artifact.capture_date,
                "retention": _safe_relative(pointer, orthophoto_root),
                "retentionSha256": _sha256_file(pointer),
                "archive": document["archive"],
                "extraction": document["extraction"],
            }
        )
        log(f"verified {artifact.filename} ({artifact.expected_bytes / 1e6:.1f} MB)")

    global_document = {
        "schemaVersion": selection.schema,
        "selectionId": selection.selection_id,
        "sheet": selection.sheet,
        "complete": True,
        "products": products,
        "workbook": {
            "source": _safe_relative(_WORKBOOK, ASSET_GEN_ROOT.parent),
            "bytes": len(workbook),
            "sha256": workbook_sha256,
            "snapshot": _safe_relative(snapshot, orthophoto_root),
            "decisions": decisions,
        },
        "selectionManifest": (
            selection_manifest if sheet in development_selections else None
        ),
        "attribution": _ATTRIBUTION,
        "temporalPolicy": selection.temporal_policy,
    }
    content_path, _ = _promote_json_content_addressed(
        stage_root / "retention", "retained.json", global_document
    )
    output = stage_root / "retained.json"
    _write_immutable(output, content_path.read_bytes())
    return output


def fetch_taevaskoda_stage1_orthophoto(base: BaseConfig, *, log=print) -> Path:
    """Fetch the exact accepted Taevaskoda Stage-1 RGB/CIR snapshot."""
    return fetch_frozen_orthophoto_sheet(base, "54472", log=log)
