"""Strict, resumable retention of the frozen Hovi public evidence tranches."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from urllib.parse import urlsplit

import requests

from ..config import CONFIG_DIR, DATA_IN, BaseConfig, FetchConfig, load_base

_SELECTION_SCHEMA = "hovi-public-target-selection/1.0.0"
_SELECTION_ID = "hovi-2024-jarvselja-hyytiala-first-conversion-v1"
_DATASET_UUID = "ace2a123-00ff-4944-951e-eddbe209b70c"
_AUTHORIZATION_URL = "https://etsin.fairdata.fi/api/v3/download/authorize"
_SIGNED_URL_HOST = "download.fairdata.fi"
_PLAN_SCHEMA = "hovi-retention-plan/1.0.0"
_MANIFEST_SCHEMA = "hovi-retained-evidence/1.0.0"
_PRIMARY_PLOT = "HY_SPRUCE4"
_TRANCHE_ORDER = ("shared", "hy-spruce4-photos", "hy-spruce4-geometry")
_PHOTO_KINDS = frozenset(
    {"semantic_quadrat_photo", "context_overview_photo", "semantic_transect_photo"}
)
_GEOMETRY_KINDS = frozenset(
    {"merged_thinned_geometry_preview", "registration_diagnostic"}
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_CONTENT_RANGE_RE = re.compile(r"^bytes ([0-9]+)-([0-9]+)/([0-9]+|\*)$")
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_TOP_LEVEL_KEYS = {
    "schema_version",
    "id",
    "purpose",
    "status",
    "source",
    "license",
    "access",
    "budget",
    "selection",
    "split_contract",
    "qualification",
    "shared_files",
    "plots",
    "validation",
}
_FILE_KEYS = {"file_id", "path", "bytes", "sha256", "kind"}


class _RetryableFetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class HoviArtifact:
    file_id: str
    pathname: str
    expected_bytes: int
    expected_sha256: str
    kind: str
    tranche: str
    plot_id: str | None

    def identity(self) -> dict[str, Any]:
        return {
            "dataset_uuid": _DATASET_UUID,
            "file_id": self.file_id,
            "path": self.pathname,
            "bytes": self.expected_bytes,
            "sha256": self.expected_sha256,
            "kind": self.kind,
            "tranche": self.tranche,
            "plot_id": self.plot_id,
        }


@dataclass(frozen=True)
class HoviRetentionPlan:
    selection_path: Path
    selection_sha256: str
    raw: dict[str, Any]
    artifacts: tuple[HoviArtifact, ...]
    plan_identity: dict[str, Any]
    plan_sha256: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_from_raw(
    item: dict[str, Any], *, tranche: str, plot_id: str | None
) -> HoviArtifact:
    if set(item) != _FILE_KEYS:
        raise ValueError("Hovi selection file tuple has unknown or missing fields")
    file_id = item["file_id"]
    pathname = item["path"]
    byte_count = item["bytes"]
    sha256 = item["sha256"]
    kind = item["kind"]
    if not isinstance(file_id, str) or not _UUID_RE.fullmatch(file_id):
        raise ValueError(f"invalid Fairdata file UUID for {pathname!r}")
    if not isinstance(pathname, str) or not pathname.startswith("/"):
        raise ValueError(f"Hovi stable pathname must be absolute: {pathname!r}")
    pure_path = PurePosixPath(pathname)
    if (
        "\\" in pathname
        or any(part in {"", ".", ".."} for part in pure_path.parts[1:])
        or "/" + "/".join(pure_path.parts[1:]) != pathname
    ):
        raise ValueError(f"unsafe or non-canonical Hovi pathname: {pathname!r}")
    if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count <= 0:
        raise ValueError(f"invalid expected byte count for {pathname}")
    if not isinstance(sha256, str) or not _SHA256_RE.fullmatch(sha256):
        raise ValueError(f"invalid expected SHA-256 for {pathname}")
    if not isinstance(kind, str) or not kind:
        raise ValueError(f"invalid evidence kind for {pathname}")
    return HoviArtifact(
        file_id=file_id,
        pathname=pathname,
        expected_bytes=byte_count,
        expected_sha256=sha256,
        kind=kind,
        tranche=tranche,
        plot_id=plot_id,
    )


def _validate_selection(raw: dict[str, Any]) -> tuple[HoviArtifact, ...]:
    if set(raw) != _TOP_LEVEL_KEYS:
        raise ValueError("unsupported or non-strict Hovi selection config")
    if raw["schema_version"] != _SELECTION_SCHEMA or raw["id"] != _SELECTION_ID:
        raise ValueError("Hovi selection schema or identity changed")
    if raw["status"] != "raw_candidate_unqualified":
        raise ValueError("Hovi selection must remain explicitly unqualified raw evidence")

    source = raw["source"]
    if (
        source.get("dataset_uuid") != _DATASET_UUID
        or source.get("dataset_doi")
        != "10.23729/9a8d90cd-73e2-438d-9230-94e10e61adc9"
        or source.get("dataset_version") != 1
        or source.get("published_revision") != 1
    ):
        raise ValueError("Hovi controlling dataset identity changed")
    license_record = raw["license"]
    if (
        license_record.get("spdx") != "CC-BY-4.0"
        or license_record.get("url")
        != "https://creativecommons.org/licenses/by/4.0/legalcode"
        or license_record.get("attribution_required") is not True
        or license_record.get("changes_must_be_indicated") is not True
    ):
        raise ValueError("Hovi controlling license identity changed")
    access = raw["access"]
    if (
        access.get("access_type") != "open"
        or access.get("authorization_url") != _AUTHORIZATION_URL
        or access.get("authorization_method") != "POST"
        or access.get("authorization_json_template")
        != {"cr_id": _DATASET_UUID, "file": "<full pathname>"}
        or access.get("signed_url_policy")
        != "Generate at fetch time; never persist signed URLs."
        or access.get("fetch_policy")
        != "Fetch to .part, verify bytes and SHA-256, then atomically rename."
    ):
        raise ValueError("Hovi access contract changed")

    selection = raw["selection"]
    plots = raw["plots"]
    plot_order = selection.get("plot_order")
    actual_plot_order = [plot.get("plot_id") for plot in plots]
    if plot_order != [_PRIMARY_PLOT, "HY_PINE2", "JS_SPRUCE1"]:
        raise ValueError("Hovi plot order changed")
    if actual_plot_order != plot_order:
        raise ValueError("Hovi plot records no longer follow their frozen order")
    if (
        raw["split_contract"].get("blind_sites") != ["hovi.jarvselja"]
        or raw["split_contract"].get("development_sites") != ["hovi.hyytiala"]
        or raw["qualification"].get("target_truth") is not False
        or raw["qualification"].get("synthesis_authorized") is not False
    ):
        raise ValueError("Hovi split or qualification boundary changed")

    shared_raw = raw["shared_files"]
    if (
        len(shared_raw) != selection.get("shared_file_count")
        or sum(item.get("bytes", 0) for item in shared_raw)
        != selection.get("shared_expected_bytes")
    ):
        raise ValueError("Hovi shared-file accounting changed")

    all_records: list[HoviArtifact] = []
    shared = [
        _artifact_from_raw(item, tranche="shared", plot_id=None) for item in shared_raw
    ]
    all_records.extend(shared)
    plot_records: dict[str, list[HoviArtifact]] = {}
    for plot in plots:
        plot_id = plot["plot_id"]
        files = [
            _artifact_from_raw(item, tranche="validation-only", plot_id=plot_id)
            for item in plot["files"]
        ]
        if (
            len(files) != plot.get("expected_file_count")
            or sum(item.expected_bytes for item in files) != plot.get("expected_bytes")
        ):
            raise ValueError(f"Hovi file accounting changed for {plot_id}")
        plot_records[plot_id] = files
        all_records.extend(files)

    if (
        len(all_records) != selection.get("file_count")
        or sum(item.expected_bytes for item in all_records)
        != selection.get("expected_bytes")
    ):
        raise ValueError("Hovi complete selection accounting changed")
    if len({item.file_id for item in all_records}) != len(all_records):
        raise ValueError("duplicate Fairdata file UUID in Hovi selection")
    if len({item.pathname.casefold() for item in all_records}) != len(all_records):
        raise ValueError("duplicate stable pathname in Hovi selection")

    primary_plot = plots[0]
    if (
        primary_plot.get("role") != "development_primary"
        or primary_plot.get("site_id") != "hovi.hyytiala"
        or primary_plot.get("sealed_until_converter_freeze") is not False
    ):
        raise ValueError("HY_SPRUCE4 is no longer the unsealed primary development plot")
    blind_plot = plots[2]
    if (
        blind_plot.get("role") != "blind_estonia_transfer"
        or blind_plot.get("site_id") != "hovi.jarvselja"
        or blind_plot.get("sealed_until_converter_freeze") is not True
    ):
        raise ValueError("JS_SPRUCE1 blind seal changed")

    primary_files = plot_records[_PRIMARY_PLOT]
    photo_files = [item for item in primary_files if item.kind in _PHOTO_KINDS]
    geometry_files = [item for item in primary_files if item.kind in _GEOMETRY_KINDS]
    unexpected = [
        item.kind for item in primary_files if item.kind not in _PHOTO_KINDS | _GEOMETRY_KINDS
    ]
    if len(photo_files) != 10 or len(geometry_files) != 2 or unexpected:
        raise ValueError("HY_SPRUCE4 does not contain the frozen 10-photo + 2-geometry split")

    # Only this explicit allow-list is returned to the network path. The remaining
    # development plot and sealed blind plot are validated above but never authorized.
    return tuple(
        shared
        + [replace(item, tranche="hy-spruce4-photos") for item in photo_files]
        + [replace(item, tranche="hy-spruce4-geometry") for item in geometry_files]
    )


def load_hovi_retention_plan(selection_path: Path | None = None) -> HoviRetentionPlan:
    path = selection_path or CONFIG_DIR / "hovi-public-targets.json"
    selection_bytes = path.read_bytes()
    raw = json.loads(selection_bytes)
    if not isinstance(raw, dict):
        raise ValueError("Hovi selection root must be a JSON object")
    artifacts = _validate_selection(raw)
    selection_sha256 = hashlib.sha256(selection_bytes).hexdigest()
    plan_identity = {
        "schema_version": _PLAN_SCHEMA,
        "selection": {
            "schema_version": raw["schema_version"],
            "id": raw["id"],
            "status": raw["status"],
            "config_sha256": selection_sha256,
        },
        "dataset": raw["source"],
        "license": raw["license"],
        "access": raw["access"],
        "ordered_tranches": list(_TRANCHE_ORDER),
        "authorized_scope": "shared-and-hy-spruce4-only",
        "files": [item.identity() for item in artifacts],
    }
    plan_sha256 = hashlib.sha256(_canonical_json(plan_identity)).hexdigest()
    return HoviRetentionPlan(
        selection_path=path,
        selection_sha256=selection_sha256,
        raw=raw,
        artifacts=artifacts,
        plan_identity=plan_identity,
        plan_sha256=plan_sha256,
    )


class _FairdataClient:
    def __init__(self, cfg: FetchConfig):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers["User-Agent"] = cfg.user_agent
        self._last_request: dict[str, float] = {}

    def _pace(self, url: str) -> None:
        host = urlsplit(url).netloc
        wait = self._last_request.get(host, 0.0) + self.cfg.min_interval_s - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_request[host] = time.monotonic()

    def _authorize(self, pathname: str) -> str:
        last_error = "authorization failed"
        for attempt in range(self.cfg.max_retries):
            self._pace(_AUTHORIZATION_URL)
            try:
                response = self.session.post(
                    _AUTHORIZATION_URL,
                    json={"cr_id": _DATASET_UUID, "file": pathname},
                    timeout=60,
                )
                if response.status_code in _RETRYABLE_STATUS:
                    raise _RetryableFetchError(f"HTTP {response.status_code}")
                if response.status_code < 200 or response.status_code >= 300:
                    raise RuntimeError(f"HTTP {response.status_code}")
                signed_url = _signed_url_from_response(response)
                parsed = urlsplit(signed_url)
                if parsed.scheme != "https" or parsed.hostname != _SIGNED_URL_HOST:
                    raise RuntimeError("authorization returned a non-Fairdata download URL")
                return signed_url
            except (requests.RequestException, _RetryableFetchError, RuntimeError) as error:
                last_error = type(error).__name__
                if attempt + 1 < self.cfg.max_retries:
                    time.sleep(2.0**attempt)
                    continue
                raise RuntimeError(
                    f"Fairdata authorization failed for stable path {pathname}: {last_error}"
                ) from None
        raise AssertionError("unreachable")

    def download(
        self,
        artifact: HoviArtifact,
        destination: Path,
        *,
        log: Callable[[str], None],
    ) -> bool:
        if destination.exists():
            size = destination.stat().st_size
            digest = _sha256_file(destination)
            if size != artifact.expected_bytes or digest != artifact.expected_sha256:
                raise ValueError(f"retained Hovi artifact failed verification: {destination}")
            return False

        destination.parent.mkdir(parents=True, exist_ok=True)
        part = destination.with_name(destination.name + ".part")
        for attempt in range(self.cfg.max_retries):
            try:
                self._download_attempt(artifact, part, log=log)
                size = part.stat().st_size
                digest = _sha256_file(part)
                if size != artifact.expected_bytes or digest != artifact.expected_sha256:
                    part.unlink(missing_ok=True)
                    raise _RetryableFetchError("downloaded bytes failed the frozen tuple")
                part.replace(destination)
                return True
            except (requests.RequestException, _RetryableFetchError) as error:
                if attempt + 1 < self.cfg.max_retries:
                    time.sleep(2.0**attempt)
                    continue
                raise RuntimeError(
                    f"Fairdata download failed for stable path {artifact.pathname}: "
                    f"{type(error).__name__}"
                ) from None
        raise AssertionError("unreachable")

    def _download_attempt(
        self,
        artifact: HoviArtifact,
        part: Path,
        *,
        log: Callable[[str], None],
    ) -> None:
        if part.exists() and part.stat().st_size > artifact.expected_bytes:
            part.unlink()
        offset = part.stat().st_size if part.exists() else 0
        if offset == artifact.expected_bytes:
            return

        signed_url = self._authorize(artifact.pathname)
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        self._pace(signed_url)
        with self.session.get(
            signed_url,
            headers=headers,
            stream=True,
            timeout=(60, 300),
        ) as response:
            status = response.status_code
            if status in _RETRYABLE_STATUS:
                raise _RetryableFetchError(f"HTTP {status}")
            if offset and status == 206:
                _validate_content_range(
                    response.headers.get("Content-Range"),
                    offset,
                    artifact.expected_bytes,
                )
                mode = "ab"
            elif status == 200:
                offset = 0
                mode = "wb"
            else:
                raise _RetryableFetchError(f"unexpected HTTP status {status}")
            if "text/html" in response.headers.get("Content-Type", "").lower():
                raise _RetryableFetchError("download endpoint returned HTML")

            written = offset
            next_progress = ((written // (256 << 20)) + 1) * (256 << 20)
            with part.open(mode) as target:
                for block in response.iter_content(1 << 20):
                    if not block:
                        continue
                    written += len(block)
                    if written > artifact.expected_bytes:
                        raise _RetryableFetchError("download exceeded expected byte count")
                    target.write(block)
                    if written >= next_progress:
                        percent = 100.0 * written / artifact.expected_bytes
                        log(f"  {written:,}/{artifact.expected_bytes:,} bytes ({percent:.1f}%)")
                        next_progress += 256 << 20
                target.flush()
                os.fsync(target.fileno())
        if written != artifact.expected_bytes:
            raise _RetryableFetchError(
                f"partial response ended at {written} of {artifact.expected_bytes} bytes"
            )


def _signed_url_from_response(response: requests.Response) -> str:
    value: Any = None
    try:
        value = response.json()
    except ValueError:
        value = response.text.strip()
    if isinstance(value, str):
        candidate = value.strip().strip('"')
    elif isinstance(value, dict):
        candidate = next(
            (
                value[key]
                for key in ("url", "download_url", "downloadUrl")
                if isinstance(value.get(key), str)
            ),
            "",
        )
    else:
        candidate = ""
    if not candidate:
        raise RuntimeError("authorization response did not contain a signed URL")
    return candidate


def _validate_content_range(
    header: str | None, expected_start: int, expected_total: int
) -> None:
    match = _CONTENT_RANGE_RE.fullmatch(header or "")
    if (
        match is None
        or int(match.group(1)) != expected_start
        or int(match.group(2)) < expected_start
        or int(match.group(2)) >= expected_total
        or match.group(3) == "*"
        or int(match.group(3)) != expected_total
    ):
        raise _RetryableFetchError("invalid HTTP Content-Range for resumed download")


def _destination(files_root: Path, artifact: HoviArtifact) -> Path:
    relative = Path(*PurePosixPath(artifact.pathname).parts[1:])
    destination = files_root / relative
    resolved_root = files_root.resolve()
    if not destination.resolve(strict=False).is_relative_to(resolved_root):
        raise ValueError(f"Hovi stable pathname escaped retention root: {artifact.pathname}")
    return destination


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with part.open("w", encoding="utf-8") as target:
        json.dump(manifest, target, indent=2, sort_keys=True)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    part.replace(path)


def _load_existing_manifest(path: Path, plan: HoviRetentionPlan) -> dict[str, Any] | None:
    if not path.exists():
        return None
    manifest = json.loads(path.read_bytes())
    if (
        manifest.get("schema_version") != _MANIFEST_SCHEMA
        or manifest.get("retention_id") != plan.plan_sha256
        or manifest.get("plan_sha256") != plan.plan_sha256
        or manifest.get("selection", {}).get("config_sha256") != plan.selection_sha256
    ):
        raise ValueError(f"existing Hovi retention manifest has a different identity: {path}")
    return manifest


def _manifest(
    plan: HoviRetentionPlan,
    through: str,
    completed: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    through_index = _TRANCHE_ORDER.index(through)
    requested_tranches = list(_TRANCHE_ORDER[: through_index + 1])
    requested = [item for item in plan.artifacts if item.tranche in requested_tranches]
    rows: list[dict[str, Any]] = []
    for artifact in plan.artifacts:
        retained = completed.get(artifact.file_id)
        row = {
            **artifact.identity(),
            "relative_path": str(
                Path("files", *PurePosixPath(artifact.pathname).parts[1:])
            ),
            "status": "verified" if retained is not None else "pending",
        }
        if retained is not None:
            row.update(retained)
        rows.append(row)
    completed_ids = set(completed)
    completed_tranches: list[str] = []
    for tranche in _TRANCHE_ORDER:
        if all(
            item.file_id in completed_ids
            for item in plan.artifacts
            if item.tranche == tranche
        ):
            completed_tranches.append(tranche)
        else:
            break
    requested_complete = all(item.file_id in completed_ids for item in requested)
    return {
        "schema_version": _MANIFEST_SCHEMA,
        "retention_id": plan.plan_sha256,
        "plan_sha256": plan.plan_sha256,
        "plan_identity": plan.plan_identity,
        "selection": {
            "schema_version": plan.raw["schema_version"],
            "id": plan.raw["id"],
            "status": plan.raw["status"],
            "config_name": plan.selection_path.name,
            "config_sha256": plan.selection_sha256,
        },
        "dataset": plan.raw["source"],
        "license": plan.raw["license"],
        "split_contract": plan.raw["split_contract"],
        "qualification": plan.raw["qualification"],
        "access": {
            "access_type": plan.raw["access"]["access_type"],
            "authorization_url": plan.raw["access"]["authorization_url"],
            "authorization_method": plan.raw["access"]["authorization_method"],
            "signed_url_persisted": False,
        },
        "authorized_scope": "shared-and-hy-spruce4-only",
        "ordered_tranches": list(_TRANCHE_ORDER),
        "requested_through": through,
        "requested_tranches": requested_tranches,
        "completed_tranches": completed_tranches,
        "requested_complete": requested_complete,
        "complete": completed_ids == {item.file_id for item in plan.artifacts},
        "expected_file_count": len(plan.artifacts),
        "expected_bytes": sum(item.expected_bytes for item in plan.artifacts),
        "retained_file_count": len(completed),
        "retained_bytes": sum(item["bytes"] for item in completed.values()),
        "updated_utc": _utc_now(),
        "files": rows,
    }


def fetch_hovi_selection(
    base: BaseConfig,
    selection_path: Path | None = None,
    *,
    through: str = "hy-spruce4-geometry",
    output_root: Path | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Retain the cumulative frozen Hovi tranche through ``through``."""
    if through not in _TRANCHE_ORDER:
        raise ValueError(f"unknown Hovi tranche {through!r}; expected one of {_TRANCHE_ORDER}")
    plan = load_hovi_retention_plan(selection_path)
    root = (output_root or DATA_IN / "evidence" / "hovi") / plan.plan_sha256
    files_root = root / "files"
    manifest_path = root / "retained.json"
    existing = _load_existing_manifest(manifest_path, plan)
    completed: dict[str, dict[str, Any]] = {}
    if existing is not None:
        artifacts_by_id = {item.file_id: item for item in plan.artifacts}
        for row in existing.get("files", []):
            if row.get("status") == "verified":
                file_id = row.get("file_id")
                artifact = artifacts_by_id.get(file_id)
                if artifact is None or any(
                    row.get(key) != value for key, value in artifact.identity().items()
                ):
                    raise ValueError("existing Hovi manifest contains a changed file tuple")
                destination = _destination(files_root, artifact)
                if destination.exists():
                    completed[file_id] = {
                        "bytes": artifact.expected_bytes,
                        "sha256": artifact.expected_sha256,
                        "verified_utc": row["verified_utc"],
                    }

    requested_tranches = set(_TRANCHE_ORDER[: _TRANCHE_ORDER.index(through) + 1])
    requested = [item for item in plan.artifacts if item.tranche in requested_tranches]
    log(
        f"Hovi retention {plan.plan_sha256}: {len(requested)} files, "
        f"{sum(item.expected_bytes for item in requested):,} bytes through {through}"
    )
    _write_manifest(manifest_path, _manifest(plan, through, completed))
    client = _FairdataClient(base.fetch)
    for index, artifact in enumerate(requested, start=1):
        destination = _destination(files_root, artifact)
        log(f"[{index}/{len(requested)}] {artifact.pathname}")
        try:
            downloaded = client.download(artifact, destination, log=log)
        except BaseException:
            completed.pop(artifact.file_id, None)
            _write_manifest(manifest_path, _manifest(plan, through, completed))
            raise
        completed[artifact.file_id] = {
            "bytes": artifact.expected_bytes,
            "sha256": artifact.expected_sha256,
            "verified_utc": _utc_now(),
        }
        log("  retained" if downloaded else "  verified existing bytes")
        _write_manifest(manifest_path, _manifest(plan, through, completed))
    return manifest_path


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Retain the frozen shared + HY_SPRUCE4 Hovi evidence tranches."
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=CONFIG_DIR / "hovi-public-targets.json",
    )
    parser.add_argument(
        "--through",
        choices=_TRANCHE_ORDER,
        default="hy-spruce4-geometry",
        help="Cumulative terminal tranche (default: %(default)s)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DATA_IN / "evidence" / "hovi",
    )
    args = parser.parse_args()
    path = fetch_hovi_selection(
        load_base(),
        args.selection,
        through=args.through,
        output_root=args.output_root,
    )
    print(f"retention manifest: {path}")


if __name__ == "__main__":
    _main()
