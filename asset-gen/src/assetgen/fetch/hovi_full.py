"""Q1-authorized retention of the exact HY_SPRUCE4 individual-scan archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

import requests

from ..config import CONFIG_DIR, DATA_IN, FetchConfig, load_base

_CONFIG_SCHEMA = "hovi-hy-spruce4-full-q1-authorization/1.0.0"
_CONFIG_ID = "hovi-hy-spruce4-full-after-q1-go-v1"
_PLAN_SCHEMA = "hovi-hy-spruce4-full-retention-plan/1.0.0"
_MANIFEST_SCHEMA = "hovi-hy-spruce4-full-retention/1.0.0"
_PROGRESS_SCHEMA = "hovi-hy-spruce4-full-progress/1.0.0"
_DATASET_UUID = "ace2a123-00ff-4944-951e-eddbe209b70c"
_AUTHORIZATION_URL = "https://etsin.fairdata.fi/api/v3/download/authorize"
_SIGNED_URL_HOST = "download.fairdata.fi"
_FILE_ID = "83b09928-c296-4719-982d-0bbd9451405a"
_FILE_PATH = "/Laboratory_and_field_data/Terrestrial_laser_scanning/Point_clouds_full/HY_SPRUCE4-full.zip"
_FILE_BYTES = 47_345_435_519
_FILE_SHA256 = "008540c96bcbea1eb98128d92df844ee6346c11e1ec99ad626ba3c2e6a0caa1e"
_Q1_BUILD_ID = "1e5f674af3fc4d09fdf619ef154f425df22d148a0c619cdc2f93253a32e6440e"
_SOURCE_RETENTION_ID = "45f35e2b747b404f0f7b0e594852c43f6493c6368bab738e8f914dffb4297f7e"
_CONTENT_RANGE_RE = re.compile(r"^bytes ([0-9]+)-([0-9]+)/([0-9]+)$")
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_PROGRESS_INTERVAL = 256 << 20


class _RetryableDownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class FullScanPlan:
    config_path: Path
    config_sha256: str
    identity: dict[str, Any]
    retention_id: str
    artifact: dict[str, Any]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Hovi full-scan {label} must be an object")
    return value


def _verify_bound_file(asset_gen_root: Path, record: Mapping[str, Any], label: str) -> Path:
    relative = record.get("path")
    expected_bytes = record.get("bytes")
    expected_sha256 = record.get("sha256")
    if (
        not isinstance(relative, str)
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or not isinstance(expected_bytes, int)
        or not isinstance(expected_sha256, str)
    ):
        raise ValueError(f"Hovi full-scan {label} binding is invalid")
    path = (asset_gen_root / relative).resolve()
    if not path.is_relative_to(asset_gen_root.resolve()) or not path.is_file():
        raise ValueError(f"Hovi full-scan {label} is missing")
    if path.stat().st_size != expected_bytes or _sha256_file(path) != expected_sha256:
        raise ValueError(f"Hovi full-scan {label} bytes changed")
    return path


def load_full_scan_plan(config_path: Path | None = None) -> FullScanPlan:
    path = config_path or CONFIG_DIR / "evidence" / "hovi-hy-spruce4-full-q1.json"
    encoded = path.read_bytes()
    raw = json.loads(encoded)
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "id",
        "status",
        "authorization",
        "source_retention",
        "dataset_identity",
        "license_identity",
        "access_identity",
        "artifact",
        "exclusions",
    }:
        raise ValueError("unsupported Hovi full-scan authorization config")
    if (
        raw["schema_version"] != _CONFIG_SCHEMA
        or raw["id"] != _CONFIG_ID
        or raw["status"] != "retention_only_authorized"
    ):
        raise ValueError("Hovi full-scan authorization identity changed")

    asset_gen_root = CONFIG_DIR.parent
    authorization = _mapping(raw["authorization"], "authorization")
    q1_record = _mapping(authorization.get("q1_manifest"), "Q1 manifest")
    q1_path = _verify_bound_file(asset_gen_root, q1_record, "Q1 manifest")
    q1 = json.loads(q1_path.read_bytes())
    if (
        authorization.get("decision") != "Q1_GO"
        or authorization.get("scope") != "retain_exact_archive_only"
        or any(
            authorization.get(key) is not False
            for key in (
                "extract_authorized",
                "target_truth",
                "synthesis_authorized",
                "usable_surface",
            )
        )
        or q1_record.get("build_id") != _Q1_BUILD_ID
        or q1_record.get("disposition") != "preview_geometry_plausible_for_q1"
        or q1.get("build_id") != _Q1_BUILD_ID
        or q1.get("status") != "complete"
        or q1.get("plot_id") != "HY_SPRUCE4"
        or q1.get("disposition") != "preview_geometry_plausible_for_q1"
        or q1.get("scientific_role") != "raw_candidate"
        or q1.get("qualification_status") != "unqualified"
        or q1.get("transfer_ceiling") != "none"
        or q1.get("synthesis_authorized") is not False
        or q1.get("usable_surface") is not False
    ):
        raise ValueError("Hovi full-scan Q1 authorization boundary changed")

    source_record = _mapping(raw["source_retention"], "source retention")
    retained_path = _verify_bound_file(asset_gen_root, source_record, "source retention")
    retained = json.loads(retained_path.read_bytes())
    if (
        source_record.get("retention_id") != _SOURCE_RETENTION_ID
        or retained.get("schema_version") != "hovi-retained-evidence/1.0.0"
        or retained.get("retention_id") != _SOURCE_RETENTION_ID
        or retained.get("plan_sha256") != _SOURCE_RETENTION_ID
        or retained.get("selection", {}).get("config_sha256")
        != source_record.get("selection_config_sha256")
    ):
        raise ValueError("Hovi full-scan source retention identity changed")

    dataset = dict(_mapping(raw["dataset_identity"], "dataset identity"))
    license_identity = dict(_mapping(raw["license_identity"], "license identity"))
    access = dict(_mapping(raw["access_identity"], "access identity"))
    if (
        dataset
        != {
            "dataset_uuid": _DATASET_UUID,
            "dataset_doi": "10.23729/9a8d90cd-73e2-438d-9230-94e10e61adc9",
            "dataset_version": 1,
            "published_revision": 1,
        }
        or any(retained.get("dataset", {}).get(key) != value for key, value in dataset.items())
        or license_identity
        != {
            "spdx": "CC-BY-4.0",
            "url": "https://creativecommons.org/licenses/by/4.0/legalcode",
            "attribution_required": True,
            "changes_must_be_indicated": True,
        }
        or any(
            retained.get("license", {}).get(key) != value
            for key, value in license_identity.items()
        )
        or access
        != {
            "access_type": "open",
            "authorization_url": _AUTHORIZATION_URL,
            "authorization_method": "POST",
            "signed_url_host": _SIGNED_URL_HOST,
            "signed_url_policy": "generate_at_fetch_time_never_persist",
            "download_policy": "resume_part_verify_bytes_sha256_atomic_rename",
        }
        or retained.get("access", {}).get("access_type") != "open"
        or retained.get("access", {}).get("authorization_url") != _AUTHORIZATION_URL
        or retained.get("access", {}).get("authorization_method") != "POST"
        or retained.get("access", {}).get("signed_url_persisted") is not False
    ):
        raise ValueError("Hovi full-scan dataset, license, or access identity changed")

    artifact = dict(_mapping(raw["artifact"], "artifact"))
    if artifact != {
        "plot_id": "HY_SPRUCE4",
        "file_id": _FILE_ID,
        "path": _FILE_PATH,
        "bytes": _FILE_BYTES,
        "sha256": _FILE_SHA256,
        "kind": "individual_full_scan_archive",
    }:
        raise ValueError("Hovi full-scan file tuple changed")
    if dict(_mapping(raw["exclusions"], "exclusions")) != {
        "fetch_hy_pine2": False,
        "fetch_js_spruce1": False,
        "extract_archive": False,
    }:
        raise ValueError("Hovi full-scan exclusion boundary changed")

    config_sha256 = hashlib.sha256(encoded).hexdigest()
    identity = {
        "schema_version": _PLAN_SCHEMA,
        "authorization_config_sha256": config_sha256,
        "q1_manifest": dict(q1_record),
        "source_retention": dict(source_record),
        "dataset_identity": dataset,
        "license_identity": license_identity,
        "access_identity": access,
        "artifact": artifact,
        "authorized_action": "retain_archive_without_extraction",
    }
    retention_id = hashlib.sha256(_canonical_json(identity)).hexdigest()
    return FullScanPlan(path, config_sha256, identity, retention_id, artifact)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", encoding="utf-8") as target:
        json.dump(value, target, indent=2, sort_keys=True)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def _safe_destination(files_root: Path) -> Path:
    relative = Path(*PurePosixPath(_FILE_PATH).parts[1:])
    destination = files_root / relative
    if not destination.resolve(strict=False).is_relative_to(files_root.resolve()):
        raise ValueError("Hovi full-scan stable path escaped the retention root")
    return destination


def _manifest(plan: FullScanPlan, destination: Path, part: Path) -> dict[str, Any]:
    final_exists = destination.is_file()
    retained_bytes = destination.stat().st_size if final_exists else (
        part.stat().st_size if part.is_file() else 0
    )
    return {
        "schema_version": _MANIFEST_SCHEMA,
        "status": "complete" if final_exists else "in_progress",
        "retention_id": plan.retention_id,
        "plan_identity": plan.identity,
        "config_name": plan.config_path.name,
        "config_sha256": plan.config_sha256,
        "authorized_scope": "HY_SPRUCE4-full.zip only",
        "extract_authorized": False,
        "target_truth": False,
        "synthesis_authorized": False,
        "usable_surface": False,
        "signed_url_persisted": False,
        "artifact": {
            **plan.artifact,
            "relative_path": destination.relative_to(destination.parents[4]).as_posix(),
            "retained_bytes": retained_bytes,
            "verified": final_exists,
        },
        "updated_utc": _utc_now(),
    }


class _FairdataFullScanClient:
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

    def _authorize(self) -> str:
        self._pace(_AUTHORIZATION_URL)
        response = self.session.post(
            _AUTHORIZATION_URL,
            json={"cr_id": _DATASET_UUID, "file": _FILE_PATH},
            timeout=60,
        )
        if response.status_code in _RETRYABLE_STATUS:
            raise _RetryableDownloadError(f"authorization HTTP {response.status_code}")
        if response.status_code < 200 or response.status_code >= 300:
            raise RuntimeError(f"Fairdata authorization HTTP {response.status_code}")
        try:
            value: Any = response.json()
        except ValueError:
            value = response.text.strip()
        if isinstance(value, str):
            signed_url = value.strip().strip('"')
        elif isinstance(value, Mapping):
            signed_url = next(
                (
                    value[key]
                    for key in ("url", "download_url", "downloadUrl")
                    if isinstance(value.get(key), str)
                ),
                "",
            )
        else:
            signed_url = ""
        parsed = urlsplit(signed_url)
        if parsed.scheme != "https" or parsed.hostname != _SIGNED_URL_HOST:
            raise RuntimeError("Fairdata returned an invalid full-scan download endpoint")
        return signed_url

    def download(
        self,
        part: Path,
        progress: Callable[[int], None],
        log: Callable[[str], None],
    ) -> None:
        part.parent.mkdir(parents=True, exist_ok=True)
        if part.exists() and part.stat().st_size > _FILE_BYTES:
            raise ValueError("Hovi full-scan partial file exceeds the frozen byte count")
        failures = 0
        while (part.stat().st_size if part.exists() else 0) < _FILE_BYTES:
            try:
                self._attempt(part, progress, log)
                failures = 0
            except (requests.RequestException, _RetryableDownloadError) as error:
                failures += 1
                retained = part.stat().st_size if part.exists() else 0
                progress(retained)
                if failures >= self.cfg.max_retries:
                    raise RuntimeError(
                        f"Fairdata full-scan transfer stopped after {failures} consecutive "
                        f"failures; {retained:,} resumable bytes retained ({type(error).__name__})"
                    ) from None
                delay = min(60.0, 2.0 ** (failures - 1))
                log(
                    f"  transient {type(error).__name__}; retaining {retained:,} bytes "
                    f"and retrying in {delay:.0f}s"
                )
                time.sleep(delay)

    def _attempt(
        self,
        part: Path,
        progress: Callable[[int], None],
        log: Callable[[str], None],
    ) -> None:
        offset = part.stat().st_size if part.exists() else 0
        if offset == _FILE_BYTES:
            return
        signed_url = self._authorize()
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
            if response.status_code in _RETRYABLE_STATUS:
                raise _RetryableDownloadError(f"download HTTP {response.status_code}")
            if offset:
                if response.status_code != 206:
                    raise RuntimeError(
                        "Fairdata refused byte-range resume; existing partial bytes were preserved"
                    )
                match = _CONTENT_RANGE_RE.fullmatch(response.headers.get("Content-Range", ""))
                if (
                    match is None
                    or int(match.group(1)) != offset
                    or int(match.group(2)) >= _FILE_BYTES
                    or int(match.group(3)) != _FILE_BYTES
                ):
                    raise RuntimeError("Fairdata returned an invalid resume range")
            elif response.status_code != 200:
                raise _RetryableDownloadError(
                    f"unexpected download HTTP {response.status_code}"
                )
            if "text/html" in response.headers.get("Content-Type", "").lower():
                raise _RetryableDownloadError("download endpoint returned HTML")

            written = offset
            next_progress = ((written // _PROGRESS_INTERVAL) + 1) * _PROGRESS_INTERVAL
            with part.open("ab" if offset else "wb") as target:
                for block in response.iter_content(8 << 20):
                    if not block:
                        continue
                    written += len(block)
                    if written > _FILE_BYTES:
                        raise _RetryableDownloadError("download exceeded the frozen byte count")
                    target.write(block)
                    if written >= next_progress:
                        target.flush()
                        os.fsync(target.fileno())
                        progress(written)
                        log(
                            f"  {written:,}/{_FILE_BYTES:,} bytes "
                            f"({100.0 * written / _FILE_BYTES:.2f}%)"
                        )
                        next_progress += _PROGRESS_INTERVAL
                target.flush()
                os.fsync(target.fileno())
            progress(written)
            if written != _FILE_BYTES:
                raise _RetryableDownloadError(
                    f"response ended at {written:,} of {_FILE_BYTES:,} bytes"
                )


def retain_hy_spruce4_full(
    config_path: Path | None = None,
    output_root: Path | None = None,
    *,
    log: Callable[[str], None] = print,
) -> Path:
    plan = load_full_scan_plan(config_path)
    root = (output_root or DATA_IN / "evidence" / "hovi-full-q1") / plan.retention_id
    destination = _safe_destination(root / "files")
    part = destination.with_name(destination.name + ".part")
    manifest_path = root / "retained.json"
    progress_path = root / "progress.json"

    if destination.exists():
        if destination.stat().st_size != _FILE_BYTES or _sha256_file(destination) != _FILE_SHA256:
            raise ValueError("retained HY_SPRUCE4 full archive failed verification")
        _atomic_json(manifest_path, _manifest(plan, destination, part))
        return manifest_path

    def write_progress(retained_bytes: int) -> None:
        _atomic_json(
            progress_path,
            {
                "schema_version": _PROGRESS_SCHEMA,
                "retention_id": plan.retention_id,
                "artifact": plan.artifact,
                "retained_bytes": retained_bytes,
                "expected_bytes": _FILE_BYTES,
                "percent": 100.0 * retained_bytes / _FILE_BYTES,
                "part_relative_path": part.relative_to(root).as_posix(),
                "signed_url_persisted": False,
                "updated_utc": _utc_now(),
            },
        )
        _atomic_json(manifest_path, _manifest(plan, destination, part))

    initial = part.stat().st_size if part.exists() else 0
    write_progress(initial)
    log(
        f"HY_SPRUCE4 full Q1 retention {plan.retention_id}: "
        f"resuming at {initial:,}/{_FILE_BYTES:,} bytes"
    )
    client = _FairdataFullScanClient(load_base().fetch)
    client.download(part, write_progress, log)
    if part.stat().st_size != _FILE_BYTES:
        raise RuntimeError("HY_SPRUCE4 full archive ended at the wrong byte count")
    log("  byte count complete; verifying SHA-256")
    digest = _sha256_file(part)
    if digest != _FILE_SHA256:
        write_progress(part.stat().st_size)
        raise ValueError("HY_SPRUCE4 full archive SHA-256 differs from the frozen tuple")
    part.replace(destination)
    progress_path.unlink(missing_ok=True)
    _atomic_json(manifest_path, _manifest(plan, destination, part))
    log(f"  verified and retained: {destination}")
    return manifest_path


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Retain only the Q1-authorized HY_SPRUCE4 full-scan archive."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DIR / "evidence" / "hovi-hy-spruce4-full-q1.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DATA_IN / "evidence" / "hovi-full-q1",
    )
    args = parser.parse_args()
    manifest = retain_hy_spruce4_full(args.config, args.output_root)
    print(f"full-scan retention manifest: {manifest}")


if __name__ == "__main__":
    _main()
