"""Retain the frozen Evo selector and explicitly authorized exact plot bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

import requests

from ..config import DATA_IN, FetchConfig, load_base
from ..evidence.evo.fallback_1065 import load_evo_1065_selection
from ..evidence.evo.selection import EvoArtifact, EvoSelection, load_evo_selection

_PLAN_SCHEMA = "evo-selector-retention-plan/1.0.0"
_MANIFEST_SCHEMA = "evo-selector-retention/1.0.0"
_PLOT_PLAN_SCHEMA = "evo-plot-retention-plan/1.0.0"
_PLOT_MANIFEST_SCHEMA = "evo-plot-retention/1.0.0"
_DATASET_RECORD_URL = (
    "https://metax.fairdata.fi/v3/datasets/"
    "b1dac2b9-93cb-407e-91f1-eeb79c8cdd92"
)
_FILE_INVENTORY_URL = (
    "https://metax.fairdata.fi/v3/files?"
    "dataset=b1dac2b9-93cb-407e-91f1-eeb79c8cdd92&limit=100"
)
_AUTHORIZATION_URL = "https://etsin.fairdata.fi/api/v3/download/authorize"
_SIGNED_URL_HOST = "download.fairdata.fi"
_DATASET_UUID = "b1dac2b9-93cb-407e-91f1-eeb79c8cdd92"
_DATASET_DOI = "10.23729/fd-5a800660-8bd8-35ef-ac9f-ac5c45f7fa77"
_SELECTOR_PATH = "/Evo_TLS_2024_stand_attributes_v2.csv"
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class EvoRetentionPlan:
    selection: EvoSelection
    artifact: EvoArtifact
    identity: dict[str, Any]
    retention_id: str


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_identity(artifact: EvoArtifact) -> dict[str, Any]:
    return {
        "file_id": artifact.file_id,
        "path": artifact.source_path,
        "bytes": artifact.bytes,
        "sha256": artifact.sha256,
        "kind": artifact.kind,
    }


def build_evo_retention_plan(selection_path: Path | None = None) -> EvoRetentionPlan:
    selection = load_evo_selection(selection_path)
    allowed = selection.authorized_retention_artifacts()
    if len(allowed) != 1 or allowed[0].source_path != _SELECTOR_PATH:
        raise ValueError("Evo retention plan must contain only the selector CSV")
    artifact = allowed[0]
    identity = {
        "schema_version": _PLAN_SCHEMA,
        "selection_config_sha256": selection.config_sha256,
        "dataset": {
            "dataset_uuid": selection.dataset_uuid,
            "dataset_doi": selection.dataset_doi,
            "dataset_version": 2,
            "published_revision": 2,
        },
        "authorized_scope": "stand_attribute_selector_index_only",
        "artifact": _artifact_identity(artifact),
        "point_cloud_fetch_authorized": False,
        "signed_url_policy": "generate_in_memory_never_persist",
    }
    retention_id = hashlib.sha256(_canonical_json(identity)).hexdigest()
    return EvoRetentionPlan(selection, artifact, identity, retention_id)


def build_evo_plot_retention_plan(
    selection_path: Path | None = None,
    *,
    authorize_exact_plot: bool = False,
    plot_id: str = "1086",
) -> EvoRetentionPlan:
    """Build the exact one-file LAZ plan only after an explicit caller authorization."""
    if not authorize_exact_plot:
        raise ValueError("Evo plot retention requires --authorize-exact-plot")
    if plot_id == "1086":
        selection = load_evo_selection(selection_path)
    elif plot_id == "1065":
        selection = load_evo_1065_selection(selection_path)
    else:
        raise ValueError(f"unsupported Evo plot {plot_id!r}")
    plot_path = f"/Evo_TLS_2024_treeanal_pointclouds/{plot_id}_pointcloud_georef.laz"
    matches = tuple(
        artifact for artifact in selection.artifacts if artifact.source_path == plot_path
    )
    if len(matches) != 1:
        raise ValueError(f"frozen Evo plot-{plot_id} artifact is absent or ambiguous")
    artifact = matches[0]
    if artifact.retention_authorized:
        raise ValueError("frozen selector config unexpectedly authorizes point-cloud retention")
    identity = {
        "schema_version": _PLOT_PLAN_SCHEMA,
        "selection_config_sha256": selection.config_sha256,
        "dataset": {
            "dataset_uuid": selection.dataset_uuid,
            "dataset_doi": selection.dataset_doi,
            "dataset_version": 2,
            "published_revision": 2,
        },
        "authorized_scope": f"exact_plot_{plot_id}_point_cloud_only",
        "authorization_basis": "explicit_operator_request_2026-07-14",
        "artifact": _artifact_identity(artifact),
        "selector_config_retention_authorized": False,
        "signed_url_policy": "generate_in_memory_never_persist",
    }
    retention_id = hashlib.sha256(_canonical_json(identity)).hexdigest()
    return EvoRetentionPlan(selection, artifact, identity, retention_id)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", encoding="utf-8") as target:
        json.dump(value, target, indent=2, sort_keys=True)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def _safe_destination(files_root: Path, source_path: str) -> Path:
    relative = Path(*PurePosixPath(source_path).parts[1:])
    destination = files_root / relative
    if not destination.resolve(strict=False).is_relative_to(files_root.resolve()):
        raise ValueError("Evo selector path escaped the retention root")
    return destination


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

    def _stable_json(self, url: str) -> Mapping[str, Any]:
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                self._pace(url)
                response = self.session.get(url, timeout=(30, 60))
            except requests.RequestException:
                response = None
            if response is not None and response.status_code not in _RETRYABLE_STATUS:
                if response.status_code != 200:
                    raise RuntimeError(f"Fairdata metadata HTTP {response.status_code}")
                try:
                    value = response.json()
                except ValueError:
                    raise RuntimeError("Fairdata metadata was not JSON") from None
                if not isinstance(value, Mapping):
                    raise RuntimeError("Fairdata metadata root was not an object")
                return value
            if attempt < self.cfg.max_retries:
                time.sleep(min(8.0, 2.0 ** (attempt - 1)))
        raise RuntimeError("Fairdata metadata request exhausted retries")

    def verify_frozen_remote(self, plan: EvoRetentionPlan) -> None:
        dataset = self._stable_json(_DATASET_RECORD_URL)
        fileset = dataset.get("fileset")
        access = dataset.get("access_rights")
        if not isinstance(fileset, Mapping) or not isinstance(access, Mapping):
            raise ValueError("Evo version-2 dataset metadata is incomplete")
        access_type = access.get("access_type")
        licenses = access.get("license")
        if not isinstance(access_type, Mapping) or not isinstance(licenses, list):
            raise ValueError("Evo version-2 access metadata is incomplete")
        if (
            dataset.get("id") != _DATASET_UUID
            or dataset.get("persistent_identifier") != f"doi:{_DATASET_DOI}"
            or dataset.get("version") != 2
            or dataset.get("published_revision") != 2
            or fileset.get("total_files_count") != 57
            or fileset.get("total_files_size") != 42_192_878_905
            or access_type.get("url")
            != "http://uri.suomi.fi/codelist/fairdata/access_type/code/open"
            or len(licenses) != 1
            or not isinstance(licenses[0], Mapping)
            or licenses[0].get("url")
            != "http://uri.suomi.fi/codelist/fairdata/license/code/CC-BY-4.0"
        ):
            raise ValueError("Evo controlling dataset version or access identity drifted")

        inventory = self._stable_json(_FILE_INVENTORY_URL)
        results = inventory.get("results")
        if (
            inventory.get("count") != 57
            or inventory.get("next") is not None
            or not isinstance(results, list)
            or len(results) != 57
        ):
            raise ValueError("Evo version-2 file inventory is incomplete or paginated")
        expected = {
            item.source_path: {
                "id": item.file_id,
                "pathname": item.source_path,
                "filename": PurePosixPath(item.source_path).name,
                "size": item.bytes,
                "checksum": f"sha256:{item.sha256}",
            }
            for item in plan.selection.artifacts
        }
        found: dict[str, Mapping[str, Any]] = {}
        for raw in results:
            if not isinstance(raw, Mapping):
                raise ValueError("Evo version-2 inventory contains a non-object record")
            pathname = raw.get("pathname")
            if pathname in expected:
                if pathname in found:
                    raise ValueError("Evo version-2 inventory duplicated a frozen path")
                found[pathname] = raw
        if set(found) != set(expected):
            raise ValueError("Evo version-2 inventory lost a frozen artifact path")
        for pathname, fields in expected.items():
            if any(found[pathname].get(key) != value for key, value in fields.items()):
                raise ValueError(f"Evo frozen file tuple drifted: {pathname}")

    def _authorize(self, plan: EvoRetentionPlan) -> str:
        self._pace(_AUTHORIZATION_URL)
        try:
            response = self.session.post(
                _AUTHORIZATION_URL,
                json={
                    "cr_id": plan.selection.dataset_uuid,
                    "file": plan.artifact.source_path,
                },
                timeout=(30, 60),
            )
        except requests.RequestException:
            raise RuntimeError("Fairdata selector authorization failed") from None
        if response.status_code != 200:
            raise RuntimeError(f"Fairdata selector authorization HTTP {response.status_code}")
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
            raise RuntimeError("Fairdata returned an invalid selector download endpoint")
        return signed_url

    def fetch_verified_selector(self, plan: EvoRetentionPlan) -> bytes:
        for attempt in range(1, self.cfg.max_retries + 1):
            signed_url = self._authorize(plan)
            try:
                self._pace(signed_url)
                response = self.session.get(
                    signed_url,
                    headers={"Accept-Encoding": "identity"},
                    timeout=(30, 60),
                )
            except requests.RequestException:
                response = None
            if response is not None and response.status_code not in _RETRYABLE_STATUS:
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Fairdata selector download HTTP {response.status_code}"
                    )
                if urlsplit(response.url).hostname != _SIGNED_URL_HOST:
                    raise RuntimeError("Fairdata selector download left the allowed host")
                if "text/html" in response.headers.get("Content-Type", "").lower():
                    raise RuntimeError("Fairdata selector endpoint returned HTML")
                content = response.content
                digest = hashlib.sha256(content).hexdigest()
                if len(content) != plan.artifact.bytes or digest != plan.artifact.sha256:
                    raise ValueError("Fairdata selector bytes differ from the frozen tuple")
                return content
            if attempt < self.cfg.max_retries:
                time.sleep(min(8.0, 2.0 ** (attempt - 1)))
        raise RuntimeError("Fairdata selector download exhausted retries")

    def fetch_verified_artifact_to(
        self,
        plan: EvoRetentionPlan,
        destination: Path,
        *,
        log: Callable[[str], None] = print,
    ) -> None:
        """Stream one frozen artifact, then atomically publish only verified bytes."""
        temporary = destination.with_name(destination.name + ".part")
        if temporary.exists():
            raise ValueError("stale Evo artifact staging file requires manual inspection")
        destination.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, self.cfg.max_retries + 1):
            signed_url = self._authorize(plan)
            digest = hashlib.sha256()
            byte_count = 0
            try:
                self._pace(signed_url)
                with self.session.get(
                    signed_url,
                    headers={"Accept-Encoding": "identity"},
                    timeout=(30, 300),
                    stream=True,
                ) as response:
                    if response.status_code in _RETRYABLE_STATUS:
                        response = None
                    elif response.status_code != 200:
                        raise RuntimeError(
                            f"Fairdata artifact download HTTP {response.status_code}"
                        )
                    elif urlsplit(response.url).hostname != _SIGNED_URL_HOST:
                        raise RuntimeError("Fairdata artifact download left the allowed host")
                    elif "text/html" in response.headers.get("Content-Type", "").lower():
                        raise RuntimeError("Fairdata artifact endpoint returned HTML")
                    else:
                        with temporary.open("xb") as target:
                            for block in response.iter_content(chunk_size=8 << 20):
                                if not block:
                                    continue
                                target.write(block)
                                digest.update(block)
                                byte_count += len(block)
                            target.flush()
                            os.fsync(target.fileno())
                        if (
                            byte_count != plan.artifact.bytes
                            or digest.hexdigest() != plan.artifact.sha256
                        ):
                            raise ValueError(
                                "Fairdata artifact bytes differ from the frozen tuple"
                            )
                        temporary.replace(destination)
                        log(f"verified Evo artifact retained: {destination}")
                        return
            except requests.RequestException:
                pass
            finally:
                if temporary.exists():
                    temporary.unlink()
            if attempt < self.cfg.max_retries:
                time.sleep(min(8.0, 2.0 ** (attempt - 1)))
        raise RuntimeError("Fairdata artifact download exhausted retries")


def _manifest(plan: EvoRetentionPlan, destination: Path, root: Path) -> dict[str, Any]:
    return {
        "schema_version": _MANIFEST_SCHEMA,
        "status": "complete",
        "retention_id": plan.retention_id,
        "plan_sha256": plan.retention_id,
        "plan_identity": plan.identity,
        "selection": {
            "config_name": plan.selection.path.name,
            "config_sha256": plan.selection.config_sha256,
        },
        "remote_preflight": {
            "dataset_version_verified": 2,
            "published_revision_verified": 2,
            "file_inventory_count_verified": 57,
            "all_frozen_artifact_tuples_verified": True,
        },
        "authorized_scope": "stand_attribute_selector_index_only",
        "point_cloud_fetch_authorized": False,
        "signed_url_persisted": False,
        "artifact": {
            **_artifact_identity(plan.artifact),
            "relative_path": destination.relative_to(root).as_posix(),
            "retained_bytes": destination.stat().st_size,
            "verified": True,
        },
    }


def retain_evo_selector(
    selection_path: Path | None = None,
    output_root: Path | None = None,
    *,
    log: Callable[[str], None] = print,
) -> Path:
    plan = build_evo_retention_plan(selection_path)
    root = (output_root or DATA_IN / "evidence" / "evo") / plan.retention_id
    destination = _safe_destination(root / "files", plan.artifact.source_path)
    manifest_path = root / "retained.json"
    client = _FairdataClient(load_base().fetch)

    client.verify_frozen_remote(plan)
    if destination.exists():
        if (
            not destination.is_file()
            or destination.stat().st_size != plan.artifact.bytes
            or _sha256_file(destination) != plan.artifact.sha256
        ):
            raise ValueError("retained Evo selector failed frozen byte verification")
    else:
        content = client.fetch_verified_selector(plan)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".part")
        if temporary.exists():
            raise ValueError("stale Evo selector staging file requires manual inspection")
        with temporary.open("xb") as target:
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
        temporary.replace(destination)
        log(f"verified Evo selector retained: {destination}")

    manifest = _manifest(plan, destination, root)
    _atomic_json(manifest_path, manifest)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    log(f"Evo selector retention id: {plan.retention_id}")
    log(f"Evo selector manifest SHA-256: {manifest_sha256}")
    return manifest_path


def retain_evo_plot(
    selection_path: Path | None = None,
    output_root: Path | None = None,
    *,
    authorize_exact_plot: bool = False,
    plot_id: str = "1086",
    log: Callable[[str], None] = print,
) -> Path:
    plan = build_evo_plot_retention_plan(
        selection_path,
        authorize_exact_plot=authorize_exact_plot,
        plot_id=plot_id,
    )
    root = (output_root or DATA_IN / "evidence" / "evo") / plan.retention_id
    destination = _safe_destination(root / "files", plan.artifact.source_path)
    manifest_path = root / "retained.json"
    client = _FairdataClient(load_base().fetch)

    client.verify_frozen_remote(plan)
    if destination.exists():
        if (
            not destination.is_file()
            or destination.stat().st_size != plan.artifact.bytes
            or _sha256_file(destination) != plan.artifact.sha256
        ):
            raise ValueError("retained Evo plot failed frozen byte verification")
    else:
        client.fetch_verified_artifact_to(plan, destination, log=log)

    manifest = {
        "schema_version": _PLOT_MANIFEST_SCHEMA,
        "status": "complete",
        "retention_id": plan.retention_id,
        "plan_sha256": plan.retention_id,
        "plan_identity": plan.identity,
        "selection": {
            "config_name": plan.selection.path.name,
            "config_sha256": plan.selection.config_sha256,
        },
        "remote_preflight": {
            "dataset_version_verified": 2,
            "published_revision_verified": 2,
            "file_inventory_count_verified": 57,
            "all_frozen_artifact_tuples_verified": True,
        },
        "authorized_scope": f"exact_plot_{plot_id}_point_cloud_only",
        "operator_authorization_required": True,
        "selector_config_retention_authorized": False,
        "signed_url_persisted": False,
        "artifact": {
            **_artifact_identity(plan.artifact),
            "relative_path": destination.relative_to(root).as_posix(),
            "retained_bytes": destination.stat().st_size,
            "verified": True,
        },
        "qualification": {
            "role": "raw_candidate",
            "status": "unqualified",
            "target_truth": False,
            "synthesis_authorized": False,
        },
    }
    _atomic_json(manifest_path, manifest)
    log(f"Evo plot retention id: {plan.retention_id}")
    log(f"Evo plot manifest SHA-256: {_sha256_file(manifest_path)}")
    return manifest_path


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Retain the frozen Evo selector or one explicitly authorized exact plot."
    )
    parser.add_argument(
        "--selection",
        type=Path,
        help="override the exact selection authority for the chosen operation",
    )
    parser.add_argument(
        "--plot-id",
        choices=("1086", "1065"),
        default="1086",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DATA_IN / "evidence" / "evo",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="retain the exact frozen --plot-id LAZ instead of the selector CSV",
    )
    parser.add_argument(
        "--authorize-exact-plot",
        action="store_true",
        help="explicitly authorize only the frozen --plot-id LAZ tuple",
    )
    args = parser.parse_args()
    if args.authorize_exact_plot and not args.plot:
        parser.error("--authorize-exact-plot requires --plot")
    if args.plot:
        manifest = retain_evo_plot(
            args.selection,
            args.output_root,
            authorize_exact_plot=args.authorize_exact_plot,
            plot_id=args.plot_id,
        )
        print(f"Evo plot retention manifest: {manifest}")
    else:
        manifest = retain_evo_selector(args.selection, args.output_root)
        print(f"Evo selector retention manifest: {manifest}")


if __name__ == "__main__":
    _main()
