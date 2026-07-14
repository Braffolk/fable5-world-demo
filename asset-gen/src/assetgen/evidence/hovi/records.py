"""Strict typed records for the frozen Hovi selection and retained artifacts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

_SHA256_LENGTH = 64
_GEOMETRY_KIND = "merged_thinned_geometry_preview"
_SELECTION_SCHEMA = "hovi-public-target-selection/1.0.0"
_SELECTION_ID = "hovi-2024-jarvselja-hyytiala-first-conversion-v1"
_DATASET_UUID = "ace2a123-00ff-4944-951e-eddbe209b70c"
_UNSEALED_RETENTION_SCOPES = {
    "shared-and-hy-spruce4-only": ("HY_SPRUCE4", "hy-spruce4"),
    "shared-and-hy-pine2-only": ("HY_PINE2", "hy-pine2"),
}


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Hovi {label} must be a non-empty string")
    return value


def _sha256(value: Any, label: str) -> str:
    digest = _required_string(value, label)
    if (
        len(digest) != _SHA256_LENGTH
        or digest.lower() != digest
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"Hovi {label} must be a lowercase SHA-256")
    return digest


def _source_path(value: Any, label: str) -> str:
    path = _required_string(value, label)
    pure = PurePosixPath(path)
    if not path.startswith("/") or ".." in pure.parts or pure.name in ("", "."):
        raise ValueError(f"Hovi {label} must be an absolute dataset pathname")
    return path


@dataclass(frozen=True)
class SelectedArtifact:
    file_id: str
    source_path: str
    bytes: int
    sha256: str
    kind: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SelectedArtifact":
        byte_count = raw.get("bytes")
        if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count <= 0:
            raise ValueError("Hovi selected artifact bytes must be a positive integer")
        return cls(
            file_id=_required_string(raw.get("file_id"), "selected artifact file_id"),
            source_path=_source_path(raw.get("path"), "selected artifact path"),
            bytes=byte_count,
            sha256=_sha256(raw.get("sha256"), "selected artifact sha256"),
            kind=_required_string(raw.get("kind"), "selected artifact kind"),
        )

    def stable_tuple(self) -> tuple[str, str, int, str]:
        return self.file_id, self.source_path, self.bytes, self.sha256


@dataclass(frozen=True)
class SelectedPlot:
    plot_id: str
    role: str
    sealed_until_converter_freeze: bool
    site_id: str
    campaign_id: str
    nominal_layout_m: tuple[float, float]
    artifacts: tuple[SelectedArtifact, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SelectedPlot":
        scan_protocol = raw.get("scan_protocol")
        if not isinstance(scan_protocol, Mapping):
            raise ValueError("Hovi plot scan_protocol must be an object")
        nominal = scan_protocol.get("nominal_layout_m")
        if (
            not isinstance(nominal, list)
            or len(nominal) != 2
            or any(not isinstance(value, (int, float)) or value <= 0 for value in nominal)
        ):
            raise ValueError("Hovi plot nominal_layout_m must contain two positive values")
        files = raw.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("Hovi plot files must be a non-empty list")
        sealed = raw.get("sealed_until_converter_freeze")
        if not isinstance(sealed, bool):
            raise ValueError("Hovi plot sealed_until_converter_freeze must be explicit")
        artifacts = tuple(SelectedArtifact.from_mapping(item) for item in files)
        identities = tuple(artifact.stable_tuple() for artifact in artifacts)
        if len(set(identities)) != len(identities):
            raise ValueError("Hovi plot contains duplicate selected artifact identities")
        return cls(
            plot_id=_required_string(raw.get("plot_id"), "plot_id"),
            role=_required_string(raw.get("role"), "plot role"),
            sealed_until_converter_freeze=sealed,
            site_id=_required_string(raw.get("site_id"), "site_id"),
            campaign_id=_required_string(raw.get("campaign_id"), "campaign_id"),
            nominal_layout_m=(float(nominal[0]), float(nominal[1])),
            artifacts=artifacts,
        )

    def geometry_artifact(self) -> SelectedArtifact:
        candidates = tuple(item for item in self.artifacts if item.kind == _GEOMETRY_KIND)
        if len(candidates) != 1:
            raise ValueError(
                f"Hovi plot {self.plot_id} must select exactly one {_GEOMETRY_KIND} artifact"
            )
        return candidates[0]


@dataclass(frozen=True)
class HoviSelection:
    path: Path
    sha256: str
    schema_version: str
    selection_id: str
    dataset_uuid: str
    development_sites: tuple[str, ...]
    blind_sites: tuple[str, ...]
    qualification_role: str
    qualification_status: str
    transfer_ceiling: str
    target_truth: bool
    synthesis_authorized: bool
    plots: tuple[SelectedPlot, ...]

    @classmethod
    def load(cls, path: Path) -> "HoviSelection":
        path = path.resolve()
        encoded = path.read_bytes()
        raw = json.loads(encoded)
        if not isinstance(raw, Mapping):
            raise ValueError("Hovi selection must be a JSON object")
        source = raw.get("source")
        split = raw.get("split_contract")
        qualification = raw.get("qualification")
        plots = raw.get("plots")
        if not all(isinstance(value, Mapping) for value in (source, split, qualification)):
            raise ValueError("Hovi selection source/split/qualification contracts are required")
        if not isinstance(plots, list) or not plots:
            raise ValueError("Hovi selection plots must be a non-empty list")
        development_sites = split.get("development_sites")
        blind_sites = split.get("blind_sites")
        if not isinstance(development_sites, list) or not isinstance(blind_sites, list):
            raise ValueError("Hovi selection development_sites and blind_sites are required")
        record = cls(
            path=path,
            sha256=sha256_bytes(encoded),
            schema_version=_required_string(raw.get("schema_version"), "schema_version"),
            selection_id=_required_string(raw.get("id"), "selection id"),
            dataset_uuid=_required_string(source.get("dataset_uuid"), "dataset_uuid"),
            development_sites=tuple(
                _required_string(value, "development site") for value in development_sites
            ),
            blind_sites=tuple(_required_string(value, "blind site") for value in blind_sites),
            qualification_role=_required_string(qualification.get("role"), "qualification role"),
            qualification_status=_required_string(
                qualification.get("qualification_status"), "qualification_status"
            ),
            transfer_ceiling=_required_string(
                qualification.get("transfer_ceiling"), "transfer_ceiling"
            ),
            target_truth=qualification.get("target_truth"),
            synthesis_authorized=qualification.get("synthesis_authorized"),
            plots=tuple(SelectedPlot.from_mapping(item) for item in plots),
        )
        if (
            record.schema_version != _SELECTION_SCHEMA
            or record.selection_id != _SELECTION_ID
            or record.dataset_uuid != _DATASET_UUID
            or record.qualification_role != "raw_candidate"
            or record.qualification_status != "unqualified"
            or record.transfer_ceiling != "none"
            or record.target_truth is not False
            or record.synthesis_authorized is not False
        ):
            raise ValueError("Hovi first conversion may only inventory unqualified raw candidates")
        return record

    def development_plot(self, plot_id: str) -> SelectedPlot:
        matches = tuple(plot for plot in self.plots if plot.plot_id == plot_id)
        if len(matches) != 1:
            raise ValueError(f"Hovi selection does not contain exactly one plot {plot_id!r}")
        plot = matches[0]
        if (
            not plot.role.startswith("development_")
            or plot.sealed_until_converter_freeze
            or plot.site_id not in self.development_sites
            or plot.site_id in self.blind_sites
        ):
            raise ValueError(f"Hovi plot {plot_id} is not an unsealed development plot")
        return plot


@dataclass(frozen=True)
class RetainedArtifact:
    file_id: str
    source_path: str
    bytes: int
    sha256: str
    kind: str
    plot_id: str | None
    tranche: str
    local_path: Path

    def matches(self, selected: SelectedArtifact) -> bool:
        return (
            self.file_id,
            self.source_path,
            self.bytes,
            self.sha256,
            self.kind,
        ) == (
            selected.file_id,
            selected.source_path,
            selected.bytes,
            selected.sha256,
            selected.kind,
        )


@dataclass(frozen=True)
class RetainedSelection:
    path: Path
    sha256: str
    retention_id: str
    selection_sha256: str
    complete: bool
    artifacts: tuple[RetainedArtifact, ...]

    @classmethod
    def load(
        cls, path: Path, *, selection: HoviSelection
    ) -> "RetainedSelection":
        path = path.resolve()
        if path.name != "retained.json" or not path.is_file():
            raise ValueError("Hovi retained evidence must be an existing retained.json")
        encoded = path.read_bytes()
        raw = json.loads(encoded)
        if not isinstance(raw, Mapping):
            raise ValueError("Hovi retained manifest must be a JSON object")
        retained_selection = raw.get("selection")
        plan_identity = raw.get("plan_identity")
        if not isinstance(retained_selection, Mapping):
            raise ValueError("Hovi retained manifest lacks its selection identity")
        if not isinstance(plan_identity, Mapping):
            raise ValueError("Hovi retained manifest lacks its stable plan identity")
        retention_id = _sha256(raw.get("retention_id"), "retention_id")
        scope = raw.get("authorized_scope")
        scope_record = _UNSEALED_RETENTION_SCOPES.get(scope)
        if scope_record is None:
            raise ValueError("Hovi retained manifest is not an unsealed development scope")
        retained_plot_id, retained_slug = scope_record
        if (
            raw.get("schema_version") != "hovi-retained-evidence/1.0.0"
            or raw.get("plan_sha256") != retention_id
            or path.parent.name != retention_id
            or sha256_bytes(canonical_json_bytes(plan_identity).rstrip(b"\n")) != retention_id
            or retained_selection.get("id") != selection.selection_id
            or retained_selection.get("config_sha256") != selection.sha256
            or plan_identity.get("authorized_scope") != scope
        ):
            raise ValueError("Hovi retained manifest does not match the frozen selection")
        completed_tranches = raw.get("completed_tranches")
        if (
            raw.get("complete") is not True
            or raw.get("requested_complete") is not True
            or raw.get("requested_through") != f"{retained_slug}-geometry"
            or not isinstance(completed_tranches, list)
            or f"{retained_slug}-geometry" not in completed_tranches
        ):
            raise ValueError(
                f"Hovi retained manifest is incomplete through {retained_plot_id} geometry"
            )
        files = raw.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("Hovi retained manifest files must be a non-empty list")
        root = path.parent.resolve()
        artifacts: list[RetainedArtifact] = []
        for item in files:
            if not isinstance(item, Mapping) or item.get("status") != "verified":
                continue
            byte_count = item.get("bytes")
            if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count <= 0:
                raise ValueError("Hovi retained artifact bytes must be a positive integer")
            source_path = _source_path(item.get("path"), "retained artifact path")
            if item.get("dataset_uuid") != selection.dataset_uuid:
                raise ValueError("Hovi retained artifact dataset_uuid changed")
            plot_id = item.get("plot_id")
            if plot_id is not None and (not isinstance(plot_id, str) or not plot_id):
                raise ValueError("Hovi retained artifact plot_id is invalid")
            expected_relative = Path("files", *PurePosixPath(source_path).parts[1:])
            relative_raw = _required_string(
                item.get("relative_path"), "retained artifact relative_path"
            )
            relative_path = Path(relative_raw)
            if relative_path != expected_relative or relative_path.is_absolute():
                raise ValueError("Hovi retained artifact relative_path is not canonical")
            local_path = (root / relative_path).resolve()
            if not local_path.is_relative_to(root) or not local_path.is_file():
                raise ValueError(f"Hovi retained artifact is missing: {relative_raw}")
            if local_path.stat().st_size != byte_count:
                raise ValueError(f"Hovi retained artifact byte count changed: {relative_raw}")
            artifacts.append(
                RetainedArtifact(
                    file_id=_required_string(item.get("file_id"), "retained file_id"),
                    source_path=source_path,
                    bytes=byte_count,
                    sha256=_sha256(item.get("sha256"), "retained artifact sha256"),
                    kind=_required_string(item.get("kind"), "retained artifact kind"),
                    plot_id=plot_id,
                    tranche=_required_string(item.get("tranche"), "retained artifact tranche"),
                    local_path=local_path,
                )
            )
        stable = tuple(
            (item.file_id, item.source_path, item.bytes, item.sha256) for item in artifacts
        )
        if len(set(stable)) != len(stable):
            raise ValueError("Hovi retained manifest contains duplicate verified identities")
        if (
            raw.get("retained_file_count") != len(artifacts)
            or raw.get("expected_file_count") != len(artifacts)
            or raw.get("retained_bytes") != sum(item.bytes for item in artifacts)
            or raw.get("expected_bytes") != sum(item.bytes for item in artifacts)
        ):
            raise ValueError("Hovi retained manifest completion accounting is inconsistent")
        return cls(
            path=path,
            sha256=sha256_bytes(encoded),
            retention_id=retention_id,
            selection_sha256=selection.sha256,
            complete=True,
            artifacts=tuple(artifacts),
        )

    def artifact_for(self, selected: SelectedArtifact) -> RetainedArtifact:
        matches = tuple(item for item in self.artifacts if item.matches(selected))
        if len(matches) != 1:
            raise ValueError(
                f"Hovi retained manifest does not contain selected artifact {selected.source_path}"
            )
        return matches[0]
