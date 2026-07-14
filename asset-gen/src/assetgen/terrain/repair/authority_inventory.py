"""Integrity-bound, bounded readers for structural and canonical authority tiles."""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...height_geom import HeightChunkId
from .baseline import BaselineTile
from .model import StructuralTile
from .pinned_baseline import PinnedDecodedBaseline, ReconstructedBaselineTile
from .storage import (
    AUTHORITY_MANIFEST_FORMAT,
    decode_baseline_tile,
    decode_structural_tile,
)


_PAIR_DOMAIN = b"laas.terrain.structural-authority-pair.v1\0"
_CANONICAL_DOMAIN = b"laas.terrain.canonical-baseline.v1\0"


def validated_sha256(value: str, name: str) -> str:
    try:
        if value != value.lower() or len(bytes.fromhex(value)) != 32:
            raise ValueError
    except (AttributeError, ValueError) as error:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest") from error
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(document: object) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


def _safe_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"authority artifact path is unsafe: {relative!r}")
    return root / candidate


@dataclass(frozen=True)
class StructuralAuthorityPair:
    chunk: HeightChunkId
    structural: StructuralTile
    paired_baseline: BaselineTile
    structural_sha256: str
    paired_baseline_sha256: str
    authority_manifest_sha256: str

    def __post_init__(self) -> None:
        validated_sha256(self.structural_sha256, "structural_sha256")
        validated_sha256(self.paired_baseline_sha256, "paired_baseline_sha256")
        validated_sha256(
            self.authority_manifest_sha256, "authority_manifest_sha256"
        )
        if self.structural.height.shape != self.paired_baseline.height.shape:
            raise ValueError("structural authority pair has mismatched tile shapes")

    @property
    def pair_sha256(self) -> str:
        digest = hashlib.sha256(_PAIR_DOMAIN)
        digest.update(bytes.fromhex(self.structural_sha256))
        digest.update(bytes.fromhex(self.paired_baseline_sha256))
        return digest.hexdigest()


@dataclass(frozen=True)
class CanonicalBaselineInput:
    chunk: HeightChunkId
    tile: BaselineTile
    artifact_sha256: str
    base_manifest_sha256: str

    def __post_init__(self) -> None:
        validated_sha256(self.artifact_sha256, "canonical baseline artifact_sha256")
        validated_sha256(self.base_manifest_sha256, "base_manifest_sha256")


@dataclass(frozen=True)
class _AuthorityRecord:
    chunk: HeightChunkId
    structural_path: str
    structural_bytes: int
    structural_sha256: str
    baseline_path: str
    baseline_bytes: int
    baseline_sha256: str


class StructuralAuthorityInventory:
    """Integrity-checked format-2 authority manifest with a bounded pair LRU."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        manifest_sha256: str,
        expected_chunks: tuple[HeightChunkId, ...],
        cache_tiles: int = 12,
    ) -> None:
        if cache_tiles < 1:
            raise ValueError("cache_tiles must be positive")
        path = Path(manifest_path)
        payload = path.read_bytes()
        expected_manifest_sha = validated_sha256(
            manifest_sha256, "authority manifest_sha256"
        )
        if _sha256_bytes(payload) != expected_manifest_sha:
            raise ValueError("authority manifest hash mismatch")
        document = json.loads(payload)
        if (
            document.get("format") != AUTHORITY_MANIFEST_FORMAT
            or document.get("role") != "structural_authority_0.25m"
            or document.get("morphology") != "absent"
            or document.get("tileAlignmentLod") != -2
            or document.get("tileCoreResolution") != 512
        ):
            raise ValueError("authority manifest has the wrong structural contract")
        baseline_authority = document.get("baselineAuthority")
        if not isinstance(baseline_authority, dict):
            raise ValueError("authority manifest lacks pinned baseline authority")
        baseline_release = baseline_authority.get("release")
        baseline_rows = baseline_authority.get("tiles")
        if not isinstance(baseline_release, dict) or not isinstance(baseline_rows, list):
            raise ValueError("authority manifest has invalid pinned baseline authority")
        base_manifest_sha = validated_sha256(
            baseline_release.get("manifestSha256"), "pinned base manifestSha256"
        )
        raw_records = document.get("tiles")
        if not isinstance(raw_records, list):
            raise ValueError("authority manifest has no tile inventory")
        records: dict[HeightChunkId, _AuthorityRecord] = {}
        for row in raw_records:
            try:
                chunk = HeightChunkId(*(int(value) for value in row["key"]))
                record = _AuthorityRecord(
                    chunk=chunk,
                    structural_path=str(row["path"]),
                    structural_bytes=int(row["bytes"]),
                    structural_sha256=validated_sha256(row["sha256"], "tile sha256"),
                    baseline_path=str(row["baselinePath"]),
                    baseline_bytes=int(row["baselineBytes"]),
                    baseline_sha256=validated_sha256(
                        row["baselineSha256"], "paired baseline sha256"
                    ),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("authority manifest contains an invalid tile row") from error
            if chunk in records or chunk.lod != -2:
                raise ValueError("authority manifest tile keys must be unique LOD-2 keys")
            records[chunk] = record
        if set(records) != set(expected_chunks) or len(records) != len(expected_chunks):
            raise ValueError("authority manifest inventory differs from planned support")

        baseline_identities: dict[HeightChunkId, dict] = {}
        for row in baseline_rows:
            try:
                chunk = HeightChunkId(*(int(value) for value in row["key"]))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("pinned baseline identity has an invalid key") from error
            if (
                chunk in baseline_identities
                or row.get("manifestSha256") != base_manifest_sha
                or row.get("reconstructionVersion")
                != baseline_release.get("reconstructionVersion")
            ):
                raise ValueError("pinned baseline identity differs from release authority")
            validated_sha256(
                row.get("dependencyRootSha256"), "baseline dependencyRootSha256"
            )
            validated_sha256(
                row.get("sourceAuthoritySha256"), "baseline sourceAuthoritySha256"
            )
            if not isinstance(row.get("dependencies"), list) or not row["dependencies"]:
                raise ValueError("pinned baseline identity has no decoded source dependencies")
            baseline_identities[chunk] = row
        if set(baseline_identities) != set(records):
            raise ValueError("pinned baseline inventory differs from structural pairs")

        self.manifest_sha256 = expected_manifest_sha
        self.base_manifest_sha256 = base_manifest_sha
        self.recipe_sha256 = validated_sha256(
            document.get("recipeSha256"), "authority recipeSha256"
        )
        self._root = path.parent
        self._records = records
        self._baseline_identities = baseline_identities
        self._cache_tiles = cache_tiles
        self._cache: OrderedDict[HeightChunkId, StructuralAuthorityPair] = OrderedDict()

    @property
    def cached_tile_count(self) -> int:
        return len(self._cache)

    def load(self, chunk: HeightChunkId) -> StructuralAuthorityPair:
        cached = self._cache.pop(chunk, None)
        if cached is not None:
            self._cache[chunk] = cached
            return cached
        try:
            record = self._records[chunk]
        except KeyError as error:
            raise ValueError(f"authority manifest lacks planned tile {chunk}") from error

        def checked(relative: str, size: int, sha256: str, role: str) -> bytes:
            path = _safe_path(self._root, relative)
            payload = path.read_bytes()
            if len(payload) != size or _sha256_bytes(payload) != sha256:
                raise ValueError(f"{role} artifact integrity mismatch: {path}")
            return payload

        structural = decode_structural_tile(
            checked(
                record.structural_path,
                record.structural_bytes,
                record.structural_sha256,
                "structural",
            )
        )
        baseline = decode_baseline_tile(
            checked(
                record.baseline_path,
                record.baseline_bytes,
                record.baseline_sha256,
                "paired baseline",
            )
        )
        unowned = ~structural.unknown_bathymetry
        if (
            not np.array_equal(structural.height[unowned], baseline.height[unowned])
            or not np.array_equal(structural.valid[unowned], baseline.valid[unowned])
        ):
            raise ValueError(f"structural artifact changed paired unowned samples: {chunk}")
        result = StructuralAuthorityPair(
            chunk,
            structural,
            baseline,
            record.structural_sha256,
            record.baseline_sha256,
            self.manifest_sha256,
        )
        self._cache[chunk] = result
        while len(self._cache) > self._cache_tiles:
            self._cache.popitem(last=False)
        return result

    def load_canonical(self, chunk: HeightChunkId) -> CanonicalBaselineInput:
        """Load the persisted canonical baseline and bind its decoded-base proof."""
        pair = self.load(chunk)
        identity = self._baseline_identities[chunk]
        digest = hashlib.sha256(_CANONICAL_DOMAIN)
        digest.update(bytes.fromhex(pair.paired_baseline_sha256))
        digest.update(_canonical_json(identity))
        return CanonicalBaselineInput(
            chunk,
            pair.paired_baseline,
            digest.hexdigest(),
            self.base_manifest_sha256,
        )


class PinnedCanonicalBaseline:
    """Adapter exposing pinned decoded-base reconstruction as tile inputs."""

    def __init__(self, source: PinnedDecodedBaseline) -> None:
        self._source = source

    def load(self, chunk: HeightChunkId) -> CanonicalBaselineInput:
        result = self._source.reconstruct(chunk)
        return CanonicalBaselineInput(
            chunk,
            result.tile,
            _pinned_result_sha256(result),
            result.manifest_sha256,
        )


def _pinned_result_sha256(result: ReconstructedBaselineTile) -> str:
    digest = hashlib.sha256(_CANONICAL_DOMAIN)
    digest.update(bytes.fromhex(result.manifest_sha256))
    digest.update(bytes.fromhex(result.dependency_root_sha256))
    digest.update(bytes.fromhex(result.authority_sha256))
    digest.update(result.reconstruction_version.encode())
    return digest.hexdigest()
