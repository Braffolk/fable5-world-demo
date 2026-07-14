"""Deterministic tile storage for the 0.25 m structural authority."""
from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ...height_geom import HeightChunkId
from .baseline import BaselineTile
from .model import StructuralTile

if TYPE_CHECKING:
    from .hierarchy import BaselineTileInput

_TILE_SIDE = 512
_STRUCTURAL_ARRAY_ORDER = (
    "height",
    "valid",
    "unknown_bathymetry",
    "forbidden_morphology",
)
_BASELINE_ARRAY_ORDER = ("height", "valid")
AUTHORITY_MANIFEST_FORMAT = 2
BASELINE_TILE_CONTRACT_VERSION = "baseline-tile-npz/1"


def baseline_tile_contract() -> dict:
    """Return the persisted baseline contract embedded in authority identities."""
    return {
        "version": BASELINE_TILE_CONTRACT_VERSION,
        "role": "unmodified_structural_baseline_0.25m",
        "arrayOrder": list(_BASELINE_ARRAY_ORDER),
        "heightDtype": "<f4",
        "validDtype": "u1",
        "invalidHeight": "canonical_nan",
        "coreResolution": _TILE_SIDE,
    }


@dataclass(frozen=True)
class BaselineTileArtifact:
    chunk: HeightChunkId
    relative_path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class StructuralTileArtifact:
    chunk: HeightChunkId
    relative_path: str
    bytes: int
    sha256: str
    evidence_sha256: str
    baseline_relative_path: str
    baseline_bytes: int
    baseline_sha256: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _npy_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(output, np.asarray(array), allow_pickle=False)
    return output.getvalue()


def _float32_height(
    height: np.ndarray,
    valid: np.ndarray,
    *,
    role: str,
) -> np.ndarray:
    encoded = np.asarray(height, dtype="<f4").copy()
    encoded[~valid] = np.float32(np.nan)
    decoded = encoded.astype(np.float64)
    if not np.isfinite(decoded[valid]).all():
        raise ValueError(f"{role} float32 conversion produced nonfinite valid height")
    peak = np.max(np.abs(encoded[valid]), initial=np.float32(0.0))
    tolerance = 2.0 * abs(float(np.spacing(peak)))
    if np.max(np.abs(decoded[valid] - height[valid]), initial=0.0) > tolerance:
        raise ValueError(f"float32 {role} conversion exceeded its spacing bound")
    return encoded


def _encode_arrays(arrays: dict[str, np.ndarray], order: tuple[str, ...]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in order:
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(arrays[name]), compresslevel=9)
    return output.getvalue()


def _decode_arrays(payload: bytes, order: tuple[str, ...], *, role: str) -> dict:
    with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
        if archive.namelist() != [f"{name}.npy" for name in order]:
            raise ValueError(f"{role} tile has an unexpected array set/order")
        return {
            name: np.load(io.BytesIO(archive.read(f"{name}.npy")), allow_pickle=False)
            for name in order
        }


def _validate_decoded_arrays(
    arrays: dict[str, np.ndarray], order: tuple[str, ...], *, role: str
) -> None:
    if arrays["height"].dtype != np.dtype("<f4"):
        raise ValueError(f"{role} height must be little-endian float32")
    if arrays["height"].shape != (_TILE_SIDE, _TILE_SIDE):
        raise ValueError(f"{role} height must be {_TILE_SIDE}x{_TILE_SIDE}")
    for name in order[1:]:
        if (
            arrays[name].dtype != np.uint8
            or arrays[name].shape != arrays["height"].shape
            or np.any(arrays[name] > 1)
        ):
            raise ValueError(f"{role} mask {name} is not canonical uint8 boolean")


def encode_structural_tile(tile: StructuralTile) -> bytes:
    """Encode one tile with fixed ZIP metadata and deterministic array order."""
    if tile.height.shape != (_TILE_SIDE, _TILE_SIDE):
        raise ValueError(f"structural authority tile must be {_TILE_SIDE}x{_TILE_SIDE}")
    height = _float32_height(tile.height, tile.valid, role="structural tile")
    arrays = {
        "height": height,
        "valid": np.asarray(tile.valid, dtype=np.uint8),
        "unknown_bathymetry": np.asarray(tile.unknown_bathymetry, dtype=np.uint8),
        "forbidden_morphology": np.asarray(tile.forbidden_morphology, dtype=np.uint8),
    }
    return _encode_arrays(arrays, _STRUCTURAL_ARRAY_ORDER)


def decode_structural_tile(payload: bytes) -> StructuralTile:
    """Decode and validate one deterministic authority tile."""
    arrays = _decode_arrays(payload, _STRUCTURAL_ARRAY_ORDER, role="structural")
    _validate_decoded_arrays(arrays, _STRUCTURAL_ARRAY_ORDER, role="structural")
    return StructuralTile(
        height=arrays["height"],
        valid=arrays["valid"].astype(bool),
        unknown_bathymetry=arrays["unknown_bathymetry"].astype(bool),
        forbidden_morphology=arrays["forbidden_morphology"].astype(bool),
    )


def encode_baseline_tile(tile: BaselineTile) -> bytes:
    """Encode one unmodified baseline tile with canonical invalid NaN samples."""
    if tile.height.shape != (_TILE_SIDE, _TILE_SIDE):
        raise ValueError(f"baseline authority tile must be {_TILE_SIDE}x{_TILE_SIDE}")
    arrays = {
        "height": _float32_height(tile.height, tile.valid, role="baseline tile"),
        "valid": np.asarray(tile.valid, dtype=np.uint8),
    }
    return _encode_arrays(arrays, _BASELINE_ARRAY_ORDER)


def decode_baseline_tile(payload: bytes) -> BaselineTile:
    """Decode and validate one deterministic unmodified baseline tile."""
    arrays = _decode_arrays(payload, _BASELINE_ARRAY_ORDER, role="baseline")
    _validate_decoded_arrays(arrays, _BASELINE_ARRAY_ORDER, role="baseline")
    return BaselineTile(
        height=arrays["height"],
        valid=arrays["valid"].astype(bool),
    )


def _write_immutable_artifact(path: Path, payload: bytes, *, role: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable {role} tile differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_baseline_tile(
    root: Path,
    chunk: HeightChunkId,
    tile: BaselineTile,
) -> BaselineTileArtifact:
    """Atomically create or byte-verify one unmodified baseline artifact."""
    if chunk.lod != -2:
        raise ValueError("0.25 m baseline tiles use the LOD-2 128 m alignment")
    payload = encode_baseline_tile(tile)
    relative = Path("baseline") / f"{chunk.cx}_{chunk.cz}.npz"
    _write_immutable_artifact(root / relative, payload, role="baseline")
    return BaselineTileArtifact(
        chunk=chunk,
        relative_path=relative.as_posix(),
        bytes=len(payload),
        sha256=_sha256_bytes(payload),
    )


def write_structural_tile(
    root: Path,
    chunk: HeightChunkId,
    tile: StructuralTile,
    *,
    evidence_sha256: str,
    baseline_artifact: BaselineTileArtifact,
) -> StructuralTileArtifact:
    """Atomically create or byte-verify one tile in a recipe-addressed root."""
    if chunk.lod != -2:
        raise ValueError("0.25 m authority tiles use the LOD-2 128 m alignment")
    try:
        if len(bytes.fromhex(evidence_sha256)) != 32:
            raise ValueError
    except ValueError as error:
        raise ValueError("evidence_sha256 must be a SHA-256 hex digest") from error
    if baseline_artifact.chunk != chunk:
        raise ValueError("structural tile baseline artifact has the wrong chunk key")
    payload = encode_structural_tile(tile)
    relative = Path("tiles") / f"{chunk.cx}_{chunk.cz}.npz"
    _write_immutable_artifact(root / relative, payload, role="structural")
    return StructuralTileArtifact(
        chunk=chunk,
        relative_path=relative.as_posix(),
        bytes=len(payload),
        sha256=_sha256_bytes(payload),
        evidence_sha256=evidence_sha256,
        baseline_relative_path=baseline_artifact.relative_path,
        baseline_bytes=baseline_artifact.bytes,
        baseline_sha256=baseline_artifact.sha256,
    )


def _artifact_path(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("authority artifact path must stay relative to its root")
    return root / relative


def load_baseline_tile_input(
    root: Path,
    artifact: StructuralTileArtifact | BaselineTileArtifact,
) -> BaselineTileInput:
    """Integrity-check and load a hierarchy-compatible baseline input."""
    from .hierarchy import BaselineTileInput

    if isinstance(artifact, StructuralTileArtifact):
        relative = artifact.baseline_relative_path
        expected_bytes = artifact.baseline_bytes
        expected_sha256 = artifact.baseline_sha256
    elif isinstance(artifact, BaselineTileArtifact):
        relative = artifact.relative_path
        expected_bytes = artifact.bytes
        expected_sha256 = artifact.sha256
    else:
        raise TypeError("artifact must be structural or baseline tile metadata")
    path = _artifact_path(root, relative)
    payload = path.read_bytes()
    if len(payload) != expected_bytes or _sha256_bytes(payload) != expected_sha256:
        raise ValueError(f"baseline artifact integrity mismatch: {path}")
    return BaselineTileInput(
        chunk=artifact.chunk,
        tile=decode_baseline_tile(payload),
        artifact_sha256=expected_sha256,
    )


def encode_authority_manifest(
    *,
    recipe_sha256: str,
    profile_qualification_sha256: str,
    campaign_content_sha256: str,
    evidence_binding_version: str,
    artifacts: tuple[StructuralTileArtifact, ...],
    baseline_authority: dict,
) -> bytes:
    """Encode the complete, unique, sorted structural tile inventory."""
    ordered = tuple(sorted(artifacts, key=lambda item: item.chunk))
    if not ordered or len({item.chunk for item in ordered}) != len(ordered):
        raise ValueError("authority manifest requires nonempty unique tile keys")
    if any(item.chunk.lod != -2 for item in ordered):
        raise ValueError("authority manifest contains a non-LOD-2 alignment tile")
    if any(item.evidence_sha256 != campaign_content_sha256 for item in ordered):
        raise ValueError("authority tiles do not share the campaign-content identity")
    if not isinstance(baseline_authority, dict):
        raise TypeError("baseline_authority must be an identity document")
    release = baseline_authority.get("release")
    source_tiles = baseline_authority.get("tiles")
    if not isinstance(release, dict) or not isinstance(source_tiles, list):
        raise ValueError("baseline authority has no pinned release/tile identity")
    required_release = {
        "manifestSha256",
        "heightIndexPath",
        "heightIndexBytes",
        "heightIndexSha256",
        "reconstructionVersion",
    }
    if set(release) != required_release:
        raise ValueError("pinned baseline release identity keys differ")
    for name in ("manifestSha256", "heightIndexSha256"):
        try:
            if len(bytes.fromhex(release[name])) != 32:
                raise ValueError
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid pinned baseline {name}") from error
    expected_keys = [[item.chunk.lod, item.chunk.cx, item.chunk.cz] for item in ordered]
    if [row.get("key") for row in source_tiles if isinstance(row, dict)] != expected_keys:
        raise ValueError("pinned baseline tile identities differ from artifact inventory")
    if len(source_tiles) != len(expected_keys):
        raise ValueError("pinned baseline tile identity count differs")
    for row in source_tiles:
        if (
            row.get("manifestSha256") != release["manifestSha256"]
            or row.get("reconstructionVersion") != release["reconstructionVersion"]
            or not isinstance(row.get("dependencies"), list)
            or not row["dependencies"]
        ):
            raise ValueError("pinned baseline tile identity is incomplete")
        required_dependency = {
            "chunk",
            "content_relative_path",
            "bytes",
            "sha256",
            "index_hash64",
            "decoded_sha256",
            "qoffset",
            "qscale",
            "flags",
        }
        dependency_keys = []
        for dependency in row["dependencies"]:
            if not isinstance(dependency, dict) or set(dependency) != required_dependency:
                raise ValueError("pinned baseline dependency identity keys differ")
            key = dependency["chunk"]
            if (
                not isinstance(key, list)
                or len(key) != 3
                or any(isinstance(value, bool) or not isinstance(value, int) for value in key)
            ):
                raise ValueError("pinned baseline dependency has an invalid chunk key")
            for name in ("sha256", "decoded_sha256"):
                try:
                    if len(bytes.fromhex(dependency[name])) != 32:
                        raise ValueError
                except (TypeError, ValueError) as error:
                    raise ValueError(f"invalid pinned dependency {name}") from error
            dependency_keys.append(tuple(key))
        if len(set(dependency_keys)) != len(dependency_keys):
            raise ValueError("pinned baseline tile has duplicate dependencies")
    document = {
        "format": AUTHORITY_MANIFEST_FORMAT,
        "role": "structural_authority_0.25m",
        "morphology": "absent",
        "recipeSha256": recipe_sha256,
        # Kept as the campaign-content field consumed by the format-2 release path.
        "evidenceSha256": campaign_content_sha256,
        "evidenceBinding": {
            "version": evidence_binding_version,
            "profileQualificationSha256": profile_qualification_sha256,
            "campaignContentSha256": campaign_content_sha256,
        },
        "tileAlignmentLod": -2,
        "tileTexelMeters": 0.25,
        "tileCoreResolution": _TILE_SIDE,
        "baselineContract": baseline_tile_contract(),
        "baselineAuthority": baseline_authority,
        "tiles": [
            {
                "key": [item.chunk.lod, item.chunk.cx, item.chunk.cz],
                "path": item.relative_path,
                "bytes": item.bytes,
                "sha256": item.sha256,
                "baselinePath": item.baseline_relative_path,
                "baselineBytes": item.baseline_bytes,
                "baselineSha256": item.baseline_sha256,
            }
            for item in ordered
        ],
    }
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
