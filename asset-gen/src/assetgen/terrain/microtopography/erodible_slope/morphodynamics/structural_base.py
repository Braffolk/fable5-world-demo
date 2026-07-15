"""Load the accepted structural authority onto Development A's fine canvas."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .....config import ASSET_GEN_ROOT
from ....repair.storage import (
    AUTHORITY_MANIFEST_FORMAT,
    baseline_tile_contract,
    decode_baseline_tile,
    decode_structural_tile,
)
from ....repair.prolong import prolong_structural_4x

ACCEPTED_AUTHORITY_RELATIVE_PATH = Path(
    "data/work/terrain-repair/taevaskoda-ahja-structural-authority-stage1/"
    "5ffa4334e2013e67ddb2e1c390e455394f1b3fe0f294f12498197f6fd45bde36/"
    "authority/manifest.json"
)
ACCEPTED_AUTHORITY_SHA256 = (
    "56a29a8e68ddda75571e825eecde749c950c170c457da95b4244a5be96a5a1ed"
)
ACCEPTED_AUTHORITY_RECIPE_SHA256 = (
    "872c58e8429aaeb5b3f224dfe9d97638671abbd480ae8c39ea7b3cb9644cb8f2"
)

CANONICAL_BBOX_EN = (679841.0, 6443797.0, 681277.0, 6445126.0)
CONTROL_BBOX_EN = (679713.0, 6443669.0, 681405.0, 6445254.0)
FINE_CANVAS_BBOX_EN = (680416.0, 6444384.0, 680736.0, 6444576.0)
FINE_TEXEL_M = 0.0625
FINE_CANVAS_SHAPE = (3073, 5121)
OUTPUT_SAMPLE_SLICE = (slice(512, 2561), slice(512, 4609))
WEST_OUTPUT_SAMPLE_SLICE = (slice(512, 2561), slice(512, 2561))
EAST_OUTPUT_SAMPLE_SLICE = (slice(512, 2561), slice(2560, 4609))

_TILE_SIDE = 512
_AUTHORITY_TEXEL_M = 0.25
_TILE_METERS = _TILE_SIDE * _AUTHORITY_TEXEL_M
_MOSAIC_WEST = 680320.0
_MOSAIC_NORTH = 6444672.0
_PARENT_START_ROW = 384
_PARENT_START_COL = 384
_PARENT_REQUEST_SHAPE = (769, 1281)
_SUPPORT_HALO = 2
REQUIRED_AUTHORITY_CHUNKS = tuple(
    (-2, cx, cz) for cz in range(1491, 1494) for cx in range(2435, 2439)
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_artifact_path(root: Path, relative_path: object) -> Path:
    if not isinstance(relative_path, str):
        raise ValueError("authority artifact path must be a string")
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("authority artifact path escapes its root")
    root_resolved = root.resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root_resolved):
        raise ValueError("authority artifact path escapes its root")
    return result


def _read_bound_artifact(
    root: Path,
    relative_path: object,
    expected_bytes: object,
    expected_sha256: object,
) -> bytes:
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
        raise ValueError("authority artifact byte count must be an integer")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("authority artifact SHA-256 is invalid")
    path = _safe_artifact_path(root, relative_path)
    payload = path.read_bytes()
    if len(payload) != expected_bytes:
        raise ValueError(f"authority artifact byte count differs: {path}")
    if _sha256(payload) != expected_sha256:
        raise ValueError(f"authority artifact SHA-256 differs: {path}")
    return payload


def _validate_manifest_header(document: dict[str, Any]) -> None:
    expected = {
        "format": AUTHORITY_MANIFEST_FORMAT,
        "role": "structural_authority_0.25m",
        "morphology": "absent",
        "tileAlignmentLod": -2,
        "tileCoreResolution": _TILE_SIDE,
        "tileTexelMeters": _AUTHORITY_TEXEL_M,
        "recipeSha256": ACCEPTED_AUTHORITY_RECIPE_SHA256,
        "baselineContract": baseline_tile_contract(),
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise ValueError(f"accepted authority manifest has wrong {key}")


@dataclass(frozen=True)
class StructuralTileIdentity:
    chunk: tuple[int, int, int]
    structural_sha256: str
    baseline_sha256: str


@dataclass(frozen=True)
class StructuralFineCanvas:
    """Exact C0 and repair masks on the fixed inclusive 0.0625 m lattice."""

    bbox_en: tuple[float, float, float, float]
    texel_m: float
    c0_height_m: np.ndarray
    forbidden_morphology: np.ndarray
    unknown_bathymetry: np.ndarray
    c0_sha256: str
    authority_manifest_sha256: str
    authority_recipe_sha256: str
    tile_identities: tuple[StructuralTileIdentity, ...]

    def __post_init__(self) -> None:
        if self.bbox_en != FINE_CANVAS_BBOX_EN or self.texel_m != FINE_TEXEL_M:
            raise ValueError("structural fine canvas has the wrong spatial identity")
        arrays = (
            self.c0_height_m,
            self.forbidden_morphology,
            self.unknown_bathymetry,
        )
        if any(value.shape != FINE_CANVAS_SHAPE for value in arrays):
            raise ValueError("structural fine canvas has the wrong inclusive shape")
        if self.c0_height_m.dtype != np.float64:
            raise ValueError("C0 must remain float64 through morphodynamics")
        if not np.isfinite(self.c0_height_m).all():
            raise ValueError("C0 contains nonfinite height")
        encoded_c0 = np.asarray(self.c0_height_m, dtype="<f8", order="C")
        if _sha256(encoded_c0.tobytes(order="C")) != self.c0_sha256:
            raise ValueError("C0 differs from its canonical float64 SHA-256")
        if any(value.dtype != np.bool_ for value in arrays[1:]):
            raise ValueError("structural masks must be boolean")
        for value in arrays:
            value.flags.writeable = False

    def require_domain(
        self,
        role: str,
        bbox_en: tuple[float, float, float, float],
    ) -> StructuralFineCanvas:
        """Bind the same fixed C0 canvas to an exact canonical/control domain."""
        expected = {"canonical": CANONICAL_BBOX_EN, "control": CONTROL_BBOX_EN}
        if role not in expected or tuple(bbox_en) != expected[role]:
            raise ValueError(f"unexpected {role!r} domain bbox")
        e0, n0, e1, n1 = bbox_en
        ce0, cn0, ce1, cn1 = self.bbox_en
        if not (e0 <= ce0 < ce1 <= e1 and n0 <= cn0 < cn1 <= n1):
            raise ValueError("fine canvas falls outside its bound domain")
        return self


def _load_required_tiles(
    document: dict[str, Any], authority_root: Path
) -> tuple[dict[tuple[int, int, int], tuple[Any, Any]], tuple[StructuralTileIdentity, ...]]:
    rows = document.get("tiles")
    if not isinstance(rows, list):
        raise ValueError("accepted authority manifest tiles must be a list")
    required = set(REQUIRED_AUTHORITY_CHUNKS)
    selected: dict[tuple[int, int, int], dict[str, Any]] = {}
    seen: set[tuple[int, int, int]] = set()
    expected_row_keys = {
        "key",
        "path",
        "bytes",
        "sha256",
        "baselinePath",
        "baselineBytes",
        "baselineSha256",
    }
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected_row_keys:
            raise ValueError("accepted authority manifest has a malformed tile row")
        key_value = row["key"]
        if (
            not isinstance(key_value, list)
            or len(key_value) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) for value in key_value)
        ):
            raise ValueError("accepted authority manifest has an invalid tile key")
        key = tuple(key_value)
        if key in seen:
            raise ValueError(f"accepted authority manifest repeats tile {key}")
        seen.add(key)
        if key in required:
            selected[key] = row
    if set(selected) != required:
        missing = sorted(required - set(selected))
        raise ValueError(f"accepted authority manifest lacks required tiles: {missing}")

    loaded: dict[tuple[int, int, int], tuple[Any, Any]] = {}
    identities: list[StructuralTileIdentity] = []
    for key in REQUIRED_AUTHORITY_CHUNKS:
        row = selected[key]
        structural_payload = _read_bound_artifact(
            authority_root, row["path"], row["bytes"], row["sha256"]
        )
        baseline_payload = _read_bound_artifact(
            authority_root,
            row["baselinePath"],
            row["baselineBytes"],
            row["baselineSha256"],
        )
        structural = decode_structural_tile(structural_payload)
        baseline = decode_baseline_tile(baseline_payload)
        if not structural.valid.all() or not baseline.valid.all():
            raise ValueError(f"required authority tile {key} contains invalid samples")
        unchanged = ~structural.unknown_bathymetry
        if not np.array_equal(structural.valid, baseline.valid) or not np.array_equal(
            structural.height[unchanged], baseline.height[unchanged]
        ):
            raise ValueError(f"structural authority changed unowned samples in {key}")
        loaded[key] = (structural, baseline)
        identities.append(
            StructuralTileIdentity(
                chunk=key,
                structural_sha256=row["sha256"],
                baseline_sha256=row["baselineSha256"],
            )
        )
    return loaded, tuple(identities)


def _assemble_parent_mosaic(
    tiles: dict[tuple[int, int, int], tuple[Any, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (3 * _TILE_SIDE, 4 * _TILE_SIDE)
    height = np.empty(shape, dtype=np.float32)
    forbidden = np.empty(shape, dtype=bool)
    unknown = np.empty(shape, dtype=bool)
    for tile_row, cz in enumerate(range(1491, 1494)):
        for tile_col, cx in enumerate(range(2435, 2439)):
            structural, _ = tiles[(-2, cx, cz)]
            rows = slice(tile_row * _TILE_SIDE, (tile_row + 1) * _TILE_SIDE)
            cols = slice(tile_col * _TILE_SIDE, (tile_col + 1) * _TILE_SIDE)
            height[rows, cols] = structural.height
            forbidden[rows, cols] = structural.forbidden_morphology
            unknown[rows, cols] = structural.unknown_bathymetry
    expected_east = _MOSAIC_WEST + shape[1] * _AUTHORITY_TEXEL_M
    expected_south = _MOSAIC_NORTH - shape[0] * _AUTHORITY_TEXEL_M
    if (expected_east, expected_south) != (680832.0, 6444288.0):
        raise AssertionError("internal authority mosaic geometry drifted")
    return height, forbidden, unknown


def _prolong_canvas(
    parent: np.ndarray,
    forbidden: np.ndarray,
    unknown: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    requested_rows, requested_cols = _PARENT_REQUEST_SHAPE
    support_row = _PARENT_START_ROW - _SUPPORT_HALO
    support_col = _PARENT_START_COL - _SUPPORT_HALO
    support = parent[
        support_row : support_row + requested_rows + 2 * _SUPPORT_HALO,
        support_col : support_col + requested_cols + 2 * _SUPPORT_HALO,
    ]
    fine = prolong_structural_4x(
        support,
        parent_rows=(_SUPPORT_HALO, _SUPPORT_HALO + requested_rows),
        parent_cols=(_SUPPORT_HALO, _SUPPORT_HALO + requested_cols),
    )[: FINE_CANVAS_SHAPE[0], : FINE_CANVAS_SHAPE[1]]
    parent_rows = slice(_PARENT_START_ROW, _PARENT_START_ROW + requested_rows)
    parent_cols = slice(_PARENT_START_COL, _PARENT_START_COL + requested_cols)
    fine_forbidden = np.repeat(
        np.repeat(forbidden[parent_rows, parent_cols], 4, axis=0), 4, axis=1
    )[: FINE_CANVAS_SHAPE[0], : FINE_CANVAS_SHAPE[1]]
    fine_unknown = np.repeat(
        np.repeat(unknown[parent_rows, parent_cols], 4, axis=0), 4, axis=1
    )[: FINE_CANVAS_SHAPE[0], : FINE_CANVAS_SHAPE[1]]
    if fine.shape != FINE_CANVAS_SHAPE:
        raise ValueError("structural prolongation did not produce the fixed canvas")
    return fine, fine_forbidden, fine_unknown


def load_development_a_structural_base() -> StructuralFineCanvas:
    """Load and reconstruct the one accepted structural authority for Development A."""
    manifest_path = ASSET_GEN_ROOT / ACCEPTED_AUTHORITY_RELATIVE_PATH
    payload = manifest_path.read_bytes()
    if _sha256(payload) != ACCEPTED_AUTHORITY_SHA256:
        raise ValueError("accepted structural-authority manifest SHA-256 differs")
    document = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(document, dict):
        raise ValueError("accepted structural-authority manifest must be an object")
    _validate_manifest_header(document)
    tiles, identities = _load_required_tiles(document, manifest_path.parent)
    parent, forbidden, unknown = _assemble_parent_mosaic(tiles)
    c0, fine_forbidden, fine_unknown = _prolong_canvas(parent, forbidden, unknown)
    canvas = StructuralFineCanvas(
        bbox_en=FINE_CANVAS_BBOX_EN,
        texel_m=FINE_TEXEL_M,
        c0_height_m=c0,
        forbidden_morphology=fine_forbidden,
        unknown_bathymetry=fine_unknown,
        c0_sha256=_sha256(
            np.asarray(c0, dtype="<f8", order="C").tobytes(order="C")
        ),
        authority_manifest_sha256=ACCEPTED_AUTHORITY_SHA256,
        authority_recipe_sha256=ACCEPTED_AUTHORITY_RECIPE_SHA256,
        tile_identities=identities,
    )
    canvas.require_domain("canonical", CANONICAL_BBOX_EN)
    canvas.require_domain("control", CONTROL_BBOX_EN)
    return canvas
