"""Frozen identity and strict raster inspection for the selected FORWARD DTM."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import rasterio

from ...config import CONFIG_DIR

_CONFIG_PATH = CONFIG_DIR / "evidence" / "forward-marrviken-harvest-dtm.json"
_SCHEMA = "forward-marrviken-dtm-selection/1.0.0"


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ForwardSelection:
    path: Path
    encoded: bytes
    raw: Mapping[str, Any]
    selection_sha256: str
    retention_id: str

    @property
    def artifact(self) -> Mapping[str, Any]:
        return self.raw["selected_artifact"]

    @property
    def raster(self) -> Mapping[str, Any]:
        return self.raw["expected_raster"]


def load_forward_selection(path: Path | None = None) -> ForwardSelection:
    selected = (path or _CONFIG_PATH).resolve()
    encoded = selected.read_bytes()
    raw = json.loads(encoded)
    if not isinstance(raw, Mapping) or raw.get("schema_version") != _SCHEMA:
        raise ValueError("FORWARD selection schema changed")
    artifact = raw.get("selected_artifact")
    raster = raw.get("expected_raster")
    qualification = raw.get("qualification")
    if not all(isinstance(item, Mapping) for item in (artifact, raster, qualification)):
        raise ValueError("FORWARD selection is incomplete")
    if (
        raw.get("dataset", {}).get("doi") != "10.71540/89rs-s553"
        or raw.get("dataset", {}).get("version") != 2
        or artifact.get("bytes") != 48_058_366
        or artifact.get("manifest_md5") != "421586d02cceda5ed4096ddaddd9d95a"
        or artifact.get("sha256")
        != "592ce462120fd2958e365fd6923bd389e0e282602eb05705fdcf68fc4ecb9e86"
        or qualification.get("status") != "unqualified"
        or qualification.get("target_truth") is not False
        or qualification.get("synthesis_authorized") is not False
    ):
        raise ValueError("FORWARD frozen source or qualification changed")
    selection_sha256 = hashlib.sha256(encoded).hexdigest()
    identity = {
        "schema_version": "forward-dtm-retention-plan/1.0.0",
        "selection_sha256": selection_sha256,
        "doi": raw["dataset"]["doi"],
        "dataset_version": raw["dataset"]["version"],
        "artifact": dict(artifact),
        "scope": "exact_marrviken_harvest_dtm_only",
    }
    retention_id = hashlib.sha256(canonical_json(identity)).hexdigest()
    return ForwardSelection(selected, encoded, raw, selection_sha256, retention_id)


def verify_source_bytes(path: Path, selection: ForwardSelection) -> None:
    artifact = selection.artifact
    if (
        not path.is_file()
        or path.stat().st_size != artifact["bytes"]
        or md5_file(path) != artifact["manifest_md5"]
        or sha256_file(path) != artifact["sha256"]
    ):
        raise ValueError("FORWARD DTM bytes differ from the frozen manifest identity")


def inspect_source_raster(path: Path, selection: ForwardSelection) -> dict[str, Any]:
    expected = selection.raster
    with rasterio.open(path) as dataset:
        observed = {
            "driver": dataset.driver,
            "crs": str(dataset.crs),
            "dtype": dataset.dtypes[0],
            "shape_yx": [dataset.height, dataset.width],
            "pixel_xy_m": [float(dataset.transform.a), float(dataset.transform.e)],
            "bounds_xy_m": [float(value) for value in dataset.bounds],
            "band_count": dataset.count,
            "nodata": dataset.nodata,
            "transform": [float(value) for value in dataset.transform[:6]],
            "block_shapes": [list(value) for value in dataset.block_shapes],
            "compression": None if dataset.compression is None else str(dataset.compression),
            "tags": dataset.tags(),
        }
    for key in (
        "driver",
        "crs",
        "dtype",
        "shape_yx",
        "pixel_xy_m",
        "bounds_xy_m",
        "band_count",
        "nodata",
    ):
        if observed[key] != expected[key]:
            raise ValueError(f"FORWARD DTM raster {key} changed")
    return observed
