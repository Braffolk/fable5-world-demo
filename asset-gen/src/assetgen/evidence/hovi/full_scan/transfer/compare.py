"""Streaming full-versus-thinned comparison without surface interpretation."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import laspy
import numpy as np

from ..candidate import SpatialObservationReader, load_verified_spatial_manifest
from .artifacts import write_machine_product
from .binding import HeldFileBinding
from .model import (
    MultiscaleObservables,
    comparison_memory_bound,
    comparison_metrics,
    frozen_hovi_grid_plan,
)


THINNED_LAZ_SHA256 = "130778345d4ee5b273ef4abe8fedab4d4e23b20238d50ce34fa1be5b17970852"
THINNED_LAZ_BYTES = 2_740_487_787
THINNED_LAZ_POINTS = 714_456_347
DEFAULT_MEMORY_CEILING_BYTES = 512 << 20
DEFAULT_BATCH_POINTS = 262_144


@dataclass(frozen=True)
class ComparisonProduct:
    metrics: dict[str, Any]
    arrays: dict[str, np.ndarray]
    sources: dict[str, Any]

    def write(self, output_dir: Path) -> Path:
        return write_machine_product(
            output_dir,
            metrics=self.metrics,
            arrays=self.arrays,
            sources=self.sources,
        )


def compare_hovi_full_vs_thinned(
    full_manifest_path: Path,
    thinned_laz_path: Path,
    *,
    batch_points: int = DEFAULT_BATCH_POINTS,
    memory_ceiling_bytes: int = DEFAULT_MEMORY_CEILING_BYTES,
    log: Callable[[str], None] = print,
) -> ComparisonProduct:
    """Compare numeric raw-observation support without asserting frame equivalence."""
    if not isinstance(batch_points, int) or batch_points <= 0:
        raise ValueError("Hovi transfer batch_points must be positive")
    if not isinstance(memory_ceiling_bytes, int) or memory_ceiling_bytes <= 0:
        raise ValueError("Hovi transfer memory ceiling must be positive")
    plan = frozen_hovi_grid_plan()
    memory_bound = comparison_memory_bound(plan, batch_points=batch_points)
    if memory_bound.peak_bytes > memory_ceiling_bytes:
        raise MemoryError(
            "Hovi transfer phase-complete peak-live bound "
            f"{memory_bound.peak_bytes} exceeds {memory_ceiling_bytes} bytes"
        )

    manifest_path = Path(os.path.abspath(os.fspath(full_manifest_path)))
    manifest_digest = manifest_path.parent.name
    with HeldFileBinding(
        manifest_path,
        expected_sha256=manifest_digest,
        expected_bytes=None,
        expected_mode=0o400,
        label="Hovi full spatial manifest",
    ) as manifest_binding:
        manifest = load_verified_spatial_manifest(manifest_path)
        if manifest.manifest_sha256 != manifest_binding.sha256:
            raise ValueError("Hovi full manifest loader and held binding disagree")
        full = MultiscaleObservables(plan)
        reader = SpatialObservationReader(manifest)
        for index, shard in enumerate(manifest.point_shards, start=1):
            for batch in reader.iter_shard_artifact(shard, batch_records=batch_points):
                full.update(batch.file_xyz[:, 0], batch.file_xyz[:, 1])
            log(f"Hovi transfer: full shard {index}/{len(manifest.point_shards)}")
        full.validate()
        expected_full = sum(shard.records for shard in manifest.point_shards)
        if full.input_points != expected_full or full.outside_aoi_points != 0:
            raise ValueError(
                "Hovi full observations do not conserve the verified AOI shard inventory"
            )
        full_source = manifest_binding.document()
        full_source.update(
            {
                "kind": "verified_full_scan_spatial_materialization",
                "point_shards": len(manifest.point_shards),
                "point_records": expected_full,
                "artifact_bytes": sum(shard.bytes for shard in manifest.point_shards),
                "artifact_verification": (
                    "candidate reader SHA-256 and stat before/after each shard"
                ),
            }
        )

    with HeldFileBinding(
        thinned_laz_path,
        expected_sha256=THINNED_LAZ_SHA256,
        expected_bytes=THINNED_LAZ_BYTES,
        expected_mode=0o644,
        label="Hovi merged thinned LAZ",
    ) as thinned_binding:
        thinned = MultiscaleObservables(plan)
        with thinned_binding.binary_stream() as source:
            with laspy.open(source, closefd=False) as las:
                if int(las.header.point_count) != THINNED_LAZ_POINTS:
                    raise ValueError("Hovi thinned LAZ point count differs from its frozen source")
                decoded = 0
                for chunk_index, points in enumerate(
                    las.chunk_iterator(batch_points), start=1
                ):
                    x = np.asarray(points.x, dtype=np.float64)
                    y = np.asarray(points.y, dtype=np.float64)
                    thinned.update(x, y)
                    decoded += len(points)
                    log(f"Hovi transfer: thinned chunk {chunk_index} ({decoded:,} points)")
        thinned.validate()
        if thinned.input_points != THINNED_LAZ_POINTS or thinned.outside_aoi_points != 0:
            raise ValueError("Hovi thinned observations do not conserve the frozen common AOI")
        thinned_source = thinned_binding.document()
        thinned_source.update(
            {
                "kind": "merged_thinned_geometry_preview",
                "point_records": THINNED_LAZ_POINTS,
                "scan_identity": "absent",
            }
        )

    metrics = comparison_metrics(full, thinned)
    arrays = full.take_arrays("full")
    arrays.update(thinned.take_arrays("thinned"))
    sources = {
        "full": full_source,
        "thinned": thinned_source,
        "comparison_boundary": {
            "numeric_coordinate_rebinning_only": True,
            "frame_equivalence_established": False,
            "point_correspondence_available": False,
        },
    }
    return ComparisonProduct(metrics=metrics, arrays=arrays, sources=sources)
