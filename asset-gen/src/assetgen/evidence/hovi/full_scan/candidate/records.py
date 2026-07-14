"""Verified streaming decode of Hovi AOI-local point shards."""
from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ..authorization import open_regular_nofollow
from .manifest import PointShard, ScanTransform, VerifiedSpatialManifest


RAW_RECORD_DTYPE = np.dtype(
    [
        ("source_ordinal", "<u4"),
        ("raw_x", "<i4"),
        ("raw_y", "<i4"),
        ("raw_z", "<i4"),
        ("intensity_bits", "<u4"),
        ("row", "<u2"),
        ("column", "<u2"),
        ("scan_ordinal", "u1"),
        ("invalid_state", "u1"),
        ("flags", "<u2"),
    ],
    align=False,
)
if RAW_RECORD_DTYPE.itemsize != 28:
    raise AssertionError("Hovi raw record dtype does not match the frozen 28-byte ABI")


@dataclass(frozen=True)
class ObservationBatch:
    scan: int
    guid: str
    shard_x: int
    shard_y: int
    source_ordinal: np.ndarray
    raw_xyz: np.ndarray
    file_xyz: np.ndarray
    range_m: np.ndarray
    beam_direction_xyz: np.ndarray
    beam_direction_valid: np.ndarray
    intensity: np.ndarray
    intensity_finite: np.ndarray
    row: np.ndarray
    column: np.ndarray

    @property
    def size(self) -> int:
        return int(self.source_ordinal.size)


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _verify_artifact(
    manifest: VerifiedSpatialManifest,
    shard: PointShard,
) -> tuple[int, int, int, int, int]:
    path = manifest.root / shard.relative_path
    descriptor = open_regular_nofollow(path, f"Hovi point shard {shard.relative_path}")
    try:
        before = os.fstat(descriptor)
        if (
            before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) != 0o400
            or before.st_nlink != 1
            or before.st_size != shard.bytes
        ):
            raise PermissionError("Hovi point shard is not the published immutable artifact")
        digest = hashlib.sha256()
        byte_count = 0
        while block := os.read(descriptor, 8 << 20):
            digest.update(block)
            byte_count += len(block)
        after = os.fstat(descriptor)
        if (
            _identity(before) != _identity(after)
            or byte_count != shard.bytes
            or digest.hexdigest() != shard.sha256
        ):
            raise ValueError("Hovi point shard changed or failed manifest digest verification")
        return _identity(before)
    finally:
        os.close(descriptor)


def _apply_pose(
    raw_xyz: np.ndarray,
    transform: ScanTransform,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    local = raw_xyz.astype(np.float64)
    local *= np.asarray(transform.raw_scale_xyz, dtype=np.float64)
    local += np.asarray(transform.raw_offset_xyz, dtype=np.float64)
    w, x, y, z = transform.quaternion_wxyz
    qv = np.asarray((x, y, z), dtype=np.float64)
    uv = np.cross(np.broadcast_to(qv, local.shape), local)
    uuv = np.cross(np.broadcast_to(qv, local.shape), uv)
    file_xyz = local + 2.0 * (w * uv + uuv)
    file_xyz += np.asarray(transform.translation_xyz, dtype=np.float64)
    beam = file_xyz - np.asarray(transform.translation_xyz, dtype=np.float64)
    range_m = np.linalg.norm(beam, axis=1)
    if np.any(~np.isfinite(file_xyz)) or np.any(~np.isfinite(range_m)):
        raise ValueError("Hovi point shard produced invalid file-frame/range geometry")
    beam_valid = range_m > 0.0
    beam[beam_valid] /= range_m[beam_valid, None]
    beam[~beam_valid] = 0.0
    return file_xyz, range_m, beam, beam_valid


class SpatialObservationReader:
    """Read verified point shards without collapsing scans or spatial competitors."""

    def __init__(self, manifest: VerifiedSpatialManifest):
        self.manifest = manifest

    def iter_scan(self, scan: int, *, batch_records: int = 1 << 20) -> Iterator[ObservationBatch]:
        if not isinstance(batch_records, int) or batch_records <= 0:
            raise ValueError("Hovi batch_records must be positive")
        shards = self.manifest.shards_for_scan(scan)
        if not shards:
            raise ValueError(f"Hovi verified materialization has no AOI shards for scan {scan}")
        for shard in shards:
            yield from self.iter_shard_artifact(shard, batch_records=batch_records)

    def iter_spatial_shard(
        self,
        shard_x: int,
        shard_y: int,
        *,
        batch_records: int = 1 << 20,
    ) -> Iterator[ObservationBatch]:
        for shard in self.manifest.shards_at(shard_x, shard_y):
            yield from self.iter_shard_artifact(shard, batch_records=batch_records)

    def iter_shard_artifact(
        self,
        shard: PointShard,
        *,
        batch_records: int = 1 << 20,
    ) -> Iterator[ObservationBatch]:
        if not isinstance(batch_records, int) or batch_records <= 0:
            raise ValueError("Hovi batch_records must be positive")
        if shard not in self.manifest.point_shards:
            raise ValueError("Hovi point shard is not bound by this verified manifest")
        expected_identity = _verify_artifact(self.manifest, shard)
        path = self.manifest.root / shard.relative_path
        descriptor = open_regular_nofollow(path, f"verified Hovi point shard {shard.relative_path}")
        try:
            if _identity(os.fstat(descriptor)) != expected_identity:
                raise ValueError("Hovi point shard changed after digest verification")
            transform = self.manifest.scans[shard.scan]
            remaining = shard.records
            previous_source = -1
            while remaining:
                count = min(remaining, batch_records)
                needed = count * RAW_RECORD_DTYPE.itemsize
                payload = bytearray()
                while len(payload) < needed:
                    block = os.read(descriptor, needed - len(payload))
                    if not block:
                        raise ValueError("Hovi point shard ended before its manifest record count")
                    payload.extend(block)
                records = np.frombuffer(payload, dtype=RAW_RECORD_DTYPE)
                source = records["source_ordinal"].astype(np.uint64)
                raw_minimum = np.iinfo(np.int32).min
                if (
                    np.any(records["scan_ordinal"] != shard.scan)
                    or np.any(records["invalid_state"] != 0)
                    or np.any((records["flags"] & np.uint16(0xFFFE)) != 0)
                    or np.any(source >= transform.publisher_records)
                    or source[0] <= previous_source
                    or np.any(source[1:] <= source[:-1])
                    or np.any(records["row"] > 11_003)
                    or np.any(records["column"] > 27_480)
                    or np.any(records["raw_x"] == raw_minimum)
                    or np.any(records["raw_y"] == raw_minimum)
                    or np.any(records["raw_z"] == raw_minimum)
                ):
                    raise ValueError("Hovi point shard contains an out-of-contract record")
                previous_source = int(source[-1])
                raw_xyz = np.column_stack(
                    (records["raw_x"], records["raw_y"], records["raw_z"])
                ).astype(np.int32, copy=False)
                file_xyz, range_m, beam, beam_valid = _apply_pose(raw_xyz, transform)
                tick_x = np.floor(file_xyz[:, 0] * 40.0).astype(np.int64)
                tick_y = np.floor(file_xyz[:, 1] * 40.0).astype(np.int64)
                assigned_x = np.floor_divide(tick_x + 271, 160)
                assigned_y = np.floor_divide(tick_y + 244, 160)
                if np.any(assigned_x != shard.shard_x) or np.any(assigned_y != shard.shard_y):
                    raise ValueError(
                        "Hovi reconstructed point disagrees with verified shard assignment"
                    )
                intensity = records["intensity_bits"].view("<f4")
                finite = (records["flags"] & np.uint16(1)) != 0
                if np.any(finite != np.isfinite(intensity)):
                    raise ValueError("Hovi intensity finite flag differs from raw single bits")
                yield ObservationBatch(
                    scan=shard.scan,
                    guid=transform.guid,
                    shard_x=shard.shard_x,
                    shard_y=shard.shard_y,
                    source_ordinal=records["source_ordinal"].astype(np.uint32, copy=True),
                    raw_xyz=raw_xyz.copy(),
                    file_xyz=file_xyz,
                    range_m=range_m,
                    beam_direction_xyz=beam,
                    beam_direction_valid=beam_valid,
                    intensity=intensity.astype(np.float32, copy=True),
                    intensity_finite=finite,
                    row=records["row"].astype(np.uint16, copy=True),
                    column=records["column"].astype(np.uint16, copy=True),
                )
                remaining -= count
            if os.read(descriptor, 1) or _identity(os.fstat(descriptor)) != expected_identity:
                raise ValueError("Hovi point shard changed or exceeds its manifest record count")
        finally:
            os.close(descriptor)
