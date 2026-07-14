"""Bounded, semantics-neutral Hovi multiscale evidence workspaces."""
from __future__ import annotations

import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .manifest import (
    AOI_X_MIN_M,
    AOI_Y_MIN_M,
    RESOLUTIONS_M,
    SHARD_M,
    VerifiedSpatialManifest,
)
from .records import ObservationBatch, RAW_RECORD_DTYPE, _apply_pose


_INDEX_DTYPE = np.dtype("<u8")
_WORKSPACE_BYTES_PER_RECORD = RAW_RECORD_DTYPE.itemsize
_LEVEL_INDEX_BYTES_PER_RECORD = _INDEX_DTYPE.itemsize
_VALIDATION_WORK_BYTES_PER_RECORD = 192


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


@dataclass(frozen=True)
class ContinuityEvidence:
    cell_linear: np.ndarray
    scan: np.ndarray
    samples: np.ndarray
    source_consecutive_pairs: np.ndarray
    same_row_source_consecutive_pairs: np.ndarray
    unit_column_source_consecutive_pairs: np.ndarray
    same_column_source_consecutive_pairs: np.ndarray
    unit_row_source_consecutive_pairs: np.ndarray


@dataclass(frozen=True)
class StratumCounts:
    edges: np.ndarray
    counts: np.ndarray
    undefined_count: np.ndarray


@dataclass(frozen=True)
class VerticalSampleBatch:
    scan: np.ndarray
    source_ordinal: np.ndarray
    raw_z: np.ndarray
    file_z_m: np.ndarray


@dataclass(frozen=True)
class VerticalSampleGroup:
    cell_x: int
    cell_y: int
    sample_count: int
    _level: "CellEvidenceLevel"
    _start: int
    _stop: int

    def iter_batches(self, *, max_records: int) -> Iterator[VerticalSampleBatch]:
        yield from self._level._iter_vertical_batches(
            self._start,
            self._stop,
            max_records=max_records,
        )


class MultiscaleShardAccumulator:
    """Append verified batches to one private, ceiling-bound 28-byte workspace."""

    def __init__(
        self,
        manifest: VerifiedSpatialManifest,
        shard_x: int,
        shard_y: int,
        *,
        workspace_root: Path,
        memory_ceiling_bytes: int,
        temp_ceiling_bytes: int,
    ):
        if not isinstance(manifest, VerifiedSpatialManifest):
            raise TypeError("Hovi accumulator requires a VerifiedSpatialManifest")
        if not 0 <= shard_x <= 11 or not 0 <= shard_y <= 9:
            raise ValueError("Hovi accumulator shard lies outside the frozen AOI")
        if memory_ceiling_bytes < 1 << 20:
            raise ValueError("Hovi accumulator memory ceiling is below 1 MiB")
        if temp_ceiling_bytes < 1 << 20:
            raise ValueError("Hovi accumulator temp ceiling is below 1 MiB")
        root = Path(os.path.abspath(os.fspath(workspace_root)))
        result = os.lstat(root)
        if not stat.S_ISDIR(result.st_mode) or stat.S_ISLNK(result.st_mode):
            raise PermissionError("Hovi workspace root must be an existing real directory")
        self.manifest = manifest
        self.shard_x = shard_x
        self.shard_y = shard_y
        self.memory_ceiling_bytes = memory_ceiling_bytes
        self.temp_ceiling_bytes = temp_ceiling_bytes
        self._directory = Path(tempfile.mkdtemp(prefix="hovi-candidate-", dir=root))
        os.chmod(self._directory, 0o700)
        self._records_path = self._directory / "observations.bin"
        self._descriptor = os.open(
            self._records_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        self._records = 0
        self._last_scan = -1
        self._last_source = -1
        self._finished = False

    def add(self, batch: ObservationBatch) -> None:
        if self._finished:
            raise RuntimeError("Hovi accumulator is already finalized")
        self._validate_batch(batch)
        projected = self._records + batch.size
        peak_temp = projected * (
            _WORKSPACE_BYTES_PER_RECORD + _LEVEL_INDEX_BYTES_PER_RECORD
        )
        if peak_temp > self.temp_ceiling_bytes:
            raise MemoryError("Hovi candidate workspace would exceed its temp ceiling")
        packed = np.empty(batch.size, dtype=RAW_RECORD_DTYPE)
        packed["source_ordinal"] = batch.source_ordinal
        packed["raw_x"] = batch.raw_xyz[:, 0]
        packed["raw_y"] = batch.raw_xyz[:, 1]
        packed["raw_z"] = batch.raw_xyz[:, 2]
        packed["intensity_bits"] = batch.intensity.view(np.uint32)
        packed["row"] = batch.row
        packed["column"] = batch.column
        packed["scan_ordinal"] = batch.scan
        packed["invalid_state"] = 0
        packed["flags"] = batch.intensity_finite.astype(np.uint16)
        view = memoryview(packed).cast("B")
        while view:
            written = os.write(self._descriptor, view)
            if written <= 0:
                raise OSError("short write in Hovi candidate workspace")
            view = view[written:]
        self._records = projected
        self._last_scan = batch.scan
        self._last_source = int(batch.source_ordinal[-1])

    def finalize(self) -> "MultiscaleShardEvidence":
        if self._finished or self._descriptor < 0:
            raise RuntimeError("Hovi accumulator is already finalized")
        if self._records == 0:
            raise ValueError("Hovi candidate workspace has no observations")
        os.fsync(self._descriptor)
        os.fchmod(self._descriptor, 0o400)
        record_result = os.fstat(self._descriptor)
        if record_result.st_size != self._records * RAW_RECORD_DTYPE.itemsize:
            raise ValueError("Hovi candidate workspace byte count is inconsistent")
        record_identity = _identity(record_result)
        os.close(self._descriptor)
        self._descriptor = -1
        self._finished = True
        return MultiscaleShardEvidence(
            manifest=self.manifest,
            shard_x=self.shard_x,
            shard_y=self.shard_y,
            directory=self._directory,
            records_path=self._records_path,
            records_identity=record_identity,
            record_count=self._records,
            memory_ceiling_bytes=self.memory_ceiling_bytes,
            temp_ceiling_bytes=self.temp_ceiling_bytes,
        )

    def abort(self) -> None:
        if self._descriptor >= 0:
            os.close(self._descriptor)
            self._descriptor = -1
        if self._records_path.exists():
            os.unlink(self._records_path)
        if self._directory.exists():
            os.rmdir(self._directory)
        self._finished = True

    def _validate_batch(self, batch: ObservationBatch) -> None:
        scan = batch.scan
        if not isinstance(scan, int) or not 0 <= scan < len(self.manifest.scans):
            raise ValueError("Hovi batch scan is outside the verified manifest")
        binding = self.manifest.scans[scan]
        if (
            batch.guid != binding.guid
            or batch.shard_x != self.shard_x
            or batch.shard_y != self.shard_y
            or batch.size <= 0
        ):
            raise ValueError("Hovi batch identity differs from its workspace binding")
        expected = {
            "source_ordinal": ((batch.size,), np.dtype(np.uint32)),
            "raw_xyz": ((batch.size, 3), np.dtype(np.int32)),
            "file_xyz": ((batch.size, 3), np.dtype(np.float64)),
            "range_m": ((batch.size,), np.dtype(np.float64)),
            "beam_direction_xyz": ((batch.size, 3), np.dtype(np.float64)),
            "beam_direction_valid": ((batch.size,), np.dtype(np.bool_)),
            "intensity": ((batch.size,), np.dtype(np.float32)),
            "intensity_finite": ((batch.size,), np.dtype(np.bool_)),
            "row": ((batch.size,), np.dtype(np.uint16)),
            "column": ((batch.size,), np.dtype(np.uint16)),
        }
        arrays: list[np.ndarray] = []
        for name, (shape, dtype) in expected.items():
            value = getattr(batch, name)
            if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype:
                raise ValueError(f"Hovi batch {name} shape/dtype is invalid")
            arrays.append(value)
        required_memory = sum(value.nbytes for value in arrays) + (
            batch.size * _VALIDATION_WORK_BYTES_PER_RECORD
        )
        if required_memory > self.memory_ceiling_bytes:
            raise MemoryError("Hovi batch validation would exceed its memory ceiling")
        source = batch.source_ordinal.astype(np.uint64)
        if (
            np.any(source >= binding.publisher_records)
            or np.any(source[1:] <= source[:-1])
            or batch.scan < self._last_scan
            or (batch.scan == self._last_scan and int(source[0]) <= self._last_source)
            or np.any(batch.raw_xyz == np.iinfo(np.int32).min)
            or np.any(batch.row > 11_003)
            or np.any(batch.column > 27_480)
            or np.any(~np.isfinite(batch.file_xyz))
            or np.any(~np.isfinite(batch.range_m))
            or np.any(~np.isfinite(batch.beam_direction_xyz))
            or np.any(batch.intensity_finite != np.isfinite(batch.intensity))
        ):
            raise ValueError("Hovi batch values/order are outside the verified contract")
        file_xyz, range_m, beam, beam_valid = _apply_pose(batch.raw_xyz, binding)
        if (
            not np.array_equal(file_xyz, batch.file_xyz)
            or not np.array_equal(range_m, batch.range_m)
            or not np.array_equal(beam, batch.beam_direction_xyz)
            or not np.array_equal(beam_valid, batch.beam_direction_valid)
        ):
            raise ValueError("Hovi batch derived geometry differs from raw scale/pose")
        tick_x = np.floor(file_xyz[:, 0] * 40.0).astype(np.int64)
        tick_y = np.floor(file_xyz[:, 1] * 40.0).astype(np.int64)
        if (
            np.any(np.floor_divide(tick_x + 271, 160) != self.shard_x)
            or np.any(np.floor_divide(tick_y + 244, 160) != self.shard_y)
        ):
            raise ValueError("Hovi batch derived geometry belongs to another shard")


class MultiscaleShardEvidence:
    """One compact observation workspace; only one disk index may be active."""

    def __init__(
        self,
        *,
        manifest: VerifiedSpatialManifest,
        shard_x: int,
        shard_y: int,
        directory: Path,
        records_path: Path,
        records_identity: tuple[int, int, int, int, int],
        record_count: int,
        memory_ceiling_bytes: int,
        temp_ceiling_bytes: int,
    ):
        self.manifest = manifest
        self.shard_x = shard_x
        self.shard_y = shard_y
        self.directory = directory
        self.records_path = records_path
        self.records_identity = records_identity
        self.record_count = record_count
        self.memory_ceiling_bytes = memory_ceiling_bytes
        self.temp_ceiling_bytes = temp_ceiling_bytes
        self._active_level: CellEvidenceLevel | None = None
        self._closed = False

    def open_level(
        self,
        resolution_m: float,
        *,
        block_records: int,
    ) -> "CellEvidenceLevel":
        if self._closed:
            raise RuntimeError("Hovi evidence workspace is closed")
        if self._active_level is not None:
            raise RuntimeError("close the active Hovi level before opening another")
        level = CellEvidenceLevel._build(self, resolution_m, block_records)
        self._active_level = level
        return level

    def close(self) -> None:
        if self._active_level is not None:
            raise RuntimeError("close the active Hovi level before its workspace")
        if not self._closed:
            os.unlink(self.records_path)
            os.rmdir(self.directory)
            self._closed = True

    def __enter__(self) -> "MultiscaleShardEvidence":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class CellEvidenceLevel:
    """Disk-indexed cell evidence at one AOI-anchored resolution."""

    def __init__(
        self,
        evidence: MultiscaleShardEvidence,
        resolution_m: float,
        cells_per_shard: int,
        block_records: int,
        index_path: Path,
        index_identity: tuple[int, int, int, int, int],
        counts: np.ndarray,
        scan_mask: np.ndarray,
    ):
        self.evidence = evidence
        self.resolution_m = resolution_m
        self.cells_per_shard = cells_per_shard
        self.block_records = block_records
        self.index_path = index_path
        self.index_identity = index_identity
        self._counts_full = counts
        self.cell_linear = np.flatnonzero(counts).astype(np.uint32)
        self.counts = counts[self.cell_linear].astype(np.uint64)
        self.scan_mask = scan_mask[self.cell_linear]
        self.unique_scan_count = np.fromiter(
            (int(value).bit_count() for value in self.scan_mask),
            dtype=np.uint8,
            count=self.scan_mask.size,
        )
        prefix = np.cumsum(counts, dtype=np.uint64)
        self._offsets_full = np.r_[np.uint64(0), prefix]
        self._closed = False

    @classmethod
    def _build(
        cls,
        evidence: MultiscaleShardEvidence,
        resolution_m: float,
        block_records: int,
    ) -> "CellEvidenceLevel":
        if resolution_m not in RESOLUTIONS_M:
            raise ValueError(f"Hovi evidence resolution must be one of {RESOLUTIONS_M}")
        if not isinstance(block_records, int) or block_records <= 0:
            raise ValueError("Hovi level block_records must be positive")
        cells = round(SHARD_M / resolution_m)
        if not np.isclose(cells * resolution_m, SHARD_M, rtol=0.0, atol=1e-12):
            raise ValueError("Hovi resolution does not partition a 4 m shard")
        cell_count = cells * cells
        fixed_memory = cell_count * (8 + 2 + 8 + 8)
        block_memory = block_records * 160
        if fixed_memory + block_memory > evidence.memory_ceiling_bytes:
            raise MemoryError("Hovi level construction would exceed its memory ceiling")
        projected_temp = evidence.record_count * (
            _WORKSPACE_BYTES_PER_RECORD + _LEVEL_INDEX_BYTES_PER_RECORD
        )
        if projected_temp > evidence.temp_ceiling_bytes:
            raise MemoryError("Hovi level construction would exceed its temp ceiling")
        records = _open_records(evidence)
        counts = np.zeros(cell_count, dtype=np.uint64)
        scan_mask = np.zeros(cell_count, dtype=np.uint16)
        for start in range(0, evidence.record_count, block_records):
            stop = min(start + block_records, evidence.record_count)
            part = records[start:stop]
            cell = _cell_ids(evidence, part, resolution_m, cells)
            counts += np.bincount(cell, minlength=cell_count).astype(np.uint64)
            np.bitwise_or.at(
                scan_mask,
                cell,
                np.left_shift(np.uint16(1), part["scan_ordinal"].astype(np.uint16)),
            )
        index_path = evidence.directory / f"level-{resolution_m:g}.indices"
        descriptor = os.open(
            index_path,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        try:
            os.ftruncate(descriptor, evidence.record_count * _INDEX_DTYPE.itemsize)
        finally:
            os.close(descriptor)
        index = np.memmap(
            index_path,
            mode="r+",
            dtype=_INDEX_DTYPE,
            shape=(evidence.record_count,),
        )
        offsets = np.r_[np.uint64(0), np.cumsum(counts, dtype=np.uint64)]
        cursor = offsets[:-1].copy()
        try:
            for start in range(0, evidence.record_count, block_records):
                stop = min(start + block_records, evidence.record_count)
                part = records[start:stop]
                cell = _cell_ids(evidence, part, resolution_m, cells)
                order = np.argsort(cell, kind="stable")
                sorted_cell = cell[order]
                boundaries = np.flatnonzero(
                    np.r_[True, sorted_cell[1:] != sorted_cell[:-1]]
                )
                ends = np.r_[boundaries[1:], order.size]
                for first, last in zip(boundaries, ends, strict=True):
                    cell_id = int(sorted_cell[first])
                    destination = int(cursor[cell_id])
                    length = int(last - first)
                    index[destination : destination + length] = (
                        np.uint64(start) + order[first:last].astype(np.uint64)
                    )
                    cursor[cell_id] += np.uint64(length)
            if not np.array_equal(cursor, offsets[1:]):
                raise ValueError("Hovi level index did not conserve cell records")
            index.flush()
            os.chmod(index_path, 0o400)
        except BaseException:
            del index
            os.unlink(index_path)
            raise
        del index
        del records
        index_result = os.lstat(index_path)
        if (
            not stat.S_ISREG(index_result.st_mode)
            or stat.S_IMODE(index_result.st_mode) != 0o400
            or index_result.st_nlink != 1
        ):
            os.unlink(index_path)
            raise PermissionError("Hovi level index did not become immutable")
        return cls(
            evidence,
            resolution_m,
            cells,
            block_records,
            index_path,
            _identity(index_result),
            counts,
            scan_mask,
        )

    @property
    def occupied_cells(self) -> int:
        return int(self.cell_linear.size)

    def iter_vertical_groups(self) -> Iterator[VerticalSampleGroup]:
        self._require_open()
        for linear, count in zip(self.cell_linear, self.counts, strict=True):
            start = int(self._offsets_full[int(linear)])
            stop = start + int(count)
            yield VerticalSampleGroup(
                cell_x=int(linear % self.cells_per_shard),
                cell_y=int(linear // self.cells_per_shard),
                sample_count=int(count),
                _level=self,
                _start=start,
                _stop=stop,
            )

    def range_strata(self, edges_m: np.ndarray) -> StratumCounts:
        edges = _edges(edges_m, "range")
        if edges[0] < 0.0:
            raise ValueError("Hovi range stratum edges may not be negative")
        counts, undefined = self._strata(edges, normals=None, absolute=False)
        return StratumCounts(edges, counts, undefined)

    def incidence_strata(
        self,
        cell_normals_xyz: np.ndarray,
        edges_cosine: np.ndarray,
        *,
        absolute: bool,
    ) -> StratumCounts:
        edges = _edges(edges_cosine, "incidence cosine")
        minimum = 0.0 if absolute else -1.0
        if edges[0] < minimum or edges[-1] > 1.0:
            raise ValueError("Hovi incidence-cosine edges exceed their physical domain")
        normals = np.asarray(cell_normals_xyz, dtype=np.float64)
        if normals.shape != (self.occupied_cells, 3) or np.any(~np.isfinite(normals)):
            raise ValueError("Hovi incidence requires one finite normal per occupied cell")
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths <= 0.0):
            raise ValueError("Hovi incidence normals may not be zero length")
        normals = normals / lengths[:, None]
        counts, undefined = self._strata(edges, normals=normals, absolute=absolute)
        return StratumCounts(edges, counts, undefined)

    def continuity(self) -> ContinuityEvidence:
        self._require_open()
        cells = self.cells_per_shard * self.cells_per_shard
        keys = cells * 16
        required = keys * (3 * 8 + 3 * 2 + 6 * 8)
        if required + self.block_records * 160 > self.evidence.memory_ceiling_bytes:
            raise MemoryError("Hovi continuity accumulation would exceed its memory ceiling")
        metrics = np.zeros((keys, 6), dtype=np.uint64)
        last_source = np.full(keys, -1, dtype=np.int64)
        last_row = np.zeros(keys, dtype=np.uint16)
        last_column = np.zeros(keys, dtype=np.uint16)
        records = self._records()
        for start in range(0, self.evidence.record_count, self.block_records):
            part = records[start : start + self.block_records]
            cell = _cell_ids(
                self.evidence,
                part,
                self.resolution_m,
                self.cells_per_shard,
            )
            key = cell * 16 + part["scan_ordinal"].astype(np.int64)
            order = np.argsort(key, kind="stable")
            sorted_key = key[order]
            source = part["source_ordinal"][order].astype(np.int64)
            row = part["row"][order].astype(np.int64)
            column = part["column"][order].astype(np.int64)
            metrics[:, 0] += np.bincount(sorted_key, minlength=keys).astype(np.uint64)
            same = sorted_key[1:] == sorted_key[:-1]
            _accumulate_transitions(
                metrics,
                sorted_key[1:][same],
                source[:-1][same],
                row[:-1][same],
                column[:-1][same],
                source[1:][same],
                row[1:][same],
                column[1:][same],
            )
            first = np.flatnonzero(np.r_[True, ~same])
            first_key = sorted_key[first]
            had_previous = last_source[first_key] >= 0
            _accumulate_transitions(
                metrics,
                first_key[had_previous],
                last_source[first_key[had_previous]],
                last_row[first_key[had_previous]].astype(np.int64),
                last_column[first_key[had_previous]].astype(np.int64),
                source[first[had_previous]],
                row[first[had_previous]],
                column[first[had_previous]],
            )
            last = np.r_[first[1:] - 1, sorted_key.size - 1]
            last_key = sorted_key[last]
            last_source[last_key] = source[last]
            last_row[last_key] = row[last].astype(np.uint16)
            last_column[last_key] = column[last].astype(np.uint16)
        present = np.flatnonzero(metrics[:, 0])
        return ContinuityEvidence(
            cell_linear=(present // 16).astype(np.uint32),
            scan=(present % 16).astype(np.uint8),
            samples=metrics[present, 0],
            source_consecutive_pairs=metrics[present, 1],
            same_row_source_consecutive_pairs=metrics[present, 2],
            unit_column_source_consecutive_pairs=metrics[present, 3],
            same_column_source_consecutive_pairs=metrics[present, 4],
            unit_row_source_consecutive_pairs=metrics[present, 5],
        )

    def close(self) -> None:
        if not self._closed:
            os.unlink(self.index_path)
            self._closed = True
            self.evidence._active_level = None

    def __enter__(self) -> "CellEvidenceLevel":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _iter_vertical_batches(
        self,
        start: int,
        stop: int,
        *,
        max_records: int,
    ) -> Iterator[VerticalSampleBatch]:
        self._require_open()
        if not isinstance(max_records, int) or max_records <= 0:
            raise ValueError("Hovi vertical max_records must be positive")
        if max_records * 160 > self.evidence.memory_ceiling_bytes:
            raise MemoryError("Hovi vertical batch would exceed its memory ceiling")
        index = self._index()
        records = self._records()
        for offset in range(start, stop, max_records):
            selected = np.asarray(index[offset : min(offset + max_records, stop)])
            part = records[selected]
            file_xyz, _, _, _ = _derive_geometry(part, self.evidence.manifest)
            yield VerticalSampleBatch(
                scan=part["scan_ordinal"].copy(),
                source_ordinal=part["source_ordinal"].copy(),
                raw_z=part["raw_z"].copy(),
                file_z_m=file_xyz[:, 2],
            )
        del records
        del index

    def _strata(
        self,
        edges: np.ndarray,
        *,
        normals: np.ndarray | None,
        absolute: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._require_open()
        bins = edges.size + 1
        result_bytes = self.occupied_cells * bins * 8
        if result_bytes + self.block_records * 192 > self.evidence.memory_ceiling_bytes:
            raise MemoryError("Hovi strata accumulation would exceed its memory ceiling")
        counts = np.zeros((self.occupied_cells, bins), dtype=np.uint64)
        undefined = np.zeros(self.occupied_cells, dtype=np.uint64)
        occupied_lookup = np.full(
            self.cells_per_shard * self.cells_per_shard,
            -1,
            dtype=np.int32,
        )
        occupied_lookup[self.cell_linear] = np.arange(self.occupied_cells, dtype=np.int32)
        records = self._records()
        for start in range(0, self.evidence.record_count, self.block_records):
            part = records[start : start + self.block_records]
            file_xyz, range_m, beam, beam_valid = _derive_geometry(
                part,
                self.evidence.manifest,
            )
            cell = _cell_ids_from_xyz(
                self.evidence,
                file_xyz,
                self.resolution_m,
                self.cells_per_shard,
            )
            occupied = occupied_lookup[cell]
            if normals is None:
                value = range_m
                valid = np.ones(part.size, dtype=np.bool_)
            else:
                value = np.einsum("ij,ij->i", -beam, normals[occupied])
                if absolute:
                    value = np.abs(value)
                valid = beam_valid
            stratum = np.searchsorted(edges, value[valid], side="right")
            np.add.at(counts, (occupied[valid], stratum), 1)
            np.add.at(undefined, occupied[~valid], 1)
        del records
        return counts, undefined

    def _records(self) -> np.memmap:
        return _open_records(self.evidence)

    def _index(self) -> np.memmap:
        result = os.lstat(self.index_path)
        if (
            _identity(result) != self.index_identity
            or not stat.S_ISREG(result.st_mode)
            or stat.S_IMODE(result.st_mode) != 0o400
            or result.st_nlink != 1
        ):
            raise PermissionError("Hovi candidate level index identity drifted")
        return np.memmap(
            self.index_path,
            mode="r",
            dtype=_INDEX_DTYPE,
            shape=(self.evidence.record_count,),
        )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Hovi cell evidence level is closed")


def _derive_geometry(
    records: np.ndarray,
    manifest: VerifiedSpatialManifest,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    file_xyz = np.empty((records.size, 3), dtype=np.float64)
    range_m = np.empty(records.size, dtype=np.float64)
    beam = np.empty((records.size, 3), dtype=np.float64)
    beam_valid = np.empty(records.size, dtype=np.bool_)
    scans = records["scan_ordinal"]
    for scan in np.unique(scans):
        selected = scans == scan
        raw_xyz = np.column_stack(
            (
                records["raw_x"][selected],
                records["raw_y"][selected],
                records["raw_z"][selected],
            )
        ).astype(np.int32, copy=False)
        values = _apply_pose(raw_xyz, manifest.scans[int(scan)])
        file_xyz[selected], range_m[selected], beam[selected], beam_valid[selected] = values
    return file_xyz, range_m, beam, beam_valid


def _open_records(evidence: MultiscaleShardEvidence) -> np.memmap:
    result = os.lstat(evidence.records_path)
    if (
        _identity(result) != evidence.records_identity
        or not stat.S_ISREG(result.st_mode)
        or stat.S_IMODE(result.st_mode) != 0o400
        or result.st_nlink != 1
    ):
        raise PermissionError("Hovi candidate observation workspace identity drifted")
    return np.memmap(
        evidence.records_path,
        mode="r",
        dtype=RAW_RECORD_DTYPE,
        shape=(evidence.record_count,),
    )


def _cell_ids(
    evidence: MultiscaleShardEvidence,
    records: np.ndarray,
    resolution_m: float,
    cells: int,
) -> np.ndarray:
    file_xyz, _, _, _ = _derive_geometry(records, evidence.manifest)
    return _cell_ids_from_xyz(evidence, file_xyz, resolution_m, cells)


def _cell_ids_from_xyz(
    evidence: MultiscaleShardEvidence,
    file_xyz: np.ndarray,
    resolution_m: float,
    cells: int,
) -> np.ndarray:
    origin_x = AOI_X_MIN_M + evidence.shard_x * SHARD_M
    origin_y = AOI_Y_MIN_M + evidence.shard_y * SHARD_M
    x = np.floor((file_xyz[:, 0] - origin_x) / resolution_m).astype(np.int64)
    y = np.floor((file_xyz[:, 1] - origin_y) / resolution_m).astype(np.int64)
    if np.any(x < 0) or np.any(x >= cells) or np.any(y < 0) or np.any(y >= cells):
        raise ValueError("Hovi reconstructed observation lies outside its bound shard")
    return y * cells + x


def _accumulate_transitions(
    metrics: np.ndarray,
    key: np.ndarray,
    source0: np.ndarray,
    row0: np.ndarray,
    column0: np.ndarray,
    source1: np.ndarray,
    row1: np.ndarray,
    column1: np.ndarray,
) -> None:
    consecutive = source1 - source0 == 1
    same_row = row1 == row0
    same_column = column1 == column0
    np.add.at(metrics[:, 1], key[consecutive], 1)
    np.add.at(metrics[:, 2], key[consecutive & same_row], 1)
    np.add.at(
        metrics[:, 3],
        key[consecutive & same_row & (np.abs(column1 - column0) == 1)],
        1,
    )
    np.add.at(metrics[:, 4], key[consecutive & same_column], 1)
    np.add.at(
        metrics[:, 5],
        key[consecutive & same_column & (np.abs(row1 - row0) == 1)],
        1,
    )


def _edges(value: np.ndarray, label: str) -> np.ndarray:
    edges = np.asarray(value, dtype=np.float64)
    if (
        edges.ndim != 1
        or edges.size == 0
        or np.any(~np.isfinite(edges))
        or np.any(np.diff(edges) <= 0.0)
    ):
        raise ValueError(f"Hovi {label} edges must be finite and strictly increasing")
    return edges.copy()
