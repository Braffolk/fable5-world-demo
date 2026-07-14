"""Strict sequential reader for the malformed published Evo plot-1086 LAZ."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterator

import laspy
import lazrs
import numpy as np

_EXPECTED_DIMENSIONS = (
    "X",
    "Y",
    "Z",
    "intensity",
    "return_number",
    "number_of_returns",
    "synthetic",
    "key_point",
    "withheld",
    "overlap",
    "scanner_channel",
    "scan_direction_flag",
    "edge_of_flight_line",
    "classification",
    "user_data",
    "scan_angle",
    "point_source_id",
    "gps_time",
    "reflectance",
    "h",
    "treeid",
)
_FIXED_LAZ_CHUNK_POINTS = 50_000
_FAST_BATCH_POINTS = 500_000
_EOF_GUARD_BYTES = 8 << 20


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _vlr_record(vlr: Any) -> dict[str, Any]:
    payload = vlr.record_data_bytes()
    return {
        "user_id": vlr.user_id,
        "record_id": int(vlr.record_id),
        "description": vlr.description,
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def inspect_evo_source(path: Path) -> dict[str, Any]:
    """Record the source header, including the exact defects that require recovery."""
    with laspy.open(path) as reader:
        header = reader.header
        dimensions = tuple(header.point_format.dimension_names)
        zip_vlrs = [
            vlr
            for vlr in header.vlrs
            if vlr.user_id == "laszip encoded" and int(vlr.record_id) == 22204
        ]
        if (
            str(header.version) != "1.4"
            or int(header.point_format.id) != 6
            or int(header.point_format.size) != 36
            or dimensions != _EXPECTED_DIMENSIONS
            or len(zip_vlrs) != 1
        ):
            raise ValueError("Evo plot-1086 LAS schema differs from the inspected source")
        laz_vlr = lazrs.LazVlr(zip_vlrs[0].record_data_bytes())
        if laz_vlr.chunk_size() != _FIXED_LAZ_CHUNK_POINTS:
            raise ValueError("Evo plot-1086 LAZ chunk size changed")
        if laz_vlr.uses_variable_size_chunks():
            raise ValueError("Evo plot-1086 unexpectedly uses variable-size chunks")
        if int(header.point_count) != 0:
            raise ValueError("Evo plot-1086 no longer has the frozen zero-count header defect")
        point_offset = int(header.offset_to_point_data)
        with path.open("rb") as source:
            source.seek(point_offset)
            declared_chunk_table_offset = int.from_bytes(
                source.read(8), "little", signed=True
            )
        if declared_chunk_table_offset != point_offset:
            raise ValueError("Evo plot-1086 invalid chunk-table pointer changed")
        return {
            "las_version": str(header.version),
            "point_format": int(header.point_format.id),
            "point_record_bytes": int(header.point_format.size),
            "declared_point_count": int(header.point_count),
            "declared_points_by_return": [
                int(value) for value in header.number_of_points_by_return
            ],
            "scales": [float(value) for value in header.scales],
            "offsets": [float(value) for value in header.offsets],
            "declared_min_xyz": [float(value) for value in header.mins],
            "declared_max_xyz": [float(value) for value in header.maxs],
            "point_data_offset": point_offset,
            "dimension_names": list(dimensions),
            "extra_dimensions": [
                {
                    "name": dimension.name,
                    "dtype": str(dimension.dtype),
                    "description": dimension.description,
                    "scales": (
                        None
                        if dimension.scales is None
                        else [float(value) for value in dimension.scales]
                    ),
                    "offsets": (
                        None
                        if dimension.offsets is None
                        else [float(value) for value in dimension.offsets]
                    ),
                }
                for dimension in header.point_format.extra_dimensions
            ],
            "crs": None,
            "vlrs": [_vlr_record(vlr) for vlr in header.vlrs],
            "source_defects": {
                "zero_extended_point_count": True,
                "placeholder_bounds": True,
                "chunk_table_pointer_equals_point_data_offset": True,
                "declared_chunk_table_offset": declared_chunk_table_offset,
            },
            "recovery": {
                "method": "fixed_chunk_sequential_decode_without_chunk_table",
                "source_bytes_changed": False,
                "chunk_points": _FIXED_LAZ_CHUNK_POINTS,
                "actual_count_and_bounds_recomputed_from_decoded_records": True,
            },
        }


class _SequentialSource:
    """Make lazrs take its fixed-chunk sequential fallback without changing bytes."""

    def __init__(self, source, point_offset: int):
        self.source = source
        self.point_offset = point_offset
        self.seek_attempts = 0

    def read(self, size: int = -1) -> bytes:
        return self.source.read(size)

    def readinto(self, buffer) -> int:
        return self.source.readinto(buffer)

    def seekable(self) -> bool:
        return False

    def seek(self, offset: int, whence: int = 0) -> int:
        if self.seek_attempts or whence != 0 or offset != self.point_offset:
            raise OSError("unexpected seek in Evo sequential LAZ recovery")
        self.seek_attempts += 1
        self.source.seek(self.point_offset)
        raise OSError("published Evo chunk table is unavailable")

    def tell(self) -> int:
        return self.source.tell()


def _point_record(buffer, point_format, scales, offsets, count: int):
    array = np.frombuffer(buffer, dtype=point_format.dtype(), count=count)
    return laspy.ScaleAwarePointRecord(array, point_format, scales, offsets)


def iter_evo_points(path: Path) -> Iterator[laspy.ScaleAwarePointRecord]:
    """Decode every recoverable record and stop only on EOF at a point boundary."""
    inspect_evo_source(path)
    file_bytes = path.stat().st_size
    with laspy.open(path) as reader:
        header = reader.header
        point_format = header.point_format
        scales = header.scales.copy()
        offsets = header.offsets.copy()
        point_offset = int(header.offset_to_point_data)
        zip_vlr = next(
            vlr
            for vlr in header.vlrs
            if vlr.user_id == "laszip encoded" and int(vlr.record_id) == 22204
        )
        record_data = zip_vlr.record_data_bytes()
    record_bytes = int(point_format.size)

    with path.open("rb") as source:
        source.seek(point_offset)
        sequential = _SequentialSource(source, point_offset)
        decoder = lazrs.LasZipDecompressor(sequential, record_data)
        fast_buffer = bytearray(_FAST_BATCH_POINTS * record_bytes)
        chunk_buffer = bytearray(_FIXED_LAZ_CHUNK_POINTS * record_bytes)

        while file_bytes - source.tell() > 64 << 20:
            try:
                decoder.decompress_many(fast_buffer)
            except lazrs.LazrsError as error:
                raise ValueError("Evo LAZ failed before the guarded EOF region") from error
            yield _point_record(
                fast_buffer, point_format, scales, offsets, _FAST_BATCH_POINTS
            )

        while file_bytes - source.tell() > _EOF_GUARD_BYTES:
            try:
                decoder.decompress_many(chunk_buffer)
            except lazrs.LazrsError as error:
                raise ValueError("Evo LAZ failed before the guarded EOF region") from error
            yield _point_record(
                chunk_buffer,
                point_format,
                scales,
                offsets,
                _FIXED_LAZ_CHUNK_POINTS,
            )

        while True:
            decoded = 0
            for index in range(_FIXED_LAZ_CHUNK_POINTS):
                record = memoryview(chunk_buffer)[
                    index * record_bytes : (index + 1) * record_bytes
                ]
                try:
                    decoder.decompress_many(record)
                except lazrs.LazrsError as error:
                    if (
                        source.tell() != file_bytes
                        or "failed to fill whole buffer" not in str(error).lower()
                    ):
                        raise ValueError("Evo LAZ sequential recovery ended before EOF") from error
                    if decoded:
                        yield _point_record(
                            chunk_buffer,
                            point_format,
                            scales,
                            offsets,
                            decoded,
                        )
                    return
                decoded += 1
            yield _point_record(
                chunk_buffer,
                point_format,
                scales,
                offsets,
                decoded,
            )
