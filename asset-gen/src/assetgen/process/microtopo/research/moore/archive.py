"""Safe archive inventory and exact Moore grid extraction."""
from __future__ import annotations

import hashlib
import os
import shutil
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.io import loadmat

from .contracts import FrozenDesign


@dataclass(frozen=True)
class SourceGrid:
    plot_id: str
    group_id: str
    archive_member: str
    archive_member_sha256: str
    z_m: np.ndarray
    raw_z_sha256: str
    filtered_z_sha256: str
    x_centers_m: np.ndarray
    y_centers_m: np.ndarray
    source_boundary_origin_xy_m: tuple[float, float]
    orientation: dict[str, object]

    @property
    def finite_count(self) -> int:
        return int(np.count_nonzero(np.isfinite(self.z_m)))


def _safe_member(info: zipfile.ZipInfo) -> None:
    path = PurePosixPath(info.filename)
    if path.is_absolute() or not path.parts or ".." in path.parts or "" in path.parts:
        raise ValueError(f"unsafe Moore archive member: {info.filename!r}")
    mode = info.external_attr >> 16
    if mode and stat.S_ISLNK(mode):
        raise ValueError(f"symlink forbidden in Moore archive: {info.filename!r}")
    if info.file_size < 0 or info.file_size > 64 * 1024 * 1024:
        raise ValueError(f"unexpected Moore member size: {info.filename!r}")


def _stream_extract(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo, destination: Path
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    written = 0
    with archive.open(info, "r") as source, destination.open("xb") as target:
        while block := source.read(1 << 20):
            digest.update(block)
            target.write(block)
            written += len(block)
        target.flush()
        os.fsync(target.fileno())
    if written != info.file_size:
        raise ValueError(f"truncated Moore archive member: {info.filename}")
    return digest.hexdigest()


def _grid(
    struct_value: object,
    *,
    plot_id: str,
    group_id: str,
    member: str,
    member_sha256: str,
) -> SourceGrid:
    if tuple(getattr(struct_value, "_fieldnames", ())) != ("x", "y", "z"):
        raise ValueError(f"unexpected MAT fields for {plot_id}")
    x = np.asarray(getattr(struct_value, "x"), dtype=np.float64)
    y = np.asarray(getattr(struct_value, "y"), dtype=np.float64)
    z = np.asarray(getattr(struct_value, "z"), dtype=np.float64)
    if x.ndim != 2 or x.shape != y.shape or x.shape != z.shape:
        raise ValueError(f"non-grid MAT arrays for {plot_id}")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(f"nonfinite coordinates for {plot_id}")
    if np.any(np.isinf(z)):
        raise ValueError(f"infinite source height for {plot_id}")
    if not np.allclose(x, x[:1, :], rtol=0, atol=1e-12):
        raise ValueError(f"x is not rectilinear for {plot_id}")
    if not np.allclose(y, y[:, :1], rtol=0, atol=1e-12):
        raise ValueError(f"y is not rectilinear for {plot_id}")
    x_axis = x[0].copy()
    y_axis = y[:, 0].copy()
    x_reversed = bool(x_axis[1] < x_axis[0])
    y_reversed = bool(y_axis[1] < y_axis[0])
    if x_reversed:
        x_axis = x_axis[::-1]
        z = z[:, ::-1]
    if y_reversed:
        y_axis = y_axis[::-1]
        z = z[::-1, :]
    if not np.allclose(np.diff(x_axis), 0.01, rtol=0, atol=2e-12):
        raise ValueError(f"x spacing is not 0.01 m for {plot_id}")
    if not np.allclose(np.diff(y_axis), 0.01, rtol=0, atol=2e-12):
        raise ValueError(f"y spacing is not 0.01 m for {plot_id}")
    if not np.any(np.isfinite(z)):
        raise ValueError(f"empty Moore surface: {plot_id}")
    raw_digest = hashlib.sha256(np.ascontiguousarray(z).view(np.uint8)).hexdigest()
    filtered = demsmooth3_exact(z)
    filtered_digest = hashlib.sha256(np.ascontiguousarray(filtered).view(np.uint8)).hexdigest()
    return SourceGrid(
        plot_id=plot_id,
        group_id=group_id,
        archive_member=member,
        archive_member_sha256=member_sha256,
        z_m=filtered,
        raw_z_sha256=raw_digest,
        filtered_z_sha256=filtered_digest,
        x_centers_m=x_axis,
        y_centers_m=y_axis,
        source_boundary_origin_xy_m=(float(x_axis[0] - 0.005), float(y_axis[0] - 0.005)),
        orientation={
            "source_rows": "y",
            "source_columns": "x",
            "materialized_y_ascending": True,
            "materialized_x_ascending": True,
            "source_x_reversed": x_reversed,
            "source_y_reversed": y_reversed,
            "unit": "metre",
            "spacing_m": 0.01,
        },
    )


def demsmooth3_exact(z_m: np.ndarray) -> np.ndarray:
    """Emulate the archived ``demsmooth(z,[3 3])`` without granting fill support.

    Only nearest-filled pixels entering a retained finite output's 3x3 stencil
    need values. Their nearest finite source is necessarily in that 3x3
    neighborhood, which lets us implement the frozen MATLAB column-major tie
    rule directly instead of relying on a library EDT's undocumented tie rule.
    """
    source = np.asarray(z_m, dtype=np.float64)
    original_nan = np.isnan(source)
    if np.any(original_nan):
        padded = np.pad(source, 2, mode="constant", constant_values=np.nan)
        finite = np.isfinite(padded)
        relevant = binary_dilation(finite, structure=np.ones((3, 3), dtype=np.bool_)) & ~finite
        nrows = padded.shape[0]
        for row, column in zip(*np.nonzero(relevant), strict=True):
            candidates: list[tuple[int, int, int, float]] = []
            for dy in (-1, 0, 1):
                rr = row + dy
                if rr < 0 or rr >= padded.shape[0]:
                    continue
                for dx in (-1, 0, 1):
                    cc = column + dx
                    if cc < 0 or cc >= padded.shape[1] or not finite[rr, cc]:
                        continue
                    squared_distance = dy * dy + dx * dx
                    matlab_index = rr + cc * nrows
                    candidates.append((squared_distance, matlab_index, rr, cc))
            if not candidates:
                raise AssertionError("nearest-filled convolution neighbor has no finite source")
            _, _, rr, cc = min(candidates)
            padded[row, column] = padded[rr, cc]
    else:
        padded = np.pad(source, 2, mode="edge")
    filtered = np.zeros(source.shape, dtype=np.float64)
    # Fixed row-major accumulation of the nine exact 1/9 weights.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            filtered += padded[2 + dy : 2 + dy + source.shape[0], 2 + dx : 2 + dx + source.shape[1]] / 9.0
    filtered[original_nan] = np.nan
    if np.any(np.isfinite(filtered) != ~original_nan):
        raise AssertionError("demsmooth changed the source finite-support mask")
    return filtered


def load_all_grids(
    design: FrozenDesign, extraction_root: Path
) -> tuple[list[SourceGrid], dict[str, dict[str, object]]]:
    group_members = design.group_members
    rec_to_group = {name: "red_earth_creek" for name in group_members["red_earth_creek"]}
    field_to_group = {
        field: group
        for group, members in group_members.items()
        if group != "red_earth_creek"
        for field in members
    }
    expected_rec = set(rec_to_group)
    expected_fields = set(field_to_group)
    member_ledger: dict[str, dict[str, object]] = {}
    grids: list[SourceGrid] = []
    seen_rec: set[str] = set()
    seen_fields: set[str] = set()
    with zipfile.ZipFile(design.archive_path, "r") as archive:
        infos = archive.infolist()
        expected_count = design.value["source"]["archive"]["member_count"]
        if len(infos) != expected_count or len({info.filename for info in infos}) != len(infos):
            raise ValueError("Moore archive member inventory changed")
        for info in infos:
            _safe_member(info)
            output = extraction_root / info.filename
            digest = _stream_extract(archive, info, output)
            member_ledger[info.filename] = {
                "sha256": digest,
                "bytes": info.file_size,
                "compressed_bytes": info.compress_size,
            }
            if info.filename in expected_rec:
                mat = loadmat(output, squeeze_me=True, struct_as_record=False)
                if set(mat) - {"__header__", "__version__", "__globals__", "DEM"}:
                    raise ValueError(f"unexpected variables in {info.filename}")
                plot_id = Path(info.filename).stem.removesuffix("_DEM")
                grids.append(
                    _grid(
                        mat["DEM"], plot_id=plot_id, group_id=rec_to_group[info.filename],
                        member=info.filename, member_sha256=digest,
                    )
                )
                seen_rec.add(info.filename)
            elif info.filename == "DEMs.mat":
                mat = loadmat(output, squeeze_me=True, struct_as_record=False)
                dem = mat["DEM"]
                actual_fields = tuple(dem._fieldnames)
                if len(actual_fields) != 18 or set(actual_fields) != expected_fields:
                    raise ValueError("DEMs.mat named-grid inventory changed")
                for field in actual_fields:
                    grids.append(
                        _grid(
                            getattr(dem, field), plot_id=field, group_id=field_to_group[field],
                            member=info.filename, member_sha256=digest,
                        )
                    )
                    seen_fields.add(field)
        if seen_rec != expected_rec or seen_fields != expected_fields:
            raise ValueError("Moore plot/group inventory does not match the frozen design")
    shutil.rmtree(extraction_root)
    grids.sort(
        key=lambda item: (
            list(group_members).index(item.group_id),
            group_members[item.group_id].index(
                item.archive_member if item.group_id == "red_earth_creek" else item.plot_id
            ),
        )
    )
    inventory = design.value["source"]["frozen_inventory"]
    finite_count = sum(grid.finite_count for grid in grids)
    if len(grids) != inventory["grid_count"] or finite_count != inventory["finite_cell_count"]:
        raise ValueError("Moore finite inventory identity failed")
    finite_area = finite_count * 0.01 * 0.01
    if abs(finite_area - inventory["finite_area_m2"]) > 1e-12:
        raise ValueError("Moore finite-area identity failed")
    operator = design.value["source"]["publisher_observation_operator"]["demsmooth"]
    if member_ledger[operator["archive_member"]]["sha256"] != operator["sha256"]:
        raise ValueError("archived demsmooth.m hash changed")
    return grids, member_ledger
