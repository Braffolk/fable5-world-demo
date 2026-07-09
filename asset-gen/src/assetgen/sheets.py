"""Maa-amet map-sheet grid indexes (epk2T / epk10T / epk20T shapefiles).

Sheets are axis-aligned squares in EPSG:3301, so a minimal pure-Python SHP bbox +
DBF reader is all we need — no GDAL dependency for the fetch planning step.
Grid zips are themselves fetched reproducibly (see fetch.maaamet.ensure_sheet_grids).
"""
from __future__ import annotations

import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path

# scale name -> (zip basename, layer basename inside the zip)
GRIDS = {
    "2k": ("epk2T_SHP.zip", "epk2T"),
    "10k": ("epk10T_SHP.zip", "epk10T"),
    "20k": ("epk20T_shp.zip", "epk20T"),
}


@dataclass(frozen=True)
class Sheet:
    nr: str  # sheet number as text, e.g. "54761" (1:10k) or "547611" (1:2000)
    bbox: tuple[float, float, float, float]  # (min_e, min_n, max_e, max_n)


def _read_dbf_field(dbf: bytes, name_wanted: str) -> list[str]:
    nrec = struct.unpack("<I", dbf[4:8])[0]
    hlen = struct.unpack("<H", dbf[8:10])[0]
    rlen = struct.unpack("<H", dbf[10:12])[0]
    fields, pos = [], 32
    while dbf[pos : pos + 1] != b"\r":
        fd = dbf[pos : pos + 32]
        fields.append((fd[:11].split(b"\x00")[0].decode(), fd[16]))
        pos += 32
    off = 1  # deletion flag
    for fname, flen in fields:
        if fname.upper() == name_wanted.upper():
            return [
                dbf[hlen + i * rlen + off : hlen + i * rlen + off + flen].decode("latin1").strip()
                for i in range(nrec)
            ]
        off += flen
    raise KeyError(f"DBF field {name_wanted!r} not found (have {[f for f, _ in fields]})")


def _read_shp_bboxes(shp: bytes) -> list[tuple[float, float, float, float]]:
    boxes, pos = [], 100
    while pos < len(shp):
        _, clen = struct.unpack(">II", shp[pos : pos + 8])
        pos += 8
        shtype = struct.unpack("<I", shp[pos : pos + 4])[0]
        if shtype == 5:  # polygon
            boxes.append(struct.unpack("<4d", shp[pos + 4 : pos + 36]))
        else:  # null shape etc. keeps record alignment with the DBF
            boxes.append((0.0, 0.0, 0.0, 0.0))
        pos += clen * 2
    return boxes


def load_sheet_grid(grids_dir: Path, scale: str) -> list[Sheet]:
    zip_name, base = GRIDS[scale]
    with zipfile.ZipFile(grids_dir / zip_name) as zf:
        names = {Path(n).name.lower(): n for n in zf.namelist()}
        shp = zf.read(names[f"{base.lower()}.shp"])
        dbf = zf.read(names[f"{base.lower()}.dbf"])
    # epk20T formats numbers as "54.57" while the download endpoint wants "5457"
    nrs = [nr.replace(".", "") for nr in _read_dbf_field(dbf, "NR")]
    boxes = _read_shp_bboxes(shp)
    if len(nrs) != len(boxes):
        raise ValueError(f"{scale}: DBF records {len(nrs)} != SHP records {len(boxes)}")
    return [Sheet(nr, box) for nr, box in zip(nrs, boxes) if nr]


def sheets_for_bbox(sheets: list[Sheet], bbox_en: tuple[float, float, float, float]) -> list[Sheet]:
    min_e, min_n, max_e, max_n = bbox_en
    return [
        s
        for s in sheets
        if not (s.bbox[2] <= min_e or s.bbox[0] >= max_e or s.bbox[3] <= min_n or s.bbox[1] >= max_n)
    ]


def grid_union_bbox(sheets: list[Sheet]) -> tuple[float, float, float, float]:
    return (
        min(s.bbox[0] for s in sheets),
        min(s.bbox[1] for s in sheets),
        max(s.bbox[2] for s in sheets),
        max(s.bbox[3] for s in sheets),
    )
