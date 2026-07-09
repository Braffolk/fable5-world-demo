"""Maa-amet geoportaal downloads: per-sheet elevation rasters + whole-country files.

URL pattern (verified live 2026-07-09):
  https://geoportaal.maaamet.ee/index.php?lang_id=1&plugin_act=otsing
      &kaardiruut=<sheet>&andmetyyp=<type>&dl=1&f=<file>&page_id=614
Sheet grids: 1:10k (5-digit) for dem_1m, 1:2000 (6-digit) for ndsm_*_1m,
1:20k (4-digit) for chm. Whole-country files use andmetyyp=mp_korgusmudelid, no sheet.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from ..config import DATA_IN
from ..sheets import GRIDS, Sheet, load_sheet_grid, sheets_for_bbox
from .http import PoliteSession

BASE = "https://geoportaal.maaamet.ee/index.php"
GRID_DOCS = "https://geoportaal.maaamet.ee/docs/pohikaart"

COUNTRY_FILES = {
    # dest subdir -> (andmetyyp, filename)
    "country": ("mp_korgusmudelid", "DTM_10m_eesti.tif"),
}


@dataclass(frozen=True)
class FetchItem:
    url: str
    dest: Path
    note: str


def sheet_url(kaardiruut: str, andmetyyp: str, filename: str, page_id: int = 614) -> str:
    kr = f"&kaardiruut={kaardiruut}" if kaardiruut else ""
    return (
        f"{BASE}?lang_id=1&plugin_act=otsing{kr}"
        f"&andmetyyp={andmetyyp}&dl=1&f={quote(filename)}&page_id={page_id}"
    )


def search_sheet_files(session: PoliteSession, kaardiruut: str, andmetyyp: str) -> list[str]:
    """List downloadable filenames for one sheet via the geoportaal search endpoint."""
    url = (
        f"{BASE}?lang_id=1&plugin_act=otsing&page_id=614"
        f"&kaardiruut={kaardiruut}&andmetyyp={andmetyyp}"
    )
    text = session.get_text(url)
    return re.findall(r"[?&]f=([^&\"']+)", html.unescape(text))


def ensure_sheet_grids(session: PoliteSession, grids_dir: Path | None = None) -> Path:
    grids_dir = grids_dir or DATA_IN / "grids"
    for zip_name, _ in GRIDS.values():
        session.download(f"{GRID_DOCS}/{zip_name}", grids_dir / zip_name)
    return grids_dir


def pick_chm(filenames: list[str]) -> str | None:
    """Newest CHM, preferring leaf-on (suvi) at equal year."""
    best: tuple[int, int, str] | None = None
    for f in filenames:
        m = re.match(r"\d+_chm_(suvi|kevad)_(\d{4})\.tif$", f)
        if m:
            key = (int(m.group(2)), 1 if m.group(1) == "suvi" else 0, f)
            if best is None or key > best:
                best = key
    return best[2] if best else None


def plan_elevation(
    session: PoliteSession,
    bbox_en: tuple[float, float, float, float] | None,
    grids_dir: Path,
    *,
    want_dem1m: bool = True,
    want_ndsm: bool = True,
    want_chm: bool = True,
) -> list[FetchItem]:
    """Build the per-sheet fetch list for an AOI bbox (None = every sheet = whole country)."""
    items: list[FetchItem] = []

    def select(scale: str) -> list[Sheet]:
        sheets = load_sheet_grid(grids_dir, scale)
        return sheets_for_bbox(sheets, bbox_en) if bbox_en else sheets

    if want_dem1m:
        for s in select("10k"):
            f = f"{s.nr}_dtm_1m.tif"
            items.append(FetchItem(sheet_url(s.nr, "dem_1m_geotiff", f), DATA_IN / "dem_1m" / f, f"DTM 1m sheet {s.nr}"))
    if want_ndsm:
        for s in select("2k"):
            f = f"{s.nr}_ndsm_1m.tif"
            items.append(FetchItem(sheet_url(s.nr, "ndsm_rel_1m_geotiff", f), DATA_IN / "ndsm_1m" / f, f"nDSM 1m sheet {s.nr}"))
    if want_chm:
        for s in select("20k"):
            names = search_sheet_files(session, s.nr, "chm_geotiff")
            f = pick_chm(names)
            if f:
                items.append(FetchItem(sheet_url(s.nr, "chm_geotiff", f), DATA_IN / "chm" / f, f"CHM sheet {s.nr}"))
    return items


def plan_country(_session: PoliteSession) -> list[FetchItem]:
    return [
        FetchItem(sheet_url("", andmetyyp, filename), DATA_IN / sub / filename, f"country {filename}")
        for sub, (andmetyyp, filename) in COUNTRY_FILES.items()
    ]


def run_fetch(session: PoliteSession, items: list[FetchItem], log=print) -> tuple[int, int]:
    got = skipped = 0
    for i, it in enumerate(items):
        _, downloaded = session.download(it.url, it.dest)
        if downloaded:
            got += 1
            log(f"[{i + 1}/{len(items)}] fetched {it.note} -> {it.dest.name} ({it.dest.stat().st_size / 1e6:.1f} MB)")
        else:
            skipped += 1
    if skipped:
        log(f"skipped {skipped} already-present files")
    return got, skipped
