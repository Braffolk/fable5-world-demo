"""ETAK (Estonian Topographic Database) whole-country GeoPackage download + unzip."""
from __future__ import annotations

import zipfile
from pathlib import Path

from ..config import DATA_IN
from .http import PoliteSession

ETAK_URL = (
    "https://geoportaal.maaamet.ee/index.php?lang_id=1&plugin_act=otsing"
    "&andmetyyp=ETAK&dl=1&f=ETAK_EESTI_GPKG.zip&page_id=609"
)


def fetch_etak(session: PoliteSession, log=print) -> Path:
    """Download + extract the whole-Estonia ETAK GeoPackage; returns the .gpkg path."""
    dest_dir = DATA_IN / "etak"
    zip_path = dest_dir / "ETAK_EESTI_GPKG.zip"
    existing = sorted(dest_dir.glob("*.gpkg"))
    if existing:
        return existing[0]
    log("fetching ETAK whole-country GeoPackage (large, one-time)...")
    session.download(ETAK_URL, zip_path, min_bytes=1 << 20)
    log(f"extracting {zip_path.name} ({zip_path.stat().st_size / 1e9:.2f} GB)...")
    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".gpkg")]
        if not members:
            raise FileNotFoundError(f"no .gpkg inside {zip_path}")
        zf.extractall(dest_dir, members=members)
    gpkg = sorted(dest_dir.glob("**/*.gpkg"))[0]
    log(f"ETAK ready: {gpkg}")
    return gpkg
