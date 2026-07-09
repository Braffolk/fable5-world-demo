"""Paged WFS 2.0 GetFeature downloads (Maa-amet / Keskkonnaagentuur GeoServers).

All Estonian public GeoServers cap at 5000 features per request; we page with
startIndex/count + a stable sortBy and write newline-delimited GeoJSON features.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode

from ..config import DATA_IN
from .http import PoliteSession

ENDPOINTS = {
    "metsaregister": "https://gsavalik.envir.ee/geoserver/metsaregister/ows",
    "eelis": "https://gsavalik.envir.ee/geoserver/eelis/ows",
}
PAGE = 5000


def fetch_wfs_layer(
    session: PoliteSession,
    service: str,
    type_name: str,
    bbox_en: tuple[float, float, float, float] | None,
    dest: Path | None = None,
    log=print,
) -> Path:
    """Page a WFS layer (optionally bbox-filtered, EPSG:3301) into an .ndjson file."""
    base = ENDPOINTS[service]
    safe = type_name.replace(":", "_")
    tag = "all" if bbox_en is None else "_".join(str(int(v)) for v in bbox_en)
    dest = dest or DATA_IN / "wfs" / service / f"{safe}.{tag}.ndjson"
    done_marker = dest.with_suffix(dest.suffix + ".done")
    if dest.exists() and done_marker.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    start, total = 0, 0
    with open(dest, "w") as out:
        while True:
            params = {
                "service": "WFS",
                "version": "2.0.0",
                "request": "GetFeature",
                "typeNames": type_name,
                "outputFormat": "application/json",
                "count": PAGE,
                "startIndex": start,
                "srsName": "EPSG:3301",
            }
            if bbox_en:
                params["bbox"] = ",".join(str(v) for v in bbox_en) + ",EPSG:3301"
            text = session.get_text(f"{base}?{urlencode(params)}", timeout=300)
            try:
                page = json.loads(text)
            except json.JSONDecodeError as err:
                raise RuntimeError(f"WFS {service}/{type_name} returned non-JSON: {text[:300]}") from err
            feats = page.get("features", [])
            for feat in feats:
                out.write(json.dumps(feat, separators=(",", ":")) + "\n")
            total += len(feats)
            log(f"  {type_name}: {total} features...")
            if len(feats) < PAGE:
                break
            start += PAGE
    done_marker.write_text(str(total) + "\n")
    log(f"{type_name}: {total} features -> {dest.name}")
    return dest


def list_layers(session: PoliteSession, service: str) -> list[str]:
    """Layer names from GetCapabilities (verify names at runtime, don't guess)."""
    base = ENDPOINTS[service]
    text = session.get_text(f"{base}?service=WFS&version=2.0.0&request=GetCapabilities", timeout=120)
    import re

    return re.findall(r"<Name>([^<]+)</Name>", text)
