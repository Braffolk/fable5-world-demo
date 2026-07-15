"""Retain and freeze official EGT 1:200k surficial coded domains."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from ..config import DATA_IN, load_base
from ..terrain.conditions.geology.domains import (
    DOMAIN_SPECS,
    SERVICE_ITEM_URL,
    build_egt_surficial_domain_snapshot,
    canonical_json_bytes,
)

_ALLOWED_HOST = "gis.egt.ee"
_RETRYABLE = {429, 500, 502, 503, 504}
_SOURCE_ROOT = DATA_IN / "egt" / "arcgis" / "Q_200_WM_MIL1"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable EGT source artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _fetch_json(session: requests.Session, url: str) -> tuple[bytes, dict[str, Any]]:
    cfg = load_base().fetch
    for attempt in range(cfg.max_retries):
        started = datetime.now(timezone.utc).isoformat()
        try:
            response = session.get(url, timeout=60)
            if response.status_code in _RETRYABLE:
                raise requests.HTTPError(str(response.status_code), response=response)
            response.raise_for_status()
            final = urlsplit(response.url)
            if final.scheme != "https" or final.netloc != _ALLOWED_HOST:
                raise ValueError(f"EGT response left the official host: {response.url}")
            if response.url != url:
                raise ValueError(f"EGT response URL changed: {response.url}")
            payload = response.content
            if len(payload) < 64 or payload.lstrip().startswith(b"<"):
                raise ValueError(f"EGT response is not plausible JSON: {url}")
            parsed = json.loads(payload)
            if not isinstance(parsed, dict):
                raise ValueError(f"EGT response is not a JSON object: {url}")
            metadata = {
                "requested_url": url,
                "final_url": response.url,
                "status": response.status_code,
                "retrieved_at": started,
                "response_headers": {
                    key: response.headers.get(key)
                    for key in (
                        "Content-Type",
                        "Content-Length",
                        "Content-Encoding",
                        "ETag",
                        "Last-Modified",
                    )
                },
                "bytes": len(payload),
                "exact_json_sha256": _sha256(payload),
            }
            return payload, metadata
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
            if attempt == cfg.max_retries - 1:
                raise
            time.sleep(2.0**attempt)
    raise AssertionError("unreachable")


def _retain_source(
    *, label: str, payload: bytes, retrieval: dict[str, Any]
) -> dict[str, Any]:
    digest = _sha256(payload)
    root = _SOURCE_ROOT / "raw" / "sha256" / digest
    body_path = root / f"{label}.json"
    retrieval_path = root / "retrieval.json"
    _write_immutable(body_path, payload)
    if retrieval_path.exists():
        frozen_retrieval = json.loads(retrieval_path.read_bytes())
        stable_keys = (
            "requested_url",
            "final_url",
            "status",
            "bytes",
            "exact_json_sha256",
        )
        if any(frozen_retrieval.get(key) != retrieval.get(key) for key in stable_keys):
            raise ValueError(f"retained EGT retrieval no longer binds {body_path}")
    else:
        _write_immutable(retrieval_path, canonical_json_bytes(retrieval))
    retrieval_sha = _sha256(retrieval_path.read_bytes())
    return {
        "requested_url": retrieval["requested_url"],
        "final_url": retrieval["final_url"],
        "status": retrieval["status"],
        "bytes": len(payload),
        "exact_json_sha256": digest,
        "retained_json_path": body_path.relative_to(DATA_IN.parent.parent).as_posix(),
        "retrieval_record_path": retrieval_path.relative_to(
            DATA_IN.parent.parent
        ).as_posix(),
        "retrieval_record_sha256": retrieval_sha,
    }


def fetch_egt_surficial_domains() -> Path:
    cfg = load_base().fetch
    session = requests.Session()
    session.headers["User-Agent"] = cfg.user_agent
    documents: list[dict[str, Any]] = []
    item_payload, item_retrieval = _fetch_json(session, SERVICE_ITEM_URL)
    documents.append(
        _retain_source(
            label="service-item", payload=item_payload, retrieval=item_retrieval
        )
    )
    layers: dict[int, bytes] = {}
    for spec in DOMAIN_SPECS:
        if cfg.min_interval_s > 0:
            time.sleep(cfg.min_interval_s)
        payload, retrieval = _fetch_json(session, spec.url)
        layers[spec.layer_id] = payload
        documents.append(
            _retain_source(
                label=f"layer-{spec.layer_id}", payload=payload, retrieval=retrieval
            )
        )
    return build_egt_surficial_domain_snapshot(
        service_item_json=item_payload,
        layer_json=layers,
        source_documents=documents,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(fetch_egt_surficial_domains())


if __name__ == "__main__":
    main()
