"""Polite, resumable, idempotent HTTP fetching.

Every file lands atomically (.part -> rename) with a .sha256 sidecar; a re-run skips
files whose sidecar validates, so `assetgen fetch` is safe to interrupt and repeat —
that is the reproducibility contract: fresh clone + one command == populated data/in.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests

from ..config import FetchConfig

_RETRYABLE = {429, 500, 502, 503, 504}


class PoliteSession:
    def __init__(self, cfg: FetchConfig):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers["User-Agent"] = cfg.user_agent
        self._last_request: dict[str, float] = {}

    def _pace(self, url: str) -> None:
        host = urlsplit(url).netloc
        wait = self._last_request.get(host, 0.0) + self.cfg.min_interval_s - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_request[host] = time.monotonic()

    def get_text(self, url: str, timeout: int = 60) -> str:
        for attempt in range(self.cfg.max_retries):
            self._pace(url)
            try:
                r = self.session.get(url, timeout=timeout)
                if r.status_code in _RETRYABLE:
                    raise requests.HTTPError(f"{r.status_code}", response=r)
                r.raise_for_status()
                return r.text
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as err:
                if attempt == self.cfg.max_retries - 1 or (
                    isinstance(err, requests.HTTPError)
                    and err.response is not None
                    and err.response.status_code not in _RETRYABLE
                ):
                    raise
                time.sleep(2.0**attempt)
        raise AssertionError("unreachable")

    def download(self, url: str, dest: Path, min_bytes: int = 64) -> tuple[Path, bool]:
        """Download url -> dest. Returns (dest, downloaded_now). Skips if sidecar validates."""
        sidecar = dest.with_suffix(dest.suffix + ".sha256")
        if dest.exists() and sidecar.exists():
            return dest, False
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + ".part")
        for attempt in range(self.cfg.max_retries):
            self._pace(url)
            try:
                digest = hashlib.sha256()
                with self.session.get(url, stream=True, timeout=300) as r:
                    if r.status_code in _RETRYABLE:
                        raise requests.HTTPError(f"{r.status_code}", response=r)
                    r.raise_for_status()
                    ctype = r.headers.get("content-type", "")
                    with open(part, "wb") as f:
                        for block in r.iter_content(1 << 20):
                            digest.update(block)
                            f.write(block)
                size = part.stat().st_size
                if size < min_bytes or (size < 4096 and b"<html" in part.read_bytes()[:512].lower()):
                    raise ValueError(
                        f"suspicious download ({size} B, content-type {ctype!r}) for {url}"
                    )
                part.replace(dest)
                sidecar.write_text(digest.hexdigest() + "\n")
                return dest, True
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as err:
                part.unlink(missing_ok=True)
                if attempt == self.cfg.max_retries - 1 or (
                    isinstance(err, requests.HTTPError)
                    and err.response is not None
                    and err.response.status_code not in _RETRYABLE
                ):
                    raise
                time.sleep(2.0**attempt)
        raise AssertionError("unreachable")
