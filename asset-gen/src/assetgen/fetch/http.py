"""Polite, resumable, idempotent HTTP fetching.

Every file lands atomically (.part -> rename) with a .sha256 sidecar; a re-run skips
files whose sidecar validates, so `assetgen fetch` is safe to interrupt and repeat —
that is the reproducibility contract: fresh clone + one command == populated data/in.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import requests

from ..config import FetchConfig

_RETRYABLE = {429, 500, 502, 503, 504}


class _RetryableIntegrityError(ValueError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_prefix(path: Path, size: int = 512) -> bytes:
    with path.open("rb") as source:
        return source.read(size)


def _validate_recorded_tuple(
    url: str,
    dest: Path,
    sidecar: Path,
    record_dest: Path,
    *,
    expected_bytes: int | None,
    expected_sha256: str | None,
    expected_magic: bytes | None,
    allowed_final_hosts: tuple[str, ...] | None,
) -> str:
    digest = _sha256_file(dest)
    size = dest.stat().st_size
    if digest != sidecar.read_text(encoding="ascii").strip():
        raise ValueError(f"retained artifact differs from its sidecar: {dest}")
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"retained artifact differs from the expected SHA-256: {dest}")
    if expected_bytes is not None and size != expected_bytes:
        raise ValueError(f"retained artifact differs from the expected byte count: {dest}")
    if expected_magic is not None and not _read_prefix(dest, len(expected_magic)).startswith(
        expected_magic
    ):
        raise ValueError(f"retained artifact has the wrong file signature: {dest}")
    record = json.loads(record_dest.read_bytes())
    final_url = urlsplit(str(record.get("finalUrl", "")))
    if (
        record.get("requestedUrl") != url
        or record.get("bytes") != size
        or record.get("sha256") != digest
        or record.get("status") != 200
        or (
            allowed_final_hosts is not None
            and (final_url.scheme != "https" or final_url.netloc not in allowed_final_hosts)
        )
    ):
        raise ValueError(f"retained artifact provenance does not bind its bytes: {dest}")
    return digest


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

    def download_recorded(
        self,
        url: str,
        dest: Path,
        record_dest: Path,
        *,
        min_bytes: int = 64,
        expected_bytes: int | None = None,
        expected_sha256: str | None = None,
        expected_magic: bytes | None = None,
        allowed_final_hosts: tuple[str, ...] | None = None,
        allow_html: bool = False,
    ) -> tuple[Path, bool]:
        """Download one evidence artifact and atomically bind its HTTP provenance."""
        sidecar = dest.with_suffix(dest.suffix + ".sha256")
        part = dest.with_suffix(dest.suffix + ".part")
        sidecar_part = sidecar.with_suffix(sidecar.suffix + ".part")
        record_part = record_dest.with_suffix(record_dest.suffix + ".part")
        if dest.exists():
            digest = _sha256_file(dest)
            if not sidecar.exists() and sidecar_part.exists():
                if sidecar_part.read_text(encoding="ascii").strip() != digest:
                    raise ValueError(f"interrupted sidecar differs from retained bytes: {dest}")
                sidecar_part.replace(sidecar)
            if not record_dest.exists() and record_part.exists() and sidecar.exists():
                _validate_recorded_tuple(
                    url,
                    dest,
                    sidecar,
                    record_part,
                    expected_bytes=expected_bytes,
                    expected_sha256=expected_sha256,
                    expected_magic=expected_magic,
                    allowed_final_hosts=allowed_final_hosts,
                )
                record_part.replace(record_dest)
        present = (dest.exists(), sidecar.exists(), record_dest.exists())
        if all(present):
            _validate_recorded_tuple(
                url,
                dest,
                sidecar,
                record_dest,
                expected_bytes=expected_bytes,
                expected_sha256=expected_sha256,
                expected_magic=expected_magic,
                allowed_final_hosts=allowed_final_hosts,
            )
            return dest, False
        if any(present):
            raise ValueError(f"incomplete retained artifact tuple; refusing overwrite: {dest}")

        dest.parent.mkdir(parents=True, exist_ok=True)
        record_dest.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(self.cfg.max_retries):
            self._pace(url)
            started = datetime.now(timezone.utc).isoformat()
            try:
                digest = hashlib.sha256()
                with self.session.get(url, stream=True, timeout=300) as response:
                    if response.status_code in _RETRYABLE:
                        raise requests.HTTPError(f"{response.status_code}", response=response)
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "")
                    with part.open("wb") as target:
                        for block in response.iter_content(1 << 20):
                            if block:
                                digest.update(block)
                                target.write(block)
                    redirects = [
                        {"url": prior.url, "status": prior.status_code}
                        for prior in response.history
                    ]
                    final_url = response.url
                    response_status = response.status_code
                    response_headers = {
                        key: response.headers.get(key)
                        for key in (
                            "Content-Type",
                            "Content-Length",
                            "Content-Disposition",
                            "ETag",
                            "Last-Modified",
                            "Accept-Ranges",
                            "Content-Encoding",
                        )
                    }
                size = part.stat().st_size
                head = _read_prefix(part)
                if response_status != 200 or size < min_bytes or (
                    not allow_html
                    and (
                        b"<html" in head.lower()
                        or "text/html" in content_type.lower()
                    )
                ):
                    raise _RetryableIntegrityError(
                        f"suspicious download ({size} B, content-type {content_type!r}) for {url}"
                    )
                final = urlsplit(final_url)
                if allowed_final_hosts is not None and (
                    final.scheme != "https" or final.netloc not in allowed_final_hosts
                ):
                    raise _RetryableIntegrityError(
                        f"download redirected outside the allowed official hosts: {final_url}"
                    )
                if expected_magic is not None and not head.startswith(expected_magic):
                    raise _RetryableIntegrityError(
                        f"download has the wrong file signature for {url}"
                    )
                content_length = response_headers.get("Content-Length")
                content_encoding = response_headers.get("Content-Encoding")
                if (
                    content_length is not None
                    and content_encoding in (None, "", "identity")
                    and size != int(content_length)
                ):
                    raise _RetryableIntegrityError(
                        f"downloaded byte count {size} != HTTP Content-Length "
                        f"{content_length} for {url}"
                    )
                actual_sha256 = digest.hexdigest()
                if expected_bytes is not None and size != expected_bytes:
                    raise _RetryableIntegrityError(
                        f"downloaded byte count {size} != expected {expected_bytes} for {url}"
                    )
                if expected_sha256 is not None and actual_sha256 != expected_sha256:
                    raise _RetryableIntegrityError(
                        f"downloaded SHA-256 {actual_sha256} != expected {expected_sha256} for {url}"
                    )
                record = {
                    "format": 1,
                    "requestedUrl": url,
                    "redirects": redirects,
                    "finalUrl": final_url,
                    "fetchStartedUtc": started,
                    "fetchCompletedUtc": datetime.now(timezone.utc).isoformat(),
                    "status": response_status,
                    "headers": response_headers,
                    "bytes": size,
                    "sha256": actual_sha256,
                }
                record_part.write_text(
                    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                sidecar_part.write_text(actual_sha256 + "\n", encoding="ascii")
                part.replace(dest)
                sidecar_part.replace(sidecar)
                record_part.replace(record_dest)
                return dest, True
            except (requests.RequestException, _RetryableIntegrityError) as err:
                part.unlink(missing_ok=True)
                sidecar_part.unlink(missing_ok=True)
                record_part.unlink(missing_ok=True)
                if attempt == self.cfg.max_retries - 1 or (
                    isinstance(err, requests.HTTPError)
                    and err.response is not None
                    and err.response.status_code not in _RETRYABLE
                ):
                    raise
                time.sleep(2.0**attempt)
            except BaseException:
                part.unlink(missing_ok=True)
                sidecar_part.unlink(missing_ok=True)
                record_part.unlink(missing_ok=True)
                raise
        raise AssertionError("unreachable")
