"""Selectively retain the one frozen FORWARD Marrviken DTM artifact."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import requests

from ..config import DATA_IN, load_base
from ..evidence.forward.source import (
    canonical_json,
    inspect_source_raster,
    load_forward_selection,
    sha256_file,
    verify_source_bytes,
)


def _download(url: str, destination: Path) -> None:
    config = load_base().fetch
    temporary = destination.with_name(destination.name + ".part")
    for attempt in range(1, config.max_retries + 1):
        try:
            with requests.get(
                url,
                headers={"User-Agent": config.user_agent},
                stream=True,
                timeout=(30, 180),
            ) as response:
                response.raise_for_status()
                with temporary.open("wb") as target:
                    for block in response.iter_content(8 << 20):
                        if block:
                            target.write(block)
                    target.flush()
                    os.fsync(target.fileno())
            temporary.replace(destination)
            return
        except (OSError, requests.RequestException):
            temporary.unlink(missing_ok=True)
            if attempt == config.max_retries:
                raise
            time.sleep(min(8.0, 2.0 ** (attempt - 1)))


def retain_forward_dtm(
    source: Path | None = None,
    output_root: Path | None = None,
) -> Path:
    selection = load_forward_selection()
    root = (output_root or DATA_IN / "evidence" / "forward") / selection.retention_id
    files = root / "files"
    destination = files / Path(selection.artifact["manifest_path"]).name
    manifest_path = root / "retained.json"
    if manifest_path.exists():
        retained = json.loads(manifest_path.read_bytes())
        verify_source_bytes(destination, selection)
        if (
            retained.get("status") != "complete"
            or retained.get("retention_id") != selection.retention_id
            or retained.get("selection_sha256") != selection.selection_sha256
        ):
            raise ValueError("existing FORWARD retention manifest conflicts")
        return manifest_path
    files.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verify_source_bytes(destination, selection)
    elif source is not None:
        verify_source_bytes(source, selection)
        temporary = destination.with_name(destination.name + ".part")
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as copied:
            os.fsync(copied.fileno())
        temporary.replace(destination)
    else:
        _download(str(selection.artifact["download_url"]), destination)
        verify_source_bytes(destination, selection)
    raster = inspect_source_raster(destination, selection)
    manifest = {
        "schema_version": "forward-dtm-retention/1.0.0",
        "status": "complete",
        "retention_id": selection.retention_id,
        "selection_sha256": selection.selection_sha256,
        "scope": "exact_marrviken_harvest_dtm_only",
        "artifact": {
            "relative_path": destination.relative_to(root).as_posix(),
            "bytes": destination.stat().st_size,
            "md5": selection.artifact["manifest_md5"],
            "sha256": sha256_file(destination),
            "source_manifest_path": selection.artifact["manifest_path"],
        },
        "raster": raster,
        "qualification": selection.raw["qualification"],
    }
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".part")
    temporary_manifest.write_bytes(canonical_json(manifest))
    temporary_manifest.replace(manifest_path)
    return manifest_path


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(retain_forward_dtm(args.source, args.output_root))


if __name__ == "__main__":
    _main()
