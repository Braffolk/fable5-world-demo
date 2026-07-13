from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from assetgen.config import CONFIG_DIR
from assetgen.fetch import als


PRIMARY = CONFIG_DIR / "taevaskoda-als.json"
ADJACENT = CONFIG_DIR / "evidence" / "taevaskoda-ahja-als-adjacent.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_bytes())


def test_both_frozen_als_selection_schemas_normalize_strictly() -> None:
    primary = als.load_als_selection(PRIMARY)
    adjacent = als.load_als_selection(ADJACENT)

    assert primary.selection_id == "taevaskoda-als-444679-stage1"
    assert [(item.snapshot_name, item.url) for item in primary.source_indexes] == [
        ("source-index.html", primary.raw["source_index_url"])
    ]
    assert all(item.manifest_metadata == {} for item in primary.files)

    assert adjacent.selection_id == "taevaskoda-ahja-als-adjacent-2019-stage1"
    assert [item.filename for item in adjacent.files] == [
        "445679_2019_tava.laz",
        "444680_2019_tava.laz",
    ]
    assert [item.manifest_metadata["tile"]["id"] for item in adjacent.files] == [
        "445679",
        "444680",
    ]
    assert [item.snapshot_name for item in adjacent.source_indexes] == [
        "source-index-445679.html",
        "source-index-444680.html",
    ]
    assert all(item.expected_bytes is None for item in adjacent.files)
    assert all(item.expected_sha256 is None for item in adjacent.files)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda raw: raw["files"].append(copy.deepcopy(raw["files"][0])),
        lambda raw: raw["files"][0]["tile"].__setitem__("id", "444679"),
        lambda raw: raw["files"][0].__setitem__(
            "source_index_url", raw["files"][1]["source_index_url"]
        ),
        lambda raw: raw["files"][0].__setitem__(
            "canonical_url", raw["files"][1]["canonical_url"]
        ),
        lambda raw: raw["files"][0].__setitem__("bytes", 1),
        lambda raw: raw["files"][0].__setitem__("sha256", "0" * 64),
        lambda raw: raw.__setitem__("repair_collar_m", 9.0),
    ],
)
def test_adjacent_selection_rejects_any_change_to_frozen_closure(mutation) -> None:
    raw = _load_json(ADJACENT)
    mutation(raw)
    with pytest.raises(ValueError):
        als.load_als_selection(ADJACENT, content=json.dumps(raw).encode())


class _RecordedSession:
    def __init__(self, _config) -> None:
        pass

    def download_recorded(self, url: str, dest: Path, record_dest: Path, **kwargs):
        if dest.suffix == ".laz":
            payload = b"LASF" + dest.name.encode("ascii") + b"\0" * 300
        else:
            payload = (f"official snapshot for {url}\n" * 8).encode()
        digest = hashlib.sha256(payload).hexdigest()
        dest.parent.mkdir(parents=True, exist_ok=True)
        record_dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        dest.with_suffix(dest.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
        record_dest.write_text(
            json.dumps(
                {
                    "requestedUrl": url,
                    "finalUrl": url,
                    "status": 200,
                    "bytes": len(payload),
                    "sha256": digest,
                }
            ),
            encoding="utf-8",
        )
        return dest, True


def test_adjacent_retention_uses_an_independent_root_and_per_tile_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_in = tmp_path / "data" / "in"
    old_manifest = (
        data_in / "public" / "als" / "taevaskoda-als-444679-stage1" / "retained.json"
    )
    old_manifest.parent.mkdir(parents=True)
    old_bytes = b'{"existing":"primary manifest bytes must not change"}\n'
    old_manifest.write_bytes(old_bytes)
    monkeypatch.setattr(als, "DATA_IN", data_in)
    monkeypatch.setattr(als, "PoliteSession", _RecordedSession)

    output = als.fetch_als_selection(
        SimpleNamespace(fetch=object()), ADJACENT, log=lambda _message: None
    )
    retained = json.loads(output.read_bytes())

    assert old_manifest.read_bytes() == old_bytes
    assert output.parent.name == "taevaskoda-ahja-als-adjacent-2019-stage1"
    assert retained["complete"] is True
    assert retained["primarySelectionId"] == "taevaskoda-als-444679-stage1"
    assert [entry["name"] for entry in retained["sourceSnapshots"]] == [
        "source-index-445679.html",
        "source-index-444680.html",
        "license.html",
    ]
    assert [entry["tile"]["id"] for entry in retained["artifacts"]] == [
        "445679",
        "444680",
    ]
    assert retained["missingFiles"] == []
    assert len(retained["artifacts"]) == 2
