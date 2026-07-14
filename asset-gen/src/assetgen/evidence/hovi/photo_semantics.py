"""Content-addressed human-QA contact sheets for retained Hovi photographs."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from PIL import Image, ImageDraw, ImageFont, ImageOps, __version__ as pillow_version

from ...config import DATA_IN, DATA_WORK
from .records import canonical_json_bytes, sha256_bytes, sha256_file

_CONDITION_SCHEMA = "hovi-condition-semantics-evidence/1.0.0"
_RECIPE_SCHEMA = "hovi-semantic-photo-qa-recipe/1.0.0"
_INDEX_SCHEMA = "hovi-semantic-photo-qa-index/1.0.0"
_DEVELOPMENT_PLOTS = ("HY_SPRUCE4", "HY_PINE2")
_PHOTO_KINDS = {
    "context_overview_photo",
    "semantic_transect_photo",
    "semantic_quadrat_photo",
}
_GROUPS = (
    ("overview", "01_overview_contact_sheet.png", "Overview context photographs"),
    ("transect", "02_transect_contact_sheet.png", "Forest-floor transect context"),
    ("quadrat", "03_quadrat_contact_sheet.png", "Fractional-cover quadrat photographs"),
)
_GROUP_KIND = {
    "overview": "context_overview_photo",
    "transect": "semantic_transect_photo",
    "quadrat": "semantic_quadrat_photo",
}
_PANEL_W = 760
_IMAGE_H = 500
_LABEL_H = 142
_HEADER_H = 100
_MARGIN = 22
_COLUMNS = 2


@dataclass(frozen=True)
class PhotoSource:
    file_id: str
    source_path: str
    kind: str
    byte_count: int
    sha256: str
    local_path: Path
    group: str
    position_label: str
    position_evidence: str
    sort_order: int
    date_label: str
    date_status: str
    date_source: str | None

    def recipe_identity(self) -> dict[str, Any]:
        return {
            "file_id": self.file_id,
            "path": self.source_path,
            "kind": self.kind,
            "bytes": self.byte_count,
            "sha256": self.sha256,
            "group": self.group,
            "position_label": self.position_label,
            "position_evidence": self.position_evidence,
            "date_label": self.date_label,
            "date_status": self.date_status,
            "date_source": self.date_source,
        }


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Hovi photo QA {label} must be an object")
    return value


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Hovi photo QA {label} must be a non-empty string")
    return value


def _sha256(value: Any, label: str) -> str:
    digest = _required_string(value, label)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"Hovi photo QA {label} must be a lowercase SHA-256")
    return digest


def _position(
    source_path: str, kind: str, plot_id: str
) -> tuple[str, str, int, str]:
    name = PurePosixPath(source_path).name
    escaped_plot = re.escape(plot_id)
    if kind == "context_overview_photo":
        match = re.fullmatch(rf"{escaped_plot}-corner_(NE|NW|SE|SW)\.JPG", name)
        order = {"NW": 0, "NE": 1, "SW": 2, "SE": 3}
        group = "overview"
        prefix = "corner"
    elif kind == "semantic_transect_photo":
        match = re.fullmatch(rf"{escaped_plot}-transect_from_(E|W)\.JPG", name)
        order = {"E": 0, "W": 1}
        group = "transect"
        prefix = "view from"
    elif kind == "semantic_quadrat_photo":
        match = re.fullmatch(rf"{escaped_plot}_quadrat([1-4])\.JPG", name)
        order = {str(index): index - 1 for index in range(1, 5)}
        group = "quadrat"
        prefix = "quadrat"
    else:
        raise ValueError(f"unsupported Hovi photo kind {kind!r}")
    if match is None:
        raise ValueError(f"Hovi source filename does not carry the frozen {kind} token: {name}")
    token = match.group(1)
    return group, f"{prefix} {token}", order[token], f"source filename token: {name}"


def _capture_date(raw: Any) -> tuple[str, str, str | None]:
    record = _require_mapping(raw, "photo capture_date")
    status = record.get("status")
    if status == "observed":
        return (
            _required_string(record.get("value"), "observed capture date"),
            "observed",
            _required_string(record.get("source_field"), "capture date source"),
        )
    if status == "unknown" and record.get("inference_allowed") is False:
        return "unknown (no source-assigned image date)", "unknown", None
    raise ValueError("Hovi photo capture date is neither observed nor fail-closed unknown")


def _load_sources(
    condition_path: Path, retained_path: Path | None
) -> tuple[dict[str, Any], str, str, str, tuple[PhotoSource, ...]]:
    condition_path = condition_path.resolve()
    condition_bytes = condition_path.read_bytes()
    condition_sha256 = sha256_bytes(condition_bytes)
    condition = _require_mapping(json.loads(condition_bytes), "condition evidence")
    plot = _require_mapping(condition.get("plot_identity"), "plot identity")
    qualification = _require_mapping(condition.get("qualification"), "qualification")
    semantic_policy = _require_mapping(condition.get("semantic_policy"), "semantic policy")
    retention = _require_mapping(condition.get("retention"), "retention identity")
    plot_id = plot.get("plot_id")
    if plot_id not in _DEVELOPMENT_PLOTS:
        raise ValueError("Hovi photo QA cannot inspect a sealed or unknown plot")
    plot_slug = plot_id.lower().replace("_", "-")
    if (
        condition.get("schema_version") != _CONDITION_SCHEMA
        or condition.get("evidence_kind") != "condition_and_semantics_raw_candidate"
        or qualification.get("role") != "raw_candidate"
        or qualification.get("qualification_status") != "unqualified"
        or qualification.get("target_truth") is not False
        or qualification.get("transfer_ceiling") != "none"
        or qualification.get("synthesis_authorized") is not False
        or semantic_policy.get("photo_pixels_classified") is not False
        or semantic_policy.get("tls_points_classified_as_target_surface") is not False
        or semantic_policy.get("soil_geology_hydrology_inference_performed") is not False
    ):
        raise ValueError("Hovi photo QA requires the unqualified HY_SPRUCE4 condition artifact")

    retention_id = _sha256(retention.get("retention_id"), "retention_id")
    retained_path = retained_path or (
        DATA_IN / "evidence" / "hovi" / retention_id / "retained.json"
    )
    retained_path = retained_path.resolve()
    if retained_path.parent.name != retention_id or retained_path.name != "retained.json":
        raise ValueError("Hovi photo QA retained path does not match the condition retention_id")
    retained = _require_mapping(json.loads(retained_path.read_bytes()), "retained manifest")
    if (
        retained.get("schema_version") != "hovi-retained-evidence/1.0.0"
        or retained.get("retention_id") != retention_id
        or retained.get("authorized_scope") != f"shared-and-{plot_slug}-only"
        or f"{plot_slug}-photos" not in retained.get("completed_tranches", ())
    ):
        raise ValueError("Hovi photo QA retained manifest identity or tranche changed")

    retained_rows = {
        row.get("file_id"): row
        for row in retained.get("files", ())
        if isinstance(row, Mapping)
        and row.get("plot_id") == plot_id
        and row.get("kind") in _PHOTO_KINDS
    }
    inventory = condition.get("photo_inventory")
    if not isinstance(inventory, list) or len(inventory) != 10:
        raise ValueError("Hovi condition artifact must inventory exactly ten photographs")
    root = retained_path.parent.resolve()
    sources: list[PhotoSource] = []
    for raw in inventory:
        item = _require_mapping(raw, "photo inventory item")
        file_id = _required_string(item.get("file_id"), "photo file_id")
        source_path = _required_string(item.get("path"), "photo source path")
        kind = _required_string(item.get("kind"), "photo kind")
        sha256 = _sha256(item.get("sha256"), "photo sha256")
        byte_count = item.get("bytes")
        if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count <= 0:
            raise ValueError("Hovi photo byte count must be a positive integer")
        if item.get("inspected_for_condition_inference") is not False:
            raise ValueError("Hovi photo inventory may not contain prior condition inference")
        row = _require_mapping(retained_rows.get(file_id), "retained photo row")
        if (
            row.get("status") != "verified"
            or row.get("tranche") != f"{plot_slug}-photos"
            or any(
                row.get(key) != value
                for key, value in (
                    ("path", source_path),
                    ("kind", kind),
                    ("bytes", byte_count),
                    ("sha256", sha256),
                )
            )
        ):
            raise ValueError(f"Hovi retained photo tuple changed for {file_id}")
        source = PurePosixPath(source_path)
        expected_relative = Path("files", *source.parts[1:])
        relative = Path(_required_string(row.get("relative_path"), "photo relative_path"))
        local_path = (root / relative).resolve()
        if (
            relative != expected_relative
            or relative.is_absolute()
            or not local_path.is_relative_to(root)
            or not local_path.is_file()
            or local_path.stat().st_size != byte_count
            or sha256_file(local_path) != sha256
        ):
            raise ValueError(f"Hovi retained photograph failed verification: {source_path}")
        group, position_label, sort_order, position_evidence = _position(
            source_path, kind, plot_id
        )
        date_label, date_status, date_source = _capture_date(item.get("capture_date"))
        sources.append(
            PhotoSource(
                file_id=file_id,
                source_path=source_path,
                kind=kind,
                byte_count=byte_count,
                sha256=sha256,
                local_path=local_path,
                group=group,
                position_label=position_label,
                position_evidence=position_evidence,
                sort_order=sort_order,
                date_label=date_label,
                date_status=date_status,
                date_source=date_source,
            )
        )
    if len({source.file_id for source in sources}) != 10:
        raise ValueError("Hovi photo inventory contains duplicate file IDs")
    for group, _, _ in _GROUPS:
        expected = {"overview": 4, "transect": 2, "quadrat": 4}[group]
        actual = sum(source.group == group for source in sources)
        if actual != expected:
            raise ValueError(f"Hovi {group} group contains {actual}, expected {expected}")
    return dict(condition), condition_sha256, retention_id, plot_id, tuple(sources)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def _render_sheet(
    sources: tuple[PhotoSource, ...], destination: Path, *, plot_id: str, title: str
) -> tuple[dict[str, Any], ...]:
    rows = math.ceil(len(sources) / _COLUMNS)
    panel_h = _IMAGE_H + _LABEL_H
    width = _MARGIN * 2 + _COLUMNS * _PANEL_W
    height = _HEADER_H + _MARGIN + rows * panel_h + _MARGIN
    canvas = Image.new("RGB", (width, height), (244, 241, 234))
    draw = ImageDraw.Draw(canvas)
    draw.text((_MARGIN, 16), f"{plot_id} | {title}", fill=(24, 38, 34), font=_font(27))
    draw.text(
        (_MARGIN, 54),
        "HUMAN QA ONLY: observed appearance; no invented labels, metric surface, or ground-height claim.",
        fill=(126, 45, 34),
        font=_font(18),
    )
    panels: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        column = index % _COLUMNS
        row = index // _COLUMNS
        left = _MARGIN + column * _PANEL_W
        top = _HEADER_H + _MARGIN + row * panel_h
        with Image.open(source.local_path) as opened:
            original_size = list(opened.size)
            original_mode = opened.mode
            orientation = opened.getexif().get(274)
            image = ImageOps.exif_transpose(opened).convert("RGB")
        image.thumbnail((_PANEL_W - 24, _IMAGE_H - 24), Image.Resampling.LANCZOS)
        image_left = left + (_PANEL_W - image.width) // 2
        image_top = top + (_IMAGE_H - image.height) // 2
        draw.rectangle(
            (left + 6, top + 6, left + _PANEL_W - 7, top + _IMAGE_H - 7),
            fill=(220, 218, 211),
            outline=(65, 76, 70),
            width=1,
        )
        canvas.paste(image, (image_left, image_top))
        label_top = top + _IMAGE_H + 9
        draw.text(
            (left + 12, label_top),
            f"{source.position_label} | date: {source.date_label}",
            fill=(18, 30, 27),
            font=_font(20),
        )
        draw.text(
            (left + 12, label_top + 31),
            f"file ID: {source.file_id}",
            fill=(43, 50, 47),
            font=_font(17),
        )
        draw.text(
            (left + 12, label_top + 58),
            f"source: {PurePosixPath(source.source_path).name}",
            fill=(43, 50, 47),
            font=_font(17),
        )
        draw.text(
            (left + 12, label_top + 85),
            f"SHA-256: {source.sha256[:20]}... (full digest in index)",
            fill=(70, 76, 72),
            font=_font(16),
        )
        panels.append(
            {
                "panel": index + 1,
                "file_id": source.file_id,
                "source_path": source.source_path,
                "source_sha256": source.sha256,
                "source_bytes": source.byte_count,
                "position_label": source.position_label,
                "position_evidence": source.position_evidence,
                "date_label": source.date_label,
                "date_status": source.date_status,
                "date_source": source.date_source,
                "original_dimensions_px": original_size,
                "original_mode": original_mode,
                "exif_orientation": orientation,
                "rendered_dimensions_px": list(image.size),
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)
    return tuple(panels)


def _atomic_write(path: Path, encoded: bytes) -> None:
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as target:
        target.write(encoded)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def _verify_existing(build_root: Path, build_id: str) -> Path | None:
    index_path = build_root / "qa" / "index.json"
    if not index_path.exists():
        return None
    recipe_path = build_root / "recipe.json"
    if not recipe_path.is_file() or sha256_file(recipe_path) != build_id:
        raise ValueError("existing Hovi photo QA recipe failed content-address verification")
    index = _require_mapping(json.loads(index_path.read_bytes()), "existing QA index")
    if (
        index.get("schema_version") != _INDEX_SCHEMA
        or index.get("status") != "complete"
        or index.get("build_id") != build_id
        or index.get("recipe_sha256") != build_id
    ):
        raise ValueError(f"existing Hovi photo QA index has conflicting identity: {index_path}")
    outputs = index.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != len(_GROUPS):
        raise ValueError("existing Hovi photo QA index has an invalid output inventory")
    for output in outputs:
        item = _require_mapping(output, "existing output")
        relative = Path(_required_string(item.get("path"), "existing output path"))
        path = (build_root / relative).resolve()
        if (
            relative.is_absolute()
            or not path.is_relative_to(build_root.resolve())
            or not path.is_file()
            or path.stat().st_size != item.get("bytes")
            or sha256_file(path) != item.get("sha256")
        ):
            raise ValueError(f"existing Hovi photo QA output failed verification: {relative}")
    sidecar = index_path.with_suffix(index_path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != sha256_file(index_path):
        raise ValueError("existing Hovi photo QA index sidecar failed verification")
    return index_path


def render_hovi_semantic_photo_evidence(
    condition_path: Path,
    *,
    retained_path: Path | None = None,
    work_root: Path = DATA_WORK,
) -> Path:
    """Render immutable semantic-observation contact sheets; never classify pixels."""
    condition, condition_sha256, retention_id, plot_id, sources = _load_sources(
        condition_path, retained_path
    )
    recipe = {
        "schema_version": _RECIPE_SCHEMA,
        "plot_id": plot_id,
        "condition_evidence": {
            "schema_version": condition["schema_version"],
            "sha256": condition_sha256,
        },
        "retention_id": retention_id,
        "qualification": condition["qualification"],
        "sources": [
            source.recipe_identity()
            for source in sorted(sources, key=lambda item: (item.group, item.sort_order))
        ],
        "rendering": {
            "pillow_version": pillow_version,
            "orientation": "Pillow ImageOps.exif_transpose",
            "resampling": "Lanczos contain; no crop",
            "columns": _COLUMNS,
            "panel_width_px": _PANEL_W,
            "image_height_px": _IMAGE_H,
            "label_height_px": _LABEL_H,
            "groups": [group for group, _, _ in _GROUPS],
            "position_policy": "source filename tokens only; not surveyed camera coordinates",
            "date_policy": "only source-observed dates; unknown is rendered explicitly",
        },
        "human_qa_scope": {
            "purpose": "inspect observed stable-surface semantic ambiguity and context",
            "invented_labels": False,
            "pixel_classification": False,
            "metric_registration": False,
            "ground_height_claim": False,
            "target_truth_claim": False,
            "synthesis_authorized": False,
        },
    }
    recipe_bytes = canonical_json_bytes(recipe)
    build_id = sha256_bytes(recipe_bytes)
    build_root = work_root / "microtopography" / "hovi" / "photo-semantics" / build_id
    existing = _verify_existing(build_root, build_id)
    if existing is not None:
        return existing

    qa_dir = build_root / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(build_root / "recipe.json", recipe_bytes)
    outputs: list[dict[str, Any]] = []
    for group, filename, title in _GROUPS:
        grouped = tuple(
            sorted(
                (source for source in sources if source.group == group),
                key=lambda source: source.sort_order,
            )
        )
        temporary = qa_dir / (filename + ".part")
        panels = _render_sheet(grouped, temporary, plot_id=plot_id, title=title)
        destination = qa_dir / filename
        temporary.replace(destination)
        with Image.open(destination) as rendered:
            dimensions = list(rendered.size)
        outputs.append(
            {
                "group": group,
                "path": destination.relative_to(build_root).as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "dimensions_px": dimensions,
                "panels": list(panels),
                "interpretation": (
                    "Human visual evidence only. Panel content is unclassified, not metric, "
                    "and makes no terrain-height or target-truth claim."
                ),
            }
        )
    index = {
        "schema_version": _INDEX_SCHEMA,
        "status": "complete",
        "build_id": build_id,
        "recipe_sha256": build_id,
        "plot_id": plot_id,
        "condition_evidence_sha256": condition_sha256,
        "retention_id": retention_id,
        "qualification": condition["qualification"],
        "source_photo_count": len(sources),
        "source_photos": [
            source.recipe_identity()
            for source in sorted(sources, key=lambda item: (item.group, item.sort_order))
        ],
        "outputs": outputs,
        "claims": recipe["human_qa_scope"],
    }
    index_path = qa_dir / "index.json"
    _atomic_write(index_path, canonical_json_bytes(index))
    _atomic_write(
        index_path.with_suffix(index_path.suffix + ".sha256"),
        (sha256_file(index_path) + "\n").encode("ascii"),
    )
    return index_path


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Render Hovi development semantic-photo evidence contact sheets."
    )
    parser.add_argument("--conditions", required=True, type=Path)
    parser.add_argument("--retained", type=Path, default=None)
    parser.add_argument("--work-root", type=Path, default=DATA_WORK)
    args = parser.parse_args()
    index = render_hovi_semantic_photo_evidence(
        args.conditions,
        retained_path=args.retained,
        work_root=args.work_root,
    )
    print(f"semantic photo QA index: {index}")


if __name__ == "__main__":
    _main()
