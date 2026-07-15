"""Decision-relevant PNG diagnostics for erodible-slope morphology research."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .bands import b1_b2
from .forms import FineSurface, FormPlan
from .model import ProcessConfig, SlopeDomain
from .process import ProcessResult

_PANEL = 960
_CANVAS = 1056


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def _scale(values: np.ndarray, *, symmetric: bool = False, log: bool = False) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    finite = data[np.isfinite(data)]
    if not finite.size:
        return np.zeros(data.shape, dtype=np.float64)
    if log:
        data = np.log1p(np.maximum(data, 0.0))
        finite = data[np.isfinite(data)]
    if symmetric:
        bound = max(float(np.quantile(np.abs(finite), 0.995)), 1e-12)
        return np.clip(0.5 + 0.5 * data / bound, 0.0, 1.0)
    low, high = np.quantile(finite, (0.005, 0.995))
    return np.clip((data - low) / max(float(high - low), 1e-12), 0.0, 1.0)


def _earth(values: np.ndarray, *, log: bool = False) -> Image.Image:
    t = _scale(values, log=log)
    rgb = np.empty((*t.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (34 + 180 * t).astype(np.uint8)
    rgb[..., 1] = (53 + 146 * t).astype(np.uint8)
    rgb[..., 2] = (41 + 92 * t).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def _diverging(values: np.ndarray) -> Image.Image:
    t = _scale(values, symmetric=True)
    low = t <= 0.5
    q = np.where(low, t * 2.0, (t - 0.5) * 2.0)
    rgb = np.empty((*t.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.where(low, 23 + 218 * q, 241 - 75 * q).astype(np.uint8)
    rgb[..., 1] = np.where(low, 80 + 161 * q, 241 - 117 * q).astype(np.uint8)
    rgb[..., 2] = np.where(low, 145 + 96 * q, 241 - 190 * q).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def _hillshade(height: np.ndarray, texel_m: float) -> Image.Image:
    south, east = np.gradient(np.asarray(height, dtype=np.float64), texel_m)
    nx, ny, nz = -east, -south, np.ones_like(east)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    light = np.asarray((-0.48, -0.52, 0.706), dtype=np.float64)
    shade = np.clip((nx * light[0] + ny * light[1] + nz * light[2]) / norm, 0, 1)
    gray = (20 + 235 * shade).astype(np.uint8)
    return Image.fromarray(np.repeat(gray[..., None], 3, axis=2), "RGB")


def _owner(values: np.ndarray) -> Image.Image:
    palette = np.asarray(
        ((65, 70, 63), (38, 111, 142), (157, 74, 45), (53, 141, 122), (199, 154, 67)),
        dtype=np.uint8,
    )
    return Image.fromarray(palette[np.asarray(values, dtype=np.uint8)], "RGB")


def _fit(image: Image.Image, size: tuple[int, int] = (_PANEL, _PANEL)) -> Image.Image:
    return image.resize(size, Image.Resampling.BILINEAR)


def _document(
    path: Path,
    *,
    title: str,
    interpretation: str,
    image: Image.Image,
    legend: str | None = None,
) -> None:
    canvas = Image.new("RGB", (_CANVAS, 1118), "#f3efe4")
    draw = ImageDraw.Draw(canvas)
    draw.text((48, 22), title, font=_font(28, True), fill="#172622")
    canvas.paste(_fit(image), (48, 78))
    draw.rectangle((47, 77, 1008, 1038), outline="#59665f", width=2)
    draw.text((48, 1052), interpretation, font=_font(15), fill="#354640")
    if legend:
        draw.text((48, 1081), legend, font=_font(13), fill="#70422f")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", compress_level=9)


def _four_panel(
    panels: tuple[tuple[str, Image.Image], ...],
) -> Image.Image:
    canvas = Image.new("RGB", (1024, 1024), "#eee9dc")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        row, col = divmod(index, 2)
        x, y = col * 512, row * 512
        panel = image.resize((496, 472), Image.Resampling.BILINEAR)
        canvas.paste(panel, (x + 8, y + 32))
        draw.text((x + 12, y + 6), label, font=_font(17, True), fill="#172622")
    return canvas


def _budget_image(process: ProcessResult, plan: FormPlan) -> Image.Image:
    generated = process.generated_sediment_kg
    deposited = float(np.sum(process.deposited_kg))
    exported = process.exported_sediment_kg
    values = (generated, deposited, exported)
    labels = ("detached", "deposited", "exported")
    colors = ("#8a4d31", "#b8944d", "#356c7c")
    canvas = Image.new("RGB", (1024, 1024), "#f3efe4")
    draw = ImageDraw.Draw(canvas)
    bound = max(values) if max(values) > 0 else 1.0
    for index, (label, value, color) in enumerate(zip(labels, values, colors, strict=True)):
        y = 130 + index * 150
        width = int(700 * value / bound)
        draw.text((70, y), f"{label}: {value:.6g} kg", font=_font(24, True), fill="#172622")
        draw.rectangle((70, y + 45, 70 + width, y + 92), fill=color)
    lines = (
        f"mass residual: {process.mass_balance_error_kg:.3e} kg",
        f"represented erosion relief: {plan.represented_erosion_volume_m3:.6g} m3",
        f"unrepresented erosion under caps/typing: {plan.unrepresented_erosion_volume_m3:.6g} m3",
        f"represented toe relief: {plan.represented_deposition_volume_m3:.6g} m3",
        f"unrepresented deposition under cap/mask: {plan.unrepresented_deposition_volume_m3:.6g} m3",
    )
    for index, line in enumerate(lines):
        draw.text((70, 620 + index * 54), line, font=_font(20), fill="#354640")
    return canvas


def _ground_closeup(surface: FineSurface) -> Image.Image:
    side = min(int(round(16.0 / surface.texel_m)), min(surface.residual_m.shape))
    energy = ndimage.uniform_filter(surface.residual_m**2, size=max(side, 1), mode="nearest")
    row, col = np.unravel_index(int(np.argmax(energy)), energy.shape)
    r0 = int(np.clip(row - side // 2, 0, surface.residual_m.shape[0] - side))
    c0 = int(np.clip(col - side // 2, 0, surface.residual_m.shape[1] - side))
    return _hillshade(surface.c1_height_m[r0 : r0 + side, c0 : c0 + side], surface.texel_m)


def _diagnostic_text(metrics: dict[str, Any]) -> Image.Image:
    canvas = Image.new("RGB", (1024, 1024), "#f3efe4")
    draw = ImageDraw.Draw(canvas)
    y = 70
    for key, value in metrics.items():
        draw.text((60, y), f"{key}: {value}", font=_font(21), fill="#243a34")
        y += 58
    return canvas


def render_qa(
    qa_root: Path,
    *,
    domain: SlopeDomain,
    process: ProcessResult,
    plan: FormPlan,
    surface: FineSurface,
    config: ProcessConfig,
    source_hashes: dict[str, str],
    recipe_sha256: str,
    diagnostics: dict[str, Any],
) -> list[dict[str, Any]]:
    # F4/R4 analyzes the 2048-square payload core; the final row/column are
    # shared edge samples retained by the 2049-square absolute surface.
    b1, b2 = b1_b2(surface.residual_m[:-1, :-1])
    conditioning = _four_panel(
        (
            ("slope magnitude", _earth(process.routing.slope)),
            ("localized seep likelihood", _earth(domain.seep_likelihood)),
            ("vegetation cover attenuation", _earth(domain.vegetation_cover)),
            (
                "hard/unknown exclusions",
                _earth((domain.hard_exclusion | domain.unknown).astype(np.float64)),
            ),
        )
    )
    drainage = _four_panel(
        (
            ("contributing area (log)", _earth(process.routing.contributing_area_m2, log=True)),
            ("event water flux (log)", _earth(process.routing.water_m3, log=True)),
            ("detachment kg", _earth(process.detached_kg, log=True)),
            ("deposition kg", _earth(process.deposited_kg, log=True)),
        )
    )
    bands = Image.new("RGB", (1024, 1024), "#eee9dc")
    bands.paste(_diverging(b1).resize((500, 970), Image.Resampling.BILINEAR), (0, 54))
    bands.paste(_diverging(b2).resize((500, 970), Image.Resampling.BILINEAR), (524, 54))
    draw = ImageDraw.Draw(bands)
    draw.text((12, 12), "B1 0.25-1 m (F4/R4)", font=_font(18, True), fill="#172622")
    draw.text((536, 12), "B2 0.0625-0.25 m (F4/R4)", font=_font(18, True), fill="#172622")
    definitions: tuple[tuple[str, str, str, Image.Image, str | None], ...] = (
        (
            "00_conditioning_and_masks.png",
            "Bound conditioning and abstention masks",
            "Seep is localized; unknown and hard-excluded cells own no generated form.",
            conditioning,
            None,
        ),
        (
            "01_absolute_c1_surface.png",
            "0.0625 m morphology on diagnostic carrier",
            "Visual diagnostic only: this cubic 1 m carrier is not accepted fine structural authority or a cookable master.",
            _hillshade(surface.c1_height_m, surface.texel_m),
            None,
        ),
        (
            "02_c1_minus_c0_residual.png",
            "Morphology residual over diagnostic C0",
            "Blue is incision/seep relief and brown is mass-linked deposition; no noise tail is added.",
            _diverging(surface.residual_m),
            None,
        ),
        (
            "03_drainage_flux_and_sediment.png",
            "Whole-domain drainage and sediment state",
            "Flux enters only through bound rainfall/seep/upstream conditions and leaves through real outlets.",
            drainage,
            None,
        ),
        (
            "04_typed_form_ownership.png",
            "Typed form ownership",
            "Ownership identifies recognizable causal forms rather than an untyped residual.",
            _owner(surface.ownership),
            "gray none | blue rill | rust headcut | green seep | ochre toe >=0.1 mm",
        ),
        (
            "05_conservative_mass_budget.png",
            "Sediment mass and relief representation budget",
            "Process mass closes independently; cap-limited unrepresented relief is explicit.",
            _budget_image(process, plan),
            None,
        ),
        (
            "06_normative_f4_r4_bands.png",
            "Normative F4/R4 residual bands",
            "B1 and B2 are analyses of the same absolute master, never separate synthesis layers.",
            bands,
            None,
        ),
        (
            "07_ground_closeup_16m.png",
            "Automatic 16 m ground-scale close-up",
            "The window is selected by maximum residual energy, not by manual scene tuning.",
            _ground_closeup(surface),
            None,
        ),
        (
            "08_partition_and_rotation.png",
            "Partition and rotation qualification",
            "Partition identity is exact; rotation error is measured on a qualified synthetic plane.",
            _diagnostic_text(diagnostics),
            None,
        ),
    )
    rows: list[dict[str, Any]] = []
    for filename, title, interpretation, image, legend in definitions:
        path = Path(qa_root) / filename
        _document(
            path,
            title=f"{domain.site_id}: {title}",
            interpretation=interpretation,
            image=image,
            legend=legend,
        )
        with Image.open(path) as saved:
            dimensions = [saved.width, saved.height]
        rows.append(
            {
                "path": filename,
                "sha256": _sha256_file(path),
                "dimensions_px": dimensions,
                "interpretation": interpretation,
            }
        )
    index = {
        "schema_version": "laas.erodible-slope-qa-index/1",
        "site_id": domain.site_id,
        "recipe_sha256": recipe_sha256,
        "source_hashes": dict(sorted(source_hashes.items())),
        "images": rows,
    }
    index_path = Path(qa_root) / "index.json"
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return rows
