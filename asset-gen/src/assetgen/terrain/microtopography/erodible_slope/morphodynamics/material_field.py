"""World-anchored material heterogeneity with ensemble normalization."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import fft, signal

from .state import FineAuthority, MorphodynamicsConfig, allocate_array
from ..solver.model import ProcessConfig


_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


@dataclass(frozen=True)
class MaterialFields:
    eta: np.ndarray
    detachment_kg_m2_s_pa: np.ndarray
    critical_shear_pa: np.ndarray
    bulk_density_kg_m3: np.ndarray
    runoff_fraction: np.ndarray
    seep_flux_m_s: np.ndarray
    maximum_erodible_depth_m: np.ndarray
    repose_gradient: np.ndarray


def _splitmix64(value: np.ndarray) -> np.ndarray:
    value = (value + np.uint64(0x9E3779B97F4A7C15)) & _MASK64
    value = ((value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _MASK64
    value = ((value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _MASK64
    return value ^ (value >> np.uint64(31))


def _world_gaussian(
    bbox_en: tuple[float, float, float, float],
    texel_m: float,
    seed: int,
    *,
    workspace: Path | None,
    name: str,
) -> np.ndarray:
    e0, n0, e1, n1 = bbox_en
    rows = int(round((n1 - n0) / texel_m))
    cols = int(round((e1 - e0) / texel_m))
    result = allocate_array(workspace, name, (rows, cols), np.float32)
    x0 = int(round(e0 / texel_m))
    y0 = int(round(n0 / texel_m))
    x = np.arange(x0, x0 + cols, dtype=np.int64).astype(np.uint64)
    block_rows = max(1, min(rows, 1_048_576 // max(cols, 1)))
    for start in range(0, rows, block_rows):
        stop = min(rows, start + block_rows)
        y = np.arange(y0 + rows - stop, y0 + rows - start, dtype=np.int64)[::-1]
        key = (
            np.uint64(seed)
            ^ (x[None, :] * np.uint64(0xD6E8FEB86659FD93))
            ^ (y.astype(np.uint64)[:, None] * np.uint64(0xA5A3564E27F8862B))
        )
        h0 = _splitmix64(key)
        h1 = _splitmix64(key ^ np.uint64(0x8CB92BA72F3D8DD7))
        u0 = ((h0 >> np.uint64(11)).astype(np.float64) + 0.5) * (2.0**-53)
        u1 = ((h1 >> np.uint64(11)).astype(np.float64) + 0.5) * (2.0**-53)
        result[start:stop] = (
            np.sqrt(-2.0 * np.log(u0)) * np.cos(2.0 * np.pi * u1)
        ).astype(np.float32)
    return result


def _matern_nu1_kernel(
    texel_m: float,
    correlation_m: float,
    radius: int,
) -> np.ndarray:
    """Return a finite SPDE kernel with exact ensemble variance one."""
    side = 2 * radius + 1
    ky = 2.0 * np.pi * fft.fftfreq(side, d=texel_m)
    kx = 2.0 * np.pi * fft.rfftfreq(side, d=texel_m)
    kappa = np.sqrt(8.0) / correlation_m
    transfer = kappa * kappa / (
        kappa * kappa + np.square(ky[:, None]) + np.square(kx[None, :])
    )
    kernel = fft.fftshift(fft.irfft2(transfer, s=(side, side), workers=1))
    variance = float(np.sum(np.square(kernel), dtype=np.float64))
    if not variance > 0.0:
        raise RuntimeError("material filter has zero theoretical variance")
    return (kernel / np.sqrt(variance)).astype(np.float32)


def _filter_tiled(
    expanded_forcing: np.ndarray,
    kernel: np.ndarray,
    output: np.ndarray,
    radius: int,
    *,
    tile_side: int = 256,
) -> None:
    """Overlap-save convolution with a bounded temporary FFT working set."""
    rows, cols = output.shape
    if expanded_forcing.shape != (rows + 2 * radius, cols + 2 * radius):
        raise ValueError("material forcing does not provide the exact filter halo")
    for row in range(0, rows, tile_side):
        row_stop = min(rows, row + tile_side)
        for col in range(0, cols, tile_side):
            col_stop = min(cols, col + tile_side)
            support = np.asarray(
                expanded_forcing[
                    row : row_stop + 2 * radius,
                    col : col_stop + 2 * radius,
                ],
                dtype=np.float32,
            )
            filtered = signal.fftconvolve(support, kernel, mode="valid")
            output[row:row_stop, col:col_stop] = filtered.astype(np.float32)


def _cell_material_rule(authority: FineAuthority) -> np.ndarray:
    rule = authority.material_rule_node
    corners = (rule[:-1, :-1], rule[1:, :-1], rule[:-1, 1:], rule[1:, 1:])
    result = corners[0].astype(np.int16, copy=True)
    homogeneous = np.logical_and.reduce(tuple(value == result for value in corners[1:]))
    owned = authority.form_active_cell | authority.hydrologic_source_active_cell
    if np.any(owned & ~homogeneous):
        raise ValueError("a physical control cell crosses a material-rule boundary")
    result[~owned] = -1
    return result


def build_material_fields(
    authority: FineAuthority,
    process: ProcessConfig,
    config: MorphodynamicsConfig,
    *,
    workspace: Path | None,
) -> MaterialFields:
    """Bind physical rules; eta changes detachment only."""
    shape = authority.cell_shape
    e0, n0, e1, n1 = authority.recipe.fine_canvas_bbox_en
    halo = config.material_halo_m
    expanded = (e0 - halo, n0 - halo, e1 + halo, n1 + halo)
    forcing = _world_gaussian(
        expanded,
        authority.recipe.fine_texel_m,
        config.material_seed,
        workspace=workspace,
        name="material-forcing-expanded",
    )
    sand_eta = allocate_array(workspace, "material-sand-expanded", forcing.shape, np.float32)
    till_eta = allocate_array(workspace, "material-till-expanded", forcing.shape, np.float32)
    margin = int(round(halo / authority.recipe.fine_texel_m))
    if margin <= 0 or not np.isclose(margin * authority.recipe.fine_texel_m, halo):
        raise ValueError("material halo is not aligned to the fine lattice")
    if halo < 4.0 * max(config.sand_correlation_m, config.till_correlation_m):
        raise ValueError("material halo must cover four correlation lengths")
    # The output arrays reuse the expanded allocation only for storage. Their
    # centered views are the bounded-convolution destinations.
    sand_crop = sand_eta[margin:-margin, margin:-margin]
    till_crop = till_eta[margin:-margin, margin:-margin]
    _filter_tiled(
        forcing,
        _matern_nu1_kernel(
            authority.recipe.fine_texel_m, config.sand_correlation_m, margin
        ),
        sand_crop,
        margin,
    )
    _filter_tiled(
        forcing,
        _matern_nu1_kernel(
            authority.recipe.fine_texel_m, config.till_correlation_m, margin
        ),
        till_crop,
        margin,
    )
    if sand_crop.shape != shape or till_crop.shape != shape:
        raise ValueError("expanded material field does not crop to the control grid")

    eta = allocate_array(workspace, "material-eta", shape, np.float32)
    fields = {
        name: allocate_array(workspace, f"material-{name}", shape, np.float32)
        for name in (
            "detachment_kg_m2_s_pa",
            "critical_shear_pa",
            "bulk_density_kg_m3",
            "runoff_fraction",
            "seep_flux_m_s",
            "maximum_erodible_depth_m",
            "repose_gradient",
        )
    }
    cell_rule = _cell_material_rule(authority)
    for index, name in enumerate(authority.material_rule_names):
        if name not in process.material_rules:
            raise ValueError(f"missing physical material rule {name}")
        selected = cell_rule == index
        form_selected = selected & authority.form_active_cell
        hydro_selected = selected & authority.hydrologic_source_active_cell
        if not np.any(form_selected | hydro_selected):
            continue
        is_sand = "sand" in name.lower()
        if not is_sand and "till" not in name.lower():
            raise ValueError(f"no heterogeneity hypothesis for material rule {name}")
        eta[form_selected] = (sand_crop if is_sand else till_crop)[form_selected]
        rule = process.material_rules[name]
        for key in (
            "detachment_kg_m2_s_pa",
            "critical_shear_pa",
            "bulk_density_kg_m3",
        ):
            fields[key][form_selected] = rule[key]
        fields["runoff_fraction"][hydro_selected] = rule["runoff_fraction"]
        fields["seep_flux_m_s"][hydro_selected] = rule["seep_flux_m_s"]
        fields["maximum_erodible_depth_m"][form_selected] = rule[
            "max_erodible_depth_m"
        ]
        fields["repose_gradient"][form_selected] = (
            config.repose_sand_gradient if is_sand else config.repose_till_gradient
        )
        sigma = config.sand_log_std if is_sand else config.till_log_std
        factor = np.clip(
            np.exp(sigma * eta[form_selected].astype(np.float64)),
            config.material_factor_min,
            config.material_factor_max,
        )
        fields["detachment_kg_m2_s_pa"][form_selected] *= factor.astype(np.float32)

    inactive = ~authority.form_active_cell
    eta[inactive] = 0.0
    for key, value in fields.items():
        if key not in ("runoff_fraction", "seep_flux_m_s"):
            value[inactive] = 0.0
    fields["runoff_fraction"][~authority.hydrologic_source_active_cell] = 0.0
    fields["seep_flux_m_s"][~authority.hydrologic_source_active_cell] = 0.0
    return MaterialFields(eta=eta, **fields)
