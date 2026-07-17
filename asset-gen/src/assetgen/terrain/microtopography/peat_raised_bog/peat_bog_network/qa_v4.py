"""v4 QA sheets: the six v3 sheets (via qa.write_all) plus a wavelength-variation sheet
(07) that shows the frozen gradient-modulated wavelength field and the MEASURED per-block
spacing variation the mire-scale spacing gate quantifies. Research-only."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from . import network, qa


def _turbo(scalar: np.ndarray, valid: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Simple perceptual-ish blue->green->yellow->red ramp for a scalar field."""
    t = np.clip((scalar - lo) / max(hi - lo, 1.0e-9), 0.0, 1.0)
    stops = np.array(
        [[0.19, 0.07, 0.23], [0.11, 0.42, 0.69], [0.20, 0.72, 0.47],
         [0.87, 0.83, 0.22], [0.78, 0.24, 0.12]]
    )
    pos = np.linspace(0.0, 1.0, len(stops))
    rgb = np.stack([np.interp(t, pos, stops[:, c]) for c in range(3)], axis=-1)
    rgb[~valid] = (0.12, 0.14, 0.12)
    return rgb


def write_all(
    root: Path,
    *,
    whole_labels: np.ndarray,
    whole_activator_norm: np.ndarray,
    whole_mire: np.ndarray,
    whole_region: np.ndarray,
    anisotropy: np.ndarray,
    d_scale: np.ndarray,
    output_labels: np.ndarray,
    topology: network.Topology,
    output_c0: np.ndarray,
    output_c1: np.ndarray,
    relief: np.ndarray,
    deviation_1m: np.ndarray,
    authority_1m: np.ndarray,
    open_water_1m: np.ndarray,
    coupling_radius_cells: float,
    window_gates: dict,
    mire_scale: dict,
    mire_scale_fields: dict,
    process_pitch: float,
) -> list[dict]:
    root.mkdir(parents=True, exist_ok=True)
    qa.write_all(
        root,
        whole_labels=whole_labels,
        whole_activator_norm=whole_activator_norm,
        whole_mire=whole_mire,
        anisotropy=anisotropy,
        output_labels=output_labels,
        topology=topology,
        output_c0=output_c0,
        output_c1=output_c1,
        relief=relief,
        deviation_1m=deviation_1m,
        authority_1m=authority_1m,
        open_water_1m=open_water_1m,
        coupling_radius_cells=coupling_radius_cells,
        gates=window_gates,
    )

    # 07 / gradient-modulated wavelength: the frozen driver and the measured variation.
    lam_factor = np.sqrt(d_scale)  # local wavelength / flat-dome wavelength
    lam_rgb = _turbo(lam_factor, whole_mire, float(lam_factor[whole_region].min()) if whole_region.any() else 0.3, 1.0)

    block = int(mire_scale_fields["block_size"])
    block_spacing = mire_scale_fields["block_spacing"]  # (nby, nbx) mean spacing m, nan where none
    valid_blocks = np.isfinite(block_spacing)
    if valid_blocks.any():
        lo_s, hi_s = np.percentile(block_spacing[valid_blocks], [5, 95])
    else:
        lo_s, hi_s = 8.0, 24.0
    block_full = np.repeat(np.repeat(block_spacing, block, axis=0), block, axis=1)
    block_full = block_full[: whole_labels.shape[0], : whole_labels.shape[1]]
    valid_full = np.isfinite(block_full)
    block_rgb = _turbo(np.nan_to_num(block_full), valid_full, float(lo_s), float(hi_s))

    spacing_field = mire_scale_fields["spacing_field"] * process_pitch
    near = mire_scale_fields["skeleton"] | mire_scale_fields["peaks_kept"]
    show = whole_region & (spacing_field > 0)
    cont_rgb = _turbo(np.clip(spacing_field, 0, hi_s), show, float(lo_s), float(hi_s))

    gb = mire_scale["gate_b_pooled_spacing_cv"]
    ga = mire_scale["gate_a_orientation_domain_dispersion"]
    gc = mire_scale["gate_c_two_regimes_present"]
    qa._sheet(
        root / "07_wavelength_variation.png",
        "07 / Gradient-modulated wavelength (v4 mechanism) and measured spacing variation",
        f"pooled spacing CV={gb['pooled_spacing_cv']:.3f} (min {gb['pooled_spacing_cv_min']}); "
        f"R_dom={ga['R_dom']:.3f} (max {ga['R_dom_max']}); "
        f"regimes labyrinth/aligned={gc['labyrinth_blocks']}/{gc['aligned_blocks']} blocks. Research-only.",
        [
            ("frozen local wavelength factor sqrt(D_scale) (blue=short/steep, red=wide/flat)", lam_rgb),
            ("measured mean string spacing per 128 m block (m)", block_rgb),
            ("measured local spacing field 2*dist-to-skeleton (m)", cont_rgb),
            ("anisotropy strength |grad dome|/slope_ref", np.stack([anisotropy] * 3, -1)),
        ],
    )

    images = []
    for name in sorted(p.name for p in root.glob("*.png")):
        data = (root / name).read_bytes()
        images.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    return images
