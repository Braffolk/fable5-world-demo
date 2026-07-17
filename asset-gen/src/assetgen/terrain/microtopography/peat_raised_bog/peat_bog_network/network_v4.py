"""v4 gradient-modulated-wavelength solve (sibling of network.py; v3 code untouched).

v3.0 used spatially CONSTANT diffusion coefficients, so the anisotropic activator-
inhibitor (Gray-Scott) Turing system locked a single characteristic wavelength mire-wide
(spacing CV ~0.20). v4 makes the reaction-diffusion LENGTH SCALE spatially heterogeneous:
the diffusion coefficients are multiplied by a bounded monotone field D_scale(x) of the
local dome gradient |grad dome| the solver already computes for anisotropy. Turing
wavelength is proportional to sqrt(D), so where the dome is steeper (D_scale smaller) the
local wavelength shortens, and on the flat central dome (D_scale ~ 1) it stays wide. This
spreads the wavelength spectrum from the mire's real gradient heterogeneity.

Direction and magnitude of the wavelength-gradient relation are frozen in
bundle-preregistration-v4.json from the peatland-patterning literature (Korotkov/Ilyasov
2026 Drones 10(2):121; Couwenberg & Joosten 2005 J Ecol 93:1238; Rietkerk 2004 / Eppinga
2009 Am Nat): strings orient perpendicular to flow and spacing decreases with slope.

Everything else (kinetics, base params, PRF nucleation, typed-form classification,
skeleton/topology, determinism) is IMPORTED unchanged from network.py. Deterministic
float64; Neumann (zero-flux) boundary; no periodicity.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .hydrology import ProcessHydrology
from .network import SolveResult  # reuse the v3 result container unchanged


def compute_d_scale(
    slope_mag: np.ndarray,
    authority: np.ndarray,
    *,
    d_min: float,
    k: float,
) -> np.ndarray:
    """Bounded monotone diffusion-modulation field from the local dome gradient.

    ``reference = k * median(slope_mag over the mire authority)`` is a robust, parameter-
    free anchor to this dome's own gradient distribution (computed once, deterministic).
    ``r(x) = clip(slope_mag / reference, 0, 1)`` is the relative gradient (monotone up),
    and ``D_scale(x) = 1 - (1 - d_min) * r(x)`` in ``[d_min, 1]`` (monotone down). Turing
    wavelength proportional to sqrt(D_scale) => steeper -> shorter wavelength.
    """
    values = slope_mag[authority]
    if values.size == 0:
        raise ValueError("authority region is empty; cannot anchor D-modulation")
    reference = k * float(np.median(values))
    r = np.clip(slope_mag / max(reference, 1.0e-12), 0.0, 1.0)
    return 1.0 - (1.0 - d_min) * r


def solve_v4(
    hydro: ProcessHydrology,
    nucleation: np.ndarray,
    params: dict,
    d_scale: np.ndarray,
    *,
    budget: int,
    dt: float,
) -> SolveResult:
    """v3 solve with per-cell diffusion coefficients D_u/D_v scaled by ``d_scale``.

    Identical to :func:`network.solve` except ``d_u`` and ``d_v`` become the spatial
    fields ``D_u * d_scale`` and ``D_v * d_scale``; the anisotropic along-flow inhibitor
    term uses the local ``D_u`` too. Base feed/kill/D_u/D_v/anisotropy/slope_ref are the
    v3.0 constants (unchanged).
    """
    feed = float(params["feed"])
    kill = float(params["kill"])
    d_u = float(params["D_u"])
    d_v = float(params["D_v"])
    aniso = float(params["anisotropy"])
    slope_ref = float(params["slope_ref"])
    seed_quantile = float(params["seed_quantile"])

    if d_scale.shape != nucleation.shape:
        raise ValueError("d_scale must match the process grid shape")
    d_u_field = d_u * d_scale
    d_v_field = d_v * d_scale

    strength = np.clip(hydro.slope_mag / slope_ref, 0.0, 1.0)
    flow_row = hydro.flow_row
    flow_col = hydro.flow_col

    inhibitor = np.ones_like(nucleation)
    activator = np.zeros_like(nucleation)
    seed = nucleation >= seed_quantile
    activator[seed] = 0.25
    inhibitor[seed] = 0.5

    trace = np.empty(budget, dtype=np.float64)
    residual = np.inf
    for step in range(budget):
        lap_u = ndimage.laplace(inhibitor, mode="nearest")
        lap_v = ndimage.laplace(activator, mode="nearest")
        # Along-flow directional second derivative of the inhibitor (anisotropic
        # water/solute spreading). pad edge = Neumann zero-flux boundary.
        pad = np.pad(inhibitor, 1, mode="edge")
        u_rr = pad[2:, 1:-1] - 2.0 * inhibitor + pad[:-2, 1:-1]
        u_cc = pad[1:-1, 2:] - 2.0 * inhibitor + pad[1:-1, :-2]
        u_rc = 0.25 * (pad[2:, 2:] - pad[2:, :-2] - pad[:-2, 2:] + pad[:-2, :-2])
        dir2 = flow_row * flow_row * u_rr + flow_col * flow_col * u_cc + 2.0 * flow_row * flow_col * u_rc

        uvv = inhibitor * activator * activator
        du = d_u_field * lap_u + aniso * strength * d_u_field * dir2 - uvv + feed * (1.0 - inhibitor)
        dv = d_v_field * lap_v + uvv - (feed + kill) * activator
        inhibitor = np.clip(inhibitor + dt * du, 0.0, None)
        activator = np.clip(activator + dt * dv, 0.0, None)
        residual = float(np.max(np.abs(dv)))
        trace[step] = residual

    return SolveResult(
        activator=activator,
        inhibitor=inhibitor,
        anisotropy_strength=strength,
        residual_trace=trace,
        final_residual=residual,
        iterations=budget,
    )
