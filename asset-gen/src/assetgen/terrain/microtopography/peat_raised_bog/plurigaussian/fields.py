"""World-PRF stationary Gaussian random fields via spectral (circulant-embedding) synthesis.

The white-noise innovations are drawn from the shared world-PRF (``peat_bog_network.prf``,
BLAKE2b personalization ``laas-micro-prf1``) keyed on integer world lattice coordinates, so
a given world cell always receives the same innovation regardless of the crop window. The
innovations are then coloured in the Fourier domain by an analytic spectral density
(circulant embedding over the fixed work domain). The work domain is fixed and derived
deterministically from the site, so the cropped core is bit-reproducible.

Fields:
  * fine isotropic Matern-like microform field (0.5-3 m),
  * coarse anisotropic ridge-hollow patterning field (10-100 m, elongated perpendicular to
    the mire flow direction),
  * a small high-frequency roughness residual (sub-microform texture).

Independent innovation streams are obtained by a per-channel integer offset of the east
lattice key (distinct keys -> distinct BLAKE2b digests -> independent streams).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri

from ..peat_bog_network.prf import world_uniform

# Distinct east-key offsets so each channel is an independent world-PRF stream. Large primes
# keep the offset lattices from colliding with plausible real world-coordinate magnitudes.
CHANNEL_OFFSET = {"fine": 0, "coarse": 700_000_003, "rough": 1_300_000_019}
_EPS = 1.0e-12


@dataclass(frozen=True)
class WorkGrid:
    """Fixed work lattice (rows north-to-south, cols west-to-east) at a given pitch."""

    east0: float  # west edge (world E, multiple of pitch)
    north1: float  # north edge (world N, multiple of pitch)
    width: int
    height: int
    pitch: float

    @property
    def east0_idx(self) -> int:
        return int(round(self.east0 / self.pitch))

    @property
    def north1_idx(self) -> int:
        # Northmost cell-CENTER lattice index (center = north1 - 0.5*pitch).
        return int(round(self.north1 / self.pitch)) - 1


def world_white(grid: WorkGrid, channel: str) -> np.ndarray:
    """Standard-normal world-keyed white noise over the grid (rows north-to-south).

    Keyed on integer world lattice coordinates via the shared world-PRF; a per-channel
    east-key offset gives independent streams. Independent of any chunking or crop window.
    """
    offset = CHANNEL_OFFSET[channel]
    cols = grid.east0_idx + offset + np.arange(grid.width, dtype=np.int64)
    rows = grid.north1_idx - np.arange(grid.height, dtype=np.int64)
    east = np.broadcast_to(cols[None, :], (grid.height, grid.width))
    north = np.broadcast_to(rows[:, None], (grid.height, grid.width))
    uniform = world_uniform(east, north)
    return ndtri(np.clip(uniform, _EPS, 1.0 - _EPS))


def _radial_freq(height: int, width: int, pitch: float) -> tuple[np.ndarray, np.ndarray]:
    """Angular spatial-frequency component grids (rad/m) for an FFT of the given shape."""
    ky = 2.0 * np.pi * np.fft.fftfreq(height, d=pitch)
    kx = 2.0 * np.pi * np.fft.fftfreq(width, d=pitch)
    return ky[:, None] * np.ones((1, width)), kx[None, :] * np.ones((height, 1))


def _colour(white: np.ndarray, amplitude: np.ndarray) -> np.ndarray:
    """Colour white noise by a Fourier amplitude spectrum and standardize to unit variance."""
    spectrum = np.fft.fft2(white) * amplitude
    field = np.real(np.fft.ifft2(spectrum))
    field = field - field.mean()
    std = field.std()
    return field / std if std > 0 else field


def fine_field(grid: WorkGrid, white: np.ndarray, *, range_m: float, nu: float) -> np.ndarray:
    """Isotropic Matern-like microform field, unit variance.

    High-frequency radial PSD slope ~ -(2*nu + 2); ``range_m`` sets the correlation length.
    """
    ky, kx = _radial_freq(grid.height, grid.width, grid.pitch)
    k2 = kx * kx + ky * ky
    alpha2 = 2.0 * nu / (range_m * range_m)
    # Matern spectral DENSITY S(k) ~ (alpha^2 + k^2)^(-(nu+1)); amplitude = sqrt(S).
    power = (alpha2 + k2) ** (-(nu + 1.0))
    return _colour(white, np.sqrt(power))


def coarse_field(
    grid: WorkGrid, white: np.ndarray, *, major_len_m: float, minor_len_m: float, flow_angle_rad: float
) -> np.ndarray:
    """Anisotropic Gaussian patterning field, unit variance.

    The MAJOR (long-correlation) axis is oriented perpendicular to the flow direction
    (ridges/hollows elongate perpendicular to rainwater flow); the MINOR axis lies along
    flow and sets the ridge-hollow alternation scale. ``flow_angle_rad`` is atan2(flow_row,
    flow_col) of the downslope unit vector on the (row=south, col=east) grid.
    """
    ky, kx = _radial_freq(grid.height, grid.width, grid.pitch)
    # Major axis (long correlation) is perpendicular to flow. Rotate freq coords so k_major
    # is along the major axis. Flow direction unit vector (fr, fc); perpendicular = (-fc, fr).
    fr, fc = np.sin(flow_angle_rad), np.cos(flow_angle_rad)
    # k component along major axis (perpendicular to flow) and along minor axis (along flow).
    k_major = kx * (-fc) + ky * (fr)
    k_minor = kx * (fr) + ky * (fc)
    power = np.exp(-0.5 * (k_major * k_major * major_len_m * major_len_m
                           + k_minor * k_minor * minor_len_m * minor_len_m))
    return _colour(white, np.sqrt(power))


def roughness_field(grid: WorkGrid, white: np.ndarray, *, corner_m: float, slope_exp: float) -> np.ndarray:
    """High-frequency sub-microform residual, unit variance.

    A high-pass-shaped red field: negligible power below ``corner_m`` scale, red roll-off
    above. ``slope_exp`` sets the high-frequency PSD slope (~ -2*slope_exp).
    """
    ky, kx = _radial_freq(grid.height, grid.width, grid.pitch)
    k = np.sqrt(kx * kx + ky * ky)
    kc = 2.0 * np.pi / corner_m
    highpass = k / np.sqrt(k * k + kc * kc)  # 0 at DC, ->1 above corner
    with np.errstate(divide="ignore", invalid="ignore"):
        red = np.where(k > 0, k ** (-slope_exp), 0.0)
    amplitude = highpass * red
    amplitude[0, 0] = 0.0
    return _colour(white, amplitude)


def _interp_axis(a: np.ndarray, factor: int, axis: int) -> np.ndarray:
    """1-D exact band-limited (trig) periodic interpolation of a CELL-CENTERED real signal by
    an integer ``factor`` along ``axis``.

    Low samples sit at cell centers (i+0.5)/N of the period; the returned samples sit at
    (j+0.5)/M with M = factor*N. This evaluates the SAME band-limited periodic function the
    low-grid spectral synthesis defines (the continuous surface sampled finer), NOT a
    nearest/bilinear upsample. The even-N Nyquist bin is split so the interpolant stays real,
    and a half-cell Fourier phase shift lands the refined samples on the correct cell-centered
    positions (the low and refined cell centers never coincide for factor>=2).
    """
    a = np.moveaxis(a, axis, -1)
    n_lo = a.shape[-1]
    m_hi = factor * n_lo
    x = np.fft.fft(a, axis=-1)
    half = n_lo // 2
    x_hi = np.zeros(a.shape[:-1] + (m_hi,), dtype=complex)
    x_hi[..., :half] = x[..., :half]  # positive freqs 0..half-1
    if half > 1:
        x_hi[..., m_hi - (half - 1):] = x[..., half + 1:]  # negative freqs -(half-1)..-1
    if n_lo % 2 == 0:  # split the Nyquist (-N/2 == +N/2) so the interpolant is real
        x_hi[..., half] = 0.5 * x[..., half]
        x_hi[..., m_hi - half] = 0.5 * x[..., half]
    else:
        x_hi[..., half] = x[..., half]
    # Half-cell alignment: natural zero-pad gives F(j/M); we want F((j + 0.5 - 0.5*factor)/M).
    modes = np.fft.fftfreq(m_hi, d=1.0 / m_hi)  # signed integer mode per index
    delta = (0.5 - 0.5 * factor) / m_hi
    x_hi = x_hi * np.exp(1j * 2.0 * np.pi * modes * delta)
    y = np.fft.ifft(x_hi, axis=-1).real * factor
    return np.moveaxis(y, -1, axis)


def fft_interp_cellcentered(field_lo: np.ndarray, factor: int) -> np.ndarray:
    """Exact 2-D band-limited (trig) periodic interpolation of a cell-centered real field by an
    integer ``factor`` (separable: applied along each axis). Real in -> real out."""
    if factor == 1:
        return np.asarray(field_lo, dtype=np.float64)
    out = _interp_axis(np.asarray(field_lo, dtype=np.float64), factor, axis=0)
    return _interp_axis(out, factor, axis=1)
