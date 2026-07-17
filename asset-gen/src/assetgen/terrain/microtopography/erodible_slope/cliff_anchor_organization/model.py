"""Measured-anchor + transferred-organization escarpment reconstruction.

Development A only (ETAK escarpment 1826743, unconsolidated sand/till bank).

The measured accepted ALS/TGV float (`a5d101c1...`) supplies WHERE and HOW MUCH
(the anchored macro deviation and its local envelope). The Biala Gora source
capacity screen (`9d27b52d...`) supplies HOW FORMS CONNECT (the connective
grammar: continuous crest-face-toe break, asymmetric bench, slump sockets and
aprons at concavities, branching chute/gully seeds at flow convergence, bounded
band-limited fine relief). No Biala amplitude is transferred: every synthesized
deviation is bounded by the LOCAL anchored ALS envelope and suppressed at
qualified measurements so withheld reality is honored exactly.

Deterministic float64; world-PRF fine seeding; a single fixed-budget pass.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import shapely
from scipy import ndimage

from ..morphodynamics.structural_base import FINE_CANVAS_BBOX_EN

MASK64 = (1 << 64) - 1


# --------------------------------------------------------------------------- #
# deterministic world-PRF value noise                                          #
# --------------------------------------------------------------------------- #
def _splitmix64(keys: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = (keys.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(MASK64)
        v = ((v ^ (v >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(MASK64)
        v = ((v ^ (v >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & np.uint64(MASK64)
        v = v ^ (v >> np.uint64(31))
    return v


def _world_white(shape: tuple[int, int], bbox, pitch: float, seed: int) -> np.ndarray:
    """Uniform white noise keyed by integer world coordinate — crop invariant."""
    rows, cols = shape
    ei = np.rint(bbox[0] / pitch).astype(np.int64) + np.arange(cols, dtype=np.int64)
    ni = np.rint(bbox[3] / pitch).astype(np.int64) - np.arange(rows, dtype=np.int64)
    ee, nn = np.meshgrid(ei, ni)
    key = (
        (ee.astype(np.uint64) + np.uint64(1 << 32)) * np.uint64(0xD6E8FEB86659FD93)
        ^ (nn.astype(np.uint64) + np.uint64(1 << 32)) * np.uint64(0xA5A3564E27F8862F)
        ^ np.uint64(seed & MASK64)
    )
    return _splitmix64(key).astype(np.float64) / float(1 << 64) - 0.5


def _smoothstep(x: np.ndarray, a: float, b: float) -> np.ndarray:
    t = np.clip((x - a) / max(b - a, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# --------------------------------------------------------------------------- #
# geometry: station frame, per-cell station / signed-normal / distance          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LineFrame:
    line: Any
    length_m: float
    station_m: np.ndarray          # (nS,)
    sx: np.ndarray
    sy: np.ndarray
    nx: np.ndarray                 # +n points to the higher (crest) side
    ny: np.ndarray
    plan_curv: np.ndarray          # signed plan curvature (concave-to-lower > 0)


def build_line_frame(line, c0: np.ndarray, bbox, pitch: float, step_m: float) -> LineFrame:
    L = float(line.length)
    S = np.arange(0.0, L + 1e-6, step_m)
    sp = shapely.line_interpolate_point(line, S)
    sb = shapely.line_interpolate_point(line, np.maximum(S - 0.5, 0.0))
    sa = shapely.line_interpolate_point(line, np.minimum(S + 0.5, L))
    tx = shapely.get_x(sa) - shapely.get_x(sb)
    ty = shapely.get_y(sa) - shapely.get_y(sb)
    tl = np.hypot(tx, ty) + 1e-9
    nx, ny = -ty / tl, tx / tl
    sx, sy = shapely.get_x(sp), shapely.get_y(sp)

    def sample(px, py):
        return ndimage.map_coordinates(
            c0, [(bbox[3] - py) / pitch, (px - bbox[0]) / pitch], order=1, mode="nearest"
        )

    hi = sample(sx + 8.0 * nx, sy + 8.0 * ny)
    lo = sample(sx - 8.0 * nx, sy - 8.0 * ny)
    if float(np.median(hi - lo)) < 0.0:
        nx, ny = -nx, -ny
    # signed plan curvature via finite differences of the unit tangent
    ux, uy = tx / tl, ty / tl
    dux = np.gradient(ux, step_m)
    duy = np.gradient(uy, step_m)
    plan_curv = dux * nx + duy * ny  # >0 : the line bends toward the lower side
    plan_curv = ndimage.uniform_filter1d(plan_curv, 7)
    return LineFrame(line, L, S, sx, sy, nx, ny, plan_curv)


@dataclass(frozen=True)
class CorridorGeometry:
    ridx: np.ndarray               # flat indices of corridor cells
    station: np.ndarray            # station (m) per corridor cell
    station_index: np.ndarray      # nearest frame station index
    signed_n: np.ndarray           # signed normal distance (m), +crest side
    distance: np.ndarray           # unsigned distance to line (m)
    dist_to_als: np.ndarray        # distance to nearest qualified ALS return (m)


def build_corridor(
    frame: LineFrame, active, bbox, pitch: float, half_width_m: float,
    px: np.ndarray, py: np.ndarray, shape: tuple[int, int],
) -> CorridorGeometry:
    H, W = shape
    xs = bbox[0] + np.arange(W) * pitch
    ys = bbox[3] - np.arange(H) * pitch
    XX, YY = np.meshgrid(xs, ys)
    ridx = np.flatnonzero(active.ravel())
    pts = shapely.points(XX.ravel()[ridx], YY.ravel()[ridx])
    st = shapely.line_locate_point(frame.line, pts)
    proj = shapely.line_interpolate_point(frame.line, st)
    si = np.clip(np.rint(st / (frame.station_m[1] - frame.station_m[0])).astype(np.int64), 0, len(frame.station_m) - 1)
    nx, ny = frame.nx[si], frame.ny[si]
    dx = XX.ravel()[ridx] - shapely.get_x(proj)
    dy = YY.ravel()[ridx] - shapely.get_y(proj)
    signed = dx * nx + dy * ny
    dist = shapely.distance(pts, frame.line)

    occ = np.zeros((H, W), dtype=bool)
    rr = np.clip(((bbox[3] - py) / pitch).astype(np.int64), 0, H - 1)
    cc = np.clip(((px - bbox[0]) / pitch).astype(np.int64), 0, W - 1)
    occ[rr, cc] = True
    dals = ndimage.distance_transform_edt(~occ) * pitch
    return CorridorGeometry(ridx, st, si, np.asarray(signed), np.asarray(dist), dals.ravel()[ridx])


# --------------------------------------------------------------------------- #
# profile-state model fit from the complete corrected base C0                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProfileState:
    crest: np.ndarray
    toe: np.ndarray
    center: np.ndarray
    width: np.ndarray
    face_h: np.ndarray
    normal_axis: np.ndarray
    profiles: np.ndarray           # (nS, nAxis) sampled C0 cross-profiles


def fit_profile_state(frame: LineFrame, c0, bbox, pitch: float, extent_m: float, step_m: float) -> ProfileState:
    ncent = np.arange(-extent_m, extent_m + 1e-6, step_m)
    nS = len(frame.station_m)

    def sample(px, py):
        return ndimage.map_coordinates(
            c0, [(bbox[3] - py) / pitch, (px - bbox[0]) / pitch], order=1, mode="nearest"
        )

    prof = np.stack(
        [sample(frame.sx + n * frame.nx, frame.sy + n * frame.ny) for n in ncent], axis=1
    )
    crest = np.median(prof[:, ncent >= extent_m - 10.0], axis=1)
    toe = np.median(prof[:, ncent <= -(extent_m - 10.0)], axis=1)
    face_h = crest - toe
    grad = np.gradient(prof, axis=1)
    center = ndimage.uniform_filter1d(ncent[np.argmax(np.abs(grad), axis=1)].astype(np.float64), 9)
    width = np.full(nS, 4.0)
    for s in range(nS):
        rng = face_h[s]
        if abs(rng) > 1.5:
            f = (prof[s] - toe[s]) / rng
            order = np.argsort(f)
            lo = np.interp(0.2, f[order], ncent[order])
            hi = np.interp(0.8, f[order], ncent[order])
            width[s] = max(abs(hi - lo) / 2.0, 1.5)
    width = np.clip(ndimage.uniform_filter1d(width, 9), 1.5, 9.0)
    return ProfileState(crest, toe, center, width, face_h, ncent, prof)


def anchored_envelope(
    corridor: CorridorGeometry, anchor_residual_flat: np.ndarray, nS: int, cfg: dict
) -> np.ndarray:
    """p95 |anchor residual| in a station window — the LOCAL measured envelope."""
    near = float(cfg["near_line_m"])
    win = float(cfg["station_window_m"])
    pct = float(cfg["percentile"])
    floor, ceil = float(cfg["floor_m"]), float(cfg["ceiling_m"])
    st = corridor.station
    env = np.full(nS, floor)
    for s in range(nS):
        m = (np.abs(st - s) <= win) & (corridor.distance <= near)
        if int(np.count_nonzero(m)) > 25:
            env[s] = np.clip(np.percentile(np.abs(anchor_residual_flat[m]), pct), floor, ceil)
    return ndimage.uniform_filter1d(env, int(cfg["smooth_stations"]))


# --------------------------------------------------------------------------- #
# flow routing on C0 (D8 steepest descent accumulation) for chute seeds          #
# --------------------------------------------------------------------------- #
def flow_accumulation(c0: np.ndarray, active: np.ndarray) -> np.ndarray:
    H, W = c0.shape
    height = np.where(active, c0, np.inf)
    idx = np.flatnonzero(active.ravel())
    order = idx[np.argsort(c0.ravel()[idx])[::-1]]  # high to low
    acc = np.where(active, 1.0, 0.0).ravel()
    steps = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    hflat = height.ravel()
    for flat in order:
        r, c = divmod(int(flat), W)
        best, bd = 0.0, -1
        for k, (dr, dc) in enumerate(steps):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                drop = (hflat[flat] - hflat[rr * W + cc]) / (np.hypot(dr, dc))
                if drop > best:
                    best, bd = drop, rr * W + cc
        if bd >= 0:
            acc[bd] += acc[flat]
    return acc.reshape(H, W)


# --------------------------------------------------------------------------- #
# whole-domain assembly                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class Reconstruction:
    residual_m: np.ndarray
    face_member: np.ndarray
    toe_member: np.ndarray
    shoulder_member: np.ndarray
    grammar_m: np.ndarray
    fine_m: np.ndarray
    proximity: np.ndarray
    ordinary_ground: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)


def synthesize(
    c0: np.ndarray, active: np.ndarray, hard: np.ndarray, mapped_face: np.ndarray,
    anchor_residual: np.ndarray, frame: LineFrame, corridor: CorridorGeometry,
    state: ProfileState, envelope: np.ndarray, bbox, pitch: float, config: dict,
) -> Reconstruction:
    H, W = c0.shape
    ridx = corridor.ridx
    si = corridor.station_index
    signed = corridor.signed_n
    dist = corridor.distance
    dals = corridor.dist_to_als
    c0v = c0.ravel()[ridx]
    Rv = anchor_residual.ravel()[ridx]

    cr, to, ce = state.crest[si], state.toe[si], state.center[si]
    wd = np.clip(state.width[si], 1.5, 9.0)
    envc = envelope[si]
    u = (signed - ce) / wd

    corr = config["corridor"]
    edge_win = np.exp(-(dist / float(corr["edge_taper_sigma_m"])) ** 2)
    st = corridor.station
    endL = float(corr["endpoint_taper_m"])
    end_taper = np.minimum(_smoothstep(st, 0.0, endL), _smoothstep(frame.length_m - st, 0.0, endL))

    prot = config["measured_protection"]
    proximity = _smoothstep(dals, float(prot["als_proximity_ramp_lo_m"]), float(prot["als_proximity_ramp_hi_m"]))
    void_w = _smoothstep(dals, float(prot["void_fill_ramp_lo_m"]), float(prot["void_fill_ramp_hi_m"]))

    # memberships (form types)
    face_member = np.exp(-(u ** 2))
    toe_member = np.exp(-((signed - (ce - 2.2 * wd)) / (1.5 * wd)) ** 2)
    shoulder_member = np.exp(-((signed - (ce + 2.2 * wd)) / (1.5 * wd)) ** 2)

    # ---- macro: continuous crest-face-toe crispening (fills soft/void face) ----
    ms = config["macro_sharpen"]
    T = to + (cr - to) * (1.0 / (1.0 + np.exp(-(signed - ce) / (wd * float(ms["sigmoid_width_scale"])))))
    crispen_full = np.clip(float(ms["gain"]) * (T - c0v), -envc, envc) * edge_win * end_taper
    face_gate = face_member
    # measured where ALS present; between points/on the sparse face, crispen toward T
    macro = Rv + proximity * face_gate * (crispen_full - Rv)
    # true ALS voids: replace measured smoothing with the connected profile target
    macro = (1.0 - void_w) * macro + void_w * crispen_full

    # ---- Biala connective grammar (locally balanced, envelope bounded) ----
    gr = config["grammar"]
    along = 0.6 + 0.4 * np.sin(si * 0.22)
    # asymmetric bench: zero-integral shelf on the toe side only
    bshift = signed - (ce - float(gr["bench_normal_offset_widths"]) * wd)
    shelf = (-(bshift) / wd) * np.exp(-((bshift / (1.3 * wd)) ** 2))
    shelf = shelf / (np.max(np.abs(shelf)) + 1e-9)
    bench = float(gr["bench_gain"]) * envc * shelf * along

    # slump-headwall sockets / aprons at plan concavities (balanced cut + fill)
    curv = frame.plan_curv[si]
    socket_gate = _smoothstep(curv, float(gr["socket_min_plan_curvature"]), 3.0 * float(gr["socket_min_plan_curvature"]))
    socket_shape = np.exp(-((signed - (ce + 0.6 * wd)) / (1.1 * wd)) ** 2) - np.exp(-((signed - (ce - 1.4 * wd)) / (1.4 * wd)) ** 2)
    socket = float(gr["socket_gain"]) * envc * socket_gate * socket_shape

    # branching chute/gully seeds at flow-convergence crossings of the face
    chute = _chute_field(
        c0, active, frame, corridor, state, envc, u, signed, ce, wd, bbox, pitch, gr
    )

    grammar = bench + socket + chute

    # ---- bounded band-limited fine (0.5-1.25 m), conditioned on form type ----
    fb = config["fine_band"]
    white = _world_white((H, W), bbox, pitch, int(config["world_prf_seed"]))
    band = ndimage.gaussian_filter(white, float(fb["sigma_low_m"]) / pitch) - ndimage.gaussian_filter(
        white, float(fb["sigma_high_m"]) / pitch
    )
    band = band / (np.std(band) + 1e-12)
    bandv = band.ravel()[ridx]
    amp = (
        float(fb["face_amplitude_m"]) * face_member
        + float(fb["toe_amplitude_m"]) * toe_member
        + float(fb["shoulder_amplitude_m"]) * shoulder_member
    )
    fine = np.clip(bandv * amp, -float(fb["clip_m"]), float(fb["clip_m"]))

    # ---- combine: inferred detail bounded by the LOCAL anchored envelope,      ----
    # ---- suppressed at measurements (proximity), tapered to C0 at edge/endpoint ----
    inferred = np.clip(grammar + fine, -envc, envc) * proximity * edge_win * end_taper
    total_v = macro + inferred
    total_v = np.clip(total_v, -1.5, 1.5)

    residual = np.zeros((H, W), dtype=np.float64)
    residual.ravel()[ridx] = total_v
    residual[hard] = 0.0
    residual[~active] = 0.0
    residual[mapped_face] = 0.0

    grammar_grid = np.zeros((H, W)); grammar_grid.ravel()[ridx] = grammar * proximity * edge_win * end_taper
    fine_grid = np.zeros((H, W)); fine_grid.ravel()[ridx] = fine * proximity * edge_win * end_taper
    prox_grid = np.zeros((H, W)); prox_grid.ravel()[ridx] = proximity
    fm = np.zeros((H, W)); fm.ravel()[ridx] = face_member
    tm = np.zeros((H, W)); tm.ravel()[ridx] = toe_member
    sm = np.zeros((H, W)); sm.ravel()[ridx] = shoulder_member
    for grid in (grammar_grid, fine_grid, fm, tm, sm):
        grid[hard | ~active | mapped_face] = 0.0

    ordinary = np.zeros((H, W), dtype=bool)
    ord_v = (dist > float(config["gates"]["dc_drift_ordinary_ground_min_distance_m"])) & (
        (face_member + toe_member + shoulder_member) < 0.05
    )
    ordinary.ravel()[ridx] = ord_v
    ordinary &= active & ~hard & ~mapped_face

    return Reconstruction(
        residual, fm, tm, sm, grammar_grid, fine_grid, prox_grid, ordinary,
        diagnostics={
            "n_chute_seeds": int(chute_seed_count(chute)),
        },
    )


def _chute_field(c0, active, frame, corridor, state, envc, u, signed, ce, wd, bbox, pitch, gr):
    """Branching incision+levee seeds where flow accumulation crosses the face."""
    acc = flow_accumulation(c0, active)
    logacc = np.log1p(acc)
    accv = logacc.ravel()[corridor.ridx]
    on_face = np.abs(u) < 1.2
    thr = np.percentile(accv[on_face], float(gr["chute_accumulation_percentile"])) if np.any(on_face) else np.inf
    seed = on_face & (accv >= thr)
    # incision along the face at seed columns, small compensating levees to the sides
    incise = -np.exp(-(u ** 2) / 0.5)
    levee = 0.35 * (np.exp(-((u - 1.1) ** 2) / 0.4) + np.exp(-((u + 1.1) ** 2) / 0.4))
    shape = incise + levee
    # spread the seed influence a little along-line so chutes read as connected channels
    seedf = seed.astype(np.float64)
    field_v = float(gr["chute_gain"]) * envc * seedf * shape
    return field_v


def chute_seed_count(chute_field: np.ndarray) -> int:
    return int(np.count_nonzero(chute_field < -1e-6))
