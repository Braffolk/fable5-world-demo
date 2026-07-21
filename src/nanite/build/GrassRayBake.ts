/**
 * GrassRayBake — CPU precomputation of the raycast tile texture (G-E, the
 * Sannikov algorithm: docs/deep-review/grass-raycast.txt).
 *
 * The article's precomputation, faithfully: a 3D texture over (x, z inside one
 * tile, normalized ray angle) storing the 2D ray PATH LENGTH to the first
 * fiber intersection (R = 1/(1+d), d in tile widths) plus the surface normal
 * (GBA = n·0.5+0.5), traced assuming INFINITELY TILED geometry. His two
 * optional bake-time heuristics are available: fibers can SHIFT along their bend
 * direction as the ray advances and THICKEN with ray distance. Production keeps
 * both at zero because the article identifies them as controlled-error extensions;
 * the base parallel extrusion is geometrically exact. Runtime answers with fixed
 * fetches — no stepping and no per-clump batteries.
 *
 * The tile = one guide texel footprint: SUB×SUB fine cells. One volume per
 * DENSITY TIER (nested cell thinning — see GrassRayBakeOpts.tiers); the
 * runtime picks the density-matched tier per pixel. Extruded profiles are 2D
 * and height-free, exactly like the article's. Rounded cushion profiles use a
 * bounded set of bake-time ray-elevation slices; runtime still performs fixed
 * atlas fetches and never steps through the volume.
 *
 * The article: "Предрасчёт вычисляется на чём угодно и каким угодно
 * алгоритмом" — so this is plain JS at boot, analytic (fiber cross-sections
 * are oriented rectangles; shift+thicken keep the ray/fiber test a LINEAR
 * interval intersection — no stepping in the bake either). 64×64×8 RGBA8 by
 * default (his numbers), ~130 KB.
 */

/** one blade of the canonical clump table (NaniteGrass BLADES — passed in to
 *  avoid a module cycle) */
interface BakeBlade {
  c: number;
  s: number;
  ox: number;
  oz: number;
  hk: number;
  lean: number;
}

export interface GrassRayBakeOpts {
  /** diagnostic/profile label for bake logs */
  label?: string;
  /** texels per tile edge (article: 64; "1 texel ≈ 1 screen px" law) */
  res: number;
  /** angle slices (article: 8..64, he uses 8) */
  angles: number;
  /** the 5-blade clump table (per-clump variety comes from runtime bombing) */
  blades: BakeBlade[];
  /** fine cells per tile edge (guide GUIDE_SUB = 8) */
  sub: number;
  /** world meters per fine cell (CELL = 0.105) */
  cellM: number;
  /** fiber shift per unit ray distance, cells/cell (article's incline heuristic) */
  shiftK: number;
  /** fiber half-width growth per cell of ray distance (article's taper heuristic) */
  thickK: number;
  /** fiber half-width at d=0, meters */
  halfW: number;
  /** fiber half-thickness at d=0, meters */
  halfT: number;
  /** fibers per cell (bake-side density — FREE at runtime; more fibers means
   *  shorter fetch distances, i.e. cheaper marches) */
  fibers: number;
  /** Analytic 2D section. Rectangle preserves the original grass bake; ellipse
   *  supplies rounded XZ sections for lichen lobes/rosette leaves and is also
   *  the footprint used by the true ellipsoid-cap cushion shape below. */
  section?: 'rectangle' | 'ellipse';
  /** Vertical shape. `extruded` is Sannikov's exact parallel extrusion.
   * `ellipsoid-cap` bakes a genuine rounded height surface and therefore
   * requires section=ellipse plus one `dropPerTile` elevation slice. */
  shape?: 'extruded' | 'ellipsoid-cap';
  /** For ellipsoid-cap: normalized height lost per horizontal TILE travelled by
   * the bake ray. Runtime selects among a bounded precomputed elevation set. */
  dropPerTile?: number;
  /** placement span inside a root cell. 1.3 is the original overlap; values
   *  below 1 cluster sections toward the cell centre. */
  spread?: number;
  /** DENSITY TIERS: one volume per entry, the fraction of CELLS populated
   *  (descending, [0] = 1 = full). The world thins grass per-cell; a single
   *  uniform-tiling LUT can't express that, so the runtime validates the denser
   *  tier's nearest root and selects one complete record from the two bracketing
   *  tiers. Tiers are NESTED (same per-cell hash, different threshold), making
   *  the sparser candidate a guaranteed-valid fallback without interpolating
   *  unrelated visibility records. */
  tiers: number[];
  /** pcg2d salt shared with the runtime root-validity test. */
  keepSalt: number;
  /** per-fiber arc magnitude scale (tip displacement ≈ 0.35..1.3 cells × this) */
  arcK: number;
}

export interface GrassRayBake {
  /** one RGBA8 volume per DENSITY TIER, index ((angle·res + z)·res + x)·4 */
  data: Uint8Array[];
  res: number;
  angles: number;
  /** max traced distance in TILE units — runtime treats d ≥ ~0.97·this as miss */
  dMaxTile: number;
}

/** Packed profile atlas. Each profile owns [last-angle guard, real angles,
 * first-angle guard], so hardware-linear filtering remains periodic within a
 * profile and can never interpolate across cover types. */
export interface GroundCoverRayAtlas {
  data: Uint8Array[];
  res: number;
  angles: number;
  angleStride: number;
  depth: number;
  dMaxTile: number;
}

/** Vector-continuous two-channel normal encoding. Scalar azimuth is not a
 * filterable quantity: values either side of its 0/1 seam interpolate through
 * the opposite direction. Octahedral XY survives the atlas's required linear
 * filtering and matches the offline GPU profile format. */
export function encodeOctNormal(x: number, y: number, z: number): [number, number] {
  const length = Math.hypot(x, y, z) || 1;
  let px = x / length;
  let py = y / length;
  const pz = z / length;
  const l1 = Math.abs(px) + Math.abs(py) + Math.abs(pz) || 1;
  px /= l1;
  py /= l1;
  if (pz < 0) {
    const oldX = px;
    px = (1 - Math.abs(py)) * (oldX < 0 ? -1 : 1);
    py = (1 - Math.abs(oldX)) * (py < 0 ? -1 : 1);
  }
  return [px * 0.5 + 0.5, py * 0.5 + 0.5];
}

export function decodeOctNormal(x01: number, y01: number): [number, number, number] {
  let x = x01 * 2 - 1;
  let y = y01 * 2 - 1;
  const z = 1 - Math.abs(x) - Math.abs(y);
  if (z < 0) {
    const oldX = x;
    x = (1 - Math.abs(y)) * (oldX < 0 ? -1 : 1);
    y = (1 - Math.abs(oldX)) * (y < 0 ? -1 : 1);
  }
  const length = Math.hypot(x, y, z) || 1;
  return [x / length, y / length, z / length];
}

/** CPU contract twin for the runtime's canonical periodic-root recovery.
 * `q*` is the layer-space ray origin in tile units; `dq*` is its normalized
 * horizontal direction; `dTile` is the baked hit path. The stored root id owns
 * one cell in the canonical SUB×SUB tile. Selecting the nearest repeated copy
 * at the hit, then inverting the fixed layer transform, makes the root address
 * independent of which surface point/view recovered the same atlas object. */
export function recoverPeriodicRootAnchor(
  qx: number,
  qz: number,
  dqx: number,
  dqz: number,
  dTile: number,
  rootId: number,
  sub: number,
  angle: number,
  scale: number,
  phaseX: number,
  phaseZ: number,
): { pTileX: number; pTileZ: number; rootU: number; rootV: number } {
  if (!Number.isInteger(sub) || sub <= 0) throw new Error('periodic root SUB must be a positive integer');
  if (!Number.isInteger(rootId) || rootId < 0 || rootId >= sub * sub) {
    throw new Error('periodic root id is outside the canonical tile');
  }
  if (!(scale > 0)) throw new Error('periodic layer scale must be positive');
  const rootV = Math.floor(rootId / sub);
  const rootU = rootId - rootV * sub;
  const localX = (rootU + 0.5) / sub;
  const localZ = (rootV + 0.5) / sub;
  const hitX = qx + dqx * dTile;
  const hitZ = qz + dqz * dTile;
  const rootQx = Math.floor(hitX - localX + 0.5) + localX;
  const rootQz = Math.floor(hitZ - localZ + 0.5) + localZ;
  const a = rootQx - phaseX;
  const b = rootQz - phaseZ;
  const cs = Math.cos(angle);
  const sn = Math.sin(angle);
  return {
    pTileX: (a * cs + b * sn) / scale,
    pTileZ: (-a * sn + b * cs) / scale,
    rootU,
    rootV,
  };
}

export function packGroundCoverRayAtlas(
  profiles: readonly GrassRayBake[],
): GroundCoverRayAtlas {
  const first = profiles[0];
  if (!first) throw new Error('GroundCoverRayAtlas: at least one profile is required');
  const tiers = first.data.length;
  for (const profile of profiles) {
    if (
      profile.res !== first.res
      || profile.angles !== first.angles
      || profile.data.length !== tiers
      || Math.abs(profile.dMaxTile - first.dMaxTile) > 1e-9
    ) {
      throw new Error('GroundCoverRayAtlas: profile layouts must match');
    }
  }
  const angleStride = first.angles + 2;
  const depth = profiles.length * angleStride;
  const sliceBytes = first.res * first.res * 4;
  const data = Array.from(
    { length: tiers },
    () => new Uint8Array(sliceBytes * depth),
  );
  for (let pi = 0; pi < profiles.length; pi++) {
    const profile = profiles[pi] as GrassRayBake;
    for (let tier = 0; tier < tiers; tier++) {
      const src = profile.data[tier] as Uint8Array;
      const dest = data[tier] as Uint8Array;
      if (src.length !== sliceBytes * first.angles) {
        throw new Error('GroundCoverRayAtlas: malformed profile volume');
      }
      for (let ai = 0; ai < first.angles; ai++) {
        const src0 = ai * sliceBytes;
        const dst0 = (pi * angleStride + 1 + ai) * sliceBytes;
        dest.set(src.subarray(src0, src0 + sliceBytes), dst0);
      }
      const prefix = pi * angleStride * sliceBytes;
      const last = (first.angles - 1) * sliceBytes;
      dest.set(src.subarray(last, last + sliceBytes), prefix);
      const suffix = (pi * angleStride + first.angles + 1) * sliceBytes;
      dest.set(src.subarray(0, sliceBytes), suffix);
    }
  }
  return {
    data,
    res: first.res,
    angles: first.angles,
    angleStride,
    depth,
    dMaxTile: first.dMaxTile,
  };
}

export function bakeGrassRayTile(o: GrassRayBakeOpts): GrassRayBake {
  const t0 = performance.now();
  const { res, angles, sub } = o;
  const section = o.section ?? 'rectangle';
  const shape = o.shape ?? 'extruded';
  const dropPerTile = o.dropPerTile ?? 0;
  if (section === 'ellipse' && o.thickK !== 0) {
    throw new Error('GrassRayBake: ellipse sections require thickK=0');
  }
  if (
    shape === 'ellipsoid-cap'
    && (section !== 'ellipse' || o.shiftK !== 0 || o.thickK !== 0 || o.arcK !== 0 || dropPerTile <= 0)
  ) {
    throw new Error('GrassRayBake: ellipsoid-cap requires ellipse, positive dropPerTile, and zero shift/thicken/arc');
  }
  const invCell = 1 / o.cellM;
  // fibers wander ≤ shiftK·dMax cells; runtime clamps hits to the CURRENT tile,
  // so 1.5 tiles of traced range is already conservative.
  const dMaxC = 1.5 * sub; // encoded path cap, cells for extrusion / normalized metric for cushion
  const pathScale = shape === 'ellipsoid-cap' ? Math.hypot(1, dropPerTile) : 1;
  const dMaxHorizontalC = dMaxC / pathScale;
  const hw0 = o.halfW * invCell;
  const ht0 = o.halfT * invCell;

  // ---- fiber list: sub² cells × spread fibers, canonical per-cell params (LCG) -------
  // Each fiber is the article's extruded 2D cross-section: an oriented rectangle
  // (width axis = the blade's side direction at its yaw, thickness ⊥).
  const XSCALE = 1.15;
  const spread = o.spread ?? 1.3;
  interface Fiber {
    cx: number;
    cz: number; // center at the ROOT, cells
    ru: number;
    rv: number; // ROOT CELL (validation anchor — density law applies to roots)
    wx: number;
    wz: number; // width axis (unit)
    tx: number;
    tz: number; // thickness axis (unit)
    fx: number;
    fz: number; // bend/shift + radial-arc direction (unit)
    arcM: number; // per-fiber tip arc displacement, cells
  }
  const fibers: Fiber[] = [];
  for (let cv = 0; cv < sub; cv++) {
    for (let cu = 0; cu < sub; cu++) {
      let s = ((cu * 127 + cv * 311 + 17) * 1664525 + 1013904223) >>> 0;
      const rnd = (): number => {
        s = (s * 1664525 + 1013904223) >>> 0;
        return s / 4294967296;
      };
      for (let k = 0; k < o.fibers; k++) {
        // SPREAD placement (user call 2026-07-04: the tight 5-blade clump per
        // cell read as dot-tufts with bare holes from above): fibers distribute
        // across the WHOLE cell footprint with overlap into neighbors, each with
        // its own yaw — coverage instead of batches. Blade-table lean still
        // flavors the per-fiber bend axis via the shift heuristic.
        const b = o.blades[k % o.blades.length] as BakeBlade;
        const cx = cu + rnd() * spread + (1 - spread) * 0.5;
        const cz = cv + rnd() * spread + (1 - spread) * 0.5;
        const yaw = rnd() * Math.PI * 2;
        const cc = Math.cos(yaw);
        const cs = Math.sin(yaw);
        // width axis (blade local (1,0) x-scaled, yaw-rotated), thickness ⊥
        let wx = cc * XSCALE;
        let wz = -cs;
        const wl = Math.hypot(wx, wz) || 1;
        wx /= wl;
        wz /= wl;
        // bend direction (blade local (0,1) yaw-rotated) — shift heuristic axis
        // AND the fiber's own radial-arc direction; sign-flavored by table lean
        const fsgn = b.lean >= 0 ? 1 : -1;
        let fx = cs * XSCALE * fsgn;
        let fz = cc * fsgn;
        const fl = Math.hypot(fx, fz) || 1;
        fx /= fl;
        fz /= fl;
        // canonical mean rounded normal (0, .25, −CS), yaw-rotated
        fibers.push({
          cx,
          cz,
          ru: cu,
          rv: cv,
          wx,
          wz,
          tx: -wz,
          tz: wx,
          fx,
          fz,
          arcM: (0.5 + rnd() * 1.1) * o.arcK, // tip displacement, cells
        });
      }
    }
  }
  // infinite tiling: test copies of every fiber in a 5×5 tile neighborhood
  // (ray reach dMaxC + fiber extent < 2·sub in every direction)
  const OFFS: number[] = [-2 * sub, -sub, 0, sub, 2 * sub];

  const missR = Math.round(255 / (1 + dMaxC / sub));
  /** Exact CPU twin of Scatter.cellHash(...).x. Keeping the root threshold
   *  reconstructible in the shader lets it select one COMPLETE tier hit record;
   *  interpolating unrelated depths/normals/root ids is not valid visibility. */
  const cellKeep = (cu: number, cv: number): number => {
    const M = 1664525;
    const C = 1013904223;
    const add = (a: number, b: number): number => (a + b) >>> 0;
    const mul = (a: number, b: number): number => Math.imul(a, b) >>> 0;
    let a = (cu + 40000 + (o.keepSalt & 0x3fff)) >>> 0;
    let b = (cv + 40000 + ((o.keepSalt >> 14) & 0x3fff)) >>> 0;
    a = add(mul(a, M), C);
    b = add(mul(b, M), C);
    a = add(a, mul(b, M));
    b = add(b, mul(a, M));
    a = (a ^ (a >>> 16)) >>> 0;
    b = (b ^ (b >>> 16)) >>> 0;
    a = add(a, mul(b, M));
    b = add(b, mul(a, M));
    a = (a ^ (a >>> 16)) >>> 0;
    return (a & 0xffffff) / 16777216;
  };
  const volumes: Uint8Array[] = [];
  for (const keepF of o.tiers) {
    // mid-height cross-section: every fiber displaced along its OWN arc
    // direction by arcM·0.25 (t=0.5 of the arc·t² law — the single height-free
    // slice that reads as curved clumps; runtime column jitter does the tips)
    const bandFibers = fibers
      .filter((f) => cellKeep(f.ru, f.rv) < keepF)
      .map((f) => ({
        ...f,
        cx: f.cx + f.fx * f.arcM * 0.25,
        cz: f.cz + f.fz * f.arcM * 0.25,
      }));
    const data = new Uint8Array(res * res * angles * 4);
    for (let ai = 0; ai < angles; ai++) {
      const th = ((ai + 0.5) / angles) * Math.PI * 2;
      const ux = Math.cos(th);
      const uz = Math.sin(th);
      for (let zi = 0; zi < res; zi++) {
        const oz_ = ((zi + 0.5) / res) * sub;
        for (let xi = 0; xi < res; xi++) {
          const ox_ = ((xi + 0.5) / res) * sub;
          let best = dMaxHorizontalC;
          let bi = -1;
          let bestNx = 0;
          let bestNy = 1;
          let bestNz = 0;
          for (let fi = 0; fi < bandFibers.length; fi++) {
            const f = bandFibers[fi] as Fiber;
            // B = u − fwd·shiftK (the fiber recedes/advances as the ray marches);
            // growth g = h0·thickK per cell of distance
            const bx = ux - f.fx * o.shiftK;
            const bz = uz - f.fz * o.shiftK;
            for (const oxT of OFFS) {
              for (const ozT of OFFS) {
                const dcx = f.cx + oxT - ox_;
                const dcz = f.cz + ozT - oz_;
                // cheap rejects: behind / beyond reach / too far off-axis
                const along = dcx * ux + dcz * uz;
                if (along < -2 || along > best + 2) continue;
                const perp = dcx * uz - dcz * ux;
                if (perp > 2.5 || perp < -2.5) continue;
                // interval test: |(A + B·d)·axis| ≤ h0·(1 + thickK·d), axis ∈ {ŵ, t̂}
                const ax = -dcx; // A = o − c0
                const az = -dcz;
                if (section === 'ellipse') {
                  const aw = (ax * f.wx + az * f.wz) / Math.max(hw0, 1e-8);
                  const at = (ax * f.tx + az * f.tz) / Math.max(ht0, 1e-8);
                  const bw = (bx * f.wx + bz * f.wz) / Math.max(hw0, 1e-8);
                  const bt = (bx * f.tx + bz * f.tz) / Math.max(ht0, 1e-8);
                  const kCell = shape === 'ellipsoid-cap' ? dropPerTile / sub : 0;
                  const qa = bw * bw + bt * bt + kCell * kCell;
                  const qb = 2 * (aw * bw + at * bt - kCell);
                  const qc = aw * aw + at * at + (shape === 'ellipsoid-cap' ? 0 : -1);
                  let hit = Number.POSITIVE_INFINITY;
                  if (qc <= 0) {
                    hit = 0;
                  } else if (qa > 1e-12) {
                    const disc = qb * qb - 4 * qa * qc;
                    if (disc >= 0) hit = (-qb - Math.sqrt(disc)) / (2 * qa);
                  }
                  const hitH = 1 - kCell * hit;
                  if (hit < 0 || hit >= best || (shape === 'ellipsoid-cap' && (hitH < 0 || hitH > 1))) continue;
                  best = hit;
                  bi = fi;
                  if (hit <= 1e-7 && shape === 'extruded') {
                    bestNx = 0;
                    bestNy = 1;
                    bestNz = 0;
                  } else {
                    const hx = ax + bx * hit;
                    const hz = az + bz * hit;
                    const vw = hx * f.wx + hz * f.wz;
                    const vt = hx * f.tx + hz * f.tz;
                    bestNx = f.wx * (vw / Math.max(hw0 * hw0, 1e-12))
                      + f.tx * (vt / Math.max(ht0 * ht0, 1e-12));
                    bestNz = f.wz * (vw / Math.max(hw0 * hw0, 1e-12))
                      + f.tz * (vt / Math.max(ht0 * ht0, 1e-12));
                    bestNy = shape === 'ellipsoid-cap' ? hitH : 0;
                    if (shape === 'ellipsoid-cap') {
                      // Horizontal bake coordinates above are in CELLS; runtime
                      // works in tile coordinates, so transform the implicit
                      // gradient before packing its azimuth/y component.
                      bestNx *= sub;
                      bestNz *= sub;
                    }
                    const nl = Math.hypot(bestNx, bestNy, bestNz) || 1;
                    bestNx /= nl;
                    bestNz /= nl;
                    bestNy /= nl;
                    if (bestNx * ux + bestNz * uz - bestNy * kCell > 0) {
                      bestNx = -bestNx;
                      bestNy = -bestNy;
                      bestNz = -bestNz;
                    }
                  }
                  continue;
                }
                let lo = 0;
                let hi = best;
                let ok = true;
                const slab = (aA: number, aB: number, h0: number): void => {
                  const g = h0 * o.thickK;
                  // (aB − g)·d ≤ h0 − aA   and   (−aB − g)·d ≤ h0 + aA
                  const k1 = aB - g;
                  const m1 = h0 - aA;
                  if (k1 > 1e-9) hi = Math.min(hi, m1 / k1);
                  else if (k1 < -1e-9) lo = Math.max(lo, m1 / k1);
                  else if (m1 < 0) ok = false;
                  const k2 = -aB - g;
                  const m2 = h0 + aA;
                  if (k2 > 1e-9) hi = Math.min(hi, m2 / k2);
                  else if (k2 < -1e-9) lo = Math.max(lo, m2 / k2);
                  else if (m2 < 0) ok = false;
                };
                slab(ax * f.wx + az * f.wz, bx * f.wx + bz * f.wz, hw0);
                if (!ok) continue;
                slab(ax * f.tx + az * f.tz, bx * f.tx + bz * f.tz, ht0);
                if (!ok || lo > hi) continue;
                if (lo < best) {
                  best = lo;
                  bi = fi;
                  if (lo <= 1e-7) {
                    // The shell entry is already inside this finite sward
                    // footprint: its first surface is the top/origin cap.
                    bestNx = 0;
                    bestNy = 1;
                    bestNz = 0;
                  } else {
                    // Store the face actually entered, not a preset per-fiber
                    // direction. At rectangle corners pick the closer slab face.
                    const hx = ax + bx * lo;
                    const hz = az + bz * lo;
                    const vw = hx * f.wx + hz * f.wz;
                    const vt = hx * f.tx + hz * f.tz;
                    const hw = hw0 * (1 + o.thickK * lo);
                    const ht = ht0 * (1 + o.thickK * lo);
                    const ew = Math.abs(Math.abs(vw) - hw) / Math.max(hw, 1e-8);
                    const et = Math.abs(Math.abs(vt) - ht) / Math.max(ht, 1e-8);
                    if (ew <= et) {
                      const sign = vw >= 0 ? 1 : -1;
                      bestNx = f.wx * sign;
                      bestNy = 0;
                      bestNz = f.wz * sign;
                    } else {
                      const sign = vt >= 0 ? 1 : -1;
                      bestNx = f.tx * sign;
                      bestNy = 0;
                      bestNz = f.tz * sign;
                    }
                    // Numerical corner cases must still face the incoming ray.
                    if (bestNx * ux + bestNz * uz > 0) {
                      bestNx = -bestNx;
                      bestNz = -bestNz;
                    }
                  }
                }
              }
            }
          }
          const base = ((ai * res + zi) * res + xi) * 4;
          if (bi < 0) {
            data[base] = missR;
            data[base + 1] = 128;
            data[base + 2] = 255; // oct(0, 1, 0); ignored for a categorical miss
            data[base + 3] = 255;
          } else {
            const f = bandFibers[bi] as Fiber;
            // depth in TILE units, article encoding 1/(1+d)
            const encodedPath = best * pathScale;
            data[base] = Math.max(missR + 1, Math.round(255 / (1 + encodedPath / sub)));
            // actual entering-face normal as filterable octahedral XY;
            // the freed A channel carries the fiber's ROOT CELL id, the runtime's
            // validation anchor (density law applies to ROOTS — arcs legally
            // overhang empty cells, exactly like the reference ring)
            const nrmL = Math.hypot(bestNx, bestNy, bestNz) || 1;
            const oct = encodeOctNormal(bestNx / nrmL, bestNy / nrmL, bestNz / nrmL);
            data[base + 1] = Math.round(oct[0] * 255);
            data[base + 2] = Math.round(oct[1] * 255);
            data[base + 3] = Math.round(((f.rv * sub + f.ru + 0.5) / (sub * sub)) * 255);
          }
        }
      }
    }
    volumes.push(data);
  }
  console.info(
    `[grass] ray tile baked${o.label ? ` (${o.label})` : ''}: ` +
      `${res}×${res}×${angles} ×${o.tiers.length} tiers ` +
      `(${o.tiers.join('/')}), ${fibers.length} fibers, ` +
      `${Math.round(performance.now() - t0)} ms`,
  );
  return { data: volumes, res, angles, dMaxTile: dMaxC / sub };
}
