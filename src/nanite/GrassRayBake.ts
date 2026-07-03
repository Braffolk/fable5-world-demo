/**
 * GrassRayBake — CPU precomputation of the raycast tile texture (G-E, the
 * Sannikov algorithm: docs/deep-review/grass-raycast.txt).
 *
 * The article's precomputation, faithfully: a 3D texture over (x, z inside one
 * tile, normalized ray angle) storing the 2D ray PATH LENGTH to the first
 * fiber intersection (R = 1/(1+d), d in tile widths) plus the surface normal
 * (GBA = n·0.5+0.5), traced assuming INFINITELY TILED geometry. His two
 * bake-time heuristics are included: fibers SHIFT along their bend direction
 * as the ray marches (approximates inclined/curved blades — exact only for
 * parallel prisms) and THICKEN with ray distance (approximates blades tapering
 * toward the tip, seen root-ward as rays descend). Runtime answers a march
 * step with ONE fetch — no stepping, no per-clump batteries.
 *
 * The tile = one guide texel footprint: SUB×SUB fine cells at FULL density
 * (every cell holds a full 5-blade clump). Real-world density (the mask law)
 * and per-cell blade HEIGHTS are applied at runtime — the bake is 2D and
 * height-free, exactly like the article's.
 *
 * The article: "Предрасчёт вычисляется на чём угодно и каким угодно
 * алгоритмом" — so this is plain JS at boot, analytic (fiber cross-sections
 * are oriented rectangles; shift+thicken keep the ray/fiber test a LINEAR
 * interval intersection — no stepping in the bake either). 64×64×8 RGBA8 by
 * default (his numbers), ~130 KB.
 */

/** one blade of the canonical clump table (NaniteGrass BLADES — passed in to
 *  avoid a module cycle) */
export interface BakeBlade {
  c: number;
  s: number;
  ox: number;
  oz: number;
  hk: number;
  lean: number;
  nm: [number, number, number];
}

export interface GrassRayBakeOpts {
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
}

export interface GrassRayBake {
  /** RGBA8, index ((angle·res + z)·res + x)·4 — Data3DTexture layout */
  data: Uint8Array;
  res: number;
  angles: number;
  /** max traced distance in TILE units — runtime treats d ≥ ~0.97·this as miss */
  dMaxTile: number;
}

export function bakeGrassRayTile(o: GrassRayBakeOpts): GrassRayBake {
  const t0 = performance.now();
  const { res, angles, sub } = o;
  const invCell = 1 / o.cellM;
  // fibers wander ≤ shiftK·dMax cells; runtime clamps hits to the CURRENT tile,
  // so 1.5 tiles of traced range is already conservative.
  const dMaxC = 1.5 * sub; // cells
  const hw0 = o.halfW * invCell;
  const ht0 = o.halfT * invCell;

  // ---- fiber list: sub² cells × blades, canonical per-cell params (LCG) --------------
  // Geometry mirrors NaniteGrass corner() at a canonical height t̄ = 0.35 (the 2D
  // cross-section the article extrudes): root = clump + rotYaw(offset + curve/lean
  // displacement), width axis = the blade's side direction, thickness ⊥.
  const TBAR = 0.35;
  const XSCALE = 1.15;
  interface Fiber {
    cx: number;
    cz: number; // center, cells
    wx: number;
    wz: number; // width axis (unit)
    tx: number;
    tz: number; // thickness axis (unit)
    fx: number;
    fz: number; // bend/shift direction (unit)
    nx: number;
    ny: number;
    nz: number; // shading normal (tile space)
  }
  const fibers: Fiber[] = [];
  for (let cv = 0; cv < sub; cv++) {
    for (let cu = 0; cu < sub; cu++) {
      let s = ((cu * 127 + cv * 311 + 17) * 1664525 + 1013904223) >>> 0;
      const rnd = (): number => {
        s = (s * 1664525 + 1013904223) >>> 0;
        return s / 4294967296;
      };
      const jx = rnd();
      const jz = rnd();
      const yaw = rnd() * Math.PI * 2;
      const cc = Math.cos(yaw);
      const cs = Math.sin(yaw);
      // clump transform (corner() verbatim): rx = x·cc + z·cs, rz = z·cc − x·cs
      const rotX = (x: number, z: number): number => x * cc + z * cs;
      const rotZ = (x: number, z: number): number => z * cc - x * cs;
      for (const b of o.blades) {
        const bby = TBAR * (1 - TBAR * TBAR * 0.06) * b.hk;
        const bbz = TBAR * TBAR * 0.28;
        // local offset at t̄ (meters, before clump scale/rot)
        const lx = (bbz * b.s + b.ox + b.lean * bby * b.c) * XSCALE;
        const lz = bbz * b.c + b.oz + b.lean * bby * b.s;
        const cx = cu + jx + rotX(lx, lz) * invCell;
        const cz = cv + jz + rotZ(lx, lz) * invCell;
        // width axis: ∂corner/∂side direction (bc, −bs) x-scaled, clump-rotated
        let wx = rotX(b.c * XSCALE, -b.s);
        let wz = rotZ(b.c * XSCALE, -b.s);
        const wl = Math.hypot(wx, wz) || 1;
        wx /= wl;
        wz /= wl;
        // bend/curve direction (bs, bc) — the shift heuristic's travel axis
        let fx = rotX(b.s * XSCALE, b.c);
        let fz = rotZ(b.s * XSCALE, b.c);
        const fl = Math.hypot(fx, fz) || 1;
        fx /= fl;
        fz /= fl;
        // shading normal = the table's mean rounded normal, clump-rotated
        const nx = rotX(b.nm[0], b.nm[2]);
        const nz = rotZ(b.nm[0], b.nm[2]);
        fibers.push({ cx, cz, wx, wz, tx: -wz, tz: wx, fx, fz, nx, ny: b.nm[1], nz });
      }
    }
  }
  // infinite tiling: test copies of every fiber in a 5×5 tile neighborhood
  // (ray reach dMaxC + fiber extent < 2·sub in every direction)
  const OFFS: number[] = [-2 * sub, -sub, 0, sub, 2 * sub];

  const data = new Uint8Array(res * res * angles * 4);
  const missR = Math.round(255 / (1 + dMaxC / sub));
  for (let ai = 0; ai < angles; ai++) {
    const th = ((ai + 0.5) / angles) * Math.PI * 2;
    const ux = Math.cos(th);
    const uz = Math.sin(th);
    for (let zi = 0; zi < res; zi++) {
      const oz_ = ((zi + 0.5) / res) * sub;
      for (let xi = 0; xi < res; xi++) {
        const ox_ = ((xi + 0.5) / res) * sub;
        let best = dMaxC;
        let bi = -1;
        for (const f of fibers) {
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
                bi = fibers.indexOf(f);
              }
            }
          }
        }
        const base = ((ai * res + zi) * res + xi) * 4;
        if (bi < 0) {
          data[base] = missR;
          data[base + 1] = 128;
          data[base + 2] = 255;
          data[base + 3] = 128; // up normal
        } else {
          const f = fibers[bi] as Fiber;
          // depth in TILE units, article encoding 1/(1+d)
          data[base] = Math.max(missR + 1, Math.round(255 / (1 + best / sub)));
          // two-sided: face the ray (horizontal flip only — keep the up term)
          const flip = f.nx * ux + f.nz * uz > 0 ? -1 : 1;
          const nl = Math.hypot(f.nx, f.ny, f.nz) || 1;
          data[base + 1] = Math.round(((f.nx * flip) / nl) * 127.5 + 127.5);
          data[base + 2] = Math.round((f.ny / nl) * 127.5 + 127.5);
          data[base + 3] = Math.round(((f.nz * flip) / nl) * 127.5 + 127.5);
        }
      }
    }
  }
  console.info(
    `[grass] ray tile baked: ${res}×${res}×${angles}, ${fibers.length} fibers, ` +
      `${Math.round(performance.now() - t0)} ms`,
  );
  return { data, res, angles, dMaxTile: dMaxC / sub };
}
