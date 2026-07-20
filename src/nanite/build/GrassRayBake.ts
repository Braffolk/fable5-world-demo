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
 * The tile = one guide texel footprint: SUB×SUB fine cells. One volume per
 * DENSITY TIER (nested cell thinning — see GrassRayBakeOpts.tiers); the
 * runtime picks the density-matched tier per pixel. Blade HEIGHT is applied
 * at runtime — the bake is 2D and height-free, exactly like the article's.
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
  /** fibers per cell (bake-side density — FREE at runtime; more fibers means
   *  shorter fetch distances, i.e. cheaper marches) */
  fibers: number;
  /** DENSITY TIERS: one volume per entry, the fraction of CELLS populated
   *  (descending, [0] = 1 = full). The world thins grass per-cell; a single
   *  uniform-tiling LUT can't express that, so the runtime picks/lerps the
   *  density-matched tier per pixel — the article-legal "fewer blades in the
   *  tiled mask", still one fetch. Tiers are NESTED (same per-cell hash,
   *  different threshold): shared blades bake identical texels, so the
   *  runtime's adjacent-tier lerp cross-fades only the blades that differ. */
  tiers: number[];
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
  /** mean normalized depth (R/255) across all tier volumes — the anti-tiling
   *  hex blend's variance-preservation pivot (Sannikov 2023: averaged-mip mean).
   *  Blending N rotated taps softens the R jump between hit/miss; pushing the
   *  blend away from this mean by 1/√(Σwᵢ²) restores the silhouette contrast. */
  meanR: number;
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

  // ---- fiber list: sub² cells × spread fibers, canonical per-cell params (LCG) -------
  // Each fiber is the article's extruded 2D cross-section: an oriented rectangle
  // (width axis = the blade's side direction at its yaw, thickness ⊥).
  const XSCALE = 1.15;
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
    nx: number;
    ny: number;
    nz: number; // shading normal (tile space)
  }
  const fibers: Fiber[] = [];
  const CSn = 0.788;
  const nl = Math.hypot(0.25, CSn);
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
        const cx = cu + rnd() * 1.3 - 0.15;
        const cz = cv + rnd() * 1.3 - 0.15;
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
          nx: (-CSn / nl) * cs,
          ny: 0.25 / nl,
          nz: (-CSn / nl) * cc,
        });
      }
    }
  }
  // infinite tiling: test copies of every fiber in a 5×5 tile neighborhood
  // (ray reach dMaxC + fiber extent < 2·sub in every direction)
  const OFFS: number[] = [-2 * sub, -sub, 0, sub, 2 * sub];

  const missR = Math.round(255 / (1 + dMaxC / sub));
  /** per-cell keep hash in [0,1) — FIXED across tiers so tier k+1's cells are a
   *  strict subset of tier k's (nested thinning ⇒ lerp-safe shared blades) */
  const cellKeep = (cu: number, cv: number): number =>
    ((((cu * 2654435761 ) ^ (cv * 2246822519) ^ 0x9e37) >>> 9) % 65536) / 65536;
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
          let best = dMaxC;
          let bi = -1;
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
                }
              }
            }
          }
          const base = ((ai * res + zi) * res + xi) * 4;
          if (bi < 0) {
            data[base] = missR;
            data[base + 1] = 0;
            data[base + 2] = 166; // up-ish normal (ny ≈ 0.3)
            data[base + 3] = 255;
          } else {
            const f = bandFibers[bi] as Fiber;
            // depth in TILE units, article encoding 1/(1+d)
            data[base] = Math.max(missR + 1, Math.round(255 / (1 + best / sub)));
            // normal as (azimuth, y) — two-sided: face the ray (horizontal flip);
            // the freed A channel carries the fiber's ROOT CELL id, the runtime's
            // validation anchor (density law applies to ROOTS — arcs legally
            // overhang empty cells, exactly like the reference ring)
            const flip = f.nx * ux + f.nz * uz > 0 ? -1 : 1;
            let azN = Math.atan2(f.nz * flip, f.nx * flip) / (Math.PI * 2);
            if (azN < 0) azN += 1;
            const nrmL = Math.hypot(f.nx, f.ny, f.nz) || 1;
            data[base + 1] = Math.round(azN * 255);
            data[base + 2] = Math.round((f.ny / nrmL) * 127.5 + 127.5);
            data[base + 3] = Math.round(((f.rv * sub + f.ru + 0.5) / (sub * sub)) * 255);
          }
        }
      }
    }
    volumes.push(data);
  }
  // global mean R (over every tier volume) for the hex blend's variance pivot
  let sumR = 0;
  let nR = 0;
  for (const v of volumes) {
    for (let i = 0; i < v.length; i += 4) {
      sumR += v[i] as number;
      nR++;
    }
  }
  const meanR = nR > 0 ? sumR / nR / 255 : 0.5;
  console.info(
    `[grass] ray tile baked: ${res}×${res}×${angles} ×${o.tiers.length} tiers ` +
      `(${o.tiers.join('/')}), ${fibers.length} fibers, meanR=${meanR.toFixed(3)}, ` +
      `${Math.round(performance.now() - t0)} ms`,
  );
  return { data: volumes, res, angles, dMaxTile: dMaxC / sub, meanR };
}
