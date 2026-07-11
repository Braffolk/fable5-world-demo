/**
 * UnderstoryScatter — Estonia's understory + debris instances, derived from the u8
 * guidance planes (SPEC-STREAMING-WORLD §3 F2 row + §7, S9a). This is the Estonia
 * analogue of the generated world's GPU scatter (Scatter.ts understory/extras/stones
 * passes): the generated source ships those as explicit records; Estonia ships
 * (categoryId, density) guidance planes, so — exactly as ChunkContent derives a tree's
 * y/yaw/lean when the source omits them — this derives ground instances from the plane
 * the source DOES ship. One derivation, no source fork: both worlds end as {a,b,idF}
 * instance words the SAME InstanceBand → pool binds.
 *
 * DEMAND LAW (§5): understory renders only to ~150 m, so it rides a TIGHT sub-chunk
 * ring (fine cells ≪ the 2048 m data chunk), NOT chunk-granular residency — a full
 * chunk's ~125 k understory would be a VRAM hog for content that is 97% culled. The
 * guidance/height PLANES are per-2048 m chunk though, so each fine cell fetches its
 * parent chunk's planes once (small LRU) and scatters only its own footprint from
 * a GLOBAL grid (position-stable → deterministic across cell/refetch boundaries).
 */

import { makeGroundDeriver, pcg2d, type GroundDeriver } from '../../world/source/RecordGround';
import { VegClass } from '../../gpu/passes/Scatter';
import { packChunkKey } from '../../world/source/Lac1';
import { pickClass, type ScatterMap } from './ScatterMap';
import type { ChunkInstances } from './ChunkContent';
import type { CellPlan } from './InstanceBand';
import type { WorldManifest, WorldSource } from '../../world/source/WorldSource';

const TAU = 6.2831853;
/** grid spacings (m) — match the generated scatter's cell grain (UNDER_CELL 2.4,
 *  STONE_CELL 2.1) so the ground reads at the same density on both sources. */
const UNDER_STEP = 2.4;
const DEBRIS_STEP = 2.5;
/** peak per-cell acceptance. Understory: the density plane (already suitability-cut)
 *  + the palette's mesh fraction (non-shrub tokens → SKIP) already carry the local
 *  lushness, so a flat peak lands forest shrubs at ~the generated 0.03/m² grain — the
 *  base_density penalty (moss/herb communities have low base) would double-count and
 *  make forest floors barren, so understory does NOT scale by base. Debris DOES scale
 *  by base (litter floors ≫ stony ground), which reads well against generated stones. */
const UNDER_PEAK = 1.0;
const DEBRIS_PEAK = 0.5;
const PLANE_TEXEL_M = 2; // enc2 guidance planes: 2 m texel (manifest texelMeters)

/** parent-chunk decoded planes + grounding deriver, cached across the fine cells that
 *  share one 2048 m chunk. null fields = the source has no such chunk (authoritative
 *  absence — that category simply scatters nothing here). */
interface ParentData {
  minX: number;
  minZ: number;
  deriver: GroundDeriver;
  uRes: number;
  uId: Uint8Array | null;
  uDen: Uint8Array | null;
  dRes: number;
  dId: Uint8Array | null;
  dDen: Uint8Array | null;
}

/** per-class size + bed-sink, mirroring the generated scatter's ranges (cosmetic). */
function sizeFor(cls: number, hy: number): { scale: number; sink: number; lean: number } {
  switch (cls) {
    case VegClass.StoneL:
      return { scale: Math.pow(hy, 1.7) * 1.6 + 0.6, sink: (Math.pow(hy, 1.7) * 1.6 + 0.6) * 0.3, lean: 0.4 };
    case VegClass.StoneM:
      return { scale: hy * 0.4 + 0.2, sink: (hy * 0.4 + 0.2) * 0.26, lean: 0.4 };
    case VegClass.StoneS:
      return { scale: hy * 0.14 + 0.06, sink: (hy * 0.14 + 0.06) * 0.22, lean: 0.4 };
    case VegClass.Boulder:
    case VegClass.Slab:
      return { scale: hy * hy * 1.9 + 0.5, sink: (hy * hy * 1.9 + 0.5) * 0.28, lean: 0.3 };
    case VegClass.Log:
      return { scale: hy * 0.6 + 0.7, sink: 0.08, lean: 0.3 };
    case VegClass.Stump:
      return { scale: hy * 0.5 + 0.6, sink: 0.05, lean: 0 };
    case VegClass.Branch:
      return { scale: hy * 0.8 + 0.6, sink: 0.05, lean: 0.3 };
    default: // shrubs (BushHazel/BushPink/Juniper)
      return { scale: Math.pow(hy, 1.4) * 0.7 + 0.6, sink: 0.03, lean: 0 };
  }
}

/**
 * Build the understory/debris CellPlan for the streamed instance band. Fine cells
 * (`cellMeters`, a divisor of the 2048 m chunk) within `bandDist` of the camera hold
 * their scattered shrubs/deadwood/stones as pool instances; the tree/boulder band
 * runs its own coarser plan against the same pool.
 */
export function understoryDebrisPlan(
  source: WorldSource,
  manifest: WorldManifest,
  map: ScatterMap,
  opts: { cellMeters: number; bandDist: number },
): CellPlan {
  const chunkM = manifest.grid.chunkMeters; // 2048 (parent stride)
  const originX = manifest.grid.originX;
  const originZ = manifest.grid.originZ;
  const cache = new Map<number, Promise<ParentData>>();
  const CACHE_CAP = 24;
  let builtUnder = 0;
  let builtDebris = 0;

  const parentOf = (pcx: number, pcz: number): Promise<ParentData> => {
    const key = packChunkKey(0, pcx, pcz);
    const hit = cache.get(key);
    if (hit) return hit;
    const p = loadParent(source, pcx, pcz, originX, originZ, chunkM);
    cache.set(key, p);
    if (cache.size > CACHE_CAP) {
      const oldest = cache.keys().next().value as number;
      cache.delete(oldest);
    }
    return p;
  };

  return {
    cellMeters: opts.cellMeters,
    bandDist: opts.bandDist,
    originX,
    originZ,
    label: 'uband',
    exists(cx: number, cz: number): boolean {
      // a fine cell exists iff its parent 2048 m chunk carries either guidance layer
      const pcx = Math.floor((cx * opts.cellMeters) / chunkM);
      const pcz = Math.floor((cz * opts.cellMeters) / chunkM);
      const k = { lod: 0, cx: pcx, cz: pcz };
      return manifest.coverage('understory', k) !== null || manifest.coverage('debris', k) !== null;
    },
    async build(cx: number, cz: number): Promise<ChunkInstances> {
      const C = opts.cellMeters;
      const x0 = originX + cx * C;
      const z0 = originZ + cz * C;
      const pcx = Math.floor((x0 - originX) / chunkM);
      const pcz = Math.floor((z0 - originZ) / chunkM);
      const parent = await parentOf(pcx, pcz);
      const a: number[] = [];
      const b: number[] = [];
      const nU = scatterGrid(a, b, x0, z0, C, UNDER_STEP, UNDER_PEAK, 0x51a3, parent.minX, parent.minZ, parent.deriver, parent.uRes, parent.uId, parent.uDen, (id, r) => pickClass(map.understory(id), r), () => 1);
      const nD = scatterGrid(a, b, x0, z0, C, DEBRIS_STEP, DEBRIS_PEAK, 0x2c9f, parent.minX, parent.minZ, parent.deriver, parent.dRes, parent.dId, parent.dDen, (id, r) => pickClass(map.debris(id), r), (id) => map.debris(id).base / map.maxDebrisBase);
      builtUnder += nU;
      builtDebris += nD;
      return { a: Float32Array.from(a), b: Float32Array.from(b), count: a.length / 4 };
    },
    extra: () => ({ 'uband.built.under': builtUnder, 'uband.built.debris': builtDebris }),
  };
}

/** fetch a parent chunk's height + guidance planes and build its grounding deriver. */
async function loadParent(
  source: WorldSource,
  pcx: number,
  pcz: number,
  originX: number,
  originZ: number,
  chunkM: number,
): Promise<ParentData> {
  const key = { lod: 0, cx: pcx, cz: pcz };
  const [h, u, d] = await Promise.all([source.fetch('height', key), source.fetch('understory', key), source.fetch('debris', key)]);
  if (!h || h.kind !== 'height') throw new Error(`UnderstoryScatter: no height chunk (0,${pcx},${pcz})`);
  const deriver = makeGroundDeriver(h.heights, h.res, chunkM, pcx, pcz);
  const up = u && u.kind === 'planes' ? u.planes : null;
  const dp = d && d.kind === 'planes' ? d.planes : null;
  return {
    minX: originX + pcx * chunkM,
    minZ: originZ + pcz * chunkM,
    deriver,
    uRes: u && u.kind === 'planes' ? u.res : 0,
    uId: up ? (up[0] as Uint8Array) : null,
    uDen: up ? (up[1] as Uint8Array) : null,
    dRes: d && d.kind === 'planes' ? d.res : 0,
    dId: dp ? (dp[0] as Uint8Array) : null,
    dDen: dp ? (dp[1] as Uint8Array) : null,
  };
}

/** scatter ONE guidance grid (understory or debris) over the fine cell [x0,x0+C)². A
 *  global grid at `step` (game space) keeps every instance position-stable regardless
 *  of which fine cell/refetch produces it; each grid cell is owned by exactly one fine
 *  cell (by its unjittered base position), so no double-emit at cell seams. Returns the
 *  instance count appended. */
function scatterGrid(
  a: number[],
  b: number[],
  x0: number,
  z0: number,
  C: number,
  step: number,
  peak: number,
  salt: number,
  minX: number,
  minZ: number,
  deriver: GroundDeriver,
  res: number,
  idPlane: Uint8Array | null,
  denPlane: Uint8Array | null,
  pick: (id: number, r01: number) => number,
  baseFrac: (id: number) => number,
): number {
  if (!idPlane || !denPlane || res <= 0) return 0;
  let n = 0;
  const gxLo = Math.floor(x0 / step);
  const gxHi = Math.floor((x0 + C) / step);
  const gzLo = Math.floor(z0 / step);
  const gzHi = Math.floor((z0 + C) / step);
  for (let gz = gzLo; gz <= gzHi; gz++) {
    const bz = gz * step;
    if (bz < z0 || bz >= z0 + C) continue;
    for (let gx = gxLo; gx <= gxHi; gx++) {
      const bx = gx * step;
      if (bx < x0 || bx >= x0 + C) continue;
      // deterministic per-cell hashes (pcg2d over the global cell + salts)
      const [j0, j1] = pcg2d((gx ^ salt) >>> 0, (gz ^ salt) >>> 0);
      const px = bx + j0 * step;
      const pz = bz + j1 * step;
      // sample the guidance plane (nearest texel; ids are categorical)
      const lx = px - minX;
      const lz = pz - minZ;
      const tx = Math.min(res - 1, Math.max(0, Math.round(lx / PLANE_TEXEL_M)));
      const tz = Math.min(res - 1, Math.max(0, Math.round(lz / PLANE_TEXEL_M)));
      const den = denPlane[tz * res + tx] as number;
      if (den === 0) continue;
      const id = idPlane[tz * res + tx] as number;
      const [rAcc, rCls] = pcg2d((gx ^ (salt + 0x9e37)) >>> 0, (gz ^ (salt + 0x85eb)) >>> 0);
      const accept = (den / 255) * baseFrac(id) * peak;
      if (rAcc >= accept) continue;
      const cls = pick(id, rCls);
      if (cls < 0) continue; // SKIP — groundcover with no mesh
      const [hScale, hVar] = pcg2d((gx ^ (salt + 0x27d4)) >>> 0, (gz ^ (salt + 0x1b56)) >>> 0);
      const variant = Math.min(3, Math.floor(hVar * 4));
      const { scale, sink, lean } = sizeFor(cls, hScale);
      const g = deriver(lx, lz);
      a.push(px, g.h - sink, pz, scale);
      b.push(j0 * TAU, g.leanX * lean, g.leanZ * lean, cls * 8 + variant);
      n++;
    }
  }
  return n;
}
