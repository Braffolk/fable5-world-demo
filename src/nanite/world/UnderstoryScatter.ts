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

/** one decoded guidance layer of a parent chunk: (categoryId, density) u8 planes. */
interface GuidancePlane {
  res: number;
  id: Uint8Array;
  den: Uint8Array;
}

/** parent-chunk decoded planes + grounding deriver, cached across the fine cells that
 *  share one 2048 m chunk. null layers = the source has no such chunk (authoritative
 *  absence — that category simply scatters nothing here). */
interface ParentData {
  minX: number;
  minZ: number;
  deriver: GroundDeriver;
  under: GuidancePlane | null;
  debris: GuidancePlane | null;
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
    // Bog understory: meshes author base at y=0 and real-metre heights (each module's
    // exported *_HEIGHT const), so the generic 3 cm shrub sink buries them and the
    // sub-1 generic curve shrinks them below their real species range. Curves below
    // land mesh(H)×scale in the REAL range per species; sink ≈ stem-litter contact.
    case VegClass.CottonGrass: // tussock culms, mesh 0.42-0.5 m ≈ real 0.3-0.6 m
      return { scale: 0.9 + hy * 0.3, sink: 0.01, lean: 0 };
    case VegClass.Heather: // Calluna, mesh 0.22-0.36 m → real 0.2-0.55 m
      return { scale: 1.0 + hy * 0.6, sink: 0.01, lean: 0 };
    case VegClass.LabradorTea: // mesh 0.3-0.55 m → real 0.5-1.2 m
      return { scale: 1.2 + hy * 0.8, sink: 0.01, lean: 0 };
    case VegClass.BogRosemary: // mesh 0.15-0.3 m ≈ real 0.1-0.4 m (stays low)
      return { scale: 1.0 + hy * 0.3, sink: 0.008, lean: 0 };
    case VegClass.Cranberry: // creeping mat, mesh 0.06-0.1 m → real 0.05-0.15 m
    case VegClass.Cloudberry: // herb, mesh 0.1-0.2 m → real 0.1-0.3 m
      return { scale: 1.0 + hy * 0.5, sink: 0.005, lean: 0 };
    default: // shrubs (BushHazel/BushPink/Juniper)
      return { scale: Math.pow(hy, 1.4) * 0.7 + 0.6, sink: 0.03, lean: 0 };
  }
}

/** grounding + placement context shared by every plan of one manifest. */
interface PlanCtx {
  chunkM: number;
  originX: number;
  originZ: number;
  parentOf: (pcx: number, pcz: number) => Promise<ParentData>;
}

/** the parent-chunk plane LRU (24 chunks ≫ the fine ring's parent count) — each
 *  plan instance owns one. */
function makePlanCtx(source: WorldSource, manifest: WorldManifest): PlanCtx {
  const chunkM = manifest.grid.chunkMeters; // 2048 (parent stride)
  const originX = manifest.grid.originX;
  const originZ = manifest.grid.originZ;
  const cache = new Map<number, Promise<ParentData>>();
  const CACHE_CAP = 24;
  return {
    chunkM,
    originX,
    originZ,
    parentOf: (pcx: number, pcz: number): Promise<ParentData> => {
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
    },
  };
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
  opts: { cellMeters: number; bandDist: number; groundHeightAt?: (x: number, z: number) => number },
): CellPlan {
  const ctx = makePlanCtx(source, manifest);
  let builtUnder = 0;
  let builtDebris = 0;

  return {
    cellMeters: opts.cellMeters,
    bandDist: opts.bandDist,
    originX: ctx.originX,
    originZ: ctx.originZ,
    label: 'uband',
    ...(opts.groundHeightAt ? { groundHeightAt: opts.groundHeightAt } : {}),
    exists(cx: number, cz: number): boolean {
      // a fine cell exists iff its parent 2048 m chunk carries either guidance layer
      const pcx = Math.floor((cx * opts.cellMeters) / ctx.chunkM);
      const pcz = Math.floor((cz * opts.cellMeters) / ctx.chunkM);
      const k = { lod: 0, cx: pcx, cz: pcz };
      return manifest.coverage('understory', k) !== null || manifest.coverage('debris', k) !== null;
    },
    async build(cx: number, cz: number): Promise<ChunkInstances> {
      const C = opts.cellMeters;
      const x0 = ctx.originX + cx * C;
      const z0 = ctx.originZ + cz * C;
      const parent = await ctx.parentOf(Math.floor((x0 - ctx.originX) / ctx.chunkM), Math.floor((z0 - ctx.originZ) / ctx.chunkM));
      const a: number[] = [];
      const b: number[] = [];
      const groundOffsets: number[] | null = opts.groundHeightAt ? [] : null;
      const shared = { a, b, groundOffsets, groundHeightAt: opts.groundHeightAt, x0, z0, C, parent };
      builtUnder += scatterGrid({
        ...shared,
        plane: parent.under,
        step: UNDER_STEP,
        peak: UNDER_PEAK,
        salt: 0x51a3,
        pick: (id, r) => pickClass(map.understory(id), r),
        baseFrac: () => 1,
      });
      builtDebris += scatterGrid({
        ...shared,
        plane: parent.debris,
        step: DEBRIS_STEP,
        peak: DEBRIS_PEAK,
        salt: 0x2c9f,
        pick: (id, r) => pickClass(map.debris(id), r),
        baseFrac: (id) => map.debris(id).base / map.maxDebrisBase,
      });
      return {
        a: Float32Array.from(a),
        b: Float32Array.from(b),
        count: a.length / 4,
        ...(groundOffsets ? { groundOffsets: Float32Array.from(groundOffsets) } : {}),
      };
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
  const layer = (r: typeof u): GuidancePlane | null =>
    r && r.kind === 'planes' ? { res: r.res, id: r.planes[0] as Uint8Array, den: r.planes[1] as Uint8Array } : null;
  return {
    minX: originX + pcx * chunkM,
    minZ: originZ + pcz * chunkM,
    deriver,
    under: layer(u),
    debris: layer(d),
  };
}

/** shared per-cell scatter inputs: the output streams + the fine-cell footprint +
 *  the parent chunk's grounding. */
interface GridCellArgs {
  a: number[];
  b: number[];
  groundOffsets: number[] | null;
  groundHeightAt?: ((x: number, z: number) => number) | undefined;
  /** fine-cell world footprint [x0,x0+C)² */
  x0: number;
  z0: number;
  C: number;
  parent: ParentData;
}

/** walk one global grid (step m) over the fine cell, yielding each owned cell's
 *  jittered position + plane sample. A global grid keeps every instance position-
 *  stable regardless of which fine cell/refetch produces it; each grid cell is
 *  owned by exactly one fine cell (by its unjittered base position), so no
 *  double-emit at cell seams. */
function walkGrid(
  args: GridCellArgs,
  plane: GuidancePlane,
  step: number,
  salt: number,
  visit: (gx: number, gz: number, px: number, pz: number, j0: number, id: number, den: number) => void,
): void {
  const { x0, z0, C, parent } = args;
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
      const res = plane.res;
      const tx = Math.min(res - 1, Math.max(0, Math.round((px - parent.minX) / PLANE_TEXEL_M)));
      const tz = Math.min(res - 1, Math.max(0, Math.round((pz - parent.minZ) / PLANE_TEXEL_M)));
      const den = plane.den[tz * res + tx] as number;
      if (den === 0) continue;
      visit(gx, gz, px, pz, j0, plane.id[tz * res + tx] as number, den);
    }
  }
}

/** append one instance to the cell's {a,b} streams. */
function emit(args: GridCellArgs, px: number, pz: number, scale: number, sink: number, yaw: number, lean: number, idF: number): void {
  const g = args.parent.deriver(px - args.parent.minX, pz - args.parent.minZ);
  args.a.push(px, (args.groundHeightAt ? args.groundHeightAt(px, pz) : g.h) - sink, pz, scale);
  args.b.push(yaw, g.leanX * lean, g.leanZ * lean, idF);
  args.groundOffsets?.push(-sink);
}

/** scatter ONE guidance grid (understory or debris) over the fine cell —
 *  PROBABILISTIC acceptance (density × palette base × peak), the discrete-plant
 *  law. Returns the instance count appended. */
function scatterGrid(
  o: GridCellArgs & {
    plane: GuidancePlane | null;
    step: number;
    peak: number;
    salt: number;
    pick: (id: number, r01: number) => number;
    baseFrac: (id: number) => number;
  },
): number {
  if (!o.plane) return 0;
  let n = 0;
  walkGrid(o, o.plane, o.step, o.salt, (gx, gz, px, pz, j0, id, den) => {
    const [rAcc, rCls] = pcg2d((gx ^ (o.salt + 0x9e37)) >>> 0, (gz ^ (o.salt + 0x85eb)) >>> 0);
    const accept = (den / 255) * o.baseFrac(id) * o.peak;
    if (rAcc >= accept) return;
    const cls = o.pick(id, rCls);
    if (cls < 0) return; // SKIP — groundcover with no mesh
    const [hScale, hVar] = pcg2d((gx ^ (o.salt + 0x27d4)) >>> 0, (gz ^ (o.salt + 0x1b56)) >>> 0);
    const variant = Math.min(3, Math.floor(hVar * 4));
    const { scale, sink, lean } = sizeFor(cls, hScale);
    emit(o, px, pz, scale, sink, j0 * TAU, lean, cls * 8 + variant);
    n++;
  });
  return n;
}
