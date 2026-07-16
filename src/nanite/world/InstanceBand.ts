/**
 * InstanceBand — the streamed near-field instance ring (SPEC-STREAMING-WORLD §5
 * A4, A12; S7 trees/boulders, S9a understory/debris). Its OWN residency ring,
 * independent of the terrain tile clipmap: the cells within `bandDist` of the camera
 * hold their instances as pool slots; cells that leave the exit radius are evicted.
 *
 * ONE ring, TWO content plans (law 6, no fork): the tree/boulder band uses 2048 m
 * data-chunk cells (`treeBoulderPlan` → buildChunkInstances), while the understory/
 * debris band uses a TIGHT sub-chunk cell (`understoryDebrisPlan` → the guidance-plane
 * scatter) because ground cover renders only to ~150 m and chunk-granular residency
 * for it would be a VRAM hog (demand law §5). Both plans emit ABSOLUTE {a,b,idF}
 * words the same buildBlocks → rewriteInstanceBlock binds into the SAME shared pool.
 *
 * WHY main-side (not the brain): the record→instance build is entangled with
 * main-thread library knowledge — SpeciesMap idF, idF→head handles, EtakBoulders'
 * per-class nominal radii, ChunkContent's height-derive — while the record FETCH
 * already decodes off-main in RemoteWorldSource's Lac1 workers. Replicating all of
 * that across the worker boundary for a trivial cell diff is large, fragile surface
 * for no win. The band's async fetch/build stays off the token bucket; only the GPU
 * block writes are bucketed (rewriteInstanceBlock), FIFO with the brain's packet drain.
 */

import type { GeometryRegistry } from './GeometryRegistry';
import { buildChunkInstances, type ChunkInstances } from './ChunkContent';
import type { WorldManifest, WorldSource } from '../../world/source/WorldSource';
import { MICRO_MORPH_BANDS } from './TerrainMorph';

/** the residency-grain + content contract for one band. Cells are integer (cx,cz)
 *  over a grid of `cellMeters` anchored at (originX,originZ); `build` produces the
 *  cell's flat instance list (ABSOLUTE game space). */
export interface CellPlan {
  cellMeters: number;
  /** enter radius (m) a cell must reach to load; exit = this + cellMeters/2. */
  bandDist: number;
  originX: number;
  originZ: number;
  /** counter prefix (e.g. 'band', 'uband') — keeps the two bands' HUD keys distinct. */
  label: string;
  /** does a cell exist (has any streamable content)? */
  exists(cx: number, cz: number): boolean;
  /** build one cell's flat instance list. */
  build(cx: number, cz: number): Promise<ChunkInstances>;
  /** Packed finest/morph surface. Absent on old manifests, which retain their
   *  original record-key grounding and incur no refresh work. */
  groundHeightAt?: (x: number, z: number) => number;
  /** optional extra HUD counters (e.g. the understory/debris split). */
  extra?(): Record<string, number>;
}

/** one block ready to write: head-resolved, ≤ blockSize instances. */
interface ReadyBlock {
  count: number;
  a: Float32Array;
  b: Float32Array;
  meshIds: Uint32Array;
  groundOffsets?: Float32Array;
  dirtyFirst?: number;
  dirtyLast?: number;
}
interface ReadyChunk {
  key: string;
  cx: number;
  cz: number;
  blocks: ReadyBlock[];
}
interface ResidentChunk {
  cx: number;
  cz: number;
  blocks: { slot: number; data: ReadyBlock }[];
  instCount: number;
}

export interface InstanceBandDeps {
  /** the residency-grain + content plan (tree/boulder or understory/debris). */
  plan: CellPlan;
  /** idF → EVERY head that renders on this instance (bark trunk + near leaf crown +
   *  mid/far voxel crown for trees; the single mesh head for a shrub/rock/deadwood).
   *  Empty ⇒ instance dropped. A tree consumes 3 pool slots, a ground instance 1. */
  headsOf: (idF: number) => number[];
  reg: GeometryRegistry;
}

const KEY = (cx: number, cz: number): string => `${cx}:${cz}`;
const GROUND_MORPH_OUTER_M = Math.max(MICRO_MORPH_BANDS[-2].outerM, MICRO_MORPH_BANDS[-1].outerM);

export class InstanceBand {
  private readonly d: InstanceBandDeps;
  private readonly blockSize: number;
  private readonly cellM: number;
  private readonly originX: number;
  private readonly originZ: number;
  private readonly label: string;

  private readonly resident = new Map<string, ResidentChunk>();
  private readonly inflight = new Set<string>();
  private readonly ready: ReadyChunk[] = [];
  private wanted = new Set<string>();
  private camX = 0;
  private camZ = 0;
  private lastDiffAt = -1e9;
  private dropWarned = false;
  private haveGroundCenter = false;
  private groundCenterX = 0;
  private groundCenterZ = 0;

  private nDropped = 0;
  private nLoaded = 0;
  private nEvicted = 0;
  private nResidentInst = 0;

  constructor(deps: InstanceBandDeps) {
    this.d = deps;
    this.blockSize = deps.reg.instancePoolBlockSize;
    if (this.blockSize <= 0) throw new Error('InstanceBand: registry has no instance pool reserved');
    this.cellM = deps.plan.cellMeters;
    this.originX = deps.plan.originX;
    this.originZ = deps.plan.originZ;
    this.label = deps.plan.label;
  }

  /** cell footprint → nearest-point distance to (px,pz). */
  private chunkDist(cx: number, cz: number, px: number, pz: number): number {
    const x0 = this.originX + cx * this.cellM;
    const z0 = this.originZ + cz * this.cellM;
    const dx = px < x0 ? x0 - px : px > x0 + this.cellM ? px - (x0 + this.cellM) : 0;
    const dz = pz < z0 ? z0 - pz : pz > z0 + this.cellM ? pz - (z0 + this.cellM) : 0;
    return Math.hypot(dx, dz);
  }

  /** pose-driven residency diff (throttled ~5 Hz). */
  update(camX: number, camZ: number): void {
    this.camX = camX;
    this.camZ = camZ;
    this.refreshGround(camX, camZ);
    const now = performance.now();
    if (now - this.lastDiffAt < 200) return;
    this.lastDiffAt = now;

    // desired = existing cells within bandDist of the camera
    const enter = this.d.plan.bandDist;
    const exit = enter + this.cellM * 0.5; // ½-cell hysteresis
    const c0x = Math.floor((camX - enter - this.originX) / this.cellM);
    const c1x = Math.floor((camX + enter - this.originX) / this.cellM);
    const c0z = Math.floor((camZ - enter - this.originZ) / this.cellM);
    const c1z = Math.floor((camZ + enter - this.originZ) / this.cellM);
    const want = new Set<string>();
    for (let cz = c0z; cz <= c1z; cz++) {
      for (let cx = c0x; cx <= c1x; cx++) {
        if (this.chunkDist(cx, cz, camX, camZ) > enter) continue;
        if (!this.d.plan.exists(cx, cz)) continue;
        const key = KEY(cx, cz);
        want.add(key);
        if (this.resident.has(key) || this.inflight.has(key) || this.ready.some((r) => r.key === key)) continue;
        this.inflight.add(key);
        void this.load(cx, cz, key);
      }
    }
    this.wanted = want;

    // evict resident cells past the exit radius (hysteresis keeps the band steady)
    for (const [key, rc] of this.resident) {
      if (want.has(key)) continue;
      if (this.chunkDist(rc.cx, rc.cz, camX, camZ) <= exit) continue;
      for (const b of rc.blocks) this.d.reg.freeInstanceBlock(b.slot);
      this.nResidentInst -= rc.instCount;
      this.resident.delete(key);
      this.nEvicted++;
    }
  }

  private async load(cx: number, cz: number, key: string): Promise<void> {
    try {
      const inst = await this.d.plan.build(cx, cz);
      // if the cell left the band while loading, drop the result
      if (!this.wanted.has(key)) {
        this.inflight.delete(key);
        return;
      }
      const blocks = this.buildBlocks(inst);
      this.inflight.delete(key);
      if (blocks.length === 0) return;
      this.ready.push({ key, cx, cz, blocks });
    } catch (e) {
      this.inflight.delete(key);
      // eslint-disable-next-line no-console
      console.warn(`[laas][${this.label}] cell ${key} load failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  /** head-resolve + drop head-less instances + split into ≤ blockSize blocks. */
  private buildBlocks(inst: ChunkInstances): ReadyBlock[] {
    const bs = this.blockSize;
    const blocks: ReadyBlock[] = [];
    let a: number[] = [];
    let b: number[] = [];
    let m: number[] = [];
    let g: number[] = [];
    const flush = (): void => {
      if (m.length === 0) return;
      blocks.push({
        count: m.length,
        a: Float32Array.from(a),
        b: Float32Array.from(b),
        meshIds: Uint32Array.from(m),
        ...(inst.groundOffsets ? { groundOffsets: Float32Array.from(g) } : {}),
      });
      a = [];
      b = [];
      m = [];
      g = [];
    };
    for (let i = 0; i < inst.count; i++) {
      const idF = inst.b[i * 4 + 3] as number;
      const heads = this.d.headsOf(idF);
      if (heads.length === 0) {
        this.nDropped++;
        continue;
      }
      // one pool slot per rendering head (trunk + crown for a tree), same A/B words
      for (const head of heads) {
        a.push(inst.a[i * 4] as number, inst.a[i * 4 + 1] as number, inst.a[i * 4 + 2] as number, inst.a[i * 4 + 3] as number);
        b.push(inst.b[i * 4] as number, inst.b[i * 4 + 1] as number, inst.b[i * 4 + 2] as number, idF);
        m.push(head);
        if (inst.groundOffsets) g.push(inst.groundOffsets[i] as number);
        if (m.length === bs) flush();
      }
    }
    flush();
    return blocks;
  }

  /** write ready blocks under the token bucket (returns bytes written). Allocates a
   *  block per write; on a dry pool evicts the farthest non-wanted resident cell. */
  drainBudget(bytesRemaining: number): number {
    let used = 0;
    while (this.ready.length > 0 && used < bytesRemaining) {
      const rc = this.ready[0] as ReadyChunk;
      const blk = rc.blocks.shift() as ReadyBlock;
      let slot = this.d.reg.allocInstanceBlock();
      if (slot < 0) slot = this.evictFarthest();
      if (slot < 0) {
        if (!this.dropWarned) {
          this.dropWarned = true;
          // eslint-disable-next-line no-console
          console.warn(`[laas][${this.label}] instance pool full of wanted cells — raise the pool block count`);
        }
        rc.blocks.length = 0; // drop the rest of this cell
      } else {
        this.d.reg.rewriteInstanceBlock(slot, blk.count, blk.a, blk.b, blk.meshIds);
        const res = this.resident.get(rc.key) ?? { cx: rc.cx, cz: rc.cz, blocks: [], instCount: 0 };
        res.blocks.push({ slot, data: blk });
        res.instCount += blk.count;
        this.resident.set(rc.key, res);
        this.nResidentInst += blk.count;
        used += blk.count * 32 + 64;
      }
      if (rc.blocks.length === 0) {
        this.ready.shift();
        this.nLoaded++;
      }
    }
    for (const rc of this.resident.values()) {
      for (const block of rc.blocks) {
        const first = block.data.dirtyFirst;
        const last = block.data.dirtyLast;
        if (first === undefined || last === undefined) continue;
        const bytes = (last - first + 1) * 32 + 64;
        if (used > 0 && used + bytes > bytesRemaining) return used;
        this.d.reg.rewriteInstanceBlockGround(block.slot, first, last - first + 1, block.data.a);
        delete block.data.dirtyFirst;
        delete block.data.dirtyLast;
        used += bytes;
      }
    }
    return used;
  }

  /** Refresh only the union of the old/new packed-morph envelopes. Outside that
   *  union the camera morph weight is unchanged, so touching those instances
   *  would waste CPU/upload bandwidth. Dirty ranges drain through the same token
   *  bucket as initial instance writes. */
  private refreshGround(camX: number, camZ: number): void {
    const heightAt = this.d.plan.groundHeightAt;
    if (!heightAt) return;
    const oldX = this.groundCenterX;
    const oldZ = this.groundCenterZ;
    const hadOld = this.haveGroundCenter;
    this.groundCenterX = camX;
    this.groundCenterZ = camZ;
    this.haveGroundCenter = true;

    const refresh = (block: ReadyBlock, resident: boolean): void => {
      const offsets = block.groundOffsets;
      if (!offsets) return;
      let first = block.count;
      let last = -1;
      for (let i = 0; i < block.count; i++) {
        const offset = offsets[i] as number;
        if (!Number.isFinite(offset)) continue;
        const d = i * 4;
        const x = block.a[d] as number;
        const z = block.a[d + 2] as number;
        const inNew = Math.max(Math.abs(x - camX), Math.abs(z - camZ)) <= GROUND_MORPH_OUTER_M;
        const inOld = hadOld && Math.max(Math.abs(x - oldX), Math.abs(z - oldZ)) <= GROUND_MORPH_OUTER_M;
        if (!inNew && !inOld) continue;
        const y = Math.fround(heightAt(x, z) + offset);
        if ((block.a[d + 1] as number) === y) continue;
        block.a[d + 1] = y;
        first = Math.min(first, i);
        last = i;
      }
      if (!resident || last < first) return;
      block.dirtyFirst = Math.min(block.dirtyFirst ?? first, first);
      block.dirtyLast = Math.max(block.dirtyLast ?? last, last);
    };

    for (const chunk of this.ready) for (const block of chunk.blocks) refresh(block, false);
    for (const chunk of this.resident.values()) for (const block of chunk.blocks) refresh(block.data, true);
  }

  /** evict the farthest non-wanted resident cell → free a block; -1 if none. */
  private evictFarthest(): number {
    let worstKey: string | null = null;
    let worstD = -1;
    for (const [key, rc] of this.resident) {
      if (this.wanted.has(key)) continue;
      const dd = this.chunkDist(rc.cx, rc.cz, this.camX, this.camZ);
      if (dd > worstD) {
        worstD = dd;
        worstKey = key;
      }
    }
    if (worstKey === null) return -1;
    const rc = this.resident.get(worstKey) as ResidentChunk;
    for (const b of rc.blocks) this.d.reg.freeInstanceBlock(b.slot);
    this.nResidentInst -= rc.instCount;
    this.resident.delete(worstKey);
    this.nEvicted++;
    return this.d.reg.allocInstanceBlock();
  }

  counters(): Record<string, number> {
    const p = this.label;
    return {
      [`${p}.chunks.resident`]: this.resident.size,
      [`${p}.chunks.inflight`]: this.inflight.size,
      [`${p}.chunks.ready`]: this.ready.length,
      [`${p}.inst.resident`]: this.nResidentInst,
      [`${p}.blocks.used`]: this.d.reg.instancePoolBlockCount - this.d.reg.instancePoolFreeBlocks,
      [`${p}.blocks.free`]: this.d.reg.instancePoolFreeBlocks,
      [`${p}.loaded`]: this.nLoaded,
      [`${p}.evicted`]: this.nEvicted,
      [`${p}.dropped`]: this.nDropped,
      ...(this.d.plan.extra ? this.d.plan.extra() : {}),
    };
  }
}

/** the S7 tree/boulder plan: 2048 m data-chunk cells → buildChunkInstances (trees +
 *  ETAK boulders). Estonia resolves species via SpeciesMap; boulder radii via lib. */
export function treeBoulderPlan(
  source: WorldSource,
  manifest: WorldManifest,
  opts: {
    idFOf: (species: number, variant: number) => number;
    boulderRadiusOf: (cls: number) => number;
    bandDist: number;
    groundHeightAt?: (x: number, z: number) => number;
  },
): CellPlan {
  return {
    cellMeters: manifest.grid.chunkMeters,
    bandDist: opts.bandDist,
    originX: manifest.grid.originX,
    originZ: manifest.grid.originZ,
    label: 'band',
    ...(opts.groundHeightAt ? { groundHeightAt: opts.groundHeightAt } : {}),
    exists(cx: number, cz: number): boolean {
      const key = { lod: 0, cx, cz };
      return manifest.coverage('trees', key) !== null || manifest.coverage('boulders', key) !== null;
    },
    build(cx: number, cz: number): Promise<ChunkInstances> {
      return buildChunkInstances(source, manifest, { lod: 0, cx, cz }, {
        idFOf: opts.idFOf,
        boulderRadiusOf: opts.boulderRadiusOf,
        ...(opts.groundHeightAt ? { groundHeightAt: opts.groundHeightAt } : {}),
      });
    },
  };
}
