/**
 * InstanceBand — the streamed tree/boulder instance band (SPEC-STREAMING-WORLD §5
 * A4, A12; S7). Its OWN residency ring, independent of the terrain tile clipmap:
 * the LOD0 chunks within ~a near radius of the camera (the "2×2 + ½-chunk
 * hysteresis" band) hold their trees + ETAK boulders as pool instances; chunks
 * that leave the exit radius are evicted (parked). Beyond the band trees pop —
 * S8 fartiles cover the mid/far field (noted for S8).
 *
 * WHY main-side (not the brain): the record→instance build is entangled with
 * main-thread library knowledge — SpeciesMap idF, idF→head handles, EtakBoulders'
 * per-class nominal radii, ChunkContent's height-derive — while the record FETCH
 * already decodes off-main in RemoteWorldSource's Lac1 workers. Replicating all of
 * that across the worker boundary for a trivial 2×2 chunk diff is large, fragile
 * surface for no win (go-up-a-level: the demand law wants band-only FETCHES + off-
 * frame processing, both satisfied here). The band's async fetch/build stays off the
 * token bucket; only the GPU block writes are bucketed (rewriteInstanceBlock), FIFO
 * with the brain's own packet drain.
 */

import type { GeometryRegistry } from './GeometryRegistry';
import { buildChunkInstances, type ChunkInstances } from './ChunkContent';
import type { ChunkKey, WorldManifest, WorldSource } from '../../world/source/WorldSource';

/** one block ready to write: head-resolved, ≤ blockSize instances. */
interface ReadyBlock {
  count: number;
  a: Float32Array;
  b: Float32Array;
  meshIds: Uint32Array;
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
  blocks: number[];
  instCount: number;
}

export interface InstanceBandDeps {
  source: WorldSource;
  manifest: WorldManifest;
  /** trees (species, variant) → library idF (SpeciesMap on Estonia). */
  idFOf: (species: number, variant: number) => number;
  /** rock class → nominal radius (lib.clsRadius) — enables ETAK boulder ingest. */
  boulderRadiusOf: (cls: number) => number;
  /** idF → EVERY head that renders on this instance (bark trunk + near leaf crown +
   *  mid/far voxel crown for trees; the single rock head for boulders). Empty ⇒
   *  instance dropped. A tree therefore consumes 3 pool slots (trunk + leaf + voxel),
   *  a boulder 1. */
  headsOf: (idF: number) => number[];
  reg: GeometryRegistry;
  /** near radius (m) a chunk must reach to load; exit = this + ½ chunk (hysteresis). */
  bandDist: number;
}

const KEY = (cx: number, cz: number): string => `${cx}:${cz}`;

export class InstanceBand {
  private readonly d: InstanceBandDeps;
  private readonly blockSize: number;
  private readonly chunkM: number;
  private readonly originX: number;
  private readonly originZ: number;

  private readonly resident = new Map<string, ResidentChunk>();
  private readonly inflight = new Set<string>();
  private readonly ready: ReadyChunk[] = [];
  private wanted = new Set<string>();
  private camX = 0;
  private camZ = 0;
  private lastDiffAt = -1e9;
  private dropWarned = false;

  private nDropped = 0;
  private nLoaded = 0;
  private nEvicted = 0;
  private nResidentInst = 0;

  constructor(deps: InstanceBandDeps) {
    this.d = deps;
    this.blockSize = deps.reg.instancePoolBlockSize;
    if (this.blockSize <= 0) throw new Error('InstanceBand: registry has no instance pool reserved');
    this.chunkM = deps.manifest.grid.chunkMeters; // LOD0 stride
    this.originX = deps.manifest.grid.originX;
    this.originZ = deps.manifest.grid.originZ;
  }

  /** chunk footprint → nearest-point distance to (px,pz). */
  private chunkDist(cx: number, cz: number, px: number, pz: number): number {
    const x0 = this.originX + cx * this.chunkM;
    const z0 = this.originZ + cz * this.chunkM;
    const dx = px < x0 ? x0 - px : px > x0 + this.chunkM ? px - (x0 + this.chunkM) : 0;
    const dz = pz < z0 ? z0 - pz : pz > z0 + this.chunkM ? pz - (z0 + this.chunkM) : 0;
    return Math.hypot(dx, dz);
  }

  private chunkExists(cx: number, cz: number): boolean {
    const key: ChunkKey = { lod: 0, cx, cz };
    return this.d.manifest.coverage('trees', key) !== null || this.d.manifest.coverage('boulders', key) !== null;
  }

  /** pose-driven residency diff (throttled ~5 Hz). */
  update(camX: number, camZ: number): void {
    this.camX = camX;
    this.camZ = camZ;
    const now = performance.now();
    if (now - this.lastDiffAt < 200) return;
    this.lastDiffAt = now;

    // desired = existing LOD0 chunks within bandDist of the camera
    const enter = this.d.bandDist;
    const exit = enter + this.chunkM * 0.5; // ½-chunk hysteresis
    const c0x = Math.floor((camX - enter - this.originX) / this.chunkM);
    const c1x = Math.floor((camX + enter - this.originX) / this.chunkM);
    const c0z = Math.floor((camZ - enter - this.originZ) / this.chunkM);
    const c1z = Math.floor((camZ + enter - this.originZ) / this.chunkM);
    const want = new Set<string>();
    for (let cz = c0z; cz <= c1z; cz++) {
      for (let cx = c0x; cx <= c1x; cx++) {
        if (this.chunkDist(cx, cz, camX, camZ) > enter) continue;
        if (!this.chunkExists(cx, cz)) continue;
        const key = KEY(cx, cz);
        want.add(key);
        if (this.resident.has(key) || this.inflight.has(key) || this.ready.some((r) => r.key === key)) continue;
        this.inflight.add(key);
        void this.load(cx, cz, key);
      }
    }
    this.wanted = want;

    // evict resident chunks past the exit radius (hysteresis keeps the band steady)
    for (const [key, rc] of this.resident) {
      if (want.has(key)) continue;
      if (this.chunkDist(rc.cx, rc.cz, camX, camZ) <= exit) continue;
      for (const b of rc.blocks) this.d.reg.freeInstanceBlock(b);
      this.nResidentInst -= rc.instCount;
      this.resident.delete(key);
      this.nEvicted++;
    }
  }

  private async load(cx: number, cz: number, key: string): Promise<void> {
    try {
      const inst = await buildChunkInstances(this.d.source, this.d.manifest, { lod: 0, cx, cz }, {
        idFOf: this.d.idFOf,
        boulderRadiusOf: this.d.boulderRadiusOf,
      });
      // if the chunk left the band while loading, drop the result
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
      console.warn(`[laas][band] chunk ${key} load failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  /** head-resolve + drop head-less instances + split into ≤ blockSize blocks. */
  private buildBlocks(inst: ChunkInstances): ReadyBlock[] {
    const bs = this.blockSize;
    const blocks: ReadyBlock[] = [];
    let a: number[] = [];
    let b: number[] = [];
    let m: number[] = [];
    const flush = (): void => {
      if (m.length === 0) return;
      blocks.push({ count: m.length, a: Float32Array.from(a), b: Float32Array.from(b), meshIds: Uint32Array.from(m) });
      a = [];
      b = [];
      m = [];
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
        if (m.length === bs) flush();
      }
    }
    flush();
    return blocks;
  }

  /** write ready blocks under the token bucket (returns bytes written). Allocates a
   *  block per write; on a dry pool evicts the farthest non-wanted resident chunk. */
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
          console.warn('[laas][band] instance pool full of wanted chunks — raise the pool block count');
        }
        rc.blocks.length = 0; // drop the rest of this chunk
      } else {
        this.d.reg.rewriteInstanceBlock(slot, blk.count, blk.a, blk.b, blk.meshIds);
        const res = this.resident.get(rc.key) ?? { cx: rc.cx, cz: rc.cz, blocks: [], instCount: 0 };
        res.blocks.push(slot);
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
    return used;
  }

  /** evict the farthest non-wanted resident chunk → free a block; -1 if none. */
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
    for (const b of rc.blocks) this.d.reg.freeInstanceBlock(b);
    this.nResidentInst -= rc.instCount;
    this.resident.delete(worstKey);
    this.nEvicted++;
    return this.d.reg.allocInstanceBlock();
  }

  counters(): Record<string, number> {
    return {
      'band.chunks.resident': this.resident.size,
      'band.chunks.inflight': this.inflight.size,
      'band.chunks.ready': this.ready.length,
      'band.inst.resident': this.nResidentInst,
      'band.blocks.used': this.d.reg.instancePoolBlockCount - this.d.reg.instancePoolFreeBlocks,
      'band.blocks.free': this.d.reg.instancePoolFreeBlocks,
      'band.loaded': this.nLoaded,
      'band.evicted': this.nEvicted,
      'band.dropped': this.nDropped,
    };
  }
}
