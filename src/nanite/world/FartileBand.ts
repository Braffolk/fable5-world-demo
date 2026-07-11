/**
 * FartileBand — the runtime FAR-TILE residency (SPEC §5 A2/F-2, S8), brain-side
 * (law 5: the splat/emit/pyramid — the 4.4 ms/tile cost — runs in the StreamBrain
 * worker, never the main thread; main only writes GPU records under the token
 * bucket). It replaces the boot all-resident fartile build on BOTH sources: the
 * subsystem never knows which source feeds it (law 3).
 *
 * RESIDENCY UNIT = a `cellMeters` (512 m) CELL, ring-GRADED (F-2, the demand law
 * applied to bricks): a cell's bake cellSize steps up (coarser bricks) with its ring
 * distance, so far cells cost ~4× fewer bricks per doubling — that grading, not the
 * pool alone, is what makes S8 a NET VRAM CUT (an ungraded ring over the world is
 * the same brick total as the boot build, just pooled). A cell re-bakes only when
 * its grade FLIPS (rare — ring boundaries are hundreds of metres apart with
 * hysteresis), so the far forest is effectively static.
 *
 * TX-OWNS-BAKE: each cell bake carries a generation token; a result whose cell was
 * evicted / re-graded mid-bake has no home and is dropped (illegal double-attach
 * unrepresentable). The brain owns the slot + granule free-lists (single residency
 * authority); a dry pool is a throw-loud provisioning bug, never backpressure.
 *
 * A bake is a TOTAL function to residency: it ALWAYS lands the cell in `resident`
 * (0..N slots — a clearing / forest-edge cell bakes to 0 slots, a TERMINAL empty
 * state) unless SUPERSEDED by a mid-flight re-grade eviction (rolled back). A cell
 * that bakes forever — perpetually re-nominated because it never reached residency —
 * is thus UNREPRESENTABLE (the pre-S8b-3 `baking=1` stuck-cell bug).
 */

import type { ChunkKey, ChunkPayload, LayerName } from '../../world/source/WorldSource';
import { makeGroundDeriver } from '../../world/source/RecordGround';
import { planFarTilesFlat, emitTile, packFarTiles } from './FarTilesCore';
import { splatTiles, type SplatPoolFlat, type TileSplatJob } from './FarTilesSplat';
import type { PackedLevel } from '../build/CrownPack';
import type { FtArmMsg, StreamPacket } from './StreamProtocol';

/** the grid geometry the band needs (a subset of WorldGrid + the LOD0 tree layer). */
export interface FtGrid {
  originX: number;
  originZ: number;
  chunkMeters: number; // LOD0 chunk footprint (trees are LOD0-only on both sources)
}

export interface FartileBandDeps {
  fetch(layer: LayerName, key: ChunkKey): Promise<ChunkPayload | null>;
  emit(packet: StreamPacket, transfer: Transferable[]): void;
  /** does a LOD0 tree chunk exist? (authoritative absence — the brain's key set). */
  treesExist(cx: number, cz: number): boolean;
  grid: FtGrid;
  /** cooperative yield so a multi-tile cell bake never freezes the brain thread. */
  yield(): Promise<void>;
}

interface SpeciesFlat {
  bricks: Float32Array; // stride 11 (FarTilesSplat SPECIES_BRICK_STRIDE), crown-local
  crownMinY: number;
  barkR: number;
  barkG: number;
  barkB: number;
}

interface ResidentCell {
  slots: number[];
  granules: number[];
  grade: number; // grade index (cellSizes/gradeRadii index)
  gen: number;
}

const CELL_KEY = (fcx: number, fcz: number): number => ((fcx & 0xffff) << 16) | (fcz & 0xffff);
/** grade re-evaluation hysteresis (m) past a ring boundary before a re-bake fires.
 *  LARGE (½ a 512 m cell): a re-grade re-splats a whole cell (expensive), so a cell must
 *  move well past a ring boundary before it flips — this bounds the re-grade churn that
 *  otherwise sweeps the whole (small) generated world on every camera move. */
const GRADE_HYST = 256;
/** cell bakes started per pose tick (bounds concurrent brain-thread splat work). */
const MAX_BAKES_PER_TICK = 2;

export class FartileBand {
  private armed = false;
  private cfg!: FtArmMsg;
  private pools = new Map<number, SpeciesFlat>();
  private grid!: FtGrid;
  private readonly deps: FartileBandDeps;

  private slotFree: number[] = [];
  private granuleFree: number[] = [];
  private readonly resident = new Map<number, ResidentCell>();
  private readonly baking = new Set<number>();
  private cellGen = 1;

  // per-2048 m tree-chunk decode cache (a chunk feeds 16 cells) — cleared when idle
  private chunkCache = new Map<string, { trees: ChunkPayload | null; height: ChunkPayload | null }>();

  private camX = 0;
  private camZ = 0;
  private nBaked = 0;
  private nEvicted = 0;
  private nRebaked = 0;
  private nHeads = 0;
  private nMaxTileClusters = 0; // peak clusters emitted by ANY one tile — sets the clusterCap floor

  constructor(deps: FartileBandDeps) {
    this.deps = deps;
  }

  arm(msg: FtArmMsg): void {
    this.cfg = msg;
    this.grid = this.deps.grid;
    for (const p of msg.pools) {
      this.pools.set(p.idF, { bricks: p.bricks, crownMinY: p.crownMinY, barkR: p.barkR, barkG: p.barkG, barkB: p.barkB });
    }
    this.slotFree = [];
    for (let s = msg.slots - 1; s >= 0; s--) this.slotFree.push(s);
    this.granuleFree = [];
    for (let g = msg.granules - 1; g >= 0; g--) this.granuleFree.push(g);
    this.armed = true;
  }

  /** ring distance (m) from the camera to a cell's box nearest point (Chebyshev). */
  private cellDist(fcx: number, fcz: number): number {
    const cm = this.cfg.cellMeters;
    const x0 = this.grid.originX + fcx * cm;
    const z0 = this.grid.originZ + fcz * cm;
    const dx = this.camX < x0 ? x0 - this.camX : this.camX > x0 + cm ? this.camX - (x0 + cm) : 0;
    const dz = this.camZ < z0 ? z0 - this.camZ : this.camZ > z0 + cm ? this.camZ - (z0 + cm) : 0;
    return Math.max(dx, dz);
  }

  /** grade index for a ring distance (first ladder rung it fits under; coarsest tail). */
  private gradeFor(dist: number): number {
    const r = this.cfg.gradeRadii;
    for (let i = 0; i < r.length; i++) if (dist < (r[i] as number)) return i;
    return this.cfg.cellSizes.length - 1;
  }

  /** does a cell keep its current grade under hysteresis? (avoid ring-boundary thrash). */
  private gradeStable(dist: number, cur: number): boolean {
    const r = this.cfg.gradeRadii;
    // the band for grade `cur` is [radii[cur-1], radii[cur]) — stay unless past it by GRADE_HYST
    const lo = cur > 0 ? (r[cur - 1] as number) - GRADE_HYST : -Infinity;
    const hi = cur < r.length ? (r[cur] as number) + GRADE_HYST : Infinity;
    return dist >= lo && dist < hi;
  }

  pose(camX: number, camZ: number): void {
    if (!this.armed) return;
    this.camX = camX;
    this.camZ = camZ;
    const cm = this.cfg.cellMeters;
    const horizon = this.cfg.horizon;
    const exit = horizon + cm * 0.5; // hysteresis
    const c0x = Math.floor((camX - horizon - this.grid.originX) / cm);
    const c1x = Math.floor((camX + horizon - this.grid.originX) / cm);
    const c0z = Math.floor((camZ - horizon - this.grid.originZ) / cm);
    const c1z = Math.floor((camZ + horizon - this.grid.originZ) / cm);
    const want = new Set<number>();
    const toBake: { fcx: number; fcz: number; grade: number; key: number }[] = [];
    for (let fcz = c0z; fcz <= c1z; fcz++) {
      for (let fcx = c0x; fcx <= c1x; fcx++) {
        const dist = this.cellDist(fcx, fcz);
        if (dist > horizon) continue;
        // parent LOD0 tree chunk must exist
        const worldX = this.grid.originX + (fcx + 0.5) * cm;
        const worldZ = this.grid.originZ + (fcz + 0.5) * cm;
        const pcx = Math.floor((worldX - this.grid.originX) / this.grid.chunkMeters);
        const pcz = Math.floor((worldZ - this.grid.originZ) / this.grid.chunkMeters);
        if (!this.deps.treesExist(pcx, pcz)) continue;
        const key = CELL_KEY(fcx, fcz);
        want.add(key);
        const res = this.resident.get(key);
        const grade = this.gradeFor(dist);
        if (res) {
          if (!this.gradeStable(dist, res.grade) && grade !== res.grade && !this.baking.has(key)) {
            toBake.push({ fcx, fcz, grade, key }); // re-grade
          }
        } else if (!this.baking.has(key)) {
          toBake.push({ fcx, fcz, grade, key });
        }
      }
    }
    // evict cells past the exit radius (or off-horizon)
    for (const [key, rc] of this.resident) {
      const fcx = this.s16((key >> 16) & 0xffff);
      const fcz = this.s16(key & 0xffff);
      if (want.has(key) && this.cellDist(fcx, fcz) <= exit) continue;
      this.evictCell(key, rc);
    }
    // nearest-first bakes, budgeted
    toBake.sort((a, b) => this.cellDist(a.fcx, a.fcz) - this.cellDist(b.fcx, b.fcz));
    let started = 0;
    for (const t of toBake) {
      if (started >= MAX_BAKES_PER_TICK) break;
      this.baking.add(t.key);
      started++;
      void this.bakeCell(t.fcx, t.fcz, t.grade, t.key);
    }
    if (this.baking.size === 0 && this.chunkCache.size > 0) this.chunkCache.clear();
  }

  private s16(v: number): number {
    return (v << 16) >> 16;
  }

  private evictCell(key: number, rc: ResidentCell): void {
    rc.gen = -1; // invalidate any in-flight re-bake token match
    if (rc.slots.length > 0) {
      // an EMPTY resident cell (a clearing baked to 0 slots) holds no GPU records — skip
      // the (empty) evict packet + free-list churn; still drop it from residency.
      this.deps.emit({ kind: 'ftEvict', slots: rc.slots.slice() }, []);
      for (const s of rc.slots) this.slotFree.push(s);
      for (const g of rc.granules) this.granuleFree.push(g);
      this.nHeads -= rc.slots.length;
    }
    this.resident.delete(key);
    this.nEvicted++;
  }

  private async fetchChunkPair(pcx: number, pcz: number): Promise<{ trees: ChunkPayload | null; height: ChunkPayload | null }> {
    const ck = `${pcx}:${pcz}`;
    const hit = this.chunkCache.get(ck);
    if (hit) return hit;
    const key: ChunkKey = { lod: 0, cx: pcx, cz: pcz };
    const [trees, height] = await Promise.all([this.deps.fetch('trees', key), this.deps.fetch('height', key)]);
    const pair = { trees, height };
    this.chunkCache.set(ck, pair);
    return pair;
  }

  /** bake ONE cell: assemble its trees, plan at the ring grade, splat+emit+pack each tile,
   *  then RESERVE THE WHOLE CELL's slots+granules AT ONCE (or defer it cleanly if the pool
   *  is transiently full) and emit ftAttach. Interleaved so the brain stays live.
   *
   *  RESERVE-WHOLE-CELL (law 9 — the partial-cell/leak is unrepresentable): the free-lists
   *  are touched only AFTER the whole cell is proven to fit; a dry pool or a mid-flight
   *  eviction drops the ENTIRE cell with nothing popped, so no tile emits without its cell
   *  and no slot/granule ever leaks (the pre-#109 mid-loop-throw stranded the slots already
   *  popped for the cell's earlier tiles).
   *
   *  A bake is a TOTAL function to residency (law 9 — no stranded-bake state): a cell
   *  with no pooled-species trees (a clearing / forest edge) or no plannable content
   *  bakes to an EMPTY resident cell (0 slots) — a TERMINAL state, so pose() never
   *  re-nominates it (the pre-fix `flats.length===0 → return-without-resident` left the
   *  cell forever re-baking, the `baking=1` stuck-cell bug). The two non-commit outcomes
   *  both drop the cell with NOTHING reserved (no leak): a bake SUPERSEDED by a mid-flight
   *  re-grade eviction (its predecessor's gen went −1, no home), and a DEFER when the pool
   *  is only transiently full (re-nominated next tick). */
  private async bakeCell(fcx: number, fcz: number, grade: number, key: number): Promise<void> {
    const myGen = this.cellGen++;
    try {
      const cm = this.cfg.cellMeters;
      const cellX0 = this.grid.originX + fcx * cm;
      const cellZ0 = this.grid.originZ + fcz * cm;
      const margin = this.cfg.reachMargin;
      const pcx = Math.floor((cellX0 + cm * 0.5 - this.grid.originX) / this.grid.chunkMeters);
      const pcz = Math.floor((cellZ0 + cm * 0.5 - this.grid.originZ) / this.grid.chunkMeters);
      const { trees, height } = await this.fetchChunkPair(pcx, pcz);

      // group this cell's trees (+reach margin) by idF → flat splat pools. No records /
      // no pooled species in the box ⇒ NO pools ⇒ an empty (0-slot) bake below.
      const flats =
        trees && trees.kind === 'records'
          ? this.buildCellPools(trees, height, pcx, pcz, cellX0, cellZ0, cm, margin)
          : [];
      const cellSize = this.cfg.cellSizes[grade] as number;
      const plan =
        flats.length > 0
          ? planFarTilesFlat({
              tileSize: this.cfg.tileSize,
              cellSize,
              pools: flats,
              bounds: { mnX: cellX0, mnZ: cellZ0, mxX: cellX0 + cm - 1, mxZ: cellZ0 + cm - 1 },
            })
          : null;

      // an in-flight re-grade replaces an existing resident cell: evict the old set
      // ONLY once the new bake is ready to attach (no uncover). Snapshot the old here.
      const old = this.resident.get(key);

      // PASS 1 — build EVERY tile (splat → emit → pack) WITHOUT touching the free-lists. The
      // heavy per-tile work (law 5: ~4 ms/tile) stays yield-interleaved; nothing is reserved
      // yet, so a dry pool or a mid-flight eviction drops the WHOLE cell with no partial emit
      // and no leaked slots/granules.
      const tiles: { levels: PackedLevel[]; center: [number, number, number]; nClusters: number }[] = [];
      let needGranules = 0;
      if (plan) {
        for (let j = 0; j < plan.jobs.length; j++) {
          const splatted = splatTiles(plan.grid, plan.poolsFlat, [plan.jobs[j] as TileSplatJob]);
          const r = splatted[0];
          if (!r) continue;
          const built = emitTile(plan.grid, r);
          if (!built) continue;
          const packed = packFarTiles([built])[0];
          if (!packed) continue;
          const levels = packed.prep.vox.levels;
          if (!levels || levels.length === 0) continue;
          let nClusters = 0;
          for (const l of levels) nClusters += l.blocks.length;
          if (nClusters > this.nMaxTileClusters) this.nMaxTileClusters = nClusters;
          tiles.push({ levels, center: packed.center, nClusters });
          needGranules += nClusters;
          if ((j & 15) === 15) await this.deps.yield();
        }
      }
      const needSlots = tiles.length;

      // stale (tx-owns-bake): our resident predecessor was evicted mid-bake (gen → −1), so
      // this bake has no home. Nothing reserved yet — just drop it (no rollback needed).
      if (old && old.gen === -1) return;

      // RESERVE-WHOLE-CELL or DEFER (law 9 — no partial cell, no leak): the cell commits
      // all-or-nothing. A cell whose need exceeds the ENTIRE pool is a genuine ring/grade
      // arithmetic bug (throw-loud — must never fire once provisioning is correct). Else if
      // the pool is only TRANSIENTLY full (other residents, or a re-grade's old set still
      // held) DEFER the whole cell: emit NOTHING and leave it un-resident so pose()
      // re-nominates it next tick. No slot/granule is popped until the cell is guaranteed
      // to land — the mid-bake throw that stranded earlier tiles' slots (#109) is gone.
      if (needSlots > this.cfg.slots)
        throw new Error(`fartile pool: cell needs ${needSlots} SLOTS > pool ${this.cfg.slots} (ring/grade arithmetic bug)`);
      if (needGranules > this.cfg.granules)
        throw new Error(`fartile pool: cell needs ${needGranules} GRANULES > pool ${this.cfg.granules} (ring/grade arithmetic bug)`);
      if (this.slotFree.length < needSlots || this.granuleFree.length < needGranules) return;

      // COMMIT PASS 2 — reserve + attach every tile (no throw path: the pool is pre-verified).
      const slots: number[] = [];
      const granules: number[] = [];
      for (const t of tiles) {
        const slot = this.slotFree.pop() as number;
        const gids = new Uint32Array(t.nClusters);
        for (let g = 0; g < t.nClusters; g++) gids[g] = this.granuleFree.pop() as number;
        slots.push(slot);
        for (const g of gids) granules.push(g);
        this.deps.emit(
          { kind: 'ftAttach', slot, granules: gids, center: t.center, levels: t.levels },
          [gids.buffer, ...t.levels.flatMap((l) => [l.words.buffer, l.occupied.buffer])],
        );
      }

      // commit: a resident cell (0..N slots) — TERMINAL. A re-grade evicts the old set
      // now that the new one is attached (never uncovered).
      if (old && old.slots.length > 0) {
        this.deps.emit({ kind: 'ftEvict', slots: old.slots.slice() }, []);
        for (const s of old.slots) this.slotFree.push(s);
        for (const g of old.granules) this.granuleFree.push(g);
        this.nHeads -= old.slots.length;
      }
      if (old) this.nRebaked++;
      this.resident.set(key, { slots, granules, grade, gen: myGen });
      this.nHeads += slots.length;
      this.nBaked++;
    } catch (e) {
      throw e instanceof Error ? e : new Error(String(e));
    } finally {
      this.baking.delete(key);
    }
  }

  /** trees within the cell box (+reach margin) grouped by idF → SplatPoolFlat[]. */
  private buildCellPools(
    trees: ChunkPayload & { kind: 'records' },
    height: ChunkPayload | null,
    pcx: number,
    pcz: number,
    cellX0: number,
    cellZ0: number,
    cm: number,
    margin: number,
  ): SplatPoolFlat[] {
    const { cols, count } = trees;
    const footprint = this.grid.chunkMeters; // LOD0
    const minX = this.grid.originX + pcx * footprint;
    const minZ = this.grid.originZ + pcz * footprint;
    const needDerive = !(cols.y && cols.yaw);
    const derive =
      needDerive && height && height.kind === 'height'
        ? makeGroundDeriver(height.heights, height.res, footprint, pcx, pcz)
        : null;
    const byId = new Map<number, { a: number[]; b: number[] }>();
    for (let i = 0; i < count; i++) {
      const xLocal = cols.x[i] as number;
      const zLocal = cols.z[i] as number;
      const x = cols.xw ? (cols.xw[i] as number) : minX + xLocal;
      const z = cols.zw ? (cols.zw[i] as number) : minZ + zLocal;
      if (x < cellX0 - margin || x > cellX0 + cm + margin || z < cellZ0 - margin || z > cellZ0 + cm + margin) continue;
      const cls = this.cfg.speciesToClass[cols.species[i] as number] ?? -1;
      if (cls < 0) continue;
      const idF = cls * 8 + ((cols.variant[i] as number) & 3);
      if (!this.pools.has(idF)) continue;
      const scale = cols.scale[i] as number;
      let y: number;
      let yaw: number;
      if (cols.y && cols.yaw) {
        y = cols.y[i] as number;
        yaw = cols.yaw[i] as number;
      } else {
        const dv = (derive as NonNullable<typeof derive>)(xLocal, zLocal);
        y = cols.y ? (cols.y[i] as number) : dv.h - scale * 0.12;
        yaw = cols.yaw ? (cols.yaw[i] as number) : dv.yaw;
      }
      let g = byId.get(idF);
      if (!g) { g = { a: [], b: [] }; byId.set(idF, g); }
      g.a.push(x, y, z, scale);
      g.b.push(yaw, 0, 0, idF);
    }
    const out: SplatPoolFlat[] = [];
    for (const [idF, g] of byId) {
      const sp = this.pools.get(idF) as SpeciesFlat;
      out.push({
        a: Float32Array.from(g.a),
        b: Float32Array.from(g.b),
        species: { bricks: sp.bricks, crownMinY: sp.crownMinY, barkR: sp.barkR, barkG: sp.barkG, barkB: sp.barkB },
      });
    }
    return out;
  }

  counters(): Record<string, number> {
    return {
      'ft.cells.resident': this.resident.size,
      'ft.cells.baking': this.baking.size,
      'ft.heads': this.nHeads,
      'ft.slots.free': this.slotFree.length,
      'ft.granules.free': this.granuleFree.length,
      'ft.baked': this.nBaked,
      'ft.rebaked': this.nRebaked,
      'ft.evicted': this.nEvicted,
      'ft.maxTileClusters': this.nMaxTileClusters,
    };
  }
}
