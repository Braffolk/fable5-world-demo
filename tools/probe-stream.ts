/**
 * N8-D2 Stage 2b-3 validation (node-only): the TerrainStreamer residency loop
 * (D-N39) — the per-frame clipmap diff that makes the 1 m detail FOLLOW the
 * camera. Drives a synthetic field through a motion path and asserts the
 * streaming invariants against a real GeometryRegistry tile pool.
 *
 *   npx tsx tools/probe-stream.ts
 *
 *  F  FOLLOW — after settling at a pose, the resident set EXACTLY equals the
 *     clipmap's desired set there (residency tracks the camera; no stale tiles).
 *  H  NO-HOLE — every desired tile is resident (the no-fallback floor at the diff
 *     level), and the finest (L0) tile at the camera is among them.
 *  E  EVICT — moving to a disjoint region drops the old fine tiles (resident set
 *     turns over; departed tiles are freed, not leaked).
 *  B  BOUNDED — resident ≤ maxTiles, AND the registry's vert/tri/cluster cursors
 *     are IDENTICAL after the whole motion as right after boot (the pool reuses
 *     slots in place — streaming never grows the mega-buffers; the memory bound).
 *  L  NO-LEAK — free slots + resident slots == pool size at every step.
 *  D  DETERMINISTIC — re-settling at the same pose loads nothing new.
 *  S  SKIP-GRACEFUL — with caps too small for a tile, the streamer SKIPS it (no
 *     throw, no slot leak, cursors bounded, built once not every frame) — the
 *     coarser ring backstops it on the GPU.
 */

import { GeometryRegistry } from '../src/nanite/GeometryRegistry';
import { TerrainStreamer } from '../src/nanite/TerrainStreamer';

let failures = 0;
const fail = (msg: string): void => {
  failures++;
  console.error(`  FAIL ${msg}`);
};
const expect = (cond: boolean, msg: string): void => {
  if (!cond) fail(msg);
};
const setsEqual = (a: Set<string>, b: Set<string>): boolean => {
  if (a.size !== b.size) return false;
  for (const k of a) if (!b.has(k)) return false;
  return true;
};

// --- synthetic field: rolling hills + a diagonal ridge band so different regions
// have genuinely different DAG densities (exercises eviction + reload of distinct
// tiles). cell=1, origin=0 ⇒ world coord == texel coord (residency is affine-
// invariant; the real cell/origin only matter for the GPU decode, not this diff).
const RES = 512;
const GRID_N = 64;
const CELL = 1;
const ORIGIN = 0;
function makeField(res: number): Float32Array {
  const h = new Float32Array(res * res);
  for (let z = 0; z < res; z++) {
    for (let x = 0; x < res; x++) {
      const u = x / res;
      const v = z / res;
      let y = Math.sin(u * 9) * 20 + Math.cos(v * 7) * 16; // rolling hills
      y += Math.abs(Math.sin((u + v) * 11)) * 22; // diagonal ridge band
      h[z * res + x] = y;
    }
  }
  return h;
}

function newStreamer(reg: GeometryRegistry, heights: Float32Array): TerrainStreamer {
  return new TerrainStreamer(
    { reg, heights, res: RES, cell: CELL, origin: ORIGIN, gridN: GRID_N, seed: null, worker: null },
    () => {}, // swallow deferred notes
  );
}

console.log('[probe-stream]');
const heights = makeField(RES);

// ============================================================================
// PART A — residency invariants (generous caps; nothing should ever skip)
// ============================================================================
const reg = new GeometryRegistry();
const streamer = newStreamer(reg, heights);
// buildBootSet is PRE-build (measures caps); reserve + build + attach mirror
// WorldRegistry's clip path exactly.
await streamer.buildBootSet(RES / 2, RES / 2);
const pm = streamer.poolMax;
const vCap = Math.ceil(pm.v * 2.0) + 512;
const tCap = Math.ceil(pm.t * 2.0) + 512;
const cCap = Math.ceil(pm.c * 2.0) + 64;
const SLOTS = streamer.maxTiles + 8;
reg.reserveTilePool('terrain', { originX: ORIGIN, originZ: ORIGIN, cellSize: CELL }, { slots: SLOTS, vertCap: vCap, triCap: tCap, clusterCap: cCap }, { label: 'terrain' });
reg.build();
streamer.attachBootSet();
console.log(
  `  cfg: res ${RES} gridN ${GRID_N} → ${streamer.clipDesc}, maxTiles ${streamer.maxTiles}, ` +
    `pool ${SLOTS}×(v${vCap}/t${tCap}/c${cCap}); boot poolMax v${pm.v}/t${pm.t}/c${pm.c}`,
);

// baseline cursors right after boot — must never grow across the whole motion
const baseV = reg.vertCount;
const baseT = reg.triCount;
const baseC = reg.clusterCount;
const slotTotal = reg.tilePoolSlotCount;

const noLeak = (where: string): void =>
  expect(
    reg.tileFreeSlotCount + streamer.residentCount === slotTotal,
    `L: leak @${where} — free ${reg.tileFreeSlotCount} + resident ${streamer.residentCount} != ${slotTotal}`,
  );

/** settle at a world pose, then assert F/H/L/B for it. */
async function visit(label: string, wx: number, wz: number): Promise<void> {
  await streamer.settleAt(wx, wz);
  const desired = streamer.desiredKeysAt(wx, wz);
  const resident = streamer.residentKeys();
  // F: residency exactly tracks the camera
  expect(setsEqual(resident, desired), `F: ${label} resident≠desired (r${resident.size}/d${desired.size})`);
  // H: the finest (L0) tile at the camera is resident (no-hole at full res)
  const l0 = [...desired].filter((k) => k.startsWith('L0:'));
  expect(l0.length > 0, `H: ${label} no L0 tile at camera`);
  expect(l0.every((k) => resident.has(k)), `H: ${label} L0 tile not resident`);
  // L: no slot leak
  noLeak(label);
  // B: bounded resident count
  expect(streamer.residentCount <= streamer.maxTiles, `B: ${label} resident ${streamer.residentCount} > maxTiles ${streamer.maxTiles}`);
}

// boot pose is the field center
await visit('center', RES / 2, RES / 2);
const centerKeys = streamer.residentKeys();
expect((streamer.counters()['terrain.stream.skipped'] as number) === 0, `S: unexpected skip with generous caps`);

// --- F/E: a motion path across + past the field --------------------------------
const path: Array<[string, number, number]> = [
  ['corner-lo', 40, 40],
  ['mid-right', 400, 256],
  ['corner-hi', 470, 470],
  ['edge-clamp', 511, 8], // near the rim — L0 block clamps, backstop must still reach
  ['back-center', RES / 2, RES / 2],
];
let prev = centerKeys;
for (const [label, wx, wz] of path) {
  await visit(label, wx, wz);
  const now = streamer.residentKeys();
  // E: a big move must TURN OVER residency (some old fine tiles evicted)
  if (label === 'corner-lo' || label === 'corner-hi') {
    expect(!setsEqual(now, prev), `E: ${label} residency did not change from a disjoint move`);
  }
  prev = now;
}
// returning to center reproduces the boot residency exactly
expect(setsEqual(streamer.residentKeys(), centerKeys), `F: back-center ≠ boot center residency`);

// D: determinism — re-settle at the same pose loads nothing new
const loadedBefore = streamer.counters()['terrain.stream.loaded'] as number;
await streamer.settleAt(RES / 2, RES / 2);
expect((streamer.counters()['terrain.stream.loaded'] as number) === loadedBefore, `D: re-settle at same pose loaded new tiles`);

// B: cursors never grew — the pool reused slots in place across all the churn
expect(reg.vertCount === baseV, `B: vertCount grew ${baseV}→${reg.vertCount} (pool must reuse in place)`);
expect(reg.triCount === baseT, `B: triCount grew ${baseT}→${reg.triCount}`);
expect(reg.clusterCount === baseC, `B: clusterCount grew ${baseC}→${reg.clusterCount}`);
expect((streamer.counters()['terrain.stream.skipped'] as number) === 0, `S: caps generous but tiles skipped`);
console.log(
  `  PART A: visited ${path.length + 1} poses; resident≡desired everywhere, ` +
    `evict ${streamer.counters()['terrain.stream.evicted']}, no leak, cursors bounded v${baseV}/t${baseT}/c${baseC}`,
);

// ============================================================================
// PART B — graceful over-cap SKIP (vertCap below EVERY tile ⇒ all skip)
// ============================================================================
const reg2 = new GeometryRegistry();
const streamer2 = newStreamer(reg2, heights);
await streamer2.buildBootSet(RES / 2, RES / 2);
const pm2 = streamer2.poolMax;
// vertCap = 8 is below any real tile's vert soup ⇒ every desired tile overflows
// and must be SKIPPED (the per-frame path; we do NOT attachBootSet, which is the
// boot path and assumes the caps were measured from the boot set).
reg2.reserveTilePool(
  'terrain',
  { originX: ORIGIN, originZ: ORIGIN, cellSize: CELL },
  { slots: streamer2.maxTiles + 8, vertCap: 8, triCap: pm2.t + 512, clusterCap: pm2.c + 64 },
  { label: 'terrain' },
);
reg2.build();
const base2 = { v: reg2.vertCount, t: reg2.triCount, c: reg2.clusterCount };
const slotTotal2 = reg2.tilePoolSlotCount;

const wantB = streamer2.desiredKeysAt(200, 320).size;
let threw = false;
try {
  await streamer2.settleAt(200, 320); // runDiff cap-checks → all skip
} catch {
  threw = true;
}
expect(!threw, `S: over-cap settle threw (must skip gracefully)`);
const skipped = streamer2.counters()['terrain.stream.skipped'] as number;
// every desired tile skipped EXACTLY once (the skipped-set prevents per-frame rebuilds)
expect(skipped === wantB, `S: expected ${wantB} unique skips, got ${skipped} (per-frame rebuild?)`);
expect(streamer2.residentCount === 0, `S: nothing should be resident when all tiles overflow (${streamer2.residentCount})`);
// no leak + cursors bounded despite the skip storm
expect(reg2.tileFreeSlotCount === slotTotal2, `S: slot leak after skips (free ${reg2.tileFreeSlotCount} != ${slotTotal2})`);
expect(reg2.vertCount === base2.v && reg2.triCount === base2.t && reg2.clusterCount === base2.c, `S: cursors grew during skip path`);
// a SECOND settle must not rebuild the skipped tiles (skipped-set holds)
const rebuilt = streamer2.counters()['terrain.stream.built'] as number;
await streamer2.settleAt(200, 320);
expect((streamer2.counters()['terrain.stream.built'] as number) === rebuilt, `S: skipped tiles rebuilt on re-settle (should be cached as skipped)`);
console.log(`  PART B: all ${skipped} over-cap tiles skipped once (no throw/leak/rebuild); cursors bounded`);

if (failures > 0) {
  console.error(`[probe-stream] ${failures} FAILURES`);
  process.exit(1);
}
console.log('[probe-stream] streamer residency: follows camera, evicts far, no holes, bounded, no leak, graceful skip');
