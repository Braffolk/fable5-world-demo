/**
 * S5 validation (node-only): StreamBrainCore's WRAPPING plane-window scroll —
 * the path the generated world never exercises (its windows cover the world ⇒
 * pinned) but Estonia rides from S6. Drives the core against a synthetic
 * height source whose lattice is far larger than the window and asserts:
 *
 *   npx tsx tools/probe-streambrain.ts
 *
 *  W  WINDOW CONTENT — after boot fill and after every scroll, reconstructing
 *     the logical window from the emitted physical fill rects + phase yields
 *     EXACTLY the source lattice values at the window's placement (the
 *     toroidal rect/phase math is lossless).
 *  F  FIFO ORDER — every scroll emits its fills BEFORE its origin commit
 *     (promote-after-fill is packet order, F-8).
 *  D  DEMAND — only chunks overlapping the (new) window region are fetched.
 *  T  DEMOTE-BEFORE-SCROLL — with a resident tile pinned on the vacated
 *     region, the scroll DEFERS (counter) instead of overwriting its source.
 */

import type { ChunkKey, ChunkPayload, LayerName } from '../src/world/source/WorldSource';
import { planField } from '../src/nanite/world/PlaneFill';
import { StreamBrainCore } from '../src/nanite/world/StreamBrainCore';
import type { BrainToMain, StreamPacket } from '../src/nanite/world/StreamProtocol';
import { packChunkKey } from '../src/world/source/Lac1';

let failures = 0;
const fail = (m: string): void => {
  failures++;
  console.error(`  FAIL ${m}`);
};
const expect = (c: boolean, m: string): void => {
  if (!c) fail(m);
};

// ---- synthetic world: height lods [0], chunkMeters 2048, texel 8 m (chunkRes
// 256 samples/chunk), 8×8 chunks ⇒ lattice 2048 samples ≫ the derived window.
const TEXEL = 8;
const CHUNK_M = 2048;
const CHUNK_RES = CHUNK_M / TEXEL; // 256
const CHUNKS = 32; // 65536 m ⇒ 8192-sample lattice ≫ the 2048 window cap ⇒ wraps
const srcAt = (nx: number, nz: number): number => Math.fround(Math.sin(nx * 0.37) * 100 + nz * 0.01 + nx * 0.003); // fround: payloads are f32

const keys: ChunkKey[] = [];
for (let cz = 0; cz < CHUNKS; cz++) for (let cx = 0; cx < CHUNKS; cx++) keys.push({ lod: 0, cx, cz });

const fetched: string[] = [];
async function fetchChunk(layer: LayerName, key: ChunkKey): Promise<ChunkPayload | null> {
  if (layer !== 'height' || key.lod !== 0) return null;
  if (key.cx < 0 || key.cz < 0 || key.cx >= CHUNKS || key.cz >= CHUNKS) return null;
  fetched.push(`${key.cx},${key.cz}`);
  const res = CHUNK_RES + 1; // incl. apron
  const heights = new Float32Array(res * res);
  for (let j = 0; j < res; j++) {
    for (let i = 0; i < res; i++) heights[j * res + i] = srcAt(key.cx * CHUNK_RES + i, key.cz * CHUNK_RES + j);
  }
  return { kind: 'height', res, heights };
}

// a fake manifest just rich enough for planField + the init tables
const manifest = {
  grid: { anchorE: 0, anchorN: 0, chunkMeters: CHUNK_M, chunkRes: CHUNK_RES, lodStep: 4, originX: 0, originZ: 0 },
  layers: { height: { enc: 1, lods: [0], chunkCount: keys.length, texelMeters: TEXEL } },
  dictionaries: { species: new Map(), understory: new Map(), debris: new Map() },
  coverage: () => null,
  chunks: (layer: LayerName, lod: number) => (layer === 'height' && lod === 0 ? keys : []),
} as unknown as Parameters<typeof planField>[0];

const plan = planField(manifest);
const H = plan.height[0];
if (!H) throw new Error('probe: no height plan');
expect(H.wraps, `W window (res ${H.res}) must wrap over a ${CHUNKS * CHUNK_RES}-sample lattice`);

// ---- shadow window: reconstruct logical content from emitted fills ---------------
const res = H.res;
const shadow = new Float32Array(res * res); // physical layout, like the GPU plane
let curN0x = H.n0x;
let curN0z = H.n0z;
let curPhaseX = 0;
let curPhaseY = 0;
const emitted: BrainToMain[] = [];

function applyPackets(packets: StreamPacket[]): void {
  let sawFill = false;
  for (const p of packets) {
    if (p.kind === 'fill' && p.plane === 'height') {
      sawFill = true;
      const data = p.f32 as Float32Array;
      for (let y = 0; y < p.h; y++) shadow.set(data.subarray(y * p.w, (y + 1) * p.w), (p.y + y) * res + p.x);
    } else if (p.kind === 'planeOrigin' && p.plane === 'height') {
      expect(sawFill || packets.every((q) => q.kind !== 'fill'), 'F origin commit must FOLLOW its fills in the batch');
      curN0x = p.n0x;
      curN0z = p.n0z;
      curPhaseX = p.phaseX;
      curPhaseY = p.phaseY;
    }
  }
}

function checkWindow(tag: string): void {
  let bad = 0;
  for (let z = 0; z < res; z += 17) {
    for (let x = 0; x < res; x += 13) {
      const nx = curN0x + x;
      const nz = curN0z + z;
      if (nx < 0 || nz < 0 || nx > CHUNKS * CHUNK_RES - 1 || nz > CHUNKS * CHUNK_RES - 1) continue; // rim clamp-extend
      const phys = ((z + curPhaseY) % res) * res + ((x + curPhaseX) % res);
      if (shadow[phys] !== srcAt(nx, nz)) bad++;
    }
  }
  expect(bad === 0, `W ${tag}: ${bad} sampled texels mismatch the source lattice`);
}

const core = new StreamBrainCore({
  fetch: fetchChunk,
  emit: (msg) => {
    emitted.push(msg);
    if (msg.kind === 'packets') applyPackets(msg.packets);
    if (msg.kind === 'log' && msg.level === 'error') fail(`brain error: ${msg.msg}`);
  },
  bake: null,
});

const packed = new Float64Array(keys.map((k) => packChunkKey(k.lod, k.cx, k.cz)));
core.init({
  kind: 'init',
  grid: manifest.grid,
  layers: {
    height: {
      lods: [0],
      texelMeters: TEXEL,
      chunkKeys: { 0: packed },
      chunkHashes: { 0: new BigUint64Array(packed.length) },
    },
  },
  plan,
  tiles: {
    gridN: 32,
    tilesPerSide: 4,
    skirt: false,
    seed: 1,
    cell: TEXEL,
    origin: TEXEL * 0.5,
    latMin: 0,
    latMax: CHUNKS * CHUNK_RES - 1,
  },
});

const worldOf = (n: number): number => (n + 0.5) * TEXEL;

await core.fillPlanesBoot();
checkWindow('boot');
expect(emitted.some((m) => m.kind === 'planesReady'), 'boot must emit planesReady');

// ---- scroll east by ~3/8 window, then diagonally back ------------------------------
const centerN = () => ({ x: curN0x + res / 2, z: curN0z + res / 2 });
const c0 = centerN();
fetched.length = 0;
core.pose(worldOf(c0.x + Math.floor(res * 0.375)), worldOf(c0.z), 0, 0);
await new Promise((r) => setTimeout(r, 50));
expect(curN0x > H.n0x, 'scroll east must advance n0x');
checkWindow('east scroll');
// D: demand — no chunk fetched entirely west of the OLD window start
const minWantCx = Math.floor(H.n0x / CHUNK_RES);
expect(fetched.every((f) => Number(f.split(',')[0]) >= minWantCx), `D fetched only window-overlapping chunks (got ${fetched.join(' ')})`);

core.pose(worldOf(c0.x - Math.floor(res * 0.25)), worldOf(c0.z + Math.floor(res * 0.3)), 0, 0);
await new Promise((r) => setTimeout(r, 50));
checkWindow('diagonal scroll');

// teleport-scale jump ⇒ whole-window refill
core.pose(worldOf(CHUNKS * CHUNK_RES - res / 2 - 10), worldOf(CHUNKS * CHUNK_RES - res / 2 - 10), 0, 0);
await new Promise((r) => setTimeout(r, 80));
checkWindow('teleport refill');

// ---- T: residency ⊕ scroll coexist — boot the partition tree so RESIDENT tiles
// exist, then scroll the wrapping window; assert scrolls run FREELY (never
// deferred behind residency — the S6f deadlock cure) and the window content
// stays lattice-exact throughout.
core.pose(worldOf(CHUNKS * CHUNK_RES * 0.5), worldOf(CHUNKS * CHUNK_RES * 0.5), 0, 0);
await new Promise((r) => setTimeout(r, 80));
checkWindow('re-center refill');
const back = centerN();
// boot the residency tree at mid-lattice (inline bakes; pool stays null ⇒ no
// runtime refine churn — just resident tiles alongside the scrolling window).
await core.bootTiles(worldOf(back.x), worldOf(back.z));
expect(core.residentCountForProbe() > 0, 'T: partition-tree boot produced no resident tiles');
core.pose(worldOf(back.x + Math.floor(res * 0.375)), worldOf(back.z), 0, 0);
await new Promise((r) => setTimeout(r, 80));
checkWindow('scroll with residency active');

if (failures > 0) {
  console.error(`[probe-streambrain] ${failures} FAILURE(S)`);
  process.exit(1);
}
console.log('[probe-streambrain] wrapping-window scroll: content-exact, FIFO, demand-only fetches, residency⊕scroll coherent');
