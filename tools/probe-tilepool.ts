/**
 * N8-D2 Stage 2a validation (node-only): the GeometryRegistry streaming tile
 * POOL (reserveTilePool / allocTileSlot / attachHeightDagTile / evictHeightDagTile,
 * D-N39). The pool is the MEMORY bound for full-res terrain — a fixed set of S
 * slots, each a constant byte block, (re)loaded in place as the camera streams.
 *
 *   npx tsx tools/probe-tilepool.ts
 *
 * Checks:
 *  P  pool claim — fixed slot bases (poolBase + slot*cap), all slots initially
 *     free, a pre-build sentinel entry sits BELOW the pool (disjoint regions)
 *  L  load — attachHeightDagTile writes a tile into its slot: mesh repoint
 *     (clusterBase = slot base, clusterCount, HASDAG, sphere contains clusters),
 *     cluster recs (meshId = slot handle, DAG|HF flags, triStart globalized),
 *     verts (word0 = global texel coord), indices rebased into the slot block,
 *     DAG cut records (own/parent error, root sentinel)
 *  E  evict — clusterCount→0, sphere parked off-world, HASDAG cleared, slot
 *     returns to the free-stack (idempotent on an already-free slot)
 *  R  reload — a freed slot reloads a DIFFERENT region IN PLACE; the new data
 *     fully overwrites; OTHER live slots + the sentinel are untouched
 *  B  BOUNDED — vert/tri/cluster cursors are IDENTICAL after a full churn of
 *     loads/evicts/reloads as right after build(): streaming never grows the
 *     buffers (THE point of the pool)
 *  G  guard — a tile exceeding the slot cap throws (explicit, never silent)
 */

import {
  CLUSTER_FLAG_DAG,
  CLUSTER_FLAG_HEIGHTFIELD,
  GeometryRegistry,
  MESH_FLAG_HASDAG,
  TILE_EVICTED_FAR,
  VERT_WORDS,
  decodeClusterCPU,
  decodeDagCPU,
  decodeMeshCPU,
  type ExplicitSource,
} from '../src/nanite/GeometryRegistry';
import { buildHeightDag, type HeightDagBuild } from '../src/nanite/BuildHeightDag';

let failures = 0;
const fail = (msg: string): void => {
  failures++;
  console.error(`  FAIL ${msg}`);
};
const expect = (cond: boolean, msg: string): void => {
  if (!cond) fail(msg);
};

// --- synthetic terrain regions (distinct shapes → distinct cluster counts) ----
type Shape = 'flat' | 'noisy' | 'ramp' | 'ridges';
function heights(shape: Shape, gridN: number): Float32Array {
  const vpa = gridN + 1;
  const h = new Float32Array(vpa * vpa);
  for (let z = 0; z < vpa; z++) {
    for (let x = 0; x < vpa; x++) {
      const u = x / gridN;
      const v = z / gridN;
      let y = 0;
      if (shape === 'flat') y = 0;
      else if (shape === 'ramp') y = u * 40 + v * 10;
      else if (shape === 'noisy') y = Math.sin(u * 21.7) * 18 + Math.cos(v * 17.3) * 14 + Math.sin((u + v) * 33) * 6;
      else y = Math.abs(Math.sin(u * 12.1)) * 30 + Math.abs(Math.cos(v * 9.4)) * 22;
      h[z * vpa + x] = y;
    }
  }
  return h;
}

const GRID_N = 32; // power of two (RTIN error pyramid)
const CELL = 1;
/** build a tile DAG for region `r` and remap its tile-local grid coords (0..gridN)
 *  to GLOBAL texel coords at the region's column (mimics WorldRegistry tiling). */
function region(shape: Shape, r: number): { build: HeightDagBuild; gridVerts: Uint32Array } {
  const tx0 = r * GRID_N; // distinct global texel column per region
  const tz0 = 0;
  const build = buildHeightDag(
    { heights: heights(shape, GRID_N), gridN: GRID_N, cellSize: CELL, originX: tx0 * CELL, originZ: tz0 * CELL },
    {},
  );
  const gridVerts = new Uint32Array(build.gridVerts.length);
  for (let i = 0; i < build.gridVerts.length; i++) {
    const p = build.gridVerts[i] as number;
    const texX = tx0 + (p & 0xffff);
    const texZ = tz0 + ((p >>> 16) & 0xffff);
    gridVerts[i] = ((texX & 0xffff) | ((texZ & 0xffff) << 16)) >>> 0;
  }
  return { build, gridVerts };
}

/** a tiny pre-build explicit mesh (sentinel that must survive all pool churn) */
function sentinelMesh(): ExplicitSource {
  // two triangles (a quad) in the XZ plane
  const positions = new Float32Array([0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]);
  const normals = new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]);
  const uvs = new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]);
  const vdata = new Uint32Array([0, 0, 0, 0]);
  const indices = new Uint32Array([0, 1, 2, 0, 2, 3]);
  return { kind: 'mesh', positions, normals, uvs, vdata, indices };
}

/** checksum a cluster range's raw record words — detects any corruption compactly */
function clusterChecksum(recs: Uint32Array, base: number, count: number): number {
  let s = 0;
  for (let i = base * 8; i < (base + count) * 8; i++) s = (s + (recs[i] as number)) >>> 0;
  return s;
}

console.log('[probe-tilepool]');

// --- build regions up front; size slot caps to the worst region + margin ------
const regions: Record<string, { build: HeightDagBuild; gridVerts: Uint32Array }> = {
  flat: region('flat', 0),
  noisy: region('noisy', 1),
  ramp: region('ramp', 2),
  ridges: region('ridges', 3),
  noisy2: region('noisy', 4),
};
let maxV = 0;
let maxT = 0;
let maxC = 0;
for (const k of Object.keys(regions)) {
  const b = (regions[k] as { build: HeightDagBuild }).build;
  maxV = Math.max(maxV, b.gridVerts.length);
  maxT = Math.max(maxT, b.indices.length / 3);
  maxC = Math.max(maxC, b.clusters.length);
}
const SLOTS = 4;
const VERT_CAP = maxV + 16;
const TRI_CAP = maxT + 32;
const CLUSTER_CAP = maxC + 8;
console.log(
  `  regions: flat ${regions.flat.build.clusters.length}cl  noisy ${regions.noisy.build.clusters.length}cl  ` +
    `ramp ${regions.ramp.build.clusters.length}cl  ridges ${regions.ridges.build.clusters.length}cl  ` +
    `| slot cap v${VERT_CAP}/t${TRI_CAP}/c${CLUSTER_CAP} × ${SLOTS}`,
);

const reg = new GeometryRegistry();
const sentinel = reg.registerMesh(sentinelMesh(), 'rock', { label: 'sentinel' });
reg.bindInstances(sentinel, { a: new Float32Array([0, 0, 0, 1]), b: new Float32Array([0, 0, 0, 0]) });
const slotHandles = reg.reserveTilePool(
  'terrain',
  { originX: 0, originZ: 0, cellSize: CELL },
  { slots: SLOTS, vertCap: VERT_CAP, triCap: TRI_CAP, clusterCap: CLUSTER_CAP },
  { label: 'terrain' },
);
reg.build();

const { arrays } = reg.debug();
const sentinelMeshRec = decodeMeshCPU(arrays.meshes, sentinel);
const sentinelSum = clusterChecksum(arrays.clusters, sentinelMeshRec.clusterStart, sentinelMeshRec.clusterCount);

// baseline cursors right after build() (with the pool region already claimed)
const baseV = reg.vertCount;
const baseT = reg.triCount;
const baseC = reg.clusterCount;

// --- P: pool claim --------------------------------------------------------------
expect(reg.tilePoolSlotCount === SLOTS, `P: slot count ${reg.tilePoolSlotCount} != ${SLOTS}`);
expect(reg.tileFreeSlotCount === SLOTS, `P: all ${SLOTS} slots should start free (${reg.tileFreeSlotCount})`);
expect(slotHandles.length === SLOTS, `P: handle table ${slotHandles.length} != ${SLOTS}`);
// sentinel sits below the pool: its cluster range ends at-or-before the pool base
expect(
  sentinelMeshRec.clusterStart + sentinelMeshRec.clusterCount <= baseC - SLOTS * CLUSTER_CAP,
  `P: sentinel cluster range overlaps the pool region`,
);

/** load region into a freshly-allocated slot; verify the full pack (L). */
function loadAndVerify(name: string, slot: number): void {
  const { build, gridVerts } = regions[name] as { build: HeightDagBuild; gridVerts: Uint32Array };
  const handle = reg.tileSlotHandle(slot);
  reg.attachHeightDagTile(slot, { gridVerts, indices: build.indices, clusters: build.clusters });
  const a = reg.debug().arrays;
  const cCount = build.clusters.length;
  const m = decodeMeshCPU(a.meshes, handle);
  expect(m.clusterCount === cCount, `L ${name}@${slot}: clusterCount ${m.clusterCount} != ${cCount}`);
  expect((m.flags & MESH_FLAG_HASDAG) !== 0, `L ${name}@${slot}: HASDAG missing`);
  const cBase = m.clusterStart;
  // cluster recs + dag recs
  let roots = 0;
  let minIdx = Infinity;
  let maxIdx = -Infinity;
  for (let c = 0; c < cCount; c++) {
    const dc = build.clusters[c];
    if (!dc) {
      fail(`L ${name}@${slot}: missing DagCluster ${c}`);
      continue;
    }
    const cc = decodeClusterCPU(a.clusters, cBase + c);
    expect(cc.meshId === handle, `L ${name}@${slot} C${c}: meshId ${cc.meshId} != ${handle}`);
    expect((cc.flags & CLUSTER_FLAG_DAG) !== 0, `L ${name}@${slot} C${c}: DAG flag missing`);
    expect((cc.flags & CLUSTER_FLAG_HEIGHTFIELD) !== 0, `L ${name}@${slot} C${c}: HF flag missing`);
    expect(cc.sphere[0] === Math.fround(dc.sx) && cc.sphere[3] === Math.fround(dc.sr), `L ${name}@${slot} C${c}: sphere drift`);
    // mesh sphere contains this cluster sphere
    const dist = Math.hypot(cc.sphere[0] - m.sphere[0], cc.sphere[1] - m.sphere[1], cc.sphere[2] - m.sphere[2]);
    expect(dist + cc.sphere[3] <= m.sphere[3] + 1e-3, `L ${name}@${slot} C${c}: cluster escapes mesh sphere`);
    const rec = decodeDagCPU(a.dag, cBase + c);
    if (!Number.isFinite(dc.parentError)) roots++;
    expect(Math.abs(rec.ownError - dc.ownError) < 1e-3, `L ${name}@${slot} C${c}: ownError drift`);
    for (let t = 0; t < dc.triCount; t++) {
      for (let k = 0; k < 3; k++) {
        const gi = a.indices[(cc.triStart + t) * 3 + k] as number;
        minIdx = Math.min(minIdx, gi);
        maxIdx = Math.max(maxIdx, gi);
      }
    }
  }
  expect(roots === build.stats.roots, `L ${name}@${slot}: root count ${roots} != ${build.stats.roots}`);
  // I: indices land inside THIS slot's OWN vertex block (disjoint from neighbours)
  const vBase = reg.tileSlotBase(slot).vert;
  expect(minIdx >= vBase, `L ${name}@${slot}: index ${minIdx} below slot vertBase ${vBase}`);
  expect(maxIdx < vBase + gridVerts.length, `L ${name}@${slot}: index ${maxIdx} past slot vert block`);
  // V: vertex word0 (raw) == the global texel coord we packed (read straight from
  // the array — decodeVertexCPU reinterprets word0 as f32, wrong for texel coords)
  expect(a.verts[vBase * VERT_WORDS] === (gridVerts[0] as number), `L ${name}@${slot}: vert0 word0 mismatch`);
}

// --- L: load three regions into slots 0,1,2 -------------------------------------
const s0 = reg.allocTileSlot();
const s1 = reg.allocTileSlot();
const s2 = reg.allocTileSlot();
expect(s0 === 0 && s1 === 1 && s2 === 2, `L: alloc order ${s0},${s1},${s2} != 0,1,2`);
expect(reg.tileFreeSlotCount === SLOTS - 3, `L: free count ${reg.tileFreeSlotCount} != ${SLOTS - 3}`);
loadAndVerify('flat', s0);
loadAndVerify('noisy', s1);
loadAndVerify('ramp', s2);
const sum0 = clusterChecksum(reg.debug().arrays.clusters, decodeMeshCPU(reg.debug().arrays.meshes, reg.tileSlotHandle(s0)).clusterStart, regions.flat.build.clusters.length);
const sum2 = clusterChecksum(reg.debug().arrays.clusters, decodeMeshCPU(reg.debug().arrays.meshes, reg.tileSlotHandle(s2)).clusterStart, regions.ramp.build.clusters.length);

// --- E: evict slot 1 (noisy) ----------------------------------------------------
reg.evictHeightDagTile(s1);
const mE = decodeMeshCPU(reg.debug().arrays.meshes, reg.tileSlotHandle(s1));
expect(mE.clusterCount === 0, `E: evicted clusterCount ${mE.clusterCount} != 0`);
expect((mE.flags & MESH_FLAG_HASDAG) === 0, `E: evicted HASDAG not cleared`);
expect(mE.sphere[0] === Math.fround(TILE_EVICTED_FAR), `E: evicted sphere not parked off-world (${mE.sphere[0]})`);
expect(reg.tileFreeSlotCount === SLOTS - 2, `E: free count ${reg.tileFreeSlotCount} != ${SLOTS - 2}`);
reg.evictHeightDagTile(s1); // idempotent
expect(reg.tileFreeSlotCount === SLOTS - 2, `E: double-evict changed free count`);

// --- R: reload a DIFFERENT region into the freed slot ---------------------------
const sR = reg.allocTileSlot();
expect(sR === s1, `R: realloc returned ${sR}, expected the just-freed ${s1}`);
loadAndVerify('ridges', sR);
// the OTHER live slots are untouched (checksums stable)
const sum0b = clusterChecksum(reg.debug().arrays.clusters, decodeMeshCPU(reg.debug().arrays.meshes, reg.tileSlotHandle(s0)).clusterStart, regions.flat.build.clusters.length);
const sum2b = clusterChecksum(reg.debug().arrays.clusters, decodeMeshCPU(reg.debug().arrays.meshes, reg.tileSlotHandle(s2)).clusterStart, regions.ramp.build.clusters.length);
expect(sum0b === sum0, `R: slot 0 corrupted by neighbour reload`);
expect(sum2b === sum2, `R: slot 2 corrupted by neighbour reload`);
// the sentinel (pre-build entry below the pool) is untouched
const sentinelSumB = clusterChecksum(reg.debug().arrays.clusters, sentinelMeshRec.clusterStart, sentinelMeshRec.clusterCount);
expect(sentinelSumB === sentinelSum, `R: pre-build sentinel corrupted by pool writes`);

// churn a few more cycles to stress reuse
for (let i = 0; i < 6; i++) {
  reg.evictHeightDagTile(s0);
  const sn = reg.allocTileSlot();
  expect(sn === s0, `R churn ${i}: realloc ${sn} != ${s0}`);
  loadAndVerify(i % 2 === 0 ? 'noisy2' : 'flat', sn);
}

// --- B: BOUNDED — cursors never grew across all the churn -----------------------
expect(reg.vertCount === baseV, `B: vertCount grew ${baseV} → ${reg.vertCount} (pool must reuse in place)`);
expect(reg.triCount === baseT, `B: triCount grew ${baseT} → ${reg.triCount}`);
expect(reg.clusterCount === baseC, `B: clusterCount grew ${baseC} → ${reg.clusterCount}`);

// --- G: capacity guard ----------------------------------------------------------
let threw = false;
try {
  const big = new Uint32Array(VERT_CAP + 100);
  reg.attachHeightDagTile(s2, { gridVerts: big, indices: new Uint32Array([0, 1, 2]), clusters: regions.flat.build.clusters });
} catch {
  threw = true;
}
expect(threw, `G: over-cap tile did not throw`);

console.log(
  `  loaded/evicted/reloaded across ${SLOTS} slots; cursors bounded at v${baseV}/t${baseT}/c${baseC}; ` +
    `sentinel + neighbours intact`,
);
if (failures > 0) {
  console.error(`[probe-tilepool] ${failures} FAILURES`);
  process.exit(1);
}
console.log('[probe-tilepool] tile pool: load/evict/reload in place, bounded, crack-free of neighbours');
