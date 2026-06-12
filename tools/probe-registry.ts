/**
 * GeometryRegistry validation probe (node-only): registers real rocks + a
 * synthetic heightfield + CPU instance streams, builds, and asserts the
 * packed-layout invariants end to end. Late registration + flush() and the
 * capacity-overflow throw are exercised too.
 *
 *   npx tsx tools/probe-registry.ts
 *
 * Checks:
 *  V  vertex blob roundtrip — pos f32 bits exact, oct16 normal within 1e-3,
 *     uv f16 within 1e-3, vdata word exact
 *  C  cluster records — sphere/cone-cos bits exact vs clusterize output,
 *     cone axis within oct16, triStart globalized, triCount/flags/meshId
 *  H  heightfield records — window spheres contain their AABB corners,
 *     gx|gz packing, implicit tri count, flags
 *  M  mesh table — ranges contiguous, channel/class/flags bytes, hf params,
 *     LOD chain linkage, instance ranges
 *  I  instance blob + instanceMesh ids
 *  L  late registration: flush() lands ranges + updateRanges, overflow throws
 */

import { Rng } from '../src/core/Seed';
import {
  CLUSTER_FLAG_HEIGHTFIELD,
  GeometryRegistry,
  LOD_NONE,
  MATERIAL_CLASS,
  MESH_FLAG_CAST_SHADOWS,
  MESH_FLAG_HEIGHTFIELD,
  TRANSFORM_CHANNEL,
  decodeClusterCPU,
  decodeMeshCPU,
  decodeVertexCPU,
  f32Bits,
  type ExplicitSource,
} from '../src/nanite/GeometryRegistry';
import { clusterize } from '../src/nanite/Clusterize';
import { buildRock } from '../src/vegetation/RockBuilder';

let failures = 0;
const fail = (msg: string): void => {
  failures++;
  console.error(`  FAIL ${msg}`);
};
const expect = (cond: boolean, msg: string): void => {
  if (!cond) fail(msg);
};

function rockSource(kind: 'boulder' | 'angular', seed: number, detail: number): ExplicitSource {
  const r = buildRock(kind, new Rng(seed), detail);
  const pos = r.geometry.attributes.position;
  const nrm = r.geometry.attributes.normal;
  const dat = r.geometry.attributes.vdata;
  const idx = r.geometry.index;
  if (!pos || !nrm || !dat || !idx) throw new Error('rock attrs missing');
  const positions = pos.array as Float32Array;
  const normals = nrm.array as Float32Array;
  const vCount = positions.length / 3;
  // synth planar UVs (rocks carry none) + quantize vdata vec4 → 4×u8 word —
  // the same packing C4 will use for the rock pools
  const uvs = new Float32Array(vCount * 2);
  const vdata = new Uint32Array(vCount);
  const d = dat.array as Float32Array;
  for (let v = 0; v < vCount; v++) {
    uvs[v * 2] = (positions[v * 3] as number) * 0.13 + 0.5;
    uvs[v * 2 + 1] = (positions[v * 3 + 2] as number) * 0.13 + 0.5;
    const q = (i: number): number =>
      Math.round(Math.max(0, Math.min(1, d[v * 4 + i] as number)) * 255);
    vdata[v] = (q(0) | (q(1) << 8) | (q(2) << 16) | (q(3) << 24)) >>> 0;
  }
  return {
    kind: 'mesh',
    positions,
    normals,
    uvs,
    vdata,
    indices: new Uint32Array(idx.array as ArrayLike<number>),
  };
}

function instStream(rng: Rng, count: number): { a: Float32Array; b: Float32Array } {
  const a = new Float32Array(count * 4);
  const b = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) {
    a[i * 4] = rng.range(-100, 100);
    a[i * 4 + 1] = rng.range(0, 10);
    a[i * 4 + 2] = rng.range(-100, 100);
    a[i * 4 + 3] = rng.range(0.5, 3);
    b[i * 4] = rng.range(0, Math.PI * 2);
    b[i * 4 + 1] = rng.range(-0.1, 0.1);
    b[i * 4 + 2] = rng.range(-0.1, 0.1);
    b[i * 4 + 3] = i + 17;
  }
  return { a, b };
}

console.log('[probe-registry]');

// --- sources ---------------------------------------------------------------
const srcA = rockSource('boulder', 41, 4);
const srcB = rockSource('angular', 42, 5);
const srcLod = rockSource('boulder', 41, 3);
const srcLate = rockSource('angular', 77, 3);

// 131 quads × winQuads 7 → 19 windows per axis, last window partial (5 quads)
const QUADS_X = 131;
const QUADS_Z = 131;
const WIN_QUADS = 7;
const WINDOWS = Math.ceil(QUADS_X / WIN_QUADS);
const CELL = 2;
const ORIGIN_X = -64;
const ORIGIN_Z = -48;
const hAt = (x: number, z: number): number => Math.sin(x * 0.05) * 6 + Math.cos(z * 0.07) * 4;
const winSpan = (g: number, total: number): number => Math.min(WIN_QUADS, total - g * WIN_QUADS);
const minMax = new Float32Array(WINDOWS * WINDOWS * 2);
for (let gz = 0; gz < WINDOWS; gz++) {
  for (let gx = 0; gx < WINDOWS; gx++) {
    let mn = Infinity;
    let mx = -Infinity;
    for (let qz = 0; qz <= winSpan(gz, QUADS_Z); qz++) {
      for (let qx = 0; qx <= winSpan(gx, QUADS_X); qx++) {
        const h = hAt(ORIGIN_X + (gx * WIN_QUADS + qx) * CELL, ORIGIN_Z + (gz * WIN_QUADS + qz) * CELL);
        mn = Math.min(mn, h);
        mx = Math.max(mx, h);
      }
    }
    const i = (gz * WINDOWS + gx) * 2;
    minMax[i] = mn;
    minMax[i + 1] = mx;
  }
}

// --- register + build --------------------------------------------------------
const reg = new GeometryRegistry({
  late: { meshes: 1, verts: 40000, tris: 80000, clusters: 700, instances: 16 },
});
const hA = reg.registerMesh(srcA, 'rock', { label: 'rockA', swayPad: 0 });
const hB = reg.registerMesh(srcB, 'rock', { label: 'rockB', castShadows: false });
const hT = reg.registerMesh(
  {
    kind: 'heightfield',
    quadsX: QUADS_X,
    quadsZ: QUADS_Z,
    winQuads: WIN_QUADS,
    cellSize: CELL,
    originX: ORIGIN_X,
    originZ: ORIGIN_Z,
    minMax,
  },
  'terrain',
  { label: 'hf-tile' },
);
const hLod = reg.registerLod(hA, srcLod, 120);

const rngI = new Rng(7);
const streamA = instStream(rngI, 5);
const streamB = instStream(rngI, 3);
const streamT = instStream(rngI, 1);
reg.bindInstances(hA, streamA);
reg.bindInstances(hB, streamB);
reg.bindInstances(hT, streamT);

const report = reg.build();
console.log(report.table);
const dbg = reg.debug();
const { verts, clusters, meshes, instances, instanceMesh, indices } = dbg.arrays;

// --- V: vertex roundtrip -----------------------------------------------------
{
  const entry = reg.meshEntry(hA);
  let maxN = 0;
  let maxUv = 0;
  let exact = true;
  for (let v = 0; v < entry.vertCount; v++) {
    const dec = decodeVertexCPU(verts, entry.vertBase + v);
    if (
      f32Bits(dec.pos[0]) !== f32Bits(srcA.positions[v * 3] as number) ||
      f32Bits(dec.pos[1]) !== f32Bits(srcA.positions[v * 3 + 1] as number) ||
      f32Bits(dec.pos[2]) !== f32Bits(srcA.positions[v * 3 + 2] as number)
    ) {
      exact = false;
    }
    for (let c = 0; c < 3; c++) {
      maxN = Math.max(maxN, Math.abs(dec.nrm[c] - (srcA.normals[v * 3 + c] as number)));
    }
    maxUv = Math.max(
      maxUv,
      Math.abs(dec.uv[0] - (srcA.uvs?.[v * 2] as number)),
      Math.abs(dec.uv[1] - (srcA.uvs?.[v * 2 + 1] as number)),
    );
    if (dec.vdata !== (srcA.vdata?.[v] as number)) exact = false;
  }
  expect(exact, 'V: pos bits / vdata word not exact');
  expect(maxN < 1e-3, `V: oct normal err ${maxN.toExponential(2)}`);
  expect(maxUv < 2e-3, `V: uv f16 err ${maxUv.toExponential(2)}`);
  console.log(`  V verts=${entry.vertCount} nrmErr=${maxN.toExponential(2)} uvErr=${maxUv.toExponential(2)}`);
}

// --- C: explicit cluster records vs a reference clusterize ------------------
{
  const ref = clusterize(srcB.positions, 3, srcB.indices, 128);
  const entry = reg.meshEntry(hB);
  expect(entry.clusterCount === ref.clusterCount, 'C: cluster count mismatch');
  let axErr = 0;
  let ok = true;
  for (let c = 0; c < ref.clusterCount; c++) {
    const dec = decodeClusterCPU(clusters, entry.clusterBase + c);
    for (let k = 0; k < 4; k++) {
      if (f32Bits(dec.sphere[k] as number) !== f32Bits(ref.sphere[c * 4 + k] as number)) ok = false;
    }
    const cosRef = ref.cone[c * 4 + 3] as number;
    if (f32Bits(dec.coneCos) !== f32Bits(cosRef)) ok = false;
    if (cosRef > -1) {
      for (let k = 0; k < 3; k++) {
        axErr = Math.max(axErr, Math.abs(dec.coneAxis[k] - (ref.cone[c * 4 + k] as number)));
      }
    }
    if (dec.triStart !== entry.triBase + (ref.triStart[c] as number)) ok = false;
    if (dec.triCount !== (ref.triCount[c] as number)) ok = false;
    if (dec.meshId !== hB) ok = false;
    if (dec.flags !== 0) ok = false;
  }
  expect(ok, 'C: record fields mismatch');
  expect(axErr < 1e-3, `C: cone axis err ${axErr.toExponential(2)}`);
  // indices globalized into this mesh's vertex window
  let inWindow = true;
  for (let i = entry.triBase * 3; i < (entry.triBase + entry.triCount) * 3; i++) {
    const v = indices[i] as number;
    if (v < entry.vertBase || v >= entry.vertBase + entry.vertCount) inWindow = false;
  }
  expect(inWindow, 'C: global indices outside vertex window');
  console.log(`  C clusters=${ref.clusterCount} axisErr=${axErr.toExponential(2)}`);
}

// --- H: heightfield records (incl. partial edge windows) ---------------------
{
  const entry = reg.meshEntry(hT);
  expect(entry.clusterCount === WINDOWS * WINDOWS, 'H: window count');
  let ok = true;
  let worst = 0;
  let partials = 0;
  for (let gz = 0; gz < WINDOWS; gz++) {
    for (let gx = 0; gx < WINDOWS; gx++) {
      const dec = decodeClusterCPU(clusters, entry.clusterBase + gz * WINDOWS + gx);
      const qx = winSpan(gx, QUADS_X);
      const qz = winSpan(gz, QUADS_Z);
      if (qx < WIN_QUADS || qz < WIN_QUADS) partials++;
      if ((dec.triStart & 0xffff) !== gx || dec.triStart >>> 16 !== gz) ok = false;
      if (dec.triCount !== qx * qz * 2) ok = false;
      if ((dec.flags & CLUSTER_FLAG_HEIGHTFIELD) === 0) ok = false;
      if (dec.meshId !== hT) ok = false;
      const i = (gz * WINDOWS + gx) * 2;
      const x0 = ORIGIN_X + gx * WIN_QUADS * CELL;
      const z0 = ORIGIN_Z + gz * WIN_QUADS * CELL;
      for (const y of [minMax[i] as number, minMax[i + 1] as number]) {
        for (const [cx, cz] of [
          [x0, z0],
          [x0 + qx * CELL, z0],
          [x0, z0 + qz * CELL],
          [x0 + qx * CELL, z0 + qz * CELL],
        ] as const) {
          const d = Math.hypot(cx - dec.sphere[0], y - dec.sphere[1], cz - dec.sphere[2]);
          worst = Math.max(worst, d - dec.sphere[3]);
        }
      }
    }
  }
  expect(ok, 'H: record fields mismatch');
  expect(partials === WINDOWS * 2 - 1, `H: expected ${WINDOWS * 2 - 1} partial windows, saw ${partials}`);
  expect(worst < 1e-4, `H: corner outside sphere by ${worst.toExponential(2)}`);
  console.log(`  H windows=${WINDOWS * WINDOWS} partials=${partials} cornerSlack=${worst.toExponential(2)}`);
}

// --- M: mesh table -----------------------------------------------------------
{
  const a = decodeMeshCPU(meshes, hA);
  const b = decodeMeshCPU(meshes, hB);
  const t = decodeMeshCPU(meshes, hT);
  const l = decodeMeshCPU(meshes, hLod);
  expect(a.clusterStart === 0, 'M: rockA clusterStart');
  expect(b.clusterStart === a.clusterStart + a.clusterCount, 'M: contiguous clusters A→B');
  expect(t.clusterStart === b.clusterStart + b.clusterCount, 'M: contiguous clusters B→T');
  expect(a.matClass === MATERIAL_CLASS.rock && t.matClass === MATERIAL_CLASS.terrain, 'M: matClass');
  expect(a.channel === TRANSFORM_CHANNEL.rigid && t.channel === TRANSFORM_CHANNEL.terrain, 'M: channel');
  expect((a.flags & MESH_FLAG_CAST_SHADOWS) !== 0, 'M: rockA castShadows default');
  expect((b.flags & MESH_FLAG_CAST_SHADOWS) === 0, 'M: rockB castShadows=false');
  expect((t.flags & MESH_FLAG_HEIGHTFIELD) !== 0 && t.winQuads === WIN_QUADS, 'M: hf flag/winQuads');
  expect(
    t.hfOriginX === ORIGIN_X && t.hfOriginZ === ORIGIN_Z && t.hfCellSize === CELL &&
      t.quadsX === QUADS_X && t.quadsZ === QUADS_Z,
    'M: hf params',
  );
  expect(a.lodNext === hLod && a.lodDist === 120, 'M: lod linkage');
  expect(l.lodNext === LOD_NONE && l.instCount === 0, 'M: lod tail');
  expect(a.instFirst === 0 && a.instCount === 5, 'M: rockA instances');
  expect(b.instFirst === 5 && b.instCount === 3, 'M: rockB instances');
  expect(t.instFirst === 8 && t.instCount === 1, 'M: hf instances');
  console.log('  M mesh table ok');
}

// --- I: instance blob ---------------------------------------------------------
{
  let ok = true;
  const check = (first: number, count: number, s: { a: Float32Array; b: Float32Array }, mesh: number): void => {
    for (let i = 0; i < count; i++) {
      for (let k = 0; k < 4; k++) {
        if (instances[(first + i) * 8 + k] !== (s.a[i * 4 + k] as number)) ok = false;
        if (instances[(first + i) * 8 + 4 + k] !== (s.b[i * 4 + k] as number)) ok = false;
      }
      if (instanceMesh[first + i] !== mesh) ok = false;
    }
  };
  check(0, 5, streamA, hA);
  check(5, 3, streamB, hB);
  check(8, 1, streamT, hT);
  expect(ok, 'I: instance blob / instanceMesh mismatch');
  console.log('  I instances ok');
}

// --- L: late registration + flush + overflow ---------------------------------
{
  const hLate = reg.registerMesh(srcLate, 'rock', { label: 'late-rock' });
  reg.bindInstances(hLate, instStream(new Rng(9), 2));
  reg.flush();
  const e = reg.meshEntry(hLate);
  const m = decodeMeshCPU(meshes, hLate);
  expect(m.clusterCount === e.clusterCount && m.instCount === 2, 'L: late mesh record');
  const dec = decodeClusterCPU(clusters, e.clusterBase);
  expect(dec.meshId === hLate && dec.triStart === e.triBase, 'L: late cluster record');
  const decV = decodeVertexCPU(verts, e.vertBase);
  expect(
    f32Bits(decV.pos[0]) === f32Bits(srcLate.positions[0] as number),
    'L: late verts landed',
  );
  expect(dbg.attrs.verts.updateRanges.length > 0, 'L: verts updateRanges queued');
  expect(dbg.attrs.instances.updateRanges.length > 0, 'L: instance updateRanges queued');
  let threw = false;
  try {
    reg.registerMesh(srcLate, 'rock', { label: 'overflow' });
  } catch {
    threw = true;
  }
  expect(threw, 'L: capacity overflow must throw');
  console.log('  L late registration ok');
}

if (failures > 0) {
  console.error(`[probe-registry] ${failures} FAILURES`);
  process.exit(1);
}
console.log('[probe-registry] all invariants hold');
