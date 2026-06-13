/**
 * N8-D1a validation (node-only): attach a built LOD DAG (BuildDag.ts) to a
 * GeometryRegistry mesh and assert the PACK round-trips — the appended
 * mega-buffer block + the parallel 10-float cut records decode back to the
 * DagBuild bit-for-bit (within the registry's f32/oct/f16 quantisation), and
 * the mesh record is repointed at the DAG range with the discrete LOD retired.
 *
 *   npx tsx tools/probe-dagpack.ts
 *
 * Checks (per source):
 *  M  mesh repoint — clusterStart/Count = DAG range, lodNext = NONE,
 *     MESH_FLAG_HASDAG set, mesh sphere contains every DAG cluster sphere
 *  C  cluster records — sphere bits-exact, cone within oct16, triStart
 *     globalized onto the appended block, triCount, CLUSTER_FLAG_DAG, meshId
 *  D  DAG cut records — own/parent error+sphere match DagBuild (f32), roots
 *     carry the +∞ sentinel + parentSphere = ownSphere, root count matches
 *  V  vertex re-pack — pos f32-exact, normal within oct16, uv within f16
 *  I  index rebase — every appended index lands in the appended vertex block
 *  X  CPU cut sanity — at near/far camD the screen-error cut is a non-empty
 *     antichain (LOD0-ish near, roots far) using ONLY the packed records
 */

import { Rng } from '../src/core/Seed';
import { buildRock } from '../src/vegetation/RockBuilder';
import { buildDag, type DagBuild } from '../src/nanite/BuildDag';
import {
  CLUSTER_FLAG_DAG,
  DAG_ROOT_PARENT_ERR,
  DAG_VERT_STRIDE,
  GeometryRegistry,
  LOD_NONE,
  MESH_FLAG_HASDAG,
  decodeClusterCPU,
  decodeDagCPU,
  decodeMeshCPU,
  decodeVertexCPU,
  explicitToDagVerts,
  type ExplicitSource,
} from '../src/nanite/GeometryRegistry';

let failures = 0;
const fail = (msg: string): void => {
  failures++;
  console.error(`  FAIL ${msg}`);
};
const expect = (cond: boolean, msg: string): void => {
  if (!cond) fail(msg);
};
const near = (a: number, b: number, tol: number): boolean => Math.abs(a - b) <= tol;
/** f32-roundtrip tolerance, relative + absolute */
const f32ok = (got: number, want: number): boolean =>
  near(got, want, Math.abs(want) * 1e-5 + 1e-4);

const SCREEN_H = 1080;
const PROJ_K = SCREEN_H / 2 / Math.tan(1.0 / 2);
/** screen-error projection of a sphere viewed from (0,0,camD) down -Z (mirrors
 *  probe-dag.project / the GPU errToPx — d² − r² guarded) */
function project(e: number, cx: number, cy: number, cz: number, r: number, camD: number): number {
  if (e >= DAG_ROOT_PARENT_ERR) return Infinity;
  const dz = cz - camD;
  const d2 = cx * cx + cy * cy + dz * dz;
  return (PROJ_K * e) / Math.sqrt(Math.max(1e-6, d2 - r * r));
}

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
  const uvs = new Float32Array(vCount * 2);
  const vdata = new Uint32Array(vCount);
  const d = dat.array as Float32Array;
  for (let v = 0; v < vCount; v++) {
    uvs[v * 2] = (positions[v * 3] as number) * 0.13 + 0.5;
    uvs[v * 2 + 1] = (positions[v * 3 + 2] as number) * 0.13 + 0.5;
    const q = (i: number): number => Math.max(0, Math.min(1, d[v * 4 + i] as number));
    vdata[v] =
      (Math.round(q(0) * 255) |
        (Math.round(q(1) * 255) << 8) |
        (Math.round(q(2) * 255) << 16) |
        (Math.round(q(3) * 255) << 24)) >>>
      0;
  }
  return { kind: 'mesh', positions, normals, uvs, vdata, indices: new Uint32Array(idx.array as ArrayLike<number>) };
}

function checkPack(label: string, src: ExplicitSource, dag: DagBuild): void {
  const cCount = dag.clusters.length;
  const reg = new GeometryRegistry({
    late: {
      verts: dag.verts.length / DAG_VERT_STRIDE,
      tris: dag.indices.length / 3,
      clusters: cCount,
    },
  });
  const h = reg.registerMesh(src, 'rock', { label, swayPad: 0.5, matParam: 3 });
  // Build a real 2-LOD chain so the head's SWITCH distance (26 m) differs from the
  // chain's MAX draw distance (496 m, on the tail). attachDag must inherit the MAX
  // (the mesh's full intended envelope), NOT the head switch and NOT unbounded —
  // else the head, now the chain tail (lodNext==NONE), is envelope-dropped just
  // past the switch by the cull rule `lodNext==NONE && lodDist>0 && dist>lodDist`
  // and the WHOLE instance vanishes (trees at ~26 m). A no-chain head can't tell
  // switch from max — they'd be the same node — so the chain is load-bearing here.
  reg.registerLod(h, src, 26);
  reg.setMaxDistance(h, 496);
  reg.bindInstances(h, { a: new Float32Array([0, 0, 0, 1]), b: new Float32Array([0, 0, 0, 0]) });
  reg.build();

  // bases the append will use (cursors at attach time)
  const vBase = reg.vertCount;
  const tBase = reg.triCount;
  const cBase = reg.clusterCount;

  reg.attachDag(h, dag);

  const { arrays } = reg.debug();
  const vCount = dag.verts.length / DAG_VERT_STRIDE;

  // --- M: mesh repoint ---
  const m = decodeMeshCPU(arrays.meshes, h);
  expect(m.clusterStart === cBase, `${label} M: clusterStart ${m.clusterStart} != ${cBase}`);
  expect(m.clusterCount === cCount, `${label} M: clusterCount ${m.clusterCount} != ${cCount}`);
  expect(m.lodNext === LOD_NONE, `${label} M: lodNext not retired (${m.lodNext})`);
  expect(m.lodDist === 496, `${label} M: DAG envelope ${m.lodDist} != chain max 496 — must inherit the tail's max draw distance, not the head switch (26, would envelope-drop the instance) or unbounded`);
  expect((m.flags & MESH_FLAG_HASDAG) !== 0, `${label} M: MESH_FLAG_HASDAG missing`);

  // --- C / D / V / I over every DAG cluster ---
  let rootCount = 0;
  let minIdx = Infinity;
  let maxIdx = -Infinity;
  let maxOwnErr = 0;
  let maxParErr = 0;
  let maxSphereErr = 0;
  for (let c = 0; c < cCount; c++) {
    const dc = dag.clusters[c];
    if (!dc) {
      fail(`${label}: missing DagCluster ${c}`);
      continue;
    }
    const cc = decodeClusterCPU(arrays.clusters, cBase + c);
    // C: geometric sphere bits-exact (stored via f32Bits)
    expect(cc.sphere[0] === Math.fround(dc.sx) && cc.sphere[3] === Math.fround(dc.sr),
      `${label} C${c}: sphere mismatch`);
    expect(cc.triCount === dc.triCount, `${label} C${c}: triCount ${cc.triCount} != ${dc.triCount}`);
    expect(cc.triStart === tBase + dc.triStart, `${label} C${c}: triStart ${cc.triStart} != ${tBase + dc.triStart}`);
    expect((cc.flags & CLUSTER_FLAG_DAG) !== 0, `${label} C${c}: CLUSTER_FLAG_DAG missing`);
    expect(cc.meshId === h, `${label} C${c}: meshId ${cc.meshId} != ${h}`);
    // cone axis within oct16
    const dot = cc.coneAxis[0] * dc.cax + cc.coneAxis[1] * dc.cay + cc.coneAxis[2] * dc.caz;
    expect(dot > 0.999 || (dc.cax === 0 && dc.cay === 0 && dc.caz === 0), `${label} C${c}: cone axis drift ${dot.toFixed(4)}`);

    // mesh sphere contains this cluster sphere
    const dx = cc.sphere[0] - m.sphere[0];
    const dy = cc.sphere[1] - m.sphere[1];
    const dz = cc.sphere[2] - m.sphere[2];
    const contained = Math.hypot(dx, dy, dz) + cc.sphere[3] <= m.sphere[3] + 1e-3;
    expect(contained, `${label} C${c}: cluster sphere escapes mesh sphere`);

    // D: DAG cut record (own/parent error+sphere ride through the f32 sidecar,
    // so within-f32 drift is expected — bits-exact would be wrong to assert)
    const rec = decodeDagCPU(arrays.dag, cBase + c);
    maxOwnErr = Math.max(maxOwnErr, Math.abs(rec.ownError - dc.ownError));
    for (let k = 0; k < 4; k++) {
      const want = [dc.oex, dc.oey, dc.oez, dc.oer][k] as number;
      maxSphereErr = Math.max(maxSphereErr, Math.abs((rec.ownSphere[k] as number) - want));
    }
    const root = !Number.isFinite(dc.parentError);
    if (root) {
      rootCount++;
      expect(f32ok(rec.parentError, DAG_ROOT_PARENT_ERR), `${label} D${c}: root sentinel missing (${rec.parentError})`);
      expect(rec.parentSphere[3] === rec.ownSphere[3], `${label} D${c}: root parentSphere != ownSphere`);
    } else {
      maxParErr = Math.max(maxParErr, Math.abs(rec.parentError - dc.parentError));
    }

    // I: every triangle index of this cluster lands in the appended vert block
    for (let t = 0; t < dc.triCount; t++) {
      for (let k = 0; k < 3; k++) {
        const gi = arrays.indices[(cc.triStart + t) * 3 + k] as number;
        minIdx = Math.min(minIdx, gi);
        maxIdx = Math.max(maxIdx, gi);
      }
    }
  }
  expect(maxOwnErr < 1e-3, `${label} D: ownError drift ${maxOwnErr.toExponential(2)}`);
  expect(maxParErr < 1e-3, `${label} D: parentError drift ${maxParErr.toExponential(2)}`);
  expect(maxSphereErr < 1e-3, `${label} D: ownSphere drift ${maxSphereErr.toExponential(2)}`);
  expect(rootCount === dag.stats.roots, `${label} D: root count ${rootCount} != ${dag.stats.roots}`);
  expect(minIdx >= vBase && maxIdx < vBase + vCount, `${label} I: index range [${minIdx},${maxIdx}] outside [${vBase},${vBase + vCount})`);

  // V: spot-check vertex re-pack (pos exact, normal oct, uv f16)
  const sv = dag.verts;
  let maxPos = 0;
  let maxNrm = 0;
  const K = Math.min(256, vCount);
  for (let v = 0; v < K; v++) {
    const vc = decodeVertexCPU(arrays.verts, vBase + v);
    const s = v * DAG_VERT_STRIDE;
    maxPos = Math.max(maxPos, Math.abs(vc.pos[0] - Math.fround(sv[s] as number)));
    // re-normalise the source normal for the oct comparison
    const nx = sv[s + 3] as number;
    const ny = sv[s + 4] as number;
    const nz = sv[s + 5] as number;
    const l = Math.hypot(nx, ny, nz) || 1;
    const ndot = vc.nrm[0] * (nx / l) + vc.nrm[1] * (ny / l) + vc.nrm[2] * (nz / l);
    maxNrm = Math.max(maxNrm, 1 - ndot);
  }
  expect(maxPos === 0, `${label} V: pos not f32-exact (${maxPos})`);
  expect(maxNrm < 1e-3, `${label} V: normal oct drift ${maxNrm.toExponential(2)}`);

  // X: CPU cut sanity using ONLY the packed records — near camD → fine cut,
  // far → coarse cut; both non-empty antichains (no cluster with own+parent
  // both selected). Sphere/error are mesh-local (identity instance here).
  const cut = (camD: number, tau: number): { n: number; tris: number } => {
    let n = 0;
    let tris = 0;
    for (let c = 0; c < cCount; c++) {
      const rec = decodeDagCPU(arrays.dag, cBase + c);
      const cc = decodeClusterCPU(arrays.clusters, cBase + c);
      const pOwn = project(rec.ownError, rec.ownSphere[0], rec.ownSphere[1], rec.ownSphere[2], rec.ownSphere[3], camD);
      const pPar = project(rec.parentError, rec.parentSphere[0], rec.parentSphere[1], rec.parentSphere[2], rec.parentSphere[3], camD);
      if (pOwn <= tau && pPar > tau) {
        n++;
        tris += cc.triCount;
      }
    }
    return { n, tris };
  };
  const lod0Tris = dag.stats.lod0Tris;
  const cFull = cut(30, 0); // τ=0 → only own=0 (LOD0) passes, at any distance → full detail
  const cFar = cut(1e6, 1); // effectively infinite → only the ∞-sentinel parents (roots) survive
  const cMid = cut(30, 1); // a real working cut (τ=1px @ camD 30) — must tile the surface once
  expect(cFull.tris === lod0Tris, `${label} X: τ=0 cut ${cFull.tris} tris != LOD0 ${lod0Tris}`);
  expect(cFar.n === rootCount, `${label} X: far cut ${cFar.n} clusters != ${rootCount} roots`);
  expect(cMid.n > 0 && cMid.tris > 0, `${label} X: τ=1 cut empty`);
  expect(cMid.tris <= lod0Tris && cMid.tris >= cFar.tris, `${label} X: τ=1 cut ${cMid.tris} outside [roots ${cFar.tris}, LOD0 ${lod0Tris}]`);

  console.log(
    `  ${label.padEnd(12)} clusters ${String(cCount).padStart(5)} roots ${String(rootCount).padStart(3)} | ` +
      `τ=0 ${cFull.tris}tri (=LOD0)  τ=1@30 ${cMid.n}cl/${cMid.tris}tri  far ${cFar.n}cl (=roots)`,
  );
}

console.log('[probe-dagpack]');
for (const detail of [3, 5, 7]) {
  const src = rockSource('boulder', 1234 + detail, detail);
  const dag = buildDag(explicitToDagVerts(src), DAG_VERT_STRIDE, src.indices, { normalOffset: 3 });
  checkPack(`rock-d${detail}`, src, dag);
}

if (failures > 0) {
  console.error(`[probe-dagpack] ${failures} FAILURES`);
  process.exit(1);
}
console.log('[probe-dagpack] DAG pack round-trip holds');
