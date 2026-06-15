/**
 * N8-HIC step 2 gate: the GPU-format hierarchy round-trips the CPU hierarchy.
 *
 * Builds a real registry (registerMesh → build → attachDag), then reads back the
 * packed arrays (dag words 10/11 = childBase/childCount bitcast u32, mesh words
 * 16/17 = rootBase/rootCount, dagLinks = global child/root ids), reconstructs the
 * DagHierarchy in mesh-LOCAL form, and re-runs validateDagHierarchy — proving the
 * bitcast, the cBase global-id offsets, and the [roots…][children…] layout all
 * survive attachDag intact. If this is green, the BFS kernel (step 3) can trust
 * the GPU data.
 *
 *   npx tsx tools/probe-hier.ts
 */
import type { BufferGeometry } from 'three';
import {
  DAG_VERT_STRIDE,
  DAG_WORDS,
  MESH_WORDS,
  GeometryRegistry,
  explicitToDagVerts,
  f32Bits,
  type ExplicitSource,
} from '../src/nanite/GeometryRegistry';
import { buildDag } from '../src/nanite/BuildDag';
import { buildDagHierarchy, validateDagHierarchy, type DagHierarchy } from '../src/nanite/DagHierarchy';
import { Rng } from '../src/core/Seed';
import { buildRock } from '../src/vegetation/RockBuilder';
import { buildTree } from '../src/vegetation/TreeBuilder';
import { BEECH } from '../src/vegetation/Species';

let failures = 0;
const fail = (m: string): void => {
  failures++;
  console.error(`  FAIL ${m}`);
};

/** BufferGeometry → ExplicitSource (pos + normal + index) */
function toSource(geo: BufferGeometry): ExplicitSource {
  const pos = geo.attributes.position;
  if (!pos) throw new Error('no position');
  if (!geo.attributes.normal) geo.computeVertexNormals();
  const positions = pos.array as Float32Array;
  const normals = geo.attributes.normal!.array as Float32Array;
  const vCount = pos.count;
  const indices = geo.index
    ? new Uint32Array(geo.index.array as ArrayLike<number>)
    : Uint32Array.from({ length: vCount }, (_, i) => i);
  return { kind: 'mesh', positions, normals, indices };
}

function roundTrip(label: string, geo: BufferGeometry): void {
  const src = toSource(geo);
  const dag = buildDag(explicitToDagVerts(src), DAG_VERT_STRIDE, src.indices, { normalOffset: 3 });

  // real registry flow
  const reg = new GeometryRegistry();
  const h = reg.registerMesh(src, 'rock', { label });
  reg.bindInstances(h, { a: new Float32Array([0, 0, 0, 1]), b: new Float32Array([0, 0, 0, 0]) });
  reg.addLate({
    verts: dag.verts.length / DAG_VERT_STRIDE,
    tris: dag.indices.length / 3,
    clusters: dag.clusters.length,
  });
  reg.build();
  reg.attachDag(h, dag);

  // read the packed GPU arrays back + reconstruct the LOCAL hierarchy
  const { dag: dagArr, meshes: meshArr, dagLinks } = reg.debug().arrays;
  const mb = h * MESH_WORDS;
  const clusterBase = meshArr[mb] as number;
  const rootBase = meshArr[mb + 16] as number;
  const rootCount = meshArr[mb + 17] as number;
  const cCount = dag.clusters.length;
  const childBlockBase = rootBase + rootCount;

  const childStart = new Uint32Array(cCount);
  const childCount = new Uint32Array(cCount);
  let childTotal = 0;
  for (let c = 0; c < cCount; c++) {
    const db = (clusterBase + c) * DAG_WORDS;
    const cbG = f32Bits(dagArr[db + 10] as number);
    const cnt = f32Bits(dagArr[db + 11] as number);
    childStart[c] = cbG - childBlockBase;
    childCount[c] = cnt;
    childTotal += cnt;
  }
  const childIndices = new Uint32Array(childTotal);
  for (let j = 0; j < childTotal; j++) childIndices[j] = (dagLinks[childBlockBase + j] as number) - clusterBase;
  const rootIndices = new Uint32Array(rootCount);
  for (let i = 0; i < rootCount; i++) rootIndices[i] = (dagLinks[rootBase + i] as number) - clusterBase;

  const gpuHier: DagHierarchy = { childStart, childCount, childIndices, rootIndices };

  // 1) the GPU hierarchy must match the CPU hierarchy exactly
  const cpu = buildDagHierarchy(dag);
  let mismatch = '';
  if (cpu.rootIndices.length !== rootIndices.length) mismatch = `root count ${rootIndices.length} != ${cpu.rootIndices.length}`;
  else if (cpu.childIndices.length !== childIndices.length) mismatch = `child count ${childIndices.length} != ${cpu.childIndices.length}`;
  else {
    for (let c = 0; c < cCount && !mismatch; c++) {
      if (childCount[c] !== cpu.childCount[c]) mismatch = `cluster ${c} childCount ${childCount[c]} != ${cpu.childCount[c]}`;
    }
  }
  if (mismatch) fail(`${label}: GPU≠CPU hierarchy — ${mismatch}`);

  // 2) the RECONSTRUCTED hierarchy still reproduces the exact cut
  const v = validateDagHierarchy(dag, gpuHier);
  if (!v.ok) fail(`${label}: reconstructed ${v.msg}`);
  else console.log(`  ${label}: GPU round-trip OK — ${v.msg} (clusterBase ${clusterBase})`);
}

console.log('[probe-hier] GPU-format hierarchy round-trip');
roundTrip('rock-small', buildRock('boulder', new Rng(1237), 3).geometry);
roundTrip('rock-hero', buildRock('boulder', new Rng(1241), 7).geometry);
roundTrip('bark-beech', buildTree(BEECH, new Rng(77), { lod: 1 }).bark);

if (failures > 0) {
  console.error(`[probe-hier] ${failures} FAILURES`);
  process.exit(1);
}
console.log('[probe-hier] GPU plumbing reproduces the validated hierarchy');
