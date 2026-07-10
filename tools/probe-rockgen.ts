/**
 * RockGen validation probe (node-only, no browser): generates the full
 * SPEC-ROCKS §B library (22 variants) and asserts the R0 gate invariants,
 * then prints a summary table + registry-VRAM estimate vs the 12 MB ceiling.
 *
 *   npx tsx tools/probe-rockgen.ts [--obj <path>]
 *
 * Invariants checked per variant:
 *  M  manifold + winding + volume audits pass (generateRock throws on fail)
 *  T  LOD0 tri count within ±25% of the §B table target
 *  V  vertex count < 65k
 *  D  determinism: a second run with the same seed is byte-identical across
 *     positions/normals/vdata/indices
 *
 * --obj writes the first Boulder graniteErratic as a Wavefront OBJ for
 * eyeballing (e.g. into the session scratchpad).
 */

import { writeFileSync } from 'node:fs';
import { generateRock, ROCK_LIBRARY } from '../src/vegetation/RockGen';
import type { RockMesh } from '../src/vegetation/RockGen';

const SEED = 1337;
const TRI_TOL = 0.25;
const VRAM_CEILING_MB = 12;

let failures = 0;
const fail = (msg: string): void => {
  failures++;
  console.error(`  FAIL ${msg}`);
};

function sameBytes(a: ArrayBufferView, b: ArrayBufferView): boolean {
  if (a.byteLength !== b.byteLength) return false;
  const ua = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
  const ub = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
  for (let i = 0; i < ua.length; i++) if (ua[i] !== ub[i]) return false;
  return true;
}

function writeObj(path: string, mesh: RockMesh): void {
  const parts: string[] = [];
  const p = mesh.positions;
  const n = mesh.normals;
  for (let i = 0; i < p.length / 3; i++) {
    parts.push(`v ${p[i * 3]} ${p[i * 3 + 1]} ${p[i * 3 + 2]}`);
    parts.push(`vn ${n[i * 3]} ${n[i * 3 + 1]} ${n[i * 3 + 2]}`);
  }
  const ix = mesh.indices;
  for (let t = 0; t < ix.length / 3; t++) {
    const a = (ix[t * 3] as number) + 1;
    const b = (ix[t * 3 + 1] as number) + 1;
    const c = (ix[t * 3 + 2] as number) + 1;
    parts.push(`f ${a}//${a} ${b}//${b} ${c}//${c}`);
  }
  writeFileSync(path, parts.join('\n'));
  console.log(`  OBJ written: ${path} (${p.length / 3} verts, ${ix.length / 3} tris)`);
}

const objIdx = process.argv.indexOf('--obj');
const objPath = objIdx >= 0 ? process.argv[objIdx + 1] : undefined;

console.log('[probe-rockgen] seed', SEED);
console.log(
  '  class        v archetype             res    tris  target      Δ    verts  evals/full     ms',
);

let totTris = 0;
let totVerts = 0;
let totMs = 0;
let totPinch = 0;
let heroMesh: RockMesh | undefined;
let objMesh: RockMesh | undefined;

for (const cls of ROCK_LIBRARY) {
  for (let v = 0; v < cls.variants.length; v++) {
    const spec = cls.variants[v];
    if (!spec) continue;
    const t0 = performance.now();
    let mesh: RockMesh;
    try {
      mesh = generateRock(spec.archetype, v, SEED, cls.gridRes, spec.mod, spec.domainScale ?? 1);
    } catch (e) {
      fail(`M: ${cls.name} v${v} ${spec.archetype}: ${(e as Error).message}`);
      continue;
    }
    const ms = performance.now() - t0;
    const st = mesh.stats;

    const dev = (st.tris - cls.targetTris) / cls.targetTris;
    if (Math.abs(dev) > TRI_TOL) {
      fail(`T: ${cls.name} v${v} tris ${st.tris} vs target ${cls.targetTris} (${(dev * 100).toFixed(0)}%)`);
    }
    if (st.verts >= 65536) fail(`V: ${cls.name} v${v} verts ${st.verts}`);
    if (st.droppedTris > 0) console.log(`  note: ${cls.name} v${v} dropped ${st.droppedTris} zero-area tris`);
    totPinch += st.pinchEdges;

    // D: determinism — regenerate, byte-compare all four arrays
    const again = generateRock(spec.archetype, v, SEED, cls.gridRes, spec.mod, spec.domainScale ?? 1);
    if (
      !sameBytes(mesh.positions, again.positions) ||
      !sameBytes(mesh.normals, again.normals) ||
      !sameBytes(mesh.vdata, again.vdata) ||
      !sameBytes(mesh.indices, again.indices)
    ) {
      fail(`D: ${cls.name} v${v} not byte-identical across runs`);
    }

    const fullEvals = (cls.gridRes + 1) ** 3;
    const modTag = spec.mod ? `:${spec.mod}` : '';
    console.log(
      `  ${cls.name.padEnd(12)} ${v} ${(spec.archetype + modTag).padEnd(21)} ${String(cls.gridRes).padStart(3)} ` +
        `${String(st.tris).padStart(7)} ${String(cls.targetTris).padStart(7)} ${((dev >= 0 ? '+' : '') + (dev * 100).toFixed(0) + '%').padStart(6)} ` +
        `${String(st.verts).padStart(8)} ${((st.fieldEvals / fullEvals) * 100).toFixed(0).padStart(9)}% ${ms.toFixed(0).padStart(6)}`,
    );
    totTris += st.tris;
    totVerts += st.verts;
    totMs += ms;
    if (cls.name === 'EtakErratic' && v === 0) heroMesh = mesh;
    if (cls.name === 'Boulder' && v === 0) objMesh = mesh;
  }
}

// vdata channel spread (R2 shading needs real variation, not flat channels)
for (const [label, mesh] of [['Boulder v0', objMesh], ['EtakErratic v0', heroMesh]] as const) {
  if (!mesh) continue;
  const mins = [255, 255, 255, 255];
  const maxs = [0, 0, 0, 0];
  const sums = [0, 0, 0, 0];
  const n = mesh.vdata.length / 4;
  for (let i = 0; i < n; i++) {
    for (let ch = 0; ch < 4; ch++) {
      const val = mesh.vdata[i * 4 + ch] as number;
      if (val < (mins[ch] as number)) mins[ch] = val;
      if (val > (maxs[ch] as number)) maxs[ch] = val;
      sums[ch] = (sums[ch] as number) + val;
    }
  }
  const chans = ['curv', 'flow', 'ao*up', 'cavity']
    .map((c, ch) => `${c} [${mins[ch]}..${maxs[ch]}] μ${((sums[ch] as number) / n).toFixed(0)}`)
    .join('  ');
  console.log(`  vdata ${label}: ${chans}  | boundR ${mesh.stats.boundRadius.toFixed(2)}`);
}

// registry VRAM estimate (§B model): vertex 24 B, tri 12 B, cluster rec
// 32 B + DAG sidecar 48 B per 128 tris; DAG total ≈ 2× LOD0.
const dagTris = totTris * 2;
const dagVerts = totVerts * 2;
const vramMB = (dagVerts * 24 + dagTris * 12 + Math.ceil(dagTris / 128) * (32 + 48)) / (1024 * 1024);
console.log(
  `  totals: LOD0 ${totTris} tris / ${totVerts} verts | ${totPinch} pinch edges ` +
    `(retired downstream) | est DAG ${dagTris} tris → registry ≈ ${vramMB.toFixed(1)} MB ` +
    `(ceiling ${VRAM_CEILING_MB} MB)`,
);
if (vramMB > VRAM_CEILING_MB) fail(`VRAM estimate ${vramMB.toFixed(1)} MB exceeds ${VRAM_CEILING_MB} MB ceiling`);
console.log(
  `  timing: ${(totMs / 1000).toFixed(2)} s single-thread for 22 variants ` +
    `(boot fans across the prepareWorldVeg worker pool; §G estimate 2-4 s cold)`,
);

if (objPath && objMesh) writeObj(objPath, objMesh);

if (failures > 0) {
  console.error(`[probe-rockgen] ${failures} FAILURES`);
  process.exit(1);
}
console.log('[probe-rockgen] all invariants hold');
