/**
 * Clusterizer validation probe (node-only, no browser): clusterizes real
 * generated rocks at three detail levels and asserts the structural
 * invariants, then prints stats + an extrapolated all-pools build time.
 *
 *   npx tsx tools/probe-clusterize.ts
 *
 * Invariants checked per mesh:
 *  P  output indices are a triangle-level permutation of the input
 *  C  cluster tri ranges are contiguous, complete, and ≤ maxTris
 *  S  every cluster vertex lies inside its bounding sphere (rel eps)
 *  N  every face normal lies inside its cluster's cone (or cone disabled)
 */

import { clusterize } from '../src/nanite/Clusterize';
import { Rng } from '../src/core/Seed';
import { buildRock } from '../src/vegetation/RockBuilder';

let failures = 0;
const fail = (msg: string): void => {
  failures++;
  console.error(`  FAIL ${msg}`);
};

function checkMesh(name: string, positions: Float32Array, posStride: number, indices: Uint32Array): void {
  const built = clusterize(positions, posStride, indices, 128);
  const st = built.stats;

  // P: permutation at triangle granularity
  const key = (arr: Uint32Array, t: number): string =>
    `${arr[t * 3]},${arr[t * 3 + 1]},${arr[t * 3 + 2]}`;
  const seen = new Map<string, number>();
  const triN = indices.length / 3;
  for (let t = 0; t < triN; t++) {
    const k = key(indices, t);
    seen.set(k, (seen.get(k) ?? 0) + 1);
  }
  for (let t = 0; t < triN; t++) {
    const k = key(built.indices, t);
    const n = seen.get(k);
    if (n === undefined || n === 0) {
      fail(`P: output tri ${t} (${k}) not in input multiset`);
      break;
    }
    seen.set(k, n - 1);
  }

  // C: ranges
  let covered = 0;
  for (let c = 0; c < built.clusterCount; c++) {
    const start = built.triStart[c] as number;
    const count = built.triCount[c] as number;
    if (start !== covered) fail(`C: cluster ${c} start ${start} != ${covered}`);
    if (count < 1 || count > 128) fail(`C: cluster ${c} count ${count}`);
    covered += count;
  }
  if (covered !== triN) fail(`C: covered ${covered} != ${triN}`);

  // S + N
  let worstS = 0;
  let worstN = 0;
  for (let c = 0; c < built.clusterCount; c++) {
    const sx = built.sphere[c * 4] as number;
    const sy = built.sphere[c * 4 + 1] as number;
    const sz = built.sphere[c * 4 + 2] as number;
    const sr = built.sphere[c * 4 + 3] as number;
    const ax = built.cone[c * 4] as number;
    const ay = built.cone[c * 4 + 1] as number;
    const az = built.cone[c * 4 + 2] as number;
    const cosA = built.cone[c * 4 + 3] as number;
    const start = built.triStart[c] as number;
    const count = built.triCount[c] as number;
    for (let t = start; t < start + count; t++) {
      for (let v = 0; v < 3; v++) {
        const p = (built.indices[t * 3 + v] as number) * posStride;
        const dx = (positions[p] as number) - sx;
        const dy = (positions[p + 1] as number) - sy;
        const dz = (positions[p + 2] as number) - sz;
        const d = Math.hypot(dx, dy, dz);
        if (d > sr * (1 + 1e-4) + 1e-6) worstS = Math.max(worstS, d - sr);
      }
      if (cosA > -1) {
        const i0 = (built.indices[t * 3] as number) * posStride;
        const i1 = (built.indices[t * 3 + 1] as number) * posStride;
        const i2 = (built.indices[t * 3 + 2] as number) * posStride;
        const e1x = (positions[i1] as number) - (positions[i0] as number);
        const e1y = (positions[i1 + 1] as number) - (positions[i0 + 1] as number);
        const e1z = (positions[i1 + 2] as number) - (positions[i0 + 2] as number);
        const e2x = (positions[i2] as number) - (positions[i0] as number);
        const e2y = (positions[i2 + 1] as number) - (positions[i0 + 1] as number);
        const e2z = (positions[i2 + 2] as number) - (positions[i0 + 2] as number);
        let fx = e1y * e2z - e1z * e2y;
        let fy = e1z * e2x - e1x * e2z;
        let fz = e1x * e2y - e1y * e2x;
        const fl = Math.hypot(fx, fy, fz);
        if (fl > 1e-12) {
          fx /= fl;
          fy /= fl;
          fz /= fl;
          const d = fx * ax + fy * ay + fz * az;
          if (d < cosA - 1e-4) worstN = Math.max(worstN, cosA - d);
        }
      }
    }
  }
  if (worstS > 0) fail(`S: vertex outside sphere by ${worstS.toExponential(2)}`);
  if (worstN > 0) fail(`N: face outside cone by ${worstN.toExponential(2)}`);

  console.log(
    `  ${name}: ${st.tris} tris -> ${st.clusters} clusters | avg ${st.avgTris.toFixed(1)} ` +
      `min ${st.minTris} full ${(st.fullFrac * 100).toFixed(0)}% | rFrac ${st.meanRadiusFrac.toFixed(3)} | ${st.buildMs.toFixed(1)} ms (adj ${st.adjMs.toFixed(0)} grow ${st.growMs.toFixed(0)} metrics ${st.metricsMs.toFixed(0)})`,
  );
}

function geoOf(detail: number, label: string): void {
  const r = buildRock('boulder', new Rng(1234 + detail), detail);
  const pos = r.geometry.attributes.position;
  const idx = r.geometry.index;
  if (!pos || !idx) throw new Error('rock not indexed');
  checkMesh(
    `${label} (detail ${detail})`,
    pos.array as Float32Array,
    pos.itemSize,
    new Uint32Array(idx.array as ArrayLike<number>),
  );
}

console.log('[probe-clusterize]');
geoOf(3, 'rock-small');
geoOf(5, 'rock-mid');
geoOf(7, 'rock-hero');

// extrapolation: time a 327k mesh, scale to the 10–20M all-pools budget
{
  const r = buildRock('boulder', new Rng(99), 7);
  const pos = r.geometry.attributes.position;
  const idx = r.geometry.index;
  if (pos && idx) {
    const indices = new Uint32Array(idx.array as ArrayLike<number>);
    const t0 = performance.now();
    clusterize(pos.array as Float32Array, pos.itemSize, indices, 128);
    const ms = performance.now() - t0;
    const perMTri = ms / (indices.length / 3 / 1e6);
    console.log(
      `  throughput ${(1000 / perMTri).toFixed(2)} Mtri/s -> 20M source tris ≈ ${((perMTri * 20) / 1000).toFixed(2)} s`,
    );
  }
}

if (failures > 0) {
  console.error(`[probe-clusterize] ${failures} FAILURES`);
  process.exit(1);
}
console.log('[probe-clusterize] all invariants hold');
