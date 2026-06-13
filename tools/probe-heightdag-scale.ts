/**
 * Terrain-DAG build SCALING probe (temporary, D2b speed work). Builds the same
 * bimodal field at several gridN and reports total build ms + Mtri/s, to reveal
 * whether throughput is flat (constant-factor cost — reseed/guards) or falls with
 * size (super-linear — per-group or level-count blowup). Decides the speed fix.
 *
 *   npx tsx tools/probe-heightdag-scale.ts
 */

import { buildHeightDag, type HeightField } from '../src/nanite/BuildHeightDag';

function synthField(gridN: number, cellSize: number): HeightField {
  const vpa = gridN + 1;
  const heights = new Float32Array(vpa * vpa);
  for (let gz = 0; gz <= gridN; gz++) {
    for (let gx = 0; gx <= gridN; gx++) {
      const fx = gx / gridN;
      const fz = gz / gridN;
      let h: number;
      if (fx < 0.4) {
        h = 0;
      } else if (fx < 0.5) {
        h = (fx - 0.4) * 40 + fz * 6;
      } else {
        const u = (fx - 0.5) * gridN * cellSize;
        const w = fz * gridN * cellSize;
        h = 6 + 5 * Math.sin(u * 0.7) * Math.cos(w * 0.55) + 2.5 * Math.sin(u * 1.9 + w * 0.3);
      }
      heights[gz * vpa + gx] = h;
    }
  }
  return { heights, gridN, cellSize, originX: 0, originZ: 0 };
}

console.log('[probe-heightdag-scale]');
console.log('  gridN   LOD0tris   levels   cl    roots   buildMs   Mtri/s');
for (const gridN of [128, 256, 512, 1024]) {
  const hf = synthField(gridN, 2560 / gridN);
  // warm once at the smallest size to amortise JIT, then measure.
  const dag = buildHeightDag(hf);
  const tris = dag.stats.lod0Tris;
  const ms = dag.stats.buildMs;
  console.log(
    `  ${String(gridN).padEnd(6)}  ${String(tris).padEnd(9)}  ${String(dag.stats.levels).padEnd(7)}  ` +
      `${String(dag.stats.totalClusters).padEnd(5)} ${String(dag.stats.roots).padEnd(7)} ${ms.toFixed(0).padEnd(8)}  ${((tris / 1e6 / ms) * 1000).toFixed(3)}`,
  );
}
