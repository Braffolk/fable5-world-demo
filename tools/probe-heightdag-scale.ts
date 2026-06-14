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

// memory cost of STORING the full DAG in the registry mega-buffers (the cut
// draws a subset, but every level is stored): indices = totalTris·3·u32; verts
// today = 6-word VERT_WORDS, terrain only needs 1 word (packed grid coord) → the
// stride-1 win. This reveals the full-res wall (a single 4096² DAG vs tiling).
console.log('[probe-heightdag-scale]');
console.log('  gridN  LOD0tris  totVerts  totTris   cl      buildMs   idxMB  v6MB  v1MB  totMB(v1)');
for (const gridN of [512, 1024, 2048]) {
  const hf = synthField(gridN, 2560 / gridN);
  const dag = buildHeightDag(hf);
  const totVerts = dag.gridVerts.length;
  const totTris = dag.stats.totalTris;
  const idxMB = (totTris * 3 * 4) / 1e6;
  const v6MB = (totVerts * 6 * 4) / 1e6;
  const v1MB = (totVerts * 4) / 1e6;
  console.log(
    `  ${String(gridN).padEnd(5)}  ${String(dag.stats.lod0Tris).padEnd(8)}  ${String(totVerts).padEnd(8)}  ` +
      `${String(totTris).padEnd(8)}  ${String(dag.stats.totalClusters).padEnd(6)}  ${dag.stats.buildMs.toFixed(0).padEnd(8)}  ` +
      `${idxMB.toFixed(0).padEnd(5)}  ${v6MB.toFixed(0).padEnd(4)}  ${v1MB.toFixed(0).padEnd(4)}  ${(idxMB + v1MB).toFixed(0)}`,
  );
}
// 4096² is the real field res; extrapolate from the gridN² LOD0 floor rather than
// build it here (minutes in node). LOD0(4096) = 33.5M tris alone.
console.log('  4096   33554432  (extrapolate ×4 from 2048: idx+vert MB scale ~×4)');
