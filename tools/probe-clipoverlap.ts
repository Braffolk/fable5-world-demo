/**
 * N8-D2 clipmap OVERLAP gate (node-only). Two clipmap tiles of DIFFERENT levels
 * must never cover the same field XZ — if they do, both rasterize the same ground
 * at different heights ⇒ Z-FIGHTING (the user's "heavy z-fight from altitude").
 * The 2d skirts seal inter-level CRACKS, not overlap. This sweeps a grid of camera
 * positions, calls clipmapTiles, and asserts ZERO cross-level rectangle overlap.
 *
 *   npx tsx tools/probe-clipoverlap.ts
 */
import { clipmapTiles, type ClipmapConfig } from '../src/nanite/TerrainClipmap';
import { HEIGHT_RES } from '../src/world/WorldConst';

const RES = HEIGHT_RES;
const GRID_N = 128;
const M = 4;
const levels = Math.max(1, Math.ceil(Math.log2((2 * RES) / (M * GRID_N))) + 1);
const cfg: ClipmapConfig = { res: RES, gridN: GRID_N, baseStride: 1, levels, tilesPerSide: M };

/** area of XZ-rectangle intersection of two tiles (0 = disjoint/touching). */
function overlapArea(a: { tx0: number; tz0: number; tileTexels: number }, b: { tx0: number; tz0: number; tileTexels: number }): number {
  const ix = Math.max(0, Math.min(a.tx0 + a.tileTexels, b.tx0 + b.tileTexels) - Math.max(a.tx0, b.tx0));
  const iz = Math.max(0, Math.min(a.tz0 + a.tileTexels, b.tz0 + b.tileTexels) - Math.max(a.tz0, b.tz0));
  return ix * iz;
}

let worstOverlap = 0;
let worstAt = '';
let badPositions = 0;
let total = 0;
const STEP = 37; // a prime-ish step to hit many sub-tile phases (aligned + misaligned)
for (let cz = 0; cz < RES; cz += STEP) {
  for (let cx = 0; cx < RES; cx += STEP) {
    total++;
    const tiles = clipmapTiles(cx, cz, cfg);
    let posWorst = 0;
    for (let i = 0; i < tiles.length; i++) {
      for (let j = i + 1; j < tiles.length; j++) {
        const ti = tiles[i]!;
        const tj = tiles[j]!;
        if (ti.level === tj.level) continue; // same level = a grid, never overlaps
        const a = overlapArea(ti, tj);
        if (a > posWorst) posWorst = a;
      }
    }
    if (posWorst > 0) {
      badPositions++;
      if (posWorst > worstOverlap) {
        worstOverlap = posWorst;
        worstAt = `(${cx},${cz})`;
      }
    }
  }
}

console.log(`[clipoverlap] ${levels} levels, gridN ${GRID_N}, M ${M}, res ${RES}; swept ${total} camera positions (step ${STEP})`);
console.log(`  positions with cross-level overlap: ${badPositions}/${total}`);
console.log(`  worst overlap area: ${worstOverlap} texels² at ${worstAt || 'n/a'}`);
if (badPositions === 0) {
  console.log('[clipoverlap] PASS — levels nest exactly, no cross-level overlap anywhere (no z-fight source)');
} else {
  console.log(`[clipoverlap] FAIL — ${badPositions} positions have overlapping LODs ⇒ z-fighting`);
  process.exit(1);
}
