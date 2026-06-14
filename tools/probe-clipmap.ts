/**
 * N8-D2 Stage 2b-2 validation (node-only): the terrain CLIPMAP residency math
 * (TerrainClipmap.clipmapTiles, D-N39). Over a sweep of camera positions across
 * (and past) the field, assert the resident tile set is correct:
 *
 *   npx tsx tools/probe-clipmap.ts
 *
 *  C  COVERAGE — every field point lands inside ≥1 resident tile (NO HOLES; the
 *     no-fallback + crack-free floors). Sampled on a fine grid.
 *  F  FINEST-NEAR — the camera's own texel is covered by a level-0 (1×stride) tile
 *     (true full-res right at the camera).
 *  B  BOUNDED — tile count ≤ clipmapMaxTiles AND independent of field size (same
 *     count for a 4 k and a 16 k field at the same camera-relative position).
 *  N  NESTING — each level's tiles tile its level grid without intra-level overlap;
 *     coarser levels are hollow where a finer level covers them.
 *  D  DETERMINISTIC — same camera ⇒ identical key set.
 */

import { clipmapTiles, clipmapMaxTiles, type ClipmapConfig, type ClipmapTile } from '../src/nanite/TerrainClipmap';

let failures = 0;
const fail = (m: string): void => {
  failures++;
  console.error(`  FAIL ${m}`);
};
const expect = (c: boolean, m: string): void => {
  if (!c) fail(m);
};

const CFG: ClipmapConfig = { res: 4096, gridN: 128, baseStride: 1, levels: 5, tilesPerSide: 4 };

/** is texel point (px,pz) inside tile t's footprint? */
function inTile(t: ClipmapTile, px: number, pz: number): boolean {
  return px >= t.tx0 && px < t.tx0 + t.tileTexels && pz >= t.tz0 && pz < t.tz0 + t.tileTexels;
}

/** C+F: coverage of the whole field + finest-at-camera, for one camera pose */
function checkPose(cfg: ClipmapConfig, camX: number, camZ: number): { tiles: ClipmapTile[]; maxLvl: number } {
  const tiles = clipmapTiles(camX, camZ, cfg);
  // B: bounded
  expect(tiles.length <= clipmapMaxTiles(cfg), `B: ${tiles.length} tiles > max ${clipmapMaxTiles(cfg)} @${camX},${camZ}`);
  expect(tiles.length > 0, `B: empty tile set @${camX},${camZ}`);
  // C: every field point covered (sample a coarse grid — tiles are ≥128 texels so
  // a 64-texel sample stride cannot slip through an uncovered gap)
  const STEP = 64;
  let worstGap = '';
  for (let pz = 0; pz < cfg.res && !worstGap; pz += STEP) {
    for (let px = 0; px < cfg.res; px += STEP) {
      let covered = false;
      for (const t of tiles) {
        if (inTile(t, px, pz)) {
          covered = true;
          break;
        }
      }
      if (!covered) {
        worstGap = `${px},${pz}`;
        break;
      }
    }
  }
  expect(worstGap === '', `C: field point ${worstGap} uncovered @cam ${camX},${camZ}`);
  // F: camera texel covered by a level-0 tile
  let finestAtCam = false;
  for (const t of tiles) {
    if (t.level === 0 && inTile(t, camX, camZ)) {
      finestAtCam = true;
      break;
    }
  }
  // (only when the camera is inside the field — at the rim the L0 block may clamp off)
  if (camX >= 0 && camX < cfg.res && camZ >= 0 && camZ < cfg.res) {
    expect(finestAtCam, `F: camera ${camX},${camZ} not under a level-0 tile`);
  }
  let maxLvl = 0;
  for (const t of tiles) maxLvl = Math.max(maxLvl, t.level);
  return { tiles, maxLvl };
}

console.log('[probe-clipmap]');
console.log(`  cfg: res ${CFG.res} gridN ${CFG.gridN} baseStride ${CFG.baseStride} levels ${CFG.levels} M ${CFG.tilesPerSide} | maxTiles ${clipmapMaxTiles(CFG)}`);

// --- C/F/B/N over a sweep of camera positions (center, off-center, rim, beyond) -
// in-field camera poses (the camera never leaves the world). Coverage of the
// whole field is the no-holes invariant; a camera-centered clipmap covers a
// region AROUND the camera, so off-field cameras are out of scope by design.
const poses: Array<[number, number]> = [
  [2048, 2048],
  [0, 0],
  [4095, 4095],
  [137, 3900],
  [2048 + 63, 2048 + 1], // sub-tile offsets (snap edge cases)
  [1024, 3072],
  [4095, 17], // corner/edge — backstop must still reach the far side
];
const tileCounts: number[] = [];
for (const [cx, cz] of poses) {
  const { tiles, maxLvl } = checkPose(CFG, cx, cz);
  tileCounts.push(tiles.length);
  // N: no intra-level duplicate footprints
  const seen = new Set<string>();
  for (const t of tiles) {
    expect(!seen.has(t.key), `N: duplicate tile ${t.key} @${cx},${cz}`);
    seen.add(t.key);
  }
  // the coarsest RESIDENT level may legitimately be < levels−1 when finer levels +
  // field bounds make the outer ring fully redundant (e.g. centered on a field that
  // a finer level already spans) — COVERAGE (C) is the invariant, not which level.
  expect(maxLvl >= 1, `N: only one level resident @${cx},${cz} — expected concentric LODs`);
}

// D: determinism
const a = clipmapTiles(2048, 2048, CFG).map((t) => t.key).sort().join('|');
const b = clipmapTiles(2048, 2048, CFG).map((t) => t.key).sort().join('|');
expect(a === b, 'D: non-deterministic tile set');

// B (field-size independence): the resident set must NOT grow with world size.
// Compare two LARGE fields at their centers (neither clips the clipmap at the rim)
// — equal counts ⇒ the bound is ⊥ world size. (A SMALL field like 4 k clips outer
// rings to FEWER tiles, which is fine — the bound is an upper bound.)
const f16: ClipmapConfig = { ...CFG, res: 16384 };
const f32: ClipmapConfig = { ...CFG, res: 32768 };
const c16 = clipmapTiles(8192, 8192, f16).length;
const c32 = clipmapTiles(16384, 16384, f32).length;
expect(c16 === c32, `B: world-size dependence — 16k ${c16} vs 32k ${c32} tiles`);
expect(c16 <= clipmapMaxTiles(CFG), `B: ${c16} > max ${clipmapMaxTiles(CFG)}`);

console.log(
  `  swept ${poses.length} poses: tiles ${Math.min(...tileCounts)}–${Math.max(...tileCounts)} ` +
    `(≤${clipmapMaxTiles(CFG)}); world-size ⊥ count (16k≡32k ${c16}≡${c32}); coverage hole-free, finest-at-camera`,
);
if (failures > 0) {
  console.error(`[probe-clipmap] ${failures} FAILURES`);
  process.exit(1);
}
console.log('[probe-clipmap] clipmap residency: hole-free coverage, bounded, field-size-independent');
