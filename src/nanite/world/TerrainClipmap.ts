/**
 * N8-D2 Stage 2b-2 (D-N39): the terrain geometry CLIPMAP — the residency policy
 * for streamed full-res terrain. The field is covered by L concentric LEVELS of
 * same-gridN tiles at DOUBLING stride (level k cell = baseStride·2^k texels), each
 * level an M×M block of tiles centered on the camera. A level's inner block is a
 * HOLE — covered by the finer level k−1 — so the resident tiles form hollow rings
 * (the finest full block at the center, coarser rings outward; the coarsest level
 * spans the whole field = the always-resident backstop, no holes ever).
 *
 * Because every level uses the SAME gridN, every tile fits ONE uniform pool-slot
 * cap (the 2a tile pool holds them all). The resident tile COUNT is bounded by
 * Σ(per-level ring counts) ≈ M² + (L−1)(M² − (M/2)²), INDEPENDENT of field size —
 * the memory/cull bound for arbitrarily large worlds.
 *
 * This module is the pure GEOMETRY of the clipmap (which tiles, at which texel
 * windows). Building/loading/evicting tiles is the registry/streamer's job. Pure
 * integer math, node-testable (probe-clipmap.ts).
 */

export interface ClipmapConfig {
  /** field texels per side (heightTex resolution) */
  res: number;
  /** tile resolution — cells per side (every level, every tile) */
  gridN: number;
  /** finest stride in texels/cell (1 = true 1 m at a 1 m field) */
  baseStride: number;
  /** number of concentric levels */
  levels: number;
  /** tiles per side per level (EVEN; the inner M/2 block is the hole) */
  tilesPerSide: number;
}

export interface ClipmapTile {
  /** clipmap level (0 = finest) */
  level: number;
  /** texels per cell at this level = baseStride << level */
  strideTexels: number;
  /** tile texel origin (may be <0 or ≥res at the field rim — clamp on sample) */
  tx0: number;
  tz0: number;
  /** tile side in texels = gridN * strideTexels */
  tileTexels: number;
  /** stable identity within a config: (level, tile-grid-x, tile-grid-z) */
  key: string;
}

/**
 * The resident clipmap tile set for a camera at texel (camX, camZ). Hollow rings:
 * a level-k tile is omitted when it lies FULLY inside the finer level (k−1)'s
 * extent. Each level snaps to 2× its tile grid, so consecutive levels NEST EXACTLY
 * (the finer extent lands on coarse tile boundaries) ⇒ every coarse tile is fully
 * inside the finer extent (dropped) or fully outside (kept), never a partial straddle
 * — no cross-level overlap, no gap. Tiles fully outside the field are dropped.
 */
export function clipmapTiles(camX: number, camZ: number, cfg: ClipmapConfig): ClipmapTile[] {
  const { res, gridN, baseStride, levels, tilesPerSide: M } = cfg;
  if (M % 2 !== 0 || M < 2) throw new Error(`clipmap: tilesPerSide must be even ≥2, got ${M}`);
  if (levels < 1) throw new Error(`clipmap: levels must be ≥1, got ${levels}`);
  const tiles: ClipmapTile[] = [];
  let prev: { x0: number; z0: number; x1: number; z1: number } | null = null;
  for (let k = 0; k < levels; k++) {
    const stride = baseStride * (1 << k);
    const Tk = gridN * stride; // tile side in texels
    const half = (M / 2) * Tk;
    // Snap each level's origin to 2× its tile grid (ROUNDED). This makes the next
    // coarser level's tiles fall EXACTLY on this level's extent boundary, so every
    // coarse tile is either fully inside the finer extent (dropped by the hollow) or
    // fully outside (kept) — NEVER a partial straddle. The old per-level snapDown(…,Tk)
    // misaligned consecutive levels (the finer extent is only M/2 coarse-tiles wide, so
    // when misaligned it fully contained NO coarse tile ⇒ the finer level OVERLAID 2-3
    // whole coarse tiles): both LODs rasterized the same ground at different heights ⇒
    // heavy z-fighting + ~2× wasted terrain draw. (Skirts seal cracks, not overlap.)
    // Rounding (not floor) keeps the camera within ±Tk of the level centre.
    const snap = 2 * Tk;
    const ox = Math.round((camX - half) / snap) * snap;
    const oz = Math.round((camZ - half) / snap) * snap;
    const ext = { x0: ox, z0: oz, x1: ox + M * Tk, z1: oz + M * Tk };
    for (let tj = 0; tj < M; tj++) {
      for (let ti = 0; ti < M; ti++) {
        const tx0 = ox + ti * Tk;
        const tz0 = oz + tj * Tk;
        // drop tiles entirely off the field
        if (tx0 >= res || tz0 >= res || tx0 + Tk <= 0 || tz0 + Tk <= 0) continue;
        // hollow: the finer level already covers this tile fully
        if (prev && tx0 >= prev.x0 && tz0 >= prev.z0 && tx0 + Tk <= prev.x1 && tz0 + Tk <= prev.z1) {
          continue;
        }
        // stable key: level + tile index on this level's grid (snap-independent)
        const gx = Math.round(tx0 / Tk);
        const gz = Math.round(tz0 / Tk);
        tiles.push({ level: k, strideTexels: stride, tx0, tz0, tileTexels: Tk, key: `L${k}:${gx},${gz}` });
      }
    }
    prev = ext;
  }
  return tiles;
}

/** upper bound on resident tiles for a config — pool slot count (+ a margin). */
export function clipmapMaxTiles(cfg: ClipmapConfig): number {
  const M = cfg.tilesPerSide;
  const inner = (M / 2) * (M / 2);
  // level 0 is full M²; each outer level sheds at most its inner M/2 block
  return M * M + Math.max(0, cfg.levels - 1) * (M * M - inner);
}
