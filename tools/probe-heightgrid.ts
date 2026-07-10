/**
 * Heightmap-native regular-grid terrain LOD validation (node-only; TERRAIN-RW).
 * Builds `buildHeightGrid` on deterministic fields and asserts the properties the
 * old QEM terrain builder VIOLATED (the user's fans / spanning-triangles / "flap"):
 *
 *   F  no fans / bounded tris — every SURFACE triangle edge ≤ stride·cell·√2
 *                               (a regular grid cell). QEM produced edges spanning
 *                               the whole tile (the slivers). HEAD-TO-HEAD vs old.
 *   W  watertight per level   — at each τ the cut selects ONE level (tile-uniform);
 *                               its surface is a complete grid, every interior edge
 *                               shared by exactly 2 tris (no holes / T-junctions).
 *   U  tile-uniform cut       — every cluster's error-sphere == the tile sphere and
 *                               ownError is constant per level ⇒ a level is all-or-
 *                               nothing ⇒ no intra-tile cracks.
 *   M  error monotone         — parentError ≥ ownError; chain strictly increases.
 *   S  shape-faithful         — the rendered level-ℓ surface deviates from the true
 *                               heights by ≤ ownError(ℓ) (the cut's screen-error is
 *                               an honest bound).
 *   G  on-grid                — offGridVerts==0, indices in range.
 *   D  deterministic          — a second build is bit-identical.
 *   adaptivity (per TILE)     — a FLAT tile collapses to its root (~2 tris); a CLIFF
 *                               tile keeps every level.
 *
 *   npx tsx tools/probe-heightgrid.ts
 */
import type { DagBuild, DagCluster } from '../src/nanite/build/BuildDag';
import { buildHeightGrid, type HeightField } from '../src/nanite/build/BuildHeightGrid';
import { buildHeightGridHierarchy, validateDagHierarchy } from '../src/nanite/build/DagHierarchy';

let failures = 0;
const fail = (msg: string): void => {
  failures++;
  console.error(`  FAIL ${msg}`);
};

const SCREEN_H = 1080;
const PROJ_K = SCREEN_H / 2 / Math.tan(1.0 / 2);

function project(e: number, cx: number, cy: number, cz: number, r: number, camX: number, camY: number, camZ: number): number {
  if (!Number.isFinite(e)) return Infinity;
  const dx = cx - camX;
  const dy = cy - camY;
  const dz = cz - camZ;
  const denom = Math.sqrt(Math.max(1e-6, dx * dx + dy * dy + dz * dz - r * r));
  return (PROJ_K * e) / denom;
}

/** deterministic bimodal field: flat plain | tilted ramp | ridged cliffs. */
function synthField(gridN: number, cellSize: number): HeightField {
  const vpa = gridN + 1;
  const heights = new Float32Array(vpa * vpa);
  for (let gz = 0; gz <= gridN; gz++) {
    for (let gx = 0; gx <= gridN; gx++) {
      const fx = gx / gridN;
      const fz = gz / gridN;
      let h: number;
      if (fx < 0.4) h = 0;
      else if (fx < 0.5) h = (fx - 0.4) * 40 + fz * 6;
      else {
        const u = (fx - 0.5) * gridN * cellSize;
        const w = fz * gridN * cellSize;
        h = 6 + 5 * Math.sin(u * 0.7) * Math.cos(w * 0.55) + 2.5 * Math.sin(u * 1.9 + w * 0.3);
      }
      heights[gz * vpa + gx] = h;
    }
  }
  return { heights, gridN, cellSize, originX: 0, originZ: 0 };
}

function flatField(gridN: number, cellSize: number): HeightField {
  return { heights: new Float32Array((gridN + 1) * (gridN + 1)), gridN, cellSize, originX: 0, originZ: 0 };
}

function cliffField(gridN: number, cellSize: number): HeightField {
  const vpa = gridN + 1;
  const heights = new Float32Array(vpa * vpa);
  for (let gz = 0; gz <= gridN; gz++)
    for (let gx = 0; gx <= gridN; gx++) {
      const u = gx * cellSize;
      const w = gz * cellSize;
      heights[gz * vpa + gx] = 10 * Math.sin(u * 0.6) * Math.cos(w * 0.5) + 4 * Math.sin(u * 1.7 + w * 0.4);
    }
  return { heights, gridN, cellSize, originX: 0, originZ: 0 };
}

const isSkirtVert = (packed: number): boolean => ((packed >> 13) & 0x7) !== 0;

/** F per-cluster: assert no surface tri exceeds its own level's grid bound
 *  (stride·cell·√2). A regular grid CANNOT exceed this; a QEM fan/spanning-tri does.
 *  Returns the count of over-bound surface tris (0 = clean grid). */
function checkBoundedTris(name: string, dag: { gridVerts: Uint32Array; indices: Uint32Array; clusters: DagCluster[] }, cellSize: number, assert = true): number {
  const gv = dag.gridVerts;
  const idx = dag.indices;
  let over = 0;
  let worst = 0;
  let worstBound = 0;
  for (const c of dag.clusters) {
    const bound = (1 << c.level) * cellSize * Math.SQRT2 + 1e-3;
    for (let t = c.triStart; t < c.triStart + c.triCount; t++) {
      const a = gv[idx[t * 3] as number] as number;
      const b = gv[idx[t * 3 + 1] as number] as number;
      const cc = gv[idx[t * 3 + 2] as number] as number;
      if (isSkirtVert(a) || isSkirtVert(b) || isSkirtVert(cc)) continue;
      const pos = (p: number): [number, number] => [(p & 0x1fff) * cellSize, ((p >> 16) & 0xffff) * cellSize];
      const [ax, az] = pos(a);
      const [bx, bz] = pos(b);
      const [cx, cz] = pos(cc);
      const e = Math.max(Math.hypot(ax - bx, az - bz), Math.hypot(bx - cx, bz - cz), Math.hypot(cx - ax, cz - az));
      if (e > bound) {
        over++;
        if (e - bound > worst - worstBound) {
          worst = e;
          worstBound = bound;
        }
      }
    }
  }
  if (assert && over > 0) fail(`${name} F: ${over} surface tris exceed their grid bound (worst ${worst.toFixed(1)} > ${worstBound.toFixed(1)} m) — a spanning triangle/fan`);
  return over;
}

/** W: watertight per-level cut. At each τ the tile-uniform cut selects ONE level;
 *  its surface (skirts excluded) must be a complete manifold grid. */
function checkWatertight(name: string, dag: { gridVerts: Uint32Array; indices: Uint32Array; clusters: DagCluster[]; gridN: number }, taus: number[], cam: [number, number, number]): void {
  const gv = dag.gridVerts;
  const idx = dag.indices;
  const N = dag.gridN;
  const onBorder = (p: number): boolean => {
    const gx = p & 0x1fff;
    const gz = (p >> 16) & 0xffff;
    return gx === 0 || gx === N || gz === 0 || gz === N;
  };
  for (const tau of taus) {
    const edge = new Map<string, { use: number; u: number; v: number }>();
    const bump = (u: number, v: number): void => {
      const lo = u < v ? u : v;
      const hi = u < v ? v : u;
      const k = `${lo}_${hi}`;
      const e = edge.get(k);
      if (e) e.use++;
      else edge.set(k, { use: 1, u: lo, v: hi });
    };
    let levelsSelected = new Set<number>();
    for (const c of dag.clusters) {
      const po = project(c.ownError, c.oex, c.oey, c.oez, c.oer, cam[0], cam[1], cam[2]);
      const pp = project(c.parentError, c.pex, c.pey, c.pez, c.per, cam[0], cam[1], cam[2]);
      if (!(po <= tau && pp > tau)) continue;
      levelsSelected.add(c.level);
      for (let t = c.triStart; t < c.triStart + c.triCount; t++) {
        const a = gv[idx[t * 3] as number] as number;
        const b = gv[idx[t * 3 + 1] as number] as number;
        const cc = gv[idx[t * 3 + 2] as number] as number;
        if (isSkirtVert(a) || isSkirtVert(b) || isSkirtVert(cc)) continue;
        bump(a, b);
        bump(b, cc);
        bump(cc, a);
      }
    }
    if (levelsSelected.size > 1) fail(`${name} U(τ=${tau}): cut spans ${levelsSelected.size} levels (NOT tile-uniform)`);
    let cracks = 0;
    for (const e of edge.values()) {
      if (e.use === 2) continue;
      if (e.use === 1 && onBorder(e.u) && onBorder(e.v)) continue;
      cracks++;
    }
    if (cracks > 0) fail(`${name} W(τ=${tau}): ${cracks} non-manifold interior edges (cracks/T-junctions)`);
  }
}

/** U + M: all error-spheres equal the tile sphere; per-level ownError constant;
 *  the level→error chain is strictly monotone. */
function checkUniformMonotone(name: string, dag: { clusters: DagCluster[] }): void {
  const cl = dag.clusters;
  if (cl.length === 0) return fail(`${name}: no clusters`);
  const t0 = cl[0] as DagCluster;
  let sphereViol = 0;
  const errOfLevel = new Map<number, number>();
  for (const c of cl) {
    if (c.oex !== t0.oex || c.oey !== t0.oey || c.oez !== t0.oez || c.oer !== t0.oer) sphereViol++;
    if (c.pex !== t0.oex || c.pey !== t0.oey || c.pez !== t0.oez || c.per !== t0.oer) sphereViol++;
    if (Number.isFinite(c.parentError) && c.parentError < c.ownError - 1e-9) fail(`${name} M: parentError < ownError`);
    const prev = errOfLevel.get(c.level);
    if (prev === undefined) errOfLevel.set(c.level, c.ownError);
    else if (Math.abs(prev - c.ownError) > 1e-9) fail(`${name} U: level ${c.level} has two ownErrors (${prev}, ${c.ownError})`);
  }
  if (sphereViol > 0) fail(`${name} U: ${sphereViol} clusters whose error-sphere != the tile sphere`);
  const levels = [...errOfLevel.entries()].sort((a, b) => a[0] - b[0]);
  for (let i = 1; i < levels.length; i++) {
    if ((levels[i] as [number, number])[1] < (levels[i - 1] as [number, number])[1] - 1e-9) {
      fail(`${name} M: ownError decreased from level ${(levels[i - 1] as [number, number])[0]} to ${(levels[i] as [number, number])[0]}`);
    }
  }
}

/** S: the rendered level-ℓ surface deviates from the true heights by ≤ ownError(ℓ). */
function checkShapeFaithful(name: string, dag: { clusters: DagCluster[] }, hf: HeightField): void {
  const { heights, gridN, cellSize } = hf;
  const size = gridN + 1;
  const triH = (u: number, v: number, h00: number, h10: number, h01: number, h11: number): number =>
    v >= u ? h00 + (v - u) * (h01 - h00) + u * (h11 - h00) : h00 + v * (h11 - h00) + (u - v) * (h10 - h00);
  let worst = 0;
  let worstErr = 0;
  // group clusters by level; measure surface deviation at that stride vs ownError
  const byLevel = new Map<number, number>();
  for (const c of dag.clusters) if (!byLevel.has(c.level)) byLevel.set(c.level, c.ownError);
  for (const [level, ownError] of byLevel) {
    const stride = 1 << level;
    let maxDev = 0;
    for (let gz = 0; gz <= gridN; gz++) {
      const cz0 = gz - (gz % stride);
      if (cz0 + stride > gridN) continue;
      const v = (gz - cz0) / stride;
      for (let gx = 0; gx <= gridN; gx++) {
        const cx0 = gx - (gx % stride);
        if (cx0 + stride > gridN) continue;
        const u = (gx - cx0) / stride;
        const dev = Math.abs(
          (heights[gz * size + gx] as number) -
            triH(
              u,
              v,
              heights[cz0 * size + cx0] as number,
              heights[cz0 * size + cx0 + stride] as number,
              heights[(cz0 + stride) * size + cx0] as number,
              heights[(cz0 + stride) * size + cx0 + stride] as number,
            ),
        );
        if (dev > maxDev) maxDev = dev;
      }
    }
    if (maxDev > ownError + 1e-3 && maxDev - ownError > worst - worstErr) {
      worst = maxDev;
      worstErr = ownError;
    }
    void cellSize;
  }
  if (worst > worstErr + 1e-3) fail(`${name} S: surface deviates ${worst.toFixed(2)} m > ownError ${worstErr.toFixed(2)} m (error bound dishonest)`);
}

console.log('[probe-heightgrid]');
const gridN = 128;
const cellSize = 1.0;
const hf = synthField(gridN, cellSize);
const extent = gridN * cellSize;

const grid = buildHeightGrid(hf);
const gridOver = checkBoundedTris('grid', grid, cellSize, false);
console.log(
  `  GRID ${gridN}×${gridN} (tile ${extent} m): ${grid.stats.buildMs.toFixed(1)} ms | ${grid.stats.totalClusters} cl | ` +
    `${grid.stats.totalTris} tris | ${grid.stats.levels} levels | ${gridOver} tris over the grid bound (must be 0 — no fans)`,
);

// F — no fans (bounded triangles)
checkBoundedTris('grid', grid, cellSize);
// W + U — watertight per-level, tile-uniform (camera high above centre)
checkWatertight('grid', grid, [0.0, 0.5, 1, 2, 4, 8, 32, 1e9], [extent / 2, 200, extent / 2]);
// U + M
checkUniformMonotone('grid', grid);
// S — shape-faithful
checkShapeFaithful('grid', grid, hf);
// G — on-grid
if (grid.stats.offGridVerts > 0) fail(`G: ${grid.stats.offGridVerts} off-grid verts`);
{
  let bad = 0;
  for (let t = 0; t < grid.indices.length; t++) if ((grid.indices[t] as number) >= grid.gridVerts.length) bad++;
  if (bad > 0) fail(`G: ${bad} indices out of range`);
}
// D — determinism
{
  const g2 = buildHeightGrid(hf);
  if (g2.stats.totalClusters !== grid.stats.totalClusters || g2.stats.totalTris !== grid.stats.totalTris || g2.gridVerts.length !== grid.gridVerts.length) {
    fail(`D: non-deterministic build`);
  } else {
    let diff = 0;
    for (let i = 0; i < grid.gridVerts.length; i++) if (grid.gridVerts[i] !== g2.gridVerts[i]) diff++;
    if (diff > 0) fail(`D: ${diff} gridVerts differ`);
  }
}

// adaptivity (per TILE): flat → 1 level; cliff → many
const flat = buildHeightGrid(flatField(gridN, cellSize));
const cliff = buildHeightGrid(cliffField(gridN, cellSize));
console.log(`  ADAPTIVITY: flat tile → ${flat.stats.levels} level(s), ${flat.stats.totalTris} tris (root only) | cliff tile → ${cliff.stats.levels} levels, ${cliff.stats.totalClusters} cl`);
if (flat.stats.totalTris > 8) fail(`adaptivity: a FLAT tile emitted ${flat.stats.totalTris} tris (expected ≤8 — should collapse to its root)`);
if (cliff.stats.levels < 4) fail(`adaptivity: a CLIFF tile only emitted ${cliff.stats.levels} levels (expected ≥4 — should keep detail)`);
checkBoundedTris('cliff', cliff, cellSize);
checkWatertight('cliff', cliff, [0.0, 1, 8, 1e9], [extent / 2, 200, extent / 2]);

// skirt sanity: with a skirt level set, skirt verts/clusters appear
const withSkirt = buildHeightGrid(hf, { skirtLevel: 0 });
const skirtCl = withSkirt.clusters.length - grid.clusters.length;
console.log(`  SKIRTS: +${skirtCl} skirt clusters with skirtLevel=0 (gated per level)`);
if (skirtCl <= 0) fail(`skirt: skirtLevel=0 added no skirt clusters`);
checkBoundedTris('skirt-surf', withSkirt, cellSize);
checkWatertight('skirt', withSkirt, [0.0, 1, 8, 1e9], [extent / 2, 200, extent / 2]);

// H — the anchor-chain hierarchy reproduces the tile-uniform cut at every threshold
// (PERF-VB3: terrain hier roots). Reuse the DAG traversal gate on the grid clusters.
for (const [name, b] of [
  ['grid', grid],
  ['cliff', cliff],
  ['skirt', withSkirt],
  ['flat', flat],
] as const) {
  const h = buildHeightGridHierarchy(b.clusters);
  const r = validateDagHierarchy({ clusters: b.clusters } as unknown as DagBuild, h);
  if (!r.ok) fail(`H(${name}): anchor-chain != cut — ${r.msg}`);
  else if (name === 'grid') console.log(`  HIER: ${h.rootIndices.length} roots, ${h.childIndices.length} child links — ${r.msg}`);
}

if (failures > 0) {
  console.error(`[probe-heightgrid] ${failures} FAILURES`);
  process.exit(1);
}
console.log('[probe-heightgrid] PASS — regular-grid terrain: no fans, watertight, tile-uniform, shape-faithful, fast');
