/**
 * Terrain heightfield DAG validation probe (node-only, no browser; NANITE-SPEC.md
 * N8-D2, D-N32). Builds buildHeightDag on a deterministic BIMODAL field (flat
 * plain + tilted ramp + ridged cliffs) and asserts the crack-free cut invariants
 * PLUS the terrain-specific ones, then prints adaptivity + build throughput.
 *
 *   npx tsx tools/probe-heightdag.ts
 *
 * Invariants (shared with probe-dag — the SAME flat kClusterCull cut):
 *   M  error monotone     — non-root cluster: parentError ≥ ownError
 *   C  containment        — non-root cluster: parentSphere ⊇ ownSphere
 *   E  sibling equality   — per reduced group: inputs' parent pair AND parents'
 *                           own pair === the group (err,sphere), bit-exact
 *   O  no orphans         — LOD0 ownError=0; structural group links sound
 *   A  cut antichain      — τ sweep: no group has BOTH an input and a parent cut
 * Terrain-specific:
 *   G  on-grid / F4       — every survivor round-trips to an exact grid vertex
 *                           (offGridVerts==0): positions reconstruct from heights
 *   W  watertight cut     — at several τ, every INTERIOR edge of the selected
 *                           triangle set (keyed by CANONICAL grid-id) is shared
 *                           by exactly 2 tris → no cracks / no T-junctions
 *   D  deterministic      — a second build is bit-identical
 *   adaptivity (reported + asserted): flat/ramp regions decimate hard, ridged
 *                           cliffs stay dense (the "plains SIGNIFICANTLY fewer
 *                           tris" mandate) — measured as a per-region tri ratio.
 */

import { buildHeightDag, type HeightDagBuild, type HeightField } from '../src/nanite/BuildHeightDag';
import type { DagCluster, DagGroup } from '../src/nanite/BuildDag';

let failures = 0;
const fail = (msg: string): void => {
  failures++;
  console.error(`  FAIL ${msg}`);
};

const SCREEN_H = 1080;
const FOV_Y = 1.0;
const PROJ_K = SCREEN_H / 2 / Math.tan(FOV_Y / 2);

function project(e: number, cx: number, cy: number, cz: number, r: number, camX: number, camZ: number): number {
  if (!Number.isFinite(e)) return Infinity;
  const dx = cx - camX;
  const dy = cy;
  const dz = cz - camZ;
  const d2 = dx * dx + dy * dy + dz * dz;
  const denom = Math.sqrt(Math.max(1e-6, d2 - r * r));
  return (PROJ_K * e) / denom;
}

function dist(ax: number, ay: number, az: number, bx: number, by: number, bz: number): number {
  return Math.hypot(ax - bx, ay - by, az - bz);
}

/** deterministic bimodal heightfield: flat plain | tilted ramp | ridged cliffs. */
function synthField(gridN: number, cellSize: number): HeightField {
  const vpa = gridN + 1;
  const heights = new Float32Array(vpa * vpa);
  for (let gz = 0; gz <= gridN; gz++) {
    for (let gx = 0; gx <= gridN; gx++) {
      const fx = gx / gridN; // 0..1 across X
      const fz = gz / gridN;
      let h: number;
      if (fx < 0.4) {
        h = 0; // FLAT plain — QEM cost 0 → collapses to a few big triangles
      } else if (fx < 0.5) {
        // TILTED ramp — still planar → must ALSO decimate (QEM cost 0 on a plane)
        h = (fx - 0.4) * 40 + fz * 6;
      } else {
        // RIDGED cliffs — high-frequency vertical detail → stuck → stays dense
        const u = (fx - 0.5) * gridN * cellSize;
        const w = fz * gridN * cellSize;
        h = 6 + 5 * Math.sin(u * 0.7) * Math.cos(w * 0.55) + 2.5 * Math.sin(u * 1.9 + w * 0.3);
      }
      heights[gz * vpa + gx] = h;
    }
  }
  return { heights, gridN, cellSize, originX: 0, originZ: 0 };
}

interface Built {
  clusters: DagCluster[];
  groups: DagGroup[];
  lod0Tris: number;
}

function checkInvariants(name: string, b: Built): void {
  const cl = b.clusters;
  const tol = 1e-4;

  // M + C
  let worstMono = 0;
  let worstContain = 0;
  for (const c of cl) {
    if (!Number.isFinite(c.parentError)) continue;
    if (c.parentError < c.ownError - 1e-9) worstMono = Math.max(worstMono, c.ownError - c.parentError);
    const gap = dist(c.oex, c.oey, c.oez, c.pex, c.pey, c.pez) + c.oer - c.per;
    if (gap > tol * Math.max(1, c.per)) worstContain = Math.max(worstContain, gap);
  }
  if (worstMono > 0) fail(`${name} M: parentError < ownError by ${worstMono.toExponential(2)}`);
  if (worstContain > 0) fail(`${name} C: ownSphere escapes parentSphere by ${worstContain.toExponential(2)}`);

  // E: exact sibling-pair equality per reduced group
  let eqViolations = 0;
  for (const g of b.groups) {
    if (!g.reduced) continue;
    for (const id of g.inputs) {
      const c = cl[id];
      if (!c) continue;
      if (c.parentError !== g.error || c.pex !== g.sx || c.pey !== g.sy || c.pez !== g.sz || c.per !== g.sr) eqViolations++;
    }
    for (const id of g.parents) {
      const c = cl[id];
      if (!c) continue;
      if (c.ownError !== g.error || c.oex !== g.sx || c.oey !== g.sy || c.oez !== g.sz || c.oer !== g.sr) eqViolations++;
    }
  }
  if (eqViolations > 0) fail(`${name} E: ${eqViolations} sibling pairs not bit-exact with their group`);

  // O: structural / no-orphan
  let orphan = 0;
  for (let i = 0; i < cl.length; i++) {
    const c = cl[i] as DagCluster;
    if (c.level === 0) {
      if (c.ownError !== 0) orphan++;
      if (c.groupAsParent !== -1) orphan++;
    } else {
      const gp = b.groups[c.groupAsParent];
      if (!gp || !gp.reduced || !gp.parents.includes(i)) orphan++;
    }
    if (Number.isFinite(c.parentError)) {
      const gi = b.groups[c.groupAsInput];
      if (!gi || !gi.reduced || !gi.inputs.includes(i)) orphan++;
      else if (c.parentError !== gi.error) orphan++;
    } else if (c.groupAsInput !== -1) {
      orphan++;
    }
  }
  for (const g of b.groups) if (g.reduced && (g.parents.length === 0 || g.inputs.length === 0)) orphan++;
  if (orphan > 0) fail(`${name} O: ${orphan} orphan/structural violations`);

  // A: cut-antichain over a τ sweep (camera off to one side, in the XZ plane)
  const camX = 64;
  const camZ = -40;
  const projOwn = cl.map((c) => project(c.ownError, c.oex, c.oey, c.oez, c.oer, camX, camZ));
  const projPar = cl.map((c) => project(c.parentError, c.pex, c.pey, c.pez, c.per, camX, camZ));
  const taus = [0.0, 0.25, 0.5, 1, 2, 4, 8, 16, 64, 1e9];
  let handoff = 0;
  const triAt: { tau: number; tris: number }[] = [];
  for (const tau of taus) {
    const selected = new Uint8Array(cl.length);
    let selTris = 0;
    for (let i = 0; i < cl.length; i++) {
      if ((projOwn[i] as number) <= tau && (projPar[i] as number) > tau) {
        selected[i] = 1;
        selTris += (cl[i] as DagCluster).triCount;
      }
    }
    triAt.push({ tau, tris: selTris });
    for (const g of b.groups) {
      if (!g.reduced) continue;
      if (g.inputs.some((id) => selected[id] === 1) && g.parents.some((id) => selected[id] === 1)) handoff++;
    }
  }
  if (handoff > 0) fail(`${name} A: ${handoff} group handoff overlaps (input+parent both cut)`);
  for (let i = 1; i < triAt.length; i++) {
    if ((triAt[i] as { tris: number }).tris > (triAt[i - 1] as { tris: number }).tris + 1) {
      fail(`${name} A: cut tris rose with τ (${(triAt[i - 1] as { tris: number }).tris} → ${(triAt[i] as { tris: number }).tris})`);
      break;
    }
  }
  if ((triAt[0] as { tris: number }).tris !== b.lod0Tris) fail(`${name} A: τ=0 cut ${(triAt[0] as { tris: number }).tris} != LOD0 ${b.lod0Tris}`);
}

/** W: geometric watertightness of the selected cut, keyed by CANONICAL grid-id.
 *  Two triangles from different clusters that meet on a grid vertex reference the
 *  SAME packed id, so a shared edge counts twice; an interior use≠2 = crack/T-junction.
 *  Border edges (a grid-perimeter vertex pair) are legal use==1. */
function checkWatertight(name: string, dag: HeightDagBuild, taus: number[]): void {
  const cl = dag.clusters;
  const gv = dag.gridVerts;
  const idx = dag.indices;
  const N = dag.gridN;
  const camX = 64;
  const camZ = -40;
  const onBorder = (packed: number): boolean => {
    const gx = packed & 0xffff;
    const gz = (packed >> 16) & 0xffff;
    return gx === 0 || gx === N || gz === 0 || gz === N;
  };
  for (const tau of taus) {
    // key = "lo_hi" of the two CANONICAL packed grid-ids (order-independent);
    // string key sidesteps 2^53 overflow when combining two ≤2^32 ids.
    const edgeUse = new Map<string, { use: number; u: number; v: number }>();
    const bump = (u: number, v: number): void => {
      const lo = u < v ? u : v;
      const hi = u < v ? v : u;
      const k = `${lo}_${hi}`;
      const e = edgeUse.get(k);
      if (e) e.use++;
      else edgeUse.set(k, { use: 1, u: lo, v: hi });
    };
    for (let ci = 0; ci < cl.length; ci++) {
      const c = cl[ci] as DagCluster;
      const po = project(c.ownError, c.oex, c.oey, c.oez, c.oer, camX, camZ);
      const pp = project(c.parentError, c.pex, c.pey, c.pez, c.per, camX, camZ);
      if (!(po <= tau && pp > tau)) continue;
      for (let t = c.triStart; t < c.triStart + c.triCount; t++) {
        const a = gv[idx[t * 3] as number] as number;
        const bb = gv[idx[t * 3 + 1] as number] as number;
        const cc = gv[idx[t * 3 + 2] as number] as number;
        bump(a, bb);
        bump(bb, cc);
        bump(cc, a);
      }
    }
    let cracks = 0;
    const locs: string[] = [];
    for (const e of edgeUse.values()) {
      if (e.use === 2) continue;
      // a use==1 edge is legal ONLY on the terrain perimeter (both ends on border)
      if (e.use === 1 && onBorder(e.u) && onBorder(e.v)) continue;
      cracks++;
      if (locs.length < 10) {
        const ux = e.u & 0xffff;
        const uz = (e.u >> 16) & 0xffff;
        const vx = e.v & 0xffff;
        const vz = (e.v >> 16) & 0xffff;
        locs.push(`(${ux},${uz})-(${vx},${vz})×${e.use}`);
      }
    }
    if (cracks > 0) {
      fail(`${name} W(τ=${tau === 1e9 ? '∞' : tau}): ${cracks} non-manifold interior edges (cracks/T-junctions)`);
      if (process.env.WDEBUG) console.error(`       edges: ${locs.join('  ')}`);
    }
  }
}

console.log('[probe-heightdag]');

const gridN = 128;
const cellSize = 1.0;
const hf = synthField(gridN, cellSize);

const dag = buildHeightDag(hf);
const lod0Tris = dag.stats.lod0Tris;
checkInvariants('terrain', { clusters: dag.clusters, groups: dag.groups, lod0Tris });
checkWatertight('terrain', dag, [0.0, 0.5, 2, 8, 1e9]);

// G: on-grid / F4
if (dag.stats.offGridVerts > 0) fail(`G: ${dag.stats.offGridVerts} survivors round-tripped OFF the grid (F4 violated)`);
if (dag.stats.maxGridResidual > 1e-3) fail(`G: worst grid residual ${dag.stats.maxGridResidual.toExponential(2)} cells (> 1e-3)`);
// index sanity
{
  let bad = 0;
  for (let t = 0; t < dag.indices.length; t++) if ((dag.indices[t] as number) >= dag.gridVerts.length) bad++;
  if (bad > 0) fail(`G: ${bad} indices out of range`);
}

// D: determinism (bit-identical second build)
{
  const d2 = buildHeightDag(hf);
  if (
    d2.stats.totalClusters !== dag.stats.totalClusters ||
    d2.stats.totalTris !== dag.stats.totalTris ||
    d2.stats.roots !== dag.stats.roots ||
    Math.abs(d2.stats.maxError - dag.stats.maxError) > 1e-9 ||
    d2.gridVerts.length !== dag.gridVerts.length
  ) {
    fail(`D: build is NON-deterministic (cl ${dag.stats.totalClusters}/${d2.stats.totalClusters})`);
  } else {
    let diff = 0;
    for (let i = 0; i < dag.gridVerts.length; i++) if (dag.gridVerts[i] !== d2.gridVerts[i]) diff++;
    if (diff > 0) fail(`D: ${diff} gridVerts differ between builds`);
  }
}

// adaptivity: per-region tris under a SCREEN-ERROR cut from a camera HIGH above
// the centre (flat & cliffs at ~equal distance, so the cut differs only by their
// intrinsic error). The flat/ramp half must select FAR fewer tris than the ridged
// cliffs — the user's "plains SIGNIFICANTLY fewer tris" mandate. (Measured at a
// real cut, NOT at the roots where everything is uniformly coarse.)
{
  const extent = gridN * cellSize;
  const camX = extent / 2;
  const camY = 100;
  const camZ = extent / 2;
  const tau = 1;
  const proj = (e: number, cx: number, cy: number, cz: number, r: number): number => {
    if (!Number.isFinite(e)) return Infinity;
    const dx = cx - camX;
    const dy = cy - camY;
    const dz = cz - camZ;
    const denom = Math.sqrt(Math.max(1e-6, dx * dx + dy * dy + dz * dz - r * r));
    return (PROJ_K * e) / denom;
  };
  let flatTris = 0;
  let cliffTris = 0;
  let selTris = 0;
  for (const c of dag.clusters) {
    const po = proj(c.ownError, c.oex, c.oey, c.oez, c.oer);
    const pp = proj(c.parentError, c.pex, c.pey, c.pez, c.per);
    if (!(po <= tau && pp > tau)) continue;
    selTris += c.triCount;
    if (c.sx < extent * 0.45) flatTris += c.triCount;
    else if (c.sx > extent * 0.55) cliffTris += c.triCount;
  }
  const reduction = 1 - selTris / lod0Tris;
  const ratio = cliffTris / Math.max(1, flatTris);
  console.log(
    `  field ${gridN}×${gridN} (${lod0Tris} LOD0 tris) → ${dag.stats.levels} levels, ${dag.stats.totalClusters} cl, ` +
      `${dag.stats.roots} roots | maxErr ${dag.stats.maxError.toExponential(2)} m | ${dag.stats.buildMs.toFixed(1)} ms`,
  );
  for (const ls of dag.levelStats) {
    console.log(
      `    L${ls.level}→${ls.level + 1}: ${ls.inTris} → ${ls.outTris} tris | ${ls.groups} groups (${ls.stuckGroups} stuck) | reduce ${(ls.triReduction * 100).toFixed(0)}%`,
    );
  }
  console.log(
    `  ADAPTIVITY (τ=1 cut, cam 100 m above centre): flat/ramp ${flatTris} tris vs cliffs ${cliffTris} tris ` +
      `(${ratio.toFixed(1)}× denser cliffs) | ${selTris}/${lod0Tris} tris drawn (${(reduction * 100).toFixed(1)}% culled)`,
  );
  console.log(`  ON-GRID: offGrid ${dag.stats.offGridVerts}, worst residual ${dag.stats.maxGridResidual.toExponential(2)} cells`);
  console.log(`  throughput ${((lod0Tris / 1e6 / dag.stats.buildMs) * 1000).toFixed(2)} Mtri/s`);

  if (reduction < 0.4) fail(`adaptivity: screen cut only reduced ${(reduction * 100).toFixed(0)}% (expected ≥40% on a half-flat field)`);
  if (ratio < 3) fail(`adaptivity: cliffs only ${ratio.toFixed(1)}× denser than flats (expected ≥3× — decimation not flatness-adaptive)`);
}

if (failures > 0) {
  console.error(`[probe-heightdag] ${failures} FAILURES`);
  process.exit(1);
}
console.log('[probe-heightdag] all terrain-DAG invariants hold (crack-free cut, on-grid F4, adaptive)');
