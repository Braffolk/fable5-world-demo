/**
 * LOD DAG validation probe (node-only, no browser; NANITE.md N8-D0).
 * Builds the cluster DAG on real generated rocks at several detail levels and
 * asserts the crack-free runtime-cut invariants, then prints per-level stats +
 * an extrapolated all-pools build time (F15 boot budget).
 *
 *   npx tsx tools/probe-dag.ts
 *
 * Invariants checked per mesh:
 *   M  error monotone   — every non-root cluster: parentError ≥ ownError
 *   C  containment      — every non-root cluster: parentSphere ⊇ ownSphere
 *   E  sibling equality — per reduced group: ALL inputs' parent pair AND ALL
 *                         parents' own pair === the group's (err,sphere), bit-exact
 *   O  no orphans       — LOD0 ownError=0; every level>0 cluster has a producing
 *                         group; every non-root cluster has a (reduced) input group
 *                         with matching error; roots have parentError=∞
 *   A  cut antichain    — across a τ sweep, no reduced group ever has BOTH an input
 *                         and a parent selected (the exact-handoff ⇒ crack-free cut)
 * Plus a projected-error monotonicity STATISTIC (reported, not failed — containment
 * + error-monotone makes it ~0 for sane cameras but is not a pointwise guarantee).
 */

import { buildDag, type DagBuild } from '../src/nanite/BuildDag';
import { Rng } from '../src/core/Seed';
import { buildRock } from '../src/vegetation/RockBuilder';
import { buildTree } from '../src/vegetation/TreeBuilder';
import { BEECH, SNAG } from '../src/vegetation/Species';
import type { BufferGeometry } from 'three';
import type { SpeciesParams } from '../src/vegetation/VegTypes';

let failures = 0;
const fail = (msg: string): void => {
  failures++;
  console.error(`  FAIL ${msg}`);
};

const SCREEN_H = 1080;
const FOV_Y = 1.0;
const PROJ_K = SCREEN_H / 2 / Math.tan(FOV_Y / 2);

/** screen-space projected error for a sphere viewed from (0,0,camD) down -Z */
function project(e: number, cx: number, cy: number, cz: number, r: number, camD: number): number {
  if (!Number.isFinite(e)) return Infinity;
  const dx = cx;
  const dy = cy;
  const dz = cz - camD;
  const d2 = dx * dx + dy * dy + dz * dz;
  const denom = Math.sqrt(Math.max(1e-6, d2 - r * r));
  return (PROJ_K * e) / denom;
}

function dist(ax: number, ay: number, az: number, bx: number, by: number, bz: number): number {
  return Math.hypot(ax - bx, ay - by, az - bz);
}

function checkMesh(name: string, dag: DagBuild): void {
  const cl = dag.clusters;
  const tol = 1e-4;

  // M + C: per non-root cluster
  let worstMono = 0;
  let worstContain = 0;
  for (const c of cl) {
    if (!Number.isFinite(c.parentError)) continue; // root
    if (c.parentError < c.ownError - 1e-9) worstMono = Math.max(worstMono, c.ownError - c.parentError);
    const gap = dist(c.oex, c.oey, c.oez, c.pex, c.pey, c.pez) + c.oer - c.per;
    if (gap > tol * Math.max(1, c.per)) worstContain = Math.max(worstContain, gap);
  }
  if (worstMono > 0) fail(`${name} M: parentError < ownError by ${worstMono.toExponential(2)}`);
  if (worstContain > 0) fail(`${name} C: ownSphere escapes parentSphere by ${worstContain.toExponential(2)}`);

  // E: exact sibling-pair equality per reduced group
  let eqViolations = 0;
  for (const g of dag.groups) {
    if (!g.reduced) continue;
    for (const id of g.inputs) {
      const c = cl[id];
      if (!c) continue;
      if (c.parentError !== g.error || c.pex !== g.sx || c.pey !== g.sy || c.pez !== g.sz || c.per !== g.sr) {
        eqViolations++;
      }
    }
    for (const id of g.parents) {
      const c = cl[id];
      if (!c) continue;
      if (c.ownError !== g.error || c.oex !== g.sx || c.oey !== g.sy || c.oez !== g.sz || c.oer !== g.sr) {
        eqViolations++;
      }
    }
  }
  if (eqViolations > 0) fail(`${name} E: ${eqViolations} sibling pairs not bit-exact with their group`);

  // O: structural / no-orphan
  let orphan = 0;
  for (let i = 0; i < cl.length; i++) {
    const c = cl[i] as DagBuild['clusters'][number];
    if (c.level === 0) {
      if (c.ownError !== 0) orphan++; // LOD0 must have zero own error
      if (c.groupAsParent !== -1) orphan++;
    } else {
      const gp = dag.groups[c.groupAsParent];
      if (!gp || !gp.reduced || !gp.parents.includes(i)) orphan++;
    }
    if (Number.isFinite(c.parentError)) {
      const gi = dag.groups[c.groupAsInput];
      if (!gi || !gi.reduced || !gi.inputs.includes(i)) orphan++;
      else if (c.parentError !== gi.error) orphan++;
    } else {
      if (c.groupAsInput !== -1) orphan++; // root must not also be an input
    }
  }
  for (const g of dag.groups) {
    if (g.reduced && (g.parents.length === 0 || g.inputs.length === 0)) orphan++;
  }
  if (orphan > 0) fail(`${name} O: ${orphan} orphan/structural violations`);

  // A: cut-antichain sweep — for each reduced group, never both an input and a
  // parent selected at the same τ (the exact handoff guarantees a clean frontier)
  const camD = 30;
  const projOwn = cl.map((c) => project(c.ownError, c.oex, c.oey, c.oez, c.oer, camD));
  const projPar = cl.map((c) => project(c.parentError, c.pex, c.pey, c.pez, c.per, camD));
  const taus = [0.0, 0.25, 0.5, 1, 2, 4, 8, 16, 64, 1e9];
  let handoffViolations = 0;
  let projMonoViolations = 0;
  for (let i = 0; i < cl.length; i++) {
    if (Number.isFinite(cl[i]?.parentError ?? Infinity) && (projPar[i] as number) < (projOwn[i] as number) - 1e-6) {
      projMonoViolations++;
    }
  }
  const triAt: { tau: number; clusters: number; tris: number }[] = [];
  for (const tau of taus) {
    const selected = new Uint8Array(cl.length);
    let selClusters = 0;
    let selTris = 0;
    for (let i = 0; i < cl.length; i++) {
      if ((projOwn[i] as number) <= tau && (projPar[i] as number) > tau) {
        selected[i] = 1;
        selClusters++;
        selTris += (cl[i] as DagBuild['clusters'][number]).triCount;
      }
    }
    triAt.push({ tau, clusters: selClusters, tris: selTris });
    for (const g of dag.groups) {
      if (!g.reduced) continue;
      const anyIn = g.inputs.some((id) => selected[id] === 1);
      const anyPar = g.parents.some((id) => selected[id] === 1);
      if (anyIn && anyPar) handoffViolations++;
    }
  }
  if (handoffViolations > 0) fail(`${name} A: ${handoffViolations} group handoff overlaps (input+parent both cut)`);

  // cut sweep should be monotone: more detail (small τ) ⇒ ≥ tris than coarse
  for (let i = 1; i < triAt.length; i++) {
    const prev = triAt[i - 1] as { tris: number };
    const cur = triAt[i] as { tris: number };
    if (cur.tris > prev.tris + 1) {
      fail(`${name} A: cut tris rose with τ (${prev.tris} → ${cur.tris})`);
      break;
    }
  }
  const lod0Tris = dag.stats.lod0Tris;
  const fine = triAt[0] as { tris: number };
  const coarse = triAt[triAt.length - 1] as { tris: number };
  if (fine.tris !== lod0Tris) fail(`${name} A: τ=0 cut ${fine.tris} tris != LOD0 ${lod0Tris}`);
  let rootTris = 0;
  for (const c of cl) if (!Number.isFinite(c.parentError)) rootTris += c.triCount;
  if (coarse.tris !== rootTris) fail(`${name} A: τ=∞ cut ${coarse.tris} tris != roots ${rootTris}`);

  // report
  const s = dag.stats;
  console.log(
    `  ${name}: ${s.lod0Tris} tris / ${s.lod0Clusters} cl (L0) -> ${s.levels} levels, ` +
      `${s.totalClusters} cl total, ${s.roots} roots | maxErr ${s.maxError.toExponential(2)} | ${s.buildMs.toFixed(1)} ms`,
  );
  for (const ls of dag.levelStats) {
    console.log(
      `    L${ls.level}→${ls.level + 1}: ${ls.inClusters} cl / ${ls.inTris} tris -> ` +
        `${ls.outClusters} cl / ${ls.outTris} tris | ${ls.groups} groups (${ls.stuckGroups} stuck) | ` +
        `tri-reduce ${(ls.triReduction * 100).toFixed(0)}%`,
    );
  }
  console.log(
    `    cut sweep (τ px → tris): ` +
      triAt.map((t) => `${t.tau === 1e9 ? '∞' : t.tau}:${t.tris}`).join('  '),
  );
  if (projMonoViolations > 0) {
    console.log(`    note: ${projMonoViolations} clusters with projected parent<own at camD=${camD} (expected small; not a defect)`);
  }
}

/** build the DAG for any BufferGeometry (pos + normal interleaved, stride 6),
 *  run the determinism + invariant checks, return it */
function dagFromGeometry(label: string, geo: BufferGeometry): DagBuild {
  const pos = geo.attributes.position;
  const nrm = geo.attributes.normal;
  if (!pos) throw new Error(`${label}: no position attribute`);
  const vCount = pos.count;
  const stride = 6; // pos(3) + normal(3) — exercises attribute carry + renorm
  const verts = new Float32Array(vCount * stride);
  const pa = pos.array as Float32Array;
  const na = nrm ? (nrm.array as Float32Array) : null;
  for (let v = 0; v < vCount; v++) {
    verts[v * stride] = pa[v * 3] as number;
    verts[v * stride + 1] = pa[v * 3 + 1] as number;
    verts[v * stride + 2] = pa[v * 3 + 2] as number;
    verts[v * stride + 3] = na ? (na[v * 3] as number) : 0;
    verts[v * stride + 4] = na ? (na[v * 3 + 1] as number) : 1;
    verts[v * stride + 5] = na ? (na[v * 3 + 2] as number) : 0;
  }
  // non-indexed geometry → trivial sequential index (the position-weld recovers
  // shared edges at level 0, exactly the seam-merge the build relies on)
  const idx = geo.index;
  const indices = idx
    ? new Uint32Array(idx.array as ArrayLike<number>)
    : Uint32Array.from({ length: vCount }, (_, i) => i);
  const dag = buildDag(verts, stride, indices, { normalOffset: 3 });

  // determinism: a second build must be bit-identical in structure
  const dag2 = buildDag(verts, stride, indices, { normalOffset: 3 });
  if (
    dag2.stats.totalClusters !== dag.stats.totalClusters ||
    dag2.stats.totalTris !== dag.stats.totalTris ||
    dag2.stats.roots !== dag.stats.roots ||
    Math.abs(dag2.stats.maxError - dag.stats.maxError) > 1e-9
  ) {
    fail(`${label}: build is NON-deterministic (cl ${dag.stats.totalClusters}/${dag2.stats.totalClusters})`);
  }
  checkMesh(label, dag);
  return dag;
}

function rockOf(detail: number, label: string): DagBuild {
  return dagFromGeometry(label, buildRock('boulder', new Rng(1234 + detail), detail).geometry);
}

function barkOf(sp: SpeciesParams, lod: 0 | 1 | 2, label: string): DagBuild {
  return dagFromGeometry(label, buildTree(sp, new Rng(77), { lod }).bark);
}

console.log('[probe-dag]');
rockOf(3, 'rock-small');
rockOf(5, 'rock-mid');
const hero = rockOf(7, 'rock-hero');
// open-tube topology (bark trunk/branches) — exercises open-boundary locking;
// SNAG stands in for the DEADWOOD class. Both are the same ExplicitSource path.
barkOf(BEECH, 1, 'bark-beech');
barkOf(SNAG, 0, 'deadwood-snag');

// boot-budget extrapolation (F15): DAG build ms per source Mtri, scaled to the
// 3–4M all-pools source-tri budget (per N1: 1.52M explicit tris today).
{
  const ms = hero.stats.buildMs;
  const mtri = hero.stats.lod0Tris / 1e6;
  const perMTri = ms / Math.max(1e-6, mtri); // ms per Mtri
  console.log(
    `  build throughput ${(1000 / perMTri).toFixed(2)} Mtri/s (DAG) -> ` +
      `1.52M explicit ≈ ${((perMTri * 1.52) / 1000).toFixed(1)} s, 4M ≈ ${((perMTri * 4) / 1000).toFixed(1)} s`,
  );
}

if (failures > 0) {
  console.error(`[probe-dag] ${failures} FAILURES`);
  process.exit(1);
}
console.log('[probe-dag] all DAG invariants hold');
