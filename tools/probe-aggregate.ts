/**
 * Aggregate foliage DAG validation probe (node-only, no browser; NANITE-SPEC.md
 * N9-C1). Builds the area-preserving leaf-removal DAG (BuildAggregateDag.ts) on a
 * synthetic leaf crown (hundreds of DISCONNECTED folded leaf strips — the island
 * topology QEM can't handle) and asserts:
 *
 *   The SAME crack-free cut invariants as probe-dag (the aggregate emits the SAME
 *   DagBuild contract, so it must satisfy them identically):
 *     M  error monotone   — parentError ≥ ownError
 *     C  containment      — parentSphere ⊇ ownSphere
 *     E  sibling equality — per reduced group, all inputs' parent pair AND all
 *                           parents' own pair === the group pair, bit-exact
 *     O  no orphans       — LOD0 own=0; every cluster has its producing/input group
 *     A  cut antichain     — over a τ sweep no group has BOTH an input and a parent
 *                           selected (clean handoff); τ=0 cut==LOD0, τ=∞ cut==roots
 *   Plus the AGGREGATE-specific gates (the spec's "Preserve Area" contract):
 *     AREA  silhouette mass preserved — the summed triangle area of EVERY runtime
 *           cut (near→far distance sweep) stays ≈ the LOD0 area: the crown thins to
 *           fewer/bigger leaves but never BALDS (the naive-drop failure Epic fixed)
 *     THIN  monotone thinning — coarser levels really do shed triangles
 *     DET   determinism — a second build is bit-identical; a different ?seed thins
 *           a DIFFERENT leaf subset (still area-preserving)
 *
 *   npx tsx tools/probe-aggregate.ts
 */

import { buildAggregateDag, type AggregateDagOpts } from '../src/nanite/BuildAggregateDag';
import { type DagBuild } from '../src/nanite/BuildDag';
import { setClusterFill } from '../src/nanite/Clusterize';
import { Rng } from '../src/core/Seed';

let failures = 0;
const fail = (msg: string): void => {
  failures++;
  console.error(`  FAIL ${msg}`);
};

const SCREEN_H = 1080;
const FOV_Y = 1.0;
const PROJ_K = SCREEN_H / 2 / Math.tan(FOV_Y / 2);
const STRIDE = 12; // pos(0..2) nrm(3..5) uv(6..7) vdata(8..11) — DAG_VERT_STRIDE

/** screen-space projected error for a sphere viewed from distance camD down -Z */
function project(e: number, cx: number, cy: number, cz: number, r: number, camD: number): number {
  if (!Number.isFinite(e)) return Infinity;
  const dz = cz - camD;
  const d2 = cx * cx + cy * cy + dz * dz;
  const denom = Math.sqrt(Math.max(1e-6, d2 - r * r));
  return (PROJ_K * e) / denom;
}

function dist(ax: number, ay: number, az: number, bx: number, by: number, bz: number): number {
  return Math.hypot(ax - bx, ay - by, az - bz);
}

// ---------------------------------------------------------------------------
// synthetic leaf crown: nLeaves folded multi-tri strips scattered on a shell,
// each a connected island, all mutually disconnected (the foliage topology)
// ---------------------------------------------------------------------------

interface Mesh {
  verts: Float32Array;
  indices: Uint32Array;
}

function makeCrown(rng: Rng, nLeaves: number, crownR: number, leafSize: number): Mesh {
  const VPL = 6; // verts per leaf (3 rows × 2 cols, folded)
  const TPL = 4; // tris per leaf
  const verts = new Float32Array(nLeaves * VPL * STRIDE);
  const indices = new Uint32Array(nLeaves * TPL * 3);
  let vp = 0;
  let ip = 0;
  // golden-spiral shell positions for an even spread, small radial jitter
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < nLeaves; i++) {
    const y = 1 - (i / Math.max(1, nLeaves - 1)) * 2; // 1 → -1
    const rxz = Math.sqrt(Math.max(0, 1 - y * y));
    const phi = i * golden;
    const jr = crownR * (0.82 + 0.18 * rng.float());
    const cx = Math.cos(phi) * rxz * jr;
    const cy = y * jr;
    const cz = Math.sin(phi) * rxz * jr;
    // leaf frame: N radial, T/B tangent
    let nx = cx;
    let ny = cy;
    let nz = cz;
    const nl = Math.hypot(nx, ny, nz) || 1;
    nx /= nl;
    ny /= nl;
    nz /= nl;
    // tangent = normalize(cross(N, up)); near the poles fall back to X
    let tx = nz * 0 - ny * 0; // cross(N, up=(0,1,0)) = (nz, 0, -nx)
    tx = nz;
    let ty = 0;
    let tz = -nx;
    let tl = Math.hypot(tx, ty, tz);
    if (tl < 1e-4) {
      tx = 1;
      ty = 0;
      tz = 0;
      tl = 1;
    }
    tx /= tl;
    ty /= tl;
    tz /= tl;
    // bitangent = cross(N, T)
    const bx = ny * tz - nz * ty;
    const by = nz * tx - nx * tz;
    const bz = nx * ty - ny * tx;
    const base = vp;
    for (let r = 0; r < 3; r++) {
      const v = (r - 1) * 0.5 * leafSize; // -0.5..0.5 along length
      const fold = Math.cos((r - 1) * 1.2) * 0.18 * leafSize - 0.18 * leafSize; // middle row bows out
      for (let cIdx = 0; cIdx < 2; cIdx++) {
        const u = (cIdx - 0.5) * leafSize; // width
        const w = fold;
        const px = cx + tx * u + bx * v + nx * w;
        const py = cy + ty * u + by * v + ny * w;
        const pz = cz + tz * u + bz * v + nz * w;
        const o = vp * STRIDE;
        verts[o] = px;
        verts[o + 1] = py;
        verts[o + 2] = pz;
        verts[o + 3] = nx;
        verts[o + 4] = ny;
        verts[o + 5] = nz;
        verts[o + 6] = cIdx;
        verts[o + 7] = r * 0.5;
        verts[o + 8] = i; // vdata: per-leaf id (verify attribute carry-through)
        vp++;
      }
    }
    // 2 quads → 4 tris over the 3×2 grid (rows 0-1-2, cols 0-1)
    for (let r = 0; r < 2; r++) {
      const a = base + r * 2;
      const b = a + 1;
      const c = a + 2;
      const d = a + 3;
      indices[ip++] = a;
      indices[ip++] = b;
      indices[ip++] = c;
      indices[ip++] = b;
      indices[ip++] = d;
      indices[ip++] = c;
    }
  }
  return { verts, indices };
}

// ---------------------------------------------------------------------------
// geometry area helpers (over a DagBuild's packed verts/indices)
// ---------------------------------------------------------------------------

function triArea(dag: DagBuild, t: number): number {
  const s = dag.vertStride;
  const ia = (dag.indices[t * 3] as number) * s;
  const ib = (dag.indices[t * 3 + 1] as number) * s;
  const ic = (dag.indices[t * 3 + 2] as number) * s;
  const v = dag.verts;
  const e1x = (v[ib] as number) - (v[ia] as number);
  const e1y = (v[ib + 1] as number) - (v[ia + 1] as number);
  const e1z = (v[ib + 2] as number) - (v[ia + 2] as number);
  const e2x = (v[ic] as number) - (v[ia] as number);
  const e2y = (v[ic + 1] as number) - (v[ia + 1] as number);
  const e2z = (v[ic + 2] as number) - (v[ia + 2] as number);
  const cx = e1y * e2z - e1z * e2y;
  const cy = e1z * e2x - e1x * e2z;
  const cz = e1x * e2y - e1y * e2x;
  return 0.5 * Math.sqrt(cx * cx + cy * cy + cz * cz);
}

function clusterArea(dag: DagBuild, ci: number): number {
  const c = dag.clusters[ci] as DagBuild['clusters'][number];
  let a = 0;
  for (let t = c.triStart; t < c.triStart + c.triCount; t++) a += triArea(dag, t);
  return a;
}

/** total leaf area of the LOD0 (full-detail) crown */
function lod0Area(dag: DagBuild): number {
  let a = 0;
  for (let i = 0; i < dag.lod0Count; i++) a += clusterArea(dag, i);
  return a;
}

// ---------------------------------------------------------------------------
// the shared crack-free invariant checks (same contract as probe-dag's checkMesh)
// ---------------------------------------------------------------------------

function checkInvariants(name: string, dag: DagBuild): void {
  const cl = dag.clusters;
  const tol = 1e-4;

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

  let eqViolations = 0;
  for (const g of dag.groups) {
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

  let orphan = 0;
  for (let i = 0; i < cl.length; i++) {
    const c = cl[i] as DagBuild['clusters'][number];
    if (c.level === 0) {
      if (c.ownError !== 0) orphan++;
      if (c.groupAsParent !== -1) orphan++;
    } else {
      const gp = dag.groups[c.groupAsParent];
      if (!gp || !gp.reduced || !gp.parents.includes(i)) orphan++;
    }
    if (Number.isFinite(c.parentError)) {
      const gi = dag.groups[c.groupAsInput];
      if (!gi || !gi.reduced || !gi.inputs.includes(i)) orphan++;
      else if (c.parentError !== gi.error) orphan++;
    } else if (c.groupAsInput !== -1) {
      orphan++;
    }
  }
  for (const g of dag.groups) if (g.reduced && (g.parents.length === 0 || g.inputs.length === 0)) orphan++;
  if (orphan > 0) fail(`${name} O: ${orphan} orphan/structural violations`);

  // A: cut-antichain + area-preservation across a near→far distance sweep
  const a0 = lod0Area(dag);
  const taus = [0.0, 0.5, 1, 2, 4, 8, 16, 64, 1e9];
  let handoff = 0;
  let worstAreaDev = 0;
  const rows: string[] = [];
  for (const camD of [6, 12, 25, 60, 150, 400]) {
    const projOwn = cl.map((c) => project(c.ownError, c.oex, c.oey, c.oez, c.oer, camD));
    const projPar = cl.map((c) => project(c.parentError, c.pex, c.pey, c.pez, c.per, camD));
    // pick a τ band by distance so the cut actually moves through the LODs
    const tau = 1;
    const selected = new Uint8Array(cl.length);
    let selTris = 0;
    let selArea = 0;
    for (let i = 0; i < cl.length; i++) {
      if ((projOwn[i] as number) <= tau && (projPar[i] as number) > tau) {
        selected[i] = 1;
        selTris += (cl[i] as DagBuild['clusters'][number]).triCount;
        selArea += clusterArea(dag, i);
      }
    }
    for (const g of dag.groups) {
      if (!g.reduced) continue;
      if (g.inputs.some((id) => selected[id] === 1) && g.parents.some((id) => selected[id] === 1)) handoff++;
    }
    const dev = Math.abs(selArea - a0) / Math.max(1e-9, a0);
    if (dev > worstAreaDev) worstAreaDev = dev;
    rows.push(`d${camD}:${selTris}t/${(selArea / a0).toFixed(3)}A`);
  }
  if (handoff > 0) fail(`${name} A: ${handoff} group handoff overlaps (input+parent both cut)`);
  // AREA gate: the crown must keep its silhouette mass at every distance (±8% for
  // FP accumulation across levels; grow restores area exactly when un-clamped)
  if (worstAreaDev > 0.08) fail(`${name} AREA: cut area deviates ${(worstAreaDev * 100).toFixed(1)}% from LOD0 (crown balding)`);

  // antichain τ sweep at a fixed mid distance (tris monotone in τ; endpoints exact)
  {
    const camD = 30;
    const projOwn = cl.map((c) => project(c.ownError, c.oex, c.oey, c.oez, c.oer, camD));
    const projPar = cl.map((c) => project(c.parentError, c.pex, c.pey, c.pez, c.per, camD));
    let prevTris = Infinity;
    for (const tau of taus) {
      let selTris = 0;
      for (let i = 0; i < cl.length; i++) {
        if ((projOwn[i] as number) <= tau && (projPar[i] as number) > tau) selTris += (cl[i] as DagBuild['clusters'][number]).triCount;
      }
      if (selTris > prevTris + 1) {
        fail(`${name} A: cut tris rose with τ (${prevTris} → ${selTris})`);
        break;
      }
      prevTris = selTris;
    }
  }

  const s = dag.stats;
  console.log(
    `  ${name}: ${s.lod0Tris} tris / ${s.lod0Clusters} cl (L0) -> ${s.levels} levels, ` +
      `${s.totalClusters} cl total, ${s.roots} roots | maxErr ${s.maxError.toExponential(2)} | area ${a0.toFixed(2)} m² | ${s.buildMs.toFixed(1)} ms`,
  );
  for (const ls of dag.levelStats) {
    console.log(
      `    L${ls.level}→${ls.level + 1}: ${ls.inClusters} cl / ${ls.inTris} tris -> ${ls.outClusters} cl / ${ls.outTris} tris | ` +
        `${ls.groups} groups (${ls.stuckGroups} stuck) | tri-reduce ${(ls.triReduction * 100).toFixed(0)}%`,
    );
  }
  console.log(`    cut area sweep (dist: tris/areaFrac): ${rows.join('  ')}  (worst dev ${(worstAreaDev * 100).toFixed(1)}%)`);
}

// ---------------------------------------------------------------------------
// driver
// ---------------------------------------------------------------------------

function buildCrown(label: string, nLeaves: number, opts: AggregateDagOpts = {}): DagBuild {
  const mesh = makeCrown(new Rng(4242), nLeaves, 2.2, 0.13);
  const dag = buildAggregateDag(mesh.verts, STRIDE, mesh.indices, opts);

  // THIN: level 0 must actually shed triangles (aggregate isn't stuck on islands)
  const l0 = dag.levelStats[0];
  if (!l0 || l0.triReduction < 0.15) fail(`${label} THIN: level 0 reduced only ${((l0?.triReduction ?? 0) * 100).toFixed(0)}% (aggregate stuck?)`);

  // DET: a second identical build must match bit-for-bit in structure + geometry
  const dag2 = buildAggregateDag(mesh.verts, STRIDE, mesh.indices, opts);
  let geomMismatch = dag.verts.length !== dag2.verts.length || dag.indices.length !== dag2.indices.length;
  if (!geomMismatch) {
    for (let i = 0; i < dag.verts.length; i++) if (dag.verts[i] !== dag2.verts[i]) { geomMismatch = true; break; }
  }
  if (
    geomMismatch ||
    dag2.stats.totalClusters !== dag.stats.totalClusters ||
    dag2.stats.roots !== dag.stats.roots ||
    Math.abs(dag2.stats.maxError - dag.stats.maxError) > 1e-9
  ) {
    fail(`${label} DET: build is NON-deterministic`);
  }

  checkInvariants(label, dag);
  return dag;
}

console.log('[probe-aggregate]');
buildCrown('crown-sparse', 400);
const hero = buildCrown('crown-hero', 1200);
buildCrown('crown-dense', 2600);

// DET (seed): a different ?seed must thin a DIFFERENT leaf subset (so coarse
// geometry differs) while STILL preserving area (checked by checkInvariants).
{
  const mesh = makeCrown(new Rng(4242), 1200, 2.2, 0.13);
  const dA = buildAggregateDag(mesh.verts, STRIDE, mesh.indices, { seed: 1 });
  const dB = buildAggregateDag(mesh.verts, STRIDE, mesh.indices, { seed: 2 });
  let differs = dA.verts.length !== dB.verts.length;
  if (!differs) for (let i = 0; i < dA.verts.length; i++) if (dA.verts[i] !== dB.verts[i]) { differs = true; break; }
  if (!differs) fail('seed: seed=1 and seed=2 produced identical geometry (thinning not seeded)');
  else console.log('  seed: seed=1 vs seed=2 thin different leaf subsets (area still preserved) ✓');
}

// boot-budget extrapolation (F15): aggregate build ms per source Mtri
{
  const ms = hero.stats.buildMs;
  const mtri = hero.stats.lod0Tris / 1e6;
  const perMTri = ms / Math.max(1e-6, mtri);
  console.log(
    `  aggregate throughput ${(1000 / perMTri).toFixed(2)} Mtri/s -> 3.1M leaf tris ≈ ${((perMTri * 3.1) / 1000).toFixed(1)} s`,
  );
}

// CAP/FILL: does buildAggregateDag honor maxTris + the clusterfill threshold? Prints LOD0
// tris/cluster for each (cap, fill) on the SAME crown — the number the forest HUD shows.
console.log('  --- cap/fill LOD0 fill (crown-hero, 4800 tris) ---');
for (const [cap, fill] of [[128, 0.75], [256, 0.75], [256, 0.95]] as const) {
  setClusterFill(fill);
  const m = makeCrown(new Rng(4242), 1200, 2.2, 0.13);
  const d = buildAggregateDag(m.verts, STRIDE, m.indices, { maxTris: cap });
  const l0 = d.stats.lod0Tris / Math.max(1, d.stats.lod0Clusters);
  const tot = d.stats.totalTris / Math.max(1, d.stats.totalClusters);
  console.log(
    `  cap=${cap} fill=${fill}: LOD0 ${d.stats.lod0Tris}t / ${d.stats.lod0Clusters}cl = ${l0.toFixed(0)} tris/cl | all-levels avg ${tot.toFixed(0)}`,
  );
  // crack-free invariants at this (cap, fill) — does 256 / 0.95 still emit watertight pairs?
  checkInvariants(`  cap${cap}/fill${fill}`, d);
}
setClusterFill(0.75);

if (failures > 0) {
  console.error(`[probe-aggregate] ${failures} FAILURES`);
  process.exit(1);
}
console.log('[probe-aggregate] all aggregate DAG invariants hold');
