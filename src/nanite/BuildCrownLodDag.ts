/**
 * Crown-LOD DAG from a PRE-PRUNED anchor-graph ladder (crown-LOD Phase 2,
 * design §5.8 OPTION A — the graph-bake passthrough).
 *
 * The pruning already happened in VEGETATION space (TreeBuilder.foliageLadder /
 * VegLibrary.CROWN_LOD_SCHEDULE): each ladder rung is the crown mesh regenerated
 * from a de-correlated random keep-subset of WHOLE anchors (leaves / needle
 * sprays), survivors bit-identical to LOD0, keep-sets NESTED (coarser ⊆ finer).
 * So this builder must NOT re-prune, NOT weld across elements, and NOT √(1/λ)-grow
 * — those are exactly the merge/spike mechanisms Phase 1 was designed to avoid
 * (design §3b, §5.7). It is a pure PASSTHROUGH: clusterize each pre-pruned rung
 * independently (Clusterize preserves every triangle + vertex; it never fuses
 * elements), then stack the rungs as the DAG's LOD levels and wire the crack-free
 * cut records the raster already consumes.
 *
 * Structure = an ANCHOR-CHAIN (mirrors the terrain grid / voxel MIP DAGs — see
 * DagHierarchy.buildHeightGridHierarchy). A crown is a small object viewed as a
 * whole, so the WHOLE crown switches rung with distance: every cluster at a rung
 * shares ONE (ownError, ownSphere) pair, so they project the same screen error and
 * flip together (no partial-rung mix). Rung L's clusters descend into rung L-1
 * (finer) via a single group whose OWNER parent carries the finer rung as children
 * — the owner-dedup buildDagHierarchy expects. The runtime BFS cut
 * (NaniteCull.makeTraverse) then selects exactly one rung by projK·ownError/d ≤ τ.
 *
 * Levels are passed FINEST-FIRST: levels[0] = λ=1 (== LOD0, ownError 0, renders
 * nearest, never holes), levels[last] = the coarsest λ (roots, renders farthest,
 * up to the mesh→voxel handoff). ownErrors are strictly increasing (finer→coarser)
 * so the cut is a clean partition; see crownLodOwnErrors for the distance mapping.
 *
 * Determinism: no RNG (the ladder's keep-order was seeded in vegetation space).
 * Pure CPU / typed-array, node-runnable (no three.js, no GPU) like BuildAggregateDag.
 */

import type { DagBuild, DagCluster, DagGroup, DagLevelStats } from './BuildDag';
import { clusterize } from './Clusterize';
import { type Sphere, now } from './DagCommon';
import { computeVoxlodAnchorL0 } from './VoxelizeCrown';

/** one pre-pruned rung of the ladder, already in DAG vertex form (DAG_VERT_STRIDE
 *  floats/vertex via explicitToDagVerts) + its triangle indices. */
export interface CrownLodLevelMesh {
  verts: Float32Array;
  indices: Uint32Array;
}

export interface CrownLodDagOpts {
  /** cluster triangle cap (matches Clusterize / the QEM DAG), default 128 */
  maxTris?: number;
  /** per-rung own simplification error (LOCAL-space metres), FINEST-FIRST and
   *  strictly increasing; [0] MUST be 0 (LOD0 always emits when near). Length ==
   *  levels.length. Compute with crownLodOwnErrors(). */
  ownErrors: number[];
}

/** τ the CAMERA cut runs at (NaniteFrame ?loderr default). The crown leaf head is
 *  a camera-path mesh, so its rung cuts land at projK·ownError/τ. Baking with this
 *  τ makes errorScale=1 place each rung at its target distance. */
const CROWN_CUT_TAU = 3;

/** engagement distance of each COARSER rung as a fraction of transitionDist, at
 *  errorScale=1 — where that rung takes over from the finer one inside the mesh
 *  band. Rung 0 (λ=1) owns 0..frac[0]. Chosen to span 0..transitionDist and meet
 *  the voxel handoff at transitionDist. The 3 entries map onto CROWN_LOD_SCHEDULE's
 *  3 coarser rungs (λ = 0.7, 0.5, 0.35). errorScale (?leaflodk) shifts the whole
 *  set: >1 pushes rungs farther (fuller near crown / more tris), <1 nearer. */
const CROWN_ENGAGE_FRAC = [0.28, 0.56, 0.84];

/**
 * Per-rung ownError (LOCAL-space metres) that lands each coarser rung's cut across
 * 0..transitionDist. rung 0 = 0 (finest, always emits near). rung L≥1 lands at
 * CROWN_ENGAGE_FRAC[L-1]·transitionDist (at errorScale=1). Reuses the SAME
 * projK-anchored formula the voxel ladder uses (computeVoxlodAnchorL0 = d·τ/projK),
 * so mesh + voxel errors share one τ with no rescale. errorScale = ?leaflodk /
 * AGG_LOD_CFG style tuning multiplier (design §5.2). Extra rungs beyond the 3
 * fractions extrapolate on the last octave (kept monotone).
 */
export function crownLodOwnErrors(
  count: number,
  cfg: { transitionDist: number; renderHeight: number; fov: number; errorScale: number },
): number[] {
  const out = new Array<number>(count).fill(0);
  // anchorErr = ownError whose cut lands at transitionDist (one crown-band unit).
  const anchorErr = computeVoxlodAnchorL0(cfg.transitionDist, cfg.renderHeight, cfg.fov, CROWN_CUT_TAU);
  const scale = Number.isFinite(cfg.errorScale) && cfg.errorScale > 0 ? cfg.errorScale : 1;
  let prev = 0;
  for (let L = 1; L < count; L++) {
    const frac =
      L - 1 < CROWN_ENGAGE_FRAC.length
        ? (CROWN_ENGAGE_FRAC[L - 1] as number)
        : // extrapolate past the table: keep stepping by the last delta
          (CROWN_ENGAGE_FRAC[CROWN_ENGAGE_FRAC.length - 1] as number) +
          (L - CROWN_ENGAGE_FRAC.length) * 0.28;
    let e = anchorErr * frac * scale;
    // strictly increasing (a degenerate anchorErr / equal fracs must not collapse
    // two rungs into one un-cuttable level).
    if (e <= prev) e = prev * (1 + 1e-4) + 1e-6;
    out[L] = e;
    prev = e;
  }
  return out;
}

/** whole-rung bounding sphere (centroid of verts + max vertex distance) — the
 *  SHARED ownSphere every cluster at this rung references, so the rung flips as a
 *  unit (anchor-chain crack-freeness). Position is at stride offset 0/1/2. */
function levelSphere(verts: Float32Array, stride: number): Sphere {
  const n = verts.length / stride;
  if (n === 0) return { x: 0, y: 0, z: 0, r: 0 };
  let cx = 0;
  let cy = 0;
  let cz = 0;
  for (let v = 0; v < n; v++) {
    const b = v * stride;
    cx += verts[b] as number;
    cy += verts[b + 1] as number;
    cz += verts[b + 2] as number;
  }
  cx /= n;
  cy /= n;
  cz /= n;
  let r2 = 0;
  for (let v = 0; v < n; v++) {
    const b = v * stride;
    const dx = (verts[b] as number) - cx;
    const dy = (verts[b + 1] as number) - cy;
    const dz = (verts[b + 2] as number) - cz;
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 > r2) r2 = d2;
  }
  return { x: cx, y: cy, z: cz, r: Math.sqrt(r2) };
}

/**
 * Assemble the crown-LOD DAG from the pre-pruned ladder. Returns the SAME DagBuild
 * contract as buildDag / buildAggregateDag (verts/indices/clusters/groups/
 * levelStats/lod0Count/stats), so the registry attaches it identically (attachDag →
 * buildDagHierarchy → the flat cut) and meshletizeDag re-packs it unchanged.
 */
export function buildCrownLodDag(
  levels: CrownLodLevelMesh[],
  vertStride: number,
  opts: CrownLodDagOpts,
): DagBuild {
  const t0 = now();
  const maxTris = opts.maxTris ?? 128;

  // clusterize each pre-pruned rung independently (NO weld across rungs, NO grow).
  // Drop rungs that clusterize to nothing (empty pruned mesh) so the chain never
  // has a childless gap — the finest rung (0) is always kept if non-empty.
  const kept: { srcLevel: number; verts: Float32Array; indices: Uint32Array; built: ReturnType<typeof clusterize> }[] =
    [];
  for (let li = 0; li < levels.length; li++) {
    const lvl = levels[li] as CrownLodLevelMesh;
    if (lvl.indices.length < 3) continue;
    const built = clusterize(lvl.verts, vertStride, lvl.indices, maxTris);
    if (built.clusterCount === 0) continue;
    kept.push({ srcLevel: li, verts: lvl.verts, indices: built.indices, built });
  }
  if (kept.length === 0) {
    // nothing to build — return a minimal empty DAG (attachDag rejects 0 clusters,
    // so the caller treats an empty crown as "no aggregate", same as today).
    return emptyDag(vertStride, t0);
  }

  // per-rung shared cut sphere (whole-rung bound).
  const rungSphere = kept.map((k) => levelSphere(k.verts, vertStride));
  const K = kept.length;
  // ownError per KEPT rung: reuse the caller's per-source-level errors, but index
  // by kept position 0..K-1 (kept[0] is the finest present rung ⇒ ownError 0).
  const ownErr = new Array<number>(K).fill(0);
  {
    let prev = 0;
    for (let ki = 0; ki < K; ki++) {
      const srcE = opts.ownErrors[(kept[ki] as { srcLevel: number }).srcLevel] ?? 0;
      let e = ki === 0 ? 0 : srcE;
      if (ki > 0 && e <= prev) e = prev * (1 + 1e-4) + 1e-6; // strictly increasing
      ownErr[ki] = e;
      prev = e;
    }
  }

  // ---- concat verts + indices into one pool; build cluster records -------------
  let vTotal = 0;
  let iTotal = 0;
  for (const k of kept) {
    vTotal += k.verts.length;
    iTotal += k.indices.length;
  }
  const verts = new Float32Array(vTotal);
  const indices = new Uint32Array(iTotal);
  const clusters: DagCluster[] = [];
  const levelStats: DagLevelStats[] = [];
  // kept position → [firstClusterId, count] (for group wiring)
  const rungClusterStart: number[] = [];
  const rungClusterCount: number[] = [];

  let vBase = 0; // vertex offset (in vertices) of the current rung in the pool
  let tBase = 0; // triangle offset (in tris) of the current rung in the pool
  for (let ki = 0; ki < K; ki++) {
    const k = kept[ki] as { built: ReturnType<typeof clusterize>; verts: Float32Array; indices: Uint32Array };
    verts.set(k.verts, vBase * vertStride);
    // rebase this rung's (permuted) indices onto the appended vertex block
    const ib = tBase * 3;
    for (let i = 0; i < k.indices.length; i++) indices[ib + i] = (k.indices[i] as number) + vBase;

    const os = rungSphere[ki] as Sphere;
    const parentE = ki + 1 < K ? (ownErr[ki + 1] as number) : Infinity;
    const ps = ki + 1 < K ? (rungSphere[ki + 1] as Sphere) : os;

    rungClusterStart[ki] = clusters.length;
    rungClusterCount[ki] = k.built.clusterCount;
    let outTris = 0;
    for (let c = 0; c < k.built.clusterCount; c++) {
      const triCount = k.built.triCount[c] as number;
      outTris += triCount;
      clusters.push({
        level: ki,
        triStart: tBase + (k.built.triStart[c] as number),
        triCount,
        // geometric sphere / cone (per-cluster, TIGHT) — frustum + cone + min-size cull
        sx: k.built.sphere[c * 4] as number,
        sy: k.built.sphere[c * 4 + 1] as number,
        sz: k.built.sphere[c * 4 + 2] as number,
        sr: k.built.sphere[c * 4 + 3] as number,
        cax: k.built.cone[c * 4] as number,
        cay: k.built.cone[c * 4 + 1] as number,
        caz: k.built.cone[c * 4 + 2] as number,
        ccos: k.built.cone[c * 4 + 3] as number,
        // DAG cut: SHARED per-rung own pair (anchor-chain — whole rung flips together)
        ownError: ownErr[ki] as number,
        oex: os.x,
        oey: os.y,
        oez: os.z,
        oer: os.r,
        // parent pair == the coarser rung's own pair, bit-for-bit (par(child) ==
        // own(parent) ⇒ the BFS cut reproduces the scalar-error partition). Coarsest
        // rung = root (parentError +∞).
        parentError: parentE,
        pex: ps.x,
        pey: ps.y,
        pez: ps.z,
        per: ps.r,
        groupAsInput: ki + 1 < K ? ki : -1, // group index = child rung position
        groupAsParent: ki > 0 ? ki - 1 : -1,
      });
    }

    levelStats.push({
      level: ki,
      inClusters: k.built.clusterCount,
      inTris: outTris,
      outClusters: 0,
      outTris: 0,
      groups: 0,
      stuckGroups: 0,
      triReduction: 0,
    });

    vBase += k.verts.length / vertStride;
    tBase += k.indices.length / 3;
  }

  // ---- groups: one per finer→coarser transition (anchor-chain child links) -----
  // group ki links child rung ki (inputs) to parent rung ki+1 (parents). The
  // OWNER = parents[0] (first coarser cluster) carries the finer rung as children;
  // buildDagHierarchy reads exactly this to seed the BFS descent.
  const groups: DagGroup[] = [];
  for (let ki = 0; ki + 1 < K; ki++) {
    const childStart = rungClusterStart[ki] as number;
    const childCount = rungClusterCount[ki] as number;
    const parStart = rungClusterStart[ki + 1] as number;
    const parCount = rungClusterCount[ki + 1] as number;
    const inputs: number[] = [];
    for (let c = 0; c < childCount; c++) inputs.push(childStart + c);
    const parents: number[] = [];
    for (let c = 0; c < parCount; c++) parents.push(parStart + c);
    const ps = rungSphere[ki + 1] as Sphere;
    groups.push({
      id: ki,
      level: ki,
      error: ownErr[ki + 1] as number,
      sx: ps.x,
      sy: ps.y,
      sz: ps.z,
      sr: ps.r,
      inputs,
      parents,
      reduced: true,
    });
  }

  const lod0Count = rungClusterCount[0] as number;
  let roots = 0;
  let totalTris = 0;
  let maxError = 0;
  for (const c of clusters) {
    if (!Number.isFinite(c.parentError)) roots++;
    totalTris += c.triCount;
    if (c.ownError > maxError) maxError = c.ownError;
  }

  return {
    verts,
    vertStride,
    indices,
    clusters,
    groups,
    levelStats,
    lod0Count,
    stats: {
      buildMs: now() - t0,
      levels: K,
      lod0Clusters: lod0Count,
      lod0Tris: (kept[0] as { indices: Uint32Array }).indices.length / 3,
      totalClusters: clusters.length,
      totalTris,
      roots,
      maxError,
    },
  };
}

function emptyDag(vertStride: number, t0: number): DagBuild {
  return {
    verts: new Float32Array(0),
    vertStride,
    indices: new Uint32Array(0),
    clusters: [],
    groups: [],
    levelStats: [],
    lod0Count: 0,
    stats: {
      buildMs: now() - t0,
      levels: 0,
      lod0Clusters: 0,
      lod0Tris: 0,
      totalClusters: 0,
      totalTris: 0,
      roots: 0,
      maxError: 0,
    },
  };
}
