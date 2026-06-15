/**
 * Hierarchical DAG traversal data (N8 — replaces the brute-force cluster cull).
 *
 * THE PROBLEM (measured): the GPU cull dispatches EVERY cluster of every visible
 * instance (the mesh's clusterStart/Count spans all ~13 LOD levels) and tests the
 * per-cluster cut on each, discarding ~90%. That over-pushes the chunk queue (5.8×
 * overflow → flicker) and wastes the cull. It also throws away the entire point of a
 * DAG: descend from the roots and refine ONLY where the screen error is too coarse,
 * pruning whole subtrees.
 *
 * THE FIX: seed the traversal with each mesh's ROOT clusters, then per cluster:
 *   project(ownError) ≤ τ  ⇒ it's the cut, EMIT it
 *   project(ownError) > τ   ⇒ too coarse, ENQUEUE its children (the finer level)
 * which visits O(cut + path) clusters, not O(all). This module derives, from a built
 * DAG, the two things that traversal needs and the GPU doesn't have yet: per-cluster
 * CHILD links and the ROOT seed list. Pure CPU; the builders stay untouched.
 *
 * THE ONE SUBTLETY — owner dedup (crack-safety): a group has M parents ← N children,
 * and all M parents share the SAME own-error sphere (the bit-exact pair that makes the
 * cut crack-free), so they refine together. Naively descending from each parent would
 * enqueue the N children M times. We bake the fix into the data: ONLY group.parents[0]
 * (the "owner") carries the child list; the other parents carry none. Since they all
 * refine together, the owner enqueues the children exactly once — no runtime logic.
 */

import type { DagBuild } from './BuildDag';

export interface DagHierarchy {
  /** per-cluster [start,count) into childIndices — the finer clusters this cluster
   *  refines into. NON-zero only for a group's OWNER parent (dedup); 0 for the rest
   *  and for LOD0 (already finest). Mesh-LOCAL cluster ids. */
  childStart: Uint32Array; // length = clusters.length
  childCount: Uint32Array; // length = clusters.length
  /** flattened child cluster ids (mesh-local) */
  childIndices: Uint32Array;
  /** root cluster ids (parentError = +∞) — the traversal seeds. Mesh-local. */
  rootIndices: Uint32Array;
}

/** Derive child links (owner-gated) + the root seed list from a built DAG. */
export function buildDagHierarchy(dag: DagBuild): DagHierarchy {
  const n = dag.clusters.length;
  const childStart = new Uint32Array(n);
  const childCount = new Uint32Array(n);

  const roots: number[] = [];
  for (let c = 0; c < n; c++) {
    if (!Number.isFinite((dag.clusters[c] as { parentError: number }).parentError)) roots.push(c);
  }

  // pass 1: each reduced group's OWNER (parents[0]) gets its inputs as children.
  let total = 0;
  for (const g of dag.groups) {
    if (!g.reduced || g.parents.length === 0 || g.inputs.length === 0) continue;
    const owner = g.parents[0] as number;
    childStart[owner] = total;
    childCount[owner] = g.inputs.length;
    total += g.inputs.length;
  }
  // pass 2: fill the flattened child ids
  const childIndices = new Uint32Array(total);
  for (const g of dag.groups) {
    if (!g.reduced || g.parents.length === 0 || g.inputs.length === 0) continue;
    const owner = g.parents[0] as number;
    const base = childStart[owner] as number;
    for (let i = 0; i < g.inputs.length; i++) childIndices[base + i] = g.inputs[i] as number;
  }

  return { childStart, childCount, childIndices, rootIndices: new Uint32Array(roots) };
}

/**
 * CORRECTNESS GATE: the hierarchical traversal must reproduce the EXACT per-cluster
 * cut for every error threshold. For a sweep of thresholds E we compute both:
 *   reference cut = { c : ownError ≤ E < parentError }   (the per-cluster test)
 *   traversal cut = seed roots; while frontier: ownError ≤ E ⇒ emit, else ⇒ enqueue
 *                   children (owner-gated)
 * and require they are IDENTICAL sets. Also checks the traversal reaches every cluster
 * (no orphan), terminates, and visits each cluster once (owner dedup holds). Returns
 * the first failure or ok. (Error-threshold form — projection is monotone in 1/dist,
 * so set-equality at all E ⇒ set-equality at every camera distance.)
 */
export function validateDagHierarchy(dag: DagBuild, h: DagHierarchy): { ok: boolean; msg: string } {
  const n = dag.clusters.length;
  const own = (c: number): number => (dag.clusters[c] as { ownError: number }).ownError;
  const par = (c: number): number => (dag.clusters[c] as { parentError: number }).parentError;

  // a representative sweep of thresholds: every distinct ownError/parentError boundary
  const bounds = new Set<number>([0, Number.MAX_VALUE]);
  for (let c = 0; c < n; c++) {
    bounds.add(own(c));
    if (Number.isFinite(par(c))) bounds.add(par(c));
  }
  // also probe just-below/above each boundary (finite only — the cut formula
  // ownError ≤ E < parentError is undefined at E=∞: it yields ∅ because nothing
  // exceeds ∞, whereas the traversal correctly renders the roots. The cut is for
  // FINITE screen-error thresholds, so MAX_VALUE stands in for "coarsest".)
  const Es: number[] = [];
  for (const b of bounds) {
    for (const e of [b, b * (1 - 1e-6), b * (1 + 1e-6)]) if (Number.isFinite(e)) Es.push(e);
  }

  for (const E of Es) {
    // reference cut
    const ref = new Set<number>();
    for (let c = 0; c < n; c++) if (own(c) <= E && par(c) > E) ref.add(c);

    // traversal cut
    const got = new Set<number>();
    const seen = new Uint8Array(n);
    const stack = Array.from(h.rootIndices);
    let guard = 0;
    while (stack.length > 0) {
      if (++guard > n * 4 + 16) return { ok: false, msg: `traversal did not terminate at E=${E}` };
      const c = stack.pop() as number;
      if (seen[c]) return { ok: false, msg: `cluster ${c} visited twice (owner dedup broke) at E=${E}` };
      seen[c] = 1;
      if (own(c) <= E) {
        // fine enough at this distance → it's the cut. EVERY parent of a group is
        // emitted (owner or not), so emission is unconditional per cluster.
        got.add(c);
      } else {
        // too coarse → refine. Only a group's OWNER carries the child list; a
        // non-owner parent (count 0) skips — its owner sibling, reached in the same
        // frontier, enqueues the children once. (LOD0 never lands here: own = 0 ≤ E.)
        const s = h.childStart[c] as number;
        const k = h.childCount[c] as number;
        for (let i = 0; i < k; i++) stack.push(h.childIndices[s + i] as number);
      }
    }
    // sets must match exactly
    if (ref.size !== got.size) return { ok: false, msg: `cut size ${got.size} != ref ${ref.size} at E=${E}` };
    for (const c of ref) if (!got.has(c)) return { ok: false, msg: `ref cluster ${c} missing from traversal at E=${E}` };
  }
  return { ok: true, msg: `${Es.length} thresholds × exact-cut match; roots ${h.rootIndices.length}, childLinks ${h.childIndices.length}` };
}
