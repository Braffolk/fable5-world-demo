/**
 * Greedy mesh clusterizer (NANITE-SPEC.md N1) — splits an indexed triangle mesh
 * into clusters of ≤ maxTris triangles, grown over shared-edge adjacency
 * with centroid-proximity priority (tight bounding spheres, good locality).
 *
 * Outputs a PERMUTED index buffer (each cluster's triangles contiguous) plus
 * per-cluster bounding sphere, backface normal cone, and tri range. Pure
 * CPU/typed-array code — no three.js imports, runs under node for the
 * validation probe (tools/probe-clusterize.ts).
 *
 * Performance notes (budget: all pools ≈ 10–20M source tris in < 2 s):
 * adjacency via open-addressing hash on typed arrays (no JS Map), frontier
 * via a small binary heap; everything O(n)–O(n log k).
 */

export interface BuiltClusters {
  /** permuted copy of the input indices; cluster c owns tris [triStart[c], triStart[c]+triCount[c]) */
  indices: Uint32Array;
  clusterCount: number;
  /** xyz + radius per cluster */
  sphere: Float32Array;
  /** unit axis xyz + cos(half-angle) per cluster; cos = -1 disables backface culling */
  cone: Float32Array;
  triStart: Uint32Array;
  triCount: Uint32Array;
  stats: ClusterStats;
}

export interface ClusterStats {
  adjMs: number;
  growMs: number;
  metricsMs: number;
  tris: number;
  clusters: number;
  avgTris: number;
  minTris: number;
  /** fraction of clusters at ≥ 0.75 × maxTris */
  fullFrac: number;
  /** mean sphere radius / mesh bounding radius (lower = tighter) */
  meanRadiusFrac: number;
  buildMs: number;
}

/** open-addressing edge map: key (loVert,hiVert) → first tri + slot reuse for the pair */
function buildAdjacency(indices: Uint32Array, triCount: number): Int32Array {
  const edgeCount = triCount * 3;
  let cap = 1;
  while (cap < edgeCount * 2) cap <<= 1;
  const mask = cap - 1;
  const keyLo = new Uint32Array(cap);
  const keyHi = new Uint32Array(cap);
  const slotTri = new Int32Array(cap).fill(-1);
  const slotEdge = new Int32Array(cap); // which edge (0..2) of slotTri
  const adj = new Int32Array(edgeCount).fill(-1);

  for (let t = 0; t < triCount; t++) {
    for (let e = 0; e < 3; e++) {
      const a = indices[t * 3 + e] as number;
      const b = indices[t * 3 + ((e + 1) % 3)] as number;
      const lo = a < b ? a : b;
      const hi = a < b ? b : a;
      // 64-bit-ish hash from the vertex pair
      let h = (lo * 0x85ebca6b) ^ (hi * 0xc2b2ae35);
      h = (h ^ (h >>> 13)) >>> 0;
      let s = h & mask;
      for (;;) {
        const st = slotTri[s] as number;
        if (st === -1) {
          // empty: claim for this edge's first side
          keyLo[s] = lo;
          keyHi[s] = hi;
          slotTri[s] = t;
          slotEdge[s] = e;
          break;
        }
        if (keyLo[s] === lo && keyHi[s] === hi) {
          // pair found: link both sides, then retire the slot (3rd+ tri on a
          // non-manifold edge re-claims a fresh slot — acceptable: first two
          // tris pair, extras stay unpaired)
          adj[t * 3 + e] = st;
          adj[st * 3 + (slotEdge[s] as number)] = t;
          slotTri[s] = -2; // tombstone: keeps probe chains intact
          break;
        }
        s = (s + 1) & mask;
      }
    }
  }
  return adj;
}

export function clusterize(
  positions: Float32Array,
  posStride: number,
  indices: Uint32Array,
  maxTris = 128,
): BuiltClusters {
  const t0 = performance.now();
  const triCount = (indices.length / 3) | 0;
  const adj = buildAdjacency(indices, triCount);
  const tAdj = performance.now();

  // per-tri centroids (flat arrays; drive the growth priority)
  const cx = new Float32Array(triCount);
  const cy = new Float32Array(triCount);
  const cz = new Float32Array(triCount);
  for (let t = 0; t < triCount; t++) {
    const i0 = (indices[t * 3] as number) * posStride;
    const i1 = (indices[t * 3 + 1] as number) * posStride;
    const i2 = (indices[t * 3 + 2] as number) * posStride;
    const x0 = positions[i0] as number;
    const y0 = positions[i0 + 1] as number;
    const z0 = positions[i0 + 2] as number;
    const x1 = positions[i1] as number;
    const y1 = positions[i1 + 1] as number;
    const z1 = positions[i1 + 2] as number;
    const x2 = positions[i2] as number;
    const y2 = positions[i2 + 1] as number;
    const z2 = positions[i2 + 2] as number;
    cx[t] = (x0 + x1 + x2) / 3;
    cy[t] = (y0 + y1 + y2) / 3;
    cz[t] = (z0 + z1 + z2) / 3;
  }

  const used = new Uint8Array(triCount);
  const inFrontier = new Uint8Array(triCount);
  // binary min-heap of (dist², tri)
  const heapT = new Int32Array(maxTris * 64);
  const heapD = new Float32Array(maxTris * 64);
  let heapN = 0;
  const heapPush = (tri: number, d: number): boolean => {
    if (heapN >= heapT.length) return false; // frontier full: caller must not mark
    let i = heapN++;
    heapT[i] = tri;
    heapD[i] = d;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if ((heapD[p] as number) <= (heapD[i] as number)) break;
      const td = heapD[p] as number;
      const tt = heapT[p] as number;
      heapD[p] = heapD[i] as number;
      heapT[p] = heapT[i] as number;
      heapD[i] = td;
      heapT[i] = tt;
      i = p;
    }
    return true;
  };
  const heapPop = (): number => {
    const top = heapT[0] as number;
    heapN--;
    if (heapN > 0) {
      heapD[0] = heapD[heapN] as number;
      heapT[0] = heapT[heapN] as number;
      let i = 0;
      for (;;) {
        const l = i * 2 + 1;
        const r = l + 1;
        let m = i;
        if (l < heapN && (heapD[l] as number) < (heapD[m] as number)) m = l;
        if (r < heapN && (heapD[r] as number) < (heapD[m] as number)) m = r;
        if (m === i) break;
        const td = heapD[m] as number;
        const tt = heapT[m] as number;
        heapD[m] = heapD[i] as number;
        heapT[m] = heapT[i] as number;
        heapD[i] = td;
        heapT[i] = tt;
        i = m;
      }
    }
    return top;
  };

  const outIndices = new Uint32Array(indices.length);
  const triStartArr: number[] = [];
  const triCountArr: number[] = [];
  const clusterTris: number[] = []; // scratch: tris of the current cluster
  let outTri = 0;
  let seedScan = 0;
  let carrySeed = -1; // leftover-frontier tri: next cluster starts adjacent

  while (true) {
    // next seed: previous cluster's frontier first (coherent surface sweep,
    // fewer islands), index scan as fallback
    let seed = -1;
    if (carrySeed >= 0 && (used[carrySeed] as number) === 0) {
      seed = carrySeed;
    } else {
      while (seedScan < triCount && (used[seedScan] as number) === 1) seedScan++;
      if (seedScan >= triCount) break;
      seed = seedScan;
    }
    carrySeed = -1;

    clusterTris.length = 0;
    heapN = 0;
    let ccx = cx[seed] as number;
    let ccy = cy[seed] as number;
    let ccz = cz[seed] as number;
    heapPush(seed, 0);
    inFrontier[seed] = 1;

    while (clusterTris.length < maxTris) {
      if (heapN === 0) {
        // frontier died with the cluster underfull: refill from the index
        // scan (generated meshes have spatially coherent index order) so
        // clusters stay near-full instead of fragmenting into islands
        if (clusterTris.length >= maxTris * 0.75) break;
        while (seedScan < triCount && (used[seedScan] as number) === 1) seedScan++;
        if (seedScan >= triCount) break;
        heapPush(seedScan, 0);
        inFrontier[seedScan] = 1;
        continue;
      }
      const t = heapPop();
      if ((used[t] as number) === 1) continue;
      used[t] = 1;
      clusterTris.push(t);
      // running centroid (cheap sphere-growth proxy)
      const k = clusterTris.length;
      ccx += ((cx[t] as number) - ccx) / k;
      ccy += ((cy[t] as number) - ccy) / k;
      ccz += ((cz[t] as number) - ccz) / k;
      for (let e = 0; e < 3; e++) {
        const n = adj[t * 3 + e] as number;
        if (n >= 0 && (used[n] as number) === 0 && (inFrontier[n] as number) === 0) {
          const dx = (cx[n] as number) - ccx;
          const dy = (cy[n] as number) - ccy;
          const dz = (cz[n] as number) - ccz;
          if (heapPush(n, dx * dx + dy * dy + dz * dz)) inFrontier[n] = 1;
        }
      }
    }
    // clear frontier marks; keep the nearest leftover as the next seed
    for (let i = 0; i < heapN; i++) {
      const t = heapT[i] as number;
      inFrontier[t] = 0;
      if (carrySeed === -1 && (used[t] as number) === 0) carrySeed = t;
    }
    for (const t of clusterTris) inFrontier[t] = 0;
    heapN = 0;

    triStartArr.push(outTri);
    triCountArr.push(clusterTris.length);
    for (const t of clusterTris) {
      outIndices[outTri * 3] = indices[t * 3] as number;
      outIndices[outTri * 3 + 1] = indices[t * 3 + 1] as number;
      outIndices[outTri * 3 + 2] = indices[t * 3 + 2] as number;
      outTri++;
    }
  }

  const tGrow = performance.now();
  // ---- per-cluster sphere + cone -------------------------------------------
  const clusterCount = triStartArr.length;
  const sphere = new Float32Array(clusterCount * 4);
  const cone = new Float32Array(clusterCount * 4);
  let radiusSum = 0;
  // mesh bounding radius for stats
  let mcx = 0;
  let mcy = 0;
  let mcz = 0;
  const vCount = (positions.length / posStride) | 0;
  for (let v = 0; v < vCount; v++) {
    mcx += positions[v * posStride] as number;
    mcy += positions[v * posStride + 1] as number;
    mcz += positions[v * posStride + 2] as number;
  }
  mcx /= vCount;
  mcy /= vCount;
  mcz /= vCount;
  let meshR2 = 0;
  for (let v = 0; v < vCount; v++) {
    const dx = (positions[v * posStride] as number) - mcx;
    const dy = (positions[v * posStride + 1] as number) - mcy;
    const dz = (positions[v * posStride + 2] as number) - mcz;
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 > meshR2) meshR2 = d2;
  }

  for (let c = 0; c < clusterCount; c++) {
    const start = triStartArr[c] as number;
    const count = triCountArr[c] as number;
    let sx = 0;
    let sy = 0;
    let sz = 0;
    for (let t = start; t < start + count; t++) {
      for (let v = 0; v < 3; v++) {
        const p = (outIndices[t * 3 + v] as number) * posStride;
        sx += positions[p] as number;
        sy += positions[p + 1] as number;
        sz += positions[p + 2] as number;
      }
    }
    const n3 = count * 3;
    sx /= n3;
    sy /= n3;
    sz /= n3;
    let r2 = 0;
    for (let t = start; t < start + count; t++) {
      for (let v = 0; v < 3; v++) {
        const p = (outIndices[t * 3 + v] as number) * posStride;
        const dx = (positions[p] as number) - sx;
        const dy = (positions[p + 1] as number) - sy;
        const dz = (positions[p + 2] as number) - sz;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 > r2) r2 = d2;
      }
    }
    sphere[c * 4] = sx;
    sphere[c * 4 + 1] = sy;
    sphere[c * 4 + 2] = sz;
    sphere[c * 4 + 3] = Math.sqrt(r2);
    radiusSum += Math.sqrt(r2);

    // cone: area-weighted mean axis, min cos vs faces. NOTE: the original tri
    // ids for this cluster are no longer needed — recompute face normals from
    // the permuted indices (cheap, avoids carrying a tri permutation array).
    let axx = 0;
    let axy = 0;
    let axz = 0;
    for (let t = start; t < start + count; t++) {
      const i0 = (outIndices[t * 3] as number) * posStride;
      const i1 = (outIndices[t * 3 + 1] as number) * posStride;
      const i2 = (outIndices[t * 3 + 2] as number) * posStride;
      const ax = (positions[i1] as number) - (positions[i0] as number);
      const ay = (positions[i1 + 1] as number) - (positions[i0 + 1] as number);
      const az = (positions[i1 + 2] as number) - (positions[i0 + 2] as number);
      const bx = (positions[i2] as number) - (positions[i0] as number);
      const by = (positions[i2 + 1] as number) - (positions[i0 + 1] as number);
      const bz = (positions[i2 + 2] as number) - (positions[i0 + 2] as number);
      axx += ay * bz - az * by;
      axy += az * bx - ax * bz;
      axz += ax * by - ay * bx;
    }
    const alen = Math.hypot(axx, axy, axz);
    if (alen < 1e-12) {
      cone[c * 4 + 3] = -1; // degenerate spread: never backface-cull
    } else {
      axx /= alen;
      axy /= alen;
      axz /= alen;
      let minCos = 1;
      for (let t = start; t < start + count; t++) {
        const i0 = (outIndices[t * 3] as number) * posStride;
        const i1 = (outIndices[t * 3 + 1] as number) * posStride;
        const i2 = (outIndices[t * 3 + 2] as number) * posStride;
        const ax = (positions[i1] as number) - (positions[i0] as number);
        const ay = (positions[i1 + 1] as number) - (positions[i0 + 1] as number);
        const az = (positions[i1 + 2] as number) - (positions[i0 + 2] as number);
        const bx = (positions[i2] as number) - (positions[i0] as number);
        const by = (positions[i2 + 1] as number) - (positions[i0 + 1] as number);
        const bz = (positions[i2 + 2] as number) - (positions[i0 + 2] as number);
        let fx = ay * bz - az * by;
        let fy = az * bx - ax * bz;
        let fz = ax * by - ay * bx;
        const fl = Math.hypot(fx, fy, fz);
        if (fl < 1e-12) continue;
        fx /= fl;
        fy /= fl;
        fz /= fl;
        const d = fx * axx + fy * axy + fz * axz;
        if (d < minCos) minCos = d;
      }
      cone[c * 4] = axx;
      cone[c * 4 + 1] = axy;
      cone[c * 4 + 2] = axz;
      // a cone only culls when the whole cluster faces away; spreads ≥ 90°
      // can never cull — disable so the GPU test can shortcut
      cone[c * 4 + 3] = minCos > 0 ? minCos : -1;
    }
  }

  let minTris = triCount;
  let fullCount = 0;
  for (const n of triCountArr) {
    if (n < minTris) minTris = n;
    if (n >= maxTris * 0.75) fullCount++;
  }
  const fullFrac = fullCount / Math.max(1, clusterCount);
  return {
    indices: outIndices,
    clusterCount,
    sphere,
    cone,
    triStart: Uint32Array.from(triStartArr),
    triCount: Uint32Array.from(triCountArr),
    stats: {
      adjMs: tAdj - t0,
      growMs: tGrow - tAdj,
      metricsMs: performance.now() - tGrow,
      tris: triCount,
      clusters: clusterCount,
      avgTris: triCount / Math.max(1, clusterCount),
      minTris: triCountArr.length ? minTris : 0,
      fullFrac,
      meanRadiusFrac: radiusSum / Math.max(1, clusterCount) / Math.max(1e-9, Math.sqrt(meshR2)),
      buildMs: performance.now() - t0,
    },
  };
}
