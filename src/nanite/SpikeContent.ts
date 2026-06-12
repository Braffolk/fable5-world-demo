/**
 * N0 spike content — one rock pool (3 variants) + one heightfield terrain
 * tile, packed into mega-buffer typed arrays with 128-tri clusters.
 *
 * Spike-grade clusterizer: SEQUENTIAL 128-tri chunks with bounding spheres
 * (the real greedy adjacency clusterizer is N1). Terrain clusters carry NO
 * vertices — they reference an 8×8-quad window of the height texture and the
 * kernels reconstruct positions implicitly (the F4 heightfield-ClusterSource
 * mechanism this spike exists to prove).
 *
 * Cluster meta layout (uvec4): [kind(0 mesh | 1 terrain), triStart, triCount,
 * window (gx | gz<<16)]. Mesh cluster spheres are LOCAL space (instance
 * transform applied in-kernel); terrain spheres are WORLD space.
 */

import { DataTexture, FloatType, RedFormat, type BufferGeometry } from 'three';
import type { Rng } from '../core/Seed';
import { buildRock } from '../vegetation/RockBuilder';

export const CLUSTER_TRIS = 128;
/** terrain tile: 256×256 quads of 1 m cells → 257² height samples */
export const TERRAIN_QUADS = 256;
export const TERRAIN_CELL = 1.0;
/** quads per terrain cluster edge (8×8 quads = 128 tris) */
export const TERRAIN_WIN = 8;

export interface SpikeContent {
  /** mesh mega-buffers (rock variants only — terrain is implicit) */
  positions: Float32Array; // vec4 padded
  indices: Uint32Array;
  /** per cluster: sphere vec4 (xyz, r) */
  clusterSphere: Float32Array;
  /** per cluster: uvec4 [kind, triStart, triCount, window] */
  clusterMeta: Uint32Array;
  clusterCount: number;
  /** per mesh: [clusterStart, clusterCount] (mesh 0..2 = rocks, 3 = terrain) */
  meshTable: Uint32Array;
  /** instances: A = (x,y,z,scale), B = (yaw, leanX, leanZ, meshId) */
  instA: Float32Array;
  instB: Float32Array;
  instanceCount: number;
  /** 257² world heights (y) for the tile, row-major, x-fastest */
  heights: Float32Array;
  heightTex: DataTexture;
  /** world origin of the tile's (0,0) sample */
  tileOrigin: { x: number; z: number };
  stats: {
    rockTris: number[];
    terrainTris: number;
    totalSourceTris: number;
    instancedTris: number;
  };
  /** ground height lookup for camera/instance placement */
  groundY(x: number, z: number): number;
}

/** deterministic value-noise fbm for the spike heightfield (CPU) */
function makeHeightField(rng: Rng): Float32Array {
  const n = TERRAIN_QUADS + 1;
  const out = new Float32Array(n * n);
  // hash-based gradient-free value noise; few octaves of rolling ground
  const seed = rng.u32();
  const hash = (xi: number, zi: number): number => {
    let h = (xi * 374761393 + zi * 668265263) ^ seed;
    h = Math.imul(h ^ (h >>> 13), 1274126177);
    h ^= h >>> 16;
    return (h >>> 0) / 4294967296;
  };
  const vnoise = (x: number, z: number): number => {
    const xi = Math.floor(x);
    const zi = Math.floor(z);
    const fx = x - xi;
    const fz = z - zi;
    const sx = fx * fx * (3 - 2 * fx);
    const sz = fz * fz * (3 - 2 * fz);
    const a = hash(xi, zi);
    const b = hash(xi + 1, zi);
    const c = hash(xi, zi + 1);
    const d = hash(xi + 1, zi + 1);
    return a + (b - a) * sx + (c - a) * sz + (a - b - c + d) * sx * sz;
  };
  for (let z = 0; z < n; z++) {
    for (let x = 0; x < n; x++) {
      let h = 0;
      let amp = 7.0;
      let freq = 1 / 90;
      for (let o = 0; o < 5; o++) {
        h += (vnoise(x * TERRAIN_CELL * freq, z * TERRAIN_CELL * freq) - 0.5) * 2 * amp;
        amp *= 0.5;
        freq *= 2.1;
      }
      out[z * n + x] = h;
    }
  }
  return out;
}

interface PackedMesh {
  triStart: number;
  triCount: number;
}

export function buildSpikeContent(rng: Rng): SpikeContent {
  // ---- rock variants -------------------------------------------------------
  const rockGeos: BufferGeometry[] = [];
  const rockTris: number[] = [];
  const presets = ['boulder', 'angular', 'boulder'] as const;
  for (let i = 0; i < 3; i++) {
    const r = buildRock(presets[i] as Parameters<typeof buildRock>[0], rng.fork(`rock${i}`), 4);
    rockGeos.push(r.geometry);
    rockTris.push(r.stats.tris);
  }

  let totalVerts = 0;
  let totalTris = 0;
  for (const g of rockGeos) {
    totalVerts += g.attributes.position?.count ?? 0;
    totalTris += (g.index?.count ?? 0) / 3;
  }

  const positions = new Float32Array(totalVerts * 4);
  const indices = new Uint32Array(totalTris * 3);
  const packed: PackedMesh[] = [];
  {
    let vOff = 0;
    let tOff = 0;
    for (const g of rockGeos) {
      const pos = g.attributes.position;
      const idx = g.index;
      if (!pos || !idx) throw new Error('spike: rock geometry must be indexed');
      for (let i = 0; i < pos.count; i++) {
        positions[(vOff + i) * 4 + 0] = pos.getX(i);
        positions[(vOff + i) * 4 + 1] = pos.getY(i);
        positions[(vOff + i) * 4 + 2] = pos.getZ(i);
        positions[(vOff + i) * 4 + 3] = 1;
      }
      const triCount = idx.count / 3;
      for (let t = 0; t < idx.count; t++) {
        indices[tOff * 3 + t] = vOff + idx.getX(t);
      }
      packed.push({ triStart: tOff, triCount });
      vOff += pos.count;
      tOff += triCount;
    }
  }

  // ---- clusters: sequential 128-tri chunks + spheres (mesh kind 0) ---------
  const sphereOf = (triStart: number, triCount: number): [number, number, number, number] => {
    let cx = 0;
    let cy = 0;
    let cz = 0;
    const vCount = triCount * 3;
    for (let t = 0; t < triCount; t++) {
      for (let v = 0; v < 3; v++) {
        const vi = indices[(triStart + t) * 3 + v] as number;
        cx += positions[vi * 4 + 0] as number;
        cy += positions[vi * 4 + 1] as number;
        cz += positions[vi * 4 + 2] as number;
      }
    }
    cx /= vCount;
    cy /= vCount;
    cz /= vCount;
    let r2 = 0;
    for (let t = 0; t < triCount; t++) {
      for (let v = 0; v < 3; v++) {
        const vi = indices[(triStart + t) * 3 + v] as number;
        const dx = (positions[vi * 4 + 0] as number) - cx;
        const dy = (positions[vi * 4 + 1] as number) - cy;
        const dz = (positions[vi * 4 + 2] as number) - cz;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 > r2) r2 = d2;
      }
    }
    return [cx, cy, cz, Math.sqrt(r2)];
  };

  const spheres: number[] = [];
  const metas: number[] = [];
  const meshTable: number[] = [];
  for (const m of packed) {
    const clusterStart = metas.length / 4;
    const nClusters = Math.ceil(m.triCount / CLUSTER_TRIS);
    for (let c = 0; c < nClusters; c++) {
      const triStart = m.triStart + c * CLUSTER_TRIS;
      const triCount = Math.min(CLUSTER_TRIS, m.triCount - c * CLUSTER_TRIS);
      spheres.push(...sphereOf(triStart, triCount));
      metas.push(0, triStart, triCount, 0);
    }
    meshTable.push(clusterStart, nClusters);
  }

  // ---- terrain tile (mesh 3, kind 1, implicit verts) -----------------------
  const heights = makeHeightField(rng.fork('heights'));
  const n = TERRAIN_QUADS + 1;
  const tileOrigin = { x: -(TERRAIN_QUADS * TERRAIN_CELL) / 2, z: -(TERRAIN_QUADS * TERRAIN_CELL) / 2 };
  const groundY = (x: number, z: number): number => {
    const gx = Math.min(Math.max((x - tileOrigin.x) / TERRAIN_CELL, 0), TERRAIN_QUADS - 1e-4);
    const gz = Math.min(Math.max((z - tileOrigin.z) / TERRAIN_CELL, 0), TERRAIN_QUADS - 1e-4);
    const xi = Math.floor(gx);
    const zi = Math.floor(gz);
    const fx = gx - xi;
    const fz = gz - zi;
    const h00 = heights[zi * n + xi] as number;
    const h10 = heights[zi * n + xi + 1] as number;
    const h01 = heights[(zi + 1) * n + xi] as number;
    const h11 = heights[(zi + 1) * n + xi + 1] as number;
    return h00 * (1 - fx) * (1 - fz) + h10 * fx * (1 - fz) + h01 * (1 - fx) * fz + h11 * fx * fz;
  };

  {
    const terrainClusterStart = metas.length / 4;
    const wins = TERRAIN_QUADS / TERRAIN_WIN; // 32×32 windows
    for (let wz = 0; wz < wins; wz++) {
      for (let wx = 0; wx < wins; wx++) {
        const gx = wx * TERRAIN_WIN;
        const gz = wz * TERRAIN_WIN;
        // world-space sphere over the window's height range
        let hMin = Infinity;
        let hMax = -Infinity;
        for (let z = gz; z <= gz + TERRAIN_WIN; z++) {
          for (let x = gx; x <= gx + TERRAIN_WIN; x++) {
            const h = heights[z * n + x] as number;
            if (h < hMin) hMin = h;
            if (h > hMax) hMax = h;
          }
        }
        const half = (TERRAIN_WIN * TERRAIN_CELL) / 2;
        const cx = tileOrigin.x + (gx + TERRAIN_WIN / 2) * TERRAIN_CELL;
        const cz = tileOrigin.z + (gz + TERRAIN_WIN / 2) * TERRAIN_CELL;
        const cy = (hMin + hMax) / 2;
        const r = Math.sqrt(half * half * 2 + ((hMax - hMin) / 2) ** 2);
        spheres.push(cx, cy, cz, r);
        metas.push(1, 0, TERRAIN_WIN * TERRAIN_WIN * 2, gx | (gz << 16));
      }
    }
    meshTable.push(terrainClusterStart, wins * wins);
  }

  const heightTex = new DataTexture(heights, n, n, RedFormat, FloatType);
  heightTex.needsUpdate = true;

  // ---- instances ------------------------------------------------------------
  // 44×44×40 clusters ≈ 77k total, ~55k in-frustum worst case < 65535 work cap
  const GRID = 44;
  const SPACING = 4.5;
  const rockCount = GRID * GRID;
  const instanceCount = rockCount + 1;
  const instA = new Float32Array(instanceCount * 4);
  const instB = new Float32Array(instanceCount * 4);
  const irng = rng.fork('instances');
  let instancedTris = 0;
  for (let i = 0; i < rockCount; i++) {
    const gx = i % GRID;
    const gz = Math.floor(i / GRID);
    const x = (gx - GRID / 2 + 0.5) * SPACING + irng.range(-1.4, 1.4);
    const z = (gz - GRID / 2 + 0.5) * SPACING + irng.range(-1.4, 1.4);
    const scale = irng.range(0.7, 1.6);
    const meshId = irng.int(3);
    instA[i * 4 + 0] = x;
    instA[i * 4 + 1] = groundY(x, z) + 0.15 * scale;
    instA[i * 4 + 2] = z;
    instA[i * 4 + 3] = scale;
    instB[i * 4 + 0] = irng.range(0, Math.PI * 2);
    instB[i * 4 + 1] = 0;
    instB[i * 4 + 2] = 0;
    instB[i * 4 + 3] = meshId;
    instancedTris += rockTris[meshId] as number;
  }
  // terrain instance (identity, meshId 3)
  const ti = rockCount;
  instA[ti * 4 + 3] = 1;
  instB[ti * 4 + 3] = 3;
  const terrainTris = TERRAIN_QUADS * TERRAIN_QUADS * 2;
  instancedTris += terrainTris;

  return {
    positions,
    indices,
    clusterSphere: new Float32Array(spheres),
    clusterMeta: new Uint32Array(metas),
    clusterCount: metas.length / 4,
    meshTable: new Uint32Array(meshTable),
    instA,
    instB,
    instanceCount,
    heights,
    heightTex,
    tileOrigin,
    stats: { rockTris, terrainTris, totalSourceTris: totalTris + terrainTris, instancedTris },
    groundY,
  };
}
