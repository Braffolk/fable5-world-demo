/**
 * WorldRegistry (N1-C4, docs/NANITE.md) — builds THE GeometryRegistry from the
 * real world's opaque pools behind ?nanite=1. No rendering changes yet (that
 * is N2/N3); this wires the content contract end to end and measures it.
 *
 * Opaque-part policy (PATH UNIFICATION AUDIT):
 *  - tree pools (cls 0–5): bark part of r0/r1/r2 as a discrete LOD chain
 *    (switch at R0_FAR=26 m, R1_FAR=150 m — Forests ring radii); foliage
 *    CARDS + hero mesh leaves DEFERRED to N9 (alpha/leaf path).
 *  - shrubs (8–10): bark part, single ring.
 *  - ferns/flowers (11–14): leafy card geometry — DEFERRED to N9 entirely.
 *  - logs/stumps (16–17), branches (23): deadwood, r1 (branch r2 is a clone
 *    that exists only for indirect-slot bookkeeping — one registration).
 *  - boulders/slabs (18–19), stones L/M (20–21): rock, r1→r2 LOD chain at
 *    EX_R1_FAR=120 m; StoneS (22): single ring.
 *  - terrain: ONE heightfield source over the full field (winQuads 7,
 *    partial edge windows), single identity instance.
 *  - GroundRing grass/debris: clipmap-instanced (not boot-static scatter) —
 *    stays on its bespoke path until N6/N10 per the audit table.
 *
 * Instances: scatter layers are GPU-resident MIXED-class buffers (idF =
 * cls·8+variant in B.w). C4 reads each layer back ONCE at boot (placements
 * are static) and partitions on CPU into per-mesh contiguous streams —
 * deterministic, order-preserving, and the registry uploads one packed blob.
 * Instances of deferred classes are counted and skipped.
 */

import type { Renderer, StorageBufferAttribute, StorageBufferNode } from 'three/webgpu';
import type { BufferGeometry } from 'three';
import type { ScatterLayer, ScatterResult } from '../gpu/passes/Scatter';
import { VegClass } from '../gpu/passes/Scatter';
import type { VegLib, PoolPart } from '../vegetation/VegLibrary';
import type { Heightfield } from '../world/Heightfield';
import { WORLD_SIZE } from '../world/WorldConst';
import {
  type BuildReport,
  type ExplicitSource,
  GeometryRegistry,
  type MaterialClassId,
  type MeshHandle,
  type TransformChannel,
} from './GeometryRegistry';
import { readBuffer } from './Tsl';

/** Forests ring radii (Forests.ts) — discrete LOD switch distances until N8 */
const R0_FAR = 26;
const R1_FAR = 150;
const EX_R1_FAR = 120;
/** terrain window size: 7 quads → 98 tris, divides 4095 exactly (4096² field) */
const TERRAIN_WIN_QUADS = 7;

const TREE_MAX_CLS = 5;
const SHRUB_CLASSES: ReadonlySet<number> = new Set([
  VegClass.BushHazel,
  VegClass.BushPink,
  VegClass.Juniper,
]);
const DEADWOOD_CLASSES: ReadonlySet<number> = new Set([
  VegClass.Log,
  VegClass.Stump,
  VegClass.Branch,
]);
const ROCK_CLASSES: ReadonlySet<number> = new Set([
  VegClass.Boulder,
  VegClass.Slab,
  VegClass.StoneL,
  VegClass.StoneM,
  VegClass.StoneS,
]);

export interface WorldRegistryResult {
  registry: GeometryRegistry;
  report: BuildReport;
  /** ms: scatter readback, CPU partition, terrain minMax scan, registry build */
  readbackMs: number;
  partitionMs: number;
  terrainMs: number;
  buildMs: number;
  totalMs: number;
  /** instance count whose class is deferred (cards/leafy) */
  deferredInstances: number;
  /** per-part notes of geometry NOT migrated yet */
  deferred: string[];
}

function attrOf(node: StorageBufferNode<'vec4'>): StorageBufferAttribute {
  return (node as unknown as { value: StorageBufferAttribute }).value;
}

/** BufferGeometry → packed ExplicitSource (vdata vec4 → 4×u8 word) */
export function geometryToSource(geo: BufferGeometry): ExplicitSource {
  const pos = geo.attributes.position;
  if (!pos || pos.itemSize !== 3) throw new Error('WorldRegistry: geometry lacks stride-3 positions');
  if (!geo.attributes.normal) geo.computeVertexNormals();
  const nrm = geo.attributes.normal;
  if (!nrm) throw new Error('WorldRegistry: normals missing after compute');
  const positions = pos.array as Float32Array;
  const normals = nrm.array as Float32Array;
  const vCount = pos.count;
  const uvAttr = geo.attributes.uv;
  let uvs: Float32Array | undefined;
  if (uvAttr && uvAttr.itemSize === 2) uvs = uvAttr.array as Float32Array;
  const datAttr = geo.attributes.vdata;
  let vdata: Uint32Array | undefined;
  if (datAttr && datAttr.itemSize === 4) {
    const d = datAttr.array as Float32Array;
    vdata = new Uint32Array(vCount);
    for (let v = 0; v < vCount; v++) {
      const q = (i: number): number =>
        Math.round(Math.max(0, Math.min(1, d[v * 4 + i] as number)) * 255);
      vdata[v] = (q(0) | (q(1) << 8) | (q(2) << 16) | (q(3) << 24)) >>> 0;
    }
  }
  const idx = geo.index;
  if (!idx) throw new Error('WorldRegistry: geometry not indexed');
  const indices =
    idx.array instanceof Uint32Array ? idx.array : new Uint32Array(idx.array as ArrayLike<number>);
  return { kind: 'mesh', positions, normals, uvs, vdata, indices };
}

function classPolicy(
  cls: number,
): { matClass: MaterialClassId; channel: TransformChannel; lodDist: number } | null {
  if (cls <= TREE_MAX_CLS) return { matClass: 'bark', channel: 'trunk', lodDist: R1_FAR };
  if (SHRUB_CLASSES.has(cls)) return { matClass: 'bark', channel: 'trunk', lodDist: 0 };
  if (DEADWOOD_CLASSES.has(cls)) return { matClass: 'deadwood', channel: 'rigid', lodDist: EX_R1_FAR };
  if (ROCK_CLASSES.has(cls)) return { matClass: 'rock', channel: 'rigid', lodDist: EX_R1_FAR };
  return null; // ferns/flowers — leafy, N9
}

async function readLayer(
  renderer: Renderer,
  layer: ScatterLayer,
): Promise<{ a: Float32Array; b: Float32Array; count: number }> {
  const n = layer.count;
  if (n === 0) return { a: new Float32Array(0), b: new Float32Array(0), count: 0 };
  const [ab, bb] = await Promise.all([
    readBuffer(renderer, attrOf(layer.bufA), 0, n * 16),
    readBuffer(renderer, attrOf(layer.bufB), 0, n * 16),
  ]);
  return { a: new Float32Array(ab), b: new Float32Array(bb), count: n };
}

export async function buildWorldRegistry(input: {
  renderer: Renderer;
  hf: Heightfield;
  scatter: ScatterResult;
  lib: VegLib;
  counters?: Record<string, number>;
}): Promise<WorldRegistryResult> {
  const { renderer, hf, scatter, lib, counters } = input;
  const t0 = performance.now();
  const deferred: string[] = [];

  // ---- scatter readback (placements are boot-static) ------------------------
  const layers = await Promise.all([
    readLayer(renderer, scatter.trees),
    readLayer(renderer, scatter.understory),
    readLayer(renderer, scatter.extras),
    readLayer(renderer, scatter.stones),
  ]);
  const tRead = performance.now();

  // ---- partition by idF = cls·8 + variant (order-preserving) ----------------
  const perId = new Map<number, { a: Float32Array; b: Float32Array; fill: number }>();
  const idCounts = new Map<number, number>();
  for (const layer of layers) {
    for (let i = 0; i < layer.count; i++) {
      const id = Math.round(layer.b[i * 4 + 3] as number);
      idCounts.set(id, (idCounts.get(id) ?? 0) + 1);
    }
  }
  for (const [id, n] of idCounts) {
    perId.set(id, { a: new Float32Array(n * 4), b: new Float32Array(n * 4), fill: 0 });
  }
  for (const layer of layers) {
    for (let i = 0; i < layer.count; i++) {
      const id = Math.round(layer.b[i * 4 + 3] as number);
      const s = perId.get(id);
      if (!s) continue;
      const d = s.fill * 4;
      s.a[d] = layer.a[i * 4] as number;
      s.a[d + 1] = layer.a[i * 4 + 1] as number;
      s.a[d + 2] = layer.a[i * 4 + 2] as number;
      s.a[d + 3] = layer.a[i * 4 + 3] as number;
      s.b[d] = layer.b[i * 4] as number;
      s.b[d + 1] = layer.b[i * 4 + 1] as number;
      s.b[d + 2] = layer.b[i * 4 + 2] as number;
      s.b[d + 3] = layer.b[i * 4 + 3] as number;
      s.fill++;
    }
  }
  const tPart = performance.now();

  // ---- register pools --------------------------------------------------------
  const reg = new GeometryRegistry();
  const heads = new Map<number, MeshHandle>(); // idF → chain head
  let deferredTris = 0;
  const notePart = (label: string, parts: PoolPart[] | null | undefined, from: number): void => {
    if (!parts) return;
    for (let p = from; p < parts.length; p++) {
      deferredTris += parts[p]?.tris ?? 0;
    }
    if (parts.length > from) deferred.push(`${label}: ${parts.length - from} card/leaf part(s)`);
  };

  for (const pool of lib.pools) {
    const policy = classPolicy(pool.cls);
    const idF = pool.cls * 8 + pool.variant;
    const label = `c${pool.cls}v${pool.variant}`;
    if (!policy) {
      notePart(label, pool.r1, 0);
      continue;
    }
    // opaque part = parts[0] by construction (bark/rock/deadwood); the rest
    // are foliage cards / mesh leaves (deferred N9)
    const isTree = pool.cls <= TREE_MAX_CLS;
    const rings: { part: PoolPart; switchAt: number }[] = [];
    if (isTree && pool.r0?.[0]) rings.push({ part: pool.r0[0], switchAt: R0_FAR });
    if (pool.r1?.[0]) rings.push({ part: pool.r1[0], switchAt: isTree ? R1_FAR : policy.lodDist });
    // branch r2 is a geometry clone (indirect-slot bookkeeping) — skip it
    if (pool.r2?.[0] && pool.cls !== VegClass.Branch) {
      rings.push({ part: pool.r2[0], switchAt: 0 });
    }
    notePart(`${label}/r0`, pool.r0, 1);
    notePart(`${label}/r1`, pool.r1, 1);
    notePart(`${label}/r2`, pool.r2, 1);
    if (rings.length === 0) continue;
    const first = rings[0] as { part: PoolPart; switchAt: number };
    const head = reg.registerMesh(geometryToSource(first.part.geo), policy.matClass, {
      transformChannel: policy.channel,
      castShadows: first.part.castShadow,
      label,
      // swayPad stays 0 until N2 sources real amplitudes from Wind.ts (F6)
    });
    for (let r = 1; r < rings.length; r++) {
      const prev = rings[r - 1] as { part: PoolPart; switchAt: number };
      const ring = rings[r] as { part: PoolPart; switchAt: number };
      reg.registerLod(head, geometryToSource(ring.part.geo), prev.switchAt);
    }
    heads.set(idF, head);
  }

  // bind partitioned instances to chain heads
  let deferredInstances = 0;
  for (const [id, s] of perId) {
    const head = heads.get(id);
    if (head === undefined) {
      deferredInstances += s.fill;
      continue;
    }
    reg.bindInstances(head, { a: s.a, b: s.b });
  }

  // ---- terrain: the REAL field as ONE heightfield source ---------------------
  const tTerr0 = performance.now();
  const heights = hf.cpuHeights;
  if (!heights) throw new Error('WorldRegistry: hf.cpuHeights missing (boot order)');
  const res = hf.res;
  const quads = res - 1;
  const cell = WORLD_SIZE / res;
  const origin = cell / 2 - WORLD_SIZE / 2; // vertex (0,0) = texel-center 0
  const w = TERRAIN_WIN_QUADS;
  const windows = Math.ceil(quads / w);
  const minMax = new Float32Array(windows * windows * 2);
  for (let gz = 0; gz < windows; gz++) {
    const z0 = gz * w;
    const z1 = Math.min(z0 + w, quads);
    for (let gx = 0; gx < windows; gx++) {
      const x0 = gx * w;
      const x1 = Math.min(x0 + w, quads);
      let mn = Infinity;
      let mx = -Infinity;
      for (let z = z0; z <= z1; z++) {
        const row = z * res;
        for (let x = x0; x <= x1; x++) {
          const h = heights[row + x] as number;
          if (h < mn) mn = h;
          if (h > mx) mx = h;
        }
      }
      const i = (gz * windows + gx) * 2;
      minMax[i] = mn;
      minMax[i + 1] = mx;
    }
  }
  const hTerrain = reg.registerMesh(
    {
      kind: 'heightfield',
      quadsX: quads,
      quadsZ: quads,
      winQuads: w,
      cellSize: cell,
      originX: origin,
      originZ: origin,
      minMax,
    },
    'terrain',
    { label: 'terrain' },
  );
  reg.bindInstances(hTerrain, {
    a: new Float32Array([0, 0, 0, 1]),
    b: new Float32Array([0, 0, 0, 0]),
  });
  const tTerr1 = performance.now();
  deferred.push('GroundRing grass/debris: clipmap-instanced — N6/N10 per audit');
  deferred.push(`card/leaf tris deferred to N9: ${deferredTris}`);

  // ---- build ------------------------------------------------------------------
  const report = reg.build(renderer, counters);
  const t1 = performance.now();
  return {
    registry: reg,
    report,
    readbackMs: tRead - t0,
    partitionMs: tPart - tRead,
    terrainMs: tTerr1 - tTerr0,
    buildMs: t1 - tTerr1,
    totalMs: t1 - t0,
    deferredInstances,
    deferred,
  };
}
