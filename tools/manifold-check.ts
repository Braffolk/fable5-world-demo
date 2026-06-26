/**
 * G1 MANIFOLD GATE (node-only, no browser).
 *
 * Welds each species' LOD0/1/2 bark at a 1e-5 position grid (exactly the seam
 * merge BuildDag's QEM relies on) and counts GLOBAL edges used by ≠2 triangles:
 *   open(use==1)        — an OPEN boundary loop  → QEM LOCKS it → far field stalls
 *   nonManifold(use>2)  — a >2-incident edge     → broken weld / inverted stitch
 *
 * The junction rework must drive BOTH to 0 (or ~0) for every species: that is
 * the literal precondition the proof identified for the DAG to collapse to ~1
 * root. A non-zero count prints a sample edge's welded position so an unwelded
 * junction is locatable.
 *
 *   npx tsx tools/manifold-check.ts
 */

import type { BufferGeometry } from 'three';
import { Rng } from '../src/core/Seed';
import { buildTree } from '../src/vegetation/TreeBuilder';
import { SPRUCE, PINE, BEECH, BIRCH, KARST_GNARL, SNAG } from '../src/vegetation/Species';
import type { SpeciesParams } from '../src/vegetation/VegTypes';

let failures = 0;

const Q = 1e5; // 1e-5 weld grid
function quant(x: number): number {
  return Math.round(x * Q);
}

interface Report {
  verts: number;
  weldedVerts: number;
  tris: number;
  open: number;
  nonManifold: number;
  sampleOpen?: [number, number, number];
  sampleNon?: [number, number, number];
}

/** weld positions at 1e-5, count global edge incidence */
function manifoldOf(geo: BufferGeometry): Report {
  const pos = geo.attributes.position;
  if (!pos) throw new Error('no position');
  const pa = pos.array as ArrayLike<number>;
  const vCount = pos.count;

  // weld: map quantized position → canonical vertex id
  const keyToId = new Map<string, number>();
  const weld = new Int32Array(vCount);
  for (let v = 0; v < vCount; v++) {
    const key = `${quant(pa[v * 3] as number)},${quant(pa[v * 3 + 1] as number)},${quant(pa[v * 3 + 2] as number)}`;
    let id = keyToId.get(key);
    if (id === undefined) {
      id = keyToId.size;
      keyToId.set(key, id);
    }
    weld[v] = id;
  }

  const idx = geo.index;
  const indices = idx
    ? (idx.array as ArrayLike<number>)
    : Array.from({ length: vCount }, (_, i) => i);
  const triCount = indices.length / 3;

  const edgeUse = new Map<number, number>();
  const NV = keyToId.size;
  const ek = (u: number, v: number): number => (u < v ? u * NV + v : v * NV + u);
  const addE = (u: number, v: number): void => {
    if (u === v) return; // degenerate after weld (ignore, not an open edge)
    const k = ek(u, v);
    edgeUse.set(k, (edgeUse.get(k) ?? 0) + 1);
  };
  for (let t = 0; t < triCount; t++) {
    const a = weld[indices[t * 3] as number] as number;
    const b = weld[indices[t * 3 + 1] as number] as number;
    const c = weld[indices[t * 3 + 2] as number] as number;
    addE(a, b);
    addE(b, c);
    addE(c, a);
  }

  let open = 0;
  let nonManifold = 0;
  let sampleOpen: [number, number, number] | undefined;
  let sampleNon: [number, number, number] | undefined;
  // reverse map id → a representative position (first welded vertex)
  const idPos = new Map<number, [number, number, number]>();
  for (let v = 0; v < vCount; v++) {
    const id = weld[v] as number;
    if (!idPos.has(id)) idPos.set(id, [pa[v * 3] as number, pa[v * 3 + 1] as number, pa[v * 3 + 2] as number]);
  }
  for (const [k, c] of edgeUse) {
    if (c === 2) continue;
    const u = Math.floor(k / NV);
    const mid = idPos.get(u);
    if (c === 1) {
      open++;
      if (!sampleOpen) sampleOpen = mid;
    } else {
      nonManifold++;
      if (!sampleNon) sampleNon = mid;
    }
  }

  return { verts: vCount, weldedVerts: NV, tris: triCount, open, nonManifold, sampleOpen, sampleNon };
}

function check(sp: SpeciesParams, lod: 0 | 1 | 2): void {
  const built = buildTree(sp, new Rng(77), { lod });
  const rep = manifoldOf(built.bark);
  const ok = rep.open === 0 && rep.nonManifold === 0;
  const tag = `${sp.id}-lod${lod}`;
  if (!ok) failures++;
  console.log(
    `  ${ok ? 'PASS' : 'FAIL'} ${tag.padEnd(16)} verts ${rep.verts} (welded ${rep.weldedVerts}) tris ${rep.tris} | ` +
      `open=${rep.open} nonManifold=${rep.nonManifold}` +
      (rep.sampleOpen ? `  openAt[${rep.sampleOpen.map((n) => n.toFixed(3)).join(',')}]` : '') +
      (rep.sampleNon ? `  nonAt[${rep.sampleNon.map((n) => n.toFixed(3)).join(',')}]` : ''),
  );
}

console.log('[manifold-check] G1 — welded edge incidence (open & non-manifold must be 0)');
const SPECIES: SpeciesParams[] = [SPRUCE, PINE, BEECH, BIRCH, KARST_GNARL, SNAG];
for (const sp of SPECIES) {
  check(sp, 0);
}
console.log('  -- LOD1/LOD2 (stride/maxLevel coherence) --');
for (const sp of SPECIES) {
  check(sp, 1);
  check(sp, 2);
}

if (failures > 0) {
  console.error(`[manifold-check] ${failures} FAILURES (open or non-manifold edges remain)`);
  process.exit(1);
}
console.log('[manifold-check] all species closed & manifold (0 open, 0 non-manifold)');
