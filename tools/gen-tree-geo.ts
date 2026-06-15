/**
 * gen-tree-geo.ts — serialize ONE leaf-crown and ONE bark geometry from the
 * project's TreeBuilder into JSON, for the geometry-vs-renderer experiment.
 *
 * The reference compute-rasterizer (reference/ref-tree.html) loads these JSON
 * blobs in place of the helmet GLTF, keeping its renderer (cull + per-instance
 * LOD + 129600 instances + single-pass SW raster) identical — so the ONLY
 * variable is the source geometry.
 *
 * We emit:
 *   - crown : the volumetric LEAF/NEEDLE crown (foliageMesh) — the flood geo
 *   - bark  : the trunk/branch tube manifold (CONTROL — should simplify cleanly)
 *
 * Both are tuned (meshAnchorTarget / barkK) so LOD0 ≈ 8-12k tris, so that the
 * reference's 5-LOD SimplifyModifier chain sums to ≤ 32768 tris (its bit cap).
 *
 * Usage: npx tsx tools/gen-tree-geo.ts
 * Writes: tools/geo/<name>.json  with { positions, normals, uvs, indices, tris, ... }
 */

import { mkdirSync, writeFileSync } from 'node:fs';
import type { BufferGeometry } from 'three';
import { Rng } from '../src/core/Seed';
import { SPRUCE } from '../src/vegetation/Species';
import { buildTree } from '../src/vegetation/TreeBuilder';

interface SerialGeo {
  name: string;
  positions: number[];
  normals: number[];
  uvs: number[];
  indices: number[];
  numVertices: number;
  tris: number;
  bbox: { min: number[]; max: number[] };
  boundingRadius: number;
}

function serialize(name: string, g: BufferGeometry): SerialGeo {
  const pos = g.attributes.position;
  const nrm = g.attributes.normal;
  const uv = g.attributes.uv;
  if (!pos || !nrm || !uv) throw new Error(`${name}: missing pos/normal/uv`);
  const positions = Array.from(pos.array as Float32Array);
  const normals = Array.from(nrm.array as Float32Array);
  const uvs = Array.from(uv.array as Float32Array);
  const idxAttr = g.index;
  const baseIdx = idxAttr
    ? Array.from(idxAttr.array as Uint16Array | Uint32Array)
    : Array.from({ length: pos.count }, (_, i) => i);
  // DOUBLE-SIDED: our leaves/needles are single-sided strips/quads, and the
  // reference's SW rasterizer backface-culls (areaNdc>0) — so half of every leaf
  // vanishes. Emit each triangle in BOTH windings so the crown renders solid from
  // any angle (this is the real geometry a foliage renderer must rasterize).
  const indices: number[] = [];
  for (let i = 0; i < baseIdx.length; i += 3) {
    const a = baseIdx[i] as number;
    const b = baseIdx[i + 1] as number;
    const c = baseIdx[i + 2] as number;
    indices.push(a, b, c, a, c, b);
  }

  // bbox + bounding radius about the geometry centroid origin (tree origin = 0,0,0)
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < pos.count; i++) {
    for (let a = 0; a < 3; a++) {
      const v = positions[i * 3 + a] as number;
      if (v < (min[a] as number)) min[a] = v;
      if (v > (max[a] as number)) max[a] = v;
    }
  }
  const cx = (min[0]! + max[0]!) / 2;
  const cy = (min[1]! + max[1]!) / 2;
  const cz = (min[2]! + max[2]!) / 2;
  let r2 = 0;
  for (let i = 0; i < pos.count; i++) {
    const dx = (positions[i * 3] as number) - cx;
    const dy = (positions[i * 3 + 1] as number) - cy;
    const dz = (positions[i * 3 + 2] as number) - cz;
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 > r2) r2 = d2;
  }
  return {
    name,
    positions,
    normals,
    uvs,
    indices,
    numVertices: pos.count,
    tris: indices.length / 3,
    bbox: { min, max },
    boundingRadius: Math.sqrt(r2),
  };
}

function write(geo: SerialGeo): void {
  const path = `tools/geo/${geo.name}.json`;
  writeFileSync(path, JSON.stringify(geo));
  console.log(
    `[gen] ${geo.name}: ${geo.tris} tris, ${geo.numVertices} verts, ` +
      `bboxY=[${geo.bbox.min[1]!.toFixed(2)}..${geo.bbox.max[1]!.toFixed(2)}], ` +
      `radius=${geo.boundingRadius.toFixed(2)} -> ${path}`,
  );
}

function main(): void {
  mkdirSync('tools/geo', { recursive: true });

  // Spruce, a canopy conifer. The crown is needle SPRAYS (disconnected single
  // quads) — exactly the volumetric flood geometry. We tune meshAnchorTarget
  // down so LOD0 lands ~8-12k tris.
  //
  // meshAnchorTarget strides the leaf anchors; each surviving anchor builds a
  // full ~30-needle spray (~30 quads = ~60 tris). The full forest uses 4000.
  const seed = 1234;

  // ---- bark (manifold CONTROL) ----
  // lod0 bark is the FULL tube hierarchy (~41k tris @ barkK 0.4) — too big for
  // the reference's 5-LOD sum cap (32768). lod1 drops the deepest tube levels
  // and yields a clean ~7k manifold trunk+branch geometry — the ideal control
  // (a connected manifold that SimplifyModifier should reduce well).
  const bark = serialize(
    'bark',
    buildTree(SPRUCE, new Rng(seed), { lod: 1, foliageMode: 'mesh' }).bark,
  );
  write(bark);

  // ---- crown (volumetric leaf/needle geometry) ----
  // tune meshAnchorTarget so the crown lands NEAR the bark control's tri count,
  // for the fairest matched-triangle leaves-vs-manifold comparison. Also emit a
  // ~11k variant ("crown_hi") so we have a second data point near the cap.
  let best: SerialGeo | null = null;
  for (const anchorTarget of [30, 38, 40, 44, 50, 62]) {
    const t = buildTree(SPRUCE, new Rng(seed), {
      lod: 0,
      foliageMode: 'mesh',
      hero: { meshAnchorTarget: anchorTarget, barkK: 0.8 },
    });
    if (!t.foliageMesh) throw new Error('no foliageMesh');
    const s = serialize(`crown_a${anchorTarget}`, t.foliageMesh);
    console.log(`  crown probe anchorTarget=${anchorTarget}: ${s.tris} tris`);
    // pick the closest to the bark control tri count (matched comparison)
    if (!best || Math.abs(s.tris - bark.tris) < Math.abs(best.tris - bark.tris)) best = s;
  }
  if (!best) throw new Error('no crown probe');
  best.name = 'crown';
  write(best);

  // a denser crown as a second leaves data point. Double-sided tris mean the
  // 5-LOD sum must stay < 32768 (the reference bit cap), so cap LOD0 ~17k.
  const crownHi = serialize(
    'crown_hi',
    buildTree(SPRUCE, new Rng(seed), {
      lod: 0,
      foliageMode: 'mesh',
      hero: { meshAnchorTarget: 45, barkK: 0.8 },
    }).foliageMesh!,
  );
  write(crownHi);

  // ---- tree (FULL tree: trunk + leaf crown merged into ONE geometry) ----
  // This is what renders as an actual forest (trunk + foliage together). The
  // reference's 5-LOD SimplifyModifier sum must stay < 32768 (double-sided tris),
  // so the combined LOD0 is kept ~14-16k: a lod1 bark (compact manifold trunk +
  // primary branches) + a needle crown tuned down to fit.
  // combined LOD0 must keep the 5-LOD sum < 32768 (≈ LOD0 × 1.76), so target
  // LOD0 ≈ 16k: a compact lod2 bark trunk (~5.7k doubled) + a ~24-anchor crown.
  // Disconnected leaf strips simplify poorly, so the LOD chain shrinks slowly —
  // keep combined LOD0 ≈ 12k so the 5-LOD sum clears the 32768 cap with margin.
  const treeCrown = buildTree(SPRUCE, new Rng(seed), {
    lod: 0,
    foliageMode: 'mesh',
    hero: { meshAnchorTarget: 18, barkK: 0.6 },
  }).foliageMesh!;
  const barkForTree = buildTree(SPRUCE, new Rng(seed), { lod: 2, foliageMode: 'mesh' }).bark;
  const tree = mergeSerial('tree', [barkForTree, treeCrown]);
  write(tree);

  console.log(
    `\n[gen] DONE. tree(full)=${tree.tris} tris, bark(control)=${bark.tris} tris, ` +
      `crown(matched)=${best.tris} tris, crown_hi=${crownHi.tris} tris.`,
  );
}

/** merge several geometries' pos/normal/uv/indices into one double-sided SerialGeo */
function mergeSerial(name: string, geos: BufferGeometry[]): SerialGeo {
  const merged = { positions: [] as number[], normals: [] as number[], uvs: [] as number[], indices: [] as number[] };
  let vbase = 0;
  for (const g of geos) {
    const s = serialize('_tmp', g); // serialize already doubles winding
    merged.positions.push(...s.positions);
    merged.normals.push(...s.normals);
    merged.uvs.push(...s.uvs);
    for (const idx of s.indices) merged.indices.push(idx + vbase);
    vbase += s.numVertices;
  }
  // recompute bbox/radius for the merged set
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  const nv = merged.positions.length / 3;
  for (let i = 0; i < nv; i++) {
    for (let a = 0; a < 3; a++) {
      const v = merged.positions[i * 3 + a] as number;
      if (v < (min[a] as number)) min[a] = v;
      if (v > (max[a] as number)) max[a] = v;
    }
  }
  const cx = (min[0]! + max[0]!) / 2, cy = (min[1]! + max[1]!) / 2, cz = (min[2]! + max[2]!) / 2;
  let r2 = 0;
  for (let i = 0; i < nv; i++) {
    const dx = (merged.positions[i * 3] as number) - cx;
    const dy = (merged.positions[i * 3 + 1] as number) - cy;
    const dz = (merged.positions[i * 3 + 2] as number) - cz;
    r2 = Math.max(r2, dx * dx + dy * dy + dz * dz);
  }
  return {
    name, positions: merged.positions, normals: merged.normals, uvs: merged.uvs, indices: merged.indices,
    numVertices: nv, tris: merged.indices.length / 3, bbox: { min, max }, boundingRadius: Math.sqrt(r2),
  };
}

main();
