/** Measure hero (lod-0 mesh, dieted) build time + tri split per species. */
import { WorldSeed } from '../src/core/Seed';
import { TREE_SPECIES } from '../src/vegetation/Species';
import { buildTree } from '../src/vegetation/TreeBuilder';
import { HERO_DIETS } from '../src/vegetation/VegLibrary';

const seed = new WorldSeed(1);
let totalMs = 0;
let totalTris = 0;
for (const sp of TREE_SPECIES) {
  const t0 = performance.now();
  const t = buildTree(sp, seed.rng(`veg/${sp.id}/0`), {
    lod: 0,
    foliageMode: 'mesh',
    hero: HERO_DIETS[sp.id] ?? { meshAnchorTarget: 1200 },
  });
  const ms = performance.now() - t0;
  totalMs += ms;
  const bark = t.bark.index ? t.bark.index.count / 3 : 0;
  const mesh = t.foliageMesh?.index ? t.foliageMesh.index.count / 3 : 0;
  totalTris += bark + mesh;
  console.log(
    `${sp.id.padEnd(10)} bark ${String(bark).padStart(7)} + mesh ${String(mesh).padStart(7)} = ${String(bark + mesh).padStart(7)}  anchors ${t.stats.anchors}  ${ms.toFixed(0)}ms`,
  );
}
console.log(
  `total ${Math.round(totalTris)} tris across 6 species, build ${totalMs.toFixed(0)}ms (×4 variants ≈ ${(totalMs * 4).toFixed(0)}ms)`,
);
