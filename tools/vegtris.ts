/** Print per-species per-LOD triangle counts for the veg pools (CPU-only). */
import { WorldSeed } from '../src/core/Seed';
import { TREE_SPECIES } from '../src/vegetation/Species';
import { buildTree } from '../src/vegetation/TreeBuilder';

const seed = new WorldSeed(1);
let totalR1 = 0;
let totalR2 = 0;
for (const sp of TREE_SPECIES) {
  const t1 = buildTree(sp, seed.rng(`veg/${sp.id}/0`), { lod: 1 });
  const t2 = buildTree(sp, seed.rng(`veg/${sp.id}/0`), { lod: 2 });
  const bark1 = t1.bark.index ? t1.bark.index.count / 3 : 0;
  const fol1 = t1.foliage?.index ? t1.foliage.index.count / 3 : 0;
  const bark2 = t2.bark.index ? t2.bark.index.count / 3 : 0;
  const fol2 = t2.foliage?.index ? t2.foliage.index.count / 3 : 0;
  totalR1 += bark1 + fol1;
  totalR2 += bark2 + fol2;
  console.log(
    `${sp.id.padEnd(10)} R1: bark ${bark1} + cards ${fol1} = ${bark1 + fol1}   R2: bark ${bark2} + cards ${fol2} = ${bark2 + fol2}  anchors ${t1.stats.anchors}`,
  );
}
console.log(`avg R1 ${Math.round(totalR1 / 6)}  avg R2 ${Math.round(totalR2 / 6)}`);
