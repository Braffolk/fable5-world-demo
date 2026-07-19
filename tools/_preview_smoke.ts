/**
 * SMOKE FIXTURE for tools/veg-preview.ts — NOT a real bog plant.
 *
 * Proves the harness renders an existing veg mesh end-to-end. Exports the same
 * `buildPreview(rng): THREE.Object3D` contract the real bog-plant modules will,
 * by wrapping an existing understory shrub (buildShrub) into meshes + simple
 * materials — mirroring how VegLibrary.ts turns a shrub's {bark, crown} into
 * meshes with a bark material and a green two-sided leaf material.
 */

import { Color, DoubleSide, Group, Mesh, MeshStandardMaterial, Object3D } from 'three';
import type { Rng } from '../src/core/Seed';
import { buildShrub, UNDERSTORY_SPECIES } from '../src/vegetation/Understory';
import type { SpeciesParams } from '../src/vegetation/VegTypes';

export function buildPreview(rng: Rng): Object3D {
  const sp = UNDERSTORY_SPECIES[1] as SpeciesParams; // pink flowering shrub
  const shrub = buildShrub(sp, rng);

  const g = new Group();

  const barkMat = new MeshStandardMaterial({ color: 0x5a4433, roughness: 0.9, metalness: 0 });
  g.add(new Mesh(shrub.bark, barkMat));

  if (shrub.crown) {
    const fc = sp.foliageColor;
    // foliageColor is a dim linear tint; brighten it a little so the smoke leaf reads green
    const leafColor = new Color(Math.min(1, fc.r * 3), Math.min(1, fc.g * 3), Math.min(1, fc.b * 3));
    const leafMat = new MeshStandardMaterial({
      color: leafColor,
      roughness: 0.7,
      metalness: 0,
      side: DoubleSide,
    });
    g.add(new Mesh(shrub.crown, leafMat));
  }

  return g;
}
