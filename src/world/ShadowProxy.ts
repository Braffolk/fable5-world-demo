/**
 * ShadowProxy — coarse terrain stand-in for the shadow cascades.
 *
 * The CDLOD tiles run 2.8M+ triangles; letting them cast re-rasterizes that
 * into all four CSM cascades (~11M tri-passes — the "terrain 20M tris" debt).
 * Mountain/ridge shadows only need macro shape, so a static 512² grid (8 m
 * quads, heights from the height buffer in the vertex stage) casts instead:
 * colorWrite/depthWrite off make its main-pass cost vertex-only, while the
 * shadow pass swaps in its depth material as usual. Near-field terrain
 * self-shadow detail below 8 m is covered by the screen-space contact
 * shadows. The real terrain keeps castShadow = false.
 */

import { BufferAttribute, BufferGeometry, Mesh } from 'three';
import { MeshStandardNodeMaterial } from 'three/webgpu';
import { positionLocal, vec2, vec3 } from 'three/tsl';
import type { Heightfield } from './Heightfield';
import { WORLD_SIZE } from './WorldConst';

const GRID = 512;

export function buildTerrainShadowProxy(hf: Heightfield): Mesh {
  const n = GRID + 1;
  const pos = new Float32Array(n * n * 3);
  for (let z = 0; z < n; z++) {
    for (let x = 0; x < n; x++) {
      const i = (z * n + x) * 3;
      pos[i] = (x / GRID - 0.5) * WORLD_SIZE;
      pos[i + 1] = 0;
      pos[i + 2] = (z / GRID - 0.5) * WORLD_SIZE;
    }
  }
  const idx = new Uint32Array(GRID * GRID * 6);
  let w = 0;
  for (let z = 0; z < GRID; z++) {
    for (let x = 0; x < GRID; x++) {
      const a = z * n + x;
      idx[w++] = a;
      idx[w++] = a + n;
      idx[w++] = a + 1;
      idx[w++] = a + 1;
      idx[w++] = a + n;
      idx[w++] = a + n + 1;
    }
  }
  const geo = new BufferGeometry();
  geo.setAttribute('position', new BufferAttribute(pos, 3));
  geo.setIndex(new BufferAttribute(idx, 1));

  const mat = new MeshStandardNodeMaterial();
  const lifted = vec3(
    positionLocal.x,
    hf.sampleHeight(vec2(positionLocal.x, positionLocal.z)),
    positionLocal.z,
  );
  mat.positionNode = lifted;
  (mat as unknown as { castShadowPositionNode: unknown }).castShadowPositionNode = lifted;
  mat.colorWrite = false;
  mat.depthWrite = false;
  mat.depthTest = false;

  const mesh = new Mesh(geo, mat);
  mesh.frustumCulled = false;
  mesh.castShadow = true;
  mesh.receiveShadow = false;
  return mesh;
}
