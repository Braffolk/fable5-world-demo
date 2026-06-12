/**
 * ?scene=rasterspike — N0 feasibility spike (docs/NANITE.md).
 *
 * Same content two ways:
 *   &sw=1 (default) — compute cull → work queue → SW raster (Option C two-pass
 *          vis buffer) → fullscreen resolve; big/near tris via the HW queue
 *          writing the SAME vis buffer from fragment stage.
 *   &sw=0 — plain hardware reference: 3 instanced rock draws + 1 terrain mesh,
 *          same flat shading, for the ≥2× r.scene GO/NO-GO comparison.
 *
 * &packing=a|c selects the vis-buffer scheme (c = Option C default).
 * &clusterdbg=1 mixes cluster hash colors into the resolve.
 *
 * Deliberately NO post stack / shadows / sky: the spike measures raster +
 * cull + resolve cost, not shading. Flat lambert with a fixed light.
 */

import {
  Color,
  InstancedMesh,
  Matrix4,
  Mesh,
  Quaternion,
  Vector2,
  Vector3,
} from 'three';
import { BufferAttribute, BufferGeometry } from 'three';
import { NodeMaterial } from 'three/webgpu';
import { dot, max, normalize, normalWorld, vec3, vec4 } from 'three/tsl';
import type { NF, NV3 } from '../gpu/TSLTypes';
import { buildSpikeContent, TERRAIN_QUADS } from '../nanite/SpikeContent';
import { buildSpikeRaster } from '../nanite/SpikeRaster';
import type { WorldContext } from './Scenes';

function flatMaterial(albedo: [number, number, number]): NodeMaterial {
  const mat = new NodeMaterial();
  const L = normalize(vec3(0.55, 0.8, 0.25)) as unknown as NV3;
  const lambert = max(dot(normalWorld as unknown as NV3, L), 0).mul(0.85).add(0.18) as unknown as NF;
  mat.fragmentNode = vec4(
    vec3(...albedo).mul(lambert),
    1,
  ) as unknown as typeof mat.fragmentNode;
  mat.fog = false;
  mat.lights = false;
  return mat;
}

export async function buildRasterSpikeScene(ctx: WorldContext): Promise<void> {
  const { engine, seed, hooks } = ctx;
  const q = new URLSearchParams(window.location.search);
  const sw = q.get('sw') !== '0';
  const packing = q.get('packing') === 'a' ? 'a' : 'c';
  const clusterTint = q.get('clusterdbg') === '1';

  ctx.progress(0.1, 'spike: content');
  const content = buildSpikeContent(seed.rng('rasterspike'));
  engine.scene.background = new Color(0.035, 0.055, 0.095);

  // eslint-disable-next-line no-console
  console.log(
    `[laas] rasterspike sw=${sw ? 1 : 0} packing=${packing} clusters=${content.clusterCount} ` +
      `srcTris=${content.stats.totalSourceTris} instancedTris=${content.stats.instancedTris} ` +
      `instances=${content.instanceCount}`,
  );
  engine.stats.counters['spike.clusters'] = content.clusterCount;
  engine.stats.counters['spike.instTrisK'] = Math.round(content.stats.instancedTris / 1000);

  if (sw) {
    ctx.progress(0.5, 'spike: SW raster pipeline');
    const size = new Vector2();
    engine.renderer.getDrawingBufferSize(size);
    const raster = buildSpikeRaster(content, size.x, size.y, packing, clusterTint);
    engine.scene.add(raster.resolveMesh);
    let frame = 0;
    let reading = false;
    engine.onUpdate(() => {
      raster.update(engine.renderer, engine.camera);
      if (frame++ % 20 === 0 && !reading) {
        reading = true;
        void raster.readCounts(engine.renderer).then((n) => {
          engine.stats.counters['spike.work'] = n.work;
          engine.stats.counters['spike.hwTris'] = n.hw;
          reading = false;
        });
      }
    });
  } else {
    ctx.progress(0.5, 'spike: HW reference meshes');
    // rocks: one InstancedMesh per variant
    const perMesh: number[][] = [[], [], []];
    for (let i = 0; i < content.instanceCount - 1; i++) {
      perMesh[content.instB[i * 4 + 3] as number]?.push(i);
    }
    const rockMat = flatMaterial([0.42, 0.41, 0.4]);
    for (let m = 0; m < 3; m++) {
      const list = perMesh[m] as number[];
      const [clusterStart, clusterCount] = [
        content.meshTable[m * 2] as number,
        content.meshTable[m * 2 + 1] as number,
      ];
      // rebuild this variant's geometry window from the mega-buffers
      const triStart = content.clusterMeta[clusterStart * 4 + 1] as number;
      const lastMeta = (clusterStart + clusterCount - 1) * 4;
      const triEnd = (content.clusterMeta[lastMeta + 1] as number) + (content.clusterMeta[lastMeta + 2] as number);
      const geo = new BufferGeometry();
      geo.setAttribute('position', new BufferAttribute(content.positions, 4));
      geo.setIndex(new BufferAttribute(content.indices.slice(triStart * 3, triEnd * 3), 1));
      geo.computeVertexNormals();
      const inst = new InstancedMesh(geo, rockMat, list.length);
      const mtx = new Matrix4();
      const quat = new Quaternion();
      const up = new Vector3(0, 1, 0);
      for (let k = 0; k < list.length; k++) {
        const i = list[k] as number;
        quat.setFromAxisAngle(up, -(content.instB[i * 4] as number));
        const s = content.instA[i * 4 + 3] as number;
        mtx.compose(
          new Vector3(content.instA[i * 4], content.instA[i * 4 + 1], content.instA[i * 4 + 2]),
          quat,
          new Vector3(s, s, s),
        );
        inst.setMatrixAt(k, mtx);
      }
      inst.instanceMatrix.needsUpdate = true;
      inst.frustumCulled = false;
      engine.scene.add(inst);
    }
    // terrain mesh from the same heights
    const n = TERRAIN_QUADS + 1;
    const pos = new Float32Array(n * n * 3);
    for (let z = 0; z < n; z++) {
      for (let x = 0; x < n; x++) {
        const i = z * n + x;
        pos[i * 3 + 0] = content.tileOrigin.x + x;
        pos[i * 3 + 1] = content.heights[i] as number;
        pos[i * 3 + 2] = content.tileOrigin.z + z;
      }
    }
    const idx = new Uint32Array(TERRAIN_QUADS * TERRAIN_QUADS * 6);
    let t = 0;
    for (let z = 0; z < TERRAIN_QUADS; z++) {
      for (let x = 0; x < TERRAIN_QUADS; x++) {
        const v0 = z * n + x;
        // matches the kernel's implicit winding (up-facing CCW):
        // even (0,0)(0,1)(1,1) / odd (0,0)(1,1)(1,0)
        idx[t++] = v0;
        idx[t++] = v0 + n;
        idx[t++] = v0 + n + 1;
        idx[t++] = v0;
        idx[t++] = v0 + n + 1;
        idx[t++] = v0 + 1;
      }
    }
    const tg = new BufferGeometry();
    tg.setAttribute('position', new BufferAttribute(pos, 3));
    tg.setIndex(new BufferAttribute(idx, 1));
    tg.computeVertexNormals();
    const terrain = new Mesh(tg, flatMaterial([0.3, 0.36, 0.22]));
    terrain.frustumCulled = false;
    engine.scene.add(terrain);
  }

  // standing pose at the tile edge, looking across the rock field
  const px = 0;
  const pz = 102;
  hooks.initialPose = {
    p: [px, content.groundY(px, Math.min(pz, 127)) + 1.8, pz],
    yaw: 0,
    pitch: -0.04,
  };
  ctx.progress(1, 'spike: ready');
}
