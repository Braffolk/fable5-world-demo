/**
 * ?scene=rasterspike — N0 feasibility spike (docs/NANITE-SPEC.md).
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
import { NodeMaterial, StorageBufferAttribute, type Renderer } from 'three/webgpu';
import { dot, max, normalize, normalWorld, storage, vec3, vec4 } from 'three/tsl';
import type { Rng } from '../core/Seed';
import type { NF, NV3 } from '../gpu/TSLTypes';
import { GeometryRegistry } from '../nanite/GeometryRegistry';
import { buildSpikeContent, TERRAIN_QUADS } from '../nanite/SpikeContent';
import { buildSpikeRaster } from '../nanite/SpikeRaster';
import { readBuffer } from '../nanite/Tsl';
import { buildRock } from '../vegetation/RockBuilder';
import type { WorldContext } from './Scenes';

/**
 * Registry GPU roundtrip: pack a real rock + heightfield records + CPU and
 * GPU instance streams, build, then (a) validateOnGpu — readVertex/readCluster
 * TSL decode vs the CPU mirrors, (b) read back the instance blob region the
 * copy kernel wrote and compare against the source arrays. Logs [regtest]
 * PASS/FAIL for the headless probe (tools/probe-registry-gpu.ts).
 */
async function runRegistryGpuTest(renderer: Renderer, rng: Rng): Promise<void> {
  const reg = new GeometryRegistry();
  const rock = buildRock('boulder', rng, 4);
  const pos = rock.geometry.attributes.position;
  const nrm = rock.geometry.attributes.normal;
  const dat = rock.geometry.attributes.vdata;
  const idx = rock.geometry.index;
  if (!pos || !nrm || !dat || !idx) throw new Error('regtest: rock attrs missing');
  const positions = pos.array as Float32Array;
  const vCount = positions.length / 3;
  const uvs = new Float32Array(vCount * 2);
  const vdata = new Uint32Array(vCount);
  const d = dat.array as Float32Array;
  for (let v = 0; v < vCount; v++) {
    uvs[v * 2] = (positions[v * 3] as number) * 0.13 + 0.5;
    uvs[v * 2 + 1] = (positions[v * 3 + 2] as number) * 0.13 + 0.5;
    const quant = (i: number): number =>
      Math.round(Math.max(0, Math.min(1, d[v * 4 + i] as number)) * 255);
    vdata[v] = (quant(0) | (quant(1) << 8) | (quant(2) << 16) | (quant(3) << 24)) >>> 0;
  }
  const h = reg.registerMesh(
    {
      kind: 'mesh',
      positions,
      normals: nrm.array as Float32Array,
      uvs,
      vdata,
      indices: new Uint32Array(idx.array as ArrayLike<number>),
    },
    'rock',
    { label: 'regtest-rock' },
  );
  const minMax = new Float32Array(32);
  for (let i = 0; i < 16; i++) {
    minMax[i * 2] = -1 - i * 0.25;
    minMax[i * 2 + 1] = 1 + i * 0.5;
  }
  reg.registerMesh(
    { kind: 'heightfield', quadsX: 32, quadsZ: 32, winQuads: 8, cellSize: 1, originX: 0, originZ: 0, minMax },
    'terrain',
    { label: 'regtest-hf' },
  );

  // 2 CPU instances + 5 GPU-copied instances on the same mesh
  const cpuA = new Float32Array([1, 2, 3, 1.5, 4, 5, 6, 0.8]);
  const cpuB = new Float32Array([0.1, 0.2, 0.3, 17, 0.4, 0.5, 0.6, 18]);
  reg.bindInstances(h, { a: cpuA, b: cpuB });
  const N = 5;
  const srcA = new Float32Array(N * 4);
  const srcB = new Float32Array(N * 4);
  for (let i = 0; i < N * 4; i++) {
    srcA[i] = i + 0.25;
    srcB[i] = 1000 - i * 0.5;
  }
  const attrA = new StorageBufferAttribute(srcA, 4);
  const attrB = new StorageBufferAttribute(srcB, 4);
  reg.bindInstances(h, {
    bufA: storage(attrA, 'vec4', N),
    bufB: storage(attrB, 'vec4', N),
    count: N,
  });

  reg.build(renderer);
  const v = await reg.validateOnGpu(renderer, 64);

  const attrs = reg.debug().attrs;
  const entry = reg.meshEntry(h);
  const gpuFirst = entry.instFirst + 2;
  const blob = new Float32Array(await readBuffer(renderer, attrs.instances, gpuFirst * 32, N * 32));
  let copyOk = true;
  for (let i = 0; i < N; i++) {
    for (let k = 0; k < 4; k++) {
      if (blob[i * 8 + k] !== (srcA[i * 4 + k] as number)) copyOk = false;
      if (blob[i * 8 + 4 + k] !== (srcB[i * 4 + k] as number)) copyOk = false;
    }
  }
  const ids = new Uint32Array(await readBuffer(renderer, attrs.instanceMesh, gpuFirst * 4, N * 4));
  let idsOk = true;
  for (let i = 0; i < N; i++) if (ids[i] !== h) idsOk = false;

  const pass = v.pass && copyOk && idsOk;
  // eslint-disable-next-line no-console
  console.log(
    `[regtest] ${pass ? 'PASS' : 'FAIL'} — decode: ${v.detail}; ` +
      `gpuCopy ${copyOk ? 'ok' : 'MISMATCH'}; instMesh ${idsOk ? 'ok' : 'MISMATCH'}`,
  );
}

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

  // &regtest=1 — GeometryRegistry GPU decode/copy roundtrip (N1-C3 validation)
  if (q.get('regtest') === '1') {
    await runRegistryGpuTest(engine.renderer, seed.rng('regtest'));
  }

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
