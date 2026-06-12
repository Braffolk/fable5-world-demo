/**
 * `?nanitedbg=hwref` — flat-lit HARDWARE reference of the migrated registry
 * content for the N3(b) silhouette parity gate (F12): the SAME mesh/LOD/
 * instance set the nanite cull selects, drawn as classic three.js instanced
 * geometry with the SAME flat shading as the nanite flat resolve (matClass
 * palette × face-normal lambert). Diff target: ≤0.05%, no structural breaks.
 *
 * Faithfulness contract (mirrors the kernels, not the old pipeline):
 *  - LOD chain walk per instance = kInstCull's: dist(camPos, A.xyz), step
 *    while lodNext != NONE && dist > lodDist, envelope-drop at the tail
 *    (D-N14). Frustum/cone/occlusion culls are NOT mirrored — they only
 *    remove off-screen/hidden geometry, so the image is identical.
 *  - Instance transform = instTransformPoint exactly:
 *    T(A.xyz) · Shear(B.y, B.z) · RotY(B.x) · Scale(A.w).
 *  - Heightfield windows rebuilt from hf.cpuHeights with fetchWorldVert's
 *    corner tables (even (0,0)(0,1)(1,1), odd (0,0)(1,1)(1,0)); windows
 *    CPU-frustum-selected per pose (full L0 is 33.5M tris — never whole).
 *
 * The visibility set FREEZES at the first rendered frame (gate shots are
 * static poses); it is a debug view — it dies after the gate.
 */

import {
  Box3,
  Frustum,
  Matrix4,
  Mesh,
  Scene,
  Vector3,
} from 'three';
import {
  BufferGeometry,
  Float32BufferAttribute,
  InstancedMesh,
  Uint32BufferAttribute,
} from 'three';
import type { WebGPURenderer } from 'three/webgpu';
import { NodeMaterial } from 'three/webgpu';
import { Fn, cross, dFdx, dFdy, dot, max, normalize, positionWorld, vec3, vec4 } from 'three/tsl';
import type { NF, NV3 } from '../gpu/TSLTypes';
import type { Engine } from '../core/Engine';
import type { Heightfield } from '../world/Heightfield';
import {
  CLUSTER_WORDS,
  LOD_NONE,
  MESH_FLAG_HEIGHTFIELD,
  VERT_WORDS,
  bitsF32,
  decodeMeshCPU,
} from './GeometryRegistry';
import type { GeometryRegistry, MeshCPU } from './GeometryRegistry';
import type { NaniteViewHandles } from './NaniteView';

/** the nanite flat resolve's palette, verbatim */
const PALETTE: Record<number, [number, number, number]> = {
  0: [0.3, 0.36, 0.22],
  1: [0.42, 0.41, 0.4],
  2: [0.36, 0.27, 0.19],
  3: [0.33, 0.28, 0.22],
};
const FALLBACK: [number, number, number] = [0.35, 0.33, 0.3];

function flatMaterial(albedo: [number, number, number], shade: boolean): NodeMaterial {
  const mat = new NodeMaterial();
  // fragmentNode (NOT colorNode): raw output exactly like the nanite flat
  // resolve — no tone mapping / color-space transform asymmetry
  mat.fragmentNode = Fn(() => {
    if (!shade) {
      // ?shade=0 — the machine gate: derivative normals are garbage at
      // silhouette pixels and average across sub-pixel triangles (the
      // resolve fetches each pixel's EXACT triangle), so the gate compares
      // pure class color — coverage/structure only
      return vec4(vec3(albedo[0], albedo[1], albedo[2]), 1);
    }
    // human/lambert mode: face normal from world-position derivatives —
    // cross(dFdx, dFdy) matches the resolve's cross(w1-w0, w2-w0)
    // orientation for CCW front faces (the flipped order shades terrain
    // ambient-only)
    const faceN = normalize(
      cross(dFdx(positionWorld) as unknown as NV3, dFdy(positionWorld) as unknown as NV3),
    ) as unknown as NV3;
    const L = normalize(vec3(0.55, 0.8, 0.25)) as unknown as NV3;
    const lambert = max(dot(faceN, L), 0).mul(0.85).add(0.18) as unknown as NF;
    return vec4(vec3(albedo[0], albedo[1], albedo[2]).mul(lambert), 1);
  })() as unknown as typeof mat.fragmentNode;
  mat.fog = false;
  mat.lights = false;
  return mat;
}

export function buildNaniteHwRef(
  engine: Engine,
  registry: GeometryRegistry,
  hf: Heightfield,
): NaniteViewHandles {
  const dbg = registry.debug();
  const meshesArr = dbg.arrays.meshes;
  const instArr = dbg.arrays.instances;
  const instMeshArr = dbg.arrays.instanceMesh;
  const vertsArr = dbg.arrays.verts;
  const idxArr = dbg.arrays.indices;
  const clustersArr = dbg.arrays.clusters;
  const meshCount = registry.meshCount;
  const instanceCount = registry.instanceCount;

  const entries: MeshCPU[] = [];
  for (let m = 0; m < meshCount; m++) entries.push(decodeMeshCPU(meshesArr, m));

  // shared world-space-local position attribute (1.07M verts ≈ 12.8 MB)
  const vertCount = registry.bytes().verts / (VERT_WORDS * 4);
  const positions = new Float32Array(vertCount * 3);
  for (let v = 0; v < vertCount; v++) {
    const b = v * VERT_WORDS;
    positions[v * 3] = bitsF32(vertsArr[b] as number);
    positions[v * 3 + 1] = bitsF32(vertsArr[b + 1] as number);
    positions[v * 3 + 2] = bitsF32(vertsArr[b + 2] as number);
  }
  const posAttr = new Float32BufferAttribute(positions, 3);

  // per-entry indexed geometry (explicit meshes; indices are GLOBAL vert ids)
  const geos: (BufferGeometry | null)[] = [];
  for (let m = 0; m < meshCount; m++) {
    const e = entries[m] as MeshCPU;
    if ((e.flags & MESH_FLAG_HEIGHTFIELD) !== 0 || e.clusterCount === 0) {
      geos.push(null);
      continue;
    }
    let tris = 0;
    for (let c = 0; c < e.clusterCount; c++) {
      tris += (clustersArr[(e.clusterStart + c) * CLUSTER_WORDS + 7] as number) & 0xff;
    }
    const idx = new Uint32Array(tris * 3);
    let w = 0;
    for (let c = 0; c < e.clusterCount; c++) {
      const cb = (e.clusterStart + c) * CLUSTER_WORDS;
      const triStart = clustersArr[cb + 6] as number;
      const triCount = (clustersArr[cb + 7] as number) & 0xff;
      for (let k = 0; k < triCount * 3; k++) idx[w++] = idxArr[triStart * 3 + k] as number;
    }
    const g = new BufferGeometry();
    g.setAttribute('position', posAttr);
    g.setIndex(new Uint32BufferAttribute(idx, 1));
    geos.push(g);
  }

  const shade = new URLSearchParams(window.location.search).get('shade') !== '0';
  const materials = new Map<number, NodeMaterial>();
  const matFor = (matClass: number): NodeMaterial => {
    let m = materials.get(matClass);
    if (!m) {
      m = flatMaterial(PALETTE[matClass] ?? FALLBACK, shade);
      materials.set(matClass, m);
    }
    return m;
  };

  const scene = new Scene();
  // background stays null: the flat view's discarded pixels are TRANSPARENT
  // canvas compositing the page background (#06080a) — an opaque black
  // scene.background here read as 12.6k "structural" diff px at spawn
  scene.background = null;

  let refDraws = 0;
  let refTris = 0;
  /** meshes of the current build (geometries are shared except terrain's) */
  const built: Mesh[] = [];

  /** CPU mirror of kInstCull's chain walk + D-N14 envelope; -1 = culled */
  const selectLod = (headId: number, dist: number): number => {
    let id = headId;
    let e = entries[id] as MeshCPU;
    for (let s = 0; s < 4 && e.lodNext !== LOD_NONE && dist > e.lodDist; s++) {
      id = e.lodNext;
      e = entries[id] as MeshCPU;
    }
    if (e.lodNext === LOD_NONE && e.lodDist > 0 && dist > e.lodDist) return -1;
    return id;
  };

  const buildInstanced = (camPos: Vector3): void => {
    const buckets = new Map<number, number[]>();
    let culled = 0;
    for (let i = 0; i < instanceCount; i++) {
      const b = i * 8;
      const dx = (instArr[b] as number) - camPos.x;
      const dy = (instArr[b + 1] as number) - camPos.y;
      const dz = (instArr[b + 2] as number) - camPos.z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const sel = selectLod(instMeshArr[i] as number, dist);
      if (sel < 0) {
        culled++;
        continue;
      }
      const arr = buckets.get(sel);
      if (arr) arr.push(i);
      else buckets.set(sel, [i]);
    }

    const T = new Matrix4();
    const Sh = new Matrix4();
    const R = new Matrix4();
    const S = new Matrix4();
    const M = new Matrix4();
    for (const [entryId, ids] of buckets) {
      const geo = geos[entryId];
      const e = entries[entryId] as MeshCPU;
      if (!geo) continue; // heightfield buckets handled separately
      const im = new InstancedMesh(geo, matFor(e.matClass), ids.length);
      im.frustumCulled = false;
      for (let k = 0; k < ids.length; k++) {
        const b = (ids[k] as number) * 8;
        T.makeTranslation(instArr[b] as number, instArr[b + 1] as number, instArr[b + 2] as number);
        const lX = instArr[b + 5] as number;
        const lZ = instArr[b + 6] as number;
        Sh.set(1, lX, 0, 0, 0, 1, 0, 0, 0, lZ, 1, 0, 0, 0, 0, 1);
        R.makeRotationY(instArr[b + 4] as number);
        const s = instArr[b + 3] as number;
        S.makeScale(s, s, s);
        M.copy(T).multiply(Sh).multiply(R).multiply(S);
        im.setMatrixAt(k, M);
      }
      im.instanceMatrix.needsUpdate = true;
      scene.add(im);
      built.push(im);
      refDraws++;
      refTris += ((geo.index as Uint32BufferAttribute).count / 3) * ids.length;
    }
    // eslint-disable-next-line no-console
    console.log(
      `[nanite] hwref: ${refDraws} instanced draws, ${culled} envelope-culled instances`,
    );
  };

  const buildTerrain = (frustum: Frustum): void => {
    const heights = hf.cpuHeights;
    if (!heights) {
      // eslint-disable-next-line no-console
      console.warn('[nanite] hwref: hf.cpuHeights missing — terrain reference skipped');
      return;
    }
    const res = hf.res;
    const terrain = entries.find((e) => (e.flags & MESH_FLAG_HEIGHTFIELD) !== 0);
    if (!terrain) return;
    const wq = terrain.winQuads;
    const wxN = Math.ceil(terrain.quadsX / wq);
    const wzN = Math.ceil(terrain.quadsZ / wq);
    const box = new Box3();

    // pass 1: frustum-select windows on their exact world boxes
    const kept: number[] = [];
    let triTotal = 0;
    for (let wz = 0; wz < wzN; wz++) {
      for (let wx = 0; wx < wxN; wx++) {
        const gx = wx * wq;
        const gz = wz * wq;
        const qx = Math.min(wq, terrain.quadsX - gx);
        const qz = Math.min(wq, terrain.quadsZ - gz);
        let mn = Infinity;
        let mx = -Infinity;
        for (let z = gz; z <= gz + qz; z++) {
          const row = z * res;
          for (let x = gx; x <= gx + qx; x++) {
            const h = heights[row + x] as number;
            if (h < mn) mn = h;
            if (h > mx) mx = h;
          }
        }
        box.min.set(
          gx * terrain.hfCellSize + terrain.hfOriginX,
          mn,
          gz * terrain.hfCellSize + terrain.hfOriginZ,
        );
        box.max.set(
          (gx + qx) * terrain.hfCellSize + terrain.hfOriginX,
          mx,
          (gz + qz) * terrain.hfCellSize + terrain.hfOriginZ,
        );
        if (frustum.intersectsBox(box)) {
          kept.push(wx, wz);
          triTotal += qx * qz * 2;
        }
      }
    }

    // pass 2: merged geometry, fetchWorldVert's exact vertex math
    const vCount = kept.length / 2;
    const pos = new Float32Array(vCount * (wq + 1) * (wq + 1) * 3);
    const idx = new Uint32Array(triTotal * 3);
    let pv = 0;
    let pi = 0;
    for (let k = 0; k < kept.length; k += 2) {
      const gx = (kept[k] as number) * wq;
      const gz = (kept[k + 1] as number) * wq;
      const qx = Math.min(wq, terrain.quadsX - gx);
      const qz = Math.min(wq, terrain.quadsZ - gz);
      const base = pv / 3;
      for (let z = 0; z <= qz; z++) {
        const sz = gz + z;
        const row = sz * res;
        for (let x = 0; x <= qx; x++) {
          const sx = gx + x;
          pos[pv++] = sx * terrain.hfCellSize + terrain.hfOriginX;
          pos[pv++] = heights[row + sx] as number;
          pos[pv++] = sz * terrain.hfCellSize + terrain.hfOriginZ;
        }
      }
      const stride = qx + 1;
      for (let z = 0; z < qz; z++) {
        for (let x = 0; x < qx; x++) {
          const v00 = base + z * stride + x;
          const v01 = v00 + stride; // (x, z+1)
          const v11 = v01 + 1; // (x+1, z+1)
          const v10 = v00 + 1; // (x+1, z)
          // even (0,0)(0,1)(1,1) / odd (0,0)(1,1)(1,0) — up-facing CCW
          idx[pi++] = v00;
          idx[pi++] = v01;
          idx[pi++] = v11;
          idx[pi++] = v00;
          idx[pi++] = v11;
          idx[pi++] = v10;
        }
      }
    }
    const g = new BufferGeometry();
    g.setAttribute('position', new Float32BufferAttribute(pos.subarray(0, pv), 3));
    g.setIndex(new Uint32BufferAttribute(idx.subarray(0, pi), 1));
    const mesh = new Mesh(g, matFor(terrain.matClass));
    mesh.frustumCulled = false;
    scene.add(mesh);
    built.push(mesh);
    refDraws++;
    refTris += pi / 3;
    // eslint-disable-next-line no-console
    console.log(`[nanite] hwref: terrain ${vCount} windows in frustum, ${(pi / 3 / 1e6).toFixed(2)}M tris`);
  };

  const frustum = new Frustum();
  const pm = new Matrix4();
  const camPos = new Vector3();
  const lastBuildPos = new Vector3(Infinity, 0, 0);
  const lastFramePos = new Vector3(Infinity, 0, 0);
  let stable = false;

  const rebuild = (): void => {
    for (const m of built) {
      // geometries are shared (geos[]) except terrain's per-build one
      if (!(m instanceof InstancedMesh)) m.geometry.dispose();
      m.removeFromParent();
      if (m instanceof InstancedMesh) m.dispose();
    }
    built.length = 0;
    refDraws = 0;
    refTris = 0;
    const cam = engine.camera;
    cam.updateMatrixWorld();
    pm.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
    frustum.setFromProjectionMatrix(pm);
    const t0 = performance.now();
    buildInstanced(camPos);
    buildTerrain(frustum);
    // eslint-disable-next-line no-console
    console.log(
      `[nanite] hwref: built at pose (${(performance.now() - t0).toFixed(0)} ms, ${(refTris / 1e6).toFixed(1)}M tris, ${refDraws} draws)`,
    );
    lastBuildPos.copy(camPos);
  };

  const render = (): void => {
    // the LOD partition must match the kernel's SETTLED pose (walk-mode
    // spawn drops to the ground over the first frames): coarse rebuilds
    // while moving, one EXACT rebuild when the camera stops, then freeze
    if (!stable) {
      engine.camera.getWorldPosition(camPos);
      const stopped = camPos.distanceTo(lastFramePos) < 1e-4;
      lastFramePos.copy(camPos);
      if (stopped && camPos.distanceTo(lastBuildPos) > 0) rebuild();
      if (stopped && camPos.distanceTo(lastBuildPos) === 0) {
        stable = true;
        // eslint-disable-next-line no-console
        console.log('[nanite] hwref: pose settled — visibility frozen');
      } else if (!stopped && camPos.distanceTo(lastBuildPos) > 0.25) {
        rebuild();
      }
    }
    engine.renderer.render(scene, engine.camera);
  };

  const meter = (_r: WebGPURenderer): void => {
    engine.stats.counters['nanite.refDraws'] = refDraws;
    engine.stats.counters['nanite.refTrisK'] = Math.round(refTris / 1000);
  };

  return { render, meter };
}
