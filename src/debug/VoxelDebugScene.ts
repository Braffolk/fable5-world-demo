/**
 * ?scene=voxdbg — THROWAWAY voxel-crown inspection scene (spec §11 Stage 1).
 *
 * This is a DELIBERATELY THROWAWAY debug view, NOT the production voxel raster.
 * The real depth-bucketed voxel-brick raster is Stage 2 (§6) — this scene exists
 * ONLY to JUDGE the offline voxelizer's output shape/quality and to print the
 * per-crown brick count + MB (§11 Stage-1 SHOT/ACCEPTANCE). It voxelizes ONE
 * conifer crown (spruce — the worst case for thin needles, §3.4 / KG-0b), then
 * renders the occupied bricks as instanced boxes at a FIXED distance beside the
 * real mesh crown so the two read side-by-side.
 *
 * It does NOT touch the registry, the nanite pipeline, or any perf harness. Boot
 * with:  ?scene=voxdbg   (optional ?voxgrid=N to sweep the detail knob;
 *        optional ?voxspecies=spruce|pine).
 */

import {
  BoxGeometry,
  Color,
  InstancedMesh,
  Matrix4,
  Mesh,
  PlaneGeometry,
  Quaternion,
  Vector3,
} from 'three';
import { MeshStandardNodeMaterial } from 'three/webgpu';
import { vec3 } from 'three/tsl';
import { PostStack } from '../render/PostStack';
import { setupSunShadows } from '../render/ShadowSetup';
import { foliageMaterial, updateSunUniforms } from '../render/VegMaterials';
import { SunSky } from '../sky/SunSky';
import { PINE, SPRUCE } from '../vegetation/Species';
import type { SpeciesParams } from '../vegetation/VegTypes';
import { buildTree } from '../vegetation/TreeBuilder';
import { geometryToSource } from '../nanite/WorldRegistry';
import { BRICK_WORDS } from '../nanite/VoxelBrick';
import {
  brickCenterLocal,
  brickWorldSize,
  DEFAULT_VOXEL_GRID_DIM,
  voxelizeCrown,
} from '../nanite/VoxelizeCrown';
import type { WorldContext } from './Scenes';

export async function buildVoxelDebugScene(ctx: WorldContext): Promise<void> {
  const { engine, params, seed } = ctx;
  const q = new URLSearchParams(window.location.search);

  // ---- sky + shadows ---------------------------------------------------------
  ctx.progress(0.05, 'voxdbg: sky');
  const sunSky = new SunSky(engine, params.timeOfDay);
  await sunSky.init(engine.renderer);
  updateSunUniforms(sunSky.sun);
  setupSunShadows(sunSky.sun, engine.camera, undefined, { maxFar: 200, lightMargin: 60 });

  // ---- neutral ground --------------------------------------------------------
  const groundMat = new MeshStandardNodeMaterial();
  groundMat.colorNode = vec3(0.1, 0.11, 0.09);
  groundMat.roughness = 0.95;
  const ground = new Mesh(new PlaneGeometry(400, 400), groundMat);
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  engine.scene.add(ground);

  // ---- pick the worst-case conifer (spruce default) --------------------------
  const speciesId = (q.get('voxspecies') ?? 'spruce').toLowerCase();
  const sp: SpeciesParams = speciesId === 'pine' ? PINE : SPRUCE;
  const gridDim = Number(q.get('voxgrid') ?? DEFAULT_VOXEL_GRID_DIM) || DEFAULT_VOXEL_GRID_DIM;

  ctx.progress(0.3, `voxdbg: growing ${sp.id} crown`);
  // FULL anchor density (§5.2) so thin needles are not pre-decimated; hybrid mode
  // produces the real mesh-leaf crown (foliageMesh).
  const built = buildTree(sp, seed.rng(`voxdbg/${sp.id}`), { foliageMode: 'hybrid' });
  if (!built.foliageMesh) {
    throw new Error(`voxdbg: ${sp.id} produced no foliage mesh to voxelize`);
  }

  // ---- reference: the REAL mesh crown on the LEFT ----------------------------
  const refMesh = new Mesh(built.foliageMesh, foliageMaterial({ color: sp.foliageColor }));
  refMesh.position.set(-7, 0, 0);
  refMesh.castShadow = true;
  refMesh.receiveShadow = true;
  engine.scene.add(refMesh);
  const refBark = new Mesh(
    built.bark,
    (() => {
      const m = new MeshStandardNodeMaterial();
      m.colorNode = vec3(0.18, 0.13, 0.09);
      m.roughness = 0.9;
      return m;
    })(),
  );
  refBark.position.copy(refMesh.position);
  refBark.castShadow = true;
  engine.scene.add(refBark);

  // ---- OFFLINE VOXELIZE the crown's foliage mesh -----------------------------
  ctx.progress(0.6, `voxdbg: voxelizing ${sp.id} @ grid ${gridDim}`);
  const src = geometryToSource(built.foliageMesh);
  const vox = voxelizeCrown(src, sp.foliageColor, gridDim);

  // ---- brick count + MB printout (the §11 Stage-1 SHOT) ----------------------
  const occBricks = vox.occupied.length;
  const totalBricks = vox.stats.totalBricks;
  // bytes if ONLY occupied bricks are uploaded (the real registry path appends only
  // non-empty bricks); also report the dense-grid count for context.
  const occBytes = occBricks * BRICK_WORDS * 4;
  const occMB = occBytes / (1024 * 1024);
  const line =
    `[voxdbg] ${sp.id} crown voxelized @ gridDim ${gridDim} ` +
    `(${vox.cellGrid.x}×${vox.cellGrid.y}×${vox.cellGrid.z} cells, ` +
    `${vox.brickGrid.x}×${vox.brickGrid.y}×${vox.brickGrid.z} bricks, cell ${(vox.cellSize * 100).toFixed(1)} cm):\n` +
    `         source tris ${vox.stats.triangles}, cells touched ${vox.stats.cellsTouched}\n` +
    `         OCCUPIED bricks ${occBricks} / ${totalBricks} grid  ⇒  ` +
    `${BRICK_WORDS} u32/brick × ${occBricks} = ${occBytes} B = ${occMB.toFixed(3)} MB\n` +
    `         mean brick density ${vox.stats.meanDensity.toFixed(3)}, ` +
    `voxelize ${vox.stats.voxelizeMs.toFixed(1)} ms (offline)`;
  // eslint-disable-next-line no-console
  console.log(line);
  engine.stats.counters['voxdbg.occBricks'] = occBricks;
  engine.stats.counters['voxdbg.totalBricks'] = totalBricks;
  engine.stats.counters['voxdbg.bytes'] = occBytes;
  engine.stats.counters['voxdbg.tris'] = vox.stats.triangles;

  // ---- THROWAWAY render: occupied bricks as instanced boxes on the RIGHT ------
  // box size scales with brick density so a low-density needle smear reads smaller;
  // tinted by the brick mean color. This is purely to EYEBALL the silhouette — it is
  // NOT the depth-bucketed raster (Stage 2).
  ctx.progress(0.85, 'voxdbg: building brick boxes');
  const brickSize = brickWorldSize(vox);
  const boxGeo = new BoxGeometry(1, 1, 1);
  const boxMat = new MeshStandardNodeMaterial();
  boxMat.roughness = 0.85;
  // per-instance color is auto-wired by three's InstanceNode when the InstancedMesh
  // carries an instanceColor attribute (setColorAt below) — no colorNode needed.
  const inst = new InstancedMesh(boxGeo, boxMat, occBricks);
  inst.castShadow = true;
  inst.receiveShadow = true;
  const m = new Matrix4();
  const qRot = new Quaternion();
  const pScale = new Vector3();
  const pPos = new Vector3();
  const col = new Color();
  const VOX_OFFSET = new Vector3(7, 0, 0); // place the voxel crown to the RIGHT
  for (let i = 0; i < occBricks; i++) {
    const bi = vox.occupied[i] as number;
    const brick = vox.bricks[bi];
    if (!brick) continue;
    const c = brickCenterLocal(vox, bi);
    // density → box fill fraction (cube-root so volume ∝ density), min floor so it's visible
    const fill = Math.max(0.25, Math.cbrt(Math.max(0.02, brick.density)));
    pPos.set(c[0] + VOX_OFFSET.x, c[1] + VOX_OFFSET.y, c[2] + VOX_OFFSET.z);
    pScale.setScalar(brickSize * fill);
    m.compose(pPos, qRot, pScale);
    inst.setMatrixAt(i, m);
    // gamma-up the linear tint so the boxes read like the lit foliage
    col.setRGB(
      Math.sqrt(brick.albedo[0]),
      Math.sqrt(brick.albedo[1]),
      Math.sqrt(brick.albedo[2]),
    );
    inst.setColorAt(i, col);
  }
  inst.instanceMatrix.needsUpdate = true;
  if (inst.instanceColor) inst.instanceColor.needsUpdate = true;
  engine.scene.add(inst);

  // ---- post + camera at a FIXED distance -------------------------------------
  ctx.progress(0.95, 'voxdbg: post');
  const post = new PostStack(engine, sunSky.atmosphere, params.timeOfDay, null);
  engine.post = post;
  ctx.hooks.setTimeOfDay = (t: number) => {
    void (async () => {
      await sunSky.setTimeOfDay(t);
      updateSunUniforms(sunSky.sun);
      post.setTimeOfDay(t);
    })();
  };

  if (params.cam === null) {
    const h = built.stats.height;
    // FIXED viewing distance so the two crowns frame side-by-side
    engine.camera.position.set(0, h * 0.55, 26);
    engine.camera.lookAt(new Vector3(0, h * 0.5, 0));
  }
  engine.onUpdate(() => {
    if (engine.camera.position.y < 0.5) engine.camera.position.y = 0.5;
  });

  ctx.progress(1, 'voxdbg ready');
}
