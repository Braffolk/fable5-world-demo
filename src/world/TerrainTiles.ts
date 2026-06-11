/**
 * Terrain rendering: CDLOD quadtree of instanced grid patches + far vista shell.
 *
 * - One InstancedMesh draws every active tile; per-tile data (origin, size,
 *   lod) lives in a CPU-writable instanced storage buffer, updated only when
 *   the quadtree changes (camera moved) — never per-frame per-instance.
 * - CDLOD vertex morphing: odd vertices slide toward their even-grid
 *   positions across the outer 35% of each LOD ring → no cracks, no pops.
 * - Far shell: radial ring 1.95–14 km, analytic macro height (far branch),
 *   blended to the baked field across the world edge.
 */

import { InstancedMesh, PlaneGeometry, RingGeometry, Mesh, type PerspectiveCamera } from 'three';
import { IrradianceNode, MeshStandardNodeMaterial, type StorageBufferNode } from 'three/webgpu';
import {
  cameraPosition,
  clamp,
  float,
  fract,
  instanceIndex,
  instancedArray,
  mix,
  positionLocal,
  positionWorld,
  screenUV,
  texture,
  transformNormalToView,
  varying,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import type { ProbeGI } from '../gpu/passes/ProbeGI';
import { buildTerrainShading } from '../render/TerrainMaterial';
import type { Heightfield } from './Heightfield';
import { macroTerrain } from './MacroMap';
import { FAR_RADIUS, WORLD_HALF, WORLD_SIZE } from './WorldConst';

const MAX_TILES = 2048;
const PATCH_SEGS = 64;
/** split while camDist < size·SPLIT_K */
const SPLIT_K = 2.1;
const MIN_TILE = 64;
/** rough/steep tiles may refine below MIN_TILE (cliff close-ups) */
const MIN_TILE_ROUGH = 32;

export class TerrainTiles {
  readonly mesh: InstancedMesh;
  readonly farShell: Mesh;
  private tileData: Float32Array;
  private tileBuf: StorageBufferNode<'vec4'>;
  private hf: Heightfield;
  private lastCamX = Infinity;
  private lastCamZ = Infinity;
  activeTiles = 0;
  /** per-level height ranges: level 0 = 64×64 grid of 64 m cells, then halves */
  private rangePyr: Float32Array[] = [];

  constructor(
    hf: Heightfield,
    debugView: string | null = null,
    opts: {
      heightBuf?: typeof hf.height;
      neutral?: boolean;
      screenHalf?: 'left' | 'right';
      gi?: ProbeGI;
    } = {},
  ) {
    this.hf = hf;
    this.buildRangePyramid();
    // ?ablate=mat → neutral clay (perf attribution for the splat material)
    const ablate = new Set(
      (new URLSearchParams(window.location.search).get('ablate') ?? '').split(','),
    );
    if (ablate.has('mat')) opts = { ...opts, neutral: true };
    // --- per-tile buffer -------------------------------------------------------
    this.tileData = new Float32Array(MAX_TILES * 4);
    this.tileBuf = instancedArray(this.tileData, 'vec4');
    const heightBuf = opts.heightBuf ?? hf.height;

    // --- patch geometry ----------------------------------------------------------
    // one extra quad ring beyond ±0.5 = skirt vertices: the shader clamps
    // them onto the edge then drops them down — hides cracks from the
    // error-biased (non-uniform) quadtree splits
    const s = 1 / PATCH_SEGS;
    const patch = new PlaneGeometry(1 + 2 * s, 1 + 2 * s, PATCH_SEGS + 2, PATCH_SEGS + 2);
    patch.rotateX(-Math.PI / 2); // local xz in [-0.5-s, 0.5+s], +y up

    // --- material ---------------------------------------------------------------
    const mat = new MeshStandardNodeMaterial();
    const tile = this.tileBuf.element(instanceIndex);
    const tileOrigin = tile.xy; // world xz of tile center
    const tileSize = tile.z;

    // CDLOD morph: world-space vertex, odd-vertex snap toward even grid.
    // Skirt verts (|local| > 0.5) clamp onto the edge, then drop down.
    const rawLocal = positionLocal.xz;
    const clampedLocal = clamp(rawLocal, -0.5, 0.5);
    const isSkirt = rawLocal
      .abs()
      .x.max(rawLocal.abs().y)
      .greaterThan(0.5001)
      .select(float(1), float(0));
    const local = clampedLocal.mul(tileSize);
    const wpos0 = local.add(tileOrigin).toVar();
    const quad = tileSize.div(PATCH_SEGS); // quad size in meters
    const gridUV = clampedLocal.add(0.5).mul(PATCH_SEGS); // 0..SEGS
    const odd = fract(gridUV.mul(0.5)).mul(2); // 1 where odd, 0 where even
    const snapped = wpos0.sub(odd.mul(quad)); // snap odd verts down-grid
    const camD = wpos0.sub(cameraPosition.xz).length();
    // morph across the outer band of this LOD's range
    const rangeEnd = tileSize.mul(SPLIT_K).mul(2); // parent split distance
    const morphK = clamp(camD.sub(rangeEnd.mul(0.7)).div(rangeEnd.mul(0.24)), 0, 1);
    const wpos = mix(wpos0, snapped, morphK);

    // instance + object matrices are identity → positionNode is world space
    const skirtDrop = isSkirt.mul(tileSize.mul(0.045).add(2.5));
    const hSample = hf.sampleHeightFrom(heightBuf, wpos).sub(skirtDrop);
    mat.positionNode = vec3(wpos.x, hSample, wpos.y);
    // shadow casting: skip the morph + bilinear (4 reads → 1); cascade texels
    // are meters wide, normalBias absorbs the nearest-fetch steps
    mat.castShadowPositionNode = vec3(
      wpos0.x,
      hf.sampleHeightNearest(wpos0).sub(skirtDrop),
      wpos0.y,
    );

    const shading = buildTerrainShading({
      normalTex: hf.normalTex,
      biomeTex: hf.biomeTex as NonNullable<typeof hf.biomeTex>,
      fieldsTex: hf.fieldsTex as NonNullable<typeof hf.fieldsTex>,
      noiseA: hf.noiseA as NonNullable<typeof hf.noiseA>,
      noiseB: hf.noiseB as NonNullable<typeof hf.noiseB>,
      mp: hf.mp,
      far: false,
    });
    mat.colorNode = shading.colorNode;
    mat.normalNode = shading.normalNode;
    mat.roughnessNode = shading.roughnessNode;
    mat.metalnessNode = float(0);
    if (opts.gi && !ablate.has('gi')) {
      // probe-GI irradiance replaces the hemisphere ambient (Phase 3) —
      // injected through the lighting context like a light map
      const irr = opts.gi.irradiance(positionWorld, shading.worldNormalNode);
      (mat as unknown as { setupLightMap: () => unknown }).setupLightMap = () =>
        new IrradianceNode(irr as unknown as ConstructorParameters<typeof IrradianceNode>[0]);
    }
    if (debugView === 'probes' && opts.gi) {
      // ambient-only view: probe irradiance × albedo, no sun/shadows
      mat.colorNode = vec3(0.0);
      mat.emissiveNode = opts.gi
        .irradiance(positionWorld, shading.worldNormalNode)
        .mul(shading.colorNode);
    }
    if (debugView === 'lod') {
      // distinct color per LOD level + faint grid along tile edges
      const lod = tile.w;
      const edge = positionLocal.xz.abs().x.max(positionLocal.xz.abs().y);
      const grid = edge.greaterThan(0.492).select(float(0.25), float(1));
      mat.colorNode = vec3(0.02);
      mat.emissiveNode = vec3(
        lod.mul(0.9173).add(0.13).fract(),
        lod.mul(0.3719).add(0.41).fract(),
        lod.mul(0.7177).add(0.79).fract(),
      ).mul(grid);
    }
    if ((debugView === 'snow' || debugView === 'bioR' || debugView === 'bioB') && hf.biomeTex) {
      // single-channel classification view: white = channel value
      const b = texture(hf.biomeTex, positionWorld.xz.div(WORLD_SIZE).add(0.5));
      mat.colorNode = vec3(0.02);
      const ch = debugView === 'bioR' ? b.r : debugView === 'bioB' ? b.b : b.g;
      mat.emissiveNode = vec3(ch);
    }
    if (opts.neutral) {
      // neutral clay shading for the erosion split view: fragment-space
      // finite-difference normals from the bound height buffer
      const eH = 1.6;
      const pxz = positionWorld.xz;
      const hC = hf.sampleHeightFrom(heightBuf, pxz);
      const hX = hf.sampleHeightFrom(heightBuf, pxz.add(vec2(eH, 0)));
      const hZ = hf.sampleHeightFrom(heightBuf, pxz.add(vec2(0, eH)));
      const nFD = vec3(hC.sub(hX), float(eH), hC.sub(hZ)).normalize();
      mat.colorNode = vec3(0.55, 0.53, 0.5);
      mat.normalNode = transformNormalToView(nFD);
      mat.roughnessNode = float(0.92);
    }
    if (opts.screenHalf) {
      // split-screen via alpha test: keep only one half of the screen
      const keepLeft = opts.screenHalf === 'left';
      const keep = keepLeft
        ? screenUV.x.lessThanEqual(0.5)
        : screenUV.x.greaterThan(0.5);
      mat.opacityNode = keep.select(float(1), float(0));
      mat.alphaTest = 0.5;
    }

    this.mesh = new InstancedMesh(patch, mat, MAX_TILES);
    this.mesh.frustumCulled = false;
    this.mesh.receiveShadow = true;
    // mountain shadows come from the coarse ShadowProxy grid — casting the
    // full CDLOD mesh re-rasterized ~11M tris across the four cascades
    this.mesh.castShadow = false;

    // --- far shell -----------------------------------------------------------------
    const ring = new RingGeometry(WORLD_HALF * 0.952, FAR_RADIUS, 160, 42);
    ring.rotateX(-Math.PI / 2);
    const farMat = new MeshStandardNodeMaterial();
    const fxz = positionLocal.xz;
    const farMacro = macroTerrain(fxz, hf.mp, 'far');
    const baked = hf.sampleHeight(fxz);
    const edgeBlend = clamp(
      fxz.abs().x.max(fxz.abs().y).sub(WORLD_HALF * 0.95).div(WORLD_HALF * 0.05),
      0,
      1,
    );
    // sit well below the tile mesh inside the world (coarse far tiles deviate
    // several meters — the shell poked through and showed far-mode shading)
    const farH = mix(baked, farMacro.height, edgeBlend).sub(
      mix(float(9), float(2.5), edgeBlend),
    );
    farMat.positionNode = vec3(fxz.x, farH, fxz.y);
    // analytic per-vertex normal (no baked maps beyond the world edge):
    // finite-difference the far macro height, interpolated via varying
    const eN = 60;
    const hX = macroTerrain(fxz.add(vec2(eN, 0)), hf.mp, 'far').height;
    const hZ = macroTerrain(fxz.add(vec2(0, eN)), hf.mp, 'far').height;
    const farNormal = vec3(farMacro.height.sub(hX), float(eN), farMacro.height.sub(hZ))
      .normalize();
    const farSlope = vec2(farMacro.height.sub(hX), farMacro.height.sub(hZ))
      .length()
      .div(eN);
    const farNS = varying(vec4(farNormal, farSlope));
    const farShading = buildTerrainShading({
      normalTex: hf.normalTex,
      biomeTex: hf.biomeTex as NonNullable<typeof hf.biomeTex>,
      fieldsTex: hf.fieldsTex as NonNullable<typeof hf.fieldsTex>,
      noiseA: hf.noiseA as NonNullable<typeof hf.noiseA>,
      noiseB: hf.noiseB as NonNullable<typeof hf.noiseB>,
      mp: hf.mp,
      far: true,
      baseNormalSlope: farNS,
    });
    farMat.colorNode = farShading.colorNode;
    farMat.normalNode = farShading.normalNode;
    farMat.roughnessNode = farShading.roughnessNode;
    farMat.metalnessNode = float(0);
    if (opts.gi && !ablate.has('gi')) {
      const farIrr = opts.gi.irradiance(positionWorld, farShading.worldNormalNode);
      (farMat as unknown as { setupLightMap: () => unknown }).setupLightMap = () =>
        new IrradianceNode(farIrr as unknown as ConstructorParameters<typeof IrradianceNode>[0]);
    }
    this.farShell = new Mesh(ring, farMat);
    this.farShell.frustumCulled = false;
    this.farShell.receiveShadow = true;
  }

  /**
   * Height-range mip pyramid from the CPU height mirror — drives error-biased
   * splits (steep/rough tiles refine deeper, flat meadows stay coarse).
   */
  private buildRangePyramid(): void {
    const heights = this.hf.cpuHeights;
    if (!heights) return;
    const res = Math.sqrt(heights.length) | 0;
    const base = 64; // cells per side; one cell = MIN_TILE meters
    const cellPx = res / base;
    const l0 = new Float32Array(base * base);
    for (let cy = 0; cy < base; cy++) {
      for (let cx = 0; cx < base; cx++) {
        let mn = Infinity;
        let mx = -Infinity;
        const x0 = cx * cellPx;
        const y0 = cy * cellPx;
        // 4-px stride: range estimate, not exact min/max (16× cheaper)
        for (let y = y0; y < y0 + cellPx; y += 4) {
          const row = y * res;
          for (let x = x0; x < x0 + cellPx; x += 4) {
            const v = heights[row + x] as number;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
          }
        }
        l0[cy * base + cx] = mx - mn;
      }
    }
    this.rangePyr = [l0];
    for (let side = base >> 1; side >= 1; side >>= 1) {
      const prev = this.rangePyr[this.rangePyr.length - 1] as Float32Array;
      const pSide = side * 2;
      const lvl = new Float32Array(side * side);
      for (let cy = 0; cy < side; cy++) {
        for (let cx = 0; cx < side; cx++) {
          lvl[cy * side + cx] = Math.max(
            prev[cy * 2 * pSide + cx * 2] as number,
            prev[cy * 2 * pSide + cx * 2 + 1] as number,
            prev[(cy * 2 + 1) * pSide + cx * 2] as number,
            prev[(cy * 2 + 1) * pSide + cx * 2 + 1] as number,
          );
        }
      }
      this.rangePyr.push(lvl);
    }
  }

  /** height range (m) within a tile (≥ MIN_TILE sizes use the exact level) */
  private heightRange(ox: number, oz: number, size: number): number {
    if (this.rangePyr.length === 0) return 0;
    const lvl = Math.max(0, Math.min(Math.round(Math.log2(Math.max(size, MIN_TILE) / MIN_TILE)), this.rangePyr.length - 1));
    const side = 64 >> lvl;
    const cell = WORLD_SIZE / side;
    const cx = Math.max(0, Math.min(Math.floor((ox + WORLD_SIZE / 2) / cell), side - 1));
    const cy = Math.max(0, Math.min(Math.floor((oz + WORLD_SIZE / 2) / cell), side - 1));
    return (this.rangePyr[lvl] as Float32Array)[cy * side + cx] as number;
  }

  /** rebuild the quadtree when the camera has moved enough */
  update(camera: PerspectiveCamera): void {
    const cx = camera.position.x;
    const cz = camera.position.z;
    if (Math.hypot(cx - this.lastCamX, cz - this.lastCamZ) < 20 && this.activeTiles > 0) return;
    this.lastCamX = cx;
    this.lastCamZ = cz;

    let n = 0;
    const data = this.tileData;
    const emit = (ox: number, oz: number, size: number, lod: number): void => {
      if (n >= MAX_TILES) return;
      data[n * 4] = ox;
      data[n * 4 + 1] = oz;
      data[n * 4 + 2] = size;
      data[n * 4 + 3] = lod;
      n++;
    };
    const cy = camera.position.y;
    const recurse = (ox: number, oz: number, size: number, lod: number): void => {
      const dx = Math.max(Math.abs(cx - ox) - size / 2, 0);
      const dz = Math.max(Math.abs(cz - oz) - size / 2, 0);
      // 3D distance: from high altitude the ground straight below does not
      // need MIN_TILE resolution (slack absorbs in-tile height spread)
      const groundY = this.hf.heightAtCpu(ox, oz);
      const dy = Math.max(Math.abs(cy - groundY) - 250, 0) * 0.8;
      const dist = Math.hypot(dx, dz, dy);
      // error bias: tiles with big internal relief split earlier AND deeper
      // (cliff close-ups got 1 m quads stretched over ~10 m vertical)
      const range = this.heightRange(ox, oz, size);
      const errBoost = Math.min(1 + (range / size) * 0.8, 1.8);
      const minTile = range > size * 0.85 ? MIN_TILE_ROUGH : MIN_TILE;
      if (size > minTile && dist < size * SPLIT_K * errBoost) {
        const q = size / 4;
        const h = size / 2;
        recurse(ox - q, oz - q, h, lod + 1);
        recurse(ox + q, oz - q, h, lod + 1);
        recurse(ox - q, oz + q, h, lod + 1);
        recurse(ox + q, oz + q, h, lod + 1);
      } else {
        emit(ox, oz, size, lod);
      }
    };
    recurse(0, 0, WORLD_SIZE, 0);

    this.activeTiles = n;
    this.mesh.count = n;
    const attr = this.tileBuf.value;
    attr.needsUpdate = true;
  }
}
