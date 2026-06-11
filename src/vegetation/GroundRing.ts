/**
 * GroundRing — camera-following near-field carpets: GRASS (≥800k blades,
 * spec floor) and DEBRIS (≥80k: cobbles/pebbles/twigs/bark chips/litter).
 *
 * Streaming without uploads: each instance slot maps to the unique world
 * cell congruent to it (mod GRID) nearest the camera — the classic clipmap
 * wrap. All per-instance parameters re-derive from pcg(worldCell), so a
 * slot's content changes only when its world cell does. A per-frame cull
 * compute samples biome/water/canopy fields, thins density toward the ring
 * edge, frustum-tests, picks the LOD band, and appends (cell, groundY) into
 * compact lists → indirect draws.
 *
 * Grass LODs: 4-seg blade ≤26 m → 2-seg ≤60 m → wide tuft cross beyond
 * (dither-crossfaded). Debris types: cobbles/pebbles bias toward stream beds
 * (flowStrength — "water-rounded near streams"), litter/twigs/chips under
 * canopy, pebbles on rocky ground.
 */

import {
  BufferAttribute,
  BufferGeometry,
  DoubleSide,
  Frustum,
  Group,
  Matrix4,
  Mesh,
  Vector3,
  Vector4,
} from 'three';
import type { DataTexture, PerspectiveCamera } from 'three';
import {
  IndirectStorageBufferAttribute,
  MeshStandardNodeMaterial,
  StorageBufferAttribute,
  type Renderer,
  type StorageBufferNode,
  type StorageTexture,
} from 'three/webgpu';
import {
  Discard,
  Fn,
  If,
  Return,
  atomicAdd,
  atomicLoad,
  atomicStore,
  cameraPosition,
  float,
  instanceIndex,
  instancedArray,
  int,
  interleavedGradientNoise,
  mix,
  normalLocal,
  positionLocal,
  screenCoordinate,
  smoothstep,
  storage,
  texture,
  uint,
  uniform,
  uniformArray,
  uv,
  varying,
  vec2,
  vec3,
} from 'three/tsl';
import { canopyAt, cellHash, cellHash2 } from '../gpu/passes/Scatter';
import { grassTranslucency, rockMaterial } from '../render/VegMaterials';
import type { NF, NI, NU, NV2, NV3, NV4 } from '../gpu/TSLTypes';
import type { Heightfield } from '../world/Heightfield';
import { WORLD_SIZE } from '../world/WorldConst';
import {
  barkChipGeometry,
  debrisMaterial,
  grassBladeGeometry,
  litterMaterial,
  twigGeometry,
} from './GroundCover';
import { buildRock } from './RockBuilder';
import type { WorldSeed } from '../core/Seed';

const GRASS_GRID = 2560;
const GRASS_CELL = 0.103; // m → ±132 m ring, ~94 slots/m² (800k-blade floor)
const GRASS_R = 124;
const G_NEAR = 24;
const G_MID = 55;
const GRASS_CAPS = [327680, 655360, 786432]; // near/mid/far compact regions

const DEB_GRID = 448;
const DEB_CELL = 0.3; // ±67 m ring
const DEB_R = 62;
// cobble / pebble / twig / chip / litter
const DEB_CAPS = [16384, 32768, 24576, 16384, 32768];

interface RingBind {
  cells: StorageBufferNode<'uint'>;
  heights: StorageBufferNode<'float'>;
  base: number;
  cell: number;
  salt: number;
}

/** world cells span ±~10k — bias before the 16-bit pack */
const CELL_BIAS = 20000;

/** vertex-stage fetch: packed world cell + ground height for this instance */
function fetchRing(bind: RingBind): { wc: NV2; y: NF; wpos: NV2 } {
  const at = instanceIndex.add(uniform(uint(bind.base)) as unknown as NU);
  const packed = bind.cells.element(at) as unknown as NU;
  const wc = vec2(
    float(packed.shiftRight(uint(16))).sub(CELL_BIAS),
    float(packed.bitAnd(uint(0xffff))).sub(CELL_BIAS),
  );
  const y = bind.heights.element(at) as unknown as NF;
  const jit = cellHash2(wc, bind.salt);
  return { wc, y, wpos: wc.add(jit).mul(bind.cell) };
}

/** dithered band crossfade by camera distance */
function bandFade(
  mat: MeshStandardNodeMaterial,
  dist: NF,
  fadeIn: number | null,
  fadeOut: number | null,
  band: number,
): void {
  let f: NF = float(1);
  if (fadeIn !== null) f = f.mul(smoothstep(fadeIn - band, fadeIn + band, dist));
  if (fadeOut !== null)
    f = f.mul(float(1).sub(smoothstep(fadeOut - band, fadeOut + band, dist)));
  const fv = varying(f);
  const prev = mat.colorNode as unknown as NV3 | null;
  mat.colorNode = Fn(() => {
    Discard(fv.lessThanEqual(interleavedGradientNoise(screenCoordinate.xy)));
    return prev ?? vec3(1, 0, 1);
  })();
}

/** simple flat litter quad (5×7 cm), uv 0..1, normal up */
function litterQuad(): BufferGeometry {
  const g = new BufferGeometry();
  const w = 0.038;
  const l = 0.05;
  g.setAttribute(
    'position',
    new BufferAttribute(new Float32Array([-w, 0, -l, w, 0, -l, w, 0, l, -w, 0, l]), 3),
  );
  g.setAttribute(
    'normal',
    new BufferAttribute(new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]), 3),
  );
  g.setAttribute('uv', new BufferAttribute(new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]), 2));
  g.setIndex([0, 2, 1, 0, 3, 2]);
  return g;
}

/** two crossed wide blades — far-band tuft (≈ a small clump in one card) */
function tuftGeometry(): BufferGeometry {
  const pos: number[] = [];
  const nrm: number[] = [];
  const uvA: number[] = [];
  const idx: number[] = [];
  const W = 0.055;
  for (let k = 0; k < 2; k++) {
    const a = k * 1.92 + 0.4;
    const c = Math.cos(a);
    const s = Math.sin(a);
    const base = pos.length / 3;
    for (const [u, v] of [
      [-W, 0],
      [W, 0],
      [W * 0.55, 1],
      [-W * 0.55, 1],
    ] as const) {
      pos.push(u * c, v, u * s);
      nrm.push(-s * 0.97, 0.25, c * 0.97);
      uvA.push(u < 0 ? 0 : 1, v);
    }
    idx.push(base, base + 2, base + 1, base, base + 3, base + 2);
  }
  const g = new BufferGeometry();
  g.setAttribute('position', new BufferAttribute(new Float32Array(pos), 3));
  g.setAttribute('normal', new BufferAttribute(new Float32Array(nrm), 3));
  g.setAttribute('uv', new BufferAttribute(new Float32Array(uvA), 2));
  g.setIndex(idx);
  return g;
}

export class GroundRing {
  readonly group = new Group();
  private kernels: object[] = [];
  private camU = uniform(new Vector3());
  private planesU = uniformArray(Array.from({ length: 6 }, () => new Vector4()));
  private frustum = new Frustum();
  private projView = new Matrix4();
  private hud: Record<string, number> = {};
  private reading = false;
  private frame = 0;
  private counters!: ReturnType<StorageBufferNode<'uint'>['toAtomic']>;
  private caps: number[] = [...GRASS_CAPS, ...DEB_CAPS];

  constructor(
    private hf: Heightfield,
    private canopyTex: StorageTexture,
    private seed: WorldSeed,
  ) {}

  init(beechAtlas: DataTexture | null): void {
    const hf = this.hf;
    const salt = this.seed.sub('groundring') & 0x7fffffff;
    const camU = this.camU;
    const planesU = this.planesU;
    const canopyTex = this.canopyTex;

    const offsets: number[] = [];
    let off = 0;
    for (const cap of this.caps) {
      offsets.push(off);
      off += cap;
    }
    const cells = instancedArray(off, 'uint');
    const heights = instancedArray(off, 'float');
    this.counters = instancedArray(this.caps.length, 'uint').toAtomic();
    const counters = this.counters;
    const capBuf = storage(
      new StorageBufferAttribute(new Uint32Array(this.caps), 1),
      'uint',
      this.caps.length,
    );
    const offBuf = storage(
      new StorageBufferAttribute(new Uint32Array(offsets), 1),
      'uint',
      this.caps.length,
    );

    const clearK = Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(this.caps.length), () => {
        Return();
      });
      atomicStore(counters.element(i), uint(0));
    })().compute(this.caps.length);
    clearK.setName('ringClear');

    /** toroidal slot → nearest congruent world cell */
    const worldCell = (sx: NF, sy: NF, grid: number, cell: number): NV2 => {
      const camC = vec2(camU.x, camU.z).div(cell);
      const wx = camC.x.sub(sx).div(grid).round().mul(grid).add(sx);
      const wy = camC.y.sub(sy).div(grid).round().mul(grid).add(sy);
      return vec2(wx, wy);
    };

    const byBio = (b: NI, vals: number[]): NF => {
      let e: NF = float(vals[5] ?? 0);
      for (let i = 4; i >= 0; i--) {
        e = b.equal(float(i).toInt()).select(float(vals[i] ?? 0), e) as NF;
      }
      return e;
    };

    const inFrustum = (center: NV3, slack: number): NF => {
      let inside: NF = float(1);
      for (let p = 0; p < 6; p++) {
        const pl = planesU.element(float(p).toInt()) as unknown as NV4;
        inside = inside.mul(
          pl.xyz.dot(center).add(pl.w).greaterThan(-slack).select(float(1), float(0)),
        );
      }
      return inside;
    };

    const appendRing = (g: NI, wc: NV2, y: NF): void => {
      const idx = atomicAdd(counters.element(g), uint(1)) as unknown as NU;
      If(idx.lessThan(capBuf.element(g) as unknown as NU), () => {
        const at = (offBuf.element(g) as unknown as NU).add(idx);
        // pack biased cell coords 16|16 (cells span ±~10k)
        cells.element(at).assign(
          wc.x.add(CELL_BIAS).toUint().shiftLeft(uint(16)).bitOr(wc.y.add(CELL_BIAS).toUint()),
        );
        heights.element(at).assign(y);
      });
    };

    // ---------------- grass cull -----------------------------------------------
    const grassK = Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(GRASS_GRID * GRASS_GRID), () => {
        Return();
      });
      const sx = float(i.mod(GRASS_GRID));
      const sy = float(i.div(GRASS_GRID));
      const wc = worldCell(sx, sy, GRASS_GRID, GRASS_CELL);
      const jit = cellHash2(wc, salt);
      const wpos = wc.add(jit).mul(GRASS_CELL);
      const dist = wpos.sub(vec2(camU.x, camU.z)).length();
      If(dist.greaterThan(GRASS_R), () => {
        Return();
      });
      const uvW = wpos.div(WORLD_SIZE).add(0.5);
      const bio = texture(
        hf.biomeTex as NonNullable<typeof hf.biomeTex>,
        uvW,
        0,
      ) as unknown as NV4;
      const fl = texture(
        hf.fieldsTex as NonNullable<typeof hf.fieldsTex>,
        uvW,
        0,
      ) as unknown as NV4;
      const ns = texture(hf.normalTex, uvW, 0) as unknown as NV4;
      const bioId = bio.x.mul(8).add(0.5).floor().toInt();
      const h = hf.sampleHeight(wpos);
      If(fl.z.greaterThan(0.3).or(fl.w.sub(h).greaterThan(0.28)), () => {
        Return();
      });
      const canopy = canopyAt(canopyTex, wpos);
      // soft bank margin — a hard depth cut prints a razor arc along streams
      const bank = float(1).sub(smoothstep(0.08, 0.28, fl.z));
      const dens = byBio(bioId, [0.04, 0.5, 0.42, 0.5, 1.0, 0.85])
        .mul(bank)
        .mul(bio.z.mul(0.85).add(0.15))
        .mul(float(1).sub(bio.w.mul(0.7)))
        .mul(float(1).sub(bio.y.mul(0.95)))
        .mul(float(1).sub(canopy.mul(0.45)))
        .mul(float(1).sub(smoothstep(0.55, 0.95, ns.w)))
        .mul(fl.x.mul(0.35).add(0.75));
      const edge = float(1).sub(smoothstep(GRASS_R * 0.55, GRASS_R, dist));
      const lodK = dist.lessThan(G_MID).select(float(1), float(0.3));
      If(cellHash(wc, salt ^ 0x77a1).greaterThanEqual(dens.mul(edge).mul(lodK)), () => {
        Return();
      });
      If(inFrustum(vec3(wpos.x, h.add(0.5), wpos.y), 1.4).lessThan(0.5), () => {
        Return();
      });
      const lod = int(0).toVar();
      If(dist.greaterThanEqual(G_NEAR), () => {
        lod.assign(1);
      });
      If(dist.greaterThanEqual(G_MID), () => {
        lod.assign(2);
      });
      appendRing(lod as unknown as NI, wc, h);
    })().compute(GRASS_GRID * GRASS_GRID);
    grassK.setName('grassRingCull');

    // ---------------- debris cull ------------------------------------------------
    const debrisK = Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(DEB_GRID * DEB_GRID), () => {
        Return();
      });
      const sx = float(i.mod(DEB_GRID));
      const sy = float(i.div(DEB_GRID));
      const wc = worldCell(sx, sy, DEB_GRID, DEB_CELL);
      const jit = cellHash2(wc, salt ^ 0x5dd5);
      const wpos = wc.add(jit).mul(DEB_CELL);
      const dist = wpos.sub(vec2(camU.x, camU.z)).length();
      If(dist.greaterThan(DEB_R), () => {
        Return();
      });
      const uvW = wpos.div(WORLD_SIZE).add(0.5);
      const bio = texture(
        hf.biomeTex as NonNullable<typeof hf.biomeTex>,
        uvW,
        0,
      ) as unknown as NV4;
      const fl = texture(
        hf.fieldsTex as NonNullable<typeof hf.fieldsTex>,
        uvW,
        0,
      ) as unknown as NV4;
      const ns = texture(hf.normalTex, uvW, 0) as unknown as NV4;
      const bioId = bio.x.mul(8).add(0.5).floor().toInt();
      const h = hf.sampleHeight(wpos);
      If(fl.w.sub(h).greaterThan(0.4), () => {
        Return();
      });
      const canopy = canopyAt(canopyTex, wpos);
      const streamK = smoothstep(0.32, 0.7, fl.y).max(smoothstep(0.02, 0.2, fl.z));
      const wCobble = streamK.mul(2.2).add(bio.w.mul(0.3)).mul(0.5);
      const wPebble = bio.w.mul(0.9).add(streamK).add(0.15).mul(0.6);
      const wTwig = canopy.mul(1.4).add(0.1).mul(float(1).sub(streamK));
      const wChip = canopy.mul(0.8).mul(float(1).sub(streamK));
      const wLitter = canopy.mul(2.0).add(0.05).mul(float(1).sub(streamK.mul(0.8)));
      const wSum = wCobble.add(wPebble).add(wTwig).add(wChip).add(wLitter);
      // streambeds are FULLY cobbled geometry (spec §9) — override biome density
      const dens = byBio(bioId, [0.25, 0.45, 0.8, 0.85, 0.5, 0.6])
        .mul(float(1).sub(bio.y.mul(0.9)))
        .mul(wSum.mul(0.5).min(1))
        .max(streamK.mul(0.95))
        .mul(float(1).sub(smoothstep(0.7, 1.05, ns.w)));
      const edge = float(1).sub(smoothstep(DEB_R * 0.72, DEB_R, dist));
      If(cellHash(wc, salt ^ 0x132f).greaterThanEqual(dens.mul(edge)), () => {
        Return();
      });
      If(inFrustum(vec3(wpos.x, h.add(0.3), wpos.y), 0.8).lessThan(0.5), () => {
        Return();
      });
      const r = cellHash(wc, salt ^ 0x4c11).mul(wSum);
      const ty = float(0).toVar();
      const acc = wCobble.toVar();
      If(r.greaterThan(acc), () => {
        ty.assign(1);
        acc.addAssign(wPebble);
        If(r.greaterThan(acc), () => {
          ty.assign(2);
          acc.addAssign(wTwig);
          If(r.greaterThan(acc), () => {
            ty.assign(3);
            acc.addAssign(wChip);
            If(r.greaterThan(acc), () => {
              ty.assign(4);
            });
          });
        });
      });
      appendRing(ty.add(3).toInt(), wc, h);
    })().compute(DEB_GRID * DEB_GRID);
    debrisK.setName('debrisRingCull');

    // ---------------- draws -------------------------------------------------------
    const draws: { geo: BufferGeometry; mat: MeshStandardNodeMaterial; g: number }[] = [];

    const grassGeos = [grassBladeGeometry(4), grassBladeGeometry(2), tuftGeometry()];
    const grassFades: [number | null, number | null][] = [
      [null, G_NEAR],
      [G_NEAR, G_MID],
      [G_MID, null],
    ];
    for (let l = 0; l < 3; l++) {
      const bindL: RingBind = {
        cells,
        heights,
        base: offsets[l] ?? 0,
        cell: GRASS_CELL,
        salt,
      };
      const mat = this.grassMaterial(bindL, grassFades[l] ?? [null, null], l === 2);
      draws.push({ geo: grassGeos[l] as BufferGeometry, mat, g: l });
    }

    const rng = this.seed.rng('groundring/geo');
    const debrisGeos: BufferGeometry[] = [
      buildRock('cobble', rng.fork('cobble'), 2).geometry,
      buildRock('cobble', rng.fork('pebble'), 1).geometry,
      twigGeometry(rng.fork('twig')),
      barkChipGeometry(rng.fork('chip')),
      litterQuad(),
    ];
    const debrisScale = [0.16, 0.05, 1, 1, 1];
    for (let t = 0; t < 5; t++) {
      let mat: MeshStandardNodeMaterial;
      if (t === 4 && beechAtlas) mat = litterMaterial(beechAtlas);
      else if (t === 2) mat = debrisMaterial('twig');
      else if (t === 3) mat = debrisMaterial('chip');
      else mat = rockMaterial({ moss: t === 0 ? 0.18 : 0.05 });
      const bindD: RingBind = {
        cells,
        heights,
        base: offsets[3 + t] ?? 0,
        cell: DEB_CELL,
        salt: salt ^ 0x5dd5,
      };
      this.debrisTransform(mat, bindD, debrisScale[t] ?? 1);
      draws.push({ geo: debrisGeos[t] as BufferGeometry, mat, g: 3 + t });
    }

    const D = draws.length;
    const indirectData = new Uint32Array(D * 5);
    const drawGroups = new Uint32Array(D);
    const indirectAttr = new IndirectStorageBufferAttribute(indirectData, 5);
    for (let d = 0; d < D; d++) {
      const spec = draws[d];
      if (!spec) continue;
      const geo = spec.geo;
      indirectData[d * 5] = geo.index ? geo.index.count : geo.attributes.position?.count ?? 0;
      drawGroups[d] = spec.g;
      geo.setIndirect(indirectAttr, d * 20);
      const mesh = new Mesh(geo, spec.mat);
      mesh.frustumCulled = false;
      mesh.castShadow = false;
      mesh.receiveShadow = true;
      this.group.add(mesh);
    }
    const indirectStore = storage(indirectAttr, 'uint', D * 5);
    const drawGroupBuf = storage(new StorageBufferAttribute(drawGroups, 1), 'uint', D);

    const indirectK = Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(D), () => {
        Return();
      });
      const g = drawGroupBuf.element(i) as unknown as NU;
      const raw = atomicLoad(counters.element(g)) as unknown as NU;
      const cap = capBuf.element(g) as unknown as NU;
      indirectStore.element(i.mul(5).add(1)).assign(raw.greaterThan(cap).select(cap, raw));
    })().compute(D);
    indirectK.setName('ringIndirect');

    this.kernels = [clearK, grassK, debrisK, indirectK];
  }

  /** blade/tuft material — color matched to the terrain grass palette */
  private grassMaterial(
    bind: RingBind,
    fades: [number | null, number | null],
    tuft: boolean,
  ): MeshStandardNodeMaterial {
    const mat = new MeshStandardNodeMaterial();
    const { wc, y, wpos } = fetchRing(bind);
    const h2 = cellHash2(wc, bind.salt ^ 0x9191);
    // patch-level (≈1.6 m) dryness/hue so meadows read as drifts, not noise
    const patch = cellHash2(wc.mul(0.125).floor(), bind.salt ^ 0x3333);
    const tilt = cellHash2(wc, bind.salt ^ 0x4545).sub(0.5).mul(0.5);
    const bladeH = h2.x.pow(1.3).mul(0.3).add(0.2).mul(tuft ? 2.3 : 1);
    const yawA = h2.y.mul(6.2831853);
    const c = yawA.cos();
    const s = yawA.sin();
    const ls = positionLocal.mul(vec3(tuft ? 1.7 : 0.9, bladeH, 1));
    const rx = ls.x.mul(c).add(ls.z.mul(s));
    const rz = ls.z.mul(c).sub(ls.x.mul(s));
    // random lean (shear) — vertical uniform blades read as planted corn
    mat.positionNode = vec3(
      rx.add(tilt.x.mul(ls.y)).add(wpos.x),
      ls.y.add(y),
      rz.add(tilt.y.mul(ls.y)).add(wpos.y),
    );

    const t = uv().y as unknown as NF;
    const fresh = mix(
      vec3(0.022, 0.06, 0.012),
      vec3(0.072, 0.135, 0.03),
      t.mul(t),
    ) as unknown as NV3;
    const dry = mix(
      vec3(0.085, 0.07, 0.024),
      vec3(0.21, 0.17, 0.075),
      t,
    ) as unknown as NV3;
    const dryK = smoothstep(0.55, 0.85, patch.x);
    let albedo = mix(fresh, dry, dryK) as unknown as NV3;
    albedo = albedo.mul(patch.y.sub(0.5).mul(0.3).add(1)) as unknown as NV3;
    mat.colorNode = albedo;
    mat.emissiveNode = grassTranslucency(albedo, t);
    mat.aoNode = smoothstep(0.0, 0.55, t).mul(0.55).add(0.45);
    mat.roughness = 0.88;
    mat.metalness = 0;
    mat.side = DoubleSide;
    const dist = wpos.sub(vec2(cameraPosition.x, cameraPosition.z)).length();
    bandFade(mat, dist, fades[0], fades[1], fades[0] === null && fades[1] === G_NEAR ? 3 : 6);
    return mat;
  }

  /** cobbles/pebbles/twigs/chips/litter placement (yaw + scale + sink) */
  private debrisTransform(
    mat: MeshStandardNodeMaterial,
    bind: RingBind,
    scaleK: number,
  ): void {
    const { wc, y, wpos } = fetchRing(bind);
    const h2 = cellHash2(wc, bind.salt ^ 0x7777);
    const scl = h2.x.mul(0.9).add(0.55).mul(scaleK);
    const yawA = h2.y.mul(6.2831853);
    const c = yawA.cos();
    const s = yawA.sin();
    const ls = positionLocal.mul(scl);
    const rx = ls.x.mul(c).add(ls.z.mul(s));
    const rz = ls.z.mul(c).sub(ls.x.mul(s));
    const sink = scl.mul(0.22);
    mat.positionNode = Fn(() => {
      const n = vec3(
        normalLocal.x.mul(c).add(normalLocal.z.mul(s)),
        normalLocal.y,
        normalLocal.z.mul(c).sub(normalLocal.x.mul(s)),
      ).toVar();
      normalLocal.assign(n);
      return vec3(rx.add(wpos.x), ls.y.add(y).sub(sink), rz.add(wpos.y));
    })();
    const dist = wpos.sub(vec2(cameraPosition.x, cameraPosition.z)).length();
    bandFade(mat, dist, null, DEB_R - 6, 5);
  }

  update(renderer: Renderer, camera: PerspectiveCamera): void {
    this.camU.value.copy(camera.position);
    this.projView.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    this.frustum.setFromProjectionMatrix(this.projView);
    const arr = this.planesU.array as Vector4[];
    for (let p = 0; p < 6; p++) {
      const pl = this.frustum.planes[p];
      if (!pl) continue;
      (arr[p] as Vector4).set(pl.normal.x, pl.normal.y, pl.normal.z, pl.constant);
    }
    for (const k of this.kernels) {
      renderer.compute(k as Parameters<Renderer['compute']>[0]);
    }
    this.frame++;
    if (this.frame % 90 === 30 && !this.reading) {
      this.reading = true;
      void this.readStats(renderer);
    }
  }

  counterSnapshot(): Record<string, number> {
    return this.hud;
  }

  private async readStats(renderer: Renderer): Promise<void> {
    try {
      const attr = (this.counters as unknown as { value: unknown }).value;
      const ab = await renderer.getArrayBufferAsync(
        attr as Parameters<Renderer['getArrayBufferAsync']>[0],
      );
      const c = new Uint32Array(ab);
      const n = (g: number): number => Math.min(c[g] ?? 0, this.caps[g] ?? 0);
      this.hud = {
        'veg.grass': n(0) + n(1) + n(2),
        'veg.g0': n(0),
        'veg.g1': n(1),
        'veg.g2': n(2),
        'veg.debris': n(3) + n(4) + n(5) + n(6) + n(7),
      };
    } finally {
      this.reading = false;
    }
  }
}
