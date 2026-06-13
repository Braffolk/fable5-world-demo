/**
 * Bark texture synthesis (GPU compute, one-time): per-species tileable
 * 2048² maps from periodic worley plates + fissures + micro grain.
 *   texA = albedo.rgb (sqrt-encoded) + cavity AO (a)
 *   texB = tangent normal xy (0..1) + roughness + height
 * Species recipes: spruce fissures, pine plates, beech smooth, birch
 * paper+lenticels, karst gnarl ridges, snag weathered cracks.
 */

import { FloatType, LinearMipmapLinearFilter, RepeatWrapping, Vector2 } from 'three';
import { StorageArrayTexture, StorageTexture, type Renderer } from 'three/webgpu';
import {
  Fn,
  float,
  instanceIndex,
  int,
  ivec2,
  mix,
  textureStore,
  uint,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { sqrt } from 'three/tsl';
import { hash12 } from '../noise/NoiseTSL';
import type { NF, NV2, NV3, NV4 } from '../TSLTypes';

export const BARK_RES = 2048;

function sqrtV3(v: NV3): NV3 {
  return sqrt(v as unknown as NF) as unknown as NV3;
}

/** value noise on a wrapped integer lattice → tiles at `period` */
function pnoise(p: NV2, period: number, seedK: number): NF {
  const cell = p.floor();
  const f = p.fract();
  const u = f.mul(f).mul(f.negate().mul(2).add(3)); // smoothstep fade
  const wrap = (c: NV2): NV2 => c.sub(c.div(period).floor().mul(period));
  const h = (ox: number, oy: number): NF =>
    hash12(wrap(cell.add(vec2(ox, oy))).add(seedK * 17.17));
  const a = h(0, 0);
  const b = h(1, 0);
  const c = h(0, 1);
  const d = h(1, 1);
  return a
    .add(b.sub(a).mul(u.x))
    .add(c.sub(a).mul(u.y))
    .add(a.sub(b).sub(c).add(d).mul(u.x).mul(u.y));
}

function pfbm(p: NV2, octaves: number, period: number, seedK: number): NF {
  let sum: NF = float(0);
  let amp = 0.5;
  let scale = 1;
  for (let i = 0; i < octaves; i++) {
    sum = sum.add(pnoise(p.mul(scale), period * scale, seedK + i * 7).mul(amp));
    amp *= 0.5;
    scale *= 2;
  }
  return sum;
}

/** periodic worley: returns F1 and edge term (F2−F1), both tileable */
function pworley(p: NV2, period: Vector2, seedK: number): { f1: NF; edge: NF } {
  const cell = p.floor();
  const f = p.fract();
  const wrapX = (v: NF): NF => v.sub(v.div(period.x).floor().mul(period.x));
  const wrapY = (v: NF): NF => v.sub(v.div(period.y).floor().mul(period.y));
  const dists: NF[] = [];
  for (let oy = -1; oy <= 1; oy++) {
    for (let ox = -1; ox <= 1; ox++) {
      const cx = cell.x.add(ox);
      const cy = cell.y.add(oy);
      const hx = hash12(vec2(wrapX(cx), wrapY(cy)).add(seedK * 31.7));
      const hy = hash12(vec2(wrapX(cx), wrapY(cy)).add(seedK * 31.7 + 911.3));
      const feat = vec2(float(ox).add(hx), float(oy).add(hy));
      const d = feat.sub(f);
      dists.push(d.dot(d));
    }
  }
  let f1: NF = dists[0] as NF;
  for (let i = 1; i < 9; i++) f1 = f1.min(dists[i] as NF);
  let f2: NF = float(9);
  for (let i = 0; i < 9; i++) {
    const di = dists[i] as NF;
    f2 = f2.min(di.add(di.lessThanEqual(f1.add(1e-5)).select(float(10), float(0))));
  }
  const f1s = f1.sqrt();
  return { f1: f1s, edge: f2.sqrt().sub(f1s) };
}

export interface BarkParams {
  /** worley plate frequency (x = around trunk, y = along) */
  plates: [number, number];
  /** uv warp amount (breaks straight fissures) */
  warp: number;
  /** fissure profile: width of the dark crevice band */
  fissureW: number;
  fissureDepth: number;
  /** per-plate dome rounding */
  plateRound: number;
  micro: number;
  /** extra long vertical cracks (spruce/snag) */
  vertCrack: number;
  /** birch lenticel dashes */
  lenticels: number;
  deep: [number, number, number];
  high: [number, number, number];
  /** low-freq hue mottling */
  mottle: number;
  roughBase: number;
  roughVar: number;
  normalK: number;
}

export const BARK_TABLE: readonly BarkParams[] = [
  { // 0 spruce: narrow vertical fissured ridges, gray-brown
    plates: [16, 4], warp: 0.5, fissureW: 0.34, fissureDepth: 0.85, plateRound: 0.25,
    micro: 0.3, vertCrack: 0.55, lenticels: 0,
    deep: [0.045, 0.032, 0.026], high: [0.21, 0.155, 0.115], mottle: 0.25,
    roughBase: 0.92, roughVar: 0.07, normalK: 2.6,
  },
  { // 1 pine: big orange plates, flaky crevices
    plates: [7, 9], warp: 0.35, fissureW: 0.42, fissureDepth: 1.0, plateRound: 0.55,
    micro: 0.22, vertCrack: 0.1, lenticels: 0,
    deep: [0.05, 0.027, 0.016], high: [0.30, 0.155, 0.075], mottle: 0.35,
    roughBase: 0.88, roughVar: 0.1, normalK: 3.0,
  },
  { // 2 beech: smooth pale gray, subtle mottling
    plates: [5, 5], warp: 0.6, fissureW: 0.85, fissureDepth: 0.12, plateRound: 0.1,
    micro: 0.12, vertCrack: 0, lenticels: 0,
    deep: [0.16, 0.15, 0.135], high: [0.30, 0.285, 0.25], mottle: 0.5,
    roughBase: 0.78, roughVar: 0.08, normalK: 0.9,
  },
  { // 3 birch: white paper bark + dark horizontal lenticels
    plates: [4, 3], warp: 0.3, fissureW: 0.9, fissureDepth: 0.06, plateRound: 0.05,
    micro: 0.1, vertCrack: 0, lenticels: 1,
    deep: [0.46, 0.44, 0.42], high: [0.80, 0.79, 0.76], mottle: 0.22,
    roughBase: 0.62, roughVar: 0.18, normalK: 0.7,
  },
  { // 4 karst gnarl: twisted deep ridges
    plates: [9, 3], warp: 1.4, fissureW: 0.5, fissureDepth: 0.9, plateRound: 0.3,
    micro: 0.34, vertCrack: 0.3, lenticels: 0,
    deep: [0.05, 0.043, 0.036], high: [0.205, 0.18, 0.15], mottle: 0.3,
    roughBase: 0.93, roughVar: 0.05, normalK: 2.8,
  },
  { // 5 snag: weathered silver-gray, long splits
    plates: [11, 2], warp: 0.4, fissureW: 0.3, fissureDepth: 0.7, plateRound: 0.15,
    micro: 0.26, vertCrack: 0.8, lenticels: 0,
    deep: [0.07, 0.065, 0.06], high: [0.26, 0.25, 0.23], mottle: 0.2,
    roughBase: 0.9, roughVar: 0.06, normalK: 2.2,
  },
];

export interface BarkTextures {
  texA: StorageTexture;
  texB: StorageTexture;
}

/** texA (albedo.rgb sqrt-enc + cavity.a) and texB (normal.xy + rough + height)
 *  for one bark pixel — shared by the per-layer 2D bake and the array bake so
 *  the nanite resolve's sampled array is bit-identical to the old material's
 *  per-layer textures. Call inside an Fn() stack. */
function barkPixelValues(
  p: BarkParams,
  uvN: NV2,
  seedK: number,
): { a: NV4; b: NV4 } {
  const h = barkHeight(p, uvN, seedK).toVar();
  const e = 1.6 / BARK_RES;
  const hx0 = barkHeight(p, uvN.add(vec2(-e, 0)), seedK);
  const hx1 = barkHeight(p, uvN.add(vec2(e, 0)), seedK);
  const hy0 = barkHeight(p, uvN.add(vec2(0, -e)), seedK);
  const hy1 = barkHeight(p, uvN.add(vec2(0, e)), seedK);
  const n = vec3(
    hx0.sub(hx1).mul(p.normalK * 0.5),
    hy0.sub(hy1).mul(p.normalK * 0.5),
    float(1),
  ).normalize();

  // cavity: darker in crevices + slight top lightening
  const cavity = h.clamp(0, 1).mul(0.7).add(0.3);
  const mott = pnoise(uvN.mul(2), 2, seedK + 201).sub(0.5).mul(p.mottle);
  let albedo: NV3 = mix(
    vec3(p.deep[0], p.deep[1], p.deep[2]),
    vec3(p.high[0], p.high[1], p.high[2]),
    h.clamp(0, 1),
  ) as unknown as NV3;
  albedo = albedo.mul(mott.add(1)) as unknown as NV3;
  if (p.lenticels > 0) {
    // horizontal dark dashes: stretched worley spots
    const lw = pworley(uvN.mul(vec2(5, 24)), new Vector2(5, 24), seedK + 77);
    const dash = float(1).sub(lw.f1.smoothstep(0.2, 0.42));
    albedo = mix(albedo, vec3(0.045, 0.04, 0.038), dash.mul(0.85)) as unknown as NV3;
  }
  const rough = float(p.roughBase).add(h.sub(0.5).mul(p.roughVar * 2));
  const albEnc = sqrtV3(albedo.clamp(0, 1) as unknown as NV3);
  return {
    a: vec4(albEnc, cavity) as unknown as NV4,
    b: vec4(n.xy.mul(0.5).add(0.5), rough.clamp(0.3, 1), h.clamp(0, 1)) as unknown as NV4,
  };
}

/** height field expression (re-evaluated at offsets for normals) */
function barkHeight(p: BarkParams, uvN: NV2, seedK: number): NF {
  const P = 1; // uv tiles at 1
  const warp = vec2(
    pfbm(uvN.mul(6), 2, 6 * P, seedK + 31).sub(0.5),
    pfbm(uvN.mul(6), 2, 6 * P, seedK + 67).sub(0.5),
  ).mul(p.warp * 0.12);
  const q = uvN.add(warp);
  const pl = pworley(
    q.mul(vec2(p.plates[0], p.plates[1])),
    new Vector2(p.plates[0], p.plates[1]),
    seedK,
  );
  // plates: high in the middle, fissure at edges
  const fissure = pl.edge.div(p.fissureW).clamp(0, 1);
  let h: NF = fissure.pow(0.65).mul(p.fissureDepth);
  h = h.add(pl.f1.mul(p.plateRound));
  if (p.vertCrack > 0) {
    // long wavy vertical cracks: thin valleys in x
    const cx = q.x.mul(Math.max(1, Math.round(p.plates[0] * 0.5))).add(pfbm(q.mul(3), 2, 3, seedK + 5).mul(1.4));
    const crack = cx.fract().sub(0.5).abs().mul(2); // 0 at crack center
    h = h.mul(crack.div(0.22).clamp(0, 1).pow(0.5).mul(p.vertCrack).add(1 - p.vertCrack));
  }
  h = h.add(pfbm(uvN.mul(24), 3, 24 * P, seedK + 91).sub(0.5).mul(p.micro));
  return h;
}

export async function bakeBarkTextures(
  renderer: Renderer,
  layer: number,
  seedK: number,
): Promise<BarkTextures> {
  const p = BARK_TABLE[layer] as BarkParams;
  const mk = (): StorageTexture => {
    const t = new StorageTexture(BARK_RES, BARK_RES);
    t.wrapS = RepeatWrapping;
    t.wrapT = RepeatWrapping;
    t.generateMipmaps = true;
    t.minFilter = LinearMipmapLinearFilter;
    t.anisotropy = 4;
    return t;
  };
  const texA = mk();
  const texB = mk();
  void FloatType;

  const kernel = Fn(() => {
    const id = instanceIndex;
    const xi = id.mod(uint(BARK_RES));
    const yi = id.div(uint(BARK_RES));
    const uvN = vec2(float(xi).add(0.5), float(yi).add(0.5)).div(BARK_RES);
    const px = barkPixelValues(p, uvN, seedK);
    textureStore(texA, ivec2(int(xi), int(yi)), px.a);
    textureStore(texB, ivec2(int(xi), int(yi)), px.b);
  })().compute(BARK_RES * BARK_RES);

  await renderer.computeAsync(kernel);
  return { texA, texB };
}

export interface BarkArrayTextures {
  /** 2D-array albedo (rgb sqrt-enc) + cavity (a); slice index == bark layer */
  texA: StorageArrayTexture;
  /** 2D-array normal.xy + roughness + height; slice index == bark layer */
  texB: StorageArrayTexture;
}

/**
 * Bark texture ARRAY for the nanite resolve übershader (N4-C3): one
 * `texture_2d_array` per map, slice L == BARK_TABLE layer L, so a single
 * sampled binding covers every species' bark (the 10-storage / 16-sampled
 * budget forbids binding 12 separate 2D bark textures). Baked from the SAME
 * `barkPixelValues` recipe as the per-layer 2D bake — pass the per-layer
 * `seedK` used on the old path (`seed.sub('bark/L') % 977`) so the array is
 * visually identical. Deadwood shares slice 5 (snag).
 */
export async function bakeBarkArray(
  renderer: Renderer,
  seedKByLayer: readonly number[],
): Promise<BarkArrayTextures> {
  const depth = BARK_TABLE.length;
  const mk = (): StorageArrayTexture => {
    const t = new StorageArrayTexture(BARK_RES, BARK_RES, depth);
    t.wrapS = RepeatWrapping;
    t.wrapT = RepeatWrapping;
    t.generateMipmaps = true;
    t.minFilter = LinearMipmapLinearFilter;
    t.anisotropy = 4;
    return t;
  };
  const texA = mk();
  const texB = mk();
  // one dispatch per layer (BarkParams are a compile-time table — cannot be
  // indexed by a runtime node); each writes its array slice via .depthNode.
  for (let layer = 0; layer < depth; layer++) {
    const p = BARK_TABLE[layer] as BarkParams;
    const seedK = seedKByLayer[layer] ?? 0;
    const kernel = Fn(() => {
      const id = instanceIndex;
      const xi = id.mod(uint(BARK_RES));
      const yi = id.div(uint(BARK_RES));
      const uvN = vec2(float(xi).add(0.5), float(yi).add(0.5)).div(BARK_RES);
      const px = barkPixelValues(p, uvN, seedK);
      // array store: textureStore(arr, uvec2, value) + array index via depthNode
      // (read at generate time, so set AFTER the helper toStacks the node).
      const sA = textureStore(texA, ivec2(int(xi), int(yi)), px.a);
      (sA as unknown as { depthNode: unknown }).depthNode = int(layer);
      const sB = textureStore(texB, ivec2(int(xi), int(yi)), px.b);
      (sB as unknown as { depthNode: unknown }).depthNode = int(layer);
    })().compute(BARK_RES * BARK_RES);
    await renderer.computeAsync(kernel);
  }
  // Regenerate the mip chain from the now-filled level 0. three's auto-mip ran
  // (empty) when these storage textures were first bound for the compute write
  // and never reran — without this, any sample at mip > 0 reads black (the
  // nanite resolve's distant/grazing trunks). Self-submits its own encoder.
  const backend = (renderer as unknown as {
    backend: { generateMipmaps(t: StorageArrayTexture): void };
  }).backend;
  backend.generateMipmaps(texA);
  backend.generateMipmaps(texB);
  return { texA, texB };
}
