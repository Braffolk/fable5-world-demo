/**
 * Vegetation materials (v1: structure-review shading; TexSynth bark/leaf
 * detail + translucency land with the texture milestone).
 *
 * All vegetation geometry carries a `vdata` vec4 attribute:
 *   x hue jitter (−1..1) · y sway flexibility · z sway phase · w baked AO.
 * Hue/AO are consumed here; sway feeds the Phase-6 wind field.
 */

import { Color, DoubleSide, type DirectionalLight, type Texture, Vector3 } from 'three';
import { MeshStandardNodeMaterial } from 'three/webgpu';
import {
  attribute,
  cameraPosition,
  clamp,
  float,
  mix,
  normalMap,
  normalWorld,
  positionWorld,
  smoothstep,
  texture,
  uniform,
  uv,
  vec3,
} from 'three/tsl';
import { fbm3, valueNoise3 } from '../gpu/noise/NoiseTSL';
import type { NF, NV3, NV4 } from '../gpu/TSLTypes';

/**
 * Shared sun uniforms for the foliage translucency term (D-2). Updated by
 * the scene on init + time-of-day changes.
 */
export const sunU = {
  dir: uniform(new Vector3(0, 1, 0)),
  color: uniform(new Color(1, 1, 1)),
  intensity: uniform(0),
};

export function updateSunUniforms(sun: DirectionalLight): void {
  sunU.dir.value.copy(sun.position).normalize();
  sunU.color.value.copy(sun.color);
  sunU.intensity.value = sun.intensity;
}

/**
 * Back-lit transmission glow: light through the blade toward a camera that
 * faces the sun. Thin-surface approximation; modest k since it is not
 * shadow-gated yet (full gating with Phase-5/6 light queries).
 */
function translucency(albedo: NV3, k: number): NV3 {
  const viewDir = positionWorld.sub(cameraPosition).normalize();
  const toward = clamp(viewDir.dot(vec3(sunU.dir).negate()), 0, 1);
  const glow = toward.pow(5).mul(sunU.intensity).mul(k);
  const sunCol = sunU.color as unknown as NV3;
  return albedo.mul(sunCol).mul(glow).mul(vec3(0.9, 1.05, 0.55));
}

/** grass variant: transmission strengthens toward the blade tip */
export function grassTranslucency(albedo: NV3, tipT: NF): NV3 {
  return translucency(albedo, 0.09).mul(tipT);
}

function vdata(): NV4 {
  return attribute('vdata', 'vec4') as unknown as NV4;
}

/** hue jitter: rotate albedo toward yellow (+) / blue-green (−) */
function hueShift(base: NV3, hue: NF, amount: number): NV3 {
  const k = hue.mul(amount);
  const warm = vec3(1.18, 1.0, 0.55);
  const cool = vec3(0.7, 0.95, 1.25);
  const shifted = base
    .mul(warm)
    .mul(clamp(k, 0, 1))
    .add(base.mul(cool).mul(clamp(k.negate(), 0, 1)))
    .add(base.mul(float(1).sub(k.abs())));
  return shifted;
}

export interface BarkMatParams {
  color: { r: number; g: number; b: number };
  roughness?: number;
}

export function barkMaterial(p: BarkMatParams): MeshStandardNodeMaterial {
  const mat = new MeshStandardNodeMaterial();
  const d = vdata();
  const base = vec3(p.color.r, p.color.g, p.color.b);
  mat.colorNode = hueShift(base, d.x, 0.18).mul(d.w.mul(0.75).add(0.25));
  mat.roughness = p.roughness ?? 0.93;
  mat.metalness = 0;
  return mat;
}

/**
 * Synthesized bark material: tileable albedo/cavity + normal/rough/height.
 * Cavity feeds `aoNode` — AO on indirect light only (DEVIATIONS D-1 close).
 */
export function barkTexturedMaterial(tex: {
  texA: Texture;
  texB: Texture;
}): MeshStandardNodeMaterial {
  const mat = new MeshStandardNodeMaterial();
  const d = vdata();
  const a = texture(tex.texA, uv() as never) as unknown as NV4;
  const b = texture(tex.texB, uv() as never) as unknown as NV4;
  const albedo = a.rgb.mul(a.rgb); // sqrt-encoded at bake
  mat.colorNode = hueShift(albedo, d.x, 0.14).mul(d.w.mul(0.45).add(0.55));
  mat.normalNode = normalMap(vec3(b.x, b.y, 1));
  mat.aoNode = a.w;
  mat.roughnessNode = b.z;
  mat.metalness = 0;
  return mat;
}

/**
 * Procedural rock shading (no UVs): strata banding from vdata.y, lichen
 * spots + dust on open faces, moss by upness (dressing rule), cavity AO via
 * aoNode. Geometric normals carry the meso detail (displaced mesh).
 */
export function rockMaterial(opts?: { moss?: number }): MeshStandardNodeMaterial {
  const mat = new MeshStandardNodeMaterial();
  const d = vdata();
  const wp = positionWorld;
  const strataT = d.y;
  const upness = normalWorld.y.max(0);
  // band tint: alternating warm/cool sediment layers + grain
  const bandTint = valueNoise3(vec3(float(0), strataT.mul(7.3), float(0)).add(wp.mul(0.02)));
  const grain = fbm3(wp.mul(2.1), 3).mul(0.5).add(0.5);
  let albedo = mix(
    vec3(0.085, 0.075, 0.07),
    vec3(0.21, 0.165, 0.12),
    bandTint.mul(0.55).add(grain.mul(0.45)).clamp(0, 1),
  ) as unknown as NV3;
  // pale lichen patches on exposed faces
  const lich = smoothstep(0.62, 0.78, valueNoise3(wp.mul(3.7)))
    .mul(d.z.mul(0.7).add(0.3));
  albedo = mix(albedo, vec3(0.16, 0.175, 0.14), lich.mul(0.55)) as unknown as NV3;
  // dust settles on up-faces
  albedo = mix(albedo, vec3(0.17, 0.15, 0.12), upness.pow(2).mul(0.3)) as unknown as NV3;
  // dirt streaks bleeding down steep faces (dressing rule)
  const steep = float(1).sub(upness);
  const streakN = valueNoise3(vec3(wp.x.mul(2.6), wp.y.mul(0.22), wp.z.mul(2.6)));
  const streak = smoothstep(0.55, 0.82, streakN)
    .mul(smoothstep(0.45, 0.8, steep))
    .mul(0.55);
  albedo = mix(albedo, albedo.mul(vec3(0.5, 0.46, 0.4)), streak) as unknown as NV3;
  const mossAmt = opts?.moss ?? 0.25;
  if (mossAmt > 0) {
    const mossN = smoothstep(0.45, 0.75, fbm3(wp.mul(1.7), 3).mul(0.5).add(0.5));
    const moss = smoothstep(0.45, 0.85, upness)
      .mul(mossN).mul(d.w).mul(mossAmt * 2).clamp(0, 1);
    albedo = mix(albedo, vec3(0.045, 0.085, 0.03), moss) as unknown as NV3;
    mat.roughnessNode = mix(float(0.93), float(1), moss).sub(lich.mul(0.06));
  } else {
    mat.roughnessNode = float(0.93).sub(lich.mul(0.06));
  }
  mat.colorNode = albedo.mul(d.w.mul(0.35).add(0.65));
  mat.aoNode = d.w;
  mat.metalness = 0;
  return mat;
}

/** deadfall wood: bark textures + moss carpet on the up-side by vdata.z */
export function deadwoodMaterial(tex: {
  texA: Texture;
  texB: Texture;
}): MeshStandardNodeMaterial {
  const mat = new MeshStandardNodeMaterial();
  const d = vdata();
  const a = texture(tex.texA, uv() as never) as unknown as NV4;
  const b = texture(tex.texB, uv() as never) as unknown as NV4;
  let albedo = a.rgb.mul(a.rgb) as unknown as NV3;
  const mossN = smoothstep(0.24, 0.58, fbm3(positionWorld.mul(2.6), 3).mul(0.5).add(0.5));
  const moss = smoothstep(0.05, 0.65, normalWorld.y).mul(d.z).mul(mossN).clamp(0, 1);
  albedo = mix(albedo, vec3(0.05, 0.1, 0.032), moss) as unknown as NV3;
  // rot darkening for heavily decayed wood
  albedo = albedo.mul(float(1).sub(d.z.mul(0.25))) as unknown as NV3;
  mat.colorNode = hueShift(albedo, d.x, 0.1);
  mat.normalNode = normalMap(vec3(b.x, b.y, 1));
  mat.aoNode = a.w;
  mat.roughnessNode = mix(b.z, float(1), moss);
  mat.metalness = 0;
  return mat;
}

/**
 * Flower shading by vdata.x part id: 0 stem/leaf, 0.5 flower center, 1 petal.
 */
export function flowerMaterial(petal: {
  r: number;
  g: number;
  b: number;
}): MeshStandardNodeMaterial {
  const mat = new MeshStandardNodeMaterial();
  const d = vdata();
  const stem = vec3(0.045, 0.1, 0.03);
  const center = vec3(0.5, 0.32, 0.045);
  const petalC = vec3(petal.r, petal.g, petal.b);
  const centerK = smoothstep(0.12, 0.02, d.x.sub(0.5).abs());
  const petalK = smoothstep(0.85, 0.95, d.x);
  let albedo = mix(stem, center, centerK) as unknown as NV3;
  albedo = mix(albedo, petalC, petalK) as unknown as NV3;
  mat.colorNode = albedo.mul(d.w.mul(0.5).add(0.5));
  mat.roughness = 0.7;
  mat.metalness = 0;
  mat.side = DoubleSide;
  return mat;
}

/** mushroom shading by vdata.x part id: 0 stem, 0.5 gills, 1 cap */
export function mushroomMaterial(): MeshStandardNodeMaterial {
  const mat = new MeshStandardNodeMaterial();
  const d = vdata();
  const stem = vec3(0.32, 0.29, 0.24);
  const gills = vec3(0.42, 0.37, 0.28);
  const cap = vec3(0.23, 0.12, 0.05);
  const gillK = smoothstep(0.12, 0.02, d.x.sub(0.5).abs());
  const capK = smoothstep(0.85, 0.95, d.x);
  let albedo = mix(stem, gills, gillK) as unknown as NV3;
  albedo = mix(albedo, cap, capK) as unknown as NV3;
  mat.colorNode = albedo.mul(d.w);
  mat.roughness = 0.62;
  mat.metalness = 0;
  return mat;
}

export interface FoliageMatParams {
  color: { r: number; g: number; b: number; hueVar: number };
}

export function foliageMaterial(p: FoliageMatParams): MeshStandardNodeMaterial {
  const mat = new MeshStandardNodeMaterial();
  const d = vdata();
  const base = vec3(p.color.r, p.color.g, p.color.b);
  const tinted = hueShift(base, d.x, p.color.hueVar).mul(d.w.mul(0.8).add(0.2));
  mat.colorNode = tinted;
  mat.emissiveNode = translucency(tinted as unknown as NV3, 0.032);
  mat.roughness = 0.62;
  mat.metalness = 0;
  mat.side = DoubleSide;
  return mat;
}

/** captured cluster-card material: sqrt-decoded atlas albedo, alpha-tested */
export function foliageCardMaterial(
  atlas: Texture,
  p: FoliageMatParams,
): MeshStandardNodeMaterial {
  const mat = new MeshStandardNodeMaterial();
  const d = vdata();
  const t = texture(atlas, uv() as never) as unknown as NV4;
  const albedo = t.rgb.mul(t.rgb); // sqrt-encoded at capture
  const tinted = hueShift(albedo, d.x, p.color.hueVar * 0.8).mul(
    d.w.mul(0.75).add(0.25),
  );
  mat.colorNode = tinted;
  mat.emissiveNode = translucency(tinted as unknown as NV3, 0.032);
  mat.opacityNode = t.w;
  mat.alphaTest = 0.32;
  mat.roughness = 0.62;
  mat.metalness = 0;
  mat.side = DoubleSide;
  return mat;
}
