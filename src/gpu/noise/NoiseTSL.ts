/**
 * TSL noise library — graph-builder functions (unrolled at shader build time).
 * Hash family: Dave Hoskins–style sinless hashes (stable across GPUs).
 * Octave counts are JS-side constants; graphs are built per call site.
 */

import { dot, float, floor, fract, mix, vec2, vec3 } from 'three/tsl';
import type { NF, NV2, NV3 } from '../TSLTypes';

/**
 * NOTE: these are pure expression builders — no `.toVar()`/`.assign()` — so they
 * work both inside compute `Fn` bodies AND in material node graphs (assign ops
 * require an active TSL stack, which materials don't have).
 */

/** vec2 → float hash, [0,1) */
export function hash12(p: NV2): NF {
  const a = fract(vec3(p.x, p.y, p.x).mul(0.1031));
  const b = a.add(dot(a, a.yzx.add(33.33)));
  return fract(b.x.add(b.y).mul(b.z));
}

/** vec3 → float hash, [0,1) */
export function hash13(p: NV3): NF {
  const a = fract(p.mul(0.1031));
  const b = a.add(dot(a, a.zyx.add(31.32)));
  return fract(b.x.add(b.y).mul(b.z));
}

/** vec2 → vec2 hash, [0,1)² */
export function hash22(p: NV2): NV2 {
  const a = fract(vec3(p.x, p.y, p.x).mul(vec3(0.1031, 0.103, 0.0973)));
  const b = a.add(dot(a, a.yzx.add(33.33)));
  return fract(b.xx.add(b.yz).mul(b.zy));
}

/** smooth value noise, vec2 domain → [0,1] */
export function valueNoise2(p: NV2): NF {
  const i = floor(p);
  const f = fract(p);
  const u = f.mul(f).mul(f.mul(-2).add(3));
  const a = hash12(i);
  const b = hash12(i.add(vec2(1, 0)));
  const c = hash12(i.add(vec2(0, 1)));
  const d = hash12(i.add(vec2(1, 1)));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

/** smooth value noise, vec3 domain → [0,1] */
export function valueNoise3(p: NV3): NF {
  const i = floor(p);
  const f = fract(p);
  const u = f.mul(f).mul(f.mul(-2).add(3));
  const n000 = hash13(i.add(vec3(0, 0, 0)));
  const n100 = hash13(i.add(vec3(1, 0, 0)));
  const n010 = hash13(i.add(vec3(0, 1, 0)));
  const n110 = hash13(i.add(vec3(1, 1, 0)));
  const n001 = hash13(i.add(vec3(0, 0, 1)));
  const n101 = hash13(i.add(vec3(1, 0, 1)));
  const n011 = hash13(i.add(vec3(0, 1, 1)));
  const n111 = hash13(i.add(vec3(1, 1, 1)));
  const x00 = mix(n000, n100, u.x);
  const x10 = mix(n010, n110, u.x);
  const x01 = mix(n001, n101, u.x);
  const x11 = mix(n011, n111, u.x);
  return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

/** fBm over 2D value noise → approx [0,1] */
export function fbm2(p: NV2, octaves: number, lacunarity = 2.02, gain = 0.5): NF {
  let amp = 0.5;
  let norm = 0;
  let freq = 1;
  let sum: NF = float(0);
  for (let o = 0; o < octaves; o++) {
    sum = sum.add(valueNoise2(p.mul(freq).add(float(o * 17.13))).mul(amp));
    norm += amp;
    amp *= gain;
    freq *= lacunarity;
  }
  return sum.div(norm);
}

/** fBm over 3D value noise → approx [0,1] */
export function fbm3(p: NV3, octaves: number, lacunarity = 2.02, gain = 0.5): NF {
  let amp = 0.5;
  let norm = 0;
  let freq = 1;
  let sum: NF = float(0);
  for (let o = 0; o < octaves; o++) {
    sum = sum.add(valueNoise3(p.mul(freq).add(float(o * 19.19))).mul(amp));
    norm += amp;
    amp *= gain;
    freq *= lacunarity;
  }
  return sum.div(norm);
}

/** ridged multifractal over 2D noise → [0,1], sharp crests */
export function ridged2(p: NV2, octaves: number, lacunarity = 2.1, gain = 0.53): NF {
  let amp = 0.5;
  let norm = 0;
  let freq = 1;
  let sum: NF = float(0);
  for (let o = 0; o < octaves; o++) {
    const n = valueNoise2(p.mul(freq).add(float(o * 13.7))).mul(2).sub(1).abs().oneMinus();
    sum = sum.add(n.mul(n).mul(amp));
    norm += amp;
    amp *= gain;
    freq *= lacunarity;
  }
  return sum.div(norm);
}

