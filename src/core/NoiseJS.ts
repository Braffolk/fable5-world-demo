/**
 * CPU-side deterministic noise — used by mesh generators (trees, rocks) where
 * geometry is built once on the CPU. Runtime per-frame noise lives on the GPU.
 */

import { mix32 } from './Seed';

function hash3(x: number, y: number, z: number, seed: number): number {
  let h = seed >>> 0;
  h = mix32(h ^ Math.imul(x | 0, 0x27d4eb2f));
  h = mix32(h ^ Math.imul(y | 0, 0x165667b1));
  h = mix32(h ^ Math.imul(z | 0, 0x9e3779b1));
  return h / 4294967296;
}

function smooth(t: number): number {
  return t * t * (3 - 2 * t);
}

/** trilinear value noise, ~[-1, 1] */
function valueNoise3(x: number, y: number, z: number, seed: number): number {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const zi = Math.floor(z);
  const xf = smooth(x - xi);
  const yf = smooth(y - yi);
  const zf = smooth(z - zi);
  let v = 0;
  let n000 = hash3(xi, yi, zi, seed);
  let n100 = hash3(xi + 1, yi, zi, seed);
  let n010 = hash3(xi, yi + 1, zi, seed);
  let n110 = hash3(xi + 1, yi + 1, zi, seed);
  let n001 = hash3(xi, yi, zi + 1, seed);
  let n101 = hash3(xi + 1, yi, zi + 1, seed);
  let n011 = hash3(xi, yi + 1, zi + 1, seed);
  let n111 = hash3(xi + 1, yi + 1, zi + 1, seed);
  const nx00 = n000 + (n100 - n000) * xf;
  const nx10 = n010 + (n110 - n010) * xf;
  const nx01 = n001 + (n101 - n001) * xf;
  const nx11 = n011 + (n111 - n011) * xf;
  const nxy0 = nx00 + (nx10 - nx00) * yf;
  const nxy1 = nx01 + (nx11 - nx01) * yf;
  v = nxy0 + (nxy1 - nxy0) * zf;
  return v * 2 - 1;
}

/** standard fBm over valueNoise3 */
export function fbm3(
  x: number,
  y: number,
  z: number,
  seed: number,
  octaves = 5,
  lacunarity = 2.02,
  gain = 0.5,
): number {
  let amp = 0.5;
  let freq = 1;
  let sum = 0;
  let norm = 0;
  for (let i = 0; i < octaves; i++) {
    sum += amp * valueNoise3(x * freq, y * freq, z * freq, mix32(seed + i * 0x9e37));
    norm += amp;
    amp *= gain;
    freq *= lacunarity;
  }
  return sum / norm;
}

/** ridged fBm in [0,1], sharp creases */
export function ridged3(
  x: number,
  y: number,
  z: number,
  seed: number,
  octaves = 5,
  lacunarity = 2.1,
  gain = 0.52,
): number {
  let amp = 0.5;
  let freq = 1;
  let sum = 0;
  let norm = 0;
  for (let i = 0; i < octaves; i++) {
    const n = 1 - Math.abs(valueNoise3(x * freq, y * freq, z * freq, mix32(seed + i * 0x51ed)));
    sum += amp * n * n;
    norm += amp;
    amp *= gain;
    freq *= lacunarity;
  }
  return sum / norm;
}
