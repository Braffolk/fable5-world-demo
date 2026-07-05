/**
 * Baked tileable noise textures — the terrain material was evaluating
 * ~35 live noise functions per pixel (≈52 ms/frame at 1080p on M1 Max);
 * these two rgba16f textures replace almost all of them with filtered
 * fetches. Gradient channels are pre-derived so normal-domain detail
 * costs one fetch instead of four finite-difference noise evaluations.
 *
 * Channel map (sample uv = worldXZ / (worldScale · PERIOD_channel)):
 *   texA. r  value noise          (PERIOD_VAL = 256)
 *   texA. g  fbm 3-oct            (PERIOD_FBM = 64)
 *   texA.ba  d(fbm3)/dx, /dz      (per noise unit, PERIOD_FBM)
 *   texB.rg  d(ridged3)/dx, /dz   (per noise unit, PERIOD_RID = 32)
 *   texB. b  ridged 3-oct         (PERIOD_RID)
 *   texB. a  worley F1            (PERIOD_WOR = 128)
 *
 * MirroredRepeatWrapping keeps the tile seamless without a periodic
 * lattice. Mirror symmetry is invisible at the scales involved.
 */

import { HalfFloatType, LinearFilter, MirroredRepeatWrapping } from 'three';
import type { Renderer } from 'three/webgpu';
import { StorageTexture } from 'three/webgpu';
import {
  Fn,
  If,
  Return,
  float,
  floor,
  instanceIndex,
  textureStore,
  uvec2,
  vec2,
  vec4,
} from 'three/tsl';
import type { NF, NV2 } from '../TSLTypes';
import { fbm2, hash22, ridged2, valueNoise2 } from '../noise/NoiseTSL';

export const NOISE_TEX_RES = 1024;
export const PERIOD_VAL = 256;
export const PERIOD_FBM = 64;
export const PERIOD_RID = 32;
export const PERIOD_WOR = 128;

export interface NoiseTextures {
  texA: StorageTexture;
  texB: StorageTexture;
}

function makeTex(nm: string): StorageTexture {
  const t = new StorageTexture(NOISE_TEX_RES, NOISE_TEX_RES);
  t.name = nm;
  t.type = HalfFloatType;
  t.wrapS = MirroredRepeatWrapping;
  t.wrapT = MirroredRepeatWrapping;
  t.magFilter = LinearFilter;
  t.minFilter = LinearFilter;
  t.generateMipmaps = false;
  return t;
}

/** worley F1 over a 3×3 cell neighbourhood (bake-time only) */
function worleyF1(p: NV2): NF {
  const cell = floor(p);
  const f = p.sub(cell);
  let best: NF = float(8);
  for (let j = -1; j <= 1; j++) {
    for (let i = 0; i < 3; i++) {
      const off = vec2(i - 1, j);
      const feat = hash22(cell.add(off));
      const d = off.add(feat).sub(f).length();
      best = best.min(d);
    }
  }
  return best;
}

export async function bakeNoiseTextures(renderer: Renderer): Promise<NoiseTextures> {
  const texA = makeTex('noiseBakeA');
  const texB = makeTex('noiseBakeB');
  const R = NOISE_TEX_RES;
  // gradient step: half a texel in noise units, per channel period
  const eFbm = (PERIOD_FBM / R) * 0.5;
  const eRid = (PERIOD_RID / R) * 0.5;

  const kernel = Fn(() => {
    const i = instanceIndex;
    If(i.greaterThanEqual(R * R), () => {
      Return();
    });
    const x = i.mod(R);
    const y = i.div(R);
    const uv = vec2(float(x).add(0.5), float(y).add(0.5)).div(R);

    const pVal = uv.mul(PERIOD_VAL);
    const pFbm = uv.mul(PERIOD_FBM);
    const pRid = uv.mul(PERIOD_RID);
    const pWor = uv.mul(PERIOD_WOR);

    const v = valueNoise2(pVal);
    const f = fbm2(pFbm, 3);
    const fdx = fbm2(pFbm.add(vec2(eFbm, 0)), 3).sub(fbm2(pFbm.sub(vec2(eFbm, 0)), 3)).div(2 * eFbm);
    const fdz = fbm2(pFbm.add(vec2(0, eFbm)), 3).sub(fbm2(pFbm.sub(vec2(0, eFbm)), 3)).div(2 * eFbm);

    const rid = ridged2(pRid, 3);
    const rdx = ridged2(pRid.add(vec2(eRid, 0)), 3).sub(ridged2(pRid.sub(vec2(eRid, 0)), 3)).div(2 * eRid);
    const rdz = ridged2(pRid.add(vec2(0, eRid)), 3).sub(ridged2(pRid.sub(vec2(0, eRid)), 3)).div(2 * eRid);
    const wor = worleyF1(pWor);

    const xy = uvec2(x.toUint(), y.toUint());
    textureStore(texA, xy, vec4(v, f, fdx, fdz)).toWriteOnly();
    textureStore(texB, xy, vec4(rdx, rdz, rid, wor)).toWriteOnly();
  })().compute(R * R);
  kernel.setName('noiseBake');
  await renderer.computeAsync(kernel);
  return { texA, texB };
}
