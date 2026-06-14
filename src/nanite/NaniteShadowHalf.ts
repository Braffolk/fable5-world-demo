/**
 * N5 shadow S0 (D-N29): HALF-RES PCSS evaluation + depth-aware bilateral upsample.
 *
 * The fixed per-pixel PCSS sample in shadowFactor (blocker search + penumbra PCF,
 * ~15 cascade-texture taps) is the STATIC shadow cost — paid every frame whether
 * the camera moves or not (R1 caches the per-cascade RASTER, not the sample). S0
 * evaluates shadowFactor at HALF resolution in a compute pass, writing (shadow,
 * camDist) into an rg32f StorageTexture, and the resolve reconstructs the full-res
 * factor with a depth-aware BILATERAL upsample (4 taps weighted by camera-distance
 * similarity). That quarters the PCSS evaluation count while keeping the contact
 * band sharp at depth discontinuities — a naive bilinear would bleed shadow across
 * silhouettes (foreground trunk vs background terrain). DAG-independent, resolve-
 * side only; ?shalfres=0 restores the full-res per-pixel path for A/B.
 *
 * The wp reconstruction mirrors the resolve EXACTLY (cam.invVp · ndc with the
 * bottom-up row flip — the verbatim NaniteFrame probe expression) so the half-res
 * sample lands on the same surface the resolve shades. shadowFactor uses the
 * surface normal ONLY for the receiver bias offset, so a cheap depth-derivative
 * geometric normal (oriented toward the camera) is ample here.
 */

import { FloatType, NearestFilter, RGFormat } from 'three';
import { StorageTexture, type Renderer } from 'three/webgpu';
import {
  Fn,
  If,
  cross,
  dot,
  exp,
  float,
  instanceIndex,
  normalize,
  screenCoordinate,
  sign,
  textureLoad,
  textureStore,
  uint,
  uvec2,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import type { NB, NF, NU, NV3 } from '../gpu/TSLTypes';
import type { NaniteCam } from './NaniteCommon';
import type { NaniteShadow } from './NaniteShadow';
import type { NaniteVisBuffers } from './NaniteRaster';
import { bcU2F, dispatch, elemU, minU, toF } from './Tsl';

interface NamedKernel {
  setName(n: string): unknown;
}

export interface ShadowHalf {
  /** the half-res shadow texture (rg32f: R = factor, G = camDist) — bound by the
   *  resolve via textureLoad in upsample(). */
  tex: StorageTexture;
  /** dispatch the half-res shadow eval. Call each frame AFTER the vis depth is
   *  final (raster.payload) and BEFORE the resolve samples it (post.render).
   *  cam.invVp must already be current for this frame (cam.update done). */
  run(renderer: Renderer): void;
  /** resolve-side full-res factor: depth-aware bilateral upsample of tex. wp =
   *  the resolve's reconstructed world pos, camDist = |wp − camPos|. */
  upsample(wp: NV3, camDist: NF): NF;
}

/** camera-distance bilateral tolerance: a half-res tap is rejected as its camDist
 *  diverges from the full-res receiver by more than ~this FRACTION of the receiver
 *  distance (relative → far cascades blend freely, near contact edges stay crisp). */
const BILATERAL_REL = 0.02;
const BILATERAL_ABS = 0.1; // metres floor so near-camera surfaces still blend a little

export function buildShadowHalf(
  vis: NaniteVisBuffers,
  cam: NaniteCam,
  shadow: NaniteShadow,
): ShadowHalf {
  const W = cam.width;
  const H = cam.height;
  const hW = Math.ceil(W / 2);
  const hH = Math.ceil(H / 2);
  const N = hW * hH;

  const tex = new StorageTexture(hW, hH);
  tex.type = FloatType;
  tex.format = RGFormat;
  tex.magFilter = NearestFilter;
  tex.minFilter = NearestFilter;
  tex.generateMipmaps = false;
  tex.name = 'nanShadowHalf';

  // reconstruct world pos at full-res pixel (x col, yTop top-down row) from the
  // stored vis depth — verbatim the resolve/probe expression (cam.invVp · ndc,
  // bottom-up row fy). Returns the point + whether the pixel is empty (sky).
  const reconstruct = (x: NU, yTop: NU): { wp: NV3; empty: NB } => {
    const fy = uint(H - 1).sub(yTop); // bottom-up row index into the vis buffer
    const pixelIndex = fy.mul(uint(W)).add(x);
    const dRaw = elemU(vis.depthV.ro, pixelIndex).toVar();
    const empty = dRaw.equal(uint(0xffffffff)) as unknown as NB;
    const visD = bcU2F(dRaw);
    const ndcX = toF(x).add(0.5).div(float(W)).mul(2).sub(1);
    const ndcY = toF(fy).add(0.5).div(float(H)).mul(2).sub(1);
    const hp = (cam.invVp as unknown as { mul(v: unknown): NV3 }).mul(
      vec4(ndcX, ndcY, visD, 1),
    ) as unknown as { x: NF; y: NF; z: NF; w: NF };
    const wp = vec3(hp.x.div(hp.w), hp.y.div(hp.w), hp.z.div(hp.w)) as unknown as NV3;
    return { wp, empty };
  };

  const kHalf = Fn(() => {
    const idx = instanceIndex;
    If(idx.lessThan(uint(N)), () => {
      const hx = idx.mod(uint(hW));
      const hy = idx.div(uint(hW));
      const x = hx.mul(2); // sample the top-left full-res pixel of the 2×2 block
      const yTop = hy.mul(2);
      const { wp, empty } = reconstruct(x, yTop);
      const sOut = float(1).toVar(); // empty (sky) → fully lit
      const dOut = float(0).toVar(); // empty → camDist 0 (matches no real receiver → bilateral drops it)
      If(empty.not(), () => {
        // cheap geometric normal from depth derivatives, oriented toward the
        // camera (shadowFactor uses normal only for the receiver bias offset).
        const wpR = reconstruct(minU(x.add(uint(2)), uint(W - 1)), yTop).wp;
        const wpD = reconstruct(x, minU(yTop.add(uint(2)), uint(H - 1))).wp;
        const toCam = (vec3(cam.camPos) as unknown as { sub(o: NV3): NV3 }).sub(wp);
        const gn = normalize(
          cross(wpR.sub(wp) as unknown as NV3, wpD.sub(wp) as unknown as NV3),
        ) as unknown as NV3;
        // faceforward: flip toward the camera (visible surfaces face the viewer)
        const facing = dot(gn, toCam as unknown as NV3) as unknown as NF;
        const n = (gn as unknown as { mul(o: NF): NV3 }).mul(sign(facing) as unknown as NF) as unknown as NV3;
        // IGN noise coord = the full-res pixel we sampled (compute has no fragCoord)
        sOut.assign(shadow.shadowFactor(wp, n, vec2(toF(x), toF(yTop))) as unknown as NF);
        dOut.assign((toCam as unknown as { length(): NF }).length());
      });
      textureStore(tex, uvec2(hx, hy), vec4(sOut, dOut, 0, 1)).toWriteOnly();
    });
  })().compute(N, [256]);
  (kHalf as unknown as NamedKernel).setName('nanShadowHalf');

  const run = (renderer: Renderer): void => {
    dispatch(renderer, kHalf);
  };

  // depth-aware bilateral upsample: bilinear footprint over the 4 nearest half-
  // texels, each spatial weight scaled by camera-distance similarity to the full-
  // res receiver. Half-texel (hx,hy) represents full-res pixel (2hx,2hy).
  const upsample = (wp: NV3, camDist: NF): NF =>
    Fn(() => {
      void wp;
      const pxf = (screenCoordinate as unknown as { x: NF }).x;
      const pyf = (screenCoordinate as unknown as { y: NF }).y; // top-down, matches the vis flip
      const h0x = uint(pxf).div(uint(2));
      const h0y = uint(pyf).div(uint(2));
      const fracx = pxf.sub(toF(h0x).mul(2)).mul(0.5); // ∈ [0,1): weight toward h0x+1
      const fracy = pyf.sub(toF(h0y).mul(2)).mul(0.5);
      const tol = camDist.mul(BILATERAL_REL).add(BILATERAL_ABS);

      const wsum = float(0).toVar();
      const ssum = float(0).toVar();
      const corners: [number, number][] = [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
      ];
      for (const [i, j] of corners) {
        const hx = minU(h0x.add(uint(i)), uint(hW - 1));
        const hy = minU(h0y.add(uint(j)), uint(hH - 1));
        const t = textureLoad(tex, uvec2(hx, hy)) as unknown as { x: NF; y: NF };
        const wx = i === 0 ? fracx.oneMinus() : fracx;
        const wy = j === 0 ? fracy.oneMinus() : fracy;
        const spatial = (wx as unknown as { mul(o: NF): NF }).mul(wy);
        // gaussian-ish camera-distance similarity; empty taps (camDist 0) drop out
        const dd = (t.y.sub(camDist) as unknown as { abs(): NF }).abs();
        const distW = exp(dd.div(tol).mul(dd.div(tol)).negate()) as unknown as NF;
        const w = (spatial as unknown as { mul(o: NF): NF }).mul(distW);
        wsum.addAssign(w);
        ssum.addAssign(w.mul(t.x));
      }
      // fallback to the nearest tap if every bilateral weight collapsed (all taps
      // across a depth discontinuity from this receiver)
      const nearest = (
        textureLoad(tex, uvec2(minU(h0x, uint(hW - 1)), minU(h0y, uint(hH - 1)))) as unknown as {
          x: NF;
        }
      ).x;
      return wsum.greaterThan(1e-4).select(ssum.div(wsum.max(1e-4)), nearest) as unknown as NF;
    })() as unknown as NF;

  return { tex, run, upsample };
}
