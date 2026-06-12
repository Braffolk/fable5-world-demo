/**
 * Hierarchical-Z pyramid over the Option C vis-depth buffer (N2-C2; example
 * port per F19 — the SW+HW paths write ONE depth buffer, so the HZB has one
 * source). Level 0 = half canvas res reduced 2×2 from visDepth (empty
 * 0xffffffff → 1.0 far), each level the 2×2 max (farthest, classic depth) of
 * the previous; ALL levels packed into one f32 storage buffer with a vec4
 * level table (offset, w, h, 0). Initial fill 1.0 occludes nothing —
 * frame-0/resize pass-through for free.
 *
 * sphereOccluded is the example's conservative test verbatim: nearest sphere
 * point vs the 2×2 footprint max at the level where the sphere's diameter
 * fits one texel; PREV-frame VP + camera position (static world: prev
 * matrices, current bounds — NANITE.md "Culling (N2)").
 */

import { StorageBufferAttribute } from 'three/webgpu';
import type { Renderer, StorageBufferNode } from 'three/webgpu';
import { Fn, If, float, instanceIndex, uint, vec4 } from 'three/tsl';
import type { NB, NF, NV3 } from '../gpu/TSLTypes';
import type { NaniteCam } from './NaniteCommon';
import { bcU2F, dispatch, elemU, minU, sF32Views, toF, uniformArrV4, uniformF } from './Tsl';
import { Vector4 } from 'three';

const MAX_LEVELS = 16;

interface ComputeKernel {
  setName(name: string): unknown;
}

export interface NaniteHzb {
  /** run the reduction chain (call AFTER the raster wrote visDepth) */
  build(renderer: Renderer): void;
  /** conservative occlusion test vs the pyramid (prev-frame matrices) */
  sphereOccluded(center: NV3, radius: NF): NB;
  levelCount: number;
}

export function buildNaniteHzb(
  visDepthRO: StorageBufferNode<'uint'>,
  cam: NaniteCam,
): NaniteHzb {
  // ---- level layout (fixed per canvas size) -----------------------------------
  const levels: { offset: number; w: number; h: number }[] = [];
  let w = Math.max(1, Math.ceil(cam.width / 2));
  let h = Math.max(1, Math.ceil(cam.height / 2));
  let offset = 0;
  while (levels.length < MAX_LEVELS) {
    levels.push({ offset, w, h });
    offset += w * h;
    if (w === 1 && h === 1) break;
    w = Math.max(1, Math.ceil(w / 2));
    h = Math.max(1, Math.ceil(h / 2));
  }
  const totalTexels = offset;
  const levelCount = levels.length;

  const table = uniformArrV4(
    Array.from({ length: MAX_LEVELS }, (_, k) => {
      const l = levels[Math.min(k, levelCount - 1)] as { offset: number; w: number; h: number };
      return new Vector4(l.offset, l.w, l.h, 0);
    }),
  );
  const levelCountU = uniformF(levelCount);

  // far plane everywhere — occludes nothing until the first build
  const hzbAttr = new StorageBufferAttribute(new Float32Array(totalTexels).fill(1), 1);
  const hzbF = sF32Views(hzbAttr, totalTexels);

  // ---- per-level reduction kernels -----------------------------------------------
  const kernels: unknown[] = [];
  for (let k = 0; k < levelCount; k++) {
    const info = levels[k] as { offset: number; w: number; h: number };
    const kn = Fn(() => {
      const lw = uint(info.w);
      const lh = uint(info.h);
      If(instanceIndex.lessThan(lw.mul(lh)), () => {
        const x = instanceIndex.mod(lw);
        const y = instanceIndex.div(lw);
        const sx = x.mul(uint(2));
        const sy = y.mul(uint(2));
        const depthMax = float(0).toVar();
        if (k === 0) {
          // source: full-res vis depth (u32 f32-bits; empty → far plane)
          const sw = uint(cam.width - 1);
          const sh = uint(cam.height - 1);
          for (let dy = 0; dy < 2; dy++) {
            for (let dx = 0; dx < 2; dx++) {
              const tx = minU(sx.add(uint(dx)), sw);
              const ty = minU(sy.add(uint(dy)), sh);
              const bits = elemU(visDepthRO, ty.mul(uint(cam.width)).add(tx));
              const d = bits.equal(uint(0xffffffff)).select(float(1), bcU2F(bits));
              depthMax.assign(depthMax.max(d));
            }
          }
        } else {
          // read the previous level through the SAME rw view that writes this
          // one — a second (read-only) view of one buffer in one dispatch is
          // a WebGPU same-scope usage violation (N0 gotcha; example does this)
          const src = levels[k - 1] as { offset: number; w: number; h: number };
          const srcW = uint(src.w);
          const swMax = uint(src.w - 1);
          const shMax = uint(src.h - 1);
          for (let dy = 0; dy < 2; dy++) {
            for (let dx = 0; dx < 2; dx++) {
              const tx = minU(sx.add(uint(dx)), swMax);
              const ty = minU(sy.add(uint(dy)), shMax);
              depthMax.assign(
                depthMax.max(hzbF.rw.element(uint(src.offset).add(ty.mul(srcW)).add(tx))),
              );
            }
          }
        }
        hzbF.rw.element(uint(info.offset).add(y.mul(lw)).add(x)).assign(depthMax);
      });
    })().compute(info.w * info.h, [64]);
    (kn as unknown as ComputeKernel).setName(`nanHzbL${k}`);
    kernels.push(kn);
  }

  const build = (renderer: Renderer): void => {
    for (const k of kernels) dispatch(renderer, k);
  };

  // ---- conservative sphere test (example-verbatim, prev-frame inputs) -------------
  const sphereOccluded = (center: NV3, radius: NF): NB => {
    const toCamera = cam.prevCamPos.sub(center).toVar();
    const dist = toCamera.length().toVar();
    const nearPoint = center.add(toCamera.div(dist).mul(radius));
    const nearClip = cam.prevVp.mul(vec4(nearPoint, 1)).toVar();
    const centerClip = cam.prevVp.mul(vec4(center, 1)).toVar();
    const nearestZ = nearClip.z.div(nearClip.w);
    const ndc = centerClip.xy.div(centerClip.w);

    // footprint in half-res pyramid texels picks the level where the sphere's
    // diameter fits one texel (2×2 window always covers it); /4 = NDC half-
    // screen factor × half-res pyramid
    const radiusTexels = radius.mul(cam.cotHalfFov).mul(cam.uH).div(4).div(dist);
    const levelF = radiusTexels.mul(2).max(1).log2().ceil().clamp(0, levelCountU.sub(1));
    const info = table.element(uint(levelF));
    const lw = uint(info.y).toVar();
    const lh = uint(info.z).toVar();
    const lo = uint(info.x).toVar();

    // NO Y flip (departure from the example): its HZB sources a TEXTURE
    // (rows top-down); ours sources the visDepth buffer whose rows are
    // bottom-up (SW raster: s.y = (ndc.y+1)/2·H). The mirrored lookup
    // over-culled valley framings against the near wall's depth (bm1 went
    // 131k → 179 clusters before this line).
    const px = ndc.x.mul(0.5).add(0.5).mul(toF(lw));
    const py = ndc.y.mul(0.5).add(0.5).mul(toF(lh));
    const x0 = uint(px.sub(0.5).clamp(0, toF(lw.sub(uint(1))))).toVar();
    const y0 = uint(py.sub(0.5).clamp(0, toF(lh.sub(uint(1))))).toVar();
    const x1 = minU(x0.add(uint(1)), lw.sub(uint(1)));
    const y1 = minU(y0.add(uint(1)), lh.sub(uint(1)));

    const z00 = hzbF.ro.element(lo.add(y0.mul(lw)).add(x0));
    const z01 = hzbF.ro.element(lo.add(y0.mul(lw)).add(x1));
    const z10 = hzbF.ro.element(lo.add(y1.mul(lw)).add(x0));
    const z11 = hzbF.ro.element(lo.add(y1.mul(lw)).add(x1));
    const maxZ = z00.max(z01).max(z10.max(z11));

    return dist
      .greaterThan(radius.mul(2)) // never occlusion-cull right at the camera
      .and(nearClip.w.greaterThan(0))
      .and(centerClip.w.greaterThan(0))
      .and(nearestZ.greaterThan(maxZ)) as unknown as NB;
  };

  return { build, sphereOccluded, levelCount };
}
