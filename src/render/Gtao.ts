/**
 * GTAO layer — faithful port of three 0.184 GTAONode's fragment math
 * (examples/jsm/tsl/display/GTAONode.js) so it can live inside the merged
 * half-res MRT pass (HalfResMrt) instead of its own render pass. Same
 * magic-square noise texture, slice/step loops, horizon math, and uniform
 * defaults — verify against the pinned source on any three upgrade.
 *
 * Deliberate differences, both required by the merge and output-equivalent:
 * - stock discards sky fragments onto a white-cleared RT; an MRT pass can't
 *   discard (it would kill the other attachments) → ao = 1 branch instead.
 * - temporal filtering stays off (stock default; _temporalDirection = 0).
 *
 * Camera matrices are LIVE uniform(camera.matrix) references like stock —
 * at pass time they carry the TRAA jitter, exactly as the old GTAONode saw.
 */

import { DataTexture, RepeatWrapping, Vector3 } from 'three';
import type { PerspectiveCamera } from 'three';
import {
  Fn,
  If,
  Loop,
  PI,
  abs,
  acos,
  add,
  clamp,
  cos,
  cross,
  div,
  dot,
  float,
  floor,
  int,
  ivec2,
  mat3,
  max,
  mix,
  mul,
  normalize,
  pow,
  screenSize,
  sin,
  sqrt,
  sub,
  texture,
  textureLoad,
  textureSize,
  uniform,
  uv,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import type { Node } from 'three/webgpu';
import { runiform } from '../gpu/RenderUniform';
import type { NF, NI, NM4, NV2, NV3, NV4 } from '../gpu/TSLTypes';

export interface GtaoOptions {
  samples: number;
  radius: number;
  distanceFallOff: number;
  thickness?: number;
  distanceExponent?: number;
  scale?: number;
  /** PERF-4: skip the whole march for pixels past this view distance (m). The
   *  consumer fades AO to 1 with distance anyway (PostStack aoFaded), so beyond
   *  the fade end the marched result is discarded — gating here is output-
   *  equivalent and skips the 12–18 sample march on the far half of a vista. */
  maxDist?: number;
}

interface DepthTexLike {
  sample(uv: unknown): unknown;
  value: unknown;
}

/**
 * Equivalents of three's PostProcessingUtils getViewPosition/getScreenPosition
 * (and a local getNormalFromDepth port) that evaluate the mat4×vec4 product
 * ONCE. The stock versions reference the un-toVar'd product node for both the
 * xyz/xy read and the w read, so the matrix multiply is emitted twice; the
 * .toVar() here forces a single evaluation. Identical operations, deduplicated
 * ⇒ bit-identical output. WebGPU coordinate system only — this project renders
 * exclusively through three/webgpu, so only that branch of stock
 * getViewPosition is ever emitted.
 */
export function getViewPositionFast(screenPosition: Node, depth: Node, projInv: Node): NV3 {
  const sp = vec2((screenPosition as unknown as NV2).x, (screenPosition as unknown as NV2).y.oneMinus())
    .mul(2.0)
    .sub(1.0);
  const clip = vec4(vec3(sp, depth as unknown as NF), 1.0);
  const v = vec4(
    (projInv as unknown as NM4).mul(clip as unknown as NV4) as unknown as NV4,
  ).toVar();
  return v.xyz.div(v.w) as unknown as NV3;
}

export function getScreenPositionFast(viewPosition: Node, proj: Node): NV2 {
  const clip = vec4(
    (proj as unknown as NM4).mul(
      vec4(viewPosition as unknown as NV3, 1.0) as unknown as NV4,
    ) as unknown as NV4,
  ).toVar();
  const sampleUv = clip.xy.div(clip.w).mul(0.5).add(0.5).toVar();
  return vec2(sampleUv.x, sampleUv.y.oneMinus());
}

/**
 * PERF-P3 view-position reconstruction from a LINEAR view-Z (the half-res
 * DepthHalf source), skipping the per-tap inverse-projection unproject.
 *
 * Derivation (standard perspective, incl. TRAA-jittered P02/P12): with the
 * projection's third row = (0,0,-1,0), clip.w = -viewZ, so
 *   projInv · vec4(ndc.xy, 0, 1) = [ (ndc.x+P02)/P00, (ndc.y+P12)/P11, -1, · ]
 * and the true view position is that ray.xyz scaled by (1/w) = -viewZ. This is
 * algebraically identical to getViewPositionFast(uv, ndc.z) — it just consumes
 * viewZ directly instead of recovering it from ndc.z, so a tap needs only the
 * scalar viewZ texel (no mat4×vec4 per sample). viewZ is independent of ndc.xy,
 * but the reconstructed x/y are not — pass the tap's own uv.
 */
export function getViewPositionFromViewZ(screenPosition: Node, viewZ: Node, projInv: Node): NV3 {
  const sp = vec2((screenPosition as unknown as NV2).x, (screenPosition as unknown as NV2).y.oneMinus())
    .mul(2.0)
    .sub(1.0);
  const ray = vec4(
    (projInv as unknown as NM4).mul(vec4(vec3(sp, 0.0), 1.0) as unknown as NV4) as unknown as NV4,
  ).toVar();
  return ray.xyz.mul((viewZ as unknown as NF).negate()) as unknown as NV3;
}

/**
 * Port of stock getNormalFromDepth, re-sourced onto the half-res LINEAR view-Z
 * texture (PERF-P3). Same op structure — 3×3 texel stencil, pick the smoother
 * neighbour per axis, cross the two view-space edge vectors — but the samples
 * are half-res view-Z texels (nearest textureLoad, hardware-clamped) and the
 * view positions come from getViewPositionFromViewZ, so no per-sample unproject.
 *
 * The discontinuity metric (2·n1−n2−c0) is evaluated on linear view-Z instead
 * of ndc.z; it is still only a monotone selector for "which side is flatter",
 * so the choice is unchanged on planar surfaces. Consequence (honesty): the
 * derived normal is a HALF-RES central difference — one-texel-thin ridges the
 * full-res port could resolve are averaged out, softening AO on thin features.
 */
function getNormalFromViewZFast(uvN: Node, viewZTexture: Node, projInv: Node): NV3 {
  const size = textureSize(
    textureLoad(viewZTexture as unknown as Parameters<typeof textureLoad>[0]),
  );
  const sizeV = size as unknown as NV2;
  const p = ivec2((uvN as unknown as NV2).mul(sizeV)).toVar();
  const load = (coord?: unknown): NF =>
    (
      textureLoad(
        viewZTexture as unknown as Parameters<typeof textureLoad>[0],
        coord as unknown as Parameters<typeof textureLoad>[1],
      ) as unknown as NV4
    ).x as unknown as NF;

  const c0 = load(p).toVar();
  const l2 = load(p.sub(ivec2(2, 0))).toVar();
  const l1 = load(p.sub(ivec2(1, 0))).toVar();
  const r1 = load(p.add(ivec2(1, 0))).toVar();
  const r2 = load(p.add(ivec2(2, 0))).toVar();
  const b2 = load(p.add(ivec2(0, 2))).toVar();
  const b1 = load(p.add(ivec2(0, 1))).toVar();
  const t1 = load(p.sub(ivec2(0, 1))).toVar();
  const t2 = load(p.sub(ivec2(0, 2))).toVar();

  const dl = abs(sub(float(2).mul(l1).sub(l2), c0)).toVar();
  const dr = abs(sub(float(2).mul(r1).sub(r2), c0)).toVar();
  const db = abs(sub(float(2).mul(b1).sub(b2), c0)).toVar();
  const dt = abs(sub(float(2).mul(t1).sub(t2), c0)).toVar();

  const ce = getViewPositionFromViewZ(uvN, c0, projInv).toVar();
  const uv2 = uvN as unknown as NV2;
  const dpdx = dl
    .lessThan(dr)
    .select(
      ce.sub(getViewPositionFromViewZ(uv2.sub(vec2(float(1).div(sizeV.x), 0)), l1, projInv)),
      ce.negate().add(getViewPositionFromViewZ(uv2.add(vec2(float(1).div(sizeV.x), 0)), r1, projInv)),
    );
  const dpdy = db
    .lessThan(dt)
    .select(
      ce.sub(getViewPositionFromViewZ(uv2.add(vec2(0, float(1).div(sizeV.y))), b1, projInv)),
      ce.negate().add(getViewPositionFromViewZ(uv2.sub(vec2(0, float(1).div(sizeV.y))), t1, projInv)),
    );

  return normalize(cross(dpdx as unknown as NV3, dpdy as unknown as NV3)) as unknown as NV3;
}

/**
 * Builds the AO fragment expression for the merged half-res pass.
 * `resolution` is the live half-res dimensions uniform (HalfResMrtNode owns
 * it) — drives the noise tiling exactly like stock GTAONode's resolution.
 */
export function gtaoLayer(
  depthTex: DepthTexLike,
  viewZTex: DepthTexLike,
  camera: PerspectiveCamera,
  resolution: ReturnType<typeof uniform>,
  opts: GtaoOptions,
): NV4 {
  const uRadius = runiform(opts.radius);
  const uThickness = runiform(opts.thickness ?? 1);
  const uDistanceExponent = runiform(opts.distanceExponent ?? 1);
  const uDistanceFallOff = runiform(opts.distanceFallOff);
  const uScale = runiform(opts.scale ?? 1);
  // DIRECTIONS/STEPS are compile-time literals, NOT a runtime uniform trip
  // count: a uniform loop bound forces Tint to emit while(true) + a u64
  // loop-counter carry chain with no unroll and no cross-iteration CSE;
  // constant bounds let Metal unroll. Same derivation as stock GTAONode
  // (select(3,5) on the sample count; ceil-divide for steps) evaluated
  // JS-side ⇒ identical values.
  const DIRECTIONS = opts.samples < 30 ? 3 : 5;
  const STEPS = Math.floor((opts.samples + (DIRECTIONS - 1)) / DIRECTIONS);
  const uMaxDist = runiform(opts.maxDist ?? 1e9);
  // live object references — read current (jittered) values at upload time,
  // matching stock GTAONode's uniform(camera.projectionMatrix)
  const uProj = runiform(camera.projectionMatrix);
  const uProjInv = runiform(camera.projectionMatrixInverse);
  const noiseNode = texture(generateMagicSquareNoise());

  return Fn((): NV4 => {
    const uvNode = uv();
    // CENTER stays full-res (output contract §3): the returned viewPosition.z is
    // the bilateral-upsample depth guide and the sky/maxDist reject must be exact.
    const sampleDepth = (uvS: unknown): NF =>
      (depthTex.sample(uvS) as NV4).r as unknown as NF;
    // MARCH taps read the half-res LINEAR view-Z source (nearest, hw-clamped) and
    // reconstruct view position without a per-tap unproject (PERF-P3).
    const sampleViewZ = (uvS: unknown): NF =>
      (viewZTex.sample(uvS) as NV4).x as unknown as NF;

    const result = float(1).toVar();
    const depth = sampleDepth(uvNode).toVar();
    const viewPosition = getViewPositionFast(
      uvNode,
      depth,
      uProjInv as unknown as Node,
    ).toVar();

    // stock: depth.greaterThanEqual(1.0).discard() onto a white-cleared RT.
    // PERF-4: also skip past uMaxDist (the consumer fades AO→1 there, so the
    // marched value is discarded) — kills the far-vista march for free.
    If(depth.lessThan(1.0).and(viewPosition.length().lessThan(uMaxDist as unknown as NF)), () => {
      const viewNormal = getNormalFromViewZFast(
        uvNode,
        viewZTex.value as unknown as Node,
        uProjInv as unknown as Node,
      ).toVar();
      // own full-res depth texel — used to reject degenerate self-samples
      // (deviation from stock, see below; depth is drawing-buffer sized)
      const ownTexel = floor(uvNode.mul(screenSize)).toVar();

      const radiusToUse = uRadius;

      const noiseResolution = textureSize(noiseNode, int(0));
      const noiseUv = vec2(uvNode.x, uvNode.y.oneMinus()).mul(
        (resolution as unknown as NV2).div(noiseResolution as unknown as NV2),
      );

      const noiseTexel = noiseNode.sample(noiseUv) as unknown as NV4;
      const randomVec = noiseTexel.xyz.mul(2.0).sub(1.0);
      const tangent = vec3(randomVec.xy, 0.0).normalize();
      const bitangent = vec3(tangent.y.mul(-1.0), tangent.x, 0.0);
      const kernelMatrix = mat3(tangent, bitangent, vec3(0.0, 0.0, 1.0));

      const ao = float(0).toVar();

      Loop(
        { start: int(0), end: int(DIRECTIONS), type: 'int', condition: '<' },
        ({ i }: { readonly i: NI }) => {
          // stock adds _temporalDirection here — always 0 with temporal
          // filtering off (our configuration), omitted
          const angle = float(i).div(float(DIRECTIONS)).mul(PI).toVar();
          const sampleDir = vec4(
            cos(angle),
            sin(angle),
            0,
            add(0.5, mul(0.5, noiseTexel.w)),
          ).toVar();
          sampleDir.xyz = normalize(kernelMatrix.mul(sampleDir.xyz));

          const viewDir = normalize(viewPosition.xyz.negate()).toVar();
          const sliceBitangent = normalize(cross(sampleDir.xyz, viewDir)).toVar();
          const sliceTangent = cross(sliceBitangent, viewDir);
          const normalInSlice = normalize(
            viewNormal.sub(sliceBitangent.mul(dot(viewNormal, sliceBitangent))),
          );

          const tangentToNormalInSlice = cross(normalInSlice, sliceBitangent).toVar();
          const cosHorizons = vec2(
            dot(viewDir, tangentToNormalInSlice),
            dot(viewDir, tangentToNormalInSlice.negate()),
          ).toVar();

          Loop(
            // 'name' missing from @types LoopNodeObjectParameter but required
            // at runtime: it names the loop var AND the destructure key
            { end: int(STEPS), type: 'int', name: 'j', condition: '<' } as unknown as Parameters<typeof Loop>[0],
            (({ j }: { readonly j: NI }) => {
              const sampleViewOffset = sampleDir.xyz
                .mul(radiusToUse as unknown as NF)
                .mul(sampleDir.w)
                .mul(
                  pow(
                    div(float(j).add(1.0), float(STEPS)),
                    uDistanceExponent as unknown as NF,
                  ),
                );

              // x
              const sampleScreenPositionX = getScreenPositionFast(
                viewPosition.add(sampleViewOffset),
                uProj as unknown as Node,
              ).toVar();
              const sampleViewZX = sampleViewZ(sampleScreenPositionX).toVar();
              const sampleSceneViewPositionX = getViewPositionFromViewZ(
                sampleScreenPositionX,
                sampleViewZX,
                uProjInv as unknown as Node,
              ).toVar();
              const viewDeltaX = sampleSceneViewPositionX.sub(viewPosition).toVar();
              // sub-texel rejection (deviation from stock, horizon-black
              // fix): past a few hundred meters the world-space radius
              // projects below one depth texel — the sample lands on the
              // center's OWN texel, passes the thickness test with a
              // quantization-dominated direction (normalize(≈0)) and drives
              // cosHorizons → 1 = "fully occluded". A same-texel sample
              // carries no horizon information; near-field offsets span
              // many texels and are unaffected.
              const offTexelX = dot(
                abs(floor(sampleScreenPositionX.mul(screenSize)).sub(ownTexel)),
                vec2(1, 1),
              ).greaterThan(0.5);
              If(abs(viewDeltaX.z).lessThan(uThickness as unknown as NF).and(offTexelX), () => {
                const sampleCosHorizon = dot(viewDir, normalize(viewDeltaX));
                cosHorizons.x.addAssign(
                  max(
                    0,
                    mul(
                      sampleCosHorizon.sub(cosHorizons.x),
                      mix(
                        1.0,
                        float(2.0).div(float(j).add(2)),
                        uDistanceFallOff as unknown as NF,
                      ),
                    ),
                  ),
                );
              });

              // y
              const sampleScreenPositionY = getScreenPositionFast(
                viewPosition.sub(sampleViewOffset),
                uProj as unknown as Node,
              ).toVar();
              const sampleViewZY = sampleViewZ(sampleScreenPositionY).toVar();
              const sampleSceneViewPositionY = getViewPositionFromViewZ(
                sampleScreenPositionY,
                sampleViewZY,
                uProjInv as unknown as Node,
              ).toVar();
              const viewDeltaY = sampleSceneViewPositionY.sub(viewPosition).toVar();
              const offTexelY = dot(
                abs(floor(sampleScreenPositionY.mul(screenSize)).sub(ownTexel)),
                vec2(1, 1),
              ).greaterThan(0.5);
              If(abs(viewDeltaY.z).lessThan(uThickness as unknown as NF).and(offTexelY), () => {
                const sampleCosHorizon = dot(viewDir, normalize(viewDeltaY));
                cosHorizons.y.addAssign(
                  max(
                    0,
                    mul(
                      sampleCosHorizon.sub(cosHorizons.y),
                      mix(
                        1.0,
                        float(2.0).div(float(j).add(2)),
                        uDistanceFallOff as unknown as NF,
                      ),
                    ),
                  ),
                );
              });
            }) as unknown as Parameters<typeof Loop>[1],
          );

          // f32 guard (deviation from stock, which carries the hazard):
          // dot(viewDir, normalize(δ)) can read 1+ε at grazing → cos² > 1
          // → sqrt(negative) = NaN AO
          cosHorizons.assign(clamp(cosHorizons as unknown as NF, -1, 1) as unknown as NV2);
          // stock: sqrt(sub(1.0, cosHorizons*cosHorizons)) — oneMinus is the
          // same WGSL; @types sqrt is float-only, runtime handles vec2
          const sinHorizons = (
            sqrt(cosHorizons.mul(cosHorizons).oneMinus() as unknown as NF) as unknown as NV2
          ).toVar();
          const nx = dot(normalInSlice, sliceTangent);
          const ny = dot(normalInSlice, viewDir);
          const nxb = mul(
            0.5,
            acos(cosHorizons.y)
              .sub(acos(cosHorizons.x))
              .add(sinHorizons.x.mul(cosHorizons.x).sub(sinHorizons.y.mul(cosHorizons.y))),
          );
          const nyb = mul(
            0.5,
            sub(2.0, cosHorizons.x.mul(cosHorizons.x)).sub(
              cosHorizons.y.mul(cosHorizons.y),
            ),
          );
          const occlusion = nx.mul(nxb).add(ny.mul(nyb));
          ao.addAssign(occlusion);
        },
      );

      ao.assign(clamp(ao.div(float(DIRECTIONS)), 0, 1));
      ao.assign(pow(ao, uScale as unknown as NF));
      result.assign(ao);
    });

    // PERF-4: pack this half-res pixel's view-space z into .y so the full-res
    // bilateral upsample (PostStack aoFaded) reads the depth guide straight from
    // the AO texture — no per-tap full-res depth re-fetch + unproject. viewZ is
    // valid even on the maxDist/sky skip branches (result stays 1 there; a huge
    // |z| just makes those taps reject in the bilateral, which is correct).
    return vec4(result, viewPosition.z, 0, 1);
  })();
}

/**
 * Stock GTAONode noise — magic-square angles baked to a 5×5 repeat texture.
 * Copied verbatim from GTAONode.js (not exported there).
 */
function generateMagicSquareNoise(size = 5): DataTexture {
  const noiseSize = Math.floor(size) % 2 === 0 ? Math.floor(size) + 1 : Math.floor(size);
  const magicSquare = generateMagicSquare(noiseSize);
  const noiseSquareSize = magicSquare.length;
  const data = new Uint8Array(noiseSquareSize * 4);

  for (let inx = 0; inx < noiseSquareSize; ++inx) {
    const iAng = magicSquare[inx] ?? 0;
    const angle = (2 * Math.PI * iAng) / noiseSquareSize;
    const randomVec = new Vector3(Math.cos(angle), Math.sin(angle), 0).normalize();
    data[inx * 4] = (randomVec.x * 0.5 + 0.5) * 255;
    data[inx * 4 + 1] = (randomVec.y * 0.5 + 0.5) * 255;
    data[inx * 4 + 2] = 127;
    data[inx * 4 + 3] = 255;
  }

  const noiseTexture = new DataTexture(data, noiseSize, noiseSize);
  noiseTexture.name = 'gtaoNoiseTex';
  noiseTexture.wrapS = RepeatWrapping;
  noiseTexture.wrapT = RepeatWrapping;
  noiseTexture.needsUpdate = true;
  return noiseTexture;
}

function generateMagicSquare(size: number): number[] {
  const noiseSize = Math.floor(size) % 2 === 0 ? Math.floor(size) + 1 : Math.floor(size);
  const noiseSquareSize = noiseSize * noiseSize;
  const magicSquare = Array<number>(noiseSquareSize).fill(0);
  let i = Math.floor(noiseSize / 2);
  let j = noiseSize - 1;

  for (let num = 1; num <= noiseSquareSize; ) {
    if (i === -1 && j === noiseSize) {
      j = noiseSize - 2;
      i = 0;
    } else {
      if (j === noiseSize) j = 0;
      if (i < 0) i = noiseSize - 1;
    }
    if (magicSquare[i * noiseSize + j] !== 0) {
      j -= 2;
      i++;
      continue;
    } else {
      magicSquare[i * noiseSize + j] = num++;
    }
    j++;
    i--;
  }

  return magicSquare;
}
