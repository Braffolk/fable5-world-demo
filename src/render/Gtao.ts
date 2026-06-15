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
  getNormalFromDepth,
  getScreenPosition,
  getViewPosition,
  int,
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
  textureSize,
  uniform,
  uv,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { runiform } from '../gpu/RenderUniform';
import type { NF, NI, NV2, NV3, NV4 } from '../gpu/TSLTypes';

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
 * Builds the AO fragment expression for the merged half-res pass.
 * `resolution` is the live half-res dimensions uniform (HalfResMrtNode owns
 * it) — drives the noise tiling exactly like stock GTAONode's resolution.
 */
export function gtaoLayer(
  depthTex: DepthTexLike,
  camera: PerspectiveCamera,
  resolution: ReturnType<typeof uniform>,
  opts: GtaoOptions,
): NV4 {
  const uRadius = runiform(opts.radius);
  const uThickness = runiform(opts.thickness ?? 1);
  const uDistanceExponent = runiform(opts.distanceExponent ?? 1);
  const uDistanceFallOff = runiform(opts.distanceFallOff);
  const uScale = runiform(opts.scale ?? 1);
  const uSamples = runiform(opts.samples);
  const uMaxDist = runiform(opts.maxDist ?? 1e9);
  // live object references — read current (jittered) values at upload time,
  // matching stock GTAONode's uniform(camera.projectionMatrix)
  const uProj = runiform(camera.projectionMatrix);
  const uProjInv = runiform(camera.projectionMatrixInverse);
  const noiseNode = texture(generateMagicSquareNoise());

  return Fn((): NV4 => {
    const uvNode = uv();
    const sampleDepth = (uvS: unknown): NF =>
      (depthTex.sample(uvS) as NV4).r as unknown as NF;

    const result = float(1).toVar();
    const depth = sampleDepth(uvNode).toVar();
    const viewPosition = getViewPosition(
      uvNode,
      depth,
      uProjInv as unknown as Parameters<typeof getViewPosition>[2],
    ).toVar();

    // stock: depth.greaterThanEqual(1.0).discard() onto a white-cleared RT.
    // PERF-4: also skip past uMaxDist (the consumer fades AO→1 there, so the
    // marched value is discarded) — kills the far-vista march for free.
    If(depth.lessThan(1.0).and(viewPosition.length().lessThan(uMaxDist as unknown as NF)), () => {
      const viewNormal = (
        getNormalFromDepth(
          uvNode,
          depthTex.value as Parameters<typeof getNormalFromDepth>[1],
          uProjInv as unknown as Parameters<typeof getNormalFromDepth>[2],
        ) as unknown as NV3
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

      const DIRECTIONS = (uSamples as unknown as NF).lessThan(30).select(3, 5).toVar();
      const STEPS = add((uSamples as unknown as NF), DIRECTIONS.sub(1)).div(DIRECTIONS).toVar();

      const ao = float(0).toVar();

      Loop(
        { start: int(0), end: DIRECTIONS as unknown as NI, type: 'int', condition: '<' },
        ({ i }: { readonly i: NI }) => {
          // stock adds _temporalDirection here — always 0 with temporal
          // filtering off (our configuration), omitted
          const angle = float(i).div(float(DIRECTIONS as unknown as NF)).mul(PI).toVar();
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
            { end: STEPS as unknown as NI, type: 'int', name: 'j', condition: '<' } as unknown as Parameters<typeof Loop>[0],
            (({ j }: { readonly j: NI }) => {
              const sampleViewOffset = sampleDir.xyz
                .mul(radiusToUse as unknown as NF)
                .mul(sampleDir.w)
                .mul(
                  pow(
                    div(float(j).add(1.0), float(STEPS as unknown as NF)),
                    uDistanceExponent as unknown as NF,
                  ),
                );

              // x
              const sampleScreenPositionX = getScreenPosition(
                viewPosition.add(sampleViewOffset),
                uProj as unknown as Parameters<typeof getScreenPosition>[1],
              ).toVar();
              const sampleDepthX = sampleDepth(sampleScreenPositionX).toVar();
              const sampleSceneViewPositionX = getViewPosition(
                sampleScreenPositionX,
                sampleDepthX,
                uProjInv as unknown as Parameters<typeof getViewPosition>[2],
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
              const sampleScreenPositionY = getScreenPosition(
                viewPosition.sub(sampleViewOffset),
                uProj as unknown as Parameters<typeof getScreenPosition>[1],
              ).toVar();
              const sampleDepthY = sampleDepth(sampleScreenPositionY).toVar();
              const sampleSceneViewPositionY = getViewPosition(
                sampleScreenPositionY,
                sampleDepthY,
                uProjInv as unknown as Parameters<typeof getViewPosition>[2],
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

      ao.assign(clamp(ao.div(DIRECTIONS as unknown as NF), 0, 1));
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
