/**
 * Exterior boundary-transfer fragment resolve.
 *
 * This draw lives in the existing scene render pass and reuses the Nanite
 * fullscreen-triangle geometry supplied by its caller. It allocates no
 * screen-sized handoff, storage buffer, pass, barrier, dispatch, or runtime
 * ground-cover geometry. The fragment performs exactly one boundary query;
 * colour and hardware depth consume that same expression graph.
 */

import { Mesh, type BufferGeometry } from 'three';
import { NodeMaterial } from 'three/webgpu';
import {
  Discard,
  Fn,
  If,
  atan,
  depth,
  dot,
  float,
  max,
  mix,
  normalize,
  positionGeometry,
  screenCoordinate,
  uint,
  vec3,
  vec4,
} from 'three/tsl';
import type { NB, NF, NU, NV2, NV3, NV4 } from '../../gpu/TSLTypes';
import { sunU } from '../../render/VegMaterials';
import type { NaniteCam } from '../NaniteCommon';
import type { NaniteVisBuffers } from '../raster/NaniteRaster';
import { elemU, toF } from '../Tsl';
import type {
  GroundCoverBoundaryQueryInput,
  GroundCoverBoundaryQueryResult,
} from './GroundCoverBoundaryRuntime';

export type GroundCoverBoundaryQuery = (
  input: GroundCoverBoundaryQueryInput,
) => GroundCoverBoundaryQueryResult;

export interface GroundCoverBoundaryResolveOptions {
  /** Reuse an existing fullscreen-triangle geometry; this builder creates no
   * vertex/index allocation of its own. */
  geometry: BufferGeometry;
  cam: NaniteCam;
  vis: NaniteVisBuffers;
  query: GroundCoverBoundaryQuery;
}

/** Build the one direct boundary-transfer draw. The caller adds the returned
 * mesh to the existing scene render pass. */
export function buildGroundCoverBoundaryResolve(
  options: GroundCoverBoundaryResolveOptions,
): Mesh {
  const { geometry, cam, vis, query } = options;
  const material = new NodeMaterial();
  material.name = 'groundCoverBoundaryResolve';
  material.vertexNode = vec4(positionGeometry.xy, 0, 1) as unknown as typeof material.vertexNode;

  const evaluateBoundary = (): {
    result: GroundCoverBoundaryQueryResult;
    rayOrigin: NV3;
    rayDirection: NV3;
  } => {
    const fy = float(cam.uH).sub(screenCoordinate.y);
    const x = uint(screenCoordinate.x).toVar() as unknown as NU;
    const y = uint(fy).toVar() as unknown as NU;
    const pixelIndex = y.mul(uint(cam.uW)).add(x).toVar() as unknown as NU;
    const ndcX = toF(x).add(0.5).div(float(cam.uW)).mul(2).sub(1) as unknown as NF;
    const ndcY = toF(y).add(0.5).div(float(cam.uH)).mul(2).sub(1) as unknown as NF;
    const farH = cam.invVp.mul(vec4(ndcX, ndcY, 1, 1)) as unknown as NV4;
    const rayOrigin = vec3(cam.camPos).toVar() as unknown as NV3;
    const rayDirection = normalize(
      farH.xyz.div(farH.w).sub(rayOrigin),
    ) as unknown as NV3;
    const elect = elemU(vis.payloadV.ro, pixelIndex).toVar() as unknown as NU;
    const hasScene = elect.notEqual(uint(0)).toVar() as unknown as NB;
    const sceneDistance = float(1e9).toVar() as unknown as NF;
    const preferredAnchor = (
      rayOrigin.xz as unknown as { toVar(): NV2 }
    ).toVar() as unknown as NV2;
    If(hasScene, () => {
      const sceneDepth = float(1).sub(
        toF(elect.shiftRight(uint(8))).div(16777215),
      ) as unknown as NF;
      const sceneH = cam.invVp.mul(
        vec4(ndcX, ndcY, sceneDepth, 1),
      ) as unknown as NV4;
      const scenePoint = sceneH.xyz.div(sceneH.w).toVar() as unknown as NV3;
      sceneDistance.assign(scenePoint.sub(rayOrigin).length());
      preferredAnchor.assign(scenePoint.xz);
    });
    const input: GroundCoverBoundaryQueryInput = {
      rayOrigin,
      rayDirection,
      sceneCutoffMetres: sceneDistance,
      hasSceneCutoff: hasScene,
      preferredAnchorXZ: preferredAnchor,
      pixelAngleRadians: float(2)
        .mul(atan(float(1).div(cam.cotHalfFov)))
        .div(float(cam.uH)) as unknown as NF,
    };
    return { result: query(input), rayOrigin, rayDirection };
  };

  material.fragmentNode = Fn(() => {
    const evaluated = evaluateBoundary();
    const result = evaluated.result;

    // One query feeds both colour and fragment depth. A separate depthNode
    // would duplicate every terminal query read in the generated fragment.
    // Mixed appearance owns an interval rather than a representative surface;
    // its far endpoint is the conservative hardware-depth choice.
    const hitDistance = (result.regularHit as unknown as {
      select(a: unknown, b: unknown): NF;
    }).select(
      result.regularDistance,
      (result.mixedHit as unknown as { select(a: unknown, b: unknown): NF })
        .select(result.mixedDistanceFar, float(1e9)),
    ) as unknown as NF;
    const depthPoint = evaluated.rayOrigin.add(
      evaluated.rayDirection.mul(hitDistance),
    ) as unknown as NV3;
    const depthClip = cam.vp.mul(vec4(depthPoint, 1)) as unknown as NV4;
    depth.assign(
      (depthClip.z as unknown as NF).div(
        (depthClip.w as unknown as NF).max(1e-6),
      ),
    );

    // Both record kinds obey the same guarded-carrier interior fade. A miss
    // contributes exactly zero; there is no fallback surface or scene colour.
    const regularGate = (result.regularHit as unknown as {
      select(a: unknown, b: unknown): NF;
    }).select(result.laneFade, float(0)).toVar() as unknown as NF;
    const mixedGate = (result.mixedHit as unknown as {
      select(a: unknown, b: unknown): NF;
    }).select(result.laneFade, float(0)).toVar() as unknown as NF;

    const hit = evaluated.rayOrigin.add(
      evaluated.rayDirection.mul(result.regularDistance),
    ) as unknown as NV3;
    const toCamera = normalize(evaluated.rayOrigin.sub(hit)) as unknown as NV3;
    const faceNormal = dot(result.regularNormal, toCamera).lessThan(0)
      .select(result.regularNormal.negate(), result.regularNormal) as unknown as NV3;
    const sunDir = normalize(vec3(sunU.dir)) as unknown as NV3;
    const direct = max(dot(faceNormal, sunDir), 0) as unknown as NF;
    const ambientUp = faceNormal.y.mul(0.5).add(0.5).clamp(0, 1) as unknown as NF;
    const ambient = mix(
      vec3(0.18, 0.16, 0.12),
      vec3(0.4, 0.5, 0.62),
      ambientUp,
    ).mul(0.5 * Math.PI) as unknown as NV3;
    const irradiance = max(
      (sunU.color as unknown as NV3).mul(float(sunU.intensity)).mul(direct),
      ambient,
    ) as unknown as NV3;
    const regularPremul = result.regularAlbedo
      .mul(irradiance)
      .mul(1 / Math.PI)
      .mul(regularGate) as unknown as NV3;

    // Mixed records are already integrated appearance. They have no
    // representative surface normal and never enter surface lighting.
    const mixedPremul = result.mixedPremulRadiance
      .mul(mixedGate) as unknown as NV3;
    const alpha = regularGate
      .add(result.mixedCoverage.mul(mixedGate))
      .clamp(0, 1)
      .toVar() as unknown as NF;
    If(alpha.lessThanEqual(0), () => {
      Discard();
    });
    return vec4(regularPremul.add(mixedPremul), alpha) as unknown as NV4;
  })() as unknown as typeof material.fragmentNode;

  material.transparent = true;
  material.premultipliedAlpha = true;
  material.depthTest = true;
  material.depthWrite = false;
  material.fog = false;
  material.lights = false;

  const mesh = new Mesh(geometry, material);
  mesh.name = 'groundCoverBoundaryResolve';
  mesh.frustumCulled = false;
  // Transparent objects are rendered after opaque geometry. This exact order
  // preserves depth testing against Nanite and ordinary scene geometry.
  mesh.renderOrder = 10_000;
  mesh.castShadow = false;
  mesh.receiveShadow = false;
  return mesh;
}
