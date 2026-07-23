/**
 * Deprecated periodic Sannikov/GCAR ground-cover renderer.
 *
 * This compatibility lane is deliberately isolated from the exterior
 * boundary-transfer renderer. It is behavior-frozen and will be removed after
 * the replacement passes visual and performance acceptance.
 */

import {
  HalfFloatType,
  NearestFilter,
  RGBAFormat,
  type PerspectiveCamera,
} from 'three';
import { StorageTexture, type Renderer } from 'three/webgpu';
import { texture, uint, vec2 } from 'three/tsl';
import type { NU, NV2, NV4 } from '../../gpu/TSLTypes';
import { dispatch, toF, uniformF } from '../Tsl';
import {
  GROUND_COVER_PROFILE_COUNT,
} from '../groundcover/GroundCoverProfiles';
import type {
  LegacyPeriodicGroundCoverBuildOpts,
  LegacyPeriodicGroundCoverField,
} from './LegacyPeriodicGroundCoverContracts';
import { createLegacyPeriodicGuideField } from './LegacyPeriodicGuideField';
import { createLegacyPeriodicProfileSamplers } from './LegacyPeriodicProfileSampler';
import { createLegacyPeriodicRayAtlas } from './LegacyPeriodicRayAtlas';
import { createLegacyPeriodicRayQuery } from './LegacyPeriodicRayQuery';

export { GRASS_FAR_BASE } from './LegacyPeriodicGroundCoverPolicy';
export type {
  LegacyPeriodicGroundCoverBuildOpts,
  LegacyPeriodicGroundCoverField,
} from './LegacyPeriodicGroundCoverContracts';

/** @deprecated Use the exterior boundary-transfer renderer for new ground cover. */
export function buildLegacyPeriodicGroundCoverField(
  options: LegacyPeriodicGroundCoverBuildOpts,
): LegacyPeriodicGroundCoverField {
  const { cam, vis, field, periodicProfiles } = options;
  if (field.hasGroundCoverClosure) {
    const arrayTexture = periodicProfiles[0]?.texture ?? null;
    const canonical = periodicProfiles.length === GROUND_COVER_PROFILE_COUNT
      && periodicProfiles.every((profile) => (
        profile.texture === arrayTexture && profile.textureLayer === profile.profileId
      ));
    const isolatedStandalone = options.forcedProfileId !== null
      && options.forcedProfileId !== undefined
      && periodicProfiles.length === 1
      && periodicProfiles[0]?.profileId === options.forcedProfileId
      && periodicProfiles[0]?.textureLayer === null;
    if (!canonical && !isolatedStandalone) {
      throw new Error(
        'legacy periodic ground cover requires one canonical GCAR array with layer == profile id',
      );
    }
  }

  const enabledUniform = uniformF(1);
  let enabled = true;
  const streamed = new URLSearchParams(window.location.search).get('src') === 'estonia';
  const guide = createLegacyPeriodicGuideField({
    cam,
    field,
    canopyTexture: options.canopyTex,
    enabledUniform,
    streamed,
  });
  const rayAtlas = createLegacyPeriodicRayAtlas();
  const periodicSamplers = createLegacyPeriodicProfileSamplers(periodicProfiles);
  const rayNormalTexture = new StorageTexture(cam.width, cam.height);
  rayNormalTexture.type = HalfFloatType;
  rayNormalTexture.format = RGBAFormat;
  rayNormalTexture.magFilter = NearestFilter;
  rayNormalTexture.minFilter = NearestFilter;
  rayNormalTexture.generateMipmaps = false;
  rayNormalTexture.name = 'legacyPeriodicGroundCoverRayNormal';

  const rayKernel = createLegacyPeriodicRayQuery({
    cam,
    vis,
    field,
    enabledUniform,
    streamed,
    guide,
    rayAtlas,
    periodicProfiles,
    periodicSamplers,
    rayNormalTexture,
    forcedProfileId: options.forcedProfileId ?? null,
  });
  const resolveRay = (pixel: NU): NV4 => {
    const uv = vec2(
      toF(pixel.mod(uint(cam.width))).add(0.5).div(cam.width),
      toF(pixel.div(uint(cam.width))).add(0.5).div(cam.height),
    ) as unknown as NV2;
    return texture(rayNormalTexture, uv, 0) as unknown as NV4;
  };

  return Object.freeze({
    batch: Object.freeze([]),
    renderHw(renderer: Renderer, camera: PerspectiveCamera): void {
      if (!enabled) return;
      guide.update(camera);
      dispatch(renderer, guide.kernel);
      dispatch(renderer, rayKernel);
    },
    resolveRay,
    authoredPackedColor: periodicSamplers?.color !== null,
    setEnabled(value: boolean): void {
      enabled = value;
      enabledUniform.value = value ? 1 : 0;
    },
    enabled: (): boolean => enabled,
    readCounts: async (_renderer: Renderer): Promise<{ clumps: number; hwTris: number }> => ({
      clumps: 0,
      hwTris: 0,
    }),
  });
}
