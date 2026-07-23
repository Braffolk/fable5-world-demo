/**
 * Material decode for the deprecated periodic O(1) ground-cover lane.
 *
 * This module owns only profile/material interpretation. The Nanite resolve
 * supplies its already reconstructed pixel, world position, camera position,
 * and shared shading accumulators; this builder neither reconstructs depth nor
 * evaluates lighting, shadows, or GI.
 */

import { Vector4 } from 'three';
import type { StorageTexture } from 'three/webgpu';
import {
  If,
  dot,
  float,
  floor,
  fract,
  mix,
  normalize,
  sin,
  smoothstep,
  uint,
  vec2,
  vec3,
} from 'three/tsl';
import type { NF, NB, NU, NV2, NV3, NV4 } from '../../gpu/TSLTypes';
import { canopyAt } from '../../gpu/passes/Scatter';
import type { TerrainField } from '../world/TerrainField';
import { GRASS_FAR_BASE } from './LegacyPeriodicGroundCoverPolicy';
import { GROUND_COVER_ID_MASK, GroundCoverId } from '../groundcover/GroundCoverTypes';
import {
  GROUND_COVER_PROFILE_FUNCTIONAL_IDS,
  GroundCoverProfileId,
} from '../groundcover/GroundCoverProfiles';
import { toF, uniformArrV4 } from '../Tsl';

const MATERIAL_STRIDE = 5;

export interface LegacyPeriodicGroundCoverSource {
  ray(pixelIndex: NU): NV4;
  readonly authoredPackedColor?: boolean;
}

export interface LegacyPeriodicGroundCoverShadeOptions {
  source: LegacyPeriodicGroundCoverSource;
  field: TerrainField;
  canopyTex: StorageTexture | null;
}

export interface LegacyPeriodicGroundCoverPixel {
  isGroundCover: NB;
  packedElectionId: NU;
  pixelIndex: NU;
  worldPosition: NV3;
  cameraPosition: NV3;
  albedo: NV3;
  worldNormal: NV3;
  ao: NF;
  backlightStrength: NF;
}

export interface LegacyPeriodicGroundCoverShadeResult {
  tip: NF;
  translucency: NF;
  typeColor: NV3 | null;
}

export interface LegacyPeriodicGroundCoverShade {
  readonly typeDebug: boolean;
  shade(pixel: LegacyPeriodicGroundCoverPixel): LegacyPeriodicGroundCoverShadeResult;
}

/** Five vec4s per exact native profile (legacy functional fixtures live at 16+id):
 *  0 fresh-base.rgb / near normal-pull
 *  1 fresh-tip.rgb  / base AO
 *  2 dry-base.rgb   / tip translucency
 *  3 dry-tip.rgb    / canopy darkening
 *  4 canopy.rgb     / reserved
 * A uniform table keeps cost independent of authored species count. */
function materialTable(grassNearPull: number): Vector4[] {
  const rows = Array.from(
    { length: 64 * MATERIAL_STRIDE },
    () => new Vector4(0, 0, 0, 0),
  );
  const put = (
    id: number,
    freshBase: readonly [number, number, number],
    freshTip: readonly [number, number, number],
    dryBase: readonly [number, number, number],
    dryTip: readonly [number, number, number],
    canopy: readonly [number, number, number],
    normalPull: number,
    aoBase: number,
    translucency: number,
    canopyDarken: number,
  ): void => {
    const base = id * MATERIAL_STRIDE;
    rows[base] = new Vector4(...freshBase, normalPull);
    rows[base + 1] = new Vector4(...freshTip, aoBase);
    rows[base + 2] = new Vector4(...dryBase, translucency);
    rows[base + 3] = new Vector4(...dryTip, canopyDarken);
    rows[base + 4] = new Vector4(...canopy, 0);
  };
  const legacy = (functionalId: number): number => 16 + functionalId;
  put(legacy(GroundCoverId.Grass),
    [0.02, 0.062, 0.011], [0.065, 0.148, 0.028],
    [0.085, 0.07, 0.024], [0.21, 0.17, 0.075],
    [0.018, 0.052, 0.014],
    grassNearPull, 0.45, 0.09, 0.55);
  put(legacy(GroundCoverId.Moss),
    [0.025, 0.065, 0.008], [0.18, 0.34, 0.035],
    [0.11, 0.075, 0.018], [0.42, 0.27, 0.045],
    [0.018, 0.05, 0.008],
    0.1, 0.28, 0.13, 0.32);
  put(legacy(GroundCoverId.Sedge),
    [0.025, 0.055, 0.01], [0.14, 0.2, 0.03],
    [0.12, 0.09, 0.025], [0.33, 0.25, 0.08],
    [0.018, 0.045, 0.009],
    0.14, 0.38, 0.1, 0.45);
  put(legacy(GroundCoverId.Lichen),
    [0.1, 0.12, 0.08], [0.32, 0.38, 0.25],
    [0.16, 0.14, 0.1], [0.46, 0.4, 0.28],
    [0.08, 0.1, 0.07],
    0.3, 0.6, 0.02, 0.24);
  put(legacy(GroundCoverId.Forb),
    [0.015, 0.05, 0.012], [0.06, 0.17, 0.04],
    [0.07, 0.055, 0.02], [0.19, 0.13, 0.05],
    [0.012, 0.04, 0.01],
    0.12, 0.4, 0.07, 0.5);
  put(legacy(GroundCoverId.DwarfShrub),
    [0.012, 0.035, 0.008], [0.045, 0.11, 0.025],
    [0.055, 0.035, 0.012], [0.13, 0.085, 0.03],
    [0.01, 0.03, 0.006],
    0.16, 0.35, 0.05, 0.48);

  // Until every species has a measured material, inherit only the broad
  // functional response. Geometry/profile identity remains exact.
  GROUND_COVER_PROFILE_FUNCTIONAL_IDS.forEach((functionalId, profileId) => {
    const source = legacy(functionalId) * MATERIAL_STRIDE;
    const target = profileId * MATERIAL_STRIDE;
    for (let i = 0; i < MATERIAL_STRIDE; i++) {
      rows[target + i] = rows[source + i]!.clone();
    }
  });
  put(GroundCoverProfileId.SphagnumCapillifolium,
    [0.15, 0.3, 0.038], [0.22, 0.42, 0.055],
    [0.27, 0.18, 0.048], [0.46, 0.3, 0.07],
    [0.09, 0.19, 0.025],
    0.08, 0.8, 0.16, 0.25);
  return rows;
}

/** Smooth, world-anchored dryness/brightness field with no hard patch cells. */
function patchField(xz: NV2): NV2 {
  const p = vec2(
    (xz.x as unknown as NF).mul(0.766).sub((xz.y as unknown as NF).mul(0.643)),
    (xz.x as unknown as NF).mul(0.643).add((xz.y as unknown as NF).mul(0.766)),
  ).mul(1 / 1.6) as unknown as NV2;
  const ip = floor(p) as unknown as NV2;
  const fp = fract(p) as unknown as NV2;
  const u = fp.mul(fp).mul(fp.mul(-2).add(3)) as unknown as NV2;
  const hash = (cell: NV2): NV2 =>
    fract(
      sin(
        vec2(
          dot(cell as unknown as NV3, vec2(127.1, 311.7) as unknown as NV3),
          dot(cell as unknown as NV3, vec2(269.5, 183.3) as unknown as NV3),
        ),
      ).mul(vec2(43758.5453, 28461.7331) as unknown as NV2),
    ) as unknown as NV2;
  const lower = mix(
    hash(ip),
    hash(ip.add(vec2(1, 0)) as unknown as NV2),
    u.x,
  ) as unknown as NV2;
  const upper = mix(
    hash(ip.add(vec2(0, 1)) as unknown as NV2),
    hash(ip.add(vec2(1, 1)) as unknown as NV2),
    u.x,
  ) as unknown as NV2;
  return mix(lower, upper, u.y) as unknown as NV2;
}

export function createLegacyPeriodicGroundCoverShade(
  options: LegacyPeriodicGroundCoverShadeOptions,
): LegacyPeriodicGroundCoverShade {
  const query = new URLSearchParams(window.location.search);
  const nearPullRaw = Number(query.get('grassnrmpull') ?? '0.18');
  const nearPull = Number.isFinite(nearPullRaw)
    ? Math.max(0, Math.min(1, nearPullRaw))
    : 0.18;
  const materials = uniformArrV4(materialTable(nearPull));
  const typeDebug = query.get('groundcoverdbg') === 'type';
  const flatResolve = query.get('grassdbg') === 'flatres';

  return {
    typeDebug,
    shade(pixel): LegacyPeriodicGroundCoverShadeResult {
      const tip = float(0.5).toVar() as unknown as NF;
      const translucency = float(0).toVar() as unknown as NF;
      const coverId = uint(GroundCoverId.Grass).toVar() as unknown as NU;
      const profileId = uint(GroundCoverProfileId.AgrostisCapillaris).toVar() as unknown as NU;
      const materialId = uint(16 + GroundCoverId.Grass).toVar() as unknown as NU;
      const typeColor = typeDebug
        ? (vec3(0.12, 0.42, 0.05).toVar() as unknown as NV3)
        : null;

      If(pixel.isGroundCover, () => {
        // The voxel-tier seed must not leak into procedural cover. Its own
        // profile-indexed tip transmission is accumulated after common light.
        pixel.backlightStrength.assign(float(0));
        const body = pixel.packedElectionId.bitAnd(uint(0x3fffffff));
        if (options.field.hasGroundCoverClosure) {
          profileId.assign(body.bitAnd(uint(options.source.authoredPackedColor ? 0xf : 0xff)));
          let functional: NU = uint(
            GROUND_COVER_PROFILE_FUNCTIONAL_IDS[
              GROUND_COVER_PROFILE_FUNCTIONAL_IDS.length - 1
            ]!,
          ) as unknown as NU;
          for (
            let id = GROUND_COVER_PROFILE_FUNCTIONAL_IDS.length - 2;
            id >= 0;
            id--
          ) {
            functional = (profileId.equal(uint(id)) as unknown as {
              select(a: unknown, b: unknown): NU;
            }).select(uint(GROUND_COVER_PROFILE_FUNCTIONAL_IDS[id]!), functional);
          }
          coverId.assign(functional);
          materialId.assign(profileId);
        } else {
          coverId.assign(body.bitAnd(uint(GROUND_COVER_ID_MASK)));
          profileId.assign(coverId);
          materialId.assign(coverId.add(uint(16)));
        }

        if (typeDebug) {
          const debugId = options.field.hasGroundCoverClosure ? profileId : coverId;
          const colors = options.field.hasGroundCoverClosure
            ? [
                [0.12, 0.42, 0.05], [0.12, 0.68, 0.46], [0.10, 0.38, 0.82],
                [0.92, 0.68, 0.08], [0.95, 0.92, 0.72], [0.42, 0.78, 0.08],
                [0.20, 0.58, 0.26], [0.68, 0.76, 0.58], [0.76, 0.18, 0.62],
                [0.82, 0.34, 0.22], [0.42, 0.21, 0.08], [0.62, 0.20, 0.74],
              ]
            : [
                [0.12, 0.42, 0.05], [0.42, 0.78, 0.08], [0.92, 0.68, 0.08],
                [0.68, 0.76, 0.58], [0.76, 0.18, 0.62], [0.42, 0.21, 0.08],
              ];
          let selected = vec3(...colors[0]!) as unknown as NV3;
          for (let id = 1; id < colors.length; id++) {
            selected = (debugId.equal(uint(id)) as unknown as {
              select(a: NV3, b: NV3): NV3;
            }).select(vec3(...colors[id]!) as unknown as NV3, selected);
          }
          (typeColor as unknown as { assign(value: NV3): void }).assign(selected);
          pixel.albedo.assign(selected);
          pixel.worldNormal.assign(vec3(0, 1, 0) as unknown as NV3);
          pixel.ao.assign(float(1));
          tip.assign(float(0.5));
          return;
        }
        if (flatResolve) {
          pixel.albedo.assign(vec3(0.05, 0.12, 0.03) as unknown as NV3);
          pixel.worldNormal.assign(vec3(0, 1, 0) as unknown as NV3);
          pixel.ao.assign(float(1));
          tip.assign(float(0.5));
          return;
        }

        // One screen-lane read provides both the authored normal and height.
        const ray = options.source.ray(pixel.pixelIndex);
        const height = ray.w as unknown as NF;
        const authoredNormal = normalize(ray.xyz as unknown as NV3) as unknown as NV3;
        const distance = pixel.worldPosition.sub(pixel.cameraPosition).length();
        const toCamera = normalize(
          pixel.cameraPosition.sub(pixel.worldPosition),
        ) as unknown as NV3;
        const facingNormal = dot(authoredNormal, toCamera).lessThan(0)
          .select(authoredNormal.negate(), authoredNormal) as unknown as NV3;
        const terrainNormal = options.field
          .fieldNormalSlopeHot(pixel.worldPosition.xz as unknown as NV2)
          .xyz as unknown as NV3;
        const base = materialId.mul(uint(MATERIAL_STRIDE));
        const m0 = materials.element(base);
        const m1 = materials.element(base.add(uint(1)));
        const m2 = materials.element(base.add(uint(2)));
        const m3 = materials.element(base.add(uint(3)));
        const m4 = materials.element(base.add(uint(4)));
        const isFar = options.field.hasGroundCoverClosure
          ? body.lessThan(uint(0))
          : body.greaterThanEqual(uint(GRASS_FAR_BASE));
        const terrainPull = isFar.select(
          float(1),
          smoothstep(8, 70, distance).mul(0.47).add(m0.w),
        ) as unknown as NF;
        const fresh = mix(m0.xyz, m1.xyz, height.mul(height)) as unknown as NV3;
        const dry = mix(m2.xyz, m3.xyz, height) as unknown as NV3;
        const patch = patchField(
          pixel.worldPosition.xz as unknown as NV2,
        ).toVar() as unknown as NV2;
        const canopy = (options.canopyTex
          ? canopyAt(options.canopyTex as StorageTexture, pixel.worldPosition.xz as unknown as NV2)
          : float(0)) as unknown as NF;
        const dryness = smoothstep(0.64, 0.82, patch.x as unknown as NF)
          .mul(float(1).sub(canopy.mul(0.85))) as unknown as NF;
        let color: NV3;
        if (options.source.authoredPackedColor) {
          const packed = body.shiftRight(uint(4)).bitAnd(uint(0xff_ffff));
          color = vec3(
            toF(packed.shiftRight(uint(16)).bitAnd(uint(0xff))).div(255),
            toF(packed.shiftRight(uint(8)).bitAnd(uint(0xff))).div(255),
            toF(packed.bitAnd(uint(0xff))).div(255),
          ) as unknown as NV3;
        } else {
          color = mix(fresh, dry, dryness) as unknown as NV3;
          color = color.mul(
            (patch.y as unknown as NF).sub(0.5).mul(0.4).add(1),
          ) as unknown as NV3;
          color = mix(color, m4.xyz, canopy.mul(m3.w)) as unknown as NV3;
        }
        pixel.albedo.assign(color);
        pixel.worldNormal.assign(
          normalize(mix(facingNormal, terrainNormal, terrainPull)) as unknown as NV3,
        );
        pixel.ao.assign(
          smoothstep(0, 0.55, height)
            .mul(float(1).sub(m1.w))
            .add(m1.w) as unknown as NF,
        );
        tip.assign(height);
        translucency.assign(height.mul(m2.w));
      });

      return { tip, translucency, typeColor };
    },
  };
}
