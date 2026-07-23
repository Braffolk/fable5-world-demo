import type { StorageTexture } from 'three/webgpu';
import {
  Fn, If, atan, atomicMax, atomicStore, float, instanceIndex, mix, normalize, smoothstep,
  texture, texture3D, textureStore, uint, uvec2, vec2, vec3, vec4,
} from 'three/tsl';
import type { NB, NF, NU, NV2, NV3, NV4 } from '../../gpu/TSLTypes';
import { cellHash } from '../../gpu/passes/Scatter';
import type { NaniteCam } from '../NaniteCommon';
import type { NaniteVisBuffers } from '../raster/NaniteRaster';
import type { TerrainField } from '../world/TerrainField';
import {
  GROUND_COVER_PROFILE_COUNT,
  GROUND_COVER_PROFILE_FUNCTIONAL_IDS,
  type LoadedPeriodicProfile,
} from '../groundcover/GroundCoverProfiles';
import { GROUND_COVER_ID_MASK } from '../groundcover/GroundCoverTypes';
import { aLoadU, bcU2F, returnIf, toF, unpackHalfU } from '../Tsl';
import type { LegacyPeriodicGuideField } from './LegacyPeriodicGuideField';
import type { LegacyPeriodicProfileSamplers } from './LegacyPeriodicProfileSampler';
import type { LegacyPeriodicRayAtlas } from './LegacyPeriodicRayAtlas';
import {
  COVER_PARAMETER_TABLE,
  GRASS_BAKE_ANGLES,
  GRASS_BAKE_RESOLUTION,
  GRASS_CELL_METRES,
  GRASS_DENSITY_TIERS,
  GRASS_ELECTION_FLAGS,
  GRASS_GRID,
  GRASS_NEAR_EPSILON,
  GRASS_RAY_END,
  GRASS_SALT,
  GRASS_TIER_KEEP_SALT,
  GUIDE_PITCH,
  GUIDE_RESOLUTION,
  GUIDE_SUB,
} from './LegacyPeriodicGroundCoverPolicy';

export interface LegacyPeriodicRayQueryOptions {
  readonly cam: NaniteCam;
  readonly vis: NaniteVisBuffers;
  readonly field: TerrainField;
  readonly enabledUniform: unknown;
  readonly streamed: boolean;
  readonly guide: LegacyPeriodicGuideField;
  readonly rayAtlas: LegacyPeriodicRayAtlas;
  readonly periodicProfiles: readonly LoadedPeriodicProfile[];
  readonly periodicSamplers: LegacyPeriodicProfileSamplers | null;
  readonly rayNormalTexture: StorageTexture;
  readonly forcedProfileId: number | null;
}

/** @deprecated The sole fixed-cost screen ray query for the legacy periodic renderer. */
export function createLegacyPeriodicRayQuery(options: LegacyPeriodicRayQueryOptions): unknown {
  const { cam, vis, field, streamed, periodicSamplers } = options;
  const uOn = options.enabledUniform;
  const guideCtx4 = options.guide.records;
  const guideFieldT1 = options.guide.field1;
  const guideFieldT2 = options.guide.field2;
  const guideFieldT3 = options.guide.field3;
  const uGFx = options.guide.originFineX;
  const uGFz = options.guide.originFineZ;
  const uRoMx = options.guide.rayOriginMetresX;
  const uRoMz = options.guide.rayOriginMetresZ;
  const uOTx = options.guide.worldTileOriginX;
  const uOTz = options.guide.worldTileOriginZ;
  const rayNrmTex = options.rayNormalTexture;
  const rayBake = {
    texs: options.rayAtlas.textures,
    dMaxTile: options.rayAtlas.maximumTileDistance,
    angleStride: options.rayAtlas.angleStride,
    atlasDepth: options.rayAtlas.depth,
  };
  const periodicProfilesById = new Map(
    options.periodicProfiles.map((profile) => [profile.profileId, profile] as const),
  );
  const standaloneProfile = options.forcedProfileId !== null
    && options.periodicProfiles.length === 1
    && options.periodicProfiles[0]?.textureLayer === null;
  const GRID = GRASS_GRID;
  const CELL = GRASS_CELL_METRES;
  const SALT = GRASS_SALT;
  const TIER_KEEP_SALT = GRASS_TIER_KEEP_SALT;
  const TIER_FRACS = GRASS_DENSITY_TIERS;
  const BAKE_RES = GRASS_BAKE_RESOLUTION;
  const BAKE_ANG = GRASS_BAKE_ANGLES;
  const RAY_END = GRASS_RAY_END;
  const GUIDE_RES = GUIDE_RESOLUTION;
  const NEAR_EPS = GRASS_NEAR_EPSILON;
  const COVER_PARAM_TABLE = COVER_PARAMETER_TABLE;
  const emitPx = (pixel: NU, clipDepth: NF, body: NU): void => {
    const depthKey = uint(float(1).sub(clipDepth).mul(16777215).clamp(0, 16777215));
    const candidate = depthKey.shiftLeft(uint(8)).bitOr(body.bitAnd(uint(0xff))).toVar();
    const previous = aLoadU(vis.payloadV.atomic.element(pixel));
    If(candidate.greaterThan(previous), () => {
      const won = atomicMax(vis.payloadV.atomic.element(pixel), candidate) as unknown as NU;
      If(candidate.greaterThan(won), () => {
        atomicStore(vis.visBV.atomic.element(pixel), uint(GRASS_ELECTION_FLAGS).bitOr(body));
      });
    });
  };
  const coverParams = (id: NU): NV4 => {
    let value = vec4(...COVER_PARAMETER_TABLE[5]) as unknown as NV4;
    for (let index = 4; index >= 0; index--) {
      value = (id.equal(uint(index)) as unknown as { select(a: unknown, b: unknown): NV4 })
        .select(vec4(...COVER_PARAMETER_TABLE[index]!) as unknown as NV4, value);
    }
    return value;
  };

  return ((): unknown => {
    const W = cam.width;
    const H = cam.height;
    const k = Fn(() => {
      returnIf((uOn as unknown as NF).lessThan(0.5) as unknown as NB);
      const qi = instanceIndex;
      returnIf(qi.greaterThanEqual(uint(W * H)));
      const xI = qi.mod(uint(W)).toVar() as unknown as NU;
      const yI = qi.div(uint(W)).toVar() as unknown as NU; // bottom-up rows
      const px = yI.mul(uint(W)).add(xI).toVar() as unknown as NU;
      const ndcX = toF(xI).add(0.5).div(W).mul(2).sub(1);
      const ndcY = toF(yI).add(0.5).div(H).mul(2).sub(1);
      const hf4 = cam.invVp.mul(vec4(ndcX, ndcY, 1, 1));
      const ro = vec3(cam.camPos).toVar() as unknown as NV3;
      const rd = (hf4.xyz.div(hf4.w).sub(ro).normalize().toVar()) as unknown as NV3;
      const tMax = float(1e9).toVar() as unknown as NF;
      const elect = aLoadU(vis.payloadV.atomic.element(px));
      If(elect.notEqual(uint(0)), () => {
        const czS = float(1).sub(toF(elect.shiftRight(uint(8))).div(16777215));
        const hs = cam.invVp.mul(vec4(ndcX, ndcY, czS, 1));
        (tMax as unknown as { assign(v: unknown): void }).assign(
          hs.xyz.div(hs.w).sub(ro).length().add(0.3),
        );
      });
      const dirL = rd.xz.length().max(1e-4).toVar() as unknown as NF;
      const tBest = float(1e9).toVar() as unknown as NF;
      const bodyBest = uint(0).toVar() as unknown as NU;
      const nrmV = vec3(0, 1, 0).toVar() as unknown as NV3;
      const tParV = float(0.5).toVar() as unknown as NF;
      const bestProfileId = field.hasGroundCoverClosure
        ? null
        : uint(0).toVar() as unknown as NU;
      const bestAntiLayer = uint(0).toVar() as unknown as NU;
      returnIf(tMax.greaterThan(1e8) as unknown as NB);
      const tScene = tMax.sub(0.3).max(0.05).toVar() as unknown as NF;
      const gfx = uGFx as unknown as NF;
      const gfz = uGFz as unknown as NF;
      returnIf(tScene.mul(dirL).greaterThan(RAY_END) as unknown as NB);
      /** ph-frame (S6c): streamed = guide-origin-relative metres (exact), else
       *  absolute world metres — every helper below takes coordinates in it */
      const phAt = (t: NF): { x: NF; z: NF } =>
        streamed
          ? {
              x: (uRoMx as unknown as NF).add(rd.x.mul(t)) as unknown as NF,
              z: (uRoMz as unknown as NF).add(rd.z.mul(t)) as unknown as NF,
            }
          : { x: ro.x.add(rd.x.mul(t)) as unknown as NF, z: ro.z.add(rd.z.mul(t)) as unknown as NF };
      /** ph-frame metres → fine-cell offset from the guide origin */
      const relCells = (ph: NF, gf: NF): NF =>
        (streamed ? ph.div(CELL) : ph.div(CELL).sub(gf)) as unknown as NF;
      /** bilinear guide sampler (ground + gradient) at a ph-frame position given as
       *  its fine-cell offset rc. The O/E/B planes were NEAREST per 0.84 m texel, so
       *  the entry, height and descent JUMPED at every guide-cell edge — the hard
       *  cell-edge clipping the user still sees, and a per-cell warp as the camera
       *  moves. Bilinear ⇒ C0-continuous across space. */
      const bguide = (rcx: NF, rcz: NF): { g: NF; grad: NV2; mixWord: NU; envWord: NU } => {
        const qx = rcx.div(GUIDE_SUB).sub(0.5) as unknown as NF;
        const qz = rcz.div(GUIDE_SUB).sub(0.5) as unknown as NF;
        const ix = qx.floor().clamp(0, GUIDE_RES - 2).toVar() as unknown as NF;
        const iz = qz.floor().clamp(0, GUIDE_RES - 2).toVar() as unknown as NF;
        const fx = qx.sub(ix).clamp(0, 1) as unknown as NF;
        const fz = qz.sub(iz).clamp(0, 1) as unknown as NF;
        const cv = (dx: number, dz: number): NV4 =>
          guideCtx4.element(
            uint(iz.add(dz).mul(GUIDE_RES).add(ix.add(dx))) as unknown as NU,
          ) as unknown as NV4;
        const c00 = cv(0, 0);
        const c10 = cv(1, 0);
        const c01 = cv(0, 1);
        const c11 = cv(1, 1);
        const g = mix(
          mix(bcU2F(c00.x as unknown as NU), bcU2F(c10.x as unknown as NU), fx),
          mix(bcU2F(c01.x as unknown as NU), bcU2F(c11.x as unknown as NU), fx),
          fz,
        ) as unknown as NF;
        const grad = mix(
          mix(
            unpackHalfU(c00.y as unknown as NU) as unknown as NV2,
            unpackHalfU(c10.y as unknown as NU) as unknown as NV2,
            fx,
          ),
          mix(
            unpackHalfU(c01.y as unknown as NU) as unknown as NV2,
            unpackHalfU(c11.y as unknown as NU) as unknown as NV2,
            fx,
          ),
          fz,
        ) as unknown as NV2;
        const nearMix0 = (fx.lessThan(0.5) as unknown as { select(a: unknown, b: unknown): NU })
          .select(c00.z, c10.z);
        const nearMix1 = (fx.lessThan(0.5) as unknown as { select(a: unknown, b: unknown): NU })
          .select(c01.z, c11.z);
        const nearEnv0 = (fx.lessThan(0.5) as unknown as { select(a: unknown, b: unknown): NU })
          .select(c00.w, c10.w);
        const nearEnv1 = (fx.lessThan(0.5) as unknown as { select(a: unknown, b: unknown): NU })
          .select(c01.w, c11.w);
        const mixWord = (fz.lessThan(0.5) as unknown as { select(a: unknown, b: unknown): NU })
          .select(nearMix0, nearMix1);
        const envWord = (fz.lessThan(0.5) as unknown as { select(a: unknown, b: unknown): NU })
          .select(nearEnv0, nearEnv1);
        return { g, grad, mixWord, envWord };
      };

      const phO = phAt(tScene);
      const phOx = phO.x.toVar() as unknown as NF;
      const phOz = phO.z.toVar() as unknown as NF;
      const rcOx = relCells(phOx, gfx).toVar() as unknown as NF;
      const rcOz = relCells(phOz, gfz).toVar() as unknown as NF;
      const og = bguide(rcOx, rcOz);
      const pOy = ro.y.add(rd.y.mul(tScene)).toVar() as unknown as NF;
      const groundO = (og.g as unknown as { toVar(): NF }).toVar() as unknown as NF;
      const gradO = (og.grad as unknown as { toVar(): NV2 }).toVar() as unknown as NV2;
      const guv = vec2(
        rcOx.div(GUIDE_SUB * GUIDE_RES),
        rcOz.div(GUIDE_SUB * GUIDE_RES),
      ).clamp(0, 1) as unknown as NV2;
      const f1 = (texture(guideFieldT1, guv, 0) as unknown as { toVar(): NV4 }).toVar();
      const f2 = (texture(guideFieldT2, guv, 0) as unknown as { toVar(): NV4 }).toVar();
      const f3 = (texture(guideFieldT3, guv, 0) as unknown as { toVar(): NV4 }).toVar();
      const dens = (f3.x as unknown as NF).clamp(0, 1).toVar() as unknown as NF;
      // The isolated volume is clipped at the reconstructed plant root.  An
      // endpoint test here cuts valid edge-facing plants into a floating sheet.
      if (!standaloneProfile) returnIf(dens.lessThan(0.02) as unknown as NB);
      const mixWordO = og.mixWord;
      const candidateMask = options.forcedProfileId === null
        ? mixWordO.shiftRight(uint(16)).bitAnd(uint(0xffff))
        : uint(1 << options.forcedProfileId);
      const vigor = field.hasGroundCover ? dens : (float(1) as unknown as NF);
      returnIf(
        gradO.x.mul(gradO.x).add(gradO.y.mul(gradO.y)).greaterThan(1.0) as unknown as NB,
      );
      const rag = ((): NF => {
        const qx = rcOx.mul(0.25).toVar() as unknown as NF;
        const qz = rcOz.mul(0.25).toVar() as unknown as NF;
        const ix = qx.floor().toVar() as unknown as NF;
        const iz = qz.floor().toVar() as unknown as NF;
        const fx = smoothstep(0, 1, qx.sub(ix)) as unknown as NF;
        const fz = smoothstep(0, 1, qz.sub(iz)) as unknown as NF;
        const h = (dx: number, dz: number): NF =>
          cellHash(
            vec2(
              ix.add(dx).add(gfx.mul(0.25)),
              iz.add(dz).add(gfz.mul(0.25)),
            ) as unknown as NV2,
            SALT ^ 0x5a17,
          ) as unknown as NF;
        return mix(mix(h(0, 0), h(1, 0), fx), mix(h(0, 1), h(1, 1), fx), fz) as unknown as NF;
      })();
      const considerProfile = (profileId: number): void => {
      const periodicProfile = periodicProfilesById.get(profileId) ?? null;
      const usesPeriodicCarrier = periodicProfile !== null;
      const profileIdU = uint(profileId);
      const coverIdValue = field.hasGroundCoverClosure
        ? GROUND_COVER_PROFILE_FUNCTIONAL_IDS[profileId]!
        : profileId;
      const coverId = uint(coverIdValue);
      const params = vec4(...COVER_PARAM_TABLE[coverIdValue]!).toVar() as unknown as NV4;
      const baseHeight = params.x as unknown as NF;
      const ragAmp = params.y as unknown as NF;
      const profileBaseBlock = params.w as unknown as NF;
      const vigorScale = field.hasGroundCover
        ? vigor.mul(0.55).add(0.65)
        : (float(1) as unknown as NF);
      const swardH = (
        standaloneProfile && periodicProfile
          ? float(periodicProfile.topH)
          : baseHeight
              .mul(vigorScale)
              .mul(rag.mul(ragAmp).add(float(1).sub(ragAmp.mul(0.5))))
              .clamp(0, 1.1)
      ).toVar() as unknown as NF;
      const hO = pOy.sub(groundO).toVar() as unknown as NF;
      // The isolated acceptance view judges the baked plant itself.  Keep its
      // carrier rigid until affine wind is reintroduced at the shared frame.
      const deformK = options.forcedProfileId !== null
        ? (float(0) as unknown as NF)
        : params.z as unknown as NF;
      const Slx = (f1.x as unknown as NF).mul(deformK).toVar() as unknown as NF;
      const Slz = (f1.y as unknown as NF).mul(deformK).toVar() as unknown as NF;
      const Sqx = (f1.z as unknown as NF).add(f2.z).mul(deformK).toVar() as unknown as NF;
      const Sqz = (f1.w as unknown as NF).add(f2.w).mul(deformK).toVar() as unknown as NF;

      const basisC = float(1)
        .sub(gradO.x.mul(Slx).add(gradO.y.mul(Slz)))
        .toVar() as unknown as NF;
      const basisB = gradO.x.mul(Sqx).add(gradO.y.mul(Sqz)).toVar() as unknown as NF;
      const qAt = (h: NF): NF => basisC.mul(h).sub(basisB.mul(h).mul(h)) as unknown as NF;
      const kGroundExact = rd.y
        .sub(gradO.x.mul(rd.x))
        .sub(gradO.y.mul(rd.z))
        .toVar() as unknown as NF;
      const kGround = (standaloneProfile ? kGroundExact : kGroundExact.min(-1e-3))
        .toVar() as unknown as NF;
      const qO = (hO.lessThanEqual(swardH) as unknown as { select(a: unknown, b: unknown): NF })
        .select(qAt(hO), hO)
        .toVar() as unknown as NF;
      const qTop = qAt(swardH).toVar() as unknown as NF;
      const dtEUnbounded = qTop.sub(qO).div(kGround) as unknown as NF;
      const dtE = (standaloneProfile
        ? dtEUnbounded
        : dtEUnbounded.clamp(
            float(GUIDE_PITCH * 3).div(dirL).negate(),
            float(GUIDE_PITCH * 3).div(dirL),
          )).toVar() as unknown as NF;
      const tE = tScene.add(dtE).max(0.05).toVar() as unknown as NF;

      const phE = phAt(tE);
      const phEx = phE.x.toVar() as unknown as NF;
      const phEz = phE.z.toVar() as unknown as NF;
      const hFromQ = (q: NF): NF => {
        const disc = basisC.mul(basisC).sub(basisB.mul(q).mul(4)).max(1e-5) as unknown as NF;
        return q.mul(2).div(basisC.add(disc.sqrt()).max(1e-4)) as unknown as NF;
      };
      const qE = qO.add(kGround.mul(tE.sub(tScene))).clamp(0, qTop.max(1e-4)) as unknown as NF;
      const hgt = hFromQ(qE).clamp(0, swardH).toVar() as unknown as NF;
      const offX = Slx.add(Sqx.mul(hgt)).mul(hgt) as unknown as NF; // Sl·h + Sq·h²
      const offZ = Slz.add(Sqz.mul(hgt)).mul(hgt) as unknown as NF;
      const wtx = (
        streamed ? phEx.div(GUIDE_PITCH).add(uOTx as unknown as NF) : phEx.div(GUIDE_PITCH)
      ).toVar() as unknown as NF;
      const wtz = (
        streamed ? phEz.div(GUIDE_PITCH).add(uOTz as unknown as NF) : phEz.div(GUIDE_PITCH)
      ).toVar() as unknown as NF;
      const Ptx = wtx.sub(offX.div(GUIDE_PITCH)).toVar() as unknown as NF;
      const Ptz = wtz.sub(offZ.div(GUIDE_PITCH)).toVar() as unknown as NF;
      const tanX = Slx.add(Sqx.mul(hgt).mul(2)) as unknown as NF;
      const tanZ = Slz.add(Sqz.mul(hgt).mul(2)) as unknown as NF;
      const basisDen = basisC.sub(basisB.mul(hgt).mul(2)).toVar() as unknown as NF;
      const basisValid = basisDen.greaterThan(0.05) as unknown as NB;
      const dhdt = kGround.div(basisDen.max(0.05)).toVar() as unknown as NF;
      const ex = rd.x.sub(tanX.mul(dhdt)).toVar() as unknown as NF;
      const ez = rd.z.sub(tanZ.mul(dhdt)).toVar() as unknown as NF;
      const eLen = vec2(ex, ez).length().max(1e-5).toVar() as unknown as NF;

      const missNorm = 1 / (1 + rayBake.dMaxTile);
      const lutSample = (
        qx: NF,
        qz: NF,
        azA: NF,
        typeSlot: NF,
        queryDens: NF,
      ): { rec: NV4; near: NV4 } => {
        const atlasZ = typeSlot
          .mul(rayBake.angleStride)
          .add(1)
          .add(azA.fract().mul(BAKE_ANG))
          .div(rayBake.atlasDepth) as unknown as NF;
        const tap = (i: number): NV4 =>
          texture3D(
            rayBake.texs[i] as unknown as Parameters<typeof texture3D>[0],
            vec3(qx, qz, atlasZ) as unknown as NV3,
            0,
          ) as unknown as NV4;
        const tc = vec3(
          qx.fract().mul(BAKE_RES).floor().add(0.5).div(BAKE_RES),
          qz.fract().mul(BAKE_RES).floor().add(0.5).div(BAKE_RES),
          typeSlot
            .mul(rayBake.angleStride)
            .add(1.5)
            .add(azA.fract().mul(BAKE_ANG).floor())
            .div(rayBake.atlasDepth),
        ) as unknown as NV3;
        const nearTap = (i: number): NV4 =>
          texture3D(
            rayBake.texs[i] as unknown as Parameters<typeof texture3D>[0],
            tc,
            0,
          ) as unknown as NV4;
        const rootKeep = (rt: NV4): NF => {
          const ri = (rt.w as unknown as NF).mul(GUIDE_SUB * GUIDE_SUB).floor().clamp(0, 63)
            .toVar() as unknown as NF;
          const rv = ri.div(GUIDE_SUB).floor() as unknown as NF;
          const ru = ri.sub(rv.mul(GUIDE_SUB)) as unknown as NF;
          return cellHash(vec2(ru, rv) as unknown as NV2, TIER_KEEP_SALT) as unknown as NF;
        };
        const miss = vec4(missNorm, 0, 166 / 255, 1) as unknown as NV4;
        const s = miss.toVar() as unknown as NV4;
        const n = miss.toVar() as unknown as NV4;
        const choosePair = (hi: number, lo: number): void => {
          const h = tap(hi);
          const l = tap(lo);
          const hn = nearTap(hi);
          const ln = nearTap(lo);
          const useHi = rootKeep(hn).lessThan(queryDens) as unknown as NB;
          (s as unknown as { assign(v: unknown): void }).assign(
            (useHi as unknown as { select(a: unknown, b: unknown): NV4 })
              .select(h, l),
          );
          (n as unknown as { assign(v: unknown): void }).assign(
            (useHi as unknown as { select(a: unknown, b: unknown): NV4 })
              .select(hn, ln),
          );
        };
        If(queryDens.greaterThanEqual(0.999), () => {
          (s as unknown as { assign(v: unknown): void }).assign(tap(0));
          (n as unknown as { assign(v: unknown): void }).assign(nearTap(0));
        })
          .ElseIf(queryDens.greaterThanEqual(TIER_FRACS[1] as number), () => {
            choosePair(0, 1);
          })
          .ElseIf(queryDens.greaterThanEqual(TIER_FRACS[2] as number), () => {
            choosePair(1, 2);
          })
          .ElseIf(queryDens.greaterThanEqual(TIER_FRACS[3] as number), () => {
            choosePair(2, 3);
          })
          .Else(() => {
            const h = tap(3);
            const hn = nearTap(3);
            const useHi = rootKeep(hn).lessThan(queryDens) as unknown as NB;
            (s as unknown as { assign(v: unknown): void }).assign(
              (useHi as unknown as { select(a: unknown, b: unknown): NV4 })
                .select(h, miss),
            );
            (n as unknown as { assign(v: unknown): void }).assign(
              (useHi as unknown as { select(a: unknown, b: unknown): NV4 })
                .select(hn, miss),
            );
          });
        return { rec: s, near: n };
      };

      const samplePeriodicLayer = (
        profile: LoadedPeriodicProfile,
        ang: number,
        scale: number,
        phaseX: number,
        phaseZ: number,
      ): {
        dt: NF; nx: NF; ny: NF; nz: NF; cr: NF; cg: NF; cb: NF;
        tip: NF; rootDx: NF; rootDz: NF;
      } => {
        const cs = Math.cos(ang);
        const sn = Math.sin(ang);
        const coordinateScale = standaloneProfile ? GUIDE_PITCH : 1;
        const qx = Ptx.mul(scale * coordinateScale).mul(cs)
          .sub(Ptz.mul(scale * coordinateScale).mul(sn)).add(phaseX) as unknown as NF;
        const qz = Ptx.mul(scale * coordinateScale).mul(sn)
          .add(Ptz.mul(scale * coordinateScale).mul(cs)).add(phaseZ) as unknown as NF;
        const profileDerivativeScale = scale * coordinateScale / GUIDE_PITCH;
        const dpx = ex.mul(profileDerivativeScale).mul(cs)
          .sub(ez.mul(profileDerivativeScale).mul(sn)) as unknown as NF;
        const dpz = ex.mul(profileDerivativeScale).mul(sn)
          .add(ez.mul(profileDerivativeScale).mul(cs)) as unknown as NF;
        const dpy = dhdt.mul(profile.topH).div(swardH.max(1e-4)) as unknown as NF;
        const metricSpeed = vec3(dpx, dpy, dpz).length().max(1e-5).toVar() as unknown as NF;
        const ndx = dpx.div(metricSpeed).toVar() as unknown as NF;
        const ndy = dpy.div(metricSpeed).toVar() as unknown as NF;
        const ndz = dpz.div(metricSpeed).toVar() as unknown as NF;
        const profileOriginY = hgt.mul(profile.topH)
          .div(swardH.max(1e-4)).toVar() as unknown as NF;
        if (!periodicSamplers) {
          throw new Error(`periodic profile ${profile.profileId} has no compatible texture carrier`);
        }
        const depthRows = Array.from({ length: profile.lattice.elevationCount }, (_, row) => {
          const slice = profile.slices[
            profile.lattice.order === 'azimuth-major'
              ? row
              : row * profile.lattice.azimuthCount
          ]!;
          return slice;
        });
        const profileDirection = vec3(ndx, ndy, ndz) as unknown as NV3;
        const profileTile = vec4(
          profile.tileOriginX, profile.tileOriginZ, profile.tileSizeX, profile.tileSizeZ,
        ) as unknown as NV4;
        const aligned = standaloneProfile && periodicSamplers.alignedStandalone
          ? periodicSamplers.alignedStandalone(vec2(qx, qz), profileDirection, profileTile)
          : null;
        const tProfile = (aligned
          ? aligned.depth
          : periodicSamplers.depth(
              vec2(qx, qz),
              profileDirection,
              profileTile,
              vec4(...depthRows.map((slice) => slice.depthMin)),
              vec4(...depthRows.map((slice) => slice.depthMax)),
              uint(profile.textureLayer ?? 0),
            )).toVar() as unknown as NF;
        const profileTip = (aligned
          ? aligned.point.y.div(profile.topH)
          : profileOriginY.add(ndy.mul(tProfile)).div(profile.topH))
          .clamp(0, 1).toVar() as unknown as NF;
        const dt = tProfile.div(metricSpeed).toVar() as unknown as NF;
        // Height and periodic-copy identity belong to the categorical point
        // X1.  Only depth compositing uses X1 projected onto the camera ray.
        const hitQx = (aligned ? aligned.point.x : qx.add(ndx.mul(tProfile))) as unknown as NF;
        const hitQz = (aligned ? aligned.point.z : qz.add(ndz.mul(tProfile))) as unknown as NF;
        const rootQx = hitQx.sub(profile.tileOriginX).div(profile.tileSizeX)
          .floor().add(0.5).mul(profile.tileSizeX).add(profile.tileOriginX) as unknown as NF;
        const rootQz = hitQz.sub(profile.tileOriginZ).div(profile.tileSizeZ)
          .floor().add(0.5).mul(profile.tileSizeZ).add(profile.tileOriginZ) as unknown as NF;
        const qa = rootQx.sub(phaseX) as unknown as NF;
        const qb = rootQz.sub(phaseZ) as unknown as NF;
        const rootPtx = qa.mul(cs).add(qb.mul(sn)).div(scale * coordinateScale) as unknown as NF;
        const rootPtz = qb.mul(cs).sub(qa.mul(sn)).div(scale * coordinateScale) as unknown as NF;
        const hitPtx = Ptx.add(ex.mul(dt).div(GUIDE_PITCH)) as unknown as NF;
        const hitPtz = Ptz.add(ez.mul(dt).div(GUIDE_PITCH)) as unknown as NF;
        const valid = tProfile.lessThan(5e5) as unknown as NB;
        const profileNormal = aligned?.normal ?? (vec3(0, 1, 0) as unknown as NV3);
        const normalX = (profileNormal.x as unknown as NF).mul(cs)
          .add((profileNormal.z as unknown as NF).mul(sn)) as unknown as NF;
        const normalZ = (profileNormal.z as unknown as NF).mul(cs)
          .sub((profileNormal.x as unknown as NF).mul(sn)) as unknown as NF;
        const normalY = (profileNormal.y as unknown as NF)
          .mul(profile.topH).div(swardH.max(1e-4)) as unknown as NF;
        const worldNormal = normalize(vec3(
          normalX.mul(profileDerivativeScale).sub(normalY.mul(gradO.x)),
          normalY,
          normalZ.mul(profileDerivativeScale).sub(normalY.mul(gradO.y)),
        ) as unknown as NV3) as unknown as NV3;
        const profileColor = aligned?.color ?? (vec3(0.05, 0.12, 0.03) as unknown as NV3);
        return {
          dt: (valid as unknown as { select(a: unknown, b: unknown): NF })
            .select(dt, float(1e6)),
          nx: worldNormal.x as unknown as NF,
          ny: worldNormal.y as unknown as NF,
          nz: worldNormal.z as unknown as NF,
          cr: profileColor.x as unknown as NF,
          cg: profileColor.y as unknown as NF,
          cb: profileColor.z as unknown as NF,
          tip: profileTip,
          rootDx: rootPtx.sub(hitPtx).mul(GUIDE_PITCH) as unknown as NF,
          rootDz: rootPtz.sub(hitPtz).mul(GUIDE_PITCH) as unknown as NF,
        };
      };
      const sampleAnalyticLayer = (
        ang: number,
        scale: number,
        phaseX: number,
        phaseZ: number,
      ): {
        dt: NF; nx: NF; ny: NF; nz: NF; cr: NF; cg: NF; cb: NF;
        tip: NF; rootDx: NF; rootDz: NF;
      } => {
        const cs = Math.cos(ang);
        const sn = Math.sin(ang);
        const qx = Ptx.mul(scale).mul(cs).sub(Ptz.mul(scale).mul(sn)).add(phaseX) as unknown as NF;
        const qz = Ptx.mul(scale).mul(sn).add(Ptz.mul(scale).mul(cs)).add(phaseZ) as unknown as NF;
        const rex = ex.mul(cs).sub(ez.mul(sn)) as unknown as NF;
        const rez = ex.mul(sn).add(ez.mul(cs)) as unknown as NF;
        const azN = (atan(rez, rex) as unknown as NF).mul(1 / (Math.PI * 2)).fract() as unknown as NF;
        const hit = lutSample(qx, qz, azN, profileBaseBlock, dens);
        const origin = (hit.near.x as unknown as NF).greaterThanEqual(254.5 / 255) as unknown as NB;
        const rec = (origin as unknown as { select(a: unknown, b: unknown): NV4 })
          .select(hit.near, hit.rec);
        const ox = (rec.y as unknown as NF).mul(2).sub(1).toVar() as unknown as NF;
        const oy = (rec.z as unknown as NF).mul(2).sub(1).toVar() as unknown as NF;
        const oz = float(1).sub(ox.abs()).sub(oy.abs()).toVar() as unknown as NF;
        If(oz.lessThan(0), () => {
          const oldX = (ox as unknown as { toVar(): NF }).toVar() as unknown as NF;
          const sx = (oldX.greaterThanEqual(0) as unknown as { select(a: unknown, b: unknown): NF })
            .select(float(1), float(-1));
          const sy = (oy.greaterThanEqual(0) as unknown as { select(a: unknown, b: unknown): NF })
            .select(float(1), float(-1));
          (ox as unknown as { assign(v: unknown): void }).assign(float(1).sub(oy.abs()).mul(sx));
          (oy as unknown as { assign(v: unknown): void }).assign(float(1).sub(oldX.abs()).mul(sy));
        });
        const octN = normalize(vec3(ox, oy, oz) as unknown as NV3) as unknown as NV3;
        const tx = octN.x as unknown as NF;
        const ny = octN.y as unknown as NF;
        const tz = octN.z as unknown as NF;
        const dParam = float(1).div((rec.x as unknown as NF).max(1 / 255)).sub(1) as unknown as NF;
        const valid = dParam.lessThan(rayBake.dMaxTile * 0.94) as unknown as NB;
        const rx = tx.mul(cs).add(tz.mul(sn)) as unknown as NF;
        const rz = tz.mul(cs).sub(tx.mul(sn)) as unknown as NF;
        const qSpeed = eLen.mul(scale / GUIDE_PITCH).max(1e-5) as unknown as NF;
        const dt = dParam.div(qSpeed) as unknown as NF;
        const rootId = (hit.near.w as unknown as NF)
          .mul(GUIDE_SUB * GUIDE_SUB).floor().clamp(0, GUIDE_SUB * GUIDE_SUB - 1) as unknown as NF;
        const rootV = rootId.div(GUIDE_SUB).floor() as unknown as NF;
        const rootU = rootId.sub(rootV.mul(GUIDE_SUB)) as unknown as NF;
        const localRootX = rootU.add(0.5).div(GUIDE_SUB) as unknown as NF;
        const localRootZ = rootV.add(0.5).div(GUIDE_SUB) as unknown as NF;
        const hitQx = qx.add(rex.div(eLen).mul(dParam)) as unknown as NF;
        const hitQz = qz.add(rez.div(eLen).mul(dParam)) as unknown as NF;
        const rootQx = hitQx.sub(localRootX).add(0.5).floor().add(localRootX) as unknown as NF;
        const rootQz = hitQz.sub(localRootZ).add(0.5).floor().add(localRootZ) as unknown as NF;
        const qa = rootQx.sub(phaseX) as unknown as NF;
        const qb = rootQz.sub(phaseZ) as unknown as NF;
        const rootPtx = qa.mul(cs).add(qb.mul(sn)).div(scale) as unknown as NF;
        const rootPtz = qb.mul(cs).sub(qa.mul(sn)).div(scale) as unknown as NF;
        const hitPtx = Ptx.add(ex.mul(dt).div(GUIDE_PITCH)) as unknown as NF;
        const hitPtz = Ptz.add(ez.mul(dt).div(GUIDE_PITCH)) as unknown as NF;
        return {
          dt: (valid as unknown as { select(a: unknown, b: unknown): NF })
            .select(dt, float(1e6)),
          nx: rx,
          ny,
          nz: rz,
          cr: float(0.05) as unknown as NF,
          cg: float(0.12) as unknown as NF,
          cb: float(0.03) as unknown as NF,
          tip: float(-1) as unknown as NF,
          rootDx: rootPtx.sub(hitPtx).mul(GUIDE_PITCH) as unknown as NF,
          rootDz: rootPtz.sub(hitPtz).mul(GUIDE_PITCH) as unknown as NF,
        };
      };
      const L0 = usesPeriodicCarrier
        ? samplePeriodicLayer(periodicProfile, 0, 1, 0, 0)
        : sampleAnalyticLayer(0, 1, 0, 0);
      // A single source copy is deliberate in the isolated acceptance graph:
      // it exposes one plant community without the second golden layer's dense
      // interference pattern, and removes its complete depth lookup at build
      // time. The production multi-profile graph retains both layers.
      const L1 = standaloneProfile
        ? L0
        : usesPeriodicCarrier
          ? samplePeriodicLayer(
              periodicProfile,
              Math.PI * 1.618033988749895,
              1.071773462536293,
              0.371,
              0.619,
            )
          : sampleAnalyticLayer(Math.PI * 1.618033988749895, 1.071773462536293, 0.371, 0.619);
      const take0 = standaloneProfile
        ? (uint(0).equal(uint(0)) as unknown as NB)
        : L0.dt.lessThanEqual(L1.dt) as unknown as NB;
      const pick = (a: NF, b: NF): NF =>
        (take0 as unknown as { select(x: unknown, y: unknown): NF }).select(a, b);
      const dtHit = pick(L0.dt, L1.dt).toVar() as unknown as NF;
      const inlinePeriodicSurface = standaloneProfile && Boolean(periodicSamplers?.alignedStandalone);
      const nWx0 = usesPeriodicCarrier && !inlinePeriodicSurface
        ? (float(0) as unknown as NF)
        : pick(L0.nx, L1.nx).toVar() as unknown as NF;
      const nWy0 = usesPeriodicCarrier && !inlinePeriodicSurface
        ? (float(1) as unknown as NF)
        : pick(L0.ny, L1.ny).toVar() as unknown as NF;
      const nWz0 = usesPeriodicCarrier && !inlinePeriodicSurface
        ? (float(0) as unknown as NF)
        : pick(L0.nz, L1.nz).toVar() as unknown as NF;
      const colorR = pick(L0.cr, L1.cr).toVar() as unknown as NF;
      const colorG = pick(L0.cg, L1.cg).toVar() as unknown as NF;
      const colorB = pick(L0.cb, L1.cb).toVar() as unknown as NF;
      const profileTip = pick(L0.tip, L1.tip).toVar() as unknown as NF;
      const rootDx = pick(L0.rootDx, L1.rootDx).toVar() as unknown as NF;
      const rootDz = pick(L0.rootDz, L1.rootDz).toVar() as unknown as NF;
      const tHit = tE.add(dtHit).toVar() as unknown as NF;
      const hitInRange = basisValid
        .and(dtHit.lessThan(1e5) as unknown as NB)
        .and(tHit.greaterThan(0.05) as unknown as NB)
        .and(tHit.lessThan(tMax) as unknown as NB) as unknown as NB;

      const preCandidate = hitInRange
        .and(tHit.lessThan(tBest) as unknown as NB) as unknown as NB;
      If(preCandidate, () => {

      const yH = ro.y.add(rd.y.mul(tHit)).toVar() as unknown as NF;
      const phB = phAt(tHit);
      const phBx = phB.x.toVar() as unknown as NF;
      const phBz = phB.z.toVar() as unknown as NF;
      const qBraw = (standaloneProfile
        ? profileTip.mul(qTop)
        : qO.add(kGround.mul(tHit.sub(tScene)))).toVar() as unknown as NF;
      const qInRange = standaloneProfile
        ? profileTip.greaterThanEqual(0).and(profileTip.lessThanEqual(1) as unknown as NB) as unknown as NB
        : qBraw.greaterThanEqual(-0.03)
            .and(qBraw.lessThanEqual(qTop.add(0.08)) as unknown as NB) as unknown as NB;
      const hB = (standaloneProfile
        ? profileTip.mul(swardH)
        : hFromQ(qBraw.clamp(0, qTop.max(1e-4)) as unknown as NF).clamp(0, swardH))
        .toVar() as unknown as NF;
      const offBX = Slx.add(Sqx.mul(hB)).mul(hB) as unknown as NF;
      const offBZ = Slz.add(Sqz.mul(hB)).mul(hB) as unknown as NF;
      const baseBx = phBx.sub(offBX).toVar() as unknown as NF;
      const baseBz = phBz.sub(offBZ).toVar() as unknown as NF;
      const rootBx = baseBx.add(rootDx).toVar() as unknown as NF;
      const rootBz = baseBz.add(rootDz).toVar() as unknown as NF;
      const baseRcx = relCells(baseBx, gfx).toVar() as unknown as NF;
      const baseRcz = relCells(baseBz, gfz).toVar() as unknown as NF;
      const rootRcx = relCells(rootBx, gfx).toVar() as unknown as NF;
      const rootRcz = relCells(rootBz, gfz).toVar() as unknown as NF;
      const surfaceGuide = bguide(baseRcx, baseRcz);
      const gB = (surfaceGuide.g as unknown as { toVar(): NF }).toVar() as unknown as NF;
      const surfaceValid = yH.sub(gB.add(hB)).abs().lessThanEqual(0.15) as unknown as NB;
      const gOB = groundO
        .add(gradO.x.mul(baseBx.sub(phOx)))
        .add(gradO.y.mul(baseBz.sub(phOz))) as unknown as NF;
      const planeValid = gB.sub(gOB).abs().lessThanEqual(0.35) as unknown as NB;
      const surfaceCandidate = standaloneProfile
        ? qInRange
        : qInRange.and(surfaceValid).and(planeValid) as unknown as NB;
      If(surfaceCandidate, () => {
      const rootGuide = bguide(rootRcx, rootRcz);
      const wcx = (streamed ? rootBx.div(CELL).floor().add(gfx) : rootBx.div(CELL).floor())
        .toVar() as unknown as NF;
      const wcz = (streamed ? rootBz.div(CELL).floor().add(gfz) : rootBz.div(CELL).floor())
        .toVar() as unknown as NF;
      const rootMix = rootGuide.mixWord;
      const rootEnv = rootGuide.envWord;
      const rootTypeA = rootMix.bitAnd(uint(0xff));
      const rootTypeB = rootMix.shiftRight(uint(8)).bitAnd(uint(0xff));
      const rootClumpLo = rootEnv.bitAnd(uint(0xff));
      const rootClumpHi = rootEnv.shiftRight(uint(8)).bitAnd(uint(0xff));
      const rootProfileA = rootEnv.shiftRight(uint(16)).bitAnd(uint(0xff));
      const rootProfileB = rootEnv.shiftRight(uint(24)).bitAnd(uint(0xff));
      const rootGuv = vec2(
        rootRcx.div(GUIDE_SUB * GUIDE_RES),
        rootRcz.div(GUIDE_SUB * GUIDE_RES),
      ).clamp(0, 1) as unknown as NV2;
      const rootControl = (texture(guideFieldT3, rootGuv, 0) as unknown as {
        toVar(): NV4;
      }).toVar();
      const rootBlend = field.hasGroundCover
        ? (rootControl.y as unknown as NF)
        : (float(0) as unknown as NF);
      const rootPick = cellHash(
        vec2(
          wcx.add(toF(rootClumpLo)),
          wcz.add(toF(rootClumpHi)),
        ) as unknown as NV2,
        SALT ^ 0x6c31,
      ) as unknown as NF;
      const rootCover = (rootPick.lessThan(rootBlend) as unknown as { select(a: unknown, b: unknown): NU })
        .select(rootTypeB, rootTypeA)
        .bitAnd(uint(GROUND_COVER_ID_MASK));
      const rootProfile = (rootPick.lessThan(rootBlend) as unknown as { select(a: unknown, b: unknown): NU })
        .select(rootProfileB, rootProfileA);
      const rootMatches = options.forcedProfileId !== null
        ? (field.hasGroundCover
            ? (rootControl.x as unknown as NF).greaterThanEqual(0.02) as unknown as NB
            : (uint(0).equal(uint(0)) as unknown as NB))
        : rootCover.equal(coverId)
            .and(rootProfile.equal(profileIdU) as unknown as NB) as unknown as NB;
      const sxs = wcx.sub(wcx.div(GRID).floor().mul(GRID));
      const sys = wcz.sub(wcz.div(GRID).floor().mul(GRID));
      If(rootMatches, () => {
        (tBest as unknown as { assign(v: unknown): void }).assign(tHit);
        (bodyBest as unknown as { assign(v: unknown): void }).assign(
          field.hasGroundCoverClosure
            ? uint(sys.mul(GRID).add(sxs))
                .bitAnd(uint(0x3fffff))
                .shiftLeft(uint(8))
                .bitOr(profileIdU)
            : uint(sys.mul(GRID).add(sxs)).shiftLeft(uint(6)).bitOr(coverId),
        );
        if (standaloneProfile) {
          (nrmV as unknown as { assign(v: unknown): void }).assign(vec3(nWx0, nWy0, nWz0));
          const authoredBody = uint(colorR.clamp(0, 1).mul(255).add(0.5).floor())
            .shiftLeft(uint(20))
            .bitOr(uint(colorG.clamp(0, 1).mul(255).add(0.5).floor()).shiftLeft(uint(12)))
            .bitOr(uint(colorB.clamp(0, 1).mul(255).add(0.5).floor()).shiftLeft(uint(4)))
            .bitOr(profileIdU.bitAnd(uint(0xf))) as unknown as NU;
          (bodyBest as unknown as { assign(v: unknown): void }).assign(authoredBody);
        } else if (usesPeriodicCarrier) {
          if (bestProfileId) {
            (bestProfileId as unknown as { assign(v: unknown): void }).assign(profileIdU);
          }
          (bestAntiLayer as unknown as { assign(v: unknown): void }).assign(
            (take0 as unknown as { select(a: unknown, b: unknown): NU })
              .select(uint(0), uint(1)),
          );
        } else {
          (nrmV as unknown as { assign(v: unknown): void }).assign(vec3(nWx0, nWy0, nWz0));
        }
        (tParV as unknown as { assign(v: unknown): void }).assign(
          usesPeriodicCarrier ? profileTip : hB.div(swardH.max(0.05)).clamp(0, 1),
        );
      });
      });
      });
      };

      if (options.forcedProfileId !== null) {
        considerProfile(options.forcedProfileId);
      } else {
        const candidateCount = field.hasGroundCoverClosure ? GROUND_COVER_PROFILE_COUNT : 6;
        for (let profileId = 0; profileId < candidateCount; profileId++) {
          If(candidateMask.bitAnd(uint(1 << profileId)).notEqual(uint(0)), () => {
            considerProfile(profileId);
          });
        }
      }

      If(tBest.lessThan(1e8), () => {
        const reconstructPeriodicNormal = (): void => {
        if (periodicSamplers && options.periodicProfiles.length > 0) {
          let winningProfileId: NU;
          let profileMeta: NV4;
          let tileSizeZ: NF;
          let bestCoverId: NU;
          if (standaloneProfile) {
            const profile = options.periodicProfiles[0]!;
            winningProfileId = uint(profile.profileId) as unknown as NU;
            profileMeta = vec4(
              profile.topH, profile.tileOriginX, profile.tileOriginZ, profile.tileSizeX,
            ) as unknown as NV4;
            tileSizeZ = float(profile.tileSizeZ) as unknown as NF;
            bestCoverId = uint(GROUND_COVER_PROFILE_FUNCTIONAL_IDS[profile.profileId]!) as unknown as NU;
          } else {
            const orderedProfiles = Array.from(
              { length: GROUND_COVER_PROFILE_COUNT },
              (_, profileId) => periodicProfilesById.get(profileId),
            );
            if (orderedProfiles.some((profile) => !profile)) {
              throw new Error('winner normal reconstruction requires the canonical profile array');
            }
            const profiles = orderedProfiles as LoadedPeriodicProfile[];
            winningProfileId = field.hasGroundCoverClosure
              ? bodyBest.bitAnd(uint(0xff)) as unknown as NU
              : bestProfileId!;
            profileMeta = vec4(
              profiles[GROUND_COVER_PROFILE_COUNT - 1]!.topH,
              profiles[GROUND_COVER_PROFILE_COUNT - 1]!.tileOriginX,
              profiles[GROUND_COVER_PROFILE_COUNT - 1]!.tileOriginZ,
              profiles[GROUND_COVER_PROFILE_COUNT - 1]!.tileSizeX,
            ) as unknown as NV4;
            tileSizeZ = float(profiles[GROUND_COVER_PROFILE_COUNT - 1]!.tileSizeZ) as unknown as NF;
            bestCoverId = uint(
              GROUND_COVER_PROFILE_FUNCTIONAL_IDS[GROUND_COVER_PROFILE_COUNT - 1]!,
            ) as unknown as NU;
            for (let profileId = GROUND_COVER_PROFILE_COUNT - 2; profileId >= 0; profileId--) {
              const isProfile = winningProfileId.equal(uint(profileId));
              const profile = profiles[profileId]!;
              profileMeta = (isProfile as unknown as { select(a: unknown, b: unknown): NV4 })
                .select(
                  vec4(profile.topH, profile.tileOriginX, profile.tileOriginZ, profile.tileSizeX),
                  profileMeta,
                );
              tileSizeZ = (isProfile as unknown as { select(a: unknown, b: unknown): NF })
                .select(float(profile.tileSizeZ), tileSizeZ);
              bestCoverId = (isProfile as unknown as { select(a: unknown, b: unknown): NU })
                .select(uint(GROUND_COVER_PROFILE_FUNCTIONAL_IDS[profileId]!), bestCoverId);
            }
          }
          const topH = profileMeta.x as unknown as NF;
          const params = coverParams(bestCoverId).toVar() as unknown as NV4;
          const bestSwardH = (standaloneProfile
            ? topH
            : (params.x as unknown as NF)
                .mul(field.hasGroundCover ? vigor.mul(0.55).add(0.65) : float(1))
                .mul(rag.mul(params.y).add(float(1).sub((params.y as unknown as NF).mul(0.5))))
                .clamp(0, 1.1)).toVar() as unknown as NF;
          const bestDeformK = standaloneProfile ? (float(0) as unknown as NF) : params.z as unknown as NF;
          const bestSlx = (f1.x as unknown as NF).mul(bestDeformK).toVar() as unknown as NF;
          const bestSlz = (f1.y as unknown as NF).mul(bestDeformK).toVar() as unknown as NF;
          const bestSqx = (f1.z as unknown as NF).add(f2.z).mul(bestDeformK).toVar() as unknown as NF;
          const bestSqz = (f1.w as unknown as NF).add(f2.w).mul(bestDeformK).toVar() as unknown as NF;
          const bestBasisC = float(1)
            .sub(gradO.x.mul(bestSlx).add(gradO.y.mul(bestSlz)))
            .toVar() as unknown as NF;
          const bestBasisB = gradO.x.mul(bestSqx).add(gradO.y.mul(bestSqz)).toVar() as unknown as NF;
          const bestQAt = (h: NF): NF =>
            bestBasisC.mul(h).sub(bestBasisB.mul(h).mul(h)) as unknown as NF;
          const bestKGroundExact = rd.y
            .sub(gradO.x.mul(rd.x))
            .sub(gradO.y.mul(rd.z))
            .toVar() as unknown as NF;
          const bestKGround = (standaloneProfile
            ? bestKGroundExact
            : bestKGroundExact.min(-1e-3)).toVar() as unknown as NF;
          const bestHO = pOy.sub(groundO).toVar() as unknown as NF;
          const bestQO = (
            bestHO.lessThanEqual(bestSwardH) as unknown as { select(a: unknown, b: unknown): NF }
          ).select(bestQAt(bestHO), bestHO).toVar() as unknown as NF;
          const bestQTop = bestQAt(bestSwardH).toVar() as unknown as NF;
          const bestDtEUnbounded = bestQTop.sub(bestQO).div(bestKGround) as unknown as NF;
          const bestDtE = (standaloneProfile
            ? bestDtEUnbounded
            : bestDtEUnbounded.clamp(
                float(GUIDE_PITCH * 3).div(dirL).negate(),
                float(GUIDE_PITCH * 3).div(dirL),
              )).toVar() as unknown as NF;
          const bestTE = tScene.add(bestDtE).max(0.05).toVar() as unknown as NF;
          const bestPhE = phAt(bestTE);
          const bestQE = bestQO.add(bestKGround.mul(bestTE.sub(tScene)))
            .clamp(0, bestQTop.max(1e-4)) as unknown as NF;
          const bestDisc = bestBasisC.mul(bestBasisC)
            .sub(bestBasisB.mul(bestQE).mul(4)).max(1e-5) as unknown as NF;
          const bestHgt = bestQE.mul(2)
            .div(bestBasisC.add(bestDisc.sqrt()).max(1e-4))
            .clamp(0, bestSwardH)
            .toVar() as unknown as NF;
          const bestOffX = bestSlx.add(bestSqx.mul(bestHgt)).mul(bestHgt) as unknown as NF;
          const bestOffZ = bestSlz.add(bestSqz.mul(bestHgt)).mul(bestHgt) as unknown as NF;
          const bestWtx = (
            streamed
              ? bestPhE.x.div(GUIDE_PITCH).add(uOTx as unknown as NF)
              : bestPhE.x.div(GUIDE_PITCH)
          ).toVar() as unknown as NF;
          const bestWtz = (
            streamed
              ? bestPhE.z.div(GUIDE_PITCH).add(uOTz as unknown as NF)
              : bestPhE.z.div(GUIDE_PITCH)
          ).toVar() as unknown as NF;
          const bestPtx = bestWtx.sub(bestOffX.div(GUIDE_PITCH)).toVar() as unknown as NF;
          const bestPtz = bestWtz.sub(bestOffZ.div(GUIDE_PITCH)).toVar() as unknown as NF;
          const bestTanX = bestSlx.add(bestSqx.mul(bestHgt).mul(2)) as unknown as NF;
          const bestTanZ = bestSlz.add(bestSqz.mul(bestHgt).mul(2)) as unknown as NF;
          const bestBasisDen = bestBasisC.sub(bestBasisB.mul(bestHgt).mul(2)).toVar() as unknown as NF;
          const bestDhdt = bestKGround.div(bestBasisDen.max(0.05)).toVar() as unknown as NF;
          const bestEx = rd.x.sub(bestTanX.mul(bestDhdt)).toVar() as unknown as NF;
          const bestEz = rd.z.sub(bestTanZ.mul(bestDhdt)).toVar() as unknown as NF;

          const useLayer0 = bestAntiLayer.equal(uint(0));
          const layer1Angle = Math.PI * 1.618033988749895;
          const layer1Scale = 1.071773462536293;
          const layerCs = (useLayer0 as unknown as { select(a: unknown, b: unknown): NF })
            .select(float(1), float(Math.cos(layer1Angle)))
            .toVar() as unknown as NF;
          const layerSn = (useLayer0 as unknown as { select(a: unknown, b: unknown): NF })
            .select(float(0), float(Math.sin(layer1Angle)))
            .toVar() as unknown as NF;
          const layerScale = (useLayer0 as unknown as { select(a: unknown, b: unknown): NF })
            .select(float(1), float(layer1Scale))
            .toVar() as unknown as NF;
          const layerScaleOverPitch = (
            useLayer0 as unknown as { select(a: unknown, b: unknown): NF }
          ).select(float(1 / GUIDE_PITCH), float(layer1Scale / GUIDE_PITCH))
            .toVar() as unknown as NF;
          const phaseX = (useLayer0 as unknown as { select(a: unknown, b: unknown): NF })
            .select(float(0), float(0.371)) as unknown as NF;
          const phaseZ = (useLayer0 as unknown as { select(a: unknown, b: unknown): NF })
            .select(float(0), float(0.619)) as unknown as NF;
          const coordinateScale = standaloneProfile ? GUIDE_PITCH : 1;
          const queryX = bestPtx.mul(coordinateScale).mul(layerScale).mul(layerCs)
            .sub(bestPtz.mul(coordinateScale).mul(layerScale).mul(layerSn)).add(phaseX) as unknown as NF;
          const queryZ = bestPtx.mul(coordinateScale).mul(layerScale).mul(layerSn)
            .add(bestPtz.mul(coordinateScale).mul(layerScale).mul(layerCs)).add(phaseZ) as unknown as NF;
          const derivativeScale = standaloneProfile ? layerScale : layerScaleOverPitch;
          const dpx = bestEx.mul(derivativeScale).mul(layerCs)
            .sub(bestEz.mul(derivativeScale).mul(layerSn)) as unknown as NF;
          const dpz = bestEx.mul(derivativeScale).mul(layerSn)
            .add(bestEz.mul(derivativeScale).mul(layerCs)) as unknown as NF;
          const dpy = bestDhdt.mul(topH).div(bestSwardH.max(1e-4)) as unknown as NF;
          const metricSpeed = vec3(dpx, dpy, dpz).length().max(1e-5).toVar() as unknown as NF;
          const nProfile = periodicSamplers.normal(
            vec2(queryX, queryZ),
            vec3(dpx.div(metricSpeed), dpy.div(metricSpeed), dpz.div(metricSpeed)),
            vec4(profileMeta.y, profileMeta.z, profileMeta.w, tileSizeZ),
            winningProfileId,
          ) as unknown as NV3;
          const nxBase = (nProfile.x as unknown as NF).mul(layerCs)
            .add((nProfile.z as unknown as NF).mul(layerSn)) as unknown as NF;
          const nzBase = (nProfile.z as unknown as NF).mul(layerCs)
            .sub((nProfile.x as unknown as NF).mul(layerSn)) as unknown as NF;
          const nyScale = (nProfile.y as unknown as NF)
            .mul(topH).div(bestSwardH.max(1e-4)) as unknown as NF;
          (nrmV as unknown as { assign(v: unknown): void }).assign(normalize(vec3(
            nxBase.mul(derivativeScale).sub(nyScale.mul(gradO.x)),
            nyScale,
            nzBase.mul(derivativeScale).sub(nyScale.mul(gradO.y)),
          ) as unknown as NV3));
          if (standaloneProfile && periodicSamplers.color) {
            const authoredColor = periodicSamplers.color(
              vec2(queryX, queryZ),
              vec3(dpx.div(metricSpeed), dpy.div(metricSpeed), dpz.div(metricSpeed)),
              vec4(profileMeta.y, profileMeta.z, profileMeta.w, tileSizeZ),
            ) as unknown as NV3;
            const authoredBody = uint((authoredColor.x as unknown as NF)
              .clamp(0, 1).mul(255).add(0.5).floor())
              .shiftLeft(uint(20))
              .bitOr(uint((authoredColor.y as unknown as NF)
                .clamp(0, 1).mul(255).add(0.5).floor()).shiftLeft(uint(12)))
              .bitOr(uint((authoredColor.z as unknown as NF)
                .clamp(0, 1).mul(255).add(0.5).floor()).shiftLeft(uint(4)))
              .bitOr(winningProfileId.bitAnd(uint(0xf))) as unknown as NU;
            (bodyBest as unknown as { assign(v: unknown): void }).assign(authoredBody);
          }
        }
        };
        const hit = ro.add(rd.mul(tBest));
        const clip = cam.vp.mul(vec4(hit, 1));
        const cz = clip.z.div(clip.w.max(NEAR_EPS));
        If(cz.greaterThanEqual(0).and(cz.lessThanEqual(1)), () => {
          if (!standaloneProfile) reconstructPeriodicNormal();
          emitPx(px as unknown as NU, cz as unknown as NF, bodyBest);
          textureStore(rayNrmTex, uvec2(xI, yI), vec4(nrmV, tParV)).toWriteOnly();
        });
      });
    })().compute(W * H, [256]);
    (k as unknown as { setName(n: string): void }).setName('legacyPeriodicGroundCoverRay');
    return k;
  })();

}
