import type { Texture } from 'three';
import { Fn, If, atan, float, int, normalize, texture, uint, vec2, vec3, vec4 } from 'three/tsl';
import type { NB, NF, NU, NV2, NV3, NV4 } from '../../gpu/TSLTypes';
import {
  GROUND_COVER_PROFILE_COUNT,
  type LoadedPeriodicProfile,
} from '../groundcover/GroundCoverProfiles';

export interface LegacyPeriodicProfileSamplers {
  readonly depth: (
    qxz: NV2,
    direction: NV3,
    tile: NV4,
    depthMin: NV4,
    depthMax: NV4,
    profileLayer: NU,
  ) => NF;
  readonly normal: (qxz: NV2, direction: NV3, tile: NV4, profileLayer: NU) => NV3;
  readonly color: ((qxz: NV2, direction: NV3, tile: NV4) => NV3) | null;
  readonly standalone: boolean;
}

interface AddressResult {
  readonly nd: NV3;
  readonly r00: NV4;
  readonly r10: NV4;
  readonly r01: NV4;
  readonly r11: NV4;
  readonly elevation0: NF;
  readonly elevation1: NF;
  readonly rowValue: (values: NV4, row: NF) => NF;
  readonly w00: NF;
  readonly w10: NF;
  readonly w01: NF;
  readonly w11: NF;
  readonly weight: NF;
}

/** @deprecated Fixed GCAR array sampler used only by the legacy periodic renderer. */
export function createLegacyPeriodicProfileSamplers(
  profiles: readonly LoadedPeriodicProfile[],
): LegacyPeriodicProfileSamplers | null {
  if (profiles.length === 0) return null;
  const layout = profiles[0]!;
  const arrayTexture = layout.texture as Texture;
  const standalone = profiles.length === 1 && layout.textureLayer === null;
  if (!standalone && (
    profiles.length !== GROUND_COVER_PROFILE_COUNT
    || profiles.some((profile) => (
      profile.texture !== arrayTexture || profile.textureLayer !== profile.profileId
    ))
  )) throw new Error('legacy periodic renderer requires canonical GCAR or one standalone profile');

  const decodeRecordNormal = (record: NV4): NV3 => {
    const coverage = (record.w as unknown as NF).clamp(0, 1) as unknown as NF;
    const missWeight = float(1).sub(coverage) as unknown as NF;
    const safeCoverage = coverage.max(1 / 65535) as unknown as NF;
    const ox = (record.y as unknown as NF)
      .sub(missWeight.mul(0.5)).div(safeCoverage).clamp(0, 1)
      .mul(2).sub(1).toVar() as unknown as NF;
    const oy = (record.z as unknown as NF)
      .sub(missWeight.mul(0.5)).div(safeCoverage).clamp(0, 1)
      .mul(2).sub(1).toVar() as unknown as NF;
    const oz = float(1).sub(ox.abs()).sub(oy.abs()).toVar() as unknown as NF;
    If(oz.lessThan(0), () => {
      const oldX = (ox as unknown as { toVar(): NF }).toVar() as unknown as NF;
      const sx = (oldX.greaterThanEqual(0) as unknown as { select(a: unknown, b: unknown): NF })
        .select(float(1), float(-1));
      const sy = (oy.greaterThanEqual(0) as unknown as { select(a: unknown, b: unknown): NF })
        .select(float(1), float(-1));
      (ox as unknown as { assign(value: unknown): void }).assign(float(1).sub(oy.abs()).mul(sx));
      (oy as unknown as { assign(value: unknown): void }).assign(float(1).sub(oldX.abs()).mul(sy));
    });
    return normalize(vec3(ox, oy, oz) as unknown as NV3) as unknown as NV3;
  };

  const address = (
    qxz: NV2,
    nd: NV3,
    tile: NV4,
    profileLayer: NU,
    textureOverride: Texture | null = null,
  ): AddressResult => {
    const azF = (atan(nd.z, nd.x) as unknown as NF)
      .mul(1 / (Math.PI * 2)).fract().mul(layout.lattice.azimuthCount)
      .toVar() as unknown as NF;
    const az0 = azF.floor().toVar() as unknown as NF;
    const az1 = az0.add(1).mod(layout.lattice.azimuthCount).toVar() as unknown as NF;
    const azMix = azF.sub(az0).clamp(0, 1).toVar() as unknown as NF;
    const elevation = (atan(
      nd.y.negate(),
      vec2(nd.x, nd.z).length().max(1e-5),
    ) as unknown as NF).toVar() as unknown as NF;
    const lastInterval = layout.lattice.elevationCount - 2;
    let elevation0: NF = float(lastInterval) as unknown as NF;
    let elevation1: NF = float(lastInterval + 1) as unknown as NF;
    for (let index = lastInterval - 1; index >= 0; index--) {
      const below = elevation.lessThan(layout.lattice.elevations[index + 1]!) as unknown as {
        select(a: unknown, b: unknown): NF;
      };
      elevation0 = below.select(float(index), elevation0);
      elevation1 = below.select(float(index + 1), elevation1);
    }
    const rowValue = (values: NV4, row: NF): NF => {
      let value = values.w as unknown as NF;
      value = (row.equal(float(2)) as unknown as { select(a: unknown, b: unknown): NF })
        .select(values.z, value);
      value = (row.equal(float(1)) as unknown as { select(a: unknown, b: unknown): NF })
        .select(values.y, value);
      return (row.equal(float(0)) as unknown as { select(a: unknown, b: unknown): NF })
        .select(values.x, value);
    };
    const elevationLo = rowValue(vec4(...layout.lattice.elevations) as unknown as NV4, elevation0);
    const elevationHi = rowValue(vec4(...layout.lattice.elevations) as unknown as NV4, elevation1);
    const elevationMix = elevation.sub(elevationLo)
      .div(elevationHi.sub(elevationLo).max(1e-5)).clamp(0, 1)
      .toVar() as unknown as NF;
    const u = qxz.x.sub(tile.x).div(tile.z).fract().toVar() as unknown as NF;
    const v = float(1).sub(qxz.y.sub(tile.y).div(tile.w).fract()).toVar() as unknown as NF;
    const atlasWidth = layout.storedTileWidth * layout.atlasColumns;
    const atlasHeight = layout.storedTileHeight * layout.atlasRows;
    const tap = (azimuth: NF, row: NF): NV4 => {
      const slice = layout.lattice.order === 'azimuth-major'
        ? azimuth.mul(layout.lattice.elevationCount).add(row) as unknown as NF
        : row.mul(layout.lattice.azimuthCount).add(azimuth) as unknown as NF;
      const column = slice.mod(layout.atlasColumns).toVar() as unknown as NF;
      const atlasRow = slice.div(layout.atlasColumns).floor().toVar() as unknown as NF;
      const uv = vec2(
        column.mul(layout.storedTileWidth).add(layout.gutter)
          .add(u.mul(layout.interiorTileWidth)).div(atlasWidth),
        atlasRow.mul(layout.storedTileHeight).add(layout.gutter)
          .add(v.mul(layout.interiorTileHeight)).div(atlasHeight),
      ) as unknown as NV2;
      const sourceTexture = textureOverride ?? arrayTexture;
      return standalone || textureOverride
        ? texture(sourceTexture, uv, 0) as unknown as NV4
        : (texture(sourceTexture, uv) as unknown as {
            depth(layer: unknown): { level(lod: unknown): NV4 };
          }).depth(int(profileLayer)).level(float(0));
    };
    const r00 = tap(az0, elevation0);
    const r10 = tap(az1, elevation0);
    const r01 = tap(az0, elevation1);
    const r11 = tap(az1, elevation1);
    const oneAz = float(1).sub(azMix) as unknown as NF;
    const oneEl = float(1).sub(elevationMix) as unknown as NF;
    const w00 = oneAz.mul(oneEl).mul(r00.w.clamp(0, 1)).toVar() as unknown as NF;
    const w10 = azMix.mul(oneEl).mul(r10.w.clamp(0, 1)).toVar() as unknown as NF;
    const w01 = oneAz.mul(elevationMix).mul(r01.w.clamp(0, 1)).toVar() as unknown as NF;
    const w11 = azMix.mul(elevationMix).mul(r11.w.clamp(0, 1)).toVar() as unknown as NF;
    return {
      nd, r00, r10, r01, r11, elevation0, elevation1, rowValue,
      w00, w10, w01, w11,
      weight: w00.add(w10).add(w01).add(w11).toVar() as unknown as NF,
    };
  };

  const depth = Fn(([
    qxz, nd, tile, depthMin, depthMax, profileLayer,
  ]: [NV2, NV3, NV4, NV4, NV4, NU]): NF => {
    const a = address(qxz, nd, tile, profileLayer);
    if (standalone) {
      const maxProjectedTiles = Math.max(...layout.slices.map((slice) =>
        slice.depthMax * Math.hypot(slice.direction[0], slice.direction[2]) / layout.tileSizeX));
      const missInverse = 1 / (1 + maxProjectedTiles);
      const hitInverse = (record: NV4): NF => {
        const coverage = (record.w as unknown as NF).clamp(0, 1) as unknown as NF;
        return (record.x as unknown as NF)
          .sub(float(1).sub(coverage).mul(missInverse))
          .div(coverage.max(1 / 65535))
          .clamp(1 / 65535, 1) as unknown as NF;
      };
      const inversePath = hitInverse(a.r00).mul(a.w00)
        .add(hitInverse(a.r10).mul(a.w10))
        .add(hitInverse(a.r01).mul(a.w01))
        .add(hitInverse(a.r11).mul(a.w11))
        .div(a.weight.max(1e-5)).max(1 / 65535) as unknown as NF;
      const projectedTiles = float(1).div(inversePath).sub(1) as unknown as NF;
      const liveHorizontal = vec2(a.nd.x, a.nd.z).length().max(1e-4) as unknown as NF;
      const liveT = projectedTiles.mul(tile.z).div(liveHorizontal) as unknown as NF;
      const valid = a.weight.greaterThan(0.02)
        .and(projectedTiles.lessThan(maxProjectedTiles * 0.97) as unknown as NB)
        .and(a.nd.y.lessThan(-1e-4) as unknown as NB) as unknown as NB;
      return (valid as unknown as { select(a: unknown, b: unknown): NF })
        .select(liveT, float(1e6));
    }
    const unpackDepth = (record: NV4, row: NF): NF => {
      const coverage = (record.w as unknown as NF).clamp(0, 1) as unknown as NF;
      const depth01 = (record.x as unknown as NF)
        .sub(float(1).sub(coverage)).div(coverage.max(1 / 65535)).clamp(0, 1) as unknown as NF;
      const low = a.rowValue(depthMin, row);
      const high = a.rowValue(depthMax, row);
      return low.add(depth01.mul(high.sub(low))) as unknown as NF;
    };
    const result = unpackDepth(a.r00, a.elevation0).mul(a.w00)
      .add(unpackDepth(a.r10, a.elevation0).mul(a.w10))
      .add(unpackDepth(a.r01, a.elevation1).mul(a.w01))
      .add(unpackDepth(a.r11, a.elevation1).mul(a.w11))
      .div(a.weight.max(1e-5)).toVar() as unknown as NF;
    const valid = a.weight.greaterThan(0.02)
      .and(a.nd.y.lessThan(-1e-4) as unknown as NB) as unknown as NB;
    return (valid as unknown as { select(a: unknown, b: unknown): NF })
      .select(result, float(1e6));
  }).setLayout({
    name: 'legacyGroundCoverPeriodicDepth',
    type: 'float',
    inputs: [
      { name: 'qxz', type: 'vec2' },
      { name: 'nd', type: 'vec3' },
      { name: 'tile', type: 'vec4' },
      { name: 'depthMin', type: 'vec4' },
      { name: 'depthMax', type: 'vec4' },
      { name: 'profileLayer', type: 'uint' },
    ],
  });
  const normal = Fn(([
    qxz, nd, tile, profileLayer,
  ]: [NV2, NV3, NV4, NU]): NV3 => {
    const a = address(qxz, nd, tile, profileLayer);
    return normalize(
      decodeRecordNormal(a.r00).mul(a.w00)
        .add(decodeRecordNormal(a.r10).mul(a.w10))
        .add(decodeRecordNormal(a.r01).mul(a.w01))
        .add(decodeRecordNormal(a.r11).mul(a.w11))
        .add(vec3(0, 1e-6, 0)) as unknown as NV3,
    ) as unknown as NV3;
  }).setLayout({
    name: 'legacyGroundCoverPeriodicNormal',
    type: 'vec3',
    inputs: [
      { name: 'qxz', type: 'vec2' },
      { name: 'nd', type: 'vec3' },
      { name: 'tile', type: 'vec4' },
      { name: 'profileLayer', type: 'uint' },
    ],
  });
  const color = standalone && layout.colorTexture
    ? Fn(([
        qxz, nd, tile,
      ]: [NV2, NV3, NV4]): NV3 => {
        const a = address(qxz, nd, tile, uint(0), layout.colorTexture as Texture);
        const contribution = (record: NV4, weightedCoverage: NF): NV3 =>
          (record.xyz as unknown as NV3).mul(
            weightedCoverage.div((record.w as unknown as NF).max(1 / 255)),
          ) as unknown as NV3;
        const rgb = contribution(a.r00, a.w00)
          .add(contribution(a.r10, a.w10))
          .add(contribution(a.r01, a.w01))
          .add(contribution(a.r11, a.w11))
          .div(a.weight.max(1 / 255)) as unknown as NV3;
        return (a.weight.greaterThan(1 / 255) as unknown as {
          select(a: unknown, b: unknown): NV3;
        }).select(rgb, vec3(0.05, 0.12, 0.03));
      }).setLayout({
        name: 'legacyGroundCoverPeriodicColor',
        type: 'vec3',
        inputs: [
          { name: 'qxz', type: 'vec2' },
          { name: 'nd', type: 'vec3' },
          { name: 'tile', type: 'vec4' },
        ],
      })
    : null;
  return Object.freeze({
    depth: depth as unknown as LegacyPeriodicProfileSamplers['depth'],
    normal: normal as unknown as LegacyPeriodicProfileSamplers['normal'],
    color: color as unknown as LegacyPeriodicProfileSamplers['color'],
    standalone,
  });
}
