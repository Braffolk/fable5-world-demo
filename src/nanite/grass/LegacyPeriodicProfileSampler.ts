import type { Texture } from 'three';
import { Fn, If, atan, cos, float, int, normalize, sin, texture, uint, vec2, vec3, vec4 } from 'three/tsl';
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
  readonly alignedStandalone: ((
    qxz: NV2,
    direction: NV3,
    tile: NV4,
  ) => { readonly depth: NF; readonly normal: NV3; readonly color: NV3 }) | null;
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
  readonly azimuth0: NF;
  readonly azimuth1: NF;
  readonly azimuthMix: NF;
  readonly elevationMix: NF;
  readonly rowValue: (values: NV4, row: NF) => NF;
  readonly tap: (azimuth: NF, row: NF, at: NV2, override?: Texture | null) => NV4;
  readonly b00: NF;
  readonly b10: NF;
  readonly b01: NF;
  readonly b11: NF;
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
    const atlasWidth = layout.storedTileWidth * layout.atlasColumns;
    const atlasHeight = layout.storedTileHeight * layout.atlasRows;
    const tap = (
      azimuth: NF,
      row: NF,
      at: NV2,
      override: Texture | null = textureOverride,
    ): NV4 => {
      const u = at.x.sub(tile.x).div(tile.z).fract().toVar() as unknown as NF;
      const v = float(1).sub(at.y.sub(tile.y).div(tile.w).fract()).toVar() as unknown as NF;
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
      const sourceTexture = override ?? arrayTexture;
      return standalone || override
        ? texture(sourceTexture, uv, 0) as unknown as NV4
        : (texture(sourceTexture, uv) as unknown as {
            depth(layer: unknown): { level(lod: unknown): NV4 };
          }).depth(int(profileLayer)).level(float(0));
    };
    const r00 = tap(az0, elevation0, qxz);
    const r10 = tap(az1, elevation0, qxz);
    const r01 = tap(az0, elevation1, qxz);
    const r11 = tap(az1, elevation1, qxz);
    const oneAz = float(1).sub(azMix) as unknown as NF;
    const oneEl = float(1).sub(elevationMix) as unknown as NF;
    const b00 = oneAz.mul(oneEl).toVar() as unknown as NF;
    const b10 = azMix.mul(oneEl).toVar() as unknown as NF;
    const b01 = oneAz.mul(elevationMix).toVar() as unknown as NF;
    const b11 = azMix.mul(elevationMix).toVar() as unknown as NF;
    const w00 = b00.mul(r00.w.clamp(0, 1)).toVar() as unknown as NF;
    const w10 = b10.mul(r10.w.clamp(0, 1)).toVar() as unknown as NF;
    const w01 = b01.mul(r01.w.clamp(0, 1)).toVar() as unknown as NF;
    const w11 = b11.mul(r11.w.clamp(0, 1)).toVar() as unknown as NF;
    return {
      nd, r00, r10, r01, r11,
      elevation0, elevation1, azimuth0: az0, azimuth1: az1,
      azimuthMix: azMix, elevationMix, rowValue, tap,
      b00, b10, b01, b11,
      w00, w10, w01, w11,
      weight: w00.add(w10).add(w01).add(w11).toVar() as unknown as NF,
    };
  };

  const depth = Fn(([
    qxz, nd, tile, depthMin, depthMax, profileLayer,
  ]: [NV2, NV3, NV4, NV4, NV4, NU]): NF => {
    const a = address(qxz, nd, tile, profileLayer);
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

  const alignedStandalone = standalone
    ? ((qxz: NV2, nd: NV3, tile: NV4): { depth: NF; normal: NV3; color: NV3 } => {
        const a = address(qxz, nd, tile, uint(0));
        const elevations = vec4(...layout.lattice.elevations) as unknown as NV4;
        const depthMins = vec4(...Array.from(
          { length: layout.lattice.elevationCount },
          (_, row) => layout.slices[
            layout.lattice.order === 'azimuth-major'
              ? row
              : row * layout.lattice.azimuthCount
          ]!.depthMin,
        )) as unknown as NV4;
        const depthMaxs = vec4(...Array.from(
          { length: layout.lattice.elevationCount },
          (_, row) => layout.slices[
            layout.lattice.order === 'azimuth-major'
              ? row
              : row * layout.lattice.azimuthCount
          ]!.depthMax,
        )) as unknown as NV4;
        const decodeDepth = (record: NV4, row: NF, missAsFar: boolean): NF => {
          const coverage = (record.w as unknown as NF).clamp(0, 1) as unknown as NF;
          const depth01 = (record.x as unknown as NF)
            .sub(float(1).sub(coverage))
            .div(coverage.max(1 / 65535))
            .clamp(0, 1) as unknown as NF;
          const low = a.rowValue(depthMins, row);
          const high = a.rowValue(depthMaxs, row);
          const hitDepth = low.add(depth01.mul(high.sub(low))) as unknown as NF;
          return missAsFar
            ? (coverage.greaterThan(1 / 65535) as unknown as {
                select(yes: unknown, no: unknown): NF;
              }).select(hitDepth, high)
            : hitDepth;
        };
        type Node = {
          readonly score: NF;
          readonly liveDepth: NF;
          readonly normal: NV3;
          readonly color: NV4;
        };
        const node = (azimuth: NF, row: NF, base: NF, initial: NV4): Node => {
          const elevation = a.rowValue(elevations, row);
          const azimuthRadians = azimuth.mul(Math.PI * 2 / layout.lattice.azimuthCount) as unknown as NF;
          const horizontal = cos(elevation) as unknown as NF;
          const canonical = vec3(
            horizontal.mul(cos(azimuthRadians)),
            sin(elevation).negate(),
            horizontal.mul(sin(azimuthRadians)),
          ).toVar() as unknown as NV3;
          const tau0 = decodeDepth(initial, row, true);
          const verticalRatio = (canonical.y as unknown as NF)
            .div((nd.y as unknown as NF).min(-1e-5)) as unknown as NF;
          const bx = verticalRatio.mul(nd.x).sub(canonical.x).toVar() as unknown as NF;
          const bz = verticalRatio.mul(nd.z).sub(canonical.z).toVar() as unknown as NF;
          const correctedAddress = vec2(
            qxz.x.add(bx.mul(tau0)),
            qxz.y.add(bz.mul(tau0)),
          ) as unknown as NV2;
          const record = a.tap(azimuth, row, correctedAddress);
          const tau1 = decodeDepth(record, row, false);
          const canonicalDotLive = (canonical.x as unknown as NF).mul(nd.x)
            .add((canonical.y as unknown as NF).mul(nd.y))
            .add((canonical.z as unknown as NF).mul(nd.z)) as unknown as NF;
          const liveDepth = bx.mul(tau0).mul(nd.x)
            .add(bz.mul(tau0).mul(nd.z))
            .add(tau1.mul(canonicalDotLive))
            .toVar() as unknown as NF;
          const colorRecord = layout.colorTexture
            ? a.tap(azimuth, row, correctedAddress, layout.colorTexture as Texture)
            : vec4(0.05, 0.12, 0.03, record.w) as unknown as NV4;
          return {
            score: base.mul((record.w as unknown as NF).clamp(0, 1)).toVar() as unknown as NF,
            liveDepth,
            normal: decodeRecordNormal(record),
            color: colorRecord,
          };
        };
        const n00 = node(a.azimuth0, a.elevation0, a.b00, a.r00);
        const n10 = node(a.azimuth1, a.elevation0, a.b10, a.r10);
        const n01 = node(a.azimuth0, a.elevation1, a.b01, a.r01);
        const n11 = node(a.azimuth1, a.elevation1, a.b11, a.r11);
        const choose = (left: Node, right: Node): Node => {
          const takeRight = right.score.greaterThan(left.score) as unknown as NB;
          const selectF = (yes: NF, no: NF): NF =>
            (takeRight as unknown as { select(a: unknown, b: unknown): NF }).select(yes, no);
          const selectV3 = (yes: NV3, no: NV3): NV3 =>
            (takeRight as unknown as { select(a: unknown, b: unknown): NV3 }).select(yes, no);
          const selectV4 = (yes: NV4, no: NV4): NV4 =>
            (takeRight as unknown as { select(a: unknown, b: unknown): NV4 }).select(yes, no);
          return {
            score: selectF(right.score, left.score),
            liveDepth: selectF(right.liveDepth, left.liveDepth),
            normal: selectV3(right.normal, left.normal),
            color: selectV4(right.color, left.color),
          };
        };
        const winner = choose(choose(n00, n10), choose(n01, n11));
        const premul = (n00.color.xyz as unknown as NV3).mul(a.b00)
          .add((n10.color.xyz as unknown as NV3).mul(a.b10))
          .add((n01.color.xyz as unknown as NV3).mul(a.b01))
          .add((n11.color.xyz as unknown as NV3).mul(a.b11)) as unknown as NV3;
        const alpha = (n00.color.w as unknown as NF).mul(a.b00)
          .add((n10.color.w as unknown as NF).mul(a.b10))
          .add((n01.color.w as unknown as NF).mul(a.b01))
          .add((n11.color.w as unknown as NF).mul(a.b11)) as unknown as NF;
        const valid = winner.score.greaterThan(1 / 255)
          .and((nd.y as unknown as NF).lessThan(-1e-4) as unknown as NB) as unknown as NB;
        return {
          depth: (valid as unknown as { select(a: unknown, b: unknown): NF })
            .select(winner.liveDepth.max(0), float(1e6)),
          normal: winner.normal,
          color: (alpha.greaterThan(1 / 255) as unknown as {
            select(a: unknown, b: unknown): NV3;
          }).select(premul.div(alpha.max(1 / 255)), vec3(0.05, 0.12, 0.03)),
        };
      })
    : null;
  return Object.freeze({
    depth: depth as unknown as LegacyPeriodicProfileSamplers['depth'],
    normal: normal as unknown as LegacyPeriodicProfileSamplers['normal'],
    color: color as unknown as LegacyPeriodicProfileSamplers['color'],
    alignedStandalone,
    standalone,
  });
}
