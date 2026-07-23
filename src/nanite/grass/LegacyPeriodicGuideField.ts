import type { PerspectiveCamera } from 'three';
import { HalfFloatType, LinearFilter, RGBAFormat } from 'three';
import { StorageBufferAttribute, StorageTexture } from 'three/webgpu';
import {
  Fn, float, instanceIndex, mix, smoothstep, textureStore, time, uint, uvec2, vec2, vec4,
} from 'three/tsl';
import type { NB, NF, NU, NV2, NV4 } from '../../gpu/TSLTypes';
import { canopyAt, cellHash2 } from '../../gpu/passes/Scatter';
import { gustAt, windContext, windExposure, windU } from '../../render/Wind';
import type { NaniteCam } from '../NaniteCommon';
import type { TerrainField } from '../world/TerrainField';
import { bcF2U, packHalfU, returnIf, sU32Views, sUvec4RO, toF, uniformF } from '../Tsl';
import {
  GRASS_CELL_METRES,
  GRASS_RAY_SWAY,
  GRASS_SALT,
  GUIDE_PITCH,
  GUIDE_RESOLUTION,
  GUIDE_SUB,
  GUIDE_TEXEL_COUNT,
  grassNumberParam,
} from './LegacyPeriodicGroundCoverPolicy';

export interface LegacyPeriodicGuideField {
  readonly kernel: unknown;
  readonly records: { element(index: NU | number): unknown };
  readonly field1: StorageTexture;
  readonly field2: StorageTexture;
  readonly field3: StorageTexture;
  readonly originFineX: ReturnType<typeof uniformF>;
  readonly originFineZ: ReturnType<typeof uniformF>;
  readonly rayOriginMetresX: ReturnType<typeof uniformF>;
  readonly rayOriginMetresZ: ReturnType<typeof uniformF>;
  readonly worldTileOriginX: ReturnType<typeof uniformF>;
  readonly worldTileOriginZ: ReturnType<typeof uniformF>;
  update(camera: PerspectiveCamera): void;
}

interface GuideOptions {
  readonly cam: NaniteCam;
  readonly field: TerrainField;
  readonly canopyTexture: StorageTexture | null;
  readonly enabledUniform: ReturnType<typeof uniformF>;
  readonly streamed: boolean;
}

/** @deprecated Camera-windowed guide used by the legacy periodic renderer. */
export function createLegacyPeriodicGuideField(options: GuideOptions): LegacyPeriodicGuideField {
  const { cam, field, streamed } = options;
  const heightAt = (position: NV2): NF => (
    streamed ? field.fieldHeightFinestHot(position) : field.fieldHeight(position, 0)
  );
  const byBiome = (biome: NF, values: number[]): NF => {
    let result = float(values[5] ?? 0) as unknown as NF;
    for (let index = 4; index >= 0; index--) {
      result = biome.equal(float(index)).select(float(values[index] ?? 0), result) as unknown as NF;
    }
    return result;
  };
  const densityAt = (position: NV2, height: NF): NF => {
    const biome = field.biomeAt(position);
    const fields = field.fieldsAt(position);
    const biomeId = biome.x.mul(255).add(0.5).floor() as unknown as NF;
    const slope = streamed ? field.fieldSlopeHot(position) : field.fieldSlope(position, 0);
    const waterY = field.fieldWaterYNearest(position);
    const above = height.sub(waterY) as unknown as NF;
    const riverDepth = waterY.sub(height).max(0) as unknown as NF;
    const bank = smoothstep(0.06, 0.5, above).mul(
      float(1).sub(smoothstep(0.2, 1.1, riverDepth).mul(0.78)),
    ) as unknown as NF;
    const canopy = options.canopyTexture
      ? canopyAt(options.canopyTexture, position)
      : float(0) as unknown as NF;
    return byBiome(biomeId, [0.18, 0.7, 0.62, 0.7, 1.5, 1.1])
      .mul(bank)
      .mul((biome.y as unknown as NF).mul(0.85).add(0.15))
      .mul(float(1).sub((fields.w as unknown as NF).mul(0.55)))
      .mul(float(1).sub(canopy.mul(0.45)))
      .mul((fields.x as unknown as NF).mul(0.35).add(0.75))
      .mul(float(1).sub((fields.z as unknown as NF).mul(0.95)))
      .mul(float(1).sub(smoothstep(0.55, 0.95, slope)))
      .mul(above.greaterThanEqual(0.04).select(float(1), float(0))) as unknown as NF;
  };

  const wordsPerRecord = 8;
  const wordCount = GUIDE_TEXEL_COUNT * wordsPerRecord;
  const attribute = new StorageBufferAttribute(new Uint32Array(wordCount), 1);
  attribute.name = 'legacyGroundCoverGuideRec';
  const writable = sU32Views(attribute, wordCount);
  const records4 = sUvec4RO(attribute, GUIDE_TEXEL_COUNT * 2);
  const records = {
    element(index: NU | number): unknown {
      const value = typeof index === 'number' ? uint(index) : index;
      return records4.element(value.mul(uint(2)) as unknown as NU);
    },
  };
  const originFineX = uniformF(0);
  const originFineZ = uniformF(0);
  const rayOriginMetresX = uniformF(0);
  const rayOriginMetresZ = uniformF(0);
  const worldTileOriginX = uniformF(0);
  const worldTileOriginZ = uniformF(0);
  const makeFieldTexture = (name: string): StorageTexture => {
    const result = new StorageTexture(GUIDE_RESOLUTION, GUIDE_RESOLUTION);
    result.type = HalfFloatType;
    result.format = RGBAFormat;
    result.magFilter = LinearFilter;
    result.minFilter = LinearFilter;
    result.generateMipmaps = false;
    result.name = name;
    return result;
  };
  const field1 = makeFieldTexture('legacyGroundCoverGuideField1');
  const field2 = makeFieldTexture('legacyGroundCoverGuideField2');
  const field3 = makeFieldTexture('legacyGroundCoverGuideField3');
  const kernel = (() => {
    const result = Fn(() => {
      returnIf((options.enabledUniform as unknown as NF).lessThan(0.5) as unknown as NB);
      const index = instanceIndex;
      returnIf(index.greaterThanEqual(uint(GUIDE_TEXEL_COUNT)));
      const texelX = index.mod(uint(GUIDE_RESOLUTION));
      const texelZ = index.div(uint(GUIDE_RESOLUTION));
      const tx = toF(texelX);
      const tz = toF(texelZ);
      const fineBase = vec2(originFineX as unknown as NF, originFineZ as unknown as NF)
        .add(vec2(tx, tz).mul(GUIDE_SUB)).toVar() as unknown as NV2;
      const worldPosition = fineBase.add(GUIDE_SUB / 2).mul(GRASS_CELL_METRES).toVar() as unknown as NV2;
      const guidePosition = vec2(tx, tz).mul(GUIDE_SUB).add(GUIDE_SUB / 2)
        .mul(GRASS_CELL_METRES).toVar() as unknown as NV2;
      const guideCamera = vec2(rayOriginMetresX as unknown as NF, rayOriginMetresZ as unknown as NF);
      const distance = (
        streamed
          ? guidePosition.sub(guideCamera)
          : worldPosition.sub(vec2(cam.camPos.x, cam.camPos.z))
      ).length().toVar() as unknown as NF;
      const ground = heightAt(worldPosition).toVar() as unknown as NF;
      const halfPitch = GUIDE_PITCH / 2;
      const slopeX = heightAt(worldPosition.add(vec2(halfPitch, 0)) as unknown as NV2)
        .sub(heightAt(worldPosition.sub(vec2(halfPitch, 0)) as unknown as NV2))
        .div(GUIDE_PITCH) as unknown as NF;
      const slopeZ = heightAt(worldPosition.add(vec2(0, halfPitch)) as unknown as NV2)
        .sub(heightAt(worldPosition.sub(vec2(0, halfPitch)) as unknown as NV2))
        .div(GUIDE_PITCH) as unknown as NF;
      const cornerDensity = (dx: number, dz: number): NF => {
        const offset = vec2(dx * halfPitch, dz * halfPitch);
        const samplePosition = worldPosition.add(offset) as unknown as NV2;
        return densityAt(samplePosition, heightAt(samplePosition)).toVar() as unknown as NF;
      };
      const proceduralDensity = cornerDensity(-1, -1).add(cornerDensity(1, -1))
        .add(cornerDensity(-1, 1)).add(cornerDensity(1, 1)).mul(0.25)
        .clamp(0, 1).toVar() as unknown as NF;
      const amplitude = windContext()
        ? (windU.strength as unknown as NF).mul(gustAt(worldPosition).mul(0.9).add(0.3))
            .mul(windExposure(worldPosition)) as unknown as NF
        : float(0) as unknown as NF;
      const controlA = field.hasGroundCover ? field.groundCoverAt(worldPosition) : vec4(0) as unknown as NV4;
      const controlB = field.hasGroundCover
        ? field.groundCoverLinearAt(worldPosition)
        : vec4(0, proceduralDensity, 0, 0) as unknown as NV4;
      const controlC = field.hasGroundCoverClosure
        ? field.groundCoverProfilesAt(worldPosition)
        : vec4(63 / 255, 0, 0, 0) as unknown as NV4;
      const coverDensity = (field.hasGroundCover ? controlB.y : proceduralDensity).clamp(0, 1) as unknown as NF;
      const top = coverDensity.greaterThanEqual(0.02).select(float(0.75), float(0)) as unknown as NF;
      const byte = (value: NF): NU => uint(value.mul(255).add(0.5).floor()) as unknown as NU;
      const typeA = byte(controlA.x as unknown as NF);
      const typeB = byte(controlA.y as unknown as NF);
      const clumpLo = byte(controlA.z as unknown as NF);
      const clumpHi = byte(controlA.w as unknown as NF);
      const maskLo = byte(controlC.x as unknown as NF);
      const maskHi = byte(controlC.y as unknown as NF);
      const profileA = field.hasGroundCoverClosure ? byte(controlC.z as unknown as NF) : typeA;
      const profileB = field.hasGroundCoverClosure ? byte(controlC.w as unknown as NF) : typeB;
      const mixWord = typeA.bitOr(typeB.shiftLeft(uint(8)))
        .bitOr(maskLo.shiftLeft(uint(16))).bitOr(maskHi.shiftLeft(uint(24))) as unknown as NU;
      const environmentWord = clumpLo.bitOr(clumpHi.shiftLeft(uint(8)))
        .bitOr(profileA.shiftLeft(uint(16))).bitOr(profileB.shiftLeft(uint(24))) as unknown as NU;
      const base = index.mul(uint(wordsPerRecord));
      writable.rw.element(base).assign(bcF2U(ground));
      writable.rw.element(base.add(uint(1))).assign(packHalfU(vec2(slopeX, slopeZ) as unknown as NV2));
      writable.rw.element(base.add(uint(2))).assign(mixWord);
      writable.rw.element(base.add(uint(3))).assign(environmentWord);
      writable.rw.element(base.add(uint(4))).assign(uint(0));
      writable.rw.element(base.add(uint(5))).assign(uint(0));
      writable.rw.element(base.add(uint(6))).assign(mixWord);
      writable.rw.element(base.add(uint(7))).assign(environmentWord);

      const smoothNoise = (salt: number): NV2 => {
        const qx = worldPosition.x.mul(1 / (GUIDE_PITCH * 4)) as unknown as NF;
        const qz = worldPosition.y.mul(1 / (GUIDE_PITCH * 4)) as unknown as NF;
        const ix = qx.floor().toVar() as unknown as NF;
        const iz = qz.floor().toVar() as unknown as NF;
        const fx = smoothstep(0, 1, qx.sub(ix)) as unknown as NF;
        const fz = smoothstep(0, 1, qz.sub(iz)) as unknown as NF;
        const corner = (dx: number, dz: number): NV2 => cellHash2(
          vec2(ix.add(dx), iz.add(dz)) as unknown as NV2,
          GRASS_SALT ^ salt,
        ) as unknown as NV2;
        return mix(mix(corner(0, 0), corner(1, 0), fx), mix(corner(0, 1), corner(1, 1), fx), fz) as unknown as NV2;
      };
      const rotation = (smoothNoise(0x0b0b).x as unknown as NF).mul(Math.PI * 2).toVar() as unknown as NF;
      let windX = float(0) as unknown as NF;
      let windZ = float(0) as unknown as NF;
      if (windContext()) {
        const strength = (windU.strength as unknown as NF).toVar() as unknown as NF;
        const shear = amplitude.mul(strength.mul(0.55).add(0.6)).mul(0.45)
          .div(top.max(0.35)).toVar() as unknown as NF;
        const direction = vec2(windU.dir as unknown as NV2);
        const noise = smoothNoise(0x5151);
        const phase = (noise.x as unknown as NF).mul(Math.PI * 2).toVar() as unknown as NF;
        const gust = time.mul(0.22).add(phase).sin().mul(0.35).add(0.65) as unknown as NF;
        const sway = time.mul(0.55).add(phase).sin().mul(gust).mul(0.45)
          .add(time.mul(2.4).add(phase.mul(1.7)).sin().mul(0.1))
          .mul(strength.mul(0.7).add(0.15)).mul(amplitude.mul(1.5).add(0.25).min(1))
          .mul(float(1).sub(smoothstep(50, 110, distance))).mul(GRASS_RAY_SWAY) as unknown as NF;
        const wobble = (noise.y as unknown as NF).sub(0.5).mul(0.8) as unknown as NF;
        windX = direction.x.mul(shear).add(direction.x.sub(direction.y.mul(wobble)).mul(sway)) as unknown as NF;
        windZ = direction.y.mul(shear).add(direction.y.add(direction.x.mul(wobble)).mul(sway)) as unknown as NF;
      }
      const randomLean = grassNumberParam('grassrandlean', 0, 0, 1);
      const leanNoise = smoothNoise(0x3333);
      const arcAngle = (leanNoise.x as unknown as NF).mul(Math.PI * 2) as unknown as NF;
      const arcMagnitude = (leanNoise.y as unknown as NF).mul(0.25 * randomLean).add(0.12 * randomLean) as unknown as NF;
      const leanAngle = (leanNoise.x as unknown as NF).mul(Math.PI * 2).add(2.1) as unknown as NF;
      const leanMagnitude = (leanNoise.y as unknown as NF).mul(0.05 * randomLean).add(0.02 * randomLean) as unknown as NF;
      textureStore(field1, uvec2(texelX, texelZ), vec4(
        leanAngle.cos().mul(leanMagnitude), leanAngle.sin().mul(leanMagnitude), windX, windZ,
      )).toWriteOnly();
      textureStore(field2, uvec2(texelX, texelZ), vec4(
        rotation.cos(), rotation.sin(), arcAngle.cos().mul(arcMagnitude), arcAngle.sin().mul(arcMagnitude),
      )).toWriteOnly();
      textureStore(field3, uvec2(texelX, texelZ), vec4(coverDensity, controlB.x, controlB.z, controlB.w)).toWriteOnly();
    })().compute(GUIDE_TEXEL_COUNT, [256]);
    (result as unknown as { setName(name: string): void }).setName('legacyGroundCoverGuide');
    return result;
  })();

  return Object.freeze({
    kernel,
    records,
    field1,
    field2,
    field3,
    originFineX,
    originFineZ,
    rayOriginMetresX,
    rayOriginMetresZ,
    worldTileOriginX,
    worldTileOriginZ,
    update(camera: PerspectiveCamera): void {
      const half = (GUIDE_RESOLUTION / 2) * GUIDE_SUB;
      originFineX.value = Math.round(camera.position.x / GUIDE_PITCH) * GUIDE_SUB - half;
      originFineZ.value = Math.round(camera.position.z / GUIDE_PITCH) * GUIDE_SUB - half;
      rayOriginMetresX.value = camera.position.x - originFineX.value * GRASS_CELL_METRES;
      rayOriginMetresZ.value = camera.position.z - originFineZ.value * GRASS_CELL_METRES;
      const tileX = originFineX.value / GUIDE_SUB;
      const tileZ = originFineZ.value / GUIDE_SUB;
      worldTileOriginX.value = tileX - Math.floor(tileX / 4096) * 4096;
      worldTileOriginZ.value = tileZ - Math.floor(tileZ / 4096) * 4096;
    },
  });
}
