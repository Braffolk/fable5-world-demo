/** Production-resolution anisotropic GBR4/v3 periodic-cap cook.
 *
 * Runtime address: (cap chart, phase X/Z, compact slope X/Z).  Phase and
 * direction use different resolutions; the complete tensor is packed as one
 * phase tile per (chart,slopeZ,slopeX).  Hardware bilinear filtering performs
 * the phase interpolation, four fixed taps perform slope interpolation, and
 * one nearest categorical tap supplies both disjoint representatives.
 */

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';
import {
  TriangleBvh,
  decodeOwnedProfileGeometry,
  type CensusVec3,
  type DecodedOwnedProfileGeometry,
} from '../../ExteriorClosureCensus';
import { periodicNearestSuccessor } from '../../OriginAwareRayTruth';
import {
  addScaled,
  align,
  attributes,
  canonicalJson,
  clamp,
  cross,
  encodeOct,
  normalized,
  sha256,
} from './cook-gbr4-brick-assets';

const WORKSPACE = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
const SOURCE = resolve(WORKSPACE, 'src/assets/groundcover/calamagrostis-canescens.gcrp');
const EXPECTED_SOURCE_SHA = '2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c';
const HEADER_BYTES = 512;
const CHART_COUNT = 2;
const PHASE_CELLS = 128;
const SLOPE_CELLS = 12;
const PHASE_SAMPLES = PHASE_CELLS + 1;
const SLOPE_SAMPLES = SLOPE_CELLS + 1;
const QUADRATURE_SIDE = 2;
const STANDOFF_METRES = 4;
const PIXEL_ANGLE = 60 * Math.PI / 180 / 1920;
const HORIZON_METRES = 155;
const CARRIER_GUARD = 0.025;
const COLOR_TEXEL_BYTES = 4;
const CATEGORICAL_TEXEL_BYTES = 8;

interface SurfaceSample {
  depth01: number;
  color: CensusVec3;
  normal: CensusVec3;
}

interface FilteredRecord {
  color: readonly [number, number, number, number];
  depth: number;
  oct: number;
}

interface NodeRecord {
  front: FilteredRecord;
  back: FilteredRecord;
}

interface PeriodicEvidence {
  tracedSubrays: number;
  hitSubrays: number;
  hitsBeyondFirstTileExit: number;
  maximumHitDistanceMetres: number;
}

function capDirection(chart: number, compactX: number, compactZ: number): CensusVec3 {
  const sx = compactX * 2 - 1;
  const sz = compactZ * 2 - 1;
  const boundaryX = Math.abs(sx) >= 1 - 1e-12;
  const boundaryZ = Math.abs(sz) >= 1 - 1e-12;
  if (boundaryX || boundaryZ) {
    return normalized([boundaryX ? Math.sign(sx) : 0, 0, boundaryZ ? Math.sign(sz) : 0]);
  }
  return normalized([
    sx / Math.max(1e-30, 1 - Math.abs(sx)),
    chart === 0 ? -1 : 1,
    sz / Math.max(1e-30, 1 - Math.abs(sz)),
  ]);
}

function cluster(samples: readonly SurfaceSample[]): readonly [SurfaceSample[], SurfaceSample[]] {
  if (samples.length < 2) return [[...samples], []];
  let c0 = Math.min(...samples.map((sample) => sample.depth01));
  let c1 = Math.max(...samples.map((sample) => sample.depth01));
  let g0: SurfaceSample[] = [];
  let g1: SurfaceSample[] = [];
  for (let iteration = 0; iteration < 8; iteration++) {
    g0 = [];
    g1 = [];
    for (const sample of samples) {
      (Math.abs(sample.depth01 - c0) <= Math.abs(sample.depth01 - c1) ? g0 : g1).push(sample);
    }
    if (g0.length) c0 = g0.reduce((sum, sample) => sum + sample.depth01, 0) / g0.length;
    if (g1.length) c1 = g1.reduce((sum, sample) => sum + sample.depth01, 0) / g1.length;
  }
  return c0 <= c1 ? [g0, g1] : [g1, g0];
}

function filtered(group: readonly SurfaceSample[], total: number): FilteredRecord {
  if (!group.length) return { color: [0, 0, 0, 0], depth: 0, oct: encodeOct([0, 1, 0]) };
  const rgb: [number, number, number] = [0, 0, 0];
  const mean = group.reduce((sum, sample) => sum + sample.depth01, 0) / group.length;
  let representative = group[0]!;
  for (const sample of group) {
    if (Math.abs(sample.depth01 - mean) < Math.abs(representative.depth01 - mean)) representative = sample;
    for (let channel = 0; channel < 3; channel++) rgb[channel] += sample.color[channel]! / total;
  }
  return {
    color: [rgb[0], rgb[1], rgb[2], group.length / total],
    depth: Math.round(clamp(representative.depth01, 0, 1) * 65535),
    oct: encodeOct(representative.normal),
  };
}

function integrate(
  bytes: Uint8Array,
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  chart: number,
  phaseX: number,
  phaseZ: number,
  compactX: number,
  compactZ: number,
  capY: number,
  evidence: PeriodicEvidence,
): NodeRecord {
  const central = capDirection(chart, compactX, compactZ);
  if (Math.abs(central[1]) <= 1e-12) {
    const miss = filtered([], QUADRATURE_SIDE ** 2);
    return { front: miss, back: miss };
  }
  const q: CensusVec3 = [
    geometry.tileOriginX + phaseX * geometry.tileSizeX,
    capY,
    geometry.tileOriginZ + phaseZ * geometry.tileSizeZ,
  ];
  const camera = addScaled(q, central, -STANDOFF_METRES);
  const right = Math.abs(central[1]) > 0.999999
    ? ([1, 0, 0] as const)
    : normalized(cross(central, [0, 1, 0]));
  const up = normalized(cross(right, central));
  const tangent = Math.tan(PIXEL_ANGLE * 0.5);
  const samples: SurfaceSample[] = [];
  for (let sy = 0; sy < QUADRATURE_SIDE; sy++) for (let sx = 0; sx < QUADRATURE_SIDE; sx++) {
    const px = (2 * (sx + 0.5) / QUADRATURE_SIDE - 1) * tangent;
    const py = (2 * (sy + 0.5) / QUADRATURE_SIDE - 1) * tangent;
    const direction = normalized([
      central[0] + right[0] * px + up[0] * py,
      central[1] + right[1] * px + up[1] * py,
      central[2] + right[2] * px + up[2] * py,
    ]);
    if ((chart === 0 && direction[1] >= -1e-12) || (chart === 1 && direction[1] <= 1e-12)) continue;
    const entryT = (capY - camera[1]) / direction[1];
    if (!(entryT > 0)) continue;
    const entry = addScaled(camera, direction, entryT);
    evidence.tracedSubrays++;
    const hit = periodicNearestSuccessor(geometry, bvh, entry, direction, HORIZON_METRES, 0);
    if (!hit) continue;
    evidence.hitSubrays++;
    evidence.maximumHitDistanceMetres = Math.max(evidence.maximumHitDistanceMetres, hit.t);
    let firstTileExit = Number.POSITIVE_INFINITY;
    for (const [axis, origin, size] of [
      [0, geometry.tileOriginX, geometry.tileSizeX],
      [2, geometry.tileOriginZ, geometry.tileSizeZ],
    ] as const) {
      const component = direction[axis]!;
      if (Math.abs(component) <= 1e-15) continue;
      const cell = Math.floor((entry[axis]! - origin) / size);
      const boundary = origin + (cell + (component > 0 ? 1 : 0)) * size;
      const distance = (boundary - entry[axis]!) / component;
      if (distance > 1e-9) firstTileExit = Math.min(firstTileExit, distance);
    }
    if (hit.t > firstTileExit + 1e-8) evidence.hitsBeyondFirstTileExit++;
    const decoded = attributes(bytes, geometry, entry, direction, hit);
    samples.push({ depth01: clamp(hit.t / HORIZON_METRES, 0, 1), ...decoded });
  }
  const groups = cluster(samples);
  return {
    front: filtered(groups[0], QUADRATURE_SIDE ** 2),
    back: filtered(groups[1], QUADRATURE_SIDE ** 2),
  };
}

function byte(value: number): number {
  return Math.round(clamp(value, 0, 1) * 255);
}

async function main(): Promise<void> {
  const started = performance.now();
  const sourceBytes = readFileSync(SOURCE);
  const sourceSha = sha256(sourceBytes);
  if (sourceSha !== EXPECTED_SOURCE_SHA) throw new Error(`accepted source changed: ${sourceSha}`);
  const geometry = decodeOwnedProfileGeometry(sourceBytes);
  const capY = [geometry.bounds.max[1] + CARRIER_GUARD, geometry.bounds.min[1] - CARRIER_GUARD] as const;
  console.error(`[gbr4-v3] BVH for ${geometry.triangleCount.toLocaleString()} triangles`);
  const bvh = TriangleBvh.build(geometry);

  const directionTileCount = CHART_COUNT * SLOPE_SAMPLES * SLOPE_SAMPLES;
  const atlasColumns = Math.ceil(Math.sqrt(directionTileCount));
  const atlasRows = Math.ceil(directionTileCount / atlasColumns);
  const atlasWidth = atlasColumns * PHASE_SAMPLES;
  const atlasHeight = atlasRows * PHASE_SAMPLES;
  const atlasTexels = atlasWidth * atlasHeight;
  const chartTableBytes = CHART_COUNT * 16;
  const scaleTableBytes = 64;
  const colorBytes = atlasTexels * COLOR_TEXEL_BYTES;
  const categoricalBytes = atlasTexels * CATEGORICAL_TEXEL_BYTES;
  const chartTableOffset = HEADER_BYTES;
  const scaleTableOffset = align(chartTableOffset + chartTableBytes);
  const color0Offset = align(scaleTableOffset + scaleTableBytes);
  const color1Offset = align(color0Offset + colorBytes);
  const categoricalOffset = align(color1Offset + colorBytes);
  const totalBytes = align(categoricalOffset + categoricalBytes);
  const recipe = {
    schema: 'laas-gbr4-periodic-cap-direct-recipe/v3',
    implementationSha256: sha256(readFileSync(fileURLToPath(import.meta.url))),
    sourceSha256: sourceSha,
    phaseCells: PHASE_CELLS,
    slopeCells: SLOPE_CELLS,
    quadratureSide: QUADRATURE_SIDE,
    standoffMetres: STANDOFF_METRES,
    pixelAngleRadians: PIXEL_ANGLE,
    horizonMetres: HORIZON_METRES,
    color: 'two disjoint premultiplied rgba8 strata',
    categorical: 'rgba16uint depth0,oct0,depth1,oct1',
  };
  const recipeSha = sha256(canonicalJson(recipe));
  const artifact = Buffer.alloc(totalBytes);
  const view = new DataView(artifact.buffer, artifact.byteOffset, artifact.byteLength);
  artifact.write('GBR4', 0, 'ascii');
  view.setUint32(4, 3, true); view.setUint32(8, HEADER_BYTES, true); view.setUint32(12, geometry.profileId, true);
  view.setUint32(16, CHART_COUNT, true); view.setUint32(20, 3, true); view.setUint32(24, 4, true);
  view.setUint32(28, PHASE_CELLS, true); view.setUint32(32, SLOPE_CELLS, true);
  view.setUint32(36, PHASE_SAMPLES, true); view.setUint32(40, SLOPE_SAMPLES, true);
  view.setUint32(44, 1, true); view.setUint32(48, 2, true);
  view.setUint32(52, chartTableOffset, true); view.setUint32(56, chartTableBytes, true);
  view.setUint32(68, scaleTableOffset, true); view.setUint32(72, scaleTableBytes, true);
  view.setUint32(76, color0Offset, true); view.setUint32(80, colorBytes, true);
  view.setUint32(84, color1Offset, true); view.setUint32(88, colorBytes, true);
  view.setUint32(92, categoricalOffset, true); view.setUint32(96, categoricalBytes, true);
  view.setUint32(100, atlasWidth, true); view.setUint32(104, atlasHeight, true);
  view.setUint32(108, PHASE_SAMPLES, true); view.setUint32(112, PHASE_SAMPLES, true);
  view.setUint32(116, directionTileCount, true); view.setUint32(120, 65535, true); view.setUint32(124, 31, true);
  view.setFloat32(128, geometry.tileOriginX, true); view.setFloat32(132, geometry.tileOriginZ, true);
  view.setFloat32(136, geometry.tileSizeX, true); view.setFloat32(140, geometry.tileSizeZ, true);
  const minimum: CensusVec3 = [geometry.tileOriginX, capY[1], geometry.tileOriginZ];
  const maximum: CensusVec3 = [geometry.tileOriginX + geometry.tileSizeX, capY[0], geometry.tileOriginZ + geometry.tileSizeZ];
  for (let axis = 0; axis < 3; axis++) view.setFloat32(144 + axis * 4, minimum[axis]!, true);
  for (let axis = 0; axis < 3; axis++) view.setFloat32(156 + axis * 4, maximum[axis]!, true);
  artifact.set(Buffer.from(sourceSha, 'hex'), 168); artifact.set(Buffer.from(recipeSha, 'hex'), 200);
  view.setFloat32(232, PIXEL_ANGLE, true); view.setFloat32(236, STANDOFF_METRES, true);
  view.setUint32(240, QUADRATURE_SIDE, true); view.setFloat32(244, HORIZON_METRES, true);
  view.setUint32(248, 2, true); view.setUint32(252, 3, true);

  const chartTable = new DataView(artifact.buffer, artifact.byteOffset + chartTableOffset, chartTableBytes);
  const tilesPerChart = SLOPE_SAMPLES * SLOPE_SAMPLES;
  for (let chart = 0; chart < CHART_COUNT; chart++) {
    chartTable.setUint8(chart * 16, chart);
    chartTable.setInt8(chart * 16 + 1, chart === 0 ? -1 : 1);
    chartTable.setUint32(chart * 16 + 4, chart * tilesPerChart, true);
    chartTable.setUint32(chart * 16 + 8, tilesPerChart, true);
  }
  const scale = new DataView(artifact.buffer, artifact.byteOffset + scaleTableOffset, scaleTableBytes);
  scale.setFloat32(0, STANDOFF_METRES, true); scale.setFloat32(4, PIXEL_ANGLE, true);
  scale.setUint32(12, directionTileCount, true); scale.setUint32(20, directionTileCount, true);
  scale.setUint32(24, atlasWidth, true); scale.setUint32(28, atlasHeight, true);

  const color0 = new Uint8Array(artifact.buffer, artifact.byteOffset + color0Offset, colorBytes);
  const color1 = new Uint8Array(artifact.buffer, artifact.byteOffset + color1Offset, colorBytes);
  const categorical = new Uint16Array(
    artifact.buffer,
    artifact.byteOffset + categoricalOffset,
    categoricalBytes / 2,
  );
  const evidence: PeriodicEvidence = {
    tracedSubrays: 0,
    hitSubrays: 0,
    hitsBeyondFirstTileExit: 0,
    maximumHitDistanceMetres: 0,
  };
  const totalNodes = directionTileCount * PHASE_SAMPLES * PHASE_SAMPLES;
  console.error(`[gbr4-v3] cooking ${totalNodes.toLocaleString()} nodes x ${QUADRATURE_SIDE ** 2} subrays`);
  for (let chart = 0; chart < CHART_COUNT; chart++) {
    for (let i3 = 0; i3 < SLOPE_SAMPLES; i3++) {
      for (let i2 = 0; i2 < SLOPE_SAMPLES; i2++) {
        const tile = (chart * SLOPE_SAMPLES + i3) * SLOPE_SAMPLES + i2;
        const tileX = (tile % atlasColumns) * PHASE_SAMPLES;
        const tileY = Math.floor(tile / atlasColumns) * PHASE_SAMPLES;
        for (let i1 = 0; i1 < PHASE_SAMPLES; i1++) for (let i0 = 0; i0 < PHASE_SAMPLES; i0++) {
          const record = integrate(
            sourceBytes,
            geometry,
            bvh,
            chart,
            i0 / PHASE_CELLS,
            i1 / PHASE_CELLS,
            i2 / SLOPE_CELLS,
            i3 / SLOPE_CELLS,
            capY[chart]!,
            evidence,
          );
          const texel = (tileY + i1) * atlasWidth + tileX + i0;
          const colorBase = texel * 4;
          for (let channel = 0; channel < 4; channel++) {
            color0[colorBase + channel] = byte(record.front.color[channel]!);
            color1[colorBase + channel] = byte(record.back.color[channel]!);
          }
          categorical[colorBase] = record.front.depth;
          categorical[colorBase + 1] = record.front.oct;
          categorical[colorBase + 2] = record.back.depth;
          categorical[colorBase + 3] = record.back.oct;
        }
      }
      console.error(`[gbr4-v3] chart ${chart + 1}/${CHART_COUNT}, slope row ${i3 + 1}/${SLOPE_SAMPLES}`);
    }
  }

  const outputRoot = resolve(
    WORKSPACE,
    'data/work/groundcover-gbr4-direct-cap-assets',
    sourceSha.slice(0, 16),
    recipeSha.slice(0, 16),
  );
  const qaRoot = resolve(outputRoot, 'qa');
  mkdirSync(qaRoot, { recursive: true });
  const assetPath = resolve(outputRoot, 'calamagrostis-canescens.gbr4');
  const stablePath = resolve(WORKSPACE, 'src/assets/groundcover/calamagrostis-canescens.gbr4');
  writeFileSync(assetPath, artifact);
  writeFileSync(stablePath, artifact);

  const verticalTile = Math.floor(SLOPE_SAMPLES / 2) * SLOPE_SAMPLES + Math.floor(SLOPE_SAMPLES / 2);
  const qaRaw = Buffer.alloc(PHASE_SAMPLES * PHASE_SAMPLES * 4);
  const qaTileX = (verticalTile % atlasColumns) * PHASE_SAMPLES;
  const qaTileY = Math.floor(verticalTile / atlasColumns) * PHASE_SAMPLES;
  for (let y = 0; y < PHASE_SAMPLES; y++) for (let x = 0; x < PHASE_SAMPLES; x++) {
    const source = ((qaTileY + y) * atlasWidth + qaTileX + x) * 4;
    const target = (y * PHASE_SAMPLES + x) * 4;
    qaRaw.set(color0.subarray(source, source + 4), target);
  }
  const qaPath = resolve(qaRoot, '01-top-vertical-front.png');
  await sharp(qaRaw, { raw: { width: PHASE_SAMPLES, height: PHASE_SAMPLES, channels: 4 } })
    .resize(PHASE_SAMPLES * 4, PHASE_SAMPLES * 4, { kernel: 'nearest' })
    .png().toFile(qaPath);
  const report = {
    schema: 'laas-gbr4-periodic-cap-direct-asset/v3',
    created: new Date().toISOString(),
    recipe,
    source: {
      path: relative(WORKSPACE, SOURCE),
      sha256: sourceSha,
      vertices: geometry.vertexCount,
      triangles: geometry.triangleCount,
    },
    container: {
      path: relative(WORKSPACE, assetPath),
      stablePath: relative(WORKSPACE, stablePath),
      sha256: sha256(artifact),
      bytes: artifact.byteLength,
      MiB: artifact.byteLength / 1048576,
      estimatedGpuResidentMiB: (colorBytes * 2 + atlasTexels * 16) / 1048576,
    },
    tensor: {
      phaseCells: PHASE_CELLS,
      phaseMetres: geometry.tileSizeX / PHASE_CELLS,
      slopeCells: SLOPE_CELLS,
      directionTiles: directionTileCount,
      atlas: [atlasWidth, atlasHeight],
      nodes: totalNodes,
    },
    evidence,
    runtimeBudget: {
      logicalReads: 9,
      filteredColorReads: 8,
      categoricalReads: 1,
      loops: 0,
      marches: 0,
      candidates: 0,
      runtimeGeometry: false,
    },
    qa: { path: relative(WORKSPACE, qaPath), sha256: sha256(readFileSync(qaPath)) },
    elapsedMilliseconds: performance.now() - started,
  };
  const reportPath = resolve(outputRoot, 'report.json');
  writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`);
  console.log(JSON.stringify({ outputRoot: relative(WORKSPACE, outputRoot), report }, null, 2));
}

await main();
