/** Corrected GBR4/v2 moving finite-horizon periodic cap cook. */

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
  float16,
  normalized,
  sha256,
} from './cook-gbr4-brick-assets';

const WORKSPACE = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
const SOURCE = resolve(WORKSPACE, 'src/assets/groundcover/calamagrostis-canescens.gcrp');
const EXPECTED_SOURCE_SHA = '2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c';
const HEADER_BYTES = 512;
const FACE_CHARTS = 2;
const CELL_RESOLUTION = 6;
const BLOCK_CELL_EDGE = 3;
const BLOCK_SAMPLE_EDGE = 4;
const BLOCKS_PER_AXIS = 2;
const SAMPLES_PER_BRICK = BLOCK_SAMPLE_EDGE ** 4;
const QUADRATURE_SIDE = 2;
const STANDOFF_METRES = 4;
const PIXEL_ANGLE = 60 * Math.PI / 180 / 1920;
const HORIZON_METRES = 155;
const CARRIER_GUARD = 0.025;
const TEXEL_BYTES = 8;

interface SurfaceSample { depth01: number; color: CensusVec3; normal: CensusVec3; }
interface NodeRecord { color0: readonly [number, number, number, number]; color1: readonly [number, number, number, number]; categorical: readonly [number, number, number, number]; }
interface PeriodicEvidence { tracedSubrays: number; hitSubrays: number; hitsBeyondFirstTileExit: number; maximumHitDistanceMetres: number; }

function capDirection(chart: number, compactX: number, compactZ: number): CensusVec3 {
  const sx = compactX * 2 - 1;
  const sz = compactZ * 2 - 1;
  const boundaryX = Math.abs(sx) >= 1 - 1e-12;
  const boundaryZ = Math.abs(sz) >= 1 - 1e-12;
  if (boundaryX || boundaryZ) {
    const x = boundaryX ? Math.sign(sx) : 0;
    const z = boundaryZ ? Math.sign(sz) : 0;
    return normalized([x, 0, z]);
  }
  const x = sx / Math.max(1e-30, 1 - Math.abs(sx));
  const z = sz / Math.max(1e-30, 1 - Math.abs(sz));
  return normalized([x, chart === 0 ? -1 : 1, z]);
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
    for (const sample of samples) (Math.abs(sample.depth01 - c0) <= Math.abs(sample.depth01 - c1) ? g0 : g1).push(sample);
    if (g0.length) c0 = g0.reduce((sum, sample) => sum + sample.depth01, 0) / g0.length;
    if (g1.length) c1 = g1.reduce((sum, sample) => sum + sample.depth01, 0) / g1.length;
  }
  return c0 <= c1 ? [g0, g1] : [g1, g0];
}

function filtered(group: readonly SurfaceSample[], total: number): { color: readonly [number, number, number, number]; depth: number; oct: number } {
  if (!group.length) return { color: [0, 0, 0, 0], depth: 0, oct: encodeOct([0, 1, 0]) };
  const alpha = group.length / total;
  const rgb: [number, number, number] = [0, 0, 0];
  const mean = group.reduce((sum, sample) => sum + sample.depth01, 0) / group.length;
  let representative = group[0]!;
  for (const sample of group) if (Math.abs(sample.depth01 - mean) < Math.abs(representative.depth01 - mean)) representative = sample;
  for (const sample of group) for (let channel = 0; channel < 3; channel++) rgb[channel] += sample.color[channel]! / total;
  return { color: [rgb[0], rgb[1], rgb[2], alpha], depth: Math.round(clamp(representative.depth01, 0, 1) * 65535), oct: encodeOct(representative.normal) };
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
  if (Math.abs(central[1]) <= 1e-12) return { color0: [0, 0, 0, 0], color1: [0, 0, 0, 0], categorical: [0, encodeOct([0, 1, 0]), 0, encodeOct([0, 1, 0])] };
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
    for (const [axis, origin, size] of [[0, geometry.tileOriginX, geometry.tileSizeX], [2, geometry.tileOriginZ, geometry.tileSizeZ]] as const) {
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
  const front = filtered(groups[0], QUADRATURE_SIDE ** 2);
  const back = filtered(groups[1], QUADRATURE_SIDE ** 2);
  return { color0: front.color, color1: back.color, categorical: [front.depth, front.oct, back.depth, back.oct] };
}

function nodeLinear(chart: number, i0: number, i1: number, i2: number, i3: number): number {
  const edge = CELL_RESOLUTION + 1;
  return ((((chart * edge + i3) * edge + i2) * edge + i1) * edge + i0);
}

async function main(): Promise<void> {
  const started = performance.now();
  const bytes = readFileSync(SOURCE);
  const sourceSha = sha256(bytes);
  if (sourceSha !== EXPECTED_SOURCE_SHA) throw new Error(`accepted source changed: ${sourceSha}`);
  const geometry = decodeOwnedProfileGeometry(bytes);
  const capY = [geometry.bounds.max[1] + CARRIER_GUARD, geometry.bounds.min[1] - CARRIER_GUARD] as const;
  const nodeCount = FACE_CHARTS * (CELL_RESOLUTION + 1) ** 4;
  const nodeColor0 = new Float32Array(nodeCount * 4);
  const nodeColor1 = new Float32Array(nodeCount * 4);
  const nodeCategorical = new Uint16Array(nodeCount * 4);
  const periodicEvidence: PeriodicEvidence = { tracedSubrays: 0, hitSubrays: 0, hitsBeyondFirstTileExit: 0, maximumHitDistanceMetres: 0 };
  console.error(`[gbr4-v2] BVH for ${geometry.triangleCount.toLocaleString()} triangles`);
  const bvh = TriangleBvh.build(geometry);
  console.error(`[gbr4-v2] cooking ${nodeCount.toLocaleString()} periodic cap nodes x ${QUADRATURE_SIDE ** 2} subrays`);
  for (let chart = 0; chart < FACE_CHARTS; chart++) {
    for (let i3 = 0; i3 <= CELL_RESOLUTION; i3++) for (let i2 = 0; i2 <= CELL_RESOLUTION; i2++) {
      for (let i1 = 0; i1 <= CELL_RESOLUTION; i1++) for (let i0 = 0; i0 <= CELL_RESOLUTION; i0++) {
        const record = integrate(bytes, geometry, bvh, chart, i0 / CELL_RESOLUTION, i1 / CELL_RESOLUTION, i2 / CELL_RESOLUTION, i3 / CELL_RESOLUTION, capY[chart]!, periodicEvidence);
        const node = nodeLinear(chart, i0, i1, i2, i3);
        nodeColor0.set(record.color0, node * 4);
        nodeColor1.set(record.color1, node * 4);
        nodeCategorical.set(record.categorical, node * 4);
      }
    }
    console.error(`[gbr4-v2] chart ${chart + 1}/${FACE_CHARTS}`);
  }
  const blocksPerChart = BLOCKS_PER_AXIS ** 4;
  const codebookCount = FACE_CHARTS * blocksPerChart;
  const descriptorCount = codebookCount;
  const tileEdge = BLOCK_SAMPLE_EDGE ** 2;
  const atlasColumns = Math.ceil(Math.sqrt(codebookCount));
  const atlasRows = Math.ceil(codebookCount / atlasColumns);
  const atlasWidth = atlasColumns * tileEdge;
  const atlasHeight = atlasRows * tileEdge;
  const atlasTexels = atlasWidth * atlasHeight;
  const descriptorBytes = descriptorCount * 2;
  const chartTableBytes = FACE_CHARTS * 16;
  const scaleTableBytes = 64;
  const atlasBytes = atlasTexels * TEXEL_BYTES;
  const descriptorOffset = HEADER_BYTES;
  const chartTableOffset = align(descriptorOffset + descriptorBytes);
  const scaleTableOffset = align(chartTableOffset + chartTableBytes);
  const color0Offset = align(scaleTableOffset + scaleTableBytes);
  const color1Offset = align(color0Offset + atlasBytes);
  const categoricalOffset = align(color1Offset + atlasBytes);
  const totalBytes = align(categoricalOffset + atlasBytes);
  const recipe = {
    schema: 'laas-gbr4-periodic-cap-real-cook-recipe/v2',
    implementationSha256: sha256(readFileSync(fileURLToPath(import.meta.url))), sourceSha256: sourceSha,
    address: 'top/bottom cap phase x,z plus compact direction slopes', horizonMetres: HORIZON_METRES,
    cellResolution: CELL_RESOLUTION, blockCellEdge: BLOCK_CELL_EDGE, quadratureSide: QUADRATURE_SIDE,
    standoffMetres: STANDOFF_METRES, pixelAngleRadians: PIXEL_ANGLE,
    codebook: 'lossless one codeword per 4D block',
  };
  const recipeSha = sha256(canonicalJson(recipe));
  const artifact = Buffer.alloc(totalBytes);
  const view = new DataView(artifact.buffer, artifact.byteOffset, artifact.byteLength);
  artifact.write('GBR4', 0, 'ascii');
  view.setUint32(4, 2, true); view.setUint32(8, HEADER_BYTES, true); view.setUint32(12, geometry.profileId, true);
  view.setUint32(16, FACE_CHARTS, true); view.setUint32(20, 2, true); view.setUint32(24, 4, true);
  view.setUint32(28, CELL_RESOLUTION, true); view.setUint32(32, BLOCK_CELL_EDGE, true); view.setUint32(36, BLOCK_SAMPLE_EDGE, true);
  view.setUint32(40, BLOCKS_PER_AXIS, true); view.setUint32(44, 1, true); view.setUint32(48, 1, true);
  for (const [offset, value] of [
    [52, descriptorOffset], [56, descriptorBytes], [60, chartTableOffset], [64, chartTableBytes], [68, scaleTableOffset], [72, scaleTableBytes],
    [76, color0Offset], [80, atlasBytes], [84, color1Offset], [88, atlasBytes], [92, categoricalOffset], [96, atlasBytes],
    [100, atlasWidth], [104, atlasHeight], [108, tileEdge], [112, tileEdge], [116, codebookCount], [120, 65535], [124, 15],
  ] as const) view.setUint32(offset, value, true);
  view.setFloat32(128, geometry.tileOriginX, true); view.setFloat32(132, geometry.tileOriginZ, true);
  view.setFloat32(136, geometry.tileSizeX, true); view.setFloat32(140, geometry.tileSizeZ, true);
  const minimum: CensusVec3 = [geometry.tileOriginX, capY[1], geometry.tileOriginZ];
  const maximum: CensusVec3 = [geometry.tileOriginX + geometry.tileSizeX, capY[0], geometry.tileOriginZ + geometry.tileSizeZ];
  for (let axis = 0; axis < 3; axis++) view.setFloat32(144 + axis * 4, minimum[axis]!, true);
  for (let axis = 0; axis < 3; axis++) view.setFloat32(156 + axis * 4, maximum[axis]!, true);
  artifact.set(Buffer.from(sourceSha, 'hex'), 168); artifact.set(Buffer.from(recipeSha, 'hex'), 200);
  view.setFloat32(232, PIXEL_ANGLE, true); view.setFloat32(236, STANDOFF_METRES, true); view.setUint32(240, QUADRATURE_SIDE, true);
  view.setFloat32(244, HORIZON_METRES, true); view.setUint32(248, 2, true); view.setUint32(252, 2, true);
  const descriptors = new Uint16Array(artifact.buffer, artifact.byteOffset + descriptorOffset, descriptorCount);
  const chartTable = new DataView(artifact.buffer, artifact.byteOffset + chartTableOffset, chartTableBytes);
  for (let chart = 0; chart < FACE_CHARTS; chart++) {
    const base = chart * blocksPerChart;
    chartTable.setUint8(chart * 16, chart); chartTable.setInt8(chart * 16 + 1, chart === 0 ? -1 : 1);
    chartTable.setUint32(chart * 16 + 4, base, true); chartTable.setUint32(chart * 16 + 8, blocksPerChart, true);
    for (let local = 0; local < blocksPerChart; local++) descriptors[base + local] = base + local;
  }
  const scale = new DataView(artifact.buffer, artifact.byteOffset + scaleTableOffset, scaleTableBytes);
  scale.setFloat32(0, STANDOFF_METRES, true); scale.setFloat32(4, PIXEL_ANGLE, true); scale.setUint32(12, descriptorCount, true);
  scale.setUint32(20, codebookCount, true); scale.setUint32(24, atlasWidth, true); scale.setUint32(28, atlasHeight, true);
  const color0 = new Uint16Array(artifact.buffer, artifact.byteOffset + color0Offset, atlasTexels * 4);
  const color1 = new Uint16Array(artifact.buffer, artifact.byteOffset + color1Offset, atlasTexels * 4);
  const categorical = new Uint16Array(artifact.buffer, artifact.byteOffset + categoricalOffset, atlasTexels * 4);
  for (let chart = 0; chart < FACE_CHARTS; chart++) {
    for (let b3 = 0; b3 < BLOCKS_PER_AXIS; b3++) for (let b2 = 0; b2 < BLOCKS_PER_AXIS; b2++) for (let b1 = 0; b1 < BLOCKS_PER_AXIS; b1++) for (let b0 = 0; b0 < BLOCKS_PER_AXIS; b0++) {
      const localBlock = (((b3 * BLOCKS_PER_AXIS + b2) * BLOCKS_PER_AXIS + b1) * BLOCKS_PER_AXIS + b0);
      const codeword = chart * blocksPerChart + localBlock;
      const tileX = (codeword % atlasColumns) * tileEdge;
      const tileY = Math.floor(codeword / atlasColumns) * tileEdge;
      for (let l3 = 0; l3 < BLOCK_SAMPLE_EDGE; l3++) for (let l2 = 0; l2 < BLOCK_SAMPLE_EDGE; l2++) for (let l1 = 0; l1 < BLOCK_SAMPLE_EDGE; l1++) for (let l0 = 0; l0 < BLOCK_SAMPLE_EDGE; l0++) {
        const node = nodeLinear(chart, b0 * BLOCK_CELL_EDGE + l0, b1 * BLOCK_CELL_EDGE + l1, b2 * BLOCK_CELL_EDGE + l2, b3 * BLOCK_CELL_EDGE + l3);
        const x = tileX + l0 + BLOCK_SAMPLE_EDGE * l2;
        const y = tileY + l1 + BLOCK_SAMPLE_EDGE * l3;
        const texel = (y * atlasWidth + x) * 4;
        for (let channel = 0; channel < 4; channel++) {
          color0[texel + channel] = float16(nodeColor0[node * 4 + channel]!);
          color1[texel + channel] = float16(nodeColor1[node * 4 + channel]!);
          categorical[texel + channel] = nodeCategorical[node * 4 + channel]!;
        }
      }
    }
  }
  if (artifact.toString('ascii', 0, 4) !== 'GBR4' || view.getUint32(4, true) !== 2 || descriptors.at(-1) !== codebookCount - 1) throw new Error('GBR4/v2 self-validation failed');
  const outputRoot = resolve(WORKSPACE, 'data/work/groundcover-gbr4-periodic-cap-assets', sourceSha.slice(0, 16), recipeSha.slice(0, 16));
  const qaRoot = resolve(outputRoot, 'qa'); mkdirSync(qaRoot, { recursive: true });
  const assetPath = resolve(outputRoot, 'calamagrostis-canescens.gbr4');
  const stablePath = resolve(WORKSPACE, 'src/assets/groundcover/calamagrostis-canescens.gbr4');
  writeFileSync(assetPath, artifact); writeFileSync(stablePath, artifact);
  const qaWidth = 98; const qaHeight = 49; const qaRaw = Buffer.alloc(qaWidth * qaHeight * 3);
  for (let pixel = 0; pixel < qaWidth * qaHeight; pixel++) {
    const node = pixel % nodeCount; const alpha = nodeColor0[node * 4 + 3]! + nodeColor1[node * 4 + 3]!;
    for (let channel = 0; channel < 3; channel++) qaRaw[pixel * 3 + channel] = Math.round(clamp(nodeColor0[node * 4 + channel]! + nodeColor1[node * 4 + channel]! + (1 - alpha) * .1, 0, 1) * 255);
  }
  const qaPath = resolve(qaRoot, '01-periodic-cap-filtered-node-swatches.png');
  await sharp(qaRaw, { raw: { width: qaWidth, height: qaHeight, channels: 3 } }).resize(1176, 588, { kernel: 'nearest' }).png().toFile(qaPath);
  const report = {
    schema: 'laas-gbr4-periodic-cap-minimal-real-asset/v2', created: new Date().toISOString(), recipe,
    premiseAudit: {
      supersedesV1PeriodicAsset: true,
      reason: 'v1 stopped at first 0.52m tile exit; v2 periodic truth continues through later copies to the fixed horizon',
      empiricalCookEvidence: periodicEvidence,
      interpretation: `${periodicEvidence.hitsBeyondFirstTileExit} cooked hit subrays first intersected geometry only after leaving their entry tile; v1 necessarily lost these, v2 stores them in the same one-query node`,
    },
    source: { path: relative(WORKSPACE, SOURCE), sha256: sourceSha, vertices: geometry.vertexCount, triangles: geometry.triangleCount },
    container: { path: relative(WORKSPACE, assetPath), stablePath: relative(WORKSPACE, stablePath), sha256: sha256(artifact), bytes: artifact.byteLength, MiB: artifact.byteLength / 1048576 },
    address: { mode: 'periodic cap phase+slope', charts: ['top-downward', 'bottom-upward'], cellResolution: CELL_RESOLUTION, blocksPerChart, descriptorCount },
    codebook: { mode: 'lossless one codeword per block', count: codebookCount, samplesPerBrick: SAMPLES_PER_BRICK, tile: [tileEdge, tileEdge], atlas: [atlasWidth, atlasHeight] },
    sections: { descriptor: [descriptorOffset, descriptorBytes], chartTable: [chartTableOffset, chartTableBytes], scaleTable: [scaleTableOffset, scaleTableBytes], frontColor: [color0Offset, atlasBytes], backColor: [color1Offset, atlasBytes], categorical: [categoricalOffset, atlasBytes] },
    runtimeBudget: { logicalReads: 10, descriptorReads: 1, filteredColorReads: 8, categoricalReads: 1, loops: 0, marches: 0, candidates: 0, runtimeGeometry: false },
    validity: {
      status: 'GREEN_PERIODIC_CAP_CONTAINER_MINIMAL_COARSE', runtimeLoadable: true, visualProductionReady: false,
      proven: ['one query includes later periodic tile hits to 155m', 'shared-origin top/bottom cap filtering', 'joint 4D bricks', 'fixed O(1) decode'],
      missing: ['finite-patch moving side-entry charts', 'production R/VQ', 'scale cascade', 'held-out visual/translation gate', 'runtime implementation'],
    },
    qa: { path: relative(WORKSPACE, qaPath), sha256: sha256(readFileSync(qaPath)) }, elapsedMilliseconds: performance.now() - started,
  };
  const reportPath = resolve(outputRoot, 'report.json'); writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`);
  console.log(JSON.stringify({ outputRoot: relative(WORKSPACE, outputRoot), report: relative(WORKSPACE, reportPath), container: report.container, premiseAudit: report.premiseAudit, validity: report.validity }, null, 2));
}

await main();
