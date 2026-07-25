/**
 * Small, honest GBR4/v4 boundary-transfer integration cook.
 *
 * This is deliberately a loadable reference asset, not the production fit.
 * It traces the accepted Calamagrostis triangle soup through six complete box
 * boundary charts, a bounded Lambert hemisphere disk, and four independently
 * integrated physical footprint levels.  Every valid address is published as
 * FILTERED_MIXED because this small cook does not own a whole-cell firstness
 * prover.  It therefore cannot fabricate categorical depth/normal/owner.
 */

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import sharp from 'sharp';
import {
  TriangleBvh,
  decodeOwnedProfileGeometry,
  type CensusVec3,
  type DecodedOwnedProfileGeometry,
} from '../../ExteriorClosureCensus';
import { periodicNearestSuccessor } from '../../OriginAwareRayTruth';
import { certifyBoundaryRayBundle, type BoundaryBundleCertificate } from '../boundary-transfer/BoundaryBundleCertifier';
import {
  ESTONIAN_GRAMINOID_PROFILE_IDS,
  makeEstonianGraminoidFixture,
  type GraminoidPrimitiveRecipe,
} from '../../EstonianGraminoids';
import type { Vec3 } from '../../ProfileFormat';
import {
  addScaled,
  align,
  attributes,
  canonicalJson,
  clamp,
  cross,
  float16,
  normalized,
  sha256,
} from './cook-gbr4-brick-assets';

const WORKSPACE = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
const SOURCE = resolve(WORKSPACE, 'src/assets/groundcover/calamagrostis-canescens.gcrp');
const EXPECTED_SOURCE_SHA = '2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c';
const HEADER_BYTES = 512;
const FACE_COUNT = 6;
const FACE_ENTRY_BYTES = 16;
const LEVEL_ENTRY_BYTES = 64;
const BOUNDARY_CELLS = 2;
const DIRECTION_CELLS = 4;
const QUADRATURE_SIDE = 4;
const PIXEL_ANGLE = 60 * Math.PI / 180 / 1920;
const STANDOFF_LEVELS = [0.025, 0.25, 4, 32] as const;
const HORIZON_METRES = 155;
const CARRIER_GUARD = 0.025;
const MIXED_TEXELS = 4;
const REGULAR_TEXELS = 5;
const ADDRESS_TEXTURE_WIDTH = 512;
const MIXED_ATLAS_WIDTH = 256;
const REGULAR_ATLAS_WIDTH = REGULAR_TEXELS;
const FINE_BOUNDARY_CELLS = 4096;
const FINE_DIRECTION_CELLS = 65535;
const CERTIFICATION_GRID = 64;
const MAX_REGULAR_CORRECTIONS = 64;

interface FaceFrame {
  readonly id: number;
  readonly axis: 0 | 1 | 2;
  readonly side: -1 | 1;
  readonly origin: CensusVec3;
  readonly edgeU: CensusVec3;
  readonly edgeV: CensusVec3;
  readonly tangentU: CensusVec3;
  readonly tangentV: CensusVec3;
  readonly inward: CensusVec3;
}

interface MixedRecord {
  readonly values: readonly number[];
  readonly hits: number;
  readonly maximumHitDistance: number;
}

interface Evidence {
  tracedSubrays: number;
  hitSubrays: number;
  maximumHitDistanceMetres: number;
  hitsByLevel: number[];
}

interface RegularCorrection {
  readonly level: number;
  readonly face: number;
  readonly boundaryU: number;
  readonly boundaryV: number;
  readonly directionU: number;
  readonly directionV: number;
  readonly triangleId: number;
  readonly copyX: number;
  readonly copyZ: number;
  readonly certificate: BoundaryBundleCertificate;
  readonly payload: readonly number[];
}

function subtract(a: CensusVec3, b: CensusVec3): CensusVec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function add(a: CensusVec3, b: CensusVec3): CensusVec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function scale(a: CensusVec3, amount: number): CensusVec3 {
  return [a[0] * amount, a[1] * amount, a[2] * amount];
}

function recipeRoot(recipe: GraminoidPrimitiveRecipe): Vec3 {
  switch (recipe.kind) {
    case 'hub': return recipe.center;
    case 'blade': return recipe.root;
    case 'tube':
    case 'rhizome': return recipe.centerline[0]!;
    case 'lanceolate-surface': return recipe.base;
    case 'hair-filament': return recipe.root;
    case 'cotton-bristle': return recipe.root;
  }
}

function recipeForTriangle(
  recipes: readonly GraminoidPrimitiveRecipe[],
  triangleId: number,
): GraminoidPrimitiveRecipe {
  let low = 0;
  let high = recipes.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    const recipe = recipes[middle]!;
    if (triangleId < recipe.sourceTriangleStart) high = middle;
    else if (triangleId >= recipe.sourceTriangleStart + recipe.sourceTriangleCount) low = middle + 1;
    else return recipe;
  }
  throw new Error(`triangle ${triangleId} has no procedural primitive owner`);
}

function triangleMeanColor(sourceBytes: Uint8Array, triangleId: number): readonly [number, number, number] {
  const view = new DataView(sourceBytes.buffer, sourceBytes.byteOffset, sourceBytes.byteLength);
  const vertexOffset = view.getUint32(84, true);
  const triangleOffset = view.getUint32(88, true) + triangleId * 16;
  const rgb = [0, 0, 0];
  for (let corner = 0; corner < 3; corner++) {
    const vertex = view.getUint32(triangleOffset + corner * 4, true);
    for (let channel = 0; channel < 3; channel++) {
      rgb[channel]! += view.getUint16(vertexOffset + vertex * 16 + 6 + channel * 2, true) / (65535 * 3);
    }
  }
  return rgb as [number, number, number];
}

function correctionKey(correction: Pick<RegularCorrection, 'level' | 'face' | 'boundaryU' | 'boundaryV' | 'directionU' | 'directionV'>): bigint {
  let key = BigInt(correction.level);
  key = (key << 3n) | BigInt(correction.face);
  key = (key << 12n) | BigInt(correction.boundaryU);
  key = (key << 12n) | BigInt(correction.boundaryV);
  key = (key << 16n) | BigInt(correction.directionU);
  key = (key << 16n) | BigInt(correction.directionV);
  return key;
}

function correctionHash(low: number, high: number, seed: number): number {
  let value = Math.imul((low ^ seed) >>> 0, 0x9e37_79b1);
  value = (value + Math.imul(high >>> 0, 0x85eb_ca6b)) >>> 0;
  value ^= value >>> 16;
  return value >>> 0;
}

function dot(a: CensusVec3, b: CensusVec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function length(a: CensusVec3): number { return Math.hypot(a[0], a[1], a[2]); }

function makeFaces(minimum: CensusVec3, maximum: CensusVec3): readonly FaceFrame[] {
  const span = subtract(maximum, minimum);
  const raw = [
    { axis: 0, side: -1, origin: [minimum[0], minimum[1], minimum[2]], edgeU: [0, span[1], 0], edgeV: [0, 0, span[2]] },
    { axis: 0, side: 1, origin: [maximum[0], minimum[1], maximum[2]], edgeU: [0, span[1], 0], edgeV: [0, 0, -span[2]] },
    { axis: 1, side: -1, origin: [minimum[0], minimum[1], maximum[2]], edgeU: [span[0], 0, 0], edgeV: [0, 0, -span[2]] },
    { axis: 1, side: 1, origin: [minimum[0], maximum[1], minimum[2]], edgeU: [span[0], 0, 0], edgeV: [0, 0, span[2]] },
    { axis: 2, side: -1, origin: [minimum[0], minimum[1], minimum[2]], edgeU: [span[0], 0, 0], edgeV: [0, span[1], 0] },
    { axis: 2, side: 1, origin: [maximum[0], minimum[1], maximum[2]], edgeU: [-span[0], 0, 0], edgeV: [0, span[1], 0] },
  ] as const;
  return raw.map((face, id) => {
    const edgeU = face.edgeU as CensusVec3;
    const edgeV = face.edgeV as CensusVec3;
    const tangentU = normalized(edgeU);
    const tangentV = normalized(edgeV);
    return Object.freeze({
      id,
      axis: face.axis as 0 | 1 | 2,
      side: face.side as -1 | 1,
      origin: face.origin as CensusVec3,
      edgeU,
      edgeV,
      tangentU,
      tangentV,
      inward: normalized(cross(tangentU, tangentV)),
    });
  });
}

function facePoint(face: FaceFrame, u: number, v: number): CensusVec3 {
  return add(add(face.origin, scale(face.edgeU, u)), scale(face.edgeV, v));
}

/** Inverse Lambert equal-area hemisphere disk for one face-local frame. */
function faceDirection(face: FaceFrame, diskU: number, diskV: number): CensusVec3 {
  const radius = Math.hypot(diskU, diskV);
  const u = radius > 1 ? diskU / radius : diskU;
  const v = radius > 1 ? diskV / radius : diskV;
  const r2 = Math.min(1, u * u + v * v);
  const tangentScale = Math.sqrt(Math.max(0, 2 - r2));
  return normalized(add(
    add(scale(face.tangentU, u * tangentScale), scale(face.tangentV, v * tangentScale)),
    scale(face.inward, 1 - r2),
  ));
}

function rayBox(
  origin: CensusVec3,
  direction: CensusVec3,
  minimum: CensusVec3,
  maximum: CensusVec3,
): { entry: CensusVec3; entryT: number } | null {
  let near = Number.NEGATIVE_INFINITY;
  let far = Number.POSITIVE_INFINITY;
  let nearFace = Number.POSITIVE_INFINITY;
  for (let axis = 0; axis < 3; axis++) {
    const d = direction[axis]!;
    if (Math.abs(d) <= 1e-15) {
      if (origin[axis]! < minimum[axis]! || origin[axis]! > maximum[axis]!) return null;
      continue;
    }
    let a = (minimum[axis]! - origin[axis]!) / d;
    let b = (maximum[axis]! - origin[axis]!) / d;
    let faceA = axis * 2;
    let faceB = axis * 2 + 1;
    if (a > b) { [a, b] = [b, a]; [faceA, faceB] = [faceB, faceA]; }
    if (a > near + 1e-12 || (Math.abs(a - near) <= 1e-12 && faceA < nearFace)) {
      near = a;
      nearFace = faceA;
    }
    far = Math.min(far, b);
    if (near > far + 1e-12) return null;
  }
  if (!(far >= Math.max(0, near))) return null;
  const entryT = Math.max(0, near);
  return { entryT, entry: addScaled(origin, direction, entryT) };
}

function integrateMixed(
  sourceBytes: Uint8Array,
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  face: FaceFrame,
  boundaryU: number,
  boundaryV: number,
  diskU: number,
  diskV: number,
  standoffMetres: number,
  minimum: CensusVec3,
  maximum: CensusVec3,
  evidence: Evidence,
  level: number,
): MixedRecord {
  const q = facePoint(face, boundaryU, boundaryV);
  const central = faceDirection(face, diskU, diskV);
  const camera = addScaled(q, central, -standoffMetres);
  const right = Math.abs(central[1]) > 0.999999
    ? ([1, 0, 0] as const)
    : normalized(cross(central, [0, 1, 0]));
  const up = normalized(cross(right, central));
  const tangent = Math.tan(PIXEL_ANGLE * 0.5);
  const color = [0, 0, 0];
  const colorSecond = [0, 0, 0];
  const normal = [0, 0, 0];
  const depths: number[] = [];
  let maximumHitDistance = 0;
  const total = QUADRATURE_SIDE ** 2;
  for (let sy = 0; sy < QUADRATURE_SIDE; sy++) {
    for (let sx = 0; sx < QUADRATURE_SIDE; sx++) {
      const px = (2 * (sx + 0.5) / QUADRATURE_SIDE - 1) * tangent;
      const py = (2 * (sy + 0.5) / QUADRATURE_SIDE - 1) * tangent;
      const direction = normalized([
        central[0] + right[0] * px + up[0] * py,
        central[1] + right[1] * px + up[1] * py,
        central[2] + right[2] * px + up[2] * py,
      ]);
      evidence.tracedSubrays++;
      const box = rayBox(camera, direction, minimum, maximum);
      if (!box) continue;
      const hit = periodicNearestSuccessor(geometry, bvh, box.entry, direction, HORIZON_METRES, 1e-7);
      if (!hit) continue;
      evidence.hitSubrays++;
      evidence.hitsByLevel[level] = (evidence.hitsByLevel[level] ?? 0) + 1;
      evidence.maximumHitDistanceMetres = Math.max(evidence.maximumHitDistanceMetres, hit.t);
      maximumHitDistance = Math.max(maximumHitDistance, hit.t);
      const decoded = attributes(sourceBytes, geometry, box.entry, direction, hit);
      const depth = clamp(hit.t / HORIZON_METRES, 0, 1);
      depths.push(depth);
      for (let channel = 0; channel < 3; channel++) {
        color[channel]! += decoded.color[channel]! / total;
        colorSecond[channel]! += decoded.color[channel]! ** 2 / total;
        normal[channel]! += decoded.normal[channel]! / total;
      }
    }
  }
  const coverage = depths.length / total;
  const meanDepth = depths.length ? depths.reduce((sum, value) => sum + value, 0) / depths.length : 0;
  const varianceDepth = depths.length
    ? depths.reduce((sum, value) => sum + (value - meanDepth) ** 2, 0) / depths.length
    : 0;
  // No continuum interval certificate exists in this integration cook.  The
  // complete [0, horizon] interval is conservative and therefore honest.
  const depthMinimum = 0;
  const depthMaximum = 1;
  return {
    hits: depths.length,
    maximumHitDistance,
    values: [
      color[0]!, color[1]!, color[2]!, coverage,
      depthMinimum, depthMaximum, meanDepth, varianceDepth,
      normal[0]!, normal[1]!, normal[2]!, coverage,
      colorSecond[0]!, colorSecond[1]!, colorSecond[2]!, coverage,
    ],
  };
}

async function main(): Promise<void> {
  const started = performance.now();
  const sourceBytes = readFileSync(SOURCE);
  const sourceSha = sha256(sourceBytes);
  if (sourceSha !== EXPECTED_SOURCE_SHA) throw new Error(`accepted Calamagrostis source changed: ${sourceSha}`);
  const geometry = decodeOwnedProfileGeometry(sourceBytes);
  const minimum: CensusVec3 = [
    geometry.tileOriginX,
    geometry.bounds.min[1] - CARRIER_GUARD,
    geometry.tileOriginZ,
  ];
  const maximum: CensusVec3 = [
    geometry.tileOriginX + geometry.tileSizeX,
    geometry.bounds.max[1] + CARRIER_GUARD,
    geometry.tileOriginZ + geometry.tileSizeZ,
  ];
  const faces = makeFaces(minimum, maximum);
  for (const face of faces) {
    if (dot(cross(face.tangentU, face.tangentV), face.inward) < 0.999999) {
      throw new Error(`face ${face.id} frame is not inward right-handed`);
    }
  }
  console.error(`[gbr4-v4] building BVH for ${geometry.triangleCount.toLocaleString()} triangles`);
  const bvh = TriangleBvh.build(geometry);
  const recordsPerLevel = FACE_COUNT * BOUNDARY_CELLS ** 2 * DIRECTION_CELLS ** 2;
  const addressRecordCount = STANDOFF_LEVELS.length * recordsPerLevel;
  const evidence: Evidence = {
    tracedSubrays: 0,
    hitSubrays: 0,
    maximumHitDistanceMetres: 0,
    hitsByLevel: STANDOFF_LEVELS.map(() => 0),
  };
  const mixedRecords: MixedRecord[] = [];
  const addressWords = new Uint32Array(
    ADDRESS_TEXTURE_WIDTH * Math.ceil(addressRecordCount / ADDRESS_TEXTURE_WIDTH),
  );
  let address = 0;
  for (let level = 0; level < STANDOFF_LEVELS.length; level++) {
    const standoff = STANDOFF_LEVELS[level]!;
    for (const face of faces) {
      for (let directionV = 0; directionV < DIRECTION_CELLS; directionV++) {
        for (let directionU = 0; directionU < DIRECTION_CELLS; directionU++) {
          const diskU = 2 * (directionU + 0.5) / DIRECTION_CELLS - 1;
          const diskV = 2 * (directionV + 0.5) / DIRECTION_CELLS - 1;
          for (let boundaryV = 0; boundaryV < BOUNDARY_CELLS; boundaryV++) {
            for (let boundaryU = 0; boundaryU < BOUNDARY_CELLS; boundaryU++) {
              const record = integrateMixed(
                sourceBytes,
                geometry,
                bvh,
                face,
                (boundaryU + 0.5) / BOUNDARY_CELLS,
                (boundaryV + 0.5) / BOUNDARY_CELLS,
                diskU,
                diskV,
                standoff,
                minimum,
                maximum,
                evidence,
                level,
              );
              const payload = mixedRecords.length;
              mixedRecords.push(record);
              addressWords[address++] = (payload << 2) | 3;
            }
          }
        }
      }
    }
    console.error(`[gbr4-v4] footprint ${level + 1}/${STANDOFF_LEVELS.length}: ${standoff} m`);
  }
  if (address !== addressRecordCount || mixedRecords.length !== addressRecordCount) {
    throw new Error('GBR4/v4 cook address cardinality mismatch');
  }

  // Sparse maximum-resolution correction proof.  These cells are not corner
  // samples: BoundaryBundleCertifier proves the full q x d product cell by a
  // support/incidence bound and a complete periodic-copy swept-capsule test.
  const regularCorrections: RegularCorrection[] = [];
  let certificationCandidates = 0;
  let certificationCenterHits = 0;
  const certificationFace = faces[3]!; // upper Y face, inward/downward
  const fineDirectionCenter = (FINE_DIRECTION_CELLS - 1) / 2;
  const boundaryHalfU = length(certificationFace.edgeU) / (2 * FINE_BOUNDARY_CELLS);
  const boundaryHalfV = length(certificationFace.edgeV) / (2 * FINE_BOUNDARY_CELLS);
  const pinholeBoundaryRadius = STANDOFF_LEVELS[0]! * Math.tan(PIXEL_ANGLE * 0.5) * Math.SQRT2;
  const boundaryRadius = Math.hypot(boundaryHalfU, boundaryHalfV) + pinholeBoundaryRadius;
  // The closed-disk inverse has vector Lipschitz constant below five.  The
  // extra angular term encloses the complete square pixel kernel.
  const directionRadius = 5 * Math.SQRT2 / FINE_DIRECTION_CELLS
    + 2 * Math.sin(PIXEL_ANGLE * Math.SQRT2 * 0.25);
  for (let gy = 0; gy < CERTIFICATION_GRID && regularCorrections.length < MAX_REGULAR_CORRECTIONS; gy++) {
    for (let gx = 0; gx < CERTIFICATION_GRID && regularCorrections.length < MAX_REGULAR_CORRECTIONS; gx++) {
      const boundaryU = Math.min(
        FINE_BOUNDARY_CELLS - 1,
        Math.floor((gx + 0.5) / CERTIFICATION_GRID * FINE_BOUNDARY_CELLS),
      );
      const boundaryV = Math.min(
        FINE_BOUNDARY_CELLS - 1,
        Math.floor((gy + 0.5) / CERTIFICATION_GRID * FINE_BOUNDARY_CELLS),
      );
      const q0 = facePoint(
        certificationFace,
        (boundaryU + 0.5) / FINE_BOUNDARY_CELLS,
        (boundaryV + 0.5) / FINE_BOUNDARY_CELLS,
      );
      const d0 = certificationFace.inward;
      certificationCandidates++;
      const winner = periodicNearestSuccessor(geometry, bvh, q0, d0, HORIZON_METRES, 1e-7);
      if (!winner) continue;
      certificationCenterHits++;
      const certificate = certifyBoundaryRayBundle(
        geometry,
        bvh,
        q0,
        d0,
        winner,
        boundaryRadius,
        directionRadius,
      );
      if (!certificate) continue;
      regularCorrections.push({
        level: 0,
        face: certificationFace.id,
        boundaryU,
        boundaryV,
        directionU: fineDirectionCenter,
        directionV: fineDirectionCenter,
        triangleId: winner.triangleId,
        copyX: winner.copyX,
        copyZ: winner.copyZ,
        certificate,
        payload: [],
      });
    }
  }
  console.error(
    `[gbr4-v4] certified ${regularCorrections.length}/${certificationCandidates} tested fine cells `
      + `(${certificationCenterHits} center hits)`,
  );

  let fixture: ReturnType<typeof makeEstonianGraminoidFixture> | null = null;
  if (regularCorrections.length > 0) {
    fixture = makeEstonianGraminoidFixture(ESTONIAN_GRAMINOID_PROFILE_IDS.CALAMAGROSTIS_CANESCENS);
    if (fixture.mesh.indices.length / 3 !== geometry.triangleCount) {
      throw new Error('procedural primitive ownership no longer matches the accepted GCRP triangle order');
    }
  }
  const completeRegularCorrections = regularCorrections.map((correction): RegularCorrection => {
    if (!fixture) throw new Error('regular correction lacks procedural root ownership');
    const recipe = recipeForTriangle(fixture.primitiveRecipes, correction.triangleId);
    const root = recipeRoot(recipe);
    const color = triangleMeanColor(sourceBytes, correction.triangleId);
    const payload = [
      ...correction.certificate.plane,
      root.x, root.y, root.z, correction.copyX,
      correction.copyZ, correction.triangleId, recipe.primitiveId, geometry.profileId,
      correction.certificate.supportMarginMetres,
      correction.certificate.incidenceMargin,
      correction.certificate.orderMarginMetres,
      0, // terrain/motion firstness margin: not certified by this flat-source cook
      color[0], color[1], color[2], recipe.disposition === 'plume' ? 1 : 0,
    ];
    return Object.freeze({ ...correction, payload: Object.freeze(payload) });
  });

  const faceTableBytes = FACE_COUNT * FACE_ENTRY_BYTES;
  const levelTableBytes = STANDOFF_LEVELS.length * LEVEL_ENTRY_BYTES;
  const addressBytes = addressWords.byteLength;
  const mixedAtlasHeight = Math.ceil(mixedRecords.length * MIXED_TEXELS / MIXED_ATLAS_WIDTH);
  const mixedAtlasTexels = MIXED_ATLAS_WIDTH * mixedAtlasHeight;
  const mixedBytes = mixedAtlasTexels * 8;
  const regularAtlasHeight = Math.max(1, completeRegularCorrections.length);
  const regularBytes = REGULAR_ATLAS_WIDTH * regularAtlasHeight * 16;
  let correctionTexels = 1;
  while (correctionTexels < Math.max(2, completeRegularCorrections.length * 4)) correctionTexels *= 2;
  const correctionTextureWidth = Math.min(256, correctionTexels);
  const correctionTextureHeight = correctionTexels / correctionTextureWidth;
  let correctionSeed = 0;
  const correctionSlots = new Int32Array(correctionTexels);
  correctionSlots.fill(-1);
  seedSearch: for (let seed = 1; seed < 1_000_000; seed++) {
    correctionSlots.fill(-1);
    for (let correction = 0; correction < completeRegularCorrections.length; correction++) {
      const key = correctionKey(completeRegularCorrections[correction]!);
      const low = Number(key & 0xffff_ffffn) >>> 0;
      const high = Number((key >> 32n) & 0xffff_ffffn) >>> 0;
      const slot = correctionHash(low, high, seed) & (correctionTexels - 1);
      if (correctionSlots[slot] >= 0) continue seedSearch;
      correctionSlots[slot] = correction;
    }
    correctionSeed = seed;
    break;
  }
  if (!(correctionSeed > 0)) throw new Error('failed to fit the fixed one-read correction hash');
  const correctionBytes = correctionTexels * 16;
  const faceTableOffset = HEADER_BYTES;
  const levelTableOffset = align(faceTableOffset + faceTableBytes);
  const addressOffset = align(levelTableOffset + levelTableBytes);
  const mixedOffset = align(addressOffset + addressBytes);
  const regularOffset = align(mixedOffset + mixedBytes);
  const correctionOffset = align(regularOffset + regularBytes);
  const totalBytes = align(correctionOffset + correctionBytes);
  const recipe = {
    schema: 'laas-gbr4-boundary-lambert-coupled-reference/v4',
    implementationSha256: sha256(readFileSync(fileURLToPath(import.meta.url))),
    sourceSha256: sourceSha,
    boundaryCharts: 'six box faces; lowest-face-id ties',
    directionChart: 'Lambert equal-area closed hemisphere disk',
    boundaryCells: BOUNDARY_CELLS,
    directionCells: DIRECTION_CELLS,
    quadratureSide: QUADRATURE_SIDE,
    pixelAngleRadians: PIXEL_ANGLE,
    standoffLevelsMetres: STANDOFF_LEVELS,
    horizonMetres: HORIZON_METRES,
    recordPolicy: 'all valid addresses FILTERED_MIXED; no sampled owner promoted to regular',
    correctionPolicy: {
      fineBoundaryCells: FINE_BOUNDARY_CELLS,
      fineDirectionCells: FINE_DIRECTION_CELLS,
      certifier: 'full product-cell analytic winner bounds plus complete periodic swept-capsule competitor exclusion',
      maximumRegularCorrections: MAX_REGULAR_CORRECTIONS,
    },
  };
  const recipeSha = sha256(canonicalJson(recipe));
  const artifact = Buffer.alloc(totalBytes);
  const view = new DataView(artifact.buffer, artifact.byteOffset, artifact.byteLength);
  artifact.write('GBR4', 0, 'ascii');
  view.setUint32(4, 4, true);
  view.setUint32(8, HEADER_BYTES, true);
  view.setUint32(12, geometry.profileId, true);
  view.setUint32(16, FACE_COUNT, true);
  view.setUint32(20, 4, true);
  view.setUint32(24, 4, true);
  view.setUint32(28, STANDOFF_LEVELS.length, true);
  view.setUint32(32, BOUNDARY_CELLS, true);
  view.setUint32(36, DIRECTION_CELLS, true);
  view.setUint32(40, 1, true);
  view.setUint32(44, MIXED_TEXELS, true);
  view.setUint32(48, REGULAR_TEXELS, true);
  view.setUint32(52, faceTableOffset, true);
  view.setUint32(56, faceTableBytes, true);
  view.setUint32(60, levelTableOffset, true);
  view.setUint32(64, levelTableBytes, true);
  view.setUint32(68, addressOffset, true);
  view.setUint32(72, addressBytes, true);
  view.setUint32(76, mixedOffset, true);
  view.setUint32(80, mixedBytes, true);
  view.setUint32(84, regularOffset, true);
  view.setUint32(88, regularBytes, true);
  view.setUint32(92, mixedRecords.length, true);
  view.setUint32(96, completeRegularCorrections.length, true);
  view.setUint32(100, addressRecordCount, true);
  view.setUint32(104, ADDRESS_TEXTURE_WIDTH, true);
  view.setUint32(108, addressWords.length / ADDRESS_TEXTURE_WIDTH, true);
  view.setUint32(112, MIXED_ATLAS_WIDTH, true);
  view.setUint32(116, mixedAtlasHeight, true);
  view.setUint32(120, REGULAR_ATLAS_WIDTH, true);
  view.setUint32(124, regularAtlasHeight, true);
  view.setFloat32(128, geometry.tileOriginX, true);
  view.setFloat32(132, geometry.tileOriginZ, true);
  view.setFloat32(136, geometry.tileSizeX, true);
  view.setFloat32(140, geometry.tileSizeZ, true);
  for (let axis = 0; axis < 3; axis++) view.setFloat32(144 + axis * 4, minimum[axis]!, true);
  for (let axis = 0; axis < 3; axis++) view.setFloat32(156 + axis * 4, maximum[axis]!, true);
  artifact.set(Buffer.from(sourceSha, 'hex'), 168);
  artifact.set(Buffer.from(recipeSha, 'hex'), 200);
  view.setFloat32(232, HORIZON_METRES, true);
  view.setUint32(236, QUADRATURE_SIDE, true);
  view.setUint32(264, 0b1_1111, true);
  view.setUint32(268, 4, true);
  view.setUint32(272, 1, true);
  view.setUint32(276, 1, true);
  view.setUint32(280, correctionOffset, true);
  view.setUint32(284, correctionBytes, true);
  view.setUint32(288, completeRegularCorrections.length, true);
  view.setUint32(292, correctionTextureWidth, true);
  view.setUint32(296, correctionTextureHeight, true);
  view.setUint32(300, FINE_BOUNDARY_CELLS, true);
  view.setUint32(304, FINE_DIRECTION_CELLS, true);
  view.setUint32(308, correctionSeed, true);
  view.setUint32(312, 0, true); // REFERENCE_RED; visible runtime binding must fail closed

  const faceTable = new DataView(artifact.buffer, artifact.byteOffset + faceTableOffset, faceTableBytes);
  for (const face of faces) {
    const offset = face.id * FACE_ENTRY_BYTES;
    faceTable.setUint8(offset, face.id);
    faceTable.setUint8(offset + 1, face.axis);
    faceTable.setInt8(offset + 2, face.side);
  }
  const levelTable = new DataView(artifact.buffer, artifact.byteOffset + levelTableOffset, levelTableBytes);
  for (let level = 0; level < STANDOFF_LEVELS.length; level++) {
    const offset = level * LEVEL_ENTRY_BYTES;
    const standoff = STANDOFF_LEVELS[level]!;
    levelTable.setFloat32(offset, standoff, true);
    levelTable.setFloat32(offset + 4, 2 * standoff * Math.tan(PIXEL_ANGLE * 0.5), true);
    levelTable.setFloat32(offset + 8, PIXEL_ANGLE, true);
    levelTable.setUint32(offset + 12, level * recordsPerLevel, true);
    levelTable.setUint32(offset + 16, recordsPerLevel, true);
    levelTable.setUint32(offset + 20, level * recordsPerLevel, true);
    levelTable.setUint32(offset + 24, recordsPerLevel, true);
  }
  new Uint32Array(artifact.buffer, artifact.byteOffset + addressOffset, addressWords.length).set(addressWords);
  const mixedWords = new Uint16Array(artifact.buffer, artifact.byteOffset + mixedOffset, mixedAtlasTexels * 4);
  for (let record = 0; record < mixedRecords.length; record++) {
    const values = mixedRecords[record]!.values;
    for (let value = 0; value < values.length; value++) {
      mixedWords[record * MIXED_TEXELS * 4 + value] = float16(values[value]!);
    }
  }
  // The v4 schema reserves a real regular atlas with a five-RGBA32F coupled
  // plane/root/copy/material/certificate payload.  Only continuum-certified
  // sparse correction cells may name these records.
  const regularWords = new Float32Array(
    artifact.buffer,
    artifact.byteOffset + regularOffset,
    regularBytes / 4,
  );
  for (let record = 0; record < completeRegularCorrections.length; record++) {
    regularWords.set(completeRegularCorrections[record]!.payload, record * REGULAR_TEXELS * 4);
  }
  const correctionWords = new Uint32Array(
    artifact.buffer,
    artifact.byteOffset + correctionOffset,
    correctionBytes / 4,
  );
  for (let slot = 0; slot < correctionSlots.length; slot++) {
    const correctionIndex = correctionSlots[slot]!;
    if (correctionIndex < 0) continue;
    const key = correctionKey(completeRegularCorrections[correctionIndex]!);
    correctionWords[slot * 4] = Number(key & 0xffff_ffffn) >>> 0;
    correctionWords[slot * 4 + 1] = Number((key >> 32n) & 0xffff_ffffn) >>> 0;
    correctionWords[slot * 4 + 2] = correctionIndex + 1;
    correctionWords[slot * 4 + 3] = correctionSeed;
  }

  const outputRoot = resolve(
    WORKSPACE,
    'data/work/groundcover-gbr4-v4-coupled-reference',
    sourceSha.slice(0, 16),
    recipeSha.slice(0, 16),
  );
  const qaRoot = resolve(outputRoot, 'qa');
  mkdirSync(qaRoot, { recursive: true });
  const assetPath = resolve(outputRoot, 'calamagrostis-canescens.reference-v4.gbr4');
  const stablePath = resolve(WORKSPACE, 'src/assets/groundcover/calamagrostis-canescens.reference-v4.gbr4');
  writeFileSync(assetPath, artifact);
  writeFileSync(stablePath, artifact);

  const qaColumns = 32;
  const qaRowsPerLevel = Math.ceil(recordsPerLevel / qaColumns);
  const qaRows = qaRowsPerLevel * STANDOFF_LEVELS.length;
  const qa = Buffer.alloc(qaColumns * qaRows * 4);
  for (let record = 0; record < mixedRecords.length; record++) {
    const level = Math.floor(record / recordsPerLevel);
    const local = record % recordsPerLevel;
    const x = local % qaColumns;
    const y = level * qaRowsPerLevel + Math.floor(local / qaColumns);
    const values = mixedRecords[record]!.values;
    const target = (y * qaColumns + x) * 4;
    qa[target] = Math.round(clamp(values[0]!, 0, 1) * 255);
    qa[target + 1] = Math.round(clamp(values[1]!, 0, 1) * 255);
    qa[target + 2] = Math.round(clamp(values[2]!, 0, 1) * 255);
    qa[target + 3] = Math.round(clamp(values[3]!, 0, 1) * 255);
  }
  const qaPath = resolve(qaRoot, '001-mixed-premultiplied-level-records.png');
  await sharp(qa, { raw: { width: qaColumns, height: qaRows, channels: 4 } })
    .resize(qaColumns * 8, qaRows * 8, { kernel: 'nearest' })
    .png()
    .toFile(qaPath);

  const report = {
    schema: 'laas-gbr4-boundary-lambert-coupled-reference-report/v4',
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
      gpuResidentBytes:
        addressWords.byteLength
        + mixedWords.byteLength
        + regularWords.byteLength
        + correctionWords.byteLength,
    },
    address: {
      faceCharts: FACE_COUNT,
      boundaryCellsPerAxis: BOUNDARY_CELLS,
      directionCellsPerAxis: DIRECTION_CELLS,
      directionChart: 'Lambert equal-area closed hemisphere disk',
      exactHorizontalRepresentable: true,
      recordsPerLevel,
      totalRecords: addressRecordCount,
    },
    footprintLevels: STANDOFF_LEVELS.map((standoffMetres, level) => ({
      level,
      standoffMetres,
      footprintDiameterMetres: 2 * standoffMetres * Math.tan(PIXEL_ANGLE * 0.5),
      independentlyIntegrated: true,
    })),
    records: {
      invalid: 0,
      certifiedMiss: 0,
      certifiedRegular: completeRegularCorrections.length,
      filteredMixed: mixedRecords.length,
      mixedPayload: [
        'premultiplied RGB + coverage',
        'conservative depth interval + conditional mean/variance',
        'premultiplied first normal moment + coverage',
        'premultiplied RGB second moments + coverage',
      ],
      regularPayloadSchema: [
        'source plane n,c',
        'root xyz, periodic copy x',
        'periodic copy z, primitive, material/species, deformation class',
        'support, incidence, order, terrain/motion firstness margins',
        'coupled constant attribute chart RGB + triangle-domain token',
      ],
      certification: {
        testedFineCells: certificationCandidates,
        centerHits: certificationCenterHits,
        acceptedRegularCells: completeRegularCorrections.length,
        acceptedFractionOfTested: certificationCandidates > 0
          ? completeRegularCorrections.length / certificationCandidates
          : 0,
        acceptedFractionOfCenterHits: certificationCenterHits > 0
          ? completeRegularCorrections.length / certificationCenterHits
          : 0,
        entireFineDomainCells:
          STANDOFF_LEVELS.length * FACE_COUNT * FINE_BOUNDARY_CELLS ** 2 * FINE_DIRECTION_CELLS ** 2,
        note: 'search is a deterministic vertical/top-face subset; it is not a whole-domain regular-fraction estimate',
      },
    },
    evidence,
    runtimeBudget: {
      terminalReadsPerLane: 6,
      schedule: 'correction hit: one hash read + five regular payload reads; correction miss: one hash read + one R32Uint base tag + four mixed payload reads',
      loops: 0,
      marches: 0,
      candidates: 0,
      runtimeGeometry: false,
    },
    validity: {
      status: 'LOADABLE_REFERENCE_RED_NOT_VISUAL_PRODUCTION',
      runtimeLoadable: true,
      visualProductionReady: false,
      proven: [
        'strict versioned container and loader layout',
        'bounded complete Lambert hemisphere coordinate chart',
        'six boundary faces including side and exact-horizontal addresses',
        'four explicit independently integrated shared-origin footprint levels',
        'regular/miss/mixed record classes cannot be confused',
        'mixed cells carry no fictitious categorical surface',
        'fixed six-read terminal schedule and exact upload-byte accounting',
        `${completeRegularCorrections.length} conservative whole-cell flat-source regular corrections`,
      ],
      missing: [
        'whole-domain regular/MISS certification (the reference proves only its sparse correction subset)',
        'tight certified mixed-cell depth intervals',
        'quadrature convergence and held-out image/depth/mm-translation gates',
        'production spatial/angular resolution and compressed macrobrick/VQ fit',
        'finite ecological-patch carrier rho_M and first-entry closure certificates',
        'positive terrain/root/deformation firstness margins (the flat-source correction payload carries zero terrain margin)',
        'runtime binding',
      ],
    },
    qa: {
      path: relative(WORKSPACE, qaPath),
      sha256: sha256(readFileSync(qaPath)),
      interpretation: 'rows grouped by 0.025/0.25/4/32 m footprint levels; each swatch is one honest filtered-mixed record',
    },
    elapsedMilliseconds: performance.now() - started,
  };
  const reportPath = resolve(outputRoot, 'report.json');
  writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`);
  const indexPath = resolve(qaRoot, 'index.json');
  writeFileSync(indexPath, `${JSON.stringify({
    schema: 'laas-groundcover-qa-index/v1',
    sourceSha256: sourceSha,
    recipeSha256: recipeSha,
    files: [
      { file: relative(qaRoot, qaPath), sha256: sha256(readFileSync(qaPath)), width: qaColumns * 8, height: qaRows * 8 },
      { file: relative(qaRoot, reportPath), sha256: sha256(readFileSync(reportPath)) },
    ],
  }, null, 2)}\n`);
  console.log(JSON.stringify({
    outputRoot: relative(WORKSPACE, outputRoot),
    asset: report.container,
    validity: report.validity,
    runtimeBudget: report.runtimeBudget,
    elapsedMilliseconds: report.elapsedMilliseconds,
  }, null, 2));
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) await main();
