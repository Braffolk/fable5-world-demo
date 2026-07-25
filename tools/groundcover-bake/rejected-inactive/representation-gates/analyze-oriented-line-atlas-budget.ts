/**
 * Offline-only storage bound for a dense nearest categorical atlas in the six
 * signed dominant-axis oriented-line charts. This performs no runtime bake and
 * proposes no shader path.
 */

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { decodeOwnedProfileGeometry } from '../../ExteriorClosureCensus';

const WORKSPACE = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
const DEFAULT_SOURCE = resolve(WORKSPACE, 'src/assets/groundcover/calamagrostis-canescens.gcrp');
const RESIDENT_CAP_BYTES = 51_121_152;
const HORIZON_METRES = 155;
const TARGET_COORDINATE_BOUNDS_METRES = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1] as const;

function sha256(bytes: Uint8Array | string): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function horizontalSlopeSensitivityIntegral(height: number, horizon: number): number {
  // J = integral_{-1}^{1} integral_{-1}^{1}
  //       min(H/|r_y|, L/sqrt(1+r_y^2+r_z^2))^2 dr_y dr_z.
  // The closed form follows by splitting at
  // |r_y|=H*sqrt(1+r_z^2)/sqrt(L^2-H^2).
  if (!(horizon > height && height > 0)) throw new Error('horizon must exceed positive height');
  const root = Math.sqrt(horizon * horizon - height * height);
  const coefficient = horizon * horizon * Math.atan(height / root) + height * root;
  return 4 * (coefficient * Math.asinh(1) - height * height);
}

function continuumCoefficient(tileX: number, tileZ: number, height: number, horizon: number): {
  vertical: number;
  horizontal: number;
  total: number;
  horizontalSlopeSensitivityIntegral: number;
} {
  const slopeIntegral = horizontalSlopeSensitivityIntegral(height, horizon);
  // For a coordinate-error budget epsilon:
  // - two signed Y charts need 8*Px*Pz*H^2 / epsilon^4 records;
  // - four signed X/Z charts need 4*Px*Pz*J / epsilon^4 records.
  // This already exploits variable slope density near grazing rather than a
  // worst-case uniform slope grid.
  const vertical = tileX * tileZ * 8 * height * height;
  const horizontal = tileX * tileZ * 4 * slopeIntegral;
  return { vertical, horizontal, total: vertical + horizontal, horizontalSlopeSensitivityIntegral: slopeIntegral };
}

function equalTensorGrid(capBytes: number, bytesPerRecord: number): {
  binsPerCoordinate: number;
  records: number;
  bytes: number;
  remainingBytes: number;
} {
  const binsPerCoordinate = Math.floor((capBytes / (6 * bytesPerRecord)) ** 0.25);
  const records = 6 * binsPerCoordinate ** 4;
  const bytes = records * bytesPerRecord;
  return { binsPerCoordinate, records, bytes, remainingBytes: capBytes - bytes };
}

const cli = new Map<string, string>();
for (let i = 2; i < process.argv.length; i++) {
  const key = process.argv[i]!;
  if (!key.startsWith('--')) throw new Error(`unexpected positional argument ${key}`);
  const value = process.argv[++i];
  if (!value) throw new Error(`missing value for ${key}`);
  cli.set(key.slice(2), value);
}

const sourcePath = resolve(WORKSPACE, cli.get('source') ?? DEFAULT_SOURCE);
const sourceBytes = readFileSync(sourcePath);
const geometry = decodeOwnedProfileGeometry(sourceBytes);
const height = geometry.topH - geometry.bounds.min[1];
const coefficients = continuumCoefficient(geometry.tileSizeX, geometry.tileSizeZ, height, HORIZON_METRES);
const toolBytes = readFileSync(fileURLToPath(import.meta.url));
const configuration = {
  schema: 'laas-groundcover-oriented-line-atlas-budget/v1',
  residentCapBytes: RESIDENT_CAP_BYTES,
  horizonMetres: HORIZON_METRES,
  signedDominantAxisCharts: 6,
  recordBytesEvaluated: [4, 6],
  targetCoordinateBoundsMetres: TARGET_COORDINATE_BOUNDS_METRES,
  toolSha256: sha256(toolBytes),
};
const sourceSha256 = sha256(sourceBytes);
const recipeSha256 = sha256(JSON.stringify({ sourceSha256, configuration }));
const outputPath = resolve(
  WORKSPACE,
  cli.get('output') ?? `data/work/groundcover-oriented-line-atlas-budget/${sourceSha256}/${recipeSha256}/report.json`,
);

const recordLayouts = configuration.recordBytesEvaluated.map((bytesPerRecord) => {
  const recordCap = RESIDENT_CAP_BYTES / bytesPerRecord;
  const coordinateErrorBoundMetres = (coefficients.total / recordCap) ** 0.25;
  return {
    bytesPerRecord,
    payloadBoundary: bytesPerRecord === 4
      ? 'strict geometry-only lower bound: u16 line parameter plus u16 appearance-or-normal; cannot carry depth, normal, authored appearance, and mark simultaneously'
      : 'minimum plausible categorical event: u16 line parameter, oct8x2 normal, u16 appearance/mark palette index; palette and metadata bytes are not charged',
    maximumRawRecords: Math.floor(recordCap),
    equalTensorGrid: equalTensorGrid(RESIDENT_CAP_BYTES, bytesPerRecord),
    optimisticAxisSpecificContinuum: {
      coordinateErrorBoundMetres,
      transverseEuclideanErrorBoundMetres: Math.SQRT2 * coordinateErrorBoundMetres,
      note: 'best-case continuum allocation with variable horizontal slope density; ignores integer bins, gutters, mips, palettes, metadata, categorical boundaries, and empty cells',
    },
  };
});

const targetBounds = TARGET_COORDINATE_BOUNDS_METRES.map((coordinateErrorBoundMetres) => {
  const records = coefficients.total / coordinateErrorBoundMetres ** 4;
  return {
    coordinateErrorBoundMetres,
    transverseEuclideanErrorBoundMetres: Math.SQRT2 * coordinateErrorBoundMetres,
    optimisticContinuumRecords: records,
    bytesAt4PerRecord: records * 4,
    bytesAt6PerRecord: records * 6,
    capMultiplesAt4PerRecord: records * 4 / RESIDENT_CAP_BYTES,
    capMultiplesAt6PerRecord: records * 6 / RESIDENT_CAP_BYTES,
  };
});

const report = {
  schema: configuration.schema,
  source: {
    path: relative(WORKSPACE, sourcePath),
    bytes: sourceBytes.byteLength,
    sha256: sourceSha256,
    tileSizeMetres: [geometry.tileSizeX, geometry.tileSizeZ],
    botanicalHeightMetres: height,
    triangles: geometry.triangleCount,
  },
  configuration,
  recipeSha256,
  mathematicalContract: {
    coordinates: 'six signed dominant-axis charts; two plane intersections are equivalent to (Q_a,Q_b,r_a,r_b), with |r_a|,|r_b|<=1',
    quantization: 'nearest/Voronoi complete marked event; line address is origin invariant; chart overlaps must be exact identity pairs',
    errorBudget: 'phase half-cell plus slope half-cell each receive half the per-coordinate transverse error bound; horizontal slope cells adapt to finite in-slab/horizon axis travel',
    verticalSignedChartsContinuumCoefficient: coefficients.vertical,
    horizontalSignedChartsContinuumCoefficient: coefficients.horizontal,
    horizontalSlopeSensitivityIntegral: coefficients.horizontalSlopeSensitivityIntegral,
    totalContinuumRecords: `${coefficients.total} / epsilon^4`,
  },
  recordLayouts,
  targetBounds,
  decision: {
    exteriorAtlas: 'reject dense direct atlas under the current resident cap',
    reasons: [
      'the optimistic 4-byte allocation permits only about a 10.2 cm per-coordinate / 14.4 cm transverse line bound',
      'the plausible 6-byte event permits only about an 11.3 cm per-coordinate / 16.0 cm transverse line bound before palette and metadata bytes',
      'these are only line-perturbation bounds; nearest categorical event error is unbounded at occlusion boundaries and can be much larger',
      'camera-inside queries remain five-dimensional successors and are not solved by this exterior atlas',
    ],
    resumeCondition: 'a non-dense analytic sharing mechanism must prove coupled event identity and a complete <=51,121,152-byte / <=4-read / <=256-FMA fixed cost before codec or shader work',
  },
};

mkdirSync(dirname(outputPath), { recursive: true });
writeFileSync(outputPath, `${JSON.stringify(report, null, 2)}\n`);
console.error(`[oriented-line-atlas-budget] ${outputPath}`);
console.error(`[oriented-line-atlas-budget] recipe ${recipeSha256}`);
