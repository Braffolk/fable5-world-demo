/**
 * Deterministic first production cook for the GBC2-K6 asset container.
 *
 * Direct modes are a conservative 16-corner census of the existing v4 owner
 * field. CUT2 records carry an explicit census-only separator flag until the
 * continuous-domain compiler derives their edge/order kind. The accepted
 * source triangle payload is fully physical and runtime-decodable. K6/head
 * sections remain absent until a source-fitted artifact is supplied; the
 * report marks that state BLOCKED, never green.
 */

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';

const WORKSPACE = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
const SOURCE = resolve(WORKSPACE, 'src/assets/groundcover/calamagrostis-canescens.gcrp');
const EXPECTED_SOURCE_SHA = '2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c';
const HEADER_BYTES = 256;
const ALIGNMENT = 256;
const DESCRIPTOR_WIDTH = 2064;
const DESCRIPTOR_HEIGHT = 2064;
const DESCRIPTOR_STRIDE = 16;
const DESCRIPTOR_BYTES = DESCRIPTOR_WIDTH * DESCRIPTOR_HEIGHT * DESCRIPTOR_STRIDE;
const PAYLOAD_STRIDE = 32;
const MODE_MISS = 0;
const MODE_REGULAR = 1;
const MODE_CUT2 = 2;
const MODE_KPLANE = 3;
const MODE_CENSUS_ONLY = 1 << 4;
const MISS_OWNER = 0xffffffff;

function sha256(value: Uint8Array | string): string {
  return createHash('sha256').update(value).digest('hex');
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  const record = value as Record<string, unknown>;
  return `{${Object.keys(record).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`).join(',')}}`;
}

function align(value: number): number {
  return Math.ceil(value / ALIGNMENT) * ALIGNMENT;
}

function writeAscii(target: Uint8Array, offset: number, value: string): void {
  for (let i = 0; i < value.length; i++) target[offset + i] = value.charCodeAt(i);
}

function rgb565(r16: number, g16: number, b16: number): number {
  const r5 = Math.round(r16 * 31 / 65535);
  const g6 = Math.round(g16 * 63 / 65535);
  const b5 = Math.round(b16 * 31 / 65535);
  return (r5 << 11) | (g6 << 5) | b5;
}

function decodeRgb565(value: number): readonly [number, number, number] {
  return [((value >>> 11) & 31) / 31, ((value >>> 5) & 63) / 63, (value & 31) / 31];
}

function quantiles(valuesInput: readonly number[]): Record<string, number> {
  const values = [...valuesInput].sort((a, b) => a - b);
  const at = (p: number): number => values[Math.floor((values.length - 1) * p)] ?? 0;
  return { p50: at(0.5), p95: at(0.95), p99: at(0.99), maximum: values.at(-1) ?? 0 };
}

async function main(): Promise<void> {
  const started = performance.now();
  const source = readFileSync(SOURCE);
  const sourceSha = sha256(source);
  if (sourceSha !== EXPECTED_SOURCE_SHA) throw new Error(`accepted GCRP changed: ${sourceSha}`);
  const sourceView = new DataView(source.buffer, source.byteOffset, source.byteLength);
  if (source.toString('ascii', 0, 4) !== 'GCRP' || sourceView.getUint32(4, true) !== 4) {
    throw new Error('GBC2 cook requires GCRP/v4');
  }
  const profileId = sourceView.getUint32(8, true);
  const storedWidth = sourceView.getUint32(12, true);
  const storedHeight = sourceView.getUint32(16, true);
  const atlasColumns = sourceView.getUint32(20, true);
  const atlasRows = sourceView.getUint32(24, true);
  const sliceCount = sourceView.getUint32(28, true);
  const interiorWidth = sourceView.getUint32(44, true);
  const interiorHeight = sourceView.getUint32(48, true);
  const gutter = sourceView.getUint32(52, true);
  const ownerOffset = sourceView.getUint32(80, true);
  const vertexOffset = sourceView.getUint32(84, true);
  const triangleOffset = sourceView.getUint32(88, true);
  const vertexCount = sourceView.getUint32(92, true);
  const triangleCount = sourceView.getUint32(96, true);
  if (
    storedWidth * atlasColumns !== DESCRIPTOR_WIDTH
    || storedHeight * atlasRows !== DESCRIPTOR_HEIGHT
    || storedWidth !== 258 || storedHeight !== 258
    || interiorWidth !== 256 || interiorHeight !== 256 || gutter !== 1
    || sliceCount !== 64
  ) throw new Error('accepted GCRP atlas no longer matches the frozen 2064x2064 descriptor layout');
  if (profileId > 0xffff) throw new Error(`profile id ${profileId} does not fit species mark`);

  const payloadBytes = triangleCount * PAYLOAD_STRIDE;
  const descriptorOffset = HEADER_BYTES;
  const payloadOffset = align(descriptorOffset + DESCRIPTOR_BYTES);
  const totalBytes = align(payloadOffset + payloadBytes);
  const artifact = Buffer.alloc(totalBytes);
  const header = new DataView(artifact.buffer, artifact.byteOffset, HEADER_BYTES);

  writeAscii(artifact, 0, 'GBC2');
  header.setUint32(4, 1, true);
  header.setUint32(8, HEADER_BYTES, true);
  header.setUint32(12, profileId, true);
  header.setUint32(16, DESCRIPTOR_WIDTH, true);
  header.setUint32(20, DESCRIPTOR_HEIGHT, true);
  header.setUint32(24, DESCRIPTOR_STRIDE, true);
  header.setUint32(28, descriptorOffset, true);
  header.setUint32(32, DESCRIPTOR_BYTES, true);
  header.setUint32(36, payloadOffset, true);
  header.setUint32(40, payloadBytes, true);
  header.setUint32(44, triangleCount, true);
  header.setUint32(48, PAYLOAD_STRIDE, true);
  header.setUint32(52, 0, true); // K6 absent in this first cook.
  header.setUint32(56, 0, true);
  header.setUint32(60, 0, true);
  header.setUint32(64, 0, true);
  header.setUint32(68, 0, true);
  header.setUint32(72, 0, true);
  header.setUint32(76, 0, true);
  header.setUint32(80, 0, true);
  header.setFloat32(84, sourceView.getFloat32(60, true), true);
  header.setFloat32(88, sourceView.getFloat32(64, true), true);
  header.setFloat32(92, sourceView.getFloat32(68, true), true);
  header.setFloat32(96, sourceView.getFloat32(72, true), true);
  for (let axis = 0; axis < 6; axis++) header.setFloat32(100 + axis * 4, sourceView.getFloat32(100 + axis * 4, true), true);
  artifact.set(Buffer.from(sourceSha, 'hex'), 128);

  const descriptorWords = new Uint32Array(
    artifact.buffer,
    artifact.byteOffset + descriptorOffset,
    DESCRIPTOR_BYTES / 4,
  );
  const elevationCount = 4;
  const azimuthCount = sliceCount / elevationCount;
  if (azimuthCount !== 16) throw new Error(`expected 16x4 direction lattice, got ${azimuthCount}x${elevationCount}`);
  const wrap = (value: number, size: number): number => ((value % size) + size) % size;
  const ownerAt = (azimuth: number, elevation: number, phaseX: number, phaseZ: number): number => {
    const slice = wrap(azimuth, azimuthCount) * elevationCount + Math.max(0, Math.min(elevationCount - 1, elevation));
    const atlasX = (slice % atlasColumns) * storedWidth + gutter + wrap(phaseX, interiorWidth);
    const atlasY = Math.floor(slice / atlasColumns) * storedHeight + gutter + wrap(phaseZ, interiorHeight);
    return sourceView.getUint32(ownerOffset + (atlasY * DESCRIPTOR_WIDTH + atlasX) * 4, true);
  };
  const census = { MISS: 0, REGULAR: 0, CUT2: 0, KPLANE: 0 };
  const unique = new Uint32Array(16);
  for (let atlasY = 0; atlasY < DESCRIPTOR_HEIGHT; atlasY++) {
    const tileY = Math.floor(atlasY / storedHeight);
    const localY = atlasY % storedHeight;
    const phaseZ = wrap(localY - gutter, interiorHeight);
    for (let atlasX = 0; atlasX < DESCRIPTOR_WIDTH; atlasX++) {
      const tileX = Math.floor(atlasX / storedWidth);
      const localX = atlasX % storedWidth;
      const phaseX = wrap(localX - gutter, interiorWidth);
      const slice = tileY * atlasColumns + tileX;
      const azimuth = Math.floor(slice / elevationCount);
      const elevation = slice % elevationCount;
      let uniqueCount = 0;
      for (let de = 0; de < 2; de++) for (let da = 0; da < 2; da++) {
        for (let dz = 0; dz < 2; dz++) for (let dx = 0; dx < 2; dx++) {
          const owner = ownerAt(azimuth + da, Math.min(elevationCount - 1, elevation + de), phaseX + dx, phaseZ + dz);
          let seen = false;
          for (let i = 0; i < uniqueCount; i++) if (unique[i] === owner) { seen = true; break; }
          if (!seen) unique[uniqueCount++] = owner;
        }
      }
      let mode: number;
      let ownerA = MISS_OWNER;
      let ownerB = MISS_OWNER;
      if (uniqueCount === 1 && unique[0] === MISS_OWNER) {
        mode = MODE_MISS;
        census.MISS++;
      } else if (uniqueCount === 1) {
        mode = MODE_REGULAR | MODE_CENSUS_ONLY;
        ownerA = unique[0]!;
        census.REGULAR++;
      } else if (uniqueCount === 2) {
        mode = MODE_CUT2 | MODE_CENSUS_ONLY;
        const a = unique[0]!;
        const b = unique[1]!;
        if (a === MISS_OWNER) { ownerA = b; ownerB = a; }
        else if (b === MISS_OWNER) { ownerA = a; ownerB = b; }
        else { ownerA = Math.min(a, b); ownerB = Math.max(a, b); }
        census.CUT2++;
      } else {
        mode = MODE_KPLANE;
        census.KPLANE++;
      }
      const word = (atlasY * DESCRIPTOR_WIDTH + atlasX) * 4;
      descriptorWords[word] = ownerA;
      descriptorWords[word + 1] = ownerB;
      descriptorWords[word + 2] = mode;
      descriptorWords[word + 3] = 0;
    }
  }

  const payloadView = new DataView(artifact.buffer, artifact.byteOffset + payloadOffset, payloadBytes);
  const colorErrors: number[] = [];
  const normalCoordinateErrors: number[] = [];
  const diagnosticColors: Array<readonly [number, number, number]> = [];
  const sampleStride = Math.max(1, Math.floor(triangleCount / 8192));
  for (let triangle = 0; triangle < triangleCount; triangle++) {
    const sourceTriangle = triangleOffset + triangle * 16;
    const target = triangle * PAYLOAD_STRIDE;
    for (let corner = 0; corner < 3; corner++) {
      const vertex = sourceView.getUint32(sourceTriangle + corner * 4, true);
      if (vertex >= vertexCount) throw new Error(`triangle ${triangle} has invalid vertex ${vertex}`);
      const sourceVertex = vertexOffset + vertex * 16;
      const positionTarget = target + corner * 6;
      payloadView.setUint16(positionTarget, sourceView.getUint16(sourceVertex, true), true);
      payloadView.setUint16(positionTarget + 2, sourceView.getUint16(sourceVertex + 2, true), true);
      payloadView.setUint16(positionTarget + 4, sourceView.getUint16(sourceVertex + 4, true), true);
      const r16 = sourceView.getUint16(sourceVertex + 6, true);
      const g16 = sourceView.getUint16(sourceVertex + 8, true);
      const b16 = sourceView.getUint16(sourceVertex + 10, true);
      const packedColor = rgb565(r16, g16, b16);
      payloadView.setUint16(target + 18 + corner * 2, packedColor, true);
      const nx16 = sourceView.getUint16(sourceVertex + 12, true);
      const ny16 = sourceView.getUint16(sourceVertex + 14, true);
      payloadView.setUint8(target + 24 + corner * 2, Math.round(nx16 / 257));
      payloadView.setUint8(target + 25 + corner * 2, Math.round(ny16 / 257));
      if (triangle % sampleStride === 0) {
        const decoded = decodeRgb565(packedColor);
        colorErrors.push(Math.max(
          Math.abs(decoded[0] - r16 / 65535),
          Math.abs(decoded[1] - g16 / 65535),
          Math.abs(decoded[2] - b16 / 65535),
        ));
        normalCoordinateErrors.push(Math.max(
          Math.abs(Math.round(nx16 / 257) / 255 - nx16 / 65535),
          Math.abs(Math.round(ny16 / 257) / 255 - ny16 / 65535),
        ));
        if (corner === 0 && diagnosticColors.length < 8192) diagnosticColors.push(decoded);
      }
    }
    payloadView.setUint16(target + 30, profileId, true);
  }

  const implementationSha = sha256(readFileSync(fileURLToPath(import.meta.url)));
  const recipe = {
    schema: 'laas-gbc2-k6-production-cook-recipe/v1',
    implementationSha256: implementationSha,
    sourceSha256: sourceSha,
    containerVersion: 1,
    descriptor: {
      width: DESCRIPTOR_WIDTH, height: DESCRIPTOR_HEIGHT, format: 'RGBA32Uint',
      mode: '16-corner owner census; REGULAR/CUT2 candidates carry MODE_CENSUS_ONLY',
    },
    payload: { strideBytes: PAYLOAD_STRIDE, format: 'two-RGBA32Uint', color: 'RGB565', normal: 'oct8x2', mark: profileId },
    k6: {
      width: 448, height: 448, layers: 6, fittedMipLevels: 9,
      order: 'mip-major/layer-major/texel/RGBA16F', present: false,
      pairs: ['uv', 'ua', 'ue', 'va', 've', 'ae'],
      head: {
        scalarType: 'float32',
        layout: 'row-major A[4x14], a[4], B[18x4], b[18]',
        bytesUnpadded: 600,
        formula: 'p=product(0.5+saturate(sample_i)); phi=[p,p^2,pairwise(p)]; h=SiLU(A*phi+a); raw=B*h+b',
        output: 'two strata x [alpha, normalized depth moment1, normalized depth moment2, premul RGB3, premul normal XYZ3]',
      },
    },
  };
  const recipeSha = sha256(canonicalJson(recipe));
  const outputRoot = resolve(
    WORKSPACE,
    'data/work/groundcover-gbc2-k6-assets',
    sourceSha.slice(0, 16),
    recipeSha.slice(0, 16),
  );
  const qaRoot = resolve(outputRoot, 'qa');
  mkdirSync(qaRoot, { recursive: true });
  const containerPath = resolve(outputRoot, 'calamagrostis-canescens.gbc2');
  const stableContainerPath = resolve(WORKSPACE, 'src/assets/groundcover/calamagrostis-canescens.gbc2');
  writeFileSync(containerPath, artifact);
  writeFileSync(stableContainerPath, artifact);
  const containerSha = sha256(artifact);

  const report = {
    schema: 'laas-gbc2-k6-production-asset/v1',
    created: new Date().toISOString(),
    source: {
      path: relative(WORKSPACE, SOURCE), sha256: sourceSha, bytes: source.byteLength,
      profileId, vertexCount, triangleCount,
    },
    recipe,
    container: {
      path: relative(WORKSPACE, containerPath), sha256: containerSha,
      stablePath: relative(WORKSPACE, stableContainerPath),
      bytes: artifact.byteLength, MiB: artifact.byteLength / 1048576,
      headerBytes: HEADER_BYTES, alignmentBytes: ALIGNMENT,
      headerLayout: {
        magic: 0, version: 4, headerBytes: 8, profileId: 12,
        descriptorWidth: 16, descriptorHeight: 20, descriptorBytesPerPixel: 24,
        descriptorOffset: 28, descriptorBytes: 32, payloadOffset: 36,
        payloadBytes: 40, triangleCount: 44, payloadStride: 48,
        k6Present: 52, k6Offset: 56, k6Bytes: 60, k6Size: 64,
        k6Layers: 68, k6MipCount: 72, headOffset: 76, headBytes: 80,
        tileOriginAndSize: 84, bounds: 100, sourceSha256: 128,
      },
    },
    sections: {
      descriptor: {
        offset: descriptorOffset, bytes: DESCRIPTOR_BYTES,
        dimensions: [DESCRIPTOR_WIDTH, DESCRIPTOR_HEIGHT], format: 'RGBA32Uint',
        texel: ['ownerA', 'ownerB', 'modeKind', 'reserved'],
        modeKindBits: {
          mode: 'bits 0..1: MISS=0 REGULAR=1 CUT2=2 KPLANE=3',
          kind: 'bits 2..3 (zero until separator compiler)',
          censusOnly: 'bit 4; set on REGULAR/CUT2 candidates',
        },
        census,
        interpretation: '16 corners of the existing 256x256x16x4 owner field; flagged analytic records must route to KPLANE until continuous separator certification clears bit 4',
      },
      payload: {
        offset: payloadOffset, bytes: payloadBytes, triangleCount, strideBytes: PAYLOAD_STRIDE,
        layout: 'xyz16x3 (18 B), RGB565x3 (6 B), oct8x2x3 (6 B), species mark u16 (2 B)',
        sampledQuantisation: {
          triangleStride: sampleStride,
          rgbMaxChannelAbsolute: quantiles(colorErrors),
          octCoordinateAbsolute: quantiles(normalCoordinateErrors),
        },
      },
      k6: {
        offset: 0, bytes: 0, present: false,
        required: 'six 448x448 RGBA16F layers with nine source-fitted scale levels plus fixed head; mip-major/layer-major/texel/RGBA16F order',
      },
      head: { offset: 0, bytes: 0, present: false },
    },
    exactProjectedCompleteBytes: {
      currentDescriptorAndPayload: artifact.byteLength,
      K6FittedLevels: 12_844_752,
      maximumHeadAndAlignment: 786_736,
      completeBeforeExternalControlAndNonTopPages: artifact.byteLength + 13 * 1048576,
    },
    validity: {
      status: 'BLOCKED_MISSING_K6',
      runtimeReady: false,
      blocker: 'KPLANE cells and census-only analytic candidates require a source-fitted K6 transfer, but no K6 scale levels/head are serialized yet',
      resumeCondition: 'fit and held-out validate the six pair-plane fitted-scale transfer, append K6/head sections, and set header presence bits',
    },
    elapsedMilliseconds: performance.now() - started,
  };
  const reportPath = resolve(outputRoot, 'report.json');
  const reportText = `${JSON.stringify(report, null, 2)}\n`;
  writeFileSync(reportPath, reportText);

  const qaFiles = [
    resolve(qaRoot, '01-descriptor-mode-census.png'),
    resolve(qaRoot, '02-payload-colour-quantisation.png'),
  ];
  const totalCells = DESCRIPTOR_WIDTH * DESCRIPTOR_HEIGHT;
  const modeSvg = `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="500"><rect width="1200" height="500" fill="#121719"/><text x="42" y="58" fill="#eff4f3" font-family="monospace" font-size="25">01 — DIRECT DESCRIPTOR 16-CORNER CENSUS</text><text x="42" y="125" fill="#cbd7d1" font-family="monospace" font-size="24">MISS     ${census.MISS.toLocaleString().padStart(9)}  ${(100 * census.MISS / totalCells).toFixed(2)}%</text><text x="42" y="180" fill="#8ed7a8" font-family="monospace" font-size="24">REGULAR  ${census.REGULAR.toLocaleString().padStart(9)}  ${(100 * census.REGULAR / totalCells).toFixed(2)}%</text><text x="42" y="235" fill="#e8c56c" font-family="monospace" font-size="24">CUT2     ${census.CUT2.toLocaleString().padStart(9)}  ${(100 * census.CUT2 / totalCells).toFixed(2)}%</text><text x="42" y="290" fill="#bd91ef" font-family="monospace" font-size="24">KPLANE   ${census.KPLANE.toLocaleString().padStart(9)}  ${(100 * census.KPLANE / totalCells).toFixed(2)}%</text><text x="42" y="370" fill="#e8c56c" font-family="monospace" font-size="20">REGULAR/CUT2 carry bit 4: census-only; runtime must use K6 until certified.</text><text x="42" y="420" fill="#cbd7d1" font-family="monospace" font-size="20">Total ${totalCells.toLocaleString()} RGBA32Uint descriptor texels.</text></svg>`;
  await sharp(Buffer.from(modeSvg)).png().toFile(qaFiles[0]!);
  const swatchWidth = 128;
  const swatchHeight = 64;
  const raw = Buffer.alloc(swatchWidth * swatchHeight * 3);
  for (let i = 0; i < swatchWidth * swatchHeight; i++) {
    const color = diagnosticColors[i % Math.max(1, diagnosticColors.length)] ?? [0, 0, 0];
    raw[i * 3] = Math.round(color[0] * 255);
    raw[i * 3 + 1] = Math.round(color[1] * 255);
    raw[i * 3 + 2] = Math.round(color[2] * 255);
  }
  await sharp(raw, { raw: { width: swatchWidth, height: swatchHeight, channels: 3 } }).resize(1024, 512, { kernel: 'nearest' }).png().toFile(qaFiles[1]!);
  const index = {
    schema: 'laas-gbc2-k6-production-qa-index/v1', sourceSha256: sourceSha, recipeSha256: recipeSha,
    report: { path: relative(WORKSPACE, reportPath), sha256: sha256(reportText) },
    images: qaFiles.map((path, index) => ({ number: index + 1, path: relative(WORKSPACE, path), sha256: sha256(readFileSync(path)) })),
  };
  writeFileSync(resolve(qaRoot, 'index.json'), `${JSON.stringify(index, null, 2)}\n`);
  console.log(JSON.stringify({ outputRoot: relative(WORKSPACE, outputRoot), report: relative(WORKSPACE, reportPath), container: report.container, validity: report.validity }, null, 2));
}

await main();
