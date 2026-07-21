import { spawn, type ChildProcess } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import sharp from 'sharp';
import { chromium, type Browser } from 'playwright';
import { decodeOct, makeDirection, PROFILE_TEXEL_BYTES } from './ProfileFormat';
import {
  ESTONIAN_LOW_COVER_PROFILE_IDS,
  makeEstonianLowCoverFixture,
  type EstonianLowCoverFixture,
  type EstonianLowCoverProfileId,
} from './EstonianLowCover';
import {
  diagnosePeriodicGpuMismatches,
  enumeratePeriodicPixelHit,
  enumeratePeriodicRasterPixelHits,
  makePeriodicSlice,
  packPeriodicProfile,
  parsePeriodicHeader,
  periodicAddress,
  type PeriodicBakeResult,
} from './PeriodicProfile';

interface Args {
  [key: string]: string | boolean;
}

const AZIMUTHS = Array.from({ length: 16 }, (_value, index) => index * 22.5);
const ELEVATIONS = [15, 35, 55, 75] as const;
const ATLAS_COLUMNS = 8;
const ATLAS_ROWS = 8;
const ADDRESS_TOLERANCE = Number.EPSILON * 32;

const PROFILE_SLUGS: Record<EstonianLowCoverProfileId, string> = {
  6: 'pleurozium-schreberi',
  7: 'cladonia-rangiferina',
  8: 'oxalis-acetosella',
  9: 'maianthemum-bifolium',
  10: 'vaccinium-myrtillus',
  11: 'calluna-vulgaris',
};

const PROVENANCE: Record<EstonianLowCoverProfileId, Array<{ source: string; url: string; binding: string }>> = {
  6: [
    {
      source: 'British Bryological Society species account and field-guide sheet',
      url: 'https://www.britishbryologicalsociety.org.uk/learning/species-finder/pleurozium-schreberi/',
      binding: 'pleurocarp mat, several-centimetre shoots, loose simple pinnation, and concave overlapping oval leaves',
    },
    {
      source: 'eElurikkus Estonia occurrence/project record',
      url: 'https://elurikkus.ee/projects/meie-naabrid',
      binding: 'Estonian presence and native low-cover palette relevance (palusammal)',
    },
  ],
  7: [
    {
      source: 'Consortium of Lichen Herbaria species account',
      url: 'https://lichenportal.org/portal/taxa/index.php?taxon=Cladonia+rangiferina',
      binding: 'densely aggregated 50-120 mm secondary thallus, 0.8-1.8 mm terete stipes, branching podetia, evanescent primary thallus',
    },
    {
      source: 'eElurikkus accepted Estonian taxon',
      url: 'https://elurikkus.ee/app/taxonomy/taxon/139351',
      binding: 'accepted Estonian identity and common name harilik põdrasamblik',
    },
  ],
  8: [
    {
      source: 'World Flora Online / Flora of China description',
      url: 'https://www.worldfloraonline.org/taxon/wfo-0000387238',
      binding: 'slender creeping rhizome, 3-15 cm petiole, three obcordate deeply emarginate leaflets, solitary five-part flowers',
    },
    {
      source: 'eElurikkus native urban-forest project record',
      url: 'https://elurikkus.ee/projects/meie-naabrid',
      binding: 'Estonian native/persistent classification and recorded woodland occurrence',
    },
  ],
  9: [
    {
      source: 'World Flora Online / Flora of China description',
      url: 'https://www.worldfloraonline.org/taxon/wfo-0000691280',
      binding: '8-25 cm rhizomatous shoots, usually two cordate cauline leaves, and 10-25-flowered erect raceme',
    },
    {
      source: 'eElurikkus occurrence record',
      url: 'https://elurikkus.ee/app/occurrences/occurrence/44461108',
      binding: 'verified Estonian occurrence and accepted taxon identity',
    },
  ],
  10: [
    {
      source: 'World Flora Online / Flora of China and Flora of North America descriptions',
      url: 'https://www.worldfloraonline.org/taxon/wfo-0000422209',
      binding: 'rhizomatous much-branched dwarf shrub, conspicuously angled green twigs, alternate ovate serrulate leaves',
    },
    {
      source: 'eElurikkus occurrence record',
      url: 'https://elurikkus.ee/app/occurrences/occurrence/66598395',
      binding: 'verified Estonian occurrence and accepted taxon identity',
    },
  ],
  11: [
    {
      source: 'World Flora Online / Flora of North America and Flora Helvetica descriptions',
      url: 'https://www.worldfloraonline.org/taxon/wfo-0000580837',
      binding: 'richly branched dwarf shrub with dense 1-3.5 mm, four-ranked, imbricate scale leaves',
    },
    {
      source: 'eElurikkus accepted Estonian taxon',
      url: 'https://elurikkus.ee/app/taxonomy/taxon/3309',
      binding: 'accepted Estonian identity and common name kanarbik',
    },
  ],
};

function parseArgs(argv: string[]): Args {
  const result: Args = {};
  for (let index = 0; index < argv.length; index++) {
    const argument = argv[index] as string;
    if (!argument.startsWith('--')) continue;
    const value = argv[index + 1];
    if (value !== undefined && !value.startsWith('--')) {
      result[argument.slice(2)] = value;
      index++;
    } else {
      result[argument.slice(2)] = true;
    }
  }
  return result;
}

function stringArg(value: string | boolean | undefined): string | undefined {
  return typeof value === 'string' ? value : undefined;
}

function selectedProfileIds(value: string | undefined): EstonianLowCoverProfileId[] {
  if (value === undefined || value === 'all') return [...ESTONIAN_LOW_COVER_PROFILE_IDS];
  const ids = value.split(',').map((item) => Number(item.trim()));
  for (const id of ids) {
    if (!ESTONIAN_LOW_COVER_PROFILE_IDS.includes(id as EstonianLowCoverProfileId)) {
      throw new Error(`unknown --profile ${id}; expected comma-separated IDs 6-11 or all`);
    }
  }
  return ids as EstonianLowCoverProfileId[];
}

async function waitForServer(url: string, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {
      // The localhost server is still starting.
    }
    await new Promise((done) => setTimeout(done, 200));
  }
  throw new Error(`localhost Vite server did not become ready within ${timeoutMs} ms`);
}

async function launchWebGPU(): Promise<Browser> {
  const recipes: Parameters<typeof chromium.launch>[0][] = [
    { headless: true, channel: 'chromium', args: [] },
    { headless: true, channel: 'chromium', args: ['--enable-unsafe-webgpu'] },
    { headless: false, args: ['--enable-unsafe-webgpu'] },
  ];
  let lastError: unknown;
  for (const recipe of recipes) {
    try {
      return await chromium.launch(recipe);
    } catch (error) {
      lastError = error;
    }
  }
  throw new Error(`could not launch WebGPU Chromium: ${lastError instanceof Error ? lastError.message : lastError}`);
}

function hashJson(value: unknown): string {
  return createHash('sha256').update(JSON.stringify(value)).digest('hex');
}

function hashFile(path: string): string {
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}

function writeAcceptedSummary(outputRoot: string): { path: string; count: number } {
  const profiles: unknown[] = [];
  for (const profileId of ESTONIAN_LOW_COVER_PROFILE_IDS) {
    const slug = PROFILE_SLUGS[profileId];
    const profileRoot = resolve(outputRoot, slug);
    if (!existsSync(profileRoot)) continue;
    const currentFixture = makeEstonianLowCoverFixture(profileId);
    const currentGeometrySha256 = hashJson(currentFixture.mesh);
    const candidates = readdirSync(profileRoot).sort().reverse();
    const artifactHash = candidates.find((candidate) => {
      const candidateIndex = resolve(profileRoot, candidate, 'qa/index.json');
      if (!existsSync(candidateIndex)) return false;
      const candidateRecord = JSON.parse(readFileSync(candidateIndex, 'utf8')) as {
        recipe?: { fixture?: string; mesh?: { indexedGeometrySha256?: string } };
      };
      return candidateRecord.recipe?.fixture === currentFixture.generator &&
        candidateRecord.recipe.mesh?.indexedGeometrySha256 === currentGeometrySha256;
    });
    if (!artifactHash) continue;
    const artifactDir = resolve(profileRoot, artifactHash);
    const indexPath = resolve(artifactDir, 'qa/index.json');
    const index = JSON.parse(readFileSync(indexPath, 'utf8')) as {
      recipe: { species: string; fixture: string; mesh: unknown; morphology: unknown };
      binary: { file: string; sha256: string; bytes: number };
      adapter: string;
      deterministicRepeat: unknown;
      selectedPixelCpuRasterAcceptance: unknown;
      periodicAddressAcceptance: unknown;
    };
    profiles.push({
      profileId,
      species: index.recipe.species,
      generator: index.recipe.fixture,
      mesh: index.recipe.mesh,
      morphology: index.recipe.morphology,
      adapter: index.adapter,
      binary: {
        path: resolve(artifactDir, index.binary.file.replace(/^\.\.\//, '')),
        sha256: index.binary.sha256,
        bytes: index.binary.bytes,
      },
      qaIndex: indexPath,
      deterministicRepeat: index.deterministicRepeat,
      selectedPixelCpuRasterAcceptance: index.selectedPixelCpuRasterAcceptance,
      periodicAddressAcceptance: index.periodicAddressAcceptance,
    });
  }
  mkdirSync(outputRoot, { recursive: true });
  const path = resolve(outputRoot, 'accepted-profiles.json');
  writeFileSync(path, `${JSON.stringify({ schema: 'laas-groundcover-lowcover-bake-summary/v1', profiles }, null, 2)}\n`);
  return { path, count: profiles.length };
}

function diagnosticPixels(result: PeriodicBakeResult, mode: 'depth' | 'normal'): Uint8Array {
  const width = result.tileWidth * result.atlasColumns;
  const height = result.tileHeight * result.atlasRows;
  const output = new Uint8Array(width * height * 4);
  for (let texel = 0; texel < width * height; texel++) {
    const source = texel * 4;
    const target = texel * 4;
    const hit = (result.pixels[source + 3] as number) > 0.5;
    if (!hit) {
      output.set([18, 8, 22, 255], target);
    } else if (mode === 'depth') {
      const depth = Math.max(0, Math.min(255, Math.round((1 - (result.pixels[source] as number)) * 255)));
      output.set([depth, depth, depth, 255], target);
    } else {
      const normal = decodeOct(result.pixels[source + 1] as number, result.pixels[source + 2] as number);
      output.set([
        Math.round((normal.x * 0.5 + 0.5) * 255),
        Math.round((normal.y * 0.5 + 0.5) * 255),
        Math.round((normal.z * 0.5 + 0.5) * 255),
        255,
      ], target);
    }
  }
  return output;
}

interface BotanicalValidation {
  selected: number;
  skippedTriangleBoundary: number;
  rasterOwnerDiffersFromAnalytic: number;
  subpixelOnlyRasterHits: number;
  comparedHits: number;
  comparedMisses: number;
  hitMismatch: number;
  maxDepthTError: number;
  rmsDepthTError: number;
  minNormalDot: number;
  meanNormalDot: number;
  mismatchSamples: Array<{ sliceIndex: number; pixelX: number; pixelY: number; cpuHit: boolean; rasterHit: boolean; gpuHit: boolean }>;
}

/**
 * The shared oracle originally skipped a raster transcription when the ideal
 * Moller-Trumbore center ray missed. Thin botanical silhouettes can still own
 * that center after the specified 8-bit window-space edge quantization. Treat
 * the submitted-f32 raster election as authoritative for those centers too.
 */
function validateBotanicalBake(result: PeriodicBakeResult, mesh: EstonianLowCoverFixture['mesh'], sampleStep = 32): BotanicalValidation {
  const atlasWidth = result.tileWidth * result.atlasColumns;
  let selected = 0;
  let skippedTriangleBoundary = 0;
  let rasterOwnerDiffersFromAnalytic = 0;
  let subpixelOnlyRasterHits = 0;
  let comparedHits = 0;
  let comparedMisses = 0;
  let hitMismatch = 0;
  let depthSq = 0;
  let maxDepthTError = 0;
  let normalSum = 0;
  let minNormalDot = 1;
  const mismatchSamples: BotanicalValidation['mismatchSamples'] = [];
  const dot = (a: { x: number; y: number; z: number }, b: { x: number; y: number; z: number }): number =>
    a.x * b.x + a.y * b.y + a.z * b.z;
  result.slices.forEach((slice, sliceIndex) => {
    const tileX = (sliceIndex % result.atlasColumns) * result.tileWidth;
    const tileY = Math.floor(sliceIndex / result.atlasColumns) * result.tileHeight;
    for (let pixelY = 3; pixelY < result.tileHeight - 3; pixelY += sampleStep) {
      for (let pixelX = 3; pixelX < result.tileWidth - 3; pixelX += sampleStep) {
        selected++;
        const cpu = enumeratePeriodicPixelHit(
          mesh,
          result.tile,
          slice,
          sliceIndex,
          pixelX,
          pixelY,
          result.tileWidth,
          result.tileHeight,
        ).nearest;
        if (cpu && cpu.edgeMargin < 0.03) {
          skippedTriangleBoundary++;
          continue;
        }
        const raster = enumeratePeriodicRasterPixelHits(
          mesh,
          result.tile,
          slice,
          pixelX,
          pixelY,
          result.tileWidth,
          result.tileHeight,
        )[0] ?? null;
        if (!cpu && raster) subpixelOnlyRasterHits++;
        if (cpu && raster && (raster.triangleId !== cpu.triangleId || raster.copy.ix !== cpu.copy.ix || raster.copy.iz !== cpu.copy.iz)) {
          rasterOwnerDiffersFromAnalytic++;
        }
        const index = ((tileY + pixelY) * atlasWidth + tileX + pixelX) * 4;
        const gpuHit = (result.pixels[index + 3] as number) > 0.5;
        const expected = raster ?? cpu;
        if (!expected) {
          if (gpuHit) {
            hitMismatch++;
            mismatchSamples.push({ sliceIndex, pixelX, pixelY, cpuHit: false, rasterHit: false, gpuHit: true });
          } else {
            comparedMisses++;
          }
          continue;
        }
        if (!gpuHit) {
          hitMismatch++;
          mismatchSamples.push({ sliceIndex, pixelX, pixelY, cpuHit: cpu !== null, rasterHit: raster !== null, gpuHit: false });
          continue;
        }
        const gpuT = slice.depthMin + (result.pixels[index] as number) * (slice.depthMax - slice.depthMin);
        const depthError = Math.abs(gpuT - expected.t);
        const gpuNormal = decodeOct(result.pixels[index + 1] as number, result.pixels[index + 2] as number);
        const normalDot = dot(expected.normal, gpuNormal);
        depthSq += depthError * depthError;
        maxDepthTError = Math.max(maxDepthTError, depthError);
        normalSum += normalDot;
        minNormalDot = Math.min(minNormalDot, normalDot);
        comparedHits++;
      }
    }
  });
  return {
    selected,
    skippedTriangleBoundary,
    rasterOwnerDiffersFromAnalytic,
    subpixelOnlyRasterHits,
    comparedHits,
    comparedMisses,
    hitMismatch,
    maxDepthTError,
    rmsDepthTError: comparedHits > 0 ? Math.sqrt(depthSq / comparedHits) : Number.POSITIVE_INFINITY,
    minNormalDot: comparedHits > 0 ? minNormalDot : -1,
    meanNormalDot: comparedHits > 0 ? normalSum / comparedHits : -1,
    mismatchSamples,
  };
}

function addressAcceptance(tile: PeriodicBakeResult['tile']): { samples: number; maxUError: number; maxVError: number } {
  let samples = 0;
  let maxUError = 0;
  let maxVError = 0;
  for (const sample of [{ x: -1.37, z: 2.81 }, { x: tile.sizeX * 0.137, z: tile.sizeZ * 0.819 }, { x: 1.003, z: -0.011 }]) {
    const base = periodicAddress(sample.x, sample.z, tile);
    for (let ix = -3; ix <= 3; ix++) {
      for (let iz = -3; iz <= 3; iz++) {
        const moved = periodicAddress(sample.x + ix * tile.sizeX, sample.z + iz * tile.sizeZ, tile);
        maxUError = Math.max(maxUError, Math.abs(base.u - moved.u));
        maxVError = Math.max(maxVError, Math.abs(base.v - moved.v));
        samples++;
      }
    }
  }
  return { samples, maxUError, maxVError };
}

async function writeArtifact(
  outputRoot: string,
  fixture: EstonianLowCoverFixture,
  first: PeriodicBakeResult,
  second: PeriodicBakeResult,
  validation: BotanicalValidation,
): Promise<{ binaryPath: string; indexPath: string; sha256: string }> {
  const packed = packPeriodicProfile(first, fixture.profileId, 1);
  const repeated = packPeriodicProfile(second, fixture.profileId, 1);
  if (packed.sha256 !== repeated.sha256) {
    throw new Error(`${fixture.species} packed repeat differs: ${packed.sha256} != ${repeated.sha256}`);
  }
  const header = parsePeriodicHeader(packed.bytes);
  const slug = PROFILE_SLUGS[fixture.profileId];
  const artifactDir = resolve(outputRoot, slug, packed.sha256.slice(0, 16));
  const qaDir = resolve(artifactDir, 'qa');
  mkdirSync(qaDir, { recursive: true });
  const binaryName = `${slug}-periodic-profile.gcrp`;
  const binaryPath = resolve(artifactDir, binaryName);
  writeFileSync(binaryPath, packed.bytes);
  const width = first.tileWidth * first.atlasColumns;
  const height = first.tileHeight * first.atlasRows;
  const depthPath = resolve(qaDir, '01-periodic-first-hit-depth.png');
  const normalPath = resolve(qaDir, '02-periodic-first-hit-normal.png');
  await sharp(diagnosticPixels(first, 'depth'), { raw: { width, height, channels: 4 } }).png().toFile(depthPath);
  await sharp(diagnosticPixels(first, 'normal'), { raw: { width, height, channels: 4 } }).png().toFile(normalPath);
  const directionSpecs = AZIMUTHS.flatMap((azimuthDeg) => ELEVATIONS.map((elevationDeg) => ({ azimuthDeg, elevationDeg })));
  const address = addressAcceptance(first.tile);
  if (address.maxUError > ADDRESS_TOLERANCE || address.maxVError > ADDRESS_TOLERANCE) {
    throw new Error(`${fixture.species} periodic address failed: ${JSON.stringify(address)}`);
  }
  const recipe = {
    fixture: fixture.generator,
    species: fixture.species,
    profileId: fixture.profileId,
    morphology: fixture.morphology,
    claimBoundary: fixture.claimBoundary,
    tile: fixture.tile,
    mesh: {
      vertices: fixture.mesh.positions.length / 3,
      triangles: fixture.mesh.indices.length / 3,
      indexedGeometrySha256: hashJson(fixture.mesh),
    },
    directionLattice: {
      ordering: 'azimuth-major/elevation-minor: sliceIndex = azimuthIndex * 4 + elevationIndex',
      azimuthDegrees: AZIMUTHS,
      elevationDegrees: ELEVATIONS,
      slices: first.slices.map((slice, index) => ({
        index,
        ...directionSpecs[index],
        direction: slice.direction,
        depthMin: slice.depthMin,
        depthMax: slice.depthMax,
        copyRange: slice.copyRange,
        copyCount: slice.copies.length,
      })),
    },
    interiorTilePixels: first.tileWidth,
    gutter: 1,
    atlas: { columns: first.atlasColumns, rows: first.atlasRows },
  };
  const storedAtlasWidth = packed.storedTileWidth * first.atlasColumns;
  const storedAtlasHeight = packed.storedTileHeight * first.atlasRows;
  const index = {
    schema: 'laas-groundcover-gpu-bake-qa/v2-periodic-top-xz-botanical',
    provenance: {
      authoredAsset: 'original deterministic procedural indexed geometry; no external mesh, image, or texture was ingested',
      publicMorphologyAndEstoniaSources: PROVENANCE[fixture.profileId],
      sourceLedger: '../../../tools/groundcover-bake/ESTONIAN-LOW-COVER-README.md',
      claimBoundary: fixture.claimBoundary,
    },
    recipeSha256: hashJson(recipe),
    binary: {
      file: `../${binaryName}`,
      sha256: packed.sha256,
      bytes: packed.bytes.byteLength,
      profileId: header.profileId,
      format: 'GCRP/v2 periodic-top-XZ, little-endian, RGBA16 payload',
      projection: 'rayOriginXZ = p.xz - d.xz * ((p.y - topH) / d.y)',
      texel: ['normalized ray parameter t (65535 means miss)', 'ray-facing oct-normal x', 'ray-facing oct-normal y', 'hit mask'],
      interiorTile: `${header.interiorTileWidth}x${header.interiorTileHeight}`,
      storedTile: `${header.storedTileWidth}x${header.storedTileHeight}`,
      gutter: 'one wrapped texel from the opposite canonical edge on all sides and corners',
    },
    diagnosticImages: [
      { number: 1, file: '01-periodic-first-hit-depth.png', sha256: hashFile(depthPath), width, height, interpretation: '64 azimuth-major direction slices; white is nearer first hit, dark purple is miss' },
      { number: 2, file: '02-periodic-first-hit-normal.png', sha256: hashFile(normalPath), width, height, interpretation: 'ray-facing decoded world normal; all slices address the same canonical XZ tile' },
    ],
    recipe,
    adapter: first.adapter,
    deterministicRepeat: { runs: 2, identicalSha256: true },
    selectedPixelCpuRasterAcceptance: {
      scope: '64 production directions sampled every 32 interior texels; strict ideal-ray interiors and subpixel-only covered silhouettes both use the submitted-f32 8-bit-subpixel raster owner',
      requirements: { comparedHitsAtLeast: 1, hitMismatch: 0, rmsDepthTErrorAtMost: 2e-5, maxDepthTErrorAtMost: 2e-4, meanNormalDotAtLeast: 0.999, minNormalDotAtLeast: 0.99 },
      observed: validation,
    },
    periodicAddressAcceptance: {
      requirements: { maxUError: ADDRESS_TOLERANCE, maxVError: ADDRESS_TOLERANCE },
      observed: address,
    },
    runtimeResourceProjection: {
      artifactAtlas: `${storedAtlasWidth}x${storedAtlasHeight} rgba16unorm`,
      payloadBytes: storedAtlasWidth * storedAtlasHeight * PROFILE_TEXEL_BYTES,
      sampling: 'four fixed direction-corner taps for one profile; a bounded A/B cover mixture is at most eight fixed taps',
      fixedCost: 'offline geometry/raster complexity adds no runtime march, data-dependent loop, or extra per-instance geometry',
    },
  };
  const indexPath = resolve(qaDir, 'index.json');
  writeFileSync(indexPath, `${JSON.stringify(index, null, 2)}\n`);
  return { binaryPath, indexPath, sha256: packed.sha256 };
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const port = Number(stringArg(args.port) ?? '5231');
  const timeoutMs = Number(stringArg(args.timeout) ?? '300000');
  const tilePixels = Number(stringArg(args.tile) ?? '64');
  const outputRoot = resolve(stringArg(args.out) ?? '/tmp/groundcover-gpu-bake-lowcover');
  const profileIds = selectedProfileIds(stringArg(args.profile));
  if (args['summarize-only'] === true) {
    const summary = writeAcceptedSummary(outputRoot);
    console.log(`[groundcover-lowcover-bake] collected ${summary.count} accepted profile(s); summary: ${summary.path}`);
    return;
  }
  const repoRoot = resolve(import.meta.dirname, '../..');
  const directionSpecs = AZIMUTHS.flatMap((azimuthDeg) => ELEVATIONS.map((elevationDeg) => ({ azimuthDeg, elevationDeg })));
  const directions = directionSpecs.map(({ azimuthDeg, elevationDeg }) => makeDirection(azimuthDeg, elevationDeg));
  const vite: ChildProcess = spawn(
    'npx',
    ['vite', '--port', String(port), '--strictPort', '--host', 'localhost'],
    { cwd: repoRoot, stdio: ['ignore', 'pipe', 'pipe'] },
  );
  vite.stdout?.on('data', (chunk: Buffer) => {
    const output = chunk.toString();
    if (/error/i.test(output)) process.stdout.write(`[vite] ${output}`);
  });
  vite.stderr?.on('data', (chunk: Buffer) => process.stderr.write(`[vite] ${chunk.toString()}`));
  let browser: Browser | null = null;
  const cleanup = (): void => {
    if (browser) browser.close().catch(() => undefined);
    if (!vite.killed) vite.kill('SIGTERM');
  };
  try {
    const url = `http://localhost:${port}/tools/groundcover-bake/periodic.html`;
    await waitForServer(url, timeoutMs);
    browser = await launchWebGPU();
    const page = await browser.newPage();
    const diagnostics: string[] = [];
    page.on('console', (message) => {
      if (message.type() === 'error' || message.type() === 'warning') diagnostics.push(`${message.type()}: ${message.text()}`);
    });
    page.on('pageerror', (error) => diagnostics.push(`pageerror: ${error.message}`));
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: timeoutMs });
    await page.waitForFunction(() => window.__groundCoverPeriodicBake?.ready === true, undefined, { timeout: timeoutMs });
    for (const profileId of profileIds) {
      const fixture = makeEstonianLowCoverFixture(profileId);
      const slices = directions.map((direction) => makePeriodicSlice(fixture.mesh, fixture.tile, direction));
      const request = {
        mesh: fixture.mesh,
        tile: fixture.tile,
        slices,
        tileWidth: tilePixels,
        tileHeight: tilePixels,
        atlasColumns: ATLAS_COLUMNS,
        atlasRows: ATLAS_ROWS,
      };
      const first = await page.evaluate(async (bakeRequest) => window.__groundCoverPeriodicBake.run(bakeRequest), request);
      if (diagnostics.length > 0) throw new Error(`browser diagnostics after ${fixture.species}: ${diagnostics.join(' | ')}`);
      const validation = validateBotanicalBake(first, fixture.mesh, 32);
      const accepted =
        validation.comparedHits >= 1 &&
        validation.hitMismatch === 0 &&
        validation.rmsDepthTError <= 2e-5 &&
        validation.maxDepthTError <= 2e-4 &&
        validation.meanNormalDot >= 0.999 &&
        validation.minNormalDot >= 0.99;
      if (!accepted) {
        mkdirSync(outputRoot, { recursive: true });
        const mismatchPath = resolve(outputRoot, `${PROFILE_SLUGS[profileId]}-diagnostic-failure.json`);
        const mismatches = diagnosePeriodicGpuMismatches(first, fixture.mesh, 32);
        writeFileSync(mismatchPath, `${JSON.stringify({ fixture: fixture.species, validation, mismatches }, null, 2)}\n`);
        throw new Error(`${fixture.species} CPU/raster acceptance failed: ${JSON.stringify(validation)}; ${mismatchPath}`);
      }
      const second = await page.evaluate(async (bakeRequest) => window.__groundCoverPeriodicBake.run(bakeRequest), request);
      if (diagnostics.length > 0) throw new Error(`browser diagnostics after repeat ${fixture.species}: ${diagnostics.join(' | ')}`);
      const artifact = await writeArtifact(outputRoot, fixture, first, second, validation);
      const summary = {
        profileId,
        species: fixture.species,
        adapter: first.adapter,
        vertices: fixture.mesh.positions.length / 3,
        triangles: fixture.mesh.indices.length / 3,
        morphology: fixture.morphology,
        validation,
        ...artifact,
      };
      console.log(`[groundcover-lowcover-bake] PASS ${fixture.species}`);
      console.log(`  mesh: ${summary.vertices} vertices, ${summary.triangles} triangles`);
      console.log(`  validation: ${JSON.stringify(validation)}`);
      console.log(`  deterministic sha256: ${artifact.sha256}`);
      console.log(`  binary: ${artifact.binaryPath}`);
      console.log(`  QA index: ${artifact.indexPath}`);
    }
    const combined = writeAcceptedSummary(outputRoot);
    console.log(`[groundcover-lowcover-bake] ${profileIds.length} profile(s) accepted; collected ${combined.count}: ${combined.path}`);
  } finally {
    cleanup();
  }
}

main().catch((error: unknown) => {
  console.error('[groundcover-lowcover-bake] FAIL', error instanceof Error ? error.stack ?? error.message : error);
  process.exit(1);
});
