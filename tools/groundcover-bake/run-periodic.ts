import { spawn, type ChildProcess } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import sharp from 'sharp';
import { chromium, type Browser } from 'playwright';
import { decodeOct, makeDirection, PROFILE_TEXEL_BYTES } from './ProfileFormat';
import {
  makeSphagnumCapillifoliumFixture,
  SPHAGNUM_CAPILLIFOLIUM_PROFILE_ID,
  SPHAGNUM_CAPILLIFOLIUM_SPECIES,
} from './SphagnumCapillifolium';
import {
  diagnosePeriodicGpuMismatches,
  makeCushionCarpetFixture,
  makePeriodicSlice,
  packPeriodicProfile,
  parsePeriodicHeader,
  periodicAddress,
  validatePeriodicBake,
  type PeriodicBakeResult,
} from './PeriodicProfile';

interface Args {
  [key: string]: string | boolean;
}

function parseArgs(argv: string[]): Args {
  const result: Args = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i] as string;
    if (!arg.startsWith('--')) continue;
    const value = argv[i + 1];
    if (value !== undefined && !value.startsWith('--')) {
      result[arg.slice(2)] = value;
      i++;
    } else {
      result[arg.slice(2)] = true;
    }
  }
  return result;
}

function stringArg(value: string | boolean | undefined): string | undefined {
  return typeof value === 'string' ? value : undefined;
}

async function waitForServer(url: string, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {
      // Local server is still starting.
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
      const d = Math.max(0, Math.min(255, Math.round((1 - (result.pixels[source] as number)) * 255)));
      output.set([d, d, d, 255], target);
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

function addressTest(tile: PeriodicBakeResult['tile']): { samples: number; maxUError: number; maxVError: number } {
  let samples = 0;
  let maxUError = 0;
  let maxVError = 0;
  const points = [{ x: -1.37, z: 2.81 }, { x: 0.137, z: 0.819 }, { x: 1.003, z: -0.011 }];
  for (const point of points) {
    const base = periodicAddress(point.x, point.z, tile);
    for (let ix = -3; ix <= 3; ix++) {
      for (let iz = -3; iz <= 3; iz++) {
        const moved = periodicAddress(point.x + ix * tile.sizeX, point.z + iz * tile.sizeZ, tile);
        maxUError = Math.max(maxUError, Math.abs(base.u - moved.u));
        maxVError = Math.max(maxVError, Math.abs(base.v - moved.v));
        samples++;
      }
    }
  }
  return { samples, maxUError, maxVError };
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const port = Number(stringArg(args.port) ?? '5228');
  const timeoutMs = Number(stringArg(args.timeout) ?? '300000');
  const fixtureName = stringArg(args.fixture) ?? 'geometry';
  if (fixtureName !== 'geometry' && fixtureName !== 'sphagnum-capillifolium') {
    throw new Error(`unknown --fixture ${fixtureName}; expected geometry or sphagnum-capillifolium`);
  }
  const speciesBake = fixtureName === 'sphagnum-capillifolium';
  const tilePixels = Number(stringArg(args.tile) ?? (speciesBake ? '64' : '96'));
  const outputRoot = resolve(stringArg(args.out) ?? '/tmp/groundcover-gpu-bake-periodic');
  const repoRoot = resolve(import.meta.dirname, '../..');
  const fixture = speciesBake ? makeSphagnumCapillifoliumFixture() : makeCushionCarpetFixture();
  const azimuths = speciesBake
    ? Array.from({ length: 16 }, (_value, index) => index * 22.5)
    : [0, 90, 180, 270];
  const elevations = speciesBake ? [15, 35, 55, 75] : [20, 55];
  // Stable azimuth-major ordering: every elevation for azimuth 0, then every
  // elevation for azimuth 22.5, and so on. Runtime lattice indexing can use
  // slice = azimuthIndex * elevationCount + elevationIndex.
  const directionSpecs = azimuths.flatMap((azimuthDeg) =>
    elevations.map((elevationDeg) => ({ azimuthDeg, elevationDeg })));
  const directions = directionSpecs.map(({ azimuthDeg, elevationDeg }) => makeDirection(azimuthDeg, elevationDeg));
  const slices = directions.map((direction) => makePeriodicSlice(fixture.mesh, fixture.tile, direction));
  const atlasColumns = speciesBake ? 8 : 4;
  const atlasRows = speciesBake ? 8 : 2;
  const validationStep = speciesBake ? 32 : 14;
  const profileId = speciesBake ? SPHAGNUM_CAPILLIFOLIUM_PROFILE_ID : 1001;
  const request = {
    mesh: fixture.mesh,
    tile: fixture.tile,
    slices,
    tileWidth: tilePixels,
    tileHeight: tilePixels,
    atlasColumns,
    atlasRows,
  };

  const vite: ChildProcess = spawn(
    'npx',
    ['vite', '--port', String(port), '--strictPort', '--host', 'localhost'],
    { cwd: repoRoot, stdio: ['ignore', 'pipe', 'pipe'] },
  );
  vite.stdout?.on('data', (chunk: Buffer) => {
    const text = chunk.toString();
    if (/error/i.test(text)) process.stdout.write(`[vite] ${text}`);
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
    const first = await page.evaluate(async (bakeRequest) => window.__groundCoverPeriodicBake.run(bakeRequest), request);
    if (diagnostics.length > 0) throw new Error(`browser diagnostics: ${diagnostics.join(' | ')}`);

    const validation = validatePeriodicBake(first, fixture.mesh, validationStep);
    const accepted =
      validation.comparedHits >= 150 &&
      validation.hitMismatch === 0 &&
      validation.rmsDepthTError <= 2e-5 &&
      validation.maxDepthTError <= 2e-4 &&
      validation.meanNormalDot >= 0.999 &&
      validation.minNormalDot >= 0.99;
    // The exhaustive mismatch expansion scans every submitted primitive again.
    // Keep it off the accepted path; pay for it only when preserving a failure.
    const mismatchDetails = accepted ? [] : diagnosePeriodicGpuMismatches(first, fixture.mesh, validationStep);
    mkdirSync(outputRoot, { recursive: true });
    const diagnosticPath = resolve(outputRoot, 'periodic-diagnostic-latest.json');
    writeFileSync(diagnosticPath, `${JSON.stringify({
      schema: 'laas-groundcover-periodic-diagnostic/v2',
      fixture: fixtureName,
      accepted,
      adapter: first.adapter,
      validation,
      mismatchCount: mismatchDetails.length,
      mismatches: mismatchDetails,
      drawOrder: 'slice-major, then slice.copies order; drawIndex is the 256-byte uniform slot',
      slices: first.slices.map((slice, index) => ({ index, direction: slice.direction, depthMin: slice.depthMin, depthMax: slice.depthMax, copies: slice.copies })),
    }, null, 2)}\n`);
    if (!accepted) {
      throw new Error(`periodic CPU triangle validation failed: ${JSON.stringify(validation)}; exact capture: ${diagnosticPath}`);
    }
    const address = addressTest(first.tile);
    if (address.maxUError > 2e-15 || address.maxVError > 2e-15) {
      throw new Error(`periodic address test failed: ${JSON.stringify(address)}`);
    }

    // Only spend the second bake after the first passes the unchanged strict gate.
    const second = await page.evaluate(async (bakeRequest) => window.__groundCoverPeriodicBake.run(bakeRequest), request);
    if (diagnostics.length > 0) throw new Error(`browser diagnostics after deterministic repeat: ${diagnostics.join(' | ')}`);
    const packed = packPeriodicProfile(first, profileId, 1);
    const packedAgain = packPeriodicProfile(second, profileId, 1);
    if (packed.sha256 !== packedAgain.sha256) {
      throw new Error(`non-deterministic periodic result: ${packed.sha256} != ${packedAgain.sha256}`);
    }
    const header = parsePeriodicHeader(packed.bytes);

    const artifactDir = resolve(outputRoot, packed.sha256.slice(0, 16));
    const qaDir = resolve(artifactDir, 'qa');
    mkdirSync(qaDir, { recursive: true });
    const binaryName = speciesBake
      ? 'sphagnum-capillifolium-periodic-profile.gcrp'
      : 'multi-cushion-periodic-profile.gcrp';
    const binaryPath = resolve(artifactDir, binaryName);
    writeFileSync(binaryPath, packed.bytes);
    const width = first.tileWidth * first.atlasColumns;
    const height = first.tileHeight * first.atlasRows;
    const depthPath = resolve(qaDir, '01-periodic-first-hit-depth.png');
    const normalPath = resolve(qaDir, '02-periodic-first-hit-normal.png');
    await sharp(diagnosticPixels(first, 'depth'), { raw: { width, height, channels: 4 } }).png().toFile(depthPath);
    await sharp(diagnosticPixels(first, 'normal'), { raw: { width, height, channels: 4 } }).png().toFile(normalPath);

    const fixtureRecipe = 'capitula' in fixture
      ? {
          fixture: fixture.generator,
          species: SPHAGNUM_CAPILLIFOLIUM_SPECIES,
          identity: {
            functionalCover: 'Moss',
            functionalCoverId: 1,
            speciesProfileId: SPHAGNUM_CAPILLIFOLIUM_PROFILE_ID,
            namespaceRule: 'functional cover IDs classify behavior; zero-based profile IDs select botanical geometry and are independent',
          },
          carpet: {
            topology: 'one connected periodic indexed height-field surface',
            vertices: fixture.carpetVertexCount,
            triangles: fixture.carpetTriangleCount,
          },
          capitula: fixture.capitula,
          morphologyRecipe: {
            distribution: 'deterministic toroidal best-candidate placement; non-grid',
            form: 'dense convex star-like capitula assembled from explicit radial flattened branches, forked branchlets, central cores, and stems above an irregular connected hummock carpet',
            negativeSpace: 'inter-branch and inter-head openings remain geometry, not a smooth cap union',
            colorCarrier: 'palettePhase is recorded per capitulum for later red/green material variation; GCRP/v2 contains geometry depth and normals, not calibrated color',
          },
        }
      : {
          fixture: 'geometric-validation-only-overlapping-rounded-cushion-carpet',
          botanicalClaim: 'none; this is not a Sphagnum species model',
          cushions: fixture.cushions,
        };
    const recipe = {
      ...fixtureRecipe,
      tile: fixture.tile,
      mesh: {
        vertices: fixture.mesh.positions.length / 3,
        triangles: fixture.mesh.indices.length / 3,
        indexedGeometrySha256: hashJson(fixture.mesh),
      },
      directionLattice: {
        ordering: 'azimuth-major/elevation-minor: sliceIndex = azimuthIndex * elevationCount + elevationIndex',
        azimuthDegrees: azimuths,
        elevationDegrees: elevations,
        slices: slices.map((slice, index) => ({
        index,
        azimuthIndex: Math.floor(index / elevations.length),
        elevationIndex: index % elevations.length,
        azimuthDeg: directionSpecs[index]!.azimuthDeg,
        elevationDeg: directionSpecs[index]!.elevationDeg,
        direction: slice.direction,
        depthMin: slice.depthMin,
        depthMax: slice.depthMax,
        copyRange: slice.copyRange,
        copyCount: slice.copies.length,
        conservativeProjectedBounds: slice.conservativeProjectedBounds,
        })),
      },
      interiorTilePixels: tilePixels,
      gutter: 1,
      atlas: { columns: atlasColumns, rows: atlasRows },
    };
    const storedAtlasWidth = packed.storedTileWidth * first.atlasColumns;
    const storedAtlasHeight = packed.storedTileHeight * first.atlasRows;
    const index = {
      schema: 'laas-groundcover-gpu-bake-qa/v2-periodic-top-xz',
      provenance: speciesBake
        ? {
            authoredAsset: 'original deterministic procedural indexed geometry; no external mesh, texture, or copyrighted geometry was ingested',
            morphologyBasis: 'accepted S. capillifolium authoring brief: connected carpet, irregular rounded hummocks, dense convex star-like capitula, branching texture, and dark inter-head crevices',
            explicitRejection: 'not a union of smooth ellipsoid caps',
            claimBoundary: 'S. capillifolium vegetative carpet/hummock/capitulum/branch geometry only; no reproductive-stage, chemistry, abundance, geographic occurrence, material color calibration, or complete ecology claim',
          }
        : {
            authoredAsset: 'original deterministic procedural validation geometry',
            claimBoundary: 'geometry/raster validation only; no botanical claim',
          },
      recipeSha256: hashJson(recipe),
      binary: {
        file: `../${binaryName}`,
        sha256: packed.sha256,
        bytes: packed.bytes.byteLength,
        profileId: header.profileId,
        format: 'GCRP/v2 periodic-top-XZ, little-endian, RGBA16 payload',
        projection: 'rayOriginXZ = p.xz - d.xz * ((p.y - topH) / d.y); depth reconstructs t = depthMin + encoded * (depthMax-depthMin)',
        texel: ['normalized ray parameter t (65535 means miss)', 'ray-facing oct-normal x', 'ray-facing oct-normal y', 'hit mask'],
        interiorTile: `${header.interiorTileWidth}x${header.interiorTileHeight}`,
        storedTile: `${header.storedTileWidth}x${header.storedTileHeight}`,
        gutterPlan: 'one wrapped texel copied from the opposite canonical edge on all four sides and corners; atlas UV clamps within each stored slice',
      },
      diagnosticImages: [
        { number: 1, file: '01-periodic-first-hit-depth.png', sha256: hashFile(depthPath), width, height, interpretation: `${slices.length} canonical top-plane tiles in the documented azimuth-major lattice; white is nearer t, dark purple is miss` },
        { number: 2, file: '02-periodic-first-hit-normal.png', sha256: hashFile(normalPath), width, height, interpretation: 'ray-facing decoded world normal; every direction addresses the same square XZ tile' },
      ],
      recipe,
      adapter: first.adapter,
      deterministicRepeat: { runs: 2, identicalSha256: true },
      selectedPixelCpuRasterAcceptance: {
        scope: 'selected canonical-tile pixel centers whose ideal Moller-Trumbore nearest hit has barycentric edge margin >= 0.03; expected owner/depth/normal then follows the submitted-f32 8-bit-subpixel fixed-function raster election over every mathematically required periodic copy',
        requirements: { comparedHitsAtLeast: 150, hitMismatch: 0, rmsDepthTErrorAtMost: 2e-5, maxDepthTErrorAtMost: 2e-4, meanNormalDotAtLeast: 0.999, minNormalDotAtLeast: 0.99 },
        observed: validation,
      },
      periodicAddressAcceptance: {
        rule: 'address(x + integer*tileSize) == address(x)',
        requirements: { maxUError: 2e-15, maxVError: 2e-15 },
        observed: address,
      },
      runtimeResourceProjection: {
        artifactAtlas: `${storedAtlasWidth}x${storedAtlasHeight} rgba16unorm (${packed.storedTileWidth}x${packed.storedTileHeight} stored slice, ${first.tileWidth}x${first.tileHeight} interior)`,
        payloadBytesPerProfile: storedAtlasWidth * storedAtlasHeight * PROFILE_TEXEL_BYTES,
        productionExample: '64x64 interior + one wrapped gutter => 66x66 x 16 azimuth x 4 elevation x 8 B = 2.13 MiB/profile; 12 profiles = 25.52 MiB before mips',
        sampling: 'four fixed direction-corner taps for one profile; bounded A/B cover mixture is at most eight fixed taps',
        bindings: 'one filterable 2D-array/3D atlas plus one sampler for every profile; offline periodic copies add zero runtime bindings or taps',
      },
    };
    const indexPath = resolve(qaDir, 'index.json');
    writeFileSync(indexPath, `${JSON.stringify(index, null, 2)}\n`);
    console.log('[groundcover-periodic-bake] PASS');
    console.log(`  adapter: ${first.adapter}`);
    console.log(`  fixture: ${fixtureName}, profileId: ${profileId}`);
    console.log(`  mesh: ${fixture.mesh.positions.length / 3} vertices, ${fixture.mesh.indices.length / 3} triangles${'capitula' in fixture ? `, ${fixture.capitula.length} capitula` : `, ${fixture.cushions.length} cushions`}`);
    console.log(`  direction lattice: ${azimuths.length} azimuth x ${elevations.length} elevation, azimuth-major`);
    console.log(`  copies/slice: ${slices.map((slice) => slice.copies.length).join(', ')}`);
    console.log(`  stored atlas: ${storedAtlasWidth}x${storedAtlasHeight}, ${packed.bytes.byteLength} bytes packed`);
    console.log(`  CPU triangle validation: ${JSON.stringify(validation)}`);
    console.log(`  periodic address: ${JSON.stringify(address)}`);
    console.log(`  deterministic sha256: ${packed.sha256}`);
    console.log(`  binary: ${binaryPath}`);
    console.log(`  QA index: ${indexPath}`);
  } finally {
    cleanup();
  }
}

main().catch((error: unknown) => {
  console.error('[groundcover-periodic-bake] FAIL', error instanceof Error ? error.stack ?? error.message : error);
  process.exit(1);
});
