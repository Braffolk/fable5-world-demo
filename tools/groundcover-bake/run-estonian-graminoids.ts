import { spawn, type ChildProcess } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { relative, resolve } from 'node:path';
import sharp from 'sharp';
import { chromium, type Browser } from 'playwright';
import {
  makeAllEstonianGraminoidFixtures,
  makeEstonianGraminoidFixture,
  type EstonianGraminoidFixture,
  type EstonianGraminoidProfileId,
} from './EstonianGraminoids';
import { decodeOct, makeDirection, PROFILE_TEXEL_BYTES } from './ProfileFormat';
import {
  diagnosePeriodicGpuMismatches,
  makePeriodicSlice,
  packPeriodicOwnedProfile,
  packPeriodicProfile,
  parsePeriodicHeader,
  periodicAddress,
  validatePeriodicBake,
  type PeriodicBakeResult,
} from './PeriodicProfile';

interface Args {
  [key: string]: string | boolean;
}

interface ArtifactSummary {
  profileId: number;
  species: string;
  adapter: string;
  sha256: string;
  binary: string;
  qaIndex: string;
  meshVertices: number;
  meshTriangles: number;
  validation: ReturnType<typeof validatePeriodicBake> | null;
}

function collectArtifactSummaries(outputRoot: string): ArtifactSummary[] {
  const byProfile = new Map<number, ArtifactSummary>();
  if (!existsSync(outputRoot)) return [];
  for (const profileEntry of readdirSync(outputRoot, { withFileTypes: true })) {
    if (!profileEntry.isDirectory() || !/^\d+-/.test(profileEntry.name)) continue;
    const profileDir = resolve(outputRoot, profileEntry.name);
    for (const hashEntry of readdirSync(profileDir, { withFileTypes: true })) {
      if (!hashEntry.isDirectory()) continue;
      const artifactDir = resolve(profileDir, hashEntry.name);
      const qaIndex = resolve(artifactDir, 'qa/index.json');
      if (!existsSync(qaIndex)) continue;
      const qa = JSON.parse(readFileSync(qaIndex, 'utf8')) as {
        adapter: string;
        binary: { file: string; profileId: number; sha256: string };
        recipe: { identity: { species: string }; mesh: { vertices: number; triangles: number } };
        selectedPixelCpuRasterAcceptance: { observed: ReturnType<typeof validatePeriodicBake> | null };
      };
      byProfile.set(qa.binary.profileId, {
        profileId: qa.binary.profileId,
        species: qa.recipe.identity.species,
        adapter: qa.adapter,
        sha256: qa.binary.sha256,
        binary: resolve(artifactDir, 'qa', qa.binary.file),
        qaIndex,
        meshVertices: qa.recipe.mesh.vertices,
        meshTriangles: qa.recipe.mesh.triangles,
        validation: qa.selectedPixelCpuRasterAcceptance.observed,
      });
    }
  }
  return [...byProfile.values()].sort((left, right) => left.profileId - right.profileId);
}

function writeSetIndex(outputRoot: string, summaries: ArtifactSummary[]): string {
  const summaryPath = resolve(outputRoot, 'index.json');
  writeFileSync(summaryPath, `${JSON.stringify({
    schema: 'laas-groundcover-estonian-graminoid-bake-set/v1',
    generatedAt: new Date().toISOString(),
    lattice: { azimuths: 16, elevations: [15, 35, 55, 75], ordering: 'azimuth-major/elevation-minor' },
    deterministicRunsPerProfile: 2,
    artifacts: summaries,
  }, null, 2)}\n`);
  return summaryPath;
}

function parseArgs(argv: string[]): Args {
  const result: Args = {};
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index] as string;
    if (!arg.startsWith('--')) continue;
    const value = argv[index + 1];
    if (value !== undefined && !value.startsWith('--')) {
      result[arg.slice(2)] = value;
      index++;
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
      // Local Vite is still starting.
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

function writeMeshTransport(
  fixture: EstonianGraminoidFixture,
  repoRoot: string,
  meshSha256: string,
): { positionsUrl: string; normalsUrl: string; colorsUrl: string; indicesUrl: string } {
  const root = resolve(repoRoot, 'data/work/groundcover-bake-transport', meshSha256);
  mkdirSync(root, { recursive: true });
  const positionsPath = resolve(root, 'positions.f32');
  const normalsPath = resolve(root, 'normals.f32');
  const colorsPath = resolve(root, 'colors.f32');
  const indicesPath = resolve(root, 'indices.u32');
  const positions = new Float32Array(fixture.mesh.positions);
  const normals = new Float32Array(fixture.mesh.normals);
  const colors = new Float32Array(fixture.mesh.colors ?? fixture.mesh.positions.map(() => 1));
  const indices = new Uint32Array(fixture.mesh.indices);
  writeFileSync(positionsPath, Buffer.from(positions.buffer, positions.byteOffset, positions.byteLength));
  writeFileSync(normalsPath, Buffer.from(normals.buffer, normals.byteOffset, normals.byteLength));
  writeFileSync(colorsPath, Buffer.from(colors.buffer, colors.byteOffset, colors.byteLength));
  writeFileSync(indicesPath, Buffer.from(indices.buffer, indices.byteOffset, indices.byteLength));
  const url = (path: string): string => `/${relative(repoRoot, path).replaceAll('\\', '/')}`;
  return {
    positionsUrl: url(positionsPath),
    normalsUrl: url(normalsPath),
    colorsUrl: url(colorsPath),
    indicesUrl: url(indicesPath),
  };
}

function slug(species: string): string {
  return species.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
}

function diagnosticPixels(result: PeriodicBakeResult, mode: 'depth' | 'normal'): Uint8Array {
  const width = result.tileWidth * result.atlasColumns;
  const height = result.tileHeight * result.atlasRows;
  const output = new Uint8Array(width * height * 4);
  for (let texel = 0; texel < width * height; texel++) {
    const source = texel * 4;
    const hit = (result.pixels[source + 3] as number) > 0.5;
    if (!hit) {
      output.set([18, 8, 22, 255], source);
    } else if (mode === 'depth') {
      const depth = Math.max(0, Math.min(255, Math.round((1 - (result.pixels[source] as number)) * 255)));
      output.set([depth, depth, depth, 255], source);
    } else {
      const normal = decodeOct(result.pixels[source + 1] as number, result.pixels[source + 2] as number);
      output.set([
        Math.round((normal.x * 0.5 + 0.5) * 255),
        Math.round((normal.y * 0.5 + 0.5) * 255),
        Math.round((normal.z * 0.5 + 0.5) * 255),
        255,
      ], source);
    }
  }
  return output;
}

async function runStoredBake(
  page: import('playwright').Page,
  request: Parameters<Window['__groundCoverPeriodicBake']['runStored']>[0],
): Promise<PeriodicBakeResult> {
  const summary = await page.evaluate(
    async (bakeRequest) => window.__groundCoverPeriodicBake.runStored(bakeRequest),
    request,
  );
  const pixels = new Float32Array(summary.pixelLength);
  const correspondencePixels: Float32Array | undefined = undefined;
  const ownerPixels = summary.ownerLength > 0 ? new Uint32Array(summary.ownerLength) : undefined;
  const chunkSize = 131_072;
  const fill = async (
    kind: 'pixels' | 'correspondencePixels' | 'ownerPixels',
    target: Float32Array | Uint32Array | undefined,
  ): Promise<void> => {
    if (!target) return;
    for (let offset = 0; offset < target.length; offset += chunkSize) {
      const chunk = await page.evaluate(
        ([storedKind, storedOffset, storedCount]) => window.__groundCoverPeriodicBake.readStored(
          storedKind,
          storedOffset,
          storedCount,
        ),
        [kind, offset, Math.min(chunkSize, target.length - offset)] as const,
      );
      target.set(chunk, offset);
    }
  };
  await fill('pixels', pixels);
  // v4 packaging consumes owner + geometry. Correspondence remains a browser-
  // local diagnostic and is intentionally not expanded through Playwright.
  await fill('ownerPixels', ownerPixels);
  await page.evaluate(() => window.__groundCoverPeriodicBake.clearStored());
  const {
    pixelLength: _pixelLength,
    correspondenceLength: _correspondenceLength,
    ownerLength: _ownerLength,
    ...metadata
  } = summary;
  return { ...metadata, pixels, correspondencePixels, ownerPixels };
}

function addressTest(tile: PeriodicBakeResult['tile']): { samples: number; maxUError: number; maxVError: number } {
  let samples = 0;
  let maxUError = 0;
  let maxVError = 0;
  for (const point of [{ x: -1.37, z: 2.81 }, { x: 0.137, z: 0.219 }, { x: 1.003, z: -0.011 }]) {
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

async function bakeFixture(
  page: Awaited<ReturnType<Browser['newPage']>>,
  fixture: EstonianGraminoidFixture,
  outputRoot: string,
  tilePixels: number,
  repoRoot: string,
  runCpuOracle: boolean,
): Promise<ArtifactSummary> {
  const azimuths = Array.from({ length: 16 }, (_value, index) => index * 22.5);
  const elevations = [15, 35, 55, 75];
  const directionSpecs = azimuths.flatMap((azimuthDeg) => elevations.map((elevationDeg) => ({ azimuthDeg, elevationDeg })));
  const slices = directionSpecs.map(({ azimuthDeg, elevationDeg }) =>
    makePeriodicSlice(fixture.mesh, fixture.tile, makeDirection(azimuthDeg, elevationDeg)));
  const meshSha256 = hashJson(fixture.mesh);
  const request = {
    ...writeMeshTransport(fixture, repoRoot, meshSha256),
    tile: fixture.tile,
    slices,
    tileWidth: tilePixels,
    tileHeight: tilePixels,
    atlasColumns: 8,
    atlasRows: 8,
  };
  const first = await runStoredBake(page, request);
  if (!/apple/i.test(first.adapter)) {
    throw new Error(`expected localhost Apple WebGPU adapter, got ${first.adapter}`);
  }
  // Avenella deliberately carries the sparsest, filiform silhouette. Sample it
  // more densely instead of weakening the minimum-hit evidence requirement.
  const validationStep = fixture.profileId === 1 ? 24 : Math.max(32, Math.round(tilePixels / 2));
  const requiredComparedHits = fixture.profileId === 1 ? 48 : 36;
  const validation = runCpuOracle ? validatePeriodicBake(first, fixture.mesh, validationStep) : null;
  const accepted = validation === null || (
    validation.comparedHits >= requiredComparedHits &&
    validation.hitMismatch === 0 &&
    validation.rmsDepthTError <= 2e-5 &&
    validation.maxDepthTError <= 2e-4 &&
    validation.meanNormalDot >= 0.999 &&
    validation.minNormalDot >= 0.99
  );
  if (!accepted) {
    const failureDir = resolve(outputRoot, `${fixture.profileId}-${slug(fixture.species)}`);
    mkdirSync(failureDir, { recursive: true });
    const mismatchPath = resolve(failureDir, 'gpu-mismatch.json');
    writeFileSync(mismatchPath, `${JSON.stringify({
      fixture: fixture.species,
      adapter: first.adapter,
      validation,
      mismatches: diagnosePeriodicGpuMismatches(first, fixture.mesh, validationStep),
    }, null, 2)}\n`);
    throw new Error(`${fixture.species} periodic CPU/raster gate failed: ${JSON.stringify(validation)}; ${mismatchPath}`);
  }
  const second = await runStoredBake(page, request);
  const correspondenceCarrier = fixture.profileId === 2 && tilePixels >= 256;
  const packed = correspondenceCarrier
    ? packPeriodicOwnedProfile(first, fixture.mesh, fixture.profileId, 1)
    : packPeriodicProfile(first, fixture.profileId, 1);
  const packedAgain = correspondenceCarrier
    ? packPeriodicOwnedProfile(second, fixture.mesh, fixture.profileId, 1)
    : packPeriodicProfile(second, fixture.profileId, 1);
  if (packed.sha256 !== packedAgain.sha256) {
    throw new Error(`${fixture.species} non-deterministic repeat: ${packed.sha256} != ${packedAgain.sha256}`);
  }
  const header = parsePeriodicHeader(packed.bytes);
  const address = addressTest(first.tile);
  if (address.maxUError > 2e-15 || address.maxVError > 2e-15) {
    throw new Error(`${fixture.species} periodic address failed: ${JSON.stringify(address)}`);
  }

  const speciesSlug = slug(fixture.species);
  const artifactDir = resolve(outputRoot, `${fixture.profileId}-${speciesSlug}`, packed.sha256.slice(0, 16));
  const qaDir = resolve(artifactDir, 'qa');
  mkdirSync(qaDir, { recursive: true });
  const binaryName = `${speciesSlug}-periodic-profile.gcrp`;
  const binaryPath = resolve(artifactDir, binaryName);
  writeFileSync(binaryPath, packed.bytes);
  const width = first.tileWidth * first.atlasColumns;
  const height = first.tileHeight * first.atlasRows;
  const depthPath = resolve(qaDir, '01-periodic-first-hit-depth.png');
  const normalPath = resolve(qaDir, '02-periodic-first-hit-normal.png');
  await sharp(diagnosticPixels(first, 'depth'), { raw: { width, height, channels: 4 } }).png().toFile(depthPath);
  await sharp(diagnosticPixels(first, 'normal'), { raw: { width, height, channels: 4 } }).png().toFile(normalPath);

  const recipe = {
    fixture: fixture.generator,
    identity: { profileId: fixture.profileId, species: fixture.species, family: fixture.family },
    tile: fixture.tile,
    mesh: {
      vertices: fixture.mesh.positions.length / 3,
      triangles: fixture.mesh.indices.length / 3,
      indexedGeometrySha256: meshSha256,
    },
    structure: fixture.structure,
    tuftCenters: fixture.tuftCenters,
    sourceKeys: fixture.sourceKeys,
    directionLattice: {
      ordering: 'azimuth-major/elevation-minor: sliceIndex = azimuthIndex * 4 + elevationIndex',
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
      })),
    },
    interiorTilePixels: tilePixels,
    gutter: 1,
    atlas: { columns: 8, rows: 8 },
  };
  const index = {
    schema: 'laas-groundcover-gpu-bake-qa/v2-periodic-estonian-graminoid',
    provenance: {
      authoredAsset: 'original deterministic procedural indexed geometry; no external mesh, image, or texture was ingested',
      sourceLedger: 'tools/groundcover-bake/ESTONIAN-GRAMINOIDS-README.md',
      claimBoundary: fixture.claimBoundary,
      explicitRejection: 'not an ellipsoid/cushion/cap primitive and not a renamed copy of another profile',
    },
    recipeSha256: hashJson(recipe),
    binary: {
      file: `../${binaryName}`,
      sha256: packed.sha256,
      bytes: packed.bytes.byteLength,
      profileId: header.profileId,
      format: 'GCRP/v2 periodic-top-XZ, little-endian, RGBA16 payload',
      projection: 'rayOriginXZ = p.xz - d.xz * ((p.y - topH) / d.y)',
      interiorTile: `${header.interiorTileWidth}x${header.interiorTileHeight}`,
      storedTile: `${header.storedTileWidth}x${header.storedTileHeight}`,
      gutterPlan: 'one exact wrapped texel from the opposite canonical edge on every side and corner',
    },
    diagnosticImages: [
      { number: 1, file: '01-periodic-first-hit-depth.png', sha256: hashFile(depthPath), width, height, interpretation: '64 canonical top-XZ views; white is nearer ray t, dark purple is miss' },
      { number: 2, file: '02-periodic-first-hit-normal.png', sha256: hashFile(normalPath), width, height, interpretation: 'ray-facing decoded world normal for the same 64 views' },
    ],
    recipe,
    adapter: first.adapter,
    execution: 'localhost Chromium WebGPU on Apple Metal; no remote service',
    deterministicRepeat: { runs: 2, identicalSha256: true },
    selectedPixelCpuRasterAcceptance: {
      scope: 'selected canonical pixel centers with strict ideal-ray triangle interiors; expected first owner follows submitted-f32 8-bit-subpixel raster transcription over all required periodic copies',
      sampleStepPixels: validationStep,
      requirements: { comparedHitsAtLeast: requiredComparedHits, hitMismatch: 0, rmsDepthTErrorAtMost: 2e-5, maxDepthTErrorAtMost: 2e-4, meanNormalDotAtLeast: 0.999, minNormalDotAtLeast: 0.99 },
      observed: validation,
      ...(validation === null ? {
        skipped: 'explicit --skip-cpu-oracle for a multi-million-triangle authoring mesh; unchanged GPU baker is covered by focused oracle tests and this artifact retains two byte-identical Apple-Metal submissions',
      } : {}),
    },
    periodicAddressAcceptance: {
      rule: 'address(x + integer*tileSize) == address(x)',
      requirements: { maxUError: 2e-15, maxVError: 2e-15 },
      observed: address,
    },
    runtimeResourceProjection: {
      payloadBytes: header.storedTileWidth * header.storedTileHeight * 64 * PROFILE_TEXEL_BYTES,
      sampling: 'four fixed direction-corner taps for one profile; no runtime ray march and no data-dependent shader loop',
    },
  };
  const indexPath = resolve(qaDir, 'index.json');
  writeFileSync(indexPath, `${JSON.stringify(index, null, 2)}\n`);
  return {
    profileId: fixture.profileId,
    species: fixture.species,
    adapter: first.adapter,
    sha256: packed.sha256,
    binary: binaryPath,
    qaIndex: indexPath,
    meshVertices: fixture.mesh.positions.length / 3,
    meshTriangles: fixture.mesh.indices.length / 3,
    validation,
  };
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const port = Number(stringArg(args.port) ?? '5231');
  const timeoutMs = Number(stringArg(args.timeout) ?? '300000');
  const tilePixels = Number(stringArg(args.tile) ?? '64');
  const outputRoot = resolve(stringArg(args.out) ?? '/tmp/groundcover-gpu-bake-estonian-graminoids');
  const runCpuOracle = args['skip-cpu-oracle'] !== true;
  const selectedProfile = stringArg(args.profile);
  const fixtures = selectedProfile === undefined
    ? makeAllEstonianGraminoidFixtures()
    : [makeEstonianGraminoidFixture(Number(selectedProfile) as EstonianGraminoidProfileId)];
  mkdirSync(outputRoot, { recursive: true });
  if (args['index-only'] === true) {
    const discovered = collectArtifactSummaries(outputRoot);
    if (discovered.length === 0) throw new Error(`no accepted graminoid artifacts found under ${outputRoot}`);
    const summaryPath = writeSetIndex(outputRoot, discovered);
    console.log(`[graminoid-bake] indexed ${discovered.length} accepted profiles: ${summaryPath}`);
    return;
  }
  const repoRoot = resolve(import.meta.dirname, '../..');
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
    const summaries: ArtifactSummary[] = [];
    for (const fixture of fixtures) {
      const before = diagnostics.length;
      const summary = await bakeFixture(page, fixture, outputRoot, tilePixels, repoRoot, runCpuOracle);
      if (diagnostics.length !== before) throw new Error(`browser/WebGPU diagnostics for ${fixture.species}: ${diagnostics.slice(before).join(' | ')}`);
      summaries.push(summary);
      console.log(`[graminoid-bake] PASS ${fixture.profileId} ${fixture.species}`);
      console.log(`  adapter: ${summary.adapter}`);
      console.log(`  mesh: ${summary.meshVertices} vertices, ${summary.meshTriangles} triangles`);
      console.log(`  validation: ${JSON.stringify(summary.validation)}`);
      console.log(`  deterministic sha256: ${summary.sha256}`);
      console.log(`  binary: ${summary.binary}`);
      console.log(`  QA index: ${summary.qaIndex}`);
    }
    const indexedSummaries = selectedProfile === undefined ? summaries : collectArtifactSummaries(outputRoot);
    const summaryPath = writeSetIndex(outputRoot, indexedSummaries);
    console.log(`  set index: ${summaryPath}`);
  } finally {
    cleanup();
  }
}

main().catch((error: unknown) => {
  console.error('[graminoid-bake] FAIL', error instanceof Error ? error.stack ?? error.message : error);
  process.exit(1);
});
