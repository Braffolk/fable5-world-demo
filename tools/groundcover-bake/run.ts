import { spawn, type ChildProcess } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import sharp from 'sharp';
import { chromium, type Browser } from 'playwright';
import {
  decodeOct,
  makeDirection,
  makeEllipsoidFixture,
  makeProjectionSlice,
  packProfile,
  PROFILE_TEXEL_BYTES,
  validateEllipsoidBake,
  type BakeResult,
  type EllipsoidFixture,
} from './ProfileFormat';

interface Args {
  [key: string]: string | boolean;
}

function parseArgs(argv: string[]): Args {
  const result: Args = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i] as string;
    if (!arg.startsWith('--')) continue;
    const key = arg.slice(2);
    const value = argv[i + 1];
    if (value !== undefined && !value.startsWith('--')) {
      result[key] = value;
      i++;
    } else {
      result[key] = true;
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

function diagnosticPixels(result: BakeResult, mode: 'depth' | 'normal'): Uint8Array {
  const width = result.tileWidth * result.atlasColumns;
  const height = result.tileHeight * result.atlasRows;
  const output = new Uint8Array(width * height * 4);
  for (let texel = 0; texel < width * height; texel++) {
    const source = texel * 4;
    const target = texel * 4;
    const hit = (result.pixels[source + 3] as number) > 0.5;
    if (!hit) {
      output.set([18, 8, 22, 255], target);
      continue;
    }
    if (mode === 'depth') {
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

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const port = Number(stringArg(args.port) ?? '5227');
  const timeoutMs = Number(stringArg(args.timeout) ?? '300000');
  const tile = Number(stringArg(args.tile) ?? '96');
  const outputRoot = resolve(stringArg(args.out) ?? '/tmp/groundcover-gpu-bake');
  const repoRoot = resolve(import.meta.dirname, '../..');
  const fixture: EllipsoidFixture = {
    center: { x: 0, y: 0.42, z: 0 },
    radii: { x: 0.36, y: 0.42, z: 0.28 },
    sectors: 128,
    stacks: 64,
  };
  const mesh = makeEllipsoidFixture(fixture);
  const directions = [
    makeDirection(0, 20), makeDirection(90, 20), makeDirection(180, 20), makeDirection(270, 20),
    makeDirection(0, 55), makeDirection(90, 55), makeDirection(180, 55), makeDirection(270, 55),
  ];
  const slices = directions.map((direction) => makeProjectionSlice(mesh, direction));
  const request = {
    mesh,
    slices,
    tileWidth: tile,
    tileHeight: tile,
    atlasColumns: 4,
    atlasRows: 2,
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
    const url = `http://localhost:${port}/tools/groundcover-bake/bake.html`;
    await waitForServer(url, timeoutMs);
    browser = await launchWebGPU();
    const page = await browser.newPage();
    const diagnostics: string[] = [];
    page.on('console', (message) => {
      if (message.type() === 'error' || message.type() === 'warning') diagnostics.push(`${message.type()}: ${message.text()}`);
    });
    page.on('pageerror', (error) => diagnostics.push(`pageerror: ${error.message}`));
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: timeoutMs });
    await page.waitForFunction(() => window.__groundCoverBake?.ready === true, undefined, { timeout: timeoutMs });
    const first = await page.evaluate(async (bakeRequest) => window.__groundCoverBake.run(bakeRequest), request);
    const second = await page.evaluate(async (bakeRequest) => window.__groundCoverBake.run(bakeRequest), request);
    if (diagnostics.length > 0) throw new Error(`browser diagnostics: ${diagnostics.join(' | ')}`);

    const packed = packProfile(first, 1);
    const packedAgain = packProfile(second, 1);
    if (packed.sha256 !== packedAgain.sha256) {
      throw new Error(`non-deterministic packed GPU result: ${packed.sha256} != ${packedAgain.sha256}`);
    }
    const validation = validateEllipsoidBake(first, fixture);
    if (
      validation.compared < 10_000 ||
      validation.missingInterior !== 0 ||
      validation.rmsDepth > 0.0025 ||
      validation.maxDepth > 0.012 ||
      validation.meanNormalDot < 0.995 ||
      validation.minNormalDot < 0.96
    ) {
      throw new Error(`analytic validation failed: ${JSON.stringify(validation)}`);
    }

    const artifactDir = resolve(outputRoot, packed.sha256.slice(0, 16));
    const qaDir = resolve(artifactDir, 'qa');
    mkdirSync(qaDir, { recursive: true });
    const binaryPath = resolve(artifactDir, 'ellipsoid-profile.gcrp');
    writeFileSync(binaryPath, packed.bytes);
    const width = first.tileWidth * first.atlasColumns;
    const height = first.tileHeight * first.atlasRows;
    const depthPath = resolve(qaDir, '01-first-hit-depth.png');
    const normalPath = resolve(qaDir, '02-first-hit-normal.png');
    await sharp(diagnosticPixels(first, 'depth'), { raw: { width, height, channels: 4 } }).png().toFile(depthPath);
    await sharp(diagnosticPixels(first, 'normal'), { raw: { width, height, channels: 4 } }).png().toFile(normalPath);

    const recipe = {
      fixture: 'analytic-ellipsoid-indexed-triangle-mesh',
      ellipsoid: fixture,
      mesh: { vertices: mesh.positions.length / 3, triangles: mesh.indices.length / 3 },
      directions: slices.map((slice) => slice.direction),
      tile,
      atlas: { columns: 4, rows: 2 },
    };
    const index = {
      schema: 'laas-groundcover-gpu-bake-qa/v1',
      recipeSha256: hashJson(recipe),
      binary: {
        file: '../ellipsoid-profile.gcrp',
        sha256: packed.sha256,
        bytes: packed.bytes.byteLength,
        profileId: 1,
        format: 'GCRP/v1, little-endian, per-slice affine frame, RGBA16 payload',
        texel: ['normalized first-hit depth (65535 means miss)', 'oct-normal x', 'oct-normal y', 'hit mask'],
      },
      diagnosticImages: [
        { number: 1, file: '01-first-hit-depth.png', sha256: hashFile(depthPath), width, height, interpretation: 'white is nearer first-hit depth; dark purple is miss' },
        { number: 2, file: '02-first-hit-normal.png', sha256: hashFile(normalPath), width, height, interpretation: 'decoded world normal mapped from [-1,1] to [0,255]; dark purple is miss' },
      ],
      recipe,
      adapter: first.adapter,
      deterministicRepeat: { runs: 2, identicalSha256: true },
      analyticAcceptance: {
        scope: 'pixels with exact ellipsoid incidence >= 0.25; tessellated mesh uses 128x64 sectors/stacks',
        requirements: { comparedAtLeast: 10_000, missingInterior: 0, rmsDepthAtMost: 0.0025, maxDepthAtMost: 0.012, meanNormalDotAtLeast: 0.995, minNormalDotAtLeast: 0.96 },
        observed: validation,
      },
      runtimeResourceProjection: {
        artifactAtlas: `${width}x${height} rgba16unorm`,
        bytesPerProfile: width * height * PROFILE_TEXEL_BYTES,
        productionExample: '64x64 tiles x 16 azimuth x 4 elevation x 8 B = 2 MiB/profile; 12 profiles = 24 MiB before guards/mips (~30 MiB budget)',
        sampling: 'four fixed direction-corner taps for one profile; bounded A/B cover mixture is at most eight fixed taps',
        bindings: 'one filterable 2D-array/3D atlas plus one sampler for all profiles; no per-profile binding',
      },
    };
    const indexPath = resolve(qaDir, 'index.json');
    writeFileSync(indexPath, `${JSON.stringify(index, null, 2)}\n`);
    console.log('[groundcover-bake] PASS');
    console.log(`  adapter: ${first.adapter}`);
    console.log(`  mesh: ${mesh.positions.length / 3} vertices, ${mesh.indices.length / 3} triangles`);
    console.log(`  atlas: ${width}x${height}, ${first.slices.length} slices, ${packed.bytes.byteLength} bytes packed`);
    console.log(`  validation: ${JSON.stringify(validation)}`);
    console.log(`  deterministic sha256: ${packed.sha256}`);
    console.log(`  binary: ${binaryPath}`);
    console.log(`  QA index: ${indexPath}`);
  } finally {
    cleanup();
  }
}

main().catch((error: unknown) => {
  console.error('[groundcover-bake] FAIL', error instanceof Error ? error.stack ?? error.message : error);
  process.exit(1);
});
