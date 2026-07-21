/**
 * Render every authored ground-cover IndexedMesh directly from four cardinal
 * directions. Outputs content-addressed PNG QA; no ray bake/runtime shader is
 * involved, so this is the authoring-quality gate for every profile iteration.
 *
 * Usage:
 *   npx tsx tools/groundcover-bake/render-mesh-review.ts [--profile 2]
 *     [--px 1024] [--out data/work/groundcover-mesh-review] [--port 5232]
 */

import { spawn, type ChildProcess } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { relative, resolve } from 'node:path';
import sharp from 'sharp';
import { chromium, type Browser } from 'playwright';
import { makeAllEstonianGraminoidFixtures } from './EstonianGraminoids';
import { makeAllEstonianLowCoverFixtures } from './EstonianLowCover';
import {
  makeSphagnumCapillifoliumFixture,
  SPHAGNUM_CAPILLIFOLIUM_PROFILE_ID,
  SPHAGNUM_CAPILLIFOLIUM_SPECIES,
} from './SphagnumCapillifolium';

interface Args { [key: string]: string | boolean }
type CardinalView = 'north' | 'east' | 'south' | 'west';
type Framing = 'detail' | 'metric' | 'apex';

interface FixtureSummary {
  profileId: number;
  species: string;
  generator: string;
  mesh: { positions: number[]; normals: number[]; colors?: number[]; indices: number[] };
}

interface FixtureTransport {
  profileId: number;
  species: string;
  generator: string;
  positionsUrl: string;
  normalsUrl: string;
  colorsUrl?: string;
  indicesUrl: string;
}

interface ReviewResult {
  profileId: number;
  species: string;
  generator: string;
  view: CardinalView;
  framing: Framing;
  dimensionsM: { x: number; y: number; z: number };
  vertices: number;
  triangles: number;
}

const REPO_ROOT = resolve(import.meta.dirname, '../..');
const VIEWS: readonly CardinalView[] = ['north', 'east', 'south', 'west'];
const FRAMINGS: readonly Framing[] = ['detail', 'metric', 'apex'];

function parseArgs(argv: string[]): Args {
  const result: Args = {};
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index] as string;
    if (!arg.startsWith('--')) continue;
    const next = argv[index + 1];
    if (next !== undefined && !next.startsWith('--')) {
      result[arg.slice(2)] = next;
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

function sha(bytes: Uint8Array | string): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function browserUrlFor(path: string): string {
  return `/${relative(REPO_ROOT, path).replaceAll('\\', '/')}`;
}

function writeFloat32(path: string, values: number[]): void {
  const typed = new Float32Array(values);
  writeFileSync(path, Buffer.from(typed.buffer, typed.byteOffset, typed.byteLength));
}

function writeUint32(path: string, values: number[]): void {
  const typed = new Uint32Array(values);
  writeFileSync(path, Buffer.from(typed.buffer, typed.byteOffset, typed.byteLength));
}

function fixtures(): FixtureSummary[] {
  const graminoids = makeAllEstonianGraminoidFixtures().map((fixture) => ({
    profileId: fixture.profileId,
    species: fixture.species,
    generator: fixture.generator,
    mesh: fixture.mesh,
  }));
  const sphagnum = makeSphagnumCapillifoliumFixture();
  const low = makeAllEstonianLowCoverFixtures().map((fixture) => ({
    profileId: fixture.profileId,
    species: fixture.species,
    generator: fixture.generator,
    mesh: fixture.mesh,
  }));
  return [
    ...graminoids,
    {
      profileId: SPHAGNUM_CAPILLIFOLIUM_PROFILE_ID,
      species: SPHAGNUM_CAPILLIFOLIUM_SPECIES,
      generator: sphagnum.generator,
      mesh: sphagnum.mesh,
    },
    ...low,
  ].sort((left, right) => left.profileId - right.profileId);
}

async function waitForServer(url: string, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {
      // The dedicated localhost module server is still starting.
    }
    await new Promise((resolveWait) => setTimeout(resolveWait, 200));
  }
  throw new Error(`ground-cover mesh-review server did not become ready within ${timeoutMs} ms`);
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

function slug(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
}

function labelSvg(width: number, height: number, title: string, footer: string): Buffer {
  const escaped = (value: string): string => value
    .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
  return Buffer.from(`
    <svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">
      <rect x="0" y="0" width="${width}" height="42" fill="rgba(15,18,14,0.82)"/>
      <text x="18" y="28" fill="#f6f4ed" font-family="Arial, sans-serif" font-size="21" font-weight="700">${escaped(title)}</text>
      <rect x="0" y="${height - 38}" width="${width}" height="38" fill="rgba(15,18,14,0.82)"/>
      <text x="18" y="${height - 13}" fill="#f6f4ed" font-family="Arial, sans-serif" font-size="17">${escaped(footer)}</text>
    </svg>
  `);
}

function panelLabelSvg(width: number, label: string): Buffer {
  return Buffer.from(`
    <svg width="${width}" height="88" xmlns="http://www.w3.org/2000/svg">
      <rect x="10" y="48" width="178" height="30" rx="3" fill="rgba(15,18,14,0.78)"/>
      <text x="20" y="70" fill="#f6f4ed" font-family="Arial, sans-serif" font-size="17" font-weight="700">${label}</text>
    </svg>
  `);
}

async function makeSheet(
  inputPaths: readonly string[],
  outputPath: string,
  pixels: number,
  title: string,
  footer: string,
): Promise<void> {
  const panel = Math.round(pixels / 2);
  const labels = ['NORTH (+Z)', 'EAST (+X)', 'SOUTH (-Z)', 'WEST (-X)'];
  const tiles = await Promise.all(inputPaths.map(async (path, index) => {
    const image = await sharp(path).resize(panel, panel).toBuffer();
    return sharp(image).composite([{ input: panelLabelSvg(panel, labels[index]!) }]).png().toBuffer();
  }));
  await sharp({
    create: { width: pixels, height: pixels, channels: 3, background: { r: 184, g: 180, b: 170 } },
  }).composite([
    { input: tiles[0], left: 0, top: 0 },
    { input: tiles[1], left: panel, top: 0 },
    { input: tiles[2], left: 0, top: panel },
    { input: tiles[3], left: panel, top: panel },
    { input: labelSvg(pixels, pixels, title, footer), left: 0, top: 0 },
  ]).png().toFile(outputPath);
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const pixels = Number(stringArg(args.px) ?? 1280);
  const port = Number(stringArg(args.port) ?? 5232);
  const timeoutMs = Number(stringArg(args.timeout) ?? 300000);
  const selected = stringArg(args.profile);
  const allFixtures = fixtures();
  const selectedFixtures = selected === undefined
    ? allFixtures
    : allFixtures.filter((fixture) => fixture.profileId === Number(selected));
  if (selectedFixtures.length === 0) throw new Error(`unknown ground-cover profile ${selected}`);

  const recipe = {
    schema: 'laas-groundcover-authoring-mesh-review/v1',
    pixels,
    views: VIEWS,
    framings: {
      detail: 'per-profile orthographic fit',
      metric: 'fixed 1.6 m orthographic span',
      apex: 'highest reproductive/vegetative apex, 0.30 m orthographic span',
    },
    renderer: 'three r184 WebGPURenderer, direct indexed triangles, double-sided studio material',
    fixtures: selectedFixtures.map((fixture) => ({
      profileId: fixture.profileId,
      species: fixture.species,
      generator: fixture.generator,
      meshSha256: sha(JSON.stringify(fixture.mesh)),
    })),
    implementationSha256: sha([
      readFileSync(resolve(import.meta.dirname, 'mesh-review-page.ts')),
      readFileSync(import.meta.filename),
    ].map((value) => sha(value)).join(':')),
  };
  const buildSha = sha(JSON.stringify(recipe));
  const outputRoot = resolve(
    REPO_ROOT,
    stringArg(args.out) ?? 'data/work/groundcover-mesh-review',
    buildSha,
  );
  const qaRoot = resolve(outputRoot, 'qa');
  mkdirSync(qaRoot, { recursive: true });
  const transportRoot = resolve(outputRoot, 'transport');
  mkdirSync(transportRoot, { recursive: true });
  const transports = new Map<number, FixtureTransport>();
  for (const fixture of selectedFixtures) {
    const prefix = resolve(transportRoot, String(fixture.profileId));
    const positionsPath = `${prefix}-positions.f32`;
    const normalsPath = `${prefix}-normals.f32`;
    const colorsPath = fixture.mesh.colors ? `${prefix}-colors.f32` : undefined;
    const indicesPath = `${prefix}-indices.u32`;
    writeFloat32(positionsPath, fixture.mesh.positions);
    writeFloat32(normalsPath, fixture.mesh.normals);
    if (colorsPath && fixture.mesh.colors) writeFloat32(colorsPath, fixture.mesh.colors);
    writeUint32(indicesPath, fixture.mesh.indices);
    transports.set(fixture.profileId, {
      profileId: fixture.profileId,
      species: fixture.species,
      generator: fixture.generator,
      positionsUrl: browserUrlFor(positionsPath),
      normalsUrl: browserUrlFor(normalsPath),
      ...(colorsPath ? { colorsUrl: browserUrlFor(colorsPath) } : {}),
      indicesUrl: browserUrlFor(indicesPath),
    });
  }

  const vite: ChildProcess = spawn(
    'npx',
    ['vite', '--port', String(port), '--strictPort', '--host', 'localhost'],
    { cwd: REPO_ROOT, stdio: ['ignore', 'pipe', 'pipe'] },
  );
  vite.stderr?.on('data', (chunk: Buffer) => process.stderr.write(`[mesh-review:vite] ${chunk.toString()}`));
  let browser: Browser | null = null;
  const cleanup = (): void => {
    if (browser) browser.close().catch(() => undefined);
    if (!vite.killed) vite.kill('SIGTERM');
  };

  try {
    const url = `http://localhost:${port}/tools/groundcover-bake/mesh-review.html?px=${pixels}`;
    await waitForServer(url, timeoutMs);
    browser = await launchWebGPU();
    const page = await browser.newPage({
      viewport: { width: pixels, height: pixels },
      deviceScaleFactor: 1,
    });
    const diagnostics: string[] = [];
    page.on('console', (message) => {
      if (message.type() === 'error' || message.type() === 'warning') {
        diagnostics.push(`${message.type()}: ${message.text()}`);
      }
    });
    page.on('pageerror', (error) => diagnostics.push(`pageerror: ${error.message}`));
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: timeoutMs });
    await page.waitForFunction(
      () => window.__groundCoverMeshReview
        && (window.__groundCoverMeshReview.ready || window.__groundCoverMeshReview.error !== null),
      undefined,
      { timeout: timeoutMs, polling: 150 },
    );
    const pageError = await page.evaluate(() => window.__groundCoverMeshReview.error);
    if (pageError) throw new Error(pageError);

    const artifacts: Array<Record<string, unknown>> = [];
    const canvas = page.locator('canvas');
    for (const fixture of selectedFixtures) {
      await page.evaluate(
        (source) => window.__groundCoverMeshReview.loadBinary(source),
        transports.get(fixture.profileId)!,
      );
      const profileDir = resolve(qaRoot, `${fixture.profileId}-${slug(fixture.species)}`);
      mkdirSync(profileDir, { recursive: true });
      let metadata: ReviewResult | null = null;
      const sheets: Array<Record<string, unknown>> = [];
      for (const framing of FRAMINGS) {
        const paths: string[] = [];
        for (const view of VIEWS) {
          metadata = await page.evaluate(
            ([cardinalView, frame]) => window.__groundCoverMeshReview.render(
              cardinalView as CardinalView,
              frame as Framing,
            ),
            [view, framing],
          );
          const path = resolve(profileDir, `${framing}-${view}.png`);
          await canvas.screenshot({ path });
          paths.push(path);
        }
        const sheetNumber = framing === 'detail' ? 1 : framing === 'metric' ? 2 : 3;
        const sheetPath = resolve(
          profileDir,
          framing === 'detail'
            ? '01-cardinal-detail.png'
            : framing === 'metric'
              ? '02-cardinal-metric-1.6m.png'
              : '03-cardinal-apex-detail.png',
        );
        const dims = metadata!.dimensionsM;
        await makeSheet(
          paths,
          sheetPath,
          pixels,
          `${fixture.profileId}  ${fixture.species}  |  ${framing.toUpperCase()}`,
          `bbox ${dims.x.toFixed(3)} x ${dims.y.toFixed(3)} x ${dims.z.toFixed(3)} m  |  ${metadata!.vertices} vertices  |  ${metadata!.triangles} triangles`,
        );
        sheets.push({
          number: sheetNumber,
          file: relative(qaRoot, sheetPath),
          sha256: sha(readFileSync(sheetPath)),
          width: pixels,
          height: pixels,
          interpretation: framing === 'detail'
            ? 'Direct authoring triangles, per-profile orthographic fit, four cardinal azimuths.'
            : framing === 'metric'
              ? 'Direct authoring triangles, fixed 1.6 m orthographic span, four cardinal azimuths; use for stature comparison.'
              : 'Direct authoring triangles, 0.30 m crop around the highest apex; use to inspect inflorescence or shoot-tip anatomy.',
        });
      }
      const browserDiagnostics = await page.evaluate(() => window.__groundCoverMeshReview.diagnostics);
      if (diagnostics.length > 0 || browserDiagnostics.length > 0) {
        throw new Error(`mesh review browser/WebGPU diagnostics: ${[...diagnostics, ...browserDiagnostics].join(' | ')}`);
      }
      artifacts.push({
        profileId: fixture.profileId,
        species: fixture.species,
        generator: fixture.generator,
        meshSha256: recipe.fixtures.find((entry) => entry.profileId === fixture.profileId)!.meshSha256,
        dimensionsM: metadata!.dimensionsM,
        vertices: metadata!.vertices,
        triangles: metadata!.triangles,
        images: sheets,
      });
      console.log(`[groundcover-mesh-review] rendered ${fixture.profileId} ${fixture.species}`);
    }
    const indexPath = resolve(qaRoot, 'index.json');
    writeFileSync(indexPath, `${JSON.stringify({
      ...recipe,
      buildSha256: buildSha,
      generatedAt: new Date().toISOString(),
      artifacts,
    }, null, 2)}\n`);
    console.log(`[groundcover-mesh-review] PASS ${selectedFixtures.length} profile(s)`);
    console.log(`  build: ${buildSha}`);
    console.log(`  QA index: ${indexPath}`);
  } finally {
    cleanup();
  }
}

main().catch((error: unknown) => {
  console.error('[groundcover-mesh-review] FAIL', error instanceof Error ? error.stack ?? error.message : error);
  process.exit(1);
});
