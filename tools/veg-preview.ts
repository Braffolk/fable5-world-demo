/**
 * Offscreen mesh-preview harness — visual QA for vegetation meshes in isolation.
 *
 * A mesh module exports `buildPreview(rng): THREE.Object3D` (a fully-assembled,
 * materialed plant). This harness renders that Object3D to PNGs from 4 views
 * (front / three-quarter / side / top) in a neutral studio, WITHOUT booting the
 * app or wiring the mesh into the scatter pipeline. It also prints the object's
 * real-world bounding-box dimensions and total triangle count.
 *
 * Render path: Playwright + Chromium WebGPU driving three r184's WebGPURenderer
 * (the app's exact renderer), so whatever material the modeller attached — plain
 * three or a TSL NodeMaterial — renders faithfully. Vite dev-serves the page so
 * bare `three` and `src/**` imports resolve exactly as in the app. The harness
 * boots its OWN throwaway Vite server (no need for a running dev server) and
 * tears it down on exit.
 *
 * Usage:
 *   npx tsx tools/veg-preview.ts --module src/vegetation/bog/CottonGrass.ts \
 *     --export buildPreview --out shots/preview/cottongrass \
 *     [--seed 1] [--stream preview] [--px 1024] [--scale-hint 0.4]
 *
 * `--scale-hint` is informational only (framing is auto-fit from the bounding
 * box); it is echoed so the modeller can sanity-check the reported dims.
 */

import { spawn, type ChildProcess } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import { relative, resolve } from 'node:path';
import sharp from 'sharp';
import { chromium, type Browser } from 'playwright';

interface Args {
  [k: string]: string | boolean;
}

function parseArgs(argv: string[]): Args {
  const out: Args = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i] as string;
    if (!a.startsWith('--')) continue;
    const key = a.slice(2);
    const next = argv[i + 1];
    if (next !== undefined && !next.startsWith('--')) {
      out[key] = next;
      i++;
    } else {
      out[key] = true;
    }
  }
  return out;
}

function str(v: string | boolean | undefined): string | undefined {
  return typeof v === 'string' ? v : undefined;
}

const REPO_ROOT = resolve(import.meta.dirname, '..');

/** Turn a user-supplied module path into the `/src/...`-style path Vite serves. */
function servedModulePath(input: string): string {
  const abs = resolve(REPO_ROOT, input);
  const rel = relative(REPO_ROOT, abs).split('\\').join('/');
  if (rel.startsWith('..')) throw new Error(`--module must live inside the repo: ${input}`);
  return `/${rel}`;
}

async function waitForServer(url: string, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(url);
      if (res.ok || res.status === 404) return; // server is up (404 = route not found, still up)
    } catch {
      /* not up yet */
    }
    await new Promise((r) => setTimeout(r, 200));
  }
  throw new Error(`Vite dev server did not come up at ${url} within ${timeoutMs} ms`);
}

async function launchWebGPUBrowser(): Promise<Browser> {
  // Full Chromium (channel:'chromium') exposes a real Metal WebGPU adapter headless;
  // the default headless "shell" does not. Mirrors tools/launch.ts findings.
  const recipes: Parameters<typeof chromium.launch>[0][] = [
    { headless: true, channel: 'chromium', args: [] },
    { headless: true, channel: 'chromium', args: ['--enable-unsafe-webgpu'] },
    { headless: false, args: ['--enable-unsafe-webgpu'] },
  ];
  let lastErr: unknown;
  for (const opts of recipes) {
    try {
      return await chromium.launch(opts);
    } catch (e) {
      lastErr = e;
    }
  }
  throw new Error(`Could not launch a Chromium: ${lastErr instanceof Error ? lastErr.message : lastErr}`);
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const moduleInput = str(args['module']);
  if (!moduleInput) {
    throw new Error(
      'Usage: npx tsx tools/veg-preview.ts --module src/vegetation/bog/X.ts --export buildPreview --out <dir> [--seed 1] [--px 1024] [--scale-hint 0.4]',
    );
  }
  const exportName = str(args['export']) ?? 'buildPreview';
  const outDir = resolve(REPO_ROOT, str(args['out']) ?? 'shots/veg-preview');
  const seed = str(args['seed']) ?? '1';
  const stream = str(args['stream']) ?? 'preview';
  const px = str(args['px']) ?? '1024';
  const scaleHint = str(args['scale-hint']);
  const port = Number(str(args['port']) ?? 5211);
  const windMode = args['wind'] === true || str(args['wind']) === '1';
  const windAB = args['ab'] === true || str(args['ab']) === '1';
  const mod = servedModulePath(moduleInput);

  mkdirSync(outDir, { recursive: true });

  console.log(`[veg-preview] module=${mod} export=${exportName} seed=${seed} out=${outDir}`);
  if (scaleHint) console.log(`[veg-preview] scale-hint=${scaleHint} m (informational; framing is auto-fit)`);

  // --- boot a throwaway Vite dev server ------------------------------------
  const vite: ChildProcess = spawn(
    'npx',
    ['vite', '--port', String(port), '--strictPort', '--host', '127.0.0.1'],
    { cwd: REPO_ROOT, stdio: ['ignore', 'pipe', 'pipe'] },
  );
  vite.stdout?.on('data', (d: Buffer) => {
    const t = d.toString();
    if (t.includes('error') || t.includes('Error')) process.stdout.write(`[vite] ${t}`);
  });
  vite.stderr?.on('data', (d: Buffer) => process.stderr.write(`[vite] ${d.toString()}`));

  let browser: Browser | null = null;
  const cleanup = (): void => {
    if (browser) browser.close().catch(() => undefined);
    if (!vite.killed) vite.kill('SIGTERM');
  };

  try {
    const base = `http://127.0.0.1:${port}`;
    await waitForServer(`${base}/tools/preview/preview.html`, 60_000);

    browser = await launchWebGPUBrowser();
    const page = await browser.newPage({
      viewport: { width: Number(px), height: Number(px) },
      deviceScaleFactor: 1,
    });
    page.on('console', (m) => {
      if (m.type() === 'error') console.log(`[page:error] ${m.text()}`);
    });
    page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));

    const q = new URLSearchParams({ mod, exp: exportName, seed, stream, px });
    if (windMode) q.set('wind', '1');
    if (windAB) q.set('windab', '1');
    const url = `${base}/tools/preview/preview.html?${q.toString()}`;
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60_000 });

    // wait for ready or a page-side error
    await page.waitForFunction(() => window.__preview && (window.__preview.ready || window.__preview.error !== null), undefined, {
      timeout: 120_000,
      polling: 150,
    });
    const perr = await page.evaluate(() => window.__preview.error);
    if (perr) throw new Error(`page build failed:\n${perr}`);

    const info = await page.evaluate(() => ({
      dims: window.__preview.dims,
      tris: window.__preview.tris,
      views: window.__preview.views,
    }));
    const d = info.dims;

    const canvas = page.locator('canvas');

    // ── wind filmstrip mode: render each (row × frame) cell of the CPU-transcribed
    //    engine sway, then composite a grid so a stuck/stretch tear is visible ────
    if (windMode) {
      const wind = await page.evaluate(() => {
        const w = window.__preview.wind;
        return w ? { rows: w.rows.map((r) => r.label), frames: w.frames } : null;
      });
      if (!wind) throw new Error('wind mode requested but window.__preview.wind is null');
      const cs = Number(px);
      const cw = Math.min(Math.round(cs / wind.frames), Math.round(cs / Math.max(wind.rows.length, 1)));
      const composites: sharp.OverlayOptions[] = [];
      for (let r = 0; r < wind.rows.length; r++) {
        for (let f = 0; f < wind.frames; f++) {
          await page.evaluate(([rr, ff]) => window.__preview.wind!.render(rr as number, ff as number), [r, f]);
          const outPath = resolve(outDir, `wind_r${r}_f${f}.png`);
          await canvas.screenshot({ path: outPath });
          const tile = await sharp(outPath).resize(cw, cw).toBuffer();
          composites.push({ input: tile, left: f * cw, top: r * cw });
        }
      }
      const sheetPath = resolve(outDir, 'wind.png');
      await sharp({ create: { width: cw * wind.frames, height: cw * wind.rows.length, channels: 3, background: { r: 140, g: 140, b: 140 } } })
        .composite(composites)
        .png()
        .toFile(sheetPath);
      console.log('');
      console.log('[veg-preview] WIND DONE');
      if (d) console.log(`  bbox (m):  x=${d.x.toFixed(3)}  y=${d.y.toFixed(3)}  z=${d.z.toFixed(3)}`);
      console.log(`  rows:      ${wind.rows.join('  |  ')}`);
      console.log(`  frames/row: ${wind.frames}`);
      console.log(`  filmstrip: ${sheetPath}`);
      return;
    }
    const written: string[] = [];
    for (const view of info.views) {
      await page.evaluate((v) => window.__preview.render(v), view);
      const outPath = resolve(outDir, `${view}.png`);
      await canvas.screenshot({ path: outPath });
      written.push(outPath);
    }

    // 2x2 contact sheet
    const [front, threeq, side, top] = written;
    const cs = Number(px);
    const half = Math.round(cs / 2);
    const tiles = await Promise.all(
      [front, threeq, side, top].map((p) => sharp(p!).resize(half, half).toBuffer()),
    );
    const sheetPath = resolve(outDir, 'contact.png');
    await sharp({ create: { width: cs, height: cs, channels: 3, background: { r: 140, g: 140, b: 140 } } })
      .composite([
        { input: tiles[0], left: 0, top: 0 },
        { input: tiles[1], left: half, top: 0 },
        { input: tiles[2], left: 0, top: half },
        { input: tiles[3], left: half, top: half },
      ])
      .png()
      .toFile(sheetPath);

    console.log('');
    console.log('[veg-preview] DONE');
    if (d) {
      console.log(
        `  bbox (m):  x=${d.x.toFixed(3)}  y=${d.y.toFixed(3)} (height)  z=${d.z.toFixed(3)}`,
      );
    }
    console.log(`  triangles: ${info.tris}`);
    console.log(`  views:     ${written.join(', ')}`);
    console.log(`  contact:   ${sheetPath}`);
  } finally {
    cleanup();
  }
}

main().catch((err: unknown) => {
  console.error('[veg-preview] FAIL', err instanceof Error ? err.message : err);
  process.exit(1);
});
