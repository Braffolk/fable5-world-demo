/**
 * Shared Playwright launcher that guarantees a WebGPU-capable Chromium.
 * Probes flag sets (headless first, headed fallback) and caches the winner
 * in .cache/webgpu-flags.json so subsequent runs start instantly.
 */

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { chromium, type Browser } from 'playwright';

interface LaunchRecipe {
  headless: boolean;
  channel?: string;
  args: string[];
}

/**
 * IMPORTANT (discovered empirically on this machine):
 *  - WebGPU requires a secure context — probe on http://localhost, never about:blank
 *    (navigator.gpu is simply absent on opaque origins).
 *  - Playwright's default headless uses the GPU-less "headless shell": adapter = null.
 *    Full Chromium new-headless via channel:'chromium' yields an apple/metal-3 adapter.
 */
const CANDIDATES: LaunchRecipe[] = [
  { headless: true, channel: 'chromium', args: [] },
  { headless: true, channel: 'chromium', args: ['--enable-unsafe-webgpu'] },
  { headless: false, args: [] },
];

const CACHE_PATH = '.cache/webgpu-flags.json';
const PROBE_BASE = 'http://localhost:5173';

async function probeRecipe(recipe: LaunchRecipe): Promise<Browser | null> {
  let browser: Browser | null = null;
  try {
    const launchOpts: Parameters<typeof chromium.launch>[0] = {
      headless: recipe.headless,
      args: recipe.args,
    };
    if (recipe.channel) launchOpts.channel = recipe.channel;
    browser = await chromium.launch(launchOpts);
    const page = await browser.newPage();
    // any path on the dev server works — we only need the secure localhost origin
    await page.goto(`${PROBE_BASE}/__webgpu_probe__`, { waitUntil: 'domcontentloaded' });
    const ok = await page.evaluate(async () => {
      const gpu = (navigator as { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
      if (!gpu) return false;
      const adapter = await gpu.requestAdapter();
      return adapter !== null;
    });
    await page.close();
    if (ok) return browser;
    await browser.close();
    return null;
  } catch {
    if (browser) await browser.close().catch(() => undefined);
    return null;
  }
}

export async function launchWebGPU(): Promise<{ browser: Browser; recipe: LaunchRecipe }> {
  // cached recipe first
  try {
    const cached = JSON.parse(readFileSync(CACHE_PATH, 'utf8')) as LaunchRecipe;
    const browser = await probeRecipe(cached);
    if (browser) return { browser, recipe: cached };
  } catch {
    /* no cache yet */
  }
  for (const recipe of CANDIDATES) {
    const browser = await probeRecipe(recipe);
    if (browser) {
      mkdirSync('.cache', { recursive: true });
      writeFileSync(CACHE_PATH, JSON.stringify(recipe, null, 2));
      console.log(
        `[launch] WebGPU OK — headless=${recipe.headless} channel=${recipe.channel ?? 'default'} args=[${recipe.args.join(' ')}]`,
      );
      return { browser, recipe };
    }
  }
  throw new Error(
    'No Chromium launch recipe produced a WebGPU adapter (requires dev server on :5173 for the secure-context probe). ' +
      'Tried channel:chromium headless and headed.',
  );
}

export interface LaasPageOptions {
  scene?: string;
  seed?: number;
  T?: number;
  cam?: string;
  hud?: boolean;
  freeze?: boolean;
  width?: number;
  height?: number;
  extra?: Record<string, string>;
}

export function laasUrl(opts: LaasPageOptions, base = 'http://localhost:5173/'): string {
  const q = new URLSearchParams();
  if (opts.scene) q.set('scene', opts.scene);
  if (opts.seed !== undefined) q.set('seed', String(opts.seed));
  if (opts.T !== undefined) q.set('T', String(opts.T));
  if (opts.cam) q.set('cam', opts.cam);
  q.set('hud', opts.hud ? '1' : '0');
  if (opts.freeze !== false) q.set('freeze', '1');
  for (const [k, v] of Object.entries(opts.extra ?? {})) q.set(k, v);
  return `${base}?${q.toString()}`;
}
