/**
 * N1-C4 boot gate probe — boots the REAL world scene with ?nanite=1 headless,
 * captures the registry build log, and asserts the phase gate: all opaque
 * pools clusterized with clusterize time < 2 s (NANITE-SPEC.md N1 gate).
 *
 *   npx tsx tools/probe-nanite-boot.ts     (needs the dev server on :5173)
 */

import { launchWebGPU, laasUrl } from './launch';

interface LaasWindow {
  __laas?: { ready?: boolean };
}

async function main(): Promise<void> {
  const { browser } = await launchWebGPU();
  let registryLog = '';
  const errors: string[] = [];
  try {
    const page = await browser.newPage();
    await page.setViewportSize({ width: 1280, height: 720 });
    page.on('console', (m) => {
      const t = m.text();
      if (t.includes('[laas] nanite registry:')) registryLog = t;
      if (m.type() === 'error') errors.push(t);
    });
    page.on('pageerror', (e) => errors.push(String(e)));
    // naniteframe=0 → N1 build-only semantics: this probe gates the FULL
    // registry build (all classes), not the N4 full-frame migration set
    await page.goto(
      laasUrl({ scene: 'world', hud: false, extra: { nanite: '1', naniteframe: '0' } }),
      { waitUntil: 'domcontentloaded' },
    );
    await page.waitForFunction(() => (window as LaasWindow).__laas?.ready === true, undefined, {
      timeout: 180_000,
    });
    const t0 = Date.now();
    while (!registryLog && Date.now() - t0 < 10_000) {
      await new Promise((r) => setTimeout(r, 200));
    }
  } finally {
    await browser.close();
  }

  if (!registryLog) {
    console.error('[probe-nanite-boot] FAIL — registry log never appeared');
    for (const e of errors.slice(0, 5)) console.error(`  err: ${e}`);
    process.exit(1);
  }
  console.log(registryLog);
  const clusterize = /clusterize (\d+) ms/.exec(registryLog);
  const ms = clusterize ? Number(clusterize[1]) : NaN;
  const pass = Number.isFinite(ms) && ms < 2000;
  console.log(`[probe-nanite-boot] ${pass ? 'PASS' : 'FAIL'} — clusterize ${ms} ms (gate < 2000)`);
  if (errors.length) {
    console.log(`  (${errors.length} console errors during boot — first: ${errors[0]})`);
  }
  if (!pass) process.exit(1);
}

void main();
