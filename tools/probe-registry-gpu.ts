/**
 * GeometryRegistry GPU validation probe — drives ?scene=rasterspike&regtest=1
 * headless and asserts the [regtest] console verdict (TSL readVertex/
 * readCluster decode vs CPU mirrors + instance copy-kernel roundtrip).
 *
 *   npx tsx tools/probe-registry-gpu.ts        (needs the dev server on :5173)
 */

import { launchWebGPU, laasUrl } from './launch';

interface LaasWindow {
  __laas?: { ready?: boolean };
}

async function main(): Promise<void> {
  const { browser } = await launchWebGPU();
  let verdict = '';
  try {
    const page = await browser.newPage();
    await page.setViewportSize({ width: 960, height: 640 });
    page.on('console', (m) => {
      const t = m.text();
      if (t.startsWith('[regtest]')) verdict = t;
    });
    page.on('pageerror', (e) => {
      console.error(`  pageerror: ${String(e)}`);
    });
    await page.goto(laasUrl({ scene: 'rasterspike', hud: false, extra: { regtest: '1' } }), {
      waitUntil: 'domcontentloaded',
    });
    await page.waitForFunction(() => (window as LaasWindow).__laas?.ready === true, undefined, {
      timeout: 90_000,
    });
    const t0 = Date.now();
    while (!verdict && Date.now() - t0 < 30_000) {
      await new Promise((r) => setTimeout(r, 200));
    }
  } finally {
    await browser.close();
  }
  console.log(`[probe-registry-gpu] ${verdict || 'NO VERDICT (regtest log never appeared)'}`);
  if (!verdict.includes('PASS')) process.exit(1);
}

void main();
