import { launchWebGPU, laasUrl } from './launch';
async function main(): Promise<void> {
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width: 960, height: 540 } });
  await page.goto(
    laasUrl({ scene: 'terrain', T: 19.3, cam: '500,520,1400,2.9,-0.25', extra: { ablate: 'veg' } }),
    { waitUntil: 'domcontentloaded' },
  );
  await page.waitForFunction(() => window.__laas && (window.__laas.ready || window.__laas.error !== null), undefined, { timeout: 180000, polling: 250 });
  await page.evaluate(async () => window.__laas.settle && (await window.__laas.settle(40)));
  await page.screenshot({ path: 'shots/wip/probe-T19.png' });
  await page.evaluate((t) => (window as any).__laas.setTimeOfDay(t), 7.2);
  await page.evaluate(async () => window.__laas.settle && (await window.__laas.settle(90)));
  await page.screenshot({ path: 'shots/wip/probe-T7.png' });
  await browser.close();
}
void main();
