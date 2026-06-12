/**
 * N2 debug-view probe — boots the world with ?nanite=1&nanitedbg=<mode>,
 * waits ready, captures the nanite.* HUD counters + a screenshot, and fails
 * loud on console errors or queue overflow.
 *
 *   npx tsx tools/probe-nanitedbg.ts [flat|cluster] [bookmarkN]
 *   (needs the dev server on :5173; screenshot → /tmp/nanitedbg-*.png)
 */
import { launchWebGPU, laasUrl } from './launch';

async function main(): Promise<void> {
  const mode = process.argv[2] ?? 'flat';
  const shot = process.argv[3] ?? '';
  const { browser } = await launchWebGPU();
  const errors: string[] = [];
  const logs: string[] = [];
  try {
    const page = await browser.newPage();
    await page.setViewportSize({ width: 1280, height: 720 });
    page.on('console', (m) => {
      const t = m.text();
      if (t.includes('[laas]') || t.includes('[nanite]')) logs.push(t);
      if (m.type() === 'error') errors.push(t);
    });
    page.on('pageerror', (e) => errors.push(String(e)));
    const extra: Record<string, string> = { nanite: '1', nanitedbg: mode };
    if (shot) extra['shot'] = shot;
    await page.goto(laasUrl({ scene: 'world', hud: false, extra }), { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => (window as unknown as { __laas?: { ready?: boolean } }).__laas?.ready === true, undefined, { timeout: 240_000 });
    await new Promise((r) => setTimeout(r, 4000)); // let meter() publish counters
    const counters = await page.evaluate(() => {
      const dbg = (window as unknown as { __laasDbg?: { engine?: { stats?: { counters?: Record<string, number> } } } }).__laasDbg;
      const c = dbg?.engine?.stats?.counters ?? {};
      return Object.fromEntries(Object.entries(c).filter(([k]) => k.startsWith('nanite')));
    });
    await page.screenshot({ path: `/tmp/nanitedbg-${mode}${shot ? `-bm${shot}` : ''}.png` });
    console.log('counters:', JSON.stringify(counters));
  } finally {
    await browser.close();
  }
  for (const l of logs) console.log('log:', l.slice(0, 300));
  for (const e of errors.slice(0, 8)) console.log('ERR:', e.slice(0, 500));
  const overflow = logs.some((l) => l.includes('OVERFLOW'));
  const pass = errors.length === 0 && !overflow;
  console.log(`[probe-nanitedbg] ${pass ? 'PASS' : 'FAIL'} (${errors.length} errors${overflow ? ', queue overflow' : ''})`);
  if (!pass) process.exit(1);
}
void main();
