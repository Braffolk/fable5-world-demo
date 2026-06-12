/**
 * Pointer-lock acquisition probe — HEADED (headless Chromium rejects every
 * pointer-lock request with WrongDocumentError; a real window opens briefly).
 *
 * Verifies the cooldown-aware re-lock path in FlyCamera: a click right after
 * exiting pointer lock must NOT be dropped — the rig defers the request past
 * the browser's post-exit cooldown (~1.25 s) and acquires the lock unaided.
 * Tests both the programmatic exit and (when the synthesized key reaches the
 * browser's accelerator) the real ESC exit.
 *
 * Run: npx tsx tools/probe-pointerlock.ts
 */

import { chromium } from 'playwright';
import { laasUrl } from './launch';

interface LaasWindow {
  __laas?: { ready?: boolean };
}

async function main(): Promise<void> {
  const browser = await chromium.launch({ headless: false });
  let failed = false;
  try {
    const page = await browser.newPage();
    await page.setViewportSize({ width: 960, height: 640 });

    const lockErrors: string[] = [];
    page.on('pageerror', (e) => {
      if (/pointer ?lock/i.test(String(e))) lockErrors.push(String(e));
    });
    page.on('console', (m) => {
      if (m.type() === 'error' && /pointer ?lock/i.test(m.text())) lockErrors.push(m.text());
    });

    // sanity scene boots in seconds and creates the same FlyCamera rig
    await page.goto(laasUrl({ scene: 'sanity', hud: false }), { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => (window as LaasWindow).__laas?.ready === true, undefined, {
      timeout: 60_000,
    });

    const locked = (): Promise<boolean> =>
      page.evaluate(() => document.pointerLockElement !== null);
    /** wait until locked; returns elapsed ms or -1 on timeout */
    const waitLocked = async (timeoutMs: number): Promise<number> => {
      const t0 = Date.now();
      try {
        await page.waitForFunction(() => document.pointerLockElement !== null, undefined, {
          timeout: timeoutMs,
          polling: 50,
        });
        return Date.now() - t0;
      } catch (e) {
        // only a TIMEOUT means "not locked" — anything else is a probe bug
        if (e instanceof Error && e.name === 'TimeoutError') return -1;
        throw e;
      }
    };
    const check = (name: string, ok: boolean, detail = ''): void => {
      console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? ` — ${detail}` : ''}`);
      if (!ok) failed = true;
    };

    // 1) plain click → lock
    await page.bringToFront();
    await page.mouse.click(480, 320);
    const t1 = await waitLocked(2000);
    check('lock on first click', t1 >= 0, `${t1} ms`);

    // 2) programmatic exit, click INSIDE the cooldown → deferred re-lock
    await page.evaluate(() => document.exitPointerLock());
    await page.waitForTimeout(120); // pointerlockchange records unlockAt
    await page.mouse.click(480, 320);
    await page.waitForTimeout(250);
    const instant = await locked();
    const t2 = await waitLocked(4000);
    check(
      'click-after-exit re-locks unaided',
      t2 >= 0,
      instant ? 'locked instantly (no deferral needed)' : `deferred, locked after ${t2 + 250} ms`,
    );

    // 3) real ESC exit (browser accelerator) — only testable if the
    // synthesized key reaches it; skip cleanly when it doesn't
    await page.keyboard.press('Escape');
    await page.waitForTimeout(150);
    if (!(await locked())) {
      await page.mouse.click(480, 320);
      await page.waitForTimeout(250);
      const t3 = await waitLocked(5000);
      check('click-after-ESC re-locks unaided', t3 >= 0, `locked after ${t3 + 250} ms`);
    } else {
      console.log('SKIP  ESC exit (synthesized key did not reach the browser accelerator)');
    }

    check('no unhandled pointer-lock errors', lockErrors.length === 0, lockErrors.join(' | '));
  } finally {
    await browser.close();
  }
  if (failed) process.exitCode = 1;
}

main().catch((e: unknown) => {
  console.error(e);
  process.exitCode = 1;
});
