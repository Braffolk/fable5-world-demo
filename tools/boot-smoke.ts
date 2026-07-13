/**
 * Real-browser WebGPU boot gate.
 *
 * Unlike the screenshot/perf harnesses, this treats browser, shader, and WebGPU
 * validation errors as fatal even when the app reaches window.__laas.ready.
 */

import { mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import { chromium } from 'playwright';

interface Args {
  [key: string]: string | boolean;
}

function parseArgs(argv: string[]): Args {
  const args: Args = {};
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i]!;
    if (!token.startsWith('--')) continue;
    const key = token.slice(2);
    const next = argv[i + 1];
    if (next !== undefined && !next.startsWith('--')) {
      args[key] = next;
      i++;
    } else {
      args[key] = true;
    }
  }
  return args;
}

function stringArg(args: Args, key: string): string | undefined {
  const value = args[key];
  return typeof value === 'string' ? value : undefined;
}

function isFavicon(url: string): boolean {
  try {
    return new URL(url).pathname === '/favicon.ico';
  } catch {
    return false;
  }
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const url = stringArg(args, 'url');
  if (!url) throw new Error('Usage: npx tsx tools/boot-smoke.ts --url <absolute-url>');

  const timeout = Number(stringArg(args, 'timeout') ?? 300_000);
  const settle = Number(stringArg(args, 'settle') ?? 24);
  const screenshot = stringArg(args, 'out') ?? 'shots/boot-smoke.png';
  const failures: string[] = [];
  let signalFatal!: (failure: string) => void;
  const firstFatal = new Promise<string>((resolve) => {
    signalFatal = resolve;
  });
  const fail = (failure: string): void => {
    failures.push(failure);
    if (failures.length === 1) signalFatal(failure);
  };
  const browser = await chromium.launch({ headless: true, channel: 'chromium' });

  try {
    const page = await browser.newPage({
      viewport: {
        width: Number(stringArg(args, 'w') ?? 1280),
        height: Number(stringArg(args, 'h') ?? 720),
      },
      deviceScaleFactor: 1,
    });

    page.on('pageerror', (error) => fail(`pageerror: ${error.message}`));
    page.on('requestfailed', (request) => {
      if (/^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?\//.test(request.url()) && !isFavicon(request.url())) {
        fail(`requestfailed: ${request.method()} ${request.url()} (${request.failure()?.errorText ?? 'unknown'})`);
      }
    });
    page.on('response', (response) => {
      if (
        response.status() >= 400 &&
        /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?\//.test(response.url()) &&
        !isFavicon(response.url())
      ) {
        fail(`http ${response.status()}: ${response.url()}`);
      }
    });
    page.on('console', (message) => {
      const text = message.text();
      const fatalText =
        text.includes('WebGPU uncaptured error') ||
        text.includes('Invalid generated code') ||
        text.includes('Boot failed');
      const location = message.location().url;
      if ((message.type() === 'error' || fatalText) && !isFavicon(location)) {
        fail(`console:${message.type()}: ${text}${location ? ` (${location})` : ''}`);
      }
    });

    await page.goto(url, { waitUntil: 'domcontentloaded', timeout });
    const hookReady = page.waitForFunction(
      () =>
        window.__laas &&
        (window.__laas.ready === true ||
          (typeof window.__laas.error === 'string' && window.__laas.error.length > 0)),
      undefined,
      { timeout, polling: 250 },
    );
    const outcome = await Promise.race([
      hookReady.then(() => 'hook' as const),
      firstFatal.then(() => 'fatal' as const),
    ]);
    if (outcome === 'fatal') {
      hookReady.catch(() => undefined);
      throw new Error(failures[0]);
    }

    const hookError = await page.evaluate(() =>
      typeof window.__laas.error === 'string' && window.__laas.error.length > 0
        ? window.__laas.error
        : null,
    );
    if (hookError) fail(`window.__laas.error: ${hookError}`);

    if (!hookError) {
      const settleOutcome = await Promise.race([
        page.evaluate(async (frames) => window.__laas.settle?.(frames), settle).then(() => 'settled' as const),
        firstFatal.then(() => 'fatal' as const),
      ]);
      if (settleOutcome === 'fatal') throw new Error(failures[0]);
      // Give validation callbacks queued by the final submit time to fire.
      const validationOutcome = await Promise.race([
        page.waitForTimeout(1_000).then(() => 'quiet' as const),
        firstFatal.then(() => 'fatal' as const),
      ]);
      if (validationOutcome === 'fatal') throw new Error(failures[0]);
    }

    const lateHookError = await page.evaluate(() =>
      typeof window.__laas.error === 'string' && window.__laas.error.length > 0
        ? window.__laas.error
        : null,
    );
    if (lateHookError) fail(`window.__laas.error after settle: ${lateHookError}`);

    const state = await page.evaluate(() => ({
      ready: window.__laas.ready,
      frame: window.__laas.stats?.frame ?? -1,
      canvas: [...document.querySelectorAll('canvas')].map((canvas) => [canvas.width, canvas.height]),
    }));
    if (!state.ready) fail('window.__laas.ready is false');
    if (state.frame < 1) fail(`no rendered frame observed (frame=${state.frame})`);
    if (!state.canvas.some(([width, height]) => width > 0 && height > 0)) fail('no non-empty canvas');

    mkdirSync(dirname(screenshot), { recursive: true });
    await page.screenshot({ path: screenshot });

    if (failures.length > 0) {
      throw new Error(`${failures.length} boot failure(s):\n${failures.map((failure) => `- ${failure}`).join('\n')}`);
    }
    console.log(`[boot-smoke] PASS frame=${state.frame} canvas=${JSON.stringify(state.canvas)} url=${url}`);
  } finally {
    await browser.close();
  }
}

main().catch((error: unknown) => {
  console.error('[boot-smoke] FAIL', error instanceof Error ? error.message : error);
  process.exit(1);
});
