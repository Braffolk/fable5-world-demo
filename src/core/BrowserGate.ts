/**
 * Pre-boot environment gate — runs before ANY engine work so users on
 * unsupported setups get a clear message instead of a broken boot screen.
 *
 * Order of checks:
 *  1. mobile/tablet → recommend a computer. Detection is capability/UA
 *     based, never screen size: UA-Client-Hints `userAgentData.mobile`
 *     where present, classic UA markers otherwise, plus the iPadOS
 *     masquerade (iPads report a desktop macOS UA; multi-touch on "Mac"
 *     exposes them).
 *  2. non-Chromium browser → Chrome required. The engine is built and
 *     tested exclusively against Chrome's WebGPU; Safari/Firefox coverage
 *     of the features used here is incomplete (user-verified: neither
 *     boots). Brand list from UA-CH when present (covers Chrome, Edge,
 *     Brave, Arc, Opera), "Chrome/" UA token as the fallback —
 *     HeadlessChrome passes both, so the Playwright tooling is unaffected.
 *  3. Chromium but `navigator.gpu` missing → the standard tactic: there
 *     is no fallback, so give the actionable checklist (update, hardware
 *     acceleration, chrome://gpu).
 *
 * The adapter-level probe (gpu present but no usable adapter) stays in
 * boot()/probeWebGPU, which reports richer diagnostics.
 *
 * `?nogate=1` skips everything (tooling/debug escape hatch).
 */

import { failLoud } from './Diagnostics';

interface UAClientHints {
  mobile?: boolean;
  brands?: { brand: string; version: string }[];
}

function clientHints(): UAClientHints | undefined {
  return (navigator as { userAgentData?: UAClientHints }).userAgentData;
}

/** phone/tablet detection — capability + UA based, never screen metrics */
export function isMobileDevice(): boolean {
  if (clientHints()?.mobile === true) return true;
  const ua = navigator.userAgent;
  // Android tablets drop "Mobile" but keep "Android"; Kindle = Silk
  if (/Android|iPhone|iPod|iPad|Windows Phone|IEMobile|Silk|Mobile/i.test(ua)) return true;
  // iPadOS 13+ masquerades as desktop macOS ("Macintosh" UA) — multi-touch
  // gives it away; real Macs report 0 touch points
  if (/Macintosh/i.test(ua) && navigator.maxTouchPoints > 2) return true;
  return false;
}

/** Chrome or any Chromium-based browser (Edge, Brave, Arc, Opera, ...) */
export function isChromiumBrowser(): boolean {
  const brands = clientHints()?.brands;
  if (brands && brands.length > 0) {
    return brands.some((b) => /Chromium|Google Chrome/i.test(b.brand));
  }
  // every Chromium UA carries "Chrome/" (incl. HeadlessChrome); Safari and
  // Firefox never do
  return /Chrome\//.test(navigator.userAgent);
}

/** @returns true when boot may proceed; false after rendering a notice */
export function browserGate(): boolean {
  if (new URLSearchParams(window.location.search).get('nogate') === '1') return true;

  if (isMobileDevice()) {
    failLoud('A computer is required', [
      'LAAS pushes desktop-class GPU work through WebGPU — phone and tablet',
      'browsers are not supported.',
      '',
      'Please revisit from a desktop or laptop running Google Chrome.',
    ]);
    return false;
  }

  if (!isChromiumBrowser()) {
    failLoud('Google Chrome is required', [
      'LAAS is built and tested against Chrome’s WebGPU implementation.',
      'Safari and Firefox currently cannot run it.',
      '',
      'Please open this page in Google Chrome 113 or newer.',
      'Chromium-based browsers (Edge, Brave, Arc, Opera) should also work.',
    ]);
    return false;
  }

  if (!('gpu' in navigator) || !navigator.gpu) {
    failLoud('WebGPU is unavailable in this browser', [
      'You are on a Chromium browser, but it does not expose WebGPU.',
      '',
      'Things to check:',
      '  • Update Chrome — WebGPU needs version 113 or newer.',
      '  • Settings → System → “Use hardware acceleration” must be ON',
      '    (relaunch after changing it).',
      '  • chrome://gpu should list WebGPU as “Hardware accelerated”.',
      '  • On Linux, recent Chrome may need chrome://flags/#enable-vulkan',
      '    or launching with --enable-features=Vulkan.',
    ]);
    return false;
  }

  return true;
}
