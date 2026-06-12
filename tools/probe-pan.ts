/**
 * N2 disocclusion probe (F13) — continuous hard STRAFE + yaw drift over the
 * nanite debug view. Pure rotation has no parallax (it reveals nothing
 * behind occluders — first version's mistake); lateral translation through
 * the forest reveals fresh geometry behind every trunk each frame, which
 * the phase-1 prev-HZB test rejects — phase 2 must bring it back THE SAME
 * FRAME. The camera pitches down so frames contain no sky below the guard
 * band (baseline verified zero): any black pixel mid-pan = disocclusion
 * hole.
 *
 * Gate: zero hole pixels in every mid-pan frame. Negative control:
 * PAN_EXTRA="phase2=0" must FAIL (proves the gate detects what phase 2
 * fixes).
 *
 *   npx tsx tools/probe-pan.ts [shots=10] [strafePerFrame=0.5] (server :5173)
 */

import { launchWebGPU, laasUrl } from './launch';
import sharp from 'sharp';

const W = 1280;
const H = 720;
/** ignore the top band (ridgeline sky can sneak in while rotating) */
const SKY_GUARD_ROWS = Math.floor(H * 0.3);

interface CamPoseLike {
  p: [number, number, number];
  yaw: number;
  pitch: number;
  fov?: number;
}
interface FlyWindow {
  __laas?: { ready?: boolean };
  __laasFly?: {
    yaw: number;
    pitch: number;
    setMode(m: 'walk' | 'fly'): void;
    getPose(): CamPoseLike;
    setPose(p: CamPoseLike): void;
  };
  __laasDbg?: { engine?: { onUpdate(fn: () => void): void } };
  __panOn?: boolean;
}

async function blackPixels(path: string): Promise<number> {
  const img = await sharp(path).raw().toBuffer({ resolveWithObject: true });
  const ch = img.info.channels;
  let black = 0;
  for (let y = SKY_GUARD_ROWS; y < img.info.height; y++) {
    for (let x = 0; x < img.info.width; x++) {
      const i = (y * img.info.width + x) * ch;
      if ((img.data[i] ?? 0) + (img.data[i + 1] ?? 0) + (img.data[i + 2] ?? 0) < 24) black++;
    }
  }
  return black;
}

async function main(): Promise<void> {
  const shots = Number(process.argv[2] ?? '10');
  const strafePerFrame = Number(process.argv[3] ?? '0.5');
  const { browser } = await launchWebGPU();
  const errors: string[] = [];
  let settledBlack = 0;
  let worst = 0;
  try {
    const page = await browser.newPage();
    await page.setViewportSize({ width: W, height: H });
    page.on('pageerror', (e) => errors.push(String(e)));
    const extra: Record<string, string> = { nanite: '1', nanitedbg: 'flat' };
    for (const kv of (process.env['PAN_EXTRA'] ?? '').split('&')) {
      const [k, v] = kv.split('=');
      if (k && v !== undefined) extra[k] = v;
    }
    await page.goto(laasUrl({ scene: 'world', hud: false, extra }), {
      waitUntil: 'domcontentloaded',
    });
    await page.waitForFunction(
      () => (window as FlyWindow).__laas?.ready === true,
      undefined,
      { timeout: 240_000 },
    );
    // fly mode, look down (no sky below the guard band), settle
    await page.evaluate(() => {
      const w = window as FlyWindow;
      w.__laasFly?.setMode('fly');
      if (w.__laasFly) w.__laasFly.pitch = -0.5;
    });
    await new Promise((r) => setTimeout(r, 1200));
    await page.screenshot({ path: '/tmp/pan-settled.png' });
    settledBlack = await blackPixels('/tmp/pan-settled.png');
    console.log(`settled frame: ${settledBlack} black px under the sky guard`);

    // continuous hard strafe (view-space right) + yaw drift, every frame —
    // translation gives the parallax that actually disoccludes
    await page.evaluate((step) => {
      const w = window as FlyWindow;
      w.__panOn = true;
      w.__laasDbg?.engine?.onUpdate(() => {
        const fly = w.__laasFly;
        if (!w.__panOn || !fly) return;
        const pose = fly.getPose();
        pose.p[0] += Math.cos(pose.yaw) * step;
        pose.p[2] -= Math.sin(pose.yaw) * step;
        pose.yaw += 0.02;
        fly.setPose(pose);
      });
    }, strafePerFrame);

    let p2Total = 0;
    for (let s = 0; s < shots; s++) {
      await new Promise((r) => setTimeout(r, 140));
      const p = `/tmp/pan-mid${s}.png`;
      await page.screenshot({ path: p });
      const holes = Math.max(0, (await blackPixels(p)) - settledBlack);
      worst = Math.max(worst, holes);
      const p2 = await page.evaluate(() => {
        const dbg = (
          window as unknown as {
            __laasDbg?: { engine?: { stats?: { counters?: Record<string, number> } } };
          }
        ).__laasDbg;
        return dbg?.engine?.stats?.counters?.['nanite.p2'] ?? 0;
      });
      p2Total += p2;
      console.log(`mid-pan ${s}: ${holes} hole px (phase-2 appends ${p2})`);
    }
    console.log(
      `phase-2 appended ${p2Total} clusters across sampled frames — the late-reveal path is ${p2Total > 0 ? 'LIVE' : 'idle (conservative HZB caught everything in phase 1)'}`,
    );
    await page.evaluate(() => {
      (window as FlyWindow).__panOn = false;
    });
  } finally {
    await browser.close();
  }
  const pct = ((worst / (W * (H - SKY_GUARD_ROWS))) * 100).toFixed(4);
  const pass = worst === 0 && errors.length === 0;
  console.log(
    `[probe-pan] ${pass ? 'PASS' : 'FAIL'} — worst ${worst} hole px (${pct}% of analysed area)${errors.length ? `, ${errors.length} page errors` : ''}`,
  );
  if (!pass) process.exit(1);
}

void main();
