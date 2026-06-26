/**
 * G5 FAR-FIELD SHOTDIFF — rework OFF (?nojunctions) vs ON, matched poses, 200k
 * forest. Far field should be ~IDENTICAL (collapsed coarse roots are sub-pixel);
 * NEAR intentionally differs (more junction detail). Saves OFF/ON PNGs per pose
 * and an amplified diff; reports mean abs pixel diff over the full frame and a
 * FAR crop (top band = horizon/far trees). Serial: one page at a time.
 *
 *   BASE=http://localhost:5180/ npx tsx tools/probe-treeconnect-shotdiff.ts
 */
import { launchWebGPU, laasUrl } from "./launch";
import type { Page } from "playwright";
import { mkdirSync } from "node:fs";
import sharp from "sharp";

const BASE = process.env.BASE ?? "http://localhost:5180/";
const OUT = "shots/treeconnect";

type Pose = { name: string; p: [number, number, number]; yaw: number; pitch: number };
const POSES: Pose[] = [
  { name: "eye", p: [0, 2, 0], yaw: 0.6, pitch: -0.02 },
  { name: "oblique", p: [0, 40, 40], yaw: 0, pitch: -0.35 },
  { name: "aerial", p: [0, 150, 0], yaw: 0, pitch: -1.45 },
];

async function boot(page: Page, off: boolean): Promise<void> {
  const extra: Record<string, string> = { trees: "200000", nanite: "1", dpr: "1.5" };
  if (off) extra.nojunctions = "1";
  await page.goto(laasUrl({ scene: "forest", freeze: false, hud: false, extra }, BASE), {
    waitUntil: "domcontentloaded",
  });
  await page.waitForFunction(
    () => window.__laas && (window.__laas.ready || window.__laas.error != null),
    undefined,
    { timeout: 300000, polling: 250 },
  );
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error("boot err: " + err);
  await page.evaluate(async () => {
    if (window.__laas.settle) await window.__laas.settle(120);
  });
}

async function shoot(page: Page, pose: Pose, tag: string): Promise<string> {
  await page.evaluate(async (po) => {
    window.__laas.setPose!({ p: po.p, yaw: po.yaw, pitch: po.pitch });
    if (window.__laas.settle) await window.__laas.settle(40);
  }, pose);
  const path = `${OUT}/${pose.name}-${tag}.png`;
  await page.screenshot({ path });
  return path;
}

async function meanAbsDiff(a: string, b: string, cropTopFrac?: number): Promise<number> {
  let ia = sharp(a).removeAlpha();
  let ib = sharp(b).removeAlpha();
  const meta = await sharp(a).metadata();
  if (cropTopFrac && meta.width && meta.height) {
    const h = Math.floor(meta.height * cropTopFrac);
    ia = ia.extract({ left: 0, top: 0, width: meta.width, height: h });
    ib = ib.extract({ left: 0, top: 0, width: meta.width, height: h });
  }
  const [ba, bb] = await Promise.all([
    ia.raw().toBuffer(),
    ib.raw().toBuffer(),
  ]);
  let sum = 0;
  const n = Math.min(ba.length, bb.length);
  for (let i = 0; i < n; i++) sum += Math.abs(ba[i]! - bb[i]!);
  return sum / n; // 0..255
}

async function main(): Promise<void> {
  mkdirSync(OUT, { recursive: true });
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width: 1512, height: 982 }, deviceScaleFactor: 1.5 });
  page.on("pageerror", (e) => console.error("[pageerror]", e.message));

  console.log("[shotdiff] boot OFF (legacy)...");
  await boot(page, true);
  const offShots: Record<string, string> = {};
  for (const po of POSES) offShots[po.name] = await shoot(page, po, "off");

  console.log("[shotdiff] boot ON (rework)...");
  await boot(page, false);
  const onShots: Record<string, string> = {};
  for (const po of POSES) onShots[po.name] = await shoot(page, po, "on");

  await browser.close();

  console.log(`\n===== shotdiff (mean abs pixel diff, 0..255) =====`);
  for (const po of POSES) {
    const full = await meanAbsDiff(offShots[po.name]!, onShots[po.name]!);
    const farTop = await meanAbsDiff(offShots[po.name]!, onShots[po.name]!, 0.4); // top 40% = far/horizon band
    // amplified diff image for eyeballing
    const diffPath = `${OUT}/${po.name}-DIFF.png`;
    await sharp(offShots[po.name]!)
      .removeAlpha()
      .composite([{ input: onShots[po.name]!, blend: "difference" }])
      .linear(6, 0)
      .toFile(diffPath);
    console.log(
      `${po.name.padEnd(8)} full=${full.toFixed(3)}  farTop40%=${farTop.toFixed(3)}  → ${diffPath}`,
    );
  }
  console.log(`\n(far/aerial poses: expect SMALL diff = far ~identical; eye: larger = intended near detail.)`);
}

main().catch((e) => {
  console.error("[shotdiff] FAILED:", e instanceof Error ? (e.stack ?? e.message) : e);
  process.exit(1);
});
