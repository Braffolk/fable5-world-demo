/**
 * G5 PERF A/B — connected-junction tree rework OFF (?nojunctions, legacy open tubes)
 * vs ON (default). Two-boot A/B (the flag is build-time): boot a config, teleport
 * across matched poses, measure honest gpuWall via __laas.measureFrames, read
 * nanite.visTris/dagTris/visClusters. Alternated rounds expose cross-boot drift.
 *
 * Canonical forest: 200k trees, dpr=1.5 (retina backing ~2268×1473). Serial; one
 * page at a time. gpuWall ONLY (per-pass profiler is a harness ARTIFACT here).
 *
 *   BASE=http://localhost:5180/ npx tsx tools/probe-treeconnect-ab.ts
 */
import { launchWebGPU, laasUrl } from "./launch";
import type { Page } from "playwright";

const BASE = process.env.BASE ?? "http://localhost:5180/";
const ROUNDS = Number(process.env.ROUNDS ?? "2");
const FRAMES = Number(process.env.FRAMES ?? "48");
const WARMUP = Number(process.env.WARMUP ?? "30");

type Pose = { name: string; p: [number, number, number]; yaw: number; pitch: number };
const POSES: Pose[] = [
  { name: "eye", p: [0, 2, 0], yaw: 0.6, pitch: -0.02 },
  { name: "oblique", p: [0, 40, 40], yaw: 0, pitch: -0.35 },
  { name: "aerial", p: [0, 150, 0], yaw: 0, pitch: -1.45 },
];

const median = (a: number[]): number => {
  const s = a.filter(Number.isFinite).sort((x, y) => x - y);
  return s.length ? (s[Math.floor(s.length / 2)] as number) : NaN;
};
const p05 = (a: number[]): number => {
  const s = a.filter(Number.isFinite).sort((x, y) => x - y);
  return s.length ? (s[Math.floor(s.length * 0.95)] as number) : NaN; // p0.95 of ms = the slow frames
};

interface Sample {
  gpu: number[];
  visTris: number;
  dagTris: number;
  visClusters: number;
  cap: number;
}

async function bootConfig(page: Page, junctionsOff: boolean): Promise<void> {
  const extra: Record<string, string> = {
    trees: "200000",
    nanite: "1",
    dpr: "1.5",
  };
  if (junctionsOff) extra.nojunctions = "1";
  const url = laasUrl({ scene: "forest", freeze: false, hud: false, extra }, BASE);
  const t0 = Date.now();
  await page.goto(url, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(
    () => window.__laas && (window.__laas.ready || window.__laas.error != null),
    undefined,
    { timeout: 300000, polling: 250 },
  );
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error(`fatal boot: ${err}`);
  // thermal/stream warmup so the workload is steady before any pose measure
  await page.evaluate(async () => {
    if (window.__laas.settle) await window.__laas.settle(120);
  });
  console.log(
    `  [boot ${junctionsOff ? "OFF/legacy" : "ON/rework"}] ${((Date.now() - t0) / 1000).toFixed(1)} s`,
  );
}

async function measurePose(page: Page, pose: Pose): Promise<Sample> {
  const frames = (await page.evaluate(
    async ({ pose, FRAMES, WARMUP }) => {
      window.__laas.setPose!({ p: pose.p, yaw: pose.yaw, pitch: pose.pitch });
      if (window.__laas.settle) await window.__laas.settle(20);
      const fs = await window.__laas.measureFrames!({ frames: FRAMES, warmup: WARMUP });
      return fs.map((f) => ({
        gpu: f.gpuWallMs,
        cap: f.capSuspect ? 1 : 0,
        c: f.counters,
      }));
    },
    { pose, FRAMES, WARMUP },
  )) as { gpu: number; cap: number; c: Record<string, number> }[];

  const good = frames.filter((f) => f.cap === 0);
  const use = good.length >= frames.length / 2 ? good : frames;
  const last = frames[frames.length - 1]!.c;
  return {
    gpu: use.map((f) => f.gpu),
    visTris: last["nanite.visTris"] ?? -1,
    dagTris: last["nanite.dagTris"] ?? -1,
    visClusters: last["nanite.visClusters"] ?? -1,
    cap: frames.length - good.length,
  };
}

async function runConfig(page: Page, junctionsOff: boolean): Promise<Record<string, Sample>> {
  await bootConfig(page, junctionsOff);
  const out: Record<string, Sample> = {};
  for (const pose of POSES) out[pose.name] = await measurePose(page, pose);
  return out;
}

async function main(): Promise<void> {
  const { browser } = await launchWebGPU();
  // retina backing: CSS 1512×982 × dpr 1.5 ≈ 2268×1473
  const page = await browser.newPage({
    viewport: { width: 1512, height: 982 },
    deviceScaleFactor: 1.5,
  });
  page.on("pageerror", (e) => console.error("[pageerror]", e.message));

  // accumulate per (config,pose) across rounds
  const acc: Record<string, Record<string, number[]>> = { ON: {}, OFF: {} };
  const counters: Record<string, Record<string, Sample>> = { ON: {}, OFF: {} };

  for (let r = 0; r < ROUNDS; r++) {
    // alternate order each round to cancel any monotone drift
    const order: ("ON" | "OFF")[] = r % 2 === 0 ? ["ON", "OFF"] : ["OFF", "ON"];
    console.log(`\n=== round ${r + 1}/${ROUNDS} order=${order.join(",")} ===`);
    for (const cfg of order) {
      const res = await runConfig(page, cfg === "OFF");
      for (const pose of POSES) {
        (acc[cfg][pose.name] ??= []).push(...res[pose.name].gpu);
        counters[cfg][pose.name] = res[pose.name]; // keep last round's counters
        const s = res[pose.name];
        console.log(
          `   ${cfg.padEnd(3)} ${pose.name.padEnd(8)} gpuWall med=${median(s.gpu).toFixed(2)} p95=${p05(s.gpu).toFixed(2)} ` +
            `visTris=${s.visTris} visCl=${s.visClusters} cap=${s.cap}`,
        );
      }
    }
  }

  console.log(`\n===================== A/B SUMMARY (gpuWall ms) =====================`);
  console.log(`pose      | OFF med  ON med  | Δ(on-off)  ratio | OFF visTris  ON visTris  | OFF visCl  ON visCl`);
  for (const pose of POSES) {
    const offMed = median(acc.OFF[pose.name] ?? []);
    const onMed = median(acc.ON[pose.name] ?? []);
    const cOff = counters.OFF[pose.name]!;
    const cOn = counters.ON[pose.name]!;
    const ratio = onMed / offMed;
    console.log(
      `${pose.name.padEnd(9)} | ${offMed.toFixed(2).padStart(7)} ${onMed.toFixed(2).padStart(6)}  | ` +
        `${(onMed - offMed).toFixed(2).padStart(8)}  ${ratio.toFixed(3)} | ` +
        `${String(cOff.visTris).padStart(10)} ${String(cOn.visTris).padStart(10)}  | ` +
        `${String(cOff.visClusters).padStart(8)} ${String(cOn.visClusters).padStart(8)}`,
    );
  }
  console.log(`\n(ratio<1 ⇒ rework ON is FASTER. visTris drop far-dominant poses ⇒ far cut to roots.)`);
  await browser.close();
}

main().catch((e) => {
  console.error("[ab] FAILED:", e instanceof Error ? (e.stack ?? e.message) : e);
  process.exit(1);
});
