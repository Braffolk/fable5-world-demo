/**
 * N8-HIC / perf-puzzle — IS THE FLOOD PRIMITIVE-BOUND? Boots the flood frame
 * (shot 7 forest interior, `?naniteleaf=1`, occl ON = default) and sweeps the cut
 * threshold τ (loderr) coarse→fine, recording frame-time AND visible-cluster count
 * AND the per-pass raster ms TOGETHER. The relationship decides the root cause:
 *
 *   frame ∝ visClusters (linear)  ⇒ RASTER is primitive-count-bound = OVER-EMISSION
 *     (the small-triangle/overdraw regime; far crowns emit ~1 cluster each → way more
 *     visible primitives than screen pixels). Fix = reduce visible primitives.
 *   frame ~flat as visClusters drops ⇒ fixed-overhead / per-pixel bound. Different fix.
 *
 *   LEAFDENSITY=800 npx tsx tools/probe-floodtau.ts   # needs dev server on :5173
 */

import { launchWebGPU, laasUrl } from './launch';

const SHOT = process.env.SHOT ?? '7';
const LEAFDENSITY = process.env.LEAFDENSITY ?? '800';
const TAUS = [32, 16, 8, 4, 2, 1];

declare global {
  interface Window {
    __laasNanite?: { setTau?(v: number): void; tau?(): number };
  }
}

async function main(): Promise<void> {
  const width = 1280;
  const height = 720;
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.error('[pageerror]', e.message));
  const url = laasUrl({
    scene: 'world',
    width,
    height,
    freeze: false,
    extra: {
      nanite: '1',
      naniteleaf: '1',
      naniteleafdensity: LEAFDENSITY,
      nanitedterrain: '0',
      nanshadow: '0',
      occl: '1',
      shot: SHOT,
      loderr: '1',
    },
  });
  console.log(`[floodtau] SHOT=${SHOT} LEAFDENSITY=${LEAFDENSITY} occl ON — px=${width * height}`);
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(
    () => window.__laas && (window.__laas.ready || window.__laas.error != null),
    undefined,
    { timeout: 240000, polling: 250 },
  );
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error(`fatal boot: ${err}`);

  const median = (a: number[]): number => {
    const s = [...a].sort((x, y) => x - y);
    return s[Math.floor(s.length / 2)] ?? 0;
  };

  const sampleAt = async (tau: number): Promise<Record<string, number>> => {
    await page.evaluate((t) => window.__laasNanite?.setTau?.(t), tau);
    const acc = new Map<string, number[]>();
    for (let i = 0; i < 26; i++) {
      const f = (await page.evaluate(async () => {
        if (window.__laas.settle) await window.__laas.settle(6);
        const st = window.__laas.stats;
        const g = st?.gpuPasses ?? {};
        const c = st?.counters ?? {};
        return {
          frameMs: st?.frameMs ?? 0,
          payload: g['c.nanRasterPayload'] ?? 0,
          depth: g['c.nanRasterDepth'] ?? 0,
          clusterCull: g['c.nanClusterCull'] ?? 0,
          vis: c['nanite.visClusters'] ?? -1,
          trisK: c['nanite.trisK'] ?? -1,
          hwTris: c['nanite.hwTris'] ?? -1,
        };
      })) as Record<string, number>;
      if (i >= 8) for (const [k, v] of Object.entries(f)) (acc.get(k) ?? acc.set(k, []).get(k))?.push(v);
    }
    const out: Record<string, number> = {};
    for (const [k, a] of acc) out[k] = median(a);
    return out;
  };

  console.log('     τ |   frameMs | visClusters | trisK | hwTris |  payload   depth  clCull (ms)  | frame/Mcl');
  const rows: { tau: number; r: Record<string, number> }[] = [];
  for (const tau of TAUS) {
    const r = await sampleAt(tau);
    rows.push({ tau, r });
    const perMcl = r.vis > 0 ? (r.frameMs / (r.vis / 1e6)).toFixed(1) : '—';
    console.log(
      `  ${String(tau).padStart(4)} | ${r.frameMs.toFixed(1).padStart(8)} | ${String(r.vis).padStart(11)} | ${String(r.trisK).padStart(5)} | ${String(r.hwTris).padStart(6)} | ${r.payload.toFixed(2).padStart(6)}  ${r.depth.toFixed(2).padStart(6)}  ${r.clusterCull.toFixed(2).padStart(6)}      | ${perMcl}`,
    );
  }

  // linearity check: correlate Δframe with ΔvisClusters across the sweep
  const coarse = rows[0]?.r;
  const fine = rows[rows.length - 1]?.r;
  if (coarse && fine && coarse.vis > 0 && fine.vis > 0) {
    const dVis = fine.vis - coarse.vis;
    const dFrame = fine.frameMs - coarse.frameMs;
    const dPayload = fine.payload - coarse.payload;
    const msPerMcl = dVis !== 0 ? (dFrame / (dVis / 1e6)).toFixed(1) : '—';
    const payPerMcl = dVis !== 0 ? (dPayload / (dVis / 1e6)).toFixed(1) : '—';
    console.log(
      `[floodtau] τ32→τ1: ΔvisClusters ${dVis} | Δframe ${dFrame.toFixed(1)}ms (${msPerMcl} ms / Mcluster) | Δpayload ${dPayload.toFixed(2)}ms (${payPerMcl} ms / Mcluster)`,
    );
    console.log(
      `[floodtau] VERDICT: ${Math.abs(dFrame) > 5 && Math.abs(dVis) > 1e5 ? 'frame TRACKS cluster count ⇒ RASTER PRIMITIVE-BOUND (over-emission/overdraw)' : 'frame ~flat vs cluster count ⇒ NOT primitive-bound (fixed-overhead/per-pixel)'}`,
    );
  }
  await browser.close();
}

main().catch((e) => {
  console.error('[floodtau] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
