/**
 * N8-D2 Stage 2e DIAGNOSTIC — is the stride-1 hf vertex buffer populated on the
 * CPU side after the clip streamer attaches its boot tiles? Boots the clip default,
 * reaches into the live registry (__laasDbg.engine.naniteRegistry), and reports
 * whether reg.debug().arrays.hfVerts has non-zero data where resident tiles wrote it.
 * CPU-populated + GPU-flat ⇒ an upload/binding bug; CPU-zero ⇒ an attach bug.
 *
 *   npx tsx tools/probe-hfdiag.ts
 */
import { laasUrl, launchWebGPU } from './launch';

async function main(): Promise<void> {
  const { browser } = await launchWebGPU();
  const page = await browser.newPage({ viewport: { width: 320, height: 200 }, deviceScaleFactor: 1 });
  page.on('console', (m) => {
    const t = m.text();
    if (t.includes('terrain DAG') && t.includes('POOL')) {
      const seg = t.match(/terrain DAG CLIPMAP[^;]*?POOL \d+×\([^)]*\)[^;]*/);
      if (seg) console.log(`  [boot] ${seg[0].trim()}`);
    }
  });
  const url = laasUrl({ scene: 'world', width: 320, height: 200, freeze: true, extra: { nanite: '1' } });
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__laas && (window.__laas.ready || window.__laas.error != null), undefined, {
    timeout: 180000,
    polling: 250,
  });
  const err = await page.evaluate(() => window.__laas.error ?? null);
  if (err) throw new Error(`fatal boot: ${err}`);
  await page.evaluate(async () => {
    if (window.__laas.settle) await window.__laas.settle(12);
  });

  const diag = await page.evaluate(async () => {
    const dbg = (window as unknown as {
      __laasDbg?: { engine?: { naniteRegistry?: unknown; renderer?: unknown } };
    }).__laasDbg;
    const reg = dbg?.engine?.naniteRegistry as
      | {
          debug(): { arrays: { hfVerts: Uint32Array; verts: Uint32Array }; attrs: { hfVerts: unknown } };
          tilePoolSlotCount: number;
          tileSlotBase(slot: number): { vert: number; tri: number; cluster: number };
          bytes(): { verts: number; hfVerts: number };
        }
      | undefined;
    if (!reg) return { error: 'no naniteRegistry on __laasDbg.engine' };
    const arrays = reg.debug().arrays;
    const hf = arrays.hfVerts;
    // GPU READBACK: read gpu.hfVerts at slot 1's base (vBase 95016) — if the GPU
    // buffer has the data the CPU array has, the upload worked; if it's zero, it didn't.
    const renderer = dbg?.engine?.renderer as
      | { getArrayBufferAsync(attr: unknown, x: null, off: number, len: number): Promise<ArrayBuffer> }
      | undefined;
    let gpuReadback: { vBase: number; cpu: number[]; gpu: number[] } | { error: string } = {
      error: 'no renderer',
    };
    if (renderer) {
      try {
        const vBase = reg.tileSlotBase(1).vert;
        const byteOff = vBase * 4;
        const ab = await renderer.getArrayBufferAsync(reg.debug().attrs.hfVerts, null, byteOff, 32);
        const gpuWords = Array.from(new Uint32Array(ab));
        const cpuWords: number[] = [];
        for (let k = 0; k < 8; k++) cpuWords.push(hf[vBase + k] ?? -1);
        gpuReadback = { vBase, cpu: cpuWords, gpu: gpuWords };
      } catch (e) {
        gpuReadback = { error: e instanceof Error ? e.message : String(e) };
      }
    }
    // global stats
    let nonZero = 0;
    let maxIdx = 0;
    for (let i = 0; i < hf.length; i++) {
      if (hf[i] !== 0) {
        nonZero++;
        if (i > maxIdx) maxIdx = i;
      }
    }
    // per-slot: how many of the first 16 words are non-zero at each slot base
    const slots = reg.tilePoolSlotCount;
    const perSlot: { slot: number; vBase: number; nz16: number; first: number }[] = [];
    for (let s = 0; s < Math.min(slots, 8); s++) {
      const vBase = reg.tileSlotBase(s).vert;
      let nz = 0;
      for (let k = 0; k < 16; k++) if (hf[vBase + k] !== 0) nz++;
      perSlot.push({ slot: s, vBase, nz16: nz, first: hf[vBase] ?? -1 });
    }
    return {
      hfLen: hf.length,
      hfNonZero: nonZero,
      hfMaxNonZeroIdx: maxIdx,
      vertsLen: arrays.verts.length,
      bytes: reg.bytes(),
      slots,
      perSlot,
      gpuReadback,
    };
  });
  console.log('[hfdiag]', JSON.stringify(diag, null, 2));
  await browser.close();
}
main().catch((e) => {
  console.error('[hfdiag] FAILED:', e instanceof Error ? e.message : e);
  process.exit(1);
});
