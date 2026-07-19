// Motion-triggered persistent-hole repro harness (bog preview, eye level).
// Phases: boot → baseline down-sweep → aggressive low-alt roam (seeded) with
// mid-motion down-captures → STOP + hold → repeated post-stop down-sweeps
// (persistent-hole check) → state dump.
//
// HOLE METRIC: camera at 1.6 m looking ~straight down (pitch -1.15): every ray
// hits ground within a few metres, so ANY sky/background-coloured pixel means
// terrain geometry is MISSING there (a hole). Robust against dark landcover
// (the old dark-pixel metric false-positived on forest-floor material).
import { chromium } from 'playwright';
import { mkdirSync, writeFileSync } from 'node:fs';
import sharp from 'sharp';

const OUT = process.env.OUT ?? '/private/tmp/claude-501/-Users-sebastian-IdeaProjects-fable-demo2/6c83ffba-f52c-4da4-9607-c956267d5fae/scratchpad/defake-qa';
const TAG = process.env.TAG ?? 'run';
const SEED = Number(process.env.SEED ?? 7);
const ROAM_FRAMES = Number(process.env.ROAM_FRAMES ?? 1400);
const URL = process.env.URL ??
  'http://localhost:5199/?scene=world&src=estonia&dataurl=http://localhost:8790&grass=0&previewclip=0&alt=1.6&x=171776&z=206080&mschart=0';
mkdirSync(OUT, { recursive: true });

// fine core box (lod -2 published chunks 1340-1343 x 1608-1611, 128 m each)
const BOX = { x0: 171520, x1: 172032, z0: 205824, z1: 206336 };
const INSET = 24;

function mulberry32(a) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rnd = mulberry32(SEED);

const DOWN_PITCH = -1.15;
// sky/void classifier: blue-dominant OR bright washed haze — never true of bog
// ground (olive/brown/dark green) or logs/stones.
function isVoid(r, g, b) {
  return (b > 110 && b > g + 8 && g >= r) || (r > 195 && g > 205 && b > 215);
}
async function holeMetric(buf) {
  const img = sharp(buf);
  const { data, info } = await img.raw().toBuffer({ resolveWithObject: true });
  const W = info.width, H = info.height, ch = info.channels;
  // skip the top-left fps chip (90x40) and top 5% (safety)
  const top = Math.floor(H * 0.05);
  let voidPix = 0, tot = 0, minX = 1e9, minY = 1e9, maxX = -1, maxY = -1;
  for (let y = top; y < H; y++) {
    for (let x = 0; x < W; x++) {
      if (x < 100 && y < 45) continue;
      const o = (y * W + x) * ch;
      tot++;
      if (isVoid(data[o], data[o + 1], data[o + 2])) {
        voidPix++;
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (y < minY) minY = y; if (y > maxY) maxY = y;
      }
    }
  }
  return { holePct: (100 * voidPix) / tot, bbox: maxX >= 0 ? [minX, minY, maxX, maxY] : null };
}

const COUNTERS = [
  'tree.fringe', 'tree.cam.level', 'tree.tx.live', 'tree.want.deferred', 'tree.refine.commit',
  'tree.refine.cancel', 'tree.merge', 'tree.stale.drop', 'tree.teleports',
  'stream.tiles.freeslots', 'stream.mailbox.depth', 'stream.bake.cache', 'stream.bake.built',
  'stream.bake.inflight', 'stream.fetch.total', 'stream.fetch.err', 'stream.fetch.inflight',
  'stream.scrolls', 'stream.ram.mb', 'stream.bucket.packets', 'stream.origin.rebases',
  'stream.win.degraded', 'stream.tiles.tainted', 'stream.heal.ok', 'stream.tile.refresh',
];

async function main() {
  const browser = await chromium.launch({ headless: true, channel: 'chromium' });
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1 });
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('response', (r) => { if (r.status() >= 400) console.log(`[http ${r.status()}]`, r.url().slice(-90)); });
  page.on('console', (m) => {
    const t = m.text();
    if (m.type() === 'error' || /WebGPU uncaptured|Invalid generated code|Boot failed/.test(t)) errors.push('console: ' + t.slice(0, 400));
    if (/pool dry|over slot cap|plane is zero|bake failed|bake worker failed|boot leaves|invariant|exceeds/.test(t)) console.log('[app] ' + t.slice(0, 400));
  });

  page.on('crash', () => {
    console.log('PAGE CRASHED');
    errors.push('page crashed');
  });
  console.log('goto', URL);
  await page.goto(URL, { waitUntil: 'domcontentloaded', timeout: 300000 });
  console.log('goto done — waiting for ready');
  // manual ready-poll: unlike waitForFunction this cannot silently hang across a
  // renderer crash — a dead page makes evaluate throw, which we surface.
  const t0 = Date.now();
  for (;;) {
    if (Date.now() - t0 > 600000) throw new Error('ready poll timed out (600s)');
    let st = null;
    try {
      st = await page.evaluate(() => ({
        ready: window.__laas?.ready === true,
        err: typeof window.__laas?.error === 'string' && window.__laas.error.length > 0 ? window.__laas.error : null,
        prog: window.__laas?.progressMsg ?? '',
      }));
    } catch (e) {
      console.log('ready poll evaluate failed:', String(e).slice(0, 160));
    }
    if (st?.err) { console.log('BOOT ERROR:', st.err); await browser.close(); process.exit(1); }
    if (st?.ready) break;
    if (st && Math.floor((Date.now() - t0) / 15000) !== Math.floor((Date.now() - t0 - 1000) / 15000)) console.log(`  boot ${(Date.now() - t0) / 1000 | 0}s: ${st.prog}`);
    await new Promise((r) => setTimeout(r, 1000));
  }
  console.log('ready after', ((Date.now() - t0) / 1000).toFixed(0), 's');
  const bootErr = await page.evaluate(() => window.__laas.error);
  if (bootErr) { console.log('BOOT ERROR:', bootErr); await browser.close(); process.exit(1); }
  await page.evaluate(() => window.__laas.flyCamEnabled?.(false));
  await page.evaluate((n) => window.__laas.settle?.(n), 60);

  const counters = () => page.evaluate((keys) => {
    const c = window.__laas.stats?.counters ?? {};
    const out = {};
    for (const k of keys) if (c[k] !== undefined) out[k] = c[k];
    return out;
  }, COUNTERS);

  // down-sweep: 4 yaws looking nearly straight down (yaw still shifts the view
  // cone footprint a bit), screenshot each, return worst metric
  async function sweep(x, z, name, save = true) {
    let worst = { holePct: -1 };
    for (let k = 0; k < 4; k++) {
      const yaw = (k * Math.PI) / 2;
      await page.evaluate((a) => {
        const g = window.__laas.groundProbe?.(a.x, a.z)?.ground ?? 0;
        window.__laas.setPose({ p: [a.x, g + 1.6, a.z], yaw: a.yaw, pitch: a.pitch });
      }, { x, z, yaw, pitch: DOWN_PITCH });
      await page.evaluate((n) => window.__laas.settle?.(n), 6);
      const buf = await page.screenshot();
      const m = await holeMetric(buf);
      if (save || m.holePct > 0.2) writeFileSync(`${OUT}/${TAG}-${name}-yaw${k}.png`, buf);
      if (m.holePct > worst.holePct) worst = { ...m, yaw, x, z, shot: `${TAG}-${name}-yaw${k}.png` };
      console.log(`  ${name} yaw${k}: hole=${m.holePct.toFixed(3)}% bbox=${JSON.stringify(m.bbox)}`);
    }
    return worst;
  }

  const cx = (BOX.x0 + BOX.x1) / 2, cz = (BOX.z0 + BOX.z1) / 2;
  console.log('=== BASELINE down-sweep at core center ===');
  const base = await sweep(cx, cz, 'baseline');
  console.log('baseline worst hole%:', base.holePct.toFixed(3), JSON.stringify(await counters()));

  // ---- aggressive low-alt roam (in-page segments of 50 frames) ----
  console.log(`=== ROAM ${ROAM_FRAMES} frames (seed ${SEED}) ===`);
  let px = cx, pz = cz;
  let wx = cx, wz = cz, speed = 20, pitch = -0.3, bob = 0;
  let midWorst = { holePct: -1 };
  for (let f = 0; f < ROAM_FRAMES; ) {
    const seg = [];
    for (let s = 0; s < 50 && f < ROAM_FRAMES; s++, f++) {
      const dx = wx - px, dz = wz - pz;
      const d = Math.hypot(dx, dz);
      if (d < 6) {
        const edge = rnd() < 0.25;
        if (edge) {
          const side = (rnd() * 4) | 0;
          wx = side === 0 ? BOX.x0 + INSET : side === 1 ? BOX.x1 - INSET : BOX.x0 + INSET + rnd() * (BOX.x1 - BOX.x0 - 2 * INSET);
          wz = side === 2 ? BOX.z0 + INSET : side === 3 ? BOX.z1 - INSET : BOX.z0 + INSET + rnd() * (BOX.z1 - BOX.z0 - 2 * INSET);
        } else {
          wx = BOX.x0 + INSET + rnd() * (BOX.x1 - BOX.x0 - 2 * INSET);
          wz = BOX.z0 + INSET + rnd() * (BOX.z1 - BOX.z0 - 2 * INSET);
        }
        speed = 4 + rnd() * 41; // 4..45 m/s
        pitch = -(0.12 + rnd() * 0.4);
        if (rnd() < 0.3) { const t = wx; wx = px - (t - px) * 0.5; wz = pz - (wz - pz) * 0.5; }
        wx = Math.min(Math.max(wx, BOX.x0 + INSET), BOX.x1 - INSET);
        wz = Math.min(Math.max(wz, BOX.z0 + INSET), BOX.z1 - INSET);
      }
      const step = Math.min(speed / 30, Math.max(d, 0.001));
      px += (dx / (d || 1)) * step;
      pz += (dz / (d || 1)) * step;
      bob += 0.11;
      const yaw = Math.atan2(-(wx - px), -(wz - pz));
      seg.push([px, pz, yaw, pitch, 1.6 + Math.max(0, Math.sin(bob)) * 0.7]);
    }
    await page.evaluate(async (frames) => {
      for (const [x, z, yaw, pitch, alt] of frames) {
        const g = window.__laas.groundProbe?.(x, z)?.ground ?? 0;
        window.__laas.setPose({ p: [x, g + alt, z], yaw, pitch });
        await window.__laas.settle?.(1);
      }
    }, seg);
    if (f % 200 < 50) {
      // mid-roam DOWN capture at the current spot (one frame settle only)
      await page.evaluate((a) => {
        const g = window.__laas.groundProbe?.(a.x, a.z)?.ground ?? 0;
        window.__laas.setPose({ p: [a.x, g + 1.6, a.z], yaw: 0, pitch: a.pitch });
      }, { x: px, z: pz, pitch: DOWN_PITCH });
      await page.evaluate((n) => window.__laas.settle?.(n), 1);
      const buf = await page.screenshot();
      const m = await holeMetric(buf);
      if (m.holePct > midWorst.holePct) {
        midWorst = { ...m, f, px, pz, counters: await counters() };
        writeFileSync(`${OUT}/${TAG}-mid-worst.png`, buf);
      }
      console.log(`  f=${f} pos=(${px.toFixed(0)},${pz.toFixed(0)}) hole=${m.holePct.toFixed(3)}% bbox=${JSON.stringify(m.bbox)}`);
    }
  }
  console.log('mid-roam worst:', JSON.stringify(midWorst));

  // ---- STOP + hold: persistent hole check (three rounds over ~600 frames) ----
  console.log('=== STOP at', px.toFixed(1), pz.toFixed(1), '===');
  let post = { holePct: -1 };
  for (let round = 0; round < 3; round++) {
    await page.evaluate((n) => window.__laas.settle?.(n), round === 0 ? 30 : 200);
    const w = await sweep(px, pz, `poststop-r${round}`, round === 0);
    console.log(`post-stop round ${round}: worst hole=${w.holePct.toFixed(3)}% counters=${JSON.stringify(await counters())}`);
    if (w.holePct > post.holePct) post = { ...w, round };
  }
  console.log('post-stop worst hole%:', post.holePct.toFixed(3), 'baseline:', base.holePct.toFixed(3));

  writeFileSync(`${OUT}/${TAG}-result.json`, JSON.stringify({ base, midWorst, post, errors }, null, 1));
  console.log(`RESULT ${TAG}: baseline=${base.holePct.toFixed(3)}% midroam=${midWorst.holePct.toFixed(3)}% poststop=${post.holePct.toFixed(3)}% errors=${errors.length}`);
  for (const e of errors.slice(0, 10)) console.log('ERR:', e);
  await browser.close();
}
main().catch((e) => { console.error('HARNESS FAIL', e); process.exit(1); });
