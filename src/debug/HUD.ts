/**
 * Diagnostics HUD (F3). DEFAULT: a minimal fps chip only; F3 swaps in the full
 * structured panel (?hud=1 boots it open). F4 toggles the frametime ms-chart.
 *
 * ─── DESIGN (2026-07-09 rethink) ────────────────────────────────────────────
 * The old F3 was one flat `white-space:pre` dump of every counter, sorted
 * alphabetically — an unreadable wall that overflowed the screen. Rethought as:
 *
 *   1. HEADER strip (always visible): p95 frame-time is THE emphasized number (large,
 *      bold, colored by the 90fps-floor thresholds — green ≤12.5ms, yellow ≤16.7, red
 *      above; perf is judged on p95, NOT instantaneous fps), with fps/ms-now secondary.
 *      Plus internal render resolution (rscale-aware — the HONEST pixel denominator) and
 *      the pathology number: submitted triangles ÷ rendered pixels, RATIO colored.
 *
 *   2. TRIANGLE BUDGET (centerpiece, default-expanded): one row per raster layer —
 *      SW-mid, splat(1px), HW-tri, HW-cluster (clhw), vox-routed, vox brick — each with
 *      count, % of submitted, ratio-to-pixels, a live sparkline, and cap utilization,
 *      then a `culled·resid` row. The layers now PARTITION submit EXACTLY:
 *      submit = SW-mid + splat + HW-tri + vox-routed + HW-cluster + culled(resid).
 *
 *   3. COLLAPSIBLE sections for everything else (cull, GPU passes, memory, CPU,
 *      misc) — grouped + aligned, no data deleted, just restructured. Collapse
 *      state persists in localStorage. Click a section header to toggle.
 *
 * ⚠️ visTris SEMANTICS (premise-audit — see report): `nanite.visTris` =
 * counters[6] = the sum of each EMITTED cluster's FULL triCount, accumulated in
 * the cull DAG-BFS for every cluster that survived frustum + backface + occlusion
 * + the LOD cut. It is the count SUBMITTED to the rasterizer — NOT tris that shaded
 * pixels. It includes sub-pixel clusters (→ splat), voxel-routed clusters (→ bricks,
 * never rasterized as tris), and cluster tris that fall off-screen or backface at
 * the triangle level (cull is per-cluster-sphere, conservative). It is now attributed
 * EXACTLY: voxel-routed clusters (matClass==7) → nanite.voxRoutedTris, clhw clusters
 * (clusterHwClass) → nanite.clhwTris — both cluster-granular Σ triCount counted at the
 * SAME routing decision the SW classifier skips on (raster slots 10/11) — and the SW
 * per-tri classifier splits the rest into mid/splat/HW-tri, leaving the residual =
 * backface / off-screen / sub-px / cap-dropped SW tris. The per-layer split disambiguates
 * overdraw vs broken LOD; the residual isolates conservative-cull waste.
 *
 * The Series data model (HudCharts.ts) already carries a running max, so a future
 * fly-through census recorder ("record window + running max per layer") is a drop-in.
 *
 * Re-renders at 4 Hz from EngineStats; DOM is built once and updated in place
 * (no per-tick churn). Floor checks read `window.__laas.stats` directly.
 */

import { Vector2 } from 'three';
import type { Engine } from '../core/Engine';
import type { LaasParams } from '../core/Params';
import { internalSize, RSCALE } from '../render/RenderScale';
import { HW_CAP, MID_CAP, SPLAT_CAP } from '../nanite/raster/Queues';
import { Series, drawSparkline } from './HudCharts';

export type HudProvider = () => string[];

const MS_N = 240;
const MS_H = 64;
const MS_MAX = 50;

const SPARK_W = 68;
const SPARK_H = 15;

const GREEN = '#4caf6e';
const YELLOW = '#e0b23e';
const RED = '#e05252';
const DIM = '#7d8f86';
const FG = '#d9e8e0';

const LS_KEY = 'laas.hud.sections';

/** ratio→pixels color: green ≤2×, yellow 2–3×, red >3× (the pathology thresholds). */
function ratioColor(r: number): string {
  if (!Number.isFinite(r)) return DIM;
  return r <= 2 ? GREEN : r <= 3 ? YELLOW : RED;
}

/** p95 frame-time color (the 90fps-floor mandate — perf is judged on p95, NOT fps):
 *  green ≤12.5 ms (80fps), yellow ≤16.7 (60fps), red above. */
function p95Color(ms: number): string {
  if (!Number.isFinite(ms)) return DIM;
  return ms <= 12.5 ? GREEN : ms <= 16.7 ? YELLOW : RED;
}

/** compact count: 8123456 → "8.1M", 4200 → "4.2k". */
function short(n: number): string {
  if (!Number.isFinite(n)) return '–';
  const a = Math.abs(n);
  if (a >= 1e6) return `${(n / 1e6).toFixed(a >= 1e7 ? 0 : 1)}M`;
  if (a >= 1e3) return `${(n / 1e3).toFixed(a >= 1e4 ? 0 : 1)}k`;
  return String(Math.round(n));
}

interface LayerDef {
  key: string;
  label: string;
  /** counter name (undefined ⇒ use a custom getter). */
  counter: string;
  color: string;
  /** cap for utilization (0 ⇒ uncapped/na). */
  cap: number;
}

// The raster layers, in pipeline order. mid/hw are per-triangle; splat is per
// sub-pixel FRAGMENT (~1 px each); voxel bricks are per-pixel elections (?voxwrites).
// The tri-partition layers (mid + splat + HW-tri + vox-routed + HW-cluster) sum with the
// per-tri-culled residual to EXACTLY the submit total; `vox brick` is a per-PIXEL brick-write
// diagnostic (not part of the tri sum). splat = 1 fragment append per elected sub-pixel tri
// (≈1px tri, so it counts as tris here). vox-routed / HW-cluster are cluster-granular counts
// (Σ triCount) of clusters the SW classifier skips (routed to the voxel raster / HW instanced
// draw), counted where the SAME routing decision is made ⇒ the budget partitions submit.
const LAYERS: LayerDef[] = [
  { key: 'mid', label: 'SW-mid', counter: 'nanite.midTris', color: 'rgb(90,169,230)', cap: MID_CAP },
  { key: 'splat', label: 'splat(1px)', counter: 'nanite.splatFrags', color: 'rgb(224,178,62)', cap: SPLAT_CAP },
  { key: 'hw', label: 'HW-tri', counter: 'nanite.hwTris', color: 'rgb(199,125,255)', cap: HW_CAP },
  { key: 'clhw', label: 'HW-cluster', counter: 'nanite.clhwTris', color: 'rgb(157,140,255)', cap: 0 },
  { key: 'voxr', label: 'vox-routed', counter: 'nanite.voxRoutedTris', color: 'rgb(76,175,110)', cap: 0 },
  { key: 'vox', label: 'vox brick', counter: 'nanite.voxBrickWrites', color: 'rgb(110,150,120)', cap: 0 },
];

interface Section {
  id: string;
  title: string;
  root: HTMLDivElement;
  body: HTMLDivElement;
  arrow: HTMLSpanElement;
  open: boolean;
}

interface BudgetRow {
  root: HTMLDivElement;
  name: HTMLSpanElement;
  count: HTMLSpanElement;
  pct: HTMLSpanElement;
  perPx: HTMLSpanElement;
  cap: HTMLSpanElement;
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D | null;
}

export class Hud {
  private el: HTMLDivElement;
  private fpsEl: HTMLDivElement;
  private providers: HudProvider[] = [];
  private visible: boolean;
  private engine: Engine;
  private params: LaasParams;
  private acc = 0;
  private _vec = new Vector2();

  // frametime chart (F4)
  private msCanvas: HTMLCanvasElement;
  private msCtx: CanvasRenderingContext2D | null = null;
  private msBuf = new Float32Array(MS_N);
  private msIdx = 0;
  private msOn = false;
  private msP50 = 0;
  private msP95 = 0;

  // header
  private hHead1!: HTMLDivElement;
  private hHead2!: HTMLDivElement;
  private hHead3!: HTMLDivElement;

  // triangle budget
  private rows = new Map<string, BudgetRow>();
  private totalRow!: BudgetRow;
  private residualRow!: BudgetRow;
  private deltaLine!: HTMLDivElement;
  private series = new Map<string, Series>();

  // collapsible sections
  private sections = new Map<string, Section>();
  private openState: Record<string, boolean> = {};

  constructor(engine: Engine, params: LaasParams) {
    this.engine = engine;
    this.params = params;
    this.visible = params.hud;
    this.openState = this.loadOpen();

    this.el = document.createElement('div');
    this.el.id = 'hud';
    this.el.style.cssText = [
      'position:fixed', 'top:10px', 'left:10px', 'z-index:1000',
      'color:' + FG, 'background:rgba(8,12,10,0.72)', 'padding:8px 10px',
      'font:11px/1.4 ui-monospace,Menlo,monospace',
      'pointer-events:none', 'border-radius:5px',
      'max-height:94vh', 'overflow-y:auto', 'width:340px',
      'box-shadow:0 2px 12px rgba(0,0,0,0.4)',
    ].join(';');
    document.body.appendChild(this.el);

    this.buildHeader();
    this.buildTriBudget();
    this.buildSections();
    this.buildFooter();

    // always-on minimal readout — just fps (F3 swaps in the full panel)
    this.fpsEl = document.createElement('div');
    this.fpsEl.id = 'hud-fps';
    this.fpsEl.style.cssText = [
      'position:fixed', 'top:10px', 'left:10px', 'z-index:1000',
      'color:' + FG, 'background:rgba(8,12,10,0.5)', 'padding:3px 8px',
      'font:12px/1.2 ui-monospace,Menlo,monospace', 'white-space:pre',
      'pointer-events:none', 'border-radius:4px',
    ].join(';');
    document.body.appendChild(this.fpsEl);
    this.applyVisibility();

    // frametime ms-chart (?mschart=1, F4 toggles): last 240 raw rAF deltas.
    this.msOn = params.mschart;
    this.msCanvas = document.createElement('canvas');
    this.msCanvas.width = MS_N;
    this.msCanvas.height = MS_H;
    this.msCanvas.style.cssText = [
      'position:fixed', 'bottom:10px', 'left:10px', 'z-index:1000',
      'background:rgba(8,12,10,0.62)', 'pointer-events:none', 'border-radius:4px',
      `width:${MS_N}px`, `height:${MS_H}px`,
    ].join(';');
    this.msCanvas.style.display = this.msOn ? 'block' : 'none';
    document.body.appendChild(this.msCanvas);
    this.msCtx = this.msCanvas.getContext('2d');

    window.addEventListener('keydown', (e) => {
      if (e.code === 'F3') {
        e.preventDefault();
        this.visible = !this.visible;
        this.applyVisibility();
      }
      if (e.code === 'F4') {
        e.preventDefault();
        this.msOn = !this.msOn;
        this.msCanvas.style.display = this.msOn ? 'block' : 'none';
      }
    });

    engine.onUpdate((dt) => {
      this.msBuf[this.msIdx % MS_N] = this.engine.stats.frameMs;
      this.msIdx += 1;
      if (this.msOn) this.drawMsChart();
      this.acc += dt;
      if (this.acc >= 0.25) {
        this.acc = 0;
        if (this.visible) this.render();
        else this.fpsEl.textContent = `${this.engine.stats.fps.toFixed(0)} fps`;
      }
    });
  }

  // ─── DOM construction (once) ──────────────────────────────────────────────

  private buildHeader(): void {
    const wrap = document.createElement('div');
    wrap.style.cssText = 'margin-bottom:6px';
    const title = document.createElement('div');
    title.style.cssText = `color:${DIM};font-size:10px;letter-spacing:0.5px`;
    title.textContent = `LAAS · seed ${this.params.seed} · ${this.params.scene}`;
    this.hHead1 = document.createElement('div');
    this.hHead1.style.cssText = 'margin:1px 0;line-height:1.15';
    this.hHead2 = document.createElement('div');
    this.hHead2.style.cssText = `color:${DIM}`;
    this.hHead3 = document.createElement('div');
    this.hHead3.style.cssText = 'margin-top:2px;font-size:12px';
    wrap.append(title, this.hHead1, this.hHead2, this.hHead3);
    this.el.appendChild(wrap);
  }

  private makeBudgetRow(label: string, color: string, bold: boolean): BudgetRow {
    const root = document.createElement('div');
    root.style.cssText = [
      'display:grid',
      `grid-template-columns:58px 46px 34px 44px ${SPARK_W}px 1fr`,
      'align-items:center', 'gap:4px', 'height:16px',
      bold ? 'font-weight:600' : '',
    ].join(';');
    const mk = (align: string, c: string): HTMLSpanElement => {
      const s = document.createElement('span');
      s.style.cssText = `text-align:${align};color:${c};overflow:hidden;white-space:nowrap`;
      return s;
    };
    const name = mk('left', color);
    name.textContent = label;
    const count = mk('right', FG);
    const pct = mk('right', DIM);
    const perPx = mk('right', FG);
    const cap = mk('right', DIM);
    const canvas = document.createElement('canvas');
    canvas.width = SPARK_W;
    canvas.height = SPARK_H;
    canvas.style.cssText = `width:${SPARK_W}px;height:${SPARK_H}px`;
    root.append(name, count, pct, perPx, canvas, cap);
    return { root, name, count, pct, perPx, cap, canvas, ctx: canvas.getContext('2d') };
  }

  private buildTriBudget(): void {
    const sec = this.makeSection('tribudget', 'TRIANGLE BUDGET', true);
    // column header
    const hdr = document.createElement('div');
    hdr.style.cssText = [
      'display:grid',
      `grid-template-columns:58px 46px 34px 44px ${SPARK_W}px 1fr`,
      `gap:4px;color:${DIM};font-size:9px;margin-bottom:1px`,
    ].join(';');
    for (const [t, a] of [['layer', 'left'], ['tris', 'right'], ['%', 'right'], ['/px', 'right'], ['trend', 'left'], ['cap%', 'right']] as const) {
      const s = document.createElement('span');
      s.style.cssText = `text-align:${a}`;
      s.textContent = t;
      hdr.appendChild(s);
    }
    sec.body.appendChild(hdr);
    for (const L of LAYERS) {
      const row = this.makeBudgetRow(L.label, L.color, false);
      this.rows.set(L.key, row);
      this.series.set(L.key, new Series());
      sec.body.appendChild(row.root);
    }
    // divider + total (submitted = visTris)
    const div = document.createElement('div');
    div.style.cssText = `border-top:1px solid rgba(217,232,224,0.18);margin:2px 0`;
    sec.body.appendChild(div);
    this.totalRow = this.makeBudgetRow('submit', FG, true);
    this.series.set('vis', new Series());
    sec.body.appendChild(this.totalRow.root);
    // per-tri culled (residual) = submit − (mid+splat+HWtri+vox-routed+HW-cluster)
    this.residualRow = this.makeBudgetRow('culled·resid', DIM, false);
    this.series.set('residual', new Series());
    sec.body.appendChild(this.residualRow.root);
    // caption
    this.deltaLine = document.createElement('div');
    this.deltaLine.style.cssText = `color:${DIM};font-size:9px;margin-top:3px;line-height:1.3`;
    sec.body.appendChild(this.deltaLine);
  }

  private makeSection(id: string, title: string, defaultOpen: boolean): Section {
    const open = this.openState[id] ?? defaultOpen;
    const root = document.createElement('div');
    root.style.cssText = 'margin-top:6px';
    const head = document.createElement('div');
    head.style.cssText = [
      'display:flex', 'align-items:center', 'gap:5px', 'cursor:pointer',
      'pointer-events:auto', 'user-select:none', `color:${FG}`,
      'font-size:10px', 'letter-spacing:0.4px', 'padding:1px 0',
      'border-bottom:1px solid rgba(217,232,224,0.12)',
    ].join(';');
    const arrow = document.createElement('span');
    arrow.style.cssText = `color:${DIM};width:8px`;
    arrow.textContent = open ? '▾' : '▸';
    const label = document.createElement('span');
    label.textContent = title;
    head.append(arrow, label);
    const body = document.createElement('div');
    body.style.cssText = 'margin-top:3px';
    body.style.display = open ? 'block' : 'none';
    const sec: Section = { id, title, root, body, arrow, open };
    head.addEventListener('click', () => this.toggle(sec));
    root.append(head, body);
    this.el.appendChild(root);
    this.sections.set(id, sec);
    return sec;
  }

  private buildSections(): void {
    for (const [id, title] of [
      ['cull', 'CULL & CLUSTERS'],
      ['gpu', 'GPU PASSES (ms)'],
      ['mem', 'MEMORY / VRAM'],
      ['stream', 'STREAMING'],
      ['cpu', 'CPU / SHADOW / MISC'],
    ] as const) {
      const sec = this.makeSection(id, title, false);
      const pre = document.createElement('div');
      pre.style.cssText = 'white-space:pre;font-size:10.5px;line-height:1.4';
      sec.body.appendChild(pre);
      (sec as unknown as { pre: HTMLDivElement }).pre = pre;
    }
  }

  private buildFooter(): void {
    const f = document.createElement('div');
    f.style.cssText = `color:${DIM};font-size:9px;margin-top:7px;line-height:1.35`;
    f.textContent = 'F3 hud · F4 ms-chart · click headers to expand · V walk/fly · 1-9 bookmarks';
    this.el.appendChild(f);
  }

  private secPre(id: string): HTMLDivElement {
    return (this.sections.get(id) as unknown as { pre: HTMLDivElement }).pre;
  }

  private toggle(sec: Section): void {
    sec.open = !sec.open;
    sec.body.style.display = sec.open ? 'block' : 'none';
    sec.arrow.textContent = sec.open ? '▾' : '▸';
    this.openState[sec.id] = sec.open;
    this.saveOpen();
  }

  private loadOpen(): Record<string, boolean> {
    try {
      return JSON.parse(localStorage.getItem(LS_KEY) ?? '{}') as Record<string, boolean>;
    } catch {
      return {};
    }
  }

  private saveOpen(): void {
    try {
      localStorage.setItem(LS_KEY, JSON.stringify(this.openState));
    } catch {
      /* ignore */
    }
  }

  // ─── frametime chart (unchanged) ──────────────────────────────────────────

  private drawMsChart(): void {
    const g = this.msCtx;
    if (!g) return;
    g.clearRect(0, 0, MS_N, MS_H);
    const y = (ms: number): number => MS_H - (Math.min(ms, MS_MAX) / MS_MAX) * MS_H;
    g.fillStyle = 'rgba(217,232,224,0.28)';
    g.fillRect(0, y(16.7), MS_N, 1);
    g.fillRect(0, y(33.3), MS_N, 1);
    const n = Math.min(this.msIdx, MS_N);
    for (let i = 0; i < n; i++) {
      const ms = this.msBuf[(this.msIdx - n + i) % MS_N] ?? 0;
      g.fillStyle = ms <= 17.5 ? GREEN : ms <= 34 ? YELLOW : RED;
      const top = y(ms);
      g.fillRect(MS_N - n + i, top, 1, MS_H - top);
    }
    if (n >= 30 && this.msIdx % 15 === 0) {
      const s = [...this.msBuf.slice(0, n)].sort((a, b) => a - b);
      this.msP50 = s[Math.floor((n - 1) * 0.5)] ?? 0;
      this.msP95 = s[Math.floor((n - 1) * 0.95)] ?? 0;
    }
    g.fillStyle = FG;
    g.font = '9px ui-monospace,Menlo,monospace';
    g.fillText(`p50 ${this.msP50.toFixed(1)}  p95 ${this.msP95.toFixed(1)}`, 4, 9);
  }

  private applyVisibility(): void {
    this.el.style.display = this.visible ? 'block' : 'none';
    this.fpsEl.style.display = this.visible ? 'none' : 'block';
  }

  addProvider(p: HudProvider): void {
    this.providers.push(p);
  }

  // ─── per-tick render (update in place) ────────────────────────────────────

  private render(): void {
    const s = this.engine.stats;
    const c = s.counters;
    const cn = (k: string): number | undefined => c[k];

    // rendered pixel count — the HONEST, rscale-aware denominator.
    const res = internalSize(this.engine.renderer, this._vec);
    const px = Math.max(1, res.x * res.y);
    const visTris = cn('nanite.visTris') ?? 0;

    // ── header ── p95 frame-time is THE number (user judges perf on p95, not fps): large,
    // bold, colored by the 90fps-floor thresholds. Instantaneous fps/ms are secondary (dim/small).
    const p95 = s.frameMsP95;
    this.hHead1.innerHTML =
      `<b style="font-size:18px;color:${p95Color(p95)}">p95 ${p95.toFixed(1)}</b>` +
      `<span style="color:${DIM};font-size:11px"> ms</span>` +
      `<span style="color:${DIM};font-size:11px;margin-left:8px">${s.fps.toFixed(0)} fps · ${s.frameMs.toFixed(1)} ms now</span>`;
    const gpu = (s.gpuPasses['render'] ?? 0) + (s.gpuPasses['compute'] ?? 0);
    this.hHead2.textContent =
      `${res.x}×${res.y}${RSCALE < 1 ? ` @${RSCALE.toFixed(2)}` : ''}` +
      ` · ${(px / 1e6).toFixed(2)} Mpx · gpu ${gpu > 0 ? gpu.toFixed(1) + 'ms' : '–'} · ${s.drawCalls} draws`;
    const visR = visTris / px;
    this.hHead3.innerHTML =
      `<span style="color:${DIM}">submitted</span> ` +
      `<b>${short(visTris)}</b> tris / ${(px / 1e6).toFixed(1)}M px = ` +
      `<b style="color:${ratioColor(visR)}">${visR.toFixed(2)}×</b>`;

    // ── triangle budget rows ──
    for (const L of LAYERS) {
      const row = this.rows.get(L.key)!;
      const raw = cn(L.counter);
      const ser = this.series.get(L.key)!;
      if (raw === undefined) {
        row.count.textContent = '–';
        row.pct.textContent = '';
        row.perPx.textContent = '';
        row.cap.textContent = L.key === 'vox' ? '?voxwrites' : 'n/a';
        row.cap.style.color = DIM;
        continue;
      }
      ser.push(raw);
      row.count.textContent = short(raw);
      row.pct.textContent = visTris > 0 ? `${Math.round((raw / visTris) * 100)}%` : '';
      const r = raw / px;
      row.perPx.textContent = `${r.toFixed(2)}×`;
      row.perPx.style.color = ratioColor(r);
      // cap utilization
      if (L.cap > 0) {
        const u = raw / L.cap;
        row.cap.textContent = raw > L.cap ? 'OVFL' : `${Math.round(u * 100)}%`;
        row.cap.style.color = raw > L.cap ? RED : u >= 0.9 ? RED : u >= 0.7 ? YELLOW : DIM;
      } else {
        row.cap.textContent = '';
      }
      if (row.ctx) {
        drawSparkline(row.ctx, SPARK_W, SPARK_H, ser.window(), ser.windowMax(), L.color);
      }
    }

    // ── total (submitted) row ──
    const visSer = this.series.get('vis')!;
    visSer.push(visTris);
    this.totalRow.name.textContent = 'submit';
    this.totalRow.count.textContent = short(visTris);
    this.totalRow.pct.textContent = '100%';
    this.totalRow.perPx.textContent = `${visR.toFixed(2)}×`;
    this.totalRow.perPx.style.color = ratioColor(visR);
    this.totalRow.cap.textContent = '';
    if (this.totalRow.ctx) {
      drawSparkline(this.totalRow.ctx, SPARK_W, SPARK_H, visSer.window(), visSer.windowMax(), FG.startsWith('#') ? 'rgb(217,232,224)' : FG);
    }
    // ── per-tri culled (residual) row ──
    // Identity: submit = mid + splat + HW-tri + vox-routed + HW-cluster + residual. So the
    // residual = SW-cluster tris the classifier dropped (backface / off-screen / sample-miss
    // / sub-pixel) + any queue-cap overflow. Voxel & clhw clusters are counted WHOLE at their
    // routing decision — the SAME condition the SW classifier skips on (raster slots 10/11) —
    // so those buckets partition submit exactly. A residual NEGATIVE beyond ~noise ⇒ a
    // double-count bug (shown in red, still displayed as it's diagnostic).
    const haveLayers = cn('nanite.midTris') !== undefined; // world1 single-pass path only
    const mid = cn('nanite.midTris') ?? 0;
    const splat = cn('nanite.splatFrags') ?? 0;
    const hw = cn('nanite.hwTris') ?? 0;
    const voxRouted = cn('nanite.voxRoutedTris') ?? 0;
    const clhw = cn('nanite.clhwTris') ?? 0;
    const residual = visTris - (mid + splat + hw + voxRouted + clhw);
    const resSer = this.series.get('residual')!;
    resSer.push(Math.max(0, residual));
    this.residualRow.name.textContent = 'culled·resid';
    if (haveLayers) {
      this.residualRow.count.textContent = short(residual);
      // negative beyond a 2%-of-submit noise floor ⇒ double-count bug → flag red
      this.residualRow.count.style.color = residual < -Math.max(1000, visTris * 0.02) ? RED : FG;
      this.residualRow.pct.textContent = visTris > 0 ? `${Math.round((residual / visTris) * 100)}%` : '';
      const rr = residual / px;
      this.residualRow.perPx.textContent = `${rr.toFixed(2)}×`;
      this.residualRow.perPx.style.color = ratioColor(rr);
    } else {
      this.residualRow.count.textContent = '–';
      this.residualRow.pct.textContent = '';
      this.residualRow.perPx.textContent = '';
    }
    this.residualRow.cap.textContent = '';
    if (this.residualRow.ctx) {
      drawSparkline(this.residualRow.ctx, SPARK_W, SPARK_H, resSer.window(), resSer.windowMax(), DIM);
    }

    // caption
    const voxCl = cn('nanite.voxClusters');
    this.deltaLine.textContent =
      `submit = SW-mid + splat + HW-tri + vox-routed + HW-cluster + culled·resid. ` +
      `resid = backface / off-screen / sub-px / cap-drop of SW clusters` +
      (voxCl !== undefined ? ` · ${short(voxCl)} vox cl` : '');

    // ── collapsible sections ──
    this.renderCull(c);
    this.renderGpu(s);
    this.renderMem(c);
    this.renderStream(c);
    this.renderCpu(c);
  }

  private fmt(n: number | undefined): string {
    return n === undefined ? '–' : n.toLocaleString('en-US');
  }

  private kv(keys: [string, string][], c: Record<string, number>): string {
    const w = Math.max(...keys.map(([, l]) => l.length));
    return keys.map(([k, l]) => `${l.padEnd(w)}  ${this.fmt(c[k])}`).join('\n');
  }

  private renderCull(c: Record<string, number>): void {
    if (!this.sections.get('cull')?.open) return;
    const keys: [string, string][] = [
      ['nanite.visClusters', 'visClusters'],
      ['nanite.dagClusters', 'dagClusters'],
      ['nanite.voxClusters', 'voxClusters'],
    ];
    // orphans / covered are ?audit-gated (only populated under ?audit — see NaniteView):
    // show the rows ONLY when a value exists, rather than a permanent '–'.
    if (c['nanite.orphans'] !== undefined) keys.push(['nanite.orphans', 'orphans']);
    if (c['nanite.covered'] !== undefined) keys.push(['nanite.covered', 'covered']);
    this.secPre('cull').textContent = this.kv(keys, c);
  }

  private renderGpu(s: Engine['stats']): void {
    if (!this.sections.get('gpu')?.open) return;
    const passes = Object.entries(s.gpuPasses)
      .filter(([k, v]) => (k.startsWith('r.') || k.startsWith('c.')) && v >= 0.02)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 14);
    const lines = [
      `render ${(s.gpuPasses['render'] ?? 0).toFixed(2)}  compute ${(s.gpuPasses['compute'] ?? 0).toFixed(2)}`,
      '',
      ...passes.map(([k, v]) => `${v.toFixed(2).padStart(6)}  ${k}`),
    ];
    this.secPre('gpu').textContent = lines.join('\n');
  }

  private renderMem(c: Record<string, number>): void {
    if (!this.sections.get('mem')?.open) return;
    this.secPre('mem').textContent = this.kv([
      ['nanite.mb', 'nanite MB'],
      ['gpu.geometries', 'geometries'],
      ['gpu.textures', 'textures'],
      ['gpu.buffers', 'buffers'],
      ['nanite.meshes', 'reg meshes'],
      ['nanite.clusters', 'reg clusters'],
      ['nanite.trisK', 'reg tris(k)'],
      ['nanite.inst', 'reg inst'],
    ], c);
  }

  /** S5 (F-9): stream-brain residency/queue/budget counters — resident tiles
   *  per LOD, in-flight fetches/decodes/bakes, LRU + brain RAM, token-bucket
   *  utilization, evictions/nacks, scroll + teleport tallies. */
  private renderStream(c: Record<string, number>): void {
    if (!this.sections.get('stream')?.open) return;
    if (c['stream.tiles.resident'] === undefined) {
      this.secPre('stream').textContent = 'no stream brain (forest/gallery scene)';
      return;
    }
    const perLevel = Object.keys(c)
      .filter((k) => k.startsWith('stream.res.L'))
      .sort()
      .map((k) => `${k.slice('stream.res.'.length)}:${c[k]}`)
      .join(' ');
    const lines = [
      `tiles      ${this.fmt(c['stream.tiles.resident'])} resident (${perLevel})  pending ${this.fmt(c['stream.tiles.pending'])}`,
      `loads      ${this.fmt(c['stream.tiles.loaded'])} loaded · ${this.fmt(c['stream.tiles.evicted'])} evicted · ` +
        `${this.fmt(c['stream.tiles.skipped'])} skipped · ${this.fmt(c['stream.tiles.nacks'])} nacks`,
      `bakes      ${this.fmt(c['stream.bake.cache'])} cached / ${this.fmt(c['stream.bake.built'])} built · ` +
        `inflight ${this.fmt(c['stream.bake.inflight'])} · prefetch ${this.fmt(c['stream.prefetched'])}`,
      `fetch      ${this.fmt(c['stream.fetch.total'])} total · inflight ${this.fmt(c['stream.fetch.inflight'])}`,
      `brain RAM  ${this.fmt(c['stream.ram.mb'])} MB (lru ${this.fmt(c['stream.lru.mb'])} MB)`,
      `bucket     ${((c['stream.bucket.ms'] ?? 0)).toFixed(2)} ms · ${this.fmt(c['stream.bucket.kb'])} KB · ` +
        `${this.fmt(c['stream.bucket.packets'])} pkts · mailbox ${this.fmt(c['stream.mailbox.depth'])}`,
      `scrolls    ${this.fmt(c['stream.scrolls'])} (${this.fmt(c['stream.scrolls.deferred'])} deferred) · ` +
        `teleports ${this.fmt(c['stream.teleports'])}`,
      `origin     (${this.fmt(c['stream.origin.x'])}, ${this.fmt(c['stream.origin.z'])}) · ` +
        `rebases ${this.fmt(c['stream.origin.rebases'])}`,
    ];
    this.secPre('stream').textContent = lines.join('\n');
  }

  private renderCpu(c: Record<string, number>): void {
    const sec = this.sections.get('cpu');
    if (!sec?.open) return;
    // known keys shown explicitly; every OTHER counter dumped below so no data is lost.
    const known = new Set([
      'nanite.visTris', 'nanite.midTris', 'nanite.splatFrags', 'nanite.hwTris',
      'nanite.voxBrickWrites', 'nanite.voxClusters', 'nanite.visClusters',
      'nanite.dagClusters', 'nanite.orphans', 'nanite.covered', 'nanite.mb', 'gpu.geometries',
      'gpu.textures', 'gpu.buffers', 'nanite.meshes', 'nanite.clusters',
      'nanite.trisK', 'nanite.inst', 'nanite.dagTris',
      'nanite.voxRoutedTris', 'nanite.clhwTris',
    ]);
    const lines = [
      `updateMs   ${((c['cpu.updateMs100'] ?? 0) / 100).toFixed(2)}`,
      `submitMs   ${((c['cpu.submitMs100'] ?? 0) / 100).toFixed(2)}`,
      `jitterIdx  ${this.fmt(c['nanite.jitterIdx'])}`,
      `shRaster   ${this.fmt(c['nanite.shRaster'])}   shTotal ${this.fmt(c['nanite.shTotal'])}`,
    ];
    const rest = Object.keys(c)
      .filter((k) => !known.has(k) && !k.startsWith('cpu.') && !k.startsWith('stream.') && k !== 'nanite.jitterIdx' && k !== 'nanite.shRaster' && k !== 'nanite.shTotal')
      .sort();
    if (rest.length > 0) {
      lines.push('—');
      for (const k of rest) lines.push(`${k}  ${this.fmt(c[k])}`);
    }
    for (const p of this.providers) {
      lines.push('—', ...p());
    }
    this.secPre('cpu').textContent = lines.join('\n');
  }
}
