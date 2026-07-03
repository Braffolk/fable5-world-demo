/**
 * Diagnostics HUD. DEFAULT: a minimal FPS chip only — the full debug panel
 * (per-pass GPU timings, counters, providers) toggles with F3 (?hud=1 boots
 * with it open for tooling shots). Subsystems contribute line providers;
 * re-renders at 4 Hz from current EngineStats. Floor checks (triangle counts
 * etc.) read `window.__laas.stats` directly — the HUD is for humans.
 */

import type { Engine } from '../core/Engine';
import type { LaasParams } from '../core/Params';

export type HudProvider = () => string[];

const MS_N = 240;
const MS_H = 64;
const MS_MAX = 50;

export class Hud {
  private el: HTMLDivElement;
  private fpsEl: HTMLDivElement;
  private providers: HudProvider[] = [];
  private visible: boolean;
  private engine: Engine;
  private params: LaasParams;
  private acc = 0;
  private msCanvas: HTMLCanvasElement;
  private msCtx: CanvasRenderingContext2D | null = null;
  private msBuf = new Float32Array(MS_N);
  private msIdx = 0;
  private msOn = false;
  private msP50 = 0;
  private msP95 = 0;

  constructor(engine: Engine, params: LaasParams) {
    this.engine = engine;
    this.params = params;
    this.visible = params.hud;
    this.el = document.createElement('div');
    this.el.id = 'hud';
    this.el.style.cssText = [
      'position:fixed', 'top:10px', 'left:10px', 'z-index:1000',
      'color:#d9e8e0', 'background:rgba(8,12,10,0.62)', 'padding:10px 12px',
      'font:11px/1.45 ui-monospace,Menlo,monospace', 'white-space:pre',
      'pointer-events:none', 'border-radius:4px', 'max-height:90vh', 'overflow:hidden',
    ].join(';');
    document.body.appendChild(this.el);

    // always-on minimal readout — just fps (F3 swaps in the full panel)
    this.fpsEl = document.createElement('div');
    this.fpsEl.id = 'hud-fps';
    this.fpsEl.style.cssText = [
      'position:fixed', 'top:10px', 'left:10px', 'z-index:1000',
      'color:#d9e8e0', 'background:rgba(8,12,10,0.5)', 'padding:3px 8px',
      'font:12px/1.2 ui-monospace,Menlo,monospace', 'white-space:pre',
      'pointer-events:none', 'border-radius:4px',
    ].join(';');
    document.body.appendChild(this.fpsEl);
    this.applyVisibility();

    // frametime ms-chart (?mschart=1, F4 toggles): last 240 raw rAF deltas as
    // 1px bars on a 0..50 ms scale with 16.7/33.3 vsync-quantum gridlines —
    // microstutters and quantum flapping read as spikes/banding that the
    // averaged fps number hides. Fed from stats.frameMs (raw, unclamped).
    this.msOn = params.mschart;
    this.msCanvas = document.createElement('canvas');
    this.msCanvas.width = MS_N;
    this.msCanvas.height = MS_H;
    this.msCanvas.style.cssText = [
      'position:fixed', 'top:34px', 'left:10px', 'z-index:1000',
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

  private drawMsChart(): void {
    const g = this.msCtx;
    if (!g) return;
    g.clearRect(0, 0, MS_N, MS_H);
    const y = (ms: number): number => MS_H - (Math.min(ms, MS_MAX) / MS_MAX) * MS_H;
    // vsync-quantum gridlines
    g.fillStyle = 'rgba(217,232,224,0.28)';
    g.fillRect(0, y(16.7), MS_N, 1);
    g.fillRect(0, y(33.3), MS_N, 1);
    // bars, oldest → newest left → right (ring unrolled)
    const n = Math.min(this.msIdx, MS_N);
    const start = this.msIdx - n;
    for (let i = 0; i < n; i++) {
      const ms = this.msBuf[(start + i) % MS_N] ?? 0;
      g.fillStyle = ms <= 17.5 ? '#4caf6e' : ms <= 34 ? '#e0b23e' : '#e05252';
      const top = y(ms);
      g.fillRect(MS_N - n + i, top, 1, MS_H - top);
    }
    // rolling p50/p95 over the visible window
    if (n >= 30 && this.msIdx % 15 === 0) {
      const s = [...this.msBuf.slice(0, n)].sort((a, b) => a - b);
      this.msP50 = s[Math.floor((n - 1) * 0.5)] ?? 0;
      this.msP95 = s[Math.floor((n - 1) * 0.95)] ?? 0;
    }
    g.fillStyle = '#d9e8e0';
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

  private render(): void {
    const s = this.engine.stats;
    const c = this.engine.camera.position;
    const fmt = (n: number): string => n.toLocaleString('en-US');
    const lines: string[] = [
      `LAAS  seed=${this.params.seed} scene=${this.params.scene} T=${this.params.timeOfDay}`,
      `${s.fps.toFixed(0)} fps  ${s.frameMs.toFixed(2)} ms (p95 ${s.frameMsP95.toFixed(2)})`,
      `draws ${fmt(s.drawCalls)}  tris ${fmt(s.triangles)}`,
      `gpu render ${s.gpuPasses['render']?.toFixed(2) ?? '–'} ms  compute ${s.gpuPasses['compute']?.toFixed(2) ?? '–'} ms`,
      `cam ${c.x.toFixed(1)}, ${c.y.toFixed(1)}, ${c.z.toFixed(1)}`,
    ];
    // per-pass GPU attribution (spec §6 HUD requirement; Phase 7 perf)
    const passes = Object.entries(s.gpuPasses)
      .filter(([k, v]) => (k.startsWith('r.') || k.startsWith('c.')) && v >= 0.005)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 16);
    if (passes.length > 0) {
      lines.push('—');
      for (const [k, v] of passes) lines.push(`${v.toFixed(2).padStart(6)} ${k}`);
    }
    const counterKeys = Object.keys(s.counters);
    if (counterKeys.length > 0) {
      lines.push('—');
      for (const k of counterKeys.sort()) lines.push(`${k}: ${fmt(s.counters[k] ?? 0)}`);
    }
    for (const p of this.providers) lines.push('—', ...p());
    lines.push('—', 'F3 hud · V walk/fly · 1-9 bookmarks · F flythrough · P pose');
    this.el.textContent = lines.join('\n');
  }
}
