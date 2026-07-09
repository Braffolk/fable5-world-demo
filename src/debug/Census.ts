/**
 * Census — tri-count CENSUS instrument (?census=1). The before/after artifact for
 * right-sizing the pathology-adapted raster caps (MID_CAP 41.9M, SPLAT/HW/…).
 *
 * ONE reproducible run: `?census=1` implies the automatic flythrough (Bookmarks.ts).
 * As the camera flies, the recorder samples the per-layer triangle budget the F3
 * overlay already partitions (SW-mid / splat / HW-tri / HW-cluster / vox-routed /
 * submit + vox-brick) at the EXISTING meterRead cadence (~every 15 frames — it reads
 * the counters already sitting in engine.stats.counters, it does NOT add readbacks).
 * Rendered-pixel denominator = internalSize() (rscale-aware) so every /px ratio is honest.
 *
 * Records EXACTLY ONE loop (fly t wraps 1→0 = done), then:
 *   1. a console summary table — per layer PEAK count, peak /px ratio, and the pose/t
 *      where the peak happened (printed as a ready-to-paste `?cam=` snippet);
 *   2. a downloadable JSON (census-<ts>.json) with the full sample array + summary;
 *   3. a live status chip so the user sees "recording… / done — peaks: …".
 *
 * Zero footprint when ?census is absent — Bookmarks only constructs this then.
 */

import { Vector2 } from 'three';
import type { Engine } from '../core/Engine';
import type { LaasParams } from '../core/Params';
import { internalSize, RSCALE } from '../render/RenderScale';

/** sample cadence in frames — matches the meterRead readback batch (NaniteFrame.ts). */
const SAMPLE_EVERY = 15;

/** the raster layers that PARTITION submit, plus submit + the vox-brick diagnostic. */
interface LayerDef {
  key: string;
  label: string;
  counter: string;
}
const LAYERS: LayerDef[] = [
  { key: 'mid', label: 'SW-mid', counter: 'nanite.midTris' },
  { key: 'splat', label: 'splat(1px)', counter: 'nanite.splatFrags' },
  { key: 'hwTri', label: 'HW-tri', counter: 'nanite.hwTris' },
  { key: 'clhw', label: 'HW-cluster', counter: 'nanite.clhwTris' },
  { key: 'voxRouted', label: 'vox-routed', counter: 'nanite.voxRoutedTris' },
  { key: 'submit', label: 'submit', counter: 'nanite.visTris' },
  { key: 'voxBrick', label: 'vox brick', counter: 'nanite.voxBrickWrites' },
  // visible-cluster count — bounds PROJ_CLUSTER_CAP (projVertBuf reserves one
  // MAX_CLUSTER_VERTS slot-region per visible cluster). Already sitting in
  // stats.counters (NaniteFrame.ts writes nanite.visClusters); read like the rest.
  { key: 'visClusters', label: 'visClusters', counter: 'nanite.visClusters' },
];

export interface CensusPose {
  x: number;
  y: number;
  z: number;
  yaw: number;
  pitch: number;
}

interface CensusSample {
  /** loop fraction [0,1) */
  t: number;
  frame: number;
  pose: CensusPose;
  /** rendered megapixels (rscale-aware) */
  mpx: number;
  /** per-layer counts (raw, as the HUD budget shows) */
  layers: Record<string, number>;
  /** per-layer count ÷ rendered pixels */
  ratios: Record<string, number>;
  /** frame-ms since the previous sample */
  frameMs: { mean: number; max: number; p95: number };
}

interface Peak {
  value: number;
  sample: CensusSample | null;
}

interface LayerSummary {
  layer: string;
  peakCount: number;
  peakCountAt: { t: number; mpx: number; pose: CensusPose } | null;
  peakRatio: number;
  peakRatioAt: { t: number; mpx: number; pose: CensusPose } | null;
  cam: string; // ?cam= snippet at the peak-RATIO pose (the pathology location)
}

function short(n: number): string {
  if (!Number.isFinite(n)) return '-';
  const a = Math.abs(n);
  if (a >= 1e6) return `${(n / 1e6).toFixed(a >= 1e7 ? 0 : 1)}M`;
  if (a >= 1e3) return `${(n / 1e3).toFixed(a >= 1e4 ? 0 : 1)}k`;
  return String(Math.round(n));
}

function camSnippet(p: CensusPose): string {
  const f = (v: number): string => v.toFixed(1);
  return `${f(p.x)},${f(p.y)},${f(p.z)},${p.yaw.toFixed(4)},${p.pitch.toFixed(4)}`;
}

export class Census {
  private engine: Engine;
  private params: LaasParams;
  private samples: CensusSample[] = [];
  private peakCount = new Map<string, Peak>();
  private peakRatio = new Map<string, Peak>();
  private frameCount = 0;
  private msAccum: number[] = [];
  private lastT = -1;
  private done = false;
  private started = false;
  private _vec = new Vector2();
  private chip: HTMLDivElement;

  constructor(engine: Engine, params: LaasParams) {
    this.engine = engine;
    this.params = params;
    for (const L of LAYERS) {
      this.peakCount.set(L.key, { value: -1, sample: null });
      this.peakRatio.set(L.key, { value: -1, sample: null });
    }
    this.chip = document.createElement('div');
    this.chip.id = 'census-chip';
    this.chip.style.cssText = [
      'position:fixed', 'top:10px', 'left:50%', 'transform:translateX(-50%)',
      'z-index:1001', 'color:#d9e8e0', 'background:rgba(8,12,10,0.8)',
      'padding:4px 10px', 'font:11px/1.3 ui-monospace,Menlo,monospace',
      'pointer-events:none', 'border-radius:4px', 'white-space:nowrap',
      'box-shadow:0 2px 12px rgba(0,0,0,0.45)', 'border:1px solid rgba(224,178,62,0.4)',
    ].join(';');
    this.chip.textContent = 'census: arming…';
    document.body.appendChild(this.chip);
  }

  /** Called every frame from the flythrough with the ACTUAL (clamped) pose + loop t. */
  tick(t: number, pose: CensusPose): void {
    if (this.done) return;

    // accumulate frame-ms for the "since last sample" stats
    this.msAccum.push(this.engine.stats.frameMs);

    // ONE-LOOP detection: t rises 0→~1 then wraps to ~0. A large backward jump
    // (only possible via the modulo wrap) after we've collected a real run = done.
    if (this.started && this.lastT >= 0 && t < this.lastT - 0.5 && this.samples.length > 4) {
      this.lastT = t;
      this.finish();
      return;
    }
    this.started = true;
    this.lastT = t;

    this.frameCount += 1;
    if (this.frameCount % SAMPLE_EVERY !== 0) return;
    this.takeSample(t, pose);
  }

  private takeSample(t: number, pose: CensusPose): void {
    const c = this.engine.stats.counters;
    const res = internalSize(this.engine.renderer, this._vec);
    const px = Math.max(1, res.x * res.y);

    const layers: Record<string, number> = {};
    const ratios: Record<string, number> = {};
    for (const L of LAYERS) {
      const v = c[L.counter] ?? 0;
      layers[L.key] = v;
      ratios[L.key] = v / px;
    }

    // frame-ms stats since the previous sample
    const acc = this.msAccum;
    this.msAccum = [];
    let mean = 0;
    let max = 0;
    let p95 = 0;
    if (acc.length > 0) {
      const sorted = [...acc].sort((a, b) => a - b);
      for (const v of acc) {
        mean += v;
        if (v > max) max = v;
      }
      mean /= acc.length;
      p95 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))] ?? max;
    }

    const sample: CensusSample = {
      t,
      frame: this.engine.stats.frame,
      pose: { ...pose },
      mpx: px / 1e6,
      layers,
      ratios,
      frameMs: { mean, max, p95 },
    };
    this.samples.push(sample);

    // update running peaks (never decaying) — count-peak and ratio-peak tracked
    // separately since px varies along the path, so their locations can differ.
    for (const L of LAYERS) {
      const pc = this.peakCount.get(L.key)!;
      if (layers[L.key]! > pc.value) {
        pc.value = layers[L.key]!;
        pc.sample = sample;
      }
      const pr = this.peakRatio.get(L.key)!;
      if (ratios[L.key]! > pr.value) {
        pr.value = ratios[L.key]!;
        pr.sample = sample;
      }
    }

    this.updateChip(t, false);
  }

  private updateChip(t: number, doneNow: boolean): void {
    if (doneNow) {
      const mid = this.peakRatio.get('mid')!;
      const sub = this.peakRatio.get('submit')!;
      const midC = this.peakCount.get('mid')!;
      const subC = this.peakCount.get('submit')!;
      this.chip.textContent =
        `census done — mid ${short(midC.value)}@${mid.value.toFixed(1)}× · ` +
        `submit ${short(subC.value)}@${sub.value.toFixed(1)}× · JSON saved (${this.samples.length} samples)`;
      this.chip.style.borderColor = 'rgba(76,175,110,0.6)';
    } else {
      this.chip.textContent =
        `census: recording… ${this.samples.length} samples · t=${t.toFixed(2)}`;
    }
  }

  private buildSummary(): LayerSummary[] {
    return LAYERS.map((L) => {
      const pc = this.peakCount.get(L.key)!;
      const pr = this.peakRatio.get(L.key)!;
      const at = (s: CensusSample | null): { t: number; mpx: number; pose: CensusPose } | null =>
        s ? { t: s.t, mpx: s.mpx, pose: s.pose } : null;
      return {
        layer: L.label,
        peakCount: Math.max(0, pc.value),
        peakCountAt: at(pc.sample),
        peakRatio: Math.max(0, pr.value),
        peakRatioAt: at(pr.sample),
        cam: pr.sample ? camSnippet(pr.sample.pose) : '',
      };
    });
  }

  private finish(): void {
    this.done = true;
    const summary = this.buildSummary();
    this.updateChip(this.lastT, true);

    // ── 1. console summary table ──
    /* eslint-disable no-console */
    console.log(
      `%c[census] ONE loop complete — ${this.samples.length} samples · ` +
        `scene=${this.params.scene} seed=${this.params.seed} rscale=${RSCALE.toFixed(2)}`,
      'color:#4caf6e;font-weight:bold',
    );
    console.table(
      summary.map((s) => ({
        layer: s.layer,
        peakTris: short(s.peakCount),
        peakCountAtT: s.peakCountAt ? s.peakCountAt.t.toFixed(3) : '-',
        'peak/px': s.peakRatio.toFixed(2) + '×',
        ratioAtT: s.peakRatioAt ? s.peakRatioAt.t.toFixed(3) : '-',
        ratioMpx: s.peakRatioAt ? s.peakRatioAt.mpx.toFixed(2) : '-',
      })),
    );
    console.log('[census] pathology PEAK locations (paste after ?scene=…&):');
    for (const s of summary) {
      if (!s.peakRatioAt || s.peakRatio <= 0) continue;
      console.log(
        `  ${s.layer.padEnd(11)} ${s.peakRatio.toFixed(2)}×  (${short(s.peakCount)} tris)  ` +
          `?cam=${s.cam}`,
      );
    }
    /* eslint-enable no-console */

    // ── 2. downloadable JSON artifact ──
    this.downloadJson(summary);
  }

  private downloadJson(summary: LayerSummary[]): void {
    const payload = {
      meta: {
        kind: 'tri-count-census',
        scene: this.params.scene,
        seed: this.params.seed,
        rscale: RSCALE,
        sampleEveryFrames: SAMPLE_EVERY,
        sampleCount: this.samples.length,
        recordedAt: new Date().toISOString(),
      },
      summary,
      samples: this.samples,
    };
    try {
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `census-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 4000);
    } catch (e) {
      // eslint-disable-next-line no-console
      console.warn('[census] JSON download failed:', e);
    }
  }
}
