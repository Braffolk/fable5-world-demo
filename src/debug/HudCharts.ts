/**
 * HudCharts — tiny dependency-free time-series ring + inline-canvas sparkline for
 * the F3 overlay. Kept separate from HUD.ts so the panel logic stays readable.
 *
 * The Series model is deliberately the shape a future fly-through census recorder
 * wants: a fixed-length ring of recent samples PLUS a running max (peak) that never
 * decays. A recorder hook ("record window + running max per layer") is then a drop-in:
 * read `.max` for the peak and `.window()` for the trace — no reshaping needed.
 */

/** A bounded time-series with a running (non-decaying) peak. */
export class Series {
  private buf: Float32Array;
  private idx = 0;
  private filled = 0;
  /** running peak since construction / last resetMax() — the census "running max". */
  max = 0;
  /** most recent pushed sample. */
  last = 0;

  constructor(capacity = 128) {
    this.buf = new Float32Array(capacity);
  }

  push(v: number): void {
    const x = Number.isFinite(v) ? v : 0;
    this.buf[this.idx % this.buf.length] = x;
    this.idx += 1;
    this.filled = Math.min(this.filled + 1, this.buf.length);
    this.last = x;
    if (x > this.max) this.max = x;
  }

  resetMax(): void {
    this.max = this.last;
  }

  /** samples oldest→newest, length = filled. */
  window(): number[] {
    const n = this.filled;
    const start = this.idx - n;
    const out: number[] = new Array(n);
    for (let i = 0; i < n; i++) out[i] = this.buf[(start + i) % this.buf.length] ?? 0;
    return out;
  }

  /** peak over just the retained window (for sparkline auto-scale). */
  windowMax(): number {
    let m = 0;
    const n = this.filled;
    const start = this.idx - n;
    for (let i = 0; i < n; i++) {
      const v = this.buf[(start + i) % this.buf.length] ?? 0;
      if (v > m) m = v;
    }
    return m;
  }
}

/**
 * Draw a filled sparkline into a canvas 2D context. `scaleMax` fixes the vertical
 * scale (0..scaleMax); pass the series' windowMax for auto-scale. Cheap — a single
 * path per tick, called at the panel's 4 Hz.
 */
export function drawSparkline(
  g: CanvasRenderingContext2D,
  w: number,
  h: number,
  samples: number[],
  scaleMax: number,
  color: string,
): void {
  g.clearRect(0, 0, w, h);
  const n = samples.length;
  if (n === 0) return;
  const denom = scaleMax > 0 ? scaleMax : 1;
  const y = (v: number): number => h - 1 - (Math.min(v, denom) / denom) * (h - 2);
  const dx = n > 1 ? w / (n - 1) : w;
  g.beginPath();
  g.moveTo(0, h);
  for (let i = 0; i < n; i++) g.lineTo(i * dx, y(samples[i] ?? 0));
  g.lineTo((n - 1) * dx, h);
  g.closePath();
  g.fillStyle = color.replace('rgb', 'rgba').replace(')', ',0.22)');
  g.fill();
  g.beginPath();
  for (let i = 0; i < n; i++) {
    const yy = y(samples[i] ?? 0);
    if (i === 0) g.moveTo(0, yy);
    else g.lineTo(i * dx, yy);
  }
  g.strokeStyle = color;
  g.lineWidth = 1;
  g.stroke();
}
