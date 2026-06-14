/**
 * Per-pass GPU timing attribution (Phase 7 perf directive: measure first).
 *
 * three 0.184 already allocates a timestamp-query pair per render context /
 * compute call and resolves per-uid durations into the backend's
 * TimestampQueryPool — but only ever exposes the per-frame SUM via
 * renderer.info. This profiler:
 *   1. wraps backend.updateTimeStampUID to remember a human label per uid
 *      (compute → ComputeNode.name; render → tagged renderTarget /
 *      texture name / 'screen'),
 *   2. after each resolveTimestampsAsync, aggregates the newest complete
 *      frame's per-uid durations by label into stats.gpuPasses,
 *   3. prunes the pool's uid→duration map (three never clears it — it
 *      grows per resolve batch) and our label map alongside.
 *
 * Pool-overflow note: the 2048-query pool only resets its write index on
 * resolve. World-gen issues thousands of compute calls in frame 0, so the
 * pool overflows once at boot (warnOnce, untimed — boot has its own
 * timers) and recovers at the first steady-state resolve. Steady state
 * stays well under the limit because the Engine now resolves EVERY frame.
 */

import type { WebGPURenderer } from 'three/webgpu';

interface PoolShape {
  timestamps?: Map<string, number>;
}

interface BackendShape {
  updateTimeStampUID(ctx: object): void;
  get(obj: object): { timestampUID?: string };
  timestampQueryPool: Record<string, PoolShape | null | undefined>;
}

interface RenderTargetShape {
  width?: number;
  height?: number;
  texture?: { name?: string };
  textures?: { name?: string }[];
}

interface CtxShape {
  isComputeNode?: boolean;
  name?: string;
  renderTarget?: RenderTargetShape | null;
  camera?: { name?: string } | null;
}

/** explicit labels for objects whose auto-name is opaque (render targets, kernel arrays) */
const tags = new WeakMap<object, string>();

export function tagGpu(obj: object, label: string): void {
  tags.set(obj, label);
}

const FRAME_RE = /:f(\d+)$/;

function frameOf(uid: string): number {
  const m = FRAME_RE.exec(uid);
  return m ? Number(m[1]) : -1;
}

export class GpuProfiler {
  private uidLabels = new Map<string, string>();
  private backend: BackendShape;
  /** RenderTarget has no id field — assign stable indices by first appearance */
  private rtIds = new WeakMap<object, number>();
  private rtCount = 0;
  /** distinct ShadowMap render targets → stable c0..cN suffix */
  private shadowIds = new WeakMap<object, number>();
  private shadowCount = 0;

  constructor(renderer: WebGPURenderer) {
    this.backend = renderer.backend as unknown as BackendShape;
    const orig = this.backend.updateTimeStampUID.bind(this.backend);
    this.backend.updateTimeStampUID = (ctx: object): void => {
      orig(ctx);
      const uid = this.backend.get(ctx).timestampUID;
      if (uid !== undefined) this.uidLabels.set(uid, this.describe(ctx));
    };
  }

  private describe(ctx: object): string {
    const tagged = tags.get(ctx);
    if (tagged !== undefined) return tagged;
    if (Array.isArray(ctx)) {
      for (const n of ctx as CtxShape[]) {
        const t = tags.get(n as object) ?? (n.name !== undefined && n.name !== '' ? n.name : undefined);
        if (t !== undefined) return t;
      }
      return 'compute[]';
    }
    const c = ctx as CtxShape;
    if (c.isComputeNode === true) {
      return c.name !== undefined && c.name !== '' ? c.name : 'compute?';
    }
    const rt = c.renderTarget;
    if (rt === null || rt === undefined) return 'screen';
    const rtTag = tags.get(rt as object);
    if (rtTag !== undefined) return rtTag;
    const tn = rt.textures?.[0]?.name ?? rt.texture?.name ?? '';
    if (tn === 'ShadowMap') {
      let n = this.shadowIds.get(rt as object);
      if (n === undefined) {
        n = this.shadowCount++;
        this.shadowIds.set(rt as object, n);
      }
      return `shadow.c${n}`;
    }
    const cam = c.camera?.name !== undefined && c.camera.name !== '' ? `@${c.camera.name}` : '';
    if (tn !== '') return tn + cam;
    let id = this.rtIds.get(rt as object);
    if (id === undefined) {
      id = this.rtCount++;
      this.rtIds.set(rt as object, id);
    }
    return `rt#${id}(${rt.width ?? 0}x${rt.height ?? 0})${cam}`;
  }

  /**
   * Rebuild `out` with the newest resolved frame's per-pass durations:
   * 'render'/'compute' totals + 'r.<label>' / 'c.<label>' entries.
   * Prunes consumed (older-frame) entries from the pools and label map.
   */
  collect(out: Record<string, number>): void {
    const fresh: Record<string, number> = {};
    for (const [type, prefix] of [
      ['render', 'r'],
      ['compute', 'c'],
    ] as const) {
      const pool = this.backend.timestampQueryPool[type];
      const ts = pool?.timestamps;
      if (!ts || ts.size === 0) continue;
      let newest = -1;
      for (const uid of ts.keys()) {
        const f = frameOf(uid);
        if (f > newest) newest = f;
      }
      let total = 0;
      let garbage = 0;
      for (const [uid, ms] of ts) {
        const f = frameOf(uid);
        if (f !== newest) {
          ts.delete(uid);
          this.uidLabels.delete(uid);
          continue;
        }
        const key = `${prefix}.${this.uidLabels.get(uid) ?? uid.replace(FRAME_RE, '')}`;
        // INTEGRITY GUARD (perf directive: "no guessing"): three occasionally
        // resolves a GARBAGE (negative / non-finite) duration for some keep-alive
        // passes — notably the dead CSM cascade maps (r.shadow.c*), which were
        // reporting ≈ −69 ms and dragging the whole `render` total to −97 ms.
        // A poisoned total silently invalidates EVERY downstream measurement
        // (and zeroes shoot.ts --gpusample, whose gate is render+compute>0), so
        // never let a single misbehaving pass corrupt the aggregate: keep the
        // key VISIBLE (as 0, so the pass is not hidden) and tally a diagnostic
        // count instead of summing the garbage.
        if (!Number.isFinite(ms) || ms < 0) {
          fresh[key] = fresh[key] ?? 0;
          garbage++;
          continue;
        }
        fresh[key] = (fresh[key] ?? 0) + ms;
        total += ms;
      }
      fresh[type] = total;
      if (garbage > 0) fresh[`${prefix}.__garbage`] = garbage;
    }
    if (Object.keys(fresh).length === 0) return; // no resolve landed yet — keep last
    for (const k of Object.keys(out)) {
      if (!(k in fresh)) delete out[k];
    }
    Object.assign(out, fresh);
    // labels for uids that never get resolved (e.g. pool overflow) would
    // otherwise accumulate forever
    if (this.uidLabels.size > 8192) this.uidLabels.clear();
  }
}
