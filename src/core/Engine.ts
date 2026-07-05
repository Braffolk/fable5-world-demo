/**
 * Engine — owns the WebGPURenderer, camera, frame loop, time, and stats.
 * All subsystems hook in via onUpdate(); per-frame stats are mirrored to
 * `window.__laas.stats` for the verification harness.
 */

import { ACESFilmicToneMapping, PerspectiveCamera, Scene } from 'three';
import { TimestampQuery, WebGPURenderer } from 'three/webgpu';
import { buildRequiredLimits } from './Diagnostics';
import { installFragmentStorageWrites, installMaterialKeyMemo } from '../render/ThreePatches';
import { installPositionInvariance } from '../render/VegPrepass';
import { GpuProfiler } from './GpuProfiler';
import type { EngineStats, LaasHooks } from './Hooks';
import type { LaasParams } from './Params';

export type UpdateFn = (dt: number, worldTime: number) => void;

const P95_WINDOW = 120;

export class Engine {
  // NOT readonly: ?profile=1 swaps the renderer (+ its device) after loading so a
  // DAWN_TRACE_DEVICE_FILTER=laas-render capture is game-only (see swapRenderer).
  renderer: WebGPURenderer;
  readonly scene: Scene;
  readonly camera: PerspectiveCamera;
  readonly params: LaasParams;
  readonly hooks: LaasHooks;
  readonly stats: EngineStats;

  /** world-simulation time (sec) — frozen when ?freeze=1 */
  worldTime = 0;
  /** wall-clock elapsed (sec) since start */
  elapsed = 0;

  /** when set, the frame loop renders through this instead of renderer.render.
   *  meterRead (optional, measure-infra W5): the frame's counter READBACKS factored
   *  out of meter() so MeasureHarness can run them OUTSIDE the timed window. */
  post: {
    render(): void;
    meter(renderer: WebGPURenderer): void;
    meterRead?(renderer: WebGPURenderer): Promise<Record<string, number>>;
  } | null = null;

  /** measure-infra W5: MeasureHarness sets this while measuring — meter() then skips
   *  its every-15th-frame async readbacks (4-5 buffer→staging submits + mapAsync) so
   *  they never land inside an isolated timed frame. Auto-exposure (frame content)
   *  still runs. Live loop unaffected. */
  meterQuiet = false;

  /** DPR cap resolved at create() — reused when ?profile swaps the renderer. */
  private dprCap = 1;
  private updateFns: UpdateFn[] = [];
  private lastT: number | null = null;
  private frameMsRing: number[] = [];
  private fpsEma = 0;
  private frameCounter = 0;
  private settleWaiters: { frames: number; resolve: () => void }[] = [];
  private timestampsSupported = false;
  private timestampPending = false;
  private profiler: GpuProfiler | null = null;

  private constructor(renderer: WebGPURenderer, params: LaasParams, hooks: LaasHooks) {
    this.renderer = renderer;
    this.params = params;
    this.hooks = hooks;
    this.scene = new Scene();
    this.camera = new PerspectiveCamera(
      55,
      window.innerWidth / window.innerHeight,
      0.3,
      30000,
    );
    this.camera.position.set(0, 10, 30);
    this.stats = {
      fps: 0,
      frameMs: 0,
      frameMsP95: 0,
      drawCalls: 0,
      triangles: 0,
      frame: 0,
      counters: {},
      gpuPasses: {},
    };
    hooks.stats = this.stats;
  }

  static async create(
    params: LaasParams,
    hooks: LaasHooks,
    deviceLabel = 'laas-render',
  ): Promise<Engine> {
    const { renderer } = await Engine.createDeviceRenderer(deviceLabel, hooks.diag ?? null);
    const dprCap = params.dpr ?? Math.min(window.devicePixelRatio, 1.5);
    renderer.setPixelRatio(dprCap);
    renderer.setSize(window.innerWidth, window.innerHeight);
    // Temporary Phase-0 output transform; replaced by the post stack (Phase 2).
    renderer.toneMapping = ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    renderer.shadowMap.enabled = true;

    const container = document.getElementById('app');
    if (!container) throw new Error('#app container missing in index.html');
    container.appendChild(renderer.domElement);

    const engine = new Engine(renderer, params, hooks);
    engine.dprCap = dprCap;
    engine.timestampsSupported = (hooks.diag?.features ?? []).includes('timestamp-query');
    if (engine.timestampsSupported) engine.profiler = new GpuProfiler(renderer);
    // depth-prepass correctness (see VegPrepass): position math must land
    // on identical depths across the depth-only and shaded pipelines
    installPositionInvariance(renderer);
    // shadow-pass render objects re-hash their material node graph every
    // frame (see ThreePatches) — memoize per material
    installMaterialKeyMemo(renderer);
    // opt-in fragment-stage storage writes (nanite vis-buffer HW path);
    // inert unless a buffer is marked via markFragmentWritable
    installFragmentStorageWrites(renderer);

    window.addEventListener('resize', () => {
      engine.camera.aspect = window.innerWidth / window.innerHeight;
      engine.camera.updateProjectionMatrix();
      // engine.renderer (not the create-time local) so resize follows a ?profile swap
      engine.renderer.setSize(window.innerWidth, window.innerHeight);
    });
    return engine;
  }

  /**
   * Create a {device, renderer} pair with the given device label, using the same
   * feature/limit/renderer recipe as create(). Shared by create() and the
   * ?profile=1 two-device swap (ProfileBoot) — the label is what
   * DAWN_TRACE_DEVICE_FILTER keys on, so the loading pass uses 'laas-loading'
   * and the traced game pass uses 'laas-render'.
   */
  static async createDeviceRenderer(
    label: string,
    diag: Parameters<typeof buildRequiredLimits>[0] | null,
  ): Promise<{ device: GPUDevice; renderer: WebGPURenderer }> {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No GPU adapter found');
    const device = await adapter.requestDevice({
      label,
      requiredFeatures: [
        ...(adapter.features as Set<GPUFeatureName>),
        'timestamp-query' as GPUFeatureName,
      ],
      requiredLimits: diag ? buildRequiredLimits(diag) : {},
    });
    const renderer = new WebGPURenderer({ antialias: false, trackTimestamp: true, device });
    await renderer.init();
    // fail-loud: surface WebGPU validation errors (otherwise: silent black frames)
    let reported = 0;
    device.onuncapturederror = (e: GPUUncapturedErrorEvent): void => {
      if (reported++ < 8) {
        // eslint-disable-next-line no-console
        console.error('[laas] WebGPU uncaptured error:', e.error.message);
      }
    };
    return { device, renderer };
  }

  /**
   * ?profile=1 two-device split: replace the live renderer (and its device)
   * AFTER loading, so a DAWN_TRACE_DEVICE_FILTER=laas-render capture contains
   * only game frames, not the ~11 GB of boot GPU compute. The new renderer's
   * backend has an empty per-object resource cache, so it re-uploads every
   * CPU-backed scene resource (geometry/nanite StorageBufferAttributes, data
   * textures) lazily on the first render. GPU-only resources with no CPU copy
   * (the terrain/bark/canopy StorageTextures) are transferred separately by
   * ProfileBoot; subsystems that STORE a renderer ref (sky/atmosphere/half-res)
   * are re-pointed by ProfileBoot, not here.
   */
  swapRenderer(newRenderer: WebGPURenderer): void {
    const old = this.renderer;
    newRenderer.setPixelRatio(this.dprCap);
    newRenderer.setSize(window.innerWidth, window.innerHeight);
    newRenderer.toneMapping = ACESFilmicToneMapping;
    newRenderer.toneMappingExposure = 1.0;
    newRenderer.shadowMap.enabled = true;
    const parent = old.domElement.parentElement;
    if (parent) {
      parent.removeChild(old.domElement);
      parent.appendChild(newRenderer.domElement);
    }
    this.renderer = newRenderer;
    // re-apply the create()-time renderer installs onto the new backend
    installPositionInvariance(newRenderer);
    installMaterialKeyMemo(newRenderer);
    installFragmentStorageWrites(newRenderer);
    if (this.timestampsSupported) this.profiler = new GpuProfiler(newRenderer);
  }

  onUpdate(fn: UpdateFn): void {
    this.updateFns.push(fn);
  }

  /** resolves after `frames` additional frames have been rendered */
  settle(frames = 8): Promise<void> {
    return new Promise((resolve) => {
      this.settleWaiters.push({ frames, resolve });
    });
  }

  start(): void {
    void this.renderer.setAnimationLoop((timeMs) => this.frame(timeMs));
  }

  /**
   * Advance + render ONE frame's worth of work (update fns → render/post),
   * recording cpu.update/submit. Shared by the rAF loop AND the manual
   * measurement harness (MeasureHarness) — the harness drives this directly,
   * draining the GPU around it, so the per-pass timestamp is honest active
   * GPU time and not the vsync cross-frame pipelining span (see MeasureHarness).
   * Returns the wall-clock submit span (ms) so the caller can attribute.
   */
  renderStep(dt: number): void {
    this.elapsed += dt;
    if (!this.params.freeze) this.worldTime += dt;

    // CPU attribution (Phase 7): update = app-side per-frame work,
    // submit = three render+encode (excl. GPU; backpressure shows as the
    // gap between frameMs and cpu.update+cpu.submit)
    const c0 = performance.now();
    for (const fn of this.updateFns) fn(dt, this.worldTime);
    const c1 = performance.now();

    if (this.post) {
      this.post.meter(this.renderer); // exposure feedback from last frame's pass
      this.post.render();
    } else {
      this.renderer.render(this.scene, this.camera);
    }
    const c2 = performance.now();
    this.stats.counters['cpu.updateMs100'] = Math.round((c1 - c0) * 100);
    this.stats.counters['cpu.submitMs100'] = Math.round((c2 - c1) * 100);
  }

  /** the WebGPU device (for GPU-drain barriers in the measurement harness) */
  get device(): GPUDevice | null {
    return (this.renderer.backend as unknown as { device?: GPUDevice }).device ?? null;
  }

  /** the live GpuProfiler, or null when timestamp-query is unsupported */
  get gpuProfiler(): GpuProfiler | null {
    return this.profiler;
  }

  private frame(timeMs: number): void {
    const t = timeMs / 1000;
    const rawDt = this.lastT === null ? 1 / 60 : t - this.lastT;
    this.lastT = t;
    const dt = Math.min(Math.max(rawDt, 0), 0.1);
    this.renderStep(dt);
    this.collectStats(rawDt);

    if (this.settleWaiters.length > 0) {
      for (const w of this.settleWaiters) w.frames -= 1;
      const done = this.settleWaiters.filter((w) => w.frames <= 0);
      this.settleWaiters = this.settleWaiters.filter((w) => w.frames > 0);
      for (const w of done) w.resolve();
    }
  }

  private collectStats(rawDt: number): void {
    const s = this.stats;
    const ms = rawDt * 1000;
    this.frameMsRing.push(ms);
    if (this.frameMsRing.length > P95_WINDOW) this.frameMsRing.shift();
    const sorted = [...this.frameMsRing].sort((a, b) => a - b);
    const p95 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))] ?? ms;
    const fpsNow = rawDt > 0 ? 1 / rawDt : 0;
    this.fpsEma = this.fpsEma === 0 ? fpsNow : this.fpsEma * 0.95 + fpsNow * 0.05;

    s.fps = this.fpsEma;
    s.frameMs = ms;
    s.frameMsP95 = p95;
    s.drawCalls = this.renderer.info.render.drawCalls;
    s.triangles = this.renderer.info.render.triangles;
    s.frame = this.frameCounter++;
    // GPU resource accounting (perf-drift hunt): if these CLIMB frame-over-frame at a
    // static camera, something is leaking GPU objects (the "accumulates with time" bug).
    const mem = this.renderer.info.memory as { geometries?: number; textures?: number; buffers?: number };
    s.counters['gpu.geometries'] = mem.geometries ?? -1;
    s.counters['gpu.textures'] = mem.textures ?? -1;
    s.counters['gpu.buffers'] = mem.buffers ?? -1;

    // resolve EVERY frame: the 2048-query pool only resets its write index
    // on resolve — the old every-10-frames cadence overflowed it (≈100
    // timed contexts/frame), killing per-pass attribution and warning once
    if (this.timestampsSupported && !this.timestampPending) {
      this.timestampPending = true;
      Promise.all([
        this.renderer.resolveTimestampsAsync(TimestampQuery.RENDER),
        this.renderer.resolveTimestampsAsync(TimestampQuery.COMPUTE),
      ])
        .then(() => {
          if (this.profiler) {
            this.profiler.collect(s.gpuPasses);
          } else {
            s.gpuPasses['render'] = this.renderer.info.render.timestamp;
            s.gpuPasses['compute'] = this.renderer.info.compute.timestamp;
          }
        })
        .catch(() => {
          /* timestamps unsupported mid-run — ignore */
        })
        .finally(() => {
          this.timestampPending = false;
        });
    }
  }
}
