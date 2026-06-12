/**
 * Global hooks contract between the running app and external tooling
 * (Playwright verification harness reads/writes `window.__laas`).
 */

export interface CamPose {
  /** world position */
  p: [number, number, number];
  /** yaw (rad, around +Y), pitch (rad) */
  yaw: number;
  pitch: number;
  /** optional fov override (deg) */
  fov?: number;
}

export interface EngineStats {
  fps: number;
  frameMs: number;
  frameMsP95: number;
  drawCalls: number;
  triangles: number;
  frame: number;
  /** named counters merged in by subsystems (instances per category, cull stats, vram…) */
  counters: Record<string, number>;
  /** per-pass GPU timings in ms when timestamp-query is available */
  gpuPasses: Record<string, number>;
}

export interface GpuDiagnostics {
  ok: boolean;
  reason?: string;
  vendor?: string;
  architecture?: string;
  device?: string;
  description?: string;
  features: string[];
  limits: Record<string, number>;
}

export interface LaasHooks {
  /** true once the first frames have rendered and the GPU pipeline is verified */
  ready: boolean;
  /** set on fatal error (also rendered as a fail-loud overlay) */
  error: string | null;
  stats: EngineStats | null;
  diag: GpuDiagnostics | null;
  /** world-gen / scene progress 0..1 (boot UI + tooling wait on this) */
  progress: number;
  progressMsg: string;
  /** tooling control surface */
  setPose: ((pose: CamPose) => void) | null;
  getPose: (() => CamPose) | null;
  /** scene-requested spawn pose (?alt/x/z/yaw/pitch) — main applies it once
   *  the fly camera exists (scenes build BEFORE the camera rig) */
  initialPose: CamPose | null;
  /** 'walk' only for the default interactive spawn (no explicit pose
   *  params) — every explicit/programmatic pose keeps fly semantics */
  initialPoseMode: 'walk' | 'fly' | null;
  /** terrain/water heights at (x, z) — walk mode + fly soft collision */
  groundProbe: ((x: number, z: number) => { ground: number; water: number }) | null;
  setTimeOfDay: ((t: number) => void) | null;
  /** settle frames (TAA/temporal effects) then resolve — call before screenshots */
  settle: ((frames?: number) => Promise<void>) | null;
  /** enable/disable fly-camera input (flythrough takes the wheel) */
  flyCamEnabled: ((on: boolean) => void) | null;
}

declare global {
  interface Window {
    __laas: LaasHooks;
  }
}

export function initHooks(): LaasHooks {
  const hooks: LaasHooks = {
    ready: false,
    error: null,
    stats: null,
    diag: null,
    progress: 0,
    progressMsg: 'boot',
    setPose: null,
    getPose: null,
    initialPose: null,
    initialPoseMode: null,
    groundProbe: null,
    setTimeOfDay: null,
    settle: null,
    flyCamEnabled: null,
  };
  window.__laas = hooks;
  return hooks;
}
