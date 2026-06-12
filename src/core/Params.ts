/** URL parameter parsing — every run is fully described by its URL. */

export type QualityPreset = 'low' | 'high' | 'ultra';

export interface LaasParams {
  /** world seed — reproduces the entire world */
  seed: number;
  /** scene to boot: world | sanity | terrain | gallery (registry in debug/Scenes.ts) */
  scene: string;
  /** time of day, hours 0..24 */
  timeOfDay: number;
  /** quality preset: low (iGPU floor), high (default), ultra (max grids) */
  preset: QualityPreset;
  /** HUD visible at boot */
  hud: boolean;
  /** camera pose: "px,py,pz,yaw,pitch[,fov]" */
  cam: string | null;
  /** bookmark index to start at (1..9) */
  shot: number | null;
  /** freeze world time/motion (deterministic screenshots) */
  freeze: boolean;
  /** device pixel ratio cap override */
  dpr: number | null;
}

function num(v: string | null, fallback: number): number {
  if (v === null) return fallback;
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

export function parseParams(search: string = window.location.search): LaasParams {
  const q = new URLSearchParams(search);
  const presetRaw = q.get('preset') ?? 'high';
  const preset: QualityPreset =
    presetRaw === 'low' || presetRaw === 'ultra' ? presetRaw : 'high';
  const shotN = num(q.get('shot'), 0);
  return {
    seed: Math.floor(num(q.get('seed'), 1)) >>> 0,
    scene: q.get('scene') ?? 'world',
    timeOfDay: Math.min(24, Math.max(0, num(q.get('T'), 11))),
    preset,
    // full debug panel hidden by default — F3 toggles it (fps chip always on)
    hud: q.get('hud') === '1',
    cam: q.get('cam'),
    shot: shotN >= 1 && shotN <= 9 ? Math.floor(shotN) : null,
    freeze: q.get('freeze') === '1',
    dpr: q.get('dpr') !== null ? num(q.get('dpr'), 1) : null,
  };
}

/** Parse a `cam` string into pose components; returns null when malformed. */
export function parseCamString(
  cam: string,
): { p: [number, number, number]; yaw: number; pitch: number; fov?: number } | null {
  const parts = cam.split(',').map(Number);
  if (parts.length < 5 || parts.some((v) => !Number.isFinite(v))) return null;
  const [px, py, pz, yaw, pitch, fov] = parts as [number, number, number, number, number, number?];
  const pose = { p: [px, py, pz] as [number, number, number], yaw, pitch };
  return fov !== undefined && Number.isFinite(fov) ? { ...pose, fov } : pose;
}
