/**
 * Composed bookmarks + flythrough (Phase 7, spec §8: "9 bookmarks,
 * 90 s flythrough"). Showcase viewpoints are COMPOSED, not found
 * (Pillar E) — each pairs a verified framing with its best time of day.
 *
 * Keys 1–9 jump to a bookmark (pose + ToD); ?shot=N boots into one.
 * ?fly=1 (or key F) runs a looping ~135 s Catmull-Rom flythrough whose
 * waypoints deliberately cover the worst-overdraw views (deep inside dense
 * forest, oblique-across-canopy, top-down, eye-level look-across, aerial).
 * The spline is re-clamped to ground/water EVERY FRAME (not just at the
 * waypoints) so the interpolated chord never cuts through rising terrain.
 * ?census=1 implies the flythrough and records the per-layer triangle
 * budget for one loop (Census.ts) — the cap right-sizing instrument.
 */

import type { PerspectiveCamera } from 'three';
import { CatmullRomCurve3, Vector3 } from 'three';
import type { Engine } from '../core/Engine';
import type { LaasHooks } from '../core/Hooks';
import type { LaasParams } from '../core/Params';
import type { Heightfield } from '../world/Heightfield';
import { Census } from './Census';

export interface Bookmark {
  name: string;
  x: number;
  z: number;
  /** meters above ground (water-guarded at apply time) */
  alt: number;
  yaw: number;
  pitch: number;
  tod: number;
}

/** nine composed viewpoints — verified framings from the phase shots */
export const BOOKMARKS: Bookmark[] = [
  { name: 'Gorge stream (scene1)', x: 620, z: 650, alt: 1.3, yaw: 0.5, pitch: -0.12, tod: 12.5 },
  { name: 'Dawn lake mist', x: 11, z: 1338, alt: 9, yaw: 1.2, pitch: -0.06, tod: 7.5 },
  { name: 'Golden vista (Witcher)', x: 1500, z: 1900, alt: 250, yaw: 0.65, pitch: -0.18, tod: 19 },
  { name: 'Morning meadow shafts', x: -870, z: 862, alt: 1.8, yaw: -1.45, pitch: 0.02, tod: 8.2 },
  { name: 'Alpine tarn', x: 805, z: -1464, alt: 2.2, yaw: 1.57, pitch: -0.4, tod: 15.5 },
  { name: 'Karst ravine mouth', x: 650, z: 700, alt: 5, yaw: 0.6, pitch: -0.06, tod: 15 },
  { name: 'Forest interior dapple', x: -850, z: 850, alt: 4, yaw: -0.785, pitch: -0.05, tod: 12.5 },
  { name: 'Lakeshore golden', x: -1400, z: 1250, alt: 2.5, yaw: 3.14, pitch: -0.12, tod: 18.5 },
  { name: 'Valley network aerial', x: -600, z: 700, alt: 260, yaw: -0.6, pitch: -0.5, tod: 17.5 },
];

function poseY(hf: Heightfield, b: Bookmark): number {
  const ground = hf.heightAtCpu(b.x, b.z) + b.alt;
  const water = hf.waterYAtCpu(b.x, b.z) + 0.6;
  return Math.max(ground, water);
}

export function installBookmarks(
  engine: Engine,
  hf: Heightfield,
  hooks: LaasHooks,
  params: LaasParams,
): void {
  const apply = (i: number): void => {
    const b = BOOKMARKS[i];
    if (!b) return;
    hooks.setPose?.({ p: [b.x, poseY(hf, b), b.z], yaw: b.yaw, pitch: b.pitch });
    hooks.setTimeOfDay?.(b.tod);
  };

  window.addEventListener('keydown', (e) => {
    const m = /^Digit([1-9])$/.exec(e.code);
    if (m) apply(Number(m[1]) - 1);
    if (e.code === 'KeyF') fly.toggle();
  });

  // ---- flythrough -------------------------------------------------------------
  // Longer than the old 92 s so the 14-segment tour (which now deliberately
  // visits the worst-overdraw views for the census) is never rushed.
  const FLY_SECONDS = 135;
  // Ground/water clearance for the PER-FRAME re-clamp (mirrors FlyCamera's
  // FLY_GROUND_CLEAR / WADE_CLEAR — kept local since those aren't exported).
  const FLY_GROUND_CLEAR = 1.4;
  const WADE_CLEAR = 0.45;

  // The tour is authored to COVER the pathology views, not just be cinematic
  // (user: "both oblique views, deep inside dense forest … all the possible
  // views!"). One continuous loop, yaws authored UNWRAPPED and monotonically
  // increasing ~one full turn so per-segment linear interp never spins.
  // The dense-forest anchor is the verified interior around (-850, 850)
  // ('Forest interior dapple' / 'Morning meadow shafts').
  const TOUR: { x: number; z: number; alt: number; yaw: number; pitch: number }[] = [
    // 1. aerial establishing vista (loop anchor)
    { x: 1500, z: 1900, alt: 250, yaw: 0.65, pitch: -0.18 },
    // 2. aerial descent into the valley
    { x: 600, z: 1500, alt: 110, yaw: 1.1, pitch: -0.14 },
    // 3. lake / water variety, low over the shore
    { x: 11, z: 1338, alt: 12, yaw: 1.2, pitch: -0.05 },
    // 4. LOW OBLIQUE skimming across the canopy toward the forest (alt ~35, shallow)
    { x: -500, z: 1050, alt: 35, yaw: 2.0, pitch: -0.06 },
    // 5. drop to canopy level at the forest edge
    { x: -780, z: 900, alt: 18, yaw: 2.4, pitch: -0.05 },
    // 6. DEEP INSIDE dense forest, low, looking ahead (2–5 m under closed canopy)
    { x: -850, z: 850, alt: 3.5, yaw: 2.6, pitch: 0.0 },
    // 7. still inside — pitched UP through the canopy shafts
    { x: -872, z: 820, alt: 3.0, yaw: 2.9, pitch: 0.35 },
    // 8. LONG HORIZONTAL dense-forest look-across at eye level
    { x: -905, z: 862, alt: 2.5, yaw: 3.6, pitch: -0.02 },
    // 9. climb into a LOW OBLIQUE across the canopy tops
    { x: -820, z: 900, alt: 42, yaw: 4.2, pitch: -0.25 },
    // 10. TOP-DOWN over dense canopy (high alt, steep pitch ≤ -1.2)
    { x: -850, z: 860, alt: 150, yaw: 4.9, pitch: -1.35 },
    // 11. AERIAL pull-out over forest + valley
    { x: -600, z: 720, alt: 240, yaw: 5.6, pitch: -0.5 },
    // 12. mid-alt valley / gorge variety
    { x: 300, z: 680, alt: 60, yaw: 6.3, pitch: -0.15 },
    // 13. gorge stream, low and close
    { x: 620, z: 655, alt: 6, yaw: 6.9, pitch: -0.06 },
    // 14. climb back to the vista to close the loop (yaw = start + 2π)
    { x: 1500, z: 1900, alt: 250, yaw: 0.65 + Math.PI * 2, pitch: -0.18 },
  ];

  /** ground/water floor at (x,z) with clearance — the per-frame lift target. */
  const floorAt = (x: number, z: number): number =>
    Math.max(hf.heightAtCpu(x, z) + FLY_GROUND_CLEAR, hf.waterYAtCpu(x, z) + WADE_CLEAR);

  class Flythrough {
    private active = false;
    private t = 0;
    private curve: CatmullRomCurve3 | null = null;
    private smoothY = 0;
    private census: Census | null = null;

    get isActive(): boolean {
      return this.active;
    }

    setCensus(c: Census): void {
      this.census = c;
    }

    toggle(): void {
      this.active = !this.active;
      hooks.flyCamEnabled?.(!this.active);
      if (this.active && !this.curve) {
        this.curve = new CatmullRomCurve3(
          TOUR.map((w) => new Vector3(w.x, poseY(hf, { ...w, tod: 0, name: '' } as Bookmark), w.z)),
          false,
          'centripetal',
          0.5,
        );
      }
      if (!this.active) this.t = 0;
    }

    update(dt: number, cam: PerspectiveCamera): void {
      if (!this.active || !this.curve) return;
      this.t = (this.t + dt / FLY_SECONDS) % 1;
      const u = this.t;
      const p = this.curve.getPointAt(u);

      // PER-FRAME ground/water re-clamp (LIFT-ONLY): the raw spline chord cuts
      // through rising terrain between low waypoints. Look a short way AHEAD
      // along the curve and take the MAX floor so the camera starts rising
      // before a ridge (no clipping), then a light lerp toward that target
      // avoids a pop on sharp crests. A final hard max() is the safety floor.
      let targetFloor = floorAt(p.x, p.z);
      for (const du of [0.004, 0.008, 0.012]) {
        const q = this.curve.getPointAt((u + du) % 1);
        targetFloor = Math.max(targetFloor, floorAt(q.x, q.z));
      }
      const desired = Math.max(p.y, targetFloor);
      if (this.smoothY === 0) this.smoothY = desired; // init on first frame
      const damp = 1 - Math.exp(-dt * 4);
      this.smoothY += (desired - this.smoothY) * damp;
      // never let the smoothed value dip below the instantaneous hard floor
      p.y = Math.max(this.smoothY, floorAt(p.x, p.z));

      cam.position.copy(p);
      // yaw/pitch: linear over the waypoint list (yaws authored unwrapped)
      const seg = u * (TOUR.length - 1);
      const i0 = Math.min(Math.floor(seg), TOUR.length - 2);
      const f = seg - i0;
      const w0 = TOUR[i0];
      const w1 = TOUR[i0 + 1];
      if (!w0 || !w1) return;
      const yaw = w0.yaw + (w1.yaw - w0.yaw) * f;
      const pitch = w0.pitch + (w1.pitch - w0.pitch) * f;
      hooks.setPose?.({ p: [p.x, p.y, p.z], yaw, pitch });
      this.census?.tick(this.t, { x: p.x, y: p.y, z: p.z, yaw, pitch });
    }
  }
  const fly = new Flythrough();
  engine.onUpdate((dt) => fly.update(dt, engine.camera));

  const q = new URLSearchParams(window.location.search);
  // ?census=1 — the tri-count census. Implies the flythrough; records EXACTLY
  // one loop, then dumps a console table + downloadable JSON. Constructed ONLY
  // when requested (zero footprint otherwise).
  const censusOn = q.get('census') === '1';
  if (censusOn) fly.setCensus(new Census(engine, params));
  if (censusOn || q.get('fly') === '1') {
    if (!fly.isActive) fly.toggle();
  }

  // boot directly into a bookmark (?shot=N) — pose via initialPose (the
  // fly rig applies it after this scene finishes building)
  if (params.shot !== null && params.cam === null) {
    const b = BOOKMARKS[params.shot - 1];
    if (b) {
      hooks.initialPose = { p: [b.x, poseY(hf, b), b.z], yaw: b.yaw, pitch: b.pitch };
    }
  }
}
