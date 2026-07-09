/** LOW / MEDIUM / HIGH quality presets.
 *
 * A preset is a named bundle of URL-param DEFAULTS. The URL params are the
 * debug/harness interface and MUST keep working, so the preset never
 * overrides a param that is explicitly present — it only fills in absent
 * keys, exactly the `expandPureAblation` pattern in main.ts (rewrite the URL
 * ONCE via history.replaceState, before any subsystem reads it).
 *
 * Applied from src/boot.ts BEFORE the main module graph loads (dynamic
 * import barrier), so both lazy and module-eval param reads see the preset,
 * and boot-cache keys pick up the RESOLVED values (WorldRegistry resolves
 * knobs from location.search live — each preset gets its own cache entries;
 * they coexist in IndexedDB).
 *
 * MEDIUM is the shipped default: it injects nothing except the one
 * user-requested tweak (slightly lower sun-shadow clipmap resolution).
 * Harness/automation boots (navigator.webdriver, or any ?scene/?quality URL)
 * resolve MEDIUM with zero UI delay — see boot.ts.
 */

export type QualityTier = 'low' | 'medium' | 'high';

/** Per-tier URL-param defaults. Every entry must be justified by a measured
 *  ms delta AND (for HIGH) a visible A/B shot difference — see
 *  docs/perf-runs notes for the 2026-07-04 quality-preset tuning run. */
const PRESET_PARAMS: Record<QualityTier, Record<string, string>> = {
  // LOW — weaker machines. Shadows down more (one fewer clipmap level + 768²),
  // cheaper grass (shorter ray band + single overlay layer), voxel handoff pulled
  // in (meshes give way to cheaper voxels sooner). PRESET VALUES ONLY — no engine
  // edits. Measured 2026-07-04: see docs/perf-runs quality-preset run.
  low: {
    shadowcliplevels: '5',
    shadowclipres: '768',
    voxnear: '40',
    grassrayend: '90',
    grasslayers: '1',
  },
  // MEDIUM — today's shipped defaults, with the ONE user-requested tweak: sun
  // shadow clipmap res 1024→896 (−1.4 ms on moving legs, invisible at stills;
  // runtime knob ⇒ shares the DEFAULT bootcache key, stays warm). See
  // docs/perf-runs 2026-07-04 shadow-trim moving-leg A/B.
  medium: {
    shadowclipres: '896',
  },
  // HIGH — maximum detail (perf downside accepted). Every knob below is justified
  // by a visible A/B still AND a measured cost; costs-but-looks-the-same knobs are
  // FORBIDDEN. 2026-07-04 A/B verdicts:
  //   voxnear=90  KEPT — mesh→voxel handoff 60→90 m: the 45-90 m band turns from
  //               rounded voxel blobs into real leaf geometry (dramatic in the
  //               oblique A/B), no regression at eye/mid poses.
  //   voxgrid=384 DROPPED — ~2× frame cost, crowns not visibly better at any pose.
  //   voxcellmin=16/0 DROPPED — volumetric far bricks invisible under auto-rscale
  //               (oblique/aerial/far-ridge stills) while costing 3-10 ms.
  //   shadowclipres stays at the engine default (1024) — full shadow res.
  high: {
    voxnear: '90',
  },
};

/** One-line UI descriptions (shown under the buttons in the pre-boot picker). */
export const PRESET_LABELS: Record<QualityTier, { title: string; desc: string }> = {
  low: { title: 'LOW', desc: 'for weaker machines — reduced shadows, grass and detail distance' },
  medium: { title: 'MEDIUM', desc: 'the intended experience — balanced quality and performance' },
  high: { title: 'HIGH', desc: 'maximum detail — real leaf geometry farther out, full-res shadows' },
};

/** Accepts 'low' | 'med' | 'medium' | 'high' (case-insensitive); null otherwise. */
export function normalizeTier(v: string | null | undefined): QualityTier | null {
  if (!v) return null;
  const s = v.toLowerCase();
  if (s === 'low') return 'low';
  if (s === 'med' || s === 'medium') return 'medium';
  if (s === 'high') return 'high';
  return null;
}

/** Rewrite the URL with the preset's params as DEFAULTS (absent keys only —
 *  explicit params always win). Must run before the main module graph loads. */
export function applyPreset(tier: QualityTier): void {
  const q = new URLSearchParams(window.location.search);
  let changed = false;
  for (const [k, v] of Object.entries(PRESET_PARAMS[tier])) {
    if (!q.has(k)) {
      q.set(k, v);
      changed = true;
    }
  }
  if (changed) {
    history.replaceState(null, '', `${window.location.pathname}?${q.toString()}`);
  }
}

/** The quality tier resolved for THIS boot. boot.ts sets it (before the engine
 *  module graph loads) so downstream readers — the heightfield grid sizing and
 *  its bootcache key — see the picker/URL choice without re-deriving it. The
 *  interactive picker's choice never reaches the URL (so a reload re-asks), which
 *  is why this rides module state rather than location.search. Defaults to MEDIUM
 *  so non-boot contexts (unit tests / workers) get the shipped grid config. */
let resolvedTier: QualityTier = 'medium';
export function setResolvedTier(tier: QualityTier): void {
  resolvedTier = tier;
}
export function activeTier(): QualityTier {
  return resolvedTier;
}
