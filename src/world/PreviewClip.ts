/**
 * Research-preview clip — "render ONLY the generated area, void beyond".
 *
 * A cooked-micro RESEARCH PREVIEW splices a tiny synthesized patch (a ~128 m
 * microtopography core) into the full Estonia base map for hierarchy closure,
 * so the whole country renders at coarse/untouched quality around the patch and
 * every reviewed fix applies only inside it — the reviewer is forced to guess
 * "bug, or just outside the high-detail region?". The cure is to render ONLY
 * the generated region: terrain, water, vegetation/scatter and far tiles are
 * discarded outside the clip box, leaving cleared depth ⇒ sky void with a hard,
 * unambiguous boundary.
 *
 * GATING (preview-only, structurally incapable of touching production):
 *   - armed ONLY from the one world-source construction site (TerrainScene)
 *     when the opened manifest is a cooked-micro preview: format 2 AND
 *     height.finestLod < 0 AND height.synthesis === 'microtopography-v1'.
 *     The full Estonia release (format 1) and the generated world (no remote
 *     manifest) can never match; `?previewclip=0` is the explicit opt-out.
 *   - consumers read the box at NODE-GRAPH BUILD time: when null (every
 *     non-preview run) the clip contributes ZERO nodes — the compiled shaders
 *     are byte-identical to a build without this module.
 *
 * BOX (single source of truth): the honest generated area is the synthesized
 * 128 m core, NOT the larger LOD −2 fine-data window (its ring is parent-
 * resampled context — exactly the ambiguous zone the clip exists to remove).
 * The core is not tile-aligned so it cannot be derived from the chunk index;
 * it is pinned here in EPSG:3301 EN from the cook and converted to game
 * coordinates through the manifest grid (gameX = E − anchorE, gameZ =
 * anchorN − N). If a future preview build's fine window does not contain this
 * core, the clip falls back to the manifest-derived fine-window extent so the
 * preview still clips to ITS generated region instead of a stale box.
 */

import type { WorldManifest } from './source/WorldSource';

/** world-space (game) XZ clip rectangle, meters. */
export interface PreviewClipBox {
  minX: number;
  minZ: number;
  maxX: number;
  maxZ: number;
}

/** Synthesized micro core, EPSG:3301 EN (E_min, N_min, E_max, N_max) — the bog
 *  preview core from the cook (asset-gen peat-raised-bog bundle, 128 m). */
const CORE_EN = { eMin: 540224, nMin: 6429504, eMax: 540352, nMax: 6429632 };

let armed: PreviewClipBox | null = null;

/** The active preview clip box, or null (= no clipping; the default). Read at
 *  shader BUILD time by NaniteResolve + WaterMaterial. */
export function previewClipBox(): PreviewClipBox | null {
  return armed;
}

/** Decide + arm the preview clip for this boot. Call ONCE from the world-source
 *  construction site right after `worldSource.open()` — pass null for the
 *  generated (non-streamed) world so a scene rebuild always re-decides instead
 *  of inheriting a previous boot's box. Returns the armed box (null = not a
 *  micro preview / opted out). */
export function armPreviewClip(manifest: WorldManifest | null): PreviewClipBox | null {
  armed = manifest === null ? null : detect(manifest);
  if (armed) {
    console.info(
      `[preview-clip] cooked-micro preview detected — rendering ONLY x ${armed.minX}..${armed.maxX}, ` +
        `z ${armed.minZ}..${armed.maxZ} (void beyond); ?previewclip=0 disables`,
    );
  }
  return armed;
}

function detect(manifest: WorldManifest): PreviewClipBox | null {
  const h = manifest.layers.height;
  if (manifest.format !== 2 || h === undefined) return null;
  const finestLod = h.finestLod ?? 0;
  if (finestLod >= 0 || h.synthesis !== 'microtopography-v1') return null;
  if (new URLSearchParams(window.location.search).get('previewclip') === '0') return null;

  const g = manifest.grid;
  // fine-data window: extent of the finest-LOD height chunks (game coords).
  const s = g.chunkMeters * g.lodStep ** finestLod;
  let fine: PreviewClipBox | null = null;
  for (const k of manifest.chunks('height', finestLod)) {
    const minX = g.originX + k.cx * s;
    const minZ = g.originZ + k.cz * s;
    fine =
      fine === null
        ? { minX, minZ, maxX: minX + s, maxZ: minZ + s }
        : {
            minX: Math.min(fine.minX, minX),
            minZ: Math.min(fine.minZ, minZ),
            maxX: Math.max(fine.maxX, minX + s),
            maxZ: Math.max(fine.maxZ, minZ + s),
          };
  }
  if (fine === null) return null;

  const core: PreviewClipBox = {
    minX: CORE_EN.eMin - g.anchorE,
    maxX: CORE_EN.eMax - g.anchorE,
    minZ: g.anchorN - CORE_EN.nMax,
    maxZ: g.anchorN - CORE_EN.nMin,
  };
  const coreInFine =
    core.minX >= fine.minX && core.maxX <= fine.maxX && core.minZ >= fine.minZ && core.maxZ <= fine.maxZ;
  return coreInFine ? core : fine;
}
