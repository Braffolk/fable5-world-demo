# asset-gen change requests — from the streaming-world integration (2026-07-10)

**OWNERSHIP (user, 07-10): there is no parallel asset-gen session — THIS session
implements these itself** via agents working in asset-gen/. Since we can't push to S3,
dev serving = a local HTTP server over `asset-gen/data/out` (CORS-enabled) that the
runtime fetches from; the cook is content-addressed so new outputs land as a NEW
manifest dir + updated latest.json without disturbing the deployed set.

Context: the LAAS runtime is integrating the cooked Estonia data as a streamed world
(spec: SPEC-STREAMING-WORLD.md in this directory). These changes came out of the
spec-criticise loop; user directive on the shape: "if LOD3 is too detailed create LOD4
and 5 or smth. simple solutions!"

## BLOCKING (gate the runtime's far-shell / far-material milestones)

1. **Height LOD4** — extend the existing quadtree pyramid one rung: texel 256 m,
   footprint 2048·4^4 = 524.288 km (one chunk covers all Estonia). Same LAC1 format,
   same `lods[]` mechanism, nothing else changes. Add LOD5 only if measured sizes say
   LOD4 is still too heavy for a first fetch.
   WHY: measured LOD3 height = 15.4 MB total (12 chunks, 1.15-4.0 MB each) — 1 cm
   quantization at 64 m texels makes huge 2D deltas that don't compress. Boot wants a
   single sub-MB fetch for the whole-country far shell.
   ALSO: consider a relaxed per-LOD `qscale` at coarse rungs (e.g. 0.25-1 m at LOD3/4/5;
   qscale is already a per-chunk header field, so this is free format-wise). Nobody can
   see centimeters at 64-256 m texels, and it shrinks the deltas dramatically.

2. **Biome pyramid, LODs 1-4** — majority `classId` + mean `vegDensity` per texel
   (same 2-plane enc2 layout as LOD0). WHY: the runtime shades far terrain by land-cover
   class × density ("material + heightmap only" far bands); today biome exists only at
   LOD0 so the far shell would be shape without material.

3. **New `canopy` layer, LODs 1-4** — 2 planes: mean canopy height (u8, meters) +
   canopy cover (u8, 0-255), derived from the CHM the pipeline already ingests.
   WHY: far forests render as canopy-displaced terrain (real geometry, no impostors) —
   this layer is the displacement + coverage source beyond ~3 km. Near-field canopy the
   runtime derives itself from tree records, so LOD0 is NOT needed.

## BLOCKING (added by round-2 verification)

4. **Water at LOD1** (majority wet/dry + level) — mid-distance lakes; BLOCKING for the
   runtime's S9 far-water (waterYFar plane). LOD0 water already exists and covers near.

## NICE-TO-HAVE

5. Per-chunk species histogram (tiny sidecar or index extension) — far tint can lean
   conifer/deciduous without fetching tree records.

## Explicitly NOT requested

- LOD0 height changes — 2.7 MB/chunk is fine (content-hashed immutable + 3×3 demand ring).
- Any format/container change — LAC1 v1 as-is.
