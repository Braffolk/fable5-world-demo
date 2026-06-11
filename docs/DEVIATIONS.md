# DEVIATIONS — spec items replaced by nearest-feasible alternatives

Per PROJECT_LAAS_v2.md: any infeasible/impractical spec item gets the closest
feasible implementation plus an entry here explaining the gap.

## D-1: GTAO "indirect lighting only" (Phase 3)

**Spec:** GTAO modulates indirect lighting only.
**Implemented:** GTAO runs as a post multiply (depth-derived normals,
distance-faded ≤1.8 km) with a *luminance mask*: pixels whose HDR luminance
indicates direct sun (≥~4× ambient) shed 75% of the AO. This approximates
"indirect-only" without restructuring the lighting pipeline.
**Why:** true indirect-only AO needs the AO factor inside the lighting loop
(`material.aoNode`), but the AO texture is produced from the same frame's
depth — a depth prepass or 1-frame-latency feedback is required. Planned for
the Phase-4 material restructure (vegetation materials need `aoNode` wiring
anyway).

## D-2: Screen-space bounce light (Phase 3)

**Spec:** screen-space bounce as part of the GI stack.
**Implemented:** bounce comes from the irradiance probe field (heightfield
ray-marched sun+sky gather, SH-L1, 256×256×6 terrain-relative). No separate
screen-space pass yet.
**Why:** on a terrain-only world the probe field already carries the
dominant bounce signal (valley walls, couloirs); a screen-space pass mostly
pays off for fine geometry (tree trunks against rock, etc.). Revisit with
Phase-4 vegetation, where it lands together with foliage translucency.

## D-3: Probe density (Phase 3)

**Spec floor:** probes ≥ 24×24×6 per chunk (≈5 m spacing at 128 m chunks).
**Implemented:** world-uniform 16 m horizontal spacing × 6 terrain-relative
height layers (256×256×6 = 393k probes, time-sliced full refresh ≈ 2 s,
ToD jumps fast-converged via invalidate()).
**Why:** 5 m spacing world-wide = 3.5M probes — refresh latency and memory
outgrow their visual payoff before vegetation exists. A camera-following
high-density L0 clipmap (5 m) is planned for Phase 4/5 when canopy-scale
occlusion makes it visible. The floor is interpreted as the final-state
near-camera density.

## D-1 UPDATE (Phase 4): aoNode landed on asset materials

All Phase-4 asset materials (bark, rock, deadwood, grass) wire baked
cavity/crevice AO through `material.aoNode` — true indirect-only occlusion
inside the lighting loop. The screen-space GTAO keeps the Phase-3 luminance
mask for terrain. Remaining gap: terrain splat material itself (no baked
cavity texture) — revisit if terrain close-ups demand it.

## D-2 UPDATE (Phase 4): screen-space bounce + translucency landed

- Screen-space bounce: half-res depth-gated radiance gather added to the
  post stack (`?ablate=bounce`), composited before TRAA, receiver-chroma
  modulated. Subtle by design — probe GI carries large-scale bounce.
- Foliage translucency: thin-surface back-transmission term on foliage
  cards, hero leaf meshes, and grass tips (shared sun uniforms). NOT yet
  shadow-gated (glows slightly in shaded foliage when looking sunward);
  proper gating needs a light-space visibility query — queued for Phase 5/6
  alongside the wind field. Coefficient kept low (0.032) until then.

## D-4: Octahedral impostor runtime (Phase 4 → 5)

**Spec:** LOD chain to octahedral impostors (≥ 8×8 views, albedo+normal+depth).
**Implemented (Phase 4):** capture rig produces 8×8 hemi-octahedral atlases
(albedo sqrt-encoded + world-normal + per-view linear depth in alpha), plus
fixed-view relit preview cards verified against the source tree in the
gallery.
**Deferred:** the runtime impostor material (3-view blend by camera
direction, depth-based parallax, dithered transitions) belongs to the
Phase-5 LOD/scatter system where impostors actually draw. Capture data and
encoding are final.
