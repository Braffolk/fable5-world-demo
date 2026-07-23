# Grass profile 2: 2026-07-23 one-hour result

## Inspectable outcome

Commit `24027f1` splits the former 2,400-line grass renderer into focused
modules and gives `grassprofile=2` a separate single-species binding.  That
binding loads the accepted 1.1765 m Calamagrostis GCRP/v4 source, uses its
filterable precomputed ray carrier, and carries its authored per-hit RGB into
the resolve.  It is fixed cost, has no ray march or runtime grass geometry, and
removes the second anti-tiling copy from the single-species graph so individual
source silhouettes are easier to judge.

The exact review URL was booted through 48 settled real-WebGPU frames with no
browser, TSL, shader, bind-group, pipeline, or WebGPU validation error:

`http://localhost:5173/?scene=world&src=estonia&dataurl=http://localhost:8787&dpr=2&grassprofile=2&x=311123&z=190723&alt=1.7&yaw=0&pitch=-0.28`

The ordinary renderer was booted separately after the split and also passed.
Typecheck, diff check, and 11 focused ground-cover format/regression tests pass.

## Honest boundary

This visible lane is an isolated recovery preview, not the GREEN exterior
boundary-transfer codec.  The current GBR4/v4 reference is correctly marked
RED: only 8 of 4096 tested fine cells certify one surface, and binding its mixed
records as geometry recreates the terrain-coloured mosaic/sheet.  The frame
loader validates that RED asset but refuses to bind it as a surface.

The preview still interpolates first-hit records from neighbouring angular
views.  Those records need not share a triangle owner.  Therefore the preview
cannot prove arbitrary-triangle-soup fidelity at every exterior angle; grazing
rows outside the baked 15–75 degree elevation range clamp to endpoint records.
The regular periodic rows visible from altitude are also real source-community
repetition, not a projection fix.

## Fastest correct continuation

1. Preserve this split and use the preview only as the visual/source baseline.
2. Work one level upstream in the cook: produce a GREEN fixed-chart asset whose
   regular cells carry one coherent owner record and whose genuinely mixed cells
   carry filtered appearance only.  Never average depth or normal across owners.
3. Require the existing publication certificate to turn GREEN before binding.
4. Transcribe that asset through the already-separated boundary fragment path;
   do not add another renderer, carrier mesh, pass, march, or candidate loop.
5. Re-run the same exterior-angle flight gate, then profile only after visual
   acceptance.

Resume condition: a certified asset with useful regular-cell coverage, bounded
mixed appearance, and memory/read counts inside the low-end budget.  Until then,
shader-side tuning cannot make the missing categorical owner information exact.
