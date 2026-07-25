# Shell-frame field gate: RED prerequisite blocker

Date: 2026-07-22
Status: **scope bounded by `GRASS-SHELL-FRAME-FIELD.md` Section 12; runtime remains parked**
Artifact: `data/work/groundcover-shell-frame-field/96c0e04d31ffde29/ddbc8512f4c13a1a/`

> Historical prerequisite-law result only. Section 12's missing measurements
> are now complete. The final reconstruction verdict and resume condition are
> in `GRASS-SHELL-FRAME-FIELD-RECONSTRUCTION-BLOCKER.md`.

## Short result for the paper designer

The shell-frame field's original *general* empirical premise does not hold
for the tested sparse/open authored Agrostis community. Exterior first hits do **not**
collapse into a millimetric, grazing-narrowing residual around a smooth
per-view shell. At 1 degree, the raw cover-only first-hit residual is
`0.194–0.201 m` p95 across four azimuths, and the p95 of the per-shell-cell
local standard deviation is `0.157–0.162 m`. The conditioned pooled crisp
p95 is `0.1964 m`. It is nearly constant or grows toward grazing instead of
decreasing.

Substitution into the proposal's own Section 5.2 law yields:

- `566` azimuth nodes on the 1-degree ring alone;
- `316` finite rings;
- `175,940` directions total for this sparse regime, versus a fixed `<=97`
  practical reconstruction budget;
- approximately `17.8 GiB` raw or `5.1 GiB` at the proposal's optimistic
  `3.5:1` compression estimate under the documented anisotropic-frame charge.

The strict 9-tap correction cannot repair this failure. It reduces
misregistration *after* a direction cell has been selected; it does not
reduce the measured residual `sigma` which sizes those cells. The 7-tap
winner-only variant and the old 6-tap count are likewise irrelevant to this
prerequisite.

## Check that the audit matches the proposed mathematics

The first execution did expose a harness transcription error: it initially
stored unit-ray distance. That quantity diverges as `1/sin(alpha)` and is not
the lever used by the slope-vector law. That execution was stopped without a
report. The recorded audit uses the corrected, exactly equivalent slope
parameterization below.

Let the direction-lattice coordinate be

\[
s_i = d_{i,xz}/(-d_{i,y})
\]

and let `B(u)=(u_x,Ebar,u_z)` be the point where the frame ray crosses the
chart mean-height plane. The same geometric line as the spec's normalized
orthographic ray is

\[
L_i(u,\lambda)=B(u)+\lambda(s_{i,x},-1,s_{i,z}).
\]

Here `lambda` is vertical shell depth. Rescaling a ray direction does not
change its line or its first hit. If a live ray of slope `s` crosses the mean
plane at phase `q`, equality of world points gives the exact epipolar
identity

\[
u=q+\lambda(s-s_i).
\]

Therefore the correction displacement is exactly

\[
\Delta u=(s-s_i)r,
\]

which is the proposal's `Delta-s times sigma` law without a hidden angular,
cosine, or unit conversion. The unit-direction form in the prose is
equivalent only when its depth and phase-rate factors are rescaled together.

The measured shell otherwise follows the frozen proposal literally:

- infinitely periodic, cover-only, orthographic first hits of the real
  deterministic Agrostis mesh;
- misses excluded from the mean and periodically extrapolated;
- `192x192` first-hit samples per direction, binned to a `32x32` per-view
  local mean shell;
- crisp families are foliage, culm, rhizome, and panicle axis; plume/fuzz is
  separate and never sizes the lattice;
- raw residuals are measured before conditioning, then the shell is smoothed
  until `max |grad o_i| * cellRadius <= 0.5`;
- pilot directions are 90, 75, 60, 45, 30, 18, 10, 5, 2, and 1 degrees,
  with four azimuths except at the vertical point;
- the stated `k=2.5`, `m0=1.6 m`, and a 60-degree/1920-pixel angular pixel
  width drive both radial and azimuthal lattice spacing.

This is an affine chart change from `Pi_i perpendicular d_i`, not a different
frame model. The horizontal mean-plane phase is one-to-one for every finite
descending direction and makes the periodic lattice exactly the authored XZ
lattice.

## Decisive measurements

| elevation | hit fraction across azimuths | raw crisp p95 | conditioned crisp p95 |
|---:|---:|---:|---:|
| 90 deg | 0.0845 | 0.1884 m | 0.2154 m |
| 75 deg | 0.0979–0.0995 | 0.0691–0.1141 m | 0.1697 m pooled |
| 18 deg | 0.3283–0.3440 | 0.1667–0.1737 m | 0.1989 m pooled |
| 5 deg | 0.6582–0.6972 | 0.1827–0.1881 m | 0.2039 m pooled |
| 2 deg | 0.8683–0.8982 | 0.1872–0.1938 m | 0.1938 m pooled |
| 1 deg | 0.9550–0.9660 | 0.1944–0.2011 m | 0.1964 m pooled |

At 1 degree, even the *median* per-cell standard deviation is
`0.101–0.113 m`, already tens of times above the claimed millimetric regime.
The contradiction is present in the raw local-mean fit, before shell
conditioning. A different smoother cannot remove it: the unsmoothed bin mean
is already the least-squares constant fit in each shell cell.

## Mathematical cause

The proposal assumes that increasing grazing occlusion selects only a thin
canopy crest. The actual periodic multi-height community does something else:
its hit probability rises from `8.5%` top-down to about `96%` at 1 degree,
but those newly intercepted first hits occur across the plant-height range.
Repeated oblique rays meet stems, leaves, panicle axes, and low crisp content
at different heights. Occlusion makes the frame nearly opaque without making
its first-hit depth single-valued around a smooth shell.

Consequently, a single smooth offset `o_i(u)` does not collapse categorical
first-hit depth. Its residual remains a substantial fraction of plant height.
This is the same multi-owner discontinuity in a new coordinate system, not a
filtering defect.

## What would have to change upstream

For this sparse regime, the frozen lattice law must replace at least one
load-bearing premise; implementation tuning cannot fix that law result. A
revised model needs one of:

1. a proved representation of the multiple height/owner strata inside each
   shell cell, with fixed small runtime cost and categorical selection;
2. a different direction law whose budget remains small for a measured
   `sigma ~= 0.20 m` and whose error bound still meets the visual contract;
3. a narrower authoring contract that excludes this actual multi-height
   community (which would no longer satisfy the arbitrary-authored-geometry
   goal); or
4. a new radiance-weighted statistic and theorem showing why high-residual
   crisp hits may be ignored while still meeting silhouette, geometry, and
   temporal thresholds. Merely renaming p95 as RMS is insufficient: local
   standard deviations are themselves roughly `0.10–0.16 m` at grazing.

## Waste/resume record

- Effort spent: one incomplete execution caught the unit-depth error; one
  corrected high-resolution prerequisite audit completed.
- Reusable result: exact slope-chart derivation, 9/7/6-tap cost separation,
  offline harness, content-addressed residual curves and source bindings.
- Exact blocker: sparse/open Agrostis emits `175,940`, not `<=97`, directions
  under the frozen law because measured crisp residual is about `0.20 m`, not
  millimetric or grazing-decreasing. This does not yet decide the dense regime
  or fixed-97 reconstruction quality.
- Why focus moved: the handoff requires parking after one premise audit and a
  still-RED result; runtime/shader implementation is unauthorized.
- Fallback now active: none selected in this track. Prior exact Class-E
  re-authoring remains documented separately, but is not silently substituted.
- Objective resume condition: a revised paper design must state an
  unambiguous residual statistic and coordinate convention, predict the
  recorded Agrostis curves, and derive a fixed-small-tap lattice/record budget
  that meets the exterior quality thresholds on this measured distribution.

No runtime, shader, render-pipeline, or grass asset file was changed by this
gate.
