# Grassprofile 2: standing-height exterior reconstruction

Date: 2026-07-23

Status: implementation contract for the first real standing-height correction.
This note is deliberately renderer-free.  It describes one periodic cover tile in
its authored coordinates and one exterior ray.  The existing medium-height result
is the control that must not regress.

## 1. Observed regime and the current mathematical defect

Let the top of the fixed cover box be the plane `y = H`.  A live unit ray enters
that plane at `p = (p_x, H, p_z)` with direction `d`, `d_y < 0`.  A baked node has
unit direction `c`, `c_y < 0`, and stores the first-hit distance `tau_c(q)` for a
ray beginning at top-plane point `q`.

The current acceptance path does not preserve `tau_c`.  It stores horizontal path

```text
lambda_c(q) = tau_c(q) |c_xz|
```

and reconstructs

```text
tau_live = lambda_c / |d_xz|.
```

For a hit at height `y`, this gives

```text
y_hat = H - (H-y) tan(alpha_live) / tan(alpha_bake).
```

Thus a live ray below the lowest 15-degree row collapses the whole plant toward
the top plane.  With `H = 1.1765 m`, `alpha_live = 9 degrees`, and
`alpha_bake = 15 degrees`, a true ground hit is reconstructed around `0.48 m`
above ground.  This is the standing-height vertical stretching.  A 10 m camera
usually supplies rays at or above 15 degrees, explaining why that regime improved.

No filtering rule can repair this identity.  The stored quantity is wrong for the
claimed camera domain.

## 2. Full-depth epipolar identity

The same world point `X` lies on both rays exactly when

```text
X = p + tau_live d = q + tau_c(q) c.
```

Equality of height gives

```text
tau_live = tau_c(q) c_y / d_y.
```

Equality in the top plane then gives the fixed-point address

```text
q = p + B_c(d) tau_c(q)
B_c(d) = (c_y / d_y) d_xz - c_xz.
```

This is the exact projective relation.  It preserves the baked hit height; it does
not reinterpret horizontal travel as live 3D depth.

For a nearby direction node, use one fixed correction:

```text
tau_0 = tau_c(p)
q_1   = p + B_c(d) tau_0
tau_1 = tau_c(q_1)
X_1   = q_1 + tau_1 c
tau_out = dot(X_1 - p, d).
```

At the exact fixed point, `tau_out = tau_1 c_y / d_y`.  The projected form is
used because it remains the correct live-ray compositing coordinate when the one
step has a residual address error.

## 3. Direction-cell and categorical rules

The four regular-lattice corners around the live direction are evaluated
independently.  For each corner:

1. read its full baked depth at `p`;
2. compute `q_1` from that corner's actual direction;
3. read the coupled depth/coverage/normal record at `q_1`.

Depth, normal, species, and owner are never numerically blended.  The geometry
record is selected categorically from the corrected corner with greatest
directional weight among covered records.  Premultiplied colour and alpha alone
may be directionally filtered across the corrected corners.  A miss remains a
miss; it is not converted into a distant surface.

This removes the proven first-order height collapse and performs one projective
address correction.  It does not claim to synthesize an owner which is invisible
at all four lattice corners.  Such an angular birth/crossing is an unresolved
direction-cell event, not permission to blend geometry.  The visual gate therefore
specifically includes slow camera translations at standing height so a coherent
wrong-owner wedge cannot hide in a still image.

## 4. Fixed cost

The isolated profile currently performs four depth reads, four normal reads, and
four colour reads: 12 texture operations.  The replacement performs:

```text
4 initial coupled-record reads
4 corrected coupled-record reads
4 corrected premultiplied-colour reads
= 12 reads
```

Normal and depth come from the same corrected record, so the correction adds no
texture traffic relative to the current isolated path.  It adds fixed arithmetic
only: four direction reconstructions, four two-component epipolar offsets, and one
four-way categorical election.  There is no loop, march, candidate list, new pass,
barrier, dispatch, binding, or runtime geometry.  Accesses remain within the same
two atlases; the second read is dependent but spatially coherent within each view.

## 5. Binding limits and gates

- Exterior only.  `d_y < 0` and the camera is above the cover box.  Entry into the
  box remains the existing top-plane solve.  The permitted whole-box interior fade
  is separate and must not activate at standing height when the eye is above `H`.
- The 15-degree floor remains a hard sampling deficiency below 15 degrees.  The
  exact identity removes the false height reconstruction; one readdress reduces
  phase error, but the angular cell is still wide.  If the standing gate retains
  coherent owner wedges after this correction, the next required data change is a
  genuine low-elevation row or a full exterior categorical boundary field, not a
  shader-side clamp or a second march step.
- Acceptance requires both: (a) no standing-height vertical stretching or
  camera-distance height band, and (b) no regression in the accepted ~10 m oblique
  view.  Top-down correctness and authored panicle colours must remain intact.

## 6. First implementation result

Implemented in the isolated `grassprofile=2` path on 2026-07-23.  The atlas now
retains full baked depth; the runtime performs four initial reads, four corrected
coupled-record reads, a categorical covered-record election, and four corrected
premultiplied-colour reads.  The old standalone follow-up normal and colour queries
are bypassed, so the fixed count remains 12.

Supporting gates passed:

- TypeScript typecheck;
- 42 focused profile/ray-math tests;
- exact real-WebGPU standing boot at `alt=1.7`, `pitch=-0.28`;
- exact real-WebGPU shallow standing boot at `alt=1.7`, `pitch=-0.04`;
- exact real-WebGPU 10 m control boot at `alt=10`, `pitch=-0.28`;
- no page, TSL, WebGPU validation, bind-group, pipeline, or command-buffer error.

The captured standing frames no longer contain the prior camera-distance height
sheet, and the 10 m oblique control retains upright plant-scale structure and
authored colour.  This remains pending the user's close-flight motion review; that
review is the acceptance gate for residual owner wedges or phase jumps that one
still image cannot certify.
