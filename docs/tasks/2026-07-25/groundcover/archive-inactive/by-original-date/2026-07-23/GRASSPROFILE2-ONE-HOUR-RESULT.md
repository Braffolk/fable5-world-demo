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

## Independent audit — canonical-depth one-step epipolar readdressing

**Candidate under audit:** replace the isolated profile's horizontal projected-
path relift with one shell-frame correction while retaining a fixed categorical
runtime. This section is a mathematical audit, not an authorization claim for
the parked shell-frame model.

### Exact address identity

Use a boundary chart plane

\[
  \Pi=\{X:n\cdot X=h\}
\]

with an orthonormal tangent basis `B` (`B^T n=0`). A canonical inward row is a
unit vector `c`, `n·c != 0`. Its frame stores **full signed canonical ray
depth**, not horizontal projected path:

\[
  X_c(a,\tau)=Q(a)+\tau c,\qquad D_c(a)=\tau_{\rm first},
\]

where `Q(a)` is the point on `Pi` with tangent coordinate `a`. Miss is a
categorical record, never a numeric far depth.

Let the unit live ray `C+t d` cross `Pi` at `Q(a)` and define

\[
  t_\Pi={h-n\cdot C\over n\cdot d},\qquad
  \gamma={n\cdot c\over n\cdot d},\qquad
  e=B^T(\gamma d-c).
\]

Equating a live-ray point with a canonical-row point gives, without an
approximation,

\[
  t-t_\Pi=\gamma\tau,\qquad a_c=a+e\tau.
\]

Therefore the exact shared hit is the fixed point

\[
  \boxed{\tau_* = D_c(a+e\tau_*)},\qquad
  \boxed{t_*=t_\Pi+\gamma\tau_*}.
\]

For the present top-plane convention (`n=+Y`) this reduces to

\[
  e=c_y{d_{xz}\over d_y}-c_{xz}.
\]

It is valid for either sign of `d_y` only when the canonical row has the same
inward sign: top/down uses `d_y,c_y<0`; bottom/up uses `d_y,c_y>0`. Opposite
signs produce a behind-boundary solution and must not be selected. Exact
`d_y=0` is not an epsilon case: use an intersected side/dominant-axis chart,
or return MISS when a horizontal ray above/below the box cannot enter it. The
current downward-only 15--75 degree GCRP cannot by itself cover upward or
horizontal exterior rays.

### Correct one-step shell iteration

Split the canonical full depth into a smooth shell and a coupled residual:

\[
  D_c(a)=O_c(a)+R_c(a).
\]

First solve the smooth fixed point

\[
  \tau_0=O_c(a+e\tau_0),\qquad a_0=a+e\tau_0.
\]

For a locally affine shell this solve is exact. Its denominator is

\[
  A=1-\nabla O_c(a_0)\cdot e,
\]

which must be certified away from zero. The first complete-record read at
`a_0` supplies `R_0`. The dimensionally correct correction is

\[
  z_1={R_0\over A},\qquad a_1=a_0+e z_1.
\]

Read one final **complete categorical record** at `a_1`, giving `R_1` and all
attached geometry/material fields. In the same locally affine shell chart the
one-step result is

\[
  z_2={R_1\over A},\qquad \tau_2=\tau_0+z_2.
\]

The division by `A` is not optional under this record convention. If `R` is
constant but `grad O·e != 0`, adding `R` directly leaves
`R(1/A-1)=O(Delta sigma)` error, so the claimed second-order result would
already fail on a smooth same-owner sheet.

With only a frame record, reconstruct its real canonical surface point

\[
  X_1=Q(a_1)+D_c(a_1)c
\]

and output live scene depth

\[
  \boxed{t_{\rm out}=d\cdot(X_1-C)}
\]

for unit `d`. Do not output canonical `D_c`, projected horizontal distance, or
`tau_2` as though it were already live-camera depth. At the exact fixed point
the projection equals `t_*`; after one step its transverse mismatch is the
remaining second-order term. If the categorical record carries the actual
owner plane/domain, exact live plane reintersection is stronger still.

### Error removed — and the premise required

Inside one Lipschitz owner sheet, write the residual fixed-point equation in
the shell-linearized coordinate:

\[
  z=G(z)={R_c(a_0+ez)\over A},\qquad
  k={e\cdot\nabla R_c\over A}.
\]

When `|k|<1`, no readdressing gives `z_1=G(0)` and

\[
  |z_1-z_*|=O(|k|\,\sigma)=O(\Delta\sigma).
\]

The dependent reread gives `z_2=G(z_1)` and hence

\[
  \boxed{|z_2-z_*|=O(|k|^2\sigma)=O(\Delta^2\sigma)}.
\]

Thus the candidate removes the first-order **same-sheet epipolar
misregistration**. It does not remove angular first-owner error. The proof
requires the initial and corrected addresses to remain on one residual sheet,
`A` to remain conditioned, and `R` to be Lipschitz there. These are precisely
the premises which the sparse Calamagrostis/Agrostis shell-frame gate measured
as false in many cells.

### Explicit limits and counterexamples

1. **Owner birth.** A finite triangle can intersect rays only for live slopes
   inside one angular cell while both sampled endpoint rows MISS it. Neither
   `R_0` nor `R_1` contains that unseen owner, so readdressing cannot create
   the correct hit.
2. **Owner/order crossing.** Two valid planes can have canonical depths
   `tau_A(s)=1+s` and `tau_B(s)=1-s` on a local slope coordinate. Row
   `s<0` categorically stores `A`, but a live `s>0` ray is first on `B`.
   Perfectly readdressing the `A` sheet remains perfectly self-consistent and
   wrong. Numerically mixing `A/B` depth would instead create a third surface.
3. **Silhouette/miss crossing.** If `a_0` is MISS there is no residual with
   which to reach a live hit at `a_1`; if it is a hit and `a_1` crosses its
   finite triangle edge, the Lipschitz proof ends at that edge.
4. **Grazing.** In the top chart `e` grows like `1/|d_y|`. For equal azimuth,
   canonical elevation `beta` and live elevation `alpha`, the ground-to-top
   address shear is

   \[
     H|\cot\alpha-\cot\beta|.
   \]

   With `H=1.1765 m`, `alpha=5 deg`, `beta=15 deg`, this is `9.0567 m`.
   `A` or the residual contraction bound can therefore fail long before
   `d_y=0`. A 5-degree row reduces this one mismatch at 5 degrees; it does
   not certify interior owner births/crossings and it cannot cover the exact
   horizontal limit.

### Smallest geometry-safe fixed read schedule

After choosing one direction node categorically, the minimum general schedule
for this one-step model is:

1. one low-resolution shell read containing `O` and its tangent gradient;
2. one complete record at `a_0` supplying the coupled residual `R_0`;
3. one complete record at `a_1` supplying final depth, coverage, normal,
   colour, species/material mark, and owner together.

That is **three fixed coherent reads for one categorical node**, one dependent
chain, no loop or march. Geometry is taken only from read 3. If three direction
nodes are retained for band-limited premultiplied radiance, strict alignment is
`3 x 3 = 9` reads; radiance may blend only after alignment, while geometry,
normal, mark, and owner still come from one max-weight node's final record.
The 7-read winner-only variant keeps geometry categorical but leaves the two
secondary colour records at first-order alignment. No read schedule, including
9 reads, repairs an owner birth or order crossing absent from the selected
canonical record.

**Verdict:** canonical full-depth epipolar readdressing is a mathematically
valid and cheap replacement for the current projected-path relift **inside a
certified same-owner shell cell**. It is not a complete arbitrary-angle
Calamagrostis reconstruction by itself, and the existing sparse-community RED
gate forbids presenting it as one. A runtime experiment may label it as a
bounded visual mitigation only; exact publication still needs categorical
owner-cell certification or the fixed-chart boundary codec.
